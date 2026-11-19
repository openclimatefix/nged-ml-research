"""Experiment 709, phase B: the era columns of every arm, per (Primary, origin day).

One as-of function builds the columns of training rows and of forecast rows. The signal at the
origin of day `D` is the entry of day `D - 1` of a per-day signal (`entry_day`), so no reading at or
after 00:00 of day `D` can reach it. Arm `oracle` (O) is the one exception: it reads the true dates
and classes, and its entry of day `d` already holds the state of day `d + 1`, so the arm is labelled
non-causal (`Signal.causal` is False).

**Arms.** `oracle` and `placebo` are built from event lists (the era table, the verified switching
events of experiment 580, or the truth of an injected world). `all`, `rule`, `soft`, `hmm`, and
`stat` are read from the per-day signals of `e00709_classify` (`process_site`). The placebo
permutes the oracle's era switches and its switching events separately: each event keeps its class,
its change vector, and its duration, and receives a random Primary and a random onset day that lies
at least 28 days from any true event of the same world.

**What the network reads.** `era_block` returns `ERA_WIDTH` columns for each row: the nine encoder
inputs (unknown flag, class weight, age, six changes), the `active` flag, the anchor (standardised),
the anchor mode, the onset position in knot months, and the era-table slot. The other values of a
row (lag translation, rows held out of stage 3, row weights) come from `translation_rows`,
`hold_mask`, and `row_weight_factor`.

**Anchor rule.** Stage 1 trains every offset knot, so an era whose onset lies before the fold's cut
is already carried by the knots, and its anchor is zero (mode 0). An era whose onset lies after the
cut is carried by no knot, and the anchor is the full level change (mode 1). A learned arm that
confirms an era after the cut, though the estimated onset lies before it, uses mode 2: the anchor
is the level change minus the offset's own change between the knot before the onset and the newest
knot, clipped between zero and the full change (applied inside the network, because it reads the
offset knots). The curriculum's injected eras use mode 1.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import e00709_classify as classify
import e00709_features as features
import e00709_inject as inject
import numpy as np
import polars as pl
from e00709_hmm import STATE_ERA, STATE_SWITCH, STATE_UNKNOWN

LearnedArmType = Literal["all", "rule", "soft", "hmm", "stat"]
LEARNED_ARMS: Final[tuple[LearnedArmType, ...]] = ("all", "rule", "soft", "hmm", "stat")
SIGNAL_ARMS: Final[tuple[str, ...]] = ("oracle", "placebo", *LEARNED_ARMS)
SIGNALS_DIR: Final[Path] = features.DATA_DIR / "signals"

PER_DAY: Final[int] = features.HALF_HOURS_PER_DAY
AGE_CAP_DAYS: Final[int] = 365
CHANGE_SCALE: Final[float] = 4.0
"""The six scaled change values are divided by this and clipped to plus or minus 2."""
CHANGE_CLIP: Final[float] = 2.0
TRANSLATE_DAYS: Final[int] = 14
"""An era translates the lags of origins up to this many days after its onset."""
BLOCK_DAYS: Final[int] = 7
"""Rows of one era or event share a translation, read at the first day of their block."""
SWITCH_TRANSLATE_DAYS: Final[int] = 120
EARLIER_ERA_WEIGHT: Final[float] = 0.1
PLACEBO_SEED_BASE: Final[dict[str, int]] = {
    "real": 709_600_000,
    "curriculum": 709_700_000,
    "bank": 709_800_000,
}
"""Placebo seeds of the real record, the curriculum worlds, and the bank passes: disjoint ranges."""
ERA_AVERAGE_DAYS: Final[int] = 28
"""An injected era's band changes are averaged over this many days from its onset."""
PLACEBO_MIN_GAP_DAYS: Final[int] = 28
PLACEBO_TRIES: Final[int] = 200
SOFT_SCALE_STEP: Final[float] = 0.25
"""The class probability of the soft arm is rounded to this step when rows share a translation."""
ID_TABLE_SIZE: Final[int] = 130
"""Slot 0 is shared by every era id first seen after the cut. Slot 1 is the forced-zero slot of a
row without an era id. Slots 2 to 129 belong to (Primary, era id) pairs seen before the cut."""
SLOT_SHARED: Final[int] = 0
SLOT_NONE: Final[int] = 1
FIRST_SLOT: Final[int] = 2

MODE_NONE: Final[int] = 0
MODE_FULL: Final[int] = 1
MODE_CLIPPED: Final[int] = 2

STATE_NONE: Final[int] = 0
ERA_WIDTH: Final[int] = 14
"""Encoder inputs (9), active flag, anchor, anchor mode, onset position, era-table slot."""
ENCODER_INPUTS: Final[int] = 9
COL_ACTIVE: Final[int] = 9
COL_ANCHOR: Final[int] = 10
COL_MODE: Final[int] = 11
COL_ONSET_TAU: Final[int] = 12
COL_SLOT: Final[int] = 13

COLUMN_NAMES: Final[tuple[str, ...]] = (
    "state", "p_era", "age", "level", "anchor", "depth", "withdrawn", "era_id", "onset_day",
    "confirm_day",
    *(f"c{i}" for i in range(features.N_FEATURES)),
    *(f"b{i}" for i in range(4)),
)  # fmt: skip
"""Per-day columns of a signal; `c0` to `c5` are the scaled changes and `b0` to `b3` the band
changes in capacity share."""
INT_COLUMNS: Final[frozenset[str]] = frozenset(
    {"state", "withdrawn", "era_id", "onset_day", "confirm_day"}
)

BAND_OF_HALF_HOUR: Final[np.ndarray] = np.zeros(PER_DAY, dtype=np.int64)
for _band, _hh in enumerate(features.BAND_HALF_HOURS):
    BAND_OF_HALF_HOUR[_hh] = _band


@dataclass(frozen=True)
class Signal:
    """The per-day signal of every Primary for one arm.

    Attributes:
        arm: The arm's name.
        causal: False for the oracle and the placebo.
        columns: Each of `COLUMN_NAMES`, shape (sites, days); the entry of day `d` is the state at
            the end of day `d` (oracle: after day `d + 1`).
        episodes: One row per episode or event: `site`, `onset`, `alarm`, `end` (-1 while open),
            and `first_era` (first day with class `era`, -1 if none).
    """

    arm: str
    causal: bool
    columns: dict[str, np.ndarray]
    episodes: pl.DataFrame

    @property
    def n_sites(self) -> int:
        """Return the number of Primaries."""
        return self.columns["state"].shape[0]

    @property
    def n_days(self) -> int:
        """Return the number of days."""
        return self.columns["state"].shape[1]


@dataclass(frozen=True)
class OracleEra:
    """One true era.

    Attributes:
        site: Primary index.
        onset_day: First day of the era.
        scaled: The six scaled changes of the era against the level before it.
        bands: The four band changes in capacity share.
        level: The day-mean change in capacity share.
    """

    site: int
    onset_day: int
    scaled: np.ndarray
    bands: np.ndarray
    level: float


@dataclass(frozen=True)
class OracleSwitch:
    """One true switching event.

    Attributes:
        site: Primary index.
        onset_day: First day of the event.
        end_day: Last day of the event.
        scaled: The six scaled changes during the event.
        bands: The four band changes in capacity share.
        level: The day-mean change in capacity share (the depth).
    """

    site: int
    onset_day: int
    end_day: int
    scaled: np.ndarray
    bands: np.ndarray
    level: float


# ---------------------------------------------------------------- empty and assembled signals


def _empty_columns(*, n_sites: int, n_days: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in COLUMN_NAMES:
        if name in INT_COLUMNS:
            fill = -1 if name in {"onset_day", "confirm_day"} else 0
            out[name] = np.full((n_sites, n_days), fill, dtype=np.int64)
        else:
            out[name] = np.zeros((n_sites, n_days), dtype=np.float64)
    return out


def _episode_frame(*, rows: list[dict[str, int]]) -> pl.DataFrame:
    schema = {
        "site": pl.Int64, "onset": pl.Int64, "alarm": pl.Int64, "end": pl.Int64,
        "first_era": pl.Int64,
    }  # fmt: skip
    return pl.DataFrame(rows, schema=schema)


def site_columns(
    *, result: classify.SiteResult, arm: str, n_days: int, spread: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, int]]]:
    """Return one Primary's per-day columns and episode rows for a learned arm.

    Args:
        result: The Primary's `process_site` result.
        arm: One of `LEARNED_ARMS`.
        n_days: Number of days of the grid.
        spread: The Primary's robust spread of the six features.

    Returns:
        The columns (each of length `n_days`) and the episode rows (without the site).
    """
    key = "stat" if arm == "stat" else arm
    sig = result.signals[key]  # ty: ignore[invalid-argument-type]
    if arm == "hmm":
        bounds = [(a, a, b) for a, b in sig.episode_bounds]
    else:
        detector = result.stat.detector if arm == "stat" and result.stat else result.detector
        bounds = [(e.onset_day, e.alarm_day, e.end_day) for e in detector.episodes]
    first_era = []
    for k in range(len(bounds)):
        days = np.flatnonzero((sig.persistent_episode == k) & (sig.state == STATE_ERA))
        first_era.append(int(days[0]) if len(days) else -1)
    cols = _empty_columns(n_sites=1, n_days=n_days)
    cols = {k: v[0] for k, v in cols.items()}
    cols["state"] = sig.state.astype(np.int64)
    cols["p_era"] = sig.p_era.astype(np.float64)
    cols["age"] = np.minimum(sig.age_days, AGE_CAP_DAYS)
    cols["anchor"] = sig.anchor.astype(np.float64)
    cols["depth"] = sig.depth.astype(np.float64)
    cols["level"] = cols["anchor"] + cols["depth"]
    cols["withdrawn"] = sig.withdrawn.astype(np.int64)
    cols["era_id"] = sig.era_id.astype(np.int64)
    change = np.nan_to_num(sig.change_vector)
    for i in range(features.N_FEATURES):
        cols[f"c{i}"] = change[:, i]
    for i in range(4):
        cols[f"b{i}"] = change[:, i] * spread[i]
    for day in range(n_days):
        state = int(sig.state[day])
        k = int(sig.episode_id[day])
        persistent = int(sig.persistent_episode[day])
        if state == STATE_ERA and persistent >= 0:
            cols["onset_day"][day] = bounds[persistent][0]
            cols["confirm_day"][day] = first_era[persistent]
        elif state in (STATE_UNKNOWN, STATE_SWITCH) and k >= 0:
            cols["onset_day"][day] = bounds[k][0]
            cols["confirm_day"][day] = bounds[k][1]
    rows = [
        {"onset": o, "alarm": a, "end": e, "first_era": f}
        for (o, a, e), f in zip(bounds, first_era, strict=True)
    ]
    return cols, rows


def assemble(
    *,
    arm: str,
    causal: bool,
    per_site: dict[int, tuple[dict[str, np.ndarray], list[dict[str, int]]]],
    n_sites: int,
    n_days: int,
) -> Signal:
    """Stack the per-Primary columns into one `Signal`; a Primary without columns stays `none`.

    Args:
        arm: The arm's name.
        causal: Whether the arm reads only earlier data.
        per_site: Columns and episode rows by Primary index.
        n_sites: Number of Primaries.
        n_days: Number of days.

    Returns:
        The signal.
    """
    columns = _empty_columns(n_sites=n_sites, n_days=n_days)
    rows: list[dict[str, int]] = []
    for site, (cols, eps) in per_site.items():
        for name in COLUMN_NAMES:
            columns[name][site] = cols[name]
        rows += [{"site": site, **e} for e in eps]
    return Signal(arm=arm, causal=causal, columns=columns, episodes=_episode_frame(rows=rows))


def learned_signals(
    *, results: dict[int, classify.SiteResult], feats: list[features.SiteFeatures], n_days: int
) -> dict[str, Signal]:
    """Return the five learned arms' signals from the per-Primary results of `process_site`.

    Args:
        results: The results by Primary index (a Primary missing from it stays `none`).
        feats: The features of every Primary (for the spread).
        n_days: Number of days.

    Returns:
        One signal per learned arm that the results hold.
    """
    out = {}
    for arm in LEARNED_ARMS:
        per_site = {
            site: site_columns(result=r, arm=arm, n_days=n_days, spread=feats[site].spread)
            for site, r in results.items()
            if arm in r.signals and feats[site].ready_day < n_days
        }
        out[arm] = assemble(
            arm=arm, causal=True, per_site=per_site, n_sites=len(feats), n_days=n_days
        )
    return out


def merge_signals(*, parts: list[Signal]) -> Signal:
    """Add signals of disjoint Primaries (one injected pass each) into one.

    Args:
        parts: Signals of one arm, each active at its own Primaries only.

    Returns:
        The merged signal.

    Raises:
        ValueError: If two parts are both active at a Primary.
    """
    first = parts[0]
    columns = {k: v.copy() for k, v in first.columns.items()}
    for part in parts[1:]:
        for site in range(part.n_sites):
            if (part.columns["state"][site] != STATE_NONE).any():
                if (columns["state"][site] != STATE_NONE).any():
                    msg = f"two passes are active at Primary {site}"
                    raise ValueError(msg)
                for name in COLUMN_NAMES:
                    columns[name][site] = part.columns[name][site]
    episodes = pl.concat([p.episodes for p in parts])
    return Signal(arm=first.arm, causal=first.causal, columns=columns, episodes=episodes)


# ---------------------------------------------------------------- the oracle and the placebo


def hindsight_vector(
    *, feat: features.SiteFeatures, onset_day: int, end_day: int | None, horizon: int = 120
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the true change of a Primary over an era or event, read with hindsight.

    The change is the mean of the scaled features from the day after the onset to `end_day` (at most
    `horizon` days) minus the mean of the 180 days before the onset.

    Args:
        feat: The Primary's features.
        onset_day: First day of the era or event.
        end_day: Last day of the event, or None for an era (the horizon applies).
        horizon: The longest span averaged.

    Returns:
        The six scaled changes, the four band changes in capacity share, and the day-mean change.
    """
    last = min(feat.scaled.shape[0], onset_day + 1 + horizon)
    if end_day is not None:
        last = min(last, end_day + 1)
    post = feat.scaled[onset_day + 1 : max(last, onset_day + 2)]
    pre = feat.scaled[max(0, onset_day - 180) : onset_day]
    with np.errstate(all="ignore"):
        vec = np.nanmean(post, axis=0) - np.nanmean(pre, axis=0)
    vec = np.nan_to_num(vec)
    spread = np.nan_to_num(feat.spread)
    weights = features.BAND_WEIGHTS * spread
    return vec, vec[:4] * spread[:4], float(weights @ vec)


def real_oracle_events(
    *, feats: list[features.SiteFeatures], era_table: pl.DataFrame, known_events: pl.DataFrame
) -> tuple[list[OracleEra], list[OracleSwitch]]:
    """Return the true eras and switching events of the real record.

    Eras are the era table's rows that have a date (the telemetry date where the table has one,
    the register date otherwise). Switching events are the verified events of experiment 580.

    Args:
        feats: The features of every Primary.
        era_table: `era_table.parquet`.
        known_events: `known_events.parquet`.

    Returns:
        The eras and the switching events.
    """
    sites = list(features.PRIMARIES)
    eras = []
    for row in era_table.iter_rows(named=True):
        when = row["telemetry_date"] or row["register_date"]
        if when is None or row["era_index"] is None or row["primary"] not in sites:
            continue
        site, day = sites.index(row["primary"]), features.day_index(day=when)
        vec, bands, level = hindsight_vector(feat=feats[site], onset_day=day, end_day=None)
        eras.append(OracleEra(site=site, onset_day=day, scaled=vec, bands=bands, level=level))
    switches = []
    verified = known_events.filter(pl.col("source") == "experiment 580 verified")
    for row in verified.iter_rows(named=True):
        site = sites.index(row["site"])
        first = int((row["onset_utc"] - features.GRID_START).total_seconds() // 86400)
        last = int((row["end_utc"] - features.GRID_START).total_seconds() // 86400)
        vec, bands, level = hindsight_vector(feat=feats[site], onset_day=first, end_day=last)
        switches.append(
            OracleSwitch(
                site=site, onset_day=first, end_day=last, scaled=vec, bands=bands, level=level
            )
        )
    return eras, switches


def pass_oracle_events(
    *,
    injected: inject.InjectedPass,
    feats: list[features.SiteFeatures],
    panel: features.Panel,
) -> tuple[list[OracleEra], list[OracleSwitch]]:
    """Return the true events of an injected pass.

    An injected era's vector is exact: the injected change (injected power minus the real power)
    averaged over each band for `ERA_AVERAGE_DAYS` days from the onset. It reads nothing after the
    onset plus those days, so no half-hour from after the cut enters a curriculum era.

    Args:
        injected: The injected pass.
        feats: The features of every Primary on the injected panel (for the spread).
        panel: The real panel.

    Returns:
        The eras, and the switching events of each donor and each metered recipient.
    """
    eras = []
    for era in injected.eras:
        day = era.onset_hh // PER_DAY
        first = day * PER_DAY
        last = min(first + ERA_AVERAGE_DAYS * PER_DAY, panel.power.shape[1])
        change = injected.power[era.host, first:last] - panel.power[era.host, first:last]
        band_of = BAND_OF_HALF_HOUR[np.arange(first, last) % PER_DAY]
        bands = np.array(
            [
                float(np.nanmean(change[band_of == b]))
                if np.isfinite(change[band_of == b]).any()
                else 0.0
                for b in range(4)
            ]
        )
        spread = np.nan_to_num(feats[era.host].spread)
        vec = np.zeros(features.N_FEATURES)
        vec[:4] = np.divide(bands, spread[:4], out=np.zeros(4), where=spread[:4] > 0)
        level = float((features.BAND_WEIGHTS[:4]) @ bands)
        eras.append(OracleEra(site=era.host, onset_day=day, scaled=vec, bands=bands, level=level))
    switches = []
    for sw in injected.switches:
        first = sw.onset_hh // PER_DAY
        last = first + round(sw.duration_days)
        for site in (sw.donor, sw.recipient):
            if site is None:
                continue
            vec, bands, level = hindsight_vector(feat=feats[site], onset_day=first, end_day=last)
            switches.append(
                OracleSwitch(
                    site=site, onset_day=first, end_day=last, scaled=vec, bands=bands, level=level
                )
            )
    return eras, switches


def oracle_signal(
    *, arm: str, eras: list[OracleEra], switches: list[OracleSwitch], n_sites: int, n_days: int
) -> Signal:
    """Build the oracle's per-day signal from true events.

    The entry of day `d` holds the state of day `d + 1`, so the origin of day `D` sees events that
    begin on day `D` itself. Inside a switching event the state is `switch`; otherwise the newest
    era whose onset has passed is the state `era`.

    Args:
        arm: `oracle` or `placebo` (the placebo passes its permuted events).
        eras: The era events.
        switches: The switching events.
        n_sites: Number of Primaries.
        n_days: Number of days.

    Returns:
        The signal, flagged non-causal.
    """
    cols = _empty_columns(n_sites=n_sites, n_days=n_days)
    day_ref = np.arange(n_days) + 1
    rows: list[dict[str, int]] = []
    for site in range(n_sites):
        mine = sorted((e for e in eras if e.site == site), key=lambda e: e.onset_day)
        for e in mine:
            live = day_ref >= e.onset_day
            for name, value in (
                ("state", STATE_ERA), ("p_era", 1.0), ("level", e.level), ("anchor", e.level),
                ("onset_day", e.onset_day), ("confirm_day", e.onset_day),
            ):  # fmt: skip
                cols[name][site, live] = value
            cols["age"][site, live] = np.minimum(day_ref[live] - e.onset_day + 1, AGE_CAP_DAYS)
            for i in range(features.N_FEATURES):
                cols[f"c{i}"][site, live] = e.scaled[i]
            for i in range(4):
                cols[f"b{i}"][site, live] = e.bands[i]
            rows.append(
                {
                    "site": site,
                    "onset": e.onset_day,
                    "alarm": e.onset_day,
                    "end": -1,
                    "first_era": e.onset_day,
                }
            )
    for s in switches:
        live = (day_ref >= s.onset_day) & (day_ref <= s.end_day)
        site = s.site
        for name, value in (
            ("state", STATE_SWITCH), ("p_era", 0.0), ("level", s.level), ("anchor", 0.0),
            ("depth", s.level), ("onset_day", s.onset_day), ("confirm_day", s.onset_day),
        ):  # fmt: skip
            cols[name][site, live] = value
        cols["age"][site, live] = day_ref[live] - s.onset_day + 1
        for i in range(features.N_FEATURES):
            cols[f"c{i}"][site, live] = s.scaled[i]
        for i in range(4):
            cols[f"b{i}"][site, live] = s.bands[i]
        rows.append(
            {
                "site": site,
                "onset": s.onset_day,
                "alarm": s.onset_day,
                "end": s.end_day,
                "first_era": -1,
            }
        )
    return Signal(arm=arm, causal=False, columns=cols, episodes=_episode_frame(rows=rows))


def _far_enough(*, site: int, day: int, taken: list[tuple[int, int]], gap: int) -> bool:
    return all(not (s == site and abs(d - day) < gap) for s, d in taken)


def placebo_events(
    *,
    eras: list[OracleEra],
    switches: list[OracleSwitch],
    seed: int,
    eligible: list[int],
    window: tuple[int, int],
) -> tuple[list[OracleEra], list[OracleSwitch]]:
    """Permute the oracle's era switches and its switching events, separately.

    Each event keeps its class, change vector, and duration and receives a random eligible Primary
    and a random onset day inside `window`, at least `PLACEBO_MIN_GAP_DAYS` from every true event
    and every other permuted event of the same Primary. The count of each class is unchanged.

    Args:
        eras: The true era events.
        switches: The true switching events.
        seed: Seed of the permutation.
        eligible: Primaries that may receive an event.
        window: First and last allowed onset day (the last is exclusive).

    Returns:
        The permuted eras and switching events.
    """
    rng = np.random.default_rng(seed)
    true_days = [(e.site, e.onset_day) for e in eras] + [(s.site, s.onset_day) for s in switches]
    true_days += [(s.site, s.end_day) for s in switches]
    new_eras, new_switches = [], []
    taken: list[tuple[int, int]] = []
    for group, source in (("era", eras), ("switch", switches)):
        for e in source:
            for _ in range(PLACEBO_TRIES):
                site = int(rng.choice(eligible))
                day = int(rng.integers(window[0], window[1]))
                if _far_enough(
                    site=site, day=day, taken=true_days + taken, gap=PLACEBO_MIN_GAP_DAYS
                ):
                    break
            else:
                msg = "no placebo date satisfies the separation rule"
                raise RuntimeError(msg)
            taken.append((site, day))
            if group == "era":
                new_eras.append(
                    OracleEra(
                        site=site, onset_day=day, scaled=e.scaled, bands=e.bands, level=e.level
                    )
                )  # ty: ignore[unresolved-attribute]
            else:
                length = e.end_day - e.onset_day  # ty: ignore[unresolved-attribute]
                new_switches.append(
                    OracleSwitch(
                        site=site,
                        onset_day=day,
                        end_day=day + length,
                        scaled=e.scaled,
                        bands=e.bands,
                        level=e.level,
                    )  # ty: ignore[unresolved-attribute]
                )
    return new_eras, new_switches


# ---------------------------------------------------------------- disk


def write_signal(*, signal: Signal, folder: Path) -> Path:
    """Write a signal to `<folder>/<arm>.parquet` (long format) and `<arm>_episodes.parquet`.

    Args:
        signal: The signal.
        folder: The destination folder.

    Returns:
        The path of the signal file.
    """
    folder.mkdir(parents=True, exist_ok=True)
    n_sites, n_days = signal.n_sites, signal.n_days
    data: dict[str, np.ndarray] = {
        "site": np.repeat(np.arange(n_sites), n_days),
        "day": np.tile(np.arange(n_days), n_sites),
    }
    for name in COLUMN_NAMES:
        values = signal.columns[name].reshape(-1)
        data[name] = values if name in INT_COLUMNS else values.astype(np.float32)
    path = folder / f"{signal.arm}.parquet"
    tmp = path.with_suffix(".tmp")
    pl.DataFrame(data).write_parquet(tmp, compression="zstd")
    tmp.rename(path)
    episodes = folder / f"{signal.arm}_episodes.parquet"
    tmp = episodes.with_suffix(".tmp")
    signal.episodes.write_parquet(tmp)
    tmp.rename(episodes)
    return path


def read_signal(*, arm: str, folder: Path = SIGNALS_DIR) -> Signal:
    """Read a signal that `write_signal` wrote.

    Args:
        arm: The arm's name.
        folder: The folder of the signal files.

    Returns:
        The signal.
    """
    table = pl.read_parquet(folder / f"{arm}.parquet")
    n_sites = int(np.asarray(table["site"]).max()) + 1
    n_days = table.height // n_sites
    columns = {}
    for name in COLUMN_NAMES:
        values = table[name].to_numpy().reshape(n_sites, n_days)
        columns[name] = values.astype(np.int64 if name in INT_COLUMNS else np.float64)
    episodes = pl.read_parquet(folder / f"{arm}_episodes.parquet")
    return Signal(
        arm=arm, causal=arm not in ("oracle", "placebo"), columns=columns, episodes=episodes
    )


# ---------------------------------------------------------------- the as-of lookups


def entry_day(*, origin_day: np.ndarray) -> np.ndarray:
    """Return the entry of the per-day signal that the origin of each day reads: day `D - 1`.

    Args:
        origin_day: The origin's day index `D`.

    Returns:
        The entry index `D - 1`.

    Raises:
        ValueError: If an origin is on the first day, which has no earlier day.
    """
    if (origin_day < 1).any():
        msg = "an origin on the first day of the grid has no earlier day"
        raise ValueError(msg)
    return origin_day - 1


def era_slots(*, signal: Signal, cut_day: int) -> dict[tuple[int, int], int]:
    """Return the era-table slot of each (Primary, era id) that was confirmed before the cut.

    The key holds the Primary, so an id of one Primary can never read the slot of another.

    Args:
        signal: The arm's signal (only the `stat` arm has era ids).
        cut_day: Day index of the fold's cut; ids first seen on this day or later use `SLOT_SHARED`.

    Returns:
        Slots from `FIRST_SLOT` upward, in order of first appearance.

    Raises:
        ValueError: If the ids do not fit the table.
    """
    ids = signal.columns["era_id"][:, :cut_day]
    slots: dict[tuple[int, int], int] = {}
    for site in range(signal.n_sites):
        for era_id in np.unique(ids[site]):
            if era_id > 0:
                slots[(site, int(era_id))] = len(slots) + FIRST_SLOT
    if len(slots) + FIRST_SLOT > ID_TABLE_SIZE:
        msg = f"{len(slots)} era ids do not fit the table"
        raise ValueError(msg)
    return slots


def era_block(
    *,
    signal: Signal,
    site: np.ndarray,
    origin_day: np.ndarray,
    cut_day: int | np.ndarray,
    target_std: np.ndarray,
    tau_of_day: np.ndarray,
    soft: bool = False,
    slots: dict[tuple[int, int], int] | None = None,
) -> np.ndarray:
    """Return the (rows, `ERA_WIDTH`) era columns of rows, read as of the end of day `D - 1`.

    The same function builds training rows and forecast rows. A `none` row is all zeros.

    Args:
        signal: The arm's signal.
        site: Primary index of each row.
        origin_day: Day index `D` of each row's origin.
        cut_day: Day index of the fold's cut (the first day without a stage 3 row), as one value
            or one per row. A row of a curriculum world's injected era takes -1, because the era
            lies after every offset knot; an onset on the cut day itself is also after every knot.
        target_std: The standardisation scale of each Primary's target, to put the anchor in
            standardised units.
        tau_of_day: Knot position (months from 2019-09-01) of the start of each day.
        soft: Scale the era vector and anchor by the class probability (arm `soft`).
        slots: Era-table slots, from `era_slots`; None gives every era id the shared slot, which is
            right for a curriculum world (its ids never map onto a real Primary's slot).

    Returns:
        The columns, float32.
    """
    d = entry_day(origin_day=origin_day)
    col = {k: v[site, d] for k, v in signal.columns.items()}
    state = col["state"]
    # a hard arm shows the vector in the states `unknown` and `era` only: a `switch` row has no
    # vector and no anchor (plan section 2); the soft arm shows every state and weights by `p`
    shown = (state != STATE_NONE) if soft else np.isin(state, (STATE_UNKNOWN, STATE_ERA))
    out = np.zeros((len(site), ERA_WIDTH), np.float32)
    p = col["p_era"]
    inputs = np.zeros((len(site), ENCODER_INPUTS), np.float32)
    inputs[:, 0] = 0.0 if soft else state == STATE_UNKNOWN
    inputs[:, 1] = p
    inputs[:, 2] = np.log1p(np.minimum(col["age"], AGE_CAP_DAYS)) / np.log1p(AGE_CAP_DAYS)
    for i in range(features.N_FEATURES):
        inputs[:, 3 + i] = np.clip(col[f"c{i}"] / CHANGE_SCALE, -CHANGE_CLIP, CHANGE_CLIP)
    out[:, :ENCODER_INPUTS] = np.where(shown[:, None], inputs, 0.0)
    out[:, COL_ACTIVE] = shown
    full = p * col["level"] if soft else col["anchor"]
    onset, confirm = col["onset_day"], col["confirm_day"]
    if signal.causal:
        mode = np.where(
            onset >= cut_day, MODE_FULL, np.where(confirm >= cut_day, MODE_CLIPPED, MODE_NONE)
        )
    else:
        mode = np.where((state == STATE_ERA) & (onset >= cut_day), MODE_FULL, MODE_NONE)
    carries = shown & (full != 0.0) & (True if soft else state == STATE_ERA)
    mode = np.where(carries, mode, MODE_NONE)
    out[:, COL_ANCHOR] = np.where(carries, full / target_std[site], 0.0)
    out[:, COL_MODE] = mode
    out[:, COL_ONSET_TAU] = np.where(onset >= 0, tau_of_day[np.clip(onset, 0, None)], 0.0)
    ids = col["era_id"]
    if slots is None:
        slot = np.where(ids > 0, SLOT_SHARED, SLOT_NONE)
    else:
        slot = np.array(
            [
                slots.get((int(s), int(i)), SLOT_SHARED) if i > 0 else SLOT_NONE
                for s, i in zip(site, ids, strict=True)
            ]
        )
    out[:, COL_SLOT] = np.where(shown, slot, SLOT_NONE)
    return out


def hold_mask(*, signal: Signal, cut_day: int) -> np.ndarray:
    """Return the days a stage 3 training row must be held out, as of the fold's cut.

    A row is held out when its valid time lies in the unknown period of a shift (from the estimated
    onset until the day the class `era` is known), inside a shift classed `switch` (or never
    decided), or inside an era whose class was later withdrawn (the reversal rule).
    Only episodes whose alarm fell before the cut count, and an episode still
    open at the cut ends on the day before it.

    Args:
        signal: The arm's signal.
        cut_day: Day index of the fold's cut.

    Returns:
        A boolean array (sites, days) that is True on days to hold out.
    """
    hold = np.zeros((signal.n_sites, signal.n_days), dtype=bool)
    state = signal.columns["state"]
    for row in signal.episodes.iter_rows(named=True):
        if row["alarm"] >= cut_day or row["onset"] < 0:
            continue
        site = row["site"]
        last_open = cut_day - 1 if row["end"] < 0 else min(row["end"], cut_day - 1)
        last_open = max(last_open, row["alarm"])
        if signal.columns["withdrawn"][site, row["onset"] : last_open + 1].any():
            # the reversal rule: a wrong `era` class was withdrawn, so the whole span is dropped
            hold[site, row["onset"] : last_open + 1] = True
            continue
        # the oracle entry of day d holds day d + 1: its event ends one entry earlier
        final = int(state[site, last_open if signal.causal else max(last_open - 1, 0)])
        first_era = row["first_era"]
        if final == STATE_ERA:
            if first_era < 0 or first_era > last_open:
                hold[site, row["onset"] : last_open + 1] = True
            else:
                hold[site, row["onset"] : first_era] = True
        elif final in (STATE_UNKNOWN, STATE_SWITCH):
            hold[site, row["onset"] : last_open + 1] = True
    return hold


def row_weight_factor(
    *, signal: Signal, cut_day: int, site: np.ndarray, valid_day: np.ndarray
) -> np.ndarray:
    """Return the weight factor of each training row: 0.1 for rows from before the current era.

    The current era is the one standing at the end of the day before the cut.

    Args:
        signal: The arm's signal.
        cut_day: Day index of the fold's cut.
        site: Primary index of each row.
        valid_day: Day index of each row's valid time.

    Returns:
        1.0 for rows in the current era or with no era, `EARLIER_ERA_WEIGHT` for earlier rows.
    """
    d = cut_day - 1
    onset = signal.columns["onset_day"][site, d]
    era_now = (signal.columns["state"][site, d] == STATE_ERA) & (onset >= 0)
    return np.where(era_now & (valid_day < onset), EARLIER_ERA_WEIGHT, 1.0).astype(np.float32)


@dataclass(frozen=True)
class Translation:
    """How the lag inputs of rows are translated to the level of the current state.

    Attributes:
        row: Index of each translated row (each row at most once).
        onset: Onset day of the era or event.
        era_scale: Factor on the band changes added to lag times before `onset` (0 for none).
        bands: The four band changes in capacity share.
        switch_scale: Factor on the depth subtracted from lag times from `onset` on (0 for none).
        depth: The depth in capacity share.
        group: Rows with the same value share one translated panel (era and switch change together).
    """

    row: np.ndarray
    onset: np.ndarray
    era_scale: np.ndarray
    bands: np.ndarray
    switch_scale: np.ndarray
    depth: np.ndarray
    group: np.ndarray


def translation_rows(
    *, signal: Signal, site: np.ndarray, origin_day: np.ndarray, soft: bool = False
) -> Translation:
    """Return the rows whose lags need translating, with the values of the first day of their block.

    Rows of one era (origins up to `TRANSLATE_DAYS` after the onset) or one switching event (all
    origins inside it, up to `SWITCH_TRANSLATE_DAYS`) are cut into blocks of `BLOCK_DAYS` days, and
    all rows of a block read the band changes or depth at the block's first origin, which is an
    earlier entry than any row of the block, so the translation stays causal. Both translations of
    a row (the soft arm can need both) apply to one panel per (Primary, onset, block, class weight).

    Args:
        signal: The arm's signal.
        site: Primary index of each row.
        origin_day: Day index of each row's origin.
        soft: Scale era translation by the class probability and switch translation by one minus it.

    Returns:
        The translation plan; `row` is empty when nothing is translated.
    """
    d = entry_day(origin_day=origin_day)
    col = {k: v[site, d] for k, v in signal.columns.items()}
    state, onset = col["state"], col["onset_day"]
    age = origin_day - onset
    in_era_reach = (age >= 0) & (age <= TRANSLATE_DAYS)
    in_switch_reach = (age >= 0) & (age <= SWITCH_TRANSLATE_DAYS)
    if soft:
        era_flag = (state != STATE_NONE) & in_era_reach
        switch_flag = (state != STATE_NONE) & in_switch_reach
    else:
        era_flag = (state == STATE_ERA) & in_era_reach
        switch_flag = (state == STATE_SWITCH) & in_switch_reach
    row = np.flatnonzero(era_flag | switch_flag)
    if len(row) == 0:
        empty = np.empty(0)
        return Translation(
            row=row, onset=empty, era_scale=empty, bands=np.empty((0, 4)), switch_scale=empty,
            depth=empty, group=empty,
        )  # fmt: skip
    s = site[row]
    block = (origin_day[row] - onset[row]) // BLOCK_DAYS
    key = pl.DataFrame(
        {
            "site": s,
            "onset": onset[row],
            "block": block,
            "era": era_flag[row],
            "switch": switch_flag[row],
            "origin": origin_day[row],
        }
    )
    keys = ["site", "onset", "block", "era", "switch"]
    first = key.group_by(*keys).agg(pl.col("origin").min().alias("first"))
    joined = key.join(first, on=keys, how="left", maintain_order="left")
    first_origin = joined["first"].to_numpy()
    first_entry = first_origin - 1
    p_first = signal.columns["p_era"][s, first_entry]
    if soft:
        p_first = np.round(p_first / SOFT_SCALE_STEP) * SOFT_SCALE_STEP
        era_scale = np.where(era_flag[row], p_first, 0.0)
        switch_scale = np.where(switch_flag[row], 1.0 - p_first, 0.0)
    else:
        era_scale = era_flag[row].astype(np.float64)
        switch_scale = switch_flag[row].astype(np.float64)
    bands = np.stack([signal.columns[f"b{i}"][s, first_entry] for i in range(4)], axis=1)
    depth = signal.columns["level"][s, first_entry]
    group_key = key.with_columns(
        first=pl.Series(first_origin),
        era_scale=pl.Series(np.round(era_scale, 3)),
        switch_scale=pl.Series(np.round(switch_scale, 3)),
    )
    group = group_key.select(
        pl.struct("site", "onset", "first", "era_scale", "switch_scale").rank("dense")
    ).to_series()
    return Translation(
        row=row, onset=onset[row], era_scale=era_scale, bands=bands, switch_scale=switch_scale,
        depth=depth, group=group.to_numpy(),
    )  # fmt: skip


def translate_panel_row(
    *,
    power: np.ndarray,
    onset_day: int,
    era_scale: float,
    bands: np.ndarray,
    switch_scale: float,
    depth: float,
    lookback_days: int = 60,
) -> np.ndarray:
    """Return one Primary's power row translated to the level of its current state.

    Args:
        power: The Primary's power over capacity on the half-hourly grid.
        onset_day: Onset day of the era or event.
        era_scale: Factor on the band changes added before the onset (0 adds nothing).
        bands: The four band changes in capacity share.
        switch_scale: Factor on the depth subtracted from the onset on (0 subtracts nothing).
        depth: The depth in capacity share.
        lookback_days: How far before the onset an era's translation reaches.

    Returns:
        A translated copy of the row.
    """
    out = power.copy()
    onset_hh = onset_day * PER_DAY
    if era_scale:
        first = max(0, onset_hh - lookback_days * PER_DAY)
        hh = np.arange(first, onset_hh)
        out[first:onset_hh] += era_scale * bands[BAND_OF_HALF_HOUR[hh % PER_DAY]]
    if switch_scale:
        out[onset_hh:] -= switch_scale * depth
    return out


# ---------------------------------------------------------------- injected worlds


def pass_features(
    *, ctx_feats: list[features.SiteFeatures], injected: inject.InjectedPass
) -> list[features.SiteFeatures]:
    """Return the Primaries' features with those that an injected pass touched recomputed.

    Args:
        ctx_feats: The real features of every Primary.
        injected: The pass.

    Returns:
        The features on the injected panel.
    """
    touched = {e.host for e in injected.eras} | {s.donor for s in injected.switches}
    touched |= {s.recipient for s in injected.switches if s.recipient is not None}
    out = list(ctx_feats)
    for site in touched:
        out[site] = features.compute_site_features(power_site=injected.power[site])
    return out


def world_deltas(
    *, injected: inject.InjectedPass, panel: features.Panel
) -> tuple[np.ndarray, np.ndarray]:
    """Split the change an injected pass makes to the panel into eras and switching events.

    An era moves the target and the lags of its host. A switching event moves only the lags (and
    the neighbours' power), because the NRA target excludes it.

    Args:
        injected: The pass.
        panel: The real panel.

    Returns:
        The era change and the switching change, each (sites, half-hours) in capacity share, zero
        where nothing changed.
    """
    both = np.isfinite(injected.power) & np.isfinite(panel.power)
    delta = np.where(both, injected.power - panel.power, 0.0)
    era_rows = [e.host for e in injected.eras]
    era = np.zeros_like(delta)
    era[era_rows] = delta[era_rows]
    return era, delta - era


def world_signals(
    *,
    injected: list[inject.InjectedPass],
    ctx_feats: list[features.SiteFeatures],
    panel: features.Panel,
    results_of: list[dict[int, classify.SiteResult]],
    pass_feats: list[list[features.SiteFeatures]],
    n_days: int,
    placebo_seed: int,
    placebo_window: tuple[int, int],
    real_eras: list[OracleEra] | None = None,
    real_switches: list[OracleSwitch] | None = None,
) -> dict[str, Signal]:
    """Return every arm's signal for a world of several disjoint injected passes.

    The oracle holds the real record's events plus the injected ones, and the placebo permutes that
    union (eras and switching events separately).

    Args:
        injected: The passes (disjoint Primaries).
        ctx_feats: The real features (for the spread of non-host Primaries).
        panel: The real panel.
        results_of: Per pass, the `process_hosts` results by host.
        pass_feats: Per pass, the features of every Primary on the injected panel.
        n_days: Number of days of the grid.
        placebo_seed: Seed of the placebo's permutation.
        placebo_window: First and last allowed placebo onset day.
        real_eras: The real record's eras (None for none).
        real_switches: The real record's switching events (None for none).

    Returns:
        A signal per arm of `SIGNAL_ARMS`.
    """
    out: dict[str, Signal] = {}
    parts = [
        learned_signals(results=r, feats=f, n_days=n_days)
        for r, f in zip(results_of, pass_feats, strict=True)
    ]
    for arm in LEARNED_ARMS:
        out[arm] = merge_signals(parts=[p[arm] for p in parts])
    eras: list[OracleEra] = list(real_eras or [])
    switches: list[OracleSwitch] = list(real_switches or [])
    for inj, feats in zip(injected, pass_feats, strict=True):
        e, s = pass_oracle_events(injected=inj, feats=feats, panel=panel)
        eras += e
        switches += s
    n_sites = len(ctx_feats)
    out["oracle"] = oracle_signal(
        arm="oracle", eras=eras, switches=switches, n_sites=n_sites, n_days=n_days
    )
    eligible = [i for i, s in enumerate(panel.sites) if s not in inject.EXCLUDED_HOSTS]
    p_eras, p_switches = placebo_events(
        eras=eras, switches=switches, seed=placebo_seed, eligible=eligible, window=placebo_window
    )
    out["placebo"] = oracle_signal(
        arm="placebo", eras=p_eras, switches=p_switches, n_sites=n_sites, n_days=n_days
    )
    return out


# ---------------------------------------------------------------- the real record


def build_real_signals() -> dict[str, Signal]:
    """Build every arm's signal over the real record (CPU, run once, stage 0's frozen models).

    Returns:
        A signal per arm of `SIGNAL_ARMS`.
    """
    import e00709_stage0 as stage0

    ctx = stage0.build_context()
    models = classify.load_models(folder=stage0.MODELS_DIR)
    results = stage0.real_results(ctx=ctx, models=models)
    n_days = ctx.panel.n_days
    out = learned_signals(results=results, feats=ctx.feats, n_days=n_days)
    known = pl.read_parquet(features.DATA_DIR / "known_events.parquet")
    eras, switches = real_oracle_events(
        feats=ctx.feats, era_table=ctx.era_table, known_events=known
    )
    n_sites = len(ctx.feats)
    out["oracle"] = oracle_signal(
        arm="oracle", eras=eras, switches=switches, n_sites=n_sites, n_days=n_days
    )
    eligible = [i for i, s in enumerate(ctx.panel.sites) if s not in inject.EXCLUDED_HOSTS]
    p_eras, p_switches = placebo_events(
        eras=eras, switches=switches, seed=PLACEBO_SEED_BASE["real"], eligible=eligible,
        window=(int(ctx.ready_days[eligible].min()), n_days - 1),
    )  # fmt: skip
    out["placebo"] = oracle_signal(
        arm="placebo", eras=p_eras, switches=p_switches, n_sites=n_sites, n_days=n_days
    )
    return out


def main() -> None:
    """Write the real record's signal tables to `signals/<arm>.parquet` and print a summary."""
    for arm, signal in build_real_signals().items():
        path = write_signal(signal=signal, folder=SIGNALS_DIR)
        state = signal.columns["state"]
        print(
            f"{arm}: {path} causal={signal.causal} episodes={signal.episodes.height} "
            f"days with era state={int((state == STATE_ERA).sum())} "
            f"switch state={int((state == STATE_SWITCH).sum())} "
            f"unknown state={int((state == STATE_UNKNOWN).sum())}"
        )


if __name__ == "__main__":
    main()

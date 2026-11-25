# ruff: noqa: ANN003, E501, PLR0912, PERF401
"""Experiment 726: the follow share, the counted contrasts T1 and L1, the verdict, and the readings.

Usage:
    e00726_score.py counted [--draws 20000] [--plan five|two]   # T1 and L1 in both groups, and the verdict
    e00726_score.py report                                       # report-only readings, read at 95%
    e00726_score.py s6                                           # F4's reading S6 on experiment 714's second generator

Everything above `main` is plain numpy and polars, so the unit tests run it on synthetic frames.
The commands read stored forecast files only. Experiment 581's phase rule is reimplemented with
the slices of the design (`e00726_common.event_columns`), experiment 581's follow share
(`follow_q50` of `e00581_tails._aggregate`) is `e00726_common.follow_share`, and the S6 reading
imports `event_crps` and `pass_bootstrap_ci` from the pinned F4 checkout (commit `ac3e768e4`).

**Same-origin rule.** The frames of every network of a contrast are inner-joined on the three keys,
so a key that any frame lacks is dropped for every network. Five properties are asserted for each
(fold month, pass): no origin before the month start, every valid time before 2026-07-01, every
quantile finite, no repeated key, and no in-month origin with origin day 0 to 20 dropped.
"""

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import e00726_common as com
import numpy as np
import polars as pl

DELTA: Final = 0.05
"""The smallest difference that counts, in shares of the depth."""
LEVEL: Final = 98.75
"""The Bonferroni level for four counted intervals."""
SPLIT_LIMIT: Final = 0.025
"""A contrast whose seeds 0-1 and seeds 2-4 estimates differ by more than this reads `cannot tell`."""
DRAWS: Final = 20_000
REPORT_LEVEL: Final = 95.0
"""The report-only readings are read at 95% (design 5.4)."""
REPORT_DRAWS: Final = 5_000
PlanType = Literal["five", "two"]
Reading = Literal[
    "cannot tell",
    "M1 follows more",
    "M1 follows less",
    "small difference",
    "no detectable difference",
]
Verdict = Literal[
    "refuted",
    "supported",
    "partly supported",
    "small gain",
    "not supported",
    "inconclusive",
]
HOLDOUT_US: Final = int(datetime(2026, 7, 1, tzinfo=UTC).timestamp() * 1_000_000)


# ---------------------------------------------------------------- the networks and their files


@dataclass(frozen=True)
class NetSpec:
    """One network on one injected world: where its shifted and unshifted forecasts are stored."""

    name: str
    kind: Literal["cand", "msa"]
    world: str
    arm: str = ""

    def shifted_paths(self, *, month: str, r: int, seed: int, n_sets: int) -> list[Path]:
        """Return the files of the forecast on the shifted panel (one file per set for the host-only world)."""
        if self.kind == "msa":
            return [com.msa_output(world=self.world, arm=self.arm, month=month, seed=seed, r=r)]
        if self.world == "e581_host":
            return [
                com.candidate_output(world=self.world, month=month, r=r, seed=seed, k=k)
                for k in range(n_sets)
            ]
        return [com.candidate_output(world=self.world, month=month, r=r, seed=seed)]

    def base_path(self, *, month: str, seed: int) -> Path:
        """Return the stored forecast on the unshifted panel."""
        if self.kind == "msa":
            return com.msa_fit_file(arm=self.arm, month=month, seed=seed)
        return com.root_of_seed(seed=seed).stored_fit(month=month, seed=seed)


C_HOST: Final = NetSpec(name="C_host", kind="cand", world="e581_host")
C_ALL: Final = NetSpec(name="C_all", kind="cand", world="e581_all")
M1: Final = NetSpec(name="M1", kind="msa", world="all", arm="M1")
K_NET: Final = NetSpec(name="K", kind="msa", world="all", arm="K")
E_NET: Final = NetSpec(name="E", kind="msa", world="all", arm="E")
NETS: Final = {n.name: n for n in (C_HOST, C_ALL, M1, K_NET, E_NET)}


# ---------------------------------------------------------------- one (month, pass): rows to follow shares


def restrict_rows(
    *, frame: pl.DataFrame, month: str, lead_lo: float, lead_hi: float
) -> pl.DataFrame:
    """Keep origin days 0 to 20 and lead hours in [`lead_lo`, `lead_hi`)."""
    ms = com.month_start_us(month=month)
    origin = pl.col("power_fcst_init_time").dt.epoch("us")
    lead = (pl.col("valid_time").dt.epoch("us") - origin) / 3_600_000_000
    day = (origin - ms) // com.DAY_US
    return frame.filter(
        (day >= 0) & (day <= com.MAX_ORIGIN_DAY) & (lead >= lead_lo) & (lead < lead_hi)
    )


def load_seed_mean(
    *,
    paths_per_seed: Sequence[Sequence[Path]],
    columns: Sequence[str],
    month: str,
    lead_lo: float,
    lead_hi: float,
) -> pl.DataFrame:
    """Return the seed-mean, sorted-across-levels frame of the files (one list of files per seed)."""
    frames = []
    for paths in paths_per_seed:
        parts = [
            restrict_rows(
                frame=com.read_forecast(path=p, columns=com.QCOLS),
                month=month,
                lead_lo=lead_lo,
                lead_hi=lead_hi,
            )
            for p in paths
        ]
        frames.append(pl.concat(parts))
    return com.seed_mean(frames=frames, columns=com.QCOLS).select(*com.KEYS, *columns)


def join_networks(
    *, per_net: Mapping[str, tuple[pl.DataFrame, pl.DataFrame]], levels: Sequence[str]
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Return the inner join of every network's shifted and base frames, and the keys each frame lost.

    Each value of `per_net` is (shifted, base), both holding the keys and the columns `levels`.
    The result holds the keys and the columns `<net>:<level>:s` and `<net>:<level>:b`.
    """
    joined: pl.DataFrame | None = None
    sizes = {}
    for name, (shifted, base) in per_net.items():
        both = shifted.rename({q: f"{name}:{q}:s" for q in levels}).join(
            base.rename({q: f"{name}:{q}:b" for q in levels}), on=list(com.KEYS), how="inner"
        )
        sizes[name] = both.height
        joined = both if joined is None else joined.join(both, on=list(com.KEYS), how="inner")
    assert joined is not None
    return joined, {name: n - joined.height for name, n in sizes.items()}


def assert_same_origin(
    *, joined: pl.DataFrame, expected_pairs: pl.DataFrame, month: str, levels: Sequence[str]
) -> None:
    """Assert the properties of the same-origin rule on a joined frame (design section 3)."""
    assert not joined.select(list(com.KEYS)).is_duplicated().any(), "a key is repeated"
    origin = joined["power_fcst_init_time"].dt.epoch("us")
    assert (origin >= com.month_start_us(month=month)).all(), (
        "an origin lies before the month start"
    )
    assert (joined["valid_time"].dt.epoch("us") < HOLDOUT_US).all(), (
        "a valid time is after 2026-06-30"
    )
    value_columns = [c for c in joined.columns if c not in com.KEYS]
    assert value_columns, "no quantile column"
    finite = joined.select(pl.all_horizontal(pl.col(c).is_finite() for c in value_columns))
    assert finite.to_series().all(), "a forecast quantile is not finite"
    kept = joined.select("time_series_id", "power_fcst_init_time").unique()
    lost = expected_pairs.join(kept, on=["time_series_id", "power_fcst_init_time"], how="anti")
    assert lost.height == 0, (
        f"{lost.height} in-month origins (days 0 to 20) were dropped in {month}"
    )
    assert levels


def expected_pairs_of(*, month: str, lead_lo: float, lead_hi: float) -> pl.DataFrame:
    """Return the (Primary, origin) pairs of origin days 0 to 20 in the Candidate's stored forecast of seed 0."""
    stored = restrict_rows(
        frame=com.read_forecast(
            path=com.root_of_seed(seed=0).stored_fit(month=month, seed=0), columns=("q50",)
        ),
        month=month,
        lead_lo=lead_lo,
        lead_hi=lead_hi,
    )
    return stored.select("time_series_id", "power_fcst_init_time").unique()


def follow_rows(
    *, joined: pl.DataFrame, nets: Sequence[str], levels: Sequence[str], month: str, r: int,
    primary_of_id: Mapping[int, int],
) -> pl.DataFrame:  # fmt: skip
    """Return `joined` with the event columns and the follow share `<net>|<level>` of every network."""
    rows = com.event_columns(frame=joined, month=month, r=r, primary_of_id=primary_of_id)
    sign = rows["sign"].to_numpy()
    depth = rows["depth"].to_numpy()
    follow = {
        f"{n}|{q}": com.follow_share(
            shifted=rows[f"{n}:{q}:s"].to_numpy().astype(np.float64),
            base=rows[f"{n}:{q}:b"].to_numpy().astype(np.float64),
            sign=sign,
            depth=depth,
        )
        for n in nets
        for q in levels
    }
    return rows.with_columns([pl.Series(k, v) for k, v in follow.items()])


@dataclass
class Collected:
    """Events and by-origin-day means of one run of the scorer, plus the dropped keys."""

    events: pl.DataFrame
    by_day: pl.DataFrame
    dropped: list[dict] = field(default_factory=list)


def collect(
    *,
    nets: Sequence[NetSpec],
    seeds: Sequence[int],
    months: Sequence[str],
    passes: Sequence[int],
    primary_of_id: Mapping[int, int],
    n_sets: int,
    levels: Sequence[str] = ("q50",),
    band: tuple[float, float] = com.LEAD_BAND_COUNTED,
    extra: Callable[[pl.DataFrame, str, int], pl.DataFrame] | None = None,
    extra_columns: Sequence[str] = (),
) -> Collected:
    """Return one row per event and slice, and per origin day, with every network's mean follow share.

    An event is one Primary in one pass of one fold month. The rows of an event in a slice are the
    rows whose origin the slice selects and whose lead lies in `band`.
    """
    event_parts, day_parts, dropped = [], [], []
    names = [n.name for n in nets]
    for month in months:
        base = {
            n.name: load_seed_mean(
                paths_per_seed=[[n.base_path(month=month, seed=s)] for s in seeds],
                columns=levels, month=month, lead_lo=band[0], lead_hi=band[1],
            )
            for n in nets
        }  # fmt: skip
        expected = expected_pairs_of(month=month, lead_lo=band[0], lead_hi=band[1])
        for r in passes:
            per_net = {
                n.name: (
                    load_seed_mean(
                        paths_per_seed=[
                            n.shifted_paths(month=month, r=r, seed=s, n_sets=n_sets) for s in seeds
                        ],
                        columns=levels, month=month, lead_lo=band[0], lead_hi=band[1],
                    ),
                    base[n.name],
                )
                for n in nets
            }  # fmt: skip
            joined, lost = join_networks(per_net=per_net, levels=levels)
            assert_same_origin(joined=joined, expected_pairs=expected, month=month, levels=levels)
            if any(lost.values()):
                dropped.append(
                    {"month": month, "pass": r, **{f"lost_{k}": v for k, v in lost.items()}}
                )
            rows = follow_rows(
                joined=joined,
                nets=names,
                levels=levels,
                month=month,
                r=r,
                primary_of_id=primary_of_id,
            )
            if extra is not None:
                rows = extra(rows, month, r)
            follow_cols = [f"{n}|{q}" for n in names for q in levels] + list(extra_columns)
            for slice_name in com.SLICES:
                part = rows.filter(pl.col(slice_name))
                if part.is_empty():
                    continue
                event_parts.append(
                    part.group_by("primary").agg(
                        pl.first("length_days"), pl.first("depth"), pl.first("sign"),
                        pl.len().alias("n_rows"), *[pl.col(c).mean() for c in follow_cols],
                    ).with_columns(slice=pl.lit(slice_name), month=pl.lit(month), pass_r=pl.lit(r))
                )  # fmt: skip
            day_parts.append(
                rows.group_by("primary", "origin_day").agg(
                    pl.first("length_days"), pl.first("depth"), pl.first("sign"),
                    pl.len().alias("n_rows"), *[pl.col(c).mean() for c in follow_cols],
                ).with_columns(month=pl.lit(month), pass_r=pl.lit(r))
            )  # fmt: skip
    return Collected(
        events=pl.concat(event_parts) if event_parts else pl.DataFrame(),
        by_day=pl.concat(day_parts) if day_parts else pl.DataFrame(),
        dropped=dropped,
    )


# ---------------------------------------------------------------- contrasts, bootstrap, readings


def month_pass_arrays(
    *, rows: pl.DataFrame, value: str, months: Sequence[str], n_passes: int = com.N_PASSES
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sums, counts) of `value` per (fold month, pass): the event sums and the event counts."""
    sums = np.zeros((len(months), n_passes))
    counts = np.zeros((len(months), n_passes))
    agg = rows.group_by("month", "pass_r").agg(pl.col(value).sum().alias("s"), pl.len().alias("n"))
    for month, r, s, n in agg.iter_rows():
        if month in months:
            sums[list(months).index(month), r] = s
            counts[list(months).index(month), r] = n
    return sums, counts


def point_estimate(*, sums: np.ndarray, counts: np.ndarray) -> float:
    """Return the mean over events within a fold month, then over fold months with equal weight."""
    with np.errstate(invalid="ignore", divide="ignore"):
        month_mean = sums.sum(axis=1) / counts.sum(axis=1)
    return float(np.nanmean(month_mean)) if np.isfinite(month_mean).any() else float("nan")


def bootstrap_interval(
    *, sums: np.ndarray, counts: np.ndarray, draws: int = DRAWS, level: float = LEVEL, seed: int = 0
) -> tuple[float, float]:
    """Return the percentile interval that resamples the fold months, then the passes inside each month."""
    n_months, n_passes = sums.shape
    rng = np.random.default_rng(seed)
    month_pick = rng.integers(0, n_months, size=(draws, n_months))
    pass_pick = rng.integers(0, n_passes, size=(draws, n_months, n_passes))
    s = sums[month_pick[:, :, None], pass_pick].sum(axis=2)
    n = counts[month_pick[:, :, None], pass_pick].sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        month_mean = np.where(n > 0, s / n, np.nan)
    valid = np.isfinite(month_mean)
    n_valid = valid.sum(axis=1)
    estimate = np.where(
        n_valid > 0, np.where(valid, month_mean, 0.0).sum(axis=1) / np.maximum(n_valid, 1), np.nan
    )
    tail = (100.0 - level) / 2.0
    low, high = np.nanpercentile(estimate[np.isfinite(estimate)], [tail, 100.0 - tail])
    return float(low), float(high)


def group_primaries(*, labels: Sequence[str]) -> dict[str, list[int]]:
    """Return the Primary indices of each reporting group from the D labels."""
    return {
        name: [i for i, lab in enumerate(labels) if lab in members]
        for name, members in com.GROUPS.items()
    }


def contrast(
    *,
    events: pl.DataFrame,
    a: str,
    b: str,
    slice_name: str,
    primaries: Sequence[int],
    months: Sequence[str],
    level: str = "q50",
    max_depth: float | None = None,
    sign: int | None = None,
    length: int | None = None,
    draws: int | None = DRAWS,
    ci_level: float = LEVEL,
    n_passes: int = com.N_PASSES,
) -> dict:
    """Return the mean paired difference `a` minus `b` of the event follow shares, with its interval.

    `draws=None` gives the point estimate alone. The filters `max_depth` (events whose depth is at
    most this), `sign`, and `length` select events for the reports by depth, sign, and length.
    """
    sel = events.filter(pl.col("slice") == slice_name, pl.col("primary").is_in(list(primaries)))
    if max_depth is not None:
        sel = sel.filter(pl.col("depth") <= max_depth + 1e-9)
    if sign is not None:
        sel = sel.filter(pl.col("sign") == sign)
    if length is not None:
        sel = sel.filter(pl.col("length_days") == length)
    sel = sel.with_columns(diff=pl.col(f"{a}|{level}") - pl.col(f"{b}|{level}")).drop_nulls("diff")
    present = [m for m in months if m in set(sel["month"].to_list())]
    out = {"n_events": sel.height, "D": float("nan"), "low": float("nan"), "high": float("nan")}
    if not present:
        return out
    sums, counts = month_pass_arrays(rows=sel, value="diff", months=present, n_passes=n_passes)
    out["D"] = point_estimate(sums=sums, counts=counts)
    if draws:
        out["low"], out["high"] = bootstrap_interval(
            sums=sums, counts=counts, draws=draws, level=ci_level
        )
    return out


def level_of(*, events: pl.DataFrame, net: str, slice_name: str, primaries: Sequence[int], months: Sequence[str],
             level: str = "q50", **filters) -> float:  # fmt: skip
    """Return the absolute follow share of one network (the mean over events, months weighted equally)."""
    sel = events.filter(pl.col("slice") == slice_name, pl.col("primary").is_in(list(primaries)))
    if "max_depth" in filters:
        sel = sel.filter(pl.col("depth") <= filters["max_depth"] + 1e-9)
    if "sign" in filters:
        sel = sel.filter(pl.col("sign") == filters["sign"])
    sel = sel.with_columns(diff=pl.col(f"{net}|{level}")).drop_nulls("diff")
    present = [m for m in months if m in set(sel["month"].to_list())]
    if not present:
        return float("nan")
    sums, counts = month_pass_arrays(rows=sel, value="diff", months=present)
    return point_estimate(sums=sums, counts=counts)


def sign_of(x: float) -> int:
    """Return -1, 0, or +1."""
    return int(np.sign(x))


def read_cell(
    *,
    d: float,
    low: float,
    high: float,
    single: Sequence[float],
    split_difference: float | None,
    d_without_050: float,
    plan: PlanType = "five",
    delta: float = DELTA,
) -> Reading:
    """Return the reading of one counted cell by the table of design section 5.3.

    `single` holds D of each single seed (pairing seed `s` of `M1` with seed `s` of `C`). Under the
    five-seed plan a reading needs the sign to hold in at least 4 of 5 single seeds, and the
    split-seed difference of the contrast may not exceed `SPLIT_LIMIT`. Under the two-seed plan both
    single seeds must carry the sign of D, and no split-seed guard applies.
    """
    if not all(np.isfinite(v) for v in (d, low, high, d_without_050, *single)):
        return "cannot tell"
    if plan == "five" and (split_difference is None or not np.isfinite(split_difference)):
        return "cannot tell"
    n_seeds = len(single)
    need = 4 if plan == "five" else 2
    assert n_seeds == (5 if plan == "five" else 2), (plan, n_seeds)
    positive = sum(1 for v in single if v > 0)
    negative = sum(1 for v in single if v < 0)
    if low > 0 and d >= delta and positive >= need:
        reading: Reading = "M1 follows more"
    elif high < 0 and d <= -delta and negative >= need:
        reading = "M1 follows less"
    elif (low > 0 or high < 0) and (positive if d > 0 else negative) >= need:
        reading = "small difference"
    elif low > -delta and high < delta:
        reading = "no detectable difference"
    else:
        reading = "cannot tell"
    if plan == "five" and split_difference is not None and abs(split_difference) > SPLIT_LIMIT:
        return "cannot tell"
    if reading in {"M1 follows more", "M1 follows less", "small difference"} and (
        sign_of(d) != sign_of(d_without_050)
    ):
        return "cannot tell"
    return reading


def verdict_of(
    *, readings: Mapping[tuple[str, str], Reading], d_of: Mapping[tuple[str, str], float]
) -> Verdict:
    """Return the verdict by the table of design section 5.3.

    `readings` is keyed by (contrast, group), with contrast `T1` or `L1`.
    """
    groups = ("six_signed_mw", "eight_mva")
    cells = [(c, g) for c in ("T1", "L1") for g in groups]
    all_more = all(readings[c] == "M1 follows more" for c in cells)
    for contrast_name in ("T1", "L1"):
        if all(readings[(contrast_name, g)] == "M1 follows less" for g in groups):
            return "refuted"
    if all_more:
        return "supported"
    if any(readings[c] == "M1 follows more" for c in cells) and not any(
        readings[c] == "M1 follows less" for c in cells
    ):
        return "partly supported"
    if all(readings[c] == "small difference" and d_of[c] > 0 for c in cells):
        return "small gain"
    if all(readings[c] == "no detectable difference" for c in cells):
        return "not supported"
    return "inconclusive"


# ---------------------------------------------------------------- the counted run


def site_table() -> tuple[dict[int, int], list[str]]:
    """Return (primary index by time series id, D labels by Primary index)."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    sites = k578.sites().sort("primary")
    ids = sites["time_series_id"].to_list()
    return {int(t): i for i, t in enumerate(ids)}, sites["label"].to_list()


def print_table(*, title: str, rows: Sequence[Mapping[str, object]]) -> None:
    """Print `rows` as a tab-separated table under `title`."""
    print(f"\n## {title}", flush=True)
    if not rows:
        print("(empty)")
        return
    cols = list(rows[0])
    print("\t".join(cols))
    for row in rows:
        print("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values()))


def split_seed_differences(
    *, events_a: pl.DataFrame, events_b: pl.DataFrame, a: str, b: str, slice_name: str,
    primaries: Sequence[int], months: Sequence[str],
) -> dict[str, float]:  # fmt: skip
    """Return each network's follow share from seeds 0-1 minus from seeds 2-4, and the contrast's own difference."""
    kw: dict[str, Any] = {"slice_name": slice_name, "primaries": primaries, "months": months}
    la, lb = level_of(events=events_a, net=a, **kw), level_of(events=events_b, net=a, **kw)
    ma, mb = level_of(events=events_a, net=b, **kw), level_of(events=events_b, net=b, **kw)
    ca = contrast(events=events_a, a=a, b=b, draws=None, **kw)["D"]
    cb = contrast(events=events_b, a=a, b=b, draws=None, **kw)["D"]
    return {f"{a}": la - lb, f"{b}": ma - mb, "contrast": ca - cb}


def run_counted(*, draws: int, plan: PlanType) -> int:
    """Compute T1 and L1 in both groups, their readings, and the verdict; print and store them."""
    primary_of_id, labels = site_table()
    groups = group_primaries(labels=labels)
    n_sets = len(com.read_sets())
    seeds_all = com.SEEDS_ALL if plan == "five" else com.SEEDS_TWO
    common: dict[str, Any] = {"nets": (C_HOST, M1), "months": com.FOLD_MONTHS, "passes": range(com.N_PASSES),
              "primary_of_id": primary_of_id, "n_sets": n_sets}  # fmt: skip
    run_all = collect(seeds=seeds_all, **common)
    single = {s: collect(seeds=(s,), **common).events for s in seeds_all}
    split = (
        {
            "01": collect(seeds=(0, 1), **common).events,
            "234": collect(seeds=(2, 3, 4), **common).events,
        }
        if plan == "five"
        else None
    )
    print_table(title="Dropped keys per fold month and pass", rows=run_all.dropped)
    cells: dict[tuple[str, str], dict] = {}
    for tag, slice_name in (("T1", com.SLICE_AFTER), ("L1", com.SLICE_LASTING)):
        for group in ("six_signed_mw", "eight_mva"):
            kw: dict[str, Any] = {"events": run_all.events, "a": "M1", "b": "C_host", "slice_name": slice_name,
                  "primaries": groups[group], "months": com.FOLD_MONTHS}  # fmt: skip
            stats = contrast(draws=draws, **kw)
            singles = [
                contrast(events=single[s], a="M1", b="C_host", slice_name=slice_name,
                         primaries=groups[group], months=com.FOLD_MONTHS, draws=None)["D"]
                for s in seeds_all
            ]  # fmt: skip
            no050 = contrast(draws=None, max_depth=0.30, **kw)["D"]
            split_diff = None
            if split is not None:
                diffs = split_seed_differences(
                    events_a=split["01"], events_b=split["234"], a="M1", b="C_host",
                    slice_name=slice_name, primaries=groups[group], months=com.FOLD_MONTHS,
                )  # fmt: skip
                split_diff = diffs["contrast"]
                print(f"split-seed differences {tag} {group}: {diffs}", flush=True)
            reading = read_cell(
                d=stats["D"], low=stats["low"], high=stats["high"], single=singles,
                split_difference=split_diff, d_without_050=no050, plan=plan,
            )  # fmt: skip
            cells[(tag, group)] = {**stats, "single": singles, "no050_D": no050,
                                   "split_difference": split_diff, "reading": reading}  # fmt: skip
    verdict = verdict_of(
        readings={k: v["reading"] for k, v in cells.items()},
        d_of={k: v["D"] for k, v in cells.items()},
    )
    print_table(
        title=f"Counted cells (M1 minus C host-only, level {LEVEL}%, plan {plan})",
        rows=[{"contrast": k[0], "group": k[1], **{f: v[f] for f in ("D", "low", "high", "n_events", "reading")}}
              for k, v in cells.items()],
    )  # fmt: skip
    print(f"\nVERDICT: {verdict}. Experiment 713's known-answer gate was waived.", flush=True)
    com.write_json_atomic(
        obj={"plan": plan, "level": LEVEL, "draws": draws, "verdict": verdict,
             "cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()}, "dropped": run_all.dropped},
        path=com.data_root() / "scores" / "counted.json",
    )  # fmt: skip
    run_all.events.write_parquet(
        com.refuse_foreign_path(com.data_root() / "scores" / "events_counted.parquet")
    )
    return 0


# ---------------------------------------------------------------- the report-only readings


def run_report() -> int:
    """Print the report-only readings (design 5.4) from stored files; none changes the verdict."""
    primary_of_id, labels = site_table()
    groups = group_primaries(labels=labels)
    n_sets = len(com.read_sets())
    months = com.FOLD_MONTHS
    base: dict[str, Any] = {"primary_of_id": primary_of_id, "n_sets": n_sets}
    # reading 1, 4, 5, 6, 7: absolute follow shares of C host-only and M1 at five seeds
    five = collect(nets=(C_HOST, M1), seeds=com.SEEDS_ALL, months=months, passes=range(com.N_PASSES),
                   levels=("q05", "q50", "q95"), **base)  # fmt: skip
    rows = []
    for slice_name in com.SLICES:
        for group, prim in groups.items():
            rows.append(
                {"slice": slice_name, "group": group,
                 **{n: level_of(events=five.events, net=n, slice_name=slice_name, primaries=prim, months=months)
                    for n in ("C_host", "M1")}}
            )  # fmt: skip
    print_table(title="Reading 1: absolute follow shares at q50 (five seeds)", rows=rows)
    rows = []
    for slice_name in (com.SLICE_AFTER, com.SLICE_LASTING):
        for group in ("six_signed_mw", "eight_mva"):
            for sign in (1, -1):
                for depth in (0.15, 0.30, 0.50):
                    kw: dict[str, Any] = {"events": five.events, "a": "M1", "b": "C_host", "slice_name": slice_name,
                          "primaries": groups[group], "months": months, "draws": REPORT_DRAWS, "ci_level": REPORT_LEVEL, "sign": sign}  # fmt: skip
                    sel = five.events.filter(pl.col("depth").round(2) == depth)
                    kw["events"] = sel
                    rows.append({"slice": slice_name, "group": group, "sign": sign, "depth": depth,
                                 **{k: v for k, v in contrast(**kw).items() if k in {"D", "low", "high", "n_events"}}})  # fmt: skip
    print_table(title="Reading 3: M1 minus C by sign and depth", rows=rows)
    rows = []
    for level in ("q05", "q95"):
        for slice_name in (com.SLICE_AFTER, com.SLICE_LASTING):
            for group in ("six_signed_mw", "eight_mva"):
                rows.append({"level": level, "slice": slice_name, "group": group,
                             **{n: level_of(events=five.events, net=n, slice_name=slice_name,
                                            primaries=groups[group], months=months, level=level)
                                for n in ("C_host", "M1")}})  # fmt: skip
    print_table(title="Reading 7: follow ratio at q05 and q95", rows=rows)
    by_day = (
        five.by_day.group_by("origin_day", "length_days")
        .agg(pl.col("C_host|q50").mean(), pl.col("M1|q50").mean(), pl.len().alias("n_events"))
        .sort("length_days", "origin_day")
    )
    print_table(title="Reading 4: follow share by origin day and length", rows=by_day.to_dicts())
    rows = [
        {"month": m, "slice": s, **{n: level_of(events=five.events, net=n, slice_name=s,
                                                primaries=groups["six_signed_mw"] + groups["eight_mva"], months=[m])
                                    for n in ("C_host", "M1")}}
        for m in months for s in (com.SLICE_AFTER, com.SLICE_LASTING)
    ]  # fmt: skip
    print_table(title="Reading 5: by fold month, all 14 Primaries", rows=rows)
    rows = [
        {"label": labels[p], "slice": s, **{n: level_of(events=five.events, net=n, slice_name=s, primaries=[p], months=months)
                                          for n in ("C_host", "M1")}}
        for p in range(16) for s in (com.SLICE_AFTER, com.SLICE_LASTING)
    ]  # fmt: skip
    print_table(title="Reading 5: by Primary (D15 and D18 included)", rows=rows)
    rows = []
    for slice_name in (com.SLICE_AFTER, com.SLICE_LASTING):
        for group in ("six_signed_mw", "eight_mva"):
            rows.append({"slice": slice_name, "group": group, "left_out": "2025-11",
                         **{k: v for k, v in contrast(events=five.events, a="M1", b="C_host", slice_name=slice_name,
                                                      primaries=groups[group], months=[m for m in months if m != "2025-11"],
                                                      draws=REPORT_DRAWS, ci_level=REPORT_LEVEL).items() if k in {"D", "low", "high", "n_events"}}})  # fmt: skip
    print_table(title="Reading 5: counted cells with fold 2025-11 left out", rows=rows)
    # reading 6: other lead bands
    for name in ("120-240h", "24-240h"):
        band = com.LEAD_BANDS[name]
        other = collect(
            nets=(C_HOST, M1),
            seeds=com.SEEDS_ALL,
            months=months,
            passes=range(com.N_PASSES),
            band=band,
            **base,
        )
        rows = [{"band": name, "slice": s, "group": g,
                 **{k: v for k, v in contrast(events=other.events, a="M1", b="C_host", slice_name=s,
                                              primaries=groups[g], months=months, draws=REPORT_DRAWS, ci_level=REPORT_LEVEL).items() if k in {"D", "low", "high", "n_events"}}}
                for s in (com.SLICE_AFTER, com.SLICE_LASTING) for g in ("six_signed_mw", "eight_mva")]  # fmt: skip
        print_table(title=f"Reading 6: M1 minus C, lead band {name}", rows=rows)
    # reading 2 and 9: attribution on the four bank folds at two seeds, and the Candidate's all-shifted world
    bank_months = com.BANK_FOLDS
    two = collect(nets=(C_HOST, C_ALL, M1, K_NET, E_NET), seeds=com.SEEDS_TWO, months=bank_months,
                  passes=range(com.N_PASSES), **base)  # fmt: skip
    rows = []
    for a, b in (
        ("M1", "K"),
        ("E", "M1"),
        ("K", "C_host"),
        ("E", "C_host"),
        ("M1", "C_all"),
        ("C_all", "C_host"),
    ):
        for s in (com.SLICE_AFTER, com.SLICE_LASTING):
            for g in ("six_signed_mw", "eight_mva"):
                rows.append({"a": a, "b": b, "slice": s, "group": g,
                             **contrast(events=two.events, a=a, b=b, slice_name=s, primaries=groups[g],
                                        months=bank_months, draws=REPORT_DRAWS, ci_level=REPORT_LEVEL)})  # fmt: skip
    print_table(
        title="Readings 2 and 9: attribution and the Candidate's all-shifted world (4 bank folds, 2 seeds)",
        rows=rows,
    )
    five.events.write_parquet(
        com.refuse_foreign_path(com.data_root() / "scores" / "events_report.parquet")
    )
    return 0


# ---------------------------------------------------------------- reading S6 on the second generator


F4_CODE_HASH: Final = "e9809f648a885ae1"
"""The code hash that F4's `g2` folders carry in their logs (design Re-check)."""


def check_f4_pin(*, f4_dir: Path) -> None:
    """Raise unless `e00715_era.py` at the F4 branch head equals the file at the pinned commit."""
    import subprocess

    prefix = subprocess.run(
        ["git", "rev-parse", "--show-prefix"],
        cwd=f4_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rel = f"{prefix}e00715_era.py"
    pinned = subprocess.run(
        ["git", "show", f"{com.HEAD_F4}:{rel}"],
        cwd=f4_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for ref in ("origin/msa-f4-neighbour-rows", "msa-f4-neighbour-rows"):
        head = subprocess.run(
            ["git", "show", f"{ref}:{rel}"], cwd=f4_dir, capture_output=True, text=True, check=False
        )
        if head.returncode == 0:
            com.require(
                head.stdout == pinned, f"e00715_era.py at {ref} differs from {com.HEAD_F4}: re-pin"
            )
            print(f"F4 pin: e00715_era.py at {ref} equals the file at {com.HEAD_F4}", flush=True)
            return
    com.require(False, "no F4 branch head found to compare with the pinned commit")


def run_s6() -> int:
    """F4's reading S6 on experiment 714's second generator: each arm minus `C` after temporary moves.

    The Candidate's files of the world `g2` are joined with each arm's stored forecast on the keys
    both hold, and F4's `event_crps`, `crps_contrast`, and `pass_bootstrap_ci` score them.
    """
    record = com.assert_pinned_f4()
    print(f"F4 checkout: {record}", flush=True)
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00715_arms as f4_arms  # ty: ignore[unresolved-import]
    import e00715_era as f4  # ty: ignore[unresolved-import]
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    f4_root = f4_arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR) / "work" / "injected" / "g2"
    check_f4_pin(f4_dir=com.module_folder(name="e00715_era"))
    rows, skipped, keys = [], [], list(com.KEYS)
    for month in com.BANK_FOLDS:
        for seed in com.SEEDS_TWO:
            folders = sorted(f4_root.glob(f"*_{month}_s{seed}"))
            for folder in folders:
                log = (folder / "log.json").read_text()
                com.require(
                    F4_CODE_HASH in log, f"{folder} was not written by code hash {F4_CODE_HASH}"
                )
            by_pass: dict[int, dict[str, Path]] = {}
            for folder in folders:
                arm = folder.name.removesuffix(f"_{month}_s{seed}")
                for path in f4.pass_files(folder=folder):
                    by_pass.setdefault(int(path.stem.removeprefix("pass")), {})[arm] = path
            for r, arms_of in sorted(by_pass.items()):
                mine_path = com.candidate_output(world="g2", month=month, r=r, seed=seed)
                if not mine_path.exists():
                    skipped.append((month, seed, r))
                    continue
                frames = {
                    arm: com.normalise_frame(frame=pl.read_parquet(path))
                    for arm, path in arms_of.items()
                }
                cand = com.normalise_frame(
                    frame=pl.read_parquet(mine_path, columns=[*keys, *com.QCOLS])
                )
                common = cand.select(keys)
                for frame in frames.values():
                    common = common.join(frame.select(keys), on=keys, how="inner")
                first = next(iter(frames.values()))
                cand = cand.join(common, on=keys, how="inner").join(
                    first.select(*keys, "truth_share", "clean_share"), on=keys, how="inner"
                )
                any_path = next(iter(arms_of.values()))
                events = pl.read_parquet(any_path.with_name(any_path.stem + "_truth.parquet"))
                scored = {"C": cand} | {
                    arm: frame.join(common, on=keys, how="inner") for arm, frame in frames.items()
                }
                for label, frame in scored.items():
                    rows += [
                        {"arm": label, "month": month, "seed": seed, "pass_id": f"pass{r:02d}", **x}
                        for x in f4.event_crps(forecast=frame, events=events, ids=ids)
                    ]
    print(f"skipped {len(skipped)} passes with no Candidate file: {skipped[:5]}", flush=True)
    crps = pl.DataFrame(rows)
    out = []
    for arm in sorted(set(crps["arm"].to_list()) - {"C"}):
        two = crps.filter(pl.col("arm").is_in([arm, "C"]))
        for slice_name in f4.SLICES_SWITCH:
            out.append({"arm": arm, "slice": slice_name,
                        **f4.crps_contrast(crps=two, treatment=arm, reference="C", slice_name=slice_name)})  # fmt: skip
    print_table(
        title="Reading S6: arm minus C after temporary moves, second generator (points of capacity)",
        rows=out,
    )
    com.write_json_atomic(
        obj={"f4": record, "rows": out}, path=com.data_root() / "scores" / "s6.json"
    )
    return 0


def main() -> int:
    """Dispatch the command."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    counted = sub.add_parser("counted")
    counted.add_argument("--draws", type=int, default=DRAWS)
    counted.add_argument("--plan", choices=("five", "two"), default="five")
    sub.add_parser("report")
    sub.add_parser("s6")
    args = parser.parse_args()
    run = {"counted": lambda: run_counted(draws=args.draws, plan=args.plan) if args.command == "counted" else 0,
           "report": run_report, "s6": run_s6}[args.command]  # fmt: skip
    status = run()
    print(f"peak memory of this process: {com.peak_memory()}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())

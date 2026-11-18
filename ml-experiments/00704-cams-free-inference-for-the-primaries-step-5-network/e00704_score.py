# ruff: noqa: E501, PLR0912, PERF401
"""Experiments 704 and 703: score the CAMS-free arms against the stored control (CPU only).

Usage: python e00704_score.py [--experiment 704|703] [--months 2025-11 ...] [--arms control proxy drop]
[--fake-arm] [--exact] [--no-protocol]

`--experiment 704` scores experiment 677's step 5 (K1-type network, monthly refits, seeds 0 to 2),
and `--experiment 703` scores experiment 667's Candidate (seeds 0 and 1). The stored forecasts of
that experiment are the files `control`. The arms are `proxy` (columns rebuilt from the ENS analysis
proxy), `drop` (columns set to zero), and `replay` (the unchanged chain, refitted).

`--control replay` (the default) makes the replay refit the control, so C2 and C3 are an arm minus the
replay, and the stored forecasts become the reference row `stored`. The contrast N1, replay minus
stored, is the noise floor. `--control stored` makes the stored forecasts the control, and the replay
may then cover only some months. A proxy or drop arm whose fits are missing for a month is skipped
with a message. `--fake-arm` aliases every missing arm to the stored forecasts, so every contrast
must then be exactly 0.

Each seed's native forecast (09:00 UTC origin) becomes the contract's 12:00 UTC forecast through
`e00667_export.contract_rows`, and every row with a valid time on or after 2026-07-01 is dropped.
The truth, the manual heuristic, and the climatology are those of `e00677_score.py`. Writes
private tables under `$DATA_PATH_INTERNAL/ml-experiments/<experiment>/score/` and prints every
number that the page quotes.
"""

import argparse
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00521_stats as s521  # ty: ignore[unresolved-import]
import e00667_export as x667  # ty: ignore[unresolved-import]
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00668_slices_primaries as s668  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.bootstrap import BOOTSTRAP_SEED
from studies.sources import ML_EXPERIMENTS_DATA_DIR

HOLDOUT: Final = datetime(2026, 7, 1, tzinfo=UTC)
"""No valid time on or after this instant is scored."""
MONTHS: Final = ("2025-11", "2026-02", "2026-04", "2026-06")
LEVEL: Final = 100.0 - 5.0 / 3.0
"""The level of each family of three contrasts, in percent (98.33)."""
N_RESAMPLES: Final = 20_000
DAY: Final = 48
CUT_START: Final = datetime(2025, 3, 1, tzinfo=UTC)
SHORT_HISTORY: Final = ("D15", "D18")
TAILS: Final = ("q01", "q05", "q95", "q99")
SCENARIOS: Final = (
    "all rows",
    "outside every logged event and 7 days after",
    "inside a logged event",
    "7 days after a logged event ends",
)
BASELINES: Final = ("climatology", "heuristic")
METRICS: Final = (
    "crps",
    "mae",
    "band_cov",
    "width",
    "mmm",
    *[f"pin_{t}" for t in TAILS],
    *[f"cov_{t}" for t in TAILS],
)
"""Per-row measures: CRPS and MAE of the median (points of capacity), q05-q95 coverage (percent),
band width, the median minus the mean forecast, and each tail's pinball loss and share at or below."""
LEVELS_WEIGHT: Final = np.diff(np.concatenate([[0.0], (lib.TAUS[1:] + lib.TAUS[:-1]) / 2, [1.0]]))
"""Weights that integrate the 11-level quantile function into a mean (as experiment 699 does)."""
SIGNED_MW: Final = ("D5", "D9", "D10", "D11", "D14", "D16")


@dataclass(frozen=True)
class Spec:
    """Where one experiment keeps its control and arm fits, and its seeds."""

    key: str
    name: str
    seeds: tuple[int, ...]
    control_root: Path
    sub: str
    arm_sub: str
    fake: frozenset[str] = frozenset()

    @property
    def root(self) -> Path:
        """Return the experiment's private data folder."""
        return ML_EXPERIMENTS_DATA_DIR / self.name

    @property
    def arms_root(self) -> Path:
        """Return the folder holding the arms' fits; `chain667.sh` writes experiment 703's there too."""
        return (
            ML_EXPERIMENTS_DATA_DIR / "00704-cams-free-inference-for-the-primaries-step-5-network"
        )

    def fit_path(self, *, arm: str, month: str, seed: int) -> Path:
        """Return the native forecast file of one arm, month, and seed."""
        arm = "control" if arm in self.fake else arm
        base = self.control_root if arm == "control" else self.arms_root / "arms" / arm
        sub = self.sub if arm == "control" else self.arm_sub
        return (
            base
            / sub.replace("{m}", month)
            / "work"
            / "fits"
            / "P5"
            / f"{month}_d0_s{seed}.parquet"
        )


def spec_of(*, key: str) -> Spec:
    """Return the spec of experiment 704 or 703."""
    root = ML_EXPERIMENTS_DATA_DIR
    if key == "704":
        return Spec(
            key="704",
            name="00704-cams-free-inference-for-the-primaries-step-5-network",
            seeds=(0, 1, 2),
            control_root=next(root.glob("00677-*")),
            sub="arms/k1_5w_sp1",
            arm_sub="arms/k1_5w_sp1",
        )
    return Spec(
        key="703",
        name="00703-cams-free-inference-for-the-primaries-candidate",
        seeds=(0, 1),
        control_root=next(root.glob("00667-*")),
        sub="P7_D50_{m}",
        arm_sub="P7_D50_{m}",
    )


# ----------------------------------------------------------------------------- reading


def read_native(*, path: Path) -> pl.DataFrame:
    """Read one native forecast file."""
    return pl.read_parquet(path, columns=[*x667.KEYS, *x667.QCOLS]).sort(x667.KEYS)


def available(*, spec: Spec, arm: str, months: list[str]) -> list[str]:
    """Return the months for which every seed's file of the arm exists."""
    return [
        m
        for m in months
        if all(spec.fit_path(arm=arm, month=m, seed=s).exists() for s in spec.seeds)
    ]


def load_arm(*, spec: Spec, arm: str, months: list[str]) -> dict[str, pl.DataFrame]:
    """Return the arm's contract rows in MW per seed (`s0`...) and for the seed mean (`mean`).

    Every row has a valid time before the holdout. The seed mean averages the native quantiles over
    seeds and sorts them within the row, as `e00667_export.seed_mean` does.
    """
    per: dict[str, list[pl.DataFrame]] = {f"s{s}": [] for s in spec.seeds} | {"mean": []}
    for month in months:
        native = [read_native(path=spec.fit_path(arm=arm, month=month, seed=s)) for s in spec.seeds]
        for s, frame in zip(spec.seeds, native, strict=True):
            per[f"s{s}"].append(x667.contract_rows(mean=frame))
        per["mean"].append(x667.contract_rows(mean=x667.seed_mean(frames=native)))
    return {
        tag: pl.concat(parts).filter(pl.col("valid_time") < HOLDOUT) for tag, parts in per.items()
    }


# ----------------------------------------------------------------------------- scoring


def row_metrics(*, y: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    """Return the per-row measures of `METRICS`; `y` and `q` are in points of capacity."""
    pin = lib.pinball(y=y, q=q)
    out = {
        "crps": 2.0 * pin.mean(axis=1),
        "mae": np.abs(y - q[:, 5]),
        "band_cov": ((y >= q[:, 1]) & (y <= q[:, 9])).astype(np.float64) * 100.0,
        "width": q[:, 9] - q[:, 1],
        "mmm": q[:, 5] - q @ LEVELS_WEIGHT,
    }
    for t in TAILS:
        i = x667.QCOLS.index(t)
        out[f"pin_{t}"] = pin[:, i]
        out[f"cov_{t}"] = (y <= q[:, i]).astype(np.float64)
    return {k: v.astype(np.float32) for k, v in out.items()}


@dataclass
class Rows:
    """The scored rows' keys and the truth, in points of capacity."""

    keys: pl.DataFrame
    ids: list[int]
    row: np.ndarray
    vidx: np.ndarray
    oidx: np.ndarray
    lead_h: np.ndarray
    y: np.ndarray
    scale: np.ndarray
    month: np.ndarray
    label: np.ndarray


def site_map() -> pl.DataFrame:
    """Return the site map of the 16 Primaries."""
    return pl.read_csv(s521.DATA / "site_map.csv")


def build_rows(*, frame: pl.DataFrame, cut_short_history: bool) -> tuple[Rows, np.ndarray]:
    """Return the rows of a control frame with the truth, and the cleaned grid."""
    site = site_map()
    ids = sorted(frame["time_series_id"].unique().to_list())
    cap = (
        pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
        .filter(pl.col("time_series_id").is_in(ids))
        .sort("time_series_id")["capacity"]
        .to_numpy()
        .astype(np.float64)
    )
    cap_mw = (
        x667.capacity_mw()
        .filter(pl.col("time_series_id").is_in(ids))
        .sort("time_series_id")["capacity_mw"]
        .to_numpy()
        .astype(np.float64)
    )
    assert len(cap) == len(ids) == len(cap_mw), (len(cap), len(ids), len(cap_mw))
    worst = float(np.abs(cap / cap_mw - 1).max())
    print("max relative difference of the two capacity tables:", worst)
    assert worst < 1e-6, worst
    grid = lib.cleaned_grid(ids=ids)
    if cut_short_history:
        for label in SHORT_HISTORY:
            tid = int(site.filter(pl.col("label") == label)["time_series_id"][0])
            grid[ids.index(tid), : lib.grid_index(times=pl.Series([CUT_START]))[0]] = np.nan
    row = frame["time_series_id"].replace_strict({t: i for i, t in enumerate(ids)}).to_numpy()
    vidx = lib.grid_index(times=frame["valid_time"])
    oidx = lib.grid_index(times=frame["power_fcst_init_time"])
    if vidx.max() >= grid.shape[1]:
        grid = np.pad(
            grid, ((0, 0), (0, int(vidx.max()) + 1 - grid.shape[1])), constant_values=np.nan
        )
    label_of = dict(zip(site["time_series_id"].to_list(), site["label"].to_list(), strict=True))
    rows = Rows(
        keys=frame.select(x667.KEYS),
        ids=ids,
        row=row,
        vidx=vidx,
        oidx=oidx,
        lead_h=(vidx - oidx) / 2.0,
        y=grid[row, vidx].copy(),
        scale=100.0 / cap[row],
        month=frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy(),
        label=np.array([label_of[ids[r]] for r in row], dtype=object),
    )
    rows.y = rows.y * rows.scale
    return rows, grid


def baseline_q(*, rows: Rows, grid: np.ndarray) -> dict[str, np.ndarray]:
    """Return the manual heuristic's and the climatology's quantiles in points of capacity.

    The climatology trains on the power before the first day of each row's origin month, which is
    the refit cut-off of a monthly refit.
    """
    starts = pl.Series([datetime(int(m[:4]), int(m[5:]), 1, tzinfo=UTC) for m in rows.month])
    cutoff = lib.grid_index(times=starts)
    heur = lib.manual_heuristic(grid=grid, row=rows.row, vidx=rows.vidx, lead_h=rows.lead_h)
    clim = lib.climatology(grid=grid, row=rows.row, vidx=rows.vidx, cutoff_idx=cutoff)
    return {
        "heuristic": heur * rows.scale[:, None],
        "climatology": clim * rows.scale[:, None],
    }


def metrics_of_frame(*, frame: pl.DataFrame, rows: Rows) -> dict[str, np.ndarray]:
    """Score one arm's MW frame, which must hold exactly the rows' keys in the same order."""
    assert frame.select(x667.KEYS).equals(rows.keys), "the arm's rows differ from the control's"
    q = frame.select(x667.QCOLS).to_numpy().astype(np.float64) * rows.scale[:, None]
    return row_metrics(y=rows.y, q=q)


# ----------------------------------------------------------------------------- bootstrap


def month_bootstrap(
    *, sums: np.ndarray, counts: np.ndarray, level: float, n_resamples: int = N_RESAMPLES
) -> tuple[float, float]:
    """Bound a mean by resampling whole months and a seed, on the stream of `studies.bootstrap`.

    `sums` holds each seed's sum of the per-row value in each month, shape (seeds, months). The
    draw order matches `bootstrap_difference_at_level`, whose resamples each concatenate the drawn
    months' rows, so the bounds agree with it up to floating-point summation order. Months without
    a row are left out, as the row-level function never sees them.
    """
    keep = counts > 0
    sums, counts = sums[:, keep], counts[keep]
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    n_months = counts.shape[0]
    out = np.empty(n_resamples)
    for i in range(n_resamples):
        seed_index = generator.integers(0, sums.shape[0])
        drawn = generator.integers(0, n_months, size=n_months)
        out[i] = sums[seed_index, drawn].sum() / counts[drawn].sum()
    tail = (100.0 - level) / 2.0
    return float(np.percentile(out, tail)), float(np.percentile(out, 100.0 - tail))


@dataclass
class Store:
    """Per-row measures of every system and tag over the scored rows."""

    rows: Rows
    months: list[str]
    seeds: tuple[int, ...]
    data: dict[tuple[str, str], dict[str, np.ndarray]]
    covered: dict[str, list[str]]
    month_idx: np.ndarray

    def value(self, *, system: str, tag: str, metric: str) -> np.ndarray:
        """Return the per-row values of one system, tag, and metric."""
        return self.data[(system, tag)][metric]

    def mean(self, *, system: str, tag: str, metric: str, mask: np.ndarray) -> float:
        """Return the mean of a measure on the masked rows."""
        v = self.value(system=system, tag=tag, metric=metric)[mask].astype(np.float64)
        return float(v.mean()) if v.size else float("nan")

    def seed_mean_of_means(self, *, system: str, metric: str, mask: np.ndarray) -> float:
        """Return the mean over seeds of the per-seed mean."""
        return float(
            np.mean(
                [
                    self.mean(system=system, tag=f"s{s}", metric=metric, mask=mask)
                    for s in self.seeds
                ]
            )
        )

    def month_sums(
        self, *, a: str, b: str | None, metric: str, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return each seed's sum over each month of `a - b` (or `a`), and the row count per month."""
        n = len(self.months)
        counts = np.bincount(self.month_idx[mask], minlength=n).astype(np.float64)
        sums = np.zeros((len(self.seeds), n))
        for j, s in enumerate(self.seeds):
            v = self.value(system=a, tag=f"s{s}", metric=metric).astype(np.float64)
            if b is not None:
                v = v - self.value(system=b, tag=f"s{s}", metric=metric).astype(np.float64)
            sums[j] = np.bincount(self.month_idx[mask], weights=v[mask], minlength=n)
        return sums, counts

    def contrast(self, *, a: str, b: str, metric: str, mask: np.ndarray) -> dict[str, float]:
        """Return the pooled mean of `a - b` over seeds and rows, with the month-block interval."""
        sums, counts = self.month_sums(a=a, b=b, metric=metric, mask=mask)
        if counts.sum() == 0:
            return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "rows": 0}
        lo, hi = month_bootstrap(sums=sums, counts=counts, level=LEVEL)
        return {
            "point": float(sums.sum() / (counts.sum() * len(self.seeds))),
            "lo": lo,
            "hi": hi,
            "rows": int(counts.sum()),
        }


def verify_bootstrap(*, store: Store) -> None:
    """Check `month_bootstrap` against `bootstrap_difference_at_level` on a thinned set of rows."""
    from studies.bootstrap import bootstrap_difference_at_level
    from studies.experiment_store import losses_for_bootstrap

    step = 37
    pick = np.arange(0, len(store.rows.row), step)
    base = store.rows.keys[pick.tolist()]
    arms: dict[str, pl.DataFrame] = {}
    for name, system in (("a", "control"), ("b", BASELINES[1])):
        parts = []
        for s in store.seeds:
            v = store.value(system=system, tag=f"s{s}", metric="crps")[pick]
            parts.append(base.with_columns(seed=pl.lit(s, pl.Int64), crps=pl.Series(v)))
        arms[name] = pl.concat(parts)
    losses = losses_for_bootstrap(arm_losses=arms)
    want = bootstrap_difference_at_level(
        losses=losses, treatment="a", reference="b", metric="crps", level=LEVEL, n_resamples=2000
    )
    mask = np.zeros(len(store.rows.row), dtype=bool)
    mask[pick] = True
    sums, counts = store.month_sums(a="control", b=BASELINES[1], metric="crps", mask=mask)
    got = month_bootstrap(sums=sums, counts=counts, level=LEVEL, n_resamples=2000)
    print(f"bootstrap check on {len(pick):,} rows: official {want}, month totals {got}")
    assert np.allclose(want, got, atol=1e-9), (want, got)


def exact_contrasts(
    *, store: Store, pairs: dict[str, tuple[str, str]]
) -> dict[tuple[str, str], tuple[float, float]]:
    """Run the pooled contrasts through `bootstrap_difference_at_level` on every row (slow).

    Asserts that each bound equals the month-totals bootstrap's bound on the all-rows scope within
    1e-6, and returns the official bounds keyed by contrast name and metric.
    """
    from studies.bootstrap import bootstrap_difference_at_level
    from studies.experiment_store import losses_for_bootstrap

    found: dict[tuple[str, str], tuple[float, float]] = {}
    everything = np.ones(len(store.rows.row), dtype=bool)
    for metric in ("crps", "mae"):
        systems = sorted({s for pair in pairs.values() for s in pair})
        arms = {
            system: pl.concat(
                [
                    store.rows.keys.with_columns(
                        seed=pl.lit(s, pl.Int64),
                        **{
                            metric: pl.Series(
                                store.value(system=system, tag=f"s{s}", metric=metric)
                            )
                        },
                    )
                    for s in store.seeds
                ]
            )
            for system in systems
        }
        losses = losses_for_bootstrap(arm_losses=arms)
        for name, (a, b) in pairs.items():
            lo, hi = bootstrap_difference_at_level(
                losses=losses,
                treatment=a,
                reference=b,
                metric=metric,
                level=LEVEL,
                n_resamples=N_RESAMPLES,
            )
            mine = store.contrast(a=a, b=b, metric=metric, mask=everything)
            print(
                f"exact {name} {metric}: [{lo:+.5f}, {hi:+.5f}] (official function, 20,000 draws);"
                f" month-totals bootstrap [{mine['lo']:+.5f}, {mine['hi']:+.5f}]"
            )
            assert abs(lo - mine["lo"]) < 1e-6, (name, metric, "lower", lo, mine["lo"])
            assert abs(hi - mine["hi"]) < 1e-6, (name, metric, "upper", hi, mine["hi"])
            found[(name, metric)] = (float(lo), float(hi))
    return found


# ----------------------------------------------------------------------------- events and masks


def events_all(*, site: pl.DataFrame) -> pl.DataFrame:
    """Return the logged events of experiment 653 that overlap the scored period, plus the transfer."""
    table = pl.read_parquet(next(s668.R.glob("00653-*/real_events_masked.parquet"))).filter(
        pl.col("start") >= s668.SCORED_FROM_653
    )
    if table["primary"].dtype == pl.String:
        label = pl.col("primary")
    else:
        label = pl.col("primary").replace_strict(
            dict(zip(site["primary"].to_list(), site["label"].to_list(), strict=True))
        )
    table = table.with_columns(
        label=label, start=pl.col("start") + s668.OFFSET_653, end=pl.col("end") + s668.OFFSET_653
    )
    first, last = s668.idx_of(datetime(2025, 11, 1, tzinfo=UTC)), s668.idx_of(HOLDOUT)
    logged = table.filter((pl.col("end") > first) & (pl.col("start") < last)).select(
        "label", "start", "end"
    )
    verified = pl.DataFrame(
        [
            {"label": k, "start": s668.idx_of(a), "end": s668.idx_of(b)}
            for k, (a, b) in s668.VERIFIED.items()
        ]
    )
    return pl.concat([logged, verified.cast(logged.schema)]).unique().sort("label", "start")


def scenario_masks(*, rows: Rows, events: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the four load-shift scenario masks over the rows."""
    frame = pl.DataFrame({"label": rows.label.astype(str), "valid_idx": rows.vidx})
    parts = s668.masks(rows=frame, events=events)
    inside, after = parts["inside"], parts["7 days after"]
    return {
        SCENARIOS[0]: np.ones(len(rows.row), dtype=bool),
        SCENARIOS[1]: ~inside & ~after,
        SCENARIOS[2]: inside,
        SCENARIOS[3]: after,
    }


def event_counts(*, rows: Rows, events: pl.DataFrame) -> pl.DataFrame:
    """Return, per logged event, the scored rows inside it and in the 7 days after it."""
    out = []
    labels = rows.label.astype(str)
    for e in events.iter_rows(named=True):
        own = labels == e["label"]
        v = rows.vidx
        inside = int((own & (v >= e["start"]) & (v < e["end"])).sum())
        after = int((own & (v >= e["end"]) & (v < e["end"] + s668.AFTER)).sum())
        out.append({**e, "rows_inside": inside, "rows_after": after})
    return pl.DataFrame(out)


def cold_night_mask(*, rows: Rows) -> np.ndarray:
    """Return the cold-winter-night rows, as `studies.eval_protocol.slices` defines them.

    The sun is more than 5 degrees below the horizon and the ERA5 temperature is 3 degrees Celsius or
    below. The context comes from the feature cache that experiments 668 and 677 use.
    """
    import e00377_adapter as adapter  # ty: ignore[unresolved-import]
    import e00668_compare_primaries as cp  # ty: ignore[unresolved-import]
    from studies.dp_features import default_input_paths, load_or_build_feature_cache
    from studies.eval_protocol import metrics, slices

    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=cp.CACHE_DIR)
    _, context = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=metrics.HOLDOUT_START,
    )
    keyed = rows.keys.select(pl.col("time_series_id").cast(pl.Int64), "valid_time").join(
        context.select(
            pl.col("time_series_id").cast(pl.Int64), "valid_time", "temperature_c", "elevation_deg"
        ),
        on=["time_series_id", "valid_time"],
        how="left",
        maintain_order="left",
    )
    assert keyed.height == rows.keys.height, (keyed.height, rows.keys.height)
    elevation = np.nan_to_num(keyed["elevation_deg"].to_numpy().astype(np.float64), nan=90.0)
    temperature = np.nan_to_num(keyed["temperature_c"].to_numpy().astype(np.float64), nan=np.inf)
    return (elevation < slices.NIGHT_ELEVATION_DEG) & (
        temperature <= slices.COLD_NIGHT_MAX_TEMPERATURE_C
    )


def curtailment_costs(
    *, spec: Spec, frames: dict[str, pl.DataFrame], rows: Rows, ok: np.ndarray
) -> pl.DataFrame:
    """Return the avoidable curtailment cost per system and signed-MW Primary (headline fractions).

    `frames` maps a system name to its MW frame over all rows. Only rows in `ok` and in the signed-MW
    group are kept.
    """
    import e00377_adapter as adapter  # ty: ignore[unresolved-import]
    import e00668_compare_primaries as cp  # ty: ignore[unresolved-import]
    from studies.dp_features import default_input_paths, load_or_build_feature_cache
    from studies.eval_protocol import metrics

    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=cp.CACHE_DIR)
    actuals, _ = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=metrics.HOLDOUT_START,
    )
    signed = series.filter(pl.col("label").is_in(list(SIGNED_MW)))
    keep = ok & np.isin(rows.label.astype(str), list(SIGNED_MW))
    inputs = {}
    for name, frame in frames.items():
        part = frame.filter(pl.Series(keep)).with_columns(pl.col("time_series_id").cast(pl.Int64))
        inputs[name] = part.join(
            actuals.with_columns(pl.col("time_series_id").cast(pl.Int64)),
            on=["time_series_id", "valid_time"],
        )
    out = []
    for fraction in (0.5, 0.9):
        cost = metrics.avoidable_curtailment_cost(
            forecasts=inputs,
            series=signed,
            flows=actuals.rename({"actual": "flow_mw"}),
            scored_start=datetime(2025, 7, 1, tzinfo=UTC),
            fraction=fraction,
        )
        out.append(cost.with_columns(limit_fraction=pl.lit(fraction)))
    return pl.concat(out)


# ----------------------------------------------------------------------------- checks


def check_equality(*, spec: Spec, months: list[str]) -> None:
    """Check that this scorer's conversion of the control reproduces the stored export exactly."""
    if spec.key == "703":
        for month in months:
            native = [
                read_native(path=spec.fit_path(arm="control", month=month, seed=s))
                for s in spec.seeds
            ]
            mine = x667.contract_rows(mean=x667.seed_mean(frames=native))
            stored = pl.read_parquet(spec.control_root / "exports" / f"NRA_{month}.parquet")
            ok = mine.sort(x667.KEYS).equals(stored.sort(x667.KEYS).select(mine.columns))
            print(f"export equality, experiment 667 NRA_{month}.parquet, seed mean, exact: {ok}")
            assert ok, month
        return
    stored = pl.read_parquet(spec.control_root / "backtests" / "step5w_with_injected_sp1.parquet")
    for month in months:
        for s in spec.seeds:
            native = read_native(path=spec.fit_path(arm="control", month=month, seed=s))
            mine = x667.contract_rows(mean=native)
            theirs = stored.filter(
                (pl.col("seed") == s)
                & (pl.col("power_fcst_init_time").dt.strftime("%Y-%m") == month)
            ).select(mine.columns)
            ok = mine.sort(x667.KEYS).equals(theirs.sort(x667.KEYS))
            print(f"export equality, experiment 677 step 5w span 1, {month} seed {s}, exact: {ok}")
            assert ok, (month, s)
    import e00677_score as s677  # ty: ignore[unresolved-import]

    mean = s677.seed_mean(name="step5w_with_injected_sp1", seeds=spec.seeds)
    mean = mean.filter(pl.col("power_fcst_init_time").dt.strftime("%Y-%m").is_in(months))
    mine = load_arm(spec=spec, arm="control", months=months)["mean"]
    mine = mine.with_columns(pl.col("time_series_id").cast(mean.schema["time_series_id"]))
    joined = mine.join(mean, on=x667.KEYS, suffix="_old")
    worst = max(
        float((joined[q] - joined[f"{q}_old"]).abs().max())
        for q in x667.QCOLS  # ty: ignore[invalid-argument-type]
    )
    print(
        f"seed-mean rows: this scorer {mine.height:,}, e00677 seed_mean {joined.height:,} in the"
        f" months; largest absolute difference {worst:.3g} MW (float32 rounding of the mean)"
    )


def truth_checks(*, rows: Rows, ok_truth: np.ndarray, months: list[str]) -> pl.DataFrame:
    """Compare the truth with the backtest's ground-truth file and count the dropped half-hours."""
    gt = pl.read_parquet(lib.FOLDER / "ground_truth_primaries.parquet").select(
        pl.col("time_series_id").cast(pl.Int64), "valid_time", "power", "drop_reason"
    )
    keyed = rows.keys.select(pl.col("time_series_id").cast(pl.Int64), "valid_time").with_columns(
        origin_month=pl.Series(rows.month), y=pl.Series(rows.y / rows.scale, nan_to_null=True)
    )
    joined = keyed.join(gt, on=["time_series_id", "valid_time"], how="left")
    both = joined.filter(pl.col("y").is_not_null() & pl.col("power").is_not_null())
    diff = float((both["y"] - both["power"]).abs().max())  # ty: ignore[invalid-argument-type]
    print(f"largest |truth - ground_truth_primaries.power| on rows with both: {diff:.3g}")
    clash = joined.filter(pl.col("y").is_not_null() & pl.col("drop_reason").is_not_null()).height
    print(f"rows that the file flags as dropped but the cleaned grid keeps: {clash}")
    unique = joined.unique(["time_series_id", "valid_time", "origin_month"])
    table = (
        unique.group_by("origin_month")
        .agg(
            half_hours=pl.len(),
            no_truth=pl.col("y").is_null().sum(),
            substation_zero=(
                pl.col("y").is_null() & (pl.col("drop_reason") == "substation_zero")
            ).sum(),
            absent_from_file=(pl.col("y").is_null() & pl.col("drop_reason").is_null()).sum(),
            other_reason=(
                pl.col("y").is_null()
                & pl.col("drop_reason").is_not_null()
                & (pl.col("drop_reason") != "substation_zero")
            ).sum(),
        )
        .sort("origin_month")
    )
    rows_missing = (
        pl.DataFrame({"origin_month": rows.month, "no_truth": ~ok_truth})
        .group_by("origin_month")
        .agg(rows=pl.len(), rows_without_truth=pl.col("no_truth").sum())
    )
    out = table.join(rows_missing, on="origin_month").filter(pl.col("origin_month").is_in(months))
    print("dropped half-hours per fold (unique series and valid time per fold; rows repeat them):")
    print(out)
    return out


# ----------------------------------------------------------------------------- main


def band_masks(*, rows: Rows) -> dict[str, np.ndarray]:
    """Return the lead-band masks and the all-rows mask."""
    out = {"all": np.ones(len(rows.row), dtype=bool)}
    for name, (lo, hi) in lib.LEAD_BANDS.items():
        out[name] = (rows.lead_h >= lo) & (rows.lead_h < hi)
    return out


def main() -> None:  # noqa: PLR0915
    """Score the arms, print every number the page quotes, and write the private tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="704", choices=("704", "703"))
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument(
        "--control",
        choices=("replay", "stored"),
        default="replay",
        help="the control: the refit of the unchanged chain (replay) or the stored forecasts",
    )
    parser.add_argument("--arms", nargs="+", default=["control", "proxy", "drop", "replay"])
    parser.add_argument("--fake-arm", action="store_true")
    parser.add_argument("--exact", action="store_true", help="also run the official bootstrap")
    parser.add_argument("--no-protocol", action="store_true", help="skip cold nights and cost")
    parser.add_argument("--no-cut", action="store_true", help="do not cut D15 and D18 in the lags")
    args = parser.parse_args()
    spec = spec_of(key=args.experiment)
    months = sorted(args.months)
    out_dir = spec.root / "score"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"experiment {spec.key}; months {months}; seeds {list(spec.seeds)}; output {out_dir}")

    check_equality(spec=spec, months=months)
    control = load_arm(spec=spec, arm="control", months=months)
    arms: dict[str, dict[str, pl.DataFrame]] = {"control": control}
    covered = {"control": months}
    for arm in ("proxy", "drop", "replay"):
        if arm not in args.arms:
            continue
        have = available(spec=spec, arm=arm, months=months)
        partial_ok = arm == "replay" and args.control == "stored"
        if (have != months and not partial_ok) or not have:
            if args.fake_arm:
                print(f"FAKE: {arm} aliased to the control")
                arms[arm], covered[arm] = control, months
                spec = replace(spec, fake=spec.fake | {arm})
            else:
                print(f"SKIP arm {arm}: fits missing for {sorted(set(months) - set(have))}")
            continue
        arms[arm] = load_arm(spec=spec, arm=arm, months=have)
        covered[arm] = have
        print(f"arm {arm}: months {have}")

    if args.control == "replay":
        if "replay" not in arms or covered["replay"] != months:
            msg = f"--control replay needs the replay fits for every month; have {covered.get('replay')}"
            raise SystemExit(msg)
        rename = {"control": "stored", "replay": "control"}
        arms = {rename.get(k, k): v for k, v in arms.items()}
        covered = {rename.get(k, k): v for k, v in covered.items()}
        print("control = the replay refit; the stored forecasts are the reference row `stored`")
    else:
        print("control = the stored forecasts")
    rows, grid = build_rows(frame=control["s0"], cut_short_history=not args.no_cut)
    store_data: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for system, frames in arms.items():
        for tag, frame in frames.items():
            if covered[system] != months:
                keep = np.isin(rows.month, covered[system])
                part = metrics_of_frame(frame=frame, rows=_subset_rows(rows=rows, keep=keep))
                store_data[(system, tag)] = {k: _scatter(v, keep) for k, v in part.items()}
            else:
                store_data[(system, tag)] = metrics_of_frame(frame=frame, rows=rows)
    base = baseline_q(rows=rows, grid=grid)
    for name, q in base.items():
        met = row_metrics(y=rows.y, q=np.where(np.isfinite(q), q, 0.0))
        bad = ~np.isfinite(q).all(axis=1)
        for k in met:
            met[k][bad] = np.nan
        for tag in [f"s{s}" for s in spec.seeds] + ["mean"]:
            store_data[(name, tag)] = met

    ok = np.isfinite(rows.y)
    ok_truth = ok.copy()
    for name, q in base.items():
        print(
            f"rows dropped because the {name} has a NaN quantile (with a finite truth): "
            f"{int((ok & ~np.isfinite(q).all(axis=1)).sum()):,}"
        )
    for (system, _tag), met in store_data.items():
        if covered.get(system, months) == months:
            ok &= np.isfinite(met["crps"])
    print(
        f"rows {len(ok):,}; scored {int(ok.sum()):,}; without truth or a baseline {int((~ok).sum()):,}"
    )
    print(
        "QUOTE: the page's absolute scores are the tag `seed_average_of_scores` (mean over seeds of each"
        " seed's score, which the contrasts use); the tag `mean` is the seed-averaged forecast"
        " (experiment 667's published forecast), quoted only where stated."
    )
    drops = truth_checks(rows=rows, ok_truth=ok_truth, months=months)
    drops.write_csv(out_dir / "dropped_half_hours.csv")

    def keep(a: np.ndarray) -> np.ndarray:
        return a[ok]

    sub = Rows(
        keys=rows.keys.filter(pl.Series(ok)),
        ids=rows.ids,
        row=keep(rows.row),
        vidx=keep(rows.vidx),
        oidx=keep(rows.oidx),
        lead_h=keep(rows.lead_h),
        y=keep(rows.y),
        scale=keep(rows.scale),
        month=keep(rows.month),
        label=keep(rows.label),
    )
    data = {k: {m: keep(v) for m, v in met.items()} for k, met in store_data.items()}
    store = Store(
        rows=sub,
        months=months,
        seeds=spec.seeds,
        data=data,
        covered=covered,
        month_idx=np.searchsorted(months, sub.month),
    )
    systems = [
        s for s in ("control", "stored", "proxy", "drop", "replay", *BASELINES) if (s, "s0") in data
    ]
    run(
        store=store,
        spec=spec,
        systems=systems,
        out_dir=out_dir,
        args=args,
        frames=arms,
        ok=ok,
        rows=rows,
        base=base,
    )


def _subset_rows(*, rows: Rows, keep: np.ndarray) -> Rows:
    """Return the rows where `keep` holds."""
    return Rows(
        keys=rows.keys.filter(pl.Series(keep)),
        ids=rows.ids,
        row=rows.row[keep],
        vidx=rows.vidx[keep],
        oidx=rows.oidx[keep],
        lead_h=rows.lead_h[keep],
        y=rows.y[keep],
        scale=rows.scale[keep],
        month=rows.month[keep],
        label=rows.label[keep],
    )


def _scatter(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Return an array over all rows that holds `values` where `keep` and NaN elsewhere."""
    out = np.full(len(keep), np.nan, dtype=np.float32)
    out[keep] = values
    return out


def run(  # noqa: PLR0915
    *,
    store: Store,
    spec: Spec,
    systems: list[str],
    out_dir: Path,
    args: argparse.Namespace,
    frames: dict[str, dict[str, pl.DataFrame]],
    ok: np.ndarray,
    rows: Rows,
    base: dict[str, np.ndarray],
) -> None:
    """Build the tables, the contrasts, and the verdict from the scored rows."""
    sub = store.rows
    site = site_map()
    events = events_all(site=site)
    scen = scenario_masks(rows=sub, events=events)
    counts = event_counts(rows=sub, events=events)
    bands = band_masks(rows=sub)
    scopes: dict[str, dict[str, np.ndarray]] = {"band": bands, "scenario": scen}
    scopes["month"] = {m: sub.month == m for m in store.months}
    scopes["primary"] = {
        str(lab): sub.label.astype(str) == lab for lab in sorted(set(sub.label.astype(str)))
    }
    scopes["group"] = {
        g: np.isin(sub.label.astype(str), site.filter(pl.col("group") == g)["label"].to_list())
        for g in site["group"].unique().sort().to_list()
    }
    if not args.no_protocol:
        cold = cold_night_mask(rows=rows)[ok]
        scopes["night"] = {"cold winter nights": cold}
        print(
            f"cold winter night rows: {int(cold.sum()):,} ({100 * cold.mean():.2f}% of scored rows)"
        )

    print("\n=== events (logged table of experiment 653 plus the verified transfer)")
    print(f"events overlapping the scored period: {events.height}")
    print(counts)
    print(
        "events with scored rows inside:",
        int((counts["rows_inside"] > 0).sum()),
        "; with scored rows in the 7 days after:",
        int((counts["rows_after"] > 0).sum()),
    )
    for name, mask in scen.items():
        print(
            f"scenario {name}: {int(mask.sum()):,} rows, {len(np.unique(sub.oidx[mask] // DAY))} origin days"
        )

    # ---- absolute scores
    records = []
    for kind, group in scopes.items():
        for scope, mask in group.items():
            for system in systems:
                msk = mask
                if system in store.covered and store.covered[system] != store.months:
                    msk = mask & np.isfinite(store.value(system=system, tag="s0", metric="crps"))
                for metric in METRICS:
                    for tag in [*(f"s{s}" for s in store.seeds), "mean"]:
                        records.append(
                            {
                                "scope_kind": kind,
                                "scope": scope,
                                "system": system,
                                "tag": tag,
                                "metric": metric,
                                "value": store.mean(
                                    system=system, tag=tag, metric=metric, mask=msk
                                ),
                                "rows": int(msk.sum()),
                            }
                        )
                    records.append(
                        {
                            "scope_kind": kind,
                            "scope": scope,
                            "system": system,
                            "tag": "seed_average_of_scores",
                            "metric": metric,
                            "value": store.seed_mean_of_means(
                                system=system, metric=metric, mask=msk
                            ),
                            "rows": int(msk.sum()),
                        }
                    )
    table = pl.DataFrame(records)
    table.write_csv(out_dir / "scores_long.csv")

    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_float_precision(4)

    def show(*, kind: str, metric: str, tag: str, title: str) -> None:
        part = table.filter(
            (pl.col("scope_kind") == kind) & (pl.col("metric") == metric) & (pl.col("tag") == tag)
        )
        wide = part.pivot(on="system", index="scope", values="value", maintain_order=True)
        print(f"\n--- {title} ({tag})")
        print(wide)

    print("\n=== absolute scores, points of capacity")
    for metric, title in (("crps", "CRPS"), ("mae", "MAE of the median")):
        for tag in ("seed_average_of_scores", "mean"):
            show(kind="band", metric=metric, tag=tag, title=f"{title} by lead band, all rows")
    for tag in [f"s{s}" for s in store.seeds]:
        show(kind="band", metric="crps", tag=tag, title="CRPS by lead band")
        show(kind="band", metric="mae", tag=tag, title="MAE by lead band")
    for metric in ("crps", "mae", "band_cov", "width"):
        show(
            kind="month",
            metric=metric,
            tag="seed_average_of_scores",
            title=f"{metric} by origin month",
        )
        show(
            kind="scenario",
            metric=metric,
            tag="seed_average_of_scores",
            title=f"{metric} by scenario",
        )
        show(
            kind="group",
            metric=metric,
            tag="seed_average_of_scores",
            title=f"{metric} by metering group",
        )
        show(
            kind="primary",
            metric=metric,
            tag="seed_average_of_scores",
            title=f"{metric} per Primary",
        )
    show(
        kind="primary",
        metric="mmm",
        tag="seed_average_of_scores",
        title="median minus mean forecast per Primary",
    )
    for metric in ("band_cov", "width", "mmm"):
        show(
            kind="band", metric=metric, tag="seed_average_of_scores", title=f"{metric} by lead band"
        )
    for t in TAILS:
        for metric in (f"cov_{t}", f"pin_{t}"):
            show(
                kind="band",
                metric=metric,
                tag="seed_average_of_scores",
                title=f"{metric} by lead band",
            )
    if "night" in scopes:
        for metric in ("crps", "mae"):
            show(
                kind="night",
                metric=metric,
                tag="seed_average_of_scores",
                title=f"{metric} on cold winter nights",
            )

    # ---- contrasts
    pairs = {
        "C1 proxy minus drop": ("proxy", "drop"),
        "C2 proxy minus control": ("proxy", "control"),
        "C3 drop minus control": ("drop", "control"),
    }
    pairs["N1 replay minus stored (noise floor)"] = (
        ("control", "stored") if args.control == "replay" else ("replay", "control")
    )
    pairs = {k: v for k, v in pairs.items() if all((s, "s0") in store.data for s in v)}

    def adjusted(mask: np.ndarray, pair: tuple[str, str]) -> np.ndarray:
        """Return the mask without the rows that a partly covered system lacks."""
        for system in pair:
            if system in store.covered and store.covered[system] != store.months:
                mask = mask & np.isfinite(store.value(system=system, tag="s0", metric="crps"))
        return mask

    full_pairs = {k: v for k, v in pairs.items() if adjusted(bands["all"], v).all()}
    verify_bootstrap(store=store)
    results = []
    if pairs:
        contrast_scopes = {
            "all": {"all": bands["all"]},
            **{k: v for k, v in scopes.items() if k in ("band", "scenario", "night", "group")},
        }
        for name, (a, b) in pairs.items():
            for kind, group in contrast_scopes.items():
                for scope, mask in group.items():
                    for metric in ("crps", "mae"):
                        r = store.contrast(a=a, b=b, metric=metric, mask=adjusted(mask, (a, b)))
                        results.append(
                            {
                                "contrast": name,
                                "scope_kind": kind,
                                "scope": scope,
                                "metric": metric,
                                **r,
                            }
                        )
        con = pl.DataFrame(results)
        con.write_csv(out_dir / "contrasts.csv")
        if args.fake_arm and set(frames) >= {"proxy", "drop"}:
            ok_zero = bool(
                (con["point"].abs().max() == 0)
                and (con["lo"].abs().max() == 0)
                and (con["hi"].abs().max() == 0)
            )
            print(f"fake-arm check, every contrast exactly 0: {ok_zero}")
            assert ok_zero
        print(
            f"\n=== contrasts (treatment minus reference, points of capacity; {LEVEL:.2f}% month-block interval, {N_RESAMPLES:,} resamples; interim: {len(store.months)} months)"
        )
        print(con.with_columns(pl.col("point", "lo", "hi").round(4)))
        exact = exact_contrasts(store=store, pairs=full_pairs) if args.exact else None
        # per month and seed for every contrast
        mrec = []
        for name, (a, b) in pairs.items():
            for metric in ("crps", "mae"):
                sums, cnt = store.month_sums(
                    a=a, b=b, metric=metric, mask=adjusted(bands["all"], (a, b))
                )
                for j, s in enumerate(store.seeds):
                    for i, m in enumerate(store.months):
                        mrec.append(
                            {
                                "contrast": name,
                                "metric": metric,
                                "month": m,
                                "seed": s,
                                "difference": float(sums[j, i] / cnt[i]),
                                "rows": int(cnt[i]),
                            }
                        )
        mtab = pl.DataFrame(mrec)
        mtab.write_csv(out_dir / "contrasts_month_seed.csv")
        print("\n--- C1 by origin month and seed")
        print(
            mtab.filter(pl.col("contrast").str.starts_with("C1")).pivot(
                on="seed", index=["metric", "month"], values="difference"
            )
        )
        verdict = decide(store=store, con=con, mtab=mtab, exact=exact)
        (out_dir / "summary.json").write_text(json.dumps(verdict, indent=2))
    else:
        print("\nno contrasts: proxy and drop are both needed")

    # ---- replay
    replay_sys, stored_sys = (
        ("control", "stored") if args.control == "replay" else ("replay", "control")
    )
    if (replay_sys, "s0") in store.data and (stored_sys, "s0") in store.data:
        replay_check(spec=spec, store=store, replay_sys=replay_sys, stored_sys=stored_sys)

    # ---- curtailment cost
    if not args.no_protocol:
        named = {}
        for system in ("control", "stored", "proxy", "drop"):
            if system in frames:
                for tag, fr in frames[system].items():
                    named[f"{system} {tag}"] = fr
        template = frames["control"]["mean"]
        for name, q in base.items():
            mw = q / rows.scale[:, None]
            named[f"{name} (no seed)"] = template.with_columns(
                [
                    pl.Series(c, np.nan_to_num(mw[:, i], nan=0.0)).cast(pl.Float32)
                    for i, c in enumerate(x667.QCOLS)
                ]
            )
        cost = curtailment_costs(spec=spec, frames=named, rows=rows, ok=ok)
        cost.write_csv(out_dir / "curtailment_cost.csv")
        pooled = (
            cost.filter(pl.col("status") == "ok")
            .group_by("limit_fraction", "arm")
            .agg(cost_gbp=pl.col("avoidable_cost_gbp").sum(), nodes=pl.len())
            .sort("limit_fraction", "arm")
        )
        print(
            "\n=== avoidable curtailment cost, summed over signed-MW Primaries with a status of ok (GBP)"
        )
        print(pooled)
        pooled.write_csv(out_dir / "curtailment_cost_pooled.csv")


def replay_check(*, spec: Spec, store: Store, replay_sys: str, stored_sys: str) -> None:
    """Report the replay's largest difference to the control and its CRPS on the replayed months."""
    print(
        "\n=== replay of the unchanged chain against the stored control (native fraction of capacity)"
    )
    for month in store.covered[replay_sys]:
        worst = np.zeros(len(x667.QCOLS))
        for s in spec.seeds:
            a = read_native(path=spec.fit_path(arm="replay", month=month, seed=s))
            b = read_native(path=spec.fit_path(arm="control", month=month, seed=s))
            assert a.select(x667.KEYS).equals(b.select(x667.KEYS)), "replay keys differ"
            d = np.abs(
                a.select(x667.QCOLS).to_numpy().astype(np.float64)
                - b.select(x667.QCOLS).to_numpy().astype(np.float64)
            )
            worst = np.maximum(worst, d.max(axis=0))
        print(
            f"{month}: max abs difference per quantile {np.round(worst, 8).tolist()}; overall {worst.max():.3g}"
        )
    mask = np.isfinite(store.value(system=replay_sys, tag="s0", metric="crps"))
    for metric in ("crps", "mae"):
        r = store.seed_mean_of_means(system=replay_sys, metric=metric, mask=mask)
        c = store.seed_mean_of_means(system=stored_sys, metric=metric, mask=mask)
        print(f"replay {metric} {r:.5f}; control on the same rows {c:.5f}; difference {r - c:+.6f}")


def decide(
    *,
    store: Store,
    con: pl.DataFrame,
    mtab: pl.DataFrame,
    exact: dict[tuple[str, str], tuple[float, float]] | None,
) -> dict:
    """Apply the pre-registered decision rule to C1 and print the verdict.

    The rule: the C1 interval lies wholly below zero, the C1 point is at most -0.01, and every
    seed's C1 is below zero in every month. With `exact`, the interval is the official function's.
    """
    c1 = con.filter(
        pl.col("contrast").str.starts_with("C1")
        & (pl.col("scope") == "all")
        & (pl.col("metric") == "crps")
    ).row(0, named=True)
    if exact is not None:
        c1 = {
            **c1,
            "lo": exact[(c1["contrast"], "crps")][0],
            "hi": exact[(c1["contrast"], "crps")][1],
        }
    print(
        f"decision interval from the {'official function' if exact else 'month-totals bootstrap'}"
    )
    month = mtab.filter(pl.col("contrast").str.starts_with("C1") & (pl.col("metric") == "crps"))
    negative = bool((month["difference"] < 0).all())
    agree = bool(
        all(
            (g["difference"] < 0).all() or (g["difference"] > 0).all()
            for _, g in month.group_by("month")
        )
    )
    wholly_below = c1["hi"] < 0
    big = c1["point"] <= -0.01
    verdict_a = wholly_below and big and negative
    print("\n=== decision rule (C1: proxy minus drop, CRPS, all rows)")
    print(f"point {c1['point']:+.5f}; interval [{c1['lo']:+.5f}, {c1['hi']:+.5f}] at {LEVEL:.2f}%")
    print(f"interval wholly below zero: {wholly_below}; point at most -0.01: {big}")
    print(
        f"every seed negative in every month: {negative}; seeds agree in sign in every month: {agree}"
    )
    print(f"VERDICT: {'provisional arm A (proxy)' if verdict_a else 'arm B (drop)'}")
    return {
        "c1_point": c1["point"],
        "c1_lo": c1["lo"],
        "c1_hi": c1["hi"],
        "wholly_below_zero": wholly_below,
        "point_at_most_minus_0.01": big,
        "all_seeds_negative_every_month": negative,
        "seeds_agree_in_sign_every_month": agree,
        "verdict": "A" if verdict_a else "B",
    }


if __name__ == "__main__":
    main()

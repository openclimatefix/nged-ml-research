# ruff: noqa: E501, PLR0911, PT018
r"""Score experiment 695: add the engine back, compute contrasts C1 to C5, and apply the verdict table.

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00695-*/e00695_score.py [--resamples N] [--workers N]

Reads each arm's `runs/<ARM>/arms/k1_5w_sp1/work/fits/P5/<month>_d0_s<seed>.parquet` (quantiles as
fractions of effective capacity, at the network's 09:00 UTC origin) and `states/<ARM>/`, relabels the
origin to the contract's 12:00 UTC, restricts every arm to the decision window, and scores against
the metered reading (D8) or the injected reading (the host). Nothing here fits a model. The verdict
function reads the leakage tests and the gates from `checks.json`, which `e00695_checks.py` writes.

Systems at D8: A0 (raw), A1 (causal cleaning plus the climatology add-back), AH (hindsight), A1n
(signed variant), and the two bounds A1+state and AH+state. At the host: W0 (raw injected), W1
(causally cleaned plus add-back), WT (the raw-host network plus the add-back of the true engine's
climatology), and the bound WT+truth. It prints no Gas A series, only aggregate scores.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00695_addback import Calendar, add_back, build_calendar
from e00695_common import (
    D8_CAPACITY_MVA,
    D8_SERIES,
    DATA_DIR,
    E677_PROTOCOL_ROWS,
    ENGINE_MONTHS,
    FADE_MONTHS,
    HOLDOUT,
    PIPELINE_GRID_START,
    QUANTILE_COLUMNS,
    SCORED_MONTHS,
    SEEDS,
    SHARED_DIR,
    STEP9A_PATH,
    d8_truth,
    e694_flags,
    month_label,
    scored_window,
)
from e00695_engine import OUTPUT_BIN_MW, drive_series, output_bins
from scipy import stats
from studies.bootstrap import bootstrap_difference_at_level, paired_differences
from studies.eval_protocol.metrics import MEDIAN_INDEX, quantile_score
from studies.experiment_store import losses_for_bootstrap, write_predictions

NATIVE_TO_CONTRACT: Final = timedelta(hours=3)
FAMILY_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 3.0)
"""The coverage of each of the three planned contrasts, a Bonferroni split of 5%."""
FAMILY_RESAMPLES: Final[int] = 20_000
DESCRIPTIVE_LEVEL: Final[float] = 95.0
SMALLEST_GAIN_SHARE: Final[float] = 0.055
"""The smallest gain worth porting, as a share of the refitted A0's own CRPS at D8."""
G0_TOLERANCE: Final[float] = 0.10
TAIL_NAMES: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "24 to 48 h": (24.0, 48.0),
    "48 to 96 h": (48.0, 96.0),
    "96 to 168 h": (96.0, 168.0),
    "168 to 240 h": (168.0, 240.001),
}
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
SCORES_DIR: Final[Path] = DATA_DIR / "scores"
CHECKS_PATH: Final[Path] = DATA_DIR / "checks.json"
ROW_SET_ID: Final[str] = "backtest_year_2025_11_to_2026_06_daily_12utc"


@dataclass(frozen=True)
class System:
    """One scored forecast: which fits, which engine states, and what is added back."""

    name: str
    fits_arm: str
    series: str
    """`d8` or `host`."""
    states_arm: str | None
    variant: str | None
    """`climatology`, `state`, `truth`, or `None` for no add-back."""
    signed: bool = False


SYSTEMS: Final[tuple[System, ...]] = (
    System("A0", "A0", "d8", None, None),
    System("A1", "A1", "d8", "A1", "climatology"),
    System("AH", "AH", "d8", "AH", "climatology"),
    System("A1n", "A1n", "d8", "A1n", "climatology", signed=True),
    System("A1+state", "A1", "d8", "A1", "state"),
    System("AH+state", "AH", "d8", "AH", "state"),
    System("W0", "W0", "host", None, None),
    System("W1", "W1", "host", "W1", "climatology"),
    System("WT", "A0", "host", "truth", "climatology"),
    System("WT+truth", "A0", "host", "truth", "truth"),
)


def mean_of(values: pl.Series) -> float:
    """Return the mean of a Polars series as a float."""
    return float(values.to_numpy().mean())


def fits_dir(*, arm: str) -> Path:
    """Return the folder that holds an arm's prediction fits."""
    return DATA_DIR / "runs" / arm / "arms" / "k1_5w_sp1" / "work" / "fits" / "P5"


def read_fits(*, arm: str, seeds: tuple[int, ...] = SEEDS) -> pl.DataFrame:
    """Return an arm's quantiles (fractions of capacity), every series, with the contract origin.

    Each row carries `seed`. The network's native origin is 09:00 UTC, relabelled to 12:00 UTC.
    The quantiles are sorted within the row.
    """
    frames = []
    for month in SCORED_MONTHS:
        for seed in seeds:
            path = fits_dir(arm=arm) / f"{month}_d0_s{seed}.parquet"
            frames.append(pl.read_parquet(path).with_columns(seed=pl.lit(seed, dtype=pl.Int64)))
    frame = pl.concat(frames)
    assert (frame["power_fcst_init_time"].dt.hour() == 9).all(), "a native origin is not 09:00 UTC"
    q = np.sort(frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64), axis=1)
    return frame.with_columns(
        time_series_id=pl.col("time_series_id").cast(pl.Int64),
        power_fcst_init_time=pl.col("power_fcst_init_time") + NATIVE_TO_CONTRACT,
        **{name: pl.Series(name, q[:, i]) for i, name in enumerate(QUANTILE_COLUMNS)},
    ).select(*KEYS, "seed", *QUANTILE_COLUMNS)


def capacity_of() -> dict[int, float]:
    """Return each series' effective capacity (MVA or MW)."""
    table = pl.read_parquet(SHARED_DIR / "series_capacity.parquet")
    return dict(zip(table["time_series_id"].cast(pl.Int64), table["capacity"], strict=True))


def host_series_id() -> int:
    """Return the host's `time_series_id`, looked up from the host named in `host.json`."""
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00677_extend as ext  # ty: ignore[unresolved-import]

    ext.redirect()
    label = json.loads((DATA_DIR / "host.json").read_text())["host"]
    table = prep.primaries().select("site", "time_series_id")
    return int(dict(zip(table["site"], table["time_series_id"], strict=True))[label])


def grid_truth(*, y: np.ndarray) -> pl.DataFrame:
    """Return a series on the pipeline grid as `valid_time`, `power` (null where not finite)."""
    start = np.datetime64(PIPELINE_GRID_START.replace(tzinfo=None), "us")
    stamps = start + np.arange(len(y)) * np.timedelta64(1800, "s")
    return pl.DataFrame(
        {
            "valid_time": pl.Series(stamps).dt.replace_time_zone("UTC"),
            "power": pl.Series(y, nan_to_null=True),
        }
    )


@dataclass(frozen=True)
class EngineStates:
    """An engine-state distribution on the whole grid and the output of each state (MVA)."""

    mass: np.ndarray
    outputs: np.ndarray


def load_states(*, arm: str, month: str) -> EngineStates:
    """Return the engine states of an arm for one fold (`all.npz` for the hindsight arm)."""
    folder = DATA_DIR / "states" / arm
    path = folder / f"{month}.npz"
    data = np.load(path if path.exists() else folder / "all.npz")
    pk = data["pk"].astype(np.float64)
    return EngineStates(mass=pk, outputs=float(data["unit"]) * np.arange(pk.shape[1]))


def truth_states(*, n_grid: int) -> EngineStates:
    """Return the host's true injected engine as one-hot states over 0.25 MW output bins."""
    g, _ = drive_series(n_grid=n_grid)
    bins = output_bins(g=g)
    n_bins = int(bins.max()) + 1
    mass = np.zeros((n_grid, n_bins))
    mass[np.arange(n_grid), bins] = 1.0
    return EngineStates(mass=mass, outputs=OUTPUT_BIN_MW * np.arange(n_bins))


def system_forecast(
    *, system: System, fits: pl.DataFrame, series_id: int, capacity: float, calendar: Calendar
) -> pl.DataFrame:
    """Return one system's forecast of one series on the decision window, in MVA, add-back applied."""
    window = scored_window(
        frame=fits.filter(pl.col("time_series_id") == series_id).with_columns(
            *[pl.col(q) * capacity for q in QUANTILE_COLUMNS]
        )
    )
    if system.variant is None:
        return window
    truth = truth_states(n_grid=calendar.n_grid) if system.states_arm == "truth" else None
    out = []
    for seed in SEEDS:
        for month in SCORED_MONTHS:
            part = window.filter(
                pl.col("seed") == seed,
                pl.col("power_fcst_init_time").dt.strftime("%Y-%m") == month,
            )
            if part.is_empty():
                continue
            states = (
                truth if truth is not None else load_states(arm=str(system.states_arm), month=month)
            )
            out.append(
                add_back(
                    forecast=part,
                    mass=states.mass,
                    outputs=states.outputs,
                    calendar=calendar,
                    variant=system.variant,
                    signed=system.signed,
                )
            )
    return pl.concat(out).sort(*KEYS, "seed")


def row_scores(*, forecast: pl.DataFrame, truth: pl.DataFrame, capacity: float) -> pl.DataFrame:
    """Return each forecast row's losses in points of capacity, on the rows with a finite reading."""
    rows = forecast.join(truth, on="valid_time", how="inner", validate="m:1").filter(
        pl.col("power").is_not_null()
    )
    scale = 100.0 / capacity
    actual = rows["power"].to_numpy().astype(np.float64) * scale
    q = rows.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64) * scale
    levels = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
    error = actual[:, None] - q
    pinball = np.maximum(levels[None, :] * error, (levels[None, :] - 1.0) * error)
    columns = {
        "crps": quantile_score(actual=actual, quantiles=q),
        "mae": np.abs(actual - q[:, MEDIAN_INDEX]),
    }
    for name in TAIL_NAMES:
        position = QUANTILE_COLUMNS.index(name)
        columns[f"pinball_{name}"] = pinball[:, position]
        columns[f"coverage_{name}"] = (actual <= q[:, position]).astype(np.float64)
    return rows.select(*KEYS, "seed").with_columns(
        [pl.Series(name, values) for name, values in columns.items()]
    )


def _score_system(args: tuple[str, int, int]) -> tuple[str, pl.DataFrame, pl.DataFrame]:
    """Return one system's forecast and per-row losses (a worker of the pool)."""
    name, series_id, n_grid = args
    system = next(s for s in SYSTEMS if s.name == name)
    capacity = D8_CAPACITY_MVA if system.series == "d8" else capacity_of()[series_id]
    fits = read_fits(arm=system.fits_arm)
    calendar = build_calendar(n_grid=n_grid)
    forecast = system_forecast(
        system=system, fits=fits, series_id=series_id, capacity=capacity, calendar=calendar
    )
    if system.series == "d8":
        truth = d8_truth()
    else:
        import e00695_clean as clean

        truth = grid_truth(y=clean.load_setup(arm="W0").y)
    return name, forecast, row_scores(forecast=forecast, truth=truth, capacity=capacity)


def scored_systems(*, workers: int) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    """Return every system's (forecast, losses), computed in parallel."""
    import e00695_clean as clean

    n_grid = len(clean.load_setup(arm="W0").y)
    host = host_series_id()
    tasks = [(s.name, D8_SERIES if s.series == "d8" else host, n_grid) for s in SYSTEMS]
    with get_context("spawn").Pool(min(workers, len(tasks))) as pool:
        results = pool.map(_score_system, tasks, chunksize=1)
    return {name: (forecast, losses) for name, forecast, losses in results}


def family_losses(*, losses: dict[str, pl.DataFrame], names: list[str]) -> pl.DataFrame:
    """Return the `losses_for_bootstrap` frame of the named systems, which must hold the same rows."""
    keep = {n: losses[n] for n in names}
    return losses_for_bootstrap(arm_losses=keep)


def contrast(
    *,
    frame: pl.DataFrame,
    treatment: str,
    reference: str,
    metric: str,
    level: float,
    resamples: int,
    months: tuple[str, ...] | None = None,
) -> dict:
    """Return a paired contrast: estimate, month-bootstrap interval, t interval, and seed signs.

    The t interval is on the monthly means of the seed-averaged difference, at the same level.
    """
    scope = frame if months is None else frame.filter(pl.col("month").is_in(list(months)))
    differences, month_of = paired_differences(
        losses=scope, treatment=treatment, reference=reference, metric=metric
    )
    lower, upper = bootstrap_difference_at_level(
        losses=scope,
        treatment=treatment,
        reference=reference,
        metric=metric,
        level=level,
        n_resamples=resamples,
    )
    per_month = np.array(
        [differences[:, month_of == m].mean() for m in sorted(set(month_of.tolist()))]
    )
    t_lower = t_upper = float("nan")
    if len(per_month) > 1:
        t_lower, t_upper = stats.t.interval(
            level / 100.0, df=len(per_month) - 1, loc=per_month.mean(), scale=stats.sem(per_month)
        )
    return {
        "treatment": treatment,
        "reference": reference,
        "metric": metric,
        "level": level,
        "estimate": float(differences.mean()),
        "lower": lower,
        "upper": upper,
        "t_lower": float(t_lower),
        "t_upper": float(t_upper),
        "n_months": len(per_month),
        "monthly": per_month.tolist(),
        "seed_estimates": [float(v) for v in differences.mean(axis=1)],
    }


def passes(*, result: dict) -> bool:
    """A planned contrast passes when both of its intervals lie below zero."""
    return bool(result["upper"] < 0.0 and result["t_upper"] < 0.0)


def below_zero(*, result: dict) -> bool:
    """A descriptive contrast counts as below zero when its month-bootstrap interval is."""
    return bool(result["upper"] < 0.0)


def seeds_agree(*, result: dict) -> bool:
    """Return whether every seed's estimate of the contrast has the same sign."""
    signs = {np.sign(v) for v in result["seed_estimates"]}
    return len(signs) == 1


@dataclass(frozen=True)
class VerdictInputs:
    """Everything the verdict table reads."""

    leakage_failed: bool
    g0_failed: bool
    g1_failed: bool
    seeds_disagree: bool
    c1_passes: bool
    c1_estimate: float
    threshold: float
    c4_passes: bool
    mae_rises: bool
    other_primary_worse: bool
    c2_below: bool
    c3_below: bool


def verdict(*, v: VerdictInputs) -> tuple[int, str]:
    """Return (branch, verdict) of the plan's table: the first branch that applies."""
    if v.leakage_failed:
        return 1, "invalid"
    if v.g0_failed:
        return 2, "failed"
    if v.g1_failed or v.seeds_disagree:
        return 3, "inconclusive"
    beyond = v.c1_estimate <= -v.threshold
    if v.c1_passes and beyond:
        if v.c4_passes and not v.mae_rises and not v.other_primary_worse:
            return 4, "supported"
        return 5, "partly supported"
    if v.c1_passes:
        return 6, "small gain"
    if v.c2_below != v.c3_below:
        return 7, "partly supported"
    return 8, "not supported"


def sister_differences() -> dict[str, float]:
    """Return each Primary's absolute CRPS difference between experiment 677's stored sister arms.

    The arms are step 9a (5w) and 5wo, monthly refits, on the decision window of experiment 677's
    shared rows.
    """
    frame = scored_window(
        frame=pl.read_parquet(
            E677_PROTOCOL_ROWS,
            columns=[
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                "label",
                "arm",
                "crps",
            ],
        ).filter(pl.col("arm").is_in(["Step 5w monthly", "Step 5wo monthly"]))
    )
    table = frame.group_by("label", "arm").agg(crps=pl.col("crps").mean())
    wide = table.pivot(on="arm", index="label", values="crps")
    return {
        row["label"]: abs(row["Step 5w monthly"] - row["Step 5wo monthly"])
        for row in wide.iter_rows(named=True)
    }


def other_primaries(*, resamples: int) -> list[dict]:
    """Return A1 minus A0 at every Primary, with the plan's "moves" rule applied."""
    a0, a1 = read_fits(arm="A0"), read_fits(arm="A1")
    caps = capacity_of()
    truth = pl.read_parquet(SHARED_DIR / "ground_truth_primaries.parquet").with_columns(
        time_series_id=pl.col("time_series_id").cast(pl.Int64)
    )
    sisters = sister_differences()
    labels = pl.read_parquet(E677_PROTOCOL_ROWS, columns=["time_series_id", "label"]).unique()
    label_of = dict(zip(labels["time_series_id"].cast(pl.Int64), labels["label"], strict=True))
    out = []
    for series_id in sorted(a0["time_series_id"].unique().to_list()):
        losses = {}
        for name, fits in (("A0", a0), ("A1", a1)):
            part = scored_window(
                frame=fits.filter(pl.col("time_series_id") == series_id).with_columns(
                    *[pl.col(q) * caps[series_id] for q in QUANTILE_COLUMNS]
                )
            )
            t = truth.filter(pl.col("time_series_id") == series_id).select("valid_time", "power")
            losses[name] = row_scores(forecast=part, truth=t, capacity=caps[series_id])
        frame = losses_for_bootstrap(arm_losses=losses)
        result = contrast(
            frame=frame,
            treatment="A1",
            reference="A0",
            metric="crps",
            level=DESCRIPTIVE_LEVEL,
            resamples=resamples,
        )
        label = label_of.get(series_id, str(series_id))
        sister = sisters.get(label, float("nan"))
        excludes = result["upper"] < 0.0 or result["lower"] > 0.0
        moves = bool(excludes and abs(result["estimate"]) > sister)
        out.append(
            {
                "series": label,
                "estimate": result["estimate"],
                "lower": result["lower"],
                "upper": result["upper"],
                "sister_difference": sister,
                "moves": moves,
                "worse": moves and result["estimate"] > 0.0,
            }
        )
    return out


def g0_check(*, a0_losses: pl.DataFrame) -> dict:
    """Compare the refitted A0's seed-mean D8 CRPS with the stored step 9a on shared rows.

    The row keys of the stored file are asserted equal to A0's before the comparison.
    """
    stored = (
        pl.read_parquet(STEP9A_PATH)
        .filter(pl.col("time_series_id") == D8_SERIES)
        .with_columns(power_fcst_init_time=pl.col("power_fcst_init_time") + NATIVE_TO_CONTRACT)
    )
    stored = scored_window(frame=stored)
    keys = a0_losses.select(*KEYS).unique()
    stored_keys = stored.select(*KEYS).unique()
    shared = keys.join(stored_keys, on=list(KEYS), how="inner")
    assert shared.height == keys.height, (
        "A0's row keys differ from the stored step 9a's",
        keys.height,
        shared.height,
    )
    truth = d8_truth()
    mean = stored.group_by(*KEYS).agg([pl.col(q).mean() for q in QUANTILE_COLUMNS])
    cap = D8_CAPACITY_MVA
    fractions = mean.with_columns(*[pl.col(q) * cap for q in QUANTILE_COLUMNS]).with_columns(
        seed=pl.lit(0, dtype=pl.Int64)
    )
    stored_score = row_scores(forecast=fractions, truth=truth, capacity=cap)
    refit = a0_losses.group_by(*KEYS).agg(crps=pl.col("crps").mean(), mae=pl.col("mae").mean())
    stored_by_key = stored_score.select(*KEYS, stored_crps=pl.col("crps"))
    joined = refit.join(stored_by_key, on=list(KEYS), how="inner")
    gap = mean_of(joined["crps"]) - mean_of(joined["stored_crps"])
    return {
        "rows": joined.height,
        "refitted_a0_seed_mean_row_crps": mean_of(joined["crps"]),
        "stored_step9a_row_crps": mean_of(joined["stored_crps"]),
        "difference": gap,
        "passes": abs(gap) <= G0_TOLERANCE,
        "note": "the stored file's seed-mean quantiles are scored as one forecast; the refit's row scores are averaged over seeds",
    }


def slice_table(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return the D8 rows' slice labels: engine and fade months, tracked state, sunny, cold nights, lead band."""
    protocol = (
        pl.scan_parquet(E677_PROTOCOL_ROWS)
        .filter(pl.col("arm") == "Step 5w monthly", pl.col("time_series_id") == D8_SERIES)
        .select(*KEYS, "cold_nights", "winter_nights", "sunny_unit_shape")
        .collect()
    )
    flags = e694_flags().select("valid_time", "flag")
    keys = losses.select(*KEYS).unique()
    return (
        keys.join(protocol, on=list(KEYS), how="left")
        .join(flags, on="valid_time", how="left")
        .with_columns(
            month=month_label(valid_time=pl.col("valid_time")),
            lead_h=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
        )
        .with_columns(
            engine_months=pl.col("month").is_in(ENGINE_MONTHS),
            fade_months=pl.col("month").is_in(FADE_MONTHS),
            sunny_fade=pl.col("sunny_unit_shape").fill_null(value=False)
            & pl.col("month").is_in(FADE_MONTHS),
        )
    )


def slice_scores(*, losses: dict[str, pl.DataFrame], names: list[str]) -> pl.DataFrame:
    """Return seed-mean scores of each system on each D8 slice."""
    sl = slice_table(losses=losses[names[0]])
    masks: dict[str, pl.Expr] = {
        "all rows": pl.lit(value=True),
        "engine months (valid time)": pl.col("engine_months"),
        "fade months (valid time)": pl.col("fade_months"),
        "tracked on (hindsight M2 flag)": pl.col("flag").fill_null(value=False),
        "tracked off (hindsight M2 flag)": ~pl.col("flag").fill_null(value=True),
        "sunny by unit shape, fade months": pl.col("sunny_fade"),
        "cold winter nights": pl.col("cold_nights").fill_null(value=False),
        **{
            f"lead {band}": (pl.col("lead_h") >= lo) & (pl.col("lead_h") < hi)
            for band, (lo, hi) in LEAD_BANDS.items()
        },
    }
    rows = []
    for name in names:
        joined = losses[name].join(sl, on=list(KEYS), how="left", validate="m:1")
        for label, mask in masks.items():
            part = joined.filter(mask)
            if part.is_empty():
                continue
            row = {"system": name, "slice": label, "rows": part.height // len(SEEDS)}
            for column in (
                "crps",
                "mae",
                *[f"pinball_{t}" for t in TAIL_NAMES],
                *[f"coverage_{t}" for t in TAIL_NAMES],
            ):
                row[column] = mean_of(part[column])
            rows.append(row)
    return pl.DataFrame(rows)


def host_slices(*, losses: dict[str, pl.DataFrame], names: list[str], n_grid: int) -> pl.DataFrame:
    """Return seed-mean scores of each host system where the injected engine is truly on and truly off."""
    g, _ = drive_series(n_grid=n_grid)
    truth = grid_truth(y=g).rename({"power": "output"})
    rows = []
    for name in names:
        joined = losses[name].join(truth, on="valid_time", how="left", validate="m:1")
        for label, mask in (
            ("all rows", pl.lit(value=True)),
            ("engine truly on", pl.col("output") > 0.0),
            ("engine truly off", pl.col("output") <= 0.0),
        ):
            part = joined.filter(mask)
            rows.append(
                {
                    "system": name,
                    "slice": label,
                    "rows": part.height // len(SEEDS),
                    "crps": mean_of(part["crps"]),
                    "mae": mean_of(part["mae"]),
                }
            )
    return pl.DataFrame(rows)


def assert_origin_time_weather(*, names: list[str]) -> dict:
    """Assert that every scored row's forecast origin is the same in every system, and report it.

    The network reads the 00Z ensemble run of the origin day, available at 09:00 UTC, and the
    fits carry no weather column, so the assertion is on what they carry: the native origin of each
    row is 09:00 UTC and equals A0's. The stored fits come from experiment 677's step-5 chain,
    whose inference reads only the run of the origin day.
    """
    base = read_fits(arm="A0").select(*KEYS).unique().sort(*KEYS)
    for arm in {s.fits_arm for s in SYSTEMS if s.name in names}:
        other = read_fits(arm=arm).select(*KEYS).unique().sort(*KEYS)
        assert base.height == other.height and base.equals(other), (
            f"arm {arm}'s rows differ from A0's"
        )
    nwp = (base["power_fcst_init_time"] - NATIVE_TO_CONTRACT).dt.hour().unique().to_list()
    assert nwp == [9], nwp
    return {"native_origin_hours": nwp, "rows_per_arm": base.height}


def climatology_reference() -> pl.DataFrame:
    """Return the stored climatology's seed-free scores at D8 on the decision window, if experiment 677's rows hold one."""
    frame = scored_window(
        frame=pl.read_parquet(
            E677_PROTOCOL_ROWS,
            columns=[
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                "label",
                "arm",
                "crps",
                "mae_median",
            ],
        ).filter(pl.col("label") == "D8")
    )
    return (
        frame.group_by("arm")
        .agg(rows=pl.len(), crps=pl.col("crps").mean(), mae=pl.col("mae_median").mean())
        .sort("arm")
    )


def main() -> None:
    """Score every system, compute the contrasts, and write the tables and the verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=FAMILY_RESAMPLES)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    import e00695_clean as clean

    n_grid = len(clean.load_setup(arm="W0").y)
    scored = scored_systems(workers=args.workers)
    losses = {name: pair[1] for name, pair in scored.items()}
    for name, (forecast, _) in scored.items():
        if name in {"A0", "A1", "AH", "W0", "W1"}:
            write_predictions(
                experiment_id=695,
                name=f"{name.lower()}-{'d8' if name[0] == 'A' else 'host'}",
                frame=forecast,
                kind="forecast",  # ty: ignore[invalid-argument-type]
                row_set_id=ROW_SET_ID,
            )
    d8_names = ["A0", "A1", "AH", "A1n", "A1+state", "AH+state"]
    host_names = ["W0", "W1", "WT", "WT+truth"]
    d8 = family_losses(losses=losses, names=d8_names)
    host = family_losses(losses=losses, names=host_names)
    c1 = contrast(
        frame=d8,
        treatment="A1",
        reference="A0",
        metric="crps",
        level=FAMILY_LEVEL,
        resamples=args.resamples,
    )
    c4 = contrast(
        frame=host,
        treatment="W1",
        reference="W0",
        metric="crps",
        level=FAMILY_LEVEL,
        resamples=args.resamples,
    )
    c5 = contrast(
        frame=host,
        treatment="WT",
        reference="W0",
        metric="crps",
        level=FAMILY_LEVEL,
        resamples=args.resamples,
    )
    c2 = contrast(
        frame=d8,
        treatment="A1",
        reference="A0",
        metric="crps",
        level=DESCRIPTIVE_LEVEL,
        resamples=2000,
        months=ENGINE_MONTHS,
    )
    c3 = contrast(
        frame=d8,
        treatment="A1",
        reference="A0",
        metric="crps",
        level=DESCRIPTIVE_LEVEL,
        resamples=2000,
        months=FADE_MONTHS,
    )
    mae = contrast(
        frame=d8,
        treatment="A1",
        reference="A0",
        metric="mae",
        level=DESCRIPTIVE_LEVEL,
        resamples=2000,
    )
    extras = {
        f"{t} minus {r}": contrast(
            frame=d8,
            treatment=t,
            reference=r,
            metric="crps",
            level=DESCRIPTIVE_LEVEL,
            resamples=2000,
        )
        for t, r in (("AH", "A1"), ("A1n", "A1"), ("A1+state", "A0"), ("AH+state", "A0"))
    } | {
        "W1 minus WT": contrast(
            frame=host,
            treatment="W1",
            reference="WT",
            metric="crps",
            level=DESCRIPTIVE_LEVEL,
            resamples=2000,
        ),
        "WT+truth minus W0": contrast(
            frame=host,
            treatment="WT+truth",
            reference="W0",
            metric="crps",
            level=DESCRIPTIVE_LEVEL,
            resamples=2000,
        ),
    }
    a0_crps = mean_of(losses["A0"]["crps"])
    threshold = SMALLEST_GAIN_SHARE * a0_crps
    primaries = other_primaries(resamples=2000)
    g0 = g0_check(a0_losses=losses["A0"])
    checks = json.loads(CHECKS_PATH.read_text()) if CHECKS_PATH.exists() else {}
    inputs = VerdictInputs(
        leakage_failed=not checks.get("leakage_passed", False),
        g0_failed=not g0["passes"],
        g1_failed=not checks.get("g1_passed", False),
        seeds_disagree=not seeds_agree(result=c1),
        c1_passes=passes(result=c1),
        c1_estimate=c1["estimate"],
        threshold=threshold,
        c4_passes=passes(result=c4),
        mae_rises=mae["lower"] > 0.0,
        other_primary_worse=any(p["worse"] and p["series"] != "D8" for p in primaries),
        c2_below=below_zero(result=c2),
        c3_below=below_zero(result=c3),
    )
    branch, name = verdict(v=inputs)
    slices = slice_scores(losses=losses, names=d8_names)
    slices.write_csv(SCORES_DIR / "d8_slices.csv")
    host_slices(losses=losses, names=host_names, n_grid=n_grid).write_csv(
        SCORES_DIR / "host_slices.csv"
    )
    report = {
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "c5": c5,
        "mae_a1_minus_a0": mae,
        "extras": extras,
        "threshold_points": threshold,
        "a0_crps": a0_crps,
        "g0": g0,
        "other_primaries": primaries,
        "origin_time_assertion": assert_origin_time_weather(names=d8_names),
        "verdict_inputs": inputs.__dict__,
        "branch": branch,
        "verdict": name,
        "mean_crps": {n: mean_of(losses[n]["crps"]) for n in losses},
        "mean_mae": {n: mean_of(losses[n]["mae"]) for n in losses},
        "rows_per_seed": {n: losses[n].height // len(SEEDS) for n in losses},
    }
    (SCORES_DIR / "report.json").write_text(json.dumps(report, indent=1, default=float))
    climatology_reference().write_csv(SCORES_DIR / "references_d8.csv")
    print(
        json.dumps({k: report[k] for k in ("branch", "verdict", "mean_crps", "mean_mae")}, indent=1)
    )
    print(f"HOLDOUT {HOLDOUT:%Y-%m-%d}: no valid time on or after it is scored")


if __name__ == "__main__":
    main()

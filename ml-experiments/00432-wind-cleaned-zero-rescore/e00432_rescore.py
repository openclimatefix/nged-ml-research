"""Experiment 00432: re-score the stored wind forecasts on raw, masked, and cleaned-zero rows.

Fits nothing. Reads the stored quantile forecasts of experiments 336, 339, 343, and 347 (all on the
same 107,121 scored half-hours of the three metered wind farms) and the experiment 210 idle-band
mask, and writes tables to `<experiment folder>/work/`.

Row sets (see the plan in `result.md`):

- raw: every half-hour, scored against the reading;
- masked: every half-hour in the experiment 210 mask dropped;
- cleaned zero: `low_wind_idle` half-hours whose observed power lies in the farm's idle band (at or
  below its 95% value) kept with the target set to 0, other half-hours kept at their reading,
  `outage_suspected` half-hours dropped. The `low_wind_idle` flag is an ERA5 wind rule alone, so
  half-hours the flag marks while the farm generated above the idle band keep their reading;
- cleaned zero ERA5 (sensitivity): every `low_wind_idle` half-hour has the target 0.

Run with experiment 303's folder on the path (it holds the quantile score):

    PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting \
        uv run --no-sync python ml-experiments/00432-*/e00432_rescore.py
"""

import hashlib
import json
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from pathlib import Path  # noqa: E402
from typing import Final  # noqa: E402

import e00303_common as c303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402

EXPERIMENT_ID: Final[int] = 432
SLUG: Final[str] = "wind-cleaned-zero-rescore"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Private tables; never committed."""
SOURCE_336: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00336-wind-research-default-1-below-cut-in-zeroing" / "work"
)
SOURCE_339: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00339-wind-research-default-1-adapter-plus-mixture-head"
)
SOURCE_343: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00343-wind-research-default-1-train-on-ensemble-mean" / "work"
)
SOURCE_347: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00347-wind-physics-hybrid-plant-then-mdn" / "s4"
)
SOURCE_210: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00210-wind-idle-noise-mva-artefact"
CORES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00230-wind-physics-core-part-1" / "work" / "b1_cores.json"
)

FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
ROW_SETS: Final[tuple[str, ...]] = ("raw", "masked", "cleaned_zero", "cleaned_zero_era5")
BANDS: Final[dict[str, tuple[float, float]]] = {
    "days_1_to_10": (24.0, 240.0),
    "hours_12_to_36": (12.0, 36.0),
}
"""Inclusive lead-hour bounds."""
METRICS: Final[tuple[str, ...]] = ("qs", "p05", "p50", "p95", "p99")
LEVEL_INDEX: Final[dict[str, int]] = {"p05": 1, "p50": 5, "p95": 9, "p99": 10}
POINTS: Final[float] = 100.0
"""Fractions of capacity are reported in points of capacity."""
N_BOOT: Final[int] = 2000
N_PLACEBO: Final[int] = 200
BOOT_SEED: Final[int] = 432
NEAR_ZERO: Final[float] = 0.005
"""A quantile at or below 0.5% of capacity counts as 'at or near zero'."""
WIND_BIN_TOP: Final[int] = 20
"""ERA5 100 m speed is binned at 1 m/s and every speed from this bin up shares the top bin."""
PLACEBOS: Final[dict[str, tuple[str, ...]]] = {
    "farm_month_hour": ("site", "month", "hour"),
    "farm_month_hour_wind": ("site", "month", "hour", "wind_bin"),
}
"""The first placebo does not know the wind speed, so it asks whether the mask beats random drops
of the same calendar profile. The second matches the wind-speed profile too, and so reproduces
most of a mask that is itself an ERA5 wind rule; it shows how much of the effect is the wind."""
REFERENCE_ARM: Final[str] = "default"
STORED_QS_COLUMNS: Final[dict[str, tuple[str, str]]] = {
    # arm -> (rows table, column) holding the stored per-row score, used as a reproduction gate.
    "x1_336": ("336", "qs__x1"),
    "head_x1": ("339", "qs__x1"),
    "head_x2": ("339", "qs__x2"),
    "head_x3": ("339", "qs__x3"),
    "default": ("339", "qs__default"),
    "plant_head_y2": ("347", "qs__y2"),
}
REPRODUCTION_TOLERANCE: Final[float] = 1e-6


def load_arms() -> dict[str, np.ndarray]:
    """Return each stored forecast as float64 quantiles, shape (seeds, cases, 11)."""
    arms = {
        "default": np.load(SOURCE_339 / "final" / "default.npy"),
        "x1_336": np.load(SOURCE_336 / "final" / "x1.npy"),
        "head_x1": np.load(SOURCE_339 / "final" / "x1.npy"),
        "head_x2": np.load(SOURCE_339 / "final" / "x2.npy"),
        "head_x3": np.load(SOURCE_339 / "final" / "x3.npy"),
        "xgb_ens_mean_M": np.load(SOURCE_343 / "final" / "M.npy"),
    }
    y2 = pl.read_parquet(SOURCE_347 / "predictions_y2.parquet")
    arms["plant_head_y2"] = y2.select(c303.LEVEL_COLUMNS).to_numpy()[None]
    for other in (
        np.load(SOURCE_336 / "final" / "default.npy"),
        np.load(SOURCE_343 / "final" / "default.npy"),
    ):
        if not np.array_equal(other, arms[REFERENCE_ARM]):
            raise ValueError("the default differs between experiments 336, 339, and 343")
    return {name: array.astype(np.float64) for name, array in arms.items()}


def load_cases() -> pl.DataFrame:
    """Return one row per scored half-hour, with the mask flag, ERA5 speed, and observed power."""
    y2 = pl.read_parquet(SOURCE_347 / "predictions_y2.parquet").select(
        "site", "time_series_id", "valid_time", "lead_hours", "actual"
    )
    rows_339 = pl.read_parquet(SOURCE_339 / "rows.parquet").select("lead_day", "month", "masked")
    rows_336 = pl.read_parquet(SOURCE_336 / "rows.parquet").select("site", "valid_time", "masked")
    rows_347 = pl.read_parquet(SOURCE_347 / "rows.parquet").select("site", "masked")
    if not (
        (rows_336["site"] == y2["site"]).all()
        and (rows_336["valid_time"] == y2["valid_time"]).all()
        and (rows_347["site"] == y2["site"]).all()
        and (rows_336["masked"] == rows_339["masked"]).all()
        and (rows_336["masked"] == rows_347["masked"]).all()
    ):
        raise ValueError("the stored row tables of experiments 336, 339, and 347 are not aligned")
    mask = pl.read_parquet(SOURCE_210 / "mask.parquet")
    joined = pl.read_parquet(SOURCE_210 / "joined.parquet").select(
        "time_series_id", "valid_time", "power", "wind_speed"
    )
    cases = (
        y2.with_row_index("row")
        .hstack(rows_339.select("lead_day", "month"))
        .join(mask, on=["time_series_id", "valid_time"], how="left")
        .join(joined, on=["time_series_id", "valid_time"], how="left")
        .sort("row")
    )
    if cases.height != y2.height:
        raise ValueError("a join changed the number of scored rows")
    if (cases["drop_reason"].is_not_null() != rows_339["masked"]).any():
        raise ValueError(
            "the stored `masked` flag differs from membership of the experiment 210 mask"
        )
    report = json.loads((SOURCE_210 / "report.json").read_text())
    floor_mw = pl.col("site").replace_strict(
        {farm: float(report[farm]["idle_floor"]["p95"]) for farm in FARMS}, return_dtype=pl.Float64
    )
    return cases.with_columns(
        in_mask=pl.col("drop_reason").is_not_null(),
        idle_reading=(pl.col("drop_reason") == "low_wind_idle") & (pl.col("power") <= floor_mw),
        idle=pl.col("drop_reason") == "low_wind_idle",
        outage=pl.col("drop_reason") == "outage_suspected",
        actual_cleaned_zero=pl.when(
            (pl.col("drop_reason") == "low_wind_idle") & (pl.col("power") <= floor_mw)
        )
        .then(0.0)
        .otherwise(pl.col("actual")),
        actual_cleaned_zero_era5=pl.when(pl.col("drop_reason") == "low_wind_idle")
        .then(0.0)
        .otherwise(pl.col("actual")),
        hour=pl.col("valid_time").dt.hour(),
        weekday=pl.col("valid_time").dt.weekday(),
        wind_bin=pl.col("wind_speed").clip(0, WIND_BIN_TOP).floor().cast(pl.Int32),
    )


def reproduction_gate(*, arms: dict[str, np.ndarray], cases: pl.DataFrame) -> dict[str, float]:
    """Check the raw-set per-row score against each stored per-row score; return the max gaps."""
    tables = {
        "336": pl.read_parquet(SOURCE_336 / "rows.parquet"),
        "339": pl.read_parquet(SOURCE_339 / "rows.parquet"),
        "347": pl.read_parquet(SOURCE_347 / "rows.parquet"),
    }
    actual = cases["actual"].to_numpy()
    gaps = {}
    for arm, (table, column) in STORED_QS_COLUMNS.items():
        mine = row_losses(quantiles=arms[arm], actual=actual)["qs"]
        gap = float(np.abs(mine - tables[table][column].to_numpy()).max())
        gaps[arm] = gap
        if gap > REPRODUCTION_TOLERANCE:
            raise ValueError(f"arm {arm} does not reproduce its stored per-row score: gap {gap}")
    return gaps


def row_losses(*, quantiles: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row losses in points of capacity, averaged over seeds as the stored scores are.

    `qs` is experiment 33's quantile score (twice the mass-weighted pinball loss of nine levels).
    `p05`, `p50`, `p95`, and `p99` are the pinball loss at that level; `below_q05`, `above_q95`,
    and `above_q99` are the 0/1 miss indicators, whose mean is the miss rate.
    """
    out: dict[str, list[np.ndarray]] = {key: [] for key in (*METRICS, "below_q05", "above_q95")}
    out["above_q99"] = []
    for seed_quantiles in quantiles:
        out["qs"].append(c303.quantile_score(quantiles=seed_quantiles, actual=actual) * POINTS)
        for key, index in LEVEL_INDEX.items():
            level = float(c303.LEVELS[index])
            difference = actual - seed_quantiles[:, index]
            out[key].append(np.where(difference >= 0, level, level - 1.0) * difference * POINTS)
        out["below_q05"].append((actual < seed_quantiles[:, 1]).astype(np.float64))
        out["above_q95"].append((actual > seed_quantiles[:, 9]).astype(np.float64))
        out["above_q99"].append((actual > seed_quantiles[:, 10]).astype(np.float64))
    return {key: np.mean(values, axis=0) for key, values in out.items()}


def month_block_interval(
    *, sums: np.ndarray, counts: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return the mean and 95% interval of `sum(sums) / sum(counts)` over resampled month blocks."""
    mean = float(sums.sum() / counts.sum())
    draws = rng.integers(0, len(sums), size=(N_BOOT, len(sums)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return mean, float(low), float(high)


def summarise(*, cases: pl.DataFrame, losses: dict[str, dict[str, np.ndarray]]) -> pl.DataFrame:
    """Score every arm on every row set, band, and farm, with paired differences to the default."""
    months = sorted(cases["month"].unique().to_list())
    month_index = cases["month"].replace_strict(months, list(range(len(months)))).to_numpy()
    farm_column = cases["site"].to_numpy()
    lead = cases["lead_hours"].to_numpy()
    in_mask = cases["in_mask"].to_numpy()
    outage = cases["outage"].to_numpy()
    actuals = {
        "raw": (cases["actual"].to_numpy(), np.ones(cases.height, dtype=bool)),
        "masked": (cases["actual"].to_numpy(), ~in_mask),
        "cleaned_zero": (cases["actual_cleaned_zero"].to_numpy(), ~outage),
        "cleaned_zero_era5": (cases["actual_cleaned_zero_era5"].to_numpy(), ~outage),
    }
    rng = np.random.default_rng(BOOT_SEED)
    records = []
    for row_set, (actual, keep_set) in actuals.items():
        per_arm = {
            arm: row_losses(quantiles=quantiles, actual=actual)
            for arm, quantiles in losses["quantiles"].items()
        }
        for band, (low, high) in BANDS.items():
            in_band = (lead >= low) & (lead <= high) & keep_set
            for farm in (*FARMS, "pooled"):
                selection = in_band if farm == "pooled" else in_band & (farm_column == farm)
                counts = np.bincount(month_index[selection], minlength=len(months)).astype(float)
                if counts.min() == 0:
                    continue
                reference = per_arm[REFERENCE_ARM]
                for arm, arm_losses in per_arm.items():
                    for metric in (*METRICS, "below_q05", "above_q95", "above_q99"):
                        values = arm_losses[metric]
                        sums = np.bincount(
                            month_index[selection], weights=values[selection], minlength=len(months)
                        )
                        mean, lo, hi = month_block_interval(sums=sums, counts=counts, rng=rng)
                        record = {
                            "row_set": row_set,
                            "band": band,
                            "farm": farm,
                            "arm": arm,
                            "metric": metric,
                            "n": int(counts.sum()),
                            "mean": mean,
                            "lo": lo,
                            "hi": hi,
                        }
                        if arm != REFERENCE_ARM:
                            diff_sums = np.bincount(
                                month_index[selection],
                                weights=(values - reference[metric])[selection],
                                minlength=len(months),
                            )
                            d_mean, d_lo, d_hi = month_block_interval(
                                sums=diff_sums, counts=counts, rng=rng
                            )
                            record |= {"diff": d_mean, "diff_lo": d_lo, "diff_hi": d_hi}
                        records.append(record)
    return pl.DataFrame(records, infer_schema_length=None)


def dropped_share_tables(cases: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """G2: the share of scored half-hours in the mask, by month, wind bin, hour, and weekday."""
    out = {}
    for name, column in (
        ("month", "month"),
        ("wind_bin", "wind_bin"),
        ("hour", "hour"),
        ("weekday", "weekday"),
    ):
        out[name] = (
            cases.group_by(column)
            .agg(
                n=pl.len(),
                share_idle=pl.col("idle").mean(),
                share_outage=pl.col("outage").mean(),
                share_dropped=pl.col("in_mask").mean(),
            )
            .sort(column)
        )
    return out


def placebo(
    *,
    cases: pl.DataFrame,
    losses: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    stratify_by: tuple[str, ...],
) -> pl.DataFrame:
    """G3: drop random half-hours with the mask's count in each stratum, then score every arm.

    A stratum is the combination of the `stratify_by` columns. Every draw drops, in each stratum,
    as many random half-hours as the mask drops there, and the pooled days-1-to-10 score of each
    arm on the kept half-hours is recorded. The mask's own value is row `draw == -1`.
    """
    strata = (
        cases.select(pl.struct(*stratify_by).rank("dense").alias("stratum"))["stratum"]
        .to_numpy()
        .astype(np.int64)
    )
    n_strata = int(strata.max()) + 1
    n_masked = np.bincount(
        strata, weights=cases["in_mask"].to_numpy().astype(float), minlength=n_strata
    )
    n_masked = n_masked.astype(np.int64)
    lead = cases["lead_hours"].to_numpy()
    in_band = (lead >= BANDS["days_1_to_10"][0]) & (lead <= BANDS["days_1_to_10"][1])
    raw_qs = {
        arm: row_losses(quantiles=quantiles, actual=cases["actual"].to_numpy())["qs"]
        for arm, quantiles in losses["quantiles"].items()
    }
    order_base = np.argsort(strata, kind="stable")
    stratum_start = np.concatenate([[0], np.cumsum(np.bincount(strata, minlength=n_strata))[:-1]])
    records = []
    mask_keep = ~cases["in_mask"].to_numpy()
    for arm, qs in raw_qs.items():
        records.append(
            {
                "draw": -1,
                "arm": arm,
                "all": float(qs[in_band].mean()),
                "kept": float(qs[in_band & mask_keep].mean()),
            }
        )
    for draw in range(N_PLACEBO):
        key = rng.random(cases.height)
        order = order_base[np.lexsort((key[order_base], strata[order_base]))]
        rank_in_stratum = np.empty(cases.height, dtype=np.int64)
        rank_in_stratum[order] = np.arange(cases.height) - stratum_start[strata[order]]
        keep = rank_in_stratum >= n_masked[strata]
        for arm, qs in raw_qs.items():
            records.append(
                {
                    "draw": draw,
                    "arm": arm,
                    "all": float(qs[in_band].mean()),
                    "kept": float(qs[in_band & keep].mean()),
                }
            )
    return pl.DataFrame(records)


def low_wind_slice(
    *,
    cases: pl.DataFrame,
    losses: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    condition_on_power: bool,
) -> pl.DataFrame:
    """G4: score the ERA5 speed slice from cut-in to 1.5 x cut-in, optionally above the floor."""
    cores = json.loads(CORES_PATH.read_text())
    report = json.loads((SOURCE_210 / "report.json").read_text())
    cut_in = {
        farm: min(max(float(cores[farm]["cut_in_m_s"]), 2.0), 5.0)
        if 2.0 <= float(cores[farm]["cut_in_m_s"]) <= 5.0
        else 3.0
        for farm in FARMS
    }
    floor_fraction = {
        farm: report[farm]["idle_floor"]["p95"] / report[farm]["capacity"] for farm in FARMS
    }
    site = cases["site"].to_numpy()
    wind = cases["wind_speed"].to_numpy()
    power_fraction = cases["power"].to_numpy() / cases["capacity_mw"].to_numpy()
    in_slice = np.zeros(cases.height, dtype=bool)
    for farm in FARMS:
        in_slice |= (
            (site == farm)
            & (wind >= cut_in[farm])
            & (wind < 1.5 * cut_in[farm])
            & ((power_fraction > floor_fraction[farm]) | (not condition_on_power))
        )
    months = sorted(cases["month"].unique().to_list())
    month_index = cases["month"].replace_strict(months, list(range(len(months)))).to_numpy()
    lead = cases["lead_hours"].to_numpy()
    selection = in_slice & (lead >= BANDS["days_1_to_10"][0]) & (lead <= BANDS["days_1_to_10"][1])
    counts = np.bincount(month_index[selection], minlength=len(months)).astype(float)
    reference = row_losses(
        quantiles=losses["quantiles"][REFERENCE_ARM], actual=cases["actual"].to_numpy()
    )
    records = []
    for arm, quantiles in losses["quantiles"].items():
        values = row_losses(quantiles=quantiles, actual=cases["actual"].to_numpy())["qs"]
        sums = np.bincount(month_index[selection], weights=values[selection], minlength=len(months))
        mean, lo, hi = month_block_interval(sums=sums, counts=np.maximum(counts, 1.0), rng=rng)
        record = {"arm": arm, "n": int(counts.sum()), "mean": mean, "lo": lo, "hi": hi}
        if arm != REFERENCE_ARM:
            diff = np.bincount(
                month_index[selection],
                weights=(values - reference["qs"])[selection],
                minlength=len(months),
            )
            d_mean, d_lo, d_hi = month_block_interval(
                sums=diff, counts=np.maximum(counts, 1.0), rng=rng
            )
            record |= {"diff": d_mean, "diff_lo": d_lo, "diff_hi": d_hi}
        records.append(record)
    return pl.DataFrame(records, infer_schema_length=None).with_columns(
        slice_cut_ins=pl.lit(json.dumps(cut_in))
    )


def zero_mass(*, cases: pl.DataFrame, arms: dict[str, np.ndarray]) -> pl.DataFrame:
    """Report each forecast's mass at or near zero, on idle-flagged and on kept half-hours."""
    groups = {
        "idle_flagged": cases["idle"].to_numpy(),
        "kept": ~cases["in_mask"].to_numpy(),
    }
    records = []
    for arm, quantiles in arms.items():
        near = quantiles <= NEAR_ZERO
        # Highest of the 11 levels whose quantile is at or near zero: a lower bound on P(zero).
        levels = np.asarray(c303.LEVELS)
        p_zero = np.where(near, levels, 0.0).max(axis=2)
        for group, rows in groups.items():
            records.append(
                {
                    "arm": arm,
                    "group": group,
                    "n": int(rows.sum()),
                    "share_q05_near_zero": float(near[:, rows, 1].mean()),
                    "share_q50_near_zero": float(near[:, rows, 5].mean()),
                    "mean_p_zero_lower_bound": float(p_zero[:, rows].mean()),
                }
            )
    return pl.DataFrame(records)


def main() -> None:
    """Run the re-score and write the tables."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    mask_hash = hashlib.sha256((SOURCE_210 / "mask.parquet").read_bytes()).hexdigest()
    print("G1 mask sha256", mask_hash, flush=True)
    arms = load_arms()
    cases = load_cases()
    gaps = reproduction_gate(arms=arms, cases=cases)
    print("reproduction gate, max per-row gaps:", gaps, flush=True)
    losses = {"quantiles": arms}
    rng = np.random.default_rng(BOOT_SEED)
    summary = summarise(cases=cases, losses=losses)
    summary.write_parquet(WORK_DIR / "summary.parquet")
    for name, table in dropped_share_tables(cases).items():
        table.write_parquet(WORK_DIR / f"dropped_share_{name}.parquet")
    for name, columns in PLACEBOS.items():
        placebo(cases=cases, losses=losses, rng=rng, stratify_by=columns).write_parquet(
            WORK_DIR / f"placebo_{name}.parquet"
        )
    for name, flag in (("conditioned", True), ("outcome_free", False)):
        low_wind_slice(cases=cases, losses=losses, rng=rng, condition_on_power=flag).write_parquet(
            WORK_DIR / f"low_wind_slice_{name}.parquet"
        )
    zero_mass(cases=cases, arms=arms).write_parquet(WORK_DIR / "zero_mass.parquet")
    census = (
        cases.group_by("site")
        .agg(
            n=pl.len(),
            idle=pl.col("idle").sum(),
            outage=pl.col("outage").sum(),
            kept=(~pl.col("in_mask")).sum(),
            idle_reading=pl.col("idle_reading").sum(),
            idle_flag_with_generation=(pl.col("idle") & ~pl.col("idle_reading")).sum(),
        )
        .sort("site")
    )
    census.write_parquet(WORK_DIR / "census.parquet")
    (WORK_DIR / "gate.json").write_text(
        json.dumps({"mask_sha256": mask_hash, "reproduction_max_gap": gaps}, indent=1)
    )
    print(census)
    print("done", flush=True)


if __name__ == "__main__":
    main()

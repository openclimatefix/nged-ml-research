# ruff: noqa: E501, PERF401
r"""Score experiment 515's arms on the ten winter folds.

Rows (`experiment_515_winter_v1`): every forecast row of a fold's test month with a lead up to 240
hours, a valid time before 2026-07-01, a cleaned actual (flagged half-hours and the D15 and D18
start cut are NaN), and no labelled event (the target is outside every corrected event padded by 48
half-hours, and outside the 96 half-hours after it). Every scored arm must hold the row, so the
contrasts stay paired. The eight MVA-only Primaries and the six signed-MW Primaries are the gating
groups; D15 and D18 are report-only.

Arms are the quantile average of seeds 0, 1, and 2 (single seeds also feed the minimum detectable
effect). W0 is this experiment's `W0` forecasts, and experiment 513's `P0` for the folds in
`REUSED_BASE`. Quantiles are widened by 1.15 about the median, as experiment 510 does.

Slices, all from the stored ensemble mean at the valid time (experiment 437's `diag_temperature`
and `diag_csi`), never from an arm's columns: `cold_night` (temperature below 5 C, local 18:00 to
08:00), `cold_sunny` and `cold_overcast` (daylight half-hours of a complete local day whose mean
temperature is below 5 C and whose mean clear-sky index is at least 0.7 or at most 0.3), `warm_sunny`
and `warm_overcast` (the same with a day mean of at least 15.5 C), `christmas` (24 December to 2
January), and horizon bands `day1` (up to 24 h), `days2_3`, and `days4_10`.

Contrast: arm minus W0, so negative favours the arm. The interval resamples the folds (blocks) with
replacement, each fold carrying its 3-seed mean, at the Bonferroni level 1 / (2 x 20).

Commands: `rows` (build per-fold row metrics), `baseline` (minimum detectable effect and counts),
`stats` (contrasts, per-fold signs, tails, bands).
"""

import argparse
import math
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_score as score404  # ty: ignore[unresolved-import]
import e00435_tailcore as tail  # ty: ignore[unresolved-import]
import e00470_score as s470  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00515_fit as fit
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = fit.W
E513_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00513-lagless-demand-network-whole-day-price-shape-at-training-time"
    / "work"
    / "seed_preds"
    / "P0"
)
E437_TRAILING: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles"
    / "work"
    / "trailing"
)
SCORE_DIR: Final[Path] = W / "scores"
LEVELS: Final[tuple[str, ...]] = s470.LEVELS
WIDEN: Final[float] = 1.15
MAX_LEAD_HOURS: Final[float] = 240.0
COLD_C, WARM_C = 5.0, 15.5
SUNNY, OVERCAST = 0.7, 0.3
GROUPS: Final[dict[str, str]] = {"eight": "mva_eight", "six": "signed_six", "report": "report_only"}
BONFERRONI: Final[int] = 20
RESAMPLES: Final[int] = 20_000
POINTS: Final[float] = 100.0
SLICES: Final[tuple[str, ...]] = (
    "all", "cold_night", "cold_sunny", "cold_overcast", "warm_sunny", "warm_overcast",
    "christmas", "day1", "days2_3", "days4_10",
)  # fmt: skip
CORE: Final[tuple[str, ...]] = ("W-T", "W-S", "W-X", "W-null")
CONDITIONAL: Final[tuple[str, ...]] = ("W-ALL", "W-ALLnoDH")
ARMS: Final[tuple[str, ...]] = ("W0", *CORE, *CONDITIONAL)
SEEDS: Final[tuple[int, ...]] = fit.SEEDS


def arm_folder(*, arm: str, month: str) -> Path:
    """Return the folder holding an arm's seed forecasts for a fold."""
    if arm == "W0" and month in fit.REUSED_BASE:
        return E513_PREDS
    return W / "seed_preds" / arm


def available_arms(*, month: str) -> list[str]:
    """Return the arms (and W-seam) that have all three seeds for a fold."""
    out = []
    for arm in (*ARMS, fit.SEAM_ARM):
        folder = arm_folder(arm=arm, month=month)
        if all((folder / f"{month}_s{s}.parquet").exists() for s in SEEDS):
            out.append(arm)
    return out


def slice_context(*, keys: pl.DataFrame) -> pl.DataFrame:
    """Return the slice flags of (time_series_id, init, valid) keys from the stored ensemble mean."""
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        t = pl.read_parquet(E437_TRAILING / f"{row['site']}.parquet").select(
            "origin_us", "valid_us", "diag_temperature", "diag_csi"
        )
        t = (
            t.with_columns(
                valid_time=(pl.col("valid_us") * 1)
                .cast(pl.Datetime("us"))
                .dt.replace_time_zone("UTC"),
                power_fcst_init_time=pl.col("origin_us")
                .cast(pl.Datetime("us"))
                .dt.replace_time_zone("UTC"),
            )
            .with_columns(
                local=pl.col("valid_time").dt.convert_time_zone("Europe/London"),
                temperature=pl.col("diag_temperature").cast(pl.Float64),
                csi=pl.col("diag_csi").cast(pl.Float64),
            )
            .with_columns(local_date=pl.col("local").dt.date(), hour=pl.col("local").dt.hour())
        )
        day = t.group_by("power_fcst_init_time", "local_date").agg(
            pl.len().alias("n"),
            pl.col("temperature").mean().alias("day_t"),
            pl.col("csi").filter(pl.col("csi").is_finite()).mean().alias("day_csi"),
            pl.col("csi").is_finite().sum().alias("n_csi"),
        )
        t = t.join(day, on=["power_fcst_init_time", "local_date"], how="left")
        complete = (pl.col("n") >= 46) & (pl.col("n_csi") >= 8)
        daylight = pl.col("csi").is_finite()
        parts.append(
            t.select(
                pl.lit(int(row["time_series_id"])).cast(pl.Int32).alias("time_series_id"),
                "power_fcst_init_time",
                "valid_time",
                cold_night=(pl.col("temperature") < COLD_C)
                & ((pl.col("hour") >= 18) | (pl.col("hour") < 8)),
                cold_sunny=complete
                & daylight
                & (pl.col("day_t") < COLD_C)
                & (pl.col("day_csi") >= SUNNY),
                cold_overcast=complete
                & daylight
                & (pl.col("day_t") < COLD_C)
                & (pl.col("day_csi") <= OVERCAST),
                warm_sunny=complete
                & daylight
                & (pl.col("day_t") >= WARM_C)
                & (pl.col("day_csi") >= SUNNY),
                warm_overcast=complete
                & daylight
                & (pl.col("day_t") >= WARM_C)
                & (pl.col("day_csi") <= OVERCAST),
                christmas=((pl.col("local").dt.month() == 12) & (pl.col("local").dt.day() >= 24))
                | ((pl.col("local").dt.month() == 1) & (pl.col("local").dt.day() <= 2)),
            )
        )
    context = pl.concat(parts)
    return keys.join(
        context,
        on=["time_series_id", "power_fcst_init_time", "valid_time"],
        how="left",
        maintain_order="left",
    ).with_columns(pl.col(s).fill_null(value=False) for s in context.columns[3:])


def month_rows(
    *, month: str, clean: np.ndarray, excluded: np.ndarray, sites: pl.DataFrame
) -> pl.DataFrame | None:
    """Return a fold's per-row metrics of every arm, single seeds included, on the common rows."""
    arms = available_arms(month=month)
    if "W0" not in arms:
        return None
    first = pl.read_parquet(arm_folder(arm="W0", month=month) / f"{month}_s0.parquet")
    keys = first.select("time_series_id", "power_fcst_init_time", "valid_time")
    table = sites.select("time_series_id", "primary", "site")
    keys = keys.join(table, on="time_series_id", how="left", maintain_order="left")
    grid = c.prep422
    origin = (
        (keys["power_fcst_init_time"].dt.epoch("us").to_numpy() - grid.GRID_START_US)
        // grid.STEP_US
    ).astype(np.int64)
    target = (
        (keys["valid_time"].dt.epoch("us").to_numpy() - grid.GRID_START_US) // grid.STEP_US
    ).astype(np.int64)
    primary = keys["primary"].to_numpy().astype(np.int64)
    actual = clean[primary, target].astype(np.float64)
    lead = (target - origin) / 2.0
    keep = (
        np.isfinite(actual) & (lead > 0) & (lead <= MAX_LEAD_HOURS) & ~excluded[primary, target]
        & (keys["valid_time"].dt.epoch("us").to_numpy() < c.us(t=common.HOLDOUT))
    )  # fmt: skip
    base = keys.with_columns(
        actual=pl.Series(actual), lead_hours=pl.Series(lead),
        group=pl.Series(np.where(keys["site"].is_in(["D15", "D18"]).to_numpy(), "report_only", np.where(keys["site"].is_in(score404.SIX_SIGNED).to_numpy(), "signed_six", "mva_eight"))),
    ).filter(pl.Series(keep))  # fmt: skip
    base = base.join(slice_context(keys=base.select("time_series_id", "power_fcst_init_time", "valid_time")), on=["time_series_id", "power_fcst_init_time", "valid_time"], how="left", maintain_order="left")  # fmt: skip
    wide = base.select("time_series_id", "power_fcst_init_time", "valid_time", "actual", "lead_hours", "group", "cold_night", "cold_sunny", "cold_overcast", "warm_sunny", "warm_overcast", "christmas")  # fmt: skip
    cols: dict[str, np.ndarray] = {}
    every = np.ones(base.height, dtype=bool)
    jobs = dict.fromkeys(arms, SEEDS) | {f"{arm}-s{s}": (s,) for arm in arms for s in SEEDS}
    for name, seeds in jobs.items():
        arm = (
            name.rsplit("-s", 1)[0]
            if name.rsplit("-s", 1)[-1].isdigit() and name not in arms
            else name
        )
        frames = []
        for s in seeds:
            f = pl.read_parquet(arm_folder(arm=arm, month=month) / f"{month}_s{s}.parquet")
            frames.append(base.select("time_series_id", "power_fcst_init_time", "valid_time", "actual").join(f, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="left", maintain_order="left"))  # fmt: skip
        stacked = np.stack(
            [fr.select(list(LEVELS)).to_numpy().astype(np.float64) for fr in frames]
        ).mean(axis=0)
        frame = base.select("actual").hstack(
            pl.DataFrame(stacked, schema=list(LEVELS), orient="row")
        )
        ok = np.isfinite(stacked).all(axis=1)
        if name in arms:
            every &= ok
        frame = tail.widen(
            frame=frame.with_columns(pl.col(q).fill_null(0.0) for q in LEVELS), factor=WIDEN
        )
        for metric, values in s470.row_metrics(frame=frame).items():
            cols[f"{name}__{metric}"] = np.where(ok, values, np.nan)
    out = wide.hstack(pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float32) for k, v in cols.items()}))
    return out.with_columns(common_rows=pl.Series(every), month=pl.lit(month)).filter(
        pl.col("common_rows")
    )


def rows_command() -> None:
    """Write each fold's per-row metrics under `scores/`."""
    clean = c.clean_stack()
    padded = score404.real_event_mask()
    excluded = np.stack([score404.post_event_mask(padded=m) for m in padded])
    sites = prep.primaries().select("primary", "time_series_id", "site")
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    for month in fit.MONTHS:
        frame = month_rows(month=month, clean=clean, excluded=excluded, sites=sites)
        if frame is None:
            print("skipped", month, "(no W0 forecasts)", flush=True)
            continue
        frame.write_parquet(SCORE_DIR / f"rows_{month}.parquet")
        print(
            "wrote",
            month,
            frame.height,
            "rows;",
            len(available_arms(month=month)),
            "arms",
            flush=True,
        )


def load_rows() -> pl.DataFrame:
    """Return the rows of every scored fold."""
    paths = sorted(SCORE_DIR.glob("rows_*.parquet"))
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal")


def mask_of(*, frame: pl.DataFrame, group: str, slice_name: str) -> pl.Series:
    """Return the boolean row mask of a group and slice."""
    base = frame["group"] == GROUPS[group]
    lead = frame["lead_hours"]
    flag = {
        "all": pl.Series(np.ones(frame.height, dtype=bool)),
        "day1": lead <= 24,
        "days2_3": (lead > 24) & (lead <= 72),
        "days4_10": lead > 72,
    }.get(slice_name)
    return base & (flag if flag is not None else frame[slice_name])


def fold_sums(*, frame: pl.DataFrame, column: str, mask: pl.Series) -> dict[str, tuple[float, int]]:
    """Return per fold the sum and count of a metric over the masked rows."""
    out = {}
    for month in sorted(frame["month"].unique().to_list()):
        sel = mask & (frame["month"] == month)
        values = frame.filter(sel)[column].to_numpy()
        out[month] = (float(np.nansum(values)), int(np.isfinite(values).sum()))
    return out


def contrast(
    *, frame: pl.DataFrame, arm: str, ref: str, group: str, slice_name: str, metric: str = "crps"
) -> dict:
    """Return the pooled contrast (arm minus ref), a fold-block bootstrap interval, and the per-fold signs."""
    mask = mask_of(frame=frame, group=group, slice_name=slice_name)
    a = fold_sums(frame=frame, column=f"{arm}__{metric}", mask=mask)
    b = fold_sums(frame=frame, column=f"{ref}__{metric}", mask=mask)
    months = [m for m in a if a[m][1] > 0]
    sa = np.array([a[m][0] for m in months])
    sb = np.array([b[m][0] for m in months])
    n = np.array([a[m][1] for m in months], dtype=np.float64)
    point = (sa.sum() - sb.sum()) / n.sum()
    rng = np.random.default_rng(515)
    draws = rng.integers(0, len(months), size=(RESAMPLES, len(months)))
    boot = (sa[draws].sum(axis=1) - sb[draws].sum(axis=1)) / n[draws].sum(axis=1)
    alpha = 0.05 / (2 * BONFERRONI)
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    per_fold = (sa - sb) / n
    return {
        "arm": arm, "ref": ref, "group": group, "slice": slice_name, "metric": metric,
        "folds": len(months), "rows": int(n.sum()), "contrast": float(point),
        "lo": float(lo), "hi": float(hi), "negative_folds": int((per_fold < 0).sum()),
    }  # fmt: skip


def mde(*, frame: pl.DataFrame, group: str, slice_name: str, arms: list[str]) -> dict:
    """Return the minimum detectable effect of a 3-seed mean from the single-seed spread of the arms."""
    mask = mask_of(frame=frame, group=group, slice_name=slice_name)
    residuals = []
    for arm in arms:
        per_seed = [fold_sums(frame=frame, column=f"{arm}-s{s}__crps", mask=mask) for s in SEEDS]
        months = [m for m in sorted(per_seed[0]) if all(p[m][1] > 0 for p in per_seed)]
        values = np.array([[p[m][0] / p[m][1] for m in months] for p in per_seed]).T
        residuals.append(values - values.mean(axis=1, keepdims=True))
    resid = np.concatenate(residuals)
    sd = math.sqrt((resid**2).sum() / (len(resid) * (len(SEEDS) - 1)))
    return {"group": group, "slice": slice_name, "folds": resid.shape[0] // len(arms), "seed_sd": sd, "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(len(SEEDS))}  # fmt: skip


def baseline_command() -> None:
    """Print the 3-seed minimum detectable effect and the row counts, before any contrast."""
    frame = load_rows()
    arms = [a for a in ("W0", "W-null") if f"{a}__crps" in frame.columns]
    print(f"folds scored: {sorted(frame['month'].unique().to_list())}")
    records = [
        mde(frame=frame, group=g, slice_name=s, arms=arms)
        for g in GROUPS
        for s in ("all", "cold_night")
    ]
    table = pl.DataFrame(records)
    print(table)
    table.write_csv(SCORE_DIR / "mde.csv")
    counts = frame.group_by("month", "group").agg(pl.len().alias("rows")).sort("month", "group")
    print(counts)


def stats_command() -> None:
    """Write and print every contrast, tails, and horizon bands against W0."""
    frame = load_rows()
    arms = [a for a in (*CORE, *CONDITIONAL) if f"{a}__crps" in frame.columns]
    records = []
    for arm in arms:
        for g in GROUPS:
            for s in SLICES:
                records.append(contrast(frame=frame, arm=arm, ref="W0", group=g, slice_name=s))
            for s in ("all", "cold_night"):
                for metric in ("pin_q95", "pin_q99", "cov_q95", "cov_q99"):
                    records.append(
                        contrast(
                            frame=frame, arm=arm, ref="W0", group=g, slice_name=s, metric=metric
                        )
                    )
    for arm in ("W-T", "W-S", "W-X"):
        for g in GROUPS:
            for s in ("all", "cold_night"):
                records.append(contrast(frame=frame, arm=arm, ref="W-null", group=g, slice_name=s))
    if f"{fit.SEAM_ARM}__crps" in frame.columns:
        for g in GROUPS:
            for s in ("all", "cold_night"):
                records.append(
                    contrast(
                        frame=frame, arm=fit.SEAM_ARM, ref=fit.SEAM_PARENT, group=g, slice_name=s
                    )
                )
    table = pl.DataFrame(records)
    table.write_csv(SCORE_DIR / "contrasts.csv")
    with pl.Config(tbl_rows=400, tbl_cols=20, fmt_str_lengths=30):
        print(table.filter(pl.col("metric") == "crps"))
        print(table.filter(pl.col("metric") != "crps"))
    absolute = []
    for arm in ("W0", *arms):
        for g in GROUPS:
            for s in ("all", "cold_night"):
                sums = fold_sums(
                    frame=frame,
                    column=f"{arm}__crps",
                    mask=mask_of(frame=frame, group=g, slice_name=s),
                )
                absolute.append({"arm": arm, "group": g, "slice": s, "crps": sum(v[0] for v in sums.values()) / max(sum(v[1] for v in sums.values()), 1)})  # fmt: skip
    print(pl.DataFrame(absolute))
    folds = []
    for arm in ("W-T", "W-S", "W-X", "W-null"):
        for g in ("eight", "six"):
            mask = mask_of(frame=frame, group=g, slice_name="cold_night")
            a = fold_sums(frame=frame, column=f"{arm}__crps", mask=mask)
            b = fold_sums(frame=frame, column=" W0__crps".strip(), mask=mask)
            folds.append({"arm": arm, "group": g} | {m: round((a[m][0] - b[m][0]) / a[m][1], 3) for m in sorted(a)})
    with pl.Config(tbl_cols=20, tbl_width_chars=200):
        print(pl.DataFrame(folds))
    print(f"Bonferroni: primary family {BONFERRONI}; reported family 84")


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rows", "baseline", "stats"))
    args = parser.parse_args()
    {"rows": rows_command, "baseline": baseline_command, "stats": stats_command}[args.command]()


if __name__ == "__main__":
    main()

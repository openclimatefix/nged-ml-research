"""Experiment 435: the cold winter night slice, on stored experiment 425 forecasts (arms R and A).

The slice is experiment 441's `cold_winter_night` mask (built from ENS weather and the calendar,
never from realised weather), read from experiment 441's `work/masks.parquet`. The mask covers the
four screen folds of experiment 437 (origins of 2025-10, 2025-12, 2026-01, and 2026-03; 09:00 UTC
runs; leads above 0 and up to 240 hours), so the scored folds here are those four.

Per arm and metering group the script reports first the slice's row count, day count, and the
minimum detectable effect, then coverage at q95 and q99 on the slice per fold and pooled, raw and
widened by 1.15, with a 95% interval from a bootstrap that resamples whole local days (three folds
hold too few months for a month-block interval). It then fits table P with the extra class
`cold_night`: for each held-out fold, a factor per (group, lead band, tail) is fitted on the cold
nights of the other folds (class `cold_night`) and on the non-cold rows of the other folds (class
`rest`). Because only three folds hold cold nights, the fit is leave-one-fold-out, which is a
departure from the earlier-folds-only rule of the main table. The non-cold rows are also scored for
the quantile score (twice the mean pinball loss over the 11 stored levels), as the CRPS stand-in.
Output: `work/cold_nights_425.parquet` and `work/cold_nights_425_intervals.parquet`.

```bash
# PYTHONPATH as in e00425_score.py plus the folder of experiment 441
uv run --no-sync python ml-experiments/00435-*/e00435_cold_nights.py
```
"""

from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep  # ty: ignore[unresolved-import]
import e00425_score as score425  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import TAIL_LEVELS, WIDENING, lead_band_expr, widen
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
)
MASKS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00441-difficult-row-rescore-of-the-neural-demand-input-bundles"
    / "work"
    / "masks.parquet"
)
FOLDS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03")
ARMS: Final[tuple[str, ...]] = ("R", "A")
TARGETS: Final[dict[str, float]] = {"q95": 0.92, "q99": 0.975}
"""The slice's minimum acceptable coverage (nominal less the accepted tolerance)."""
LEVELS: Final[dict[str, float]] = {
    "q01": 0.01, "q05": 0.05, "q10": 0.10, "q25": 0.25, "q40": 0.40, "q50": 0.50,
    "q60": 0.60, "q75": 0.75, "q90": 0.90, "q95": 0.95, "q99": 0.99,
}  # fmt: skip
UPPER: Final[tuple[str, ...]] = ("q95", "q99")
N_DRAWS: Final[int] = 2000
SEED: Final[int] = 435


def load(*, arm: str) -> pl.DataFrame:
    """Return the arm's seed-averaged forecast with actuals, the mask, group, and lead band."""
    series = adapter.series_table()
    actuals = prep.cleaned_scoring_context(series=series)["actuals"].filter(
        pl.col("actual").is_finite()
    )
    raw = score425.load_frames(series=series, months=FOLDS, with_references=False)[f"{arm}_raw"]
    masks = pl.read_parquet(MASKS).select(
        "power_fcst_init_time", "valid_time", "label", "fold", "local_date", "cold_winter_night"
    )
    frame = (
        raw.join(actuals, on=["time_series_id", "valid_time"])
        .join(series.select("time_series_id", "label"), on="time_series_id")
        .join(masks, on=["power_fcst_init_time", "valid_time", "label"])
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
            group=pl.col("label").replace_strict(
                {lab: g for g, m in score425.SUBSETS.items() for lab in m}, default=None
            ),
        )
        .with_columns(lead_band=lead_band_expr())
    )
    return frame.filter(pl.col("group").is_not_null() & pl.col("lead_band").is_not_null())


def quantile_score(*, frame: pl.DataFrame) -> float:
    """Return twice the mean pinball loss over the 11 stored levels, in MW."""
    total = 0.0
    for name, tau in LEVELS.items():
        e = frame["actual"] - frame[name]
        total += float(np.where(e >= 0, tau * e, (tau - 1.0) * e).mean())
    return 2.0 * total / len(LEVELS)


def fit_factor(*, fit: pl.DataFrame, name: str) -> float:
    """Return the factor that makes coverage at `name` nominal on `fit` (see experiment 435's P)."""
    tau = LEVELS[name]
    spread = (fit[name] - fit["q50"]).abs()
    use = fit.filter(spread > 1e-6)
    spread = (use[name] - use["q50"]).abs()
    if name in UPPER:
        ratio, level = (use["actual"] - use["q50"]) / spread, tau
    else:
        ratio, level = (use["q50"] - use["actual"]) / spread, 1.0 - tau
    return float(ratio.quantile(level))  # ty: ignore[invalid-argument-type]


def apply(*, frame: pl.DataFrame, factors: dict[tuple[str, str, str], float]) -> pl.DataFrame:
    """Scale each tail quantile about the median by the factor of its (group, lead band)."""
    out = frame
    for name in TAIL_LEVELS:
        f = pl.DataFrame(
            [
                {"group": g, "lead_band": b, "factor": v}
                for (g, b, n), v in factors.items()
                if n == name
            ],
            schema={"group": pl.String, "lead_band": pl.String, "factor": pl.Float64},
        )
        out = (
            out.join(f, on=["group", "lead_band"], how="left")
            .with_columns(
                (pl.col("q50") + pl.col("factor").fill_null(1.0) * (pl.col(name) - pl.col("q50")))
                .cast(pl.Float64)
                .alias(name)
            )
            .drop("factor")
        )
    return out


def day_bootstrap(
    *, frame: pl.DataFrame, name: str, rng: np.random.Generator
) -> tuple[float, float]:
    """Return a 95% interval of coverage at `name`, resampling whole (fold, local day) blocks."""
    blocks = (
        frame.with_columns(hit=(pl.col("actual") <= pl.col(name)).cast(pl.Float64))
        .group_by("fold", "local_date")
        .agg(pl.col("hit").sum().alias("s"), pl.len().alias("n"))
    )
    s, n = blocks["s"].to_numpy(), blocks["n"].to_numpy()
    idx = rng.integers(0, len(s), size=(N_DRAWS, len(s)))
    draws = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    """Report the slice, its coverage, and table P with the cold-night class."""
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    intervals: list[dict] = []
    for arm in ARMS:
        data = load(arm=arm).with_columns(
            *[pl.col(c).cast(pl.Float64) for c in ("actual", *LEVELS)]
        )
        cold = data.filter(pl.col("cold_winter_night"))
        for group, part in cold.group_by("group"):
            days = part.select(pl.struct("fold", "local_date").n_unique()).item()
            for variant, frame in (("raw", part), ("w1.15", widen(frame=part, factor=WIDENING))):
                for name in UPPER:
                    low, high = day_bootstrap(frame=frame, name=name, rng=rng)
                    p = float((frame["actual"] <= frame[name]).mean())  # ty: ignore[invalid-argument-type]
                    intervals.append(
                        {
                            "arm": arm, "group": group[0], "variant": variant, "tail": name,
                            "rows": part.height, "days": days, "coverage": p,
                            "low": low, "high": high, "mde": 2.8 * (high - low) / 3.92,
                            "target": TARGETS[name],
                        }
                    )  # fmt: skip
        for held in FOLDS:
            others = data.filter(pl.col("fold") != held)
            test = data.filter(pl.col("fold") == held)
            for group in sorted(data["group"].unique()):
                for band in sorted(data["lead_band"].unique()):
                    for cls_name, mask in (("cold_night", True), ("rest", False)):
                        fit = others.filter(
                            (pl.col("group") == group)
                            & (pl.col("lead_band") == band)
                            & (pl.col("cold_winter_night") == mask)
                        )
                        part = test.filter(
                            (pl.col("group") == group)
                            & (pl.col("lead_band") == band)
                            & (pl.col("cold_winter_night") == mask)
                        )
                        if part.is_empty() or fit.height < 200:
                            continue
                        factors = {
                            (group, band, n): fit_factor(fit=fit, name=n) for n in TAIL_LEVELS
                        }
                        for variant, frame in (
                            ("raw", part),
                            ("w1.15", widen(frame=part, factor=WIDENING)),
                            ("P", apply(frame=part, factors=factors)),
                        ):
                            row = {
                                "arm": arm, "fold": held, "group": group, "lead_band": band,
                                "cls": cls_name, "variant": variant, "n": frame.height,
                                "qs": quantile_score(frame=frame),
                            }  # fmt: skip
                            for name in TAIL_LEVELS:
                                row[f"cov_{name}"] = float((frame["actual"] <= frame[name]).mean())  # ty: ignore[invalid-argument-type]
                            rows.append(row)
        print("done", arm, flush=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(WORK_DIR / "cold_nights_425.parquet")
    pl.DataFrame(intervals).write_parquet(WORK_DIR / "cold_nights_425_intervals.parquet")


if __name__ == "__main__":
    main()

"""Experiment 456, step 5: what the candidate steps change for the stored forecasts (no fit).

Reads the real-world forecasts of experiment 404 (arms A0-inj, the lagged head, and B-mix, the
50/50 mixture of the lagged and lagless forecasts) for the five scored months, and splits the rows
by whether the origin day or the target day lies inside a labelled event, inside a candidate step
of the scan (first 60 days of the step), both, or neither. For each row set it reports the share of
rows, each arm's mean score (points of capacity, as experiment 404's `pinball_points`), and B-mix's
lead (lagged minus B-mix; positive means B-mix is better) with a day-cluster bootstrap interval.
"""

from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00456-primary-demand-unmirrored-level-step-scan" / "work"
)
PRED_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00404-mdn-lagless-and-calibration-module-trust-weights"
    / "work"
    / "predictions"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99)
NAMES: Final[tuple[str, ...]] = tuple(f"q{round(100 * x):02d}" for x in LEVELS)
STEP_PERSISTENCE_DAYS: Final[int] = 60
N_RESAMPLES: Final[int] = 2000


def score_expr() -> pl.Expr:
    """Return experiment 404's score: twice the mass-weighted pinball loss, in points."""
    terms = []
    for level, name, mass in zip(LEVELS, NAMES, common.ATOM_MASSES, strict=True):
        diff = pl.col("actual") - pl.col(name)
        terms.append(float(mass) * pl.max_horizontal(level * diff, (level - 1.0) * diff))
    return 200.0 * pl.sum_horizontal(terms)


def step_mask(*, steps: pl.DataFrame, sites: list[str]) -> np.ndarray:
    """Return a Primaries-by-days boolean array: inside a candidate step's first 60 days."""
    mask = np.zeros((len(sites), 121_248 // 48), dtype=bool)
    for row in steps.iter_rows(named=True):
        if row["class"] == "unassessable":
            continue
        i = sites.index(row["site"])
        mask[
            i,
            row["start_day"] : min(row["end_day"], row["start_day"] + STEP_PERSISTENCE_DAYS - 1)
            + 1,
        ] = True
    return mask


def main() -> None:
    """Score the two arms on each row set and write the table."""
    steps = pl.read_csv(WORK_DIR / "candidate_steps.csv")
    sites = sorted(
        pl.read_csv(WORK_DIR / "spreads.csv")["site"].to_list(), key=lambda x: int(x[1:])
    )
    in_step = step_mask(steps=steps, sites=sites)
    frames = []
    for month in MONTHS:
        parts = {}
        for arm in ("A0-inj", "B-mix"):
            frame = pl.read_parquet(PRED_DIR / arm / f"{month}_s0_w-1.parquet")
            parts[arm] = frame.with_columns(score=score_expr())
        base = parts["A0-inj"].select(
            "primary", "origin", "target", "lead_hours", "win_real", "lag_real", "tgt_real",
            pl.col("score").alias("lagged"),
        )  # fmt: skip
        mixed = parts["B-mix"].select("primary", "origin", "target", pl.col("score").alias("mixed"))
        frames.append(
            base.join(mixed, on=["primary", "origin", "target"], how="inner").with_columns(
                month=pl.lit(month)
            )
        )
    data = pl.concat(frames)
    p = data["primary"].to_numpy()
    scan_flag = (
        in_step[p, data["target"].to_numpy() // 48] | in_step[p, data["origin"].to_numpy() // 48]
    )
    labelled = (data["win_real"] | data["lag_real"] | data["tgt_real"]).to_numpy()
    data = data.with_columns(
        scan=pl.Series(scan_flag), labelled=pl.Series(labelled), day=pl.col("target") // 48
    ).with_columns(
        row_set=pl.when(pl.col("labelled") & pl.col("scan")).then(pl.lit("labelled and scan step"))
        .when(pl.col("labelled")).then(pl.lit("labelled only"))
        .when(pl.col("scan")).then(pl.lit("scan step only (unlabelled)"))
        .otherwise(pl.lit("step-free")),
    )  # fmt: skip
    rng = np.random.default_rng(456)
    rows = []
    for name in (
        "step-free",
        "scan step only (unlabelled)",
        "labelled only",
        "labelled and scan step",
        "all rows",
        "not labelled",
    ):
        sub = (
            data if name == "all rows"
            else data.filter(~pl.col("labelled")) if name == "not labelled"
            else data.filter(pl.col("row_set") == name)
        )  # fmt: skip
        if sub.height == 0:
            continue
        per_day = sub.group_by("primary", "day", "month").agg(
            pl.col("lagged").sum().alias("lagged_sum"), pl.col("mixed").sum().alias("mixed_sum"),
            pl.len().alias("n"),
        )  # fmt: skip
        n = per_day["n"].to_numpy()
        lag, mix = per_day["lagged_sum"].to_numpy(), per_day["mixed_sum"].to_numpy()
        draws = []
        for _ in range(N_RESAMPLES):
            idx = rng.integers(0, len(n), len(n))
            draws.append((lag[idx].sum() - mix[idx].sum()) / n[idx].sum())
        lag_mean, mix_mean = float(lag.sum() / n.sum()), float(mix.sum() / n.sum())
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append(
            {
                "row_set": name, "rows": sub.height, "share_of_rows": sub.height / data.height,
                "lagged_score": lag_mean, "bmix_score": mix_mean,
                "bmix_lead": lag_mean - mix_mean,
                "lead_lo": float(lo), "lead_hi": float(hi),
            }
        )  # fmt: skip
    pl.DataFrame(rows).write_csv(WORK_DIR / "forecast_impact.csv")


if __name__ == "__main__":
    main()

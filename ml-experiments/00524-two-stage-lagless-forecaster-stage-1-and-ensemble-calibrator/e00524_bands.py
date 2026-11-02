# ruff: noqa: E501
"""Coarse-band contrasts of experiment 524 with an origin-day block bootstrap.

Reads the per-row scores that `e00524_score.py rows` wrote and, for each contrast, group (the eight
MVA-only Primaries, the six signed-MW Primaries with class `clean`), and coarse lead band (under 48
h, 48 to 120 h, 120 to 240 h), returns the mean CRPS difference (first arm minus second; negative
is better) with a 95% interval, a Bonferroni interval for the counted family, per-fold signs, and
the 5 whole-month blocks' range. The counted family is S2-F-B-nll, S2-J-0.4x, and S2-J-0.1x
against L-Ww: 3 arms x 2 groups x 3 bands = 18. Every other contrast is descriptive.
"""

import math
from typing import Final

import e00524_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

FOLDS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
BANDS: Final[dict[str, tuple[float, float]]] = {
    "under 48 h": (0, 48),
    "48-120 h": (48, 120),
    "120-240 h": (120, 240.001),
}
COUNTED: Final[tuple[tuple[str, str], ...]] = tuple(
    (a, "L-Ww") for a in ("S2-F-B-nll", "S2-J-0.4x", "S2-J-0.1x")
)
RESAMPLES: Final[int] = 4_000
SEED: Final[int] = 524


def intervals(
    *, total: np.ndarray, count: np.ndarray, level: float, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return the mean difference and its bootstrap interval, resampling whole origin days."""
    n = len(total)
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        pick = rng.integers(0, n, n)
        draws[i] = total[pick].sum() / count[pick].sum()
    tail = (1 - level) / 2
    return (
        float(total.sum() / count.sum()),
        float(np.quantile(draws, tail)),
        float(np.quantile(draws, 1 - tail)),
    )


def main() -> None:
    """Print and write the coarse-band contrasts."""
    score.setup()
    rows = pl.concat(
        [
            pl.read_parquet(score.s510.score470.SCORE_DIR / f"rows_{m}.parquet").with_columns(
                fold=pl.lit(m)
            )
            for m in FOLDS
        ]
    )
    rows = rows.filter(pl.col("main16"))
    rng = np.random.default_rng(SEED)
    records = []
    arms = list(score.s510.ARMS)
    pairs = [*score.s510.CONTRASTS]
    for first, second in pairs:
        if first not in arms or second not in [*arms, "K2-S1"]:
            continue
        for group, flt in (
            ("eight", pl.col("group") == "mva_eight"),
            ("six", (pl.col("group") == "signed_six") & (pl.col("cls") == "clean")),
        ):
            for band, (lo, hi) in BANDS.items():
                part = rows.filter(flt & (pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi))
                diff = part.select(
                    day=pl.col("origin").cast(pl.Int64) // 48,
                    fold="fold",
                    d=pl.col(f"{first}__crps") - pl.col(f"{second}__crps"),
                ).drop_nulls()
                by_day = diff.group_by("fold", "day").agg(
                    total=pl.col("d").sum(), count=pl.col("d").len()
                )
                total, count = by_day["total"].to_numpy(), by_day["count"].to_numpy().astype(float)
                counted = (first, second) in COUNTED
                mean, lo95, hi95 = intervals(total=total, count=count, level=0.95, rng=rng)
                _, lob, hib = (
                    intervals(total=total, count=count, level=1 - 0.05 / 18, rng=rng)
                    if counted
                    else (math.nan, math.nan, math.nan)
                )
                folds = diff.group_by("fold").agg(m=pl.col("d").mean()).sort("fold")["m"].to_numpy()
                records.append(
                    {"first": first, "second": second, "group": group, "band": band, "counted": counted, "n": diff.height, "days": len(total),
                     "difference": mean, "low95": lo95, "high95": hi95, "low_bonferroni18": lob, "high_bonferroni18": hib,
                     "fold_min": float(folds.min()), "fold_max": float(folds.max()), "folds_negative": int((folds < 0).sum())}
                )  # fmt: skip
    table = pl.DataFrame(records)
    table.write_csv(score.WORK_DIR / "scores" / "band_contrasts.csv")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    print(table.with_columns(pl.col(pl.Float64).round(3)).drop("n", "counted"))


if __name__ == "__main__":
    main()

"""Numbers for the Checks and caveats of experiment 683 (CPU only).

PYTHONPATH=ml-experiments/00668-*:ml-experiments/00683-* uv run --no-sync python \
    ml-experiments/00683-*/e00683_checks.py
Prints: fitted-scale ranges, clipped and fallback cells, arm 0 bias (the fitted shifts), the share
of cells where arm 2 is closer to 90% than arm 1, coverage in the first month after each refit
step for the affected Primaries, and scored row counts against experiment 668's.
"""

import polars as pl
from e00683_calibrate import EXPERIMENT_DIR, FIRST_SCORED_MONTH
from e00683_tables import FILES, load, means, period


def main() -> None:
    """Print the check numbers."""
    pl.Config.set_tbl_rows(100)
    fits = pl.concat(
        [pl.read_parquet(p) for p in sorted((EXPERIMENT_DIR / "fits").glob("*.parquet"))]
    )
    fits = fits.filter(pl.col("origin_month") >= FIRST_SCORED_MONTH)
    print(
        fits.group_by("file")
        .agg(
            cells=pl.len(),
            scale_min=pl.col("scale_arm2").min(),
            scale_median=pl.col("scale_arm2").median(),
            scale_max=pl.col("scale_arm2").max(),
            fallback=pl.col("fallback_scale_1").sum(),
            clipped2=pl.col("arm2_clipped").sum(),
            clipped3=pl.col("arm3_clipped").sum(),
            mean_shift=pl.col("shift_arm3_pct_capacity").mean(),
            abs_shift=pl.col("shift_arm3_pct_capacity").abs().mean(),
        )
        .with_columns(pl.col(pl.Float64).round(3))
    )
    cells = pl.read_csv(EXPERIMENT_DIR / "tables" / "cells.csv").filter(pl.col("period") == "dev")
    cells = cells.with_columns(gap=(pl.col("coverage") - 90).abs())
    wide = cells.pivot(on="arm", index=["file", "band"], values="gap")
    print(
        "cells where arm 2 is closer to 90 than arm 1:",
        wide.filter(pl.col("2") < pl.col("1")).height,
        "of",
        wide.height,
    )
    print(
        "cells where arm 2 is closer to 90 than arm 0:",
        wide.filter(pl.col("2") < pl.col("0")).height,
        "of",
        wide.height,
    )
    nra = load(file="primaries_NRA")
    steps = {  # month of the refit step, Primaries (steps outside 0.83 to 1.2 in e00683_refit.py)
        "2026-02": [8, 15, 17, 18, 24, 25],
        "2026-04": [11],
        "2026-09": [13],
    }
    for month, ids in steps.items():
        sub = nra.filter(pl.col("time_series_id").is_in(ids))
        by = means(frame=sub, by=["arm", "origin_month"]).filter(pl.col("arm").is_in([0, 1, 2]))
        print(f"\ncoverage of Primaries {ids}, step month {month}")
        print(
            by.select("arm", "origin_month", pl.col("coverage").round(1)).pivot(
                on="origin_month", index="arm", values="coverage"
            )
        )
    for file in FILES:
        d = period(frame=load(file=file), which="dev").filter(pl.col("arm") == 0)
        print(file, "scored half-hours 2025-12 to 2026-06:", int(d["n"].sum()))
    for file in FILES:
        d = load(file=file).filter(pl.col("arm") == 0)
        print(file, "scored half-hours 2025-12 to 2026-10:", int(d["n"].sum()))


if __name__ == "__main__":
    main()

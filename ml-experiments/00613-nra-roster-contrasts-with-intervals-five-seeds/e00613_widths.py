# ruff: noqa: D103, E501
"""Experiment 613: the width factors that put the class 2 candidate's band at K1's width (CPU).

For each month, the factor is the mean q05-q95 width of K1 (`k1_d5`, system `K1`) divided by the mean
width of the candidate's own band (`c2_e3`, system `K2`), on the injected events and phases both
networks were scored on (24-120 h), weighted by the rows of each cell. Two files are written:

- `c2_width_factors.csv`: each month's own factor, as experiment 612 did (in sample: the factor is
  fitted on the rows it is later scored on);
- `c2_width_factors_loo.csv`: each month's factor from the *other* months only (the mean of their
  factors), so that the scored months do not choose their own width.

Usage: python e00613_widths.py [--suffix _s01]
"""

import argparse

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00613-nra-roster-contrasts-with-intervals-five-seeds"
_p = argparse.ArgumentParser()
_p.add_argument("--suffix", default="", help="score folders of a seed subset, such as _s01")
SFX = _p.parse_args().suffix
SCORE = ML_EXPERIMENTS_DATA_DIR / NAME / "score"
KEYS = ["month", "pass", "primary", "phase", "length_days", "depth", "sign"]


def cells(*, arm: str, system: str) -> pl.DataFrame:
    return (
        pl.read_csv(SCORE / arm / "bank_event_phase.csv")
        .filter((pl.col("system") == system) & (pl.col("band") == "24-120 h"))
        .select(*KEYS, "n_rows", "width90")
    )


def main() -> None:
    k1, c2 = cells(arm="k1_d5" + SFX, system="K1"), cells(arm="c2_e3" + SFX, system="K2")
    j = k1.join(c2, on=KEYS, suffix="_c")
    out = (
        j.group_by("month")
        .agg(
            ((pl.col("width90") * pl.col("n_rows")).sum() / pl.col("n_rows").sum()).alias("k1"),
            ((pl.col("width90_c") * pl.col("n_rows_c")).sum() / pl.col("n_rows_c").sum()).alias("c2"),
            pl.len().alias("n_cells"),
        )
        .with_columns(factor=pl.col("k1") / pl.col("c2"))
        .sort("month")
    )
    print(out)
    out.select("month", "factor").write_csv(SCORE / f"c2_width_factors{SFX}.csv")
    assert out.height > 1
    f = dict(out.select("month", "factor").iter_rows())
    loo = [
        {"month": m, "factor": sum(v for k, v in f.items() if k != m) / (len(f) - 1)} for m in f
    ]
    pl.DataFrame(loo).write_csv(SCORE / f"c2_width_factors_loo{SFX}.csv")


if __name__ == "__main__":
    main()

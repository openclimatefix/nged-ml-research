# ruff: noqa: E501, D103
"""Experiment 591: the write-up's checks.

Error of the median forecast beside each CRPS contrast, and the median minus the mean-of-quantiles
forecast per Primary group.
"""

import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00565_design6 as d6  # ty: ignore[unresolved-import]
import e00572_score as sc  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00591-shared-lag-encoder-auxiliary-shift-depth-loss"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME
ARMS = {
    "A0": (ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6" / "c1" / "work", "P5"),
    **{f"A{i}": (ML_EXPERIMENTS_DATA_DIR / NAME / f"A{i}" / "work", "P5") for i in range(1, 5)},
}
LEVELS = ["q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95"]
TAILS = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}


def main() -> None:
    sites = build470.site_map()
    rows = []
    for month in d6.MONTHS:
        base = pl.read_parquet(
            d6.W559 / "scores" / f"rows_{month}.parquet",
            columns=[*d6.KEYS, "group", "lead_hours", "main16"],
        )
        act = score454.month_base(month=month).select(*d6.KEYS, "actual")
        base = base.join(act, on=d6.KEYS, how="left", maintain_order="left")
        for arm, (root, folder) in ARMS.items():
            for seed in (0, 1):
                fr = build470.pool_stored(
                    folder=root / "seed_preds" / folder, month=month, sites=sites, seeds=(seed,)
                )
                j = base.join(
                    fr.select(*d6.KEYS, *LEVELS, "q01", "q99"),
                    on=d6.KEYS,
                    how="inner",
                    maintain_order="left",
                ).filter(pl.col("q50").is_not_null() & pl.col("actual").is_not_null())
                mean_q = sum(j[q] for q in LEVELS) / len(LEVELS)
                rows.append(
                    j.select(
                        pl.lit(arm).alias("arm"), pl.lit(seed).alias("seed"), "group",
                        band=sc.band_of(pl.col("lead_hours")),
                        ae=(100 * (j["q50"] - j["actual"]).abs()),
                        gap=100 * (j["q50"] - mean_q),
                        **{f"pin_{q}": 100 * pl.max_horizontal(tau * (j["actual"] - j[q]), (tau - 1) * (j["actual"] - j[q])) for q, tau in TAILS.items()},
                        **{f"below_{q}": (j["actual"] < j[q]).cast(pl.Float64) for q in TAILS},
                    )
                )  # fmt: skip
    t = pl.concat(rows)
    g = (
        t.group_by("arm", "group", "band")
        .agg(
            mae_q50=pl.col("ae").mean(),
            median_minus_quantile_mean=pl.col("gap").mean(),
            **{c: pl.col(c).mean() for c in t.columns if c.startswith(("pin_", "below_"))},
        )
        .sort("group", "band", "arm")
    )
    g.write_csv(OUT / "checks_median_error.csv")
    g.filter(pl.col("band") == "24-120h").write_csv(OUT / "checks_tails_24_120h.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=160):
        print(
            g.filter(pl.col("group").str.contains("eight|six") & (pl.col("band") == "24-120h"))
            .drop("median_minus_quantile_mean")
            .with_columns(pl.col(pl.Float64).round(3))
        )


if __name__ == "__main__":
    main()

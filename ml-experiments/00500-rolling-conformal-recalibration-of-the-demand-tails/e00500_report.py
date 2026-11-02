# ruff: noqa: E501
"""Experiment 500: print every number the write-up quotes (run after the scoring commands)."""

import json

import e00483_tails as tails  # ty: ignore[unresolved-import]
import e00500_conformal as conformal  # ty: ignore[unresolved-import]
import polars as pl

ARMS = ["raw", "w1.15", "P", "two-constant", "RC7", "RC14", "RC28", "RC14+w1.05", "RC28-bias"]
COLS = [
    "n",
    "crps",
    "cov_q01",
    "cov_q05",
    "cov_q95",
    "cov_q99",
    "pin_q01",
    "pin_q05",
    "pin_q95",
    "pin_q99",
]


def show(title: str, frame: pl.DataFrame) -> None:
    """Print a titled table."""
    print(f"\n== {title}")
    with pl.Config(tbl_rows=100, tbl_width_chars=250, tbl_cols=30):
        print(frame.with_columns(pl.col(pl.Float64).round(4)))


def main() -> None:
    """Print the tables."""
    scores = pl.read_parquet(conformal.WORK / "scores.parquet")
    five = scores.filter(pl.col("rowset") == "five")
    for group in ("signed_six", "mva_eight", "report_only"):
        for slice_name in ("clean", "clean_12_36h", "clean_cold", "clean_xmas", "real_event_nra"):
            part = five.filter(
                (pl.col("group") == group)
                & (pl.col("slice") == slice_name)
                & (pl.col("by") == "group")
                & pl.col("arm").is_in(ARMS)
            )
            if part.height:
                show(
                    f"{group} / {slice_name} (five folds)",
                    part.select("arm", *COLS).sort(
                        pl.col("arm").replace_strict({a: i for i, a in enumerate(ARMS)})
                    ),
                )
    four = scores.filter(
        (pl.col("rowset") == "four")
        & (pl.col("group") == "signed_six")
        & (pl.col("slice") == "clean")
        & (pl.col("by") == "group")
    )
    show(
        "signed_six / clean (four folds, adds earlier-only)", four.select("arm", *COLS).sort("arm")
    )
    bands = five.filter(
        (pl.col("group") == "signed_six")
        & (pl.col("slice") == "clean")
        & (pl.col("by") == "group+lead_band")
        & pl.col("arm").is_in(["raw", "P", "RC28", "RC28-bias"])
    )
    show(
        "six by lead band",
        bands.select(
            "arm", "lead_band", "n", "crps", "cov_q05", "cov_q95", "cov_q99", "pin_q99"
        ).sort("lead_band", "arm"),
    )
    months = five.filter(
        (pl.col("group") == "signed_six")
        & (pl.col("slice") == "clean")
        & (pl.col("by") == "group+month")
        & pl.col("arm").is_in(["raw", "P", "RC28", "RC28-bias"])
    )
    show(
        "six by fold",
        months.select("arm", "month", "n", "cov_q01", "cov_q05", "cov_q95", "cov_q99").sort(
            "month", "arm"
        ),
    )
    spread = months.group_by("arm").agg(
        *[(pl.col(c).max() - pl.col(c).min()).alias(f"range_{c}") for c in ("cov_q95", "cov_q99")],
        *[
            pl.col(c).is_between(*conformal_range).sum().alias(f"folds_in_{c}")
            for c, conformal_range in (("cov_q95", (0.93, 0.97)), ("cov_q99", (0.98, 0.995)))
        ],
    )
    show("coverage stability across the five folds (six)", spread.sort("arm"))
    contrasts = pl.read_csv(conformal.WORK / "contrasts.csv")
    show(
        "counted contrasts against raw (six, five folds)",
        contrasts.filter(pl.col("counted")).select(
            "arm",
            "metric",
            "difference",
            "low_bonferroni",
            "high_bonferroni",
            "relative",
            "rel_low_bonferroni",
            "rel_high_bonferroni",
            "rel_low_bonferroni_14d_blocks",
            "rel_high_bonferroni_14d_blocks",
        ),
    )
    show(
        "post hoc and other contrasts against raw (six, five folds)",
        contrasts.filter(
            (pl.col("rowset") == "five")
            & (pl.col("group") == "signed_six")
            & (pl.col("against") == "raw")
            & ~pl.col("counted")
        ).select(
            "arm",
            "metric",
            "difference",
            "low_95",
            "high_95",
            "low_bonferroni",
            "high_bonferroni",
            "relative",
        ),
    )
    show(
        "descriptive contrasts against other recalibrations (six, 95%)",
        contrasts.filter((pl.col("rowset") == "five") & (pl.col("against") != "raw")).select(
            "arm",
            "against",
            "metric",
            "difference",
            "low_95",
            "high_95",
            "relative",
            "rel_low_95",
            "rel_high_95",
        ),
    )
    show(
        "contrasts against raw at the eight, 95% (post hoc bias arms)",
        contrasts.filter(
            (pl.col("rowset") == "five")
            & (pl.col("group") == "mva_eight")
            & (pl.col("against") == "raw")
            & pl.col("arm").is_in(["RC28", "RC28-bias", "P", "two-constant"])
        ).select("arm", "metric", "difference", "low_95", "high_95", "relative"),
    )
    best = five.filter(
        (pl.col("group") == "signed_six")
        & (pl.col("slice") == "clean")
        & (pl.col("by") == "group")
        & pl.col("arm").is_in(ARMS)
    )["crps"].min()
    print("\nbest CRPS among the nine arms (six, clean, five folds):", best)
    print(
        "\n== diagnostics:",
        json.dumps(json.loads((conformal.WORK / "diagnostics.json").read_text())),
    )
    print("\n== posthoc.csv (median MAE, cold nights)")
    print(
        pl.read_csv(conformal.WORK / "posthoc.csv").select(
            "arm", "group", "median_mae", "cold_cov_q95", "cold_cov_q99", "cold_crps"
        )
    )
    # Buffer months per fold (28-day buffer).
    sh = pl.read_parquet(conformal.shifts_path(days=28))
    for fold in conformal.FOLDS:
        origins = tails.load(arm="A-pool").filter(pl.col("month") == fold)["origin"].unique()
        cells = sh.filter(pl.col("origin").is_in(origins))
        lo, hi = cells["init_min"].min(), cells["init_max"].max()
        print(
            f"{fold}: 28-day buffer holds forecasts initiated at grid index {lo} to {hi}; smallest cell {cells['n'].min()}"
        )
    curtailment = pl.read_csv(conformal.WORK / "curtailment_bootstrap.csv")
    print("\n== curtailment columns:", curtailment.columns)
    show("curtailment totals", curtailment.filter(~pl.col("item").str.contains("minus")))


def per_primary() -> None:
    """Print RC28-bias minus raw per Primary at the six (leave-one-Primary-out check)."""
    raw = conformal.load_arm(arm="raw")
    bias = conformal.load_arm(arm="RC28-bias")
    rc28 = conformal.load_arm(arm="RC28")
    rows = []
    for primary in sorted(
        raw.filter(pl.col("group") == "signed_six")["primary"].unique().to_list()
    ):
        out = {"primary": primary}
        for name, frame in (("raw", raw), ("RC28", rc28), ("RC28-bias", bias)):
            part = tails.slices(frame=frame)["clean"].filter(pl.col("primary") == primary)
            s = tails.summarise(frame=part, by=["group"]).row(0, named=True)
            out |= {
                f"{name}_crps": s["crps"],
                f"{name}_cov_q95": s["cov_q95"],
                f"{name}_pin_q99": s["pin_q99"],
            }
        rows.append(out)
    table = pl.DataFrame(rows).with_columns(
        d_crps=pl.col("RC28-bias_crps") - pl.col("raw_crps"),
        d_pin_q99=pl.col("RC28-bias_pin_q99") - pl.col("raw_pin_q99"),
    )
    show("per Primary at the six: RC28-bias minus raw", table)


def dropped() -> None:
    """Print how many flagged exact-zero half-hours fall among each fold's scored valid times."""
    import e00407_score as score407  # ty: ignore[unresolved-import]
    import e00470_build as build470  # ty: ignore[unresolved-import]

    flagged = pl.read_parquet(score407.FLAGGED_KEYS).select("time_series_id", "valid_time")
    sites = build470.site_map().select("time_series_id")
    flagged = flagged.join(sites, on="time_series_id")
    base = conformal.load_base()
    for fold in conformal.FOLDS:
        part = base.filter(pl.col("month") == fold)
        keys = part.select("time_series_id", "valid_time").unique()
        n = (
            flagged.join(keys.select("valid_time").unique(), on="valid_time")
            .join(keys.select("time_series_id").unique(), on="time_series_id")
            .height
        )
        print(
            f"{fold}: {n} flagged exact-zero half-hours (16 Primaries) within the fold's scored valid times; "
            f"{keys.height} scored (series, valid time) pairs"
        )


if __name__ == "__main__":
    main()
    per_primary()
    dropped()

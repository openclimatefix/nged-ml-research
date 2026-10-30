"""Score experiment 387's real-data arms under demand-v1 (copies experiment 381's scoring calls).

The arms are the plain quantile model, F0 (unfolded custom objective), and F2 (folded pinball,
kappa 1.0) at Q0 = 0, 2, and 4 points of capacity, refitted monthly, scored on the 73 runs of
`fold_a_every5d_v1` (the runs the fits forecast). Prints the reference gate against the stored
monthly member, the pre-registered contrasts (99.75% for k = 4, 20,000 resamples), per-Primary
tables, tails, sunny bias, and the load-shift slices.

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
        uv run --no-sync python ml-experiments/00387-*/e00387_score.py
"""

import sys
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00387-xgboost-folded-latent-loss-at-mva-only-primaries" / "work"
)
ARMS: Final[dict[str, str]] = {
    "plain": "plain",
    "f0": "f0",
    "f2_q0": "f2_k1_m0_q0",
    "f2_q2": "f2_k1_m0_q2",
    "f2_q4": "f2_k1_m0_q4",
}
LEVEL: Final[float] = 100 * (1 - 0.01 / 4)
RESAMPLES: Final[int] = 20_000
HEADLINE: Final[str] = base.HEADLINE_BAND
MVA_ONLY_SITES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "f2_q2", "plain"),
    ("C3", "f2_q2", "f0"),
    ("Q0=0 vs 2", "f2_q0", "f2_q2"),
    ("Q0=4 vs 2", "f2_q4", "f2_q2"),
    ("F0 vs plain", "f0", "plain"),
)
SELECTIONS: Final[dict[str, pl.Expr]] = {
    "MVA-only with D8": pl.col("group") == "MVA-only",
    "MVA-only without D8": (pl.col("group") == "MVA-only") & (pl.col("label") != "D8"),
    "D12 alone": pl.col("label") == "D12",
}


def arm_frame(*, folder: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's forecast in MW over the MVA-only Primaries."""
    files = sorted((WORK / "preds" / "real" / folder).glob("*.parquet"))
    stored = (
        pl.concat([pl.read_parquet(f) for f in files])
        .select("time_series_id", "power_fcst_init_time", "valid_time", *s381.QUANTILE_COLUMNS)
        .filter(pl.col("power_fcst_init_time") >= base.SCORED_FIRST)
    )
    return adapter.forecast_frame(stored=stored, series=series, unit="fraction")


def main() -> None:
    """Print every number the write-up quotes."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    series_all = adapter.series_table()
    series = series_all.filter(pl.col("label").is_in(MVA_ONLY_SITES))
    ctx = s381.build_context(series=series)
    arms = {name: arm_frame(folder=f, series=series) for name, f in ARMS.items()}
    stored = s381.stored_frame(name="only_M", series=series)
    gate = {"plain": arms["plain"], "stored_monthly": stored}
    joined = (
        gate["plain"]
        .select("time_series_id", "power_fcst_init_time", "valid_time", "q50")
        .join(
            stored.select("time_series_id", "power_fcst_init_time", "valid_time", "q50"),
            on=["time_series_id", "power_fcst_init_time", "valid_time"],
            suffix="_stored",
        )
    )
    gap = (joined["q50"] - joined["q50_stored"]).abs()
    print(
        f"GATE (refit plain vs stored only_M): {joined.height} rows, mean |gap| {gap.mean():.4f} MW"
    )
    table, scored, rows = s381.score_arms(arms=arms, series=series, ctx=ctx)
    print("\n== absolute CRPS, days 1 to 10, per Primary ==")
    print(
        pl.concat(
            [
                t.filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE).select(
                    "label", "group", pl.lit(n).alias("arm"), "value"
                )
                for n, t in scored.items()
            ]
        )
        .pivot(on="arm", index=["label", "group"], values="value")
        .sort("group", "label")
    )
    for name, treat, ref in CONTRASTS:
        print(f"\n== {name}: {treat} minus {ref}, CRPS, points, level {LEVEL:.2f}% ==")
        for sl in (HEADLINE, "sunny_unit_shape", "sunny_april_june"):
            where_slice = pl.col(HEADLINE) if sl == HEADLINE else pl.col(sl) & pl.col(HEADLINE)
            for gname, where in SELECTIONS.items():
                r = metrics.month_bootstrap_contrast(
                    treatment=rows[treat],
                    reference=rows[ref],
                    metric="crps",
                    where=where & where_slice,
                    level_percent=LEVEL,
                    n_resamples=RESAMPLES,
                    slice_table=table,
                )
                print(
                    f"{sl:18s} {gname:20s} {r['difference']:+.3f} "
                    f"({r['lower']:+.3f} to {r['upper']:+.3f}) n={r['n_half_hours']}"
                )
        print("per Primary, 95%, days 1 to 10")
        by_primary = metrics.reported_contrast_by_primary(
            treatment=rows[treat],
            reference=rows[ref],
            metric="crps",
            where=pl.col(HEADLINE),
            slice_table=table,
        )
        print(by_primary)
        (WORK / "scores").mkdir(exist_ok=True)
        by_primary.with_columns(contrast=pl.lit(name)).write_parquet(
            WORK / "scores" / f"by_primary_{name.replace(' ', '_').replace('=', '')}.parquet"
        )
    print("\n== leave-one-out, C1 (f2_q2 minus plain), MVA-only, 95% ==")
    for left_out in MVA_ONLY_SITES:
        keep = pl.col("label").is_in([x for x in MVA_ONLY_SITES if x != left_out])
        r = metrics.month_bootstrap_contrast(
            treatment=rows["f2_q2"],
            reference=rows["plain"],
            metric="crps",
            where=keep & pl.col(HEADLINE),
            slice_table=table,
        )
        print(
            f"without {left_out:4s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f})"
        )
    tails = ["pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99",
             "coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99",
             "twcrps", "mae_median"]  # fmt: skip
    print("\n== tails, twCRPS, MAE of the median: mean over Primaries, days 1 to 10 ==")
    for name in arms:
        t = scored[name].filter(pl.col("slice") == HEADLINE, pl.col("metric").is_in(tails))
        print(name)
        print(t.group_by("metric").agg(pl.col("value").mean()).sort("metric"))
    print("\n== D12 tails and MAE ==")
    for name in arms:
        t = scored[name].filter(
            pl.col("slice") == HEADLINE, pl.col("metric").is_in(tails), pl.col("label") == "D12"
        )
        print(name, dict(zip(t["metric"], t["value"].round(3), strict=True)))
    print("\n== slice CRPS, MVA-only mean over Primaries: plain, f0, f2_q2 ==")
    for sl in ("sunny_unit_shape", "sunny_april_june", "winter_nights", "cold_nights",
               "economy_7_step", "event_rows", "event_following_7_days"):  # fmt: skip
        parts = [
            rows[n]
            .filter(pl.col(sl), pl.col(HEADLINE))
            .group_by("group")
            .agg(pl.col("crps").mean().alias(n), n_half_hours=pl.len())
            for n in ("plain", "f0", "f2_q2")
        ]
        out = parts[0]
        for p in parts[1:]:
            out = out.join(p.drop("n_half_hours"), on="group")
        print(sl)
        print(out)
    print("\n== sunny bias (median minus actual), points of capacity ==")
    s381.diagnose.__globals__["arm_frame"] = lambda **kw: arms[kw["arm"]]
    s381.diagnose(series=series, ctx=ctx, arm_names=["plain", "f0", "f2_q2"])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

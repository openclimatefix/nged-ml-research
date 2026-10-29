"""Contrasts 1 to 4 of experiment 393, printed for the write-up, from the rows of `e00393_build`.

DATA_PATH_INTERNAL=... uv run --no-sync python \
        ml-experiments/00393-gam-mva-fold-data-analysis/e00393_analysis.py
"""

from datetime import UTC, datetime

import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.eval_protocol.metrics import QUANTILE_COLUMNS
from studies.eval_protocol.slices import KEY_COLUMNS, MVA_ONLY_LABELS, SIGNED_MW_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = ML_EXPERIMENTS_DATA_DIR / "00393-gam-mva-fold-data-analysis" / "work"
SCORED_FIRST = datetime(2025, 7, 1, tzinfo=UTC)
BAND = "lead_days_1_10"
COUNTABLE = (*SIGNED_MW_LABELS, *MVA_ONLY_LABELS)
NEAR_ZERO_FRACTION = 0.10
RNG = np.random.default_rng(393)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Spearman rank correlation of two vectors (average ranks)."""
    rx = pl.Series(x).rank().to_numpy()
    ry = pl.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n: int = 10_000) -> tuple[float, float]:
    """Return a 95% interval for the Spearman correlation, resampling Primaries."""
    out = []
    for _ in range(n):
        idx = RNG.integers(0, len(x), len(x))
        if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
            continue
        out.append(spearman(x[idx], y[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def contrast(
    *,
    treat: pl.DataFrame,
    ref: pl.DataFrame,
    table: pl.DataFrame,
    where: pl.Expr,
    headline: bool = False,
) -> str:
    """Return a formatted paired month-bootstrap contrast of crps, treat minus ref."""
    fn = metrics.headline_month_bootstrap_contrast if headline else metrics.month_bootstrap_contrast
    r = fn(treatment=treat, reference=ref, metric="crps", where=where, slice_table=table)
    return (
        f"{r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) "
        f"n={r['n_half_hours']}, months={r['n_months']}"
    )


def folded_rows(
    *,
    name: str,
    series: pl.DataFrame,
    actuals: pl.DataFrame,
    table: pl.DataFrame,
    thresholds: pl.DataFrame,
) -> pl.DataFrame:
    """Return scored rows of an arm with |quantiles| sorted and |actual| as the target."""
    forecast = pl.read_parquet(WORK / f"forecast_{name}.parquet").filter(
        pl.col("time_series_id").is_in(series["time_series_id"])
    )
    folded = np.sort(
        np.abs(forecast.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)), axis=1
    )
    forecast = forecast.with_columns(
        [pl.Series(c, folded[:, i].astype(np.float32)) for i, c in enumerate(QUANTILE_COLUMNS)]
    )
    return metrics.scored_rows(
        forecast=forecast,
        actuals=actuals.with_columns(pl.col("actual").abs()),
        series=series,
        thresholds=thresholds,
        slice_table=table,
    )


def main() -> None:  # noqa: PLR0915
    """Print every number the write-up quotes."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    series = pl.read_parquet(WORK / "series.parquet")
    actuals = pl.read_parquet(WORK / "actuals.parquet")
    thresholds = pl.read_parquet(WORK / "thresholds.parquet")
    table = pl.read_parquet(WORK / "slice_table.parquet")
    arms = ("stack", "bg2", "manual_heuristic", "climatology")
    rows = {a: pl.read_parquet(WORK / f"rows_{a}.parquet") for a in arms}
    cap = series.select("time_series_id", "label", "capacity_mw")
    peak = (
        actuals.filter(pl.col("valid_time") < SCORED_FIRST)
        .group_by("time_series_id")
        .agg(peak_mw=pl.col("actual").abs().max())
        .join(cap, on="time_series_id")
        .with_columns(peak_points=pl.col("peak_mw") / pl.col("capacity_mw") * 100.0)
        .select("time_series_id", "peak_points")
    )
    base = rows["stack"].select(
        *KEY_COLUMNS,
        "label",
        "group",
        "actual_points",
        "lead_hours",
        *[c for c in rows["stack"].columns if c.startswith("lead_day")],
    )
    base = base.join(peak, on="time_series_id").with_columns(
        near_zero=pl.col("actual_points").abs() < NEAR_ZERO_FRACTION * pl.col("peak_points")
    )
    keys = list(KEY_COLUMNS)

    def paired(ref: str, col: str = "crps") -> pl.DataFrame:
        return (
            rows["stack"]
            .select(*keys, pl.col(col).alias("s"))
            .join(rows[ref].select(*keys, pl.col(col).alias("r")), on=keys, validate="1:1")
            .join(
                base.select(
                    *keys,
                    "label",
                    "group",
                    "near_zero",
                    "lead_hours",
                    BAND,
                    "lead_days_8_10",
                    "lead_day_1",
                    "lead_day_2",
                    "lead_days_3_4",
                ),
                on=keys,
                validate="1:1",
            )
            .with_columns(d=pl.col("s") - pl.col("r"))
        )

    pb = paired("bg2")
    pm = paired("manual_heuristic")
    countable = pl.col("label").is_in(COUNTABLE)

    # ---------- Absolute scores and the demand-v2 floor ----------
    print("\n== CRPS, days 1 to 10, per Primary, four arms; floor test ==")
    absolute = pl.concat(
        [
            rows[a]
            .filter(pl.col(BAND))
            .group_by("label", "group")
            .agg(
                crps=pl.col("crps").mean(),
                mae=pl.col("mae_median").mean(),
                n=pl.len(),
                cov05=pl.col("coverage_q05").mean(),
                cov95=pl.col("coverage_q95").mean(),
                cov99=pl.col("coverage_q99").mean(),
            )
            .with_columns(arm=pl.lit(a))
            for a in arms
        ]
    )
    wide = (
        absolute.pivot(on="arm", index=["label", "group"], values="crps")
        .with_columns(
            stack_over_bg2=pl.col("stack") / pl.col("bg2"),
            floor_ok=(pl.col("stack") <= pl.col("manual_heuristic"))
            & (pl.col("stack") <= 1.10 * pl.col("bg2")),
        )
        .sort("group", "label")
    )
    print(wide)
    print("\n== coverage and MAE of the median, stack and bg2, days 1 to 10 ==")
    print(
        absolute.filter(pl.col("arm").is_in(["stack", "bg2"]))
        .select("label", "group", "arm", "mae", "cov05", "cov95", "cov99")
        .sort("label", "arm")
    )

    # ---------- Contrast 2: near-zero share ----------
    print("\n== Contrast 2: near-zero share and the gap inside and outside, days 1 to 10 ==")
    c2 = (
        pb.filter(pl.col(BAND))
        .join(pm.filter(pl.col(BAND)).select(*keys, dm=pl.col("d")), on=keys)
        .group_by("label", "group")
        .agg(
            share_nz=pl.col("near_zero").mean(),
            gap_bg2=pl.col("d").mean(),
            gap_bg2_nz=pl.col("d").filter(pl.col("near_zero")).mean(),
            gap_bg2_other=pl.col("d").filter(~pl.col("near_zero")).mean(),
            gap_mh=pl.col("dm").mean(),
            n_nz=pl.col("near_zero").sum(),
        )
        .sort("group", "label")
    )
    print(c2)
    c2.write_parquet(WORK / "contrast2_table.parquet")
    c14 = c2.filter(pl.col("label").is_in(COUNTABLE))
    x = c14["share_nz"].to_numpy()
    for tag, col in (("gap vs BG2", "gap_bg2"), ("gap vs manual heuristic", "gap_mh")):
        y = c14[col].to_numpy()
        lo, hi = bootstrap_spearman(x, y)
        print(
            f"P1 {tag}: Spearman rho = {spearman(x, y):+.3f} (95% bootstrap over Primaries "
            f"{lo:+.3f} to {hi:+.3f}), n = {len(x)}"
        )
        for g in ("signed-MW", "MVA-only"):
            sub = c14.filter(pl.col("group") == g)
            print(
                f"   {g}: rho = {spearman(sub['share_nz'].to_numpy(), sub[col].to_numpy()):+.3f}"
                f" (n={sub.height})"
            )
    pooled_nz = pb.filter(pl.col(BAND) & countable & pl.col("near_zero")).select(pl.col("d").mean())
    pooled_ot = pb.filter(pl.col(BAND) & countable & ~pl.col("near_zero")).select(
        pl.col("d").mean()
    )
    print(
        "pooled gap vs BG2 near zero / elsewhere (14 Primaries):",
        pooled_nz.item(),
        pooled_ot.item(),
    )

    # ---------- Contrast 1: fold the signed series ----------
    print("\n== Contrast 1: signed against folded target at the six signed-MW Primaries ==")
    six = series.filter(pl.col("label").is_in(SIGNED_MW_LABELS))
    six_table = table.join(six.select("time_series_id"), on="time_series_id", how="semi")
    folded = {
        a: folded_rows(name=a, series=six, actuals=actuals, table=six_table, thresholds=thresholds)
        for a in ("stack", "bg2", "manual_heuristic")
    }
    signed = {
        a: rows[a].join(six.select("time_series_id"), on="time_series_id", how="semi")
        for a in ("stack", "bg2", "manual_heuristic")
    }
    where = pl.col(BAND)
    for ref in ("bg2", "manual_heuristic"):
        print(
            f"stack - {ref}, signed target, pooled six: ",
            contrast(
                treat=signed["stack"], ref=signed[ref], table=six_table, where=where, headline=True
            ),
        )
        print(
            f"stack - {ref}, folded target, pooled six: ",
            contrast(
                treat=folded["stack"], ref=folded[ref], table=six_table, where=where, headline=True
            ),
        )

    # difference of the gaps: (folded gap) - (signed gap), paired on rows
    def gap_rows(src: dict[str, pl.DataFrame], ref: str) -> pl.DataFrame:
        j = src["stack"].join(src[ref].select(*keys, rc=pl.col("crps")), on=keys, validate="1:1")
        return j.with_columns(crps=pl.col("crps") - pl.col("rc"))

    for ref in ("bg2", "manual_heuristic"):
        fg, sg = gap_rows(folded, ref), gap_rows(signed, ref)
        sg = sg.select(*keys, "crps").rename({"crps": "sg"})
        both = fg.join(sg, on=keys, validate="1:1")
        t = both.with_columns(crps=pl.col("crps"))
        r = both.with_columns(crps=pl.col("sg"))
        print(
            f"gap(folded) minus gap(signed), vs {ref}, pooled six (99%): ",
            contrast(treat=t, ref=r, table=six_table, where=where, headline=True),
        )
    print(
        "\nper Primary (95%, 2,000 resamples), stack - BG2, days 1 to 10, "
        "signed | folded | folded-minus-signed:"
    )
    fg, sg = gap_rows(folded, "bg2"), gap_rows(signed, "bg2").select(*keys, sg=pl.col("crps"))
    both = fg.join(sg, on=keys)
    near = c2.select("label", "share_nz")
    for lab in SIGNED_MW_LABELS:
        w = where & (pl.col("label") == lab)
        t = both.with_columns(crps=pl.col("crps"))
        r = both.with_columns(crps=pl.col("sg"))
        share = near.filter(pl.col("label") == lab)["share_nz"].item()
        print(
            lab,
            f"near-zero share {share:.3f}",
            "| signed",
            contrast(treat=signed["stack"], ref=signed["bg2"], table=six_table, where=w),
            "| folded",
            contrast(treat=folded["stack"], ref=folded["bg2"], table=six_table, where=w),
            "| diff",
            contrast(treat=t, ref=r, table=six_table, where=w),
        )
    fd = (
        both.filter(pl.col(BAND))
        .group_by("label")
        .agg(folded_gap=pl.col("crps").mean(), signed_gap=pl.col("sg").mean())
        .with_columns(diff=pl.col("folded_gap") - pl.col("signed_gap"))
        .join(near, on="label")
        .sort("label")
    )
    print(fd)
    fd.write_parquet(WORK / "contrast1_table.parquet")
    print(
        "Spearman near-zero share vs (folded - signed gap), six:",
        spearman(fd["share_nz"].to_numpy(), fd["diff"].to_numpy()),
    )
    print(
        "folded MAE of median, stack vs bg2 (days 1 to 10, pooled six):",
        folded["stack"].filter(where)["mae_median"].mean(),
        folded["bg2"].filter(where)["mae_median"].mean(),
        "; signed:",
        signed["stack"].filter(where)["mae_median"].mean(),
        signed["bg2"].filter(where)["mae_median"].mean(),
    )
    print("Do signed-MW Primaries have negative flow? share of negative actuals before folding:")
    neg = actuals.join(six.select("time_series_id", "capacity_mw", "label"), on="time_series_id")
    print(
        neg.filter(pl.col("valid_time") >= SCORED_FIRST)
        .group_by("label")
        .agg(
            neg_share=(pl.col("actual") < 0).mean(),
            min_pts=(pl.col("actual").min() / pl.col("capacity_mw").first() * 100),
        )
        .sort("label")
    )

    # ---------- Contrast 3: residual shape ----------
    print("\n== Contrast 3: stack residual by level of |actual| / peak, days 1 to 10 ==")
    fc = pl.read_parquet(WORK / "forecast_stack.parquet").select(*keys, "q50")
    res = (
        base.filter(pl.col(BAND))
        .join(fc, on=keys)
        .join(cap.select("time_series_id", "capacity_mw"), on="time_series_id")
        .join(rows["stack"].select(*keys, sc="crps"), on=keys)
        .join(rows["bg2"].select(*keys, bc="crps"), on=keys)
        .with_columns(
            resid=pl.col("actual_points") - pl.col("q50") / pl.col("capacity_mw") * 100.0,
            level=(pl.col("actual_points").abs() / pl.col("peak_points")),
        )
        .with_columns(
            level_bin=pl.col("level")
            .cut(
                [0.1, 0.2, 0.4, 0.6, 0.8],
                labels=["0-10%", "10-20%", "20-40%", "40-60%", "60-80%", "80%+"],
            )
            .cast(pl.String)
        )
    )
    c3 = (
        res.filter(pl.col("label").is_in(COUNTABLE))
        .group_by("group", "level_bin")
        .agg(
            n=pl.len(),
            mean_abs_resid=pl.col("resid").abs().mean(),
            mean_resid=pl.col("resid").mean(),
            stack_crps=pl.col("sc").mean(),
            bg2_crps=pl.col("bc").mean(),
            gap=(pl.col("sc") - pl.col("bc")).mean(),
        )
        .sort("group", "level_bin")
    )
    print(c3)
    c3.write_parquet(WORK / "contrast3_level_table.parquet")
    share_big = (
        res.filter(pl.col("label").is_in(COUNTABLE))
        .with_columns(big=pl.col("resid").abs() > pl.col("resid").abs().quantile(0.95))
        .group_by("group")
        .agg(
            big_in_low=pl.col("level").lt(0.2).filter(pl.col("big")).mean(),
            low_all=pl.col("level").lt(0.2).mean(),
        )
    )
    print(
        "share of the largest 5% |residual| rows with |actual| below 20% of peak, "
        "vs share of all rows:"
    )
    print(share_big)
    # ---------- Contrast 4: lead bands ----------
    print("\n== Contrast 4: gap by lead band, stack - BG2, per Primary ==")
    bands = {
        "days_1_3": pl.col("lead_hours") <= 72.0,
        "days_8_10": pl.col("lead_days_8_10"),
        "days_1_10": pl.col(BAND),
    }
    c4 = (
        pb.filter(pl.col("label").is_in(COUNTABLE))
        .group_by("label", "group")
        .agg(**{f"gap_{k}": pl.col("d").filter(v).mean() for k, v in bands.items()})
        .sort("group", "label")
    )
    print(c4)
    c4.write_parquet(WORK / "contrast4_lead_table.parquet")
    for g in ("signed-MW", "MVA-only"):
        gg = pl.col("group") == g
        for k, v in bands.items():
            print(
                g,
                k,
                contrast(
                    treat=rows["stack"],
                    ref=rows["bg2"],
                    table=table,
                    where=gg & (v if k != "days_1_3" else pl.col("lead_hours").le(72.0)),
                    headline=True,
                ),
            )
    print("\nstack and BG2 pooled by group, days 1 to 10 CRPS, 99%:")
    for g in ("signed-MW", "MVA-only"):
        gg = pl.col("group") == g
        print(
            g,
            "vs bg2",
            contrast(
                treat=rows["stack"],
                ref=rows["bg2"],
                table=table,
                where=gg & pl.col(BAND),
                headline=True,
            ),
        )
        print(
            g,
            "vs manual",
            contrast(
                treat=rows["stack"],
                ref=rows["manual_heuristic"],
                table=table,
                where=gg & pl.col(BAND),
                headline=True,
            ),
        )
        print(
            g,
            "vs climatology",
            contrast(
                treat=rows["stack"],
                ref=rows["climatology"],
                table=table,
                where=gg & pl.col(BAND),
                headline=True,
            ),
        )


if __name__ == "__main__":
    main()

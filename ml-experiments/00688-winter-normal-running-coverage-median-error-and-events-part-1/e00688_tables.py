"""Tables of experiment 688 (CPU only): coverage by stage, share explained, spread, sensitivities.

PYTHONPATH=ml-experiments/00668-*:ml-experiments/00683-*:ml-experiments/00688-* \
    DATA_PATH_INTERNAL=<production data folder> uv run --no-sync python e00688_tables.py

Prints every number the write-up quotes and writes CSV tables to `<data>/00688-*/tables/`, and the
per-day counts to `<data>/00688-*/daily.parquet` (private).
"""

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00683_calibrate as cal  # ty: ignore[unresolved-import]
import e00688_diagnose as dg  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MONTHS = dg.WINTER + dg.REST
PERIODS: dict[str, tuple[str, ...]] = {m: (m,) for m in MONTHS} | {
    "winter": dg.WINTER,
    "2026-02 to 2026-06": dg.REST,
}


def reproduction_gate(*, daily: pl.DataFrame) -> pl.DataFrame:
    """Compare stage `none` of both sets with experiment 683's per-day sums (arms 0 and 2)."""
    ref = pl.read_parquet(dg.DATA_683 / "daily" / "primaries_NRA.parquet").filter(
        pl.col("arm").is_in([0, 2])
        & (pl.col("origin_month") >= pl.lit(dg.FIRST_MONTH))
        & (pl.col("origin_month") <= pl.lit(dg.LAST_MONTH))
    )
    ref = (
        ref.group_by("arm", "origin_month")
        .agg(pl.col("n").sum(), pl.col("sum_coverage").sum())
        .with_columns(ref_coverage=pl.col("sum_coverage") / pl.col("n"), ref_n=pl.col("n"))
        .select("arm", "origin_month", "ref_coverage", "ref_n")
    )
    mine = (
        daily.filter(pl.col("stage") == "none")
        .with_columns(arm=pl.when(pl.col("set") == "stored").then(0).otherwise(2).cast(pl.Int8))
        .group_by("arm", pl.col("month").alias("origin_month"))
        .agg(
            pl.col("n").sum(), (100.0 * pl.col("cover").sum() / pl.col("n").sum()).alias("coverage")
        )
        .sort("arm", "origin_month")
    )
    out = mine.join(ref, on=["arm", "origin_month"]).with_columns(
        diff=pl.col("coverage") - pl.col("ref_coverage"), n_diff=pl.col("n") - pl.col("ref_n")
    )
    return out.sort("arm", "origin_month")


def coverage_table(*, boot: dg.Boot) -> pl.DataFrame:
    """Return pooled coverage and one-sided misses per set, stage, and period, with intervals."""
    rows = []
    for which in dg.SETS:
        for stage in dg.STAGES:
            for name, months in PERIODS.items():
                cell = boot.cell(which=which, stage=stage, months=months)
                rec = {"set": which, "stage": stage, "period": name, "n": int(cell["n"][0])}
                for k, label in (
                    ("cover", "coverage"),
                    ("below", "below_q05"),
                    ("above", "above_q95"),
                ):
                    point, lo, hi = dg.interval(cell[k])
                    rec.update({label: point, f"{label}_lo": lo, f"{label}_hi": hi})
                rows.append(rec)
    return pl.DataFrame(rows)


def explained_table(*, boot: dg.Boot) -> pl.DataFrame:
    """Return the share of the shortfall (90 minus arm 2 coverage) that each step explains."""
    rows = []
    for name, months in PERIODS.items():
        cells = {s: boot.cell(which="scaled", stage=s, months=months)["cover"] for s in dg.STAGES}
        base = cells["none"]
        short = 90.0 - base
        for label, num in (
            ("step 1 alone (events)", cells["events"] - base),
            ("step 2 alone (shift)", cells["shift"] - base),
            ("both steps", cells["events+shift"] - base),
            ("step 2 after step 1", cells["events+shift"] - cells["events"]),
            ("step 1 after step 2", cells["events+shift"] - cells["shift"]),
        ):
            share = np.where(short > 1.0, 100.0 * num / np.where(short > 1.0, short, 1.0), np.nan)
            point, lo, hi = float(share[0]), *np.nanpercentile(share[1:], [2.5, 97.5]).tolist()
            rows.append(
                {
                    "period": name,
                    "shortfall_points": float(short[0]),
                    "step": label,
                    "gain_points": float(num[0]),
                    "share_percent": point,
                    "share_lo": lo,
                    "share_hi": hi,
                }
            )
    return pl.DataFrame(rows)


def series_table(*, boot: dg.Boot, series: list[int]) -> pl.DataFrame:
    """Return coverage and one-sided misses per series, period, and stage (arm 2)."""
    rows = []
    for tid in series:
        for stage in dg.STAGES[:4]:
            for name in ("2025-12", "2026-01", "winter", "2026-02 to 2026-06"):
                cell = boot.cell(which="scaled", stage=stage, months=PERIODS[name], series=tid)
                rec = {
                    "time_series_id": tid,
                    "stage": stage,
                    "period": name,
                    "n": int(cell["n"][0]),
                }
                for k, label in (
                    ("cover", "coverage"),
                    ("below", "below_q05"),
                    ("above", "above_q95"),
                ):
                    point, lo, hi = dg.interval(cell[k])
                    rec.update({label: point, f"{label}_lo": lo, f"{label}_hi": hi})
                rows.append(rec)
    return pl.DataFrame(rows)


def spread_table(*, rows: dict[str, np.ndarray], evt: np.ndarray) -> pl.DataFrame:
    """Return the band half-width, the empirical residual half-width, and the scale still needed.

    For each series and lead band, over the two winter months, in percent of capacity. `ratio` is
    the median half-width of the band `(q95 - q05) / 2` over the empirical half-width
    `(P95 - P05) / 2` of the residuals `y - q50 - shift`. `needed_scale` is the 0.90 quantile of
    the distance of the shifted observation from `q50` over the half-width on its side
    (experiment 683's fitting statistic), which is the factor that would bring the band to 90%.
    """
    out = []
    winter = np.isin(rows["month"], dg.WINTER)
    for stage in ("none", "events+shift"):
        for which in dg.SETS:
            q, use = dg.stage_quantiles(rows=rows, which=which, stage=stage, evt=evt)
            sel_all = winter & use
            shift = q[:, 5] - rows[which][:, 5]
            for tid in [*sorted(set(rows["series"].tolist())), -1]:
                for b in [*range(4), -1]:
                    sel = sel_all & ((rows["series"] == tid) if tid >= 0 else True)
                    if b >= 0:
                        sel = sel & (rows["band"] == b)
                    if sel.sum() < 100:
                        continue
                    y, qq = rows["y"][sel], q[sel]
                    resid = y - qq[:, 5]
                    half = (qq[:, 9] - qq[:, 1]) / 2.0
                    emp = (np.percentile(resid, 95) - np.percentile(resid, 5)) / 2.0
                    if emp <= 0:
                        continue
                    z = cal.z_scores(y=y, q05=qq[:, 1], q50=qq[:, 5], q95=qq[:, 9])
                    z = z[np.isfinite(z)]
                    if len(z) == 0:
                        continue
                    out.append(
                        {
                            "set": which,
                            "stage": stage,
                            "time_series_id": tid,
                            "band": cal.BAND_LABELS[b] if b >= 0 else "all",
                            "n": int(sel.sum()),
                            "median_half_width": float(np.median(half)),
                            "empirical_half_width": float(emp),
                            "ratio": float(np.median(half) / emp),
                            "mean_half_width": float(np.mean(half)),
                            "mean_abs_residual_over_half_width": float(
                                np.mean(np.abs(resid) / np.maximum(half, 1e-9))
                            ),
                            "needed_scale": float(np.quantile(z, 0.90)),
                            "mean_abs_shift": float(np.abs(shift[sel]).mean()),
                        }
                    )
    return pl.DataFrame(out)


def mask_counts(*, rows: dict[str, np.ndarray], evt: np.ndarray) -> pl.DataFrame:
    """Return the scored and masked half-hours per series and origin month."""
    frame = pl.DataFrame({"series": rows["series"], "month": rows["month"], "evt": evt})
    return (
        frame.group_by("series", "month")
        .agg(pl.len().alias("scored"), pl.col("evt").sum().alias("masked"))
        .with_columns(share=pl.col("masked") / pl.col("scored"))
        .sort("series", "month")
    )


def main() -> None:
    """Run the diagnostic and write the tables."""
    dg.TABLES.mkdir(parents=True, exist_ok=True)
    rows = dg.load_rows()
    ids = sorted(set(rows["series"].tolist()))
    spans = dg.event_spans(kind="primary")
    evt = dg.event_mask(spans=spans, series=rows["series"], vidx=rows["vidx"])
    spans.write_csv(dg.TABLES / "event_spans_primary.csv")
    print("rows", len(rows["y"]), "masked", int(evt.sum()), "spans", spans.height)
    daily = dg.daily_table(rows=rows, evt=evt)
    daily.write_parquet(dg.DATA / "daily.parquet")
    gate = reproduction_gate(daily=daily)
    gate.write_csv(dg.TABLES / "reproduction_gate.csv")
    print(
        "reproduction gate (max |diff| coverage points, max |n diff|):",
        float(gate["diff"].abs().max()),  # ty: ignore[invalid-argument-type]
        int(gate["n_diff"].abs().max()),  # ty: ignore[invalid-argument-type]
    )
    early = gate.filter(pl.col("origin_month") < pl.lit("2026-06"))
    if early["n_diff"].abs().max() != 0 or early["diff"].abs().max() > 1e-3:
        msg = "stage none does not reproduce experiment 683 before 2026-06"
        raise ValueError(msg)
    print(gate.filter(pl.col("origin_month") == pl.lit("2026-06")))
    boot = dg.Boot(daily=daily)
    cov = coverage_table(boot=boot)
    cov.write_csv(dg.TABLES / "coverage_by_stage.csv")
    exp = explained_table(boot=boot)
    exp.write_csv(dg.TABLES / "explained.csv")
    ser = series_table(boot=boot, series=ids)
    ser.write_csv(dg.TABLES / "series_coverage.csv")
    spread = spread_table(rows=rows, evt=evt)
    spread.write_csv(dg.TABLES / "spread.csv")
    counts = mask_counts(rows=rows, evt=evt)
    counts.write_csv(dg.TABLES / "mask_counts.csv")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_width_chars(220)
    print(
        cov.filter(
            (pl.col("set") == "scaled") & pl.col("period").is_in(["2025-12", "2026-01", "winter"])
        )
    )
    print(
        cov.filter(
            (pl.col("set") == "stored") & pl.col("period").is_in(["2025-12", "2026-01", "winter"])
        )
    )
    print(exp)
    below = ser.filter(
        (pl.col("coverage") < dg.THRESHOLD * 100) & pl.col("period").is_in(["2025-12", "2026-01"])
    )
    print("series below 0.86 by stage and month")
    print(
        below.select("stage", "period", "time_series_id", "coverage", "coverage_lo", "coverage_hi")
    )
    print(spread.filter(pl.col("time_series_id") == -1))
    print(counts.filter(pl.col("masked") > 0))
    sensitivities(rows=rows, boot=boot)
    print("lib", lib.FOLDER)


def sensitivities(*, rows: dict[str, np.ndarray], boot: dg.Boot) -> None:
    """Report the wider masks and the per-band shift on the pooled winter coverage."""
    out = []
    for name, months in (
        ("2025-12", ("2025-12",)),
        ("2026-01", ("2026-01",)),
        ("winter", dg.WINTER),
    ):
        for stage in ("none", "events", "events+shift", "events+shift per band"):
            cell = boot.cell(which="scaled", stage=stage, months=months)
            out.append(
                {
                    "mask": "primary",
                    "stage": stage,
                    "period": name,
                    "coverage": float(cell["cover"][0]),
                }
            )
    wide = dg.event_spans(kind="wide")
    evt_wide = dg.event_mask(spans=wide, series=rows["series"], vidx=rows["vidx"])
    daily = dg.daily_table(rows=rows, evt=evt_wide)
    boot_wide = dg.Boot(daily=daily)
    for name, months in (
        ("2025-12", ("2025-12",)),
        ("2026-01", ("2026-01",)),
        ("winter", dg.WINTER),
    ):
        for stage in ("events", "events+shift"):
            cell = boot_wide.cell(which="scaled", stage=stage, months=months)
            point, lo, hi = dg.interval(cell["cover"])
            out.append(
                {
                    "mask": "wide",
                    "stage": stage,
                    "period": name,
                    "coverage": point,
                    "lo": lo,
                    "hi": hi,
                }
            )
    table = pl.DataFrame(out, strict=False)
    table.write_csv(dg.TABLES / "sensitivities.csv")
    wide.write_csv(dg.TABLES / "event_spans_wide.csv")
    print(table)
    print("wide mask rows", int(evt_wide.sum()))


if __name__ == "__main__":
    main()

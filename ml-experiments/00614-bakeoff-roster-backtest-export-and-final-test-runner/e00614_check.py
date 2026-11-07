# ruff: noqa: E501, D103, PERF401
"""Experiment 614: recompute event-free coverage from the exported files and compare it with the stored cells.

Reads only the exported Parquet files and experiment 613's `score/*/event_free_cells.csv`. For each
network, Primary, fold month, and lead band, coverage is the share of event-free rows with the
normal-running-arrangement-corrected power `nra_power` inside q05-q95 (experiment 613 scores the same
quantity under the column `obs`, which is the cleaned power, so this script uses `observed_power`
as the scoring target, exactly as experiment 613's event-free cells do). Writes `checks.txt` beside
the export and prints the largest differences.

Usage: python e00614_check.py
"""

import polars as pl
from e00614_export import MONTHS, OUT, ROOT613, SEEDS

STORED = {  # network -> (score folder, system column of event_free_cells.csv)
    "K1": ("k1_d5", "K1"),
    "Candidate": ("c2_e3w", "K2w"),
    "K3": ("k3_d5", "K2"),
}
TOL = 0.003
BANDS = {"0-24 h": (0, 24), "24-120 h": (24, 120), "120-240 h": (120, 240)}


def exported(*, network: str) -> pl.DataFrame:
    frames = [
        pl.read_parquet(OUT / f"{network}_{m}.parquet").with_columns(month=pl.lit(m))
        for m in MONTHS
    ]
    sub = pl.read_csv(OUT / "substations.csv").select("time_series_id", "primary_name", "group")
    df = pl.concat(frames).join(sub, on="time_series_id")
    band = (
        pl.when(pl.col("lead_hours") < 24).then(pl.lit("0-24 h"))
        .when(pl.col("lead_hours") < 120).then(pl.lit("24-120 h"))
        .otherwise(pl.lit("120-240 h"))
    )  # fmt: skip
    y = pl.col("observed_power")
    return (
        df.filter(pl.col("event_free") & y.is_not_null() & y.is_finite())
        .with_columns(band=band, hit=((pl.col("q05") <= y) & (y <= pl.col("q95"))).cast(pl.Float64))
        .group_by("primary_name", "month", "band")
        .agg(pl.len().alias("n_rows_export"), pl.col("hit").mean().alias("cover_export"))
    )


def stored(*, folder: str, system: str) -> pl.DataFrame:
    return (
        pl.read_csv(ROOT613 / "score" / folder / "event_free_cells.csv")
        .filter(pl.col("system") == system)
        .select(
            pl.col("label").alias("primary_name"),
            "group",
            "month",
            "band",
            pl.col("n_rows").alias("n_rows_stored"),
            pl.col("cover90").alias("cover_stored"),
        )
    )


def main() -> None:
    lines: list[str] = []
    worst = 0.0
    for network, (folder, system) in STORED.items():
        j = exported(network=network).join(
            stored(folder=folder, system=system),
            on=["primary_name", "month", "band"],
            how="full",
            coalesce=True,
        )
        missing = j.filter(pl.col("cover_export").is_null() | pl.col("cover_stored").is_null())
        j = j.drop_nulls(["cover_export", "cover_stored"]).with_columns(
            diff=pl.col("cover_export") - pl.col("cover_stored")
        )
        pooled = (
            j.group_by("group", "band")
            .agg(
                (
                    (pl.col("cover_export") * pl.col("n_rows_export")).sum()
                    / pl.col("n_rows_export").sum()
                ).alias("export"),
                (
                    (pl.col("cover_stored") * pl.col("n_rows_stored")).sum()
                    / pl.col("n_rows_stored").sum()
                ).alias("stored"),
                pl.col("n_rows_export").sum().alias("n_export"),
                pl.col("n_rows_stored").sum().alias("n_stored"),
            )
            .with_columns(diff=pl.col("export") - pl.col("stored"))
            .sort("group", "band")
        )
        for r in missing.iter_rows(named=True):
            lines.append(
                f"  in only one source: {r['primary_name']} {r['month']} {r['band']} (export rows {r['n_rows_export']}, stored rows {r['n_rows_stored']})"
            )
        max_cell = float(j["diff"].abs().max())  # ty: ignore[invalid-argument-type]
        worst = max(worst, max_cell, float(pooled["diff"].abs().max()))  # ty: ignore[invalid-argument-type]
        lines.append(
            f"{network}: cells compared {j.height}, cells in only one source {missing.height}, largest cell difference {max_cell:.5f}, row-count mismatches {int((j['n_rows_export'] != j['n_rows_stored']).sum())}"
        )
        for r in pooled.iter_rows(named=True):
            lines.append(
                f"  {r['group']:<20} {r['band']:<10} export {r['export']:.4f} stored {r['stored']:.4f} diff {r['diff']:+.5f} rows {r['n_export']} / {r['n_stored']}"
            )
    lines.append(
        "Experiment 612 printed event-free 24-120 h coverage on four months (2025-12 to 2026-06) from seeds 0 and 1 only:"
    )
    lines.append(
        f"  K1 0.887 (eight MVA) and 0.860 (six MW); K3 0.868 and 0.836. The files of seeds {SEEDS} give:"
    )
    for network in ("K1", "K3"):
        e = exported(network=network).filter(
            (pl.col("month") != MONTHS[0]) & (pl.col("band") == "24-120 h")
        )
        for group in ("eight MVA Primaries", "six MW Primaries"):
            s = e.join(
                pl.read_csv(OUT / "substations.csv").select("primary_name", "group"),
                on="primary_name",
            ).filter(pl.col("group") == group)
            cov = float((s["cover_export"] * s["n_rows_export"]).sum() / s["n_rows_export"].sum())  # ty: ignore[unsupported-operator]
            lines.append(f"  {network} {group}: {cov:.3f}")
    lines.append(
        f"Largest absolute difference over all cells and pooled groups: {worst:.5f} (tolerance {TOL}): {'PASS' if worst <= TOL else 'MISMATCH'}"
    )
    text = "\n".join(lines)
    print(text)
    (OUT / "checks.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()

# ruff: noqa: D103, E501
"""Experiment 677: score the GAM and XGBoost reference lines beside the ladder on their shared rows (private CSV).

Usage: python e00677_references.py

The GAM is Primary demand research default 3 (experiment 322's stored per-row CRPS) and XGBoost is BG2
(`p_s6_bg2` of experiment 87), both on the six MW Primaries and 09:00 UTC origins. The ladder's origins are
12:00 UTC, so the two sets are joined on the Primary, the valid time, and the origin day; a ladder row
therefore leads its stored partner by 3 hours. Only CRPS is compared, because experiment 322 stores no
median. Rows are the valid times from 2025-11-01 12:00 UTC up to the last valid time that the stored
forecasts reach, with lead at most 168 hours. Writes `scores/references_crps.csv`.
"""

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00668_slices_primaries as s668  # ty: ignore[unresolved-import]
import e00668_stored_comparison as stored  # ty: ignore[unresolved-import]
import e00677_export as exp
import e00677_score as sc
import numpy as np
import polars as pl

STEPS = ("1", "2", "3a", "3b", "4", "5w", "6w", "7w", "8w", "8wo")


def ladder_crps(*, step: str, label_of: dict[int, str]) -> pl.DataFrame:
    tag = "s01234" if step[0] in "5678" else "s012"
    path = sc.ROOT / "scores" / f"step{step}_{exp.VERSION[step]}_sp3_{tag}" / "rows.parquet"
    start = lib.GRID_START
    return (
        pl.read_parquet(path)
        .filter(pl.col("ok") & pl.col("time_series_id").cast(pl.Int64).is_in(list(label_of)))
        .with_columns(
            site=pl.col("time_series_id").cast(pl.Int64).replace_strict(label_of),
            valid_time=pl.lit(start) + pl.duration(minutes=30) * pl.col("valid_idx").cast(pl.Int64),
            origin_day=(
                pl.lit(start) + pl.duration(minutes=30) * pl.col("origin_idx").cast(pl.Int64)
            ).dt.date(),
        )
        .select(
            "site",
            "valid_time",
            "origin_day",
            pl.col("Model_crps").alias(f"step{step}"),
            pl.col("Manual heuristic_crps").alias("Manual heuristic"),
        )
    )


def main() -> None:
    site = pl.read_csv(next(s668.R.glob("00521-*/site_map.csv")))
    label_of = dict(zip(site["time_series_id"].to_list(), site["label"].to_list(), strict=True))
    k = stored.load().filter(pl.col("loss_bg2").is_finite() & (pl.col("lead_hours") <= 168.0))
    k = k.with_columns(origin_day=pl.col("power_fcst_init_time").dt.date()).select(
        "site",
        "valid_time",
        "origin_day",
        "lead_hours",
        pl.col("loss_d3").alias("GAM"),
        pl.col("loss_bg2").alias("XGBoost"),
    )
    for step in STEPS:
        d = ladder_crps(step=step, label_of=label_of)
        k = k.join(
            d.drop("Manual heuristic") if step != STEPS[0] else d,
            on=["site", "valid_time", "origin_day"],
            how="inner",
        )
    names = ["GAM", "XGBoost", "Manual heuristic", *[f"step{s}" for s in STEPS]]
    k = k.filter(pl.all_horizontal([pl.col(n).is_finite() for n in names]))
    print(
        k.height,
        "common rows;",
        k["valid_time"].min(),
        "to",
        k["valid_time"].max(),
        "; Primaries",
        sorted(k["site"].unique().to_list()),
    )
    day = (k["origin_day"].cast(pl.Int32)).to_numpy()
    lead = k["lead_hours"].to_numpy()
    rows = []

    def add(mask: np.ndarray, **tag: str) -> None:
        draws = lib.day_bootstrap(
            day=day[mask], values={n: k[n].to_numpy().astype(np.float64)[mask] for n in names}
        )
        for n in names:
            lo, hi = np.quantile(draws[n], [0.025, 0.975])
            rows.append(
                {
                    **tag,
                    "system": n,
                    "crps": float(k[n].to_numpy()[mask].mean()),
                    "lo": float(lo),
                    "hi": float(hi),
                    "rows": int(mask.sum()),
                }
            )

    add(np.ones(k.height, dtype=bool), slice="all rows", group="all")
    for name, (lo, hi) in lib.LEAD_BANDS.items():
        if lo < 168.0:
            add((lead >= lo) & (lead < min(hi, 168.5)), slice="lead band", group=name)
    out = pl.DataFrame(rows)
    out.write_csv(sc.ROOT / "scores" / "references_crps.csv")
    print(out.filter(pl.col("slice") == "all rows").select("system", "crps", "lo", "hi"))


if __name__ == "__main__":
    main()

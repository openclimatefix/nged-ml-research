# ruff: noqa: ANN001, D103, E501
"""Compare the prevailing rows before and after the extra-lag head: CRPS, normalised MAE, coverage.

uv run --no-sync python e00668_extra_lags_compare.py

Reads the per-row scores of the old file (`scores_668_pre_extra_lags/primaries_prevailing/rows.parquet`)
and the new file (`scores_668/...`), plus the two files for q05 to q95 coverage, and prints markdown
tables. Windows: `2025-11 to 2025-12` (fallback head, unchanged), `2026-01 to 2026-06` (before the
final-test window), and `2026-07 to 2026-10` (inside it; descriptive only).
"""

from typing import Final

import e00668_score_lib as lib
import numpy as np
import polars as pl

OLD_DIR: Final = lib.FOLDER / "scores_668_pre_extra_lags" / "primaries_prevailing"
NEW_DIR: Final = lib.FOLDER / "scores_668" / "primaries_prevailing"
BACKUP: Final = lib.FOLDER / "pre_extra_lags" / "primaries.parquet"
CURRENT: Final = lib.FOLDER / "primaries.parquet"
WINDOWS: Final = {
    "2025-11 to 2025-12": ("2025-11", "2025-12"),
    "2026-01 to 2026-06": ("2026-01", "2026-06"),
    "2026-07 to 2026-10": ("2026-07", "2026-10"),
    "all months": ("2025-11", "2026-10"),
}


def load(*, rows_dir, path, kind: str = "prevailing") -> pl.DataFrame:
    scores = pl.read_parquet(rows_dir / "rows.parquet")
    forecast = pl.read_parquet(path).filter(pl.col("NRA_or_prevailing") == kind)
    assert forecast.height == scores.height
    cap = pl.read_parquet(lib.FOLDER / "series_capacity.parquet").select(
        "time_series_id", "capacity"
    )
    forecast = forecast.join(cap, on="time_series_id", how="left")
    ids = sorted(forecast["time_series_id"].unique().to_list())
    grid = lib.cleaned_grid(ids=ids)
    row = forecast["time_series_id"].replace_strict({t: i for i, t in enumerate(ids)}).to_numpy()
    vidx = lib.grid_index(times=forecast["valid_time"])
    if vidx.max() >= grid.shape[1]:
        grid = np.pad(
            grid, ((0, 0), (0, int(vidx.max()) + 1 - grid.shape[1])), constant_values=np.nan
        )
    y = grid[row, vidx]
    scale = 100.0 / forecast["capacity"].to_numpy()
    q05 = forecast["q05"].to_numpy().astype(np.float64) * scale
    q95 = forecast["q95"].to_numpy().astype(np.float64) * scale
    y = y * scale
    month = forecast["power_fcst_init_time"].dt.strftime("%Y-%m")
    return scores.select(
        "time_series_id", "lead_h", "ok", "Model_crps", "Model_mae", "Manual heuristic_crps"
    ).with_columns(
        month=month,
        inside=pl.Series((y >= q05) & (y <= q95)),
        below=pl.Series(y < q05),
        above=pl.Series(y > q95),
    )


def band_of(lead: pl.Expr) -> pl.Expr:
    expr = pl.lit("days 10 to 14")
    for name, (lo, hi) in reversed(list(lib.LEAD_BANDS.items())):
        expr = pl.when((lead >= lo) & (lead < hi)).then(pl.lit(name)).otherwise(expr)
    return expr


def summary(*, df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (
        df.filter(pl.col("ok"))
        .group_by(by)
        .agg(
            half_hours=pl.len(),
            crps=pl.col("Model_crps").mean(),
            mae=pl.col("Model_mae").mean(),
            coverage=pl.col("inside").mean(),
        )
        .sort(by)
    )


def main() -> None:
    old = load(rows_dir=OLD_DIR, path=BACKUP).with_columns(band=band_of(pl.col("lead_h")))
    new = load(rows_dir=NEW_DIR, path=CURRENT).with_columns(band=band_of(pl.col("lead_h")))
    # The cleaned power store gained readings between the two scoring runs, so score the common rows.
    both = (old["ok"] & new["ok"]).to_numpy()
    old = old.with_columns(ok=pl.Series(both))
    new = new.with_columns(ok=pl.Series(both))
    for label, (m0, m1) in WINDOWS.items():
        for kind, frame in (("old", old), ("new", new)):
            sub = frame.filter((pl.col("month") >= m0) & (pl.col("month") <= m1))
            allb = summary(df=sub.with_columns(g=pl.lit("all")), by=["g"])
            bands = summary(df=sub, by=["band"])
            print(f"## {label} / {kind}")
            print(allb)
            print(bands)
    nra = load(rows_dir=lib.FOLDER / "scores_668" / "primaries_NRA", path=CURRENT, kind="NRA")
    nra = nra.with_columns(ok=pl.Series(both))
    for label, (m0, m1) in WINDOWS.items():
        sub = nra.filter((pl.col("month") >= m0) & (pl.col("month") <= m1))
        print(f"## NRA {label}")
        print(summary(df=sub.with_columns(g=pl.lit("all")), by=["g"]))
    pl.Config.set_tbl_rows(40)
    for kind, frame in (("old", old), ("new", new)):
        print(f"## by month / {kind}")
        print(summary(df=frame, by=["month"]))
        print(f"## by tail / {kind}")
        ok = frame.filter(pl.col("ok"))
        print(ok.select(below=pl.col("below").mean(), above=pl.col("above").mean()))


if __name__ == "__main__":
    main()

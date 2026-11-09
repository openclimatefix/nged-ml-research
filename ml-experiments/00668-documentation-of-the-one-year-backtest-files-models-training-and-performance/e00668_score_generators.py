"""Score `wind.parquet` or `solar.parquet` against observed power.

uv run --no-sync python ml-experiments/00668-*/e00668_score_generators.py wind
uv run --no-sync python ml-experiments/00668-*/e00668_score_generators.py solar

Systems: the model in the file and climatology, both on the rows where observed power is finite.
No manual heuristic exists for generators. Wind rows inside experiment 210's idle mask (to
2026-09-13 17:30 UTC) are scored with the observed power set to zero, which is the fits' target;
the same rows excluded are reported as `wind_idle_excluded`. Solar is scored on daylight
half-hours: the sun is above the horizon at the middle of the half-hour. Series are labelled A, B,
and C (wind) or A to F (solar) in order of identifier, and the identifier is never written.
Writes CSV tables to `<backtest folder>/scores_668/<name>/` (private).
"""

import sys

import e00668_score_lib as lib
import numpy as np
import polars as pl
from e00668_score_demand import season_label
from studies.pv_dataset import pv_sites
from studies.solar import zenith


def main(name: str, *, idle_as_zero: bool = True) -> None:  # noqa: PLR0915
    """Score one generator file."""
    forecast = pl.read_parquet(lib.FOLDER / f"{name}.parquet")
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
    truth = pl.read_parquet(lib.FOLDER / f"ground_truth_{name}.parquet")
    ids = sorted(forecast["time_series_id"].unique().to_list())
    cap = capacity.filter(pl.col("time_series_id").is_in(ids)).sort("time_series_id")["capacity"]
    cap = cap.to_numpy().astype(np.float64)
    grid = lib.cleaned_grid(ids=ids)
    if name == "wind":
        idle = truth.filter(pl.col("research_flag").is_not_null())
        gi = lib.grid_index(times=idle["valid_time"])
        gr = idle["time_series_id"].replace_strict({t: i for i, t in enumerate(ids)}).to_numpy()
        grid[gr, gi] = 0.0 if idle_as_zero else np.nan
    row = forecast["time_series_id"].replace_strict({t: i for i, t in enumerate(ids)}).to_numpy()
    vidx = lib.grid_index(times=forecast["valid_time"])
    oidx = lib.grid_index(times=forecast["power_fcst_init_time"])
    if vidx.max() >= grid.shape[1]:
        grid = np.pad(
            grid, ((0, 0), (0, int(vidx.max()) + 1 - grid.shape[1])), constant_values=np.nan
        )
    lead_h = (vidx - oidx) / 2.0
    y = grid[row, vidx].copy()
    cutoff = lib.grid_index(times=forecast["power_fcst_init_time"].dt.truncate("1mo"))
    scale = 100.0 / cap[row]
    q_model = forecast.select(lib.QCOLS).to_numpy().astype(np.float64) * scale[:, None]
    q_clim = (
        lib.climatology(grid=grid, row=row, vidx=vidx, cutoff_idx=cutoff, use_weekend=False)
        * scale[:, None]
    )
    y = y * scale
    ok = np.isfinite(y)
    for q in (q_model, q_clim):
        ok &= np.isfinite(q).all(axis=1)
    if name == "solar":
        sites = pv_sites().sort("time_series_id")
        sites = sites.filter(pl.col("time_series_id").is_in(ids))
        daylight = np.zeros(len(y), dtype=bool)
        for i, rec in enumerate(sites.iter_rows(named=True)):
            sel = np.flatnonzero(row == i)
            mid = forecast["valid_time"].gather(sel.tolist()).dt.offset_by("-15m")
            z = zenith(stamps=mid, latitude=rec["latitude"], longitude=rec["longitude"])
            daylight[sel] = np.asarray(z) < 90.0
        ok &= daylight
    no_truth = ~np.isfinite(y)
    if name == "solar":
        no_truth &= daylight
    label = name if (name != "wind" or idle_as_zero) else "wind_idle_excluded"
    out_dir = lib.FOLDER / "scores_668" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    month = forecast["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    pl.DataFrame({"origin_month": month, "scored": ok, "no_truth": no_truth}).group_by(
        "origin_month"
    ).agg(rows=pl.len(), scored=pl.col("scored").sum(), no_truth=pl.col("no_truth").sum()).sort(
        "origin_month"
    ).write_csv(out_dir / "row_counts.csv")
    print(label, "rows", len(y), "scored", int(ok.sum()), flush=True)
    per_system = {
        "Model": lib.losses(y=y, q=q_model),
        "Climatology": lib.losses(y=y, q=np.where(ok[:, None], q_clim, 0.0)),
    }
    day = oidx // 48
    band = np.select(
        [(lo <= lead_h) & (lead_h < hi) for lo, hi in lib.LEAD_BANDS.values()],
        list(lib.LEAD_BANDS),
        default="days 10 to 14",
    )
    season = season_label(origin=forecast["power_fcst_init_time"])
    season = np.where(season == "autumn (Nov 2025)", "autumn (first month)", season)
    season = np.where(season == "autumn (Sep to Oct 2026)", "autumn (last two months)", season)
    letters = "ABCDEF"
    rows: list[dict] = []

    def add(mask: np.ndarray, **tag: str) -> None:
        rows.extend(
            lib.summarise(
                mask=ok & mask, day=day, per_system=per_system, reference="Model", label=tag
            )
        )

    add(np.ones(len(y), dtype=bool), slice="all rows", group="all")
    for b in lib.LEAD_BANDS:
        add(band == b, slice="lead band", group=b)
    lead_day = np.minimum(np.floor(lead_h / 24.0), 13).astype(int)
    for k in range(14):
        add(lead_day == k, slice="lead day", group=f"{k:02d}")
    month_number = forecast["power_fcst_init_time"].dt.month().to_numpy()
    for m in range(1, 13):
        add(month_number == m, slice="month number", group=f"{m:02d}")
    for s in dict.fromkeys(season):
        add(season == s, slice="season", group=s)
        for b in lib.LEAD_BANDS:
            add((season == s) & (band == b), slice="season and lead band", group=f"{s} | {b}")
    for i in range(len(ids)):
        add(row == i, slice="series", group=f"{name.capitalize()} {letters[i]}")
    pl.DataFrame(rows).write_csv(out_dir / "scores.csv")
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], idle_as_zero=len(sys.argv) < 3)

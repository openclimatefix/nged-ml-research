# ruff: noqa: D103, E501
"""Print every number on the experiment 607 page: the newest date and row check of each extended input.

Reads the new tables and the stored tables; writes `inputs_after.parquet`. No forecast is read.
"""

from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

OUT: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*"))
WORK: Final[Path] = OUT / "work"
R: Final[Path] = REPO_DATA_DIR / "studies/downloads/reanalysis"


def stitched(*, stored: Path, new: Path, keys: list[str], values: list[str]) -> dict[str, object]:
    """Check a stored table plus its new rows: continuity, duplicates, and the shared hours."""
    a = pl.read_parquet(stored)
    b = pl.read_parquet(new)
    tcol = "time"
    both = pl.concat([a.select(b.columns), b])
    dup = both.height - both.unique(subset=[*keys, tcol]).height
    nulls = int(b.select(pl.col(values).null_count()).sum_horizontal().item())
    per = both.group_by(keys).agg(n=pl.len(), first=pl.col(tcol).min(), last=pl.col(tcol).max())
    return {
        "stored_last": str(a[tcol].max()),
        "new_first": str(b[tcol].min()),
        "new_last": str(b[tcol].max()),
        "new_rows": b.height,
        "duplicate_keys_after_stitch": dup,
        "null_cells_in_new_rows": nulls,
        "groups": per.height,
        "groups_not_ending_on_last": int((per["last"] != per["last"].max()).sum()),
    }


def main() -> None:
    out: dict[str, dict[str, object]] = {}
    out["ERA5 irradiance at solar sites (Open-Meteo)"] = stitched(
        stored=R / "ERA5/site_points/beam_diffuse_open_meteo.parquet",
        new=R / "ERA5/site_points/beam_diffuse_open_meteo_2026-09-11_2026-10-31.parquet",
        keys=["latitude", "longitude"],
        values=["ghi_w_m2", "bhi_w_m2", "temp_c"],
    )
    out["ERA5 wind at wind sites (Open-Meteo)"] = stitched(
        stored=R / "ERA5/site_points/wind_era5.parquet",
        new=R / "ERA5/site_points/wind_era5_2026-09-11_2026-10-31.parquet",
        keys=["site"],
        values=["wind_speed_100m", "wind_direction_100m", "wind_speed_10m"],
    )
    out["CAMS irradiance at solar sites"] = stitched(
        stored=R / "CAMS/site_points/beam_diffuse_cams.parquet",
        new=R / "CAMS/site_points/beam_diffuse_cams_2026-09-11_2026-10-31.parquet",
        keys=["site"],
        values=["ghi_w_m2"],
    )
    out["CAMS irradiance at demand points"] = stitched(
        stored=R / "CAMS/demand_points/cams_demand_points.parquet",
        new=R / "CAMS/demand_points/cams_demand_points_2026-09-11_2026-10-31.parquet",
        keys=["site"],
        values=["ghi_w_m2", "clear_sky_ghi_w_m2", "reliability"],
    )
    obs = pl.read_parquet(WORK / "observed.parquet")
    out["Observed-power cache"] = {
        "new_last": str(obs["valid_time"].max()),
        "rows": obs.height,
        "series": obs["time_series_id"].n_unique(),
    }
    for name, folder in (
        ("ENS summary", "ens_summary"),
        ("ENS summary with wind components", "ens_summary_uv"),
        ("ENS total cloud cover", "ens_summary_tcc"),
    ):
        files = sorted((WORK / folder).glob("init=*.parquet"))
        out[name] = {
            "runs": len(files),
            "first_run": files[0].stem[5:],
            "last_run": files[-1].stem[5:],
        }
    box = {}
    for p in sorted((WORK / "era5_stage1_box_cds").glob("*.parquet")):
        f = pl.read_parquet(p)
        box[p.stem] = f"{f.height:,} rows to {f['time'].max()}"
    out["ERA5 stage-1 box, new rows (CDS)"] = box  # ty: ignore[invalid-assignment]
    print(__import__("json").dumps(out, indent=2, default=str))
    (OUT / "inputs_after.json").write_text(__import__("json").dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

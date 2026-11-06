# ruff: noqa: D103
"""The newest timestamp of each derived and pretraining input, against the test window's end.

Reads only the time column of each stored table. Writes `inputs_end.parquet`.
"""

from pathlib import Path
from typing import Final

import polars as pl
from e00605_power import OUT_DIR
from studies.sources import REPO_DATA_DIR

REANALYSIS: Final[Path] = REPO_DATA_DIR / "studies" / "downloads" / "reanalysis"
STAGE1: Final[Path] = REPO_DATA_DIR / "studies" / "per_study" / "stage1_pilot" / "demand" / "inputs"
WORK_397: Final[Path] = next((REPO_DATA_DIR / "ml-experiments").glob("00397-*")) / "work"
TEST_END: Final = pl.datetime(2026, 10, 31, 23, 30, time_zone="UTC")

# (name, role, path, time column)
TABLES: Final[list[tuple[str, str, Path, str]]] = [
    ("observed power cache", "demand lags and scoring", STAGE1 / "observed.parquet", "valid_time"),
    (
        "ERA5 stage-1 features",
        "pretraining weather",
        STAGE1 / "era5_features.parquet",
        "valid_time",
    ),
    (
        "ERA5 wind components and cloud",
        "pretraining weather",
        STAGE1 / "era5_wind_components.parquet",
        "valid_time",
    ),
    (
        "ERA5 farm cells",
        "wind and solar pretraining",
        REANALYSIS / "ERA5" / "farm_cells_00288" / "era5_farm_cells.parquet",
        "time",
    ),
    (
        "ERA5 irradiance at solar sites",
        "solar pretraining",
        REANALYSIS / "ERA5" / "site_points" / "beam_diffuse_open_meteo.parquet",
        "time",
    ),
    (
        "ERA5 wind at wind sites",
        "wind pretraining",
        REANALYSIS / "ERA5" / "site_points" / "wind_era5.parquet",
        "time",
    ),
    (
        "CAMS irradiance at demand points",
        "demand pretraining",
        REANALYSIS / "CAMS" / "demand_points" / "cams_demand_points.parquet",
        "time",
    ),
    (
        "CAMS irradiance at solar sites",
        "solar pretraining",
        REANALYSIS / "CAMS" / "site_points" / "beam_diffuse_cams.parquet",
        "time",
    ),
]
DAILY_DIRS: Final[list[tuple[str, str, Path]]] = [
    ("ENS summary (experiment 397)", "ensemble mean and spread inputs", WORK_397 / "ens_summary"),
    (
        "ENS summary with wind components (experiment 561)",
        "derived input uv",
        WORK_397 / "ens_summary_uv",
    ),
    ("ENS total cloud cover (experiment 561)", "derived input tcc", WORK_397 / "ens_summary_tcc"),
]


def main() -> None:
    rows = []
    for name, role, path, col in TABLES:
        end = pl.scan_parquet(path).select(pl.col(col).max()).collect().item()
        rows.append({"input": name, "role": role, "newest": end})
    for name, role, folder in DAILY_DIRS:
        files = sorted(folder.glob("init=*.parquet"))
        run = pl.read_parquet(files[-1], columns=["valid_time"])["valid_time"].max()
        rows.append(
            {
                "input": name,
                "role": role,
                "newest": run,
                "newest_run": files[-1].stem.removeprefix("init="),
            }
        )
    out = pl.DataFrame(rows, infer_schema_length=None)
    out.write_parquet(OUT_DIR / "inputs_end.parquet")
    pl.Config.set_tbl_rows(50)
    pl.Config.set_fmt_str_lengths(60)
    print(out)


if __name__ == "__main__":
    main()

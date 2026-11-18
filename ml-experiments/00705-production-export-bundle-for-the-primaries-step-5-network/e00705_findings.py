# ruff: noqa: D103, E501, SIM115, PTH123
"""Experiment 705: numerical evidence for findings (1) and (2) of the export-bundle brief.

Usage: `python e00705_findings.py OUT_JSON`. It prints the evidence and writes it as JSON; the
prose of `findings.txt` quotes these numbers. Finding (1): for ENS run 2026-04-01 the stored summary
equals the member-wise interpolation followed by the member mean and std, and differs from
averaging the native members first. Finding (2): on the 13 runs before the origin 2026-04-01
09:00, the research analysis proxy (ensemble mean of each run's first day) is compared with
production's `select_analysis_proxy` (control member, freshest run), and with a prototype of the
smallest production change.
"""

import json
import sys
from datetime import UTC, datetime, timedelta

import e00048_build_frames as build  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00677_extend as ext  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from weather_utils.analysis_proxy import select_analysis_proxy

RUN = datetime(2026, 4, 1, tzinfo=UTC)
NINE_HOURS = timedelta(hours=9)
VARS = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)


def as_float(value: object) -> float:
    return float(value)  # ty: ignore[invalid-argument-type]


def interpolation_order() -> dict:
    targets = prep.primaries().select("time_series_id", "h3_res_5", "latitude", "longitude")
    cells = targets["h3_res_5"].unique().to_list()
    native = (
        pl.scan_parquet(build.partition_path(run=RUN) / "*.parquet")
        .select(build.SOURCE_COLUMNS)
        .filter(pl.col("h3_index").is_in(cells))
        .join(
            targets.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
        )
        .drop("h3_index")
        .with_columns(nwp_init_time=pl.lit(RUN).cast(pl.Datetime("us", "UTC")))
    )
    coordinates = targets.lazy().select("time_series_id", "latitude", "longitude")
    rename = {v: k for k, v in build.ENS_COLUMNS.items()}
    member_wise = (
        _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates).collect().rename(rename)
    )
    member_wise = member_wise.with_columns(
        wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt()
    )
    keys = ["time_series_id", "nwp_init_time", "valid_time"]
    interp_then_mean = (
        member_wise.group_by(keys)
        .agg(
            *[pl.col(v).mean().alias(f"mean_{v}") for v in VARS],
            *[pl.col(v).std(ddof=0).alias(f"std_{v}") for v in VARS],
        )
        .sort(keys)
    )
    # alternative: mean (and std) over the 51 members of the native values first, then one interpolation
    nat = native.collect()
    value_cols = [
        c
        for c in nat.columns
        if c
        not in {
            "time_series_id",
            "nwp_init_time",
            "valid_time",
            "ensemble_member",
            "categorical_precipitation_type_surface",
        }
    ]
    nat_mean = (
        nat.group_by("time_series_id", "nwp_init_time", "valid_time")
        .agg(
            *[pl.col(c).mean() for c in value_cols],
            pl.col("categorical_precipitation_type_surface").mean().round().cast(pl.Int16),
        )
        .with_columns(ensemble_member=pl.lit(0, pl.Int8))
    )
    mean_first = (
        _upsample_nwp_to_half_hourly(
            nat_mean.select(nat.columns).lazy(), solar_coordinates=coordinates
        )
        .collect()
        .rename(rename)
    )
    mean_first = mean_first.with_columns(
        wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt()
    ).sort(keys)
    stored = pl.read_parquet(ext.E607 / "ens_summary" / f"init={RUN:%Y-%m-%d}.parquet").sort(
        "time_series_id", "valid_time"
    )
    joined = interp_then_mean.join(
        mean_first.select(*keys, *[pl.col(v).alias(f"alt_{v}") for v in VARS]), on=keys, how="inner"
    ).join(
        stored.select(
            "time_series_id",
            "valid_time",
            *[pl.col(f"mean_{v}").alias(f"stored_{v}") for v in VARS],
        ),
        on=["time_series_id", "valid_time"],
        how="inner",
    )
    out = {}
    for v in VARS:
        a = joined[f"mean_{v}"].to_numpy().astype(np.float64)
        out[v] = {
            "member_wise_vs_stored_max_abs": float(
                np.abs(a - joined[f"stored_{v}"].to_numpy().astype(np.float64)).max()
            ),
            "mean_first_vs_stored_max_abs": float(
                np.abs(
                    joined[f"alt_{v}"].to_numpy().astype(np.float64)
                    - joined[f"stored_{v}"].to_numpy().astype(np.float64)
                ).max()
            ),
            "stored_abs_max": float(np.abs(joined[f"stored_{v}"].to_numpy()).max()),
        }
    return out


def research_history(*, series_id: int, runs: list[datetime]) -> pl.DataFrame:
    """Return the research proxy rows: each run's ensemble mean first day, valid in (origin, origin + 24 h]."""
    parts = []
    for run in runs:
        s = pl.read_parquet(ext.E607 / "ens_summary" / f"init={run:%Y-%m-%d}.parquet").filter(
            pl.col("time_series_id") == series_id
        )
        origin = run + NINE_HOURS
        parts.append(
            s.filter(
                (pl.col("valid_time") > origin)
                & (pl.col("valid_time") <= origin + timedelta(hours=24))
            ).select(
                "valid_time",
                t_research=pl.col("mean_temperature_2m").cast(pl.Float64),
                s_research=pl.col("mean_shortwave").cast(pl.Float64),
            )
        )
    return pl.concat(parts).sort("valid_time")


def proxy_comparison() -> dict:
    row = prep.primaries().filter(pl.col("site") == "D2").row(0, named=True)
    series_id = int(row["time_series_id"])
    targets = (
        prep.primaries()
        .select("time_series_id", "h3_res_5", "latitude", "longitude")
        .filter(pl.col("time_series_id") == series_id)
    )
    runs = [RUN - timedelta(days=d) for d in range(13)][::-1]
    origin = RUN + NINE_HOURS
    frames = []
    for run in runs:
        half, _ = build._half_hourly_rows(run=run, targets=targets)
        frames.append(
            half.select(
                "time_series_id",
                "ensemble_member",
                "valid_time",
                "nwp_init_time",
                "temperature_2m",
                "shortwave",
            )
        )
    nwp = pl.concat(frames)
    research = research_history(series_id=series_id, runs=runs)
    research = research.filter(pl.col("valid_time") <= origin)

    def compare(frame: pl.DataFrame, label: str) -> dict:
        j = research.join(frame.select("valid_time", "t", "s"), on="valid_time", how="left")
        dt = (j["t"] - j["t_research"]).abs()
        ds = (j["s"] - j["s_research"]).abs()
        return {
            "label": label,
            "half_hours": j.height,
            "missing": int(j["t"].is_null().sum()),
            "temperature_max_abs_diff_c": as_float(dt.max()),
            "temperature_mean_abs_diff_c": as_float(dt.mean()),
            "shortwave_max_abs_diff_w_m2": as_float(ds.max()),
            "shortwave_mean_abs_diff_w_m2": as_float(ds.mean()),
        }

    prod = (
        select_analysis_proxy(
            nwp.lazy().select(
                "time_series_id",
                "ensemble_member",
                "valid_time",
                "nwp_init_time",
                "temperature_2m",
                "shortwave",
            ),
            group_key="time_series_id",
            init_time_col="nwp_init_time",
        )
        .collect()
        .select(
            "valid_time",
            t=pl.col("temperature_2m").cast(pl.Float64),
            s=pl.col("shortwave").cast(pl.Float64),
        )
    )
    out = {
        "production_unmodified": compare(
            prod,
            "select_analysis_proxy as installed: control member, freshest run, per-column fallback",
        )
    }
    # prototype of the smallest change: mean over members of the interpolated half-hours before the freshest-run reduction, and a window of valid times in (init + 9 h, init + 33 h]
    windowed = nwp.filter(
        (pl.col("valid_time") > pl.col("nwp_init_time") + NINE_HOURS)
        & (pl.col("valid_time") <= pl.col("nwp_init_time") + timedelta(hours=33))
    )
    mean = (
        windowed.group_by("time_series_id", "valid_time", "nwp_init_time")
        .agg(pl.col("temperature_2m").mean(), pl.col("shortwave").mean())
        .with_columns(ensemble_member=pl.lit(0, pl.Int8))
    )
    proto = (
        select_analysis_proxy(
            mean.lazy(), group_key="time_series_id", init_time_col="nwp_init_time"
        )
        .collect()
        .select(
            "valid_time",
            t=pl.col("temperature_2m").cast(pl.Float64),
            s=pl.col("shortwave").cast(pl.Float64),
        )
    )
    out["prototype_member_mean_and_window"] = compare(
        proto,
        "ensemble mean of the interpolated members, valid in (init + 9 h, init + 33 h], then the installed reduction",
    )
    # which run the installed function picks at the origin: the 00Z run of the origin day itself
    return out


def main() -> None:
    evidence = {"interpolation_order": interpolation_order(), "analysis_proxy": proxy_comparison()}
    text = json.dumps(evidence, indent=1)
    print(text)
    open(sys.argv[1], "w").write(text)


if __name__ == "__main__":
    main()

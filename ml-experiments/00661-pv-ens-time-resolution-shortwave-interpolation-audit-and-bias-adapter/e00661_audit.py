# ruff: noqa: E501
"""Experiment 661, Part 1: audit how ensemble shortwave becomes half-hourly values.

Three questions, all on the six signed-MW Primaries and development months only:

1. `identity`: does the experiment pipeline (experiment 48's `_half_hourly_rows`, which feeds every
   summary table and the experiment 644 rows) give the same half-hourly shortwave as production's
   `_radiation.py`, and as a fresh call of the pinned `ml_core` resampler on the native archive?
2. `lag`: which shift (in half-hours) of the reconstructed ensemble shortwave maximises its
   correlation with CAMS, per Primary, and which time-stamp convention does CAMS follow?
3. `score`: error of four reconstruction methods against CAMS (production, linear flux, step-hold)
   in two settings: (A) known weather degraded to 3-hour and 6-hour windows and rebuilt, which
   isolates the interpolation, and (B) the ensemble-mean forecast against CAMS by lead band.

Usage: python e00661_audit.py identity|lag|score
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
for _d in sorted(ROOT.iterdir()):
    if _d.is_dir() and _d.name[:1] == "0" and str(_d) not in sys.path:
        sys.path.append(str(_d))

import e00048_build_frames as b  # ty: ignore[unresolved-import]  # noqa: E402
import e00397_prep as prep  # ty: ignore[unresolved-import]  # noqa: E402
import e00418_prep as p418  # ty: ignore[unresolved-import]  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from ml_core.features._nwp import _upsample_nwp_to_half_hourly  # noqa: E402
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402

NAME: Final = "00661-pv-ens-time-resolution-shortwave-interpolation-audit-and-bias-adapter"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work" / "audit"
SW: Final = "downward_short_wave_radiation_flux_surface"
TEST_MONTHS: Final[tuple[str, ...]] = ("2025-07", "2026-03", "2026-06")
HOLDOUT_START: Final = datetime(2026, 7, 1, tzinfo=UTC)
"""Half-hours ending after this stamp are never scored."""
SUNNY_KT: Final = 0.80
"""A day is sunny when CAMS daily energy is at least this share of CAMS clear-sky energy."""
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "day 1-2": (0.0, 48.0), "day 3-5": (48.0, 120.0), "day 6-10": (120.0, 240.0),
}  # fmt: skip


def six_sites() -> list[str]:
    """Return the six signed-MW Primaries' site labels (experiment 634's list)."""
    import e00634_common as k634  # ty: ignore[unresolved-import]

    return list(k634.six_indices())


def six_table() -> pl.DataFrame:
    """Return the six Primaries' ids, labels, coordinates, and ENS grid cells."""
    sites = six_sites()
    return prep.primaries().filter(pl.col("site").is_in(sites)).select(
        "time_series_id", "site", "latitude", "longitude", "h3_res_5"
    )  # fmt: skip


def run_days(*, months: tuple[str, ...], every: int = 3) -> list[datetime]:
    """Return 00:00 UTC runs on every `every`-th day of the given months."""
    out = []
    for m in months:
        d = datetime.strptime(m + "-01", "%Y-%m-%d").replace(tzinfo=UTC)
        while d.strftime("%Y-%m") == m:
            if d.day % every == 0:
                out.append(d)
            d += timedelta(days=1)
    return out


def load_cams() -> pl.DataFrame:
    """Return CAMS global horizontal irradiance for the six Primaries (hourly, stamped as in the file)."""
    return (
        pl.read_parquet(p418.CAMS_PATH)
        .filter(pl.col("site").is_in(six_sites()))
        .sort("site", "time")
    )


def native_ens(*, run: datetime, cells: list[int]) -> pl.DataFrame:
    """Return native shortwave rows (cell, member, valid_time, W/m2) of one run, lead above zero."""
    return (
        pl.scan_parquet(b.partition_path(run=run) / "*.parquet")
        .select("h3_index", "ensemble_member", "valid_time", SW)
        .filter(pl.col("h3_index").is_in(cells))
        .collect()
        .filter(pl.col("valid_time") > run)
    )


def _frame(*, native: pl.DataFrame, run: datetime, table: pl.DataFrame) -> pl.LazyFrame:
    """Join native rows to the six Primaries so the resampler sees `time_series_id` and `nwp_init_time`."""
    return (
        native.lazy()
        .join(table.lazy().select("time_series_id", "h3_res_5"), left_on="h3_index", right_on="h3_res_5")
        .drop("h3_index")
        .with_columns(nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC")))
    )  # fmt: skip


def method_production(*, native: pl.LazyFrame, table: pl.DataFrame) -> pl.DataFrame:
    """Return production's half-hourly shortwave (ml_core `_radiation`: window-mean index interpolation, energy conserved)."""
    coords = table.lazy().select("time_series_id", "latitude", "longitude")
    return _upsample_nwp_to_half_hourly(native, solar_coordinates=coords).collect()


def method_step(*, native: pl.DataFrame) -> pl.DataFrame:
    """Return step-hold: every half-hour of a native window takes the window mean (energy conserved, no shape)."""
    leads = [0, *range(3, 145, 3), *range(150, 361, 6)]
    width = {leads[i + 1]: leads[i + 1] - leads[i] for i in range(len(leads) - 1)}
    frames = []
    for lead_end, w in width.items():
        sub = native.filter(
            (pl.col("valid_time") - pl.col("nwp_init_time")).dt.total_hours() == lead_end
        )
        if sub.is_empty():
            continue
        grid = pl.DataFrame({"_k": list(range(w * 2))})
        frames.append(
            sub.join(grid, how="cross").with_columns(
                valid_time=pl.col("valid_time") - pl.duration(minutes=30 * (w * 2 - 1 - pl.col("_k")))
            ).drop("_k")
        )  # fmt: skip
    return pl.concat(frames)


def method_linear(*, native: pl.DataFrame) -> pl.DataFrame:
    """Return linear flux interpolation between native stamps, treating each period-ending mean as a value at its stamp.

    This is how the resampler worked before the 2026-10-12 radiation change: no de-accumulation, no
    clear-sky shape, no energy conservation.
    """
    keys = [c for c in native.columns if c not in ("valid_time", SW)]
    start = native.group_by(keys).agg(
        pl.col("valid_time").min().alias("_a"), pl.col("valid_time").max().alias("_z")
    )
    grid = (
        start.with_columns(
            valid_time=pl.datetime_ranges(pl.col("_a"), pl.col("_z"), interval="30m")
        )
        .drop("_a", "_z")
        .explode("valid_time")
    )
    joined = grid.join(native, on=[*keys, "valid_time"], how="left").sort([*keys, "valid_time"])
    return joined.with_columns(pl.col(SW).interpolate().over(keys))


def cams_half_hourly(
    *, cams: pl.DataFrame, table: pl.DataFrame, stamp: str = "end"
) -> pl.DataFrame:
    """Put hourly CAMS on the half-hour-ending grid; both half-hours of an hour take the hour's value.

    `stamp="end"` is experiment 14's convention (the file's time is the end of the hour averaged);
    `stamp="start"` shifts the hour one step earlier.
    """
    shift = 0 if stamp == "end" else 1
    both = pl.DataFrame({"_o": [0, 30]})
    return (
        cams.join(table.select("time_series_id", "site"), on="site")
        .join(both, how="cross")
        .with_columns(valid_time=pl.col("time") + pl.duration(hours=shift) - pl.duration(minutes=pl.col("_o")))
        .select("time_series_id", "valid_time", cam=pl.col("ghi_w_m2").cast(pl.Float64), cam_clear=pl.col("clear_sky_ghi_w_m2").cast(pl.Float64))
    )  # fmt: skip


def ens_mean_half_hourly(*, run: datetime, table: pl.DataFrame, method: str) -> pl.DataFrame:
    """Return the ensemble-mean half-hourly shortwave of one run for the six Primaries by one method."""
    native_raw = native_ens(run=run, cells=table["h3_res_5"].unique().to_list())
    native = _frame(native=native_raw, run=run, table=table)
    if method == "production":
        hh = method_production(native=native, table=table)
    elif method == "step":
        hh = method_step(native=native.collect())
    elif method == "linear":
        hh = method_linear(native=native.collect())
    else:
        raise ValueError(method)
    return (
        hh.filter(pl.col("valid_time") <= HOLDOUT_START)
        .group_by("time_series_id", "valid_time").agg(pl.col(SW).mean().alias("ens"))
        .with_columns(run=pl.lit(run).cast(pl.Datetime("us", "UTC")))
    )  # fmt: skip


def sunny_days(*, cams: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """Return the (time_series_id, date) pairs whose CAMS daily energy is at least `SUNNY_KT` of clear sky."""
    hh = cams_half_hourly(cams=cams, table=table)
    day = hh.with_columns(date=(pl.col("valid_time") - pl.duration(minutes=1)).dt.date()).group_by("time_series_id", "date").agg(
        pl.col("cam").sum().alias("e"), pl.col("cam_clear").sum().alias("c")
    )  # fmt: skip
    return day.filter((pl.col("c") > 0) & (pl.col("e") / pl.col("c") >= SUNNY_KT)).select(
        "time_series_id", "date"
    )


def cmd_lag() -> None:
    """Find the shift that maximises the correlation of ensemble shortwave with CAMS, per Primary, and test CAMS's own stamp."""
    table = six_table()
    cams = load_cams()
    sunny = sunny_days(cams=cams, table=table)
    runs = run_days(months=TEST_MONTHS, every=3)
    ens = pl.concat([ens_mean_half_hourly(run=r, table=table, method="production") for r in runs])
    ens = ens.with_columns(
        lead=(pl.col("valid_time") - pl.col("run")).dt.total_minutes() / 60.0
    ).filter(pl.col("lead") <= 30.0)
    result: dict = {"ens_vs_cams": {}, "cams_vs_clear_sky": {}}
    for stamp in ("end", "start"):
        hh = cams_half_hourly(cams=cams, table=table, stamp=stamp)
        for site, tid in zip(table["site"], table["time_series_id"], strict=True):
            row = {}
            for k in range(-8, 9):
                shifted = hh.filter(pl.col("time_series_id") == tid).with_columns(
                    valid_time=pl.col("valid_time") - pl.duration(minutes=30 * k)
                )
                j = (
                    ens.filter(pl.col("time_series_id") == tid)
                    .join(shifted.select("valid_time", "cam"), on="valid_time")
                    .with_columns(date=(pl.col("valid_time") - pl.duration(minutes=1)).dt.date())
                )
                j = j.join(sunny.filter(pl.col("time_series_id") == tid), on=["date"]).filter(
                    (pl.col("ens") > 5) | (pl.col("cam") > 5)
                )
                row[k] = float(np.corrcoef(j["ens"].to_numpy(), j["cam"].to_numpy())[0, 1])
            best = max(row, key=row.get)  # ty: ignore[no-matching-overload]
            result["ens_vs_cams"][f"{site}/{stamp}"] = {
                "best_lag_half_hours": best,
                "corr_at_best": row[best],
                "corr_at_0": row[0],
                "all": row,
            }
            print(
                site,
                stamp,
                "best lag",
                best,
                round(row[best], 4),
                "at 0:",
                round(row[0], 4),
                flush=True,
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lag.json").write_text(json.dumps(result, indent=1))


METHODS: Final[tuple[str, ...]] = ("production", "linear", "step")


def reconstruct(*, native: pl.DataFrame, table: pl.DataFrame, method: str) -> pl.DataFrame:
    """Rebuild half-hourly shortwave from native window means by one method."""
    if method == "production":
        return method_production(native=native.lazy(), table=table)
    if method == "step":
        return method_step(native=native)
    return method_linear(native=native)


def degraded_native(
    *, cams: pl.DataFrame, table: pl.DataFrame, width_h: int, days: list
) -> pl.DataFrame:
    """Average CAMS to native windows of `width_h` hours on the ENS schedule, one trajectory per day.

    A 3-hour window sits at leads 3 to 24 of a run at 00:00 UTC. A 6-hour window sits at leads 150
    to 168 of a run that started 144 hours before the day, so the production resampler sees the
    6-hourly part of the ENS schedule.
    """
    hh = cams_half_hourly(cams=cams, table=table)
    out = []
    for d in days:
        start = datetime(d.year, d.month, d.day, tzinfo=UTC)
        init = start if width_h == 3 else start - timedelta(hours=144)
        windows = [start + timedelta(hours=width_h * (i + 1)) for i in range(24 // width_h)]
        for end in windows:
            sel = hh.filter(
                (pl.col("valid_time") > end - timedelta(hours=width_h))
                & (pl.col("valid_time") <= end)
            )
            m = sel.group_by("time_series_id").agg(pl.col("cam").mean().alias(SW))
            out.append(
                m.with_columns(
                    valid_time=pl.lit(end).cast(pl.Datetime("us", "UTC")),
                    nwp_init_time=pl.lit(init).cast(pl.Datetime("us", "UTC")),
                    ensemble_member=pl.lit(0, dtype=pl.Int64),
                )
            )
    return pl.concat(out).select(
        "time_series_id", "ensemble_member", "valid_time", SW, "nwp_init_time"
    )


def score_frame(*, joined: pl.DataFrame, group: list[str]) -> pl.DataFrame:
    """Return error of `ens` against `cam` by group on sunny-day half-hours: RMSE, MAE, bias, % of peak, midday peak bias, energy bias."""
    d = joined.with_columns(
        date=(pl.col("valid_time") - pl.duration(minutes=1)).dt.date()
    ).with_columns(err=pl.col("ens") - pl.col("cam"))
    extra = ["run"] if "run" in d.columns else []
    peak = d.group_by("time_series_id", "date", *extra, *group).agg(
        pl.col("cam").max().alias("cam_peak"), pl.col("ens").max().alias("ens_peak")
    )
    base = d.group_by(group).agg(
        (pl.col("err") ** 2).mean().sqrt().alias("rmse_w_m2"), pl.col("err").abs().mean().alias("mae_w_m2"),
        pl.col("err").mean().alias("bias_w_m2"), pl.len().alias("n"),
    )  # fmt: skip
    pk = peak.group_by(group).agg(
        pl.col("cam_peak").mean().alias("cam_peak_mean"), (pl.col("ens_peak") - pl.col("cam_peak")).mean().alias("peak_bias_w_m2"),
    )  # fmt: skip
    out = base.join(pk, on=group) if group else base.join(pk, how="cross")
    return out.with_columns(
        rmse_pct_peak=100 * pl.col("rmse_w_m2") / pl.col("cam_peak_mean"),
        mae_pct_peak=100 * pl.col("mae_w_m2") / pl.col("cam_peak_mean"),
        peak_bias_pct=100 * pl.col("peak_bias_w_m2") / pl.col("cam_peak_mean"),
    )  # fmt: skip


def cmd_score() -> None:
    """Score the reconstructions against CAMS: (A) degraded known weather, (B) ensemble forecast by lead band."""
    table = six_table()
    cams = load_cams()
    sunny = sunny_days(cams=cams, table=table)
    hh = cams_half_hourly(cams=cams, table=table)
    OUT.mkdir(parents=True, exist_ok=True)
    # (A) degrade known weather
    days_all = sorted(
        {
            d
            for m in TEST_MONTHS
            for d in [
                datetime.strptime(m + "-01", "%Y-%m-%d").replace(tzinfo=UTC).date()
                + timedelta(days=i)
                for i in range(31)
            ]
            if d.strftime("%Y-%m") == m
        }
    )
    days = [d for d in days_all if d.day % 2 == 0]
    rows = []
    for width in (3, 6):
        native = degraded_native(cams=cams, table=table, width_h=width, days=days)
        for method in METHODS:
            rec = reconstruct(native=native, table=table, method=method)
            j = rec.select("time_series_id", "valid_time", pl.col(SW).alias("ens")).join(
                hh.select("time_series_id", "valid_time", "cam"),
                on=["time_series_id", "valid_time"],
            )
            j = (
                j.with_columns(date=(pl.col("valid_time") - pl.duration(minutes=1)).dt.date())
                .join(sunny, on=["time_series_id", "date"])
                .drop("date")
            )
            sc = score_frame(joined=j, group=[]).with_columns(
                window_hours=pl.lit(width), method=pl.lit(method)
            )
            rows.append(sc)
            print(
                "A",
                width,
                method,
                sc.select("rmse_w_m2", "rmse_pct_peak", "peak_bias_pct").row(0),
                flush=True,
            )
    pl.concat(rows).write_csv(OUT / "score_degraded_cams.csv")
    # (B) ensemble forecasts against CAMS by lead band
    runs = run_days(months=TEST_MONTHS, every=3)
    rows = []
    ens_by = {
        mt: pl.concat([ens_mean_half_hourly(run=r, table=table, method=mt) for r in runs])
        for mt in METHODS
    }
    keys = ["time_series_id", "valid_time", "run"]
    common_keys = ens_by[METHODS[0]].select(keys)
    for mt in METHODS[1:]:
        common_keys = common_keys.join(
            ens_by[mt].select(keys), on=keys
        )  # all methods scored on the same half-hours
    for method in METHODS:
        ens = ens_by[method].join(common_keys, on=keys)
        j = ens.join(hh.select("time_series_id", "valid_time", "cam"), on=["time_series_id", "valid_time"]).with_columns(
            date=(pl.col("valid_time") - pl.duration(minutes=1)).dt.date(), lead=(pl.col("valid_time") - pl.col("run")).dt.total_minutes() / 60.0
        ).join(sunny, on=["time_series_id", "date"])  # fmt: skip
        for band, (lo, hi) in LEAD_BANDS.items():
            sc = score_frame(
                joined=j.filter((pl.col("lead") > lo) & (pl.col("lead") <= hi)), group=[]
            ).with_columns(band=pl.lit(band), method=pl.lit(method))
            rows.append(sc)
            print(
                "B",
                band,
                method,
                sc.select("rmse_w_m2", "rmse_pct_peak", "peak_bias_pct", "bias_w_m2").row(0),
                flush=True,
            )
    pl.concat(rows).write_csv(OUT / "score_ens_vs_cams.csv")


def cmd_identity() -> None:
    """Compare the pipeline's stored summary and `_half_hourly_rows` with a fresh production call."""
    table = six_table()
    rows_out = []
    for run in run_days(months=("2024-06", "2025-02", "2025-07", "2026-03", "2026-06"), every=9):
        mine, _ = b._half_hourly_rows(
            run=run, targets=table.select("time_series_id", "h3_res_5", "latitude", "longitude")
        )
        native = _frame(
            native=native_ens(run=run, cells=table["h3_res_5"].unique().to_list()),
            run=run,
            table=table,
        )
        fresh = (
            method_production(native=native, table=table)
            .select("time_series_id", "ensemble_member", "valid_time", pl.col(SW).alias("fresh"))
        )  # fmt: skip
        j = mine.select("time_series_id", "ensemble_member", "valid_time", "shortwave").join(
            fresh, on=["time_series_id", "ensemble_member", "valid_time"], how="inner"
        )
        d = (j["shortwave"] - j["fresh"]).abs()
        summ = pl.read_parquet(prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet").select(
            "time_series_id", "valid_time", "mean_shortwave"
        )
        mean_fresh = fresh.group_by("time_series_id", "valid_time").agg(pl.col("fresh").mean())
        js = summ.join(mean_fresh, on=["time_series_id", "valid_time"])
        ds = (js["mean_shortwave"] - js["fresh"]).abs()
        rows_out.append({
            "run": f"{run:%Y-%m-%d}", "rows": j.height, "max_abs_diff_rows": d.max(),
            "summary_rows": js.height, "max_abs_diff_summary": ds.max(),
        })  # fmt: skip
        print(rows_out[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "identity.json").write_text(json.dumps(rows_out, indent=1))


def main() -> None:
    """Run one subcommand."""
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["identity", "lag", "score"])
    a = p.parse_args()
    os.environ.setdefault("POLARS_MAX_THREADS", "4")
    {"identity": cmd_identity, "lag": cmd_lag, "score": cmd_score}[a.cmd]()


if __name__ == "__main__":
    main()

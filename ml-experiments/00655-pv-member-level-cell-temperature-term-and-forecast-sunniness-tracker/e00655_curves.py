# ruff: noqa: E501, ANN001
"""Experiment 655: the physics curves, per member, and the cell-temperature scale fitted on the farms.

Curves (all per MW of alternating-current capacity before the per-cutoff normalisation):

- plane of array: Erbs split of the shortwave irradiance, then `studies.pv_physics` at tilt 20
  degrees, azimuth 180 degrees;
- alternating-current output: direct-current to alternating-current ratio 1.15, clipped at 1;
- derate: output times `max(1 - 0.004 (T - 25), 0)`, with T the 2 m air temperature (`air`) or the
  cell temperature `T_air + s * G_poa / (25 + 6.84 * wind)` (`cell`).

Variants of a table: `c0` mean-sky (member means of irradiance, temperature, and wind), air
derate; `c` member-mean of the per-member curve, air derate; `d` as `c` with the cell temperature;
`sd` the standard deviation over members of the `d` curve.

Usage: python e00655_curves.py build --cutoff 2026-06
"""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00655_extract as ex
import numpy as np
import polars as pl
import pvlib
from studies.pv_physics import SunAndSky, ac_power_mw, plane_of_array_w_m2
from studies.solar import azimuth as solar_azimuth
from studies.solar import zenith as solar_zenith

TILT_DEG: Final = 20.0
PANEL_AZIMUTH_DEG: Final = 180.0
DC_AC_RATIO: Final = 1.15
DERATING_PER_KELVIN: Final = 0.004
REFERENCE_CELL_C: Final = 25.0
FAIMAN_U0: Final = 25.0
FAIMAN_U1: Final = 6.84
FARM_WIND_M_S: Final = 3.0
S_PRIOR_SD_LOG: Final = 0.25
FARM_LATITUDE: Final = 52.5
FARM_LONGITUDE: Final = -1.5
"""A nominal central-GB location: the farm frame holds no coordinates (the farms are anonymised)."""
DAY_US: Final = 86_400_000_000
STEP_US: Final = 1_800_000_000
FIRST_ORIGIN_US: Final = int((ex.FIRST_RUN + ex.ORIGIN_DELAY).timestamp() * 1e6)
CURVES_DIR: Final[Path] = ex.WORK / "curves"
VARIANTS: Final[tuple[str, ...]] = ("c0", "c", "d", "sd")


def _unit(*, ghi, zenith_deg, sun_az_deg, doy, temp_air, wind, s: float | None) -> np.ndarray:
    """Return the derated AC output per MW of AC capacity; `s=None` derates on the air temperature."""
    ghi = np.where(zenith_deg < 90.0, np.clip(ghi, 0.0, None), 0.0)
    split = pvlib.irradiance.erbs(ghi=ghi, zenith=zenith_deg, datetime_or_doy=doy)
    sky = SunAndSky(
        half_hour_end_time=np.zeros(ghi.shape, dtype="datetime64[us]"),
        ghi_w_m2=ghi,
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"], dtype=np.float64)),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"], dtype=np.float64)),
        zenith_deg=zenith_deg,
        sun_azimuth_deg=sun_az_deg,
    )
    poa = plane_of_array_w_m2(
        sky=sky, tilt_deg=TILT_DEG, azimuth_deg=PANEL_AZIMUTH_DEG, tracker=False
    )
    ac = ac_power_mw(poa_w_m2=poa, dc_capacity_mw=DC_AC_RATIO, ac_capacity_mw=1.0)
    if s is None:
        cell = temp_air
    else:
        cell = temp_air + s * poa / (FAIMAN_U0 + FAIMAN_U1 * np.clip(wind, 0.0, None))
    return ac * np.clip(1.0 - DERATING_PER_KELVIN * (cell - REFERENCE_CELL_C), 0.0, None)


def geometry(*, primary: int, run_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (zenith, sun azimuth, day of year), each (runs, 702), for the runs' valid times."""
    import e00397_prep as prep  # ty: ignore[unresolved-import]

    row = prep.primaries().filter(pl.col("primary") == primary).row(0, named=True)
    origin = FIRST_ORIGIN_US + run_index.astype(np.int64) * DAY_US
    valid = origin[:, None] + (np.arange(ex.N_STEPS)[None, :] + 1) * STEP_US
    stamps = pl.Series(valid.ravel()).cast(pl.Datetime("us", "UTC"))
    z = solar_zenith(stamps=stamps, latitude=row["latitude"], longitude=row["longitude"])
    a = solar_azimuth(stamps=stamps, latitude=row["latitude"], longitude=row["longitude"])
    doy = stamps.dt.ordinal_day().to_numpy().astype(np.float64)
    return z.reshape(valid.shape), a.reshape(valid.shape), doy.reshape(valid.shape)


def run_list() -> list[datetime]:
    """Return every extracted run."""
    return ex.runs(first=ex.FIRST_RUN, last=ex.LAST_RUN)


def load_members(*, primary: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (shortwave, temperature, wind), each (runs, 702, 51) float32, for one Primary."""
    sw, t, w = [], [], []
    for run in run_list():
        z = np.load(ex.run_path(run=run))
        sw.append(z[f"p{primary}_sw"])
        t.append(z[f"p{primary}_t"])
        w.append(z[f"p{primary}_w"])
    return np.stack(sw), np.stack(t), np.stack(w)


def build_tables(*, primary: int, s: float) -> dict[str, np.ndarray]:
    """Return the curve tables (runs, 702) of one Primary for the scale `s`."""
    sw, t, w = load_members(primary=primary)
    n_runs = sw.shape[0]
    z, a, doy = geometry(primary=primary, run_index=np.arange(n_runs))
    out = {k: np.empty((n_runs, ex.N_STEPS), dtype=np.float32) for k in VARIANTS}
    for lo in range(0, n_runs, 20):
        hi = min(lo + 20, n_runs)
        zz, aa, dd = (x[lo:hi, :, None] for x in (z, a, doy))
        args = {"zenith_deg": zz, "sun_az_deg": aa, "doy": dd}
        member_air = _unit(
            ghi=sw[lo:hi].astype(np.float64),
            temp_air=t[lo:hi].astype(np.float64),
            wind=w[lo:hi].astype(np.float64),
            s=None,
            **args,
        )
        member_cell = _unit(
            ghi=sw[lo:hi].astype(np.float64),
            temp_air=t[lo:hi].astype(np.float64),
            wind=w[lo:hi].astype(np.float64),
            s=s,
            **args,
        )
        mean_sky = _unit(
            ghi=np.nanmean(sw[lo:hi].astype(np.float64), axis=2, keepdims=True),
            temp_air=np.nanmean(t[lo:hi].astype(np.float64), axis=2, keepdims=True),
            wind=np.nanmean(w[lo:hi].astype(np.float64), axis=2, keepdims=True), s=None, **args,
        )  # fmt: skip
        out["c0"][lo:hi] = mean_sky[:, :, 0]
        # A member with a missing value is left out of the mean, as the cached summaries do (polars skips nulls).
        out["c"][lo:hi] = np.nanmean(member_air, axis=2)
        out["d"][lo:hi] = np.nanmean(member_cell, axis=2)
        out["sd"][lo:hi] = np.nanstd(member_cell, axis=2)
    return out


# --- the cell-temperature scale ---------------------------------------------------------------


def air_temperature_proxy(*, stop: datetime) -> pl.DataFrame:
    """Return the air temperature (degrees C) by valid time before `stop`, for the farm fit.

    The farm frame holds no temperature, so the proxy is the ensemble-mean 2 m temperature of each
    run's first 24 hours, averaged over the six MW Primaries (central-GB scale). It uses only runs
    whose first 24 hours end before `stop`.
    """
    import e00634_common as k634  # ty: ignore[unresolved-import]

    six = list(k634.six_indices().values())
    times, temps = [], []
    for run in run_list():
        origin = run + ex.ORIGIN_DELAY
        if origin + timedelta(hours=24) >= stop:
            break
        z = np.load(ex.run_path(run=run))
        t = np.mean([z[f"p{p}_t"][:48].mean(axis=1) for p in six], axis=0)
        base = int(origin.timestamp() * 1e6)
        times.append(base + (np.arange(48) + 1) * STEP_US)
        temps.append(t)
    return pl.DataFrame(
        {"t_us": np.concatenate(times), "temp_air": np.concatenate(temps).astype(np.float64)}
    ).unique(subset=["t_us"])


def fit_scale(*, cutoff: str) -> dict:
    """Fit the cell-temperature scale on the farms' sunny-midday output before the cutoff.

    Returns the posterior mean and standard deviation of `s` on a grid, with the prior (log-normal,
    sd 0.25 in the log) and a free amplitude per farm.
    """
    import e00405_prep as p5  # ty: ignore[unresolved-import]

    stop = datetime.fromisoformat(f"{cutoff}-01").replace(tzinfo=UTC)
    frame = pl.read_parquet(p5.FARM_FRAME).filter(pl.col("kind") == "farm")
    proxy = air_temperature_proxy(stop=stop)
    grid = np.exp(np.linspace(-2 * S_PRIOR_SD_LOG, 2 * S_PRIOR_SD_LOG, 81))
    per_farm = []
    n_rows = 0
    n_days = 0
    for farm in sorted(frame["site"].unique().to_list()):
        f = frame.filter(pl.col("site") == farm, pl.col("time") < stop).sort("time")
        f = (
            f.with_columns(t_us=pl.col("time").dt.epoch("us"))
            .drop("temperature_2m")
            .join(proxy, on="t_us", how="inner")
        )
        t_us, norm = p5.farm_norm_series(farm=farm)
        f = f.with_columns(
            norm=pl.Series(
                np.interp(f["time"].dt.epoch("us").to_numpy(), t_us, norm, left=np.nan, right=np.nan)
            ),
            hour=pl.col("time").dt.hour(),
            kt=pl.col("ghi_u") / pl.col("ghi_u_clear").clip(lower_bound=1e-6),
        ).filter(
            pl.col("norm").is_finite(), pl.col("ghi_u").is_finite(), pl.col("kt") >= 0.7, pl.col("hour").is_between(10, 15), pl.col("ghi_u_clear") > 50.0
        )  # fmt: skip
        if f.height < 200:
            continue
        stamps = f["time"]
        zen = solar_zenith(stamps=stamps, latitude=FARM_LATITUDE, longitude=FARM_LONGITUDE)
        az = solar_azimuth(stamps=stamps, latitude=FARM_LATITUDE, longitude=FARM_LONGITUDE)
        doy = stamps.dt.ordinal_day().to_numpy().astype(np.float64)
        y = f["norm"].to_numpy()
        errs = []
        for s in grid:
            m = _unit(
                ghi=f["ghi_u"].to_numpy().astype(np.float64), zenith_deg=zen, sun_az_deg=az, doy=doy,
                temp_air=f["temp_air"].to_numpy().astype(np.float64), wind=np.full(len(y), FARM_WIND_M_S), s=s,
            )  # fmt: skip
            amp = float((m * y).sum() / max((m * m).sum(), 1e-12))
            errs.append(float(((y - amp * m) ** 2).sum()))
        per_farm.append((np.array(errs), len(y)))
        n_rows += len(y)
        n_days += int(f["time"].dt.date().n_unique())
    assert per_farm, "no farm has enough sunny-midday rows before the cutoff"
    sse = sum(e for e, _ in per_farm)
    # Half-hours of one farm-day are strongly correlated, so the likelihood counts one observation
    # per farm-day (the mean number of sunny-midday half-hours per farm-day is `inflate`).
    sigma2 = float(sse.min() / n_rows)  # ty: ignore[unresolved-attribute]
    log_post = -sse / (2 * sigma2 * (n_rows / n_days)) - (np.log(grid) ** 2) / (
        2 * S_PRIOR_SD_LOG**2
    )
    post = np.exp(log_post - log_post.max())
    post /= post.sum()
    mean = float((post * grid).sum())
    sd = float(np.sqrt((post * (grid - mean) ** 2).sum()))
    return {
        "cutoff": cutoff,
        "s_mean": mean,
        "s_sd": sd,
        "s_mle": float(grid[np.argmin(sse)]),
        "grid_lo": float(grid[0]),
        "grid_hi": float(grid[-1]),
        "farm_days": n_days,
        "rows": n_rows,
        "farms": len(per_farm),
    }


def scale_path(*, cutoff: str) -> Path:
    """Return where the fitted scale is stored."""
    return CURVES_DIR / f"scale_{cutoff}.json"


def tables_path(*, cutoff: str, primary: int) -> Path:
    """Return where a Primary's curve tables are stored for a cutoff."""
    return CURVES_DIR / cutoff / f"p{primary}.npz"


def main() -> None:
    """Fit the scale and build the tables of one cutoff."""
    import e00634_common as k634  # ty: ignore[unresolved-import]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--cutoff", required=True)
    a = parser.parse_args()
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    sp = scale_path(cutoff=a.cutoff)
    if not sp.exists():
        res = fit_scale(cutoff=a.cutoff)
        sp.write_text(json.dumps(res))
    res = json.loads(sp.read_text())
    print(res, flush=True)
    for primary in k634.six_indices().values():
        tp = tables_path(cutoff=a.cutoff, primary=primary)
        if tp.exists():
            continue
        tp.parent.mkdir(parents=True, exist_ok=True)
        tables = build_tables(primary=primary, s=res["s_mean"])
        tmp = tp.with_name(tp.name + ".tmp.npz")
        np.savez(tmp, **tables)  # ty: ignore[invalid-argument-type]
        tmp.rename(tp)
        print(f"p{primary} done", flush=True)


if __name__ == "__main__":
    main()

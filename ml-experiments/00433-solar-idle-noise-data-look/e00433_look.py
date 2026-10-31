"""Look at the night-time and daytime-idle readings of the six metered solar farms (no fits).

Reads the stored power of the six farms and the CAMS clear-sky index. Writes private tables and a
private gallery of farm-days under `data/ml-experiments/00433-*/`, and aggregated, normalised
tables (shares only, no raw farm-day) under `docs/ml-experiments/00433-*/`. Farms appear only as
Solar A to Solar F.
"""

import json
from datetime import timedelta
from pathlib import Path
from typing import Final

import matplotlib as mpl
import numpy as np
import pandas as pd  # noqa: TID251  (pvlib and matplotlib take pandas indexes)
import polars as pl
import pvlib
from studies.power import scan_power
from studies.pv_dataset import CAMS_PATH, METADATA_PATH, pv_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR

mpl.use("Agg")
import matplotlib.pyplot as plt

SLUG: Final[str] = "00433-solar-idle-noise-data-look"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / SLUG
NIGHT_ELEVATION_DEG: Final[float] = -6.0
CSI_MIN: Final[float] = 0.7
DAY_HOURS: Final[tuple[int, int]] = (10, 15)
MIN_QUALIFYING_PER_DAY: Final[int] = 3
IDLE_MARGIN_OF_CAPACITY: Final[float] = 0.02
QUANT_MIN_SHARE: Final[float] = 0.01
LATTICE_TOL_MW: Final[float] = 1e-6
N_BOOT: Final[int] = 2000
SEED: Final[int] = 433
GALLERY_DAYS: Final[int] = 10
SUMMER: Final[tuple[int, ...]] = (5, 6, 7, 8)
WINTER: Final[tuple[int, ...]] = (11, 12, 1, 2)


def prepare(
    *, tsid: int, lat: float, lon: float, site: str, power: pl.DataFrame, cams: pl.DataFrame
) -> pl.DataFrame:
    """Return one farm's half-hours with elevation and the CAMS clear-sky index."""
    p = (
        power.filter(pl.col("time_series_id") == tsid)
        .select("time", "power")
        .unique("time")
        .sort("time")
    )
    mid = pd.DatetimeIndex(
        (p["time"] - timedelta(minutes=15)).dt.convert_time_zone("UTC").to_numpy(), tz="UTC"
    )
    sp = pvlib.solarposition.get_solarposition(mid, lat, lon)
    p = p.with_columns(
        elevation=pl.Series(sp["apparent_elevation"].to_numpy()),
        # A half-hour ending at T belongs to the hour ending at ceil(T).
        hour_stamp=pl.col("time").dt.offset_by("-1ns").dt.truncate("1h").dt.offset_by("1h"),
    )
    c = (
        cams.filter(pl.col("site") == site)
        .select(
            pl.col("time").alias("hour_stamp"),
            csi=(pl.col("ghi_w_m2") / pl.col("clear_sky_ghi_w_m2")).clip(0, 2),
            cs=pl.col("clear_sky_ghi_w_m2"),
        )
        .with_columns(csi=pl.when(pl.col("cs") > 50).then(pl.col("csi")))
    )
    return p.join(c.select("hour_stamp", "csi"), on="hour_stamp", how="left")


def lattice_step(values: np.ndarray) -> float:
    """Smallest gap between adjacent distinct values that at least 1% of readings sit above."""
    uniq, counts = np.unique(np.round(values, 6), return_counts=True)
    if len(uniq) < 2:
        return float("nan")
    gaps = np.round(np.diff(uniq), 6)
    weights = counts[1:]
    order = np.argsort(gaps, kind="stable")
    cum = np.cumsum(weights[order]) / len(values)
    idx = int(np.searchsorted(cum, QUANT_MIN_SHARE))
    return float(gaps[order][min(idx, len(order) - 1)])


def lattice_share(values: np.ndarray, step: float) -> float:
    """Share of readings that are an integer multiple of the step."""
    if not np.isfinite(step) or step <= 0:
        return float("nan")
    k = values / step
    return float(np.mean(np.abs(k - np.round(k)) * step < LATTICE_TOL_MW))


def night_stats(df: pl.DataFrame, cap: float) -> dict[str, float]:
    """Return the pre-registered night statistics for the rows of `df` (night rows only)."""
    v = df["power"].to_numpy()
    prev = df["prev_power"].to_numpy()
    zero = v == 0
    rep_nonzero = (v == prev) & ~zero
    step = lattice_step(v)
    return {
        "n": float(len(v)),
        "zero_share": float(zero.mean()),
        "nonzero_repeat_share": float(rep_nonzero.mean()),
        "zero_or_repeat_share": float((zero | rep_nonzero).mean()),
        "negative_share": float((v < 0).mean()),
        "median_frac_cap": float(np.median(v) / cap),
        "p95_frac_cap": float(np.quantile(v, 0.95) / cap),
        "step_mw": step,
        "lattice_share": lattice_share(v, step),
    }


def summer_winter_ratio(df: pl.DataFrame, cap: float) -> float:
    """Return max/min of summer and winter night medians (1.0 if both are zero, inf if one is)."""
    s = df.filter(pl.col("month").is_in(SUMMER))["power"].to_numpy()
    w = df.filter(pl.col("month").is_in(WINTER))["power"].to_numpy()
    a, b = abs(float(np.median(s))) / cap, abs(float(np.median(w))) / cap
    lo, hi = min(a, b), max(a, b)
    if hi == 0:
        return 1.0
    return float("inf") if lo == 0 else hi / lo


def bootstrap(
    df: pl.DataFrame, cap: float, rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
    """Year-block bootstrap 90% intervals for the four criteria's statistics."""
    years = sorted(df["year"].unique().to_list())
    parts = {y: df.filter(pl.col("year") == y) for y in years}
    rows = []
    for _ in range(N_BOOT):
        pick = rng.choice(years, size=len(years), replace=True)
        b = pl.concat([parts[int(y)] for y in pick])
        st = night_stats(b, cap)
        rows.append(
            (
                st["zero_or_repeat_share"],
                st["lattice_share"],
                st["median_frac_cap"],
                st["negative_share"],
                summer_winter_ratio(b, cap),
            )
        )
    a = np.array(rows, dtype=float)
    names = [
        "zero_or_repeat_share",
        "lattice_share",
        "median_frac_cap",
        "negative_share",
        "summer_winter_ratio",
    ]
    return {
        n: (float(np.nanquantile(a[:, i], 0.05)), float(np.nanquantile(a[:, i], 0.95)))
        for i, n in enumerate(names)
    }


def classify(
    st: dict[str, float], ci: dict[str, tuple[float, float]], ratio: float
) -> tuple[str, str]:
    """Apply the pre-registered rule; return the label and the failing criteria."""
    c1 = ci["zero_or_repeat_share"][0] >= 0.20
    c2 = ci["lattice_share"][0] >= 0.80
    c3 = ci["median_frac_cap"][0] >= 0.001 or ci["negative_share"][0] >= 0.05
    c4 = ci["summer_winter_ratio"][1] <= 2.0
    failing = [
        n
        for n, ok in zip(
            ["1 zeros/repeats", "2 lattice", "3 floor", "4 season"], [c1, c2, c3, c4], strict=True
        )
        if not ok
    ]
    if all([c1, c2, c3, c4]):
        return "the same noise", ""
    if st["zero_share"] >= 0.90 or (
        st["zero_or_repeat_share"] < 0.20 and st["lattice_share"] < 0.80
    ):
        return "not the same", ""
    return "mixed", ", ".join(failing)


def run_lengths(days: pl.DataFrame) -> pl.DataFrame:
    """Group idle days into runs of consecutive idle days; days without enough sun are skipped."""
    d = days.filter(pl.col("qualifying") >= MIN_QUALIFYING_PER_DAY).sort("date")
    d = d.with_columns(
        idle_day=pl.col("idle") == pl.col("qualifying"),
    )
    d = d.with_columns(run_break=(pl.col("idle_day") != pl.col("idle_day").shift()).cum_sum())
    runs = (
        d.filter(pl.col("idle_day"))
        .group_by("run_break")
        .agg(
            start=pl.col("date").min(),
            end=pl.col("date").max(),
            n_days=pl.len(),
            half_hours=pl.col("idle").sum(),
        )
        .sort("start")
    )
    return runs.with_columns(calendar_days=(pl.col("end") - pl.col("start")).dt.total_days() + 1)


def half_hours_in_runs_over(runs: pl.DataFrame, hours: int) -> int:
    """Count the idle half-hours that lie in runs spanning more than `hours` hours."""
    return int(runs.filter(pl.col("calendar_days") * 24 > hours)["half_hours"].sum() or 0)


def main() -> None:  # noqa: D103, PLR0915
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sites = pv_sites().sort("site")
    meta = pl.read_parquet(METADATA_PATH).select("time_series_id", "units")
    power = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .collect()
    )
    cams = pl.read_parquet(CAMS_PATH)
    summary, by_year, neg_hour, day_runs, season, daily_tables, csi_bins, wd_hour = (
        [],
        [],
        [],
        [],
        [],
        {},
        [],
        [],
    )
    verdicts = {}
    for row in sites.iter_rows(named=True):
        site, tsid, cap = row["site"], row["time_series_id"], float(row["effective_capacity_mw"])
        label = f"Solar {site}"
        unit = meta.filter(pl.col("time_series_id") == tsid)["units"][0]
        df = prepare(
            tsid=tsid, lat=row["latitude"], lon=row["longitude"], site=site, power=power, cams=cams
        )
        df = df.with_columns(
            prev_power=pl.col("power").shift(),
            prev_gap=(pl.col("time") - pl.col("time").shift()).dt.total_minutes(),
            year=pl.col("time").dt.year(),
            month=pl.col("time").dt.month(),
            hour=pl.col("time").dt.offset_by("-15m").dt.hour(),
            weekday=pl.col("time").dt.offset_by("-15m").dt.weekday(),
            date=pl.col("time").dt.offset_by("-15m").dt.date(),
        )
        night = df.filter(
            (pl.col("elevation") < NIGHT_ELEVATION_DEG) & (pl.col("prev_gap") == 30)
        ).drop_nulls("prev_power")
        st: dict[str, object] = dict(night_stats(night, cap))
        ci = bootstrap(night, cap, rng)
        ratio = summer_winter_ratio(night, cap)
        label_verdict, failing = classify(night_stats(night, cap), ci, ratio)
        # lag-1 autocorrelation and rolling sd of first differences, within consecutive night runs.
        d = night.with_columns(diff=pl.col("power") - pl.col("prev_power"))
        d = d.with_columns(
            diff_prev=pl.col("diff").shift(),
            ok=(pl.col("time") - pl.col("time").shift()).dt.total_minutes() == 30,
        ).filter(pl.col("ok"))
        a, b = d["diff"].to_numpy(), d["diff_prev"].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        ac1 = (
            float(np.corrcoef(a[m], b[m])[0, 1])
            if m.sum() > 2 and np.std(a[m]) > 0 and np.std(b[m]) > 0
            else float("nan")
        )
        rsd = d.with_columns(rsd=pl.col("diff").rolling_std(6))["rsd"].drop_nulls()
        st.update(
            {
                "lag1_autocorr_diff": ac1,
                "rolling_sd_diff_median_frac_cap": float(np.median(rsd.to_numpy()) / cap)
                if rsd.len()
                else float("nan"),
                "summer_winter_ratio": ratio,
            }
        )
        for k, (lo, hi) in ci.items():
            st[f"{k}_lo"], st[f"{k}_hi"] = lo, hi
        st.update({"site": label, "units": str(unit), "label": label_verdict, "failing": failing})
        summary.append(st)
        verdicts[label] = label_verdict
        by_year.append(
            night.group_by("year")
            .agg(
                n=pl.len(),
                zero_share=(pl.col("power") == 0).mean(),
                median_frac_cap=pl.col("power").median() / cap,
                p95_frac_cap=pl.col("power").quantile(0.95) / cap,
                negative_share=(pl.col("power") < 0).mean(),
            )
            .sort("year")
            .with_columns(site=pl.lit(label))
        )
        neg_hour.append(
            night.group_by("hour")
            .agg(n=pl.len(), negative_share=(pl.col("power") < 0).mean())
            .sort("hour")
            .with_columns(site=pl.lit(label))
        )
        season.append(
            night.filter(pl.col("month").is_in(SUMMER + WINTER))
            .with_columns(
                season=pl.when(pl.col("month").is_in(SUMMER))
                .then(pl.lit("summer"))
                .otherwise(pl.lit("winter"))
            )
            .group_by("season")
            .agg(
                n=pl.len(),
                median_frac_cap=pl.col("power").median() / cap,
                zero_share=(pl.col("power") == 0).mean(),
            )
            .with_columns(site=pl.lit(label))
        )
        # Daytime idle: clear-sky index and the 10:00-15:00 window.
        floor = float(np.quantile(night["power"].to_numpy(), 0.95)) + IDLE_MARGIN_OF_CAPACITY * cap
        q = df.filter(
            (pl.col("csi") >= CSI_MIN)
            & (pl.col("hour") >= DAY_HOURS[0])
            & (pl.col("hour") < DAY_HOURS[1])
            & (pl.col("elevation") > 0)
        ).with_columns(idle=pl.col("power") < floor)
        days = q.group_by("date").agg(qualifying=pl.len(), idle=pl.col("idle").sum())
        runs = run_lengths(days)
        total_idle = int(q["idle"].sum())
        day_runs.append(
            {
                "site": label,
                "qualifying_half_hours": q.height,
                "idle_half_hours": total_idle,
                "idle_share": total_idle / max(q.height, 1),
                "share_idle_in_runs_over_24h": half_hours_in_runs_over(runs, 24)
                / max(total_idle, 1),
                "share_idle_in_runs_over_7d": half_hours_in_runs_over(runs, 24 * 7)
                / max(total_idle, 1),
                "n_runs": runs.height,
                "longest_runs_days": sorted(runs["calendar_days"].to_list(), reverse=True)[:5],
                "median_run_days": float(np.median(runs["calendar_days"].to_numpy()))
                if runs.height
                else None,
                "outage_class_present": (half_hours_in_runs_over(runs, 24) / max(total_idle, 1))
                >= 0.5
                and total_idle > 0,
            }
        )
        daily_tables[label] = (
            days.with_columns(idle_day=pl.col("idle") == pl.col("qualifying"))
            .filter(pl.col("qualifying") >= MIN_QUALIFYING_PER_DAY)
            .select("date", "idle_day")
        )
        runs.with_columns(site=pl.lit(label)).write_parquet(DATA_DIR / f"runs_{site}.parquet")
        wd_hour.append(
            q.group_by("weekday", "hour")
            .agg(n=pl.len(), idle_share=pl.col("idle").mean())
            .sort("weekday", "hour")
            .with_columns(site=pl.lit(label))
        )
        # Idle share by clear-sky index bin over all daylight half-hours (CSI from 0.1).
        dl = df.filter((pl.col("elevation") > 10) & pl.col("csi").is_not_null()).with_columns(
            csi_bin=(pl.col("csi").clip(0, 1.2) / 0.1).floor() * 0.1, idle=pl.col("power") < floor
        )
        csi_bins.append(
            dl.group_by("csi_bin")
            .agg(n=pl.len(), idle_share=pl.col("idle").mean())
            .sort("csi_bin")
            .with_columns(site=pl.lit(label))
        )
        # Gallery (private): 10 sure-night days, 10 daytime days disagreeing with irradiance.
        _gallery(df=df, site=site, cap=cap, night=night, q=q, rng=rng)
        # Share of scored half-hours the noise would touch.
        st["share_of_all_half_hours_night"] = night.height / df.height
        st["share_of_daylight_half_hours_idle"] = day_runs[-1]["idle_share"]
    # Agreement between farms: Jaccard of idle days.
    agree = []
    labs = list(daily_tables)
    for i, x in enumerate(labs):
        for y in labs[i + 1 :]:
            j = daily_tables[x].join(daily_tables[y], on="date", suffix="_y")
            both = int((j["idle_day"] & j["idle_day_y"]).sum())
            either = int((j["idle_day"] | j["idle_day_y"]).sum())
            only_x, only_y = (
                int((j["idle_day"] & ~j["idle_day_y"]).sum()),
                int((~j["idle_day"] & j["idle_day_y"]).sum()),
            )
            agree.append(
                {
                    "farm_x": x,
                    "farm_y": y,
                    "days_both": j.height,
                    "idle_both": both,
                    "idle_either": either,
                    "idle_only_x": only_x,
                    "idle_only_y": only_y,
                    "jaccard": both / either if either else None,
                }
            )
    out = {
        "summary": pl.DataFrame(summary),
        "by_year": pl.concat(by_year),
        "negative_by_hour": pl.concat(neg_hour),
        "season": pl.concat(season),
        "daytime_runs": pl.DataFrame(day_runs),
        "daytime_weekday_hour": pl.concat(wd_hour),
        "idle_share_by_csi": pl.concat(csi_bins),
        "pair_agreement": pl.DataFrame(agree),
    }
    for name, frame in out.items():
        frame.write_csv(DATA_DIR / f"{name}.csv")
        frame.write_csv(DOCS_DIR / f"{name}.csv")
    (DATA_DIR / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    print(json.dumps(verdicts, indent=2))


def _gallery(
    *,
    df: pl.DataFrame,
    site: str,
    cap: float,
    night: pl.DataFrame,
    q: pl.DataFrame,
    rng: np.random.Generator,
) -> None:
    """Write the private PNG gallery: 10 sure-night and 10 daytime-disagreement farm-days."""
    gdir = DATA_DIR / "gallery" / f"solar_{site}"
    gdir.mkdir(parents=True, exist_ok=True)
    night_days = night.group_by("date").agg(n=pl.len()).filter(pl.col("n") >= 20)["date"].to_list()
    idle_days = (
        q.group_by("date")
        .agg(n=pl.len(), idle=pl.col("idle").sum())
        .filter((pl.col("n") >= MIN_QUALIFYING_PER_DAY) & (pl.col("idle") >= 1))["date"]
        .to_list()
    )
    for kind, days in (("night", night_days), ("day", idle_days)):
        if not days:
            continue
        picks = rng.choice(len(days), size=min(GALLERY_DAYS, len(days)), replace=False)
        fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharey=False)
        for ax, i in zip(axes.ravel(), picks, strict=False):
            d = df.filter(pl.col("date") == days[int(i)])
            ax.step(d["time"].to_numpy(), d["power"].to_numpy() / cap, where="post", lw=1)
            ax2 = ax.twinx()
            ax2.plot(d["time"].to_numpy(), d["csi"].to_numpy(), color="orange", lw=0.8)
            ax2.set_ylim(0, 1.3)
            ax.set_title(f"{days[int(i)]}", fontsize=8)
        fig.suptitle(
            f"Solar {site}, {kind} days: power/capacity (blue), clear-sky index (orange). Private."
        )
        fig.savefig(gdir / f"{kind}.png", dpi=70)
        plt.close(fig)


if __name__ == "__main__":
    main()

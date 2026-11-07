"""Experiment 641 step 1: diagnose K1's error at D15 and D12 from the stored export (CPU only).

Reads the K1, Candidate and K3 files of the 2026-11-07 backtest export (experiments 612 and 613).
Writes tables to the experiment's private data folder and prints the headline numbers.
"""

import glob
import math
import sys
from pathlib import Path
from typing import Final

import polars as pl

EXPORT: Final = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/backtest_export_roster_2026-11-07"
)
OUT: Final = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/"
    "00641-d15-d12-error-diagnosis-and-cheapest-fixes/tables"
)
NGED: Final = Path("/home/jack/dev/nged-substation-forecast/data/NGED")
TARGETS: Final = ["D15", "D12"]
SHORT_HISTORY_CUT: Final = pl.datetime(2025, 3, 1, time_zone="UTC")
LEVELS: Final = [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99]
QCOLS: Final = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]


def load(network: str) -> pl.DataFrame:
    """Load one network's five months, joined to capacity, with errors in percent of capacity."""
    sub = pl.read_csv(EXPORT / "substations.csv").select("time_series_id", "primary_name", "capacity")
    frames = [
        pl.read_parquet(f).with_columns(pl.lit(Path(f).stem.split("_")[1]).alias("month"))
        for f in sorted(glob.glob(str(EXPORT / f"{network}_*.parquet")))
    ]
    df = pl.concat(frames).join(sub, on="time_series_id")
    df = df.filter(
        pl.col("observed_power").is_not_null(),
        pl.col("lead_hours") >= 24,
        pl.col("lead_hours") < 240,
        ~((pl.col("primary_name").is_in(["D15", "D18"])) & (pl.col("valid_time") < SHORT_HISTORY_CUT)),
    )
    cap = pl.col("capacity")
    pinball = pl.sum_horizontal(
        [
            pl.max_horizontal(
                pl.lit(tau) * (pl.col("observed_power") - pl.col(c)),
                (pl.lit(tau) - 1) * (pl.col("observed_power") - pl.col(c)),
            )
            for tau, c in zip(LEVELS, QCOLS, strict=True)
        ]
    )
    return df.with_columns(
        (pinball * 2 / len(LEVELS) / cap * 100).alias("crps"),
        ((pl.col("observed_power") - pl.col("q50")).abs() / cap * 100).alias("ae"),
        ((pl.col("observed_power") - pl.col("q50")) / cap * 100).alias("err"),
        pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.hour().alias("hour"),
        pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date().alias("day"),
        (pl.col("lead_hours") // 72).cast(pl.Int32).alias("lead_band"),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    k1 = load("K1")
    out = {}
    # 0. per-Primary summary and rows dropped
    summ = k1.group_by("primary_name").agg(
        pl.len().alias("rows"),
        pl.col("crps").mean(),
        pl.col("ae").mean().alias("mae"),
        pl.col("err").mean().alias("bias"),
        (pl.col("observed_power") > pl.col("q95")).mean().alias("above_q95"),
        (pl.col("observed_power") < pl.col("q05")).mean().alias("below_q05"),
        (pl.col("observed_power") > pl.col("q99")).mean().alias("above_q99"),
        (pl.col("observed_power") < pl.col("q01")).mean().alias("below_q01"),
    ).sort("crps", descending=True)
    summ.write_csv(OUT / "summary_by_primary.csv")
    print(summ)
    for name in TARGETS:
        d = k1.filter(pl.col("primary_name") == name)
        print("=====", name)
        for key in ["hour", "month", "lead_band"]:
            t = d.group_by(key).agg(
                pl.col("crps").mean(), pl.col("ae").mean().alias("mae"), pl.col("err").mean().alias("bias")
            ).sort(key)
            t.write_csv(OUT / f"{name}_by_{key}.csv")
            print(t.to_pandas().round(2).T.to_string())
        # concentration of error on days
        daily = d.group_by("day").agg(
            pl.col("crps").mean(), pl.col("err").mean().alias("bias"),
            pl.col("observed_power").median().alias("obs_med"),
            pl.col("q50").median().alias("fc_med"),
            pl.col("event_free").all().alias("event_free"),
            pl.col("hour").n_unique().alias("hours"),
        ).sort("day")
        daily.write_csv(OUT / f"{name}_daily.csv")
        tot = daily["crps"].sum()
        s = daily.sort("crps", descending=True)["crps"].cum_sum()
        n = daily.height
        for frac in (0.05, 0.1, 0.2):
            k = max(1, math.ceil(frac * n))
            print(f"{name}: worst {frac:.0%} of days ({k} of {n}) carry {s[k-1] / tot:.1%} of CRPS")
        # autocorrelation of daily mean error
        e = daily["bias"].to_numpy()
        ac = [float(pl.DataFrame({"a": e[:-k], "b": e[k:]}).select(pl.corr("a", "b")).item()) for k in (1, 2, 3, 7, 14)]
        print(name, "daily-bias autocorrelation at lags 1,2,3,7,14:", [round(x, 2) for x in ac])
        # level shifts: daily median observed vs rolling 28-day median of earlier days
        lv = daily.with_columns(
            pl.col("obs_med").rolling_median(28, min_samples=14).shift(1).alias("prev28")
        ).with_columns((pl.col("obs_med") - pl.col("prev28")).alias("shift"))
        mad = lv["shift"].drop_nulls().abs().median() * 1.4826
        lv = lv.with_columns((pl.col("shift").abs() > 3 * mad).alias("flag"))
        lv.write_csv(OUT / f"{name}_level_shift_flags.csv")
        flagged = lv.filter(pl.col("flag"))
        print(f"{name}: robust sd of daily shift {mad:.2f} MW; flagged days {flagged.height} of {lv.height};",
              f"CRPS share on flagged days {flagged['crps'].sum() / tot:.1%};",
              f"event_free among flagged: {flagged['event_free'].mean():.2f}")
        # midday vs night
        mid = d.filter(pl.col("hour").is_between(10, 14))["ae"].mean()
        night = d.filter(pl.col("hour").is_in([0, 1, 2, 3, 4]))["ae"].mean()
        print(f"{name}: MAE midday 10-14h {mid:.2f}, night 0-4h {night:.2f}, ratio {mid / night:.2f}")
        bym = d.filter(pl.col("hour").is_between(10, 14)).group_by("month").agg(pl.col("err").mean())
        print(bym.sort("month"))
        # step sizes: first difference of daily median observed
        steps = daily.with_columns(pl.col("obs_med").diff().abs().alias("step")).filter(pl.col("step") > 0)
        print(name, "daily-median step quantiles (MW) 50/90/99:", steps["step"].quantile(0.5), steps["step"].quantile(0.9), steps["step"].quantile(0.99))
        # comparison networks
    for net in ("Candidate", "K3"):
        o = load(net).filter(pl.col("primary_name").is_in(TARGETS)).group_by("primary_name").agg(pl.col("crps").mean(), pl.col("err").mean().alias("bias"))
        print(net, o)
    # metered generators near each target
    m = pl.read_parquet(NGED / "metadata.parquet").select(
        "time_series_id", "time_series_name", "time_series_type", "substation_type", "latitude", "longitude", "units"
    )
    for name, tsid in (("D15", 12), ("D12", 14)):
        row = m.filter(pl.col("time_series_id") == tsid).row(0, named=True)
        g = m.filter(pl.col("time_series_type") != "Disaggregated Demand").with_columns(
            (((pl.col("latitude") - row["latitude"]) * 111.0) ** 2
             + ((pl.col("longitude") - row["longitude"]) * 67.0) ** 2).sqrt().alias("km")
        ).sort("km").head(5)
        print(name, "nearest metered non-demand series (km; names withheld, type only):")
        print(g.select("time_series_type", "units", "km"))


if __name__ == "__main__":
    sys.exit(main())

# ruff: noqa: D103, E501
"""Experiment 677: days for experiment 667's prevailing output to recover half of a confirmed
event's new level (private CSV).

Usage: python e00677_recovery.py

Only experiment 667's prevailing output is meant to follow a shift, so only it is timed; the ladder steps
have no prevailing output, and the NRA output is not meant to follow a shift. For each of the 12 events
(10 logged shifts lasting over 7 days with 2 units or more, and the verified D5 and D16 transfer),
the new level
is the mean observed power over days 3 to the event end (or day 14, whichever is earlier) minus the
mean over the 7 days
before the start. An event is timed only if that change is at least 10% of the series' capacity. The
prevailing
forecast's change is its daily mean (forecasts made 0 to 36 hours ahead) minus its mean over the
same 7 days before. The time to recover half is the
first whole day since the event start on which the forecast's change reaches half of the observed
change. Writes
`scores/recovery_prevailing.csv` (one row per event) and prints the median with a 95% interval from
resampling events.
"""

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00668_slices_primaries as s668  # ty: ignore[unresolved-import]
import e00677_event_story as story
import e00677_score as sc
import numpy as np
import polars as pl

DAY = s668.DAY


def main() -> None:
    site = pl.read_csv(next(s668.R.glob("00521-*/site_map.csv")))
    ev, _ = story.events()
    truth = pl.read_parquet(lib.FOLDER / "ground_truth_primaries.parquet").select(
        "time_series_id", "valid_time", "power"
    )
    f = (
        pl.scan_parquet(lib.FOLDER / "primaries.parquet")
        .filter(pl.col("NRA_or_prevailing") == "prevailing")
        .with_columns(lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours())
        .filter(pl.col("lead") < 36)
        .group_by("time_series_id", "valid_time")
        .agg(pl.col("q50").mean().alias("fc"))
        .collect()
    )
    d = truth.join(f, on=["time_series_id", "valid_time"], how="left").join(
        site.select("time_series_id", "label", "capacity_mw"), on="time_series_id"
    )
    # grid index of each valid time, in half-hours since the grid start
    start_us = int(lib.GRID_START.timestamp() * 1_000_000)
    d = d.with_columns(
        idx=((pl.col("valid_time").dt.epoch("us") - start_us) // (30 * 60_000_000)).cast(pl.Int64)
    )
    rows = []
    for e in ev.iter_rows(named=True):
        s = d.filter(pl.col("label") == e["label"])
        cap = s["capacity_mw"][0]
        length = (e["end"] - e["start"]) // DAY
        pre = s.filter((pl.col("idx") >= e["start"] - 7 * DAY) & (pl.col("idx") < e["start"]))
        lvl = s.filter(
            (pl.col("idx") >= e["start"] + 3 * DAY)
            & (pl.col("idx") < e["start"] + min(length, 14) * DAY)
        )
        pre_mean, lvl_mean = pre["power"].mean(), lvl["power"].mean()
        if pre_mean is None or lvl_mean is None:
            rows.append(
                {
                    "event": e["event"],
                    "label": e["label"],
                    "length_days": length,
                    "observed_change_mw": None,
                    "capacity_mw": float(cap),
                    "timed": False,
                    "days_to_half": None,
                }
            )
            continue
        delta = lvl_mean - pre_mean  # ty: ignore[unsupported-operator]
        fc_pre = pre["fc"].drop_nulls().mean()
        rec = None
        if abs(delta) >= 0.1 * cap and fc_pre is not None and length >= 4:
            for day in range(int(min(length, 21))):
                w = s.filter(
                    (pl.col("idx") >= e["start"] + day * DAY)
                    & (pl.col("idx") < e["start"] + (day + 1) * DAY)
                )["fc"].drop_nulls()
                if len(w) and (w.mean() - fc_pre) / delta >= 0.5:  # ty: ignore[unsupported-operator]
                    rec = day
                    break
        rows.append(
            {
                "event": e["event"],
                "label": e["label"],
                "length_days": length,
                "observed_change_mw": float(delta),
                "capacity_mw": float(cap),
                "timed": abs(delta) >= 0.1 * cap and length >= 4,
                "days_to_half": rec,
            }
        )
    out = pl.DataFrame(rows, schema_overrides={"days_to_half": pl.Int64})
    out.write_csv(sc.ROOT / "scores" / "recovery_prevailing.csv")
    print(out)
    timed = out.filter(pl.col("timed"))
    got = timed["days_to_half"].drop_nulls().to_numpy()
    print(f"timed events {timed.height}, reached half within the event {len(got)}")
    if len(got):
        rng = np.random.default_rng(1)
        draws = [np.median(rng.choice(got, size=len(got))) for _ in range(2000)]
        print(
            "median days",
            float(np.median(got)),
            "95% interval",
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        )


if __name__ == "__main__":
    main()

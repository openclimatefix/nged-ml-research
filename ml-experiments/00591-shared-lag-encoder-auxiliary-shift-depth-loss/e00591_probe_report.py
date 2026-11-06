# ruff: noqa: B023, D103, E501, PERF401, PT018
"""Experiment 591: how well the auxiliary heads recover the injected depth, and the detector view.

Reads the probe files that `e00591_arm.py ARM probe` wrote (rows 24 to 120 h ahead, one file per month,
pass, refit day, and seed; the two seeds are averaged). Pass 0 to 23 is experiment 581's injected bank
(every event is held out: the folds' test months are never trained on); pass 99 uses the observed lags
and labels them with the verified real events of experiment 580.

Outputs (printed, and written to `probe_report_<arm>.txt`):
- R squared and bias of the heads by metering group, injected depth, and event length (origins from 7
  days before the onset to 14 days after the end, both lags pooled, all lag-rows of the window);
- the detector view: an origin day "fires" when the mean over its rows of the larger of the two heads'
  absolute outputs exceeds a threshold. Detection rate, median onset delay, and false triggers per
  Primary-year on origin days with no labelled or verified event (the `free` flag of experiment 583's
  row table) with observed lags;
- the verified December pair: mean head output and mean verified depth inside each event.

Usage: python e00591_probe_report.py ARM [ARM ...]   (a1 .. a5)
"""

import io
import sys
from contextlib import redirect_stdout

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = ML_EXPERIMENTS_DATA_DIR / "00591-shared-lag-encoder-auxiliary-shift-depth-loss"
SM = pl.read_csv(
    ML_EXPERIMENTS_DATA_DIR
    / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
).select("primary", "group", "label")
THRESHOLDS = (0.05, 0.075, 0.10)
REAL = 99
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)


def month_start_idx(month: str) -> int:
    start = k578.c.month_start(month=month)
    return int((k578.c.us(t=start) - k578.prep.GRID_START_US) // k578.prep.STEP_US)


def load_month(*, arm: str, month: str) -> pl.DataFrame:
    """Return the seed-averaged probe rows of the month, with the pass number."""
    parts = []
    for p in sorted((ROOT / arm.upper() / "work" / "probe").glob(f"{month}_r*_d*_s*.parquet")):
        r = int(p.name.split("_r")[1][:2])
        parts.append(pl.read_parquet(p).with_columns(r=pl.lit(r)))
    keys = ["r", "primary", "target_idx", "origin_idx"]
    return (
        pl.concat(parts)
        .group_by(keys)
        .agg(pl.col("aux_day", "aux_week", "true_day", "true_week").mean())
    )


def free_origins() -> dict[str, set[tuple[int, int]]]:
    """Return per month the (Primary, origin index) pairs whose scored rows are all event-free (experiment 583's flag)."""
    z = np.load(ROOT / "score_a0" / "row_arrays.npz", allow_pickle=True)
    df = pl.DataFrame(
        {"prim": z["prim"], "origin_idx": z["origin_idx"], "free": z["free"], "month": z["month"]}
    )
    ok = df.group_by("month", "prim", "origin_idx").agg(pl.col("free").all()).filter(pl.col("free"))
    return {
        m: set(zip(g["prim"].to_list(), g["origin_idx"].to_list(), strict=True))
        for (m,), g in ok.group_by("month")
    }


def main(arm: str) -> None:
    months = k578.MONTHS
    free = free_origins()
    r2_rows, det_rows, fa_rows, real_rows = [], [], [], []
    for month in months:
        d = load_month(arm=arm, month=month)
        if not d.height:
            continue
        end = month_start_idx(month) + bank.END_DAY * bank.PER_DAY
        d = d.join(SM, on="primary").with_columns(
            score=pl.max_horizontal(pl.col("aux_day").abs(), pl.col("aux_week").abs())
        )
        inj = d.filter(pl.col("r") < bank.N_PASSES)
        ev = pl.DataFrame(
            [
                {
                    "r": r,
                    "primary": p,
                    "depth": abs(bank.event(primary=p, r=r)[0]),
                    "length_days": bank.event(primary=p, r=r)[1],
                }
                for r in range(bank.N_PASSES)
                for p in range(16)
            ]
        )
        inj = inj.join(ev, on=["r", "primary"]).with_columns(
            onset=pl.lit(end) - pl.col("length_days") * bank.PER_DAY
        )
        inj = inj.with_columns(
            d_on=(pl.col("origin_idx") - pl.col("onset")) / bank.PER_DAY,
            d_end=(pl.col("origin_idx") - end) / bank.PER_DAY,
        ).filter((pl.col("d_on") >= -7) & (pl.col("d_end") < 14))
        long = pl.concat(
            [
                inj.select("group", "depth", "length_days", pred=f"aux_{n}", true=f"true_{n}")
                for n in ("day", "week")
            ]
        ).filter(pl.col("true").is_not_null() & pl.col("true").is_finite())
        r2_rows.append(
            long.group_by("group", "depth", "length_days").agg(
                n=pl.len(),
                sse=((pl.col("pred") - pl.col("true")) ** 2).sum(),
                sst=((pl.col("true") - pl.col("true").mean()) ** 2).sum(),
                in_pred=pl.col("pred").filter(pl.col("true").abs() > 0).mean(),
                in_true=pl.col("true").filter(pl.col("true").abs() > 0).mean(),
                out_pred=pl.col("pred").filter(pl.col("true") == 0).mean(),
            )
        )
        day = inj.group_by(
            "r", "primary", "group", "depth", "length_days", "origin_idx", "d_on", "d_end"
        ).agg(score=pl.col("score").mean())
        for thr in THRESHOLDS:
            fired = day.filter(pl.col("score") > thr)
            first = (
                fired.filter((pl.col("d_on") >= 0) & (pl.col("d_end") < 0))
                .group_by("r", "primary")
                .agg(delay=pl.col("d_on").min())
            )
            det_rows.append(
                ev.join(SM, on="primary")
                .join(first, on=["r", "primary"], how="left")
                .with_columns(thr=pl.lit(thr), month=pl.lit(month))
            )
        obs = d.filter(pl.col("r") == REAL)
        ok = free[month]
        od = obs.group_by("primary", "origin_idx", "group").agg(score=pl.col("score").mean())
        od = od.filter(
            pl.struct("primary", "origin_idx").map_elements(
                lambda s: (s["primary"], s["origin_idx"]) in ok, return_dtype=pl.Boolean
            )
        )
        for thr in THRESHOLDS:
            fa_rows.append(
                od.group_by("group")
                .agg(n_days=pl.len(), n_fire=(pl.col("score") > thr).sum())
                .with_columns(thr=pl.lit(thr))
            )
        real_rows.append(
            obs.filter(pl.col("true_day").abs() > 0)
            .group_by("primary")
            .agg(
                n=pl.len(),
                head_day=pl.col("aux_day").mean(),
                true_day=pl.col("true_day").mean(),
                head_week=pl.col("aux_week").mean(),
                true_week=pl.col("true_week").mean(),
            )
            .with_columns(month=pl.lit(month))
        )
    r2 = (
        pl.concat(r2_rows)
        .group_by("group", "depth", "length_days")
        .agg(
            pl.col("n").sum(),
            pl.col("sse").sum(),
            pl.col("sst").sum(),
            pl.col("in_pred").mean(),
            pl.col("in_true").mean(),
            pl.col("out_pred").mean(),
        )
        .with_columns(r2=1 - pl.col("sse") / pl.col("sst"))
        .sort("group", "depth", "length_days")
    )
    print(
        f"\n== {arm.upper()}: R squared of the heads on held-out injected events (both lags pooled)"
    )
    print(
        r2.select(
            "group",
            "depth",
            "length_days",
            "n",
            pl.col("r2").round(3),
            pl.col("in_pred").round(3),
            pl.col("in_true").round(3),
            pl.col("out_pred").round(3),
        )
    )
    allg = (
        pl.concat(r2_rows)
        .group_by("depth")
        .agg(pl.col("sse").sum(), pl.col("sst").sum())
        .with_columns(r2=1 - pl.col("sse") / pl.col("sst"))
        .sort("depth")
    )
    print(
        "R squared by depth, both groups:", allg.select("depth", pl.col("r2").round(3)).to_dicts()
    )
    det = (
        pl.concat(det_rows)
        .group_by("thr", "group", "depth", "length_days")
        .agg(
            n_events=pl.len(),
            detected=pl.col("delay").is_not_null().mean(),
            median_delay_days=pl.col("delay").median(),
        )
        .sort("thr", "group", "depth", "length_days")
    )
    print(
        f"\n== {arm.upper()}: detector view on the injected bank (an origin day fires when the head score exceeds the threshold)"
    )
    print(det.with_columns(pl.col(pl.Float64).round(3)))
    fa = (
        pl.concat(fa_rows)
        .group_by("thr", "group")
        .agg(pl.col("n_days").sum(), pl.col("n_fire").sum())
        .with_columns(per_primary_year=pl.col("n_fire") / (pl.col("n_days") / 365.0))
        .sort("thr", "group")
    )
    print(
        f"\n== {arm.upper()}: false triggers on event-free origin days with observed lags (note: Primary-days summed over the five folds)"
    )
    print(fa.with_columns(pl.col(pl.Float64).round(3)))
    print(
        f"\n== {arm.upper()}: verified December pair, rows with a verified shift in the lag (descriptive; labels imperfect)"
    )
    rr = pl.concat(real_rows).join(SM, on="primary")
    print(rr.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    for arm in sys.argv[1:]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(arm)
        (ROOT / f"probe_report_{arm}.txt").write_text(buf.getvalue())
        print(buf.getvalue())

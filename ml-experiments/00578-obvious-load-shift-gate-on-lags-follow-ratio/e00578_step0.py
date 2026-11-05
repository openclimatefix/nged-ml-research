# ruff: noqa: E501, D103
"""Experiment 578, Step 0: define the long, deep, obvious subset (E-obv) from the labels.

CPU only. Prints the label counts, the quartiles, the median depth of events lasting 7 days or more
(printed before any score), and the E-obv table. Writes `work/step0_events.csv` and
`work/step0_gap_sd.csv`.
"""

import e00578_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MIN_DAYS = 7.0
OBVIOUS_K = 3.0
OBVIOUS_RUN = 3
MIN_ROWS = 40  # complete-ish day: at least 40 of 48 finite half-hours


def daily_mean(*, x: np.ndarray) -> np.ndarray:
    """Return the per-day mean of (16, grid) with NaN where fewer than MIN_ROWS half-hours are finite."""
    n = x.shape[1] // k.PER_DAY
    r = x[:, : n * k.PER_DAY].reshape(16, n, k.PER_DAY)
    cnt = np.isfinite(r).sum(-1)
    with np.errstate(all="ignore"):
        m = np.nanmean(r, axis=-1)
    return np.where(cnt >= MIN_ROWS, m, np.nan)


def main() -> None:
    k.WORK.mkdir(parents=True, exist_ok=True)
    sm = k.sites()
    lab2p = dict(zip(sm["label"].to_list(), sm["primary"].to_list(), strict=True))
    cl = k.clean()
    exp = k.expected_lag_free(shape=cl.shape)
    gap_d = daily_mean(x=cl - exp)
    ev = pl.read_parquet(k.EVENTS).with_columns(
        primary=pl.col("label").replace_strict(lab2p, return_dtype=pl.Int64),
        onset_day=pl.col("onset_idx") // k.PER_DAY,
        end_day=pl.col("end_idx") // k.PER_DAY,
    )
    print(
        "events",
        ev.height,
        "median depth share",
        ev["depth_share"].abs().median(),
        "median days",
        ev["length_days"].median(),
    )
    q = [0, 25, 50, 75, 100]
    print("depth quartiles", np.percentile(ev["depth_share"].abs().to_numpy(), q).round(1).tolist())
    print("duration quartiles", np.percentile(ev["length_days"].to_numpy(), q).round(2).tolist())
    long = ev.filter(pl.col("length_days") >= MIN_DAYS)
    med = float(long["depth_share"].abs().median())  # ty: ignore[invalid-argument-type]
    print("events >= 7 d:", long.height, "median depth share of those:", round(med, 2))
    # event-free days per Primary: not within an event (+-1 day)
    n_days = gap_d.shape[1]
    free = np.ones((16, n_days), bool)
    for r in ev.iter_rows(named=True):
        free[r["primary"], max(r["onset_day"] - 1, 0) : r["end_day"] + 2] = False
    sd = np.array(
        [
            np.nanstd(gap_d[p][free[p]]) if np.isfinite(gap_d[p][free[p]]).sum() > 20 else np.nan
            for p in range(16)
        ]
    )
    pl.DataFrame(
        {
            "primary": range(16),
            "label": sm["label"],
            "event_free_gap_sd": sd,
            "n_event_free_days": [int(np.isfinite(gap_d[p][free[p]]).sum()) for p in range(16)],
        }
    ).write_csv(k.WORK / "step0_gap_sd.csv")
    rows = []
    for r in ev.iter_rows(named=True):
        p = r["primary"]
        days = np.arange(r["onset_day"] + 1, r["end_day"])  # whole UTC days inside the event
        g = gap_d[p, days] * np.sign(r["depth_mw"])
        have = int(np.isfinite(g).sum())
        big = np.nan_to_num(g, nan=-1.0) > OBVIOUS_K * sd[p]
        run = best = 0
        for b in big:
            run = run + 1 if b else 0
            best = max(best, run)
        rows.append(
            {
                **r,
                "gap_days_available": have,
                "longest_visible_run": best,
                "obvious": bool(best >= OBVIOUS_RUN),
                "long": r["length_days"] >= MIN_DAYS,
                "deep": abs(r["depth_share"]) >= med,
                "event_sd": sd[p],
            }
        )
    t = pl.DataFrame(rows).with_columns(long_deep=pl.col("long") & pl.col("deep"))
    t.write_csv(k.WORK / "step0_events.csv")
    sub = t.filter(pl.col("long_deep"))
    print(
        "long and deep:",
        sub.height,
        "with available gap days >= 3:",
        sub.filter(pl.col("gap_days_available") >= 3).height,
    )
    print("E-obv:", sub.filter(pl.col("obvious")).height)
    with pl.Config(tbl_rows=60, tbl_cols=20, fmt_str_lengths=30):
        print(
            sub.select(
                "event_id",
                "label",
                "length_days",
                "depth_share",
                "gap_days_available",
                "longest_visible_run",
                "obvious",
            )
        )
    print(
        "events with any gap day available:",
        t.filter(pl.col("gap_days_available") >= 3).height,
        "of",
        t.height,
    )


main()

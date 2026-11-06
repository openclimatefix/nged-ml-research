# ruff: noqa: E501, D103
"""Experiment 592: print every number the write-up quotes. Run `e00592_score.py` first."""

import e00592_common as g  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.stats import chi2

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(30)


def read(name: str) -> pl.DataFrame:
    return pl.read_csv(
        g.WORK / name, schema_overrides={"on_trace": pl.String}, infer_schema_length=None
    )


def poisson_upper(k: int, years: float) -> float:
    """Return the one-sided 95% upper bound on a rate from `k` events in `years`."""
    return float(chi2.ppf(0.95, 2 * (k + 1)) / 2 / years)


def main() -> None:
    b = read("bank_summary.csv").filter(
        (pl.col("group") == "both groups") & (pl.col("band") == "24-120 h")
    )
    cols = [
        "phase",
        "arm",
        "n_events",
        "cover90",
        "cover90_lo",
        "cover90_hi",
        "nra_side95",
        "width90",
    ]
    for cls in ("15 and 30 days", "all lengths", "7 days"):
        print(f"== coverage by phase, both groups, injected {cls}, 24-120 h")
        print(b.filter(pl.col("length_class") == cls).select(cols).sort("phase", "arm"))
    for grp in ("MVA-only eight", "signed-MW six"):
        print(f"== days4-end coverage, {grp}, 15 and 30 days")
        d = read("bank_summary.csv").filter(
            (pl.col("group") == grp)
            & (pl.col("band") == "24-120 h")
            & (pl.col("length_class") == "15 and 30 days")
            & (pl.col("phase") == "days4-end")
        )
        print(d.select("arm", "n_events", "cover90", "cover90_lo", "cover90_hi"))
    t = read("trigger_times.csv")
    bk = t.filter(pl.col("source").str.starts_with("bank") & (pl.col("length_days") >= 7))
    print("== onset and end delay, bank events of 7, 15, 30 days")
    print(
        bk.group_by("gate")
        .agg(
            n=pl.len(),
            detected=pl.col("triggered").mean(),
            median_onset_delay=pl.col("days_onset_to_trigger").median(),
            mean_onset_delay=pl.col("days_onset_to_trigger").mean(),
            released=pl.col("snapped_back").mean(),
            median_end_delay=pl.col("days_end_to_snap_back").median(),
            gate_on_before_onset=(pl.col("days_onset_to_trigger") == 0).mean(),
        )
        .sort("gate")
    )
    print(
        "== share of injected events (7, 15, 30 days) whose gate released at least once before the event's end"
    )
    early = []
    for r in bk.iter_rows(named=True):
        tr = r["on_trace"]
        start = -r["trace_start_rel"]  # index of the onset day
        length = int(r["end_day"] - r["onset_day"])
        seg = tr[start : start + length]
        early.append({"gate": r["gate"], "early": "1" in seg and "0" in seg[seg.index("1") :]})
    print(pl.DataFrame(early).group_by("gate").agg(share=pl.col("early").mean()).sort("gate"))
    ft = read("false_triggers.csv")
    print("== false triggers per Primary-year, with the one-sided 95% Poisson upper bound")
    rows = [
        {
            **r,
            "upper95": poisson_upper(r["false_triggers"], r["event_free_origin_days"] / 365.25),
            "primary_years": r["event_free_origin_days"] / 365.25,
        }
        for r in ft.iter_rows(named=True)
    ]
    print(
        pl.DataFrame(rows)
        .select(
            "gate",
            "group",
            "primary_years",
            "false_triggers",
            "false_triggers_per_primary_year",
            "upper95",
            "share_days_event_mode",
        )
        .sort("group", "gate")
    )
    print("== event-free pinball loss, 24-120 h (points of capacity)")
    print(
        read("event_free.csv")
        .sort("group", "gate")
        .select(
            "group",
            "gate",
            "counted",
            "n_rows",
            "pin05",
            "pin95",
            "pin05_minus_G0",
            "pin05_minus_G0_lo",
            "pin05_minus_G0_hi",
            "pin95_minus_G0",
            "pin95_minus_G0_lo",
            "pin95_minus_G0_hi",
            "width90",
            "cover90",
        )
    )
    print("== verified December pair, 24-120 h")
    print(
        read("real_event_phase.csv")
        .filter(pl.col("band") == "24-120 h")
        .select(
            "event",
            "phase",
            "gate",
            "n_rows",
            "cover90",
            "nra_side95",
            "share_origin_days_event_mode",
        )
        .sort("event", "phase", "gate")
    )
    z = np.load(g.WORK / "gate_state.npz")
    print("== thresholds (share of capacity), old and fresh gap, per Primary 0-15")
    print(np.round(z["old_thr"], 3), np.round(z["fresh_thr"], 3))
    off = ~z["G3_on"] & ~z["G0_on"]
    for name in ("fresh", "old"):
        gap, thr = z[f"{name}_gap"], z[f"{name}_thr"]
        m = np.array([np.nanmean(np.where(off[p], gap[p], np.nan)) for p in range(16)])
        print(
            name,
            "mean gap on days both gates are off, as a share of the threshold: max",
            float(np.max(np.abs(m) / thr)),
            "median",
            float(np.median(np.abs(m) / thr)),
        )


if __name__ == "__main__":
    main()

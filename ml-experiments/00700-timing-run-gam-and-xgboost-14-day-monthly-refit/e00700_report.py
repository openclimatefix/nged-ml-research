# ruff: noqa: E501
"""Experiment 700: print every number that `result.md` quotes.

    uv run --no-sync python ml-experiments/00700-*/e00700_report.py

Reads the timing files (`timing/`), the scores (`scores/protocol_year/`), and the forecast file of the
GAM (`backtests/gam_year_sp1.parquet`) under this experiment's private folder, and prints the timing
of the January 2026 run, the timing of the 8-month GAM run, the pooled scores, the coverage, and the
error by forecast day.
"""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Path = ML_EXPERIMENTS_DATA_DIR / "00700-timing-run-gam-and-xgboost-14-day-monthly-refit"
TIMING: Path = ROOT / "timing"
SCORES: Path = ROOT / "scores" / "protocol_year"
ARMS = {
    "GAM": "GAM",
    "XGBoost BG2 (weekly W member)": "XGBoost BG2",
    "XGBoost with monthly refits only": "XGBoost monthly only",
    "Step 5 (quarterly refits)": "Step 5 quarterly",
    "Step 8 (quarterly refits)": "Step 8 quarterly",
    "Step 5 monthly (9a)": "Step 5 monthly",
    "Step 8 monthly (9b)": "Step 8 monthly",
    "Manual heuristic": "Manual heuristic",
    "Climatology": "Climatology",
}
SECTIONS = {"lead_day_1": "Day 1", "lead_days_8_10": "Days 8-10", "all": "All half-hours"}


def january_timing() -> None:
    """Print the wall time per Primary of each GAM stage and of each XGBoost job in January."""
    print("== January 2026 timing, GAM (seconds per Primary, mean and maximum) ==")
    refit = {}
    for name in ("control", "mva", "d15"):
        data = json.loads((TIMING / f"gam_refit_{name}.json").read_text())["sites"]
        refit.update({site: v["wall_s"] for site, v in data.items()})
    print(
        "refit (T-all):", f"mean {np.mean(list(refit.values())):.2f}, max {max(refit.values()):.2f}"
    )
    stage1, stage2, members = {}, {}, {}
    for path in TIMING.glob("gam_stage1_*_D*_D*.json"):
        stage1.update(json.loads(path.read_text()))
    for path in TIMING.glob("gam_stage2_*_D*_D*.json"):
        stage2.update(json.loads(path.read_text()))
    for path in TIMING.glob("gam_members_*.json"):
        members.update(json.loads(path.read_text()))
    s1 = [v["setup_wall_s"] + v["forecast_wall_s"] for v in stage1.values()]
    fc = [v["forecast_wall_s"] for v in stage1.values()]
    print(
        "stage 1 setup plus forecast:",
        f"mean {np.mean(s1):.1f}, max {max(s1):.1f}; forecast alone mean {np.mean(fc):.2f}",
    )
    s2 = [v["wall_s"] for v in stage2.values()]
    print(
        "stage 2:",
        f"mean {np.mean(s2):.1f}, max {max(s2):.1f} at {max(stage2, key=lambda k: stage2[k]['wall_s'])}",
        f"({len(s2)} Primaries)",
    )
    m = [v["wall_s_low"] + v["wall_s_long"] for v in members.values()]
    print("member responses:", f"mean {np.mean(m):.1f}, max {max(m):.1f}", f"({len(m)} Primaries)")
    print(
        "max |stored - rerun|: stage 1 q50",
        max(v["max_abs_q50_diff_vs_stored"] for v in stage1.values()),
        "; stage 2 q95",
        max(v["max_abs_q95_diff_vs_stored"] for v in stage2.values()),
    )
    for path in sorted(TIMING.glob("time_*e00378_arms*.txt")):
        print("solar summary and blend replay", path.name, path.read_text().strip())
    print("\n== January 2026 timing, XGBoost (wall seconds for 20 series, 6 workers) ==")
    for path in sorted(TIMING.glob("time_xgb_*.txt")):
        print(path.name, path.read_text().strip())


def projection() -> None:
    """Print the time for one origin month and the projection for 12 months, from the January timing."""

    def wall(name: str) -> float:
        match = re.search(r"wall=([\d.]+)", (TIMING / name).read_text())
        assert match is not None, name
        return float(match.group(1))

    refit = np.mean(
        [
            v["wall_s"]
            for n in ("control", "mva", "d15")
            for v in json.loads((TIMING / f"gam_refit_{n}.json").read_text())["sites"].values()
        ]
    )
    stage1 = {}
    for path in TIMING.glob("gam_stage1_*_D*_D*.json"):
        stage1.update(json.loads(path.read_text()))
    stage2, members = {}, {}
    for path in TIMING.glob("gam_stage2_*_D*_D*.json"):
        stage2.update(json.loads(path.read_text()))
    for path in TIMING.glob("gam_members_*.json"):
        members.update(json.loads(path.read_text()))
    per_site = (
        refit
        + np.mean([v["setup_wall_s"] + v["forecast_wall_s"] for v in stage1.values()])
        + np.mean([v["wall_s"] for v in stage2.values()])
        + np.mean([v["wall_s_low"] + v["wall_s_long"] for v in members.values()])
    )
    replay_total = sum(
        wall(n)
        for n in (
            "time_e00378_arms.py.txt",
            "time_e00378_arms_low.py.txt",
            "time_mva_e00378_arms.py.txt",
            "time_mva_e00378_arms_low.py.txt",
        )
    )
    gam_month = 15 * per_site + replay_total / 15
    print("\n== Projection from the January timing ==")
    print(
        f"GAM per Primary: {per_site:.1f} s; 15 Primaries {15 * per_site:.0f} s; replay jobs {replay_total:.0f} s over 15 fold months = {replay_total / 15:.0f} s per month"
    )
    print(
        f"GAM one origin month: {gam_month:.0f} s = {gam_month / 60:.1f} min; 12 months {12 * gam_month / 3600:.1f} h of single-thread time"
    )
    refits = sum(
        wall(n)
        for n in (
            "time_xgb_n1h.txt",
            "time_xgb_an_monthly.txt",
            "time_xgb_pt_monthly.txt",
            "time_xgb_n1h_monthly_--offset.txt",
            "time_xgb_an_monthly_--offset.txt",
        )
    )
    assemble = wall("time_xgb_assemble.txt")
    print(
        f"XGBoost monthly refits of M, AN, PT, and two offset twins: {refits:.0f} s = {refits / 60:.1f} min; with the assembly {refits + assemble:.0f} s = {(refits + assemble) / 60:.1f} min"
    )
    print(f"XGBoost monthly-only, 12 months: {12 * (refits + assemble) / 3600:.1f} h")
    weekly = 4.3 * wall("time_xgb_n1h.txt")
    print(
        f"BG2 with weekly W refits (about 4.3 per month, each as long as the monthly M refit): extra {weekly:.0f} s per month; one month {(refits + assemble + weekly) / 60:.0f} min; 12 months {12 * (refits + assemble + weekly) / 3600:.1f} h"
    )


def january_reproduction() -> None:
    """Print how far the January XGBoost refit's median is from the stored forecast."""
    print(
        "\n== January 2026 XGBoost refit against the stored forecast (q50, fraction of capacity) =="
    )
    keys = ["site", "power_fcst_init_time", "valid_time"]
    january = (pl.col("power_fcst_init_time") >= pl.datetime(2026, 1, 1, time_zone="UTC")) & (
        pl.col("power_fcst_init_time") < pl.datetime(2026, 2, 1, time_zone="UTC")
    )
    for arm in ("p_s6_bg2", "p_a3_monthly_in_weekly_slot"):
        new, old = (
            pl.scan_parquet(next(root.glob("00087-*")) / "work" / "arms" / f"{arm}.parquet")
            .filter(january & (pl.col("group") == "Primary"))
            .select(*keys, "q50")
            .collect()
            for root in (ROOT / "sandbox" / "ml-experiments", ML_EXPERIMENTS_DATA_DIR)
        )
        both = new.join(old, on=keys, suffix="_old").with_columns(
            d=(pl.col("q50") - pl.col("q50_old")).abs()
        )
        print(arm, f"rows {both.height:,}; max {both['d'].max():.3f}; mean {both['d'].mean():.4f}")


def year_timing() -> None:
    """Print the wall time of the 8-month GAM stages from the logs."""
    print("\n== 8-month GAM run (origin months 2025-11 to 2026-06) ==")
    for stage in ("stage1", "stage2"):
        walls = []
        for path in TIMING.glob(f"year_e00700_gam_{stage}.py_D*.log"):
            match = re.search(r"wall=([\d.]+) user=([\d.]+) maxrss_kb=(\d+)", path.read_text())
            assert match is not None, path
            walls.append([float(g) for g in match.groups()])
        a = np.array(walls)
        if stage == "stage2":
            diffs = [
                float(m)
                for path in TIMING.glob("year_e00700_gam_stage2.py_D*.log")
                for m in re.findall(r"max_abs_q95_diff_vs_stored': ([\d.]+)", path.read_text())
            ]
            print(
                "stage 2: largest q95 difference from the stored rows (points of capacity):",
                max(diffs),
            )
        print(
            stage,
            f"{len(a)} Primaries; wall mean {a[:, 0].mean():.1f} s, max {a[:, 0].max():.1f} s; CPU mean {a[:, 1].mean():.1f} s; peak memory max {a[:, 2].max() / 1e6:.1f} GB",
        )


def scores() -> None:
    """Print the pooled and grouped scores, the coverage, and the error by forecast day."""
    rows = pl.scan_parquet(SCORES / "rows.parquet")
    print("\n== Pooled over the 15 Primaries all arms share ==")
    print(
        "half-hours per arm:",
        rows.filter(pl.col("arm") == "GAM").select(pl.len()).collect().item(),
        "; Primaries:",
        sorted(rows.select("label").unique().collect()["label"].to_list()),
    )
    for slice_name, label in SECTIONS.items():
        table = (
            rows.filter(pl.col(slice_name))
            .group_by("arm")
            .agg(pl.col("mae_median").mean(), pl.col("crps").mean(), pl.col("twcrps").mean())
            .collect()
        )
        print("--", label)
        for arm, short in ARMS.items():
            r = table.filter(pl.col("arm") == arm).row(0, named=True)
            print(
                f"  {short:24s} MAE {r['mae_median']:.2f}  CRPS {r['crps']:.2f}  twCRPS {r['twcrps']:.3f}"
            )
    print("\n== Groups of experiment 677 (scores.csv) ==")
    scores_csv = pl.read_csv(SCORES / "scores.csv")
    for slice_name in ("all", "lead_day_1", "lead_days_8_10", "cold_nights"):
        s = scores_csv.filter(pl.col("slice") == slice_name)
        for group in s["group"].unique().sort().to_list():
            print("--", group, "/", slice_name)
            for arm, short in ARMS.items():
                r = s.filter((pl.col("group") == group) & (pl.col("arm") == arm))
                vals = {
                    m: r.filter(pl.col("metric") == m)["mean"].item()
                    for m in ("mae_median", "crps", "twcrps")
                }
                print(
                    f"  {short:24s} MAE {vals['mae_median']:.2f}  CRPS {vals['crps']:.2f}  twCRPS {vals['twcrps']:.3f}"
                )
    print("\n== Coverage of the q05 to q95 and q01 to q99 bands (nominal 0.90 and 0.98) ==")
    print(pl.read_csv(SCORES / "extra_coverage.csv").with_columns(pl.col(pl.Float64).round(3)))
    by_day = (
        rows.with_columns(
            day=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours() // 24
            ).cast(pl.Int32)
        )
        .group_by("arm", "day")
        .agg(pl.col("mae_median").mean())
        .collect()
    )
    print("\n== MAE of the median by forecast day, pooled over the 15 Primaries ==")
    wide = (
        by_day.filter(pl.col("day").is_in(list(range(14))))
        .pivot(on="day", index="arm", values="mae_median")
        .sort("arm")
    )
    print(wide.with_columns(pl.col(pl.Float64).round(2)))
    print("MAE by lead day: lowest first, per day:")
    for day in range(14):
        col = by_day.filter(pl.col("day") == day).sort("mae_median")
        print(
            day,
            [
                (ARMS[r["arm"]], round(r["mae_median"], 2))
                for r in col.head(3).iter_rows(named=True)
            ],
        )
    gam = pl.read_parquet(
        ROOT / "backtests" / "gam_year_sp1.parquet",
        columns=["time_series_id", "power_fcst_init_time"],
    )
    per = gam.group_by("time_series_id", "power_fcst_init_time").len()
    short = per.filter(pl.col("len") < 672)
    by_month = (
        short.group_by(pl.col("power_fcst_init_time").dt.strftime("%Y-%m").alias("month"))
        .len()
        .sort("month")
    )
    print("\nGAM pairs with fewer than 672 half-hours, by origin month:", by_month.rows())
    counts = per.group_by("time_series_id").len()["len"]
    print(
        "GAM Primaries:",
        per["time_series_id"].n_unique(),
        "; origins per Primary (min, max):",
        counts.min(),
        counts.max(),
    )
    print(
        "\nGAM file:",
        gam.height,
        "rows;",
        per.height,
        "Primary-origin pairs;",
        per.filter(pl.col("len") < 672).height,
        "with fewer than 672 half-hours;",
        gam["power_fcst_init_time"].n_unique(),
        "origins",
    )


if __name__ == "__main__":
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    january_timing()
    january_reproduction()
    projection()
    year_timing()
    scores()

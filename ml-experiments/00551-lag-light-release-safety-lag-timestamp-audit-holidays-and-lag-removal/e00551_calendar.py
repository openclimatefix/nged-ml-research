# ruff: noqa: ANN001, ANN201, B007, D103, E501, PERF401
"""Part 2: calendar slices of stored per-row CRPS (CPU, no fit).

Arms: L-1d (511), R-DW (539), R-DROP (544, readings present), A-pool (425 on 533's rows), and arm E of 548
(lag-free plus trailing temperature windows). Slices, by target date: (a) Christmas week (24 Dec to
1 Jan), (b) each bank holiday of England and Wales with the day before and the day after, (c) cold nights
(the stored `cold` flag), (d) rows with no labelled event (`cls == "clean"`).

Reads experiment 521's loader for L-1d and A-pool, and the stored per-row score files of 539, 544 and 548.
"""

import json
import math
import os
from datetime import timedelta
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

DD = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
OUT = DD / "00551-lag-light-release-safety-lag-timestamp-audit-holidays-and-lag-removal/scores"
E524 = DD / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator/work/scores"
K = ["primary", "origin", "target"]
ARMS = {
    "L-1d": "L-1d__crps",
    "R-DW": "R-DW__crps",
    "R-DROP": "R-DROP__crps",
    "A-pool": "A-pool__crps",
    "E (548)": "E__crps",
}
BANK_HOLIDAYS = {
    "Christmas Day 2025": "2025-12-25",
    "Boxing Day 2025": "2025-12-26",
    "New Year's Day 2026": "2026-01-01",
    "Good Friday 2026": "2026-04-03",
    "Easter Monday 2026": "2026-04-06",
}
"""Bank holidays of England and Wales whose day, or neighbouring days, fall in a scored target window (July 2026 is unused)."""
RESAMPLES = 20_000


def load() -> pl.DataFrame:
    base = s.load()
    keep = [
        c
        for c in base.columns
        if c.startswith("A-pool__")
        or c
        in (
            *K,
            "L-1d__crps",
            "L-1d-seed0__crps",
            "L-1d-seed1__crps",
            "month",
            "group",
            "cls",
            "lead_hours",
            "cold",
            "xmas",
            "main16",
            "tgt_real",
        )
    ]
    base = base.select(sorted(set(keep), key=keep.index))
    a = pl.concat(
        [
            pl.read_parquet(E524 / f"rows_{m}.parquet", columns=[*K, "A-pool__crps"])
            for m in s.MONTHS
        ]
    )
    base = base.drop("A-pool__crps", strict=False).join(
        a, on=K, how="left", validate="1:1", maintain_order="left"
    )
    for tag, path, cols in (
        ("dw", DD / "00539-l1d-plus-day-and-week-lags/work/scores", ["R-DW__crps"]),
        (
            "drop",
            DD
            / "00544-lag-light-demand-network-robust-same-slot-median-and-lag-dropout/work/scores",
            ["R-DROP__crps"],
        ),
        (
            "e",
            DD
            / "00548-lag-free-release-candidate-assembly-with-trailing-temperature-windows/work/scores",
            ["E__crps"],
        ),
    ):
        extra = pl.concat(
            [pl.read_parquet(path / f"rows_{m}.parquet", columns=[*K, *cols]) for m in s.MONTHS]
        )
        base = base.join(extra, on=K, how="left", validate="1:1", maintain_order="left")
    start = common.GRID_START
    base = base.with_columns(
        date=(
            pl.lit(start).dt.replace_time_zone(None) + pl.duration(minutes=30) * pl.col("target")
        ).dt.date(),
    )
    fin = pl.all_horizontal([pl.col(c).is_finite() for c in ARMS.values()])
    return base.with_columns(ok=fin & pl.col("main16"))


def row_sets(t: pl.DataFrame) -> dict[str, pl.DataFrame]:
    ok = t.filter(pl.col("ok"))
    return {
        "eight": ok.filter(pl.col("group") == "mva_eight"),
        "six": ok.filter((pl.col("group") == "signed_six") & (pl.col("cls") == "clean")),
        "report (D15, D18)": ok.filter(pl.col("group") == "report_only"),
    }


def mde(part: pl.DataFrame) -> float:
    seeds = ["L-1d-seed0__crps", "L-1d-seed1__crps"]
    m = (
        part.group_by("month")
        .agg([pl.col(c).mean() for c in seeds])
        .sort("month")
        .select(seeds)
        .to_numpy()
    )
    f, k = m.shape
    r = m - m.mean(axis=1, keepdims=True)
    return 1.96 * math.sqrt(2) * math.sqrt(float((r**2).sum()) / (f * (k - 1))) / math.sqrt(k)


def month_block_interval(part: pl.DataFrame, a: str, b: str, rng) -> tuple[float, float]:
    d = (
        part.select("month", d=pl.col(ARMS[a]) - pl.col(ARMS[b]))
        .group_by("month")
        .agg(n=pl.len(), s=pl.col("d").sum())
    )
    n, sm = d["n"].to_numpy().astype(float), d["s"].to_numpy()
    draws = rng.integers(0, len(n), (RESAMPLES, len(n)))
    boot = sm[draws].sum(axis=1) / n[draws].sum(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def slice_defs(t: pl.DataFrame) -> dict[str, pl.Expr]:
    d = pl.col("date")
    out = {
        "Christmas week (24 Dec to 1 Jan)": (d >= pl.date(2025, 12, 24))
        & (d <= pl.date(2026, 1, 1))
    }
    for name, day in BANK_HOLIDAYS.items():
        y, m, dd = map(int, day.split("-"))
        from datetime import date

        c = date(y, m, dd)
        out[f"{name} and the days before and after"] = (d >= c - timedelta(days=1)) & (
            d <= c + timedelta(days=1)
        )
    out["all bank-holiday windows"] = pl.any_horizontal(
        [v for k_, v in out.items() if k_ != "Christmas week (24 Dec to 1 Jan)"]
    )
    out["cold nights"] = pl.col("cold")
    out["rows with no labelled event"] = pl.col("cls") == "clean"
    out["all rows (reference)"] = pl.lit(True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = load()
    for a, col in ARMS.items():
        print(
            a,
            "null or non-finite rows by month:",
            t.filter(~pl.col(col).is_finite() | pl.col(col).is_null())
            .group_by("month")
            .len()
            .sort("month")
            .rows(),
        )
    print("rows", t.height, "all arms finite and main16:", int(t["ok"].sum()))
    sets = row_sets(t)
    rng = np.random.default_rng(0)
    recs, daily = [], []
    for sname, part in sets.items():
        m_all = mde(part)
        for slname, expr in slice_defs(t).items():
            sub = part.filter(expr)
            if sub.height == 0:
                continue
            days = sub["date"].n_unique()
            rec = {
                "set": sname,
                "slice": slname,
                "rows": sub.height,
                "target_days": days,
                "mde_all_rows": round(m_all, 3),
            }
            for a, col in ARMS.items():
                rec[a] = float(sub[col].mean())  # ty: ignore[invalid-argument-type]
            for a in ("R-DW", "R-DROP", "A-pool", "E (548)"):
                rec[f"{a} - L-1d"] = rec[a] - rec["L-1d"]
            # per-day signs against L-1d
            dd = sub.group_by("date").agg([pl.col(c).mean() for c in ARMS.values()]).sort("date")
            for a, col in ARMS.items():
                if a == "L-1d":
                    continue
                diff = (dd[col] - dd["L-1d__crps"]).to_numpy()
                rec[f"{a} worse than L-1d on days"] = int((diff > 0).sum())
            if days >= 10 and sub["month"].n_unique() >= 5:
                for a in ("R-DW", "R-DROP", "A-pool", "E (548)"):
                    lo, hi = month_block_interval(sub, a, "L-1d", rng)
                    rec[f"{a} - L-1d lo"], rec[f"{a} - L-1d hi"] = lo, hi
            fold = (
                sub.group_by("month").agg([pl.col(c).mean() for c in ARMS.values()]).sort("month")
            )
            rec["R-DW - L-1d by fold"] = json.dumps(
                {
                    r["month"]: round(r["R-DW__crps"] - r["L-1d__crps"], 3)
                    for r in fold.iter_rows(named=True)
                }
            )
            recs.append(rec)
            if sname == "eight" and slname.startswith("Christmas"):
                for r in dd.iter_rows(named=True):
                    daily.append({"date": str(r["date"]), **{a: r[c] for a, c in ARMS.items()}})
    tab = pl.DataFrame(recs, infer_schema_length=None)
    tab.write_csv(OUT / "calendar_slices.csv")
    pl.DataFrame(daily).write_csv(OUT / "calendar_christmas_week_daily_eight.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=260, tbl_cols=30, fmt_str_lengths=60):
        print(
            tab.select(
                "set",
                "slice",
                "rows",
                "target_days",
                *ARMS,
                "R-DW - L-1d",
                "R-DW worse than L-1d on days",
            ).with_columns(pl.col(pl.Float64).round(3))
        )
    # also the headline 24-120 h band and the eight, Christmas week
    chk = {"rows": t.height, "rows_ok": int(t["ok"].sum()), "bank_holidays": BANK_HOLIDAYS}
    (OUT / "calendar_checks.json").write_text(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()

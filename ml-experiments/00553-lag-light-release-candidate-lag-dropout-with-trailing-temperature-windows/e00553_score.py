# ruff: noqa: E501, PERF401
"""Score the experiment 553 arms against L-1d, R-DW, R-DROP, E, and A-pool on experiment 533's rows.

Reads the per-row scores written by `e00553_rows.py` and the stored per-row scores of experiments
524 (A-pool), 539 (R-DW), 544 (R-DROP and R-DROP with every reading absent), and 548 (arm E, the
lag-free recipe with trailing windows). Uses experiment 521's loader, lead bands, row sets, and
month-block bootstrap (95%, 5 blocks, near the minimum to maximum of the folds), widened for the
Bonferroni count `TESTS`. Output under this experiment's private data folder, `scores/`:
`absolute.csv`, `contrasts.csv`, `gap_closed.csv`, and `checks.json`.
"""

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD: Final[Path] = ML_EXPERIMENTS_DATA_DIR
ROOT: Final[Path] = (
    DD / "00553-lag-light-release-candidate-lag-dropout-with-trailing-temperature-windows"
)
E524: Final[Path] = (
    DD / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator" / "work" / "scores"
)
E539: Final[Path] = DD / "00539-l1d-plus-day-and-week-lags" / "work" / "scores"
E544: Final[Path] = (
    DD
    / "00544-lag-light-demand-network-robust-same-slot-median-and-lag-dropout"
    / "work"
    / "scores"
)
E548: Final[Path] = (
    DD
    / "00548-lag-free-release-candidate-assembly-with-trailing-temperature-windows"
    / "work"
    / "scores"
)
K: Final[list[str]] = ["primary", "origin", "target"]
METRICS: Final[tuple[str, ...]] = ("crps", "pin_q95", "pin_q99", "cov_q95", "cov_q99")
RATES: Final[tuple[str, ...]] = ("RDWT-D30", "RDWT-D15")
OWN: Final[tuple[str, ...]] = tuple(a for r in RATES for a in (r, f"{r}-absent", f"{r}-s2"))
ARMS: Final[tuple[str, ...]] = (
    *OWN, "R-DW", "R-DROP", "R-DROP-absent", "E", "A-pool", "L-1d", "manual_heuristic",
)  # fmt: skip
SEEDED: Final[tuple[str, ...]] = tuple(a for a in OWN if "-s2" not in a)
CONTRASTS: Final[tuple[tuple[str, str, bool], ...]] = (
    *[(r, "L-1d", True) for r in RATES],
    *[(r, "R-DW", True) for r in RATES],
    *[(f"{r}-absent", "E", True) for r in RATES],
    *[(r, "A-pool", False) for r in RATES],
    *[(r, "R-DROP", False) for r in RATES],
    *[(f"{r}-absent", "L-1d", False) for r in RATES],
    *[(f"{r}-absent", "R-DROP-absent", False) for r in RATES],
    ("RDWT-D15", "RDWT-D30", False),
    ("RDWT-D15-absent", "RDWT-D30-absent", False),
    ("E", "L-1d", False),
)
TESTS: Final[int] = 84
"""2 rates x 3 counted contrasts x 7 lead bands x 2 row sets (eight and six)."""
BANK_HOLIDAYS: Final[dict[str, str]] = {
    "Christmas Day 2025": "2025-12-25",
    "Boxing Day 2025": "2025-12-26",
    "New Year's Day 2026": "2026-01-01",
    "Good Friday 2026": "2026-04-03",
    "Easter Monday 2026": "2026-04-06",
}
HEADLINE: Final[str] = "24-120h"


def slices() -> dict[str, pl.Expr]:
    """Return the row slices: all rows, cold nights, no labelled event, Christmas week, bank holidays."""
    d = pl.col("date")
    windows = []
    for day in BANK_HOLIDAYS.values():
        c = date.fromisoformat(day)
        windows.append((d >= c - timedelta(days=1)) & (d <= c + timedelta(days=1)))
    return {
        "all rows": pl.col("main16"),
        "cold winter nights": pl.col("cold"),
        "rows with no labelled event": pl.col("cls") == "clean",
        "Christmas week (24 Dec to 1 Jan)": (d >= pl.date(2025, 12, 24))
        & (d <= pl.date(2026, 1, 1)),
        "bank-holiday windows": pl.any_horizontal(windows),
    }


def table() -> pl.DataFrame:
    """Return experiment 521's table joined with the per-row scores of every other arm."""
    base = s.load()
    extra = pl.concat(
        [
            pl.read_parquet(
                E524 / f"rows_{m}.parquet",
                columns=[
                    *K,
                    *[f"{a}__{x}" for a in ("A-pool", "manual_heuristic") for x in METRICS],
                ],
            )
            for m in s.MONTHS
        ]
    )
    assert base.select(K).equals(extra.select(K))
    drop = [c for c in extra.columns[3:] if c in base.columns]
    out = pl.concat([base.drop(drop), extra.drop(K)], how="horizontal")
    n0 = out.height
    sources = (
        (E539, ("R-DW",)),
        (E544, ("R-DROP", "R-DROP-absent")),
        (E548, ("E",)),
        (ROOT / "work" / "scores", OWN),
    )
    for folder, names in sources:
        cols = [f"{a}__{x}" for a in names for x in METRICS]
        part = pl.concat(
            [pl.read_parquet(folder / f"rows_{m}.parquet", columns=[*K, *cols]) for m in s.MONTHS]
        )
        out = out.join(part, on=K, how="left", validate="1:1", maintain_order="left")
    assert out.height == n0
    start = common.GRID_START
    return out.with_columns(
        date=(
            pl.lit(start).dt.replace_time_zone(None) + pl.duration(minutes=30) * pl.col("target")
        ).dt.date(),
        headline=(pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 120),
    )


def mde(*, sub: pl.DataFrame, cols: list[str]) -> float:
    """Return the minimum detectable effect from the spread of the seeds' per-fold means."""
    mm = (
        sub.group_by("month")
        .agg([pl.col(c).mean() for c in cols])
        .sort("month")
        .select(cols)
        .to_numpy()
    )
    f, k = mm.shape
    r = mm - mm.mean(axis=1, keepdims=True)
    return 1.96 * math.sqrt(2) * math.sqrt(float((r**2).sum()) / (f * (k - 1))) / math.sqrt(k)


def main() -> None:
    """Write the absolute scores, the contrasts, and the share of L-1d's gap to A-pool each arm closes."""
    t = table().with_columns(band=s.band_expr())
    need = [f"{a}__crps" for a in ARMS if a != "manual_heuristic"]
    fin = pl.all_horizontal([pl.col(c).is_finite() for c in need])
    t = t.with_columns(main16=pl.col("main16") & fin)
    checks = {
        "rows": t.height,
        "rows with every arm finite": int(t["main16"].sum()),
        "tests": TESTS,
    }
    (ROOT / "scores").mkdir(exist_ok=True, parents=True)
    (ROOT / "scores" / "checks.json").write_text(json.dumps(checks, indent=1))
    print(checks)
    rng = np.random.default_rng(0)
    absolute, contrasts, gap = [], [], []
    seedcols = ["L-1d-seed0__crps", "L-1d-seed1__crps"]
    for name in ("eight", "six", "report"):
        part = s.row_set(table=t, name=name)
        for slice_name, expr in slices().items():
            sliced = part.filter(expr.fill_null(value=False))
            for band in (*s.BANDS, s.ALL, HEADLINE):
                sub = (
                    sliced.filter(pl.col("headline"))
                    if band == HEADLINE
                    else s.in_band(part=sliced, band=band)
                )
                if sub.height < 200:
                    continue
                m = mde(sub=sub, cols=seedcols)
                for arm in ARMS:
                    for metric in METRICS:
                        col = f"{arm}__{metric}"
                        if col in sub.columns:
                            absolute.append({"set": name, "slice": slice_name, "band": band, "arm": arm, "metric": metric, "n": sub.height, "value": float(sub.filter(pl.col(col).is_finite())[col].mean())})  # fmt: skip
                if slice_name == "all rows":
                    l1d, apool = sub["L-1d__crps"].mean(), sub["A-pool__crps"].mean()
                    for arm in (*OWN, "R-DW", "R-DROP", "R-DROP-absent", "E"):
                        gap.append({"set": name, "band": band, "arm": arm, "closed": float((l1d - sub[f"{arm}__crps"].mean()) / (l1d - apool)), "l1d_minus_apool": float(l1d - apool)})  # fmt: skip
                for a, b, counted in CONTRASTS:
                    for metric in ("crps", "pin_q95", "pin_q99"):
                        ca, cb = f"{a}__{metric}", f"{b}__{metric}"
                        d = sub.select("month", "day", d=pl.col(ca) - pl.col(cb)).filter(
                            pl.col("d").is_finite()
                        )
                        if d.height < 200:
                            continue
                        st = s.block_stats(diff=d, rng=rng, tests=TESTS)
                        contrasts.append({"set": name, "slice": slice_name, "band": band, "metric": metric, "contrast": f"{a} - {b}", "counted": counted and band in s.BANDS and name != "report" and slice_name == "all rows" and metric == "crps", "tests": TESTS, "mde": m, **st})  # fmt: skip
    pl.DataFrame(absolute).write_csv(ROOT / "scores" / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(ROOT / "scores" / "contrasts.csv")
    pl.DataFrame(gap).write_csv(ROOT / "scores" / "gap_closed.csv")
    print("counted contrasts:", sum(1 for c in contrasts if c["counted"]))


if __name__ == "__main__":
    main()

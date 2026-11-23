# ruff: noqa: E501, PT018
"""Experiment 720, rungs 0a and 0d: the census of reverse flow and the gate table.

`python e00720_census.py --selftest` runs on synthetic arrays only. `python e00720_census.py --run`
reads the stored panel (read-only) and writes two tables under the private folder, then prints the
substation-level aggregates. The first real run waits for the code review.

Rung 0a counts negative half-hours per signed-MW Primary in the training span and the scored span of
each fold, and gives the lowest reading of each MVA-only Primary and the counts under 2 and 5 points
(continuity: a reading that never comes near zero rules out a sign change).

Rung 0d computes experiment 420's gate statistic per Primary and fold cutoff from the training
window: the Spearman correlation of the reading with `u` on sunny summer-midday rows, and the 1st
percentile of those readings. The gate is open when the correlation exceeds 0.3 or the 1st
percentile is under 3 points, and it needs 200 rows.
"""

import argparse
import sys
from pathlib import Path
from typing import Final

import e00720_real as real
import numpy as np
import polars as pl
from scipy.stats import spearmanr

GATE_CORRELATION: Final[float] = 0.3
GATE_PERCENTILE_POINTS: Final[float] = 3.0
GATE_K_MIN: Final[float] = 0.7
MIN_GATE_ROWS: Final[int] = 200
MIDDAY_HOURS: Final[tuple[int, int]] = (10, 14)
SUMMER_MONTHS: Final[tuple[int, int]] = (4, 9)


def hour_month(*, stamp_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the UTC hour and month of each half-hour's START (the stamp marks its end)."""
    start = pl.Series(stamp_us - real.US_PER_HALF_HOUR).cast(pl.Datetime("us"))
    return start.dt.hour().to_numpy(), start.dt.month().to_numpy()


def census_table(
    *, power_pct: np.ndarray, sites: list[str], stamp_us: np.ndarray, folds: tuple[str, ...]
) -> pl.DataFrame:
    """Return per-site, per-fold, per-span counts of readings, negatives, and low readings."""
    rows = []
    for site_index, site in enumerate(sites):
        y = power_pct[site_index]
        signed = site in real.SIGNED_SIX or site in real.SIGNED_REPORT_ONLY
        for fold in folds:
            w = real.fold_windows(fold=fold)
            spans = {
                "train": (w.train_start_us < stamp_us) & (stamp_us <= w.train_end_us),
                "scored": (w.cutoff_us < stamp_us) & (stamp_us <= w.score_end_us),
            }
            for name, sel in spans.items():
                v = y[sel & np.isfinite(y)]
                rows.append(
                    {
                        "primary": site,
                        "group": "signed MW" if signed else "MVA only",
                        "fold": fold,
                        "span": name,
                        "n_rows": len(v),
                        "n_negative": int((v < 0.0).sum()),
                        "share_negative": float((v < 0.0).mean()) if len(v) else float("nan"),
                        "min_reading_points": float(v.min()) if len(v) else float("nan"),
                        "n_below_2_points": int((np.abs(v) < 2.0).sum()),
                        "n_below_5_points": int((np.abs(v) < 5.0).sum()),
                    }
                )
    return pl.DataFrame(rows)


def gate_statistic(
    *, y: np.ndarray, u: np.ndarray, k: np.ndarray, stamp_us: np.ndarray
) -> dict[str, float | bool | int]:
    """Return the gate decision of one Primary from rows of one training window."""
    hour, month = hour_month(stamp_us=stamp_us)
    ok = np.isfinite(y) & np.isfinite(u) & np.isfinite(k)
    sunny = (
        ok
        & (hour >= MIDDAY_HOURS[0])
        & (hour < MIDDAY_HOURS[1])
        & (month >= SUMMER_MONTHS[0])
        & (month <= SUMMER_MONTHS[1])
        & (k >= GATE_K_MIN)
    )
    n = int(sunny.sum())
    if n < MIN_GATE_ROWS:
        return {
            "rho_spearman": float("nan"),
            "p1_points": float("nan"),
            "n_rows": n,
            "gate_open": False,
        }
    rho = float(spearmanr(y[sunny], u[sunny]).statistic)
    p1 = float(np.percentile(y[sunny], 1.0))
    return {
        "rho_spearman": rho,
        "p1_points": p1,
        "n_rows": n,
        "gate_open": bool(rho > GATE_CORRELATION or p1 < GATE_PERCENTILE_POINTS),
    }


def gate_table(
    *,
    sites: list[str],
    stamp_us: np.ndarray,
    power_pct: np.ndarray,
    u: np.ndarray,
    k: np.ndarray,
    folds: tuple[str, ...],
) -> pl.DataFrame:
    """Return the gate statistic of every Primary at each fold cutoff, from the training window only."""
    rows = []
    for i, site in enumerate(sites):
        for fold in folds:
            w = real.fold_windows(fold=fold)
            sel = (w.train_start_us < stamp_us) & (stamp_us <= w.train_end_us)
            stat = gate_statistic(
                y=power_pct[i][sel], u=u[i][sel], k=k[i][sel], stamp_us=stamp_us[sel]
            )
            rows.append({"primary": site, "fold": fold, **stat})
    return pl.DataFrame(rows)


def synthetic_arrays(*, seed: int = 0) -> dict:
    """Return a 3-year synthetic panel: an exporter, a non-exporter, a signed Primary with a few negatives."""
    rng = np.random.default_rng(seed)
    start = int(np.datetime64("2022-09-01T00:30:00", "us").astype(np.int64))
    n = 3 * 365 * 48
    stamp = start + np.arange(n, dtype=np.int64) * real.US_PER_HALF_HOUR
    hour, month = hour_month(stamp_us=stamp)
    day = ((stamp - real.US_PER_HALF_HOUR) // real.US_PER_DAY).astype(np.int64)
    k_day = rng.uniform(0.1, 1.0, day.max() + 1)
    k = k_day[day - day.min()]
    sun = np.clip(np.sin(np.pi * (hour - 5) / 14.0), 0.0, 1.0) * (
        ((month >= 4) & (month <= 9)) * 1.0 + 0.3
    )
    u = sun * k
    base = 30.0 + 5.0 * rng.normal(size=n)
    exporter = np.abs(base - 120.0 * u) + 2.0
    non_exporter = base + 4.0 * rng.normal(size=n) - 5.0 * u
    signed = base - 40.0 * u
    return {
        "sites": ["EXPORT", "NOEXP", "SIGNED"],
        "stamp_us": stamp,
        "power_pct": np.stack([exporter, non_exporter, signed]),
        "u": np.stack([u, u, u]),
        "k": np.stack([k, k, k]),
    }


def selftest() -> int:
    """Check the census and the gate on synthetic arrays with known answers, and the day matrix."""
    a = synthetic_arrays()
    a["sites"] = ["D12", "D2", "D5"]  # exporter (MVA), non-exporter (MVA), signed
    census = census_table(
        power_pct=a["power_pct"], sites=a["sites"], stamp_us=a["stamp_us"], folds=("2025-06",)
    )
    got = census.filter((pl.col("primary") == "D5") & (pl.col("span") == "scored"))
    w = real.fold_windows(fold="2025-06")
    sel = (w.cutoff_us < a["stamp_us"]) & (a["stamp_us"] <= w.score_end_us)
    want = int((a["power_pct"][2][sel] < 0).sum())
    assert got["n_negative"][0] == want and want > 0, (got["n_negative"][0], want)
    assert census.filter(pl.col("group") == "MVA only")["n_negative"].sum() == 0
    gates = gate_table(
        sites=a["sites"],
        stamp_us=a["stamp_us"],
        power_pct=a["power_pct"],
        u=a["u"],
        k=a["k"],
        folds=("2025-06",),
    )
    by = {r["primary"]: r["gate_open"] for r in gates.iter_rows(named=True)}
    assert by["D12"] is True, gates
    assert by["D2"] is False, gates
    day_matrix, days = real.to_day_matrix(values=np.arange(100.0), stamp_us=a["stamp_us"][:100])
    assert day_matrix.shape[1] == 48 and days[1] - days[0] == real.US_PER_DAY
    assert np.isfinite(day_matrix).sum() == 100
    assert real.fold_windows(fold="2026-06").score_end_us == real.HOLDOUT_US
    print(
        f"PASS census and gate on synthetic arrays: {want} negatives counted; exporter gate open, non-exporter shut; day matrix and holdout cap"
    )
    return 0


def run(*, dense_raw: Path) -> int:
    """Read the stored panel and write the census and gate tables under the private folder."""
    import e00720_guard as guard

    panel = real.load_panel(dense_raw_path=dense_raw)
    census = census_table(
        power_pct=panel.power_pct, sites=panel.sites, stamp_us=panel.stamp_us, folds=real.FOLDS
    )
    gates = gate_table(
        sites=panel.sites,
        stamp_us=panel.stamp_us,
        power_pct=panel.power_pct,
        u=panel.u,
        k=panel.k,
        folds=real.FOLDS,
    )
    census.write_csv(guard.private_path(name="rung0a_census.csv"))
    gates.write_csv(guard.private_path(name="rung0d_gate_table.csv"))
    print(
        census.filter(pl.col("span") == "scored")
        .group_by("primary", "group")
        .agg(pl.col("n_rows").sum(), pl.col("n_negative").sum(), pl.col("min_reading_points").min())
        .sort("primary")
    )
    print(gates.pivot(on="fold", index="primary", values="gate_open").sort("primary"))
    return 0


def main() -> int:
    """Parse arguments and run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--run", action="store_true", help="read the stored panel (after the code review)"
    )
    ap.add_argument("--dense-raw", type=Path, default=None)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.run:
        if args.dense_raw is None:
            ap.error("--run needs --dense-raw (the dense_raw.npy of experiment 713's cache)")
        return run(dense_raw=args.dense_raw)
    ap.error("give --selftest or --run")
    return 2


if __name__ == "__main__":
    sys.exit(main())

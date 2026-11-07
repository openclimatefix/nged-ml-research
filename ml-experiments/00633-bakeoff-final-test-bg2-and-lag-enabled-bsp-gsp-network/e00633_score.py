# ruff: noqa: D103, E501, PLR0912, PLR0915
"""Experiment 633: score experiment 620's three networks, BG2, and the two comparators on the final-test window, once (CPU).

A copy of experiment 620's `e00620_score.py` (an unmerged experiment) with BG2 added as a fourth forecast.
The networks' forecasts are read from experiment 620's export folder, read only. Original docstring:

Input: the export tables `K1_<month>`, `Candidate_<month>` and `K3_<month>` that `e00620_assemble.py`
wrote (forecasts in MW or MVA). Output: CSV tables under the scores folder.

Comparators are regenerated from the cleaned power with the definitions of the `baseline_forecasters`
package, in numpy:

- `manual_heuristic`: the ensemble of observed powers at the valid time minus 1, 2, 3, 4, 5 and 6
  weeks and minus 49 to 55 weeks; a lag no longer than the lead is absent. The 11 quantile levels are
  taken from the members that exist (linear interpolation). A row with no member is dropped.
- `climatology`: for each Primary, the quantiles of the cleaned power before the cutoff in the cell
  (local month, local half-hour of day, local weekend flag), each sample counted in the 9 cells within
  one month and one half-hour (the package's pooling).

All scores use the rows on which every system has a forecast and the observed power is a finite,
unflagged reading. Scores are shares of capacity (in percent where printed). The interval of every
mean is a day-block bootstrap over forecast origin days; every contrast prints its Bonferroni count.

Usage: python e00620_score.py MONTH CUTOFF EXPORT_DIR OUT_DIR   (CUTOFF = the first day not trained on)
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import e00087_common as common87  # ty: ignore[unresolved-import]
import e00582_arms as a  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00601_common as k601  # ty: ignore[unresolved-import]
import e00633_common as k633
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
QCOLS = list(a.QCOLS)
TAUS = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
LAGS_H = (168, 336, 504, 672, 840, 1008, 8232, 8400, 8568, 8736, 8904, 9072, 9240)
SYSTEMS = ("K1", "Candidate", "K3", "BG2", "Manual heuristic", "Climatology")
NETWORKS = ("K1", "Candidate", "K3")
CONTRAST_LEFT = ("K1", "Candidate", "K3", "BG2")
BG2_ARM = "bg2_single_fit"
N_BOOT = 4000
SUBSTATIONS = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07" / "substations.csv"
START_US, STEP_US = c.k578.prep.GRID_START_US, c.k578.prep.STEP_US


def pinball(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    d = y[:, None] - q
    return np.maximum(TAUS * d, (TAUS - 1.0) * d)


def crps(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 2.0 * pinball(y=y, q=q).mean(axis=1)


def twcrps(*, y: np.ndarray, q: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Threshold-weighted CRPS stand-in of experiment 319: the CRPS of the values clipped below at each Primary's threshold."""
    d = np.maximum(y, thr)[:, None] - np.maximum(q, thr[:, None])
    return 2.0 * np.maximum(TAUS * d, (TAUS - 1.0) * d).mean(axis=1)


def manual_heuristic(
    *, clean: np.ndarray, prim: np.ndarray, idx: np.ndarray, lead_h: np.ndarray
) -> np.ndarray:
    members = np.full((len(prim), len(LAGS_H)), np.nan, np.float32)
    for k, h in enumerate(LAGS_H):
        j = idx - 2 * h
        ok = (lead_h < h) & (j >= 0)
        members[ok, k] = clean[prim[ok], j[ok]]
    have = np.isfinite(members).any(axis=1)
    out = np.full((len(prim), len(TAUS)), np.nan, np.float32)
    out[have] = np.nanquantile(members[have], TAUS, axis=1, method="linear").T
    return out


def climatology(
    *, clean: np.ndarray, prim: np.ndarray, idx: np.ndarray, cutoff_idx: int
) -> np.ndarray:
    stamps = pl.datetime_range(
        START_US_DT,
        START_US_DT + (clean.shape[1] - 1) * STEP_DT,
        interval="30m",
        time_zone="UTC",
        eager=True,
    )
    local = stamps.dt.convert_time_zone("Europe/London")
    month = (local.dt.month() - 1).to_numpy()
    hh = (local.dt.hour() * 2 + local.dt.minute() // 30).to_numpy()
    wk = (local.dt.weekday() >= 6).to_numpy().astype(int)
    out = np.full((len(prim), len(TAUS)), np.nan, np.float32)
    offsets = [(dm, dh) for dm in (-1, 0, 1) for dh in (-1, 0, 1)]
    for p in range(clean.shape[0]):
        train = np.flatnonzero(np.isfinite(clean[p, :cutoff_idx]))
        pooled = pl.concat(
            [
                pl.DataFrame(
                    {
                        "cell": (((month[train] + dm) % 12) * 48 + (hh[train] + dh) % 48) * 2
                        + wk[train],
                        "v": clean[p, train],
                    }
                )
                for dm, dh in offsets
            ]
        )
        table = pl.DataFrame({"cell": pooled["cell"].unique().sort()})
        qs = pooled.group_by("cell").agg(
            [pl.col("v").quantile(float(t), "linear").alias(f"q{i}") for i, t in enumerate(TAUS)]
        )
        look = {r[0]: r[1:] for r in qs.sort("cell").iter_rows()}
        rows = np.flatnonzero(prim == p)
        cells = (month[idx[rows]] * 48 + hh[idx[rows]]) * 2 + wk[idx[rows]]
        out[rows] = np.array([look.get(int(x), (np.nan,) * len(TAUS)) for x in cells], np.float32)
        del table
    return out


START_US_DT = datetime(2019, 9, 1, tzinfo=UTC)
STEP_DT = __import__("datetime").timedelta(minutes=30)


def day_boot(
    *, day: np.ndarray, vals: np.ndarray, seed: int = 633
) -> tuple[float, float, float, np.ndarray]:
    """Return the mean, and the day-block bootstrap draws of the mean (resampling whole origin days)."""
    days, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=vals)
    n = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(days), size=(N_BOOT, len(days)))
    draws = s[pick].sum(axis=1) / n[pick].sum(axis=1)
    return (
        float(vals.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        draws,
    )


def interval(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def bg2_quantiles(*, base: pl.DataFrame, cap: dict) -> np.ndarray:
    """Return BG2-single-fit's 11 quantiles (shares of capacity) on the export's rows, spread factor 1.15 about the median."""
    series = common87.series_table().select("site", "time_series_id", "effective_capacity_mw")
    names = dict(zip(series["time_series_id"].to_list(), series["site"].to_list(), strict=True))
    parts = []
    for t in sorted(base["time_series_id"].unique().to_list()):
        site = names[int(t)]
        f = pl.read_parquet(k633.ROOT / "preds" / BG2_ARM / f"{site}.parquet")
        assert abs(float(f["capacity"][0]) - cap[int(t)]) <= 1e-4 * cap[int(t)], (
            site,
            float(f["capacity"][0]),
            cap[int(t)],
        )
        parts.append(f.select(*KEYS, *QCOLS))
    bg2 = pl.concat(parts)
    joined = base.select(*KEYS).join(bg2, on=KEYS, how="left", validate="1:1")
    assert joined.height == base.height
    arr = joined.select(QCOLS).to_numpy().astype(np.float64)
    arr = np.sort(arr, axis=1)
    median = arr[:, [5]]
    return median + k633.SPREAD_FACTOR * (arr - median)


def main(month: str, cutoff: str, export: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("E00633_PHASE") == "predict":
        k633.redirect()  # the extended grid and observed power hold the window
        assert (cutoff == "2026-07-01") == (month == "2026-07"), (month, cutoff)
    sub = pl.read_csv(SUBSTATIONS)
    cap = dict(zip(sub["time_series_id"].to_list(), sub["capacity"].to_list(), strict=True))
    grp = dict(zip(sub["time_series_id"].to_list(), sub["group"].to_list(), strict=True))
    name = dict(zip(sub["time_series_id"].to_list(), sub["primary_name"].to_list(), strict=True))
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    clean = c.k578.clean()
    cutoff_idx = int((datetime.fromisoformat(cutoff).replace(tzinfo=UTC) - START_US_DT) / STEP_DT)
    frames = {n: pl.read_parquet(export / f"{n}_{month}.parquet") for n in NETWORKS}
    base = frames["K1"].select(*KEYS, "lead_hours", "observed_power", "unit")
    for n in NETWORKS:
        assert frames[n].select(KEYS).equals(base.select(KEYS)), f"{n}: rows differ from K1's"
    tid = base["time_series_id"].to_numpy()
    prim = np.array([prim_of[int(t)] for t in tid])
    capv = np.array([cap[int(t)] for t in tid])
    idx = ((base["valid_time"].dt.epoch("us") - START_US) // STEP_US).to_numpy().astype(np.int64)
    oidx = (
        ((base["power_fcst_init_time"].dt.epoch("us") - START_US) // STEP_US)
        .to_numpy()
        .astype(np.int64)
    )
    lead = base["lead_hours"].to_numpy().astype(np.float64)
    y = base["observed_power"].to_numpy().astype(np.float64) / capv
    q = {
        n: np.sort(frames[n].select(QCOLS).to_numpy().astype(np.float64) / capv[:, None], axis=1)
        for n in NETWORKS
    }
    q["Manual heuristic"] = manual_heuristic(clean=clean, prim=prim, idx=idx, lead_h=lead)
    q["Climatology"] = climatology(clean=clean, prim=prim, idx=idx, cutoff_idx=cutoff_idx)
    q["BG2"] = bg2_quantiles(base=base, cap=cap)
    ok = np.isfinite(y)
    dropped = {"no observed reading": int((~ok).sum())}
    for n in SYSTEMS:
        miss = ok & ~np.isfinite(q[n]).all(axis=1)
        dropped[f"{n}: no forecast"] = int(miss.sum())
        ok &= ~miss
    print("rows", len(y), "common scored rows", int(ok.sum()), dropped, flush=True)
    pl.DataFrame({"what": list(dropped), "rows": list(dropped.values())}).write_csv(
        out / "dropped_rows.csv"
    )
    sel = np.flatnonzero(ok)
    y, lead, prim, tid, oidx, idx = y[sel], lead[sel], prim[sel], tid[sel], oidx[sel], idx[sel]
    q = {n: v[sel] for n, v in q.items()}
    day = oidx // 48
    # threshold of the twCRPS stand-in: each Primary's 90th percentile of the cleaned power before the cutoff
    thr_p = np.array([np.nanquantile(clean[p, :cutoff_idx], 0.9) for p in range(16)])
    thr = thr_p[prim]
    peak = np.nanmax(clean[:, :cutoff_idx], axis=1)
    band = np.digitize(
        lead, [24, 48, 72, 96, 120, 144, 168, 192, 216, 240], right=False
    )  # 0 = under 24 h
    stats = {}
    for n in SYSTEMS:
        pin = pinball(y=y, q=q[n])
        stats[n] = {
            "crps": 2.0 * pin.mean(axis=1) * 100,
            "mae": np.abs(y - q[n][:, 5]) * 100,
            "twcrps": twcrps(y=y, q=q[n], thr=thr) * 100,
            "cover90": ((q[n][:, 1] <= y) & (y <= q[n][:, 9])).astype(float),
            "cover98": ((q[n][:, 0] <= y) & (y <= q[n][:, 10])).astype(float),
            "ge05": (y >= q[n][:, 1]).astype(float),
            "le95": (y <= q[n][:, 9]).astype(float),
            "pin01": pin[:, 0] * 100,
            "pin05": pin[:, 1] * 100,
            "pin95": pin[:, 9] * 100,
            "pin99": pin[:, 10] * 100,
            "width90": (q[n][:, 9] - q[n][:, 1]) * 100,
        }
    # headline table with day-block intervals
    rows = []
    for n in SYSTEMS:
        for m, v in stats[n].items():
            mean, lo, hi, _ = day_boot(day=day, vals=v)
            rows.append(
                {"system": n, "measure": m, "mean": mean, "lo": lo, "hi": hi, "rows": len(v)}
            )
    pl.DataFrame(rows).write_csv(out / "headline.csv")
    # by lead band and by Primary
    for label, key, groups in (("lead_band", band, range(11)), ("primary", prim, range(16))):
        rows = []
        for g in groups:
            m_ = key == g
            if not m_.any():
                continue
            for n in SYSTEMS:
                rows.append(
                    {
                        label: int(g),
                        "system": n,
                        "rows": int(m_.sum()),
                        **{k: float(v[m_].mean()) for k, v in stats[n].items()},
                    }
                )
        pl.DataFrame(rows).write_csv(out / f"by_{label}.csv")
    pl.DataFrame(
        {
            "primary": list(range(16)),
            "label": [next((name[t] for t in prim_of if prim_of[t] == p), "") for p in range(16)],
            "group": [next((grp[t] for t in prim_of if prim_of[t] == p), "") for p in range(16)],
        }
    ).write_csv(out / "primaries.csv")
    # paired contrasts of CRPS and MAE and twCRPS, Bonferroni-corrected intervals
    pairs = [
        (a_, b_)
        for a_ in CONTRAST_LEFT
        for b_ in SYSTEMS
        if a_ != b_
        and (b_ not in CONTRAST_LEFT or CONTRAST_LEFT.index(b_) > CONTRAST_LEFT.index(a_))
    ]
    metrics = ("crps", "mae", "twcrps")
    n_tests = len(pairs) * len(metrics)
    alpha = 0.05 / n_tests
    rows = []
    for a_, b_ in pairs:
        for m in metrics:
            mean, lo, hi, draws = day_boot(day=day, vals=stats[a_][m] - stats[b_][m])
            blo, bhi = interval(draws, alpha)
            rows.append(
                {
                    "a": a_,
                    "b": b_,
                    "measure": m,
                    "diff_a_minus_b": mean,
                    "lo95": lo,
                    "hi95": hi,
                    "lo_bonferroni": blo,
                    "hi_bonferroni": bhi,
                    "n_tests": n_tests,
                    "excludes_zero_bonferroni": bool(blo > 0 or bhi < 0),
                }
            )
    pl.DataFrame(rows).write_csv(out / "contrasts.csv")
    print("Bonferroni count:", n_tests, flush=True)
    # curtailment cost: six MW Primaries, 12-36 h, q95 as the planning level, limits as fractions of the pre-cutoff peak
    rows = []
    lead_ok = (lead >= k601.LEAD_COST[0]) & (lead < k601.LEAD_COST[1])
    for p in range(16):
        t0 = next((t for t in prim_of if prim_of[t] == p), None)
        if t0 is None or grp[t0] != "six MW Primaries":
            continue
        cp = float(cap[t0])
        s_ = lead_ok & (prim == p)
        if s_.sum() < 100:
            continue
        for n in SYSTEMS:
            rec = {"primary": p, "label": name[t0], "system": n, "rows": int(s_.sum())}
            for f in k601.FRACTIONS:
                lim = np.full(int(s_.sum()), f * peak[p] * cp)
                rec[f"cost_gbp_per_day_{f}"] = float(
                    k601.avoidable_cost(u=q[n][s_][:, 9] * cp, y=y[s_] * cp, limit=lim).mean() * 48
                )
            rows.append(rec)
    pl.DataFrame(rows).write_csv(out / "curtailment_cost.csv")
    # per-row scores for the figures
    np.savez_compressed(
        out / "rows.npz",
        prim=prim,
        lead=lead,
        day=day,
        idx=idx,
        y=y,
        **{f"crps_{n}": stats[n]["crps"] for n in SYSTEMS},  # ty: ignore[invalid-argument-type]
        **{f"q50_{n}": q[n][:, 5] for n in SYSTEMS},
    )
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))

# ruff: noqa: E501, D103
"""Experiment 687, step 1: causal feature tables for the stored banks of experiments 650 and 652 (CPU).

For each development month, writes `feats/<month>_<bank>.npz` where `<bank>` is `M` (the stored bank,
neighbour signals carry the mirror), `X` (no mirror: neighbour signals read the neighbour's unshifted power), `S` (mirror at 60% of its stored strength), or `real` (experiment 647's real rows, no shift).
Keys, with one row per (pass, origin, Primary) of the *full grid* of that month's origins and all 16
Primaries, so a placebo can read another Primary's row:

- `prim`, `oi`, `pass_`: the row's key.
- `feat` (n, 4, N_FEATURES), `flag` (n, 4): the features and the explicit flag at the four cutoffs
  `day` (origin - 48), `week` (origin - 336), `new4` (origin - 8), and `new6` (origin - 12).
- `lags` (n, 8), `lags_miss` (n, 8): the residual (x10, clipped) of the power at 6, 9, 12, 18, 24, 36,
  48, and 72 hours before the origin (experiment 681's L1 block).
- `in_day`, `in_week`, `in_new4` (n,): 1 where that cutoff lies inside the Primary's injected shift
  (zero for `real`).

Usage: python e00687_tables.py MONTH
"""

import sys
import time

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00647_data as d647  # ty: ignore[unresolved-import]
import e00650_common as c650  # ty: ignore[unresolved-import]
import e00687_features as f
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00687-causal-detector-features-for-the-prevailing-forecast-heads"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
CUTOFFS = {"day": 48, "week": 336, "new4": 8, "new6": 12}
LAG_HOURS = (6, 9, 12, 18, 24, 36, 48, 72)
LAG_SCALE = 10.0
LAG_CLIP = 10.0
PRE_DAYS = 100
MIRROR_S = 0.6


def neighbour_map() -> dict[int, list[int]]:
    nb_of, _cap, _mw = c650.world()
    return nb_of


def noise_units(*, clean: np.ndarray, months: tuple[str, ...]) -> dict[int, np.ndarray]:
    """Return the per-Primary noise unit for each block width, from the uninjected gap of `months`."""
    gaps = []
    for m in months:
        tab = d647.lag_free_table(month=m)
        a = c650.month_start_idx(month=m)
        b = min(a + 31 * 48, c650.WINDOW_START_IDX)
        gaps.append((clean[:, a:b] - tab.median[:, a:b]).astype(np.float64))
    units = {w: f.block_unit(gaps=gaps, w=w) for w in f.W}
    assert all(np.isfinite(u).all() for u in units.values()), units
    return units


def passes_of(*, month: str) -> list[int]:
    return list(c650.ALL_PASSES)


def build_rows(
    *, grids: f.Grids, gap_own: np.ndarray, oi: np.ndarray, neighbours: dict
) -> dict[str, np.ndarray]:
    """Return the feature arrays of every (origin in `oi`, Primary) row, Primary-major within an origin."""
    n_p = gap_own.shape[0]
    prim = np.tile(np.arange(n_p), len(oi))
    o = np.repeat(oi, n_p)
    feat = np.zeros((len(o), len(CUTOFFS), f.N_FEATURES), np.float32)
    flag = np.zeros((len(o), len(CUTOFFS)), bool)
    for k, (_name, back) in enumerate(CUTOFFS.items()):
        assert (o - back - grids.start >= 0).all(), "cutoff before the window"
        assert (o - back <= o - 8).all(), "a cutoff later than origin minus 4 hours"
        feat[:, k], flag[:, k] = f.features_at(
            g=grids, prim=prim, t=o - back, neighbours=neighbours
        )
    lags = np.zeros((len(o), len(LAG_HOURS)), np.float32)
    miss = np.zeros((len(o), len(LAG_HOURS)), bool)
    for j, h in enumerate(LAG_HOURS):
        assert 2 * h >= 12, "a lag newer than 6 hours before the origin"
        v = gap_own[prim, o - 2 * h - grids.start]
        bad = ~np.isfinite(v)
        miss[:, j] = bad
        lags[:, j] = np.clip(np.where(bad, 0.0, v) * LAG_SCALE, -LAG_CLIP, LAG_CLIP)
    return {"prim": prim.astype(np.int8), "oi": o.astype(np.int32), "feat": feat, "flag": flag, "lags": lags, "lags_miss": miss}  # fmt: skip


def main(month: str) -> None:
    t0 = time.time()
    clean = k578.clean()
    table = d647.lag_free_table(month=month)
    units = noise_units(clean=clean, months=c650.MONTHS)
    nb_of = neighbour_map()
    bank = np.load(c650.WORK / "head_data" / f"bank_{month}.npz")
    real = np.load(d647.DATA / f"real_{month}.npz")
    out_dir = WORK / "feats"
    out_dir.mkdir(parents=True, exist_ok=True)
    ms = c650.month_start_idx(month=month)
    start = ms - PRE_DAYS * 48
    end_idx = min(ms + 31 * 48, c650.WINDOW_START_IDX)
    assert end_idx <= c650.WINDOW_START_IDX, "the final-test window is never read"
    med = table.median[:, start:end_idx].astype(np.float64)
    results: dict[str, list] = {b: [] for b in ("M", "X", "S")}
    ev_all = []
    for r in c650.ALL_PASSES:
        own, mirror, _events = c650.shifts_for(clean=clean, month=month, r=r)
        sel = bank["org_pass"] == r
        oi = np.unique(bank["org_oi"][sel]).astype(np.int64)
        c = clean[:, start:end_idx]
        fixed = np.where(np.isfinite(c), c + own[:, start:end_idx], c).astype(np.float64)
        mir = mirror[:, start:end_idx]
        gap_own = fixed - med
        gap_ref = c.astype(np.float64) - med
        ev = pl.read_parquet(c650.ROWS / f"events_{month}_r{r:02d}.parquet")
        ev_all.append(ev)
        for name, scale in (("M", 1.0), ("X", 0.0), ("S", MIRROR_S)):
            sig = np.where(np.isfinite(c), c + scale * mir, c) - med
            grids = f.build_grids(
                gap=gap_own, gap_sig=sig, gap_ref=gap_ref, units=units, start=start
            )
            rows = build_rows(grids=grids, gap_own=gap_own, oi=oi, neighbours=nb_of)
            rows["pass_"] = np.full(len(rows["oi"]), r, np.int16)
            on = {
                int(p): (int(a), int(b))
                for p, a, b in zip(ev["primary"], ev["onset_idx"], ev["end_idx"], strict=True)
            }
            for k, key in ((0, "in_day"), (1, "in_week"), (2, "in_new4")):
                t = rows["oi"] - list(CUTOFFS.values())[k]
                lo = np.array([on[int(p)][0] for p in rows["prim"]])
                hi = np.array([on[int(p)][1] for p in rows["prim"]])
                rows[key] = ((t >= lo) & (t < hi)).astype(np.int8)
            results[name].append(rows)
        print(f"{month} r{r:02d} {time.time() - t0:.0f}s", flush=True)
    for name, parts in results.items():
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(out_dir / f"{month}_{name}.npz", **cat)  # ty: ignore[invalid-argument-type]
    # real rows: no shift
    c = clean[:, start:end_idx].astype(np.float64)
    gap = c - med
    grids = f.build_grids(gap=gap, gap_sig=gap, gap_ref=gap, units=units, start=start)
    oi = np.unique(real["org_oi"]).astype(np.int64)
    rows = build_rows(grids=grids, gap_own=gap, oi=oi, neighbours=nb_of)
    rows["pass_"] = np.zeros(len(rows["oi"]), np.int16)
    np.savez(out_dir / f"{month}_real.npz", **rows)  # ty: ignore[invalid-argument-type]
    np.savez(out_dir / "units.npz", **{f"w{w}": u for w, u in units.items()})  # ty: ignore[invalid-argument-type]
    print(f"{month} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

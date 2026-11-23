# ruff: noqa: E501, PT018
"""Experiment 716: choose the rung of the synthetic worlds from the NumPy references (WP5a; design section 8.2).

Usage: run.sh e00716_world_rung.py [--fold 2026-01] [--rungs 1 2 3 4 5] [--force]
       run.sh e00716_world_rung.py --selftest

This is a CPU task that reads the dense weather table and fits no neural network. For each rung in
the fixed order, it builds worlds WH and WT, fits the naive and the oracle ridge reference of each
Primary, and prints the `gap` of each world (the naive reference's MAE of the median minus the
oracle's, on solar-block validation tokens pooled over the four Primaries). It stops at the first
rung where both `gap` values are at least 0.010 and writes `work/world_rung.json` with every `gap` it
computed. If no rung passes, it writes the file with `rung: null` and exits with status 2, so that the
manager is told and F7 stops before any PV fit. A rung whose file `work/world_rung/rung<N>.json`
exists is not recomputed unless `--force` is given.

`--selftest` checks the block ridge against a direct solve and the world's formula on small arrays.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00716_arms as arms
import e00716_pv as pv
import e00716_worlds as wd
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME_713_CACHE = "00713-msa-axial-attention-demand-network-part-1"


def fold_cutoff_idx(*, fold: str) -> int:
    """Return the grid index of the first day of the fold's month at 00:00 UTC."""
    year, month = (int(x) for x in fold.split("-"))
    stamp_us = int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1e6)
    return int((stamp_us - lay.GRID_START_US) // lay.STEP_US)


def load_dense4() -> wd.Dense4:
    """Read the stored dense table of experiment 713 (read-only) and the calendar of the whole grid."""
    path = ML_EXPERIMENTS_DATA_DIR / NAME_713_CACHE / "cache" / "dense_raw.npy"
    raw = np.load(path, mmap_mode="r")
    n_grid = raw.shape[1]
    holidays = lay.england_bank_holidays(years=range(2019, 2029))
    day_type = lay.day_types_for_grid(
        grid_start_us=lay.GRID_START_US, n_grid=n_grid, holidays=holidays
    )
    half_hour = lay.local_half_hours_for_grid(grid_start_us=lay.GRID_START_US, n_grid=n_grid)
    doy = pv.day_of_year(grid_start_us=lay.GRID_START_US, n_grid=n_grid)
    return wd.dense4_from_raw(
        raw=raw, day_type=day_type.astype(np.int64), half_hour=half_hour.astype(np.int64), doy=doy
    )


def run_rungs(
    *, dense: wd.Dense4, cutoff_idx: int, rungs: list[int], out_dir: Path, force: bool
) -> dict:
    """Compute the `gap` of WH and WT per rung in order and stop at the first rung that passes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    chosen: int | None = None
    for number in rungs:
        rung = wd.RUNGS[number - 1]
        path = out_dir / f"rung{number}.json"
        if path.exists() and not force:
            record = json.loads(path.read_text())
            print(f"rung {number}: read from {path.name}", flush=True)
        else:
            started = time.monotonic()
            world = wd.build_world(dense=dense, rung=rung)
            gaps = {}
            for kind in ("WH", "WT"):
                res = wd.gap_of_world(world=world, dense=dense, kind=kind, cutoff_idx=cutoff_idx)
                gaps[kind] = {
                    "gap": res.gap, "gap_per_primary": res.gap_per_primary, "mae_naive": res.mae_naive,
                    "mae_oracle": res.mae_oracle, "penalties": res.penalties,
                }  # fmt: skip
                print(
                    f"rung {number} {kind}: gap {res.gap:.4f} (naive {res.mae_naive:.4f}, oracle {res.mae_oracle:.4f})",
                    flush=True,
                )
            record = {
                "rung": number, "settings": vars(rung), "gaps": gaps, "seconds": round(time.monotonic() - started, 1),
                "passes": all(g["gap"] >= wd.GAP_MIN for g in gaps.values()),  # ty: ignore[unsupported-operator]
            }  # fmt: skip
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(record, indent=1, default=str))
            tmp.rename(path)
        results[str(number)] = record
        if record["passes"]:
            chosen = number
            break
    return {"rung": chosen, "gap_min": wd.GAP_MIN, "results": results}


def selftest() -> int:
    """Check the block ridge against a direct solve, the world's formula, and the order of the gap on a toy world."""
    rng = np.random.default_rng(0)
    n, p, k = 60, 12, 3
    hs = rng.normal(size=(n, p))
    z = rng.normal(size=(n, wd.N_LEADS, 48, k))
    z[..., 0] = 1.0
    y = rng.normal(size=(n, wd.N_LEADS, 48)) + z[..., 1] * 0.5 + (hs[:, 0])[:, None, None]
    piece = wd.piece_of(hs=hs, z=z, y=y)
    fit = wd.prepare(piece=piece)
    lam = 37.0
    w, v = wd.solve(fit=fit, penalty=lam)
    for h in (0, 17, 47):
        rows = np.concatenate(
            [np.repeat(hs, wd.N_LEADS, axis=0), z[:, :, h].reshape(-1, k)], axis=1
        )
        gram = rows.T @ rows
        gram[:p, :p] += lam * np.eye(p)
        gram[p:, p:] += 1e-8 * np.eye(k)
        direct = np.linalg.solve(gram, rows.T @ y[:, :, h].reshape(-1))
        assert np.allclose(direct[:p], w[h], atol=1e-8) and np.allclose(
            direct[p:], v[h], atol=1e-8
        ), h
    a, b = hs[:40], hs[40:]
    sub = wd.piece_of(hs=a, z=z[:40], y=y[:40]) + wd.piece_of(hs=b, z=z[40:], y=y[40:])
    assert (
        np.allclose(sub.a, piece.a) and np.allclose(sub.b, piece.b) and np.allclose(sub.q, piece.q)
    )
    print("block ridge equals the direct normal equations (3 half-hours), and pieces add")
    # a toy world on a small grid
    n_grid = 48 * 400
    steps = np.arange(n_grid)
    hour = (steps % 48) / 2.0
    elevation = np.sin(np.pi * np.clip((hour - 4.0) / 16.0, 0.0, 1.0))
    season = 0.6 + 0.4 * np.sin(2 * np.pi * steps / (48 * 365))
    cosz = np.clip(0.9 * elevation * season, 0.0, 1.0) - 0.2 * (elevation <= 0)
    sky = np.repeat(rng.uniform(0.2, 1.0, size=(4, n_grid // 48)), 48, axis=1)
    dense = wd.Dense4(
        ghi=1000.0 * np.clip(cosz, 0, None)[None] * sky, temperature_c=np.tile(8.0 + 6 * np.sin(steps / 48 / 58.0), (4, 1)),
        dew_c=np.tile(4.0 + 3 * np.sin(steps / 48 / 40.0), (4, 1)), precip=np.zeros((4, n_grid)),
        cos_zenith=np.tile(cosz, (4, 1)), sin_azimuth=np.tile(np.sin(np.pi + (hour - 12) / 12 * np.pi / 2), (4, 1)),
        cos_azimuth=np.tile(np.cos(np.pi + (hour - 12) / 12 * np.pi / 2), (4, 1)),
        day_type=rng.integers(0, 8, n_grid), half_hour=(steps % 48).astype(np.int64), doy=(1.0 + (steps // 48) % 365),
    )  # fmt: skip
    world = wd.build_world(dense=dense, rung=wd.RUNGS[4])
    assert np.isfinite(world.y).all()
    night = dense.cos_zenith <= 0
    assert np.abs(world.resid[night]).max() < 6 * wd.NOISE_SD, (
        "the night tokens must carry only noise"
    )
    assert (world.d >= 0).all() and world.d.max() <= 1.0 + 1e-9
    res = wd.gap_of_world(world=world, dense=dense, kind="WH", cutoff_idx=48 * 300, log=False)
    assert res.mae_oracle <= res.mae_naive + 0.002, (res.mae_naive, res.mae_oracle)
    res_t = wd.gap_of_world(world=world, dense=dense, kind="WT", cutoff_idx=48 * 300, log=False)
    assert res_t.mae_oracle <= res_t.mae_naive + 0.002, (res_t.mae_naive, res_t.mae_oracle)
    print(
        f"toy world: WH naive {res.mae_naive:.4f} oracle {res.mae_oracle:.4f}; WT naive {res_t.mae_naive:.4f} oracle {res_t.mae_oracle:.4f}"
    )
    print("selftest passed")
    return 0


def main() -> int:
    """Run the self-test or the ladder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="2026-01")
    parser.add_argument("--rungs", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    dense = load_dense4()
    cutoff = fold_cutoff_idx(fold=args.fold)
    work = arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR, arm="Z") / "work"
    summary = run_rungs(
        dense=dense,
        cutoff_idx=cutoff,
        rungs=args.rungs,
        out_dir=work / "world_rung",
        force=args.force,
    )
    summary["fold"] = args.fold
    summary["code"] = arms.code_record()
    path = work / "world_rung.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1, default=str))
    tmp.rename(path)
    if summary["rung"] is None:
        print("NO RUNG PASSED: tell the manager; F7 stops before any PV fit", flush=True)
        return 2
    print(f"chosen rung: {summary['rung']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

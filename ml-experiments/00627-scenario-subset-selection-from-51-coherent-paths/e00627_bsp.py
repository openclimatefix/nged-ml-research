# ruff: noqa: E501
"""Experiment 627, part 2: choose K of the 51 paths by the peak export of a BSP group sum.

A bulk supply point (BSP) group is the sum, in MW, of the Primaries beneath it that report real
power: D13 is D9, D10, and D11; D17 is D5, D14, and D16 (experiment 161's covered children, so the
sum misses unmetered load and the Primaries that report apparent power). Export is minus the sum.
Per run, BSP, and lead band the path summary is the peak export over the band. Arms: `R`, `S-bsp`
(strata of that BSP's peak export), `C` (the part 1 medoids), and `Q` (K = 4 only: the paths whose
peak export sits at the q50, q90, q95, and q99 of the 51). Scores: the part 1 coverage and tail
measures on the peak export, and the gate, the share of observed daily peak exports above the
largest daily peak export of the K chosen paths (`All` is the same share for the 51 paths).

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00627-*/env.sh
OMP_NUM_THREADS=1 nice -n 19 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00627-*/e00627_bsp.py --workers 4
```
"""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00436_build as build  # ty: ignore[unresolved-import]
import e00627_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

BSPS: Final[dict[str, tuple[str, ...]]] = {"D13": ("D9", "D10", "D11"), "D17": ("D5", "D14", "D16")}
Q_LEVELS: Final[tuple[float, ...]] = (0.5, 0.9, 0.95, 0.99)
STEPS_PER_DAY: Final[int] = 48
MIN_OBSERVED_STEPS: Final[int] = 44
KS_BSP: Final[tuple[int, ...]] = (3, 4, 5, 10)


def group_indices() -> tuple[dict[str, list[int]], np.ndarray]:
    """Return each BSP's child node indices and the capacity (MW) of the 16 Primaries."""
    series = build.adapter.series_table()
    labels = series["label"].to_list()
    cap = np.array(series["capacity_mw"].to_list(), dtype=np.float64)
    return {b: [labels.index(c) for c in kids] for b, kids in BSPS.items()}, cap


def daily_peaks(*, export: np.ndarray, first_full: int, n_days: int) -> np.ndarray:
    """Return the peak export per full valid day, (..., n_days), from a (..., 702) export array."""
    seg = export[..., first_full : first_full + n_days * STEPS_PER_DAY]
    return seg.reshape(*seg.shape[:-1], n_days, STEPS_PER_DAY).max(axis=-1)


def q_weights(*, summary: np.ndarray) -> np.ndarray:
    """Return (1, 51) weights on the paths at the q50, q90, q95, and q99 of the 51 summaries."""
    order = np.argsort(summary, kind="stable")
    w = np.zeros((1, sc.N_MEMBERS))
    for q in Q_LEVELS:
        w[0, order[int(np.ceil(q * sc.N_MEMBERS)) - 1]] += 1.0 / len(Q_LEVELS)
    return w


def process_run(*, file: str) -> list[dict]:  # noqa: PLR0912, PLR0915
    """Return the BSP-group rows of one run for both couplings."""
    groups, cap = group_indices()
    z = np.load(file)
    init = datetime.fromisoformat(str(z["init"]) + "+00:00").astimezone(UTC)
    day_tag = f"{init:%Y-%m-%d}"
    actual = z["actual"].astype(np.float64)
    valid_time = np.array([init + (i + 1) * sc.STEP for i in range(sc.N_STEPS)])
    actual[:, valid_time >= sc.HOLDOUT] = np.nan
    first_full = next(i for i, t in enumerate(valid_time) if t.hour == 0 and t.minute == 30)
    n_days = (sc.N_STEPS - first_full) // STEPS_PER_DAY
    day_start = first_full + np.arange(n_days) * STEPS_PER_DAY
    run_index = int(init.toordinal())
    rows: list[dict] = []
    for coupling in sc.COUPLINGS:
        paths = np.load(sc.WORK436 / "paths" / coupling / f"{day_tag}.npy").astype(np.float64)
        for bsp, kids in groups.items():
            mw = paths[:, kids, :] / 100.0 * cap[kids][None, :, None]
            export = -mw.sum(axis=1)  # (51, 702)
            obs_kids = actual[kids, :] / 100.0 * cap[kids][:, None]
            obs_export = -obs_kids.sum(axis=0)
            obs_ok = (~np.isnan(obs_kids)).all(axis=0)
            dp_paths = daily_peaks(export=export, first_full=first_full, n_days=n_days)  # (51, days)
            obs_day = np.full(n_days, np.nan)
            for d in range(n_days):
                s = slice(day_start[d], day_start[d] + STEPS_PER_DAY)
                if obs_ok[s].sum() >= MIN_OBSERVED_STEPS:
                    obs_day[d] = np.nanmax(np.where(obs_ok[s], obs_export[s], np.nan))
            full_summ = {b: export[:, sc.band_steps(band=b)].max(axis=1) for b in sc.BANDS}
            sel_summary = full_summ["d1_10"]
            day_ok = ~np.isnan(obs_day)
            all_exceed = float((obs_day[day_ok] > dp_paths.max(axis=0)[day_ok]).mean()) if day_ok.any() else np.nan
            base = {"power_fcst_init_time": init, "coupling": coupling, "bsp": bsp}
            rows.append({**base, "arm": "All", "k": sc.N_MEMBERS, "band": "d1_10", "exceed": all_exceed, "n_days": int(day_ok.sum()), "w1": 0.0, "gap05": 0.0, "gap50": 0.0, "gap95": 0.0, "tail_low": 1.0, "tail_high": 1.0})
            for k in KS_BSP:
                rng = np.random.default_rng([sc.SEED, run_index, k, 7])
                arms = {
                    "R": sc.random_weights(k=k, rng=rng, n=sc.N_DRAWS),
                    "S-bsp": sc.stratified_weights(summary=sel_summary, k=k, rng=rng, n=sc.N_DRAWS),
                    "C": sc.medoid_weights(paths=paths, steps=sc.band_steps(band="d1_10"), k=k, rng=np.random.default_rng([sc.SEED, k, 3])),
                }
                if k == len(Q_LEVELS):
                    arms["Q"] = q_weights(summary=sel_summary)
                for arm, w in arms.items():
                    for band in sc.BANDS:
                        m = sc.distribution_rows(weights=w, summary=full_summ[band])
                        row = {**base, "arm": arm, "k": k, "band": band, **{key: float(np.nanmean(v)) for key, v in m.items()}, "n_days": 0, "exceed": np.nan}
                        if band == "d1_10" and day_ok.any():
                            chosen = w > 0
                            top = np.where(chosen[:, :, None], dp_paths[None], -np.inf).max(axis=1)  # (D, days)
                            row["exceed"] = float((obs_day[day_ok][None] > top[:, day_ok]).mean())
                            row["n_days"] = int(day_ok.sum())
                        rows.append(row)
    return rows


def process_run_kw(file: str) -> list[dict]:
    """Return `process_run` for one file (a positional wrapper for `pool.map`)."""
    return process_run(file=file)


def main() -> None:
    """Score every stored run in parallel worker processes and write `bsp.parquet`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    files = [str(f) for f in sorted((sc.WORK436 / "runs").glob("*.npz"))[: args.max_runs]]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for file, result in zip(files, pool.map(process_run_kw, files), strict=True):
            rows.extend(result)
            print(f"{Path(file).stem} scored", flush=True)
    pl.DataFrame(rows).write_parquet(sc.OUT / f"bsp{args.suffix}.parquet")


if __name__ == "__main__":
    main()

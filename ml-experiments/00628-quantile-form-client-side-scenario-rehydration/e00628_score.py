# ruff: noqa: E501, ANN001, PLR0915
"""Experiment 628: score client-side scenarios drawn from the 11 stored quantiles.

For each of experiment 436's 44 runs the client holds `work/q11/<date>.npy` (25 nodes, 702
half-hours, 11 quantiles) and a dependence table, and draws 1,000 scenarios per arm. Arms are
coupling (IND independent; COP the fitted copula of experiment 436 with no ensemble-rank component;
SS a Schaake shuffle of 51 past observed paths, jittered within each rank) times tail rule (clamp,
linear, gaussian). The in-pipeline reference arms are experiment 436's stored 51-path J0 (IND), J5
(COP), and J3 (SS). Writes four tables to this experiment's `work/` folder:

- `crps.parquet`: CRPS sums by arm, family, and lead band;
- `exceed.parquet`: exceedance counts at the outer levels, by arm, family, and lead band;
- `coherence.parquet`: energy and variogram scores at 51 paths, as in experiment 625;
- `query.parquet`: the lowest-wind and 5th-percentile-solar picks on day 3.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00628-*/env.sh
OMP_NUM_THREADS=1 nice -n 19 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00628-*/e00628_score.py --workers 3
```
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import client_rehydrate as client  # noqa: E402  # ty: ignore[unresolved-import]
import e00434_score as s434  # noqa: E402  # ty: ignore[unresolved-import]
import e00436_build as build  # noqa: E402  # ty: ignore[unresolved-import]
import e00436_scenarios as scen  # noqa: E402  # ty: ignore[unresolved-import]
import e00436_score as score  # noqa: E402  # ty: ignore[unresolved-import]
import e00625_score as s625  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402

SLUG: Final[str] = "00628-quantile-form-client-side-scenario-rehydration"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
K: Final[int] = 1000
N: Final[int] = build.N_NODES
S: Final[int] = build.N_STEPS
KIND: Final[np.ndarray] = build.KIND
COUPLINGS: Final[tuple[str, ...]] = ("IND", "COP", "SS")
TAILS: Final[tuple[str, ...]] = ("clamp", "linear", "gaussian")
REFERENCE: Final[dict[str, str]] = {"IND": "J0", "COP": "J5", "SS": "J3"}
BANDS: Final[dict[str, tuple[float, float]]] = {
    "h0_12": (0.0, 12.0),
    "h12_36": (12.0, 36.0),
    "h36_120": (36.0, 120.0),
    "h120_240": (120.0, 240.0),
}
LOWER: Final[tuple[float, ...]] = (0.002, 0.01, 0.05)
UPPER: Final[tuple[float, ...]] = (0.95, 0.99, 0.998)
KNOT_COLUMN: Final[dict[float, int]] = {0.01: 0, 0.05: 1, 0.95: 9, 0.99: 10}
DAY3: Final[slice] = slice(96, 144)
FAMILIES: Final[dict[str, int]] = {"demand": 0, "wind": 1, "solar": 2}
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
BOUNDS: Final[np.ndarray] = np.array(
    [[np.nan, np.nan]] * 16 + [[0.0, 100.0]] * 3 + [[0.0, np.inf]] * 6
)
"""Demand is not clipped, wind is clipped to [0, 100], and solar to zero only: the stored solar quantiles exceed 100 in 11% of daylight rows."""


def band_steps() -> dict[str, np.ndarray]:
    """Return the step indices (0 = lead 0.5 h) of each lead band."""
    lead = (np.arange(S) + 1) * 0.5
    return {n: np.flatnonzero((lead > lo) & (lead <= hi)) for n, (lo, hi) in BANDS.items()}


def uniforms(
    *,
    coupling: str,
    chol: np.ndarray,
    phi: np.ndarray,
    history: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return K uniform levels (K, N, S) from the named coupling."""
    if coupling == "IND":
        return rng.uniform(size=(K, N, S)).astype(np.float32)
    if coupling == "COP":
        return client.copula_uniforms(chol=chol, phi=phi, k=K, steps=S, rng=rng)
    return client.schaake_uniforms(history=history, k=K, rng=rng).astype(np.float32)


def crps_rows(*, label, path, actual, init, extra) -> list[dict]:
    """Return CRPS sums by family and lead band for K paths (K, N, S)."""
    have = ~np.isnan(actual)
    crps = s625.plain_crps(paths=path.astype(np.float64), actual=np.nan_to_num(actual))
    rows = []
    for band, steps in band_steps().items():
        for fam, f in FAMILIES.items():
            nodes = np.flatnonzero(f == KIND)
            ok = have[np.ix_(nodes, steps)]
            if ok.any():
                rows.append(
                    {"power_fcst_init_time": init, "arm": label, **extra, "band": band,
                     "family": fam, "crps_sum": float(crps[np.ix_(nodes, steps)][ok].sum()),
                     "n": int(ok.sum())}
                )  # fmt: skip
    return rows


def exceed_rows(*, label, quant, actual, usable, init, extra) -> list[dict]:
    """Return exceedance counts; `quant[level]` is the arm's quantile function at that level (N, S)."""
    have = ~np.isnan(actual) & usable
    rows = []
    for band, steps in band_steps().items():
        for fam, f in FAMILIES.items():
            nodes = np.flatnonzero(f == KIND)
            ok = have[np.ix_(nodes, steps)]
            if not ok.any():
                continue
            a = actual[np.ix_(nodes, steps)]
            for level, qv in quant.items():
                q = qv[np.ix_(nodes, steps)]
                hit = (a < q) if level < 0.5 else (a > q)
                rows.append(
                    {"power_fcst_init_time": init, "arm": label, **extra, "band": band,
                     "family": fam, "level": level, "hits": int((hit & ok).sum()), "n": int(ok.sum())}
                )  # fmt: skip
    return rows


def stratify(*, u: np.ndarray) -> np.ndarray:
    """Return 51 levels (51, N, S): the rank of each draw among the 51, as (rank + 1) / 52.

    Experiment 436's reference paths take the 51 values at the levels k/52, so the 51-path client
    set is stratified the same way and the comparison is not confounded by sampling noise.
    """
    ranks = np.argsort(np.argsort(u, axis=0, kind="stable"), axis=0, kind="stable")
    return ((ranks + 1) / (u.shape[0] + 1)).astype(np.float64)


BSP: Final[pl.DataFrame] = pl.read_csv(OUT / "bsp_mapping.csv")


def picks(*, path: np.ndarray) -> dict[str, float]:
    """Return the day-3 picks of K paths (K, N, S).

    The picks are lowest wind, 5th-percentile solar, and the 95th percentile of each BSP's summed
    Primaries.
    """
    wind = path[:, KIND == 1, DAY3].mean(axis=(1, 2))
    solar = path[:, KIND == 2, DAY3].sum(axis=2).mean(axis=1)
    out = {"wind_min": float(wind.min()), "solar_p05": float(np.quantile(solar, 0.05))}
    for name in BSP["bsp"].unique().sort().to_list():
        rows = BSP.filter(pl.col("bsp") == name)
        mw = (
            path[:, rows["node"].to_numpy(), DAY3]
            * rows["capacity_mw"].to_numpy()[None, :, None]
            / 100.0
        )
        out["bsp_q95_" + name.split()[0].lower()] = float(
            np.quantile(mw.sum(axis=1).mean(axis=1), 0.95)
        )
    return out


def process_run(*, file: str, chol: np.ndarray, phi: np.ndarray) -> dict[str, list[dict]]:
    """Score every arm on one run; return the rows of the four tables."""
    run = score.load_run(path=Path(file))
    init = datetime.fromisoformat(str(run["init"]) + "+00:00").astimezone(UTC)
    tag = f"{init:%Y-%m-%d}"
    observed = np.load(score.WORK / "observed.npz")["observed"]
    kind = run["kind"]
    actual = run["actual"].astype(np.float64)
    valid = np.array([init + (i + 1) * build.STEP for i in range(S)])
    actual[:, valid >= HOLDOUT] = np.nan
    q11 = np.load(OUT / "q11" / f"{tag}.npy").astype(np.float64)
    # check: the 11 quantiles reproduce experiment 436's 51 values at the levels k/52
    levels = np.arange(1, 52) / 52
    worst = 0.0
    for node in range(N):
        check = client.invert(
            q=q11[node], u=np.broadcast_to(levels[:, None], (51, S)), tail="clamp", bounds=None
        )
        scored = ~np.isnan(run["actual"][node])
        diff = np.abs(check.T - run["values"][node].astype(np.float64))[scored]
        worst = max(worst, float(diff.max(initial=0.0)))
    if worst > 0.01:
        msg = f"{tag}: the stored 11 quantiles do not reproduce experiment 436's 51 values"
        raise ValueError(msg)
    usable = (q11[:, :, 10] - q11[:, :, 0]) > 1e-6
    pool = score.schaake_pool(init=init, observed=observed, kind=kind)
    history = np.stack([pool[k % len(pool)][2] for k in range(s434.N_MEMBERS)]).astype(np.float64)
    out: dict[str, list[dict]] = {k: [] for k in ("crps", "exceed", "coherence", "query")}
    knots = {lv: q11[:, :, KNOT_COLUMN[lv]] for lv in (0.01, 0.05, 0.95, 0.99)}
    out["exceed"].extend(
        exceed_rows(label="N_knots", quant=knots, actual=actual, usable=usable, init=init, extra={"coupling": "all", "tail": "none"})
    )  # fmt: skip
    ref = {
        c: np.load(score.WORK / "paths" / REFERENCE[c] / f"{tag}.npy").astype(np.float64)
        for c in COUPLINGS
    }
    coh_paths: dict[str, np.ndarray] = {}
    for c in COUPLINGS:
        label = REFERENCE[c]
        coh_paths[label] = ref[c]
        out["crps"].extend(crps_rows(label=label, path=ref[c], actual=actual, init=init, extra={"coupling": c, "tail": "pipeline", "k": 51}))  # fmt: skip
        qs = {lv: np.quantile(ref[c], lv, axis=0) for lv in (0.01, 0.05, 0.95, 0.99)}
        out["exceed"].extend(exceed_rows(label=label, quant=qs, actual=actual, usable=usable, init=init, extra={"coupling": c, "tail": "pipeline"}))  # fmt: skip
        p = picks(path=ref[c])
        out["query"].append({"power_fcst_init_time": init, "arm": label, "coupling": c, "tail": "pipeline", "set": "N51", **p})  # fmt: skip
    seed0 = int(init.timestamp()) % (2**31)
    for ci, c in enumerate(COUPLINGS):
        u_sets = {}
        for si, set_name in enumerate(("A", "B")):
            rng = np.random.default_rng(seed0 + 17 * ci + 1000 * si)
            u_sets[set_name] = uniforms(coupling=c, chol=chol, phi=phi, history=history, rng=rng)
        for tail in TAILS:
            label = f"{c}-{tail}"
            extra = {"coupling": c, "tail": tail}
            drawn = {
                s: client.scenarios(q=q11, u=u_sets[s], tail=tail, bounds=BOUNDS)
                for s in ("A", "B")
            }
            path = drawn["A"]
            out["crps"].extend(crps_rows(label=label, path=path, actual=actual, init=init, extra={**extra, "k": K}))  # fmt: skip
            levels_all = LOWER + UPPER
            qa = np.quantile(path, levels_all, axis=0)
            quant = {lv: qa[i] for i, lv in enumerate(levels_all)}
            out["exceed"].extend(exceed_rows(label=label, quant=quant, actual=actual, usable=usable, init=init, extra=extra))  # fmt: skip
            for s in ("A", "B"):
                out["query"].append({"power_fcst_init_time": init, "arm": label, **extra, "set": s, **picks(path=drawn[s])})  # fmt: skip
            out["query"].append({"power_fcst_init_time": init, "arm": label, **extra, "set": "A51", **picks(path=path[:51])})  # fmt: skip
            u51 = stratify(u=u_sets["A"][:51])
            path51 = client.scenarios(q=q11, u=u51, tail=tail, bounds=BOUNDS).astype(np.float64)
            out["crps"].extend(crps_rows(label=label, path=path51, actual=actual, init=init, extra={**extra, "k": 51}))  # fmt: skip
            coh_paths[label] = path51
    for row in score.rows_for_run(paths=coh_paths, actual=actual, kind=kind, init=init):
        if row["block"] != "joint":
            out["coherence"].append(row)
    out["coherence"].extend(s625.joint_rows(paths=coh_paths, actual=actual, init=init))
    return out


def masked_actual(*, run: dict) -> np.ndarray:
    """Return the run's actual with every valid time on or after the hold-out set to NaN."""
    init = datetime.fromisoformat(str(run["init"]) + "+00:00").astimezone(UTC)
    valid = np.array([init + (i + 1) * build.STEP for i in range(S)])
    actual = run["actual"].astype(np.float64)
    actual[:, valid >= HOLDOUT] = np.nan
    return actual


def month_copulas(*, files: list[Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return experiment 436's held-out-month copula (Cholesky factor, AR(1) terms) per month."""
    runs = [score.load_run(path=f) for f in files]
    inits = [datetime.fromisoformat(f"{r['init']}+00:00").astimezone(UTC) for r in runs]
    z_all = [
        scen.pit_scores(values=r["values"].astype(np.float64), actual=masked_actual(run=r), seed=k)
        for k, r in enumerate(runs)
    ]
    result = {}
    for month in sorted({f"{t:%Y-%m}" for t in inits}):
        start = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
        train = [
            i for i, t in enumerate(inits)
            if f"{t:%Y-%m}" != month and not (start - timedelta(days=15) < t < start)
        ]  # fmt: skip
        result[month] = scen.fit_copula(z_runs=[z_all[i] for i in train])
    return result


def run_one(args: tuple[str, np.ndarray, np.ndarray]) -> dict[str, list[dict]]:
    """Return `process_run` for one file (a positional wrapper for `pool.map`)."""
    return process_run(file=args[0], chol=args[1], phi=args[2])


def main() -> None:
    """Score every stored run, in parallel worker processes, and write the four tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    files = sorted((score.WORK / "runs").glob("*.npz"))
    copulas = month_copulas(files=files)
    todo = files[: args.max_runs]
    jobs = []
    for f in todo:
        month = f.stem[:7]
        chol, phi = copulas[month]
        jobs.append((str(f), chol, phi))
    print(f"{len(jobs)} runs to score", flush=True)
    tables: dict[str, list[dict]] = {k: [] for k in ("crps", "exceed", "coherence", "query")}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for job, result in zip(jobs, pool.map(run_one, jobs), strict=True):
            for name, rows in result.items():
                tables[name].extend(rows)
            print(f"{Path(job[0]).stem} scored", flush=True)
    for name, rows in tables.items():
        pl.DataFrame(rows).write_parquet(OUT / f"{name}{args.suffix}.parquet")


if __name__ == "__main__":
    main()

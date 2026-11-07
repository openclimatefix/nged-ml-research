# ruff: noqa: E501
"""Experiment 627: choose K of the 51 coherent paths, and score how well the K cover the 51.

Reads experiment 436's `work/runs/<date>.npz` (the observed vector) and
`work/paths/<coupling>/<date>.npy` (51 coherent paths of 25 nodes and 702 half-hours, in points of
capacity). Nothing is written to either folder. Every arm is a weight vector over the 51 paths
(zero for a path not chosen), so every score is arithmetic on the 51 stored paths:

- `R`: K paths drawn at random without replacement, equal weights;
- `S-wind`, `S-solar`, `S-net`: the 51 paths ranked by one summary, cut into K equal-count strata,
  one path drawn from each stratum, equal weights;
- `C`: k-medoids on the 51 joint paths, each medoid weighted by its cluster size;
- `Oracle`: the best of 5,000 random K-subsets on the gap itself, a floor and not a usable rule;
- `All`: the 51 paths, equal weights.

Writes `selection.parquet` (coverage and tail rows per run), `joint.parquet` (energy and variogram
scores per run), and `strata.parquet` (the stratum of each member per run), and `summ.parquet` (the four summaries of
each of the 51 paths per run and band) to this experiment's
`work/` folder.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00627-*/env.sh
OMP_NUM_THREADS=1 nice -n 19 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00627-*/e00627_score.py --workers 4
```
"""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00627-scenario-subset-selection-from-51-coherent-paths"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
WORK436: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00436-joint-scenarios-across-demand-wind-solar-and-leads" / "work"
)
SEED: Final[int] = 627
N_MEMBERS: Final[int] = 51
N_STEPS: Final[int] = 702
STEP: Final[timedelta] = timedelta(minutes=30)
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
COUPLINGS: Final[tuple[str, ...]] = ("J3", "J2")
KS: Final[tuple[int, ...]] = (3, 5, 10)
N_DRAWS: Final[int] = 1000
N_SCORE_DRAWS: Final[int] = 200
"""Draws of R and S that are also scored against the observed vector."""
BANDS: Final[dict[str, tuple[float, float]]] = {
    "d1_10": (0.0, 240.0),
    "h12_36": (12.0, 36.0),
    "d1_3": (0.0, 72.0),
    "d4_10": (72.0, 240.0),
}
"""Lead bands in hours (open below, closed above). `d1_10` is the headline band and the selection window."""
JOINT_STEPS_PER_BAND: Final[int] = 24
CLUSTER_STRIDE: Final[int] = 4
KMEDOIDS_RESTARTS: Final[int] = 10
MIN_JOINT_DIMS: Final[int] = 100
SUMMARIES: Final[tuple[str, ...]] = ("wind", "solar", "net", "demand")
S_ARMS: Final[dict[str, str]] = {
    "S-wind": "wind",
    "S-solar": "solar",
    "S-net": "net",
    "S-demand": "demand",
}
TAIL_LOW: Final[float] = 0.10
TAIL_HIGH: Final[float] = 0.90
PERCENTILES: Final[tuple[float, ...]] = (0.05, 0.5, 0.95)


def band_steps(*, band: str) -> np.ndarray:
    """Return the step indices (0 = lead 0.5 h) whose lead lies in the band."""
    low, high = BANDS[band]
    lead = (np.arange(N_STEPS) + 1) * 0.5
    return np.flatnonzero((lead > low) & (lead <= high))


def summaries(*, paths: np.ndarray, kind: np.ndarray, steps: np.ndarray) -> dict[str, np.ndarray]:
    """Return four summaries (51,) of each path over the steps, in points of capacity.

    `wind` and `solar` are the mean over the farms of that kind and over the steps. `net` is the
    peak over the steps of the Primaries' mean minus the wind-farm mean minus the solar-farm mean,
    an unweighted proxy for net demand because MW weights are not available. `demand` is the peak
    over the steps of the Primaries' mean.
    """
    p = paths[:, :, steps].astype(np.float64)
    demand, wind, solar = (p[:, kind == k].mean(axis=1) for k in (0, 1, 2))
    return {
        "wind": wind.mean(axis=1),
        "solar": solar.mean(axis=1),
        "net": (demand - wind - solar).max(axis=1),
        "demand": demand.max(axis=1),
    }


def lower_quantile_index(*, cdf: np.ndarray, q: float) -> np.ndarray:
    """Return, per row, the index of the first atom whose cumulative weight reaches q."""
    return (cdf >= q - 1e-9).argmax(axis=-1)


def distribution_rows(*, weights: np.ndarray, summary: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-draw coverage and tail measures of weighted atoms against the 51 equal atoms.

    `weights` is (D, 51) and `summary` is (51,). Gaps and the Wasserstein-1 distance are divided by
    the spread (95th minus 5th percentile, lower inverse CDF) of the 51 values.
    """
    order = np.argsort(summary, kind="stable")
    z = summary[order]
    cdf = np.cumsum(weights[:, order], axis=1)
    ref = np.arange(1, N_MEMBERS + 1) / N_MEMBERS
    spread = z[int(np.ceil(0.95 * N_MEMBERS)) - 1] - z[int(np.ceil(0.05 * N_MEMBERS)) - 1]
    spread = spread if spread > 1e-12 else np.nan
    w1 = (np.abs(cdf[:, :-1] - ref[:-1]) * np.diff(z)).sum(axis=1) / spread
    out = {"w1": w1}
    for q in PERCENTILES:
        gap = z[lower_quantile_index(cdf=cdf, q=q)] - z[int(np.ceil(q * N_MEMBERS)) - 1]
        out[f"gap{round(q * 100):02d}"] = np.abs(gap) / spread
        out[f"sgap{round(q * 100):02d}"] = gap / spread
    low_cut = z[int(np.ceil(TAIL_LOW * N_MEMBERS)) - 1]
    high_cut = z[int(np.ceil(TAIL_HIGH * N_MEMBERS)) - 1]
    out["tail_low"] = ((weights > 0) & (summary <= low_cut)).any(axis=1).astype(float)
    out["tail_high"] = ((weights > 0) & (summary >= high_cut)).any(axis=1).astype(float)
    return out


def random_weights(*, k: int, rng: np.random.Generator, n: int) -> np.ndarray:
    """Return (n, 51) weights of n simple random samples of K paths, without replacement."""
    pick = np.argsort(rng.random((n, N_MEMBERS)), axis=1)[:, :k]
    w = np.zeros((n, N_MEMBERS))
    np.put_along_axis(w, pick, 1.0 / k, axis=1)
    return w


def strata_of(*, summary: np.ndarray, k: int) -> np.ndarray:
    """Return the stratum (0 to K-1) of each of the 51 paths: equal-count cuts of the ranked summary."""
    order = np.argsort(summary, kind="stable")
    label = np.empty(N_MEMBERS, dtype=np.int64)
    for s, chunk in enumerate(np.array_split(order, k)):
        label[chunk] = s
    return label


def stratified_weights(*, summary: np.ndarray, k: int, rng: np.random.Generator, n: int) -> np.ndarray:
    """Return (n, 51) weights of n stratified samples: one path from each of K equal-count strata."""
    label = strata_of(summary=summary, k=k)
    w = np.zeros((n, N_MEMBERS))
    for s in range(k):
        members = np.flatnonzero(label == s)
        pick = members[rng.integers(0, len(members), size=n)]
        w[np.arange(n), pick] += 1.0 / k
    return w


def cluster_features(*, paths: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Return (51, d) features: every 4th half-hour, centred across the 51 paths, one scale per node."""
    x = paths[:, :, steps[::CLUSTER_STRIDE]].astype(np.float64)
    scale = x.std(axis=(0, 2), keepdims=True)
    z = np.where(scale > 1e-9, (x - x.mean(axis=0, keepdims=True)) / np.maximum(scale, 1e-9), 0.0)
    return z.reshape(N_MEMBERS, -1)


def k_medoids(*, dist: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (medoid indices, cluster labels) of the cheapest of several k-medoids++ starts."""
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for _ in range(KMEDOIDS_RESTARTS):
        medoids = [int(rng.integers(0, N_MEMBERS))]
        while len(medoids) < k:
            d2 = dist[:, medoids].min(axis=1) ** 2
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(N_MEMBERS, 1.0 / N_MEMBERS)
            medoids.append(int(rng.choice(N_MEMBERS, p=p)))
        m = np.array(medoids)
        for _ in range(100):
            label = dist[:, m].argmin(axis=1)
            new = m.copy()
            for c in range(k):
                members = np.flatnonzero(label == c)
                if len(members):
                    new[c] = members[dist[np.ix_(members, members)].sum(axis=1).argmin()]
            if np.array_equal(new, m):
                break
            m = new
        label = dist[:, m].argmin(axis=1)
        cost = float(dist[np.arange(N_MEMBERS), m[label]].sum())
        if best is None or cost < best[0]:
            best = (cost, m, label)
    assert best is not None  # noqa: S101
    return best[1], best[2]


def medoid_weights(*, paths: np.ndarray, steps: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return (1, 51) weights: each k-medoids medoid weighted by its cluster size over 51."""
    f = cluster_features(paths=paths, steps=steps)
    sq = (f**2).sum(axis=1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * f @ f.T, 0.0))
    medoids, label = k_medoids(dist=dist, k=k, rng=rng)
    w = np.zeros((1, N_MEMBERS))
    for c, m in enumerate(medoids):
        w[0, m] += float((label == c).sum()) / N_MEMBERS
    return w


def arm_weights(  # noqa: PLR0913
    *, paths: np.ndarray, kind: np.ndarray, select_band: str, k: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Return {arm: (D, 51) weights} for R, the three S arms, and C, selecting on `select_band`."""
    steps = band_steps(band=select_band)
    summ = summaries(paths=paths, kind=kind, steps=steps)
    out = {"R": random_weights(k=k, rng=rng, n=N_DRAWS)}
    for arm, name in S_ARMS.items():
        out[arm] = stratified_weights(summary=summ[name], k=k, rng=rng, n=N_DRAWS)
    out["C"] = medoid_weights(paths=paths, steps=steps, k=k, rng=np.random.default_rng([SEED, k, 3]))
    out["Oracle"] = oracle_weights(summ=summ, k=k, rng=rng)
    return out


def oracle_weights(*, summ: dict[str, np.ndarray], k: int, rng: np.random.Generator) -> np.ndarray:
    """Return (1, 51) weights of the best of 5,000 random K-subsets on the selection window.

    The cost is the mean over the four summaries and the 5th, 50th, and 95th percentiles of the
    absolute relative gap. It uses the 51 paths themselves, so it is a floor on the gap that no
    selection rule can beat by much, not an arm anyone could run.
    """
    w = random_weights(k=k, rng=rng, n=5000)
    cost = np.zeros(5000)
    for v in summ.values():
        m = distribution_rows(weights=w, summary=v)
        cost += np.nan_to_num(sum(m[f"gap{round(q * 100):02d}"] for q in PERCENTILES), nan=0.0)
    return w[[int(np.nanargmin(cost))]]


class JointBlock:
    """Precomputed distances from the 51 paths to the observed vector, for one lead band."""

    def __init__(self, *, paths: np.ndarray, actual: np.ndarray, band: str, rng: np.random.Generator) -> None:
        steps = band_steps(band=band)
        stride = max(1, len(steps) // JOINT_STEPS_PER_BAND)
        sub = steps[::stride]
        x = paths[:, :, sub].reshape(N_MEMBERS, -1).astype(np.float64)
        y = actual[:, sub].reshape(-1).astype(np.float64)
        ok = ~np.isnan(y)
        self.valid = int(ok.sum()) >= MIN_JOINT_DIMS
        if not self.valid:
            return
        x, y = x[:, ok], y[ok]
        n = x.shape[1]
        self.dy = np.sqrt(((x - y) ** 2).sum(axis=1) / n)
        sq = (x**2).sum(axis=1)
        self.dxx = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0) / n)
        ia, ib = np.triu_indices(n, k=1)
        if len(ia) > 60_000:
            pick = np.sort(rng.choice(len(ia), size=60_000, replace=False))
            ia, ib = ia[pick], ib[pick]
        self.vo = np.abs(y[ia] - y[ib]) ** 0.5
        self.vx = np.abs(x[:, ia] - x[:, ib]) ** 0.5
        self.npairs = len(ia)

    def scores(self, *, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the weighted energy score and variogram score (order 0.5) per weight row."""
        es = weights @ self.dy - 0.5 * np.einsum("di,ij,dj->d", weights, self.dxx, weights)
        m = weights @ self.vx
        vs = ((self.vo[None] - m) ** 2).sum(axis=1) / self.npairs
        return es, vs


def process_run(*, file: str) -> dict[str, list[dict]]:  # noqa: PLR0912, PLR0915
    """Score every coupling, selection window, arm, and K on one run; return the table rows."""
    z = np.load(file)
    init = datetime.fromisoformat(str(z["init"]) + "+00:00").astimezone(UTC)
    day_tag = f"{init:%Y-%m-%d}"
    kind = z["kind"]
    actual = z["actual"].astype(np.float64)
    valid_time = np.array([init + (i + 1) * STEP for i in range(N_STEPS)])
    actual[:, valid_time >= HOLDOUT] = np.nan  # the holdout is never scored
    out: dict[str, list[dict]] = {"selection": [], "joint": [], "strata": [], "summ": []}
    run_index = int(init.toordinal())
    for coupling in COUPLINGS:
        paths = np.load(WORK436 / "paths" / coupling / f"{day_tag}.npy").astype(np.float64)
        summ_by_band = {b: summaries(paths=paths, kind=kind, steps=band_steps(band=b)) for b in BANDS}
        jb_rng = np.random.default_rng([SEED, run_index, 0])
        joint = {b: JointBlock(paths=paths, actual=actual, band=b, rng=jb_rng) for b in BANDS}
        base = {"power_fcst_init_time": init, "coupling": coupling}
        for b, sm in summ_by_band.items():
            for name, v in sm.items():
                out["summ"].append({**base, "band": b, "summary": name, "values": v.tolist()})
        for k in KS:
            sel_summ = summaries(paths=paths, kind=kind, steps=band_steps(band="d1_10"))
            for name, s in sel_summ.items():
                out["strata"].append(
                    {**base, "k": k, "summary": name,
                     "stratum": strata_of(summary=s, k=k).tolist()}
                )  # fmt: skip
        for select_band in ("d1_10", "band"):
            for k in KS:
                rng = np.random.default_rng([SEED, run_index, k, 1 if select_band == "d1_10" else 2])
                if select_band == "d1_10":
                    once = arm_weights(paths=paths, kind=kind, select_band="d1_10", k=k, rng=rng)
                    chosen = dict.fromkeys(BANDS, once)  # one selection, reused for every band
                else:
                    chosen = {
                        b: arm_weights(paths=paths, kind=kind, select_band=b, k=k, rng=rng)
                        for b in BANDS
                    }
                for band in BANDS:
                    arms = dict(chosen[band])
                    for arm, w in arms.items():
                        for sname in SUMMARIES:
                            m = distribution_rows(weights=w, summary=summ_by_band[band][sname])
                            out["selection"].append(
                                {**base, "selection": select_band, "arm": arm, "k": k, "band": band,
                                 "summary": sname, **{key: float(np.nanmean(v)) for key, v in m.items()},
                                 **{f"{key}_sd": float(np.nanstd(v)) for key, v in m.items()},
                                 **{f"{key}_p05": float(np.nanquantile(v, 0.05)) for key, v in m.items()},
                                 **{f"{key}_p95": float(np.nanquantile(v, 0.95)) for key, v in m.items()},
                                 "degenerate": bool(np.isnan(m["w1"]).all()),
                                 "draws": int(w.shape[0])}
                            )  # fmt: skip
                        jb = joint[band]
                        if jb.valid:
                            es, vs = jb.scores(weights=w[:N_SCORE_DRAWS])
                            out["joint"].append(
                                {**base, "selection": select_band, "arm": arm, "k": k, "band": band,
                                 "energy": float(es.mean()), "variogram": float(vs.mean())}
                            )  # fmt: skip
        # the 51-path reference, once per band
        ref = np.full((1, N_MEMBERS), 1.0 / N_MEMBERS)
        for band in BANDS:
            jb = joint[band]
            if jb.valid:
                es, vs = jb.scores(weights=ref)
                out["joint"].append(
                    {**base, "selection": "d1_10", "arm": "All", "k": N_MEMBERS, "band": band,
                     "energy": float(es[0]), "variogram": float(vs[0])}
                )  # fmt: skip
    return out


def process_run_kw(file: str) -> dict[str, list[dict]]:
    """Return `process_run` for one file (a positional wrapper for `pool.map`)."""
    return process_run(file=file)


def main() -> None:
    """Score every stored run in parallel worker processes and write the three tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = [str(f) for f in sorted((WORK436 / "runs").glob("*.npz"))[: args.max_runs]]
    print(f"{len(files)} runs to score", flush=True)
    tables: dict[str, list[dict]] = {"selection": [], "joint": [], "strata": [], "summ": []}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for file, result in zip(files, pool.map(process_run_kw, files), strict=True):
            for name, rows in result.items():
                tables[name].extend(rows)
            print(f"{Path(file).stem} scored", flush=True)
    for name, rows in tables.items():
        pl.DataFrame(rows).write_parquet(OUT / f"{name}{args.suffix}.parquet")


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Tier 1 of experiment 436: the error-copula arm J5 and the member-wise arms J6 and J6-median.

Fits a Gaussian copula on the probability-integral-transform (PIT) residuals of the stored
marginals, holding out one calendar month at a time (nested: the correlation, the AR(1)
coefficients, and the mixing weights are all chosen from the other months' runs), and writes the
scenarios of each arm to `work/paths/<arm>/<date>.npy` as float32 (51, 25, 702).

- The PIT of an actual under its 51 values is the linear interpolation of the actual's position among
  the values at levels k/52, clipped to [0.5/52, 1 - 0.5/52]; `z` is its standard-normal score.
- The copula is separable: the 25-node correlation `R` of `z` at the same half-hour (pooled over the
  training runs, made positive definite), and a per-node AR(1) coefficient `phi` from the lag-1
  autocorrelation of `z`. A latent path `c` of one scenario starts from `L eps` and follows
  `c_t = phi c_{t-1} + sqrt(1 - phi^2) L eps_t`, so each node has unit variance at every half-hour.
- `J5`: scenario m takes, at each node and half-hour, the rank of `sqrt(w) g + sqrt(1 - w) c` among
  the 51 scenarios, where `g` is the normal score of the member's ECC rank (the `J2` ordering) and
  `w` is chosen per family and lead band (12 to 36 hours, 36 to 240 hours) from {0, 0.25, 0.5, 0.75, 1}
  by the within-family spatial plus temporal variogram score on the training runs.
- `J6`: for wind and solar, scenario k is ensemble member k's own conditional quantile function at
  the level `Phi(c_k)`; demand follows `J2`. `J6-median` takes each member's median instead.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00436-*/env.sh
nice -n 19 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00436-*/e00436_scenarios.py
```
"""

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Final

import e00434_score as s434  # ty: ignore[unresolved-import]
import e00436_build as build  # ty: ignore[unresolved-import]
import e00436_members as members  # ty: ignore[unresolved-import]
import e00436_score as score  # ty: ignore[unresolved-import]
import numpy as np
from scipy.stats import norm

N: Final[int] = build.N_NODES
S: Final[int] = build.N_STEPS
M: Final[int] = build.N_MEMBERS
WEIGHTS: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 1.0)
BANDS: Final[dict[str, tuple[int, int]]] = {"h12_36": (23, 72), "h36_240": (72, 480)}
"""Step ranges (0 = lead 0.5 h) of the two lead bands the mixing weight is chosen for."""
LEVELS11: Final[np.ndarray] = s434.QUANTILE_LEVELS
KIND: Final[np.ndarray] = build.KIND
MIN_PAIR_STEPS: Final[int] = 200
FIT_STRIDE: Final[int] = 4
"""Every fourth step is used to fit `R` and `phi`, because neighbouring steps are nearly copies."""


def pit_scores(*, values: np.ndarray, actual: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return the normal score of each actual under its 51 values (25, 702); NaN where no actual.

    A tie between the actual and several of the 51 values draws the PIT at random within the tied levels.
    """
    rng = np.random.default_rng(seed)
    levels = np.arange(1, M + 1) / (M + 1)
    z = np.full(actual.shape, np.nan)
    for node in range(N):
        for step in np.flatnonzero(~np.isnan(actual[node])):
            row = values[node, step]
            lo = int(np.searchsorted(row, actual[node, step], side="left"))
            hi = int(np.searchsorted(row, actual[node, step], side="right"))
            if hi - lo >= 2:  # a tie among the values: randomise within the tied levels
                u = rng.uniform(levels[lo], levels[hi - 1])
            else:
                u = np.interp(actual[node, step], row, levels)
            z[node, step] = norm.ppf(np.clip(u, 0.5 / (M + 1), 1 - 0.5 / (M + 1)))
    return z


def fit_copula(*, z_runs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the Cholesky factor of `R` (25, 25) and `phi` (25,) from the training runs' scores."""
    z = np.concatenate([zr[:, ::FIT_STRIDE] for zr in z_runs], axis=1)
    corr = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            ok = ~np.isnan(z[i]) & ~np.isnan(z[j])
            if ok.sum() >= MIN_PAIR_STEPS:
                corr[i, j] = corr[j, i] = np.corrcoef(z[i, ok], z[j, ok])[0, 1]
    vals, vecs = np.linalg.eigh(corr)
    corr = (vecs * np.clip(vals, 1e-3, None)) @ vecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / d[:, None] / d[None, :]
    phi = np.zeros(N)
    for node in range(N):
        pairs = []
        for zr in z_runs:
            a, b = zr[node, :-1], zr[node, 1:]
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() > 20:
                pairs.append(np.stack([a[ok], b[ok]]))
        if pairs:
            both = np.concatenate(pairs, axis=1)
            phi[node] = np.clip(np.corrcoef(both[0], both[1])[0, 1], 0.0, 0.995)
    return np.linalg.cholesky(corr), phi


def latent_paths(*, chol: np.ndarray, phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return the copula latent paths (51, 25, 702): AR(1) in the half-hour, correlated across nodes."""
    eps = rng.standard_normal((M, N, S))
    innov = np.einsum("ij,mjs->mis", chol, eps) * np.sqrt(1 - phi**2)[None, :, None]
    c = np.empty((M, N, S))
    c[:, :, 0] = np.einsum("ij,mj->mi", chol, eps[:, :, 0])
    for t in range(1, S):
        c[:, :, t] = phi[None, :] * c[:, :, t - 1] + innov[:, :, t]
    return c


def ranks_from_latent(*, latent: np.ndarray) -> np.ndarray:
    """Return the rank (0 to 50) of each scenario among the 51 along axis 0 of (51, 25, 702)."""
    return np.argsort(np.argsort(latent, axis=0, kind="stable"), axis=0)


def j5_paths(
    *, values: np.ndarray, ecc: np.ndarray, c: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Return J5 scenarios (51, 25, 702).

    Args:
        values: The 51 sorted values, (25, 702, 51).
        ecc: Each member's ECC rank, (25, 702, 51), the member index being the last axis.
        c: Copula latent paths, (51, 25, 702).
        weights: Mixing weight `w` per node and step, (25, 702).

    Returns:
        The 51 scenarios, (51, 25, 702).
    """
    g = norm.ppf((ecc.astype(np.float64) + 0.5) / M).transpose(2, 0, 1)
    w = weights[None]
    latent = np.sqrt(w) * g + np.sqrt(1 - w) * c
    ranks = ranks_from_latent(latent=latent)  # (51, 25, 702): rank of scenario m
    return np.take_along_axis(values.transpose(2, 0, 1), ranks, axis=0)


def weight_grid(*, table: dict[tuple[int, str], float]) -> np.ndarray:
    """Return `w` per node and step (25, 702) from the table of (family, band) weights."""
    out = np.ones((N, S))
    for (fam, band), w in table.items():
        lo, hi = BANDS[band]
        lo = 0 if band == "h12_36" else lo  # leads under 12 h take the 12 to 36 hour weight
        hi = S if band == "h36_240" else hi  # leads past 240 h take the 36 to 240 hour weight
        out[fam == KIND, lo:hi] = w
    return out


def train_score(*, path: np.ndarray, actual: np.ndarray, fam: int, band: str) -> float:
    """Return the family's spatial plus temporal variogram score for one band (lower is better)."""
    lo, hi = BANDS[band]
    nodes = np.flatnonzero(fam == KIND)
    have = ~np.isnan(actual)
    y_all = np.nan_to_num(actual)
    steps = np.arange(lo, hi)
    total, count = 0.0, 0
    keep = have[nodes][:, steps].sum(0) >= score.MIN_SPATIAL[fam]
    if keep.any():
        cols = steps[keep]
        x = path[:, nodes][:, :, cols].transpose(2, 0, 1)
        _, vs = s434.score_blocks(
            x=x, y=y_all[nodes][:, cols].T, mask=have[nodes][:, cols].T, chunk=60
        )
        total += vs.mean()
        count += 1
    sub = steps[::4]
    tmask = have[nodes][:, sub]
    good = tmask.sum(1) >= 4
    if good.any():
        x = path[:, nodes][:, :, sub].transpose(1, 0, 2)[good]
        _, vs = s434.score_blocks(x=x, y=y_all[nodes][:, sub][good], mask=tmask[good], chunk=4)
        total += vs.mean()
        count += 1
    return total / max(count, 1)


def choose_weights(
    *, runs: list[dict], z_unused: None, chol: np.ndarray, phi: np.ndarray, seed: int
) -> dict[tuple[int, str], float]:
    """Return the (family, band) mixing weights that minimise the training runs' variogram score."""
    sums = {(fam, band, w): 0.0 for fam in range(3) for band in BANDS for w in WEIGHTS}
    for k, run in enumerate(runs):
        rng = np.random.default_rng(seed + k)
        c = latent_paths(chol=chol, phi=phi, rng=rng)
        values = run["values"].astype(np.float64)
        actual = run["actual"].astype(np.float64)
        for w in WEIGHTS:
            path = j5_paths(values=values, ecc=run["ecc"], c=c, weights=np.full((N, S), w))
            for fam in range(3):
                for band in BANDS:
                    sums[(fam, band, w)] += train_score(
                        path=path, actual=actual, fam=fam, band=band
                    )
    if not all(np.isfinite(v) for v in sums.values()):
        msg = "a training score is not finite"
        raise ValueError(msg)
    return {
        (fam, band): min(WEIGHTS, key=lambda w, f=fam, b=band: sums[(f, b, w)])
        for fam in range(3)
        for band in BANDS
    }


def quantile_at(*, quantiles: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Return piecewise-linear quantile functions (..., 11) at `level` (...), clamped to q01 to q99."""
    idx = np.clip(np.searchsorted(LEVELS11, level) - 1, 0, len(LEVELS11) - 2)
    low, high = LEVELS11[idx], LEVELS11[idx + 1]
    frac = np.clip((level - low) / (high - low), 0.0, 1.0)
    lo = np.take_along_axis(quantiles, idx[..., None], axis=-1)[..., 0]
    hi = np.take_along_axis(quantiles, (idx + 1)[..., None], axis=-1)[..., 0]
    return lo * (1 - frac) + hi * frac


def j6_paths(*, j2: np.ndarray, member_q: np.ndarray, c: np.ndarray, median: bool) -> np.ndarray:
    """Return J6 (or J6-median) scenarios (51, 25, 702): J2 for demand, member-wise for the farms.

    `member_q` is (9, 702, 51, 11): scenario k is member k's quantile at `Phi(c_k)`.
    """
    out = j2.copy()
    n_farm = member_q.shape[0]
    first = build.N_PRIMARIES
    if median:
        out[:, first:] = member_q[:, :, :, 5].transpose(2, 0, 1)
        return out
    level = norm.cdf(c[:, first:, :]).transpose(1, 2, 0)  # (9, 702, 51)
    out[:, first : first + n_farm] = quantile_at(quantiles=member_q, level=level).transpose(2, 0, 1)
    return out


def main() -> None:
    """Fit the copula leaving one month out and write J5, J6, and J6-median scenarios."""
    observed = np.load(score.WORK / "observed.npz")["observed"]  # noqa: F841
    files = sorted((score.WORK / "runs").glob("*.npz"))
    runs = [score.load_run(path=f) for f in files]
    inits = [datetime.fromisoformat(f"{r['init']}+00:00").astimezone(UTC) for r in runs]
    months = sorted({f"{t:%Y-%m}" for t in inits})
    z_all = [
        pit_scores(
            values=r["values"].astype(np.float64), actual=r["actual"].astype(np.float64), seed=k
        )
        for k, r in enumerate(runs)
    ]
    table_lines = ["month  w[demand h12_36, h36_240]  [wind ...]  [solar ...]"]
    for month in months:
        start = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
        train = [
            i
            for i, t in enumerate(inits)
            if f"{t:%Y-%m}" != month and not (start - timedelta(days=15) < t < start)
        ]  # a run that ends inside the held-out month is not training data
        test = [i for i, t in enumerate(inits) if f"{t:%Y-%m}" == month]
        if all(
            score.scenario_path(arm=arm, init=inits[i]).exists()
            for i in test
            for arm in ("J5", "J6", "J6-median")
        ):
            continue
        chol, phi = fit_copula(z_runs=[z_all[i] for i in train])
        seed = int(hashlib.sha256(month.encode()).hexdigest()[:6], 16)
        table = choose_weights(
            runs=[runs[i] for i in train], z_unused=None, chol=chol, phi=phi, seed=seed
        )
        table_lines.append(
            f"{month}  " + "  ".join(str([table[(fam, b)] for b in BANDS]) for fam in range(3))
        )
        print(table_lines[-1], flush=True)
        with (score.WORK / "mixing_weights.txt").open("a") as handle:
            handle.write(table_lines[-1] + "\n")
        weights = weight_grid(table=table)
        for i in test:
            run, init = runs[i], inits[i]
            rng = np.random.default_rng(int(init.timestamp()) % (2**32) + 7)
            c = latent_paths(chol=chol, phi=phi, rng=rng)
            values = run["values"].astype(np.float64)
            member_q = np.load(members.MEMBER_DIR / files[i].name)["member_q"].astype(np.float64)
            j2 = s434.assign(values=values, ranks=run["ecc"].astype(np.int64))
            outputs = {
                "J5": j5_paths(values=values, ecc=run["ecc"], c=c, weights=weights),
                "J6": j6_paths(j2=j2, member_q=member_q, c=c, median=False),
                "J6-median": j6_paths(j2=j2, member_q=member_q, c=c, median=True),
            }
            for arm, path in outputs.items():
                file = score.scenario_path(arm=arm, init=init)
                file.parent.mkdir(parents=True, exist_ok=True)
                np.save(file, path.astype(np.float32))
            print(f"{init:%Y-%m-%d} written", flush=True)


if __name__ == "__main__":
    main()

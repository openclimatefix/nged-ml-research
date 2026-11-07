# ruff: noqa: E501, PLR0912, PLR0915, RUF005
"""Build the BSP and GSP sums of Primary scenarios for experiment 629.

For each demand network and each daily run of the development months, writes
`work/runs/<network>/<date>.npz` with, for the four groups (BSP D17, BSP D13, GSP D6, GSP D1):
`sums` (4 arms J0, J2, J3, J5; 4 groups; 51; 480) of scenario sums in MW, `median_sum` (4, 480),
`meter` (4, 480) the cleaned group meter, `prim_obs_sum` (4, 480) the Primaries' observed sum
(NaN where any Primary beneath the group is absent), `direct` (5 sources, 4, 480, 11).

Arms (all re-order the same 51 values per Primary and half-hour, which are the stored 11 quantiles
interpolated linearly at the levels k/52, as in experiment 434):

- J0: independent random order for each Primary and half-hour.
- J2: ensemble copula coupling by member 2 m temperature (shortwave in daylight), as experiment 436.
- J3: Schaake shuffle with the ranks of the 51 observed 480-step paths nearest in day of year.
- J5: Gaussian copula on probability-integral-transform scores, AR(1) in the half-hour, fitted
  leaving the test month out (no ensemble mixing: the weight is 0, so no ECC ranks enter).

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00629-*/env.sh
nice -n 19 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00629-*/e00629_build.py
```
"""

import hashlib
import os
from datetime import UTC, datetime, timedelta

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00434_score as s434  # noqa: E402  # ty: ignore[unresolved-import]
import e00629_common as c  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from studies.power import scan_power  # noqa: E402

M = c.N_MEMBERS
S = c.N_STEPS
FIT_STRIDE = 4


def observed_history(*, ids: list[int]) -> tuple[np.ndarray, datetime]:
    """Return cleaned power (len(ids), n) in MW on a half-hourly grid from 2025-03-01 to 2026-07-01."""
    start = c.SCHAAKE_FIRST
    n = int((c.END_OF_DEV - start) / c.STEP)
    out = np.full((len(ids), n), np.nan)
    df = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(ids)
            & (pl.col("time") >= start)
            & (pl.col("time") < c.END_OF_DEV)
        )
        .collect()
    )
    for p, sid in enumerate(ids):
        sub = df.filter(pl.col("time_series_id") == sid)
        idx = ((sub["time"] - start).dt.total_minutes() // 30).to_numpy()
        ok = (idx >= 0) & (idx < n)
        out[p, idx[ok]] = sub["power"].to_numpy()[ok]
    return out, start


def schaake_ranks(*, init: datetime, grid: np.ndarray, start: datetime) -> tuple[np.ndarray, int]:
    """Return Schaake ranks (P, 480, 51) and the number of qualifying dates for a run at `init`."""
    last = init - c.SCHAAKE_BUFFER
    t0 = c.SCHAAKE_FIRST.replace(hour=init.hour, minute=init.minute)
    cand = []
    while t0 <= last:
        i = int((t0 - start) / c.STEP) + 1
        path = grid[:, i : i + S]
        if path.shape[1] == S and np.isnan(path).mean(axis=1).max() <= 0.05:
            doy = abs((t0.timetuple().tm_yday - init.timetuple().tm_yday + 183) % 365 - 183)
            cand.append((doy, t0, path))
        t0 += timedelta(days=1)
    if not cand:
        msg = f"{init}: no Schaake date qualifies"
        raise ValueError(msg)
    cand.sort(key=lambda x: (x[0], x[1]))
    x = np.arange(S)
    paths = []
    for k in range(M):
        out = cand[k % len(cand)][2].copy()
        for p in range(out.shape[0]):
            bad = np.isnan(out[p])
            if bad.any():
                out[p, bad] = np.interp(x[bad], x[~bad], out[p, ~bad])
        paths.append(out)
    rng = np.random.default_rng(int(hashlib.sha256(init.isoformat().encode()).hexdigest()[:8], 16))
    key = np.stack(paths).transpose(1, 2, 0)
    return s434.ranks_by_key(key=key, rng=rng), len(cand)


def pit_scores(*, values: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the standard-normal score of each actual under its 51 sorted values (P, 480); NaN if absent."""
    from scipy.stats import norm

    k = (values < actual[..., None]).sum(-1)
    kc = np.clip(k, 1, M - 1)
    lo = np.take_along_axis(values, (kc - 1)[..., None], -1)[..., 0]
    hi = np.take_along_axis(values, kc[..., None], -1)[..., 0]
    frac = np.where(hi > lo, np.clip((actual - lo) / np.where(hi > lo, hi - lo, 1.0), 0, 1), 0.5)
    u = np.where(k == 0, 0.5 / (M + 1), np.where(k >= M, 1 - 0.5 / (M + 1), (kc + frac) / (M + 1)))
    z = norm.ppf(np.clip(u, 0.5 / (M + 1), 1 - 0.5 / (M + 1)))
    z[np.isnan(actual) | np.isnan(values).any(-1)] = np.nan
    return z


def fit_copula(*, z_runs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the Cholesky factor of the Primaries' score correlation and the AR(1) coefficients."""
    p = z_runs[0].shape[0]
    z = np.concatenate([zr[:, ::FIT_STRIDE] for zr in z_runs], axis=1)
    corr = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            ok = ~np.isnan(z[i]) & ~np.isnan(z[j])
            if ok.sum() >= 200:
                corr[i, j] = corr[j, i] = np.corrcoef(z[i, ok], z[j, ok])[0, 1]
    vals, vecs = np.linalg.eigh(corr)
    corr = (vecs * np.clip(vals, 1e-3, None)) @ vecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / d[:, None] / d[None, :]
    phi = np.zeros(p)
    for node in range(p):
        pairs = []
        for zr in z_runs:
            a, b = zr[node, :-1], zr[node, 1:]
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() > 20:
                pairs.append(np.stack([a[ok], b[ok]]))
        both = np.concatenate(pairs, axis=1)
        phi[node] = np.clip(np.corrcoef(both[0], both[1])[0, 1], 0.0, 0.995)
    return np.linalg.cholesky(corr), phi


def latent_ranks(*, chol: np.ndarray, phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return J5 ranks (P, 480, 51): the order of 51 latent AR(1) copula paths at each Primary and half-hour."""
    p = chol.shape[0]
    eps = rng.standard_normal((M, p, S))
    innov = np.einsum("ij,mjs->mis", chol, eps) * np.sqrt(1 - phi**2)[None, :, None]
    lat = np.empty((M, p, S))
    lat[:, :, 0] = np.einsum("ij,mj->mi", chol, eps[:, :, 0])
    for t in range(1, S):
        lat[:, :, t] = phi[None, :] * lat[:, :, t - 1] + innov[:, :, t]
    rank = np.argsort(np.argsort(lat, axis=0, kind="stable"), axis=0)  # (51, P, S): rank of path m
    return rank.transpose(1, 2, 0)


def main() -> None:
    """Write one run table per network and daily run."""
    prim = c.primaries()
    ids = prim["time_series_id"].to_list()
    nodes = c.group_nodes(prim=prim)
    print({g: len(v) for g, v in nodes.items()}, flush=True)
    grid, start = observed_history(ids=ids)
    direct = {key: {m: c.read_direct(key=key, month=m) for m in c.MONTHS} for key in c.DIRECT}
    for network in c.NETWORKS:
        out_dir = c.WORK / "runs" / network
        out_dir.mkdir(parents=True, exist_ok=True)
        qcols = list(c.QUANTILES)
        runs: dict[datetime, dict] = {}
        for month in c.MONTHS:
            frame = c.read_network(network=network, month=month)
            q = c.run_arrays(frame=frame, ids=ids, columns=qcols)
            o = c.run_arrays(frame=frame, ids=ids, columns=["observed_power"])
            for init in q:
                runs[init] = {
                    "q": np.sort(q[init], axis=-1),
                    "obs": o[init][..., 0],
                    "month": month,
                }
        inits = sorted(runs)
        print(network, len(inits), "runs", flush=True)
        # probability-integral-transform scores of every run
        for init in inits:
            r = runs[init]
            r["values"] = s434.fifty_one_values(quantiles=r["q"])
            r["z"] = pit_scores(values=r["values"], actual=r["obs"])
        copulas = {}
        step_idx = np.arange(S)
        for month in c.MONTHS:
            start_m = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
            end_m = (start_m + timedelta(days=32)).replace(day=1)
            train = []
            for t in inits:
                if runs[t]["month"] == month or (start_m - timedelta(days=15) < t < start_m):
                    continue
                z = runs[t]["z"].copy()
                vt = np.array([t + (i + 1) * c.STEP for i in step_idx])  # ty: ignore[unsupported-operator]
                # never read the final-test window or the test month's own observations
                z[:, (vt >= c.END_OF_DEV) | ((vt >= start_m) & (vt < end_m))] = np.nan
                train.append(z)
            copulas[month] = fit_copula(z_runs=train)
            print(network, month, "copula fitted on", len(train), "runs", flush=True)
        for init in inits:
            out = out_dir / f"{init:%Y-%m-%d}.npz"
            if out.exists():
                continue
            r = runs[init]
            rng = np.random.default_rng(int(init.timestamp()) % (2**32))
            key_rng = np.random.default_rng(int(init.timestamp()) % (2**32) + 1)
            t_mem, sw_mem = s434.run_weather(init=init, ids=ids)
            day = np.median(sw_mem, axis=-1, keepdims=True) > 20.0
            key = np.where(day, -sw_mem, -t_mem)[:, :S]
            ecc = s434.ranks_by_key(key=key, rng=key_rng)
            sch, _ = schaake_ranks(init=init, grid=grid, start=start)
            ind = rng.permuted(np.broadcast_to(np.arange(M), (len(ids), S, M)), axis=-1)
            chol, phi = copulas[r["month"]]
            j5 = latent_ranks(chol=chol, phi=phi, rng=rng)
            values = r["values"]
            sums = np.full((4, 4, M, S), np.nan)
            for a, ranks in enumerate((ind, ecc, sch, j5)):
                path = s434.assign(values=np.nan_to_num(values), ranks=ranks.astype(np.int64))
                for gi, g in enumerate(c.GROUP_IDS):
                    s = path[:, nodes[g]].sum(1)
                    bad = np.isnan(values[nodes[g]]).any(axis=(0, 2))
                    s[:, bad] = np.nan
                    sums[a, gi] = s
            med = np.stack([r["q"][nodes[g], :, 5].sum(0) for g in c.GROUP_IDS])
            prim_obs = np.stack([r["obs"][nodes[g]].sum(0) for g in c.GROUP_IDS])
            dir_arr = np.full((len(c.DIRECT), 4, S, 11), np.nan)
            meter = np.full((4, S), np.nan)
            for k, key_d in enumerate(c.DIRECT):
                sub = direct[key_d][r["month"]].filter(pl.col("power_fcst_init_time") == init)
                qd = c.run_arrays(
                    frame=sub, ids=list(c.GROUP_IDS), columns=qcols + ["observed_power"]
                )
                arr = qd[init]
                dir_arr[k] = np.sort(arr[..., :11], axis=-1)
                if key_d == "d622_lagged":
                    meter = arr[..., 11]
            np.savez_compressed(
                out,
                sums=sums.astype(np.float32),
                median_sum=med.astype(np.float32),
                meter=meter.astype(np.float32),
                prim_obs_sum=prim_obs.astype(np.float32),
                direct=dir_arr.astype(np.float32),
                month=r["month"],
            )
            print(network, f"{init:%Y-%m-%d} written", flush=True)


if __name__ == "__main__":
    main()

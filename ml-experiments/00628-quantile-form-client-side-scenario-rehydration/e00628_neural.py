# ruff: noqa: E501, ANN001, ANN202, PLW2901
"""Experiment 628, neural demand stage: client scenarios from the stored quantiles of the roster networks.

Reads the backtest export of the three roster networks (`backtest_export_roster_2026-11-07`): 11
quantiles per Primary, origin (09:00 UTC daily), and half-hour, for the five development months, and
repeats the client recipe of `e00628_score.py` on the 16 Primaries only. The in-pipeline reference
at 51 paths takes the 51 values at the levels k/52 (inside the stored q01 to q99), so it equals the
client's stratified 51-path draw with any tail rule. The copula is fitted with each month left out,
from the probability-integral-transform (PIT) scores of the stored quantiles. The Schaake shuffle
uses 51 past observed days from experiment 436's observed grid (the same nodes and units).

Rows scored are event-free half-hours with a cleaned reading, before 2026-07-01. Writes
`work/neural_<network>_<table>.parquet` for tables `crps`, `exceed`, and `query`.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00628-*/env.sh
OMP_NUM_THREADS=1 nice -n 10 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00628-*/e00628_neural.py --network K1 --workers 3
```
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import client_rehydrate as client  # noqa: E402  # ty: ignore[unresolved-import]
import e00628_score as s628  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.special import ndtri  # noqa: E402
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402

EXPORT: Final = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
Q_COLS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
P: Final[int] = 16
T: Final[int] = 479
K: Final[int] = s628.K
HOLDOUT: Final[datetime] = s628.HOLDOUT
BANDS: Final[dict[str, tuple[float, float]]] = s628.BANDS
LEVELS: Final[tuple[float, ...]] = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.05,
    0.95,
    0.99,
    0.995,
    0.998,
    0.999,
)
DAY3: Final[slice] = slice(96, 144)
SCHAAKE_FIRST: Final[datetime] = datetime(2025, 3, 1, 9, tzinfo=UTC)
GRID_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
MAPPING: Final[pl.DataFrame] = pl.read_csv(s628.OUT / "bsp_mapping.csv")
NODE_IDS: Final[list[int]] = MAPPING["time_series_id"].to_list()
CAPACITY: Final[np.ndarray] = MAPPING["capacity_mw"].to_numpy()


def band_steps() -> dict[str, np.ndarray]:
    """Return the step indices (lead = step / 2 hours) of each lead band."""
    lead = (np.arange(T) + 1) * 0.5
    return {n: np.flatnonzero((lead > lo) & (lead <= hi)) for n, (lo, hi) in BANDS.items()}


def load_runs(*, network: str) -> list[dict]:
    """Return one dict per origin: init, month, q (P, T, 11) and actual (P, T), in points of capacity."""
    caps = dict(zip(NODE_IDS, CAPACITY, strict=True))
    order = {sid: i for i, sid in enumerate(NODE_IDS)}
    runs = []
    for month in MONTHS:
        df = pl.read_parquet(EXPORT / f"{network}_{month}.parquet")
        df = df.filter(pl.col("time_series_id").is_in(NODE_IDS)).with_columns(
            p=pl.col("time_series_id").replace_strict(order, return_dtype=pl.Int64),
            step=(pl.col("lead_hours") * 2).round().cast(pl.Int64) - 1,
            cap=pl.col("time_series_id").replace_strict(caps, return_dtype=pl.Float64),
        )
        for init, sub in df.partition_by(
            "power_fcst_init_time", as_dict=True, maintain_order=True
        ).items():
            sub = sub.sort("p", "step")
            if sub.height != P * T:
                print(
                    f"{network}: skipped origin {init}: {sub.height} rows, expected {P * T}",
                    flush=True,
                )
                continue
            cap = sub["cap"].to_numpy().reshape(P, T)
            q = (
                sub.select(Q_COLS).to_numpy().astype(np.float64).reshape(P, T, 11)
                / cap[:, :, None]
                * 100.0
            )
            q = np.sort(q, axis=-1)
            y = np.where(
                sub["event_free"].to_numpy(),
                sub["observed_power"].to_numpy().astype(np.float64),
                np.nan,
            )
            actual = (y.reshape(P, T) / cap) * 100.0
            init_t = init[0].astimezone(UTC)
            valid = np.array([init_t + timedelta(minutes=30 * (i + 1)) for i in range(T)])
            actual[:, valid >= HOLDOUT] = np.nan
            runs.append({"init": init_t, "month": month, "q": q, "actual": actual})
    return runs


def pit_z(*, q: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the normal score of each actual under its 11 stored quantiles.

    The score is linear between the stored levels, with 0.005 and 0.995 outside them.
    """
    lv = client.LEVELS
    idx = (q < actual[..., None]).sum(-1)
    lo_i, hi_i = np.clip(idx - 1, 0, 10), np.clip(idx, 0, 10)
    ql, qh = (
        np.take_along_axis(q, lo_i[..., None], -1)[..., 0],
        np.take_along_axis(q, hi_i[..., None], -1)[..., 0],
    )
    w = np.where(qh > ql, (actual - ql) / np.where(qh > ql, qh - ql, 1.0), 0.5)
    u = lv[lo_i] * (1 - w) + lv[hi_i] * w
    u = np.where(idx == 0, 0.005, np.where(idx == 11, 0.995, u))
    return np.where(np.isnan(actual), np.nan, ndtri(u))


def fit_copula(*, z_runs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the Cholesky factor of the node correlation (P, P) and the AR(1) terms (P,).

    Both are fitted as experiment 436 fits them.
    """
    z = np.concatenate([zr[:, ::4] for zr in z_runs], axis=1)
    corr = np.eye(P)
    for i in range(P):
        for j in range(i + 1, P):
            ok = ~np.isnan(z[i]) & ~np.isnan(z[j])
            if ok.sum() >= 200:
                corr[i, j] = corr[j, i] = np.corrcoef(z[i, ok], z[j, ok])[0, 1]
    vals, vecs = np.linalg.eigh(corr)
    corr = (vecs * np.clip(vals, 1e-3, None)) @ vecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / d[:, None] / d[None, :]
    phi = np.zeros(P)
    for node in range(P):
        a = np.concatenate([zr[node, :-1] for zr in z_runs])
        b = np.concatenate([zr[node, 1:] for zr in z_runs])
        ok = ~np.isnan(a) & ~np.isnan(b)
        phi[node] = np.clip(np.corrcoef(a[ok], b[ok])[0, 1], 0.0, 0.995)
    return np.linalg.cholesky(corr), phi


def schaake_history(*, init: datetime, observed: np.ndarray) -> np.ndarray:
    """Return 51 past observed paths (51, P, T) at the origin's time of day.

    The paths are nearest in day of year and end 16 days before the origin.
    """
    last = init - timedelta(days=16)
    t0 = SCHAAKE_FIRST
    cands = []
    while t0 <= last:
        i = int((t0 - GRID_START) / STEP) + 1  # step 0 is lead 0.5 h
        path = observed[:P, i : i + T]
        if path.shape[1] == T and (np.isnan(path).mean(axis=1) <= 0.05).all():
            doy = abs((t0.timetuple().tm_yday - init.timetuple().tm_yday + 183) % 365 - 183)
            cands.append((doy, t0, path))
        t0 += timedelta(days=1)
    cands.sort(key=lambda c: (c[0], c[1]))
    x = np.arange(T)
    out = []
    for k in range(51):
        path = cands[k % len(cands)][2].copy()
        for p in range(P):
            bad = np.isnan(path[p])
            if bad.any() and (~bad).any():
                path[p, bad] = np.interp(x[bad], x[~bad], path[p, ~bad])
        out.append(path)
    return np.stack(out)


def crps_k(*, path: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the CRPS (P, T) of K paths (K, P, T)."""
    k = path.shape[0]
    srt = np.sort(path, axis=0)
    coeff = (2 * np.arange(1, k + 1) - k - 1).reshape(-1, 1, 1)
    return (
        np.abs(path - np.nan_to_num(actual)[None]).mean(0) - 0.5 * (coeff * srt).sum(0) * 2 / k**2
    )


def process_run(
    *, run: dict, chol: np.ndarray, phi: np.ndarray, observed: np.ndarray, network: str
) -> dict[str, list[dict]]:
    """Score every arm on one origin."""
    q, actual, init = run["q"], run["actual"], run["init"]
    have = ~np.isnan(actual)
    usable = (q[:, :, 10] - q[:, :, 0]) > 1e-6
    history = schaake_history(init=init, observed=observed)
    out: dict[str, list[dict]] = {"crps": [], "exceed": [], "query": []}
    base = {"power_fcst_init_time": init, "month": run["month"], "network": network}
    # reference: the 51 values at the levels k/52
    levels = np.arange(1, 52) / 52
    ref51 = np.stack(
        [
            client.invert(
                q=q[n], u=np.broadcast_to(levels[:, None], (51, T)), tail="clamp", bounds=None
            )
            for n in range(P)
        ],
        axis=1,
    )
    knots = {0.01: q[:, :, 0], 0.05: q[:, :, 1], 0.95: q[:, :, 9], 0.99: q[:, :, 10]}

    def add_crps(label, path, k, tail, coupling):
        c = crps_k(path=path, actual=actual)
        for band, steps in band_steps().items():
            ok = have[:, steps]
            if ok.any():
                out["crps"].append(
                    {
                        **base,
                        "arm": label,
                        "k": k,
                        "tail": tail,
                        "coupling": coupling,
                        "band": band,
                        "crps_sum": float(c[:, steps][ok].sum()),
                        "n": int(ok.sum()),
                    }
                )

    add_crps("N51", ref51, 51, "pipeline", "IND")
    for band, steps in band_steps().items():
        ok = (have & usable)[:, steps]
        for level, qv in knots.items():
            hit = (
                (actual[:, steps] < qv[:, steps])
                if level < 0.5
                else (actual[:, steps] > qv[:, steps])
            )
            out["exceed"].append(
                {
                    **base,
                    "arm": "N_knots",
                    "tail": "none",
                    "band": band,
                    "level": level,
                    "hits": int((hit & ok).sum()),
                    "n": int(ok.sum()),
                }
            )
    for tail in s628.TAILS:
        quant = {
            lv: np.stack(
                [
                    client.invert(q=q[n], u=np.full((1, T), lv), tail=tail, bounds=None)[0]
                    for n in range(P)
                ]
            )
            for lv in LEVELS
        }
        for band, steps in band_steps().items():
            ok = (have & usable)[:, steps]
            for lv, qv in quant.items():
                hit = (
                    (actual[:, steps] < qv[:, steps])
                    if lv < 0.5
                    else (actual[:, steps] > qv[:, steps])
                )
                out["exceed"].append(
                    {
                        **base,
                        "arm": f"Q11-{tail}",
                        "tail": tail,
                        "band": band,
                        "level": lv,
                        "hits": int((hit & ok).sum()),
                        "n": int(ok.sum()),
                    }
                )
    seed0 = int(init.timestamp()) % (2**31)
    for ci, c in enumerate(("IND", "COP", "SS")):
        sets = {}
        for si, name in enumerate(("A", "B")):
            rng = np.random.default_rng(seed0 + 17 * ci + 1000 * si)
            if c == "IND":
                sets[name] = rng.uniform(size=(K, P, T))
            elif c == "COP":
                sets[name] = client.copula_uniforms(chol=chol, phi=phi, k=K, steps=T, rng=rng)
            else:
                sets[name] = client.schaake_uniforms(history=history, k=K, rng=rng)
        bounds = np.full((P, 2), np.nan)
        for tail in s628.TAILS:
            drawn = {s: client.scenarios(q=q, u=sets[s], tail=tail, bounds=bounds) for s in sets}
            add_crps(f"{c}-{tail}", drawn["A"].astype(np.float64), K, tail, c)
            p51 = client.scenarios(
                q=q, u=s628.stratify(u=sets["A"][:51]), tail=tail, bounds=bounds
            ).astype(np.float64)
            for s, pth in (("A", drawn["A"]), ("B", drawn["B"]), ("A51", p51)):
                out["query"].append(
                    {
                        **base,
                        "arm": f"{c}-{tail}",
                        "coupling": c,
                        "tail": tail,
                        "set": s,
                        **picks(path=pth),
                    }
                )
    return out


def picks(*, path: np.ndarray) -> dict[str, float]:
    """Return the 95th percentile over paths of each BSP's day-3 summed Primaries (MW or MVA).

    The single-Primary maximum of that quantity is returned too.
    """
    out = {}
    for name in MAPPING["bsp"].unique().sort().to_list():
        rows = MAPPING.filter(pl.col("bsp") == name)
        mw = (
            path[:, rows["node"].to_numpy(), DAY3]
            * rows["capacity_mw"].to_numpy()[None, :, None]
            / 100.0
        )
        out["bsp_q95_" + name.split()[0].lower()] = float(
            np.quantile(mw.sum(axis=1).mean(axis=1), 0.95)
        )
    return out


def run_one(args: tuple) -> dict[str, list[dict]]:
    """Positional wrapper for `pool.map`."""
    run, chol, phi, observed, network = args
    return process_run(run=run, chol=chol, phi=phi, observed=observed, network=network)


def main() -> None:
    """Score every origin of one network and write the three tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="K1")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    observed = np.load(s628.score.WORK / "observed.npz")["observed"]
    runs = load_runs(network=args.network)
    z_all = [pit_z(q=r["q"], actual=r["actual"]) for r in runs]
    copulas = {}
    for month in MONTHS:
        start = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
        train = [
            i
            for i, r in enumerate(runs)
            if r["month"] != month and not (start - timedelta(days=15) < r["init"] < start)
        ]
        copulas[month] = fit_copula(z_runs=[z_all[i] for i in train])
    todo = runs[: args.max_runs]
    jobs = [(r, *copulas[r["month"]], observed, args.network) for r in todo]
    print(f"{len(jobs)} origins", flush=True)
    tables: dict[str, list[dict]] = {"crps": [], "exceed": [], "query": []}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for job, result in zip(jobs, pool.map(run_one, jobs), strict=True):
            for name, rows in result.items():
                tables[name].extend(rows)
            print(f"{job[0]['init']:%Y-%m-%d} scored", flush=True)
    for name, rows in tables.items():
        pl.DataFrame(rows).write_parquet(
            s628.OUT / f"neural_{args.network}_{name}{args.suffix}.parquet"
        )


if __name__ == "__main__":
    main()

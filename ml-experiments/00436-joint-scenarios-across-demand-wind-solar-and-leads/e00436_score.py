# ruff: noqa: E501
"""Tier 0 scoring of experiment 436: re-order the 51 values of 25 nodes into scenarios and score them.

No fit. Reads `work/runs/*.npz` and `work/observed.npz` (written by `e00436_build.py`), assigns the
51 values at every node and half-hour to 51 scenarios in four ways, and scores how well the
scenarios match the observed joint behaviour:

- `J0`: an independent random permutation per node and half-hour;
- `J1`: every node and half-hour takes the same rank (comonotone);
- `J2`: ensemble copula coupling on a shared member index (the `ecc` array of the run file);
- `J3`: one Schaake template: 51 past observed paths of all 25 nodes, ranked jointly.

Scores (points of capacity; lower is better) are the energy score and the variogram score (order
0.5), each divided by the square root of the block size and computed by experiment 434's code. Four
blocks: `spatial` (the nodes of one family at each half-hour; label = family), `cross` (the family
means of the three families at each half-hour, solar daylight only), `temporal` (each node across
leads; label = node), and `joint` (all nodes at coarse leads).

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00436-*/env.sh
nice -n 19 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00436-*/e00436_score.py --arms J0 J1 J2 J3
```
"""

import argparse
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00434_score as s434  # ty: ignore[unresolved-import]
import e00436_build as build  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

N_NODES: Final[int] = build.N_NODES
N_STEPS: Final[int] = build.N_STEPS
N_MEMBERS: Final[int] = build.N_MEMBERS
FAMILY_NAMES: Final[tuple[str, ...]] = ("demand", "wind", "solar")
NODE_NAMES: Final[list[str]] = [
    *[f"D{i}" for i in range(build.N_PRIMARIES)],
    *build.WIND_FARMS,
    *[f"S{f}" for f in build.SOLAR_FARMS],
]
"""Anonymised node labels: Primaries in table order, then wind farms W1 to W3, then solar farms."""
MAX_MISSING: Final[dict[int, float]] = {0: 0.05, 1: 0.25, 2: 0.05}
"""Largest share of missing half-hours a node may have on a Schaake date, by family."""
DAYLIGHT_UTC_HOURS: Final[tuple[float, float]] = (10.0, 14.0)
SCHAAKE_FIRST: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
SCHAAKE_BUFFER: Final[timedelta] = timedelta(days=16)
WORK: Final[Path] = build.WORK_DIR
MIN_SPATIAL: Final[dict[int, int]] = {0: 8, 1: 2, 2: 3}
"""Fewest nodes with an actual that a family needs for a spatial vector at one half-hour."""
SLICES: Final[tuple[str, ...]] = ("all", "daylight")


def step_masks(*, init: datetime) -> dict[str, np.ndarray]:
    """Return the half-hour masks (702,) of each slice for a run at `init`."""
    hours = np.array(
        [
            (init + (i + 1) * build.STEP).hour + (init + (i + 1) * build.STEP).minute / 60
            for i in range(N_STEPS)
        ]
    )
    low, high = DAYLIGHT_UTC_HOURS
    return {"all": np.ones(N_STEPS, dtype=bool), "daylight": (hours >= low) & (hours < high)}


def schaake_ranks(
    *, init: datetime, observed: np.ndarray, kind: np.ndarray
) -> tuple[np.ndarray, int]:
    """Return the Schaake ranks (25, 702, 51) and the number of qualifying dates.

    Candidate dates start at the run's time of day, from 2025-03-01 to 16 days before the run, so
    every past path ends before the run. A date is dropped if any node misses more of its half-hours
    than `MAX_MISSING` for its family; gaps are filled by linear interpolation along the path. The 51
    dates nearest in day of year are used, repeated cyclically if fewer qualify.
    """
    last = init - SCHAAKE_BUFFER
    t0 = SCHAAKE_FIRST.replace(hour=init.hour, minute=init.minute)
    limit = np.array([MAX_MISSING[int(k)] for k in kind])
    candidates = []
    while t0 <= last:
        i = build.grid_index(t=t0) + 1
        path = observed[:, i : i + N_STEPS]
        if path.shape[1] == N_STEPS and (np.isnan(path).mean(axis=1) <= limit).all():
            doy = abs((t0.timetuple().tm_yday - init.timetuple().tm_yday + 183) % 365 - 183)
            candidates.append((doy, t0, path))
        t0 += timedelta(days=1)
    if not candidates:
        msg = f"{init}: no Schaake date qualifies"
        raise ValueError(msg)
    candidates.sort(key=lambda c: (c[0], c[1]))
    x = np.arange(N_STEPS)
    pool = []
    for k in range(N_MEMBERS):
        out = candidates[k % len(candidates)][2].copy()
        for p in range(out.shape[0]):
            bad = np.isnan(out[p])
            if bad.any():
                out[p, bad] = np.interp(x[bad], x[~bad], out[p, ~bad])
        pool.append(out)
    rng = np.random.default_rng(int(hashlib.sha256(init.isoformat().encode()).hexdigest()[:8], 16))
    stack = np.stack(pool).transpose(1, 2, 0)  # (node, step, 51)
    return s434.ranks_by_key(key=stack, rng=rng), len(candidates)


def make_ranks(
    *, arm: str, run: dict, init: datetime, observed: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return the rank each scenario takes at every node and half-hour, (25, 702, 51)."""
    shape = (N_NODES, N_STEPS, N_MEMBERS)
    identity = np.broadcast_to(np.arange(N_MEMBERS), shape)
    if arm == "J0":
        return rng.permuted(identity, axis=-1)
    if arm == "J1":
        return identity
    if arm == "J2":
        return run["ecc"].astype(np.int64)
    if arm == "J3":
        return schaake_ranks(init=init, observed=observed, kind=run["kind"])[0]
    msg = f"unknown arm {arm}"
    raise ValueError(msg)


def rows_for_run(
    *, paths: dict[str, np.ndarray], actual: np.ndarray, kind: np.ndarray, init: datetime
) -> list[dict]:
    """Return the score rows of one run: every arm, block, window, and slice."""
    rows: list[dict] = []
    have = ~np.isnan(actual)
    masks = step_masks(init=init)
    y_all = np.nan_to_num(actual)
    for window in s434.WINDOWS:
        steps = s434.window_steps(window=window)
        for slice_name in SLICES:
            sel = steps[masks[slice_name][steps]]
            if sel.size == 0:
                continue
            # spatial: one vector per family at each half-hour
            for fam, name in enumerate(FAMILY_NAMES):
                nodes = np.flatnonzero(kind == fam)
                keep = have[nodes][:, sel].sum(0) >= MIN_SPATIAL[fam]
                if not keep.any():
                    continue
                cols = sel[keep]
                y = y_all[nodes][:, cols].T
                mask = have[nodes][:, cols].T
                for arm, path in paths.items():
                    x = path[:, nodes][:, :, cols].transpose(2, 0, 1)
                    es, vs = s434.score_blocks(x=x, y=y, mask=mask, chunk=60)
                    rows.append(
                        _row(
                            init,
                            "spatial",
                            window,
                            slice_name,
                            arm,
                            name,
                            es.mean(),
                            vs.mean(),
                            int(keep.sum()),
                        )
                    )
            # cross: the family means at each half-hour, daylight slice only (solar is zero at night)
            weights = np.zeros((3, N_NODES, sel.size))
            ok = np.ones(sel.size, dtype=bool)
            for fam in range(3):
                nodes = np.flatnonzero(kind == fam)
                count = have[nodes][:, sel].sum(0)
                ok &= count >= 1
                weights[fam, nodes] = have[nodes][:, sel] / np.maximum(count, 1)
            if ok.any() and slice_name == "daylight":
                cols = sel[ok]
                w = weights[:, :, ok]
                y = np.einsum("fns,ns->sf", w, y_all[:, cols])
                mask = np.ones_like(y, dtype=bool)
                for arm, path in paths.items():
                    agg = np.einsum("fns,mns->smf", w, path[:, :, cols])
                    es, vs = s434.score_blocks(x=agg, y=y, mask=mask, chunk=120)
                    rows.append(
                        _row(
                            init,
                            "cross",
                            window,
                            slice_name,
                            arm,
                            "all",
                            es.mean(),
                            vs.mean(),
                            int(ok.sum()),
                        )
                    )
            # temporal: each node across leads
            sub = sel[:: s434.TEMPORAL_STRIDE[window]]
            tmask = have[:, sub]
            need = s434.MIN_DIM["temporal"] if window == "days_1_10" else 6
            good = tmask.sum(1) >= (need if slice_name == "all" else max(need // 2, 4))
            if good.any():
                y = y_all[:, sub][good]
                for arm, path in paths.items():
                    x = path[:, :, sub].transpose(1, 0, 2)[good]
                    es, vs = s434.score_blocks(x=x, y=y, mask=tmask[good], chunk=4)
                    for k, node in enumerate(np.flatnonzero(good)):
                        rows.append(
                            _row(
                                init,
                                "temporal",
                                window,
                                slice_name,
                                arm,
                                NODE_NAMES[node],
                                es[k],
                                vs[k],
                                int(tmask[node].sum()),
                            )
                        )
        # joint: all nodes at coarse leads
        sub = steps[:: s434.JOINT_STRIDE[window]]
        jm = have[:, sub].reshape(1, -1)
        if jm.sum() >= (100 if window == "days_1_10" else 60):
            y = y_all[:, sub].reshape(1, -1)
            for arm, path in paths.items():
                x = path[:, :, sub].transpose(1, 0, 2).reshape(N_MEMBERS, -1)[None]
                es, vs = s434.score_blocks(x=x, y=y, mask=jm, chunk=1)
                rows.append(
                    _row(init, "joint", window, "all", arm, "all", es[0], vs[0], int(jm.sum()))
                )
    return rows


def _row(
    init: datetime,
    block: str,
    window: str,
    slice_name: str,
    arm: str,
    label: str,
    es: float,
    vs: float,
    dim: int,
) -> dict:
    return {
        "power_fcst_init_time": init, "block": block, "window": window, "slice": slice_name,
        "arm": arm, "label": label, "energy": float(es), "variogram": float(vs), "n_dim": dim,
    }  # fmt: skip


def load_run(*, path: Path) -> dict:
    """Return a run file's arrays as a dict."""
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main() -> None:
    """Score the requested arms on every joint run and write `scores_tier0.parquet`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["J0", "J1", "J2", "J3"])
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--out", default="scores_tier0")
    args = parser.parse_args()
    observed = np.load(WORK / "observed.npz")["observed"]
    files = sorted((WORK / "runs").glob("*.npz"))[: args.max_runs]
    rows: list[dict] = []
    marginal: list[dict] = []
    for file in files:
        run = load_run(path=file)
        init = datetime.fromisoformat(str(run["init"]) + "+00:00").astimezone(UTC)
        rng = np.random.default_rng(
            int(hashlib.sha256(init.isoformat().encode()).hexdigest()[8:16], 16)
        )
        values = run["values"].astype(np.float64)
        paths = {}
        for arm in args.arms:
            ranks = make_ranks(arm=arm, run=run, init=init, observed=observed, rng=rng)
            paths[arm] = s434.assign(values=values, ranks=ranks)
        reference = np.sort(paths[args.arms[0]], axis=0)
        for arm, path in paths.items():
            if not np.array_equal(np.sort(path, axis=0), reference):
                msg = f"{init}: arm {arm} is not a re-ordering of the same 51 values"
                raise ValueError(msg)
        actual = run["actual"].astype(np.float64)
        rows.extend(rows_for_run(paths=paths, actual=actual, kind=run["kind"], init=init))
        crps = s434.crps_of_sorted(values=values, actual=np.nan_to_num(actual))
        for node in range(N_NODES):
            ok = ~np.isnan(actual[node])
            marginal.append(
                {
                    "power_fcst_init_time": init,
                    "label": NODE_NAMES[node],
                    "crps51": float(crps[node][ok].mean()) if ok.any() else float("nan"),
                    "n": int(ok.sum()),
                }
            )
        print(f"{init:%Y-%m-%d} scored", flush=True)
    pl.DataFrame(rows).write_parquet(WORK / f"{args.out}.parquet")
    pl.DataFrame(marginal).write_parquet(WORK / f"{args.out}_marginal.parquet")


if __name__ == "__main__":
    main()

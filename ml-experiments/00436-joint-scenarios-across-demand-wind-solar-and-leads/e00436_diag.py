# ruff: noqa: E501
"""Diagnostics of experiment 436: marginal check, rank histograms, dependence, spread-skill, proxies.

No fit. Reads the scenarios in `work/paths/<arm>/<date>.npy` and the run files, and writes four
tables to `work/`:

- `diag_marginal.parquet`: per run, arm, and family, the 51-value CRPS and the exceedance shares
  beyond the scenarios' 1%, 5%, 95%, and 99% quantiles (days 1 to 10 only);
- `diag_rank.parquet`: per arm and block, the histogram (52 bins) of the observation's average-rank
  and band-depth pre-rank among the 52 vectors (51 scenarios and the observation);
- `diag_dependence.parquet`: per arm, the Gaussian-score correlation matrix of the scenarios, and of
  the observations (arm `obs`), across nodes at one half-hour and across lags of 0.5, 3, 12, and 24
  hours; and the spread-skill ratio per family and lead day;
- `diag_proxy.parquet`: per run, arm, and bulk supply point group, the Brier scores of simultaneous
  exceedance, of group net flow below its export proxy, and of ramps, and the group-sum CRPS and
  exceedance shares.

The limit proxies are stated in `proxies.json`, written before any proxy is scored: a Primary's
static limit is the 95th percentile of its observed power over 2025-03-01 to 2025-08-31 (before the
first run), and a group's export proxy is the 5th percentile of its observed net flow over the same
months (Primaries minus wind farms, in MW). Solar farms are not mapped to a bulk supply point and are
left out of the group sums.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00436-*/env.sh
nice -n 19 uv run --no-sync python ml-experiments/00436-*/e00436_diag.py --arms J0 J1 J2 J3 J4 J5 J6 J6-median
```
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from typing import Final

import e00303_common as c303  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00436_build as build  # ty: ignore[unresolved-import]
import e00436_scenarios as scen  # ty: ignore[unresolved-import]
import e00436_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.stats import norm
from studies.sources import ML_EXPERIMENTS_DATA_DIR

N: Final[int] = build.N_NODES
S: Final[int] = build.N_STEPS
M: Final[int] = build.N_MEMBERS
KIND: Final[np.ndarray] = build.KIND
LEAD_STEPS: Final[int] = 480
"""Half-hours of days 1 to 10 (lead up to 240 hours)."""
LAGS: Final[dict[str, int]] = {"0.5h": 1, "3h": 6, "12h": 24, "24h": 48}
HIERARCHY: Final = (
    ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "substation_hierarchy_2026-10-15" / "Substations.csv"
)
METADATA: Final = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet"
PROXY_START: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
PROXY_END: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
MIN_LIMIT_SHARE: Final[float] = 0.01
"""A Primary whose q95 proxy limit is under this share of its capacity is left out of the exceedance count."""
RAMP_POINTS: Final[float] = 20.0
RAMP_STEPS: Final[int] = 6
RANK_STRIDE: Final[int] = 6
BLOCKS: Final[tuple[str, ...]] = (
    "spatial_demand",
    "spatial_wind",
    "spatial_solar",
    "cross",
    "temporal",
    "joint",
)


def groups() -> tuple[dict[str, list[int]], np.ndarray]:
    """Return the nodes of each bulk supply point group and each node's capacity in MW."""
    series = adapter.series_table()
    sites = c303.site_table()
    md = pl.read_parquet(METADATA).select("time_series_id", "substation_number")
    hier = pl.read_csv(HIERARCHY).select("substation_number", "bsp_name")
    capacity = np.full(N, np.nan)
    members: dict[str, list[int]] = {}
    rows = [
        (i, series["time_series_id"][i], series["capacity_mw"][i]) for i in range(build.N_PRIMARIES)
    ] + [
        (
            build.N_PRIMARIES + f,
            sites.filter(pl.col("site") == farm)["time_series_id"][0],
            sites.filter(pl.col("site") == farm)["effective_capacity_mw"][0],
        )
        for f, farm in enumerate(build.WIND_FARMS)
    ]
    for node, sid, cap in rows:
        capacity[node] = cap
        name = (
            md.filter(pl.col("time_series_id") == sid)
            .join(hier, on="substation_number", how="left")["bsp_name"]
            .to_list()
        )
        if name and name[0]:
            members.setdefault("A" if "BOSTON" in name[0] else "B", []).append(node)
    return members, capacity


def proxies(*, observed: np.ndarray, members: dict[str, list[int]], capacity: np.ndarray) -> dict:
    """Return the limit proxies in MW and write them to `proxies.json` before any scoring."""
    lo = build.grid_index(t=PROXY_START)
    hi = build.grid_index(t=PROXY_END)
    mw = observed[:, lo:hi] / 100.0 * capacity[:, None]
    out: dict = {"primary_q95_mw": {}, "group_export_q05_mw": {}, "groups": members}
    for node in range(build.N_PRIMARIES):
        out["primary_q95_mw"][str(node)] = float(np.nanpercentile(mw[node], 95))
    for name, nodes in members.items():
        demand = [n for n in nodes if n < build.N_PRIMARIES]
        wind = [n for n in nodes if n >= build.N_PRIMARIES]
        net = (
            np.nansum(mw[demand], axis=0) - np.nansum(mw[wind], axis=0)
            if wind
            else np.nansum(mw[demand], axis=0)
        )
        out["group_export_q05_mw"][name] = float(np.nanpercentile(net, 5))
    (score.WORK / "proxies.json").write_text(json.dumps(out, indent=2))
    return out


def pre_ranks(*, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Return the average-rank and band-depth pre-rank (0 to 51) of the observation y among 52 vectors.

    Args:
        x: Scenarios, (51, d).
        y: Observation, (d,).
        rng: Breaks ties in the final rank.
    """
    full = np.concatenate([x, y[None]], axis=0)  # (52, d)
    jitter = rng.random(full.shape) * 1e-9
    rank = np.argsort(np.argsort(full + jitter, axis=0), axis=0) + 1  # 1 to 52 per coordinate
    avg = rank.mean(axis=1)
    depth = ((rank - 1) * (M + 1 - rank)).mean(axis=1)
    tie = rng.random(M + 1) * 1e-9
    return int(np.sum((avg[:-1] + tie[:-1]) < (avg[-1] + tie[-1]))), int(
        np.sum((depth[:-1] + tie[:-1]) < (depth[-1] + tie[-1]))
    )


def rank_instances(
    *, path: np.ndarray, actual: np.ndarray, rng: np.random.Generator
) -> dict[str, list[tuple[int, int]]]:
    """Return the (average-rank, band-depth) pre-ranks of one run's instances, by block."""
    have = ~np.isnan(actual)
    y_all = np.nan_to_num(actual)
    steps = np.arange(0, LEAD_STEPS, RANK_STRIDE)
    out: dict[str, list[tuple[int, int]]] = {b: [] for b in BLOCKS}
    for fam, name in enumerate(("demand", "wind", "solar")):
        nodes = np.flatnonzero(KIND == fam)
        for t in steps:
            ok = have[nodes, t]
            if ok.sum() >= score.MIN_SPATIAL[fam]:
                out[f"spatial_{name}"].append(
                    pre_ranks(x=path[:, nodes[ok], t], y=y_all[nodes[ok], t], rng=rng)
                )
    for t in steps:
        agg_x, agg_y, good = [], [], True
        for fam in range(3):
            nodes = np.flatnonzero((KIND == fam) & have[:, t])
            if nodes.size == 0:
                good = False
                break
            agg_x.append(path[:, nodes, t].mean(1))
            agg_y.append(y_all[nodes, t].mean())
        if good:
            out["cross"].append(pre_ranks(x=np.stack(agg_x, axis=1), y=np.array(agg_y), rng=rng))
    coarse = np.arange(0, LEAD_STEPS, 12)
    for node in range(N):
        ok = have[node, coarse]
        if ok.sum() >= 10:
            out["temporal"].append(
                pre_ranks(x=path[:, node, coarse[ok]], y=y_all[node, coarse[ok]], rng=rng)
            )
    ok = have[:, coarse]
    if ok.sum() >= 100:
        out["joint"].append(
            pre_ranks(
                x=path[:, :, coarse].transpose(0, 1, 2)[:, ok], y=y_all[:, coarse][ok], rng=rng
            )
        )
    return out


def gauss_scores(*, path: np.ndarray) -> np.ndarray:
    """Return the normal score of each scenario's rank among the 51 at every node and half-hour."""
    rank = np.argsort(np.argsort(path, axis=0, kind="stable"), axis=0)
    return norm.ppf((rank + 0.5) / M)


def corr_blocks(*, z: np.ndarray, ok: np.ndarray) -> dict[str, np.ndarray]:
    """Return the correlation matrices of z (samples, 25, steps): lag 0 across nodes, and lagged within nodes.

    `ok` marks where each node has an actual, so that scenarios and observations use the same cells.
    """
    out: dict[str, np.ndarray] = {}
    steps = np.arange(0, LEAD_STEPS, 4)
    flat = np.where(ok[None, :, :], z, np.nan)[:, :, steps]  # (samples, node, steps)
    mat = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(i, N):
            both = ~np.isnan(flat[:, i]) & ~np.isnan(flat[:, j])
            if both.sum() > 50:
                mat[i, j] = mat[j, i] = np.corrcoef(flat[:, i][both], flat[:, j][both])[0, 1]
    out["lag0"] = mat
    for name, lag in LAGS.items():
        vals = np.full(N, np.nan)
        for i in range(N):
            a = np.where(ok[i, :-lag][None, :], z[:, i, :-lag], np.nan)[:, : LEAD_STEPS - lag]
            b = np.where(ok[i, lag:][None, :], z[:, i, lag:], np.nan)[:, : LEAD_STEPS - lag]
            both = ~np.isnan(a) & ~np.isnan(b)
            if both.sum() > 50:
                vals[i] = np.corrcoef(a[both], b[both])[0, 1]
        out[f"lag_{name}"] = vals
    return out


def main() -> None:  # noqa: PLR0914, PLR0915
    """Write the four diagnostic tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms", nargs="+", default=["J0", "J1", "J2", "J3", "J4", "J5", "J6", "J6-median"]
    )
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    observed = np.load(score.WORK / "observed.npz")["observed"]
    members, capacity = groups()
    limits = proxies(observed=observed, members=members, capacity=capacity)
    files = sorted((score.WORK / "runs").glob("*.npz"))[: args.max_runs]
    marginal, proxy_rows = [], []
    corr_scen: dict[str, dict[str, list]] = {arm: {} for arm in args.arms}
    corr_obs: dict[str, list] = {}
    rank_hist = {arm: {b: np.zeros((2, M + 1)) for b in BLOCKS} for arm in args.arms}
    for file in files:
        run = score.load_run(path=file)
        init = datetime.fromisoformat(f"{run['init']}+00:00").astimezone(UTC)
        actual = run["actual"].astype(np.float64)
        have = ~np.isnan(actual)
        month = f"{init:%Y-%m}"
        z_obs = scen.pit_scores(values=run["values"].astype(np.float64), actual=actual, seed=1)
        for arm in args.arms:
            path = np.load(score.scenario_path(arm=arm, init=init)).astype(np.float64)
            rng = np.random.default_rng(
                int(init.timestamp()) % (2**31)
                + int(hashlib.sha256(arm.encode()).hexdigest()[:4], 16)
            )
            # marginal and spread-skill
            if arm in ("J2", "J6", "J6-median"):
                q = np.quantile(path, [0.01, 0.05, 0.95, 0.99], axis=0)  # (4, 25, 702)
                srt = np.sort(path, axis=0)
                coeff = (2 * np.arange(1, M + 1) - M - 1)[:, None, None]
                crps = (
                    np.abs(srt - np.nan_to_num(actual)[None]).mean(0) - (coeff * srt).sum(0) / M**2
                )
                for fam, name in enumerate(("demand", "wind", "solar")):
                    nodes = np.flatnonzero(KIND == fam)
                    for day in range(10):
                        sl = slice(day * 48, (day + 1) * 48)
                        ok = have[nodes][:, sl]
                        if not ok.any():
                            continue
                        y = actual[nodes][:, sl][ok]
                        mean = path[:, nodes][:, :, sl].mean(0)[ok]
                        var = path[:, nodes][:, :, sl].var(0, ddof=1)[ok]
                        marginal.append({
                            "month": month, "init": init, "arm": arm, "family": name, "lead_day": day + 1,
                            "crps": float(crps[nodes][:, sl][ok].mean()),
                            "above_q95": float((y > q[2][nodes][:, sl][ok]).mean()),
                            "above_q99": float((y > q[3][nodes][:, sl][ok]).mean()),
                            "below_q05": float((y < q[1][nodes][:, sl][ok]).mean()),
                            "below_q01": float((y < q[0][nodes][:, sl][ok]).mean()),
                            "mse": float(((mean - y) ** 2).mean()), "var": float(var.mean()), "n": int(ok.sum()),
                        })  # fmt: skip
            # rank histograms
            inst = rank_instances(path=path, actual=actual, rng=rng)
            for block, pairs in inst.items():
                for a, d in pairs:
                    rank_hist[arm][block][0, a] += 1
                    rank_hist[arm][block][1, d] += 1
            # dependence matrices (scenario Gaussian scores of ranks)
            zs = gauss_scores(path=path)
            for key, value in corr_blocks(z=zs, ok=have).items():
                corr_scen[arm].setdefault(key, []).append(value)
            # proxies
            for name, nodes in members.items():
                demand = [n for n in nodes if n < build.N_PRIMARIES]
                wind = [n for n in nodes if n >= build.N_PRIMARIES]
                cap = capacity[:, None]
                mw_path = path[:, :, :LEAD_STEPS] / 100.0 * cap[None, :, :]
                mw_act = actual[:, :LEAD_STEPS] / 100.0 * cap
                ok_nodes = have[nodes][:, :LEAD_STEPS].all(0)
                if not ok_nodes.any():
                    continue
                lim = np.array([limits["primary_q95_mw"][str(n)] for n in demand])
                use = lim > MIN_LIMIT_SHARE * capacity[demand]  # a limit near zero is a data defect
                demand_used = [n for n, u in zip(demand, use, strict=True) if u]
                lim = lim[use]
                over_obs = (mw_act[demand_used] > lim[:, None]).sum(0) >= 2
                over_scen = ((mw_path[:, demand_used] > lim[None, :, None]).sum(1) >= 2).mean(0)
                net_path = mw_path[:, demand].sum(1) - (mw_path[:, wind].sum(1) if wind else 0.0)
                net_act = np.nansum(mw_act[demand], axis=0) - (
                    np.nansum(mw_act[wind], axis=0) if wind else 0.0
                )
                exp_lim = limits["group_export_q05_mw"][name]
                low_obs = net_act < exp_lim
                low_scen = (net_path < exp_lim).mean(0)
                srt = np.sort(net_path, axis=0)
                coeff = (2 * np.arange(1, M + 1) - M - 1)[:, None]
                crps = np.abs(srt - net_act[None]).mean(0) - (coeff * srt).sum(0) / M**2
                q95, q99 = np.quantile(net_path, [0.95, 0.99], axis=0)
                ramp_brier = []
                for n in nodes:
                    ramp_obs = (
                        np.abs(
                            actual[n, RAMP_STEPS:LEAD_STEPS] - actual[n, : LEAD_STEPS - RAMP_STEPS]
                        )
                        > RAMP_POINTS
                    )
                    ramp_scen = (
                        np.abs(
                            path[:, n, RAMP_STEPS:LEAD_STEPS]
                            - path[:, n, : LEAD_STEPS - RAMP_STEPS]
                        )
                        > RAMP_POINTS
                    ).mean(0)
                    okr = have[n, RAMP_STEPS:LEAD_STEPS] & have[n, : LEAD_STEPS - RAMP_STEPS]
                    if okr.any():
                        ramp_brier.append(float(((ramp_scen - ramp_obs) ** 2)[okr].mean()))
                excess_obs = np.maximum(exp_lim - net_act, 0.0)[ok_nodes].sum() * 0.5
                excess_scen = np.maximum(exp_lim - net_path, 0.0)[:, ok_nodes].sum(1).mean() * 0.5
                proxy_rows.append({
                    "month": month, "init": init, "arm": arm, "group": name,
                    "brier_two_over": float(((over_scen - over_obs) ** 2)[ok_nodes].mean()),
                    "brier_export": float(((low_scen - low_obs) ** 2)[ok_nodes].mean()),
                    "brier_ramp": float(np.mean(ramp_brier)) if ramp_brier else float("nan"),
                    "sum_crps": float(crps[ok_nodes].mean()),
                    "sum_above_q95": float((net_act > q95)[ok_nodes].mean()),
                    "sum_above_q99": float((net_act > q99)[ok_nodes].mean()),
                    "excess_volume_abs_error": float(abs(excess_scen - excess_obs)),
                    "excess_volume_obs": float(excess_obs), "n": int(ok_nodes.sum()),
                })  # fmt: skip
        for key, value in corr_blocks(z=z_obs[None], ok=have).items():
            corr_obs.setdefault(key, []).append(value)
        print(f"{init:%Y-%m-%d} diagnosed", flush=True)
    pl.DataFrame(marginal).write_parquet(score.WORK / "diag_marginal.parquet")
    pl.DataFrame(proxy_rows).write_parquet(score.WORK / "diag_proxy.parquet")
    hist_rows = [
        {
            "arm": arm,
            "block": block,
            "kind": kind,
            "bin": b,
            "count": float(rank_hist[arm][block][k, b]),
        }
        for arm in args.arms
        for block in BLOCKS
        for k, kind in enumerate(("average_rank", "band_depth"))
        for b in range(M + 1)
    ]
    pl.DataFrame(hist_rows).write_parquet(score.WORK / "diag_rank.parquet")
    dep_rows = []
    obs_mean = {k: np.nanmean(np.stack(v), axis=0) for k, v in corr_obs.items()}
    for arm in args.arms:
        for key, values in corr_scen[arm].items():
            mean = np.nanmean(np.stack(values), axis=0)
            ref = obs_mean[key]
            if key == "lag0":
                for label, mask in _blocks().items():
                    diff = (mean - ref)[mask]
                    dep_rows.append(
                        {
                            "arm": arm,
                            "item": f"lag0_{label}",
                            "rmsd": float(np.sqrt(np.nanmean(diff**2))),
                            "scenario": float(np.nanmean(mean[mask])),
                            "observed": float(np.nanmean(ref[mask])),
                        }
                    )
            else:
                diff = mean - ref
                dep_rows.append(
                    {
                        "arm": arm,
                        "item": key,
                        "rmsd": float(np.sqrt(np.nanmean(diff**2))),
                        "scenario": float(np.nanmean(mean)),
                        "observed": float(np.nanmean(ref)),
                    }
                )
    pl.DataFrame(dep_rows).write_parquet(score.WORK / "diag_dependence.parquet")


def _blocks() -> dict[str, np.ndarray]:
    """Return boolean masks over the 25 x 25 matrix of node pairs: within-family and cross-family blocks."""
    same = KIND[:, None] == KIND[None, :]
    off = ~np.eye(N, dtype=bool)
    out = {
        "within_demand": same & (KIND[:, None] == 0) & off,
        "within_wind": same & (KIND[:, None] == 1) & off,
        "within_solar": same & (KIND[:, None] == 2) & off,
    }
    out["demand_wind"] = (KIND[:, None] == 0) & (KIND[None, :] == 1)
    out["demand_solar"] = (KIND[:, None] == 0) & (KIND[None, :] == 2)
    out["wind_solar"] = (KIND[:, None] == 1) & (KIND[None, :] == 2)
    return out


if __name__ == "__main__":
    main()

"""The lag-encroachment sweep of experiment 392, applied to the networks of experiment 404.

For each clean row of the clean (real-data) world, a step of D = 5% of capacity (also 2.5%, both
signs) is added to the power over the span that holds the row's `k` most recent lags: the last 3
hours before the newest usable reading for k = 1, the last 24 hours for k = 2, and the last 7 days
for k = 3. The lag values and every trust input are recomputed from the shifted power, so the trust
head sees what a real event would show it. The response is the shift of the median forecast as a
share of the step, the ratio of the 90% band's width to its width without the step, and where the
forecast's 95% quantile falls. A lag-masked control hides all three lags (and their trust inputs) of
the same network, so its shift must be exactly 0.

Only the row's own Primary is shifted, one Primary at a time. The shifted rows are the clean-world
rows of every third origin day (`ORIGIN_STRIDE`), the clean class of `e00404_report`.

```bash
PYTHONPATH=<as e00404_fit.py shows> \
    uv run --no-sync python ml-experiments/00404-*/e00404_sweep.py run --month 2026-06 --seed 0 \
    --arms A0-inj B1 B5 --device cuda
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_rows as rows_mod  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

ORIGIN_STRIDE: Final[int] = 3
SPAN_STEPS: Final[dict[int, int]] = {1: 6, 2: 48, 3: 336}
"""Half-hours before the newest usable reading that a step must cover to reach `k` lags."""
STEPS: Final[tuple[float, ...]] = (0.05, -0.05)
CONTROL: Final[str] = "masked"


def subset(*, pool: fit.Pool, idx: torch.Tensor) -> fit.Pool:
    """Return the rows `idx` of a pool."""
    return fit.Pool(
        primary=pool.primary[idx],
        origin=pool.origin[idx],
        target=pool.target[idx],
        lead_hours=pool.lead_hours[idx],
        base=pool.base[idx],
        lead=pool.lead[idx],
        y=pool.y[idx],
        ll=pool.ll[idx],
        weight=pool.weight[idx],
    )


@torch.no_grad()
def band(
    *,
    net: torch.nn.Module,
    pool: fit.Pool,
    power: torch.Tensor,
    stack: rows_mod.Stack,
    lag_stats: np.ndarray,
    scalers: fit.Scalers,
    metered: torch.Tensor,
    device: str,
    masked: tuple[int, ...],
) -> np.ndarray:
    """Return q05, q50, and q95 (rows, 3) in capacity fractions for a pool on a power array."""
    lags, lag_grid = fit.lag_features(pool=pool, power=power, lag_stats=lag_stats, device=device)
    mu = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    sd = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    out = fit.forward(
        net=net,
        pool=pool,
        idx=torch.arange(len(pool), device=device),
        lags=lags,
        stack=stack,
        lag_grid=lag_grid,
        scalers=scalers,
        metered=metered,
        masked=masked,
    )
    log_w, mean, std = out[0] if isinstance(out[0], tuple) else out
    q = score.mixture_quantiles_capacity(
        log_w=log_w, mean=mean, std=std, mu=mu[pool.primary], sd=sd[pool.primary]
    )
    return q[:, [1, 5, 9]].cpu().numpy()


def run(*, month: str, seed: int, arms: list[str], device: str) -> pl.DataFrame:
    """Run the sweep for the arms and return one row per arm, k, step, and lead band."""
    clean = pl.read_parquet(
        score.prediction_path(arm=arms[0], month=month, seed=seed, world=-1)
    ).with_row_index("row")
    day = clean["origin"].to_numpy().astype(np.int64) // 48
    chosen = (day % ORIGIN_STRIDE == 0) & ~(
        clean["tgt_real"].to_numpy() | clean["tgt_post"].to_numpy() | clean["win_real"].to_numpy()
    )
    row_ids = clean["row"].to_numpy().astype(np.int64)[chosen]
    saved_by_arm = {
        a: score.load_network(arm=a, month=month, seed=seed, device=device) for a in arms
    }
    any_saved = saved_by_arm[arms[0]][1]
    _train, test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=1, saved=any_saved
    )
    sub = subset(pool=test, idx=torch.tensor(row_ids, device=device))
    base_power, _ = fit.world_power(seed=None)
    table = pl.read_parquet(fit.lagless.table_path(month=month, seed=seed))
    neighbours = rows_mod.neighbour_matrix(
        sites=fit.prep.primaries()["site"].to_list(), path=fit._neighbour_path()
    )
    metered = torch.tensor(fit.metered_indicator(), device=device)
    origins = np.unique(sub.origin.cpu().numpy())
    records = []
    variants = [(a, ()) for a in arms] + [(a, (0, 1, 2)) for a in arms if a.startswith("B")]
    nets = {a: saved_by_arm[a][0] for a in arms}
    trust_scalers = {a: saved_by_arm[a][1] for a in arms}

    def evaluate(
        *, power_np: np.ndarray, rows: torch.Tensor
    ) -> dict[tuple[str, tuple], np.ndarray]:
        stack = rows_mod.build_stack(
            power=power_np, table=table, neighbours=neighbours, device=device
        )
        power = torch.tensor(power_np, device=device)
        pool = subset(pool=sub, idx=rows)
        out = {}
        for arm, masked in variants:
            scalers.trust_mean = trust_scalers[arm]["trust_mean"]
            scalers.trust_std = trust_scalers[arm]["trust_std"]
            out[(arm, masked)] = band(
                net=nets[arm], pool=pool, power=power, stack=stack, lag_stats=lag_stats,
                scalers=scalers, metered=metered, device=device,
                masked=(0,) if arm == "B5" and not masked else masked,
            )  # fmt: skip
        return out

    for origin in origins:
        origin_rows = torch.nonzero(sub.origin == int(origin)).flatten()
        baseline = evaluate(power_np=base_power, rows=origin_rows)
        for p in range(16):
            rows = origin_rows[sub.primary[origin_rows] == p]
            if len(rows) == 0:
                continue
            newest = int(origin) - 7
            for k, span in SPAN_STEPS.items():
                for step in STEPS:
                    shifted = base_power.copy()
                    shifted[p, newest - span + 1 : newest + 1] += step
                    result = evaluate(power_np=shifted, rows=rows)
                    local = torch.isin(origin_rows, rows).cpu().numpy()
                    lead = sub.lead_hours[rows].cpu().numpy()
                    band_id = np.digitize(lead, [48.0, 120.0])
                    for key, q in result.items():
                        b = baseline[key][local]
                        records.append(
                            pl.DataFrame(
                                {
                                    "arm": key[0] + (f"-{CONTROL}" if key[1] else ""),
                                    "k": k,
                                    "step": step,
                                    "lead_band": band_id,
                                    "primary": p,
                                    "shift": (q[:, 1] - b[:, 1]) / step,
                                    "width_ratio": (q[:, 2] - q[:, 0]) / (b[:, 2] - b[:, 0]),
                                }
                            )
                        )
    return pl.concat(records)


def main() -> None:
    """Run the sweep and print its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    frame = run(month=args.month, seed=args.seed, arms=args.arms, device=args.device)
    out = fit.WORK_DIR / "sweep" / f"{args.month}_s{args.seed}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out)
    summary = frame.group_by("arm", "k", "step", "lead_band").agg(
        pl.len().alias("rows"),
        pl.col("shift").mean().alias("shift_share_of_step"),
        pl.col("width_ratio").mean().alias("width_ratio"),
    )
    pl.Config.set_tbl_rows(200)
    print(summary.sort("arm", "k", "step", "lead_band"))
    print(f"-> {Path(out)}")


if __name__ == "__main__":
    main()

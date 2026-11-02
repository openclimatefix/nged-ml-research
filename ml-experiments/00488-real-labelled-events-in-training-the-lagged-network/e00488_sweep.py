"""Readings-level encroachment sweep and bad-data injections for experiment 488 (plain networks).

Experiment 404's sweep (`e00404_sweep.py`) adds a step of 5% of capacity (both signs) to the
Primary's power over the span that holds the `k` newest lags (k = 1: the last 3 hours before the
newest usable reading; k = 2: 24 hours; k = 3: 7 days), rebuilds the three lags from the shifted
power, and reports the shift of the median forecast as a share of the step and the ratio of the 90%
band's width to its width without the step. This module does the same for the plain networks of
experiment 488 (no trust inputs, so no stack is built), for the stored `A0-inj`, and for the 50/50
mixture with the lagless table (`+LL` and `B-mix`). It adds the three bad-data injections of
experiment 465, on the readings that reach the lags: `stuck` (the 48 newest usable readings repeat
the reading just before them), `dropout` (the 12 newest usable readings are zero), and `spike` (the
4 newest usable readings rise by 4 or 20 times the Primary's median absolute change from the same
half-hour a week earlier, over the 8 weeks before). Only the row's own Primary is changed, one
Primary at a time, on the clean-world rows of every third origin day that lie outside every padded
event and the 96 half-hours after it.

```bash
PYTHONPATH=<as run488.sh> DATA_PATH_INTERNAL=<private data> OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00488-*/e00488_sweep.py run --month 2026-06 --seed 0 \
    --arms A0-inj RealMix PT-FT RealMix-shuf
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00404_sweep as sweep404  # ty: ignore[unresolved-import]
import e00488_fit as fit488  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

fit = fit488.fit
ORIGIN_STRIDE: Final[int] = 3
STEP_SHARE: Final[float] = 0.05
STUCK_STEPS: Final[int] = 48
DROPOUT_STEPS: Final[int] = 12
SPIKE_STEPS: Final[int] = 4
SPIKE_MULTIPLES: Final[tuple[int, ...]] = (4, 20)
WEEK: Final[int] = 7 * common.STEPS_PER_DAY
OUT_DIR: Final[Path] = fit488.WORK_DIR / "sweep"


def week_spread(*, power: np.ndarray, end: int) -> float:
    """Return the median absolute change from the same half-hour a week earlier, up to `end`."""
    lo = max(WEEK, end - 8 * WEEK)
    diff = np.abs(power[lo:end] - power[lo - WEEK : end - WEEK])
    return float(np.nanmedian(diff[np.isfinite(diff)]))


def injections(
    *, power: np.ndarray, primary: int, origin: int
) -> list[tuple[str, int, float, np.ndarray]]:
    """Return (kind, k, step, new power) for every injection into one Primary at one origin.

    `step` is the nominal corruption in capacity fractions (0 for stuck and dropout).
    """
    end = origin - common.NEWEST_AGE
    out = []
    for k, span in sweep404.SPAN_STEPS.items():
        for sign in (1.0, -1.0):
            new = power.copy()
            new[primary, end - span + 1 : end + 1] += sign * STEP_SHARE
            out.append(("step", k, sign * STEP_SHARE, new))
    new = power.copy()
    new[primary, end - STUCK_STEPS + 1 : end + 1] = power[primary, end - STUCK_STEPS]
    out.append(("stuck", 0, 0.0, new))
    new = power.copy()
    new[primary, end - DROPOUT_STEPS + 1 : end + 1] = 0.0
    out.append(("dropout", 0, 0.0, new))
    spread = week_spread(power=power[primary], end=end)
    for m in SPIKE_MULTIPLES:
        new = power.copy()
        new[primary, end - SPIKE_STEPS + 1 : end + 1] += np.float32(m * spread)
        out.append((f"spike{m}x", 0, float(m * spread), new))
    return out


@torch.no_grad()
def quantiles(
    *,
    net: torch.nn.Module,
    pool: fit.Pool,
    power: torch.Tensor,
    lag_stats: np.ndarray,
    scalers: fit.Scalers,
    device: str,
    mix: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return q05, q50, q95 (rows, 3) in capacity fractions, and the raw lag readings (rows, 3)."""
    lags, lag_grid = fit.lag_features(pool=pool, power=power, lag_stats=lag_stats, device=device)
    log_w, mean, std = net(pool.base, lags, pool.primary)
    if mix:
        ll_w, ll_mean, ll_std = models.mixture_parts(out=pool.ll, components=models.COMPONENTS)
        half = float(np.log(2.0))
        log_w = torch.cat([log_w - half, ll_w - half], dim=1)
        mean, std = torch.cat([mean, ll_mean], dim=1), torch.cat([std, ll_std], dim=1)
    mu = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    sd = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    q = score.mixture_quantiles_capacity(
        log_w=log_w, mean=mean, std=std, mu=mu[pool.primary], sd=sd[pool.primary]
    )
    size = power.shape[-1]
    readings = power[pool.primary[:, None], lag_grid.clamp(0, size - 1)]
    return q[:, [1, 5, 9]].cpu().numpy(), readings.cpu().numpy()


def run(*, month: str, seed: int, arms: list[str], device: str) -> pl.DataFrame:
    """Run the sweep for the arms (`X+LL` and `B-mix` are mixtures) and return one row per case."""
    clean = pl.read_parquet(
        score.prediction_path(arm="A0-inj", month=month, seed=seed, world=-1)
    ).with_row_index("row")
    day = clean["origin"].to_numpy().astype(np.int64) // 48
    chosen = (day % ORIGIN_STRIDE == 0) & ~(
        clean["tgt_real"].to_numpy() | clean["tgt_post"].to_numpy() | clean["win_real"].to_numpy()
    )
    row_ids = clean["row"].to_numpy().astype(np.int64)[chosen]
    names = {a: ("A0-inj" if a == "B-mix" else a.removesuffix("+LL")) for a in arms}
    link = fit.MODEL_DIR / "A0-inj"
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(fit488.STORED_DIR / "A0-inj", target_is_directory=True)
    saved = {
        n: score.load_network(arm=n, month=month, seed=seed, device=device)
        for n in set(names.values())
    }
    _train, test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=1, saved=saved[names[arms[0]]][1]
    )
    sub = sweep404.subset(pool=test, idx=torch.tensor(row_ids, device=device))
    base_power, _ = fit.world_power(seed=None)
    base_t = torch.tensor(base_power, device=device)
    records = []
    for origin in np.unique(sub.origin.cpu().numpy()):
        origin_rows = torch.nonzero(sub.origin == int(origin)).flatten()
        for p in range(16):
            rows = origin_rows[sub.primary[origin_rows] == p]
            if len(rows) == 0:
                continue
            pool = sweep404.subset(pool=sub, idx=rows)
            lead = pool.lead_hours.cpu().numpy()
            band_id = np.digitize(lead, [48.0, 120.0])
            actual = pool.y.cpu().numpy()
            base = {
                a: quantiles(net=saved[names[a]][0], pool=pool, power=base_t, lag_stats=lag_stats,
                             scalers=scalers, device=device, mix=a.endswith("+LL") or a == "B-mix")
                for a in arms
            }  # fmt: skip
            for kind, k, step, shifted in injections(
                power=base_power, primary=p, origin=int(origin)
            ):
                shifted_t = torch.tensor(shifted, device=device)
                for a in arms:
                    q, readings = quantiles(
                        net=saved[names[a]][0], pool=pool, power=shifted_t, lag_stats=lag_stats,
                        scalers=scalers, device=device, mix=a.endswith("+LL") or a == "B-mix",
                    )  # fmt: skip
                    b, b_readings = base[a]
                    change = readings - b_readings
                    corruption = change[np.arange(len(change)), np.abs(change).argmax(1)]
                    shift = q[:, 1] - b[:, 1]
                    records.append(
                        pl.DataFrame(
                            {
                                "arm": a, "kind": kind, "k": k, "step": step, "lead_band": band_id,
                                "primary": p, "shift": shift, "corruption": corruption,
                                "width_ratio": (q[:, 2] - q[:, 0]) / (b[:, 2] - b[:, 0]),
                                "covered": ((q[:, 0] <= actual) & (actual <= q[:, 2])).astype(float),  # noqa: E501
                                "covered_event": (
                                    (q[:, 0] <= actual + step) & (actual + step <= q[:, 2])
                                ).astype(float),
                            }
                        )
                    )  # fmt: skip
    return pl.concat(records)


def summarise(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the shift as a share of the step (or of the corruption), width ratio, and coverage."""
    return (
        frame.with_columns(
            share=pl.when(pl.col("step") != 0.0)
            .then(pl.col("shift") / pl.col("step"))
            .when(pl.col("corruption").abs() > 1e-6)
            .then(pl.col("shift") / pl.col("corruption"))
            .otherwise(None)
        )
        .group_by("arm", "kind", "k", "step", "lead_band")
        .agg(
            pl.len().alias("rows"),
            pl.col("share").mean().alias("shift_share"),
            pl.col("width_ratio").mean(),
            pl.col("covered").mean().alias("coverage_clean"),
            pl.col("covered_event").mean().alias("coverage_event"),
        )
        .sort("arm", "kind", "k", "step", "lead_band")
    )


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.month}_s{args.seed}.parquet"
    frame.write_parquet(out)
    pl.Config.set_tbl_rows(300)
    print(summarise(frame=frame))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

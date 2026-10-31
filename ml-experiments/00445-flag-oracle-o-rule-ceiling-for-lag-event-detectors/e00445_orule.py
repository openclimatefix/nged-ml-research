"""Build the flag-oracle mixtures (O-rule) of experiment 445 from experiment 404's two heads.

The lagged head is experiment 404's saved A0-inj network. The lagless head is the frozen
cross-fitted lagless table of the same cell. B-mix is their fixed 50/50 density mixture. For each
row the O-rule counts how many of the row's 3 lags lie inside a load-shift event (exact event codes
on an injected world, the padded real-event mask on the clean world -1) and then:

- hard: uses the lagless head when the count is 1 or more, and the lagged head when it is 0;
- soft: mixes the two densities with weight count / 3 on the lagless head.

Only forward passes run (CPU, one thread). A third arm, `O-lagless`, is the lagless head alone on
every row. Each arm is written in experiment 404's layout, one
parquet per month, fit seed, and world, with the row keys, the class flags of experiment 404, the
per-lag flags `flag_0` to `flag_2`, and q01 to q99.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
<experiment 404 folder> \
DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00445-*/e00445_orule.py build --month 2026-06 --seed 0
```
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00445_fit as fit  # ty: ignore[unresolved-import]
import e00445_models as models  # ty: ignore[unresolved-import]
import e00445_rows as rows_mod  # ty: ignore[unresolved-import]
import e00445_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00445-flag-oracle-o-rule-ceiling-for-lag-event-detectors" / "work"
)
PRED_DIR: Final[Path] = OUT_DIR / "predictions"
ARMS: Final[tuple[str, ...]] = ("O-hard", "O-soft", "O-lagless")
CELLS: Final[tuple[tuple[str, int], ...]] = (
    ("2026-06", 0),
    ("2026-03", 0),
    ("2026-01", 0),
    ("2025-12", 0),
    ("2025-10", 0),
    ("2026-06", 1),
    ("2026-06", 2),
)
BATCH: Final[int] = 16384
CHECK_ROWS: Final[int] = 20000
"""Rows per cell and world on which the recomputed 50/50 mixture is compared with stored B-mix."""
LEVEL_NAMES: Final[tuple[str, ...]] = score.LEVEL_NAMES


def prediction_path(*, arm: str, month: str, seed: int, world: int) -> Path:
    """Return where an O-rule arm's forecast frame is stored."""
    return PRED_DIR / arm / f"{month}_s{seed}_w{world}.parquet"


def lag_flags(*, frame: pl.DataFrame, code: np.ndarray, real: np.ndarray, world: int) -> np.ndarray:
    """Return the (rows, 3) boolean flags of the newest lag, the day lag, and the week lag.

    A lag is flagged when its valid time lies inside an injected event (world 200 to 202) or inside
    a padded real event (clean world -1), as experiment 404's `lag_inj` and `lag_real` flags do.
    """
    primary = frame["primary"].to_numpy().astype(np.int64)
    grid = rows_mod.lag_grids(
        target=frame["target"].to_numpy().astype(np.int64),
        origin=frame["origin"].to_numpy().astype(np.int64),
    )
    safe = np.clip(grid, 0, real.shape[1] - 1)
    if world < 0:
        return real[primary[:, None], safe]
    return code[primary[:, None], safe] != 0


def quantiles_of(
    *,
    log_w: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    mu: torch.Tensor,
    sd: torch.Tensor,
    primary: torch.Tensor,
) -> np.ndarray:
    """Return quantiles in capacity fractions for mixtures in z units, in batches."""
    out = []
    for start in range(0, len(log_w), BATCH):
        sl = slice(start, start + BATCH)
        out.append(
            score.mixture_quantiles_capacity(
                log_w=log_w[sl], mean=mean[sl], std=std[sl], mu=mu[primary[sl]], sd=sd[primary[sl]]
            )
        )
    return torch.cat(out).numpy() if out else np.zeros((0, len(LEVEL_NAMES)))


@dataclass
class Heads:
    """The lagged head and the lagless head of a cell's rows, as mixtures in z units.

    Attributes:
        lagged: Log weights, means, and standard deviations of the lagged head (rows, components).
        lagless: The same for the lagless head.
        primary: Primary index per row.
        mu: Per-Primary target mean of the network's scaler.
        sd: Per-Primary target standard deviation of the network's scaler.
    """

    lagged: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    lagless: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    primary: torch.Tensor
    mu: torch.Tensor
    sd: torch.Tensor

    def mix(self, *, rows: np.ndarray, share: float) -> np.ndarray:
        """Return quantiles of the mixture with weight `share` on the lagless head, for `rows`."""
        sel = torch.from_numpy(rows)
        if share <= 0.0:
            parts = tuple(t[sel] for t in self.lagged)
        elif share >= 1.0:
            parts = tuple(t[sel] for t in self.lagless)
        else:
            w = torch.cat(
                [self.lagged[0][sel] + np.log(1.0 - share), self.lagless[0][sel] + np.log(share)],
                dim=1,
            )
            m = torch.cat([self.lagged[1][sel], self.lagless[1][sel]], dim=1)
            s = torch.cat([self.lagged[2][sel], self.lagless[2][sel]], dim=1)
            parts = (w, m, s)
        return quantiles_of(
            log_w=parts[0], mean=parts[1], std=parts[2], mu=self.mu, sd=self.sd,
            primary=self.primary[sel],
        )  # fmt: skip


def load_a0_inj(*, month: str, seed: int) -> tuple[torch.nn.Module, dict]:
    """Load the saved A0-inj network on the CPU, in evaluation mode (dropout off)."""
    saved = torch.load(
        fit.MODEL_DIR / "A0-inj" / f"{month}_s{seed}.pt", map_location="cpu", weights_only=False
    )
    net = fit.build_network(arm="A0-inj", n_base=saved["n_base"], n_trust=rows_mod.N_TRUST)
    net.load_state_dict(saved["state"])
    net.eval()
    return net, saved


@torch.no_grad()
def lagged_head(
    *, net: torch.nn.Module, test: fit.Pool, power: torch.Tensor, scalers: fit.Scalers,
    lag_stats: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """Return the lagged mixture of every test row for a world's power."""
    lags, lag_grid = fit.lag_features(pool=test, power=power, lag_stats=lag_stats, device="cpu")
    metered = torch.tensor(fit.metered_indicator())
    parts = []
    for start in range(0, len(test), BATCH):
        idx = torch.arange(start, min(start + BATCH, len(test)))
        parts.append(
            fit.forward(
                net=net, pool=test, idx=idx, lags=lags, stack=None, lag_grid=lag_grid,
                scalers=scalers, metered=metered,
            )
        )  # fmt: skip
    return (
        torch.cat([p[0] for p in parts]),
        torch.cat([p[1] for p in parts]),
        torch.cat([p[2] for p in parts]),
    )


def o_rule_quantiles(
    *, heads: Heads, stored: np.ndarray, lagless_all: np.ndarray, count: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the hard and soft O-rule quantiles given the number of flagged lags per row."""
    hard = stored.copy()
    soft = stored.copy()
    flagged = np.flatnonzero(count >= 1)
    hard[flagged] = lagless_all[flagged]
    for n_flagged in (1, 2, 3):
        rows = np.flatnonzero(count == n_flagged)
        if len(rows):
            soft[rows] = heads.mix(rows=rows, share=n_flagged / 3.0)
    return hard, soft


def build_cell(*, month: str, seed: int, real: np.ndarray, worlds: list[int]) -> None:
    """Write the three arms for one cell and each world, after checking the heads."""
    net, saved = load_a0_inj(month=month, seed=seed)
    _train, test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device="cpu", stride=1, saved=saved
    )
    mu = torch.tensor(scalers.target_mean, dtype=torch.float32)
    sd = torch.tensor(scalers.target_std, dtype=torch.float32)
    lagless = models.mixture_parts(out=test.ll, components=models.COMPONENTS)
    names = list(LEVEL_NAMES)
    lagless_all = quantiles_of(
        log_w=lagless[0], mean=lagless[1], std=lagless[2], mu=mu, sd=sd, primary=test.primary
    ).astype(np.float64)
    rng = np.random.default_rng(0)
    check = np.sort(rng.choice(len(test), size=min(CHECK_ROWS, len(test)), replace=False))
    for world in worlds:
        started = time.time()
        power_np, code = fit.world_power(seed=None if world < 0 else world)
        stored = pl.read_parquet(
            score.prediction_path(arm="A0-inj", month=month, seed=seed, world=world)
        )
        for key in ("primary", "origin", "target"):
            rebuilt = getattr(test, key).numpy()
            assert np.array_equal(stored[key].to_numpy(), rebuilt), f"{key} differs from stored"
        flags = lag_flags(frame=stored, code=code, real=real, world=world)
        count = flags.sum(1)
        lagged = lagged_head(
            net=net, test=test, power=torch.tensor(power_np), scalers=scalers, lag_stats=lag_stats
        )
        heads = Heads(lagged=lagged, lagless=lagless, primary=test.primary, mu=mu, sd=sd)
        q_lagged = stored[names].to_numpy().astype(np.float64)
        bmix = pl.read_parquet(
            score.prediction_path(arm="B-mix", month=month, seed=seed, world=world)
        )
        lagged_err = np.abs(heads.mix(rows=check, share=0.0) - q_lagged[check]).max()
        bmix_err = np.abs(heads.mix(rows=check, share=0.5) - bmix[names].to_numpy()[check]).max()
        print(
            f"{month} s{seed} w{world}: max |recomputed - stored| A0-inj {lagged_err:.2e}, "
            f"B-mix {bmix_err:.2e} (capacity fractions)"
        )
        hard, soft = o_rule_quantiles(
            heads=heads, stored=q_lagged, lagless_all=lagless_all, count=count
        )
        out = stored.select([c for c in stored.columns if c not in names]).with_columns(
            flag_0=pl.Series(flags[:, 0]),
            flag_1=pl.Series(flags[:, 1]),
            flag_2=pl.Series(flags[:, 2]),
            n_flagged=pl.Series(count.astype(np.int8)),
        )
        for arm, q in (("O-hard", hard), ("O-soft", soft), ("O-lagless", lagless_all)):
            frame = out.with_columns(**{n: pl.Series(q[:, i]) for i, n in enumerate(names)})
            target = prediction_path(arm=arm, month=month, seed=seed, world=world)
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(target)
        print(
            f"  wrote {len(test):,} rows, flagged rows with 1/2/3 lags: "
            f"{(count == 1).sum():,}/{(count == 2).sum():,}/{(count == 3).sum():,}, "
            f"{time.time() - started:.0f} s",
            flush=True,
        )


def main() -> None:
    """Build the O-rule arms for the requested cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--month", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worlds", type=int, nargs="*", default=[*score.SCORING_SEEDS, -1])
    args = parser.parse_args()
    torch.set_num_threads(1)
    real = score.real_event_mask()
    cells = [(args.month, args.seed)] if args.month else list(CELLS)
    for month, seed in cells:
        build_cell(month=month, seed=seed, real=real, worlds=args.worlds)


if __name__ == "__main__":
    main()

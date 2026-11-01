"""Score the lag-dropout network in five passes and cache every head the arms need.

For each month and fit seed, on the real world and on the injected scoring worlds (experiment 428's
seeds 200 to 202), the saved S1d network is scored with all lags, with each lag alone, and with no
lags. The output per (month, seed, world) is a table with the row keys, the experiment 428 row
flags, the actual, the 11 quantiles of the all-lags pass, the median of each pass, and the two
spread signals. For the real world the mixture parameters of the all-lags and no-lags passes are
cached, together with the heads of experiment 404's stored A0-inj and S1 networks and the frozen
lagless table, in capacity fractions, so the arms are built without re-running any network.

Spread signals: `spread_mean` is the mean over the four other passes of the absolute difference
between that pass's median and the all-lags median, divided by the all-lags q95 minus q05;
`spread_max` is the maximum over the ten pairs of passes of the absolute difference in median,
divided by the same width.

```bash
source /tmp/e460_env.sh
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00460-*/e00460_heads.py --month 2026-06
```
"""

import argparse
import itertools
from pathlib import Path
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00460_models as net_mod  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00460-lag-dropout-network-forecast-disagreement-as-a-trust-and-width-signal"
    / "work"
)
MODEL_DIR: Final[Path] = WORK_DIR / "models"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
WORLDS: Final[tuple[int, ...]] = (-1, *score.SCORING_SEEDS)
BATCH: Final[int] = 16384
WIDTH_FLOOR: Final[float] = 1e-6
KEY_COLUMNS: Final[tuple[str, ...]] = ("primary", "origin", "target")
Head = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def passes_path(*, month: str, seed: int, world: int) -> Path:
    """Return where the five-pass table of a cell is stored."""
    return WORK_DIR / "passes" / f"{month}_s{seed}_w{world}.parquet"


def heads_path(*, kind: str, month: str, seed: int) -> Path:
    """Return where the real-world heads of a kind ("Sd", "A", "S", "L") are cached."""
    return WORK_DIR / "heads" / kind / f"{month}_s{seed}.pt"


def load_net(*, arm: str, month: str, seed: int, device: str) -> tuple[torch.nn.Module, dict]:
    """Load the S1d network (this experiment's folder) or a stored A0-inj or S1 network (404's)."""
    folder = MODEL_DIR if arm == "S1d" else fit.MODEL_DIR
    saved = torch.load(folder / arm / f"{month}_s{seed}.pt", weights_only=False)
    if arm == "S1d":
        net = net_mod.DropoutS1Mdn(n_base=saved["n_base"], n_primaries=16)
    else:
        net = models.PlainLagMdn(
            n_base=saved["n_base"], n_primaries=16, n_extra=3 if arm == "S1" else 0
        )
    net.load_state_dict(saved["state"])
    return net.to(device).eval(), saved


@torch.no_grad()
def pass_head(
    *,
    net: torch.nn.Module,
    test: fit.Pool,
    lags: torch.Tensor,
    scalers: fit.Scalers,
    hidden: tuple[int, ...],
) -> Head:
    """Return the network's mixture in capacity fractions, with the lags in `hidden` masked."""
    parts = []
    mask = torch.zeros(3, dtype=torch.bool, device=lags.device)
    mask[list(hidden)] = True
    for start in range(0, len(test), BATCH):
        idx = torch.arange(start, min(start + BATCH, len(test)), device=lags.device)
        rows = net_mod.mask_lags(lags=lags[idx], hidden=mask.expand(len(idx), 3))
        parts.append(net(test.base[idx], rows, test.primary[idx]))
    log_w, mean, std = (torch.cat([p[i] for p in parts]).double() for i in range(3))
    mu = torch.tensor(scalers.target_mean, device=lags.device)[test.primary][:, None]
    sd = torch.tensor(scalers.target_std, device=lags.device)[test.primary][:, None]
    return log_w, mean * sd + mu, std * sd


@torch.no_grad()
def quantiles_of(*, head: Head) -> torch.Tensor:
    """Return the sorted 11 quantiles (rows, 11) in capacity fractions of a head."""
    out = []
    for start in range(0, len(head[0]), BATCH):
        sl = slice(start, start + BATCH)
        out.append(models.mixture_quantiles(log_w=head[0][sl], mean=head[1][sl], std=head[2][sl]))
    return torch.sort(torch.cat(out), dim=1).values


def lagless_head(*, month: str, seed: int, primary: np.ndarray, target: np.ndarray) -> Head:
    """Return the frozen lagless table's mixture in capacity fractions for the rows."""
    arrays = fit.table_arrays(table=pl.read_parquet(lagless.table_path(month=month, seed=seed)))
    pos = target - arrays.g0
    inside = (pos >= 0) & (pos < arrays.raw.shape[1])
    pos = np.clip(pos, 0, arrays.raw.shape[1] - 1)
    assert (inside & arrays.present[primary, pos]).all(), f"lagless seed {seed} misses rows"
    raw = torch.from_numpy(arrays.raw[primary, pos].astype(np.float64))
    t_mean = torch.from_numpy(arrays.y_mean[primary, pos].astype(np.float64))[:, None]
    t_std = torch.from_numpy(arrays.y_std[primary, pos].astype(np.float64))[:, None]
    return (
        torch.log_softmax(raw[:, :3], dim=1),
        raw[:, 3:6] * t_std + t_mean,
        torch.exp(raw[:, 6:9].clamp(-3.0, 3.0)) * t_std,
    )


def lag_inputs(
    *,
    arm: str,
    month: str,
    seed: int,
    world: int,
    test: fit.Pool,
    lag_stats: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Return the lag tensor of a world for the rows, the world's power, and its event codes."""
    power_np, code = fit.world_power(seed=None if world < 0 else world)
    surprise = None
    if arm in ("S1", "S1d"):
        table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
        surprise = fit.surprise_array(power=power_np, table=table)
    lags, _grid = fit.lag_features(
        pool=test, power=torch.tensor(power_np, device=device), lag_stats=lag_stats,
        device=device, surprise=surprise,
    )  # fmt: skip
    return lags, power_np, code


def spread_columns(*, medians: dict[str, np.ndarray], width: np.ndarray) -> dict[str, np.ndarray]:
    """Return `spread_mean` and `spread_max` from the five pass medians and the all-lags width."""
    width = np.maximum(width, WIDTH_FLOOR)
    names = list(net_mod.PASSES)
    others = [np.abs(medians[n] - medians["all"]) for n in names if n != "all"]
    pairs = [np.abs(medians[a] - medians[b]) for a, b in itertools.combinations(names, 2)]
    return {
        "spread_mean": np.mean(others, axis=0) / width,
        "spread_max": np.max(pairs, axis=0) / width,
    }


def score_cell(*, month: str, seed: int, device: str, real: np.ndarray) -> dict[str, np.ndarray]:
    """Write the five-pass tables of one cell and cache the real-world heads; return the keys."""
    net, saved = load_net(arm="S1d", month=month, seed=seed, device=device)
    _train, test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=1, saved=saved
    )
    keys = {k: getattr(test, k).cpu().numpy() for k in KEY_COLUMNS}
    for world in WORLDS:
        lags, power_np, code = lag_inputs(
            arm="S1d", month=month, seed=seed, world=world, test=test, lag_stats=lag_stats,
            device=device,
        )  # fmt: skip
        heads = {
            name: pass_head(net=net, test=test, lags=lags, scalers=scalers, hidden=hidden)
            for name, hidden in net_mod.PASSES.items()
        }
        medians = {n: quantiles_of(head=h)[:, 5].cpu().numpy() for n, h in heads.items()}
        quant = quantiles_of(head=heads["all"]).cpu().numpy()
        width = quant[:, 9] - quant[:, 1]
        flags = score.Flags(code=code, real=real, power=power_np, device=device).of(
            primary=test.primary, origin=test.origin, target=test.target
        )
        columns: dict[str, np.ndarray] = {
            "primary": keys["primary"].astype(np.int16),
            "origin": keys["origin"].astype(np.int32),
            "target": keys["target"].astype(np.int32),
            "lead_hours": test.lead_hours.cpu().numpy(),
            "actual": test.y.cpu().numpy(),
            **{name: quant[:, i] for i, name in enumerate(score.LEVEL_NAMES)},
            **{f"med_{n}": m for n, m in medians.items()},
            **spread_columns(medians=medians, width=width),
            **{k: v.cpu().numpy() for k, v in flags.items()},
        }
        target = passes_path(month=month, seed=seed, world=world)
        target.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(columns).write_parquet(target)
        if world < 0:
            path = heads_path(kind="Sd", month=month, seed=seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"all": [t.cpu() for t in heads["all"]], "none": [t.cpu() for t in heads["none"]],
                 "keys": keys}, path,
            )  # fmt: skip
        print(f"{month} s{seed} w{world}: {test.primary.shape[0]:,} rows", flush=True)
    return keys


def cache_reference_heads(
    *, month: str, seed: int, keys: dict[str, np.ndarray], device: str
) -> None:
    """Cache the stored A0-inj and S1 heads and the lagless head of a cell (real world)."""
    for kind, arm in (("A", "A0-inj"), ("S", "S1")):
        net, saved = load_net(arm=arm, month=month, seed=seed, device=device)
        _t, test, scalers, _a, lag_stats = fit.build_pools(
            month=month, seed=seed, device=device, stride=1, saved=saved
        )
        for k in KEY_COLUMNS:
            assert np.array_equal(getattr(test, k).cpu().numpy(), keys[k]), f"{arm} {k} differs"
        lags, _p, _c = lag_inputs(
            arm=arm, month=month, seed=seed, world=-1, test=test, lag_stats=lag_stats, device=device
        )
        head = pass_head(net=net, test=test, lags=lags, scalers=scalers, hidden=())
        path = heads_path(kind=kind, month=month, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"head": [t.cpu() for t in head], "keys": keys}, path)
    head = lagless_head(
        month=month, seed=seed, primary=keys["primary"].astype(np.int64),
        target=keys["target"].astype(np.int64),
    )  # fmt: skip
    path = heads_path(kind="L", month=month, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head, "keys": keys}, path)


def main() -> None:
    """Score and cache one month for the given seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, choices=MONTHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    real = score.real_event_mask()
    for seed in args.seeds:
        keys = score_cell(month=args.month, seed=seed, device=args.device, real=real)
        cache_reference_heads(month=args.month, seed=seed, keys=keys, device=args.device)


if __name__ == "__main__":
    main()

"""Build the mixture and ensemble constructs of experiment 454 from experiment 404's networks.

A head is a Gaussian mixture of 3 components in capacity fractions, for every row of a scored month
on the real (clean) world. The lagged heads are experiment 404's A0-inj and S1 networks (fit seeds
0 to 4) and the lagless head is the frozen cross-fitted lagless table of the same seed. A construct
is the equal-weight density mixture of several heads, with its 11 quantiles solved exactly. A
construct is scored in one or more cells; a cell of a two-head construct uses a seed (or a seed
pair), so the cells give the seed spread.

Heads are cached under `work/heads/`, constructs are written under `work/predictions/<arm>/` in
experiment 404's layout (`<month>_s<cell>_w-1.parquet`), and `work/checks/<month>.txt` records the
comparison of recomputed heads with the stored 404 forecasts.

```bash
<PYTHONPATH as e00404_fit.py shows> DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00454-*/e00454_build.py --month 2026-06
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00454-lag-trust-mixture-and-ensemble-constructs-for-lagged-and-lagless-heads"
    / "work"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
CELL_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
PAIRS: Final[tuple[tuple[int, ...], ...]] = ((0, 1), (0, 2), (1, 2))
BATCH: Final[int] = 16384
CHECK_TOLERANCE: Final[float] = 5e-4
"""Largest accepted difference (capacity fraction) between a recomputed and a stored quantile."""
KEY_COLUMNS: Final[tuple[str, ...]] = ("primary", "origin", "target")
LAGGED_ARMS: Final[dict[str, str]] = {"A": "A0-inj", "S": "S1"}
type Head = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def head_path(*, kind: str, month: str, seed: int) -> Path:
    """Return where a cached head ("A", "S", or "L") is stored."""
    return WORK_DIR / "heads" / kind / f"{month}_s{seed}.pt"


def prediction_path(*, arm: str, month: str, cell: int) -> Path:
    """Return where a construct's forecast frame is stored."""
    return WORK_DIR / "predictions" / arm / f"{month}_s{cell}_w-1.parquet"


@torch.no_grad()
def lagged_head(*, arm: str, month: str, seed: int) -> tuple[Head, dict[str, np.ndarray]]:
    """Return a saved network's mixture (capacity fractions, real world) and the row keys."""
    net, saved = score.load_network(arm=arm, month=month, seed=seed, device="cpu")
    _train, test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device="cpu", stride=1, saved=saved
    )
    power_np, _code = fit.world_power(seed=None)
    table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
    lags, lag_grid = fit.lag_features(
        pool=test,
        power=torch.tensor(power_np),
        lag_stats=lag_stats,
        device="cpu",
        surprise=fit.surprise_array(power=power_np, table=table)
        if arm == fit.SURPRISE_ARM
        else None,
    )
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
    log_w, mean, std = (torch.cat([p[i] for p in parts]).double() for i in range(3))
    mu = torch.tensor(scalers.target_mean)[test.primary][:, None]
    sd = torch.tensor(scalers.target_std)[test.primary][:, None]
    keys = {k: getattr(test, k).numpy() for k in KEY_COLUMNS}
    return (log_w, mean * sd + mu, std * sd), keys


def lagless_head(*, month: str, seed: int, primary: np.ndarray, target: np.ndarray) -> Head:
    """Return a lagless table's mixture in capacity fractions for the rows (primary, target)."""
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


def mix_equal(*, heads: list[Head]) -> np.ndarray:
    """Return the sorted 11 quantiles (rows, 11) of the equal-weight mixture of the heads."""
    k = len(heads)
    log_w = torch.cat([h[0] - float(np.log(k)) for h in heads], dim=1)
    mean = torch.cat([h[1] for h in heads], dim=1)
    std = torch.cat([h[2] for h in heads], dim=1)
    out = []
    for start in range(0, len(log_w), BATCH):
        sl = slice(start, start + BATCH)
        out.append(models.mixture_quantiles(log_w=log_w[sl], mean=mean[sl], std=std[sl]))
    return torch.sort(torch.cat(out), dim=1).values.numpy().astype(np.float64)


def load_heads(*, month: str) -> tuple[dict[str, dict[int, Head]], dict[str, np.ndarray]]:
    """Return every head of a month ("A", "S", "L" by fit seed) and the shared row keys.

    Every head must be on identical rows; a mismatch stops the build.
    """
    heads: dict[str, dict[int, Head]] = {"A": {}, "S": {}, "L": {}}
    keys: dict[str, np.ndarray] | None = None
    for seed in SEEDS:
        for kind, arm in LAGGED_ARMS.items():
            path = head_path(kind=kind, month=month, seed=seed)
            if path.exists():
                saved = torch.load(path, weights_only=False)
                head, these = saved["head"], saved["keys"]
            else:
                head, these = lagged_head(arm=arm, month=month, seed=seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"head": head, "keys": these}, path)
            if keys is None:
                keys = these
            for k in KEY_COLUMNS:
                assert np.array_equal(keys[k], these[k]), f"{arm} s{seed} {month}: {k} differs"
            heads[kind][seed] = head
        assert keys is not None
        heads["L"][seed] = lagless_head(
            month=month, seed=seed, primary=keys["primary"].astype(np.int64),
            target=keys["target"].astype(np.int64),
        )  # fmt: skip
    assert keys is not None
    return heads, keys


def constructs(*, heads: dict[str, dict[int, Head]]) -> dict[str, list[list[Head]]]:
    """Return each construct's cells; a cell is the list of heads mixed with equal weights."""
    a, s, ll = heads["A"], heads["S"], heads["L"]
    out: dict[str, list[list[Head]]] = {
        "A0-inj": [[a[c]] for c in CELL_SEEDS],
        "S1": [[s[c]] for c in CELL_SEEDS],
        "LL": [[ll[c]] for c in CELL_SEEDS],
        "B-mix": [[a[c], ll[c]] for c in CELL_SEEDS],
        "S1-mix": [[s[c], ll[c]] for c in CELL_SEEDS],
        "A-ens2": [[a[i], a[j]] for i, j in PAIRS],
        "S1-ens2": [[s[i], s[j]] for i, j in PAIRS],
        "LL-ens2": [[ll[i], ll[j]] for i, j in PAIRS],
        "K2-A": [[a[i], a[j], ll[i], ll[j]] for i, j in PAIRS],
        "K2-S1": [[s[i], s[j], ll[i], ll[j]] for i, j in PAIRS],
    }
    for k in (3, 5):
        seeds = SEEDS[:k]
        out[f"A-ens{k}"] = [[a[x] for x in seeds]]
        out[f"S1-ens{k}"] = [[s[x] for x in seeds]]
        out[f"LL-ens{k}"] = [[ll[x] for x in seeds]]
        out[f"K{k}-A"] = [[a[x] for x in seeds] + [ll[x] for x in seeds]]
        out[f"K{k}-S1"] = [[s[x] for x in seeds] + [ll[x] for x in seeds]]
    return out


def check_against_stored(*, month: str, heads: dict[str, dict[int, Head]], keys: dict) -> list[str]:
    """Compare recomputed heads and B-mix with experiment 404's stored forecasts; return lines."""
    names = list(score.LEVEL_NAMES)
    lines = []
    for seed in SEEDS:
        for label, arm, build in (
            ("A0-inj", "A0-inj", lambda s: [heads["A"][s]]),
            ("S1", "S1", lambda s: [heads["S"][s]]),
            ("B-mix", "B-mix", lambda s: [heads["A"][s], heads["L"][s]]),
        ):
            path = score.prediction_path(arm=arm, month=month, seed=seed, world=-1)
            if not path.exists():
                continue
            stored = pl.read_parquet(path)
            for k in KEY_COLUMNS:
                assert np.array_equal(stored[k].to_numpy(), keys[k]), f"stored {arm} {k} differs"
            err = float(np.abs(mix_equal(heads=build(seed)) - stored[names].to_numpy()).max())
            lines.append(f"{month} seed {seed} {label}: max |recomputed - stored| {err:.2e}")
            assert err < CHECK_TOLERANCE, lines[-1]
    return lines


def build_month(*, month: str) -> None:
    """Cache the heads of a month, check them, and write every construct's cells."""
    started = time.time()
    heads, keys = load_heads(month=month)
    lines = check_against_stored(month=month, heads=heads, keys=keys)
    (WORK_DIR / "checks").mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "checks" / f"{month}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    stored = pl.read_parquet(score.prediction_path(arm="A0-inj", month=month, seed=0, world=-1))
    base = stored.select([c for c in stored.columns if c not in score.LEVEL_NAMES])
    names = list(score.LEVEL_NAMES)
    for arm, cells in constructs(heads=heads).items():
        for cell, members in enumerate(cells):
            q = mix_equal(heads=members)
            frame = base.with_columns(
                **{n: pl.Series(q[:, i]).cast(pl.Float32) for i, n in enumerate(names)}
            )
            target = prediction_path(arm=arm, month=month, cell=cell)
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(target)
        print(f"{month} {arm}: {len(cells)} cell(s), {time.time() - started:.0f} s", flush=True)


def main() -> None:
    """Build the constructs of one month."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, choices=MONTHS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    build_month(month=args.month)


if __name__ == "__main__":
    main()

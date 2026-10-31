"""Build the no-fit ensembling control of experiment 445: mixtures of two or more heads of one kind.

B-mix averages a lagged head and a lagless head. To tell an ensembling gain from a lagged-plus-
lagless gain, this module averages (equal weights, density mixtures) several lagged heads (the
A0-inj networks of experiment 404, fit seeds 0 to 2, which exist for 2026-06 only), or several
lagless heads (the frozen cross-fitted tables of seeds 0 to 4, which exist for every month). Heads
are mixtures of 3 Gaussians in capacity fractions, so any set of heads mixes exactly.

Arms are written like those of `e00445_orule`, under the names `E-lag-<seeds>` and `E-ll-<seeds>`
(for example `E-lag-012`, `E-ll-01234`) and the single lagless heads `LL-s<seed>`.

```bash
<PYTHONPATH as e00445_orule.py shows> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00445-*/e00445_ensemble.py build --month 2026-06
```
"""

import argparse
from typing import Final

import e00445_fit as fit  # ty: ignore[unresolved-import]
import e00445_lagless as lagless  # ty: ignore[unresolved-import]
import e00445_models as models  # ty: ignore[unresolved-import]
import e00445_orule as orule  # ty: ignore[unresolved-import]
import e00445_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

Head = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
LAGLESS_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
LAGGED_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
LAGLESS_SETS: Final[tuple[tuple[int, ...], ...]] = (
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1, 2),
    (0, 1, 2, 3, 4),
)
LAGGED_SETS: Final[tuple[tuple[int, ...], ...]] = ((0, 1), (0, 2), (1, 2), (0, 1, 2))
SINGLE_LAGLESS: Final[tuple[int, ...]] = (0, 1, 2)
LAGGED_MONTH: Final[str] = "2026-06"
"""The only month in which experiment 404 saved more than one A0-inj seed."""


def name_of(*, kind: str, seeds: tuple[int, ...]) -> str:
    """Return the arm name of a mixture of heads of `kind` ("lag" or "ll") over `seeds`."""
    return f"E-{kind}-{''.join(str(s) for s in seeds)}"


def mix_equal(*, heads: list[Head]) -> np.ndarray:
    """Return the 11 quantiles (rows, 11) of the equal-weight mixture of the heads."""
    k = len(heads)
    log_w = torch.cat([h[0] - float(np.log(k)) for h in heads], dim=1)
    mean = torch.cat([h[1] for h in heads], dim=1)
    std = torch.cat([h[2] for h in heads], dim=1)
    out = []
    for start in range(0, len(log_w), orule.BATCH):
        sl = slice(start, start + orule.BATCH)
        out.append(models.mixture_quantiles(log_w=log_w[sl], mean=mean[sl], std=std[sl]))
    return torch.cat(out).numpy().astype(np.float64)


def lagless_head(*, month: str, seed: int, primary: np.ndarray, target: np.ndarray) -> Head:
    """Return a lagless table's mixture in capacity fractions for rows (primary, target)."""
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


def lagged_head_capacity(
    *, month: str, seed: int, world_power: np.ndarray, keys: dict[str, np.ndarray]
) -> Head:
    """Return the A0-inj network of a fit seed on a world as a mixture in capacity fractions."""
    net, saved = orule.load_a0_inj(month=month, seed=seed)
    _train, test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device="cpu", stride=1, saved=saved
    )
    for key, value in keys.items():
        assert np.array_equal(getattr(test, key).numpy(), value), f"seed {seed}: {key} differs"
    parts = orule.lagged_head(
        net=net, test=test, power=torch.tensor(world_power), scalers=scalers, lag_stats=lag_stats
    )
    mu = torch.tensor(scalers.target_mean, dtype=torch.float64)[test.primary]
    sd = torch.tensor(scalers.target_std, dtype=torch.float64)[test.primary]
    return (
        parts[0].double(),
        parts[1].double() * sd[:, None] + mu[:, None],
        parts[2].double() * sd[:, None],
    )


def write_arm(*, arm: str, base: pl.DataFrame, q: np.ndarray, month: str, world: int) -> None:
    """Write an arm's forecast frame with the row keys and flags of `base`."""
    names = list(orule.LEVEL_NAMES)
    frame = base.with_columns(
        **{n: pl.Series(q[:, i]).cast(pl.Float32) for i, n in enumerate(names)}
    )
    target = orule.prediction_path(arm=arm, month=month, seed=0, world=world)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)


def build_month(*, month: str, worlds: list[int]) -> None:
    """Write every ensemble arm of one month."""
    names = list(orule.LEVEL_NAMES)
    ref = pl.read_parquet(score.prediction_path(arm="A0-inj", month=month, seed=0, world=-1))
    keys = {k: ref[k].to_numpy() for k in ("primary", "origin", "target")}
    primary, target = keys["primary"].astype(np.int64), keys["target"].astype(np.int64)
    ll = {
        s: lagless_head(month=month, seed=s, primary=primary, target=target) for s in LAGLESS_SEEDS
    }
    q_ll = {}
    for seeds in LAGLESS_SETS:
        q_ll[name_of(kind="ll", seeds=seeds)] = mix_equal(heads=[ll[s] for s in seeds])
    for s in SINGLE_LAGLESS:
        q_ll[f"LL-s{s}"] = mix_equal(heads=[ll[s]])
    stored_o = pl.read_parquet(
        orule.prediction_path(arm="O-lagless", month=month, seed=0, world=-1)
    )
    err = np.abs(q_ll["LL-s0"] - stored_o[names].to_numpy()).max()
    assert err < orule.CHECK_TOLERANCE, "lagless head differs from stored O-lagless"
    print(
        f"{month}: lagless head recomputed from the table vs O-lagless, max diff {err:.2e}",
        flush=True,
    )
    for world in worlds:
        base = pl.read_parquet(
            score.prediction_path(arm="A0-inj", month=month, seed=0, world=world)
        )
        base = base.select([c for c in base.columns if c not in names])
        for arm, q in q_ll.items():
            write_arm(arm=arm, base=base, q=q, month=month, world=world)
        if month != LAGGED_MONTH:
            continue
        power, _code = fit.world_power(seed=None if world < 0 else world)
        lagged = {
            s: lagged_head_capacity(month=month, seed=s, world_power=power, keys=keys)
            for s in LAGGED_SEEDS
        }
        stored = pl.read_parquet(
            score.prediction_path(arm="A0-inj", month=month, seed=0, world=world)
        )
        err = np.abs(mix_equal(heads=[lagged[0]]) - stored[names].to_numpy()).max()
        assert err < orule.CHECK_TOLERANCE, "recomputed A0-inj differs from stored"
        print(
            f"  w{world}: lagged seed 0 recomputed vs stored A0-inj, max diff {err:.2e}", flush=True
        )
        for seeds in LAGGED_SETS:
            q = mix_equal(heads=[lagged[s] for s in seeds])
            write_arm(
                arm=name_of(kind="lag", seeds=seeds), base=base, q=q, month=month, world=world
            )
        i = 0  # seed-matched B-mix check on seed 0
        bm = pl.read_parquet(score.prediction_path(arm="B-mix", month=month, seed=i, world=world))
        q = mix_equal(heads=[lagged[i], ll[i]])
        err = np.abs(q - bm[names].to_numpy()).max()
        assert err < orule.CHECK_TOLERANCE, "recomputed B-mix differs from stored"
        print(f"  w{world}: recomputed B-mix seed {i} vs stored, max diff {err:.2e}", flush=True)


def main() -> None:
    """Build the ensemble arms for a month."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--month", required=True)
    parser.add_argument("--worlds", type=int, nargs="*", default=[*score.SCORING_SEEDS, -1])
    args = parser.parse_args()
    torch.set_num_threads(1)
    build_month(month=args.month, worlds=args.worlds)


if __name__ == "__main__":
    main()

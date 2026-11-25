# ruff: noqa: D205, E501, PT018
"""Experiment 725: build the fold's world (experiment 713's arrays plus the live-rule arrays) for a phase.

`load_world` runs experiment 713's pipeline for one fold and phase, takes the fold's standardiser
from the Candidate's rows, standardises the ENS cache of the phase with it, and wraps experiment 713's
`Source` in a `LiveSource`. It does not set `E00667_PHASE`: the calling script does that before it
imports this module, so that no code path can read a held-out reading in the training phase.
"""

from dataclasses import dataclass
from pathlib import Path

import e00713_arms as arms713  # ty: ignore[unresolved-import]
import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00725_batch as lb
import e00725_frozen as fz
import e00725_sources as src
import numpy as np
import torch  # ty: ignore[unresolved-import]


@dataclass
class LiveWorld:
    """A fold's `World` with its `LiveSource`, and what the checks need besides.

    Attributes:
        world: Experiment 713's `World`, whose source is the `LiveSource`.
        data: Experiment 713's `FoldData` (its source is the same `LiveSource`).
        live: The `LiveSource`.
        cache: The raw ENS cache of the phase.
        stats: The fold's statistics used to standardise the cache.
        pipe: Experiment 713's pipeline modules.
    """

    world: tr.World
    data: inp.FoldData
    live: lb.LiveSource
    cache: src.EnsCache
    stats: dict[str, np.ndarray]
    pipe: inp.Pipeline


def fold_statistics(*, data: inp.FoldData, pipe: inp.Pipeline) -> dict[str, np.ndarray]:
    """Return the fold's weather and B1 statistics, and release the Candidate's rows.

    The weather statistics are the first seven columns of the Candidate's standardiser (the columns
    that experiment 632's `standardised_inputs` standardises); the B1 statistics are the stored
    `stats/B1/<month>.npz` of experiment 667's chain.
    """
    assert data.rows is not None, "build_fold_data must be called with keep_rows=True"
    standardiser = data.rows[0].base.standardiser
    stats = {
        "weather_mean": np.asarray(standardiser.mean[: src.N_WEATHER]),
        "weather_std": np.asarray(standardiser.std[: src.N_WEATHER]),
    }
    npz = np.load(pipe.e.stats_file(month=pipe.month))
    stats["b1_mean"] = np.asarray(npz["mean"])
    stats["b1_std"] = np.asarray(npz["std"])
    data.rows = None
    return stats


def cams_day_array(
    *, pipe: inp.Pipeline, b1_mean: np.ndarray, b1_std: np.ndarray, device: str
) -> torch.Tensor:
    """Return the standardised CAMS day summaries (16, N_DAYS, 2) float32, standardised by
    experiment 632's own `b1_scale`.
    """
    _, cams = pipe.d632.dense_b1()
    _, z = pipe.d632.b1_scale(
        grid=np.zeros((0, pipe.d632.N_B1_GRID), dtype=np.float32), cams=cams, mean=b1_mean, std=b1_std,
        device=device,
    )  # fmt: skip
    return z


def load_world(
    *, month: str, seed: int, rule_name: str, phase: str, device: str, data_dir: Path,
    target_summary_back_days: int | None = None, check_standardiser: bool = True,
) -> LiveWorld:  # fmt: skip
    """Run experiment 713's pipeline for (`month`, `phase`) and return the world with a `LiveSource`.

    Args:
        month: The fold month.
        seed: The seed (it selects the Candidate chain that holds the encoders).
        rule_name: `L`, `Z`, or `E`.
        phase: `train` or `predict`.
        device: `cuda` or `cpu`.
        data_dir: The experiments' data folder.
        target_summary_back_days: Read-out `T3`, see `LiveSource`.
        check_standardiser: Assert that the fold's standardiser equals the one experiment 713 stored.

    Returns:
        The world for the fold, with a `LiveSource` as its source.
    """
    fz.assert_frozen_checkout()
    fz.install_own_cache(data_dir=data_dir)
    pipe = inp.load_pipeline(month=month, phase=phase, chain_root=inp.chain_root_of_seed(seed=seed))
    data = inp.build_fold_data(pipe=pipe, device=device, keep_rows=True)
    if check_standardiser:
        saved = arms713.root(data_dir=data_dir) / "work" / "standardiser" / f"{month}.sha256"
        assert saved.exists() and saved.read_text() == data.fingerprint, (
            "the fold's standardiser differs from experiment 713's"
        )
    stats = fold_statistics(data=data, pipe=pipe)
    cache = src.load_cache(phase=phase, data_dir=data_dir)
    ens_z = src.standardise_cache(
        cache=cache, weather_mean=stats["weather_mean"], weather_std=stats["weather_std"],
        b1_mean=stats["b1_mean"], b1_std=stats["b1_std"],
    )  # fmt: skip
    n_grid = data.source.n_grid
    assert ens_z.shape[1] == n_grid, (
        f"the ENS cache has {ens_z.shape[1]} half-hours, the grid {n_grid}"
    )
    import e00422_prep as prep422  # ty: ignore[unresolved-import]

    live = lb.build_live_source(
        base=data.source, rule_name=rule_name, phase=phase,
        ens_z=torch.from_numpy(ens_z).to(device),
        ens_found=torch.from_numpy(np.asarray(cache.found)).to(device),
        ens_origin=torch.from_numpy(np.asarray(cache.origin_idx).astype(np.int64)).to(device),
        cams_days=cams_day_array(pipe=pipe, b1_mean=stats["b1_mean"], b1_std=stats["b1_std"], device=device),
        day0=int(pipe.d632.DAY0), grid_start_us=int(prep422.GRID_START_US),
        target_summary_back_days=target_summary_back_days,
    )  # fmt: skip
    data.source = live
    world = tr.World(
        source=live, cutoff_idx=data.cutoff_idx, clean=data.clean, target_mean=data.target_mean,
        target_std=data.target_std, month=month, placebo_pool=tr.block_pool(clean=data.clean),
    )  # fmt: skip
    return LiveWorld(world=world, data=data, live=live, cache=cache, stats=stats, pipe=pipe)


def check_cache_matches_grid(*, cache: src.EnsCache, n_grid: int) -> None:
    """Assert that the cache covers the grid of experiment 713."""
    assert cache.raw.shape[1] == n_grid and cache.found.shape[1] == n_grid, (
        cache.raw.shape,
        n_grid,
    )

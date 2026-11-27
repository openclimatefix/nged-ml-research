# ruff: noqa: E501
"""Experiment 733: the loaders that turn the real files into the inputs of `fit_real` (torch).

`real_loader` follows experiment 716's loader: experiment 713's pipeline and fold arrays, wrapped
in an
`UntSource` that carries the channel tensors of the arm. The channel tensors come from the channel
table
(`channels/table.parquet`), the re-run outputs of the fold's origins (`channels/rerun/<fold>.npz`),
and the
channel choice (`channels.json`). Each arm's tensors follow its spec:

- F0r reads no channel (`absent_tensors`);
- FU reads every usable channel with the re-run of days D - 2 and D - 1;
- FS reads P(X<0) only;
- FU-S reads the table 28 days earlier and no re-run.

Only the real-fold loader is here. The loader of the synthetic world (the `SegmentArrays` of
experiment 730's
banks with their ERA5 and CAMS history weather, ensemble target rows, and re-run channels) is not
built yet.
"""

import json
from pathlib import Path
from typing import Any, Final

import e00733_channels as channels
import e00733_core as core
import e00733_fit as fit
import e00733_inputs as inputs
import e00733_leakage as leak
import e00733_paths as paths
import e00733_rerun as rerun
import e00733_stages as st
import e00733_train as train
import numpy as np

D8_LABEL: Final = "D8"


def arm_tensors_spec(*, arm: str, usable: tuple[str, ...]) -> tuple[tuple[str, ...], bool, int]:
    """Return what an arm reads: the channels kept.

    Whether the re-run is read, and the placebo shift in days.

    Args:
        arm: F0r, FU, FU-S, or FS.
        usable: The channels the rule left usable.

    Returns:
        `(channels_kept, reads_rerun, shift_days)`; F0r keeps no channel.
    """
    spec = core.ARMS[arm]
    if arm == "F0r":
        return (), False, 0
    kept = spec.channel_subset if spec.channel_subset is not None else usable
    kept = tuple(c for c in kept if c in usable)
    return kept, spec.shifted_days == 0, spec.shifted_days


def d8_index() -> int:
    """Return the Primary index of D8 from experiment 578's site map."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    sites = k578.sites()
    labels = [str(x) for x in sites.sort("primary")["label"].to_list()]
    return fit.d8_index_from(labels=labels)


def arm_tensors_for(
    *, arm: str, base: Path, data: object, device: str, variant: str | None = None,
    poison_idx: np.ndarray | None = None,
) -> tuple[inputs.ChannelTensors, tuple[str, ...]]:  # fmt: skip
    """Return the channel tensors of one arm on one fold, and the channels the arm keeps.

    Args:
        arm: F0r, FU, FU-S, or FS.
        base: This experiment's data folder (the channel choice, table, and re-run live there).
        data: Experiment 713's `FoldData` of the fold.
        device: The device.
        variant: None for the arm itself, "masked" for reading R4 (every channel and flag zero), or
            "stored" for reading R12 (the stored outputs of days D - 2 and D - 1 in place of the
            re-run).
        poison_idx: Grid indices whose table values are set to NaN (the target-row poison check of
            `e00733_leakage`); None leaves the table as stored.

    Returns:
        The tensors and the kept channels.

    Raises:
        ValueError: When `variant` is unknown.
    """
    if variant not in (None, "masked", "stored"):
        msg = f"unknown prediction variant {variant}"
        raise ValueError(msg)
    choice = channels.read_choice(path=paths.channels_json(root=base))
    kept, reads_rerun, shift_days = arm_tensors_spec(arm=arm, usable=choice.usable)
    index = d8_index()
    n_primaries = int(data.clean.shape[0])  # ty: ignore[unresolved-attribute]
    n_grid = int(data.clean.shape[1])  # ty: ignore[unresolved-attribute]
    std = float(data.target_std[index])  # ty: ignore[unresolved-attribute]
    if not kept:
        return (
            inputs.absent_tensors(
                n_primaries=n_primaries, n_grid=n_grid, d8_index=index, device=device
            ),
            kept,
        )
    values, ok = channels.read_table(path=paths.table_path(root=base)).dense(n_grid=n_grid)
    outputs = None
    if poison_idx is not None:
        poisoned, _ = leak.poison_target_cells(
            values=values[None], ok=ok[None], primary=0, target_idx=poison_idx
        )
        values = poisoned[0]
    if shift_days:
        before = values
        values, ok = channels.placebo(
            values=values, ok=ok, shift_steps=shift_days * core.STEPS_PER_DAY
        )
        check_placebo(
            before=before, after=values, ok=ok, shift_steps=shift_days * core.STEPS_PER_DAY
        )
    elif reads_rerun and variant != "stored":
        outputs = rerun.read_rerun(path=paths.rerun_dir(root=base) / f"{data.month}.npz")  # ty: ignore[unresolved-attribute]
    tensors = inputs.build_tensors(
        dense_values=values, dense_ok=ok, n_primaries=n_primaries, d8_index=index, target_std=std,
        rerun_outputs=outputs, channels_kept=kept, device=device,
    )  # fmt: skip
    if variant == "masked":
        tensors = inputs.mask_all(tensors=tensors)
    return tensors, kept


def check_placebo(
    *, before: np.ndarray, after: np.ndarray, ok: np.ndarray, shift_steps: int
) -> int:
    """Check that each placebo value is the stored value from `shift_steps` earlier, and 28 days at least.

    Args:
        before: The stored values `(n_grid, 3)`.
        after: The placebo values.
        ok: The placebo's presence flag.
        shift_steps: The shift in half-hours.

    Returns:
        The number of tokens checked.

    Raises:
        CorrectnessFailureError: When a placebo value differs from the stored value it should copy.
    """
    token = np.flatnonzero(ok)
    source = token - shift_steps
    if not np.array_equal(after[token], before[source], equal_nan=True):
        msg = "a placebo value is not the stored value from the shifted half-hour"
        raise st.CorrectnessFailureError(msg)
    return leak.assert_placebo_from_28_days_earlier(source_idx=source, token_idx=token)


def table_flags(*, base: Path, n_grid: int) -> np.ndarray:
    """Return whether the channel table holds a value at each grid index, `(n_grid,)`."""
    return channels.read_table(path=paths.table_path(root=base)).dense(n_grid=n_grid)[1]


def build_fold_data_with_run_files(*, pipe: object, device: str, keep_rows: bool = False) -> Any:
    """Build experiment 713's fold data and attach the run times that the run files record.

    Experiment 713 sets each origin day's run time to the origin minus 9 hours, so a check against
    that array holds by construction. This function reads, for every origin day that holds target
    rows, the run's initialisation time (from the file's name) and availability time (from the file's
    own `power_fcst_init_time` column) into `ens.file_run_init_us` and `ens.file_run_avail_us`,
    `(16, n_days)` arrays that are 0 where no target row exists. The target rows' other columns come
    from the same stored run, so `ens.file_other_valid_us` is the run's initialisation time.

    Args:
        pipe: The pipeline of experiment 713.
        device: "cpu" or "cuda".
        keep_rows: Whether to keep the Candidate rows.

    Returns:
        Experiment 713's `FoldData`.
    """
    import e00713_inputs as inp713  # ty: ignore[unresolved-import]

    data = inp713.build_fold_data(pipe=pipe, device=device, keep_rows=keep_rows)
    ens = data.source.ens  # ty: ignore[unresolved-attribute]
    has_rows = ens.filled.any(dim=-1).cpu().numpy()  # (16, n_days)
    first_epoch = int(ens.first_day) + channels.GRID_START_EPOCH_DAY
    days = np.flatnonzero(has_rows.any(axis=0))
    read = leak.read_run_files(
        summary_dirs=paths.ens_summary_dirs(), epoch_days=[first_epoch + int(d) for d in days]
    )
    init = np.zeros(has_rows.shape, dtype=np.int64)
    avail = np.zeros(has_rows.shape, dtype=np.int64)
    for d in days:
        i, a = read[first_epoch + int(d)]
        init[:, d] = np.where(has_rows[:, d], i, 0)
        avail[:, d] = np.where(has_rows[:, d], a, 0)
    ens.file_run_init_us, ens.file_run_avail_us, ens.file_other_valid_us = init, avail, init.copy()
    return data


def real_loader(*, root: Path | None = None) -> fit.RealLoader:
    """Return the loader of the real-fold fits: `(arm, fold, seed, device) -> RealInputs`.

    Args:
        root: This experiment's data folder (default: `paths.data_root()`).

    Returns:
        The loader.
    """
    base = paths.data_root() if root is None else root

    def load(arm: str, fold: str, seed: int, device: str) -> fit.RealInputs:
        import e00713_arms as arms713  # ty: ignore[unresolved-import]
        import e00713_inputs as inp713  # ty: ignore[unresolved-import]
        import e00713_train as tr713  # ty: ignore[unresolved-import]
        from studies.sources import ML_EXPERIMENTS_DATA_DIR

        pipe = inp713.load_pipeline(
            month=fold, phase="train", chain_root=inp713.chain_root_of_seed(seed=seed)
        )
        if device == "cpu":
            patch_cpu_tables(pipe=pipe)
        data = build_fold_data_with_run_files(pipe=pipe, device=device)
        leak.assert_training_power_cut(power=data.clean, fold_start_idx=data.cutoff_idx)
        tensors, kept = arm_tensors_for(arm=arm, base=base, data=data, device=device)
        world = tr713.World(
            source=train.UntSource(source=data.source, tensors=tensors, root=base), cutoff_idx=data.cutoff_idx,
            clean=data.clean, target_mean=data.target_mean, target_std=data.target_std, month=fold, placebo_pool=None,
        )  # fmt: skip
        star = arms713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
        return fit.RealInputs(
            world=world,
            stage1_state=arms713.model_file(data_dir=ML_EXPERIMENTS_DATA_DIR, arm=star, month=fold, seed=seed, stage=1),
            dropped=data.dropped_per_fold, fingerprint=data.fingerprint,
            encoders=arms713.SCREEN_ARMS[star][1], has_channels=bool(kept),
        )  # fmt: skip

    return load


def patch_cpu_tables(*, pipe: object) -> None:
    """Make experiment 574's expected-input table live on the CPU (its `attach_tables` names "cuda").

    The patch applies only to a run on the CPU (the CPU smoke test), where no CUDA device exists. The
    table holds the same numbers; only the device differs.
    """
    import torch

    e = pipe.e  # ty: ignore[unresolved-attribute]

    def attach_tables(*, standardiser: object) -> None:
        cols, origin = e.d560.raw_table(source="ft", cache=e.W560 / "expected")
        e.CTX["origin_ft"] = e.np.asarray(origin)
        e.CTX["table_ft"] = torch.from_numpy(
            e.d560.standardised_table(cols=cols, standardiser=standardiser)
        )

    e.attach_tables = attach_tables


def build_weather_cache(
    *, root: Path, fold: str = core.WORLD_WEATHER_FOLD, device: str = "cpu"
) -> Path:
    """Build `world/weather_d8.npz` from experiment 713's pipeline of one fold.

    The synthetic world standardises D8's history weather and date-time columns, and the ensemble
    rows of its target rows, with that fold's constants.

    Args:
        root: This experiment's data folder.
        fold: The fold whose pipeline is read.
        device: "cpu" or "cuda".

    Returns:
        The path of the cache file.
    """
    import e00713_inputs as inp713  # ty: ignore[unresolved-import]
    import e00733_world as world

    pipe = inp713.load_pipeline(
        month=fold, phase="train", chain_root=inp713.chain_root_of_seed(seed=0)
    )
    if device == "cpu":
        patch_cpu_tables(pipe=pipe)
    data = build_fold_data_with_run_files(pipe=pipe, device=device, keep_rows=True)
    mr, _ = data.rows
    return world.write_weather_cache(
        root=root, data=data, standardiser=mr.base.standardiser, d8_index=d8_index()
    )


def read_choice_usable(*, root: Path) -> tuple[str, ...]:
    """Return the channels `channels.json` lists as usable."""
    return tuple(json.loads(paths.channels_json(root=root).read_text())["usable"])

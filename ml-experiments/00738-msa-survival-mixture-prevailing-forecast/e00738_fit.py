# ruff: noqa: E501
"""One fit job per (arm, fold, seed): W0, or a head arm on the frozen stored stage 3 state of `M1`.

Run through `e00738_run.py one-task` in a child process of its own, one process per job. It sets `E00667_PHASE=train`, under which
experiment 713's pipeline sets the cleaned power to NaN from the fold month's start, so no code
path can read a held-out reading. It writes the state to `networks/<arm>/<month>_s<seed>.pt` and the
log to `fits/<arm>/<month>_s<seed>.json`, each to a temporary name and then renamed. The stored
states of `M1` are read, never written.
"""

from __future__ import annotations

import dataclasses
import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Final

import e00738_common as common
import e00738_panel as panel
import e00738_phases as phases
import e00738_survival as surv
import e00738_world as world
import numpy as np

REQUIRED_CACHE_FILES: Final[tuple[str, ...]] = (
    "dense_b1_cams.npy",
    "dense_b1_grid.npy",
    "dense_raw.npy",
)
"""The dense arrays in experiment 713's cache folder. Experiment 713's pipeline builds a missing
one there, so a missing file stops the job before it can write into that folder."""
N_DURATION_PASSES: Final[int] = 100
"""Training-world passes whose drawn durations the survival curve is fitted to."""


def fold_world() -> tuple[dict[int, list[int]], np.ndarray, np.ndarray]:
    """Return experiment 650's (neighbours, capacity in MW, is-signed-MW flag) of the 16 Primaries."""
    import e00650_common as c650  # ty: ignore[unresolved-import]

    return c650.world()


def window_end_idx() -> int:
    """Return the grid index of 2026-07-01, which no scoring shift may reach."""
    import e00650_common as c650  # ty: ignore[unresolved-import]

    return int(c650.WINDOW_START_IDX)


def fold_curve(*, month: str, seed: int, cutoff_idx: int, is_mw: np.ndarray) -> surv.Curve:
    """Fit the survival curve to the drawn durations of the fold's training worlds.

    Args:
        month: The fold month.
        seed: The fit's seed.
        cutoff_idx: The fold's cutoff.
        is_mw: Per Primary, whether it reports signed megawatts.

    Returns:
        The best of the four families, fitted to every shift's drawn (unclipped) duration.
    """
    durations: list[float] = []
    for pass_no in range(N_DURATION_PASSES):
        for p in range(16):
            durations.extend(
                s.drawn_days
                for s in world.renewal_shifts(
                    space="train",
                    fit_seed=seed,
                    month_idx=common.month_index(month=month),
                    pass_no=pass_no,
                    primary=p,
                    window_start_idx=0,
                    window_end_idx=cutoff_idx,
                    is_mw=is_mw,
                )
            )
    return surv.fit_curve(durations_days=np.array(durations))


def load_fold_data(*, month: str, phase: str, seed: int, device: str) -> Any:
    """Return experiment 713's `FoldData` of a fold in a phase, after the standardiser check.

    Args:
        month: The fold month.
        phase: `train` (cleaned power is NaN from the month's start) or `predict`.
        seed: The seed, which selects the chain of pre-trained encoders.
        device: The device.

    Returns:
        The `FoldData`.
    """
    import e00713_arms as arms713  # ty: ignore[unresolved-import]
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    missing = [f for f in REQUIRED_CACHE_FILES if not (inp.CACHE / f).exists()]
    assert not missing, f"experiment 713's cache lacks {missing}; a build there would write to it"
    pipe = inp.load_pipeline(month=month, phase=phase, chain_root=inp.chain_root_of_seed(seed=seed))
    data = inp.build_fold_data(pipe=pipe, device=device)
    saved = (
        arms713.root(data_dir=ML_EXPERIMENTS_DATA_DIR) / "work" / "standardiser" / f"{month}.sha256"
    )
    assert saved.exists(), "the standardiser fingerprint is missing"
    assert saved.read_text() == data.fingerprint, f"the standardiser differs in the {phase} phase"
    return data


def site_table() -> Any:
    """Return the 16 Primaries' `primary`, `label`, and `time_series_id`, sorted by `primary`."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    return k578.sites().sort("primary")


def new_w0_model(*, month: str, seed: int, device: str) -> Any:
    """Return `W0`'s network: experiment 713's `M1` architecture at `M1`'s stored stage 1 state."""
    import e00713_arms as arms713  # ty: ignore[unresolved-import]
    import e00713_fit as fit713  # ty: ignore[unresolved-import]
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    spec = arms713.resolve(arm="M1", data_dir=ML_EXPERIMENTS_DATA_DIR)
    model = fit713.new_model(spec=spec, seed=seed, month=month, device=device)
    model.load_state_dict(m1_state(month=month, seed=seed, stage=1, device=device))
    return model


def m1_state(*, month: str, seed: int, stage: int, device: str) -> dict:
    """Read a stored `M1` state of experiment 713 (read only)."""
    import torch  # ty: ignore[unresolved-import]

    path = common.ROOT_713 / "work" / "models" / "M1" / f"{month}_s{seed}_stage{stage}.pt"
    assert path.exists(), path
    return torch.load(path, map_location=device)


def head_init_seed(*, seed: int) -> int:
    """Return the seed of the level head's initial weights, which differs between run seeds."""
    return seed * 7 + 3


def head_model(  # noqa: ANN201 - `LevelNet` needs torch, imported lazily
    *,
    arm: str,
    month: str,
    seed: int,
    target_std: np.ndarray,
    noise: float,
    curve: surv.SurvivalFunction | None,
    device: str,
):
    """Return a `LevelNet` with the stored stage 3 `M1` base loaded and frozen.

    Args:
        arm: `LP`, `LD`, `LM`, or `LR`.
        month: The fold month.
        seed: The seed.
        target_std: Each Primary's standard deviation of power over capacity.
        noise: A1's noise unit.
        curve: The fitted survival curve (`LD` and `LM`).
        device: The device.

    Returns:
        The network.
    """
    import e00738_model as lmdl
    import torch  # ty: ignore[unresolved-import]

    torch.manual_seed(head_init_seed(seed=seed))
    net = lmdl.LevelNet(mode=arm, target_std=target_std, noise=noise, curve=curve).to(device)
    net.load_base(state=m1_state(month=month, seed=seed, stage=3, device=device))
    net.freeze_base()
    return net


def save_state_atomic(*, state: dict, path: Path) -> None:
    """Write a state file under a temporary name and rename it."""
    import torch  # ty: ignore[unresolved-import]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(state, tmp)
    tmp.rename(path)


def fit_job(*, arm: str, month: str, seed: int, device: str, remedy: bool = False) -> None:
    """Fit one arm on one fold and one seed, and write the state and the log.

    Args:
        arm: `W0`, `LP`, `LD`, `LM`, or `LR`.
        month: The fold month.
        seed: The seed.
        device: `cuda` or `cpu`.
        remedy: Use the gate's remedy (cap 120, patience 15) for a head arm.
    """
    os.environ["E00667_PHASE"] = "train"
    os.environ["E00667_MONTH"] = month
    code713 = panel.assert_713_code()
    import e00713_train as tr  # ty: ignore[unresolved-import]
    import e00738_train as train738
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    paths = common.fit_paths(arm=arm, month=month, seed=seed)
    if paths["state"].exists() and paths["log"].exists():
        return
    started = time.time()
    data = load_fold_data(month=month, phase="train", seed=seed, device=device)
    tw = tr.World(
        source=data.source,
        cutoff_idx=data.cutoff_idx,
        clean=data.clean,
        target_mean=data.target_mean,
        target_std=data.target_std,
        month=month,
        placebo_pool=tr.block_pool(clean=data.clean),
    )
    _, _, is_mw = fold_world()
    log: dict = {
        "arm": arm,
        "month": month,
        "seed": seed,
        "device": device,
        "code_hash_713": code713,
        "code_hash": phases.code_hash(),
        "remedy": remedy,
        "start": time.strftime("%FT%TZ", time.gmtime(started)),
    }
    if arm == common.ARM_W0:
        model = new_w0_model(month=month, seed=seed, device=device)
        logs = train738.fit_w0(model=model, tw=tw, clean=data.clean, is_mw=is_mw, seed=seed)
        log["stages"] = [dataclasses.asdict(s) for s in logs]
        log["passes"] = sum(s.passes for s in logs)
        log["cap_hit"] = any(s.cap_hit for s in logs)
        state = model.state_dict()
    else:
        z = (data.clean - data.target_mean[:, None]) / data.target_std[:, None]
        noise = surv.noise_unit_from_panel(z=z, cutoff_idx=data.cutoff_idx)
        curve = (
            fold_curve(month=month, seed=seed, cutoff_idx=data.cutoff_idx, is_mw=is_mw)
            if arm in {"LD", "LM"}
            else None
        )
        net = head_model(
            arm=arm,
            month=month,
            seed=seed,
            target_std=data.target_std,
            noise=noise,
            curve=curve,
            device=device,
        )
        head_log = train738.fit_head_arm(
            model=net, tw=tw, clean=data.clean, is_mw=is_mw, seed=seed, remedy=remedy
        )
        log["stages"] = [dataclasses.asdict(head_log)]
        log["passes"] = head_log.passes
        log["cap_hit"] = head_log.cap_hit
        log["noise"] = noise
        log["curve"] = (
            None if curve is None else {"family": curve.family, "params": list(curve.params)}
        )
        log["curve_error"] = None if curve is None else surv.curve_error(curve=curve)
        state = {
            "mode": arm,
            "noise": noise,
            "curve": None
            if curve is None
            else {"family": curve.family, "params": list(curve.params), "nll": curve.nll},
            "level_head": net.level_head.state_dict(),
        }
    log["seconds"] = round(time.time() - started, 1)
    log["end"] = time.strftime("%FT%TZ", time.gmtime())
    log["peak_host_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if device.startswith("cuda"):
        log["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
    save_state_atomic(state=state, path=paths["state"])
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    common.atomic_json(obj=log, path=paths["log"])


def load_for_prediction(*, arm: str, month: str, seed: int, data: Any, device: str):  # noqa: ANN201
    """Return (network, model function) of an arm for the prediction phase.

    Args:
        arm: Any of `common.ALL_ARMS`.
        month: The fold month.
        seed: The seed.
        data: The fold's `FoldData` (its target standard deviation sets the depth limit).
        device: The device.

    Returns:
        The network in evaluation mode, and a function from an `MsaInputs` to (standardised
        quantiles, diagnostics).
    """
    import e00713_model as mdl  # ty: ignore[unresolved-import]
    import e00738_model as lmdl
    import torch  # ty: ignore[unresolved-import]

    if arm in {common.ARM_M1, common.ARM_W0}:
        net = mdl.MsaNet(switches=mdl.ARM_SWITCHES["M"]).to(device)
        if arm == common.ARM_M1:
            net.load_state_dict(m1_state(month=month, seed=seed, stage=3, device=device))
        else:
            path = common.fit_paths(arm=arm, month=month, seed=seed)["state"]
            net.load_state_dict(torch.load(path, map_location=device))
        net.eval()

        def plain(inp_: Any) -> tuple[object, dict[str, np.ndarray]]:
            return mdl.standardised_quantiles(out=net(inp_).out), {}  # ty: ignore[invalid-argument-type]

        return net, plain
    saved = torch.load(
        common.fit_paths(arm=arm, month=month, seed=seed)["state"], map_location=device
    )
    curve = None
    if saved["curve"] is not None:
        c = saved["curve"]
        curve = surv.Curve(family=c["family"], params=tuple(c["params"]), nll=float(c["nll"]))
    head = head_model(
        arm=arm,
        month=month,
        seed=seed,
        target_std=data.target_std,
        noise=float(saved["noise"]),
        curve=curve,
        device=device,
    )
    head.level_head.load_state_dict(saved["level_head"])
    head.eval()

    def level(inp_: Any) -> tuple[object, dict[str, np.ndarray]]:
        lo = head.level_forward(inp_)
        limit = head.depth_limit[inp_.primary]
        depth_share = (lo.depth * common.LEVEL_DEPTH_LIMIT / limit).detach().cpu().numpy()
        diag = {
            "age": lo.age.detach().cpu().numpy(),
            "depth_share": depth_share,
            "p_mean": lo.p.mean(dim=1).detach().cpu().numpy(),
        }
        return lmdl.quantiles_of(out=lo.out, components=lo.components), diag

    return head, level


def read_log(*, arm: str, month: str, seed: int, out: Path | None = None) -> dict:
    """Read a fit's JSON log."""
    return json.loads(common.fit_paths(arm=arm, month=month, seed=seed, out=out)["log"].read_text())

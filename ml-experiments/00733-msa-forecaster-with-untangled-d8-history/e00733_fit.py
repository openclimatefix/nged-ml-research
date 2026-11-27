# ruff: noqa: E501
"""Experiment 733: one fit, its stored files, and the runner that records a failed fit (torch).

A fit stores its record (`fits/<job>.json`, with the per-pass losses of every stage and the pass
whose state was stored), its network (`networks/<job>.pt`, one file per stage on the real folds),
and,
on the synthetic world, nothing else: prediction runs in its own process (`e00733_predict`). A fit
whose record exists is skipped. A fit that raises leaves `fits/<job>.failed.json` with the message
and
the traceback, and the runner goes on; a correctness failure, a stop, and a guard always propagate.

The GPU cap goes through experiment 729's `limit_gpu_memory`, in the one function `prepare_device`.
"""

import json
import os
import resource
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00733_channels as channels
import e00733_core as core
import e00733_model as model
import e00733_paths as paths  # noqa: F401
import e00733_synth as sy
import e00733_train as train
import numpy as np

# isort: split
import e00713_train as tr713  # ty: ignore[unresolved-import]
import e00729_train as train729  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from e00730_leakage import CorrectnessFailureError  # ty: ignore[unresolved-import]

CAP_FIRST: Final = 60
CAP_RERUN: Final = 120
"""A stage that ends at the cap of 60 passes is rerun from its starting state at a cap of 120, as
experiment 716 does."""


class GuardError(RuntimeError):
    """A gate is closed (a device or a review gate).

    The line stops and nothing is recorded as a failed fit.
    """


# --------------------------------------------------------------------------- the files


def job_name(*, arm: str, fold: str | None, seed: int) -> str:
    """Return the name of one fit.

    `<arm>_<fold>_s<seed>` on the real folds, `<arm>_s<seed>` on the synthetic world.
    """
    safe = arm.replace("*", "star")
    return f"{safe}_{fold}_s{seed}" if fold else f"{safe}_s{seed}"


def record_path(*, root: Path, arm: str, fold: str | None, seed: int) -> Path:
    """Return the path of a fit's record."""
    return root / "fits" / f"{job_name(arm=arm, fold=fold, seed=seed)}.json"


def network_path(*, root: Path, arm: str, fold: str | None, seed: int, stage: int = 0) -> Path:
    """Return the path of a fit's network state (`stage` 2 or 3 on the real folds.

    0 on the synthetic world).
    """
    name = job_name(arm=arm, fold=fold, seed=seed)
    return root / "networks" / (f"{name}_stage{stage}.pt" if stage else f"{name}.pt")


def write_json(*, path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file under a temporary name that carries the process id, then rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.rename(path)


def save_state(*, state: Mapping[str, torch.Tensor], path: Path) -> None:
    """Save a state dictionary atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(dict(state), tmp)
    tmp.rename(path)


def peak_rss_gb() -> float:
    """Return the peak resident memory of this process in gigabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


# --------------------------------------------------------------------------- the device


def prepare_device(*, device: str) -> None:
    """Check the device and cap this process's share of the GPU at the slot's 6 GB.

    Args:
        device: "cuda" or "cpu".

    Raises:
        GuardError: When "cuda" is asked for and no GPU is available, or the device is neither.
    """
    if device not in {"cuda", "cpu"}:
        msg = f"the device {device} is neither cuda nor cpu"
        raise GuardError(msg)
    if device == "cuda":
        if not torch.cuda.is_available():
            msg = "a fit asked for the GPU and none is available"
            raise GuardError(msg)
        train729.limit_gpu_memory(gigabytes=core.GPU_GB)


def peak_gpu_gb(*, device: str) -> float:
    """Return the peak GPU memory of this process in gigabytes (zero on the CPU)."""
    return torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0


# --------------------------------------------------------------------------- the real folds


@dataclass
class RealInputs:
    """What a real-fold fit reads, besides the stored states.

    Attributes:
        world: The fold's training world, whose `source` is an `UntSource` with the arm's tensors.
        stage1_state: M*'s stored stage 1 state of the fold and seed.
        dropped: The per-fold counts of dropped half-hours.
        fingerprint: The hash of the fold's standardiser.
        encoders: "frozen", "trained", or "random", as M* resolves.
        has_channels: Whether the arm reads any untangled channel (F0r reads none).
    """

    world: object
    stage1_state: Path
    dropped: Mapping[str, object]
    fingerprint: str
    encoders: str
    has_channels: bool


RealLoader = Callable[[str, str, int, str], RealInputs]
"""A loader of (arm, fold, seed, device) returning the fit's inputs."""


def train_with_rerun(
    *, net: model.UntangledNet, world: object, stage: int, samples: object, seed: int, encoders: str,
    start_state: Mapping[str, torch.Tensor], max_passes: int | None = None,
    new_lr_scale: float = 1.0,
) -> list[train.History]:  # fmt: skip
    """Train a stage at a cap of 60 passes.

    A stage that ends at the cap is rerun from its starting state at 120. `max_passes` (the smoke
    test's
    one pass) replaces both caps. `new_lr_scale` multiplies the new parameters' rates.
    """
    cap_first, cap_rerun = (
        (CAP_FIRST, CAP_RERUN) if max_passes is None else (max_passes, max_passes)
    )
    _, first = train.train_stage(
        net=net, world=world, stage=stage, samples=samples, seed=seed, encoders=encoders,
        max_passes=cap_first, new_lr_scale=new_lr_scale,
    )  # fmt: skip
    if not first.cap_hit:
        return [first]
    net.load_state_dict(start_state)
    _, again = train.train_stage(
        net=net, world=world, stage=stage, samples=samples, seed=seed, encoders=encoders,
        max_passes=cap_rerun, new_lr_scale=new_lr_scale,
    )  # fmt: skip
    return [first, again]


def assert_new_parameters_unchanged(
    *, net: model.UntangledNet, before: Mapping[str, torch.Tensor], arm: str
) -> None:
    """Raise when any new parameter (the gain `u_gamma` included) differs from its initial value.

    Args:
        net: The fitted network.
        before: The new parameters as the network was built.
        arm: The arm, for the message.

    Raises:
        CorrectnessFailureError: When a new parameter has moved (verdict row 1).
    """
    for name, value in net.named_parameters():
        if name in before and not torch.equal(value.detach().cpu(), before[name].cpu()):
            msg = f"{arm}: the new parameter {name} moved during a fit that reads no channel"
            raise CorrectnessFailureError(msg)


def fit_real(
    *, root: Path, arm: str, fold: str, seed: int, device: str, loader: RealLoader,
    make_samples: Callable[..., object] | None = None, max_passes: int | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Fit stages 2 and 3 of one real-fold arm from M*'s stored stage 1 state, and write the record.

    Args:
        root: This experiment's data folder.
        arm: The arm (F0r, FU, FU-S, or FS).
        fold: The fold.
        seed: The seed.
        device: "cuda" or "cpu".
        loader: Builds the fold's inputs.
        make_samples: Experiment 713's `make_samples` (default: `e00713_train.make_samples`).
        max_passes: Replaces the caps of 60 and 120 passes (the smoke test runs one pass).

    Returns:
        The record. A fit whose record exists is skipped and its record returned.
    """
    spec = core.ARMS[arm]
    if spec.where != "real" or spec.source == "stored":
        msg = f"{arm} is not a fitted real-fold arm"
        raise GuardError(msg)
    rec = record_path(root=root, arm=arm, fold=fold, seed=seed)
    if rec.exists():
        return json.loads(rec.read_text())
    prepare_device(device=device)
    started = time.monotonic()
    inputs = loader(arm, fold, seed, device)
    net = model.UntangledNet(seed=seed)
    net.load_m_star(state=torch.load(inputs.stage1_state, map_location="cpu"))
    if inputs.encoders == "frozen":
        net.freeze_encoders()
    net = net.to(device)
    sampler = make_samples if make_samples is not None else tr713.make_samples
    new_lr_scale = 1.0 if inputs.has_channels else 0.0
    new_before = {
        n: p.detach().clone() for n, p in net.named_parameters() if n.startswith(model.NEW_PREFIXES)
    }
    stages: list[dict[str, Any]] = []
    for stage in (2, 3):
        samples = sampler(world=inputs.world, stage=stage, placebo=False)
        scale = None
        if stage == 2 and inputs.has_channels:
            cut = tr713.target_end(world=inputs.world, stage=2)
            scale = net.calibrate_scale(
                batches=list(
                    train.calibration_batches(
                        samples=samples,
                        source=inputs.world.source,  # ty: ignore[unresolved-attribute]
                        cut=cut,  # ty: ignore[unresolved-attribute]
                    )
                )
            )
        start_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        histories = train_with_rerun(
            net=net, world=inputs.world, stage=stage, samples=samples, seed=seed,
            encoders=inputs.encoders, start_state=start_state, max_passes=max_passes,
            new_lr_scale=new_lr_scale,
        )  # fmt: skip
        if not inputs.has_channels:
            assert_new_parameters_unchanged(net=net, before=new_before, arm=arm)
        final = histories[-1]
        save_state(
            state=net.state_dict(),
            path=network_path(root=root, arm=arm, fold=fold, seed=seed, stage=stage),
        )
        stages.append({
            "stage": stage, "scale": scale, "histories": [h.as_dict() for h in histories],
            "stored_pass": final.stored_pass, "cap_hit": final.cap_hit,
            "still_falling": final.still_falling, "passes": len(final.val),
        })  # fmt: skip
    record = {
        "code_version": core.CODE_VERSION, "arm": arm, "fold": fold, "seed": seed,
        "device": device, "stages": stages, "dropped": dict(inputs.dropped),
        "fingerprint": inputs.fingerprint, "seconds": round(time.monotonic() - started, 1),
        "peak_rss_gb": peak_rss_gb(), "peak_gpu_gb": peak_gpu_gb(device=device),
    }  # fmt: skip
    write_json(path=rec, payload=record)
    return record


# --------------------------------------------------------------------------- the synthetic world


@dataclass
class SyntheticInputs:
    """What a synthetic fit reads.

    Attributes:
        train_segments: The segments of the training banks of seeds 3 and 4.
        train_index: Their samples.
        val_segments: The validation bank's segments.
        val_index: Its samples.
        d8_index: The Primary index of D8.
        target_std: D8's target standard deviation.
        clim_logit: The logit of B-clim's overall share of reverse flow (the sign head's start).
    """

    train_segments: Sequence[sy.SegmentArrays]
    train_index: sy.SampleIndex
    val_segments: Sequence[sy.SegmentArrays]
    val_index: sy.SampleIndex
    d8_index: int
    target_std: float
    clim_logit: float


SyntheticLoader = Callable[[str, int, str], SyntheticInputs]
"""A loader of (arm, seed, device) returning the fit's inputs."""


def synthetic_mode(*, arm: str) -> str:
    """Return the channel mode of a synthetic arm.

    The modes are `absent` (Y0), `untangled` (YU), `true` (YT), and `p_neg` (YS, which reads P(X<0)
    alone).
    """
    spec = core.ARMS[arm]
    if spec.where != "synthetic":
        msg = f"{arm} is not a synthetic arm"
        raise GuardError(msg)
    if spec.channel_subset is not None:
        if spec.channel_subset != ("p_neg",):
            msg = f"the synthetic arm {arm} reads a channel subset that has no mode"
            raise GuardError(msg)
        return "p_neg"
    return "true" if spec.true_channels else "untangled" if spec.channels else "absent"


def fit_synthetic(
    *,
    root: Path,
    arm: str,
    seed: int,
    device: str,
    loader: SyntheticLoader,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Fit one synthetic arm from a random start, and write its state and record.

    The network starts from random weights seeded by `seed`, trains on the training banks of seeds 3
    and 4, stops on the validation bank, and reads no test row.

    Args:
        root: This experiment's data folder.
        arm: Y0, YU, YT, or YS.
        seed: The seed.
        device: "cuda" or "cpu".
        loader: Builds the fit's inputs.
        max_epochs: Replaces the cap of 40 passes (the smoke test runs one pass).

    Returns:
        The record. A fit whose record exists is skipped and its record returned.
    """
    mode = synthetic_mode(arm=arm)
    rec = record_path(root=root, arm=arm, fold=None, seed=seed)
    if rec.exists():
        return json.loads(rec.read_text())
    prepare_device(device=device)
    started = time.monotonic()
    inputs = loader(arm, seed, device)
    sy.assert_no_world_leak(input_names=list(model.UntInputs.__dataclass_fields__))
    torch.manual_seed(seed)
    net = model.UntangledNet(
        seed=seed, path=mode != "absent", sign_head=True, clim_logit=inputs.clim_logit
    ).to(device)
    history = train.train_synthetic(
        net=net, train_segments=inputs.train_segments, train_index=inputs.train_index,
        val_segments=inputs.val_segments, val_index=inputs.val_index, seed=seed, mode=mode,
        d8_index=inputs.d8_index, target_std=inputs.target_std, device=device,
        **({} if max_epochs is None else {"max_epochs": max_epochs}),
    )  # fmt: skip
    save_state(state=net.state_dict(), path=network_path(root=root, arm=arm, fold=None, seed=seed))
    record = {
        "code_version": core.CODE_VERSION, "arm": arm, "fold": None, "seed": seed,
        "device": device, "mode": mode, "history": history.as_dict(),
        "stored_pass": history.stored_pass, "cap_hit": history.cap_hit,
        "still_falling": history.still_falling, "n_train": len(inputs.train_index),
        "n_val": len(inputs.val_index), "seconds": round(time.monotonic() - started, 1),
        "peak_rss_gb": peak_rss_gb(), "peak_gpu_gb": peak_gpu_gb(device=device),
    }  # fmt: skip
    write_json(path=rec, payload=record)
    return record


# --------------------------------------------------------------------------- loss curves


def loss_curve_data(*, record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the per-pass losses of a fit record, with the stored pass marked.

    For the loss-curve figure.

    A real-fold record gives one series per stage; a synthetic record gives one series with its two
    validation parts.

    Args:
        record: A fit record.

    Returns:
        `{"series": [{"label", "train", "val", "stored_pass", ...}]}`.

    Raises:
        ValueError: When the record holds no pass (nothing to draw).
    """
    series = []
    if "history" in record:
        h = record["history"]
        series.append(
            {
                "label": "one stage",
                **{k: h[k] for k in ("train", "val", "val_nll", "val_bce", "stored_pass")},
            }
        )
    else:
        for stage in record["stages"]:
            h = stage["histories"][-1]
            series.append(
                {
                    "label": f"stage {stage['stage']}",
                    **{k: h[k] for k in ("train", "val", "stored_pass")},
                }
            )
    if not series or not any(s["val"] for s in series):
        msg = "the fit record holds no pass, so there is no loss curve"
        raise ValueError(msg)
    return {"arm": record["arm"], "fold": record["fold"], "seed": record["seed"], "series": series}


# --------------------------------------------------------------------------- the runner


def failure_payload(*, exc: BaseException) -> dict[str, str]:
    """Return what a failure record stores: the exception's type, its message, and its traceback."""
    return {
        "error": type(exc).__name__, "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }  # fmt: skip


def run_jobs(
    *,
    root: Path,
    jobs: Sequence[tuple[str, str | None, int]],
    run_one: Callable[[str, str | None, int], dict],
    log: Callable[[str], None],
    rerun: bool = False,
) -> dict[str, int]:
    """Run fits one at a time; a fit that raises is recorded as failed and the run goes on.

    A failed fit leaves `<job>.failed.json` and is not run again unless `rerun` is true. A
    `CorrectnessFailureError` (verdict row 1), a `StoppedError` (verdict row 0), and a `GuardError`
    are never hidden as a failed fit: each propagates.

    Args:
        root: The data folder.
        jobs: The (arm, fold or None, seed) of each fit.
        run_one: Runs one fit.
        log: Receives progress lines.
        rerun: Whether to run a fit again that has a `.failed.json`.

    Returns:
        The counts of fits run, skipped, skipped as already failed, and failed.
    """
    counts = {"run": 0, "skipped": 0, "skipped_failed": 0, "failed": 0}
    for arm, fold, seed in jobs:
        rec = record_path(root=root, arm=arm, fold=fold, seed=seed)
        failed = rec.with_suffix(".failed.json")
        if rec.exists():
            counts["skipped"] += 1
            continue
        if failed.exists() and not rerun:
            counts["skipped_failed"] += 1
            continue
        try:
            run_one(arm, fold, seed)
            failed.unlink(missing_ok=True)
            counts["run"] += 1
        except CorrectnessFailureError, channels.StoppedError, GuardError:
            raise
        except Exception as exc:  # noqa: BLE001
            counts["failed"] += 1
            write_json(path=failed, payload=failure_payload(exc=exc))
            log(f"fit {rec.stem} failed: {type(exc).__name__}: {exc}")
    return counts


def arm_failed(*, root: Path, arm: str, folds: Sequence[str | None], seeds: Sequence[int]) -> bool:
    """Return whether an arm has a job whose last attempt failed: a `.failed.json` and no record."""
    for fold in folds:
        for seed in seeds:
            rec = record_path(root=root, arm=arm, fold=fold, seed=seed)
            if rec.with_suffix(".failed.json").exists() and not rec.exists():
                return True
    return False


def chain_exit_code(*, status: Mapping[str, Any]) -> int:
    """Return the exit status of a chain run.

    0 only when every fit succeeded, never 1 (the launcher's "slot busy").
    """
    return 0 if status.get("state") == "complete" else core.CHAIN_FAILED_EXIT


def fits_starting_line(*, family: str, hours: float, cores: str, now: str) -> str:
    """Return the line sent to the manager before each family of fits."""
    return f"fits starting: experiment 733, family {family}, projection {hours:.2f} GPU-process-hours at the cap, core {cores}, {now}"


def last_fit_ended_line(*, family: str, counts: Mapping[str, int], now: str) -> str:
    """Return the line printed after the last fit of a family."""
    return f"last fit ended: experiment 733, family {family}, {dict(counts)}, {now}"


def assert_fits_after_limits(*, root: Path, fit_records: Sequence[Path]) -> None:
    """Fail when a fit record is older than `limits.json`.

    (a fit must start after the limits were written).
    """
    limits = root / "limits.json"
    if not limits.exists():
        msg = "limits.json is missing, so no fit may start"
        raise GuardError(msg)
    older = [p for p in fit_records if p.exists() and p.stat().st_mtime < limits.stat().st_mtime]
    if older:
        msg = f"{len(older)} fit records are older than limits.json"
        raise CorrectnessFailureError(msg)


def d8_index_from(*, labels: Sequence[str]) -> int:
    """Return the Primary index of D8 among the Primaries' labels (sorted by Primary number)."""
    if labels.count(core.SCORED_PRIMARY) != 1:
        msg = "the Primaries' labels do not hold D8 exactly once"
        raise GuardError(msg)
    return list(labels).index(core.SCORED_PRIMARY)


def zero_check(*, values: np.ndarray) -> bool:
    """Return whether an array is all zero.

    (the check that a dropped channel stays zero in a fit input).
    """
    return not bool(np.any(values))

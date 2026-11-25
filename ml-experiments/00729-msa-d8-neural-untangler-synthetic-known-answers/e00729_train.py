"""Experiment 729: the training driver, the predictor, the timing gate, and the fit schedule.

A fit trains one network of one architecture on one arm and seed: AdamW (learning rate 1e-3, weight
decay 1e-4), batches of 256, an epoch of 8,000 windows, gradient clipping at 1.0, the learning rate
halved after 10 epochs with no gain, and a stop after 25 epochs with no gain in the validation
signed CRPS or at 200 epochs. The best epoch's weights are kept and saved atomically. Predictions
are stored decoded (lobe parameters or quantiles), so scoring needs no torch.
"""

import copy
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import e00729_arms as arms
import e00729_core as core
import e00729_dist as dist
import e00729_nets as nets
import numpy as np
import torch  # ty: ignore[unresolved-import]

INFER_BATCH: Final = 512
FITS_PER_ARCH_IN_SCREEN: Final = 6
"""Arms A and D, 3 seeds each."""
LADDER_FITS: Final = 29
"""The 9 ladder arms times 3 seeds, plus seeds 3 and 4 of arm D."""
EARLY_STOP_EPOCHS: Final = 70


def arch_seed(*, arch: str, seed: int) -> int:
    """Return the torch seed of a fit: it depends on the architecture and the seed."""
    return 1000 * (core.ARCHITECTURES.index(arch) + 1) + seed


def limit_gpu_memory(*, gigabytes: float = core.GPU_MEMORY_GB) -> None:
    """Cap this process's share of the GPU at `gigabytes` (no effect without a GPU)."""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gigabytes * 1024**3 / total))


def to_device(*, array: np.ndarray, device: str) -> torch.Tensor:
    """Return a float32 tensor on the device."""
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(device)


@torch.no_grad()
def predict_raw(
    *, net: torch.nn.Module, prepared: Sequence[arms.PreparedSegment], spec: arms.ArmSpec,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:  # fmt: skip
    """Return raw outputs `(P, outputs, 48)` and targets `(P, 4, 48)` of every (segment, day) pick.

    Args:
        net: The network (put in evaluation mode here).
        prepared: The prepared segments.
        spec: The arm.
        device: "cuda" or "cpu".

    Returns:
        The cropped raw outputs, the targets, and the picks, in segment then day order.
    """
    net.eval()
    picks = arms.all_picks(prepared=prepared)
    rng = np.random.default_rng(0)
    outs, tgts = [], []
    for i in range(0, len(picks), INFER_BATCH):
        x, t = arms.make_batch(
            prepared=prepared, picks=picks[i : i + INFER_BATCH], spec=spec, rng=rng, training=False
        )
        y = net(to_device(array=x, device=device))[:, :, arms.middle_slice(spec=spec)]
        outs.append(y.float().cpu().numpy())
        tgts.append(t)
    if not picks:
        return np.zeros((0, 1, 48)), np.zeros((0, 4, 48)), picks
    return np.concatenate(outs), np.concatenate(tgts), picks


def decode(*, raw: np.ndarray, spec: arms.ArmSpec) -> dict[str, np.ndarray]:
    """Decode raw outputs `(N, outputs, 48)` into the arrays that predictions store.

    Args:
        raw: Cropped raw outputs.
        spec: The arm.

    Returns:
        For a two-lobe head `p`, `mu_i`, `s_i`, `mu_r`, `s_r` and, with an auxiliary head, `aux_mu`
        and `aux_s`; for a quantile head `q` with the levels on the last axis. All `(N, 48)` or
        `(N, 48, 11)`.
    """
    t = torch.from_numpy(raw.astype(np.float32))
    if spec.head == "quantile":
        q = nets.quantile_values(raw=t)
        return {"q": q.permute(0, 2, 1).numpy()}
    out = {k: v.numpy() for k, v in nets.lobe_parameters(raw=t).items()}
    if spec.aux != "none":
        aux = t[:, nets.LOBE_OUTPUTS :]
        out["aux_mu"] = (nets.MEAN_SCALE * 5.0 * torch.nn.functional.softplus(aux[:, 0])).numpy()
        out["aux_s"] = (
            core.FLOOR_POINTS + nets.SPREAD_SCALE * torch.nn.functional.softplus(aux[:, 1])
        ).numpy()
    return out


def flatten_valid(
    *, decoded: dict[str, np.ndarray], targets: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Keep the valid half-hours, in pick then half-hour order (the order of `Rows`).

    Args:
        decoded: The output of `decode`.
        targets: The `(P, 4, 48)` targets.

    Returns:
        The decoded arrays restricted to valid half-hours, `(M,)` or `(M, 11)`, and the truth `(M,)`.
    """
    ok = targets[:, 1] > 0.5
    return {k: v[ok] for k, v in decoded.items()}, targets[:, 0][ok]


def validation_crps(*, decoded: dict[str, np.ndarray], targets: np.ndarray) -> float:
    """Return the mean signed CRPS of a decoded prediction over the valid half-hours."""
    flat, x = flatten_valid(decoded=decoded, targets=targets)
    if "q" in flat:
        return float(dist.quantile_crps(q=flat["q"], y=x).mean())
    lobes = dist.Lobes(flat["p"], flat["mu_i"], flat["s_i"], flat["mu_r"], flat["s_r"])
    return float(dist.mixture_crps(lobes=lobes, y=x).mean())


@dataclass
class FitResult:
    """What a fit records.

    Attributes:
        arm: The arm.
        arch: The architecture.
        seed: The seed.
        epochs: The epochs run.
        best_epoch: The epoch with the lowest validation signed CRPS.
        best_val_crps: That CRPS, in points.
        n_weights: The number of trainable weights.
        seconds: Wall-clock seconds.
        seconds_per_epoch: Mean seconds per epoch (training and validation).
        history: The validation CRPS of every epoch.
    """

    arm: str
    arch: str
    seed: int
    epochs: int
    best_epoch: int
    best_val_crps: float
    n_weights: int
    seconds: float
    seconds_per_epoch: float
    history: list[float]

    def as_dict(self) -> dict:
        """Return the fields as a plain dictionary."""
        return asdict(self)


def save_state(*, state: dict, path: Path) -> None:
    """Save a state dictionary atomically (temporary file in the same folder, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def fit(
    *, spec: arms.ArmSpec, arch: str, seed: int, train: Sequence[arms.PreparedSegment],
    val: Sequence[arms.PreparedSegment], device: str, max_epochs: int = core.MAX_EPOCHS,
    out_path: Path | None = None, log: Callable[[str], None] = lambda _: None,
) -> tuple[torch.nn.Module, FitResult]:  # fmt: skip
    """Fit one network.

    Args:
        spec: The arm.
        arch: "dae", "unet", or "bilstm".
        seed: The seed (it sets the initial weights and the sampling).
        train: The prepared segments of the seed's training bank.
        val: The prepared segments of the validation bank.
        device: "cuda" or "cpu".
        max_epochs: The cap on epochs (200; the timing gate passes 5).
        out_path: Where to save the best weights, or None.
        log: Receives one progress line per epoch (never a month or a date).

    Returns:
        The network with its best weights, and the record.
    """
    seed_all = arch_seed(arch=arch, seed=seed)
    torch.manual_seed(seed_all)
    rng = np.random.default_rng(seed_all)
    net = nets.build_net(
        arch=arch, c_in=arms.n_channels(spec=spec), steps=spec.steps,
        n_out=nets.n_outputs(head=spec.head, aux=spec.aux),
    ).to(device)  # fmt: skip
    opt = torch.optim.AdamW(net.parameters(), lr=core.LEARNING_RATE, weight_decay=core.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=core.LR_PATIENCE
    )
    crop = arms.middle_slice(spec=spec)
    best, best_epoch, best_state = float("inf"), 0, copy.deepcopy(net.state_dict())
    history: list[float] = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        net.train()
        picks = arms.sample_picks(prepared=train, n=core.EPOCH_WINDOWS, rng=rng)
        for i in range(0, len(picks), core.BATCH_SIZE):
            x, t = arms.make_batch(
                prepared=train, picks=picks[i : i + core.BATCH_SIZE], spec=spec, rng=rng,
                training=True,
            )  # fmt: skip
            raw = net(to_device(array=x, device=device))[:, :, crop]
            loss = nets.batch_loss(
                raw=raw, targets=to_device(array=t, device=device), head=spec.head, aux=spec.aux
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), core.CLIP_NORM)
            opt.step()
        raw_v, t_v, _ = predict_raw(net=net, prepared=val, spec=spec, device=device)
        score = validation_crps(decoded=decode(raw=raw_v, spec=spec), targets=t_v)
        history.append(score)
        sched.step(score)
        if score < best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(net.state_dict())
        log(f"{spec.name} {arch} seed {seed} epoch {epoch}: validation CRPS {score:.4f}")
        if epoch - best_epoch >= core.STOP_PATIENCE:
            break
    net.load_state_dict(best_state)
    if out_path is not None:
        save_state(state=best_state, path=out_path)
    seconds = time.perf_counter() - started
    return net, FitResult(
        arm=spec.name, arch=arch, seed=seed, epochs=len(history), best_epoch=best_epoch,
        best_val_crps=best, n_weights=nets.count_weights(net=net), seconds=seconds,
        seconds_per_epoch=seconds / max(len(history), 1), history=history,
    )  # fmt: skip


def predictions(
    *, net: torch.nn.Module, prepared: Sequence[arms.PreparedSegment], spec: arms.ArmSpec,
    device: str,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Return the decoded predictions on every valid half-hour of a bank, in `Rows` order."""
    raw, tgt, _ = predict_raw(net=net, prepared=prepared, spec=spec, device=device)
    flat, _ = flatten_valid(decoded=decode(raw=raw, spec=spec), targets=tgt)
    return flat


def save_predictions(*, arrays: dict[str, np.ndarray], path: Path) -> None:
    """Save decoded predictions atomically as a compressed `.npz`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **{k: v.astype(np.float32) for k, v in arrays.items()})
    tmp.replace(path)


# --------------------------------------------------------------------------- the timing gate


def projected_hours(*, seconds_per_epoch: dict[str, float], epochs: int) -> float:
    """Return the GPU-process-hours of all 47 fits at `epochs` epochs each.

    The 6 screen fits of each architecture use that architecture's measured time per epoch, and the
    29 ladder fits use the slowest architecture's time (the chosen architecture is not yet known).

    Args:
        seconds_per_epoch: The measured seconds per epoch of each architecture on arm D.
        epochs: The epochs of every fit.

    Returns:
        The projection in hours.
    """
    screen = sum(FITS_PER_ARCH_IN_SCREEN * t for t in seconds_per_epoch.values())
    ladder = LADDER_FITS * max(seconds_per_epoch.values())
    return (screen + ladder) * epochs / 3600.0


def timing_gate_report(*, seconds_per_epoch: dict[str, float]) -> dict[str, float | bool]:
    """Return the projections at the epoch cap and at an early stop, and whether they fit the budget."""
    cap = projected_hours(seconds_per_epoch=seconds_per_epoch, epochs=core.MAX_EPOCHS)
    early = projected_hours(seconds_per_epoch=seconds_per_epoch, epochs=EARLY_STOP_EPOCHS)
    return {
        "hours_at_cap": cap, "hours_at_early_stop": early, "budget_hours": core.GPU_BUDGET_HOURS,
        "within_budget": cap <= core.GPU_BUDGET_HOURS,
    }  # fmt: skip


def timing_gate(
    *, train: Sequence[arms.PreparedSegment], val: Sequence[arms.PreparedSegment], device: str,
    log: Callable[[str], None] = lambda _: None,
) -> dict[str, float | bool]:  # fmt: skip
    """Run `TIMING_EPOCHS` epochs of each architecture on arm D and project the hours."""
    spec = arms.ARM_SPECS["D"]
    per_epoch = {}
    for arch in core.ARCHITECTURES:
        _, result = fit(
            spec=spec, arch=arch, seed=0, train=train, val=val, device=device,
            max_epochs=core.TIMING_EPOCHS, log=log,
        )  # fmt: skip
        per_epoch[arch] = result.seconds_per_epoch
    return {**timing_gate_report(seconds_per_epoch=per_epoch)} | {
        f"seconds_per_epoch_{a}": t for a, t in per_epoch.items()
    }


# --------------------------------------------------------------------------- the fit schedule


def screen_jobs() -> list[tuple[str, str, int]]:
    """Return the 18 screen fits as (arm, architecture, seed): arms A and D, 3 architectures."""
    return [
        (arm, arch, s) for arch in core.ARCHITECTURES for arm in ("A", "D") for s in core.ARCH_SEEDS
    ]


def ladder_jobs(*, arch: str) -> list[tuple[str, str, int]]:
    """Return the 29 ladder fits on the chosen architecture: 9 arms x 3 seeds, and D seeds 3, 4."""
    jobs = [(arm, arch, s) for arm in core.LADDER_ARMS for s in core.ARCH_SEEDS]
    return jobs + [("D", arch, s) for s in core.MARGIN_SEEDS if s not in core.ARCH_SEEDS]


def all_jobs_count(*, arch: str) -> int:
    """Return the number of fits of the whole plan (47)."""
    return len(screen_jobs()) + len(ladder_jobs(arch=arch))

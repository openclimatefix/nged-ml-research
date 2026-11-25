"""Experiment 730: the fit loop, the predictions, the timing gate, and the fit schedule (torch).

New arms fit with a cap of 300 epochs and stop early on the validation Brier score of P(X < 0), so
that the demand head cannot move the stopping point through a metric it does not report. Experiment
729's arms A, D, and D-sign are refitted with experiment 729's own `fit` (cap 200, early stop on
the validation signed CRPS). The causal arm validates on 4 fixed cuts per validation window.
"""

import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import e00730_paths  # noqa: F401

# isort: split
import e00729_arms as arms729
import e00729_core as core729
import e00729_train as train729
import e00730_arms as arms
import e00730_core as core
import e00730_nets as nets
import numpy as np
import torch  # ty: ignore[unresolved-import]

INFER_BATCH: Final = 256
MIN_FREE_FRACTION: Final = 0.0
STATE_SUFFIX: Final = ".pt"


def fit_seed(*, arm: str, arch: str, seed: int) -> int:
    """Return the integer seed of one fit, the same for every run of the job."""
    return train729.arch_seed(arch=arch, seed=seed) + sum(map(ord, arm)) * 7919


@dataclass
class FitRecord:
    """What a fit records (counts and scores only).

    Attributes:
        arm: The arm.
        arch: The architecture.
        seed: The seed.
        epochs: The epochs run.
        best_epoch: The epoch with the lowest validation Brier score.
        best_val_brier: That score.
        n_weights: The number of trainable weights.
        seconds: Wall-clock seconds.
        seconds_per_epoch: Mean seconds per epoch.
        peak_memory_gb: Peak GPU memory in gigabytes (0 on a CPU).
        history: The validation Brier score of every epoch.
        pretrained_digest: SHA-256 of the pre-trained weight file the fit started from, or "".
    """

    arm: str
    arch: str
    seed: int
    epochs: int
    best_epoch: int
    best_val_brier: float
    n_weights: int
    seconds: float
    seconds_per_epoch: float
    peak_memory_gb: float
    history: list[float]
    pretrained_digest: str = ""

    def as_dict(self) -> dict:
        """Return the fields as a plain dictionary."""
        return asdict(self)


def brier(*, p: np.ndarray, x: np.ndarray) -> float:
    """Return the mean squared difference between P(X < 0) and the indicator of X < 0."""
    return float(np.mean((p - (x < 0.0)) ** 2))


def validation_cuts(*, n_picks: int) -> np.ndarray:
    """Return the `CAUSAL_VALIDATION_CUTS` fixed cuts of a causal arm, `(n_picks, cuts)`.

    The cuts are evenly spaced over the scored day's 48 half-hours and never drawn at random, so
    every epoch validates on the same cuts.
    """
    step = core729.WINDOW_STEPS_24 // core.CAUSAL_VALIDATION_CUTS
    cuts = core729.MIDDLE_START + step * np.arange(core.CAUSAL_VALIDATION_CUTS)
    return np.tile(cuts, (n_picks, 1))


def predict(
    *,
    net: torch.nn.Module,
    prepared: Sequence[arms.PreparedSegment],
    spec: arms.ArmSpec,
    device: str,
    hidden: np.ndarray | None = None,
    cut_pos: np.ndarray | None = None,
    picks: list[tuple[int, int]] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[tuple[int, int]]]:
    """Return decoded outputs `(P, 48)` and the targets `(P, 6, 48)` of every pick.

    Args:
        net: The network (put in evaluation mode here).
        prepared: The prepared segments.
        spec: The arm.
        device: "cuda" or "cpu".
        hidden: Fixed hidden positions `(P, 96)`, or None.
        cut_pos: Fixed causal cuts `(P,)`, or None.
        picks: The picks, or None for every scored day.

    Returns:
        The decoded arrays, the targets, and the picks.
    """
    net.eval()
    picks = arms729.all_picks(prepared=prepared) if picks is None else picks
    outs, tgts = [], []
    with torch.no_grad():
        for i in range(0, len(picks), INFER_BATCH):
            j = slice(i, i + INFER_BATCH)
            x, t, _ = arms.make_eval_batch(
                prepared=prepared,
                picks=picks[j],
                spec=spec,
                hidden=None if hidden is None else hidden[j],
                cut_pos=None if cut_pos is None else cut_pos[j],
            )
            y = net(torch.from_numpy(x).to(device))[:, :, arms.middle()]
            outs.append(y.float().cpu())
            tgts.append(t)
    if not picks:
        return {}, np.zeros((0, 6, 48), dtype=np.float32), picks
    decoded = nets.decode(raw=torch.cat(outs))
    return {k: v.numpy() for k, v in decoded.items()}, np.concatenate(tgts), picks


def valid_flat(*, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return `values` `(P, 48)` on the valid half-hours, in pick then half-hour order."""
    return values[targets[:, 1] > 0.5]


def validation_brier(
    *,
    net: torch.nn.Module,
    prepared: Sequence[arms.PreparedSegment],
    spec: arms.ArmSpec,
    device: str,
) -> float:
    """Return the validation Brier score (the mean over the fixed cuts for a causal arm)."""
    if not spec.causal:
        decoded, t, _ = predict(net=net, prepared=prepared, spec=spec, device=device)
        return brier(
            p=valid_flat(values=decoded["p"], targets=t), x=valid_flat(values=t[:, 0], targets=t)
        )
    picks = arms729.all_picks(prepared=prepared)
    cuts = validation_cuts(n_picks=len(picks))
    scores = []
    for k in range(cuts.shape[1]):
        decoded, t, _ = predict(
            net=net, prepared=prepared, spec=spec, device=device, cut_pos=cuts[:, k], picks=picks
        )
        scores.append(
            brier(
                p=valid_flat(values=decoded["p"], targets=t),
                x=valid_flat(values=t[:, 0], targets=t),
            )
        )
    return float(np.mean(scores))


def save_state(*, state: dict, path: Path) -> None:
    """Save a state dictionary atomically."""
    train729.save_state(state=state, path=path)


def fit(
    *,
    spec: arms.ArmSpec,
    arch: str,
    seed: int,
    train: Sequence[arms.PreparedSegment],
    val: Sequence[arms.PreparedSegment],
    device: str,
    max_epochs: int = core.MAX_EPOCHS,
    out_path: Path | None = None,
    frozen_encoders: bool = False,
    pretrained: Path | None = None,
    log: Callable[[str], None] = lambda _: None,
) -> tuple[torch.nn.Module, FitRecord]:
    """Fit one network of experiment 730.

    Args:
        spec: The arm.
        arch: One of `core.ARCHITECTURES`.
        seed: The seed.
        train: The prepared segments of the seed's training bank.
        val: The prepared segments of the validation bank.
        device: "cuda" or "cpu".
        max_epochs: The cap on epochs (300; the timing gate passes 5).
        out_path: Where to save the best weights, or None.
        frozen_encoders: Freeze the encoders (arm SDe-FR).
        pretrained: The pre-trained weight file the encoders start from (arms SDe-PT and SDe-FR).
        log: Receives one progress line per epoch (never a month or a date).

    Returns:
        The network with its best weights, and the record.
    """
    seed_all = fit_seed(arm=spec.name, arch=arch, seed=seed)
    torch.manual_seed(seed_all)
    rng = np.random.default_rng(seed_all)
    net = nets.build_net(arch=arch, spec=spec).to(device)
    digest = ""
    if spec.inputs == "enc" and pretrained is not None:
        import e00730_pretrain as pretrain

        digest = pretrain.weight_digest(path=pretrained)
        net.front.load_pretrained(state=torch.load(pretrained, map_location=device))
        for p in net.front.parameters():
            p.requires_grad = not frozen_encoders
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=core.LEARNING_RATE, weight_decay=core.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=core.LR_PATIENCE
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, best_epoch, best_state = float("inf"), 0, copy.deepcopy(net.state_dict())
    history: list[float] = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        net.train()
        picks = arms729.sample_picks(prepared=train, n=core.EPOCH_WINDOWS, rng=rng)
        for i in range(0, len(picks), core.BATCH_SIZE):
            x, t, k = arms.make_batch(
                prepared=train,
                picks=picks[i : i + core.BATCH_SIZE],
                spec=spec,
                rng=rng,
                training=True,
            )
            raw = net(torch.from_numpy(x).to(device))[:, :, arms.middle()]
            loss = nets.batch_loss(
                raw=raw,
                targets=torch.from_numpy(t).to(device),
                keeps=torch.from_numpy(k).to(device),
                lam=spec.lam,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, core.CLIP_NORM)
            opt.step()
        score = validation_brier(net=net, prepared=val, spec=spec, device=device)
        history.append(score)
        sched.step(score)
        if score < best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(net.state_dict())
        log(f"{spec.name} {arch} seed {seed} epoch {epoch}: validation Brier {score:.5f}")
        if epoch - best_epoch >= core.STOP_PATIENCE:
            break
    net.load_state_dict(best_state)
    if out_path is not None:
        save_state(state=best_state, path=out_path)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    return net, FitRecord(
        arm=spec.name,
        arch=arch,
        seed=seed,
        epochs=len(history),
        best_epoch=best_epoch,
        best_val_brier=best,
        n_weights=nets.count_weights(net=net),
        seconds=seconds,
        seconds_per_epoch=seconds / max(len(history), 1),
        peak_memory_gb=peak,
        history=history,
        pretrained_digest=digest,
    )


def refit_729_arm(
    *,
    arm: str,
    arch: str,
    seed: int,
    train: Sequence[arms729.PreparedSegment],
    val: Sequence[arms729.PreparedSegment],
    device: str,
    out_path: Path | None,
) -> tuple[torch.nn.Module, dict]:
    """Refit one of experiment 729's arms (A, D, D-sign) with experiment 729's own `fit`."""
    net, record = train729.fit(
        spec=arms729.ARM_SPECS[arm], arch=arch, seed=seed, train=train, val=val, device=device,
        max_epochs=core729.MAX_EPOCHS, out_path=out_path,
    )  # fmt: skip
    return net, record.as_dict()


# --------------------------------------------------------------------------- the schedule


def screen_jobs() -> list[tuple[str, str, int]]:
    """Return the architecture screen: arms S-A and S on 4 architectures and 3 seeds (24 fits)."""
    return [
        (a, arch, s)
        for a in core.SCREEN_ARMS
        for arch in core.ARCHITECTURES
        for s in core.ARCH_SEEDS
    ]


def style_jobs(*, arch: str) -> list[tuple[str, str, int]]:
    """Return the masking-style fits SD-0, SD-K, SD-W, and SD-R (5 seeds each)."""
    return [(f"SD-{s}", arch, seed) for s in core.STYLES for seed in core.MARGIN_SEEDS]


def timing_gate(
    *, seconds_per_epoch: dict[str, float], peak_memory_gb: float, manager_approved: bool
) -> dict[str, float | bool]:
    """Return the timing gate's report and whether it passes (see `core.timing_gate_report`)."""
    report = core.timing_gate_report(
        seconds_per_epoch=seconds_per_epoch, peak_memory_gb=peak_memory_gb
    )
    return {
        **report,
        "pass": core.timing_gate_passes(report=report, manager_approved=manager_approved),
    }

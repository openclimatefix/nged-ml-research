"""Experiment 730: the fit loop, the predictions, the timing gate, and the fit schedule (torch).

New arms fit with a cap of 300 epochs and stop early on the validation Brier score of P(X < 0), so
that the demand head cannot move the stopping point through a metric it does not report. Experiment
729's arms A, D, and D-sign are refitted with experiment 729's own `fit` (cap 200, early stop on
the validation signed CRPS). The causal arm validates on 4 fixed cuts per validation window.
"""

import copy
import time
import zlib
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
import e00730_masks as masks
import e00730_nets as nets
import e00730_score as score
import numpy as np
import torch

INFER_BATCH: Final = 256
MIN_FREE_FRACTION: Final = 0.0
STATE_SUFFIX: Final = ".pt"


def fit_seed(*, arm: str, arch: str, seed: int) -> int:
    """Return the integer seed of one fit: the same for every run of a job, distinct per job."""
    arm_code = zlib.crc32(arm.encode()) % 1_000_003
    return 1000 * (core.ARCHITECTURES.index(arch) + 1) + seed + arm_code * 7919


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
    """Return the fixed validation cuts of a causal arm, `(n_picks, CAUSAL_VALIDATION_CUTS)`.

    The cuts are scored half-hours `0..47` drawn from seed 730 (`masks.fixed_cuts`), 4 per window,
    and never drawn again, so every epoch validates on the same cuts.
    """
    return masks.fixed_cuts(n_windows=n_picks)


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
    """Return the validation Brier score (a causal arm reads each window at its 4 fixed cuts)."""
    if not spec.causal:
        decoded, t, _ = predict(net=net, prepared=prepared, spec=spec, device=device)
        return brier(
            p=valid_flat(values=decoded["p"], targets=t), x=valid_flat(values=t[:, 0], targets=t)
        )
    picks = arms729.all_picks(prepared=prepared)
    cuts = validation_cuts(n_picks=len(picks))
    scores = []
    for j in range(cuts.shape[1]):
        decoded, t, _ = predict(
            net=net, prepared=prepared, spec=spec, device=device,
            cut_pos=core729.MIDDLE_START + cuts[:, j], picks=picks,
        )  # fmt: skip
        ok = score.read_at_cuts(values=t[:, 1], cuts=cuts[:, j]) > 0.5
        p = score.read_at_cuts(values=decoded["p"], cuts=cuts[:, j])[ok]
        x = score.read_at_cuts(values=t[:, 0], cuts=cuts[:, j])[ok]
        scores.append(brier(p=p, x=x))
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
    pretrained: Path | None = None,
    d8_index: int = 0,
    n_primaries: int = 16,
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
        pretrained: The pre-trained weight file the encoders start from (arm SDe-PT only).
        d8_index: D8's row of the embedding (its index in experiment 397's table of Primaries).
        n_primaries: The rows of the embedding (16).
        log: Receives one progress line per epoch (never a month or a date).

    Returns:
        The network with its best weights, and the record.
    """
    seed_all = fit_seed(arm=spec.name, arch=arch, seed=seed)
    torch.manual_seed(seed_all)
    rng = np.random.default_rng(seed_all)
    if spec.name == "SDe-FR" and pretrained is not None:
        msg = "SDe-FR starts from fresh weights and takes no pre-trained file"
        raise ValueError(msg)
    if spec.name == "SDe-PT" and pretrained is None:
        msg = "SDe-PT starts from the pre-trained weights and needs the file"
        raise ValueError(msg)
    net = nets.build_net(arch=arch, spec=spec, d8_index=d8_index, n_primaries=n_primaries)
    net = net.to(device)
    digest = ""
    if spec.inputs == "enc" and pretrained is not None:
        import e00730_pretrain as pretrain

        digest = pretrain.weight_digest(path=pretrained)
        if net.front is None:
            msg = "an encoder arm builds a network with the encoder front"
            raise ValueError(msg)
        net.front.load_pretrained(state=torch.load(pretrained, map_location=device))
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
        spec=arms729.ARM_SPECS[core.ARM_729_KEY.get(arm, arm)], arch=arch, seed=seed,
        train=train, val=val, device=device, max_epochs=core729.MAX_EPOCHS, out_path=out_path,
    )  # fmt: skip
    return net, record.as_dict()


# --------------------------------------------------------------------------- the schedule


def masked_demand(
    *,
    net: torch.nn.Module,
    prepared: Sequence[arms.PreparedSegment],
    spec: arms.ArmSpec,
    device: str,
    style: str,
) -> dict[str, np.ndarray]:
    """Return the demand output on every valid half-hour with the fixed masks of `style`.

    The mask of a window depends on the style and the window id (segment and base day) alone, so
    every arm sees the same masks, whatever the order of the picks.

    Args:
        net: The network.
        prepared: The prepared segments of a bank.
        spec: The arm.
        device: "cuda" or "cpu".
        style: The masking style of the evaluation.

    Returns:
        `demand_mu`, the true `base` demand, `hidden` (the half-hour was hidden), and `p`, each
        flattened over the valid half-hours in `Rows` order.
    """
    picks = arms729.all_picks(prepared=prepared)
    hidden = masks.fixed_hidden_for_picks(
        style=style,
        picks=picks,
        offsets=[p.offsets for p in prepared],
        segment_ids=[p.source_index for p in prepared],
    )
    decoded, t, _ = predict(
        net=net, prepared=prepared, spec=spec, device=device, hidden=hidden, picks=picks
    )
    ok = t[:, 1] > 0.5
    scored = hidden[:, core729.MIDDLE_START : core729.MIDDLE_START + 48]
    return {
        "demand_mu": decoded["demand_mu"][ok],
        "base": t[:, arms.BASE_ROW][ok],
        "hidden": scored[ok],
        "p": decoded["p"][ok],
    }


ATTENTION_GROUPS: Final = ("engine_reversal", "solar_reversal", "import_only", "single_engine_day")
"""The attention maps of figure 8: three day types averaged, and the first engine-reversal day."""


def attention_group(*, target: np.ndarray) -> str:
    """Return the day type of one pick from its targets `(6, 48)`: truth, valid, engine, solar, ....

    A day with a reverse-flow half-hour is an engine day when its largest engine output is at least
    its largest solar output, and a solar day otherwise; any other day is import-only.
    """
    reverse = bool(((target[0] < 0.0) & (target[1] > 0.5)).any())
    if not reverse:
        return "import_only"
    return "engine_reversal" if target[2].max() >= target[3].max() else "solar_reversal"


def attention_maps(
    *,
    net: torch.nn.Module,
    prepared: Sequence[arms.PreparedSegment],
    spec: arms.ArmSpec,
    device: str,
) -> dict[str, np.ndarray]:
    """Return figure 8's attention maps `(48, 96)`: scored half-hour by window position.

    The transformer returns its head-averaged attention in each layer when asked. The map of a day
    is the mean over the layers of the rows of the scored day, and a map of a day type is the mean
    over the days of that type in the bank. A group with no day is left out.

    Args:
        net: A network whose backbone is a transformer.
        prepared: The prepared segments of the scored bank.
        spec: The arm.
        device: "cuda" or "cpu".

    Returns:
        The maps by `ATTENTION_GROUPS`.
    """
    net.eval()
    picks = arms729.all_picks(prepared=prepared)
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    single: np.ndarray | None = None
    with torch.no_grad():
        for i in range(0, len(picks), INFER_BATCH):
            x, t, _ = arms.make_eval_batch(
                prepared=prepared, picks=picks[i : i + INFER_BATCH], spec=spec
            )
            _, layers = net(torch.from_numpy(x).to(device), return_attention=True)
            weights = torch.stack(layers).mean(dim=0)[:, arms.middle(), :].float().cpu().numpy()
            for w, target in zip(weights, t, strict=True):
                group = attention_group(target=target)
                sums[group] = sums.get(group, 0.0) + w
                counts[group] = counts.get(group, 0) + 1
                if single is None and group == "engine_reversal":
                    single = w
    out = {g: sums[g] / counts[g] for g in sums}
    if single is not None:
        out["single_engine_day"] = single
    return out


def causal_predictions(
    *,
    net: torch.nn.Module,
    prepared: Sequence[arms.PreparedSegment],
    spec: arms.ArmSpec,
    device: str,
    k: int,
) -> np.ndarray:
    """Return P(X < 0) on every valid half-hour when `k` half-hours after each one are visible.

    For each scored half-hour `t` the window is cut at `t + k` and the output is read at `t`, so
    every window takes 48 passes. At `k = 48` nothing is cut (the two-sided protocol).
    """
    picks = arms729.all_picks(prepared=prepared)
    columns: list[np.ndarray] = []
    targets = np.zeros((0, 6, 48), dtype=np.float32)
    for t in range(core729.WINDOW_STEPS_24):
        pos = score.causal_cut(t=t, k=k)
        cut = None if pos is None else np.full(len(picks), core729.MIDDLE_START + pos)
        decoded, targets, _ = predict(
            net=net, prepared=prepared, spec=spec, device=device, cut_pos=cut, picks=picks
        )
        columns.append(decoded["p"][:, t])
    return valid_flat(values=np.stack(columns, axis=1), targets=targets)

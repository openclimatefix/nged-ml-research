"""Experiment 732: the fit loop, the predictions, the per-epoch record, and the timing gate (torch).

Every fit uses experiment 730's optimiser and stopping rule: AdamW with a learning rate of 1e-3,
weight decay of 1e-4, batch size 256, gradient clipping at 1.0, the learning rate halved after 10
epochs with no gain, and a stop after 25 epochs with no gain in the validation Brier score of
P(X<0) on the plain validation bank (or at 300 epochs). Each fit also records, at every epoch, the
validation solar error on sun-up half-hours and the validation Brier score of the step event on the
validation transfer overlay, so the write-up can say whether the solar head was still improving.
"""

import copy
import resource
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import e00729_train as train729
import e00730_train as train730
import e00732_arms as arms
import e00732_core as core
import e00732_losses as losses
import e00732_nets as nets
import e00732_overlays as ov
import numpy as np
import torch

INFER_BATCH: Final = 256
LR_PATIENCE: Final = train730.core.LR_PATIENCE
STOP_PATIENCE: Final = train730.core.STOP_PATIENCE
LEARNING_RATE: Final = train730.core.LEARNING_RATE
WEIGHT_DECAY: Final = train730.core.WEIGHT_DECAY
CLIP_NORM: Final = train730.core.CLIP_NORM


@dataclass
class FitRecord:
    """What a fit records (counts and scores only).

    Attributes:
        arm: The arm.
        arch: The architecture.
        seed: The seed.
        epochs: The epochs run.
        best_epoch: The epoch with the lowest validation Brier score of the sign.
        best_val_brier: That score.
        n_weights: The trainable weights.
        seconds: Wall-clock seconds.
        seconds_per_epoch: Mean seconds an epoch.
        peak_memory_gb: Peak GPU memory in gigabytes (0 on a CPU).
        history: The validation Brier score of the sign at every epoch.
        solar_history: The validation solar mean absolute error on sun-up half-hours at every epoch.
        step_history: The validation Brier score of the step event at every epoch.
        unpaired_share: The share of training windows that took no pair, over the whole fit.
        train_terms: For each loss term ("sign", "solar", "step", "size", "injection", "night", "monotone")
            that the arm has, and for "total" (the weighted loss), the mean unweighted value on the
            training batches of each epoch.
        val_terms: The same terms on one fixed set of validation triples, evaluated after each epoch.
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
    solar_history: list[float] = field(default_factory=list)
    step_history: list[float] = field(default_factory=list)
    unpaired_share: float = 0.0
    train_terms: dict[str, list[float]] = field(default_factory=dict)
    val_terms: dict[str, list[float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Return the fields as a plain dictionary."""
        return asdict(self)


def tensor(*, array: np.ndarray | None, device: str) -> torch.Tensor | None:
    """Return a float32 tensor on the device (None stays None)."""
    if array is None:
        return None
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(device)


def forward_members(*, net: torch.nn.Module, batch: arms.Batch, device: str) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run the plain, injected, and raised windows through the network in one pass and crop the scored day.

    Both members of a pair pass through the same function in the same pass, because the networks
    have no dropout and the U-net's group normalisation normalises each window on its own.
    """
    members = [batch.x] + [m for m in (batch.xi, batch.xc) if m is not None]
    stacked = tensor(array=np.concatenate(members), device=device)
    assert stacked is not None
    raw = net(stacked)[:, :, arms.MID : arms.MID + 48]
    parts = list(torch.split(raw, len(batch.x)))
    raw_i = parts[1] if batch.xi is not None else None
    raw_c = parts[-1] if batch.xc is not None else None
    return parts[0], raw_i, raw_c


def batch_loss(*, net: torch.nn.Module, spec: core.ArmSpec732, batch: arms.Batch, device: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the weighted loss of a batch of triples and its unweighted terms."""
    raw_w, raw_i, raw_c = forward_members(net=net, batch=batch, device=device)
    values = losses.terms(
        raw_w=raw_w, raw_i=raw_i, raw_c=raw_c, t=tensor(array=batch.t, device=device),  # ty: ignore[invalid-argument-type]
        ti=tensor(array=batch.ti, device=device), keeps=tensor(array=batch.keeps, device=device),  # ty: ignore[invalid-argument-type]
        delta=tensor(array=batch.delta, device=device), pair=tensor(array=batch.pair, device=device),  # ty: ignore[invalid-argument-type]
    )  # fmt: skip
    return losses.total(values=values, weights=spec.weights), values


@dataclass
class Predictions:
    """The decoded outputs of a net on every pick, and what was fed to it.

    Attributes:
        plain: The decoded outputs of the plain windows, each `(P, 48)`.
        inj: The decoded outputs of the injected windows, or None.
        raised: The decoded outputs of the raised-sky windows, or None.
        t: The plain targets `(P, 8, 48)`.
        ti: The injected targets, or None.
        pair: 1 where the window has a pair `(P,)`.
        delta: The scored delta `(P, 48)`.
        cosz: The cosine of the solar zenith angle `(P, 48)`.
        csi: The clear-sky index `(P, 48)` of the plain window.
        picks: The picks.
    """

    plain: dict[str, np.ndarray]
    inj: dict[str, np.ndarray] | None
    raised: dict[str, np.ndarray] | None
    t: np.ndarray
    ti: np.ndarray | None
    pair: np.ndarray
    delta: np.ndarray
    cosz: np.ndarray
    csi: np.ndarray
    picks: list[tuple[int, int]]


@torch.no_grad()
def predict(
    *, net: torch.nn.Module, prepared: Sequence[arms.Prepared], device: str, index: ov.PairIndex | None = None,
    amount: float | None = None, eps: float | None = None, picks: list[tuple[int, int]] | None = None,
    hidden: np.ndarray | None = None, extra_heads: bool = True, hide_bsp: bool = False,
) -> Predictions:  # fmt: skip
    """Return the net's outputs on every scored day, with the injected and raised members when asked.

    Args:
        net: The network (put in evaluation mode here).
        prepared: The prepared segments.
        device: "cuda" or "cpu".
        index: The pair index of the bank, for `amount`.
        amount: The injected amount of the test pair, or None for no injected member.
        eps: The raised-sky factor of the raised member, or None for no raised member.
        picks: The picks, or None for every scored day.
        hidden: Fixed hidden positions `(P, 96)`, or None.
        extra_heads: Whether the net has the extra heads (the members need them).
        hide_bsp: Whether to zero the BSP channel and its mask (table T5).

    Returns:
        The predictions.
    """
    net.eval()
    picks = arms.all_picks(prepared=prepared) if picks is None else picks
    spec = arms.eval_spec(inject=amount is not None and extra_heads, raised=eps is not None and extra_heads)
    chunks: dict[str, list] = {k: [] for k in ("plain", "inj", "raised", "t", "ti", "pair", "delta", "cosz", "csi")}
    for i in range(0, len(picks), INFER_BATCH):
        j = slice(i, i + INFER_BATCH)
        batch = arms.make_eval_batch(
            prepared=prepared, picks=picks[j], spec=spec, index=index, amount=amount, eps=eps,
            hidden=None if hidden is None else hidden[j], hide_bsp=hide_bsp,
        )  # fmt: skip
        raw_w, raw_i, raw_c = forward_members(net=net, batch=batch, device=device)
        chunks["plain"].append({k: v.float().cpu().numpy() for k, v in nets.decode(raw=raw_w).items()})
        if raw_i is not None:
            chunks["inj"].append({k: v.float().cpu().numpy() for k, v in nets.decode(raw=raw_i).items()})
        if raw_c is not None:
            chunks["raised"].append({k: v.float().cpu().numpy() for k, v in nets.decode(raw=raw_c).items()})
        chunks["t"].append(batch.t)
        if batch.ti is not None:
            chunks["ti"].append(batch.ti)
        chunks["pair"].append(batch.pair)
        chunks["delta"].append(batch.delta)
        chunks["cosz"].append(batch.x[:, arms.ROW["cosz"], arms.MID : arms.MID + 48])
        chunks["csi"].append(batch.x[:, arms.ROW["csi"], arms.MID : arms.MID + 48])

    def join(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray] | None:
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None

    empty = np.zeros((0, 8, 48), dtype=np.float32)
    return Predictions(
        plain=join(chunks["plain"]) or {}, inj=join(chunks["inj"]), raised=join(chunks["raised"]),
        t=np.concatenate(chunks["t"]) if chunks["t"] else empty,
        ti=np.concatenate(chunks["ti"]) if chunks["ti"] else None,
        pair=np.concatenate(chunks["pair"]) if chunks["pair"] else np.zeros(0),
        delta=np.concatenate(chunks["delta"]) if chunks["delta"] else np.zeros((0, 48)),
        cosz=np.concatenate(chunks["cosz"]) if chunks["cosz"] else np.zeros((0, 48)),
        csi=np.concatenate(chunks["csi"]) if chunks["csi"] else np.zeros((0, 48)),
        picks=picks,
    )  # fmt: skip


def valid_flat(*, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return `values` `(P, 48)` on the valid half-hours, in pick then half-hour order (the `Rows` order)."""
    return values[targets[:, arms.T["valid"]] > 0.5]


def validation_scores(
    *, net: torch.nn.Module, spec: core.ArmSpec732, val_plain: Sequence[arms.Prepared],
    val_transfer: Sequence[arms.Prepared], device: str,
) -> dict[str, float]:  # fmt: skip
    """Return the validation Brier score of the sign, and (with the extra heads) the solar error and step Brier score."""
    pred = predict(net=net, prepared=val_plain, device=device)
    x = valid_flat(values=pred.t[:, arms.T["x"]], targets=pred.t)
    p = valid_flat(values=pred.plain["p"], targets=pred.t)
    out = {"brier": float(np.mean((p - (x < 0.0)) ** 2))}
    if spec.heads:
        up = valid_flat(values=pred.cosz, targets=pred.t) > core.SUN_UP_COSZ
        err = np.abs(valid_flat(values=pred.plain["pv"], targets=pred.t) - valid_flat(values=pred.t[:, arms.T["solar"]], targets=pred.t))
        out["solar_mae"] = float(err[up].mean()) if up.any() else float("nan")
        tp = predict(net=net, prepared=val_transfer, device=device)
        ev = valid_flat(values=tp.t[:, arms.T["event"]], targets=tp.t)
        sp = valid_flat(values=tp.plain["step_p"], targets=tp.t)
        out["step_brier"] = float(np.mean((sp - ev) ** 2))
    return out


VAL_TRIPLES: Final = 512
"""The number of validation triples whose loss terms are recorded after each epoch."""
VAL_BATCH_SEED: Final = 7320


@torch.no_grad()
def validation_terms(
    *, net: torch.nn.Module, spec: core.ArmSpec732, prepared: Sequence[arms.Prepared], index: ov.PairIndex | None,
    device: str, n: int = VAL_TRIPLES,
) -> dict[str, float]:  # fmt: skip
    """Return the mean unweighted loss terms, and the weighted total, on one fixed set of validation triples.

    The triples come from a generator seeded with `VAL_BATCH_SEED`, so every epoch reads the same
    windows, masks, and pairs and the curve shows the weights changing, not the draw.
    """
    net.eval()
    rng = np.random.default_rng(VAL_BATCH_SEED)
    picks = arms.sample_picks(prepared=prepared, n=n, rng=rng)
    sums: dict[str, float] = {}
    count = 0
    for i in range(0, len(picks), core.BATCH_SIZE):
        batch = arms.make_batch(prepared=prepared, picks=picks[i : i + core.BATCH_SIZE], spec=spec, rng=rng, index=index if spec.inject else None)
        loss, values = batch_loss(net=net, spec=spec, batch=batch, device=device)
        for k, v in {**values, "total": loss}.items():
            sums[k] = sums.get(k, 0.0) + float(v) * len(batch.x)
        count += len(batch.x)
    return {k: v / max(count, 1) for k, v in sums.items()}


def save_state(*, state: dict, path: Path) -> None:
    """Save a state dictionary atomically."""
    train729.save_state(state=state, path=path)


def fit(
    *, spec: core.ArmSpec732, seed: int, train: Sequence[arms.Prepared], index: ov.PairIndex,
    val_plain: Sequence[arms.Prepared], val_transfer: Sequence[arms.Prepared], device: str,
    val_index: ov.PairIndex | None = None, max_epochs: int = core.MAX_EPOCHS, out_path: Path | None = None,
    log: Callable[[str], None] = lambda _: None,
) -> tuple[torch.nn.Module, FitRecord]:  # fmt: skip
    """Fit one network of experiment 732.

    Args:
        spec: The arm.
        seed: The seed (sets the initial weights and the draws).
        train: The prepared segments of the seed's training bank, with this experiment's transfer overlay.
        index: The training bank's pair index.
        val_plain: The prepared plain validation bank (the early stop reads its sign Brier score).
        val_transfer: The prepared validation transfer overlay.
        device: "cuda" or "cpu".
        val_index: The validation bank's pair index (for the injected terms of the validation loss); None gives the plain terms only.
        max_epochs: The cap on epochs (300; the timing gate passes 5).
        out_path: Where to save the best weights, or None.
        log: Receives one progress line per epoch (never a month or a date).

    Returns:
        The network with its best weights, and the record.
    """
    seed_all = train730.fit_seed(arm=spec.name, arch=spec.arch, seed=seed)
    torch.manual_seed(seed_all)
    rng = np.random.default_rng(seed_all)
    net = nets.build_net(spec=spec, c_in=arms.N_CHANNELS).to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=LR_PATIENCE)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, best_epoch, best_state = float("inf"), 0, copy.deepcopy(net.state_dict())
    history: list[float] = []
    solar_history: list[float] = []
    step_history: list[float] = []
    n_windows = n_unpaired = 0
    train_terms: dict[str, list[float]] = {}
    val_terms: dict[str, list[float]] = {}
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        net.train()
        picks = arms.sample_picks(prepared=train, n=core.EPOCH_WINDOWS, rng=rng)
        epoch_sums: dict[str, float] = {}
        epoch_count = 0
        for i in range(0, len(picks), core.BATCH_SIZE):
            batch = arms.make_batch(
                prepared=train, picks=picks[i : i + core.BATCH_SIZE], spec=spec, rng=rng,
                index=index if spec.inject else None,
            )  # fmt: skip
            n_windows += len(batch.x)
            n_unpaired += int((batch.pair == 0).sum()) if spec.inject else 0
            loss, values = batch_loss(net=net, spec=spec, batch=batch, device=device)
            for k, v in {**values, "total": loss}.items():
                epoch_sums[k] = epoch_sums.get(k, 0.0) + float(v.detach()) * len(batch.x)
            epoch_count += len(batch.x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
            opt.step()
        scores = validation_scores(net=net, spec=spec, val_plain=val_plain, val_transfer=val_transfer, device=device)
        history.append(scores["brier"])
        for k, v in epoch_sums.items():
            train_terms.setdefault(k, []).append(v / max(epoch_count, 1))
        for k, v in validation_terms(net=net, spec=spec, prepared=val_transfer, index=val_index, device=device).items():
            val_terms.setdefault(k, []).append(v)
        solar_history.append(scores.get("solar_mae", float("nan")))
        step_history.append(scores.get("step_brier", float("nan")))
        sched.step(scores["brier"])
        if scores["brier"] < best:
            best, best_epoch, best_state = scores["brier"], epoch, copy.deepcopy(net.state_dict())
        log(f"{spec.name} seed {seed} epoch {epoch}: validation Brier {scores['brier']:.5f}")
        if epoch - best_epoch >= STOP_PATIENCE:
            break
    net.load_state_dict(best_state)
    if out_path is not None:
        save_state(state=best_state, path=out_path)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    return net, FitRecord(
        arm=spec.name, arch=spec.arch, seed=seed, epochs=len(history), best_epoch=best_epoch, best_val_brier=best,
        n_weights=nets.count_weights(net=net), seconds=seconds, seconds_per_epoch=seconds / max(len(history), 1),
        peak_memory_gb=peak, history=history, solar_history=solar_history, step_history=step_history,
        unpaired_share=n_unpaired / max(n_windows, 1), train_terms=train_terms, val_terms=val_terms,
    )  # fmt: skip


def peak_rss_gb() -> float:
    """Return this process's peak resident memory in gigabytes (Linux reports kilobytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def time_pair_building(
    *, spec: core.ArmSpec732, train: Sequence[arms.Prepared], index: ov.PairIndex, n_windows: int = 512, seed: int = 0
) -> float:
    """Return the seconds an epoch of 8,000 windows would take to build on one CPU thread (batch building alone)."""
    rng = np.random.default_rng(seed)
    picks = arms.sample_picks(prepared=train, n=n_windows, rng=rng)
    started = time.perf_counter()
    for i in range(0, len(picks), core.BATCH_SIZE):
        arms.make_batch(prepared=train, picks=picks[i : i + core.BATCH_SIZE], spec=spec, rng=rng, index=index if spec.inject else None)
    return (time.perf_counter() - started) * core.EPOCH_WINDOWS / n_windows


def measure_seconds_per_epoch(
    *, spec: core.ArmSpec732, train: Sequence[arms.Prepared], index: ov.PairIndex,
    val_plain: Sequence[arms.Prepared], val_transfer: Sequence[arms.Prepared], device: str,
    epochs: int = core.TIMING_EPOCHS,
) -> float:  # fmt: skip
    """Return the mean seconds an epoch of the arm takes, with the overlay and pair building inside the time."""
    _, record = fit(spec=spec, seed=0, train=train, index=index, val_plain=val_plain, val_transfer=val_transfer, device=device, max_epochs=epochs)
    return record.seconds_per_epoch

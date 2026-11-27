# ruff: noqa: E501
"""Experiment 732: the seven loss terms of a training triple and their weights.

This module needs torch. The loss of one triple (the plain window `w`, its injected member `w+`, and
its raised-sky member `wc`) is the weighted sum of: the sign cross-entropy on `w` only; the solar
Huber loss on `w` and on `w+`; the step cross-entropy on `w` and on `w+`; the step-size Huber loss on
the true edge half-hours of `w`; the injection term, the Huber loss of `(PV(w+) - PV(w)) - delta`;
the night term, the mean of `PV(w)` and `PV(w+)` on night half-hours; and the monotone term, the mean
of `max(0, PV(w) - PV(wc))` on the raisable half-hours. A Huber loss has a scale of 1 point. The
terms of the members `w+` and `wc` apply only where the window has a pair (`pair` is 1).
"""

from collections.abc import Mapping

import e00732_paths  # noqa: F401

# isort: split
import e00729_nets as nets729
import e00732_arms as arms
import e00732_core as core
import e00732_nets as nets
import torch
import torch.nn.functional as F  # noqa: N812

K = arms.K
T = arms.T


def masked_mean(*, per: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Return the mean of `per` over the half-hours where `keep` is 1 (0 when none is kept)."""
    return (per * keep).sum() / keep.sum().clamp_min(1.0)


def huber(*, error: torch.Tensor) -> torch.Tensor:
    """Return the Huber loss with a scale of `HUBER_SCALE` points, per half-hour."""
    return F.huber_loss(error, torch.zeros_like(error), reduction="none", delta=core.HUBER_SCALE)


def pooled_mean(*, parts: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Return the mean over the kept half-hours of several members pooled together."""
    total = sum((per * keep).sum() for per, keep in parts)
    count = sum(keep.sum() for _, keep in parts)
    return total / torch.as_tensor(count).clamp_min(1.0)  # ty: ignore[invalid-argument-type]


def terms(
    *, raw_w: torch.Tensor, raw_i: torch.Tensor | None, raw_c: torch.Tensor | None,
    t: torch.Tensor, ti: torch.Tensor | None, keeps: torch.Tensor, delta: torch.Tensor, pair: torch.Tensor,
) -> dict[str, torch.Tensor]:  # fmt: skip
    """Return the unweighted value of each loss term that the outputs allow.

    Args:
        raw_w: The raw outputs of the plain windows on the scored day, `(B, outputs, 48)`.
        raw_i: The raw outputs of the injected windows, or None.
        raw_c: The raw outputs of the raised-sky windows, or None.
        t: The targets of the plain windows `(B, 8, 48)`.
        ti: The targets of the injected windows, or None.
        keeps: Where each term applies `(B, 7, 48)`.
        delta: The scored extra injection `(B, 48)`.
        pair: 1 where the window has a pair `(B,)`.

    Returns:
        The terms by name (`sign` always; the others only for an arm with the extra heads).
    """
    out = {
        "sign": masked_mean(
            per=nets729.sign_bce(raw=raw_w[:, :1], x=t[:, T["x"]]), keep=keeps[:, K["sign"]]
        )
    }
    if raw_w.shape[1] == 1:
        return out
    pv_w = nets.solar_of(raw=raw_w[:, 1])
    event = t[:, T["event"]]
    pair_col = pair[:, None]
    solar = [(huber(error=pv_w - t[:, T["solar"]]), keeps[:, K["solar"]])]
    step = [
        (
            F.binary_cross_entropy_with_logits(raw_w[:, 2], event, reduction="none"),
            keeps[:, K["step"]],
        )
    ]
    pv_i = None
    if raw_i is not None and ti is not None:
        pv_i = nets.solar_of(raw=raw_i[:, 1])
        solar.append((huber(error=pv_i - ti[:, T["solar"]]), keeps[:, K["solar"]] * pair_col))
        step.append(
            (
                F.binary_cross_entropy_with_logits(raw_i[:, 2], event, reduction="none"),
                keeps[:, K["step"]] * pair_col,
            )
        )
    out["solar"] = pooled_mean(parts=solar)
    out["step"] = pooled_mean(parts=step)
    size_pred = core.SIZE_SCALE * raw_w[:, 3]
    out["size"] = masked_mean(
        per=huber(error=size_pred - t[:, T["size"]]), keep=keeps[:, K["size"]]
    )
    night = [(pv_w, keeps[:, K["night"]])]
    if pv_i is not None:
        out["injection"] = masked_mean(
            per=huber(error=(pv_i - pv_w) - delta), keep=keeps[:, K["injection"]] * pair_col
        )
        night.append((pv_i, keeps[:, K["night"]] * pair_col))
    out["night"] = pooled_mean(parts=night)
    if raw_c is not None:
        pv_c = nets.solar_of(raw=raw_c[:, 1])
        out["monotone"] = masked_mean(per=torch.relu(pv_w - pv_c), keep=keeps[:, K["raisable"]])
    return out


def total(*, values: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> torch.Tensor:
    """Return the weighted sum of the terms whose weight is above zero."""
    parts = [weights[name] * v for name, v in values.items() if weights[name] > 0.0]
    if not parts:
        msg = "no loss term has a weight above zero"
        raise ValueError(msg)
    return sum(parts[1:], parts[0])

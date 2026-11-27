# ruff: noqa: E501
"""Experiment 733: the history rows' untangled channels, assembled per arm.

`ChannelTensors` holds, for every Primary and every grid index of the fold, the three untangled
channels and their presence flag. Only D8's row is ever nonzero: the other 15 Primaries get zeros and
a flag that is off. `gather` cuts the channels of the 14 history rows of a batch out of the tensors,
overlays the re-run outputs of days D - 2 and D - 1, and returns the channels in the units of the
forecaster's input (P(X<0), the solar output over D8's capacity and the fold's target standard
deviation clipped to 0 to 10, and the step probability), each times the flag.

There is no function here that returns an untangled channel for a target row, so no target row can
read one. The masks of experiment 716's token path are applied through its own `apply_ok_mask`.
"""

from dataclasses import dataclass
from typing import Final

import e00733_channels as channels
import e00733_core as core
import e00733_paths as paths  # noqa: F401
import e00733_rerun as rerun
import numpy as np
import torch
from e00733_leakage import CorrectnessFailureError

# isort: split
import e00716_batch as batch716

VALUE_MAX: Final = core.SOLAR_CLIP_MAX
STEPS: Final = core.STEPS_PER_DAY
DROPOUT_KEY: Final = 733


class InputsError(CorrectnessFailureError):
    """An input check failed (a correctness failure, verdict row 1)."""


@dataclass
class ChannelTensors:
    """The channels of one arm and fold, on the device.

    Attributes:
        values: `(P, G, 3)` float32: P(X<0), the solar output in points, and the step probability.
        ok: `(P, G)` bool: whether the half-hour holds an untangled value.
        d8_index: The Primary index of D8.
        target_std: D8's target standard deviation in the fold (the divisor of the solar channel).
        rerun_values: `(R, 2, 48, 3)` re-run outputs of UTC days D - 2 and D - 1 per origin, or None.
        rerun_ok: `(R, 2, 48)` bool, or None.
        rerun_first_day: The grid day number of origin row 0 of the re-run arrays.
    """

    values: torch.Tensor
    ok: torch.Tensor
    d8_index: int
    target_std: float
    rerun_values: torch.Tensor | None = None
    rerun_ok: torch.Tensor | None = None
    rerun_first_day: int = 0


@dataclass
class History:
    """The channels of a batch's history rows.

    Attributes:
        channels: `(B, 14, 48, 3)` float32: `[u, v, w]`, each times the flag (zero where the flag is off).
        ok: `(B, 14, 48)` bool, the presence flag.
    """

    channels: torch.Tensor
    ok: torch.Tensor


def build_tensors(
    *,
    dense_values: np.ndarray,
    dense_ok: np.ndarray,
    n_primaries: int,
    d8_index: int,
    target_std: float,
    rerun_outputs: rerun.RerunOutputs | None = None,
    channels_kept: tuple[str, ...] | None = None,
    device: str = "cpu",
) -> ChannelTensors:  # fmt: skip
    """Return the channel tensors of one arm: D8's dense values at row `d8_index`, zeros and a flag off elsewhere.

    Args:
        dense_values: `(G, 3)` D8's values on the grid (the stored table, or the placebo's).
        dense_ok: `(G,)` bool.
        n_primaries: The number of Primaries (16).
        d8_index: The Primary index of D8.
        target_std: D8's target standard deviation in the fold.
        rerun_outputs: The re-run of days D - 2 and D - 1, or None (the placebo and F0r read none).
        channels_kept: The channels the arm reads (None keeps every one that is present); the others
            are zero, in the stored table and in the re-run alike.
        device: The device.

    Returns:
        The tensors.
    """
    if not 0 <= d8_index < n_primaries:
        msg = "D8's Primary index is outside the Primaries"
        raise InputsError(msg)
    names = core.CHANNEL_NAMES if channels_kept is None else channels_kept
    values, ok = channels.select_channels(values=dense_values, ok=dense_ok, names=names)
    full = np.zeros((n_primaries, *values.shape), dtype=np.float32)
    full_ok = np.zeros((n_primaries, len(ok)), dtype=bool)
    full[d8_index], full_ok[d8_index] = values, ok
    out = ChannelTensors(
        values=torch.from_numpy(full).to(device), ok=torch.from_numpy(full_ok).to(device),
        d8_index=d8_index, target_std=float(target_std),
    )  # fmt: skip
    if rerun_outputs is not None:
        n_rows = rerun_outputs.values.shape[0]
        kept = [
            channels.select_channels(
                values=rerun_outputs.values[:, k].reshape(-1, core.N_CHANNELS),
                ok=rerun_outputs.ok[:, k].reshape(-1), names=names,
            )
            for k in range(2)
        ]  # fmt: skip
        rv = np.stack([v.reshape(n_rows, STEPS, core.N_CHANNELS) for v, _ in kept], axis=1)
        rok = np.stack([f.reshape(n_rows, STEPS) for _, f in kept], axis=1)
        order = np.argsort(rerun_outputs.origin_days)
        days = rerun_outputs.origin_days[order]
        if len(days) and (np.diff(days) != 1).any():
            msg = "the re-run outputs do not cover consecutive origin days"
            raise InputsError(msg)
        out.rerun_values = torch.from_numpy(rv[order].astype(np.float32)).to(device)
        out.rerun_ok = torch.from_numpy(rok[order]).to(device)
        out.rerun_first_day = int(days[0]) if len(days) else 0
    return out


def absent_tensors(
    *, n_primaries: int, n_grid: int, d8_index: int, device: str = "cpu"
) -> ChannelTensors:
    """Return tensors with every channel absent: zeros and a flag that is off (arm F0r, and M*'s equivalent)."""
    return ChannelTensors(
        values=torch.zeros((n_primaries, n_grid, core.N_CHANNELS), device=device),
        ok=torch.zeros((n_primaries, n_grid), dtype=torch.bool, device=device),
        d8_index=d8_index, target_std=1.0,
    )  # fmt: skip


def to_input_units(*, raw: torch.Tensor, target_std: float) -> torch.Tensor:
    """Return `[u, v, w]` from `[P(X<0), solar points, step probability]`.

    `v` is the solar output over D8's capacity and the fold's target standard deviation, clipped to 0 to
    10, as experiment 716's `to_p` builds its estimate. A point is 1% of the capacity.
    """
    u, solar, w = raw[..., 0], raw[..., 1], raw[..., 2]
    v = torch.clamp(solar / channels.POINTS_PER_CAPACITY / target_std, 0.0, VALUE_MAX)
    return torch.stack([u, v, w], dim=-1)


def gather(
    *,
    tensors: ChannelTensors,
    primary: torch.Tensor,
    hist_idx: torch.Tensor,
    c_idx: torch.Tensor,
    power_visible: torch.Tensor,
    cams_rule: torch.Tensor | None = None,
    convention: torch.Tensor | None = None,
    power_shifted: torch.Tensor | None = None,
) -> History:  # fmt: skip
    """Return the channels of the 14 history rows of a batch.

    Args:
        tensors: The arm's tensors.
        primary: `(B,)` long.
        hist_idx: `(B, 14, 48)` long, the grid index of each history token.
        c_idx: `(B,)` long, the grid index of the 12:00 UTC contract origin.
        power_visible: `(B, 14, 48)` bool, whether the forecaster's own power token is visible. An
            untangled channel is present only where it is.
        cams_rule: `(B,)` bool, the mask of the age-2 row and part of the age-3 row (training dropout).
        convention: `(B,)` bool, the mask of ages 2 to 5.
        power_shifted: `(B, 14, 48)` bool, where stage 3's synthetic shift changed the history
            power. The channel is absent there, because the stored output belongs to the unshifted
            power.

    Returns:
        The channels and the flag.

    Raises:
        InputsError: When an origin of a D8 sample lies outside the re-run outputs.
    """
    p3 = primary[:, None, None]
    raw = tensors.values[p3, hist_idx]
    ok = tensors.ok[p3, hist_idx]
    is_d8 = primary == tensors.d8_index
    if tensors.rerun_values is not None and tensors.rerun_ok is not None:
        origin_day = c_idx // STEPS
        row = origin_day - tensors.rerun_first_day
        n_rows = tensors.rerun_values.shape[0]
        if bool((is_d8 & ((row < 0) | (row >= n_rows))).any()):
            msg = "an origin of a D8 sample lies outside the re-run outputs"
            raise InputsError(msg)
        token_day = (hist_idx - 1) // STEPS
        hh = (hist_idx - 1) % STEPS
        age_days = origin_day[:, None, None] - token_day
        use = ((age_days == 1) | (age_days == 2)) & is_d8[:, None, None]
        k = (2 - age_days).clamp(0, 1)
        r = row.clamp(0, n_rows - 1)[:, None, None].expand_as(hist_idx)
        raw = torch.where(use[..., None], tensors.rerun_values[r, k, hh], raw)
        ok = torch.where(use, tensors.rerun_ok[r, k, hh], ok)
    by_time = hist_idx <= (c_idx - core.NEWEST_VISIBLE_BEFORE_CONTRACT)[:, None, None]
    ok = ok & power_visible & by_time & is_d8[:, None, None]
    if power_shifted is not None:
        ok = ok & ~power_shifted
    ok = batch716.apply_ok_mask(ok=ok, cams_rule=cams_rule, convention=convention)
    units = to_input_units(raw=raw, target_std=tensors.target_std)
    out = torch.where(ok[..., None], units, torch.zeros_like(units))
    history = History(channels=out, ok=ok)
    check_history(history=history, primary=primary, d8_index=tensors.d8_index)
    return history


def check_history(*, history: History, primary: torch.Tensor, d8_index: int) -> None:
    """Row-by-row assertions on every batch.

    A channel is only where its flag says, finite, in range, and only at D8.

    Args:
        history: The channels of a batch's history rows.
        primary: `(B,)` the Primary of each sample.
        d8_index: The Primary index of D8.

    Raises:
        InputsError: When a masked token holds a value, a value is non-finite or out of range, or a
            Primary other than D8 holds a channel or a flag.
    """
    ch, ok = history.channels, history.ok
    if not bool(torch.isfinite(ch).all()):
        msg = "a history channel is non-finite"
        raise InputsError(msg)
    if bool((ch[~ok] != 0).any()):
        msg = "a token whose flag is off carries an untangled value"
        raise InputsError(msg)
    if bool(
        (ch[..., 0] < 0).any()
        or (ch[..., 0] > 1).any()
        or (ch[..., 2] < 0).any()
        or (ch[..., 2] > 1).any()
    ):
        msg = "P(X<0) or the step probability leaves 0 to 1"
        raise InputsError(msg)
    if bool((ch[..., 1] < 0).any() or (ch[..., 1] > VALUE_MAX + 1e-6).any()):
        msg = "the solar channel leaves 0 to 10"
        raise InputsError(msg)
    other = primary != d8_index
    if bool(ok[other].any()) or bool((ch[other] != 0).any()):
        msg = "a Primary other than D8 holds an untangled channel or a flag"
        raise InputsError(msg)


def dropout_masks(
    *, seed: int, stage: int, epoch: int, batch: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two mask draws of one training batch, each `(n,)` bool.

    The age-2 mask and the mask of ages 2 to 5.

    The draw uses its own generator keyed by (733, seed, stage, pass, batch), never the global torch
    stream, so every arm of one seed sees the same trunk dropout. A sample gets the first mask with
    probability 0.10, and another 0.10 get the second.
    """
    u = np.random.default_rng([DROPOUT_KEY, seed, stage, epoch, batch]).random(n)
    share = core.MASK_DROPOUT_SHARE
    return u < share, (u >= share) & (u < 2 * share)


def mask_all(*, tensors: ChannelTensors) -> ChannelTensors:
    """Return the tensors of reading R4: every untangled channel and flag of D8 set to zero, nothing else changed.

    The re-run arrays are zeroed too. The flags are off, so a trained network that uses the channels
    reads the absent-channel input.
    """
    return ChannelTensors(
        values=torch.zeros_like(tensors.values), ok=torch.zeros_like(tensors.ok),
        d8_index=tensors.d8_index, target_std=tensors.target_std,
        rerun_values=None if tensors.rerun_values is None else torch.zeros_like(tensors.rerun_values),
        rerun_ok=None if tensors.rerun_ok is None else torch.zeros_like(tensors.rerun_ok),
        rerun_first_day=tensors.rerun_first_day,
    )  # fmt: skip


def arm_tensors(
    *,
    arm: str,
    choice: channels.ChannelChoice,
    table: channels.ChannelTable,
    rerun_outputs: rerun.RerunOutputs | None,
    n_grid: int,
    n_primaries: int,
    d8_index: int,
    target_std: float,
    device: str = "cpu",
) -> ChannelTensors:  # fmt: skip
    """Return the tensors of a real-fold arm.

    F0r (and M*, whose stored forecasts read nothing) has every channel absent. FU reads the usable
    channels and the re-run. FU-S reads the usable channels taken 28 days earlier from the stored
    table alone. FS reads P(X<0) alone, with the re-run.

    Args:
        arm: The arm's name.
        choice: The rule's choice.
        table: The channel table.
        rerun_outputs: The re-run outputs, or None.
        n_grid: The number of grid half-hours.
        n_primaries: The number of Primaries.
        d8_index: The Primary index of D8.
        target_std: D8's target standard deviation in the fold.
        device: The device.

    Returns:
        The tensors.

    Raises:
        InputsError: When `arm` is not a real-fold arm that reads channels or is absent.
    """
    spec = core.ARMS.get(arm)
    if spec is None or spec.where != "real":
        msg = f"{arm} is not a real-fold arm"
        raise InputsError(msg)
    if not spec.channels:
        return absent_tensors(
            n_primaries=n_primaries, n_grid=n_grid, d8_index=d8_index, device=device
        )
    dense, ok = table.dense(n_grid=n_grid)
    keep = tuple(n for n in core.CHANNEL_NAMES if n in choice.usable)
    if spec.channel_subset is not None:
        keep = tuple(n for n in keep if n in spec.channel_subset)
    if spec.shifted_days:
        dense, ok = channels.placebo(values=dense, ok=ok, shift_steps=spec.shifted_days * STEPS)
        outputs = None
    else:
        outputs = rerun_outputs
    return build_tensors(
        dense_values=dense, dense_ok=ok, n_primaries=n_primaries, d8_index=d8_index, target_std=target_std,
        rerun_outputs=outputs, channels_kept=keep, device=device,
    )  # fmt: skip

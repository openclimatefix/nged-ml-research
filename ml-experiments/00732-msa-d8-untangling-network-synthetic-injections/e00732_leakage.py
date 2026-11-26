# ruff: noqa: E501
"""Experiment 732: the leakage and correctness assertions added to experiments 729 and 730's.

A failure raises `CorrectnessFailureError`, which is verdict row 1. No function prints a date, a series
identifier, or a value; the messages hold counts and rule names. The assertions of experiments 729 and
730 on the banks run unchanged through their own modules.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import e00730_leakage as leak730
import e00732_arms as arms
import e00732_core as core
import e00732_neighbours as nb
import e00732_overlays as ov
import numpy as np

CorrectnessFailureError = leak730.CorrectnessFailureError
assert_files_older = leak730.assert_files_older
assert_nothing_written_to = leak730.assert_nothing_written_to
FORBIDDEN_INPUT_WORDS = ("month", "doy", "ordinal", "year")


def _fail(*, message: str) -> None:
    raise CorrectnessFailureError(message)


def assert_inputs_not_targets(
    *, windows: np.ndarray, targets: np.ndarray, delta: np.ndarray | None = None
) -> None:
    """Check that no input channel of any window equals the solar target, a step target, delta, or the base demand."""
    for i in range(len(windows)):
        if arms.input_equals_target(
            window=windows[i], targets=targets[i], delta=None if delta is None else delta[i]
        ):
            _fail(message="an input channel equals a target row of the scored day")


def assert_no_month_or_day_of_year() -> None:
    """Check that no channel of the stack reads the month, the day of the year, or the year."""
    bad = [n for n in arms.NAMES if any(w in n for w in FORBIDDEN_INPUT_WORDS)]
    if bad:
        _fail(message=f"{len(bad)} input channels read the month or the day of the year")


def assert_seed_spaces_disjoint(*, train_seeds: Sequence[int] = tuple(range(5))) -> None:
    """Check that the training, validation, and test overlays draw from seed spaces that do not meet.

    Args:
        train_seeds: The training seeds whose overlay seeds are compared with the validation and test seeds.
    """
    if not train_seeds:
        _fail(message="no training seed was compared")
    train = [ov.overlay_seed(kind="train", seed=s) for s in train_seeds]
    validation, test = (
        ov.overlay_seed(kind="validation", seed=None),
        ov.overlay_seed(kind="test", seed=None),
    )
    if len(set(train)) != len(train):
        _fail(message="two training seeds share an overlay seed")
    if validation == test:
        _fail(message="the validation and test overlays share a seed")
    if {validation, test} & set(train):
        _fail(message="a training overlay shares a seed with the validation or test overlay")


def assert_months_disjoint(
    *, used: Mapping[str, Iterable[str]], test_months: Iterable[str]
) -> None:
    """Check that no month of the test bank is used by the training banks, the pre-training, or the validation bank.

    Args:
        used: The months each consumer uses, by consumer name (training, pre-training, validation).
        test_months: The months of the test bank.
    """
    test = set(test_months)
    if not test or not used or any(not set(months) for months in used.values()):
        _fail(message="a month list is empty, so the disjointness check read nothing")
    for name, months in used.items():
        if set(months) & test:
            _fail(message=f"{len(set(months) & test)} test months are used by {name}")


def assert_overlay_depends_only_on_its_segment(
    *, segments: Sequence[object], kind: str, seed: int | None, neighbours: Mapping[str, np.ndarray]
) -> None:
    """Check that the overlay of a segment does not change when the other segments of the bank change.

    The overlay of segment `i` is built alone and compared with the overlay built inside the whole list.

    Args:
        segments: The bank's segments.
        kind: "train", "validation", or "test".
        seed: The training seed (None for validation and test).
        neighbours: The neighbours' series on D8's base calendar.
    """
    whole = ov.overlay_segments(segments=segments, kind=kind, seed=seed, neighbours=neighbours)  # ty: ignore[invalid-argument-type]
    for i in sorted({0, len(segments) // 2, len(segments) - 1}):
        alone = ov.overlay_segments(
            segments=[segments[i]],
            kind=kind,
            seed=seed,
            first_index=i,
            neighbours=neighbours,  # ty: ignore[invalid-argument-type]
        )
        if ov.overlay_digest(overlays=alone) != ov.overlay_digest(overlays=[whole[i]]):
            _fail(message="an overlay depends on the other segments of its bank")


def assert_pair_index_from_own_bank(*, index: ov.PairIndex, segments: Sequence[object]) -> None:
    """Check that the pair index covers exactly the segments of the bank it serves (no other bank's solar output)."""
    if len(index.months) != len(segments):
        _fail(message="the pair index was built from another bank's segments")


def assert_stage_wrote_nothing_in(
    *, written: Sequence[Path], forbidden_roots: Sequence[Path]
) -> None:
    """Check that no written path lies in experiment 730's or experiment 729's data folder."""
    for root in forbidden_roots:
        assert_nothing_written_to(written=written, forbidden_root=root)


def assert_limits_before_fits(*, limits: Path, fit_records: Sequence[Path]) -> None:
    """Check that `limits.json` is older than every fit record."""
    assert_files_older(older=limits, newer=list(fit_records), what="limits before fits")


def assert_sign_ignores_the_extra_heads(*, sign_before: np.ndarray, sign_after: np.ndarray) -> None:
    """Check that the sign logits did not change when the extra heads' weights did."""
    if not np.allclose(sign_before, sign_after, atol=1e-6):
        _fail(message="the sign logits depend on the extra heads")


def assert_unit_ids_stable(*, names: Sequence[str]) -> None:
    """Check that no panel, arm, or file name holds an asterisk (a name that a shell expands)."""
    if any("*" in n for n in names):
        _fail(message="a file name holds an asterisk")


def assert_no_test_choice(*, events: Sequence[str]) -> None:
    """Check that B-T's fit and the stops of the fits came before the first test score."""
    if "test_score" not in events:
        return
    first = events.index("test_score")
    for needed in ("limits", "bt", "fits"):
        if needed not in events[:first]:
            _fail(message=f"{needed} was not fixed before the first test score")


def assert_stop_gradient_blocks_backbone(*, backbone_grad_sums: Sequence[float]) -> None:
    """Check that the extra heads' losses gave zero gradient on every backbone weight (U-SG)."""
    if any(g != 0.0 for g in backbone_grad_sums):
        _fail(message="a loss of an extra head reached the backbone of the stop-gradient arm")
    if not len(backbone_grad_sums):
        _fail(message="the stop-gradient check read no backbone weight")


def assert_arm_weights(*, arm: str) -> None:
    """Check that an arm's loss weights match the plan's table (the seven terms, one row per arm)."""
    spec = core.ARMS[arm]
    if set(spec.weights) - set(core.LOSS_TERMS):
        _fail(message=f"arm {arm} weights an unknown term")


NEIGHBOUR_ROWS: Final = tuple(arms.NB_ROW[n] for n in arms.NEIGHBOUR_NAMES)
"""The 12 neighbour channels (value, change, sign-aligned change, and mask of each of D3, D5, and D14)."""
INJECTION_MAY_CHANGE: Final = (
    arms.ROW["r"],
    arms.ROW["d_bsp"],
    *(arms.NB_ROW[f"{n}_aligned"] for n in nb.NEIGHBOURS),
)
"""The channels an injected member may differ from its plain window in: the reading, the BSP step, and the three sign-aligned changes."""


def assert_tensor_checks(
    *, batch: arms.Batch, prepared: Sequence[arms.Prepared], spec: core.ArmSpec732
) -> dict[str, ov.Share]:
    """Run `arms.tensor_checks` on a built batch and fail on any check below its bar (an empty counter fails).

    Args:
        batch: A batch built by `make_batch` or `make_eval_batch`.
        prepared: The prepared segments the batch was built from.
        spec: The arm the batch was built for.

    Returns:
        The share of cases that pass each check.
    """
    checks = arms.tensor_checks(batch=batch, prepared=prepared, spec=spec)
    problems = arms.tensor_problems(checks=checks)
    if problems:
        _fail(message=f"{len(problems)} tensor checks failed: {', '.join(problems)}")
    return checks


def _members(*, batch: arms.Batch) -> list[np.ndarray]:
    return [m for m in (batch.x, batch.xi, batch.xc) if m is not None]


def assert_hidden_neighbours_zero(*, batch: arms.Batch) -> None:
    """Check that wherever a window is hidden, all 12 neighbour channels of every member are zero there.

    A hidden span sets D8's reading and mask to zero, and the neighbours go with them, so a neighbour
    value from a hidden span never reaches the input. The check fails when the batch hides nothing.
    """
    hidden = np.asarray(batch.hidden) > 0.5
    if not hidden.any():
        _fail(message="the batch hides no half-hour, so the hidden-span check read nothing")
    for member in _members(batch=batch):
        nb_channels = member[:, list(NEIGHBOUR_ROWS), :]
        if (nb_channels != 0.0)[np.broadcast_to(hidden[:, None, :], nb_channels.shape)].any():
            _fail(message="a neighbour channel is not zero in a hidden span")


def assert_neighbours_unsigned_and_masks_binary(*, batch: arms.Batch) -> None:
    """Check that every neighbour value channel is non-negative (the fold) and every neighbour mask is 0 or 1."""
    for member in _members(batch=batch):
        for n in nb.NEIGHBOURS:
            value, mask = member[:, arms.NB_ROW[f"{n}_value"]], member[:, arms.NB_ROW[f"{n}_mask"]]
            if (value < 0.0).any():
                _fail(message="a neighbour value channel carries a sign")
            if not np.isin(mask, (0.0, 1.0)).all():
                _fail(message="a neighbour mask is not 0 or 1")


def assert_injection_changes_only_allowed_channels(*, batch: arms.Batch) -> None:
    """Check that an injected member differs from its plain window in the reading, the BSP step, and the sign-aligned changes only.

    The plain window and the injected window share every other channel: the clear-sky index, the
    time and calendar channels, the weather, the masks, and the neighbours' values and changes. The
    sign-aligned change of a neighbour is rebuilt from the injected reading, so it may differ. The
    check fails when the batch holds no injected window or when the injection changed nothing.
    """
    if batch.xi is None:
        _fail(message="the batch has no injected member")
    paired = np.asarray(batch.pair) > 0.5
    if not paired.any():
        _fail(message="the batch holds no paired window, so the injection check read nothing")
    other = [c for c in range(batch.x.shape[1]) if c not in INJECTION_MAY_CHANGE]
    plain, injected = batch.x[paired], batch.xi[paired]
    if (plain[:, other] != injected[:, other]).any():
        _fail(
            message="an injected member differs from its plain window in a channel that may not change"
        )
    if not (plain[:, [arms.ROW["r"]]] != injected[:, [arms.ROW["r"]]]).any():
        _fail(message="the injection left the reading unchanged")


def assert_neighbours_move_only_by_mirror(*, overlays: Sequence[ov.TransferOverlay]) -> None:
    """Check that a neighbour's series differs from its real series only inside a transfer whose drawn partner it is.

    The engine, the solar output, and delta sit below D8 alone, so none of them reaches a neighbour.
    The check fails when no overlay holds a mirrored transfer, because it would then read nothing.
    """
    mirrored = 0
    for o in overlays:
        for name in nb.NEIGHBOURS:
            real = np.asarray(o.neighbours_real[name]).ravel()
            now = np.asarray(o.neighbours[name]).ravel()
            moved = ~np.isclose(real, now, equal_nan=True)
            allowed = np.zeros(moved.shape, dtype=bool)
            for t in o.transfers:
                if t.partner == name:
                    allowed[t.start_half_hour : t.end_half_hour + 1] = True
                    mirrored += 1
            if (moved & ~allowed).any():
                _fail(
                    message=f"a neighbour series changes outside a transfer that names it ({name})"
                )
    if mirrored == 0:
        _fail(message="no overlay holds a mirrored transfer, so the mirror check read nothing")

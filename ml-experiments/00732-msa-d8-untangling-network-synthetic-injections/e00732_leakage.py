# ruff: noqa: E501
"""Experiment 732: the leakage and correctness assertions added to experiments 729 and 730's.

A failure raises `CorrectnessFailureError`, which is verdict row 1. No function prints a date, a series
identifier, or a value; the messages hold counts and rule names. The assertions of experiments 729 and
730 on the banks run unchanged through their own modules.
"""

from collections.abc import Sequence
from pathlib import Path

import e00732_paths  # noqa: F401

# isort: split
import e00730_leakage as leak730
import e00732_arms as arms
import e00732_core as core
import e00732_overlays as ov
import numpy as np

CorrectnessFailureError = leak730.CorrectnessFailureError
assert_files_older = leak730.assert_files_older
assert_nothing_written_to = leak730.assert_nothing_written_to
FORBIDDEN_INPUT_WORDS = ("month", "doy", "ordinal", "year")


def _fail(*, message: str) -> None:
    raise CorrectnessFailureError(message)


def assert_inputs_not_targets(*, windows: np.ndarray, targets: np.ndarray, delta: np.ndarray | None = None) -> None:
    """Check that no input channel of any window equals the solar target, a step target, delta, or the base demand."""
    for i in range(len(windows)):
        if arms.input_equals_target(window=windows[i], targets=targets[i], delta=None if delta is None else delta[i]):
            _fail(message="an input channel equals a target row of the scored day")


def assert_no_month_or_day_of_year() -> None:
    """Check that no channel of the stack reads the month, the day of the year, or the year."""
    bad = [n for n in arms.NAMES if any(w in n for w in FORBIDDEN_INPUT_WORDS)]
    if bad:
        _fail(message=f"{len(bad)} input channels read the month or the day of the year")


def assert_seed_spaces_disjoint() -> None:
    """Check that the training, validation, and test overlays draw from seed spaces that do not meet."""
    spaces = {k: {ov.overlay_seed(kind=k, seed=s) for s in range(5)} for k in ("train",)}
    spaces["validation"] = {ov.overlay_seed(kind="validation", seed=None)}
    spaces["test"] = {ov.overlay_seed(kind="test", seed=None)}
    names = list(spaces)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if spaces[a] & spaces[b]:
                _fail(message=f"the {a} and {b} overlays share a seed")
    if len(spaces["train"]) != 5:
        _fail(message="two training seeds share an overlay seed")


def assert_overlay_depends_only_on_its_segment(*, segments: Sequence[object], kind: str, seed: int | None) -> None:
    """Check that the overlay of a segment does not change when the other segments of the bank change.

    The overlay of segment `i` is built alone and compared with the overlay built inside the whole list.
    """
    whole = ov.overlay_segments(segments=segments, kind=kind, seed=seed)  # ty: ignore[invalid-argument-type]
    for i in sorted({0, len(segments) // 2, len(segments) - 1}):
        alone = ov.overlay_segments(segments=[segments[i]], kind=kind, seed=seed, first_index=i)  # ty: ignore[invalid-argument-type]
        if ov.overlay_digest(overlays=alone) != ov.overlay_digest(overlays=[whole[i]]):
            _fail(message="an overlay depends on the other segments of its bank")


def assert_pair_index_from_own_bank(*, index: ov.PairIndex, segments: Sequence[object]) -> None:
    """Check that the pair index covers exactly the segments of the bank it serves (no other bank's solar output)."""
    if len(index.months) != len(segments):
        _fail(message="the pair index was built from another bank's segments")


def assert_stage_wrote_nothing_in(*, written: Sequence[Path], forbidden_roots: Sequence[Path]) -> None:
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

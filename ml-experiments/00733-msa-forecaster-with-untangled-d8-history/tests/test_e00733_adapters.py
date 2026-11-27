"""Tests of `e00733_adapters`: what each arm reads, with the channel names that the rule leaves
usable."""

import e00733_adapters as ad
import pytest


@pytest.mark.parametrize(
    ("arm", "usable", "want"),
    [
        ("F0r", ("p_neg", "step_p"), ((), False, 0)),
        ("FU", ("p_neg", "step_p"), (("p_neg", "step_p"), True, 0)),
        ("FS", ("p_neg", "step_p"), (("p_neg",), True, 0)),
        ("FU-S", ("p_neg", "step_p"), (("p_neg", "step_p"), False, 28)),
    ],
)
def test_arm_reads(arm: str, usable: tuple[str, ...], want: tuple) -> None:
    """F0r reads nothing, FU every usable channel with the re-run, FS only P(X<0), and the placebo
    reads the shifted table without the re-run."""
    assert ad.arm_tensors_spec(arm=arm, usable=usable) == want


def test_fs_without_p_neg_usable_keeps_nothing() -> None:
    """A subset channel that the rule dropped is not read."""
    assert ad.arm_tensors_spec(arm="FS", usable=("solar",)) == ((), True, 0)


def test_check_placebo_reads_the_shifted_table_and_refuses_a_copy_of_the_present() -> None:
    """The placebo of the channel module passes the check; the unshifted table does not."""
    import e00733_channels as channels
    import e00733_core as core
    import numpy as np
    from e00733_stages import CorrectnessFailureError

    shift = core.PLACEBO_SHIFT_DAYS * core.STEPS_PER_DAY
    rng = np.random.default_rng(0)
    values = rng.random((shift + 200, 3)).astype(np.float32)
    ok = np.ones(shift + 200, dtype=bool)
    shifted, ok_shifted = channels.placebo(values=values, ok=ok, shift_steps=shift)
    assert ad.check_placebo(before=values, after=shifted, ok=ok_shifted, shift_steps=shift) == 200
    with pytest.raises(CorrectnessFailureError):
        ad.check_placebo(before=values, after=values, ok=ok_shifted, shift_steps=shift)
    with pytest.raises(CorrectnessFailureError):
        ad.check_placebo(before=values, after=shifted, ok=ok_shifted, shift_steps=shift - 1)

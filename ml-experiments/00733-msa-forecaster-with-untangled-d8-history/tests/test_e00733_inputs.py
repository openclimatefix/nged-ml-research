# ruff: noqa: E501
"""Tests of `e00733_inputs`.

The channel gather, the visibility and D8-only masks, the dropout, and the arms' tensors.
"""

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import e00733_channels as ch  # noqa: E402
import e00733_core as core  # noqa: E402
import e00733_inputs as inp  # noqa: E402
import e00733_rerun as rr  # noqa: E402

P, G, D8 = 4, 48 * 60, 2
C0 = 48 * 40 + 24


def _tensors(rng: np.random.Generator, *, rerun: bool = False) -> inp.ChannelTensors:
    dense = np.zeros((G, 3), dtype=np.float32)
    dense[:, 0] = rng.random(G)
    dense[:, 1] = rng.random(G) * 40
    dense[:, 2] = rng.random(G)
    ok = np.ones(G, dtype=bool)
    outputs = None
    if rerun:
        n = 5
        outputs = rr.RerunOutputs(
            origin_days=np.arange(38, 38 + n), values=np.full((n, 2, 48, 3), 0.5, dtype=np.float32),
            ok=np.ones((n, 2, 48), dtype=bool),
        )  # fmt: skip
    return inp.build_tensors(
        dense_values=dense,
        dense_ok=ok,
        n_primaries=P,
        d8_index=D8,
        target_std=2.0,
        rerun_outputs=outputs,
    )


def _batch(primary: list[int], c_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prim = torch.tensor(primary)
    c = torch.tensor(c_idx)
    ages = torch.arange(core.MIN_AGE, core.MAX_AGE + 1)
    hist = c[:, None, None] - 48 * ages[None, :, None] + torch.arange(48)[None, None, :]
    return prim, c, hist


def test_only_d8_holds_a_channel(rng: np.random.Generator) -> None:
    """A sample at another Primary gets no channel and no flag, and a D8 sample gets some."""
    t = _tensors(rng)
    prim, c, hist = _batch([0, 1, 3, D8], [C0] * 4)
    h = inp.gather(
        tensors=t,
        primary=prim,
        hist_idx=hist,
        c_idx=c,
        power_visible=torch.ones(hist.shape, dtype=torch.bool),
    )
    assert not h.ok[:3].any()
    assert (h.channels[:3] == 0).all()
    assert h.ok[3].any()
    assert int(h.ok[3].sum()) > 0


def test_nothing_after_c_minus_54_is_visible(rng: np.random.Generator) -> None:
    """Every present token is at or before the newest visible stamp, even if the table holds a value later."""
    t = _tensors(rng)
    prim, c, hist = _batch([D8], [C0])
    h = inp.gather(
        tensors=t,
        primary=prim,
        hist_idx=hist,
        c_idx=c,
        power_visible=torch.ones(hist.shape, dtype=torch.bool),
    )
    assert bool((hist[h.ok] <= C0 - core.NEWEST_VISIBLE_BEFORE_CONTRACT).all())
    assert bool((~h.ok & (hist > C0 - core.NEWEST_VISIBLE_BEFORE_CONTRACT)).any())


def test_a_channel_needs_the_power_token(rng: np.random.Generator) -> None:
    """A token whose own power is hidden carries no channel."""
    t = _tensors(rng)
    prim, c, hist = _batch([D8], [C0])
    vis = torch.ones(hist.shape, dtype=torch.bool)
    vis[:, 5] = False
    h = inp.gather(tensors=t, primary=prim, hist_idx=hist, c_idx=c, power_visible=vis)
    assert not h.ok[:, 5].any()
    assert h.ok[:, 6].any()


def test_input_units_clip_and_scale_the_solar_output() -> None:
    """The solar channel is points / 100 / target_std clipped to 0 to 10; the other two pass through."""
    raw = torch.tensor([[0.3, 100.0, 0.2], [0.3, 5000.0, 0.2], [0.3, -3.0, 0.2]])
    out = inp.to_input_units(raw=raw, target_std=2.0)
    assert out[0].tolist() == pytest.approx([0.3, 0.5, 0.2])
    assert out[1, 1].item() == 10.0
    assert out[2, 1].item() == 0.0


def test_rerun_overlays_ages_one_and_two_at_d8_only(rng: np.random.Generator) -> None:
    """For a D8 origin, tokens of the two UTC days before the origin read the re-run values (0.5), where visible."""
    t = _tensors(rng, rerun=True)
    prim, c, hist = _batch([D8], [48 * 40 + 24])
    h = inp.gather(
        tensors=t,
        primary=prim,
        hist_idx=hist,
        c_idx=c,
        power_visible=torch.ones(hist.shape, dtype=torch.bool),
    )
    token_day = (hist - 1) // 48
    near = ((40 - token_day) == 1) | ((40 - token_day) == 2)
    present_near = h.ok & near
    assert int(present_near.sum()) > 0
    assert torch.allclose(h.channels[present_near][:, 0], torch.tensor(0.5))
    far = h.ok & ~near
    assert int(far.sum()) > 0
    assert not torch.allclose(h.channels[far][:, 0], torch.tensor(0.5))


def test_an_origin_outside_the_rerun_raises(rng: np.random.Generator) -> None:
    """A D8 origin the re-run does not cover raises instead of reading a wrong row."""
    t = _tensors(rng, rerun=True)
    prim, c, hist = _batch([D8], [48 * 55 + 24])
    with pytest.raises(inp.InputsError):
        inp.gather(
            tensors=t,
            primary=prim,
            hist_idx=hist,
            c_idx=c,
            power_visible=torch.ones(hist.shape, dtype=torch.bool),
        )


def test_check_history_catches_each_violation() -> None:
    """The row-by-row check fails on a masked token with a value.

    A non-finite value, an out-of-range value, and a channel at another Primary.
    """
    ch_ = torch.zeros(2, 14, 48, 3)
    ok = torch.zeros(2, 14, 48, dtype=torch.bool)
    prim = torch.tensor([D8, 0])
    inp.check_history(history=inp.History(channels=ch_, ok=ok), primary=prim, d8_index=D8)
    bad = ch_.clone()
    bad[0, 0, 0, 0] = 0.4
    with pytest.raises(inp.InputsError):
        inp.check_history(history=inp.History(channels=bad, ok=ok), primary=prim, d8_index=D8)
    ok2 = ok.clone()
    ok2[0, 0, 0] = True
    for value, slot in ((float("nan"), 0), (1.5, 0), (11.0, 1)):
        b = ch_.clone()
        b[0, 0, 0, slot] = value
        with pytest.raises(inp.InputsError):
            inp.check_history(history=inp.History(channels=b, ok=ok2), primary=prim, d8_index=D8)
    ok3 = ok.clone()
    ok3[1, 0, 0] = True
    with pytest.raises(inp.InputsError):
        inp.check_history(history=inp.History(channels=ch_, ok=ok3), primary=prim, d8_index=D8)


def test_the_age_two_mask_zeroes_the_channels(rng: np.random.Generator) -> None:
    """With the age-2 mask on, the first history row carries no channel.

    The convention mask removes ages 2 to 5.
    """
    t = _tensors(rng)
    prim, c, hist = _batch([D8, D8], [C0, C0])
    vis = torch.ones(hist.shape, dtype=torch.bool)
    plain = inp.gather(tensors=t, primary=prim, hist_idx=hist, c_idx=c, power_visible=vis)
    masked = inp.gather(
        tensors=t, primary=prim, hist_idx=hist, c_idx=c, power_visible=vis,
        cams_rule=torch.tensor([True, False]), convention=torch.tensor([False, True]),
    )  # fmt: skip
    assert plain.ok[0, 0].sum() >= masked.ok[0, 0].sum()
    assert int(masked.ok[1, :4].sum()) == 0
    assert int(masked.ok[1, 4:].sum()) == int(plain.ok[1, 4:].sum())


def test_dropout_masks_are_reproducible_and_disjoint() -> None:
    """The same key gives the same draw, the two masks never overlap, and each is about 10%."""
    a = inp.dropout_masks(seed=1, stage=2, epoch=3, batch=4, n=20000)
    b = inp.dropout_masks(seed=1, stage=2, epoch=3, batch=4, n=20000)
    assert all((x == y).all() for x, y in zip(a, b, strict=True))
    assert not (a[0] & a[1]).any()
    assert a[0].mean() == pytest.approx(0.10, abs=0.01)
    assert a[1].mean() == pytest.approx(0.10, abs=0.01)
    other = inp.dropout_masks(seed=2, stage=2, epoch=3, batch=4, n=20000)
    assert not (a[0] == other[0]).all()


def test_mask_all_gives_the_absent_input(rng: np.random.Generator) -> None:
    """Reading R4 zeroes every channel and flag, so a gather yields no present token."""
    t = inp.mask_all(tensors=_tensors(rng, rerun=True))
    prim, c, hist = _batch([D8], [C0])
    h = inp.gather(
        tensors=t,
        primary=prim,
        hist_idx=hist,
        c_idx=c,
        power_visible=torch.ones(hist.shape, dtype=torch.bool),
    )
    assert not h.ok.any()
    assert (h.channels == 0).all()


def _table(rng: np.random.Generator) -> ch.ChannelTable:
    idx = np.arange(1, G)
    return ch.ChannelTable(
        grid_index=idx,
        values=np.stack(
            [rng.random(len(idx)), rng.random(len(idx)) * 30, rng.random(len(idx))], axis=1
        ).astype(np.float32),
    )


def _choice(*usable: str) -> ch.ChannelChoice:
    return ch.ChannelChoice(
        p_neg_source="U*",
        usable=usable,
        dropped=tuple(n for n in core.CHANNEL_NAMES if n not in usable),
        reasons={},
        fu_equals_fs=usable == ("p_neg",),
    )


def test_arm_tensors_per_arm(rng: np.random.Generator) -> None:
    """F0r holds nothing.

    FU holds the usable channels; FS holds P(X<0) alone; FU-S reads 28 days earlier and no re-
    run.
    """
    table, choice = _table(rng), _choice("p_neg", "solar")
    kw: dict[str, Any] = {
        "choice": choice,
        "table": table,
        "rerun_outputs": None,
        "n_grid": G,
        "n_primaries": P,
        "d8_index": D8,
        "target_std": 2.0,
    }
    f0r = inp.arm_tensors(arm="F0r", **kw)
    assert not f0r.ok.any()
    fu = inp.arm_tensors(arm="FU", **kw)
    assert fu.ok[D8].any()
    assert not fu.ok[[0, 1, 3]].any()
    assert (fu.values[D8][:, 2] == 0).all()
    assert fu.values[D8][:, 0].abs().sum() > 0
    fs = inp.arm_tensors(arm="FS", **kw)
    assert (fs.values[D8][:, 1:] == 0).all()
    fus = inp.arm_tensors(arm="FU-S", **kw)
    shift = 28 * 48
    assert torch.equal(fus.values[D8, shift:], fu.values[D8, :-shift])
    assert not fus.ok[D8, :shift].any()
    with pytest.raises(inp.InputsError):
        inp.arm_tensors(arm="YU", **kw)


def test_the_module_never_calls_numpy_before_cpu() -> None:
    """The conversion helpers used here return CPU tensors, so a later `.numpy()` needs no device move."""
    t = inp.absent_tensors(n_primaries=P, n_grid=G, d8_index=D8)
    assert t.values.device.type == "cpu"
    assert t.values.shape == (P, G, 3)

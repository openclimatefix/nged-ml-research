"""Tests of `e00733_signdata`: the join of the stored sign forecasts to the world's samples."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import e00733_scoring_synth as ss
import e00733_signdata as sd
import e00733_synth as sy
import e00733_world as world
import numpy as np
import pytest
from synth_fixture import make_segment


def _archive(*, index: sy.SampleIndex, shift: str | None = None) -> dict[str, np.ndarray]:
    n = len(index)
    got = {
        "p_neg": np.full((n, 48), 0.3, dtype=np.float32),
        "sign_truth": np.zeros((n, 48), dtype=np.float32),
        "seg": index.seg.copy(),
        "origin_day": index.origin_day.copy(),
        "lead_index": index.lead_index.copy(),
    }
    if shift is not None:
        got[shift] = got[shift].copy()
        got[shift][0] += 1
    return got


def _patch(monkeypatch: pytest.MonkeyPatch, *, archive: dict[str, np.ndarray]) -> sy.SampleIndex:
    segments = [make_segment(n_days=40, seed=0)]
    index = sy.enumerate_samples(segments=segments)
    monkeypatch.setattr(world, "read_world", lambda **_: (segments, {}))
    monkeypatch.setattr(world, "read_meta", lambda **_: {"constants": {"std": 1.0}})
    monkeypatch.setattr(ss, "seeds_held", lambda **_: (0,))
    monkeypatch.setattr(ss, "read_arm", lambda **_: archive)
    zeros = np.zeros((len(index), 48), dtype=np.float32)
    monkeypatch.setattr(
        ss, "baseline_probabilities", lambda **_: {"b_hist": zeros, "b_clim": zeros}
    )
    monkeypatch.setattr(ss, "clim_from_limits", lambda **_: None)
    monkeypatch.setattr(sd, "segment_sources", lambda **_: ["S2"])
    import e00733_adapters as adapters

    monkeypatch.setattr(adapters, "d8_index", lambda: 3)
    return index


def test_the_archive_of_an_arm_must_hold_the_samples_of_the_world_in_the_same_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same count with another segment, origin day, or lead index is refused."""
    index = sy.enumerate_samples(segments=[make_segment(n_days=40, seed=0)])
    _patch(monkeypatch, archive=_archive(index=index))
    assert sd.load_sign_data(root=tmp_path, rows="validation").index.seg.shape == index.seg.shape
    for key in ("seg", "origin_day", "lead_index"):
        _patch(monkeypatch, archive=_archive(index=index, shift=key))
        with pytest.raises(sd.SignDataError, match=key):
            sd.load_sign_data(root=tmp_path, rows="validation")


def test_a_nan_in_the_solar_output_of_the_bank_does_not_break_the_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A segment whose injected solar output holds NaN equals the bank's copy of it."""
    import e00729_banks as b729  # ty: ignore[unresolved-import]
    import e00730_banks as b730  # ty: ignore[unresolved-import]

    seg = make_segment(n_days=40, seed=0)
    seg.solar_points[5] = np.nan
    bank_seg: Any = SimpleNamespace(
        dates=np.array(["2024-01-01"], dtype="datetime64[D]"),
        cal_index=np.array([1]),
        solar=seg.solar_points.copy().reshape(-1, 48),
        source="S2",
    )
    monkeypatch.setattr(b729, "load_bank", lambda **_: SimpleNamespace(segments=[bank_seg]))
    monkeypatch.setattr(b730, "bank_path", lambda **_: tmp_path / "bank")
    monkeypatch.setattr(
        sd, "_run_days", lambda: np.array([np.datetime64("2024-01-01", "D").astype(np.int64)])
    )
    assert sd.segment_sources(root=tmp_path, kind="test", segments=[seg]) == ["S2"]

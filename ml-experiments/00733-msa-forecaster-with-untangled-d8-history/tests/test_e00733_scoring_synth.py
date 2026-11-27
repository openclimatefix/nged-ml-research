# ruff: noqa: PT018, E501
"""Tests of `e00733_scoring_synth`: seeds held, the scored lead days, the baselines, and the readings
read from stored archives on a small random world."""

import json
import sys
import types
from pathlib import Path

import e00733_baselines as baselines
import e00733_core as core
import e00733_predict as pred
import e00733_scoring_synth as ss
import e00733_store as store
import e00733_synth as sy
import e00733_world as world
import numpy as np
import pytest
from synth_fixture import make_segment

pytest.importorskip("torch")


def _archive(
    *,
    root: Path,
    arm: str,
    seed: int,
    index: sy.SampleIndex,
    segments: list,
    rng: np.random.Generator,
) -> None:
    n = len(index)
    truth = np.stack(
        [
            segments[int(index.seg[r])].x_neg[
                sy.target_slots(
                    origin_day=index.origin_day[r : r + 1], lead_index=index.lead_index[r : r + 1]
                )[0]
            ]
            for r in range(n)
        ]
    )
    arrays = {
        "p_neg": rng.random((n, 48)).astype(np.float32), "quantiles": np.sort(rng.normal(size=(n, 48, 11)), axis=-1).astype(np.float32),
        "mean": rng.normal(size=(n, 48)).astype(np.float32), "target_z": rng.normal(size=(n, 48)).astype(np.float32),
        "sign_truth": truth.astype(np.float32), "seg": index.seg, "origin_day": index.origin_day, "lead_index": index.lead_index,
    }  # fmt: skip
    pred.write_synthetic(
        arrays=arrays, path=ss.archive_path(root=root, arm=arm, seed=seed, rows="validation")
    )


@pytest.fixture
def world_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator) -> Path:
    """A root with limits.json, a two-segment world, and stored archives of Y0, YU, and YT seed 0."""
    segments = [make_segment(seed=1), make_segment(seed=2)]
    index = sy.enumerate_samples(segments=segments)
    (tmp_path / "limits.json").write_text(
        json.dumps({"b_clim": baselines.fit_clim(segments=segments).as_dict()})
    )
    monkeypatch.setattr(world, "read_world", lambda **_k: (segments, {}))
    monkeypatch.setattr(world, "read_meta", lambda **_k: {"constants": {"mean": 0.0, "std": 1.0}})
    import e00733_adapters as adapters

    monkeypatch.setattr(adapters, "d8_index", lambda: 0)
    sites = types.SimpleNamespace(
        sites=lambda: __import__("polars").DataFrame({"label": ["D8"], "capacity_mw": [10.0]})
    )
    monkeypatch.setitem(sys.modules, "e00578_common", sites)
    for arm in ss.SYNTHETIC_ARMS:
        _archive(root=tmp_path, arm=arm, seed=0, index=index, segments=segments, rng=rng)
    return tmp_path


def test_seeds_held_requires_every_seed_on_test_rows(world_root: Path) -> None:
    """Validation rows read the seeds that exist; test rows read none unless all three exist."""
    assert ss.seeds_held(root=world_root, arm="YU", rows="validation") == (0,)
    assert ss.seeds_held(root=world_root, arm="YU", rows="test") == ()
    assert ss.seeds_held(root=world_root, arm="Y0", rows="validation") == (0,)


def test_scored_mask_is_lead_days_2_to_10() -> None:
    """Lead index 1 to 9 are lead days 2 to 10."""
    mask = ss.scored_mask(lead_index=np.arange(14))
    assert mask.tolist() == [False] + [True] * 9 + [False] * 4


def test_readings_come_from_stored_archives(world_root: Path) -> None:
    """Every contrast of the synthetic world reads, the sign inputs and R11 are present, and a
    missing YU leaves them void."""
    out = ss.synthetic_readings(root=world_root, rows="validation", n_resamples=50)
    assert set(out["contrasts"]) == {"C3", "C4", "R5", "R6", "R7"}
    assert all(v is not None and np.isfinite(v.reading.point) for v in out["contrasts"].values())
    assert out["n_scored"] > 0 and out["sign_inputs"]["seeds_present"] is False
    assert set(out["r11"]["by_reading_band"]) == {"under 1 MVA", "1 to 3 MVA", "3 MVA and above"}
    assert (
        out["r11"]["import_half_hours_of_at_least_1_mva"]["harm_limit_730"] == core.HARM_LIMIT_730
    )
    assert set(out["log_loss"]) == {"C3", "C4"}
    for p in (world_root / "predictions").glob("yu_s0/validation.npz"):
        p.unlink()
    void = ss.synthetic_readings(root=world_root, rows="validation", n_resamples=50)
    assert void["sign_inputs"] is None and all(v is None for v in void["contrasts"].values())
    assert store.set_name(arm="YU", fold=None, seed=0) == "yu_s0"


def test_a_bank_of_another_size_than_the_archive_raises(
    world_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An archive that does not match the bank's samples is a scoring failure."""
    segments = [make_segment(seed=1)]
    monkeypatch.setattr(world, "read_world", lambda **_k: (segments, {}))
    with pytest.raises(ss.sc.ScoreError):
        ss.synthetic_readings(root=world_root, rows="validation", n_resamples=50)

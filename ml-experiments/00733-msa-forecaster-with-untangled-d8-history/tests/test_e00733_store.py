# ruff: noqa: PT018, E501
"""Tests of `e00733_store`: the set names, the row sets, and the key check against M*'s stored file."""

from datetime import UTC, datetime
from pathlib import Path

import e00733_baselines as baselines
import e00733_core as core
import e00733_store as store
import polars as pl
import pytest

from studies import experiment_store


def test_set_names_match_the_store_pattern_and_carry_the_suffixes() -> None:
    """Every arm, variant, and row kind gives a name that the experiment store accepts."""
    seen = set()
    for arm in core.ARMS:
        for variant in store.VARIANTS:
            for rows in ("test", "validation"):
                name = store.set_name(arm=arm, fold="2025-11", seed=0, variant=variant, rows=rows)
                assert experiment_store.SET_NAME_PATTERN.fullmatch(name), name
                seen.add(name)
    assert len(seen) == len(core.ARMS) * 3 * 2
    assert store.set_name(arm="FU-S", fold="2025-11", seed=1) == "fu-s_2025-11_s1"
    assert (
        store.set_name(arm="FU", fold="2026-05", seed=0, variant="masked", rows="validation")
        == "fu_2026-05_s0_masked_val"
    )
    assert store.set_name(arm="YU", fold=None, seed=2) == "yu_s2"
    with pytest.raises(store.PredictionError):
        store.set_name(arm="FU", fold="2025-11", seed=0, variant="other")
    with pytest.raises(store.PredictionError):
        store.set_name(arm="FU", fold="2025-11", seed=0, rows="train")


def test_job_name_equals_the_fit_module_name() -> None:
    """The torch-free job name matches `e00733_fit.job_name` (the fits' file names)."""
    pytest.importorskip("torch")
    import e00733_fit as fit

    for arm, fold in (("M*", "2025-11"), ("FU-S", "2026-01"), ("YT", None)):
        assert store.job_name(arm=arm, fold=fold, seed=1) == fit.job_name(
            arm=arm, fold=fold, seed=1
        )


def _frame(*, shift: int = 0) -> pl.DataFrame:
    t0 = datetime(2025, 11, 1, 9, tzinfo=UTC)
    return pl.DataFrame({
        "time_series_id": [1, 1, 2],
        "power_fcst_init_time": [t0, t0, t0],
        "valid_time": [datetime(2025, 11, 2, 9 + shift, tzinfo=UTC), datetime(2025, 11, 2, 10, tzinfo=UTC), datetime(2025, 11, 2, 9, tzinfo=UTC)],
    })  # fmt: skip


def test_keys_must_equal_m_star_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The frame's keys match M*'s stored file or the check raises; an empty file raises too."""
    stored = tmp_path / "m.parquet"
    _frame().write_parquet(stored)
    monkeypatch.setattr(baselines, "m_star_file", lambda **_k: stored)
    assert store.assert_keys_match_m_star(frame=_frame().reverse(), fold="2025-11", seed=0) == 3
    with pytest.raises(store.PredictionError):
        store.assert_keys_match_m_star(frame=_frame(shift=1), fold="2025-11", seed=0)
    _frame().head(0).write_parquet(stored)
    with pytest.raises(store.PredictionError):
        store.assert_keys_match_m_star(frame=_frame().head(0), fold="2025-11", seed=0)


def test_a_set_round_trips_through_the_experiment_store(tmp_path: Path) -> None:
    """A set written through `write_predictions` is read by `read_set`, and a wrong row set raises."""
    root = tmp_path / "ml-experiments" / "00733-test-folder"
    (root / "predictions").mkdir(parents=True)
    frame = _frame().with_columns(valid_time=pl.col("valid_time").dt.replace_time_zone("UTC"))
    name = store.set_name(arm="FU", fold="2025-11", seed=0, rows="validation")
    experiment_store.write_predictions(
        experiment_id=733, name=name, frame=frame, kind="quantiles",
        row_set_id=store.row_set_id(rows="validation"), data_dir=store.data_dir(root=root),
    )  # fmt: skip
    got = store.read_set(root=root, name=name, rows="validation")
    assert got.height == 3 and store.set_folder(root=root, name=name).is_dir()
    with pytest.raises(ValueError, match="row set"):
        store.read_set(root=root, name=name, rows="test")


def test_the_smoke_scratch_folder_is_named_like_the_data_folder(tmp_path: Path) -> None:
    """The store finds experiment 733 by folder name, so a smoke prediction needs that name."""
    import e00733_smokerun as sr

    root = tmp_path / "ml-experiments" / "00733-slug"
    scratch = sr.scratch_root(outer=tmp_path / "smoke", root=root)
    (scratch / "predictions").mkdir(parents=True)
    frame = _frame().with_columns(valid_time=pl.col("valid_time").dt.replace_time_zone("UTC"))
    experiment_store.write_predictions(
        experiment_id=733, name="f0r_2025-11_s0_val", frame=frame, kind="quantiles",
        row_set_id=store.row_set_id(rows="validation"), data_dir=store.data_dir(root=scratch),
    )  # fmt: skip
    assert store.read_set(root=scratch, name="f0r_2025-11_s0_val", rows="validation").height == 3

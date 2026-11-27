"""Tests of the loader paths on tiny on-disk fixtures, the figures, and the series rules."""

from datetime import UTC, datetime
from pathlib import Path

import e00730_core as core
import e00730_figures as figures
import e00730_pretrain as pretrain
import e00730_real as real
import e00730_series_figures as sf
import numpy as np
import pytest


def test_removed_mask_reads_the_grid_of_experiment_087(monkeypatch: pytest.MonkeyPatch) -> None:
    common = pytest.importorskip("e00087_common")
    monkeypatch.setattr(common, "GRID_START", datetime(2024, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(common, "GRID_END", datetime(2024, 3, 31, 23, 30, tzinfo=UTC))
    mask = pretrain.build_removed_mask(
        held_out_months=["2024-02"], record_days=np.array([], dtype="datetime64[D]")
    )
    grid = common.grid_times()
    assert mask.shape == (len(grid),)
    days = grid.dt.date().to_numpy().astype("datetime64[D]")
    assert mask[days == np.datetime64("2024-02-15")].all()
    assert mask[days == np.datetime64("2024-01-30")].all()
    assert not mask[days == np.datetime64("2024-01-20")].any()
    assert not mask[days == np.datetime64("2024-03-10")].any()


def test_encoder_table_and_standardiser_round_trip(tmp_path: Path) -> None:
    dates = np.datetime64("2024-01-01", "D") + np.arange(3)
    table = (
        np.random.default_rng(0).normal(size=(3, 48, core.ENCODER_INPUT_COLUMNS)).astype(np.float32)
    )
    path = tmp_path / "pretrained" / pretrain.D8_TABLE_FILE
    pretrain.write_table(table=table, base_dates=dates, path=path)
    back_dates, back = pretrain.read_table(path=path)
    assert np.array_equal(back_dates, dates)
    assert np.allclose(back, table)
    s = pretrain.fit_standardiser(
        x=np.random.default_rng(1).normal(size=(50, 4)), mask=np.ones(4, bool)
    )
    pretrain.write_standardiser(standardiser=s, folder=tmp_path / "pretrained")
    t = pretrain.read_standardiser(folder=tmp_path / "pretrained")
    assert np.allclose(t.mean, s.mean)
    weights = pretrain.weight_path(folder=tmp_path / "pretrained", seed=1)
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"abc")
    assert len(pretrain.weight_digest(path=weights)) == 64


def test_d8_table_builder_needs_the_experiment_545_chain() -> None:
    pytest.importorskip("torch")
    assert callable(pretrain.build_d8_encoder_table)


def test_share_by_gives_aggregates_only() -> None:
    out = real.share_by(
        values=np.array([True, False, True, True]), groups=np.array(["a", "a", "b", "b"])
    )
    assert out == {"a": 0.5, "b": 1.0}


def test_window_figure_has_no_calendar_date_and_is_svg(tmp_path: Path) -> None:
    n = 96
    x = np.linspace(-1, 1, n)
    chart = sf.window_figure(
        window={
            "truth": x,
            "reading": np.abs(x),
            "demand": np.ones(n),
            "demand_mean": np.ones(n),
            "p_S": np.full(n, 0.2),
            "p_SD*": np.full(n, 0.4),
            "hidden": x > 0.5,
        },
        title="S1: a window",
    )
    path = tmp_path / "w.svg"
    figures.save_svg(chart=chart, path=path)
    text = path.read_text()
    assert "<svg" in text
    assert "20" + "22-" not in text
    assert "svgo" in figures.optimise_command(path=path)


def test_result_figures_draw() -> None:
    assert figures.screen_chart(
        scores={"S": {"bilstm": 0.03, "unet": 0.031}, "S-A": {"bilstm": 0.04, "unet": 0.041}}
    )
    assert figures.contrast_chart(readings={"C1": (-0.01, -0.02, -0.005)}, margin=0.005)
    assert figures.causal_readout_chart(brier_by_k={0: 0.04, 6: 0.03}, two_sided=0.02, p0=0.05)
    assert figures.attention_chart(attention=np.eye(8), title="a map")
    assert figures.demand_readout_chart(errors={"K": {"K": 1.0, "W": 2.0}}, climatology=3.0)
    assert figures.reliability_panels(curves={"S": ([0.1, 0.9], [0.1, 0.8])})


def test_gallery_s1_is_aggregate_only_in_the_public_set() -> None:
    out = figures.GalleryResult()
    out.private["G3_private"] = figures.window_chart(series={"a": np.arange(96.0)}, title="x")
    assert "G3_private" not in out.public


def table(n: int = 40) -> sf.WindowTable:
    rng = np.random.default_rng(0)
    return sf.WindowTable(
        world=np.array(["none", "engine", "solar", "both"])[np.arange(n) % 4],
        solar_run=np.arange(n) % 4 >= 2,
        solar_out=np.arange(n) % 4 >= 1,
        source=np.array(["none", "S1", "S2"])[np.arange(n) % 3],
        depth=rng.uniform(0, 5, n) * (np.arange(n) % 4 > 0),
        near_zero=rng.integers(0, 8, n),
        evening=np.arange(n) % 2 == 0,
        sd_max_p=rng.uniform(size=n),
        sd_mean_p=rng.uniform(size=n),
        gain=rng.normal(size=n),
        real=np.zeros(n, dtype=bool),
    )


def test_the_rules_are_fixed_and_the_published_set_avoids_s1() -> None:
    t = table()
    a, b = sf.plan_windows(t=t), sf.plan_windows(t=t)
    assert a == b
    assert set(a) == set(sf.RULES)
    assert len(sf.RULES) >= 15
    for rule, idx in a.items():
        if idx is not None and rule not in {"S14", "S15"}:
            assert t.source[idx] != "S1"


def test_median_and_extreme_pick_the_expected_window() -> None:
    mask = np.array([True, True, True, False])
    depth = np.array([3.0, 1.0, 2.0, 0.0])
    assert sf.median_depth_index(mask=mask, depth=depth) == 2
    assert sf.extreme_index(mask=mask, key=depth, largest=True) == 0
    assert sf.extreme_index(mask=mask, key=depth, largest=False) == 1
    assert sf.extreme_index(mask=np.zeros(4, bool), key=depth, largest=True) is None

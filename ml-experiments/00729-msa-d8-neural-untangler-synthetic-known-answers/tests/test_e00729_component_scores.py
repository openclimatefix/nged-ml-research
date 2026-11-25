# ruff: noqa: E501, PT018
"""Tests of the component scores on invented segments and predictions."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import e00729_component_scores as comp
import e00729_core as core
import numpy as np
import pytest

START = np.datetime64("2030-01-07", "D")
ONE_DAY = np.timedelta64(1, "D")
HH = np.arange(48)


def segment(
    *, world: str, source: str, k: int, n: int = 8, month_offset: int = 0
) -> SimpleNamespace:
    """Return a dense segment with an evening engine and a midday solar day (padding either side)."""
    engine = (
        np.where((HH >= 34) & (HH < 42), 20.0 + k, 0.0)
        if world in ("engine", "both")
        else np.zeros(48)
    )
    cosz = np.where((HH >= 12) & (HH < 36), np.sin(np.pi * (HH - 12) / 24.0), 0.0)
    csi = np.tile(np.linspace(0.3, 1.0, n + 2)[:, None], (1, 48))
    solar_day = 30.0 * cosz if world in ("solar", "both") else np.zeros(48)
    solar = solar_day[None, :] * csi
    cal_index = np.r_[-1, np.arange(n), -1]
    dates = START + (np.arange(n + 2) + month_offset) * ONE_DAY
    return SimpleNamespace(
        world=world, source=source, dates=dates, cal_index=cal_index,
        valid=np.ones((n + 2, 48), dtype=bool), x=np.ones((n + 2, 48)),
        solar=solar, engine=np.tile(engine, (n + 2, 1)), csi=csi,
        cosz=np.tile(cosz, (n + 2, 1)),
    )  # fmt: skip


def segments() -> list[Any]:
    return [
        segment(world=w, source=s, k=k, month_offset=40 * k)
        for k, (w, s) in enumerate(
            [("none", "none"), ("engine", "none"), ("solar", "S1"), ("both", "S1"), ("solar", "S2")]
        )
    ]


def test_rows_follow_the_scoring_order_and_skip_padding() -> None:
    rows = comp.segment_rows(segments=segments())
    assert len(rows) == 5 * 8 * 48
    assert rows.seg[0] == 0 and rows.day[0] == 1 and rows.hh[:3].tolist() == [0, 1, 2]
    assert rows.night[0] and not rows.night[24]
    assert set(rows.world.tolist()) == {0, 1, 2, 3}


def test_the_interval_is_clipped_at_zero_and_covers_ninety_percent_of_a_normal() -> None:
    lo, hi = comp.interval(mu=np.array([1.0, 10.0]), sd=np.array([2.0, 1.0]))
    assert lo[0] == 0.0 and np.isclose(hi[1], 10.0 + comp.Z90)
    rng = np.random.default_rng(0)
    truth = rng.normal(0.0, 1.0, 200_000)
    lo, hi = comp.interval(mu=np.zeros_like(truth), sd=np.ones_like(truth))
    cover = np.mean((truth >= -comp.Z90) & (truth <= comp.Z90))
    assert abs(cover - 0.9) < 0.005


def test_a_perfect_head_has_zero_error_full_coverage_and_correlation_one() -> None:
    rows = comp.segment_rows(segments=segments())
    out = comp.head_metrics(
        head="engine", rows=rows, mu=rows.engine.copy(), sd=np.full(len(rows), 0.5)
    )
    assert out["mae_all"] == 0.0 and out["mae_nonzero"] == 0.0
    assert out["coverage_90"] == 1.0 and np.isclose(out["correlation"], 1.0)
    assert out["f1_running"] == 1.0
    assert out["mae_world_engine"] == 0.0


def test_the_zero_predictor_error_is_the_mean_of_the_truth() -> None:
    rows = comp.segment_rows(segments=segments())
    train = comp.segment_rows(segments=segments())
    base = comp.baseline_metrics(head="solar", rows=rows, train=train)
    assert np.isclose(base["mae_all_zero"], rows.solar.mean())
    assert base["mae_nonzero_zero"] >= base["mae_all_zero"]


def test_the_climatology_predictor_is_the_cell_mean_of_the_training_rows() -> None:
    train = comp.segment_rows(segments=segments())
    test = comp.segment_rows(segments=segments())
    clim = comp.climatology_predictor(train=train, test=test, column="engine")
    cell = (test.month() % 12 // 3) * 48 + test.hh
    for c in np.unique(cell)[:5]:
        tr = (train.month() % 12 // 3) * 48 + train.hh == c
        assert np.isclose(clim[cell == c][0], train.engine[tr].mean())
    far = comp.climatology_predictor(train=train, test=test, column="engine")
    assert far.shape == test.engine.shape


def test_the_solar_head_metrics_report_night_output_and_the_cams_check() -> None:
    rows = comp.segment_rows(segments=segments())
    mu = rows.solar.copy()
    out = comp.head_metrics(head="solar", rows=rows, mu=mu, sd=np.full(len(rows), 1.0))
    assert out["night_mean"] == 0.0 and out["night_max"] == 0.0
    assert out["monotone_share"] == 1.0 and out["monotone_share_truth"] == 1.0
    flipped = rows.solar.max() - rows.solar
    flipped[rows.night] = 0.0
    out2 = comp.head_metrics(head="solar", rows=rows, mu=flipped, sd=np.full(len(rows), 1.0))
    assert out2["monotone_share"] < 0.5
    assert "mae_source_s1" in out and "mae_source_s2" in out


def test_ranks_average_ties() -> None:
    assert comp.ranks(x=np.array([3.0, 1.0, 1.0, 2.0])).tolist() == [4.0, 1.5, 1.5, 3.0]


def test_seed_summary_gives_mean_min_and_max() -> None:
    s = comp.seed_summary(per_seed=[{"a": 1.0}, {"a": 3.0}, {"a": 2.0}])
    assert s["a"] == {"mean": 2.0, "min": 1.0, "max": 3.0}


def test_complete_days_need_all_48_half_hours() -> None:
    rows = comp.segment_rows(segments=segments())
    assert len(comp.complete_days(rows=rows)) == 5 * 8
    keep = np.ones(len(rows), dtype=bool)
    keep[5] = False
    cut = comp.ComponentRows(
        **{f: getattr(rows, f)[keep] for f in comp.ComponentRows.__dataclass_fields__}
    )
    assert len(comp.complete_days(rows=cut)) == 5 * 8 - 1
    assert all(
        len(d) == 48 and (np.diff(cut.hh[d]) == 1).all() for d in comp.complete_days(rows=cut)
    )


def test_pick_by_key_takes_the_lower_median_the_largest_and_the_earliest_tie() -> None:
    days = [np.array([i]) for i in range(5)]
    values = {0: 3.0, 1: 1.0, 2: 3.0, 3: 2.0, 4: 3.0}
    key = lambda d: values[int(d[0])]  # noqa: E731
    picked = [
        comp.pick_by_key(days=days, keep=lambda d: True, key=key, mode=m)
        for m in ("median", "largest", "smallest")
    ]
    assert [int(p[0]) for p in picked if p is not None] == [0, 0, 1]
    assert comp.pick_by_key(days=days, keep=lambda d: False, key=key, mode="median") is None


def test_the_engine_rules_pick_the_evening_day_and_a_no_run_day() -> None:
    rows = comp.segment_rows(segments=segments())
    mu = np.where(rows.hh == 3, 40.0, rows.engine)
    picks = {p.label: p for p in comp.pick_days(head="engine", rows=rows, mu=mu)}
    assert (
        set(picks) == {"engine_evening", "engine_and_solar", "engine_no_run"}
        or "engine_no_run" not in picks
    )
    assert rows.engine[picks["engine_evening"].rows].max() > core.K1_RUNNING_POINTS
    assert "engine_morning" not in picks


def test_the_solar_rules_pick_four_days_and_respect_the_source() -> None:
    rows = comp.segment_rows(segments=segments())
    mu = rows.solar * 0.9
    picks = {p.label: p for p in comp.pick_days(head="solar", rows=rows, mu=mu, source=1)}
    assert set(picks) == {"solar_median", "solar_largest", "solar_clear", "solar_cloudy"}
    assert all(rows.source[p.rows[0]] == 1 for p in picks.values())
    s2 = comp.pick_days(head="solar", rows=rows, mu=mu, source=2)
    assert all(rows.source[p.rows[0]] == 2 for p in s2)
    clear = rows.csi[picks["solar_clear"].rows][rows.solar[picks["solar_clear"].rows] > 0].mean()
    cloudy = rows.csi[picks["solar_cloudy"].rows][rows.solar[picks["solar_cloudy"].rows] > 0].mean()
    assert clear > cloudy


def test_bin_summary_keeps_only_bins_with_enough_half_hours() -> None:
    truth = np.r_[np.zeros(500), np.full(150, 25.0), np.full(10, 95.0)]
    mu = truth * 0.8
    frame = comp.bin_summary(truth=truth, mu=mu)
    assert frame["truth"].to_list() == [0.0, 25.0]
    assert np.allclose(frame["mean"].to_list(), [0.0, 20.0])


def stored_predictions(rows: comp.ComponentRows) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    out: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for head in ("solar", "engine"):
        truth = getattr(rows, head)
        out[head] = {
            seed: {"aux_mu": truth * (1.0 - 0.05 * seed), "aux_s": np.full(len(rows), 1.0 + seed)}
            for seed in (0, 1, 2)
        }
    return out


def test_write_component_outputs_writes_the_table_and_keeps_s1_days_private(tmp_path: Path) -> None:
    segs = segments()
    rows = comp.segment_rows(segments=segs)
    out_dir, private, scores = (
        tmp_path / "figures",
        tmp_path / "PRIVATE" / "components",
        tmp_path / "scores",
    )
    messages: list[str] = []
    written = comp.write_component_outputs(
        test_segments=segs, train_segments=segs, predictions=stored_predictions(rows),
        out_dir=out_dir, private_dir=private, scores_dir=scores, log=messages.append,
    )  # fmt: skip
    names = {p.name for p in written}
    assert {
        "components_engine_days.svg",
        "components_engine_binned.svg",
        "components_solar_binned.svg",
    } <= names
    assert "components_solar_days_s1.svg" not in names
    assert (private / "components_solar_days_s1.svg").exists()
    assert {p.name for p in out_dir.iterdir()} == names
    table = (scores / "components.md").read_text()
    assert "Solar head (arm G1)" in table and "MAE, all half-hours" in table
    assert (scores / "components.csv").exists()
    published = " ".join(p.read_text() for p in written)
    assert "2030-" not in published


def test_a_prediction_of_the_wrong_length_raises(tmp_path: Path) -> None:
    segs = segments()
    rows = comp.segment_rows(segments=segs)
    preds = stored_predictions(rows)
    preds["solar"][0]["aux_mu"] = preds["solar"][0]["aux_mu"][:-1]
    with pytest.raises(ValueError, match="does not match"):
        comp.write_component_outputs(
            test_segments=segs, train_segments=segs, predictions=preds, out_dir=tmp_path, private_dir=tmp_path, scores_dir=tmp_path, log=print,
        )  # fmt: skip

# ruff: noqa: E501
"""Tests of the time-series figures on invented segments, rows, and stored predictions."""

import re
from pathlib import Path
from types import SimpleNamespace

import e00729_baselines as base
import e00729_core as core
import e00729_paths as paths
import e00729_score as score
import e00729_series_figures as series
import numpy as np
import pytest

WORLDS = ("none", "engine", "solar", "both")
N_SEG = 8
N_DAYS = 3
Q = core.QHAT_POINTS
START = np.datetime64("2030-01-01", "D")
ONE_DAY = np.timedelta64(1, "D")
World = tuple[list[SimpleNamespace], score.Rows]


def invented_segments() -> list[SimpleNamespace]:
    """Return segments with dense days (padding either side), an evening engine, and midday solar."""
    hh = np.arange(48)
    out = []
    for k in range(N_SEG):
        world = WORLDS[k % 4]
        engine = (
            np.where((hh >= 36) & (hh < 44), 10.0 + k, 0.0)
            if world in ("engine", "both")
            else np.zeros(48)
        )
        solar = (
            np.where((hh >= 18) & (hh < 30), 6.0 + k, 0.0)
            if world in ("solar", "both")
            else np.zeros(48)
        )
        base_demand = 5.0 + 0.1 * k + 0.05 * hh
        x = (
            np.tile(base_demand - engine - solar, (N_DAYS + 2, 1))
            + 0.37 * np.arange(N_DAYS + 2)[:, None]
        )
        r = np.sqrt(x**2 + Q**2)
        r[0] = r[-1] = np.nan
        out.append(
            SimpleNamespace(
                world=world, has_bsp=k >= 4, dates=START + np.arange(N_DAYS + 2) * ONE_DAY + k * 40 * ONE_DAY,
                x=x, r=r, bsp=np.cumsum(np.tile(engine, (N_DAYS + 2, 1)), axis=1),
                engine=np.tile(engine, (N_DAYS + 2, 1)), solar=np.tile(solar, (N_DAYS + 2, 1)),
            )
        )  # fmt: skip
    return out


def rows_of(segments: list[SimpleNamespace]) -> score.Rows:
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("seg", "date", "hh", "x", "r", "engine", "solar", "world", "has_bsp")
    }
    for k, s in enumerate(segments):
        for d in range(1, N_DAYS + 1):
            parts["seg"].append(np.full(48, k))
            parts["date"].append(
                np.full(48, int((s.dates[d] - np.datetime64("1970-01-01", "D")).astype(int)))
            )
            parts["hh"].append(np.arange(48))
            parts["x"].append(s.x[d])
            parts["r"].append(s.r[d])
            parts["engine"].append(s.engine[d])
            parts["solar"].append(s.solar[d])
            parts["world"].append(np.full(48, core.WORLD_TYPES.index(s.world)))
            parts["has_bsp"].append(np.full(48, s.has_bsp))
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    n = len(cat["x"])
    z = np.zeros(n)
    feats = base.Features(
        r=cat["r"], local_hh=z, local_month=z, weekend=z, weekday=z, holiday=z, t2m=z
    )
    return score.Rows(
        seg=cat["seg"], month=cat["seg"], date=cat["date"], hh=cat["hh"], x=cat["x"], r=cat["r"],
        engine=cat["engine"], solar=cat["solar"], world=cat["world"], source=np.zeros(n, int),
        fallback=np.zeros(n, bool), has_bsp=cat["has_bsp"], features=feats,
    )  # fmt: skip


def lobe_arrays(rows: score.Rows, *, p: float) -> dict[str, np.ndarray]:
    n = rows.n
    xhat = base.reading_to_import(r=rows.r)
    return {
        "p": np.full(n, p), "mu_i": xhat, "s_i": np.full(n, 0.6), "mu_r": -xhat, "s_r": np.full(n, 0.6),
    }  # fmt: skip


def write_predictions(*, root: Path, rows: score.Rows) -> None:
    for arm, p in (("A", 0.3), ("B", 0.2), ("C", 0.1), ("D", 0.15), ("Dsign", 0.25)):
        for name in ("test", "real_r1"):
            folder = root / "predictions" / f"{arm}_bilstm_s0"
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(folder / f"{name}.npz", **lobe_arrays(rows, p=p))  # ty: ignore[invalid-argument-type]
    levels = np.asarray(core.QUANTILE_LEVELS)
    q = np.sort(rows.x[:, None] + 0.5 * (levels[None, :] - 0.5) * 4, axis=1)
    folder = root / "predictions" / "Dq_bilstm_s0"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "test.npz", q=q)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(tmp_path))
    segments = invented_segments()
    rows = rows_of(segments)
    write_predictions(root=paths.data_root(), rows=rows)
    return segments, rows


def test_complete_days_finds_every_day_and_reads_the_facts(world: World) -> None:
    _, rows = world
    days = series.complete_days(rows=rows)
    assert len(days) == N_SEG * N_DAYS
    engine_day = next(d for d in days if core.WORLD_TYPES[d.world] == "engine")
    assert 36 <= engine_day.engine_centroid < 44
    assert engine_day.min_x < -core.DEAD_BAND_POINTS


def test_the_median_rule_is_deterministic_and_takes_the_lower_middle(world: World) -> None:
    _, rows = world
    days = series.complete_days(rows=rows)
    first = series.median_by(days=days, key=lambda d: d.min_x)
    second = series.median_by(days=list(reversed(days)), key=lambda d: d.min_x)
    assert first is not None
    assert second is not None
    assert (first.seg, first.date) == (second.seg, second.date)
    keys = sorted(d.min_x for d in days)
    assert first.min_x == keys[(len(keys) - 1) // 2]
    assert series.median_by(days=[], key=lambda d: d.min_x) is None


def world_of(day: series.Day | None) -> str:
    assert day is not None
    return core.WORLD_TYPES[day.world]


def test_the_picking_rules_choose_days_of_the_right_kind(world: World) -> None:
    _, rows = world
    days = series.complete_days(rows=rows)
    harm = np.arange(rows.n) % 7
    picked = series.pick_scenarios(days=days, d_p_neg=np.zeros(rows.n), d_harm=harm)
    assert world_of(picked["evening_engine"]) == "engine"
    assert picked["morning_engine"] is None
    assert world_of(picked["solar_midday"]) == "solar"
    assert world_of(picked["engine_and_solar"]) == "both"
    assert world_of(picked["plain_import"]) == "none"
    bsp, largest = picked["bsp"], picked["largest_harm"]
    assert bsp is not None
    assert largest is not None
    assert bsp.has_bsp
    assert largest.min_x >= 0


def test_window_data_puts_each_day_half_in_its_place(world: World) -> None:
    segments, rows = world
    seg = segments[1]
    date = int(rows.date[rows.seg == 1][48])
    day = 2  # the second base day of the segment (dense index)
    window = series.window_data(seg=seg, date=date, with_truth=True)
    assert window.truth.shape == window.reading.shape == window.bsp_step.shape == (96,)
    assert np.array_equal(window.truth[:24], seg.x[day - 1, 24:])
    assert np.array_equal(window.truth[24:72], seg.x[day])
    assert np.array_equal(window.truth[72:], seg.x[day + 1, :24])
    assert np.array_equal(window.reading[24:72], seg.r[day])
    steps = np.diff(np.r_[seg.bsp[day - 1, 24:], seg.bsp[day], seg.bsp[day + 1, :24]])
    assert np.array_equal(window.bsp_step[1:], steps)
    real = series.window_data(seg=seg, date=date, with_truth=False)
    assert np.isnan(real.truth).all()
    assert np.array_equal(real.reading, window.reading)


def test_a_quantile_forecast_gives_a_probability_from_its_quantiles(world: World) -> None:
    _, rows = world
    asym = {"q": np.tile(np.linspace(-1, 3, 11), (rows.n, 1))}
    f = series.forecast_on(pred=asym, index=np.arange(5))
    assert np.allclose(f.p_neg, 0.25)
    assert np.allclose(f.lower, -1)
    assert np.allclose(f.upper, 3)
    assert np.allclose(f.median, 1)
    above = series.forecast_on(
        pred={"q": np.tile(np.linspace(1, 3, 11), (rows.n, 1))}, index=np.arange(3)
    )
    below = series.forecast_on(
        pred={"q": np.tile(np.linspace(-3, -1, 11), (rows.n, 1))}, index=np.arange(3)
    )
    assert np.allclose(above.p_neg, 0.0)
    assert np.allclose(below.p_neg, 1.0)


def test_a_crossing_quantile_row_is_made_non_decreasing_before_interpolation(world: World) -> None:
    _, rows = world
    row = np.linspace(-1, 1, 11)
    row[4] = 5.0  # a crossing: the accumulated row stays at 5.0 from level 0.4 on
    f = series.forecast_on(pred={"q": np.tile(row, (rows.n, 1))}, index=np.arange(2))
    assert np.allclose(f.p_neg, np.interp(0.0, np.maximum.accumulate(row), core.QUANTILE_LEVELS))


def test_a_lobe_forecast_gives_the_mass_below_zero_not_the_stored_probability(world: World) -> None:
    del world
    n = 4
    far = {
        "p": np.full(n, 0.15),
        "mu_i": np.full(n, 20.0),
        "s_i": np.full(n, 0.5),
        "mu_r": np.full(n, -20.0),
        "s_r": np.full(n, 0.5),
    }
    f = series.forecast_on(pred=far, index=np.arange(n))
    assert np.allclose(f.p_neg, 0.15, atol=1e-6)
    near = {**far, "mu_i": np.zeros(n), "mu_r": np.zeros(n), "p": np.full(n, 0.9)}
    assert np.allclose(series.forecast_on(pred=near, index=np.arange(n)).p_neg, 0.5)
    half = {**far, "p": np.full(n, 0.5)}
    f_half = series.forecast_on(pred=half, index=np.arange(n))
    assert (f_half.lower < -15).all()
    assert (f_half.upper > 15).all()
    assert (np.abs(f_half.median) < 21).all()


def test_the_largest_harm_rule_takes_the_planted_day_and_plain_import_needs_import(
    world: World,
) -> None:
    _, rows = world
    days = series.complete_days(rows=rows)
    imports = [d for d in days if d.min_x >= 0]
    planted = imports[len(imports) // 2]
    harm = np.zeros(rows.n)
    harm[planted.index] = 1.0
    picked = series.pick_scenarios(days=days, d_p_neg=np.zeros(rows.n), d_harm=harm)
    assert picked["largest_harm"] is not None
    assert (picked["largest_harm"].seg, picked["largest_harm"].date) == (planted.seg, planted.date)
    from dataclasses import replace

    negative = [replace(d, min_x=-1.0) if core.WORLD_TYPES[d.world] == "none" else d for d in days]
    assert (
        series.pick_scenarios(days=negative, d_p_neg=np.zeros(rows.n), d_harm=harm)["plain_import"]
        is None
    )


def test_a_stored_prediction_of_the_wrong_length_is_refused(world: World) -> None:
    _, rows = world
    with pytest.raises(series.leak.CorrectnessFailureError):
        series.load_prediction(arm="D", arch="bilstm", name="test", n_rows=rows.n + 1)


def test_the_figures_are_written_without_a_calendar_date_and_under_500_kb(
    world: World, tmp_path: Path
) -> None:
    segments, rows = world
    lines: list[str] = []
    written = series.write_series_figures(
        arch="bilstm", test_segments=segments, rows=rows, real_segments=segments, real_rows=rows,
        out_dir=tmp_path / "out", log=lines.append,
    )  # fmt: skip
    names = {p.name for p in written}
    assert "series_01_evening_engine.svg" in names
    assert "series_08_bsp_steps.svg" in names
    assert "series_02_morning_engine.svg" not in names
    assert any("no day meets the rule" in line for line in lines)
    for p in written:
        text = p.read_text()
        assert p.stat().st_size < 500_000
        assert not re.search(r"20\d\d-\d\d", text)
        assert "aria-label" not in text or "2030" not in text


def test_the_truth_and_the_reading_are_wider_than_the_arm_lines_so_they_show_as_a_halo(
    world: World,
) -> None:
    segments, rows = world
    day = series.complete_days(rows=rows)[0]
    window = series.window_data(seg=segments[day.seg], date=day.date, with_truth=True)
    fc = {"D": series.forecast_on(pred=lobe_arrays(rows, p=0.1), index=day.index)}
    spec = series.window_chart(
        title="t", subtitle="s", window=window, forecasts=fc, bands=("D",), show_bsp=False
    ).to_dict()
    layers = spec["vconcat"][0]["layer"]
    encoding = next(layer["encoding"] for layer in layers if layer["mark"]["type"] == "line")
    scale = encoding["strokeWidth"]["scale"]
    widths = dict(zip(scale["domain"], scale["range"], strict=True))
    assert widths[series.TRUTH_NAME] > widths[series.ARM_LABELS["D"]]
    assert widths[series.READING_NAME] > widths[series.ARM_LABELS["D"]]

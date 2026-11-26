# ruff: noqa: E501
"""Tests of `e00733_rerun`.

The visibility limit on the windows, the hiding, the ensemble units, and the orchestration.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import e00733_core as core
import e00733_rerun as rr
import numpy as np
import pytest


@pytest.mark.parametrize("origin_day", [10, 57, 400])
def test_limit_matches_the_layout_rule(origin_day: int) -> None:
    """The limit of this module is experiment 713's c - 54 rule."""
    rr.assert_limit_matches_layout(origin_day=origin_day)


def test_hidden_columns_of_the_two_windows() -> None:
    """Columns 90..95 of day D-2's window and 42..95 of day D-1's window lie after the limit."""
    d = 100
    h2 = np.flatnonzero(rr.window_hidden(origin_day=d, day=d - 2))
    h1 = np.flatnonzero(rr.window_hidden(origin_day=d, day=d - 1))
    assert h2.tolist() == list(range(90, 96))
    assert h1.tolist() == list(range(42, 96))


def test_visible_middle_counts() -> None:
    """Day D-2's middle is fully visible, and day D-1's middle shows 18 half-hours."""
    d = 100
    assert rr.visible_middle(origin_day=d, day=d - 2).all()
    assert int(rr.visible_middle(origin_day=d, day=d - 1).sum()) == 18


def test_hide_window_zeroes_inputs_and_replaces_weather() -> None:
    """Hidden columns hold zeros (reading rows) or ensemble values (weather rows); visible ones
    keep theirs."""
    names = ("r", "r_mask", "csi", "cams_mask", "d_bsp", "bsp_mask", "t2m", "wind10", "precip")
    rows = {n: i for i, n in enumerate(names)}
    window = np.ones((len(names), rr.STEPS))
    hidden = np.zeros(rr.STEPS, dtype=bool)
    hidden[50:] = True
    era5 = {k: np.full(rr.STEPS, 7.0) for k in rr.ERA5_ROWS}
    rr.hide_window(window=window, rows=rows, hidden=hidden, era5=era5)
    assert (window[rows["r"], 50:] == 0).all()
    assert (window[rows["t2m"], 50:] == 7.0).all()
    assert (window[:, :50] == 1.0).all()


def test_hide_window_with_nan_ensemble_gives_zero() -> None:
    """A missing ensemble value becomes zero, never NaN."""
    names = ("r", "r_mask", "csi", "cams_mask", "d_bsp", "bsp_mask", "t2m", "wind10", "precip")
    rows = {n: i for i, n in enumerate(names)}
    window = np.ones((len(names), rr.STEPS))
    hidden = np.ones(rr.STEPS, dtype=bool)
    rr.hide_window(
        window=window,
        rows=rows,
        hidden=hidden,
        era5={k: np.full(rr.STEPS, np.nan) for k in rr.ERA5_ROWS},
    )
    assert np.isfinite(window).all()
    assert not window.any()


def test_standardise_ens_converts_precipitation_units() -> None:
    """One kg m-2 s-1 is 3600 mm per hour, so the precipitation channel is log1p(3600) / 0.5."""
    out = rr.standardise_ens(
        temperature=np.array([10.0]), wind=np.array([4.0]), precip_kg_m2_s=np.array([1.0]),
        t_mean=5.0, t_sd=5.0, w_mean=2.0, w_sd=2.0,
    )  # fmt: skip
    assert out["t2m"][0] == pytest.approx(1.0)
    assert out["wind10"][0] == pytest.approx(1.0)
    assert out["precip"][0] == pytest.approx(np.log1p(3600.0) / 0.5)


def _run(init_day: int, valid_us: np.ndarray, value: float) -> rr.EnsRun:
    n = len(valid_us)
    return rr.EnsRun(
        init_day=init_day,
        valid_us=valid_us,
        temperature=np.full(n, value),
        wind=np.full(n, value),
        precip=np.full(n, value),
    )


def test_ens_for_window_takes_the_previous_run_up_to_the_native_origin() -> None:
    """Stamps up to 09:00 UTC on D come from the D-1 run and later stamps from the D run.

    A missing stamp is NaN.
    """
    from e00733_channels import GRID_START_EPOCH_DAY

    origin_epoch = GRID_START_EPOCH_DAY + 100
    stamps = rr.window_stamps_us(day_epoch=origin_epoch)
    prev = _run(99, stamps, 1.0)
    cur = _run(100, stamps[:-3], 2.0)
    got = rr.ens_for_window(
        previous=prev, current=cur, origin_epoch_day=origin_epoch, window_stamps_us=stamps
    )
    native = origin_epoch * rr.US_PER_DAY + 9 * rr.US_PER_HOUR
    assert (got["temperature"][stamps <= native] == 1.0).all()
    after = stamps > native
    assert (got["temperature"][after][:-3] == 2.0).all()
    assert np.isnan(got["temperature"][-3:]).all()
    with pytest.raises(rr.RerunError):
        rr.ens_for_window(
            previous=cur, current=prev, origin_epoch_day=origin_epoch, window_stamps_us=stamps
        )


def test_rerun_origins_masks_by_visibility_and_reading(rng: np.random.Generator) -> None:
    """The outputs of two origin days have the registered shapes and the flag is the visible.

    Read half-hours only.
    """
    origins = [100, 101]

    def build(origin_day: int, day: int, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reading = np.ones(core.STEPS_PER_DAY, dtype=bool)
        reading[:4] = False
        return np.zeros((3, rr.STEPS)), reading

    def predict(windows: np.ndarray) -> np.ndarray:
        return rng.random((3, len(windows), core.N_CHANNELS, core.STEPS_PER_DAY))

    out = rr.rerun_origins(
        origin_days=origins, build_window=build, predict=predict, n_seeds=3, batch=3
    )
    assert out.values.shape == (2, 2, 48, 3)
    assert out.ok.shape == (2, 2, 48)
    assert not out.ok[:, :, :4].any()
    assert out.ok[:, 0, 4:].all()
    assert int(out.ok[0, 1].sum()) == 18 - 4


def test_rerun_origins_rejects_a_wrong_shape() -> None:
    """A network output of another shape raises."""
    with pytest.raises(rr.RerunError):
        rr.rerun_origins(
            origin_days=[100], build_window=lambda *_: (np.zeros((1, rr.STEPS)), np.ones(48, dtype=bool)),
            predict=lambda w: np.zeros((1, len(w), 3, 47)), n_seeds=1,
        )  # fmt: skip


def test_rerun_roundtrip(tmp_path: Path, rng: np.random.Generator) -> None:
    """The archive survives a write and a read."""
    out = rr.RerunOutputs(
        origin_days=np.array([5, 6]),
        values=rng.random((2, 2, 48, 3)).astype(np.float32),
        ok=rng.random((2, 2, 48)) > 0.5,
    )
    rr.write_rerun(outputs=out, path=tmp_path / "r.npz")
    back = rr.read_rerun(path=tmp_path / "r.npz")
    assert np.array_equal(back.values, out.values)
    assert np.array_equal(back.ok, out.ok)


def test_poison_check_passes_when_outputs_ignore_the_poison_and_fails_otherwise() -> None:
    """The poison check passes on a run that ignores the poison.

    Fails on one that reads it, and fails with no poison value.
    """
    base = np.linspace(0.0, 1.0, 20)
    assert rr.assert_rerun_ignores_inputs_after_limit(run=lambda _v: base) == 2
    with pytest.raises(rr.RerunError):
        rr.assert_rerun_ignores_inputs_after_limit(run=lambda v: base if v is None else base + 1.0)
    with pytest.raises(rr.RerunError):
        rr.assert_rerun_ignores_inputs_after_limit(
            run=lambda v: base if v is None else np.full(20, np.nan)
        )
    with pytest.raises(rr.RerunError):
        rr.assert_rerun_ignores_inputs_after_limit(run=lambda _v: base, poison_values=())
    with pytest.raises(rr.RerunError):
        rr.assert_rerun_ignores_inputs_after_limit(run=lambda _v: np.zeros(0))


def test_read_ens_run_searches_several_folders(tmp_path: Path) -> None:
    """A run in the second folder (the 707 back-fill) is found when the first folder lacks it."""
    import polars as pl

    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    day = 18000
    frame = pl.DataFrame(
        {
            "time_series_id": [26, 26],
            "valid_time": [datetime(2019, 4, 14, 0, 30), datetime(2019, 4, 14, 1, 0)],
            rr.ENS_COLUMNS["temperature"]: [1.0, 2.0],
            rr.ENS_COLUMNS["wind"]: [3.0, 4.0],
            rr.ENS_COLUMNS["precip"]: [0.0, 0.0],
        }
    )
    name = f"init={date(1970, 1, 1) + timedelta(days=day)}.parquet"
    frame.write_parquet(second / name)
    run = rr.read_ens_run(summary_dir=[first, second], run_epoch_day=day, series_id=26)
    assert run.temperature.tolist() == [1.0, 2.0]
    with pytest.raises(rr.RerunError):
        rr.read_ens_run(summary_dir=first, run_epoch_day=day, series_id=26)

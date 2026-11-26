# ruff: noqa: E501, PT018, RUF012, PERF401
"""Tests of `e00733_real` with a fake of experiment 732's window code (no real file is read)."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import e00733_channels as ch
import e00733_core as core
import e00733_real as rl
import e00733_rerun as rr
import numpy as np
import polars as pl
import pytest

ROWS = ("r", "r_mask", "csi", "cams_mask", "d_bsp", "bsp_mask", "t2m", "wind10", "precip")
ORIGIN = 100
EPOCH0 = ch.GRID_START_EPOCH_DAY
FIRST = EPOCH0 + ORIGIN - 6
LAST = EPOCH0 + ORIGIN + 3
N_DAYS = LAST - FIRST + 3


class _FakeArms:
    """Experiment 732's window functions on a `(9, columns)` stack.

    The reading and its mask are hidden by `plain_window`, as `masks.apply_hidden` does.
    """

    ROW = {n: i for i, n in enumerate(ROWS)}

    @staticmethod
    def plain_window(
        *, p: SimpleNamespace, day: int, spec: object, hidden: np.ndarray | None
    ) -> np.ndarray:
        start = day * 48 - 24
        out = p.stack[:, start : start + 96].copy()
        if hidden is not None:
            out[0, hidden] = 0.0
            out[1, hidden] = 0.0
        return out


def _calendar(*, seed: int = 0) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    dates = (np.arange(FIRST - 1, LAST + 2)).astype("datetime64[D]")
    shape = (len(dates), 48)
    cal = SimpleNamespace(dates=dates)
    for name in ("reading", "bsp", "csi", "t2m", "wind10", "precip"):
        setattr(cal, name, rng.normal(size=shape).astype(np.float32))
    return cal


def _fake_windows(
    cal: SimpleNamespace, _neighbours: object, *, summary_dir: Path, extra_ok: object = None
) -> rl.RealWindows:
    """Build a stack whose dense day i holds the calendar day `FIRST - 1 + i` (the same axis as `span_segment`)."""
    parts = []
    for name in ("reading", "bsp", "csi", "t2m", "wind10", "precip"):
        parts.append(np.asarray(getattr(cal, name), dtype=np.float32))
    cols = {
        "r": parts[0],
        "d_bsp": parts[1],
        "csi": parts[2],
        "t2m": parts[3],
        "wind10": parts[4],
        "precip": parts[5],
    }
    stack = np.zeros((len(ROWS), cal.dates.shape[0] * 48), dtype=np.float32)
    for name, arr in cols.items():
        stack[ROWS.index(name)] = arr.reshape(-1)
    for mask in ("r_mask", "cams_mask", "bsp_mask"):
        stack[ROWS.index(mask)] = 1.0
    stats = SimpleNamespace(t_mean=10.0, t_sd=5.0, w_mean=4.0, w_sd=2.0)
    return rl.RealWindows(
        prepared=SimpleNamespace(stack=stack), spec=None, first_epoch_day=FIRST + 1 - 1, arms=_FakeArms, stats=stats,
        summary_dir=summary_dir, series_id=26, extra_ok=extra_ok,  # ty: ignore[invalid-argument-type]
    )  # fmt: skip


def _write_summaries(folder: Path, *, days: list[int], temperature: float = 12.0) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for day in days:
        start = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(days=day)
        times = [start + timedelta(minutes=30 * k) for k in range(1, 48 * 3 + 1)]
        frame = pl.DataFrame(
            {
                "time_series_id": [26] * len(times), "valid_time": times,
                "mean_temperature_2m": [temperature] * len(times), "mean_wind_speed_10m": [6.0] * len(times),
                "mean_precipitation": [0.0] * len(times),
            }
        ).with_columns(pl.col("valid_time").cast(pl.Datetime("us", "UTC")))  # fmt: skip
        frame.write_parquet(folder / f"init={date(1970, 1, 1) + timedelta(days=day)}.parquet")


def _builder(tmp_path: Path, cal: SimpleNamespace | None = None, **kw: object) -> rl.RealWindows:
    _write_summaries(tmp_path / "ens", days=[EPOCH0 + ORIGIN - 1, EPOCH0 + ORIGIN])
    return _fake_windows(cal or _calendar(), None, summary_dir=tmp_path / "ens", **kw)  # ty: ignore[invalid-argument-type]


def test_window_hides_everything_after_the_limit_and_fills_weather_from_the_ensemble(
    tmp_path: Path,
) -> None:
    """Hidden columns hold zeros in the reading rows and the ensemble mean in the weather rows.

    Visible columns keep the real values.
    """
    builder = _builder(tmp_path)
    window, reading_ok = builder.window(origin_day=ORIGIN, day=ORIGIN - 1)
    hidden = rr.window_hidden(origin_day=ORIGIN, day=ORIGIN - 1)
    assert window.shape == (9, 96) and reading_ok.shape == (48,) and reading_ok.all()
    for name in ("r", "r_mask", "csi", "cams_mask", "d_bsp", "bsp_mask"):
        assert (
            not window[ROWS.index(name), hidden].any() and window[ROWS.index(name), ~hidden].any()
        )
    assert np.allclose(window[ROWS.index("t2m"), hidden], (12.0 - 10.0) / 5.0)
    assert np.allclose(window[ROWS.index("wind10"), hidden], (6.0 - 4.0) / 2.0)
    assert builder.missing_ens_columns == 0
    visible = builder.window(origin_day=ORIGIN, day=ORIGIN - 2)[0]
    assert visible[ROWS.index("t2m")].any() and (visible[ROWS.index("r_mask")] == 1).sum() == 90


def test_extra_reading_mask_is_intersected(tmp_path: Path) -> None:
    """Experiment 713's reading mask removes half-hours from the middle's reading mask."""
    extra = np.ones(48, bool)
    extra[:5] = False
    builder = _builder(tmp_path, extra_ok=lambda _day: extra)
    assert int(builder.window(origin_day=ORIGIN, day=ORIGIN - 2)[1].sum()) == 43


def test_a_hidden_window_whose_columns_disagree_with_the_limit_is_refused(tmp_path: Path) -> None:
    """The callback checks the hidden columns it is given against the visibility limit."""
    builder = _builder(tmp_path)
    with pytest.raises(rl.RealWindowError):
        builder.build_window(ORIGIN, ORIGIN - 1, np.zeros(96, bool))


def test_rerun_origins_runs_on_real_windows_and_the_poison_run_is_unchanged(tmp_path: Path) -> None:
    """Poisoning every input after the limit with NaN and 1e6 leaves the re-run unchanged.

    And a leaking builder is caught.
    """

    def predict(w: np.ndarray) -> np.ndarray:
        mid = w[:, 0, rr.MID : rr.MID + 48] + w[:, 6, rr.MID : rr.MID + 48]
        return np.broadcast_to(np.tanh(mid)[None, :, None, :], (2, len(w), 3, 48)).copy()

    cal = _calendar()
    make = lambda c, nb: _builder(tmp_path, c)  # noqa: E731
    n = rl.poisoned_origin_run(
        make_windows=make,
        cal=cal,
        neighbours={"D3": np.zeros((len(cal.dates), 48), np.float32)},
        origin_day=ORIGIN,
        predict=predict,
        n_seeds=2,
    )
    assert n == 2

    class Leaky(rl.RealWindows):
        def window(self, *, origin_day: int, day: int) -> tuple[np.ndarray, np.ndarray]:
            window, ok = super().window(origin_day=origin_day, day=day)
            window[0, 50] = self.prepared.stack[
                0, self.dense(epoch_day=day + EPOCH0) * 48 - 24 + 50
            ]  # an input after the limit, put back
            return window, ok

    def make_leaky(c: SimpleNamespace, nb: object) -> rl.RealWindows:
        good = _builder(tmp_path, c)
        return Leaky(
            **{
                k: getattr(good, k)
                for k in (
                    "prepared",
                    "spec",
                    "first_epoch_day",
                    "arms",
                    "stats",
                    "summary_dir",
                    "series_id",
                )
            }
        )

    with pytest.raises(rr.RerunError):
        rl.poisoned_origin_run(
            make_windows=make_leaky,
            cal=cal,
            neighbours={"D3": np.zeros((len(cal.dates), 48), np.float32)},
            origin_day=ORIGIN,
            predict=predict,
            n_seeds=2,
        )


def test_reproduce_stored_compares_unhidden_windows_and_fails_on_a_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing hidden the recomputed outputs equal the stored ones.

    And a stored row that differs fails the check.
    """
    builder = _builder(tmp_path)
    days = [FIRST + 2, FIRST + 3]
    rng = np.random.default_rng(1)
    fake = rng.uniform(size=(2, 2, 3, 48)).astype(np.float32)  # (seeds, n, 3, 48)
    monkeypatch.setattr(rr, "predict_windows", lambda **_: fake)
    mean = fake.mean(axis=0)
    day = np.repeat(days, 48)
    hh = np.tile(np.arange(48), 2)
    values = np.stack([mean[:, c, :].reshape(-1) for c in range(3)], axis=1)
    stored = ch.StoredOutputs(day=day, hh=hh, values=values.astype(np.float32))
    out = rl.reproduce_stored(windows=builder, nets=[object()], stored=stored, epoch_days=days)
    assert out["rows"] == 96 and out["max_p_neg"] < 1e-6
    bad = ch.StoredOutputs(day=day, hh=hh, values=(values + 0.05).astype(np.float32))
    with pytest.raises(rl.RealWindowError):
        rl.reproduce_stored(windows=builder, nets=[object()], stored=bad, epoch_days=days)
    nothing = ch.StoredOutputs(day=day + 999, hh=hh, values=stored.values)
    with pytest.raises(rl.RealWindowError, match="nothing was reproduced"):
        rl.reproduce_stored(windows=builder, nets=[object()], stored=nothing, epoch_days=days)


def test_ensemble_coverage_counts_gaps_as_data(tmp_path: Path) -> None:
    """An origin whose run file is missing is counted as not covered, per split, and nothing raises."""
    _write_summaries(tmp_path / "ens", days=[EPOCH0 + ORIGIN - 1, EPOCH0 + ORIGIN])
    got = rl.ens_coverage_by_split(
        origin_days=np.array([ORIGIN, ORIGIN + 50, ORIGIN + 51]), splits=np.array(["test", "test", "train"]), summary_dir=tmp_path / "ens",
    )  # fmt: skip
    assert got == {"test": {"origins": 2, "covered": 1}, "train": {"origins": 1, "covered": 0}}
    assert core.STEPS_PER_DAY == 48

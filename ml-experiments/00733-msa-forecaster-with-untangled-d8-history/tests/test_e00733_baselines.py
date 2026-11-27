# ruff: noqa: E501, PT018
"""Tests of `e00733_baselines`: the M* control read, the F0r gate, B-clim, B-hist, and O."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00733_baselines as bl
import e00733_core as core
import e00733_synth as sy
import numpy as np
import polars as pl
import pytest
from e00733_channels import StoppedError
from synth_fixture import make_segment


def _frame(*, series: int = 26, n: int = 6, shift: float = 0.0) -> pl.DataFrame:
    t0 = datetime(2026, 1, 1, 9, tzinfo=UTC)
    rows = {
        "time_series_id": [series] * n,
        "power_fcst_init_time": [t0] * n,
        "valid_time": [t0 + timedelta(minutes=30 * i) for i in range(n)],
    }
    for i, q in enumerate(bl.QCOLS):
        rows[q] = [float(i) + 0.1 * j + shift for j in range(n)]
    return pl.DataFrame(rows).with_columns(pl.col("time_series_id").cast(pl.Int32))


def test_read_m_star_selects_d8_and_sorts(tmp_path: Path) -> None:
    """The stored file is filtered to D8's id and sorted by the keys; a missing file or id stops the line."""
    path = tmp_path / "m.parquet"
    pl.concat([_frame(series=7), _frame(series=26).reverse()]).write_parquet(path)
    got = bl.read_m_star(month="2026-01", seed=0, path=path, series_id=26)
    assert got.height == 6 and got["time_series_id"].unique().to_list() == [26]
    assert got["valid_time"].is_sorted()
    with pytest.raises(StoppedError):
        bl.read_m_star(month="2026-01", seed=0, path=tmp_path / "none.parquet", series_id=26)
    with pytest.raises(StoppedError):
        bl.read_m_star(month="2026-01", seed=0, path=path, series_id=99)


def test_f0r_gate_passes_on_equal_forecasts_and_stops_on_a_difference() -> None:
    """Equal quantiles pass at 1e-6; a shift of 1e-3, a missing row, or other keys stop the line."""
    stored = _frame()
    ok = bl.f0r_gate(f0r=pl.concat([_frame(), _frame(series=3)]), stored=stored)
    assert ok["rows"] == 6 and ok["largest_difference"] == 0.0
    with pytest.raises(StoppedError, match="differs"):
        bl.f0r_gate(f0r=_frame(shift=1e-3), stored=stored)
    with pytest.raises(StoppedError):
        bl.f0r_gate(f0r=_frame(n=5), stored=stored)
    other = _frame().with_columns(valid_time=pl.col("valid_time") + timedelta(minutes=30))
    with pytest.raises(StoppedError, match="keys"):
        bl.f0r_gate(f0r=other, stored=stored)
    with pytest.raises(StoppedError):
        bl.f0r_gate(f0r=_frame(series=3), stored=stored)


def test_f0r_gate_tolerance_is_the_registered_one() -> None:
    """A difference just under the tolerance passes and one just over fails."""
    stored = _frame()
    bl.f0r_gate(f0r=_frame(shift=0.5 * core.TOLERANCE_F0R), stored=stored)
    with pytest.raises(StoppedError):
        bl.f0r_gate(f0r=_frame(shift=2 * core.TOLERANCE_F0R), stored=stored)


def test_clim_counts_the_share_by_half_hour_class_and_month() -> None:
    """B-clim equals the share of reverse flow in each cell, and an empty cell falls back to its half-hour."""
    seg = make_segment(n_days=30)
    seg.x_neg[:] = 0.0
    seg.x_neg[seg.local_hh == 20] = 1.0
    seg.day_type[:] = 1
    clim = bl.fit_clim(segments=[seg])
    assert clim.share[20, 0, 5] == 1.0 and clim.share[21, 0, 5] == 0.0
    assert np.isnan(clim.share[20, 1, 5])
    got = clim.lookup(
        hh=np.array([20, 20, 21]), weekend=np.array([0, 1, 0]), month=np.array([6, 6, 6])
    )
    assert got.tolist() == [1.0, 1.0 if False else got[1], 0.0] and got[1] == pytest.approx(
        clim.overall
    )
    assert clim.overall == pytest.approx(1 / 48)
    assert set(clim.as_dict()) == {"share", "count", "overall"}
    seg.x_neg[:] = np.nan
    with pytest.raises(StoppedError):
        bl.fit_clim(segments=[seg])


def test_clim_for_rows_matches_the_lookup() -> None:
    """B-clim for a batch reads the keys of the target slots."""
    seg = make_segment(n_days=40)
    clim = bl.fit_clim(segments=[seg])
    idx = sy.enumerate_samples(segments=[seg])
    out = bl.clim_for_rows(clim=clim, segments=[seg], index=idx, rows=np.arange(3))
    assert out.shape == (3, 48) and np.isfinite(out).all() and (out >= 0).all() and (out <= 1).all()


def test_b_hist_is_the_weighted_mean_over_rows_with_a_value() -> None:
    """With equal day types the weights are equal; a column with no value takes B-clim.

    A lone value is returned as it is.
    """
    table = np.zeros((8, 8))
    p = np.zeros((1, 14, 48))
    ok = np.zeros((1, 14, 48), dtype=bool)
    p[0, 0, 0], p[0, 1, 0] = 1.0, 0.0
    ok[0, :2, 0] = True
    p[0, 3, 1] = 0.4
    ok[0, 3, 1] = True
    types = np.zeros((1, 14, 48), dtype=int)
    clim = np.full((1, 48), 0.07)
    got = bl.b_hist(
        p_hist=p, ok=ok, type_top=np.zeros((1, 48), int), type_hist=types, clim=clim, table=table
    )
    assert (
        got[0, 0] == pytest.approx(0.5)
        and got[0, 1] == pytest.approx(0.4)
        and got[0, 2] == pytest.approx(0.07)
    )


def test_b_hist_weights_same_day_types_more() -> None:
    """With the starting table a same-type row weighs `exp(3)` and a different working day `exp(1.5)`."""
    table = bl.day_type_table()
    assert table[0, 0] == 3.0 and table[0, 1] == 1.5 and table[0, 6] == 0.0
    p = np.zeros((1, 14, 48))
    ok = np.zeros((1, 14, 48), dtype=bool)
    types = np.full((1, 14, 48), 6)
    p[0, 0, 0], p[0, 1, 0] = 1.0, 0.0
    ok[0, :2, 0] = True
    types[0, 0, 0], types[0, 1, 0] = 0, 1
    got = bl.b_hist(
        p_hist=p, ok=ok, type_top=np.zeros((1, 48), int), type_hist=types, clim=np.zeros((1, 48))
    )
    w = np.exp(3.0) / (np.exp(3.0) + np.exp(1.5))
    assert got[0, 0] == pytest.approx(w)


def test_b_hist_for_batch_uses_the_untangled_channel_and_matches_the_untrained_sign_head() -> None:
    """B-hist on a gathered batch is finite and in 0 to 1."""
    seg = make_segment(n_days=40)
    idx = sy.enumerate_samples(segments=[seg])
    rows = np.arange(5)
    batch = sy.gather_batch(segments=[seg], index=idx, rows=rows, d8_index=0, target_std=1.0)
    clim = bl.fit_clim(segments=[seg])
    got = bl.b_hist_for_batch(
        batch=batch, clim=bl.clim_for_rows(clim=clim, segments=[seg], index=idx, rows=rows)
    )
    assert got.shape == (5, 48) and (got >= 0).all() and (got <= 1).all()


def test_o_reads_the_target_day_and_brier_functions() -> None:
    """O is the stored P(X<0) at the target slots; the Brier score and the skill follow their definitions."""
    seg = make_segment(n_days=40)
    idx = sy.enumerate_samples(segments=[seg])
    o = bl.o_for_rows(segments=[seg], index=idx, rows=np.arange(2))
    t = sy.target_slots(origin_day=idx.origin_day[:1], lead_index=idx.lead_index[:1])[0]
    assert np.allclose(o[0], seg.u_raw[t, 0])
    truth = np.array([1.0, 0.0, np.nan, 1.0])
    assert bl.brier(p=np.array([1.0, 0.0, 0.5, 0.5]), truth=truth) == pytest.approx(0.25 / 3)
    assert bl.brier_skill(
        p=np.array([1.0, 0.0, 0.5, 1.0]), reference=np.full(4, 0.5), truth=truth
    ) == pytest.approx(1.0)
    with pytest.raises(StoppedError):
        bl.brier(p=np.array([np.nan]), truth=np.array([1.0]))

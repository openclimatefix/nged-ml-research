"""Tests of the world builder on invented segments: the +1 index, the fold, engine and solar."""
# ruff: noqa: D103, E402

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _n in ("732", "734", "653", "660"):
    for _d in HERE.parent.glob(f"00{_n}-*"):
        sys.path.append(str(_d))
import e00736_common as common
import e00736_world as world

ov = pytest.importorskip("e00732_overlays")

DAYS = 9
GRID = world.GRID_LENGTH


def segment(*, month: str = "2022-04", engine: float = 0.0, solar: float = 0.0) -> SimpleNamespace:
    dates = np.datetime64("2022-04-09") + np.arange(DAYS).astype("timedelta64[D]")
    shape = (DAYS, 48)
    return SimpleNamespace(
        month=month,
        dates=dates,
        cal_index=np.array([-1, 0, 1, 2, 3, 4, 5, 6, -1]),
        valid=np.ones(shape, bool),
        x=np.full(shape, 50.0, dtype=np.float32),
        base=np.full(shape, 50.0, dtype=np.float32),
        engine=np.full(shape, engine, dtype=np.float32),
        solar=np.full(shape, solar, dtype=np.float32),
    )


def overlay_with(*, seg: SimpleNamespace, transfers: list) -> SimpleNamespace:
    arrays = ov.transfer_arrays(transfers=transfers, shape=seg.x.shape)
    return SimpleNamespace(t=arrays.t, transfers=transfers)


def transfer(
    *, tick: int, ticks: int, size: float, partner: str = "D5", jitter: float = 1.0
) -> Any:
    return ov.Transfer(start_tick=tick, ticks=ticks, size=size, partner=partner, jitter=jitter)


SITES = ["D8", "D3", "D5", "D14"]


def test_additions_sit_at_the_bank_half_hour_plus_one_for_a_mid_day_onset() -> None:
    seg = segment()
    day0 = world.grid_day_of(seg=seg)
    # 13:20 on dense day 2, and 00:10 on dense day 3
    t1 = transfer(tick=2 * 288 + 13 * 12 + 4, ticks=12 * 6, size=20.0)
    t2 = transfer(tick=3 * 288 + 2, ticks=12 * 4, size=-10.0, partner="D3")
    o = overlay_with(seg=seg, transfers=[t1, t2])
    add = world.world_additions(
        segments=[seg], overlays=[o], draw=0, design=world.DESIGN_TRANSFERS_ONLY, sites=SITES
    )
    first = day0 * 48 + 2 * 48 + 26 + 1
    assert add["D8"][first - 1] == 0.0
    assert add["D8"][first] > 0.0
    start2 = day0 * 48 + 3 * 48 + 0 + 1
    assert add["D8"][start2 - 1] == 0.0
    assert add["D8"][start2] < 0.0
    # the mirror at D5 is minus the transfer at the same indices
    assert add["D5"][first] == pytest.approx(-add["D8"][first], rel=1e-6)
    assert add["D3"][start2] == pytest.approx(-add["D8"][start2], rel=1e-6)


def test_g2_arithmetic_for_an_mva_primary_in_float64() -> None:
    seg = segment()
    t = transfer(tick=288 + 100, ticks=12 * 30, size=33.3)
    o = overlay_with(seg=seg, transfers=[t])
    add = world.world_additions(
        segments=[seg], overlays=[o], draw=0, design=world.DESIGN_TRANSFERS_ONLY, sites=SITES
    )
    day0 = world.grid_day_of(seg=seg)
    lo = day0 * 48 + t.start_half_hour + common.INDEX_OFFSET
    level = t.mean_levels() * common.POINT_MW
    got = add["D8"][lo : lo + len(level)]
    assert np.all(got != 0.0)
    assert np.max(np.abs(got - level) / np.abs(level)) <= world.TOLERANCE_G2
    assert add["D8"].dtype == np.float64
    # an MVA Primary takes |p + add|; a signed Primary keeps the sign
    power = {s: np.full(GRID, 1.0) for s in SITES}
    mva = {"D8": True, "D3": True, "D5": False, "D14": False}
    neg = {s: np.zeros(GRID) for s in SITES}
    neg["D8"][lo] = -3.0
    neg["D5"][lo] = -3.0
    out = world.build_world_power(power=power, additions=neg, is_mva=mva)
    assert out["D8"][lo] == 2.0
    assert out["D5"][lo] == -2.0
    assert out["D8"][lo + 1] == 1.0


def test_full_design_subtracts_engine_and_solar_and_transfers_only_does_not() -> None:
    seg = segment(engine=4.0, solar=1.0)
    o = overlay_with(seg=seg, transfers=[])
    full = world.world_additions(
        segments=[seg], overlays=[o], draw=0, design=world.DESIGN_FULL, sites=SITES
    )
    only = world.world_additions(
        segments=[seg], overlays=[o], draw=0, design=world.DESIGN_TRANSFERS_ONLY, sites=SITES
    )
    day0 = world.grid_day_of(seg=seg)
    idx = day0 * 48 + 100 + 1
    assert full["D8"][idx] == pytest.approx(-5.0 * common.POINT_MW)
    assert not only["D8"].any()


def test_probe_selects_the_full_design_when_the_code_takes_the_arrays() -> None:
    pytest.importorskip("e00734_decode")
    record = world.probe_design_a()
    assert record["design"] == world.DESIGN_FULL
    assert all(record["checks"].values())


def test_months_and_draws_rank_segments_within_a_month() -> None:
    old = segment(month="2020-02")
    old.dates = old.dates - np.timedelta64(800, "D")
    segs = [old, segment(month="2022-04"), segment(month="2022-04"), segment(month="2022-07")]
    frame = world.months_and_draws(segments=segs)
    assert frame["seg"].to_list() == [1, 2, 3]
    assert frame["draw"].to_list() == [0, 1, 0]
    assert frame["month_idx"].to_list() == [0, 0, 1]


def test_transfer_table_bands_and_counted_cell() -> None:
    seg = segment()
    long_big = transfer(tick=288 + 10, ticks=12 * 30, size=20.0)
    short_big = transfer(tick=3 * 288 + 10, ticks=12 * 3, size=20.0)
    long_small = transfer(tick=5 * 288 + 10, ticks=12 * 30, size=5.0)
    o = overlay_with(seg=seg, transfers=[long_big, short_big, long_small])
    o.transfers = [long_big, short_big, long_small]
    table = world.transfer_table(segments=[seg], overlays=[o], noise_d8=3.4176)
    assert table["counted"].to_list() == [True, False, False]
    assert table["band"].to_list() == ["24 to 96 hours", "under 6 hours", "24 to 96 hours"]
    assert table["start_panel"].to_list()[0] == (
        world.grid_day_of(seg=seg) * 48 + long_big.start_half_hour + 1
    )
    gapped = world.with_gaps(table=table, d8_power=np.full(GRID, np.nan))
    assert gapped["gap"].all()
    assert isinstance(table, pl.DataFrame)

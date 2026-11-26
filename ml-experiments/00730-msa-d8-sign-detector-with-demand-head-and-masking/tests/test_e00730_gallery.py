"""Tests of the gallery's automatic checks and of the panel file that opens the fits."""

from pathlib import Path

import e00730_banks as banks
import e00730_core as core
import e00730_gallery as gallery
import numpy as np


def test_missing_panels_keep_the_fits_closed(tmp_path: Path) -> None:
    record = gallery.build_panel_check(results={"G1_ramp": True}, notes={})
    assert not record["all_passed"]
    gallery.write_panel_check(root=tmp_path, record=record)
    assert not banks.gallery_passed(root=tmp_path)


def test_every_panel_passed_opens_them(tmp_path: Path) -> None:
    record = gallery.build_panel_check(
        results=dict.fromkeys(core.PANEL_REQUIREMENTS, True), notes={}
    )
    gallery.write_panel_check(root=tmp_path, record=record)
    assert banks.gallery_passed(root=tmp_path)


def test_the_panel_file_goes_to_the_private_folder(tmp_path: Path) -> None:
    path = gallery.write_panel_check(
        root=tmp_path, record=gallery.build_panel_check(results={}, notes={})
    )
    assert path.parent == tmp_path / "PRIVATE" / "gallery"


def test_engine_checks() -> None:
    engine = np.zeros(48)
    engine[20:23] = [1.0, 3.0, 5.0]
    engine[23:30] = 5.0
    assert gallery.check_ramp(engine=engine)
    assert not gallery.check_ramp(engine=np.where(engine > 0, 5.0, 0.0))
    one_step = engine.copy()
    one_step[20:23] = [1.0, 5.0, 5.0]  # a partial first half-hour, then the level: not a ramp
    assert not gallery.check_ramp(engine=one_step)
    assert gallery.check_blocks(engine=engine, before=slice(0, 20), block=slice(20, 30))
    assert gallery.check_part_load(engine=engine, unit_sizes=[3.0, 3.0])
    assert not gallery.check_part_load(engine=engine, unit_sizes=[3.0])
    assert not gallery.check_part_load(engine=engine, unit_sizes=[5.2])  # within 0.9 of the total
    tripped = engine.copy()
    tripped[25] = 0.0
    assert gallery.check_trip(engine=tripped)
    assert not gallery.check_trip(engine=engine)
    two_blocks = np.zeros(48)
    two_blocks[10:14] = 3.0
    two_blocks[16:20] = 5.0  # a short gap between two blocks at different levels is not a trip
    assert not gallery.check_trip(engine=two_blocks)
    long_gap = engine.copy()
    long_gap[24:30] = 0.0  # a zero run longer than 4 half-hours is not a trip
    assert not gallery.check_trip(engine=long_gap)
    window = np.zeros(4 * 48)
    window[48:144] = 3.0
    assert gallery.check_spell(engine=window)
    window[100] = 0.0
    assert not gallery.check_spell(engine=window)
    at_start = np.zeros(4 * 48)
    at_start[:96] = 3.0  # the spell starts before the window: it is cut at the left edge
    assert not gallery.check_spell(engine=at_start)
    evening = np.zeros(96)
    evening[36:44] = 4.0
    assert gallery.check_evening(engine=evening)
    assert not gallery.check_morning(engine=evening)
    morning = np.zeros(96)
    morning[12:16] = 4.0
    assert gallery.check_morning(engine=morning)
    assert gallery.check_weekend(engine=morning, first_day_is_weekend=True)
    assert not gallery.check_weekend(engine=morning, first_day_is_weekend=False)
    assert not gallery.check_weekend(engine=np.zeros(96), first_day_is_weekend=True)


def test_weather_and_fold_checks() -> None:
    clear = np.sin(np.linspace(0, np.pi, 48)) * 10
    cloudy = clear * 0.3
    assert gallery.check_cloudy(cloudy=cloudy, clear_same_season=clear)
    assert not gallery.check_cloudy(cloudy=clear, clear_same_season=cloudy)
    assert gallery.check_aggregate_bands(q10=np.zeros(3), q50=np.ones(3), q90=np.full(3, 2.0))
    assert not gallery.check_aggregate_bands(q10=np.ones(3), q50=np.zeros(3), q90=np.full(3, 2.0))
    truth = np.array([-10.0, -0.1, 0.1, 10.0])
    folded = np.sqrt(truth**2 + 0.5**2)
    assert gallery.check_fold(reading=folded, truth=truth, floor=0.5, q_max=0.5)
    assert not gallery.check_fold(
        reading=np.array([10.0, 0.1, 0.5, 10.0]), truth=truth, floor=0.5, q_max=0.5
    )
    assert not gallery.check_fold(
        reading=np.array([12.0, 0.5, 0.5, 10.0]), truth=truth, floor=0.5, q_max=0.5
    )
    unfolded = np.array([-10.0, 0.5, 0.5, 10.0])  # a negative reading: the fold was not applied
    assert not gallery.check_fold(reading=unfolded, truth=truth, floor=0.5, q_max=0.5)
    assert gallery.check_base_equals_truth(truth=clear, base=clear)
    reading = np.ones(96)
    hidden = np.zeros(96, dtype=bool)
    hidden[24:72] = True
    assert not gallery.check_mask_panel(reading=reading, hidden=hidden)
    reading[hidden] = 0.0
    assert gallery.check_mask_panel(reading=reading, hidden=hidden)


def test_s1_is_published_only_as_seasonal_bands() -> None:
    values = np.random.default_rng(0).uniform(size=(40, 48))
    seasons = np.array(core.COMPONENT_SEASONS)[np.arange(40) % 4]
    bands = gallery.seasonal_bands(values=values, seasons=seasons)
    assert set(bands) == set(core.COMPONENT_SEASONS)
    for q in bands.values():
        assert gallery.check_aggregate_bands(q10=q[0], q50=q[1], q90=q[2])


def test_levels_count_plateaus_inside_a_run_not_every_ramp_value() -> None:
    one = np.zeros(48)
    one[10:14] = [2.0, 4.0, 5.0, 5.0]  # a ramp, then one steady level
    one[30:36] = 3.0  # a second run at another share of the one unit
    several = np.zeros(48)
    several[10:20] = [2.0, 4.0, 4.0, 4.0, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5]
    assert gallery.run_plateaus(engine=one) == 1
    assert gallery.run_plateaus(engine=several) == 2
    assert gallery.check_units(one_unit_levels=[1] * 19 + [2], several_unit_levels=[2, 2, 1, 2])
    assert not gallery.check_units(one_unit_levels=[1, 2, 2], several_unit_levels=[2, 1])
    assert not gallery.check_units(one_unit_levels=[1, 1], several_unit_levels=[1, 1])
    assert not gallery.check_units(one_unit_levels=[], several_unit_levels=[2])
    assert gallery.run_plateaus(engine=np.zeros(48)) == 0

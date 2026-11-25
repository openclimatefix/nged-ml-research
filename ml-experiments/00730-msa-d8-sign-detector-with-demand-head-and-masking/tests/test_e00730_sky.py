"""Tests of the sky-gap fill, its bound, its marker entry, and the one fresh gate attempt."""

import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import e00729_sim as sim
import e00730_banks as banks
import e00730_sky as sky
import numpy as np
import pytest
from world730 import make_calendar

MIDDAY = 24  # a half-hour inside the test calendar's sun-up window


def calendar_with_gaps(*, length: int = 2) -> sim.Calendar:
    """Return a test calendar whose irradiance is zero on `length` mid-day half-hours of day 3."""
    cal = make_calendar()
    farms = cal.farms
    ghi, dni, dhi = farms.ghi.copy(), farms.dni.copy(), farms.dhi.copy()
    for arr in (ghi, dni, dhi):
        arr[:, 3, MIDDAY : MIDDAY + length] = 0.0
    return dataclasses.replace(cal, farms=dataclasses.replace(farms, ghi=ghi, dni=dni, dhi=dhi))


def test_a_short_sun_up_gap_is_interpolated_and_the_notch_share_drops() -> None:
    cal = calendar_with_gaps(length=2)
    assert sky.zeroed_sky_share(farms=cal.farms) > 0.0
    filled = sky.fill_sky_gaps(cal=cal).farms
    assert sky.zeroed_sky_share(farms=filled) == 0.0
    left, right = cal.farms.ghi[0, 3, MIDDAY - 1], cal.farms.ghi[0, 3, MIDDAY + 2]
    got = filled.ghi[0, 3, MIDDAY : MIDDAY + 2]
    assert (got > 0).all()
    assert min(left, right) * 0.5 < got.min()
    assert got.max() < max(left, right) * 1.5
    assert (filled.dni[0, 3, MIDDAY : MIDDAY + 2] > 0).all()
    assert (filled.dhi[0, 3, MIDDAY : MIDDAY + 2] > 0).all()
    assert filled.sky_ok.all() == cal.farms.sky_ok.all()
    untouched = np.ones(48, dtype=bool)
    untouched[MIDDAY : MIDDAY + 2] = False
    np.testing.assert_array_equal(filled.ghi[0, 3, untouched], cal.farms.ghi[0, 3, untouched])


def test_a_gap_longer_than_the_limit_excludes_the_farm_day_and_is_never_zero_filled() -> None:
    cal = calendar_with_gaps(length=sky.MAX_SKY_GAP_HH + 1)
    out = sky.fill_sky_gaps(cal=cal).farms
    assert not out.sky_ok[:, 3].any()
    assert np.isnan(out.ghi[0, 3, MIDDAY : MIDDAY + sky.MAX_SKY_GAP_HH + 1]).all()
    assert out.sky_ok[:, 4].all()
    limit = sky.fill_sky_gaps(cal=calendar_with_gaps(length=sky.MAX_SKY_GAP_HH)).farms
    assert limit.sky_ok[:, 3].all()


def test_nothing_outside_the_sun_up_window_is_filled() -> None:
    cal = make_calendar()
    farms = cal.farms
    zenith = farms.zenith.copy()
    zenith[:, 3, MIDDAY - 4 : MIDDAY + 4] = 85.0  # dawn-like: sun at 5 degrees elevation
    ghi = farms.ghi.copy()
    ghi[:, 3, MIDDAY - 1 : MIDDAY + 1] = 0.0
    cal = dataclasses.replace(cal, farms=dataclasses.replace(farms, zenith=zenith, ghi=ghi))
    out = sky.fill_sky_gaps(cal=cal).farms
    np.testing.assert_array_equal(out.ghi[:, 3], ghi[:, 3])
    assert out.sky_ok.all()


def test_a_gap_at_the_first_sun_up_half_hour_holds_the_neighbour_clearness() -> None:
    cal = make_calendar()
    farms = cal.farms
    first = int(np.flatnonzero(farms.zenith[0, 3] < sky.SUN_UP_ZENITH_MAX_DEG)[0])
    ghi = farms.ghi.copy()
    ghi[:, 3, first] = 0.0
    cal = dataclasses.replace(cal, farms=dataclasses.replace(farms, ghi=ghi))
    out = sky.fill_sky_gaps(cal=cal).farms
    assert (out.ghi[:, 3, first] > 0).all()
    assert out.sky_ok.all()


def test_a_short_power_gap_is_interpolated_a_long_one_excludes_the_farm_day_and_zeros_stay() -> (
    None
):
    cal = make_calendar()
    farms = cal.farms
    power = farms.power.copy()
    power[:, 3, MIDDAY : MIDDAY + 3] = np.nan
    power[:, 4, MIDDAY : MIDDAY + 4] = np.nan
    power[:, 6, MIDDAY] = 0.0  # a genuine metered zero
    cal = dataclasses.replace(cal, farms=dataclasses.replace(farms, power=power))
    out = sky.fill_sky_gaps(cal=cal).farms
    assert np.isfinite(out.power[:, 3, MIDDAY : MIDDAY + 3]).all()
    assert out.power_ok[:, 3].all()
    assert not out.power_ok[:, 4].any()
    assert np.isnan(out.power[:, 4, MIDDAY : MIDDAY + 4]).all()
    assert (out.power[:, 6, MIDDAY] == 0.0).all()


def bank_of(*, cal: sim.Calendar, n: int = 12) -> SimpleNamespace:
    segments = []
    for seed in range(200):
        seg = sim.simulate_segment(
            cal=cal, month="2022-03", use=np.ones(len(cal.dates), dtype=bool), world="solar",
            rng=np.random.default_rng(seed),
        )  # fmt: skip
        if seg.source == "S2":
            segments.append(seg)
        if len(segments) == n:
            break
    return SimpleNamespace(segments=segments)


def test_an_unfixed_bank_breaks_the_zeroed_share_bound_and_a_filled_one_meets_it() -> None:
    raw = make_calendar()
    farms = raw.farms
    ghi, dni, dhi = farms.ghi.copy(), farms.dni.copy(), farms.dhi.copy()
    for arr in (ghi, dni, dhi):
        arr[:, ::2, MIDDAY : MIDDAY + 2] = 0.0  # a notch on every other day
    notched = dataclasses.replace(raw, farms=dataclasses.replace(farms, ghi=ghi, dni=dni, dhi=dhi))
    unfixed = banks.zeroed_sunup_share(bank=bank_of(cal=notched))
    fixed = banks.zeroed_sunup_share(bank=bank_of(cal=sky.fill_sky_gaps(cal=notched)))
    assert unfixed > sky.MAX_ZEROED_SHARE
    assert fixed <= sky.MAX_ZEROED_SHARE


def test_the_component_check_fails_an_unfixed_bank(monkeypatch: pytest.MonkeyPatch) -> None:
    cal = make_calendar()
    bank = bank_of(cal=cal)
    notched = copy.deepcopy(bank)
    for seg in notched.segments:
        seg.solar[:, MIDDAY : MIDDAY + 2] = 0.0
        seg.solar[:, MIDDAY - 3] = 1.0
    assert banks.zeroed_sunup_share(bank=notched) > sky.MAX_ZEROED_SHARE
    assert banks.zeroed_sunup_share(bank=bank) <= sky.MAX_ZEROED_SHARE


def test_the_marker_carries_the_fill_version_and_any_limit_change_changes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = banks.run_marker_payload()
    assert before["sky_gap_fill"] == sky.SKY_FILL_VERSION
    monkeypatch.setattr(sky, "SKY_FILL_VERSION", "other")
    assert banks.run_marker_payload() != before


def write_old_marker(root: Path) -> None:
    old = {k: v for k, v in banks.run_marker_payload().items() if k != "sky_gap_fill"}
    (root / banks.RUN_MARKER).write_text(json.dumps(old))


def test_the_rebuild_after_the_fix_needs_the_earlier_outputs_moved_and_nothing_else(
    tmp_path: Path,
) -> None:
    write_old_marker(tmp_path)
    (tmp_path / "banks").mkdir()
    (tmp_path / "banks" / "summary.json").write_text("{}")
    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "gate.json").write_text("{}")
    (tmp_path / "limits.json").write_text("{}")
    with pytest.raises(RuntimeError, match="superseded_run_2"):
        banks.guard_run(root=tmp_path, stage="banks")
    moved = tmp_path / banks.SKY_FIX_FOLDER
    moved.mkdir()
    for name in ("banks", "scores", "limits.json"):
        (tmp_path / name).rename(moved / name)
    banks.guard_run(root=tmp_path, stage="banks")  # allowed: the three outputs are moved


def test_the_rebuild_refuses_fits_a_stray_output_or_another_marker_change(tmp_path: Path) -> None:
    write_old_marker(tmp_path)
    moved = tmp_path / banks.SKY_FIX_FOLDER
    moved.mkdir()
    (moved / "x").write_text("")
    (tmp_path / "margin.json").write_text("{}")
    with pytest.raises(RuntimeError, match=r"margin\.json"):
        banks.guard_run(root=tmp_path, stage="banks")
    (tmp_path / "margin.json").unlink()
    (tmp_path / "selection.json").write_text("{}")
    with pytest.raises(RuntimeError, match=r"selection\.json"):
        banks.guard_run(root=tmp_path, stage="banks")
    (tmp_path / "selection.json").unlink()
    other = {**banks.run_marker_payload(), "ranges": "other"}
    other.pop("sky_gap_fill")
    (tmp_path / banks.RUN_MARKER).write_text(json.dumps(other))
    with pytest.raises(RuntimeError, match="verdict row 0"):
        banks.guard_run(root=tmp_path, stage="banks")


def test_after_the_fix_the_gate_has_one_attempt_and_no_new_retune(tmp_path: Path) -> None:
    assert banks.gate_attempts_allowed(root=tmp_path) == banks.GATE_ATTEMPTS
    (tmp_path / banks.RANGES_OVERRIDE_FILE).write_text("{}")
    assert banks.gate_attempts_allowed(root=tmp_path) == banks.GATE_ATTEMPTS
    (tmp_path / banks.SKY_FIX_FOLDER).mkdir()
    assert banks.gate_attempts_allowed(root=tmp_path) == 1
    with pytest.raises(RuntimeError, match="exists already"):
        banks.write_retune_override(root=tmp_path)


def test_a_gap_at_the_first_sun_up_half_hour_ignores_a_positive_dawn_neighbour() -> None:
    cal = make_calendar()
    farms = cal.farms
    first = int(np.flatnonzero(farms.zenith[0, 3] < sky.SUN_UP_ZENITH_MAX_DEG)[0])
    zenith, ghi = farms.zenith.copy(), farms.ghi.copy()
    zenith[:, 3, first - 1] = 85.0  # dawn: the sun is up but outside the fill window
    ghi[:, 3, first - 1] = 5.0  # a positive, biased-low dawn value
    ghi[:, 3, first] = 0.0
    cal = dataclasses.replace(cal, farms=dataclasses.replace(farms, zenith=zenith, ghi=ghi))
    out = sky.fill_sky_gaps(cal=cal).farms
    np.testing.assert_allclose(out.ghi[:, 3, first], farms.ghi[:, 3, first + 1])
    assert out.ghi[0, 3, first - 1] == 5.0


def test_nan_irradiance_at_sun_up_is_a_gap_too() -> None:
    cal = make_calendar()
    ghi = cal.farms.ghi.copy()
    ghi[:, 3, MIDDAY] = np.nan
    cal = dataclasses.replace(cal, farms=dataclasses.replace(cal.farms, ghi=ghi))
    out = sky.fill_sky_gaps(cal=cal).farms
    assert np.isfinite(out.ghi[:, 3, MIDDAY]).all()
    assert (out.ghi[:, 3, MIDDAY] > 0).all()


def test_the_wiring_in_bank_inputs_fills_the_calendar() -> None:
    import e00730_stages as stages

    cal = calendar_with_gaps(length=2)
    ctx = SimpleNamespace(cal=cal, months=lambda *, split: [split], use={})
    inputs = stages.bank_inputs(ctx=ctx)  # ty: ignore[invalid-argument-type]
    assert sky.zeroed_sky_share(farms=cal.farms) > 0.0
    assert sky.zeroed_sky_share(farms=inputs.cal.farms) == 0.0


def test_the_rebuild_after_the_fix_cannot_be_repeated(tmp_path: Path) -> None:
    old = {**banks.run_marker_payload(), "sky_gap_fill": "an earlier version"}
    (tmp_path / banks.RUN_MARKER).write_text(json.dumps(old))
    moved = tmp_path / banks.SKY_FIX_FOLDER
    moved.mkdir()
    (moved / "x").write_text("")
    with pytest.raises(RuntimeError, match="verdict row 0"):
        banks.guard_run(root=tmp_path, stage="banks")


def _digest_root(root: Path) -> None:
    (root / "banks").mkdir(parents=True, exist_ok=True)
    (root / "banks" / "summary.json").write_text(json.dumps({"digests": {"test": "abc"}}))


def test_a_gallery_record_of_other_banks_does_not_open_the_gate(tmp_path: Path) -> None:
    import e00730_core as core

    _digest_root(tmp_path)
    folder = tmp_path / "PRIVATE" / "gallery"
    folder.mkdir(parents=True)
    panels = dict.fromkeys(core.PANEL_REQUIREMENTS, True)
    record = {"panels": panels, "banks_digest": banks.banks_digest_id(root=tmp_path)}
    (folder / "panel_check.json").write_text(json.dumps(record))
    assert banks.gallery_passed(root=tmp_path)
    (folder / "panel_check.json").write_text(json.dumps({"panels": panels}))
    assert not banks.gallery_passed(root=tmp_path)
    (folder / "panel_check.json").write_text(json.dumps({**record, "banks_digest": "other"}))
    assert not banks.gallery_passed(root=tmp_path)


def test_record_panel_check_refuses_an_auto_check_of_other_banks(tmp_path: Path) -> None:
    import e00730_core as core
    import e00730_stages as stages

    _digest_root(tmp_path)
    folder = tmp_path / "PRIVATE" / "gallery"
    folder.mkdir(parents=True)
    auto = {"panels": dict.fromkeys(core.PANEL_REQUIREMENTS, True)}
    (folder / "auto_check.json").write_text(json.dumps(auto))
    with pytest.raises(RuntimeError, match="other banks"):
        stages.record_panel_check(root=tmp_path, design_lead_confirms={})
    auto["banks_digest"] = banks.banks_digest_id(root=tmp_path)
    (folder / "auto_check.json").write_text(json.dumps(auto))
    done = stages.record_panel_check(
        root=tmp_path, design_lead_confirms=dict.fromkeys(core.PANEL_REQUIREMENTS, True)
    )
    assert done["banks_digest"] == banks.banks_digest_id(root=tmp_path)
    assert banks.gallery_passed(root=tmp_path)

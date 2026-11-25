"""Each leakage assertion must fail on a crafted violation and pass on a clean input."""

import copy
import dataclasses
from datetime import date

import e00729_arms as arms
import e00729_banks as banks
import e00729_core as core
import e00729_leakage as leak
import e00729_sim as sim
import numpy as np
import pytest
from synthetic_world import make_calendar


def seg_of(*, world: str = "both", source_s1: bool = True) -> tuple[sim.Segment, sim.Calendar]:
    cal = make_calendar()
    use = np.array([str(d)[:7] == "2022-03" for d in cal.dates])
    for seed in range(40):
        s = sim.simulate_segment(
            cal=cal, month="2022-03", use=use, world=world, rng=np.random.default_rng(seed)
        )
        if not source_s1 or s.source == "S1":
            return s, cal
    return s, cal


def test_split_integrity_passes_a_clean_split_and_fails_each_violation() -> None:
    days = (
        [date(2022, 1, d) for d in (1, 15, 31)]
        + [date(2022, 2, d) for d in (1, 15)]
        + [date(2022, 3, 15)]
    )
    months = {"2022-01": "training", "2022-02": "test", "2022-03": "training"}
    good = core.assign_day_splits(days=days, splits=months)  # ty: ignore[invalid-argument-type]
    leak.assert_split_integrity(days=days, day_splits=good, month_split=months)
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_split_integrity(
            days=[*days, days[0]], day_splits=[*good, "training"], month_split=months
        )
    bad = list(good)
    bad[2] = "training"  # 31 January is within 2 days of February's test month
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_split_integrity(days=days, day_splits=bad, month_split=months)
    wrong = list(good)
    wrong[5] = "test"
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_split_integrity(days=days, day_splits=wrong, month_split=months)
    months_v = {"2022-01": "validation", "2022-02": "test", "2022-03": "training"}
    vsplit = ["validation", "validation", "validation", "test", "test", "training"]
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_split_integrity(days=days, day_splits=vsplit, month_split=months_v)


def test_lockdown_months_must_be_training() -> None:
    leak.assert_not_in_lockdown(month_split={"2020-05": "training", "2022-01": "test"})
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_not_in_lockdown(month_split={"2020-05": "test"})
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_not_in_lockdown(month_split={"2021-07": "validation"})


def test_seed_spaces_fail_on_overlap_and_on_a_mislabelled_bank() -> None:
    cal = make_calendar()
    use = {"2022-03": np.array([str(d)[:7] == "2022-03" for d in cal.dates])}
    test = banks.build_bank(cal=cal, kind="test", seed=None, months=["2022-03"], month_use=use)  # ty: ignore[invalid-argument-type]
    leak.assert_seed_spaces(train_seeds=[0, 1, 2], evaluation=[test])
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_seed_spaces(train_seeds=[729_000 - core.TRAIN_SEED_BASE], evaluation=[test])
    train = banks.build_bank(cal=cal, kind="train", seed=0, months=["2022-03"], month_use=use)  # ty: ignore[invalid-argument-type]
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_seed_spaces(train_seeds=[0], evaluation=[train])


def test_hashes_must_match_across_arms() -> None:
    same = {"A": {"test": "x", "shifted": "y"}, "B": {"test": "x", "shifted": "y"}}
    leak.assert_hashes_match(digests_by_arm=same)
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_hashes_match(digests_by_arm={**same, "C": {"test": "x", "shifted": "z"}})
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_hashes_match(digests_by_arm={**same, "C": {"test": "x"}})


def test_cams_must_differ_from_the_series_at_d8() -> None:
    cal = make_calendar()
    leak.assert_cams_differs(cal=cal)
    cal.farms.ghi[0][cal.farms.ghi[0] <= 0] = 1.0
    same = dataclasses.replace(cal, d8_ghi=cal.farms.ghi[0].copy())
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_cams_differs(cal=same)


def test_s1_must_read_the_same_base_day_only() -> None:
    s, cal = seg_of(world="solar", source_s1=True)
    if s.source != "S1":
        pytest.skip("no S1 draw in the synthetic world")
    leak.assert_s1_same_day(seg=s, cal=cal)
    shifted = copy.deepcopy(s)
    shifted.solar = np.roll(shifted.solar, 1, axis=0)  # the solar row of the previous day
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_s1_same_day(seg=shifted, cal=cal)


def test_bsp_construction_fails_when_beta_or_the_injection_is_wrong() -> None:
    s, cal = seg_of(world="both", source_s1=False)
    leak.assert_bsp_construction(seg=s, cal=cal)
    other = copy.deepcopy(s)
    other.params["beta"] = s.params["beta"] * 0.5
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_bsp_construction(seg=other, cal=cal)
    other = copy.deepcopy(s)
    other.bsp = other.bsp + 1.0
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_bsp_construction(seg=other, cal=cal)


def test_parameter_ranges_fail_outside_the_page_ranges() -> None:
    s, _ = seg_of(world="both", source_s1=False)
    leak.assert_parameter_ranges(segments=[s])
    other = copy.deepcopy(s)
    other.params["beta"] = 1.5
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_parameter_ranges(segments=[other])


def test_simulator_source_must_not_name_the_record_or_constants(tmp_path) -> None:  # noqa: ANN001
    clean = tmp_path / "clean.py"
    clean.write_text("x = 1\n")
    leak.assert_simulator_source_clean(paths=[clean])
    for bad in leak.FORBIDDEN_IN_SIMULATOR:
        dirty = tmp_path / "dirty.py"
        dirty.write_text(f"import {bad}\n")
        with pytest.raises(leak.CorrectnessFailureError):
            leak.assert_simulator_source_clean(paths=[clean, dirty])


def test_the_real_simulator_modules_are_clean() -> None:
    from pathlib import Path

    folder = Path(sim.__file__).parent
    leak.assert_simulator_source_clean(paths=[folder / "e00729_sim.py", folder / "e00729_banks.py"])


def test_cut_record_and_holdout_checks() -> None:
    d = np.array(["2022-01-01", "2022-02-01"], dtype="datetime64[D]")
    leak.assert_cut_applied(base_dates=d, cut_day=np.datetime64("2022-03-01"))
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_cut_applied(base_dates=d, cut_day=np.datetime64("2022-02-01"))
    leak.assert_record_not_base(
        record_days=np.array(["2023-01-01"], dtype="datetime64[D]"), base_dates=d
    )
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_record_not_base(record_days=d[:1], base_dates=d)
    leak.assert_before_holdout(dates=np.array(["2026-06-30"], dtype="datetime64[D]"))
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_before_holdout(dates=np.array(["2026-07-01"], dtype="datetime64[D]"))


def test_calendar_memorisation_check_fails_on_a_month_row(monkeypatch) -> None:  # noqa: ANN001
    leak.assert_no_calendar_memorisation(specs=arms.ARM_SPECS)
    monkeypatch.setattr(
        arms,
        "stack_names",
        lambda *, spec: (
            [*arms.stack_names.__wrapped__(spec=spec), "month_sin"]
            if hasattr(arms.stack_names, "__wrapped__")
            else ["r", "month_sin"]
        ),
    )
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_no_calendar_memorisation(specs=arms.ARM_SPECS)


def test_choices_must_precede_the_first_test_score() -> None:
    leak.assert_choices_before_test(events=["architecture", "bstar", "margin", "test_score"])
    leak.assert_choices_before_test(events=["architecture"])
    for missing in ("architecture", "bstar", "margin"):
        events = [e for e in ("architecture", "bstar", "margin") if e != missing] + ["test_score"]
        with pytest.raises(leak.CorrectnessFailureError):
            leak.assert_choices_before_test(events=events)
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_choices_before_test(events=["architecture", "bstar", "test_score", "margin"])

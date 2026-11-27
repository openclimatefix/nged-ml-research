"""Tests that each G1 panel picks its own window and checks the arrays it plots."""

from types import SimpleNamespace

import e00730_figures as figures
import numpy as np
import pytest

N_DAYS = 12  # base days of a fake segment, plus one padding day each side
WEEKDAY_1970 = 4  # day numbers are days since 1970-01-01, a Thursday (0): 2 and 3 are the weekend


def fake_segment(
    *, month: str, feature: str | None, n_units: int = 1, start_day: int = 19000
) -> SimpleNamespace:
    """Return a segment whose engine array holds one feature (or nothing) on its day 3."""
    engine = np.zeros((N_DAYS + 2, 48))
    day = engine[3]
    if feature == "ramp":
        day[20:23] = [1.0, 3.0, 5.0]
        day[23:30] = 5.0
    elif feature == "evening":
        day[36:44] = 4.0
    elif feature == "morning":
        day[12:16] = 4.0
    elif feature == "trip":
        day[20:32] = 5.0
        day[26] = 0.0
    elif feature == "spell":
        engine[3:6] = 3.0
        engine[6:8] = 0.0
    elif feature == "part_load":
        day[20:30] = 2.0
    elif feature == "several":
        day[20:25] = 2.0
        day[25:32] = 6.0
        engine[3, 20:25] = 2.0
    dates = np.datetime64("1970-01-01") + np.arange(start_day, start_day + N_DAYS + 2)
    return SimpleNamespace(
        world="engine", month=month, engine=engine, base=np.full_like(engine, 50.0),
        dates=dates.astype("datetime64[D]"),
        params={"n_units": n_units, "total_points_cap": 6.0 if feature == "part_load" else 5.0},
    )  # fmt: skip


def bank(feature: str | None, n: int = 4, n_units: int = 1) -> list[SimpleNamespace]:
    return [
        fake_segment(month=f"m{i}", feature=feature, start_day=19000 + i * 40, n_units=n_units)
        for i in range(n)
    ]  # ty: ignore[invalid-argument-type]


def run(segments: list[SimpleNamespace]) -> figures.GalleryResult:
    out = figures.GalleryResult()
    figures.engine_gallery(segments=segments, out=out)  # ty: ignore[invalid-argument-type]
    return out


def test_a_bank_without_a_feature_fails_that_panel_and_a_bank_with_it_passes() -> None:
    full = [
        *bank("ramp"), *bank("evening"), *bank("morning"), *bank("trip"), *bank("spell"),
        *bank("part_load"),
    ]  # fmt: skip
    out = run(full)
    for key in ("G1_ramp", "G1_evening", "G1_morning", "G1_trip", "G1_spell", "G1_part_load"):
        assert out.results[key], (key, out.notes[key])
    for key, missing in (
        ("G1_ramp", "ramp"), ("G1_trip", "trip"), ("G1_spell", "spell"),
        ("G1_evening", "evening"), ("G1_morning", "morning"),
    ):  # fmt: skip
        without = run([s for s in full if not _has(s, missing)])
        assert not without.results[key], key
    assert not run(bank(None)).results["G1_trip"]
    assert not run(bank(None)).results["G1_spell"]


def _has(seg: SimpleNamespace, feature: str) -> bool:
    probe = fake_segment(month="x", feature=feature)
    return bool(np.array_equal(seg.engine, probe.engine))


def test_each_panel_draws_three_distinct_windows_and_the_panels_differ() -> None:
    segments = [*bank("ramp", 6), *bank("trip", 6), *bank("spell", 6), *bank("evening", 6)]
    used: set[tuple[int, int]] = set()
    seen: dict[str, list[tuple[int, int]]] = {}
    for key, n_days, check in (
        ("ramp", 2, lambda s, d, e: figures.gallery.check_ramp(engine=e)),
        ("trip", 2, lambda s, d, e: figures.gallery.check_trip(engine=e)),
        ("spell", 5, lambda s, d, e: figures.gallery.check_spell(engine=e)),
        ("evening", 2, lambda s, d, e: figures.gallery.check_evening(engine=e)),
    ):
        picked = figures.pick_windows(segments=segments, n_days=n_days, check=check, used=used)  # ty: ignore[invalid-argument-type]
        assert len(picked) == figures.N_DRAWS
        assert len({str(seg.month) for seg, _ in picked}) == figures.N_DRAWS
        for seg, day in picked:  # the check holds on the exact array that is plotted
            window = figures.window_array(seg=seg, day=day, n_days=n_days, name="engine")  # ty: ignore[invalid-argument-type]
            assert check(seg, day, window)
        seen[key] = [(id(seg), day) for seg, day in picked]
    flat = [w for windows in seen.values() for w in windows]
    assert len(flat) == len(set(flat))  # no window appears in two panels


def test_a_gap_between_two_blocks_is_not_a_trip_and_two_partial_days_are_not_a_spell() -> None:
    segments = bank("evening") + bank("morning")
    out = run(segments)
    assert not out.results["G1_trip"]
    assert not out.results["G1_spell"]


def test_the_part_load_panel_draws_the_sum_of_the_unit_sizes() -> None:
    out = run(bank("part_load"))
    spec = str(out.public["G1_part_load"].to_dict())
    assert "sum of unit sizes" in spec


def test_the_spell_panel_shows_five_days() -> None:
    out = run(bank("spell"))
    assert out.results["G1_spell"]
    assert "120" in str(out.public["G1_spell"].to_dict())  # the x axis reaches 5 days = 120 hours


def test_the_units_panel_needs_one_and_several_unit_draws() -> None:
    segments = [*bank("part_load", 4, n_units=1), *bank("several", 4, n_units=3)]
    out = run(segments)
    assert out.notes["G1_units"] == "3 one-unit and 3 several-unit draws"
    assert not run(bank("part_load", 4, n_units=1)).results["G1_units"]


@pytest.mark.parametrize("n_days", [2, 5])
def test_window_chart_axis_length_follows_the_window(n_days: int) -> None:
    chart = figures.window_chart(
        series={"a": np.arange(48 * n_days, dtype=float)}, title="t", n_days=n_days
    )
    values = chart.to_dict()["encoding"]["x"]["axis"]["values"]
    assert values[-1] == 24 * n_days

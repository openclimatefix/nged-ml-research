# ruff: noqa: E501
"""Tests of the gallery panels: each check passes on a sound bank and fails when a component is silent."""

import dataclasses
import itertools
import json
from typing import NamedTuple

import e00729_arms as arms729
import e00729_sim as sim
import e00732_arms as arms
import e00732_core as core
import e00732_gallery as g
import e00732_neighbours as nb
import e00732_overlays as ov
import numpy as np
import pytest
from world732 import make_bank, make_neighbours


class World(NamedTuple):
    """An invented bank with its test overlays, training overlays, and pair indexes."""

    cal: sim.Calendar
    segs: list[sim.Segment]
    series: dict[str, np.ndarray]
    test: list[ov.TransferOverlay]
    train: list[ov.TransferOverlay]
    prepared: list[arms.Prepared]
    index_s2: ov.PairIndex
    index_all: ov.PairIndex
    train_index: ov.PairIndex
    medians: dict[str, float]


@pytest.fixture(scope="module")
def world() -> World:
    cal, segs = make_bank(draws_per_world=6)
    series = make_neighbours(cal=cal)
    stats = arms729.input_stats(cal=cal, train_days=np.arange(len(cal.dates)))
    medians = {
        n: nb.training_median(
            folded=nb.fold(signed=series[n], floor=0.0), train_rows=np.arange(len(cal.dates))
        )
        for n in nb.NEIGHBOURS
    }
    train = ov.overlay_segments(segments=segs, kind="train", seed=0, neighbours=series)
    return World(
        cal, segs, series,
        ov.overlay_segments(segments=segs, kind="test", neighbours=series), train,
        arms.prepare(overlays=train, holidays=cal.holidays, stats=stats, medians=medians),
        ov.build_pair_index(segments=segs, sources=core.TEST_SIBLING_SOURCES),
        ov.build_pair_index(segments=segs), ov.build_pair_index(segments=segs), medians,
    )  # fmt: skip


def test_g10_passes_and_fails_when_the_transfer_is_silent(world: World) -> None:
    charts, ok, _ = g.panel_g10(overlays=world.test, originals=world.segs)
    assert ok
    assert len(charts) == 4
    silent = [dataclasses.replace(o, t=np.zeros_like(o.t)) for o in world.test]
    assert not g.panel_g10(overlays=silent, originals=world.segs)[1]


def test_g10_fails_when_the_bsp_meter_does_not_follow_another_bsp(world: World) -> None:
    broken = [dataclasses.replace(o, t_bsp=np.zeros_like(o.t_bsp)) for o in world.test]
    wins = g.choose_windows(
        overlays=broken,
        originals=world.segs,
        keep=lambda o, tr: tr.partner == core.OTHER_BSP_PARTNER,
    )
    assert wins
    assert not g.check_g10(wins=wins, expect_bsp_moves=True)


def test_pair_panels_pass_and_fail_when_the_injection_is_silent(world: World) -> None:
    assert g.panel_g11(segments=world.segs, index=world.index_s2)[1]
    pairs = g.pair_windows(
        segments=world.segs,
        overlays=None,
        index=world.index_s2,
        amount=20.0,
        source="S2",
        public=True,
    )
    assert g.check_pairs(pairs=pairs)
    silent = [dataclasses.replace(p, delta=np.zeros_like(p.delta)) for p in pairs]
    assert not g.check_pairs(pairs=silent)
    unchanged = [dataclasses.replace(p, r1=p.r0.copy()) for p in pairs]
    assert not g.check_pairs(pairs=unchanged)


def test_g17_holds_s1_pairs_and_stays_out_of_the_public_set(world: World) -> None:
    charts, ok, _ = g.panel_g11(
        segments=world.segs, index=world.index_all, public=False, source="S1"
    )
    assert ok
    assert all(k.startswith("g17") for k in charts)
    assert g.check_private_only(public={"g11_pairs_5": 1}, private=charts)
    assert not g.check_private_only(public={"g17_pairs_5": 1}, private=charts)
    assert not g.check_private_only(public={}, private={})


def test_g12_passes_and_fails_when_the_sky_is_not_raised(world: World) -> None:
    charts, ok, _ = g.panel_g12(segments=world.segs)
    assert ok
    assert charts
    csi = np.array([0.5, 0.5, 0.0])
    cosz = np.array([0.5, 0.5, 0.0])
    good = {"csi": csi, "cosz": cosz, "raised": csi * 1.2}
    assert g.check_g12(windows=[good, good])
    assert not g.check_g12(windows=[{**good, "raised": csi}])
    assert not g.check_g12(windows=[{**good, "raised": csi * 1.2 + np.array([0, 0, 0.3])}])
    assert not g.check_g12(windows=[])


def test_g13_passes_and_fails_when_the_solar_target_is_silent(world: World) -> None:
    assert g.panel_g13(overlays=world.test, originals=world.segs)[1]
    silent = [
        dataclasses.replace(o, seg=dataclasses.replace(o.seg, solar=np.zeros_like(o.seg.solar)))
        for o in world.test
    ]
    assert not g.panel_g13(overlays=silent, originals=world.segs)[1]
    no_events = [dataclasses.replace(o, event=np.zeros_like(o.event)) for o in world.test]
    assert not g.panel_g13(overlays=no_events, originals=world.segs)[1]


def test_g14_passes_and_fails_outside_the_registered_ranges(world: World) -> None:
    charts, ok, _, table = g.panel_g14(
        overlays_by_name={"train seed 0": world.train, "test": world.test}
    )
    assert ok
    assert charts
    assert {r["direction"] for r in table} == {"D8 gains load", "D8 loses load"}
    params = {n: g.parameter_arrays(overlays=o) for n, o in {"test": world.test}.items()}
    assert g.check_g14(params=params, table=table)
    too_big = {"test": {**params["test"], "points": params["test"]["points"] * 3}}
    assert not g.check_g14(params=too_big, table=table)
    assert not g.check_g14(params=params, table=[])
    one_direction = [r for r in table if r["direction"] == "D8 gains load"]
    assert not g.check_g14(params=params, table=one_direction)


def test_g15_passes_and_fails_when_a_component_is_silent(world: World) -> None:
    charts, ok, _ = g.panel_g15(overlays=world.test, originals=world.segs, index=world.index_s2)
    assert ok
    assert charts
    silent = [
        dataclasses.replace(o, seg=dataclasses.replace(o.seg, engine=np.zeros_like(o.seg.engine)))
        for o in world.test
    ]
    assert not g.panel_g15(overlays=silent, originals=world.segs, index=world.index_s2)[1]


def test_g16_reads_the_pairing_shares_and_fails_below_the_bar() -> None:
    def summary(share: float) -> dict:
        return {"banks": {"train_s0": {"share_unpaired_by_month_position": [share, share]}}}

    assert g.panel_g16(summary=summary(0.01))[1]
    assert not g.panel_g16(summary=summary(0.2))[1]
    assert not g.panel_g16(summary={"banks": {}})[1]


def test_g18_passes_and_fails_when_a_mirror_is_silent(world: World) -> None:
    charts, ok, _ = g.panel_g18(overlays=world.test, originals=world.segs, medians=world.medians)
    assert ok
    assert len(charts) == 4
    silent = [dataclasses.replace(o, neighbours=o.neighbours_real) for o in world.test]
    assert not g.panel_g18(overlays=silent, originals=world.segs, medians=world.medians)[1]


def test_g19_passes_and_fails_when_the_shapes_collapse(world: World) -> None:
    shapes = g.shape_windows(overlays=world.test, originals=world.segs)
    assert g.check_g19(shapes=shapes)
    assert shapes is not None
    flat = dict(shapes)
    flat["demand"] = shapes["constant"]
    assert not g.check_g19(shapes=flat)
    assert not g.check_g19(shapes=None)


def test_g20_passes_and_fails_when_a_neighbour_goes_missing(world: World) -> None:
    reading = world.cal.reading
    kwargs = {
        "base_dates": world.cal.dates, "reading": reading, "arrays": world.series,
        "overlays_by_name": {"train seed 0": world.train, "test": world.test}, "noise_d8": 5.0,
    }  # fmt: skip
    charts, ok, _ = g.panel_g20(**kwargs)
    assert ok
    assert charts
    gone = {**world.series, "D3": np.full_like(world.series["D3"], np.nan)}
    assert not g.panel_g20(**{**kwargs, "arrays": gone})[1]  # ty: ignore[invalid-argument-type]


def test_g21_passes_and_fails_when_the_hidden_neighbours_leak(world: World) -> None:
    tensor = g.build_tensor(prepared=world.prepared, index=world.train_index)
    assert tensor is not None
    assert g.check_g21(t=tensor)
    leaked = tensor.plain.copy()
    leaked[arms.NB_ROW["D3_value"], np.flatnonzero(tensor.hidden)[0]] = 1.0
    assert not g.check_g21(t=dataclasses.replace(tensor, plain=leaked))
    assert not g.check_g21(t=dataclasses.replace(tensor, injected=tensor.plain.copy()))
    assert not g.check_g21(t=dataclasses.replace(tensor, no_bsp=tensor.plain.copy()))
    assert not g.check_g21(t=None)
    charts = g.tensor_charts(t=tensor)
    assert set(charts) == {"g21_tensor", "g21_variants"}


def test_g21_is_built_from_published_siblings_only(world: World) -> None:
    import e00732_leakage as leak

    segments = [o.seg for o in world.train]
    index_s2 = ov.build_pair_index(segments=world.segs, sources=core.TEST_SIBLING_SOURCES)
    g.assert_siblings_are_published(index=index_s2, segments=world.segs)
    with_s1 = ov.build_pair_index(segments=world.segs)  # the default sources hold S1 and S2
    assert any(world.segs[int(j)].source == "S1" for s in with_s1.siblings.values() for j in s)
    with pytest.raises(leak.CorrectnessFailureError, match="published source"):
        g.assert_siblings_are_published(index=with_s1, segments=world.segs)
    with pytest.raises(leak.CorrectnessFailureError, match="published source"):
        g.panel_g21(prepared=world.prepared, index=with_s1, segments=segments)
    with pytest.raises(leak.CorrectnessFailureError, match="not built from"):
        g.assert_siblings_are_published(index=index_s2, segments=world.segs[:-1])
    charts, ok, _ = g.panel_g21(prepared=world.prepared, index=index_s2, segments=segments)
    assert ok
    assert set(charts) == {"g21_tensor", "g21_variants"}


def test_auto_check_lists_every_required_panel_and_fails_on_a_missing_one() -> None:
    panels = g.Panels(results=dict.fromkeys(core.PANEL_REQUIREMENTS, True))
    assert g.auto_check_record(panels=panels, banks_digest="d")["all_passed"]
    panels.results.pop("G21_input_tensor")
    record = g.auto_check_record(panels=panels, banks_digest="d")
    assert not record["all_passed"]
    assert not record["panels"]["G21_input_tensor"]
    json.dumps(record)


def test_no_panel_names_a_calendar_date(world: World) -> None:
    chart = g.panel_g10(overlays=world.test, originals=world.segs)[0]["g10_other_bsp"]
    text = json.dumps(chart.to_dict())
    assert "temporal" not in text
    assert "T00:00" not in text
    assert "2022-" not in text


def test_g11_draws_differ_and_the_same_windows_recur_at_each_amount(world: World) -> None:
    def draw(amount: float) -> list[g.PairWin]:
        return g.pair_windows(
            segments=world.segs, overlays=None, index=world.index_s2, amount=amount,
            source="S2", public=True,
        )  # fmt: skip

    small, large = draw(5.0), draw(50.0)
    assert len(small) == len(large) >= 2
    for a, b in itertools.combinations(small, 2):
        assert not np.array_equal(a.x0, b.x0, equal_nan=True)
        assert not np.array_equal(a.delta / 5.0, b.delta / 5.0)
    for a, b in zip(small, large, strict=True):
        assert np.array_equal(a.x0, b.x0, equal_nan=True)
    twin = [dataclasses.replace(small[0]), dataclasses.replace(small[0]), *small[1:]]
    assert not g.check_pairs(pairs=twin)


def test_g21_window_shows_a_bsp_step_and_draws_the_neighbours_absent_from_u_n(world: World) -> None:
    tensor = g.build_tensor(prepared=world.prepared, index=world.train_index)
    assert tensor is not None
    assert g.bsp_step_outside(x=tensor.plain, hidden=tensor.hidden)
    assert not g.bsp_step_outside(x=tensor.no_bsp, hidden=tensor.hidden)
    flat = tensor.plain.copy()
    flat[[arms.ROW["d_bsp"], arms.ROW["bsp_mask"]]] = 0.0
    assert not g.check_g21(t=dataclasses.replace(tensor, plain=flat))
    spec = json.dumps(g.tensor_charts(t=tensor)["g21_variants"].to_dict())
    assert "sibling of source S2" in spec
    for name in nb.NEIGHBOURS:
        assert f"{name}_change" in spec


def test_g10_truth_lines_have_distinct_colours_and_the_sign_is_explained(world: World) -> None:
    wins = g.choose_windows(overlays=world.test, originals=world.segs, keep=lambda o, tr: True)
    spec = g.g10_charts(wins=wins, title="t", n_days=2).to_dict()
    text = json.dumps(spec)
    assert "Import is positive" in text
    scale = spec["vconcat"][0]["layer"][-2]["encoding"]["color"]["scale"]
    colours = dict(zip(scale["domain"], scale["range"], strict=True))
    assert colours["truth without the transfer"] != colours["truth with the transfer"]


def test_g14_histogram_draws_the_registered_rate_not_the_realised_count(world: World) -> None:
    for overlays in (world.train, world.test):
        p = g.parameter_arrays(overlays=overlays)
        lo, hi = core.TRANSFER_RATE_TRAIN
        assert p["drawn_rate"].min() >= lo - 1e-9
        assert p["drawn_rate"].max() <= hi + 1e-9


def test_g20_coverage_axis_reaches_the_lowest_monthly_coverage(world: World) -> None:
    thin = {
        **world.series,
        "D14": np.where(
            np.arange(world.series["D14"].shape[0])[:, None] < 40, np.nan, world.series["D14"]
        ),
    }
    charts, _, _ = g.panel_g20(
        base_dates=world.cal.dates, reading=world.cal.reading, arrays=thin,
        overlays_by_name={"train seed 0": world.train, "test": world.test}, noise_d8=5.0,
    )  # fmt: skip
    text = json.dumps({k: v.to_dict() for k, v in charts.items()})
    cov = g.monthly_coverage(base_dates=world.cal.dates, reading=world.cal.reading, arrays=thin)
    lowest = min(float(v.min()) for v in cov.values())
    assert f'"domain": [{min(0.5, float(np.floor(lowest * 10) / 10))}, 1.0]' in text

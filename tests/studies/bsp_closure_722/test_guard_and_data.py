# ruff: noqa: E501
"""The anonymisation guard, the order of reads, the timing gate, and the topology join (synthetic)."""

from datetime import UTC, datetime
from pathlib import Path

import e00722_common as common
import e00722_core as core
import e00722_d8core as d8core
import e00722_data as data
import e00722_guard as guard
import e00722_u0 as u0
import numpy as np
import polars as pl
import pytest


def test_a_public_table_with_a_timestamp_or_a_date_is_refused(tmp_path: Path):
    ok = pl.DataFrame({"time_series_id": [1], "slope": [0.5], "label": ["x"]})
    guard.write_public_csv(frame=ok, name="ok.csv", out_dir=tmp_path)
    assert (tmp_path / "ok.csv").is_file()
    stamped = pl.DataFrame({"t": [datetime(2025, 1, 1, tzinfo=UTC)], "v": [1.0]})
    with pytest.raises(guard.AnonymisationError):
        guard.write_public_csv(frame=stamped, name="bad.csv", out_dir=tmp_path)
    named = pl.DataFrame({"time": [1], "v": [1.0]})
    with pytest.raises(guard.AnonymisationError):
        guard.write_public_csv(frame=named, name="bad2.csv", out_dir=tmp_path)
    textual = pl.DataFrame({"note": ["on 2025-03-04 it ran"]})
    with pytest.raises(guard.AnonymisationError):
        guard.write_public_csv(frame=textual, name="bad3.csv", out_dir=tmp_path)
    with pytest.raises(guard.AnonymisationError):
        guard.write_public_json(payload={"x": "2025-03-04"}, name="bad.json", out_dir=tmp_path)
    assert not (tmp_path / "bad.csv").exists()


def test_stdout_prints_aggregates_and_refuses_a_date(capsys: pytest.CaptureFixture[str]):
    guard.say("slope", 0.97, "interval", (0.9, 1.0), "count", 12)
    assert "slope 0.97" in capsys.readouterr().out
    with pytest.raises(guard.AnonymisationError):
        guard.say("event on 2025-03-04")


def test_dated_detail_goes_under_private(tmp_path: Path):
    frame = pl.DataFrame({"time": [datetime(2025, 1, 1, tzinfo=UTC)], "v": [1.0]})
    path = guard.write_private_parquet(
        frame=frame, name="d.parquet", private_dir=tmp_path / "PRIVATE"
    )
    assert path.parent.name == "PRIVATE"
    assert guard.PRIVATE_DIR.name == "PRIVATE"
    assert guard.PRIVATE_DIR.parent == guard.OUT_DIR


def test_the_read_log_proves_the_class_audit_came_before_any_primary_reading():
    log = guard.ReadLog()
    log.record(name="cams", kind="sky")
    log.record(name="farms", kind="farm")
    log.mark_audit_done()
    log.record(name="primaries", kind="primary")
    log.assert_audit_before_primary_reads()
    assert [e["kind"] for e in log.as_list()] == ["sky", "farm", "marker", "primary"]
    late = guard.ReadLog()
    late.record(name="primaries", kind="primary")
    late.mark_audit_done()
    with pytest.raises(guard.ReadOrderError):
        late.assert_audit_before_primary_reads()
    never = guard.ReadLog()
    never.record(name="cams", kind="sky")
    with pytest.raises(guard.ReadOrderError):
        never.assert_audit_before_primary_reads()


def test_the_timing_gate_projects_the_loop_and_counts_setup_once():
    """B1: a large setup and a small per-unit time must not trip the gate."""
    gate = guard.TimingGate(cap_core_seconds=3600.0, start=0.0)
    gate.mark(now=500.0)
    assert gate.check(done=1, total=10, now=510.0) == 600.0
    old_style = 510.0 / 1 * 10
    assert old_style > 3600.0
    with pytest.raises(guard.TimingGateError):
        gate.check(done=1, total=10, now=900.0)


def test_the_run_log_holds_the_code_hash_input_hashes_memory_and_core_seconds(tmp_path: Path):
    inp = tmp_path / "in.csv"
    inp.write_text("a,b\n1,2\n")
    gate = guard.TimingGate()
    path = guard.write_run_log(
        job="test",
        folder=Path(guard.__file__).parent,
        inputs=[inp, tmp_path / "missing"],
        gate=gate,
        out_dir=tmp_path,
    )
    text = path.read_text()
    for key in ("code_hash", "peak_memory_mb", "core_seconds", "inputs", "n_boot"):
        assert key in text
    assert len(guard.code_hash(folder=Path(guard.__file__).parent)) == 64
    assert guard.file_fingerprint(path=inp) == guard.file_fingerprint(path=inp)


def test_a_job_refuses_without_the_launch_wrapper_settings():
    with pytest.raises(common.JobRefusedError):
        common.assert_environment(env={})
    common.assert_environment(env=dict(common.ENV_DEFAULTS))
    with pytest.raises(common.JobRefusedError):
        common.read_checks(out_dir=Path("/nonexistent-722"))
    with pytest.raises(common.JobRefusedError):
        common.require_class_audit(checks={"class_audit_ok": False})
    with pytest.raises(common.JobRefusedError):
        common.require_stamps(checks={"stamp_alignment_ok": False})


def _frames(*, move_d3: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    labels = [*core.BSP_A_PRIMARIES, *core.BSP_B_PRIMARIES]
    units = dict.fromkeys((*core.BSP_A_UNSIGNED, *core.BSP_B_UNSIGNED), "MVA")
    site_map = pl.DataFrame(
        {"label": labels, "time_series_id": list(range(1, 17)), "capacity_mw": [20.0] * 16}
    )
    meta_rows, hier_rows = [], []
    for i, label in enumerate(labels):
        bsp = 900 if (label in core.BSP_A_PRIMARIES) != (move_d3 and label == "D3") else 901
        meta_rows.append(
            (i + 1, "Disaggregated Demand", units.get(label, "MW"), 100 + i, "Primary")
        )
        hier_rows.append((100 + i, bsp))
    for bsp, ts in ((900, 50), (901, 51)):
        meta_rows.append((ts, "Raw Flow", "MW", bsp, "BSP"))
        hier_rows.append((bsp, bsp))
    n = 60
    for bsp, count, kind in ((900, 8, "PV"), (901, 5, "Wind")):
        for _ in range(count):
            n += 1
            meta_rows.append((n, kind, "MW", 500 + n, "EHV Customer"))
            hier_rows.append((500 + n, bsp))
    metadata = pl.DataFrame(
        meta_rows,
        schema=[
            "time_series_id",
            "time_series_type",
            "units",
            "substation_number",
            "substation_type",
        ],
        orient="row",
    )
    hierarchy = pl.DataFrame(
        hier_rows, schema=["substation_number", "bsp_substation_number"], orient="row"
    )
    return site_map, metadata, hierarchy


def test_the_topology_join_assigns_each_primary_to_its_bsp_and_lists_generators():
    site_map, metadata, hierarchy = _frames()
    topology = data.build_topology(site_map=site_map, metadata=metadata, hierarchy=hierarchy)
    flags = data.assert_topology(topology=topology)
    assert flags == {"generators_under_a_are_8": True, "generators_under_b_are_5": True}
    assert topology.bsp_meter == {"A": 50, "B": 51}
    assert topology.bsp_of(label="D12") == "A"
    assert topology.bsp_of(label="D4") == "B"
    assert topology.ids_of(labels=("D3",)) == {"D3": 1}
    assert topology.capacity(label="D3") == 20.0
    assert topology.generators.height == 13


def test_a_topology_that_differs_from_the_design_stops_the_slice():
    site_map, metadata, hierarchy = _frames(move_d3=True)
    topology = data.build_topology(site_map=site_map, metadata=metadata, hierarchy=hierarchy)
    with pytest.raises(data.TopologyError):
        data.assert_topology(topology=topology)


def test_engine_changes_ignore_the_first_row_of_an_era_and_gaps():
    k = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0])
    has = np.array([True, True, True, True, True, False, True, True])
    era = np.array(["a", "a", "a", "a", "b", "b", "b", "b"], dtype=object)
    got = data.engine_changes(k_path=k, has_row=has, era=era)
    assert got.tolist() == [False, False, True, False, False, False, False, True]


def test_isolated_events_drop_crowded_windows_and_gaps():
    n = 200
    k = np.zeros(n)
    k[60:] = 1.0
    k[100:] = 2.0
    k[103:] = 3.0
    has = np.ones(n, dtype=bool)
    era = np.full(n, "net", dtype=object)
    change = data.engine_changes(k_path=k, has_row=has, era=era)
    h, dk, ev_era, cand = d8core.isolated_events(
        change=change, k_path=k, era=era, has_row=has, in_drop=np.zeros(n, dtype=bool)
    )
    assert h.tolist() == [60]
    assert dk.tolist() == [1.0]
    assert ev_era.tolist() == ["net"]
    assert not cand[55:67].any()
    assert cand[20]


def test_job_one_helpers_on_synthetic_arrays():
    x = np.full(1000, np.nan)
    x[10:20] = 1.0
    x[200:210] = 1.0
    x[400:410] = 1.0
    assert u0.long_gaps(x=x) == 2
    assert u0.long_gaps(x=np.full(5, np.nan)) == 0
    span = np.zeros(1000, dtype=bool)
    span[:100] = True
    assert u0.generator_coverage(x=x, span=span) == 0.1
    rng = np.random.default_rng(13)
    record = np.full(2000, np.nan)
    record[100:1600] = rng.normal(5, 2, 1500)
    gens = {7: record + rng.normal(0, 0.01, 2000), 8: rng.normal(0, 1, 2000)}
    assert u0.identify_engine_series(record=record, gens=gens) == 7
    assert u0.identify_engine_series(record=record, gens={8: gens[8]}) is None
    frame = pl.DataFrame({"Substation Number": ["1", "2", "1"], "Start": ["a", "b", "c"]})
    table = u0.switching_windows_per_primary(
        sheets={"s": frame}, numbers={"D1": 1, "D2": 2, "D3": 9}
    )
    assert dict(zip(table["label"], table["n_windows"], strict=True)) == {"D1": 2, "D2": 1, "D3": 0}
    assert table["readable"].all()
    empty = u0.switching_windows_per_primary(
        sheets={"s": pl.DataFrame({"x": [1]})}, numbers={"D1": 1}
    )
    assert not empty["readable"][0]


def test_the_era_split_table_divides_each_excess_by_d8s_own_excess():
    rows = []
    for events, d8, meter in (("on_iso_net", -2.0, -1.5), ("off_iso_net", 2.0, 2.2)):
        rows.append(
            {
                "series": "d8_reading",
                "events": events,
                "n_events": 50,
                "excess": d8,
                "excess_ci_lo": d8 - 0.1,
                "excess_ci_hi": d8 + 0.1,
            }
        )
        rows.append(
            {
                "series": "meter",
                "events": events,
                "n_events": 50,
                "excess": meter,
                "excess_ci_lo": meter - 0.3,
                "excess_ci_hi": meter + 0.3,
            }
        )
    table = u0.era_split_from_results(results=pl.DataFrame(rows)).sort("events")
    got = dict(zip(table["events"], table["ratio"], strict=True))
    assert got == {"off_iso_net": 1.1, "on_iso_net": 0.75}
    assert (table["ratio_lo"] <= table["ratio"]).all()
    assert (table["ratio"] <= table["ratio_hi"]).all()


def test_the_dominance_flag_never_reports_a_capacity():
    assert core.dominance_flag(largest_farm_mw=4.0, total_pv_mw=12.0) == "yes"
    assert core.dominance_flag(largest_farm_mw=3.0, total_pv_mw=12.0) == "no"
    assert core.dominance_flag(largest_farm_mw=3.0, total_pv_mw=None) == "unknown"
    assert core.dominance_flag(largest_farm_mw=3.0, total_pv_mw=0.0) == "unknown"

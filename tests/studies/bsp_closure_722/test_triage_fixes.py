# ruff: noqa: E501
"""One test for each fix of the first code review (A1 to A4, B1 to B13) that jobs 1 to 3 carry."""

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import e00722_common as common
import e00722_core as core
import e00722_data as data
import e00722_guard as guard
import e00722_u0 as u0
import e00722_u2a as u2a
import numpy as np
import polars as pl
import pytest
from e00722_testkit import calendar, make_weights

EXPERIMENT_DIR = Path(data.__file__).resolve().parent


# ---- A1: the matched key exists for the whole grid ----------------------------------------------
def test_a_key_that_is_minus_one_outside_a_subspan_makes_the_engine_test_void():
    """A1: the failure of the first version, shown on purpose, and its repair."""
    n_days = 120
    cal = calendar(n_days=n_days)
    n = n_days * 48
    rng = np.random.default_rng(31)
    record = np.full(n, np.nan)
    a, b = 70 * 48, 70 * 48 + 44 * 48
    record[a:b] = np.cumsum(rng.normal(0, 0.8, b - a)) + 10.0
    residual = -np.nan_to_num(record) + rng.normal(0, 0.2, n)
    weights = make_weights(n_days=n_days)
    frame_only_key = np.where(np.arange(n) < 40 * 48, cal["key"], -1)
    broken = u2a.engine_test(
        residual=residual,
        d8_reading=np.zeros(n),
        record=record,
        key=frame_only_key,
        quiet=cal["quiet"],
        day=cal["day"],
        weights=weights,
    )
    assert broken["main"].n_rows == 0
    assert (
        core.engine_bar(
            lo=float(broken["main"].lo[0]),
            hi=float(broken["main"].hi[0]),
            se=float(broken["main"].se[0]),
            shifted={7: (-0.1, 0.1), 14: (-0.1, 0.1)},
        )
        == "void"
    )
    fixed = u2a.engine_test(
        residual=residual,
        d8_reading=np.zeros(n),
        record=record,
        key=cal["key"],
        quiet=cal["quiet"],
        day=cal["day"],
        weights=weights,
    )
    assert fixed["main"].n_rows > 40 * 48
    assert abs(fixed["main"].coef[0] + 1.0) < 0.15


def test_the_calendar_key_covers_the_whole_grid_and_uses_the_local_calendar():
    key = data.calendar_key()
    assert key.shape == (core.N_GRID,)
    assert (key >= 0).all()
    # The first half-hour starts 00:00 UTC on a Sunday of September, which is 01:00 BST: local half-hour 2.
    assert key[0] == (2 * 13 + 9) * 2 + 1
    assert key[1] == (3 * 13 + 9) * 2 + 1


# ---- A2: the same-sky pairs are computed for metered farms too ----------------------------------
def test_every_host_and_farm_pair_is_computed_and_the_metered_ones_are_flagged():
    n = 48 * 40
    rng = np.random.default_rng(32)
    shared = rng.uniform(0.2, 1.0, n)
    k_of = {"D5": shared, "F1": shared + rng.normal(0, 0.01, n), "F2": rng.uniform(0.2, 1.0, n)}
    lit = {name: np.ones(n, dtype=bool) for name in k_of}
    table = u0.same_sky_pair_table(
        k_of=k_of, lit_of=lit, hosts=("D5",), farms=["F1", "F2"], metered_farms=["F1"]
    )
    assert table.height == 2
    rows = {r["farm"]: r for r in table.iter_rows(named=True)}
    assert rows["F1"]["farm_metered_under_a_bsp"]
    assert rows["F1"]["same_sky"]
    assert not rows["F2"]["farm_metered_under_a_bsp"]
    assert not rows["F2"]["same_sky"]
    assert int(table["same_sky"].sum()) == 1
    assert int(table.filter(~pl.col("farm_metered_under_a_bsp"))["same_sky"].sum()) == 0


# ---- A3: no generator id outside PRIVATE/ -------------------------------------------------------
def test_a_public_frame_with_a_generator_id_or_a_generator_time_series_id_is_refused(
    tmp_path: Path,
):
    with pytest.raises(guard.AnonymisationError):
        guard.write_public_csv(
            frame=pl.DataFrame({"generator_id": [3], "v": [1.0]}), name="a.csv", out_dir=tmp_path
        )
    mixed = pl.DataFrame(
        {"kind": ["primary", "generator"], "time_series_id": [1, 2], "v": [1.0, 2.0]}
    )
    with pytest.raises(guard.AnonymisationError):
        guard.write_public_parquet(frame=mixed, name="b.parquet", out_dir=tmp_path)
    anonymous = mixed.with_columns(
        time_series_id=pl.when(pl.col("kind") == "primary").then(pl.col("time_series_id"))
    )
    guard.write_public_parquet(frame=anonymous, name="c.parquet", out_dir=tmp_path)
    assert (tmp_path / "c.parquet").is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_generator_labels_are_anonymous_in_id_order_and_the_map_lives_in_private(tmp_path: Path):
    labels = core.generator_labels(bsp=["B", "A", "A", "B"], ids=[40, 30, 10, 20])
    assert labels == {10: "GA1", 30: "GA2", 20: "GB1", 40: "GB2"}
    classes = pl.DataFrame({c: [1] for c in common.CLASS_COLUMNS}).with_columns(
        generator=pl.lit("GA1")
    )
    guard.write_public_parquet(frame=classes, name=common.CLASSES_NAME, out_dir=tmp_path)
    assert "generator_id" not in pl.read_parquet(tmp_path / common.CLASSES_NAME).columns
    private = tmp_path / "PRIVATE"
    guard.write_private_parquet(
        frame=pl.DataFrame({"generator": ["GA1"], "generator_id": [10]}),
        name=common.LABELS_NAME,
        private_dir=private,
    )
    joined = common.read_classes(out_dir=tmp_path, private_dir=private)
    assert joined["generator_id"].to_list() == [10]
    pl.DataFrame({"generator": ["GA1"]}).write_parquet(tmp_path / common.CLASSES_NAME)
    with pytest.raises(common.JobRefusedError):
        common.read_classes(out_dir=tmp_path, private_dir=private)


# ---- A4: both add-back forms are recovered ------------------------------------------------------
@pytest.mark.parametrize("truth", ["abs_x_then_plus_a", "abs_of_sum"])
def test_each_addback_form_is_recovered_by_its_own_regressor_only(truth: str):
    n_days = 100
    cal = calendar(n_days=n_days)
    n = n_days * 48
    rng = np.random.default_rng(33 if truth == "abs_of_sum" else 34)
    x = rng.normal(0, 2.0, n)
    a = np.abs(rng.normal(3.0, 1.0, n))
    if truth == "abs_x_then_plus_a":
        reading, folded = np.abs(x) + a, np.abs(x)
    else:
        reading, folded = np.abs(x + a), np.abs(x + a)
    y = folded + rng.normal(0, 0.05, n)
    weights = make_weights(n_days=n_days)
    slope: dict[str, float] = {}
    for form in core.ADDBACK_FORMS:
        reg = core.addback_regressor(reading=reading, added_back=a, form=form)
        fit = core.bootstrap_ols(y=y, x=reg, day=cal["day"], weights=weights, names=["s"])
        slope[form] = float(fit.coef[0])
    assert abs(slope[truth] - 1.0) < 0.03
    other = next(f for f in core.ADDBACK_FORMS if f != truth)
    assert abs(slope[other] - 1.0) > 0.1


def test_the_binding_addback_form():
    assert core.binding_addback_form(n_candidates=0, n_below=5) == "not_needed"
    assert core.binding_addback_form(n_candidates=2, n_below=5) == "abs_of_sum"
    assert core.binding_addback_form(n_candidates=2, n_below=0) == "unresolved"


# ---- B4 and B5: engine-like and sign ------------------------------------------------------------
def test_engine_like_generators_and_the_generation_sign_are_not_subtracted():
    assert core.engine_like(corr=0.7, n_overlap=1000, generator_type="PV")
    assert not core.engine_like(corr=0.1, n_overlap=1000, generator_type="CHP")
    assert core.engine_like(corr=float("nan"), n_overlap=40, generator_type="CHP")
    assert not core.engine_like(corr=float("nan"), n_overlap=40, generator_type="PV")
    assert core.generation_sign_ok(corr=-0.2)
    assert not core.generation_sign_ok(corr=0.2)
    assert not core.generation_sign_ok(corr=float("nan"))

    def sub(
        *,
        cls: core.GeneratorClass = "outside_or_added_back",
        engine_like: bool = False,
        sign_ok: bool = True,
    ) -> bool:
        return core.subtract_generator(
            cls=cls,
            units="MW",
            coverage=0.95,
            is_engine=False,
            engine_like=engine_like,
            sign_ok=sign_ok,
        )

    assert sub()
    assert not sub(engine_like=True)
    assert not sub(sign_ok=False)


# ---- B6: holdout counts per input ---------------------------------------------------------------
def test_the_holdout_counts_separate_rows_read_from_rows_dropped(monkeypatch: pytest.MonkeyPatch):
    times = [
        datetime(2026, 6, 30, 23, 30, tzinfo=UTC),
        datetime(2026, 7, 1, 0, 30, tzinfo=UTC),
        datetime(2026, 7, 1, 1, 0, tzinfo=UTC),
        datetime(2026, 6, 30, 23, 0, tzinfo=UTC),
    ]
    frame = pl.DataFrame(
        {"time_series_id": [1, 1, 1, 2], "time": times, "power": [1.0, 2.0, 3.0, 4.0]}
    ).lazy()
    monkeypatch.setattr(data, "scan_power", lambda: frame)
    log = guard.ReadLog()
    out = data.load_power(ids=[1], log=log, name="p")
    assert np.isfinite(out[1]).sum() == 1
    assert log.holdout["p"] == {"read_on_or_after_holdout": 0, "dropped_by_filter": 2}
    assert log.holdout_clean()
    log.record_holdout(name="bad", read_after=3, dropped=0)
    assert not log.holdout_clean()


# ---- B7: the dominance band comes from the register table ---------------------------------------
def test_the_dominance_band_is_unknown_without_the_register_table_and_computed_with_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(data, "ML_EXPERIMENTS_DATA_DIR", tmp_path)
    assert data.register_pv_capacity_mw(primary="D12") is None
    folder = tmp_path / "00716-x"
    folder.mkdir()
    pl.DataFrame({"primary": ["D12", "D3"], "capacity_mw": [12.0, 5.0]}).write_parquet(
        folder / "pv_register_capacity.parquet"
    )
    total = data.register_pv_capacity_mw(primary="D12")
    assert total == 12.0
    assert core.dominance_band(largest_farm_mw=5.0, total_pv_mw=total) == "a_third_or_more"
    assert core.dominance_band(largest_farm_mw=3.0, total_pv_mw=total) == "below_a_third"
    assert core.dominance_band(largest_farm_mw=3.0, total_pv_mw=None) == "unknown"


# ---- B10 and B12: atomic writes and a run log that survives a crash -----------------------------
def test_a_failed_write_leaves_no_file_under_the_real_name(tmp_path: Path):
    def boom(tmp: Path) -> None:
        tmp.write_text("half")
        msg = "crash"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError):
        guard.atomic_write(path=tmp_path / "x.csv", write=boom)
    assert not (tmp_path / "x.csv").exists()
    guard.atomic_write(path=tmp_path / "x.csv", write=lambda tmp: tmp.write_text("ok"))
    assert (tmp_path / "x.csv").read_text() == "ok"
    assert not list(tmp_path.glob("*.tmp"))


def _crash(*, inputs: list[Path], info: dict[str, object], seen: Path, out_dir: Path) -> None:
    with guard.run_log(
        job="t",
        folder=EXPERIMENT_DIR,
        gate=guard.TimingGate(),
        inputs=inputs,
        info=info,
        out_dir=out_dir,
    ):
        inputs.append(seen)
        info["power_table_delta_version"] = "7"
        msg = "crash"
        raise RuntimeError(msg)


def test_the_run_log_is_written_after_a_crash_with_what_was_read(tmp_path: Path):
    seen = tmp_path / "seen.csv"
    seen.write_text("a\n1\n")
    inputs: list[Path] = []
    info: dict[str, object] = {}
    with pytest.raises(RuntimeError):
        _crash(inputs=inputs, info=info, seen=seen, out_dir=tmp_path)
    text = (tmp_path / "run_log_t.json").read_text()
    assert '"completed": false' in text
    assert '"power_table_delta_version": "7"' in text
    assert "seen.csv" in text


def test_the_code_hash_covers_the_launch_wrapper(tmp_path: Path):
    (tmp_path / "e00722_a.py").write_text("x = 1\n")
    before = guard.code_hash(folder=tmp_path)
    (tmp_path / "run.sh").write_text("echo hi\n")
    assert guard.code_hash(folder=tmp_path) != before


# ---- B11: the launch wrapper takes one cpu ------------------------------------------------------
@pytest.mark.parametrize("cpus", ["0-7", "0,1", "a"])
def test_the_launch_wrapper_refuses_more_than_one_cpu(cpus: str):
    env = {"PATH": "/usr/bin:/bin", "DATA_PATH_INTERNAL": "x", "SLOT_CPU": cpus}
    done = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "run.sh"), "--go", "e00722_u0.py"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 64
    assert "exactly one cpu" in done.stderr


# ---- B2, B3, B4, B5: job 1's regression stage on a synthetic world ------------------------------
def _world(
    n: int = core.N_GRID,
) -> tuple[
    Any, Any, dict[str, np.ndarray], dict[str, np.ndarray], dict[int, np.ndarray], np.ndarray
]:
    from e00722_data import EngineFrame, Topology

    rng = np.random.default_rng(35)
    sky_change = rng.normal(0, 1, n)
    gen_out = np.cumsum(rng.normal(0, 0.3, n))  # outside generator: moves the BSP only
    gen_in = np.cumsum(rng.normal(0, 0.3, n))  # embedded generator: lowers one Primary's demand
    d1 = np.cumsum(0.5 * sky_change + rng.normal(0, 0.1, n)) - gen_in
    d2 = np.cumsum(rng.normal(0, 0.3, n))
    bsp = d1 + d2 - gen_out
    prim = pl.DataFrame(
        {
            "label": ["D1", "D2"],
            "time_series_id": [1, 2],
            "capacity_mw": [10.0, 10.0],
            "units": ["MW", "MW"],
            "bsp": ["A", "A"],
        }
    )
    gens = pl.DataFrame(
        {
            "time_series_id": [11, 12],
            "type": ["PV", "Wind"],
            "units": ["MW", "MW"],
            "bsp": ["A", "A"],
            "substation_number": [1, 2],
        }
    )
    topology = Topology(primaries=prim, generators=gens, bsp_meter={"A": 99})
    eng = EngineFrame(
        has_row=np.ones(n, dtype=bool),
        k_path=np.zeros(n),
        era=np.full(n, "net", dtype=object),
        change=np.zeros(n, dtype=bool),
        key=np.arange(n) % 48,
        frame_key=np.arange(n) % 48,
        in_drop=np.zeros(n, dtype=bool),
        n_after_holdout=0,
    )
    return topology, eng, {"D1": d1, "D2": d2}, {"A": bsp}, {11: gen_in, 12: gen_out}, sky_change


def test_the_addback_stage_classes_sign_engine_and_joint_columns():
    topology, eng, series, bsp_series, gens, sky_change = _world()
    record = np.full(core.N_GRID, np.nan)
    classes = u0.addback_stage(
        topology=topology,
        eng=eng,
        series=series,
        bsp_series=bsp_series,
        gens=gens,
        record=record,
        du_of={"D1": sky_change, "D2": sky_change},
        engine_id=None,
        labels={11: "GA1", 12: "GA2"},
        n_days=core.N_GRID // 48 + 1,
        gate=guard.TimingGate(),
    )
    needed = common.CLASS_COLUMNS | {
        "joint_coef",
        "joint_lo",
        "joint_hi",
        "joint_label",
        "joint_generator_class",
        "class_differs_joint",
        "sign_corr",
        "engine_corr",
    }
    assert needed <= set(classes.columns)
    by = {
        r["generator"]: r for r in classes.filter(pl.col("primary") == "D1").iter_rows(named=True)
    }
    assert by["GA1"]["label"] == "embedded_not_added_back"
    assert by["GA2"]["label"] == "added_back_or_not_below"
    assert by["GA1"]["generator_class"] == "embedded_not_added_back"
    assert by["GA2"]["generator_class"] == "outside_or_added_back"
    assert by["GA2"]["sign_ok"]
    assert by["GA2"]["subtract"]
    assert not by["GA1"]["subtract"]
    assert not by["GA2"]["engine_like"]

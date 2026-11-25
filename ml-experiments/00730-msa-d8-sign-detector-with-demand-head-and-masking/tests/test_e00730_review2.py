"""Tests of the fixes that the second Opus code review asked for (pure Python, no torch)."""

import ast
import copy
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import e00729_banks as banks729
import e00729_score as score729
import e00729_sim as sim
import e00729_train as train729
import e00730_banks as banks
import e00730_cli as cli
import e00730_core as core
import e00730_figures as figures
import e00730_leakage as leak
import e00730_paths as paths
import e00730_pretrain as pretrain
import e00730_score as score
import e00730_stages as stages
import numpy as np
import pytest
from fixture730 import make_root, write_choices, write_predictions, write_real
from test_e00730_stages import fit_invented_baselines

HERE = Path(__file__).resolve().parent
CODE = HERE.parent


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    return make_root(tmp_path, monkeypatch)


# --------------------------------------------------------------------------- M2


def other_experiment_modules() -> list[str]:
    """Return every module of another experiment that this experiment's code imports."""
    found: set[str] = set()
    for path in sorted(CODE.glob("e00730_*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            found |= {n.split(".")[0] for n in names if n.startswith("e00") and "e00730" not in n}
    return sorted(found)


def test_m2_the_sibling_prefixes_hold_the_folders_of_the_545_chain() -> None:
    for prefix in ("00377-", "00381-", "00391-", "00400-", "00402-", "00410-"):
        assert prefix in paths.SIBLING_PREFIXES
        assert any(paths.ROOT.glob(f"{prefix}*"))


@pytest.mark.parametrize("module", other_experiment_modules())
def test_m2_every_module_from_another_experiment_imports(module: str) -> None:
    paths.install()
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "mlflow"}:
            pytest.skip(f"{exc.name} is not installed")
        raise


# --------------------------------------------------------------------------- M3


def test_m3_the_holiday_column_is_found_by_name() -> None:
    import e00428_data as data428

    _, names, _ = data428.non_lag_layout()
    column = pretrain.holiday_raw_column()
    assert names[column] == "nearest_days"
    assert names[column + 1] == "nearest_inside"
    in_table = column - pretrain.DATETIME_SLICE.start + core.ENCODER_WEATHER_COLUMNS
    assert in_table == leak.HOLIDAY_COLUMN


def test_m3_the_holiday_check_needs_a_finite_clipped_column_that_reaches_the_clip() -> None:
    leak.assert_holiday_clip(nearest_days=np.array([-1.0, 0.25, 1.0]))
    for bad, match in (
        (np.array([1.0, np.nan]), "non-finite"),
        (np.array([1.0, 1.2]), "clipped"),
        (np.array([0.2, 0.7]), "never reaches"),
    ):
        with pytest.raises(leak.CorrectnessFailureError, match=match):
            leak.assert_holiday_clip(nearest_days=bad)


# --------------------------------------------------------------------------- M4


def test_m4_a_correctness_failure_inside_a_fit_is_not_a_failed_fit(world: tuple) -> None:
    root, _, _ = world

    def run_one(arm: str, arch: str, seed: int) -> dict:
        msg = "a leakage assertion failed"
        raise leak.CorrectnessFailureError(msg)

    with pytest.raises(leak.CorrectnessFailureError):
        stages.run_jobs(root=root, jobs=[("S", "bilstm", 0)], run_one=run_one, log=lambda _: None)
    assert not list(stages.fit_dir(root=root).glob("*.failed.json"))


def test_m4_the_invalid_record_is_write_once_and_closes_the_fit_stages(world: tuple) -> None:
    root, _, _ = world
    assert not banks.invalid_recorded(root=root)
    stages.write_invalid_verdict(root=root, rule="first rule")
    stages.write_invalid_verdict(root=root, rule="second rule")
    assert banks.read_json(path=root / banks.INVALID_FILE) == {"rule": "first rule"}
    for stage in banks.GALLERY_GATED_STAGES:
        with pytest.raises(RuntimeError, match="correctness check"):
            banks.guard_run(root=root, stage=stage)
    banks.guard_run(root=root, stage="score")


def test_m4_a_later_score_cannot_overwrite_a_row_one_verdict(world: tuple) -> None:
    root, holidays, _ = world
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    write_real(root)
    stages.write_invalid_verdict(root=root, rule="a check failed")
    verdict = stages.stage_score(root=root, holidays=holidays, n_resamples=50, log=lambda _: None)
    assert verdict["row"] == 1
    assert banks.read_json(path=root / "scores" / "verdict.json")["row"] == 1


def test_m4_a_failed_fit_record_of_a_correctness_failure_makes_the_run_invalid(
    world: tuple,
) -> None:
    root, _, _ = world
    banks.write_json(
        path=stages.fit_dir(root=root) / "S_bilstm_s0.failed.json",
        payload={"error": "ValueError"},
    )
    assert not stages.invalid_run(root=root)
    banks.write_json(
        path=stages.fit_dir(root=root) / "S_bilstm_s1.failed.json",
        payload={"error": "CorrectnessFailureError"},
    )
    assert stages.invalid_run(root=root)


# --------------------------------------------------------------------------- M5


def make_engine_bank(
    *, n: int, seed: int, silence: bool, holidays: np.ndarray, n_days: int = 33
) -> SimpleNamespace:
    """Return a bank of `n` engine segments drawn by experiment 729's `draw_engine`."""
    rng = np.random.default_rng(seed)
    segments = []
    for _ in range(n):
        start = np.datetime64("2023-01-01", "D") + np.timedelta64(int(rng.integers(0, 700)), "D")
        dates = start + np.arange(n_days).astype("timedelta64[D]")
        weekday = (dates.astype(int) + 3) % 7
        quiet = (weekday >= 5) | np.isin(dates, holidays)
        params: dict[str, float] = {}
        engine = np.zeros((n_days, 48))
        engine[1:-1] = sim.draw_engine(
            rng=rng, utc_offset=np.zeros(n_days - 2), quiet_day=quiet[1:-1], record=params
        )
        if silence:
            engine[:] = 0.0
        base = np.full((n_days, 48), 10.0)
        segments.append(
            SimpleNamespace(
                world="engine", cal_index=np.r_[-1, np.arange(n_days - 2), -1], dates=dates,
                params=params, engine=engine, valid=np.ones((n_days, 48), bool),
                x=base - engine, base=base,
            )
        )  # fmt: skip
    return SimpleNamespace(kind="test", segments=segments)


def engine_keys_short(*, bank: SimpleNamespace, holidays: np.ndarray) -> list[str]:
    counter = banks.engine_output_shares(bank=bank, cal=SimpleNamespace(holidays=holidays))
    return [
        k
        for k, v in counter.shares().items()
        if k.startswith(("weekday", "quiet")) and v < core.MIN_OUTPUT_SHARE
    ]


def test_m5_a_healthy_engine_bank_passes_and_a_silenced_one_fails() -> None:
    holidays = np.datetime64("2023-01-01", "D") + np.array(
        [100, 101, 400, 401, 402], "timedelta64[D]"
    )
    for seed in range(8):
        healthy = make_engine_bank(n=240, seed=seed, silence=False, holidays=holidays)
        assert engine_keys_short(bank=healthy, holidays=holidays) == [], seed
    silent = make_engine_bank(n=240, seed=0, silence=True, holidays=holidays)
    short = engine_keys_short(bank=silent, holidays=holidays)
    assert any(k.startswith("weekday") for k in short)
    assert any(k.startswith("quiet") for k in short)


def test_m5_the_day_chance_uses_the_drawn_block_probabilities_and_quiet_days() -> None:
    params = {
        "no_run_share": 0.5, "evening_prob": 0.8, "morning_prob": 0.5, "midday_prob": 0.0,
        "weekend_factor": 0.5,
    }  # fmt: skip
    chance = banks.engine_day_chance(params=params, quiet=np.array([False, True]))
    assert chance[0] == pytest.approx(0.5 * (1.0 - 0.2 * 0.5))
    assert chance[1] == pytest.approx(0.5 * (1.0 - 0.6 * 0.75))
    assert chance[1] < chance[0]


def test_m5_bank_holidays_are_read_as_quiet_days() -> None:
    holidays = np.datetime64("2023-01-01", "D") + np.arange(0, 800).astype("timedelta64[D]")
    healthy = make_engine_bank(n=240, seed=3, silence=False, holidays=holidays)
    assert engine_keys_short(bank=healthy, holidays=holidays) == []
    # the same bank read with no holidays would count every quiet day as a weekday
    none = np.array([], dtype="datetime64[D]")
    assert engine_keys_short(bank=healthy, holidays=none) != []


# --------------------------------------------------------------------------- M6, S1, S2


def import_smoke():
    import smoke_real_files

    return smoke_real_files


def test_m6_the_smoke_bank_check_reads_this_experiments_data_folder(
    world: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = world
    smoke = import_smoke()
    monkeypatch.setattr(paths, "data_root", lambda: tmp_path / "nothing_here")
    assert smoke.check_bank_s2() == "test bank present: False"
    monkeypatch.setattr(paths, "data_root", lambda: root)
    line = smoke.check_bank_s2()
    assert "test bank present: True" in line
    assert "False" not in line


def test_s1_the_smoke_weather_check_reads_each_weather_column() -> None:
    smoke = import_smoke()
    names = ["shortwave", "longwave", "t2m"]
    table = np.random.default_rng(0).normal(size=(3, 48, 6))
    table[:, :, 5] = 1.0  # a constant datetime column is not read
    column, varying = smoke.weather_flags(table=table, names=names)
    assert column == 0
    assert varying == [True, True, True]
    table[:, :, 1] = 4.0
    assert smoke.weather_flags(table=table, names=names)[1] == [True, False, True]
    with pytest.raises(ValueError, match="shortwave"):
        smoke.weather_flags(table=table, names=["a", "b", "c"])


def test_s2_the_smoke_counts_clear_days_by_split_and_season(world: tuple) -> None:
    _, _, cal = world
    smoke = import_smoke()
    used = {"training": np.ones(len(cal.dates), dtype=bool), "test": np.zeros(len(cal.dates), bool)}
    counts = smoke.clear_day_counts(cal=cal, used=used)
    assert sum(v for k, v in counts.items() if k.startswith("training")) > 0
    assert sum(v for k, v in counts.items() if k.startswith("test")) == 0


# --------------------------------------------------------------------------- S3, S4


def test_s3_the_fixture_has_missing_readings_a_missing_farm_day_and_a_varying_index(
    world: tuple,
) -> None:
    cal = world[2]
    assert np.isnan(cal.reading).sum() == 2 * len(cal.dates)
    assert not cal.farms.power_ok.all()
    inside = cal.csi[0][np.isfinite(cal.csi[0])]
    assert np.ptp(inside) > 0.0
    test = banks729.load_bank(path=banks.bank_path(root=world[0], kind="test"))
    assert any((~s.valid[s.cal_index >= 0]).any() for s in test.segments)


def test_s4_a_nan_reading_is_not_counted_as_not_near_zero() -> None:
    low = core.DECIDABLE_READING_POINTS / 2
    assert stages.share_near_zero(reading=np.array([np.nan, low, 100.0])) == pytest.approx(0.5)
    assert np.isnan(stages.share_near_zero(reading=np.array([np.nan, np.nan])))
    assert np.isnan(stages.share_near_zero(reading=np.array([])))


def test_s5_g8_draws_the_share_by_utc_hour_and_a_world_without_readings_fails(world: tuple) -> None:
    root, _, _ = world
    day = np.full((3, 48), 100.0)
    day[:, 4:6] = 0.0
    by_hour = stages.near_zero_by_hour(reading_days=day)
    assert by_hour.shape == (24,)
    assert by_hour[2] == pytest.approx(1.0)
    assert by_hour[3] == pytest.approx(0.0)
    out = figures.GalleryResult()
    figures.aggregate_gallery(
        shares_synthetic={"none": 0.1, "engine": 0.2}, shares_real={"none": 0.1, "engine": 0.3},
        component_shares={"a": 1.0}, out=out, hourly_synthetic=by_hour, hourly_real=by_hour,
    )  # fmt: skip
    assert out.results["G8_realism"]
    assert "vconcat" in out.public["G8_realism"].to_dict()
    empty = figures.GalleryResult()
    figures.aggregate_gallery(
        shares_synthetic={"none": float("nan")}, shares_real={"none": 0.1},
        component_shares={"a": 1.0}, out=empty,
    )  # fmt: skip
    assert not empty.results["G8_realism"]
    del root


# --------------------------------------------------------------------------- S6, S7


def test_s6_a_failed_fit_counts_only_on_an_architecture_that_enters_a_contrast(
    world: tuple,
) -> None:
    root, _, _ = world
    folder = stages.fit_dir(root=root)
    for name in (
        "S_unet_s0", "S_gru_transformer_s1", "SD-K_unet_s0", "D_bilstm_s0", "SD-W_transformer_s0"
    ):  # fmt: skip
        banks.write_json(path=folder / f"{name}.failed.json", payload={"error": "ValueError"})
    names = stages.failed_arm_names(root=root, arch="transformer", variant="transformer")
    assert names == {"D", "SD-W"}
    banks.write_json(path=folder / "S_bilstm_s0.failed.json", payload={"error": "ValueError"})
    names = stages.failed_arm_names(root=root, arch="transformer", variant="transformer")
    assert "S" in names


def test_s7_the_projection_adds_the_measured_causal_read_out() -> None:
    secs = dict.fromkeys(core.ARCHITECTURES, 1.0)
    base = core.projected_hours(seconds_per_epoch=secs, chosen=None)
    slow = core.projected_hours(seconds_per_epoch=secs, chosen=None, causal_seconds_per_call=600.0)
    assert slow - base == pytest.approx(
        core.CAUSAL_READOUTS * 600.0 / 3600.0 - core.CAUSAL_ESTIMATE_HOURS
    )
    report = core.timing_gate_report(
        seconds_per_epoch=secs, peak_memory_gb=1.0, causal_seconds_per_call=600.0
    )
    assert report["hours_at_cap"] == pytest.approx(slow)
    assert "causal_seconds_per_call" in inspect.signature(stages.stage_timing).parameters


def test_s7_a_timing_rerun_starts_from_an_empty_scratch_folder(tmp_path: Path) -> None:
    scratch = tmp_path / "timing_scratch" / "pretrained"
    scratch.mkdir(parents=True)
    (scratch / "old_seed.pt").write_text("x")
    folder = cli.fresh_scratch(root=tmp_path)
    assert folder == tmp_path / "timing_scratch"
    assert not folder.exists()


# --------------------------------------------------------------------------- S9


class FlatClimatology:
    """A climatology whose single cell holds the demands 10 and 20 (median 15)."""

    cells = {0: np.array([10.0, 20.0])}  # noqa: RUF012

    def cell_id(self, *, month: np.ndarray, hh: np.ndarray, weekend: np.ndarray) -> np.ndarray:
        return np.zeros(len(hh), dtype=int)


def test_s9_figure_6_gets_the_validation_demand_climatology(world: tuple) -> None:
    root, holidays, _ = world
    write_choices(root)
    val = banks729.load_bank(path=banks.bank_path(root=root, kind="validation"))
    rows = score729.rows_from_segments(segments=val.segments, holidays=holidays)
    rng = np.random.default_rng(0)
    base = rng.uniform(5.0, 25.0, rows.n)
    hidden = np.arange(rows.n) % 3 == 0
    for style in core.STYLES:
        folder = root / "predictions" / stages.job_name(arm=f"SD-{style}", arch="bilstm", seed=0)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"validation_masked_{style}.npz",
            demand_mu=base, base=base, hidden=hidden, p=np.zeros(rows.n),
        )  # fmt: skip
    out = stages.demand_climatology_error(
        root=root, arch="bilstm", holidays=holidays, climatology=FlatClimatology()
    )
    assert out == pytest.approx(float(np.abs(15.0 - base)[hidden].mean()))
    assert np.isnan(
        stages.demand_climatology_error(
            root=root, arch="unet", holidays=holidays, climatology=FlatClimatology()
        )
    )


def test_s9_r3_reads_sd_star_c_in_the_causal_protocol_on_the_real_eras(world: tuple) -> None:
    root, holidays, _ = world
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    real_rows = dict.fromkeys(("r1", "k1", "k2", "eras"), rows)
    era = np.array(["net", "solar"])[np.arange(rows.n) % 2]
    out = stages.stage_real(
        root=root, real_rows=real_rows, era_of_row=era,
        baselines=fit_invented_baselines(rows), log=lambda _: None,
    )  # fmt: skip
    assert set(out["r3"]["causal"]) == {"share_p_above_half_by_era", "share_by_era_and_hour"}
    (stages.fit_dir(root=root).parent / "predictions").mkdir(exist_ok=True)
    for seed in core.ARCH_SEEDS:
        job = stages.job_name(arm="SD*-C", arch="bilstm", seed=seed)
        (root / "predictions" / job / stages.REAL_CAUSAL_FILE).unlink()
    out = stages.stage_real(
        root=root, real_rows=real_rows, era_of_row=era,
        baselines=fit_invented_baselines(rows), log=lambda _: None,
    )  # fmt: skip
    assert "causal" not in out["r3"]


def test_s9_figure_8_is_drawn_from_stored_attention_maps(world: tuple) -> None:
    root, holidays, _ = world
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    write_real(root)
    stages.stage_score(root=root, holidays=holidays, n_resamples=50, log=lambda _: None)
    assert stages.attention_maps_stored(root=root) is None
    job = stages.job_name(arm="SD-K", arch="transformer", seed=0)
    rng = np.random.default_rng(0)
    maps = {g: rng.dirichlet(np.ones(96), size=48) for g in ("engine_reversal", "import_only")}
    train729.save_predictions(arrays=maps, path=root / "predictions" / job / stages.ATTENTION_FILE)
    stored = stages.attention_maps_stored(root=root)
    assert stored is not None
    assert set(stored) == set(maps)
    written = stages.stage_figures(root=root, log=lambda _: None)
    assert {p.name for p in written} >= {
        "attention_engine_reversal.svg",
        "attention_import_only.svg",
    }


# --------------------------------------------------------------------------- N1, N6


def test_n1_the_dead_code_is_gone() -> None:
    assert not hasattr(score, "CAUSAL_PROTOCOL_ROWS_FROM_K")
    assert "def timing_gate(" not in (CODE / "e00730_train.py").read_text()


def test_n6_a_bank_whose_truth_does_not_subtract_the_solar_output_fails(
    world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    cal = world[2]
    test = banks729.load_bank(path=banks.bank_path(root=world[0], kind="test"))
    monkeypatch.setattr(banks, "MIN_CASES", 1)
    ok = banks.component_checks(banks_by_name={"test": test}, cal=cal)
    assert not [k for k in ok["short"] if "solar_subtract" in k], ok["short"]
    broken = copy.deepcopy(test)
    for seg in broken.segments:
        if seg.world in ("solar", "both"):
            seg.x = seg.base - seg.engine  # the solar output is no longer subtracted
    out = banks.component_checks(banks_by_name={"test": broken}, cal=cal)
    assert any("solar_subtract" in k for k in out["short"])


# ---------------------------------------------------------------------- real-file smoke findings


def test_the_d8_encoder_table_ignores_missing_weather_that_no_base_day_reads() -> None:
    import e00087_common as common
    import e00418_prep as prep418

    grid = common.grid_times()
    weather = np.random.default_rng(0).normal(size=(grid.len(), len(prep418.WEATHER)))
    weather = weather.astype(np.float32)
    weather[0] = np.nan  # the dense weather is missing at the grid's own edges
    weather[-1] = np.nan
    derived = common.derived_columns(valid_time=grid, latitude=54.0, longitude=-1.5)
    day_start = np.array([48 * 400, 48 * 401, 48 * 402])
    table = pretrain.raw_table_on_days(weather=weather, derived=derived, day_start=day_start)
    assert table.shape == (3, 48, core.ENCODER_INPUT_COLUMNS)
    assert np.isfinite(table).all()
    weather[48 * 401 + 5] = np.nan  # a missing value inside a base day is still refused
    with pytest.raises(ValueError, match="not finite"):
        pretrain.raw_table_on_days(weather=weather, derived=derived, day_start=day_start)


def test_a_season_without_a_clear_base_day_is_not_expected_but_a_silent_one_still_fails(
    world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    cal = world[2]
    bank = banks729.load_bank(path=banks.bank_path(root=world[0], kind="test"))
    monkeypatch.setattr(banks, "MIN_CASES", 1)
    keys = banks.expected_solar_keys(bank=bank)
    assert keys
    assert {k.split("/")[2] for k in keys if not k.startswith("solar_subtract")} == {"S1", "S2"}
    cloudy = copy.deepcopy(bank)
    for seg in cloudy.segments:
        if seg.source == "S1":
            seg.csi[:] = 0.1  # a month of the real record with no clear sky
    assert not [k for k in banks.expected_solar_keys(bank=cloudy) if "/S1/" in k]
    assert [k for k in banks.expected_solar_keys(bank=cloudy) if "/S2/" in k]
    out = banks.component_checks(banks_by_name={"test": cloudy}, cal=cal)
    assert not [k for k in out["short"] if "/solar/" in k], out["short"]
    silent = copy.deepcopy(bank)
    for seg in silent.segments:
        seg.solar[:] = 0.0
    short = banks.component_checks(banks_by_name={"test": silent}, cal=cal)["short"]
    assert any("/solar/" in k for k in short)


def test_the_minimum_cases_bar_applies_to_the_pool_of_a_kind_and_source() -> None:
    counts = {
        "solar/test/S2/winter": 12, "solar/test/S2/spring": 3, "both/test/S2/summer": 6,
        "solar/test/S1/winter": 2, "solar_subtract/solar/test/S2/winter": 1,
    }  # fmt: skip
    pooled = banks.pooled_solar_cases(counts=counts)
    assert pooled == {"test/S2": 21, "test/S1": 2}


def zero_metered_world(world: tuple) -> tuple:
    """Return the test bank and a calendar where one reporting farm is metered at zero on a day.

    Every S1 segment is also given zero solar output on that calendar day, as experiment 729's S1
    draw would give it.
    """
    cal = copy.deepcopy(world[2])
    bank = banks729.load_bank(path=banks.bank_path(root=world[0], kind="test"))
    seg = next(s for s in bank.segments if s.world in ("solar", "both") and s.source == "S1")
    _, clear_day = banks.clear_sun_up(csi=seg.csi, cosz=seg.cosz)
    day = int(np.flatnonzero((seg.cal_index >= 0) & clear_day & ~seg.s1_fallback)[0])
    cal_day = int(seg.cal_index[day])
    cal.farms.power_ok[1, cal_day] = True
    cal.farms.power[1, cal_day] = 0.0
    for s in bank.segments:
        if s.world in ("solar", "both") and s.source == "S1":
            for d in np.flatnonzero(s.cal_index == cal_day):
                s.solar[d] = 0.0
    return bank, cal, cal_day


def test_s1_days_metered_at_zero_in_the_real_record_are_excluded_and_bounded(
    world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(banks, "MIN_CASES", 1)
    bank, cal, cal_day = zero_metered_world(world)
    assert banks.metered_zero_days(cal=cal)[cal_day]
    monkeypatch.setattr(banks, "METERED_ZERO_MAX_SHARE", 1.0)  # the fixture has few clear days
    out = banks.component_checks(banks_by_name={"test": bank}, cal=cal)
    assert not [k for k in out["short"] if "/solar/" in k], out["short"]
    assert out["shares"]["test/solar_metered_zero"]["test/S1"] > 0.0
    monkeypatch.setattr(banks, "METERED_ZERO_MAX_SHARE", 0.0)
    bounded = banks.component_checks(banks_by_name={"test": bank}, cal=cal)
    assert any("metered_zero_share" in k for k in bounded["short"])


def test_a_silent_s1_generator_still_fails_although_some_real_days_are_metered_at_zero(
    world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(banks, "MIN_CASES", 1)
    bank, cal, _ = zero_metered_world(world)
    monkeypatch.setattr(banks, "METERED_ZERO_MAX_SHARE", 1.0)
    for seg in bank.segments:
        if seg.source == "S1":
            seg.solar[:] = 0.0
    out = banks.component_checks(banks_by_name={"test": bank}, cal=cal)
    assert any("/solar/" in k and "S1" in k for k in out["short"])


def test_the_stop_line_is_16_gpu_process_hours() -> None:
    assert core.STOP_LINE_HOURS == 16.0
    secs = dict.fromkeys(core.ARCHITECTURES, 1.0)
    report = core.timing_gate_report(seconds_per_epoch=secs, peak_memory_gb=1.0)
    assert report["stop_line_hours"] == 16.0
    assert report["within_stop_line"]


def test_the_engine_case_bar_applies_to_the_subtract_cases_pooled_over_seasons() -> None:
    counts = {
        "subtract/engine/test/summer": 9, "subtract/engine/test/winter": 60,
        "subtract/both/test/summer": 4, "weekday/engine/test/summer": 1,
        "quiet/engine/test/summer": 1,
    }  # fmt: skip
    assert banks.pooled_engine_cases(counts=counts) == {"engine/test": 69, "both/test": 4}


def test_a_small_engine_season_passes_a_pooled_short_bank_fails_and_a_silent_one_fails(
    world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, cal = world[2], world[2]
    del bank
    test = banks729.load_bank(path=banks.bank_path(root=world[0], kind="test"))
    counter = banks.engine_output_shares(bank=test, cal=cal)
    pooled = banks.pooled_engine_cases(counts=counter.cases())
    assert pooled
    monkeypatch.setattr(banks, "MIN_CASES", min(pooled.values()))
    ok = banks.component_checks(banks_by_name={"test": test}, cal=cal)
    assert not [k for k in ok["short"] if "/engine/" in k], ok["short"]
    monkeypatch.setattr(banks, "MIN_CASES", max(pooled.values()) + 1)
    few = banks.component_checks(banks_by_name={"test": test}, cal=cal)
    assert any("/engine/pooled/" in k and k.endswith("/too_few") for k in few["short"])
    monkeypatch.setattr(banks, "MIN_CASES", 1)
    silent = copy.deepcopy(test)
    for seg in silent.segments:
        seg.engine[:] = 0.0
    out = banks.component_checks(banks_by_name={"test": silent}, cal=cal)
    assert any("/engine/" in k for k in out["short"])

"""Tests of the fixes that the first Opus code review asked for (pure Python, no torch)."""

import copy
import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import e00729_arms as arms729
import e00729_banks as banks729
import e00729_core as core729
import e00729_score as score729
import e00729_sim as sim
import e00730_arms as arms
import e00730_banks as banks
import e00730_core as core
import e00730_figures as figures
import e00730_leakage as leak
import e00730_masks as masks
import e00730_score as score
import e00730_stages as stages
import numpy as np
import pytest
from fixture730 import make_root, open_gallery, write_choices, write_predictions, write_real
from world730 import make_calendar

HERE = Path(__file__).resolve().parent
STATS = arms729.InputStats(10.0, 3.0, 5.0, 2.0)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    return make_root(tmp_path, monkeypatch)


def test_m1_run_sh_reads_slot_cpus_and_not_a_misspelling() -> None:
    text = (HERE.parent / "run.sh").read_text()
    assert "SLOT_CPUSS" not in text
    done = subprocess.run(
        ["bash", str(HERE.parent / "run.sh"), "--go", "x.py"],
        env={"PATH": "/usr/bin:/bin", "DATA_PATH_INTERNAL": "/nowhere", "SLOT_CPUS": "abc"},
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 64
    assert "unbound variable" not in done.stderr
    assert "cpu list" in done.stderr


def test_m6_bank_leakage_checks_pass_on_clean_banks_and_fail_on_a_broken_bsp(world: tuple) -> None:
    root, _, cal = world
    built = {
        k: banks729.load_bank(path=banks.bank_path(root=root, kind=k))
        for k in ("validation", "test", "shifted")
    }
    built["train_s0"] = banks729.load_bank(path=banks.bank_path(root=root, kind="train", seed=0))
    banks.bank_leakage_checks(built=built, cal=cal, train_seeds=[0])
    broken = copy.deepcopy(built)
    seg = next(s for s in broken["test"].segments if s.has_bsp and s.world != "none")
    seg.bsp = np.where(np.isfinite(seg.bsp), seg.bsp + 5.0, seg.bsp)
    with pytest.raises(leak.CorrectnessFailureError):
        banks.bank_leakage_checks(built=broken, cal=cal, train_seeds=[0])


def test_m6_a_failed_leakage_check_is_verdict_row_one(world: tuple) -> None:
    root, holidays, _ = world
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    write_real(root)
    path = root / "banks" / "summary.json"
    summary = banks.read_json(path=path)
    summary["leakage_pass"] = False
    banks.write_json(path=path, payload=summary)
    verdict = stages.stage_score(root=root, holidays=holidays, n_resamples=50, log=lambda _: None)
    assert verdict["row"] == 1
    stages.write_invalid_verdict(root=root)
    assert banks.read_json(path=root / "scores" / "verdict.json")["row"] == 1
    assert "no score was read" in stages.stage_report(root=root)


def test_m6_limits_written_after_a_fit_make_row_one(world: tuple) -> None:
    root, holidays, _ = world
    banks.write_json(path=stages.fit_dir(root=root) / "S_bilstm_s0.json", payload={"epochs": 1})
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    write_real(root)
    older = root / "limits.json"
    fit = stages.fit_dir(root=root) / "S_bilstm_s0.json"
    stamp = fit.stat().st_mtime_ns - 10_000_000_000
    import os

    os.utime(older, ns=(stamp + 20_000_000_000, stamp + 20_000_000_000))
    verdict = stages.stage_score(root=root, holidays=holidays, n_resamples=50, log=lambda _: None)
    assert verdict["row"] == 1


def test_m7_the_causal_cut_is_t_plus_k_and_two_sided_at_48() -> None:
    assert [score.causal_cut(t=10, k=k) for k in core.CAUSAL_K] == [10, 12, 16, 22, 34, None]
    assert score.causal_cut(t=47, k=2) == 47


def test_m7_c7_reads_the_causal_predictions_not_the_two_sided_fit(world: tuple) -> None:
    root, holidays, _ = world
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    write_real(root)
    for seed in core.ARCH_SEEDS:
        folder = root / "predictions" / stages.job_name(arm="SD*-C", arch="bilstm", seed=seed)
        np.savez(folder / "test_causal_k0.npz", p=np.zeros(rows.n))
    verdict = stages.stage_score(root=root, holidays=holidays, n_resamples=50, log=lambda _: None)
    two_sided = verdict["report_only"]["R10"]["point"]
    c7 = verdict["contrasts"]["C7"]["point"]
    assert c7 != pytest.approx(two_sided)
    assert verdict["log_loss_contrasts"]["C7"]["n_rows"] == rows.n


def test_m8_validation_cuts_are_the_fixed_seed_730_cuts_and_are_read_at_the_cut() -> None:
    cuts = masks.fixed_cuts(n_windows=5)
    assert cuts.shape == (5, core.CAUSAL_VALIDATION_CUTS)
    assert np.array_equal(cuts, masks.fixed_cuts(n_windows=5))
    values = np.arange(5 * 48).reshape(5, 48)
    read = score.read_at_cuts(values=values, cuts=cuts[:, 0])
    assert (read == values[np.arange(5), cuts[:, 0]]).all()


def test_m11_the_dsign_refit_finds_experiment_729s_spec() -> None:
    key = stages.ARM_729_KEY.get("D-sign", "D-sign")
    assert key == "Dsign"
    assert key in arms729.ARM_SPECS


def test_m12_only_needed_fits_make_row_two_and_a_rerun_clears_the_record(world: tuple) -> None:
    root, holidays, _ = world
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    write_real(root)
    failed = stages.fit_dir(root=root) / "SD*-lambda_bilstm_s0.failed.json".replace("*", "star")
    banks.write_json(path=failed, payload={"error": "X"})
    verdict = stages.stage_score(root=root, holidays=holidays, n_resamples=50, log=lambda _: None)
    assert verdict["row"] != 2
    failed.unlink()
    banks.write_json(
        path=stages.fit_dir(root=root) / "SDstar-C_bilstm_s0.failed.json", payload={"error": "X"}
    )
    verdict = stages.stage_score(root=root, holidays=holidays, n_resamples=50, log=lambda _: None)
    assert verdict["row"] != 2
    assert verdict["void_contrasts"] == ["C7"]
    assert "C7" not in verdict["contrasts"]
    # a successful rerun deletes the failed record
    record = stages.fit_dir(root=root) / "S_bilstm_s0.failed.json"
    banks.write_json(path=record, payload={"error": "X"})
    stages.run_jobs(root=root, jobs=[("S", "bilstm", 0)], run_one=lambda *_: {}, log=lambda _: None)
    assert not record.exists()


def test_m13_intervals_use_the_bonferroni_level_and_seed_730(world: tuple) -> None:
    root, holidays, _ = world
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=rows.n), rng.normal(size=rows.n)
    keep = np.ones(rows.n, dtype=bool)
    narrow = score.contrast_at_level(rows=rows, a=a, b=b, keep=keep, n_resamples=300, level=0.5)
    wide = score.contrast_at_level(rows=rows, a=a, b=b, keep=keep, n_resamples=300)
    assert (wide.reading.upper - wide.reading.lower) > (narrow.reading.upper - narrow.reading.lower)
    table = score729.cell_table(rows=rows, diff=a - b, keep=keep, months=None)
    same = score729.read_contrast(
        table=table, n_resamples=300, seed=core.GENERATOR_SEED, level=core.LEVEL
    )
    assert same.reading == wide.reading
    assert pytest.approx(1 - 0.05 / 7) == core.LEVEL
    assert core.GENERATOR_SEED == 730


def test_m14_score_needs_h2_and_real_comes_first(world: tuple) -> None:
    names = list(stages.stage_names())
    assert names.index("real") < names.index("score")
    assert names.index("choose-style") < names.index("later")
    assert names.index("margin-fits") < names.index("margin") < names.index("styles")
    assert names.index("timing") < names.index("pretrain")


def test_m15_arms_that_read_the_style_refuse_without_style_json(world: tuple) -> None:
    root, _, _ = world
    write_choices(root)
    (root / "style.json").unlink()
    banks.write_json(path=root / "scores" / "timing.json", payload={"pass": True})
    for arm in stages.STYLE_ARMS:
        with pytest.raises(RuntimeError, match="choose-style"):
            stages.fit_one(
                root=root, ctx=None, arm=arm, arch="bilstm", seed=0, device="cpu", log=print
            )  # ty: ignore[invalid-argument-type]
    with pytest.raises(RuntimeError, match="choose-style"):
        stages.stage_later(root=root, run_one=lambda *_: {}, log=print)


def test_s4_a_fit_needs_the_timing_gate(world: tuple) -> None:
    root, _, _ = world
    write_choices(root)
    with pytest.raises(RuntimeError, match="timing"):
        stages.fit_one(root=root, ctx=None, arm="S", arch="bilstm", seed=0, device="cpu", log=print)  # ty: ignore[invalid-argument-type]


def test_m16_c6_reads_the_transformer_variant_with_the_better_validation_score(
    world: tuple,
) -> None:
    root, _, _ = world
    open_gallery(root)
    write_choices(root)
    (root / "selection.json").unlink()
    for arch in core.ARCHITECTURES:
        for arm in core.SCREEN_ARMS:
            for seed in core.ARCH_SEEDS:
                val = 0.03 + (0.01 if (arch == "transformer" and arm == "S") else 0.0)
                banks.write_json(
                    path=stages.fit_dir(root=root)
                    / f"{stages.job_name(arm=arm, arch=arch, seed=seed)}.json",
                    payload={"arm": arm, "best_val_brier": val, "n_weights": 100},
                )
    record = stages.choose_architecture(root=root)
    assert record["transformer_variant"] == "gru_transformer"


def test_m17_pretraining_does_not_call_a_missing_function() -> None:
    import e00730_cli as cli

    text = inspect.getsource(cli.run_pretrain)
    assert "record_days()" not in text
    assert "ty: ignore" not in text


def test_m18_a_window_takes_its_source_from_its_segment(world: tuple) -> None:
    root, holidays, _ = world
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    for seg in test.segments:
        if seg.world in ("solar", "both"):
            seg.source = "S1"
    banks729.save_bank(bank=test, path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    built, _ = stages.series_inputs(root=root, holidays=holidays, real_rows=None)
    table: Any = built
    solar_windows = np.isin(table.world, ["solar", "both"])
    assert (table.source[solar_windows] == "S1").all()
    import e00730_series_figures as sf

    plan = sf.plan_windows(t=table, published=True)
    for idx in plan.values():
        assert idx is None or table.source[idx] != "S1"


def test_m18_public_gallery_panels_never_draw_s1_segments(world: tuple) -> None:
    root, _, _ = world
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    segments = [copy.deepcopy(s) for s in test.segments]
    for seg in segments:
        seg.source = "S1"
    out = figures.GalleryResult()
    figures.bsp_fold_base_gallery(segments=segments, out=out)
    figures.mask_gallery(segments=segments, out=out)
    assert "G4_bsp" not in out.public
    assert "G5_fold" not in out.public
    assert "G7_masks" not in out.public


def test_m19_the_gallery_gate_can_pass_with_real_shares(world: tuple) -> None:
    from e00729_data import ERA_NET, ERA_PRE, ERA_SOLAR

    reading = np.linspace(0.5, 30.0, 300).reshape(-1, 1) * np.ones((1, 48))
    era = np.array([ERA_PRE, ERA_NET, ERA_SOLAR] * 100)
    ctx = SimpleNamespace(table=SimpleNamespace(calendar=SimpleNamespace(reading=reading), era=era))
    shares = stages.real_gallery_shares(ctx=ctx)  # ty: ignore[invalid-argument-type]
    assert set(shares) == {"none", "engine", "both"}
    root, _, _ = world
    results = stages.stage_gallery(root=root, shares_real=shares, log=lambda _: None)
    assert results["G8_realism"]
    none = stages.stage_gallery(root=root, shares_real=None, log=lambda _: None)
    assert not none["G8_realism"]


def test_m20_a_causal_window_is_blind_to_observed_values_after_the_cut() -> None:
    cal = make_calendar()
    segment = sim.simulate_segment(
        cal=cal, month="2022-03", use=np.ones(len(cal.dates), dtype=bool), world="both",
        rng=np.random.default_rng(1),
    )  # fmt: skip
    spec = arms.arm_specs(style="K")["SD*-C"]
    base = arms.prepare_segments(segments=[segment], spec=spec, holidays=cal.holidays, stats=STATS)[
        0
    ]
    names = arms.stack_names(spec=spec)
    observed = [i for i, n in enumerate(names) if n in masks.OBSERVED_ROWS]
    rng = np.random.default_rng(5)
    for t in (0, 7, 23, 40):
        cut = core729.MIDDLE_START + t
        day = int(base.days[3])
        start = day * 48 - core729.MIDDLE_START
        w1, _, _ = arms.build_one(prepared=base, dense_day=day, spec=spec, cut_pos=cut)
        stack = base.stack.copy()
        for row in observed:
            stack[row, start + cut + 1 :] = rng.normal(size=stack.shape[1] - start - cut - 1) * 50
        hourly = base.hourly.copy()
        hour = (start + cut) // 2
        hourly[:3, hour + 1 :] = rng.normal(size=hourly[:3, hour + 1 :].shape) * 50
        hourly[3, hour:] = rng.normal(size=hourly[3, hour:].shape) * 50
        scrambled = arms.PreparedSegment(**{**base.__dict__, "stack": stack, "hourly": hourly})  # ty: ignore[invalid-argument-type]
        w2, _, _ = arms.build_one(prepared=scrambled, dense_day=day, spec=spec, cut_pos=cut)
        assert np.array_equal(w1, w2), f"window differs after a cut at {t}"
    # the test is not vacuous: values at or before the cut's hour do change the window
    hourly = base.hourly.copy()
    hourly[1, (start + cut) // 2 - 1] += 5.0
    changed = arms.PreparedSegment(**{**base.__dict__, "hourly": hourly})  # ty: ignore[invalid-argument-type]
    w3, _, _ = arms.build_one(prepared=changed, dense_day=day, spec=spec, cut_pos=cut)
    assert not np.array_equal(w1, w3)


def test_s1_a_cut_window_is_not_also_masked(prepared_style_w: tuple) -> None:
    spec, prep = prepared_style_w
    rng = np.random.default_rng(0)
    picks = [(0, int(prep[0].days[2]))] * 600
    x, _, _ = arms.make_batch(prepared=prep, picks=picks, spec=spec, rng=rng, training=True)
    names = arms.stack_names(spec=spec)
    future = x[:, names.index(arms.FUTURE_ROW)]
    cut = future.max(axis=1) > 0.5
    first_scored_mask = x[:, names.index("r_mask"), core729.MIDDLE_START]
    assert cut.mean() == pytest.approx(0.5, abs=0.07)
    assert (first_scored_mask[cut] > 0.5).all()  # a cut window is not also masked (style W)
    assert (first_scored_mask[~cut] < 0.5).mean() > 0.5  # the other windows carry the style


@pytest.fixture
def prepared_style_w() -> tuple:
    cal = make_calendar()
    segment = sim.simulate_segment(
        cal=cal, month="2022-03", use=np.ones(len(cal.dates), dtype=bool), world="both",
        rng=np.random.default_rng(1),
    )  # fmt: skip
    spec = arms.arm_specs(style="W")["SD*-C"]
    return spec, arms.prepare_segments(
        segments=[segment], spec=spec, holidays=cal.holidays, stats=STATS
    )


def test_s6_stale_style_and_timing_files_stop_a_new_banks_stage(tmp_path: Path) -> None:
    root = tmp_path / "exp"
    root.mkdir()
    (root / "style.json").write_text(json.dumps({"chosen": "K"}))
    with pytest.raises(RuntimeError, match="earlier run"):
        banks.guard_run(root=root, stage="banks")


def test_s5_a_ranges_override_changes_the_marker_and_is_applied(tmp_path: Path) -> None:
    root = tmp_path / "exp"
    root.mkdir()
    name = next(iter(core729.RANGES))
    before = banks.run_marker_payload()
    (root / banks.RANGES_OVERRIDE_FILE).write_text(json.dumps({name: [0.1, 0.2]}))
    try:
        banks.apply_ranges_override(root=root)
        assert core729.RANGES[name] == (0.1, 0.2)
        assert banks.run_marker_payload()["ranges"] != before["ranges"]
    finally:
        (root / banks.RANGES_OVERRIDE_FILE).unlink()
        banks.apply_ranges_override(root=root)
    assert banks.run_marker_payload() == before


def test_s7_the_encoder_table_reports_its_coverage_of_a_bank() -> None:
    cal = make_calendar()
    segment = sim.simulate_segment(
        cal=cal, month="2022-03", use=np.ones(len(cal.dates), dtype=bool), world="none",
        rng=np.random.default_rng(1),
    )  # fmt: skip
    base_dates = segment.dates[segment.cal_index >= 0]
    full = arms.EncoderTable(
        dates=base_dates, values=np.zeros((len(base_dates), 48, 28), np.float32)
    )
    assert full.coverage(segments=[segment]) == 1.0
    half = arms.EncoderTable(
        dates=base_dates[::2], values=np.zeros((len(base_dates[::2]), 48, 28), np.float32)
    )
    assert half.coverage(segments=[segment]) < core.MIN_OUTPUT_SHARE


def test_n1_the_dead_band_is_open_at_two_points() -> None:
    rows = SimpleNamespace(x=np.array([-2.0, 2.0, -2.5, 3.0]), n=4)
    p = np.array([0.0, 0.0, 1.0, 0.0])
    out = score.calibration(rows=rows, p=p, bar=1.0)  # ty: ignore[invalid-argument-type]
    assert float(np.sum(out["count"])) == 2.0


def test_n5_refitted_arms_are_scored_on_the_head_own_probability() -> None:
    pred = {"p": np.array([0.2, 0.9]), "mu_i": np.ones(2), "s_i": np.ones(2),
            "mu_r": -np.ones(2), "s_r": np.ones(2)}  # fmt: skip
    assert stages.p_neg_of(pred=pred).tolist() == [0.2, 0.9]

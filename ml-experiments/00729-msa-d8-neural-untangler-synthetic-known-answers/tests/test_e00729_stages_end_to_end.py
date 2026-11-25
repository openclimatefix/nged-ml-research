# ruff: noqa: E501, ANN001, PLR0915
"""A synthetic end-to-end run of the stages: banks, gate, tiny CPU fits, margin, scoring, verdict.

The day table, the metered record, and the private cut are replaced by invented arrays, so no real
file is read. The fits run for one epoch on a few windows. The test checks that the stages run, write
their files, and route a failed fit and a correctness error to their verdict rows.
"""

import pytest

torch = pytest.importorskip("torch")

from datetime import datetime, timedelta  # noqa: E402

import e00729_banks as banks  # noqa: E402
import e00729_core as core  # noqa: E402
import e00729_data as data  # noqa: E402
import e00729_leakage as leak  # noqa: E402
import e00729_paths as paths  # noqa: E402
import e00729_report as report  # noqa: E402
import e00729_stages as stages  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from synthetic_world import make_calendar  # noqa: E402

SPLITS = {
    "2022-03": "training", "2022-04": "training", "2022-05": "validation", "2022-06": "test",
    "2022-07": "test",
}  # fmt: skip
N_DAYS = 150
N_BASE = 130  # days from the start; the metered record lies after them


def write_private_splits(root) -> None:
    lines = ["| Number | Position | Month | Split | Carries BSP steps |", "|---|---|---|---|---|"]
    lines += [f"| {i} | {i} | {m} | {sp} | True |" for i, (m, sp) in enumerate(SPLITS.items())]
    (root / "PRIVATE").mkdir(exist_ok=True)
    (root / "PRIVATE" / "splits.md").write_text("\n".join(lines))


def fake_world(monkeypatch, tmp_path) -> None:
    cal = make_calendar(n_days=N_DAYS)
    era = np.array(["pre"] * 60 + ["net"] * 30 + ["solar"] * 40 + ["fade"] * 20)
    table = data.DayTable(
        calendar=cal, era=era, is_base=np.arange(N_DAYS) < N_BASE,
        dropped_half_hours=np.zeros(N_DAYS, dtype=int),
    )  # fmt: skip
    rng = np.random.default_rng(5)
    times = [
        datetime(2022, 3, 1) + timedelta(days=N_BASE + 1, minutes=30 * (i + 1))
        for i in range(48 * 15)
    ]
    units = np.repeat(rng.integers(0, 3, 15 * 6), 8)[: 48 * 15]
    record = pl.DataFrame(
        {"time": times, "power": units * 10.0 * 0.1, "units": units}
    ).with_columns(pl.col("time").dt.replace_time_zone("UTC").dt.cast_time_unit("us"))
    monkeypatch.setattr(paths, "data_root", lambda: tmp_path)
    write_private_splits(tmp_path)
    monkeypatch.setattr(data, "load_closure_days", lambda: (frozenset(), "missing"))
    monkeypatch.setattr(data, "load_day_table", lambda *, closure_days=frozenset(): table)
    monkeypatch.setattr(data, "private_cut", lambda: datetime(2022, 3, 1) + timedelta(days=N_BASE))
    monkeypatch.setattr(data, "metered_record", lambda: record)
    monkeypatch.setattr(core, "month_splits", lambda **_: dict(SPLITS))
    monkeypatch.setattr(core, "gate_ratio_ok", lambda **_: True)
    monkeypatch.setattr(core, "reachability_ok", lambda **_: True)
    monkeypatch.setattr(core, "EPOCH_WINDOWS", 32)
    monkeypatch.setattr(core, "BATCH_SIZE", 16)
    monkeypatch.setattr(core, "SEGMENT_DRAWS_TRAIN", 4)
    monkeypatch.setattr(core, "SEGMENT_DRAWS_EVAL", 4)
    monkeypatch.setattr(paths, "sibling_available", lambda *, prefix: True)
    import e00729_baselines as base

    monkeypatch.setattr(
        base, "m2_p_any", lambda *, reading_points: np.clip(reading_points / 100.0, 0.0, 1.0)
    )
    import e00729_train as train

    monkeypatch.setattr(train, "limit_gpu_memory", lambda **_: None)


def test_the_stages_run_end_to_end_on_invented_arrays(monkeypatch, tmp_path) -> None:
    fake_world(monkeypatch, tmp_path)
    logs: list[str] = []
    ctx = stages.build_context()
    assert ctx.closure_status == "missing"
    assert not ctx.closure_on_base
    summary = stages.stage_banks(ctx=ctx, train_seeds=core.ARCH_SEEDS, log=logs.append)
    assert summary["leakage_pass"]
    assert summary["injected_share"]["test"] == pytest.approx(
        7 / 8
    )  # quota of 8 test draws: 1 without injection
    assert summary["digests_unchanged"] is None  # no earlier summary to compare with
    first_digests = dict(summary["digests"])
    again = stages.stage_banks(ctx=ctx, train_seeds=core.ARCH_SEEDS, log=logs.append)
    assert again["digests"] == first_digests
    assert again["digests_unchanged"] is True
    assert (tmp_path / "banks" / "summary_before.json").exists()
    assert any("digests unchanged from the earlier summary: True" in line for line in logs)
    gate = stages.stage_gate(ctx=ctx, log=logs.append)
    assert "void" in gate["bsp"]
    assert gate["bsp"]["n_synthetic_stretches"] == gate["bsp"]["n_finite_fits"]
    assert (tmp_path / "scores" / "gate.json").exists()
    for arm in core.ARMS:
        for seed in core.ARCH_SEEDS:
            stages.run_fit(
                ctx=ctx, arm=arm, arch="dae", seed=seed, device="cpu", log=logs.append, max_epochs=1
            )
    stages.write_json(
        path=tmp_path / "selection.json",
        payload={"chosen": "dae", "scores": {"dae": 1.0}, "weights": {"dae": 1}},
    )
    core.write_margin_file(
        path=tmp_path / "margin.json", val_crps_by_seed=[1.0, 1.1, 0.9, 1.05, 0.95]
    )
    pis: list[float] = []
    original_metrics = stages.baseline_metrics

    def capture(**kw):  # noqa: ANN003, ANN202
        pis.append(kw["pi"])
        return original_metrics(**kw)

    monkeypatch.setattr(stages, "baseline_metrics", capture)
    record = stages.stage_score(ctx=ctx, n_resamples=40, log=logs.append)
    monkeypatch.setattr(stages, "baseline_metrics", original_metrics)
    test_share = banks.injected_share(bank=banks.load_bank(path=stages.bank_path(kind="test")))
    assert test_share != core.TEST_MIX_INJECTED
    assert test_share in pis  # the test rows take the bank's realised share, not 0.85
    assert core.TEST_MIX_INJECTED not in pis
    assert record["row"] in range(9)
    assert (tmp_path / "verdict.json").exists()
    assert {"C1", "C12", "C13"} <= set(record["contrasts"])
    ambiguity = stages.read_json(path=tmp_path / "scores" / "tables.json")["ambiguity"]
    assert set(ambiguity) == {*(a for a in core.ARMS if a != "Dq"), "Bstar"}
    assert set(ambiguity["Dsign"]) == {"reading", "truth_magnitude"}
    assert len(ambiguity["Dsign"]["reading"]) == 4
    assert "M2" in stages.read_json(path=tmp_path / "scores" / "tables.json")["tables"]
    assert not {"mae_head", "mae_zero", "n_running"} & set(record["real"]["k1"])
    realised = stages.read_json(path=tmp_path / "scores" / "tables.json")["realised_injected_share"]
    assert realised == pytest.approx(7 / 8)
    written = stages.stage_figures(ctx=ctx, arch="dae", log=logs.append)
    assert any(p.name == "figure_6_windows.svg" for p in written)
    assert all(p.exists() for p in written)

    # the report-only arm D-sign: a missing fit voids C13 and leaves the verdict untouched
    verdict_row = record["row"]
    tables = stages.read_json(path=tmp_path / "scores" / "tables.json")
    text = report.render_report(
        verdict=record,
        tables=tables,
        gate=stages.read_json(path=tmp_path / "scores" / "gate.json"),
        timing=None,
    )
    assert "C13 (report only)" in text
    assert "Ambiguity read-out" in text
    assert "| Dsign | reading |" in text
    for seed in core.ARCH_SEEDS:
        (tmp_path / "fits" / f"Dsign_dae_s{seed}.json").rename(
            tmp_path / "fits" / f"Dsign_dae_s{seed}.moved"
        )
    (tmp_path / "fits" / "Dsign_dae_s0.failed.json").write_text("{}")
    assert "Dsign_dae_s0" not in stages.needed_job_names()
    assert not stages.needed_fits_failed()
    voided = stages.stage_score(ctx=ctx, n_resamples=40, log=logs.append)
    assert voided["row"] == verdict_row
    assert voided["contrasts"]["C13"]["all"]["void"] is True
    assert "Dsign" not in stages.read_json(path=tmp_path / "scores" / "tables.json")["ambiguity"]
    assert "| C13 (report only) | all | void |" in report.render_report(
        verdict=voided,
        tables=stages.read_json(path=tmp_path / "scores" / "tables.json"),
        gate=stages.read_json(path=tmp_path / "scores" / "gate.json"),
        timing=None,
    )
    for seed in core.ARCH_SEEDS:
        (tmp_path / "fits" / f"Dsign_dae_s{seed}.moved").rename(
            tmp_path / "fits" / f"Dsign_dae_s{seed}.json"
        )
    (tmp_path / "fits" / "Dsign_dae_s0.failed.json").unlink()
    # a failure record of a needed fit without a fit record reads row 2
    (tmp_path / "fits" / "A_dae_s0.json").rename(tmp_path / "fits" / "A_dae_s0.moved")
    (tmp_path / "fits" / "A_dae_s0.failed.json").write_text("{}")
    assert stages.stage_score(ctx=ctx, n_resamples=40, log=logs.append)["row"] == 2
    # a stale failure record next to a fit record is ignored, as is one for a fit nobody needs
    (tmp_path / "fits" / "A_dae_s0.moved").rename(tmp_path / "fits" / "A_dae_s0.json")
    (tmp_path / "fits" / "Z_unet_s9.failed.json").write_text("{}")
    assert not stages.needed_fits_failed()
    (tmp_path / "fits" / "A_dae_s0.failed.json").unlink()
    (tmp_path / "fits" / "Z_unet_s9.failed.json").unlink()

    # a correctness error inside scoring reads row 1
    summary_path = tmp_path / "banks" / "summary.json"
    original = stages.read_json(path=summary_path)

    def raise_error(**_: object) -> None:
        raise leak.CorrectnessFailureError("crafted")

    monkeypatch.setattr(leak, "assert_hashes_match", raise_error)
    assert stages.stage_score(ctx=ctx, n_resamples=40, log=logs.append)["row"] == 1
    assert original["leakage_pass"]
    assert all("/home/" not in line for line in logs)


def test_a_failed_fit_is_recorded_and_not_retried(monkeypatch, tmp_path) -> None:
    fake_world(monkeypatch, tmp_path)
    ctx = stages.build_context()

    def boom(**_: object) -> dict:
        raise RuntimeError("crafted")

    monkeypatch.setattr(stages, "run_fit", boom)
    logs: list[str] = []
    assert (
        stages.run_fit_or_record_failure(
            ctx=ctx, arm="A", arch="dae", seed=0, device="cpu", log=logs.append
        )
        is None
    )
    assert (tmp_path / "fits" / "A_dae_s0.failed.json").exists()
    assert "RuntimeError" in logs[0]
    stages.write_json(path=tmp_path / "selection.json", payload={"chosen": "dae"})
    assert stages.stage_score(ctx=ctx, log=logs.append)["row"] == 2

    def correctness(**_: object) -> dict:
        raise leak.CorrectnessFailureError("crafted")

    monkeypatch.setattr(stages, "run_fit", correctness)
    with pytest.raises(leak.CorrectnessFailureError):
        stages.run_fit_or_record_failure(
            ctx=ctx, arm="B", arch="dae", seed=0, device="cpu", log=logs.append
        )


def test_the_context_refuses_a_split_that_differs_from_the_private_file(
    monkeypatch, tmp_path
) -> None:
    fake_world(monkeypatch, tmp_path)
    assert stages.build_context().month_split == SPLITS
    path = tmp_path / "PRIVATE" / "splits.md"
    path.write_text(path.read_text().replace("2022-06 | test", "2022-06 | validation"))
    with pytest.raises(leak.CorrectnessFailureError, match="differs"):
        stages.build_context()
    path.unlink()
    with pytest.raises(leak.CorrectnessFailureError, match="missing"):
        stages.build_context()
    assert banks.N_SIZE_BINS == 6

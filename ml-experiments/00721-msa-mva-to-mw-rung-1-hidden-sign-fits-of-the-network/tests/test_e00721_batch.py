# ruff: noqa: E501, PT018
"""Tests of the batch builder, the probe provider, the sky columns, and the driver's file handling.

The batch tests build the mini world of `e00721_e2e` (random weather, a synthetic signed process) and
read no private data. They need `torch` and are skipped without it.
"""

import json
from pathlib import Path

import e00721_core as core
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import e00721_paths  # noqa: E402

e00721_paths.pin_threads()

import e00721_arms as arms  # noqa: E402
import e00721_driver as drv  # noqa: E402
import e00721_e2e as e2e  # noqa: E402
import e00721_fit as fit  # noqa: E402
import e00721_real as real  # noqa: E402
import e00721_train as trn  # noqa: E402


def _arm_world(*, arm: str, world: str = "W2") -> trn.ArmWorld:
    job = fit.Job(arm=arm, world=world, fold="2025-12", seed=0)
    return e2e.make_world_builder()(job)


def test_sky_columns_are_the_shortwave_the_solar_gain_and_the_geometry_columns() -> None:
    cols = real.sky_columns()
    assert cols["weather"] == (0, 1, 5, 15, 16, 17, 18, 19, 20)
    assert cols["datetime"] == (0, 1, 2)
    assert cols["spread"] == (0, 1, 5)
    assert cols["weather"] == arms.SKY_WEATHER_COLUMNS


def test_probe_provider_names_the_missing_file_and_never_fabricates(tmp_path: Path) -> None:
    provider = real.ProbeProvider(root=tmp_path)
    with pytest.raises(real.ProbeFileMissingError, match=r"W2_2026-05\.json"):
        provider.capacity_mw(world="W2", fold="2026-05")
    path = provider.path(world="W2", fold="2026-05")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"D5": 3.0}))
    with pytest.raises(ValueError, match="D10"):
        provider.capacity_mw(world="W2", fold="2026-05")
    path.write_text(json.dumps({"D5": 3.0, "D10": 2.0}))
    assert provider.capacity_mw(world="W2", fold="2026-05") == {"D5": 3.0, "D10": 2.0}


def test_share_scale_adds_the_injected_size() -> None:
    scale = real.share_scale_with_injection(
        caps_mw=np.array([2.0, 0.0]),
        power_cap_mw=np.array([10.0, 10.0]),
        increment_pct=np.array([5.0, 0.0]),
        scale=0.7,
    )
    assert scale == pytest.approx([0.2 + 0.7 * 0.05, 0.0])


def test_target_row_u_gate_reads_no_u_at_or_after_the_native_origin() -> None:
    u = np.arange(100.0)[None, :]
    got = real.target_row_u_gate(u_grid=u, origin_idx=np.array([50, 60]))
    assert got.max() < 60 and got[0].tolist() == [44.0, 54.0]


def test_batch_marks_folded_history_and_labels_only_supervised_primaries() -> None:
    aw = _arm_world(arm="SG-A")
    src = aw.sources[2]
    visible = e2e.site_index(*core.VISIBLE_SIGNED)
    hidden = e2e.site_index(*core.HIDDEN_PAIR)
    c = torch.tensor([60 * 48 + 24] * 2)
    lead = torch.zeros(2, dtype=torch.long)
    prim = torch.tensor([visible[0], hidden[0]])
    plain = src.assemble(primary=prim, c_idx=c, lead_day=lead, fold_hist=None)
    folded = src.assemble(
        primary=prim, c_idx=c, lead_day=lead, fold_hist=torch.ones(2, dtype=torch.bool)
    )
    assert (
        float(plain.inputs.hist_u[0].abs().sum()) == 0.0
    )  # a visible Primary's plain history is signed
    assert (
        float(plain.inputs.hist_u[1].abs().sum()) > 0.0
    )  # the hidden pair's history is always folded
    assert float(folded.inputs.hist_u[0].abs().sum()) > 0.0  # folding the sample counts its tokens
    assert bool(torch.isfinite(folded.hist_sign[0]).any()) and not bool(
        torch.isfinite(folded.hist_sign[1]).any()
    )
    assert not bool(torch.equal(plain.inputs.hist_power[0], folded.inputs.hist_power[0]))
    assert bool(torch.equal(plain.inputs.hist_power[1], folded.inputs.hist_power[1]))


def test_one_training_pass_and_prediction_rows_have_the_scored_columns() -> None:
    aw = _arm_world(arm="SG-A")
    inputs = e2e.jobs.WorldInputs(make_world=lambda job: aw)
    job = fit.Job(arm="SG-A", world="W2", fold="2025-12", seed=0)
    trainable = inputs.build_trainable(job=job, stage=2)
    loss0 = trainable.validate()
    trainable.train_pass()
    assert np.isfinite(loss0) and np.isfinite(trainable.validate())
    table = inputs.predict(job=job)
    assert {"actual", "reading", "q50", "fq50", "seed", "fold"} <= set(table.columns)
    lead = (table["valid_time"] - table["power_fcst_init_time"]).dt.total_hours()
    assert int(lead.to_numpy().min()) >= 24
    assert int(lead.to_numpy().max()) < 240
    assert table.select("q01", "q99").to_numpy().shape[1] == 2


def test_driver_counts_a_missing_curve_as_not_evaluated(tmp_path: Path) -> None:
    flags = drv.flags_of(data_dir=tmp_path, folds=("2025-12",), seeds=(0,))
    assert flags == [{"evaluated": False, "flagged": False, "reason": "missing curve"}]


def test_driver_synthetic_gate_without_token_files_is_not_evaluable(tmp_path: Path) -> None:
    gate = drv.synthetic_gate(data_dir=tmp_path, seeds=(0, 1))
    assert not gate["pass"] and gate["missing"] and not gate["failed"]

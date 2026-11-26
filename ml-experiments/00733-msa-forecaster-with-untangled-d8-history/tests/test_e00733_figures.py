# ruff: noqa: PT018
"""Tests of `e00733_figures` and `e00733_attention`: the plotted rows, the stored pass marker, the
device branches, and the attention shapes."""

import json
from pathlib import Path

import e00733_attention as att
import e00733_figures as figs
import e00733_model as model
import numpy as np
import pytest
import torch
from torch_inputs import make_inputs


def _real_record(*, arm: str = "FU", seed: int = 0, n: int = 6) -> dict:
    stages = [
        {
            "stage": s,
            "histories": [
                {
                    "train": list(np.linspace(2, 1, n)),
                    "val": list(np.linspace(2.1, 1.1, n)),
                    "stored_pass": n - 1,
                }
            ],
            "stored_pass": n - 1,
        }
        for s in (2, 3)
    ]
    return {"arm": arm, "fold": "2026-01", "seed": seed, "stages": stages}


def _synth_record(*, arm: str = "YU", n: int = 8) -> dict:
    h = {
        "train": list(np.linspace(3, 1, n)),
        "val": list(np.linspace(3, 1.5, n)),
        "val_nll": list(np.linspace(2, 1, n)),
        "val_bce": list(np.linspace(1, 0.5, n)),
        "stored_pass": n,
    }
    return {"arm": arm, "fold": None, "seed": 0, "history": h, "stored_pass": n}


def test_curve_rows_mark_the_stored_pass_and_split_the_synthetic_validation_loss() -> None:
    """A real record gives one panel per stage; a synthetic record adds the mixture and sign parts;
    the stored pass is in the rows."""
    rows = figs.curve_rows(record=_real_record())
    assert set(rows["panel"].unique()) == {"stage 2", "stage 3"} and set(
        rows["stored_pass"].unique()
    ) == {5}
    syn = figs.curve_rows(record=_synth_record())
    assert {"validation mixture", "validation sign"} <= set(syn["split"].unique())


def test_curve_rows_reject_bad_histories() -> None:
    """An empty history, a non-finite loss, or a stored pass outside the passes run is an error."""
    bad = _synth_record()
    bad["history"]["val"][2] = float("nan")
    with pytest.raises(figs.HistoryError):
        figs.curve_rows(record=bad)
    out = _synth_record()
    out["history"]["stored_pass"] = 99
    with pytest.raises(figs.HistoryError):
        figs.curve_rows(record=out)
    with pytest.raises(figs.HistoryError):
        figs.curve_rows(
            record={
                "arm": "FU",
                "fold": "x",
                "seed": 0,
                "stages": [
                    {
                        "stage": 2,
                        "histories": [{"train": [], "val": [], "stored_pass": 0}],
                        "stored_pass": 0,
                    }
                ],
            }
        )


def test_write_loss_curves_picks_seed_0_of_each_arm_and_writes_svg(tmp_path: Path) -> None:
    """Records of FU and FU-S at several seeds give one panel each for seed 0; the SVG exists."""
    folder = tmp_path / "fits"
    folder.mkdir()
    for name, rec in [
        ("FU_2026-01_s0", _real_record()),
        ("FU_2026-01_s1", _real_record(seed=1)),
        ("FU-S_2026-01_s0", _real_record(arm="FU-S")),
        ("YU_s0", _synth_record()),
    ]:
        (folder / f"{name}.json").write_text(json.dumps(rec))
    (folder / "FU_2026-02_s0.failed.json").write_text("{}")
    picked = figs.pick_loss_records(root=tmp_path)
    assert [r["arm"] for r in picked] == ["FU", "FU-S", "YU"] and all(
        r["seed"] == 0 for r in picked
    )
    out = figs.write_loss_curves(root=tmp_path, out=tmp_path / "loss.svg")
    assert out.read_text().startswith("<svg") or "<svg" in out.read_text()[:400]
    with pytest.raises(figs.HistoryError):
        figs.pick_loss_records(root=tmp_path / "empty")


def test_contrast_chart_skips_void_readings_and_refuses_an_empty_set() -> None:
    """A void reading is left out of the rows; a chart of only void readings is an error."""
    readings = {
        "C1": {"relative": {"point": -0.02, "lower": -0.04, "upper": 0.0}},
        "C2": "void",
        "R1": {"point": 0.01, "lower": -0.02, "upper": 0.03},
    }
    rows = figs.contrast_rows(readings=readings)
    assert rows["contrast"].to_list() == ["C1", "R1"]
    figs.contrast_dot_chart(
        readings=readings, names=["C1", "C2", "R1"], title="t", x_title="x"
    ).to_dict()
    with pytest.raises(figs.HistoryError):
        figs.contrast_dot_chart(readings=readings, names=["C2"], title="t", x_title="x")


def test_sign_series_and_reliability() -> None:
    """The sign-forecast rows hold the truth and each forecast; a forecast of another length is an
    error; the reliability bins are finite."""
    truth = np.r_[np.ones(10), np.zeros(10)]
    rows = figs.sign_series_rows(truth=truth, forecasts={"YU": truth * 0.8 + 0.1})
    assert set(rows["series"].unique()) == {"true sign", "YU"}
    figs.sign_series_chart(rows=rows, title="t").to_dict()
    with pytest.raises(figs.HistoryError):
        figs.sign_series_rows(truth=truth, forecasts={"YU": truth[:5]})
    xs, ys = figs.reliability_rows(p=truth * 0.8 + 0.1, truth=truth)
    assert xs == pytest.approx([0.1, 0.9]) and ys == [0.0, 1.0]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
        ),
    ],
)
def test_attention_extract_shapes_and_gain(device: str) -> None:
    """The extraction returns head-averaged maps in numpy (through `.detach().cpu()`), the read-out
    weights, and the gain; the figure builds."""
    net = model.UntangledNet(seed=0).to(device)
    maps = att.extract(net=net, inp=make_inputs(b=2), device=device)
    assert maps["across_0"].shape == (48, 15) and maps["within_0"].shape == (48, 48)
    assert maps["alpha"].shape == (14, 48) and maps["gamma"].shape == (64,)
    assert np.allclose(maps["across_0"].sum(axis=1), 1.0, atol=1e-4)
    assert "learned gain" in att.gamma_line(
        gamma=maps["gamma"]
    ) and "no untangled" in att.gamma_line(gamma=np.zeros(0))
    att.attention_figure(maps=maps, label="case").to_dict()


def test_attention_extract_cpu_stub_for_the_cuda_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The device argument is passed to `inp.to`, so the cuda branch is covered on a CPU-only host
    by a recording stub."""
    seen: list[str] = []
    inp = make_inputs(b=1)
    orig = inp.to

    def spy(device: str | torch.device) -> model.UntInputs:
        seen.append(str(device))
        return orig("cpu")

    monkeypatch.setattr(type(inp), "to", lambda self, d: spy(d))
    att.extract(net=model.UntangledNet(seed=0), inp=inp, device="cuda")
    assert seen == ["cuda"]


def test_heatmap_refuses_non_finite() -> None:
    """A map with NaN is an error."""
    with pytest.raises(att.AttentionError):
        att.heatmap(matrix=np.array([[np.nan, 1.0]]), x_title="x", y_title="y", title="t")

# ruff: noqa: E501
"""Tests of the report and figure stage on a fixture shaped like the scoring stage's files."""

from pathlib import Path

import e00729_core as core
import e00729_report as report
import pytest


def reading(*, point: float) -> dict[str, float]:
    return {
        "point": point, "lower": point - 0.1, "upper": point + 0.1, "kish_blocks": 30.0,
        "kish_segments": 60.0, "widening": 1.0, "n_rows": 1000,
    }  # fmt: skip


def verdict_fixture(*, void_c13: bool = False) -> dict:
    """Return an invented `verdict.json` with the keys, nesting, and reading labels of the real one."""
    contrasts: dict[str, dict] = {}
    for k, name in enumerate(core.CONTRASTS):
        if name in core.BSP_CONTRASTS:
            contrasts[name] = {
                label: reading(point=-0.1 * k) for label in
                ("bsp_all", "bsp_engine", "bsp_solar", "bsp_both")
            }  # fmt: skip
        else:
            contrasts[name] = {"all": reading(point=-0.1 * k)}
    contrasts["C3"]["source_S1"] = reading(point=-0.3)
    if void_c13:
        contrasts["C13"] = {"all": {**reading(point=0.0), "void": True}}
    hours = {f"{era}_{h}": 0.1 for era in ("net", "solar", "fade") for h in range(24)}
    return {
        "row": 4, "verdict": "supported", "margin": 0.25, "bstar": "B1", "architecture": "unet",
        "contrasts": contrasts,
        "calibration": {
            "reliability": 0.002, "max_gap": 0.05, "count": [10.0] * 10,
            "mean_forecast": [0.05 + 0.1 * i for i in range(10)],
            "frequency": [0.04 + 0.1 * i for i in range(10)], "pass": True,
        },
        "lobe": {"coverage": 0.97, "pass": True},
        "h1": {"crps_excess": 0.03, "mae_excess": 0.01, "false_sign_share": 0.0, "pass": True},
        "reachability": {"n_all": 4000, "n_bsp": 1000, "pass": True},
        "shifted": {},
        "real": {
            "h2": {"pass": True}, "k1": {"status": "pass"}, "k2": {"status": "pass"},
            "r2": {"net": 0.99},
            "r3": {
                "share_above_half_by_era_and_hour": hours,
                "share_below_1mva_by_era_and_hour": hours,
            },
        },
    }  # fmt: skip


def tables_fixture() -> dict:
    subset = {"n": 1000.0, "crps": 0.5, "pinball": 0.4, "mae": 0.3}
    systems = ("P0", "P0f", "B1", "B2", "Bstar", "A", "B", "C", "D", "H", "Dq", "Dsign")
    return {
        "tables": {
            s: {k: dict(subset) for k in ("all", "import", "reverse", "dead_band")} for s in systems
        },
        "coverage": {s: {"q05": 0.05, "q95": 0.95} for s in systems},
        "lobe_width_mean": 3.0,
        "ambiguity": {
            "D": {
                "reading": {
                    "MVA_0_to_0.3": {
                        "n": 10.0, "mean_forecast": 0.5, "frequency": 0.5, "reliability": 0.0,
                        "sharpness": 0.0, "confident_share": 0.0,
                    },
                    "MVA_3_to_inf": {"n": 0.0},
                },
            }
        },
    }  # fmt: skip


GATE = {
    "pass": True,
    "shares": {
        e: {
            "real_day": 0.2,
            "real_half_hour": 0.02,
            "synthetic_day": 0.15,
            "synthetic_half_hour": 0.02,
        }
        for e in ("net", "solar")
    },
    "bsp": {"pass": True},
}
SELECTION = {"chosen": "unet", "scores": {"dae": 1.0, "unet": 0.9, "bilstm": 0.95}}


def test_the_headline_reading_of_a_bsp_contrast_is_its_bsp_all_reading() -> None:
    points = report.contrast_points(contrasts=verdict_fixture()["contrasts"])
    assert "C4 (BSP months)" in points
    assert "C12 (BSP months)" in points
    assert "C13 (report only)" in points
    assert "C4" not in points
    assert points["C1"] == (0.0, -0.1, 0.1)
    assert len(points) == len(core.CONTRASTS)


def test_a_void_contrast_is_left_out_of_the_contrast_points() -> None:
    points = report.contrast_points(contrasts=verdict_fixture(void_c13=True)["contrasts"])
    assert "C13 (report only)" not in points
    assert len(points) == len(core.CONTRASTS) - 1


@pytest.mark.parametrize("void_c13", [False, True])
def test_write_figures_runs_on_a_fixture_shaped_like_the_scoring_files(
    tmp_path: Path, void_c13: bool
) -> None:
    written = report.write_figures(
        verdict=verdict_fixture(void_c13=void_c13), tables=tables_fixture(), gate=GATE,
        selection=SELECTION, out_dir=tmp_path,
    )  # fmt: skip
    names = {p.name for p in written}
    assert "figure_5_contrasts.svg" in names
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


@pytest.mark.parametrize("void_c13", [False, True])
def test_render_report_runs_on_the_same_fixture(void_c13: bool) -> None:
    text = report.render_report(
        verdict={**verdict_fixture(void_c13=void_c13), "real": {**verdict_fixture()["real"], "r2": {}}},
        tables=tables_fixture(), gate=GATE, timing=None,
    )  # fmt: skip
    assert "C4 |" in text
    assert ("| C13 (report only) | all | void |" in text) is void_c13
    assert "| D | reading | MVA_0_to_0.3 | 10 |" in text


def test_dot_chart_legend_labels_are_not_clipped() -> None:
    """The era figure's legend once showed "reading below 1.0 M…", so the label limit must fit."""
    import e00729_figures as figures

    arms = ["fade: P(X<0) above 0.5", "fade: reading below 1.0 MVA", "solar: reading below 1.0 MVA"]
    chart = figures.dot_chart(
        scores={a: {"0": 0.1} for a in arms}, rows=["0"], title="t", x_title="x"
    )
    legend = chart.to_dict()["layer"][1]["encoding"]["color"]["legend"]
    assert legend["labelLimit"] >= 7 * max(len(a) for a in arms)

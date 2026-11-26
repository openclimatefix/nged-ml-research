# ruff: noqa: PT018
"""Tests of `e00733_report`: every number is printed with its half-width, void readings stay void,
and a missing file is an error."""

from pathlib import Path

import e00733_report as rep
import e00733_score as sc
import pytest
from test_e00733_score import _frame


def test_report_prints_half_width_void_and_words(tmp_path: Path) -> None:
    """The report prints C1 relative with its half-width, `void` for a dropped arm's reading, and
    both reading words."""
    ref = _frame(loss=2.0)
    c1 = sc.real_contrast(
        treatment=ref.with_columns(crps=ref["crps"] - 0.1), reference=ref, n_resamples=500
    )
    verdict, readings = sc.assemble(
        stopped=False,
        invalid=False,
        failed=False,
        c={"C1": c1, "C2": c1, "C3": None, "C4": None, "R1": None},
        guards={"all": True},
        sign_inputs=None,
        dropped_arms=("FS",),
        coverage={"n_gap_base_dates": 4},
    )
    sc.write_results(root=tmp_path, verdict=verdict, readings=readings)
    path = rep.write_report(root=tmp_path, counted_read=False)
    text = path.read_text()
    assert "C1: -5." in text and "half-width" in text and "R1: void" in text and "(interim)" in text
    assert "Sign-forecast word: void" in text and "n_gap_base_dates" in text
    with pytest.raises(rep.ReportError):
        rep.write_report(root=tmp_path / "none")

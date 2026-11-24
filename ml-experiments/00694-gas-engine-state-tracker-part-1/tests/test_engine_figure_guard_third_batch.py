"""The guard covers the Gas A levels-and-ramps figure of experiment 697 and keeps it value-free."""

from pathlib import Path
from typing import Any, Final

import altair as alt
import engine_figure_guard as guard
import polars as pl
import pytest

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
FIGURE: Final[str] = "f4_gas_a_levels_and_ramps.svg"


def _chart(data: pl.DataFrame) -> Any:
    """Return a chart typed as `Any`, because `ty` loses the type after a `mark_*` call."""
    return alt.Chart(data)


def test_a_figure_with_megawatt_values_is_refused(tmp_path: Path):
    target = tmp_path / "00697-x"
    target.mkdir()
    levels = (
        _chart(pl.DataFrame({"power": [1.0, 2.0], "n": [3, 4]}))
        .mark_bar(aria=False)
        .encode(x=alt.X("power:Q", title="Output (MW)"), y=alt.Y("n:Q", title="Half-hours"))
    )
    assert guard.is_guarded(path=target / FIGURE)
    with pytest.raises(guard.GuardError):
        guard.save_chart(chart=levels, path=target / FIGURE)
    assert not (target / FIGURE).exists()


def test_a_figure_with_per_mark_labels_is_refused(tmp_path: Path):
    target = tmp_path / "00697-x"
    target.mkdir()
    labelled = (
        _chart(pl.DataFrame({"fraction": [0.5, 1.0], "n": [3, 4]}))
        .mark_bar()
        .encode(x=alt.X("fraction:Q", title="Fraction"), y=alt.Y("n:Q", title="Share"))
    )
    with pytest.raises(guard.GuardError):
        guard.save_chart(chart=labelled, path=target / FIGURE)


def test_an_aggregate_without_units_or_labels_is_written(tmp_path: Path):
    target = tmp_path / "00697-x"
    target.mkdir()
    aggregate = (
        _chart(pl.DataFrame({"fraction": [0.5, 1.0], "n": [0.3, 0.4]}))
        .mark_bar(aria=False)
        .encode(
            x=alt.X("fraction:Q", title="Fraction of the highest"), y=alt.Y("n:Q", title="Share")
        )
    )
    guard.save_chart(chart=aggregate, path=target / FIGURE)
    assert (target / FIGURE).exists()


def test_the_committed_figure_passes_the_text_check():
    docs = ROOT.parent / "docs" / "ml-experiments"
    (path,) = docs.glob(f"00697-*/{FIGURE}")
    assert guard.svg_problems(svg=path.read_text()) == [], path

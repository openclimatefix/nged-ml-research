"""The guard refuses a dated series for each figure added in the second anonymisation batch."""

import ast
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import altair as alt
import engine_figure_guard as guard
import numpy as np
import polars as pl
import pytest

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
NEW_FIGURES: Final[tuple[tuple[str, str], ...]] = (
    ("00194-x", "figure_1.svg"),
    ("00194-x", "figure_4.svg"),
    ("00194-x", "figure_5.svg"),
    ("00196-x", "figure_3_flagged_share_by_month.svg"),
    ("00361-x", "figure_3_falls_per_month.svg"),
    ("00291-x", "figure_3.svg"),
    ("00159-x", "figure_1_history.svg"),
    ("00694-x", "figure_5_gas_a_levels.svg"),
)
SCRIPTS: Final[tuple[str, ...]] = (
    "00196-*/e00196_report.py",
    "00361-*/e00361_charts.py",
    "00291-*/e00291_charts.py",
    "00159-*/e00159_charts.py",
    "00294-*/e00294_charts.py",
    "00694-*/e00694_look.py",
)
COMMITTED: Final[tuple[tuple[str, str], ...]] = (
    ("00194-*", "figure_1.svg"),
    ("00194-*", "figure_4.svg"),
    ("00194-*", "figure_5.svg"),
    ("00196-*", "figure_3_flagged_share_by_month.svg"),
    ("00361-*", "figure_3_falls_per_month.svg"),
    ("00291-*", "figure_3.svg"),
    ("00159-*", "figure_1_history.svg"),
    ("00694-*", "figure_5_gas_a_levels.svg"),
)
DELETED: Final[tuple[tuple[str, str], ...]] = (
    ("00196-*", "figure_4_known_answer_week.svg"),
    ("00294-*", "figure_2_engine_offset.svg"),
    ("00294-*", "figure_4_example_weeks.svg"),
    ("00159-*", "figure_5_depth.svg"),
    ("00159-*", "figure_7_alarms.svg"),
)


def _chart(data: pl.DataFrame) -> Any:
    """Return a chart typed as `Any`, because `ty` loses the type after a `mark_*` call."""
    return alt.Chart(data)


def _dated_series(*, n: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    start = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "day": [start + timedelta(days=7 * i) for i in range(n)],
            "share": rng.uniform(0.0, 0.5, n),
        }
    )


@pytest.mark.parametrize(("folder", "name"), NEW_FIGURES)
def test_a_dated_series_is_refused_for_each_new_figure(tmp_path: Path, folder: str, name: str):
    target = tmp_path / folder
    target.mkdir()
    dated = (
        _chart(_dated_series())
        .mark_line()
        .encode(x=alt.X("day:T", title="Week"), y=alt.Y("share:Q", title="Share"))
    )
    assert guard.is_guarded(path=target / name)
    with pytest.raises(guard.GuardError):
        guard.save_chart(chart=dated, path=target / name)
    assert not (target / name).exists()


@pytest.mark.parametrize(("folder", "name"), NEW_FIGURES)
def test_an_undated_pooled_aggregate_is_written_for_each_new_figure(
    tmp_path: Path, folder: str, name: str
):
    target = tmp_path / folder
    target.mkdir()
    pooled = pl.DataFrame({"era": ["Era 1", "Era 2"], "share": [0.1, 0.4]})
    chart = (
        _chart(pooled)
        .mark_bar(aria=False)
        .encode(x=alt.X("era:N", title=None), y=alt.Y("share:Q", title="Mean share"))
    )
    guard.save_chart(chart=chart, path=target / name)
    assert (target / name).exists()


@pytest.mark.parametrize("pattern", SCRIPTS)
def test_every_writer_in_the_second_batch_of_scripts_goes_through_the_guard(pattern: str):
    (script,) = ROOT.glob(pattern)
    tree = ast.parse(script.read_text())
    saves = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"save", "write_image", "to_json"}
        and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "guard")
    ]
    assert saves == [], f"{script.name} writes a chart without the guard"
    assert "guard.save_chart(" in script.read_text()


@pytest.mark.parametrize(("folder", "name"), COMMITTED)
def test_the_committed_redrawn_figures_pass_the_text_check(folder: str, name: str):
    docs = ROOT.parent / "docs" / "ml-experiments"
    (path,) = docs.glob(f"{folder}/{name}")
    assert guard.svg_problems(svg=path.read_text()) == [], path


@pytest.mark.parametrize(("folder", "name"), DELETED)
def test_the_deleted_figures_stay_deleted(folder: str, name: str):
    docs = ROOT.parent / "docs" / "ml-experiments"
    assert not list(docs.glob(f"{folder}/{name}"))

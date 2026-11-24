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
    ("00291-x", "figure_2.svg"),
    ("00291-x", "figure_3.svg"),
    ("00199-x", "figure_5_flagged_hours.svg"),
    ("00159-x", "figure_1_history.svg"),
    ("00694-x", "figure_5_gas_a_levels.svg"),
)
COMMITTED: Final[tuple[tuple[str, str], ...]] = (
    ("00194-*", "figure_1.svg"),
    ("00194-*", "figure_4.svg"),
    ("00194-*", "figure_5.svg"),
    ("00196-*", "figure_3_flagged_share_by_month.svg"),
    ("00361-*", "figure_3_falls_per_month.svg"),
    ("00291-*", "figure_2.svg"),
    ("00291-*", "figure_3.svg"),
    ("00199-*", "figure_5_flagged_hours.svg"),
    ("00159-*", "figure_1_history.svg"),
    ("00694-*", "figure_5_gas_a_levels.svg"),
)
DELETED: Final[tuple[tuple[str, str], ...]] = (
    ("00196-*", "figure_4_known_answer_week.svg"),
    ("00294-*", "figure_2_engine_offset.svg"),
    ("00294-*", "figure_4_example_weeks.svg"),
    ("00159-*", "figure_5_depth.svg"),
    ("00159-*", "figure_7_alarms.svg"),
    ("00196-*", "figure_2_unit_size_profile.svg"),
    ("00182-*", "figure_1.svg"),
    ("00361-*", "figure_4_d8_demand_with_detected_steps.svg"),
    ("00361-*", "figure_6_forecasts_with_and_without_steps.svg"),
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


def _guarded_folders() -> list[Path]:
    """Return every experiment folder whose prefix is named in `GUARDED`."""
    prefixes = {prefix for prefix, _ in guard.GUARDED}
    return sorted(
        folder for prefix in prefixes for folder in ROOT.glob(f"{prefix}*") if folder.is_dir()
    )


def _bare_saves(*, script: Path) -> list[int]:
    """Return the line of each chart write in a script that does not go through the guard."""
    lines = []
    for node in ast.walk(ast.parse(script.read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in {"save", "write_image", "to_json", "write_html"}:
            continue
        target = node.func.value
        if isinstance(target, ast.Name) and target.id in {"guard", "np", "numpy"}:
            continue
        lines.append(node.lineno)
    return lines


def test_every_script_in_a_guarded_experiment_folder_writes_charts_through_the_guard():
    scripts = [
        script
        for folder in _guarded_folders()
        for script in folder.rglob("*.py")
        if "tests" not in script.relative_to(folder).parts
    ]
    assert len(scripts) > 50  # the glob found the experiment folders
    bare = {script.name: _bare_saves(script=script) for script in scripts}
    assert {name: lines for name, lines in bare.items() if lines} == {}


def test_the_bare_save_check_flags_a_script_that_writes_without_the_guard(tmp_path: Path):
    script = tmp_path / "e_demo.py"
    script.write_text("chart = make()\nchart.save('figure_1.svg')\nnp.save('a.npy', x)\n")
    assert _bare_saves(script=script) == [2]


@pytest.mark.parametrize(("folder", "name"), COMMITTED)
def test_the_committed_redrawn_figures_pass_the_text_check(folder: str, name: str):
    docs = ROOT.parent / "docs" / "ml-experiments"
    (path,) = docs.glob(f"{folder}/{name}")
    assert guard.svg_problems(svg=path.read_text()) == [], path


def test_every_guarded_figure_that_exists_passes_the_text_check():
    docs = ROOT.parent / "docs" / "ml-experiments"
    for prefix, name in guard.GUARDED:
        for path in docs.glob(f"{prefix}*/{name}"):
            assert guard.svg_problems(svg=path.read_text()) == [], path


@pytest.mark.parametrize(("folder", "name"), DELETED)
def test_the_deleted_figures_stay_deleted(folder: str, name: str):
    docs = ROOT.parent / "docs" / "ml-experiments"
    assert not list(docs.glob(f"{folder}/{name}"))

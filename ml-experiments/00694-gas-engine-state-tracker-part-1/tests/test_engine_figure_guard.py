"""The guard refuses a dated generator series and accepts an aggregate; every writer uses it."""

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
SCRIPTS: Final[tuple[str, ...]] = (
    "00194-*/e00194_charts.py",
    "00694-*/e00694_report.py",
    "00186-*/e00186_charts.py",
)


def _chart(data: pl.DataFrame) -> Any:
    """Return a chart typed as `Any`, because `ty` loses the type after a `mark_*` call."""
    return alt.Chart(data)


def _series(*, n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    start = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "day": [start + timedelta(days=i) for i in range(n)],
            "index": list(range(n)),
            "power": rng.uniform(0.0, 5.0, n),
        }
    )


def test_a_dated_generator_series_is_refused():
    chart = (
        _chart(_series())
        .mark_line()
        .encode(x=alt.X("day:T", title="Date"), y=alt.Y("power:Q", title="Output"))
    )
    assert any("datetime" in found for found in guard.problems(chart=chart))
    with pytest.raises(guard.GuardError):
        guard.check_chart(chart=chart)


def test_a_per_point_label_and_a_power_unit_are_refused():
    labelled = (
        _chart(_series())
        .mark_line()
        .encode(x=alt.X("index:Q", title="Day"), y=alt.Y("power:Q", title="Output"))
    )
    assert any("per-point" in found for found in guard.problems(chart=labelled))
    unit = (
        _chart(_series())
        .mark_line(aria=False)
        .encode(x=alt.X("index:Q", title="Day"), y=alt.Y("power:Q", title="Output (MW)"))
    )
    assert any("power unit" in found for found in guard.problems(chart=unit))


def test_an_iso_date_in_a_title_is_refused():
    chart = (
        _chart(_series())
        .mark_bar(aria=False)
        .encode(x="index:O", y="power:Q")
        .properties(title="Output from 2024-01-01")
    )
    with pytest.raises(guard.GuardError):
        guard.check_chart(chart=chart)


def test_a_pooled_aggregate_is_accepted():
    pooled = pl.DataFrame({"period": ["Before", "After"], "share": [0.1, 0.4]})
    chart = (
        _chart(pooled)
        .mark_bar(aria=False)
        .encode(x=alt.X("period:N", title=None), y=alt.Y("share:Q", title="Share of half-hours"))
        .properties(title="Share of half-hours flagged as running, pooled")
    )
    assert guard.problems(chart=chart) == []
    guard.check_chart(chart=chart)


def test_save_chart_refuses_a_guarded_path_and_writes_an_aggregate(tmp_path: Path):
    folder = tmp_path / "00194-example"
    folder.mkdir()
    dated = _chart(_series()).mark_line().encode(x="day:T", y="power:Q")
    with pytest.raises(guard.GuardError):
        guard.save_chart(chart=dated, path=folder / "figure_2.svg")
    assert not (folder / "figure_2.svg").exists()
    aggregate = (
        _chart(pl.DataFrame({"band": ["a", "b"], "share": [0.2, 0.8]}))
        .mark_bar(aria=False)
        .encode(x="band:N", y="share:Q")
    )
    guard.save_chart(chart=aggregate, path=folder / "figure_2.svg")
    assert (folder / "figure_2.svg").exists()


def test_the_four_figure_paths_are_guarded():
    names = {
        ("00194-x", "figure_2.svg"),
        ("00694-x", "figure_6_injected_example.svg"),
        ("00694-x", "figure_9_d8_running_share.svg"),
        ("00186-x", "figure_3_spread_and_drops.svg"),
    }
    for folder, name in names:
        assert guard.is_guarded(path=Path(folder) / name)
        assert guard.is_guarded(path=(Path(folder) / name).with_suffix(".png"))
    assert not guard.is_guarded(path=Path("00194-x") / "figure_3.svg")


@pytest.mark.parametrize("pattern", SCRIPTS)
def test_every_writer_in_the_four_scripts_goes_through_the_guard(pattern: str):
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


def test_the_committed_figures_pass_the_text_check():
    docs = ROOT.parent / "docs" / "ml-experiments"
    for folder, name in (
        ("00194-*", "figure_2.svg"),
        ("00694-*", "figure_9_d8_running_share.svg"),
        ("00186-*", "figure_3_spread_and_drops.svg"),
    ):
        (path,) = docs.glob(f"{folder}/{name}")
        assert guard.svg_problems(svg=path.read_text()) == [], path
    assert not list(docs.glob("00694-*/figure_6_injected_example.svg"))

"""Refuse to write a chart that shows a metered generator's record with dates or megawatt values.

NGED has asked that a metered generator's output leaves the private data store only anonymised, and
a gas engine's record is identifiable from public data once it sits on calendar dates. Four figure
files held the engine's record or a series inferred from it. Their writers now go through
`save_chart`, which renders the chart to SVG and raises `GuardError` when the text, the
accessibility labels, or the chart specification carry a date, a datetime axis, a per-point value,
or a power unit.

A path listed in `GUARDED` is checked on every write, so a script cannot switch the check off.
`check_chart` is the same check without writing, for a test.
"""

import io
import re
from pathlib import Path
from typing import Any, Final

import altair as alt

# (experiment folder prefix, file name) of each figure that once held a generator record.
GUARDED: Final[tuple[tuple[str, str], ...]] = (
    ("00194-", "figure_2.svg"),
    ("00694-", "figure_6_injected_example.svg"),
    ("00694-", "figure_9_d8_running_share.svg"),
    ("00186-", "figure_3_spread_and_drops.svg"),
)

_MONTHS: Final[str] = (
    "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    "|January|February|March|April|June|July|August|September|October|November|December"
)
_DAYS: Final[str] = (
    "Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
)
_FORBIDDEN_TEXT: Final[dict[str, re.Pattern[str]]] = {
    "an ISO date or year-month": re.compile(r"\b(?:19|20)\d{2}-\d{2}(?:-\d{2})?\b"),
    "a month name": re.compile(rf"\b(?:{_MONTHS})\b"),
    "a weekday name": re.compile(rf"\b(?:{_DAYS})\b"),
    "a four-digit year": re.compile(r"\b(?:19|20)\d{2}\b"),
    "a power unit": re.compile(r"\b(?:MW|MVA|kW|kVA|GW)\b"),
}
# Vega writes `day: 3; MVA: 6.9` into the label of each mark unless `aria=False`.
_PER_POINT_LABEL: Final[re.Pattern[str]] = re.compile(
    r'aria-label="(?!(?:X-axis|Y-axis|Legend|Title text))[^"]*\w: -?\d'
)
_AXIS_TEXT: Final[re.Pattern[str]] = re.compile(r">([^<>]+)<")


class GuardError(ValueError):
    """Raised when a guarded chart would show a generator record with dates or values."""


def _walk(node: Any) -> list[dict[str, Any]]:
    """Return every dict inside a nested specification."""
    found: list[dict[str, Any]] = []
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            found.append(item)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return found


def svg_problems(*, svg: str) -> list[str]:
    """List what in the SVG text would identify a generator record."""
    found: list[str] = []
    text = " ".join(_AXIS_TEXT.findall(svg))
    for label, pattern in _FORBIDDEN_TEXT.items():
        if pattern.search(text):
            found.append(f"{label} in the chart text")
    if _PER_POINT_LABEL.search(svg):
        found.append("a per-point accessibility label that carries a value")
    return found


def problems(*, chart: alt.TopLevelMixin) -> list[str]:
    """List what in the chart would identify a generator record; an empty list means it is clean."""
    found: list[str] = []
    for node in _walk(chart.to_dict()):
        if node.get("type") == "temporal" or "timeUnit" in node:
            found.append("a datetime axis or time unit in the specification")
            break
    buffer = io.StringIO()
    chart.save(buffer, format="svg")
    return found + svg_problems(svg=buffer.getvalue())


def check_chart(*, chart: alt.TopLevelMixin) -> None:
    """Raise `GuardError` when the chart shows a date, a per-point value, or a power unit."""
    found = problems(chart=chart)
    if found:
        raise GuardError("Refusing to write a chart with " + "; ".join(sorted(set(found))))


def is_guarded(*, path: Path) -> bool:
    """Return whether the path is one of the figure files the guard protects."""
    parts = path.resolve().parts
    return any(
        path.with_suffix(".svg").name == name and any(part.startswith(prefix) for part in parts)
        for prefix, name in GUARDED
    )


def save_chart(*, chart: alt.TopLevelMixin, path: Path, **kwargs: Any) -> None:
    """Write a chart; check it first when the path is a guarded figure."""
    if is_guarded(path=path):
        check_chart(chart=chart)
    chart.save(path, **kwargs)

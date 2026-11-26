# ruff: noqa: E501, PERF401
"""Experiment 733: the numbers of the write-up, printed from `verdict.json` and `readings.json`.

(no torch).

The report prints every number the page quotes, labelled interim until the counted reading is read,
with the
half-width of each counted interval beside it (a C2 that includes zero only because it is wider
than C1 is then
visible), the normalised mean absolute error of the median beside each CRPS contrast, the log loss
beside each
Brier contrast, the two reading words, and the synthetic world's ensemble-coverage counts as data.
A reading
that is `void` is printed as `void`, never as a number. The report prints no date, series
identifier, or generator value.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import e00733_core as core
import e00733_paths as paths  # noqa: F401

PERCENT: Final = 100.0


class ReportError(RuntimeError):
    """A score file the report needs is missing or malformed."""


def read_scores(*, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read `scores/verdict.json` and `scores/readings.json`.

    Args:
        root: The data folder.

    Returns:
        The verdict and the readings.

    Raises:
        ReportError: When either file is missing.
    """
    out = []
    for name in ("verdict.json", "readings.json"):
        path = root / "scores" / name
        if not path.exists():
            msg = f"{path.name} is missing: run `score` first"
            raise ReportError(msg)
        out.append(json.loads(path.read_text()))
    return out[0], out[1]


def format_reading(*, name: str, entry: object) -> str:
    """Return one line for a reading.

    The relative point and interval in percent where the entry has one, else the raw ones.

    A `void` entry is printed as `void`.
    """
    if not isinstance(entry, Mapping):
        return f"{name}: void" if entry == "void" or entry is None else f"{name}: {entry}"
    rel = entry.get("relative")
    if rel is not None:
        line = f"{name}: {PERCENT * rel['point']:+.2f}% of the control's mean (interval {PERCENT * rel['lower']:+.2f}% to {PERCENT * rel['upper']:+.2f}%), half-width {PERCENT * entry['half_width']:.2f}%, {entry['n_rows']} rows, widening x{entry['widening']:.3f}"
    elif "point" in entry:
        line = f"{name}: {entry['point']:+.5f} (interval {entry['lower']:+.5f} to {entry['upper']:+.5f})"
        if "kish_blocks" in entry:
            line += f", Kish blocks {entry['kish_blocks']:.1f}, Kish segments {entry['kish_segments']:.1f}, widening x{entry['widening']:.3f}, {entry['n_rows']} rows"
    else:
        line = f"{name}: {dict(entry)}"
    return line


def render(
    *, verdict: Mapping[str, Any], readings: Mapping[str, Any], counted_read: bool = True
) -> str:
    """Return the report text.

    Args:
        verdict: `verdict.json`.
        readings: `readings.json`.
        counted_read: Whether the counted contrasts were read (false labels every number interim).

    Returns:
        The text, one line per number.
    """
    tag = "" if counted_read else " (interim)"
    lines = [
        f"Experiment {core.EXPERIMENT_ID}: verdict row {verdict['row']}, `{verdict['word']}` (front matter: {verdict['front_matter']}){tag}"
    ]
    lines.append(
        f"Sign-forecast word: {verdict['sign_forecast_word']}. History word: {verdict['history_word']}. {verdict['history_note']}."
    )
    if verdict.get("note_row4"):
        lines.append(f"Note: {verdict['note_row4']}.")
    lines.append(f"Guards: {verdict.get('guards') or 'not read'}")
    lines.append(
        f"Dropped or unfitted arms (their readings are void): {verdict.get('dropped_arms') or 'none'}"
    )
    cov = verdict.get("synthetic_coverage", {})
    lines.append(
        f"Synthetic world, ensemble-run coverage of the base dates (data, no date printed): {cov}"
    )
    for name in [*core.CONTRASTS, *core.READINGS]:
        if name in readings:
            lines.append(format_reading(name=name, entry=readings[name]) + tag)
    for key in ("mae_of_median", "log_loss"):
        for name, value in (readings.get(key) or {}).items():
            lines.append(f"{key} beside {name}: {value}")
    return "\n".join(lines) + "\n"


def write_report(*, root: Path, counted_read: bool = True) -> Path:
    """Write `scores/report.txt` from the score files and return its path."""
    verdict, readings = read_scores(root=root)
    path = root / "scores" / "report.txt"
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(render(verdict=verdict, readings=readings, counted_read=counted_read))
    tmp.rename(path)
    return path

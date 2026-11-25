"""Experiment 730: the printed report, of counts, scores, and verdict inputs (never a date)."""

from collections.abc import Mapping

import e00730_paths  # noqa: F401

# isort: split
import e00730_core as core


def format_contrast(*, name: str, reading: Mapping[str, float]) -> str:
    """Return one line of a contrast: its point, its interval at the Bonferroni level, its rows."""
    return (
        f"{name} {reading['point']:+.5f} [{reading['lower']:+.5f}, {reading['upper']:+.5f}] "
        f"({int(reading['n_rows'])} rows, level {core.LEVEL:.4f})"
    )


def format_verdict(*, record: Mapping[str, object]) -> str:
    """Return the text of the verdict record: each counted contrast, H1, the bars, and the row."""
    lines = [f"verdict row {record['row']}: {record['verdict']} (margin {record['margin']:.5f})"]
    contrasts = record["contrasts"]
    assert isinstance(contrasts, Mapping)
    lines += [format_contrast(name=k, reading=v) for k, v in contrasts.items()]
    h1, det, cal = record["h1"], record["detection"], record["calibration"]
    assert isinstance(h1, Mapping)
    assert isinstance(det, Mapping)
    assert isinstance(cal, Mapping)
    lines.append(
        f"H1 mean {h1['mean_p']:.5f}, share above half {h1['share_above_half']:.5f}, "
        f"pass {h1['pass']}"
    )
    lines.append(f"detection d {det['d']:.4f} (bar {core.DETECTION_BAR}), pass {det['pass']}")
    lines.append(f"calibration reliability {cal['reliability']:.5f}, pass {cal['pass']}")
    return "\n".join(lines)

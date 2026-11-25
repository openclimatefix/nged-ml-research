# ruff: noqa: E501
"""Experiment 721: assemble the verdict from stored readings and print the report.

The module reads the JSON files that the scoring step wrote (counted readings G1 to G3, the flag
results
of the `SG-A` fits in `W2`, and the checks) and applies the verdict table of
`e00721_core.verdict_row`.
It prints no generator series: the report holds contrasts in points, areas, and booleans only.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import e00721_core as core

READING_KEYS: Final = ("g1", "g2")
"""The counted readings that carry a bar against the minimum detectable effect."""


def reading_of(*, body: Mapping, mde: float) -> core.Reading:
    """Return `pass`, `miss`, or `unresolved` for a stored contrast `{point, upper}` against `mde`."""
    return core.bar_g1_g2(point=float(body["point"]), upper=float(body["upper"]), mde=mde)


def any_flagged(*, flags: list[Mapping]) -> bool:
    """Return whether any counted `SG-A` stage in `W2` is flagged.

    A skipped continuation counting as flagged.
    """
    return any(bool(f.get("flagged", False)) or not bool(f.get("evaluated", True)) for f in flags)


def inputs_from_files(*, summary: Mapping, mde: float) -> core.VerdictInputs:
    """Build the verdict table's inputs from the stored summary of the scoring step."""
    stops = core.Stops(**{k: bool(v) for k, v in summary.get("stops", {}).items()})
    checks = [
        core.G3Check(**c) for c in summary.get("g3_checks", [])
    ]  # each check: reading, scope, point, lower, limit
    g3 = core.check_g3(checks=checks) if checks else "unresolved"
    return core.VerdictInputs(
        stops=stops,
        correctness_failed=bool(summary.get("correctness_failed", False)),
        synthetic_failed=bool(summary.get("synthetic_failed", False)),
        synthetic_cap_hit=bool(summary.get("synthetic_cap_hit", False)),
        fit_failed=bool(summary.get("fit_failed", False)),
        g1=reading_of(body=summary["g1"], mde=mde),
        g2=reading_of(body=summary["g2"], mde=mde),
        g3=g3,
        sg_a_flagged=any_flagged(flags=summary.get("flags", [])),
    )


def format_report(*, summary: Mapping, mde: float) -> str:
    """Return the printed report: the verdict row and word, each reading, and the futility test."""
    v = inputs_from_files(summary=summary, mde=mde)
    result = core.verdict_row(v=v)
    lines = [
        f"verdict row {result.row}: {result.verdict}",
        f"correctness failure in the first line: {result.correctness_in_first_line}",
        f"MDE {mde:.3f} points",
    ]
    for key in READING_KEYS:
        body = summary[key]
        lines.append(
            f"{key.upper()}: point {body['point']:+.3f}, upper {body['upper']:+.3f} -> {getattr(v, key)}"
        )
    lines.append(f"G3: {v.g3}")
    lines.append(f"SG-A flagged in W2: {v.sg_a_flagged}")
    lines.append(
        f"futility test fires: {core.futility_fires(g1_point=float(summary['g1']['point']))}"
    )
    lines.append(
        f"seeds 2 to 4 may run: {core.seeds_2_to_4_may_run(result=result, g1_point=float(summary['g1']['point']))}"
    )
    return "\n".join(lines)


def main(*, summary_path: Path, mde_path: Path) -> None:
    """Print the report of the stored summary."""
    summary = json.loads(summary_path.read_text())
    print(format_report(summary=summary, mde=core.read_mde_file(path=mde_path)))

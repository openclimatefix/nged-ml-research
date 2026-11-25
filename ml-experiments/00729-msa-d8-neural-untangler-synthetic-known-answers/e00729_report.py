# ruff: noqa: E501
"""Experiment 729: the printed report (every number the page quotes) and the figure stage.

`render_report` is pure: it turns the aggregate JSON files of the scoring stage into Markdown. It
prints points of capacity, counts, and shares, and never a month, a date, an engine size, a series
identifier, or a per-day real value. Interim numbers are labelled so until the counted reading is
read.
"""

from collections.abc import Mapping
from pathlib import Path

import e00729_core as core
import e00729_figures as figures
import e00729_paths as paths


def fmt(*, value: float, digits: int = 3) -> str:
    """Format a number with a fixed number of decimals."""
    return f"{value:.{digits}f}"


def contrast_lines(*, contrasts: Mapping[str, Mapping[str, Mapping[str, float]]]) -> list[str]:
    """Return one Markdown table row per contrast reading."""
    lines = [
        "| Contrast | Reading | Point | Lower | Upper | Rows | Kish blocks | Kish segments |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, readings in contrasts.items():
        for label, r in readings.items():
            lines.append(
                f"| {name} | {label} | {fmt(value=r['point'])} | {fmt(value=r['lower'])} | "
                f"{fmt(value=r['upper'])} | {int(r['n_rows'])} | {fmt(value=r['kish_blocks'], digits=1)} | "
                f"{fmt(value=r['kish_segments'], digits=1)} |"
            )
    return lines


def score_lines(*, tables: Mapping[str, Mapping[str, Mapping[str, float]]]) -> list[str]:
    """Return the table of signed CRPS, pinball loss, and median error on all, import, and reverse rows."""
    lines = [
        "| System | Subset | Rows | Signed CRPS | Pinball (11 quantiles) | Median error |",
        "|---|---|---|---|---|---|",
    ]
    for system, subsets in tables.items():
        for subset in ("all", "import", "reverse"):
            m = subsets[subset]
            lines.append(
                f"| {system} | {subset} | {int(m['n'])} | {fmt(value=m['crps'])} | "
                f"{fmt(value=m['pinball'])} | {fmt(value=m['mae'])} |"
            )
    return lines


def render_report(
    *, verdict: Mapping, tables: Mapping, gate: Mapping, timing: Mapping | None
) -> str:
    """Return the report as Markdown.

    Args:
        verdict: The contents of `verdict.json`.
        tables: The contents of `scores/tables.json`.
        gate: The contents of `scores/gate.json`.
        timing: The contents of `scores/timing.json`, or None before the timing gate ran.

    Returns:
        The Markdown text.
    """
    out = ["# Experiment 729 printed report (interim until the counted readings are read)", ""]
    out.append(
        f"Verdict row {verdict['row']}: {verdict['verdict']}. Margin m = {fmt(value=verdict['margin'])} points."
    )
    out.append(f"Architecture: {verdict['architecture']}. B*: {verdict['bstar']}.")
    out += ["", "## Realism gate", "", f"Pass: {gate['pass']}.", ""]
    for era, s in gate["shares"].items():
        out.append(
            f"- {era}: real daily-minimum share {fmt(value=s['real_day'])}, synthetic {fmt(value=s['synthetic_day'])}; "
            f"real half-hour share {fmt(value=s['real_half_hour'])}, synthetic {fmt(value=s['synthetic_half_hour'])}."
        )
    out.append(f"- BSP steps: {gate['bsp']}.")
    if timing is not None:
        out += ["", f"Timing gate: {timing}."]
    out += ["", "## Contrasts (points of capacity; negative is a gain)", ""]
    out += contrast_lines(contrasts=verdict["contrasts"])
    out += ["", "## Scores", "", *score_lines(tables=tables["tables"])]
    out += ["", "## Bars and harm checks", ""]
    out.append(f"- Sign calibration: {verdict['calibration']}")
    out.append(
        f"- Lobe coverage: {verdict['lobe']}; mean lobe width {fmt(value=tables['lobe_width_mean'])} points."
    )
    out.append(f"- H1: {verdict['h1']}")
    out.append(f"- Reachability: {verdict['reachability']}")
    out.append(f"- H2: {verdict['real']['h2']}")
    out.append(f"- K1: {verdict['real']['k1']}")
    out.append(f"- K2: {verdict['real']['k2']}")
    out.append(f"- R2: {verdict['real']['r2']}")
    out.append(f"- Shifted-mix read-out: {verdict['shifted']}")
    out.append(f"- q05 and q95 coverage: {tables['coverage']}")
    return "\n".join(out) + "\n"


def write_figures(
    *, verdict: Mapping, tables: Mapping, gate: Mapping, selection: Mapping, out_dir: Path
) -> list[Path]:
    """Write the figures that need only aggregates and return their paths (run `svgo` on each)."""
    written = []
    subsets = ["all", "import", "reverse"]
    scores = {
        name: {s: t[s]["crps"] for s in subsets}
        for name, t in tables["tables"].items()
        if name in ("P0f", "Bstar", "A", "B", "C", "D", "H")
    }
    charts = {
        "figure_4_scores.svg": figures.dot_chart(
            scores=scores,
            rows=subsets,
            title="Signed CRPS on the test banks (smaller is better)",
            x_title="Signed CRPS (points of D8's capacity)",
        ),
        "figure_5_contrasts.svg": figures.contrast_chart(
            readings={
                n: (r["all"]["point"], r["all"]["lower"], r["all"]["upper"])
                for n, r in verdict["contrasts"].items()
            },
            margin=verdict["margin"],
        ),
        "figure_7_reliability.svg": figures.reliability_chart(
            mean_forecast=verdict["calibration"]["mean_forecast"],
            frequency=verdict["calibration"]["frequency"],
        ),
        "figure_9_architectures.svg": figures.architecture_chart(scores=selection["scores"]),
        "figure_3_gate.svg": figures.share_pair_chart(
            groups=list(gate["shares"]),
            synthetic=[s["synthetic_day"] for s in gate["shares"].values()],
            real=[s["real_day"] for s in gate["shares"].values()],
            title="Share of days with a daily minimum below 1.0 MVA, synthetic against real",
        ),
    }
    for name, chart in charts.items():
        path = out_dir / name
        figures.save_svg(chart=chart, path=path)
        written.append(path)
    return written


def report_paths() -> dict[str, Path]:
    """Return the report and figure folders under the private data folder."""
    root = paths.data_root()
    return {"report": root / "scores" / "report.md", "figures": root / "figures"}


def margin_floor() -> float:
    """Return the margin floor the report quotes."""
    return core.MARGIN_FLOOR_POINTS

# ruff: noqa: D103, E501
"""Rewrite the Primaries score tables of the experiment 668 page from the current score files.

uv run --no-sync python e00668_extra_lags_page_tables.py PAGE.md

Each table is found by the line that introduces it (a fixed list below, matched by its text and by
the order of appearance) and replaced by the table that `e00668_tables.py` renders from
`scores_668/` (the pooled scores) or `scores_668/primaries_compare/` (the protocol comparison) or the
slice file. Nothing else in the page is touched.
"""

import sys
from pathlib import Path
from typing import Final

import e00668_figures_dots as dots
import e00668_score_lib as lib
import e00668_tables as tabs
import polars as pl

SCORES: Final = lib.FOLDER / "scores_668"
SITE: Final = next(lib.FOLDER.parent.glob("00521-*/site_map.csv"))

# (introducing line, which occurrence in the page, node, slice, metric)
POOLED: Final = [
    ("Normalised MAE of the median:", 0, "primaries_NRA", "lead band", "mae"),
    ("CRPS (2 × mean pinball loss over the 11 levels):", 0, "primaries_NRA", "lead band", "crps"),
    ("Normalised MAE of the median:", 1, "primaries_prevailing", "lead band", "mae"),
    ("CRPS:", 0, "primaries_prevailing", "lead band", "crps"),
    ("Normalised MAE of the median, normal running:", 0, "primaries_NRA", "season", "mae"),
    ("CRPS, normal running:", 0, "primaries_NRA", "season", "crps"),
    ("CRPS, prevailing:", 0, "primaries_prevailing", "season", "crps"),
    ("CRPS by month of the forecast origin, normal running:", 0, "primaries_NRA", "origin month", "crps"),
    ("CRPS by month of the forecast origin, prevailing:", 0, "primaries_prevailing", "origin month", "crps"),
    ("Normalised MAE of the median, normal running, by Primary:", 0, "primaries_NRA", "series", "mae"),
    ("Normalised MAE of the median, prevailing, by Primary:", 0, "primaries_prevailing", "series", "mae"),
]  # fmt: skip
COMPARE: Final = [
    ("**Six MW Primaries, MAE of the median:**", "signed-MW six", "mae_median"),
    ("**Six MW Primaries, CRPS:**", "signed-MW six", "crps"),
    ("**Six MW Primaries, twCRPS:**", "signed-MW six", "twcrps"),
    ("**Seven MVA-only Primaries, MAE of the median:**", "MVA-only seven", "mae_median"),
    ("**Seven MVA-only Primaries, CRPS:**", "MVA-only seven", "crps"),
    ("**Seven MVA-only Primaries, twCRPS:**", "MVA-only seven", "twcrps"),
]  # fmt: skip


def pooled_table(*, node: str, slice_name: str, metric: str) -> str:
    frame = pl.read_csv(SCORES / node / "scores.csv")
    if slice_name == "series":
        site = pl.read_csv(SITE)
        labels = {str(i): s for i, s in zip(site["time_series_id"], site["label"], strict=True)}
        frame = frame.with_columns(group=pl.col("group").cast(pl.String).replace(labels))
    return tabs.table(frame, slice_name, metric)


def slice_cells(*, d: pl.DataFrame, part: str, metric: str) -> list[str]:
    row = d.filter((pl.col("part") == part) & (pl.col("metric") == metric))
    out = []
    for system in (
        "Backtest, normal running",
        "Backtest, prevailing",
        "Manual heuristic",
        "Climatology",
    ):
        s = row.filter(pl.col("system") == system)
        out.append(f"{s['mean'][0]:.2f}" if s.height and s["mean"][0] is not None else "n/a")
    return out


def slice_table(*, set_name: str, parts: list[str], metric: str, counts: bool) -> str:
    d = pl.read_csv(SCORES / "primaries_slices" / "slices.csv").filter(pl.col("set") == set_name)
    head = "| Slice | Half-hours | Origin days | Normal running | Prevailing | Manual heuristic | Climatology |"
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|"]
    if not counts:
        lines = [
            "| Slice | Normal running | Prevailing | Manual heuristic | Climatology |",
            "|---|---:|---:|---:|---:|",
        ]
    for part in parts:
        sub = d.filter((pl.col("part") == part) & (pl.col("metric") == metric))
        n = sub["rows"][0] if sub.height else 0
        days = sub["origin_days"][0] if sub.height else 0
        cells = slice_cells(d=d, part=part, metric=metric)
        if counts:
            lines.append(f"| {part} | {n:,} | {days} | " + " | ".join(cells) + " |")
        else:
            lines.append(f"| {part} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def replace_table(*, lines: list[str], intro: str, occurrence: int, new: str) -> None:
    hits = [i for i, ln in enumerate(lines) if ln.strip() == intro.strip()]
    start = hits[occurrence]
    i = start + 1
    while not lines[i].startswith("|"):
        i += 1
    j = i
    while j < len(lines) and lines[j].startswith("|"):
        j += 1
    lines[i:j] = new.split("\n")


def main(page: Path) -> None:
    lines = page.read_text().split("\n")
    for intro, occ, node, slice_name, metric in POOLED:
        replace_table(
            lines=lines, intro=intro, occurrence=occ,
            new=pooled_table(node=node, slice_name=slice_name, metric=metric),
        )  # fmt: skip
    for intro, group, metric in COMPARE:
        replace_table(
            lines=lines, intro=intro, occurrence=0,
            new=tabs.compare_table(group=group, slices={**dots.HORIZON, **dots.CONDITION}, metric=metric),
        )  # fmt: skip
    three = ["inside", "7 days after", "outside (same Primaries)"]
    event = "event table, over 7 days, 2 units or more"
    for intro, set_name, parts in (
        ("**Event table, over 7 days, 2 noise units or more** (10 events at 7 Primaries).", event, three),
        ("**Verified transfer, D5 and D16** (2 events).", "verified transfer", three),
        ("**Gate-on origin days** (the G4 gate, fold months with stored gate states).", "gate-on days of experiment 592 (G4)", ["gate-on origin days", "gate-off origin days"]),
    ):  # fmt: skip
        replace_table(
            lines=lines, intro=intro, occurrence=0,
            new=slice_table(set_name=set_name, parts=parts, metric="crps", counts=True),
        )  # fmt: skip
    mae_intro = next(ln for ln in lines if ln.startswith("Normalised MAE of the median by slice"))
    replace_table(
        lines=lines, intro=mae_intro, occurrence=0,
        new=slice_table(set_name=event, parts=three, metric="mae", counts=False),
    )  # fmt: skip
    first = lines.index(mae_intro)
    i = next(k for k in range(first, len(lines)) if lines[k].startswith("|"))
    j = next(k for k in range(i, len(lines)) if not lines[k].startswith("|"))
    k = next(k for k in range(j, len(lines)) if lines[k].startswith("|"))
    m = next(n for n in range(k, len(lines)) if not lines[n].startswith("|"))
    lines[k:m] = slice_table(set_name="verified transfer", parts=three, metric="mae", counts=False).split("\n")
    page.write_text("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

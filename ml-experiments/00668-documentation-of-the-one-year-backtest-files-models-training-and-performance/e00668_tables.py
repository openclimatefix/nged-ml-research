"""Render the scores of one demand node as markdown tables for the experiment 668 page."""

import sys

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FOLDER = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10"


def generator_table(d: pl.DataFrame) -> str:
    """One table for a generator file, which has no manual heuristic."""
    lines = [
        "| Group | Half-hours | Model | Climatology | Gain over climatology (%) |",
        "|---|---:|---:|---:|---|",
    ]
    for g in d["group"].unique(maintain_order=True):
        s = d.filter(pl.col("group") == g)
        m = s.filter(pl.col("system") == "Model").row(0, named=True)
        c = s.filter(pl.col("system") == "Climatology").row(0, named=True)
        lines.append(
            f"| {g} | {m['rows']:,} | {m['mean']:.2f} | {c['mean']:.2f} "
            f"| {c['gain_pct']:.1f} ({c['gain_lo']:.1f} to {c['gain_hi']:.1f}) |"
        )
    return "\n".join(lines)


def table(df: pl.DataFrame, slice_name: str, metric: str) -> str:
    """One table: a row per group, columns for the three systems and the two gains."""
    d = df.filter((pl.col("slice") == slice_name) & (pl.col("metric") == metric))
    if "Manual heuristic" not in set(d["system"]):
        return generator_table(d)
    lines = [
        (
            "| Group | Half-hours | Model | Manual heuristic | Climatology "
            "| Gain over heuristic (%) | Gain over climatology (%) |"
        ),
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for g in d["group"].unique(maintain_order=True):
        s = d.filter(pl.col("group") == g)
        m = s.filter(pl.col("system") == "Model").row(0, named=True)
        h = s.filter(pl.col("system") == "Manual heuristic").row(0, named=True)
        c = s.filter(pl.col("system") == "Climatology").row(0, named=True)
        lines.append(
            f"| {g} | {m['rows']:,} | {m['mean']:.2f} | {h['mean']:.2f} | {c['mean']:.2f} "
            f"| {h['gain_pct']:.1f} ({h['gain_lo']:.1f} to {h['gain_hi']:.1f}) "
            f"| {c['gain_pct']:.1f} ({c['gain_lo']:.1f} to {c['gain_hi']:.1f}) |"
        )
    return "\n".join(lines)


COMPARE_ARMS = {
    "backtest_NRA": "Backtest, normal running",
    "backtest_prevailing": "Backtest, prevailing",
    "default_3": "Default 3",
    "bg2": "BG2",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}


def compare_table(*, group: str, slices: dict[str, str], metric: str) -> str:
    """One table of the protocol comparison: a row per slice, a column per forecast."""
    d = pl.read_csv(FOLDER / "scores_668" / "primaries_compare" / "scores.csv").filter(
        (pl.col("group") == group) & (pl.col("metric") == metric)
    )
    lines = [
        "| Slice | Half-hours | " + " | ".join(COMPARE_ARMS.values()) + " |",
        "|---|---:|" + "---:|" * len(COMPARE_ARMS),
    ]
    for key, name in slices.items():
        s = d.filter(pl.col("slice") == key)
        cells = []
        for arm in COMPARE_ARMS:
            v = s.filter(pl.col("arm") == arm)
            cells.append(f"{v['mean'][0]:.2f}" if v.height else "")
        n = s.filter(pl.col("arm") == "default_3")["half_hours"]
        lines.append(f"| {name} | {n[0]:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        from e00668_figures_dots import CONDITION, HORIZON

        group, block, metric = sys.argv[2:5]
        print(
            compare_table(
                group=group, slices=HORIZON if block == "horizon" else CONDITION, metric=metric
            )
        )
    else:
        name, slice_name, metric = sys.argv[1:4]
        frame = pl.read_csv(FOLDER / "scores_668" / name / "scores.csv")
        if name.startswith("primaries") and slice_name == "series":
            site = pl.read_csv(next(FOLDER.parent.glob("00521-*/site_map.csv")))
            labels = {str(i): s for i, s in zip(site["time_series_id"], site["label"], strict=True)}
            frame = frame.with_columns(group=pl.col("group").cast(pl.String).replace(labels))
        print(table(frame, slice_name, metric))

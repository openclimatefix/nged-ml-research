# ruff: noqa: D103, E501
"""Experiment 677: write the regime table of the ladder as Markdown (private).

Usage: python e00677_regime_table.py OUT.md [TITLE_PREFIX]

Reads `scores/slices_sp3_s01234.csv` and writes one table per metric (MAE of the median and CRPS, points
of capacity): a row per regime slice, a column per ladder step, the manual heuristic, and climatology.
"""

import sys
from pathlib import Path

import e00677_score as sc
import polars as pl

GROUPS = {
    "ladder (quarterly refits, trained with injected events from step 5)": {
        "step1": "1",
        "step2": "2",
        "step3a": "3a",
        "step3b": "3b",
        "step4": "4",
        "step5w": "5",
        "step6w": "6",
        "step7w": "7",
        "step8w": "8",
        "Manual heuristic": "Manual heuristic",
        "Climatology": "Climatology",
    },
    "extra arms (quarterly refits, trained without injected events; monthly refits, seeds 0 to 2)": {
        "step5wo": "5 without",
        "step6wo": "6 without",
        "step7wo": "7 without",
        "step8wo": "8 without",
        "step5w@1": "9a",
        "step6w@1": "6 monthly",
        "step7w@1": "7 monthly",
        "step8w@1": "9b",
        "step5wo@1": "5 without monthly",
        "step8wo@1": "8 without monthly",
    },
}


def main() -> None:
    out, prefix = Path(sys.argv[1]), (sys.argv[2] + " " if len(sys.argv) > 2 else "")
    t = pl.read_csv(sc.ROOT / "scores" / "slices_sp3_s01234.csv")
    lines = [
        f"# {prefix}Regime table of the build-up ladder (points of capacity; smaller is better)",
        "",
    ]
    for group, order in GROUPS.items():
        for metric, name in (("mae", "MAE of the median"), ("crps", "CRPS")):
            lines += [
                f"## {name}: {group}",
                "",
                "| Slice | Rows | " + " | ".join(order.values()) + " |",
                "|---|---|" + "---|" * len(order),
            ]
            keys = (
                t.filter(pl.col("metric") == metric)
                .select("set", "part", "rows")
                .unique(maintain_order=True)
            )
            for set_name, part, rows in keys.iter_rows():
                if rows == 0:
                    lines.append(f"| {set_name}: {part} | 0 |" + " |" * len(order))
                    continue
                sub = t.filter(
                    (pl.col("metric") == metric)
                    & (pl.col("set") == set_name)
                    & (pl.col("part") == part)
                )
                cells = []
                for system in order:
                    r = sub.filter(pl.col("system") == system)
                    cells.append(f"{r['mean'][0]:.2f}" if r.height else "")
                lines.append(f"| {set_name}: {part} | {rows:,} | " + " | ".join(cells) + " |")
            lines.append("")
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()

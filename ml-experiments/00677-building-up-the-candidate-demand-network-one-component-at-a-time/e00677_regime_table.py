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

ORDER = ["step1", "step2", "step3a", "step3b", "step4", "step5w", "step5wo", "step6w", "step6wo", "step7w", "step7wo", "step8w", "step8wo", "Manual heuristic", "Climatology"]


def main() -> None:
    out, prefix = Path(sys.argv[1]), (sys.argv[2] + " " if len(sys.argv) > 2 else "")
    t = pl.read_csv(sc.ROOT / "scores" / "slices_sp3_s01234.csv")
    lines = [f"# {prefix}Regime table of the build-up ladder (points of capacity; smaller is better)", ""]
    for metric, name in (("mae", "MAE of the median"), ("crps", "CRPS")):
        lines += [f"## {name}", "", "| Slice | Rows | " + " | ".join(ORDER) + " |", "|---|---|" + "---|" * len(ORDER)]
        keys = t.filter(pl.col("metric") == metric).select("set", "part", "rows").unique(maintain_order=True)
        for set_name, part, rows in keys.iter_rows():
            if rows == 0:
                lines.append(f"| {set_name}: {part} | 0 |" + " |" * len(ORDER))
                continue
            sub = t.filter((pl.col("metric") == metric) & (pl.col("set") == set_name) & (pl.col("part") == part))
            cells = []
            for system in ORDER:
                r = sub.filter(pl.col("system") == system)
                cells.append(f"{r['mean'][0]:.2f}" if r.height else "")
            lines.append(f"| {set_name}: {part} | {rows:,} | " + " | ".join(cells) + " |")
        lines.append("")
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()

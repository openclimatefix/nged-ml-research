"""Score the students of experiment 490 beside the stored references, on the clean world.

Reuses experiment 454's row classes, widening variants, and tail summaries (`e00454_score`) by
pointing its prediction paths at this experiment's folder for the student arms and at experiment
454's stored constructs for the references (A0-inj, S1, B-mix, K2-S1). The writes go to this
experiment's `work/scores/`.

`cells` writes one row per arm, month, cell, variant, basis, group, and row class. `contrasts` reads
those and writes the contrasts of each student against K2-S1 and against B-mix, with a month-block
bootstrap interval at the Bonferroni level of the 18 contrasts the plan fixes (2 student arms times
9 tests), plus the minimum detectable effect from the seed spread of B-mix.

```bash
PYTHONPATH=<as e00454_score.py shows, plus ml-experiments/00454-*:ml-experiments/00435-*:\
ml-experiments/00490-*> DATA_PATH_INTERNAL=... uv run --no-sync python \
    ml-experiments/00490-*/e00490_score.py cells
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00454_build as build  # ty: ignore[unresolved-import]
import e00454_report as report454  # ty: ignore[unresolved-import]
import e00454_score as scoring  # ty: ignore[unresolved-import]
import e00490_distill as distill  # ty: ignore[unresolved-import]
import polars as pl

SCORE_DIR: Final[Path] = distill.WORK_DIR / "scores"
STUDENTS: Final[tuple[str, ...]] = ("D-K2", "D-Bmix", "D-K2-corr")
REFERENCES: Final[tuple[str, ...]] = ("A0-inj", "S1", "B-mix", "K2-S1")
CONTRAST_ARMS: Final[tuple[str, ...]] = ("D-K2", "D-Bmix")
BONFERRONI: Final[int] = 18
HEADLINE_CLASSES: Final[tuple[str, ...]] = ("clean", "harm", "event")
_original_prediction_path = build.prediction_path


def route_prediction_path(*, arm: str, month: str, cell: int) -> Path:
    """Return a student's forecast file in this experiment's folder, else experiment 454's."""
    if arm in STUDENTS or arm == "D-0":
        return distill.PRED_DIR / arm / f"{month}_s{cell}_w-1.parquet"
    return _original_prediction_path(arm=arm, month=month, cell=cell)


def install() -> None:
    """Point experiment 454's scoring at the student forecasts and this experiment's folder."""
    build.prediction_path = route_prediction_path
    scoring.SCORE_DIR = SCORE_DIR


def cells() -> None:
    """Score every student and reference cell and write the long table."""
    install()
    bases = {m: scoring.month_base(month=m) for m in build.MONTHS}
    for m, b in bases.items():
        counts = b.group_by("group", "cls").len().sort("group", "cls")
        print(m, {(r["group"], r["cls"]): r["len"] for r in counts.iter_rows(named=True)})
    parts: list[pl.DataFrame] = []
    for arm in (*STUDENTS, *REFERENCES):
        n_cells = 3
        for cell in range(n_cells):
            parts.extend(scoring.score_arm_cell(arm=arm, cell=cell, bases=bases))
        print("scored", arm, flush=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(parts).write_parquet(SCORE_DIR / "cells.parquet")


def contrasts() -> None:
    """Write each student's pooled contrasts against K2-S1 and B-mix, and the minimum effects."""
    install()
    scores = report454.month_scores()
    mde = report454.minimum_detectable_effect()
    mde.write_csv(SCORE_DIR / "mde.csv", float_precision=4)
    level = 1.0 - 0.05 / BONFERRONI
    rows = []
    for reference in ("K2-S1", "B-mix", "A0-inj"):
        value_columns = ["crps", *[f"{k}_{t}" for k in ("cov", "pin") for t in report454.TAILS]]
        for column in value_columns:
            refc = scores.filter(pl.col("arm") == reference).select(
                "month", "variant", "basis", "group", "cls", ref=column, ref_n="n"
            )
            joined = scores.filter(pl.col("arm").is_in([*STUDENTS, *REFERENCES])).join(
                refc, on=["month", "variant", "basis", "group", "cls"]
            )
            for key, group_part in joined.group_by(
                "arm", "variant", "basis", "group", "cls", maintain_order=True
            ):
                part = group_part.sort("month")
                assert (part["n"] == part["ref_n"]).all(), f"{key}: rows differ from {reference}"
                diff = (part[column] - part["ref"]).to_numpy()
                low, high = report454.month_block_interval(diffs=diff, level=level)
                rows.append(
                    {
                        "arm": key[0], "variant": key[1], "basis": key[2], "group": key[3],
                        "cls": key[4], "reference": reference, "metric": column,
                        "months": len(diff), "absolute": float(part[column].mean()),  # ty: ignore[invalid-argument-type]
                        "reference_value": float(part["ref"].mean()),  # ty: ignore[invalid-argument-type]
                        "diff": float(diff.mean()), "low": low, "high": high,
                        "months_better": int((diff < 0).sum()),
                    }
                )  # fmt: skip
    frame = pl.DataFrame(rows)
    frame.write_csv(SCORE_DIR / "contrasts.csv", float_precision=4)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    print(mde)
    print(
        frame.filter(
            (pl.col("metric") == "crps")
            & (pl.col("variant") == "raw")
            & (pl.col("basis") == "metered")
            & (pl.col("group") != "report_only")
            & (pl.col("arm") != pl.col("reference"))
        ).sort("reference", "group", "cls", "arm")
    )


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cells", "contrasts"))
    args = parser.parse_args()
    {"cells": cells, "contrasts": contrasts}[args.command]()


if __name__ == "__main__":
    main()

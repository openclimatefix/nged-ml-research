"""Pool the contrasts of experiment 404 across months and seeds.

For each arm, row class, and Primary group, the table lists the score difference against the
reference arm in every scored month and seed where both arms have forecasts, then the mean of those
differences. A cell whose real class holds a single event has no meaningful interval, so the cell is
marked `*`. Seeds within a month are averaged first, so the pooled mean is a mean of month means
(`e00404_audit.py` adds the event-weighted pooling).

```bash
PYTHONPATH=<as e00404_fit.py shows> \
    uv run --no-sync python ml-experiments/00404-*/e00404_pool.py \
    --arms B0 B1 B3 B6 --reference B-mix
```
"""

import argparse
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_report as report  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

CELLS: Final[tuple[tuple[str, int], ...]] = (
    ("2026-06", 0),
    ("2026-06", 1),
    ("2026-06", 2),
    ("2026-03", 0),
    ("2026-01", 0),
    ("2025-12", 0),
    ("2025-10", 0),
)
BOOTSTRAP: Final[int] = 2000
CLASSES: Final[tuple[str, ...]] = ("harm_real", "tgt_real", "clean_real", "harm_inj", "tgt_inj")


def has_forecasts(*, arm: str, month: str, seed: int, worlds: list[int]) -> bool:
    """Return whether an arm's forecasts exist for the cell on every world."""
    return all(
        score.prediction_path(arm=arm, month=month, seed=seed, world=w).exists() for w in worlds
    )


def pool(*, arms: list[str], reference: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the contrasts and the absolute scores, one row per arm, group, class, and cell."""
    real = score.real_event_mask()
    worlds = [*score.SCORING_SEEDS, -1]
    rows, absolute = [], []
    for month, seed in CELLS:
        usable = [
            a
            for a in [reference, *arms]
            if has_forecasts(arm=a, month=month, seed=seed, worlds=worlds)
        ]
        if reference not in usable:
            continue
        frames = {a: report.load(arm=a, month=month, seed=seed, worlds=worlds) for a in usable}
        ref = frames[reference]
        clusters = report.cluster_ids(frame=ref, real=real)
        for arm in usable:
            for group in ("signed_six", "mva_eight"):
                for cls in CLASSES:
                    mask = ((ref["group"] == group) & (ref["cls"] == cls)).to_numpy()
                    if mask.sum() == 0:
                        continue
                    absolute.append(
                        {
                            "arm": arm,
                            "group": group,
                            "cls": cls,
                            "month": month,
                            "seed": seed,
                            "score": float(frames[arm].filter(pl.Series(mask))["crps"].mean()),  # ty: ignore[invalid-argument-type]
                        }
                    )
            if arm == reference:
                continue
            for group in ("signed_six", "mva_eight"):
                for cls in CLASSES:
                    mask = ((ref["group"] == group) & (ref["cls"] == cls)).to_numpy()
                    if mask.sum() == 0:
                        continue
                    point, low, high = report.contrast(
                        a=frames[arm].filter(pl.Series(mask)),
                        b=ref.filter(pl.Series(mask)),
                        cluster=clusters[mask],
                    )
                    rows.append(
                        {
                            "arm": arm,
                            "group": group,
                            "cls": cls,
                            "month": month,
                            "seed": seed,
                            "diff": point,
                            "low": low,
                            "high": high,
                            "single_event": bool(np.isclose(low, high)),
                        }
                    )
    return pl.DataFrame(rows), pl.DataFrame(absolute)


def month_block_interval(*, values: pl.DataFrame, column: str) -> tuple[float, float, float]:
    """Return the mean of a column over months (seeds within a month averaged) and a 95% interval.

    The interval resamples whole months with replacement, so it is wide when few months exist.
    """
    per_month = values.group_by("month").agg(pl.col(column).mean())[column].to_numpy()
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(per_month), size=(BOOTSTRAP, len(per_month)))
    estimates = per_month[draws].mean(1)
    return (
        float(per_month.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def main() -> None:
    """Print the pooled table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--reference", default="B-mix")
    args = parser.parse_args()
    frame, absolute = pool(arms=args.arms, reference=args.reference)
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_cols(20)
    lines = []
    for (arm, group, cls), part in frame.group_by("arm", "group", "cls", maintain_order=True):
        mean, low, high = month_block_interval(values=part, column="diff")
        score_part = absolute.filter(
            (pl.col("arm") == arm) & (pl.col("group") == group) & (pl.col("cls") == cls)
        )
        lines.append(
            {
                "group": group,
                "cls": cls,
                "arm": arm,
                "cells": len(part),
                "months": part["month"].n_unique(),
                "absolute": float(score_part["score"].mean()),  # ty: ignore[invalid-argument-type]
                "minus_ref": mean,
                "low": low,
                "high": high,
                "cells_better": int((part["diff"] < 0).sum()),
                "single_event_cells": int(part["single_event"].sum()),
            }
        )
    ref_rows = []
    for row in lines:
        arm_cells = frame.filter(
            (pl.col("arm") == row["arm"])
            & (pl.col("group") == row["group"])
            & (pl.col("cls") == row["cls"])
        ).select("month", "seed")
        ref_part = absolute.filter(
            (pl.col("arm") == args.reference)
            & (pl.col("group") == row["group"])
            & (pl.col("cls") == row["cls"])
        ).join(arm_cells, on=["month", "seed"])
        ref_rows.append(float(ref_part["score"].mean()))  # ty: ignore[invalid-argument-type]
    summary = pl.DataFrame(lines).with_columns(reference_absolute=pl.Series(ref_rows))
    summary = summary.sort("group", "cls", "arm")
    summary.write_csv(fit.WORK_DIR / f"pooled_summary_vs_{args.reference}.csv", float_precision=3)
    print(f"-> {fit.WORK_DIR / f'pooled_summary_vs_{args.reference}.csv'}")
    out = fit.WORK_DIR / f"pooled_vs_{args.reference}.parquet"
    frame.write_parquet(out)
    absolute.write_parquet(fit.WORK_DIR / "pooled_absolute.parquet")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

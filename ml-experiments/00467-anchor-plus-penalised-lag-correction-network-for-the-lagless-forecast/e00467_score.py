"""Score experiment 467's correction arms against the anchor (LL) and B-mix, with 454's scorer.

Reuses experiment 454's row base, widening and table-P variants, tail summaries, month-block
intervals and minimum detectable effect. The arms are the three penalties (fit seeds 0 and 1), and
the references LL (the anchor alone), B-mix, and A0-inj, each at fit seeds 0 and 1 from experiment
454's stored forecasts. Only the real world is scored (world -1), because experiment 454's stored
references exist for it only.

```bash
<PYTHONPATH as e00467_arm.py shows, plus the 454 and 435 folders> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00467-*/e00467_score.py
```
"""

from pathlib import Path
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_report as report404  # ty: ignore[unresolved-import]
import e00454_build as build  # ty: ignore[unresolved-import]
import e00454_report as report  # ty: ignore[unresolved-import]
import e00454_score as scoring  # ty: ignore[unresolved-import]
import e00467_arm as arm_mod  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SCORE_DIR: Final[Path] = arm_mod.WORK_DIR / "scores"
CELLS: Final[tuple[int, ...]] = (0, 1)
ARMS: Final[tuple[str, ...]] = (
    "C1e-3",
    "C1e-2",
    "C1e-1",
    "LL",
    "B-mix",
    "A0-inj",
    "S1",
    "S1-mix",
    "C1",
    "Z1e-2",
)
_ORIGINAL = build.prediction_path


def prediction_path(*, arm: str, month: str, cell: int) -> Path:
    """Return a stored forecast file: experiment 467's own for the C arms, 454's otherwise."""
    if arm.startswith(("C", "Z")):
        return fit.PRED_DIR / arm / f"{month}_s{cell}_w-1.parquet"
    return _ORIGINAL(arm=arm, month=month, cell=cell)


def shift_summary() -> pl.DataFrame:
    """Return the share of rows whose shift is below 0.01 of capacity, per penalty and group."""
    sites = fit.prep.primaries()["site"].to_list()
    groups = report404.group_of(sites=sites)
    rows = []
    for arm in (*ARMS[:3], "C1", "Z1e-2"):
        for month in build.MONTHS:
            for cell in CELLS:
                frame = pl.read_parquet(prediction_path(arm=arm, month=month, cell=cell))
                group = groups[frame["primary"].to_numpy().astype(np.int64)]
                small = (frame["shift_capacity"].abs() < arm_mod.ZERO_SHIFT_CAPACITY).to_numpy()
                for g in np.unique(group):
                    sel = group == g
                    rows.append(
                        {
                            "arm": arm, "month": month, "cell": cell, "group": str(g),
                            "share_below_001": float(small[sel].mean()),
                            "mean_abs_shift": float(
                                frame["shift_capacity"].abs().to_numpy()[sel].mean()
                            ),
                            "mean_log_scale_change": float(
                                frame["log_scale_change"].to_numpy()[sel].mean()
                            ),
                        }
                    )  # fmt: skip
    return pl.DataFrame(rows)


def score_all(*, band: tuple[float, float] | None) -> pl.DataFrame:
    """Score every arm cell, over all leads or over a lead band in hours."""
    bases = {m: scoring.month_base(month=m) for m in build.MONTHS}
    if band is not None:
        bases = {
            m: b.filter(pl.col("lead_hours").is_between(band[0], band[1])) for m, b in bases.items()
        }
    parts: list[pl.DataFrame] = []
    for arm in ARMS:
        for cell in CELLS:
            parts.extend(scoring.score_arm_cell(arm=arm, cell=cell, bases=bases))
        print("scored", arm, flush=True)
    return pl.concat(parts)


def main() -> None:
    """Write the score, contrast, and shift tables."""
    build.prediction_path = prediction_path
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    shift = shift_summary()
    shift.write_csv(SCORE_DIR / "shift.csv", float_precision=4)
    print(shift.group_by("arm", "group").agg(pl.col("share_below_001").mean()).sort("arm", "group"))
    value_columns = ["crps", *[f"{k}_{t}" for k in ("cov", "pin") for t in report.TAILS]]
    for name, band in (("all", None), ("lead_12_36", (12.0, 36.0))):
        cells = score_all(band=band)
        cells.write_parquet(SCORE_DIR / f"cells_{name}.parquet")
        absolute = (
            cells.filter((pl.col("variant") == "raw") & (pl.col("basis") == "metered"))
            .group_by("arm", "month", "group", "cls")
            .agg(pl.col("crps").mean())
            .group_by("arm", "group", "cls")
            .agg(pl.col("crps").mean())
            .sort("group", "cls", "arm")
        )
        absolute.write_csv(SCORE_DIR / f"absolute_{name}.csv", float_precision=4)
        print(absolute.filter(pl.col("group") != "report_only"))
        for reference in ("LL", "B-mix"):
            report.REFERENCE = reference
            keep = [*ARMS[:3], "C1", "Z1e-2", reference]
            scores = (
                cells.filter(pl.col("arm").is_in(keep))
                .group_by("arm", "month", "variant", "basis", "group", "cls")
                .agg(pl.col("n").first(), *[pl.col(c).mean() for c in value_columns])
            )
            crps = report.contrasts(scores=scores)
            crps.write_csv(
                SCORE_DIR / f"contrasts_crps_{name}_vs_{reference}.csv", float_precision=4
            )
            tails = pl.concat(
                [
                    report.contrasts(scores=scores, column=f"{k}_{t}").with_columns(
                        metric=pl.lit(f"{k}_{t}")
                    )
                    for k in ("pin", "cov")
                    for t in report.TAILS
                ]
            )
            tails.write_csv(
                SCORE_DIR / f"contrasts_tails_{name}_vs_{reference}.csv", float_precision=4
            )
            scores.write_csv(
                SCORE_DIR / f"month_scores_{name}_vs_{reference}.csv", float_precision=4
            )
            print(name, "vs", reference)
            print(
                crps.filter(
                    (pl.col("variant") == "raw")
                    & (pl.col("basis") == "metered")
                    & (pl.col("group") != "report_only")
                )
                .sort("group", "cls", "arm")
                .select(
                    "group", "cls", "arm", "absolute", "reference", "diff", "low", "high", "better"
                )
            )


if __name__ == "__main__":
    main()

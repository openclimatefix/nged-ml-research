"""Tail calibration of experiment 495's arms: coverage and pinball loss at q01, q05, q95, q99.

Reads each arm's forecasts for the seven cells (five months at seed 0; 2026-06 at seeds 0 to 2),
takes the signed-MW six, and reports for rows with no labelled event (clean real world) and for
injected harm rows (w200 to w202) the mean over cells of: the share of actuals at or below each
quantile (coverage), the pinball loss in points of capacity, and the number of rows. Raw
quantiles, and quantiles widened by 1.15 about the median, are both reported.

```bash
PYTHONPATH=<as e00495_fit.py shows> \
    uv run --no-sync python ml-experiments/00495-*/e00495_tails.py
```
"""

from typing import Final

import e00404_report as report  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00495_pool_cells as cells  # ty: ignore[unresolved-import]
import e00495_score as score495  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

LEVELS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
BANDS: Final[dict[str, tuple[float, float]]] = {"all": (0.0, 1e9), "12-36 h": (12.0, 36.0)}
WIDENING: Final[float] = 1.15
NEW = ("HN",)
ARMS: Final[tuple[str, ...]] = ("A0-inj", "B-mix", *NEW, *(f"{a}+lagless" for a in NEW))


def tails(*, frame: pl.DataFrame, widen: float) -> dict[str, float]:
    """Return coverage and pinball loss (points of capacity) at the four tail levels."""
    out = {}
    median = frame["q50"].to_numpy()
    actual = frame["actual"].to_numpy()
    for name, level in LEVELS.items():
        q = median + widen * (frame[name].to_numpy() - median)
        out[f"{name}_cover"] = float(np.mean(actual <= q))
        diff = actual - q
        out[f"{name}_pinball"] = float(
            100.0 * np.mean(np.maximum(level * diff, (level - 1.0) * diff))
        )
    return out


def main() -> None:
    """Print the table and write it to `work/scores/tails.csv`."""
    score495.point_at_495()
    rows = []
    for month, seed in cells.CELLS:
        for arm in ARMS:
            frame = report.load(arm=arm, month=month, seed=seed, worlds=[*score.SCORING_SEEDS, -1])
            frame = frame.filter(
                (pl.col("group") == "signed_six") & pl.col("cls").is_in(["clean_real", "harm_inj"])
            )
            for cls, part in frame.group_by("cls"):
                for band, (low, high) in BANDS.items():
                    sel = part.filter(
                        (pl.col("lead_hours") >= low) & (pl.col("lead_hours") <= high)
                    )
                    rows.extend(
                        {"arm": arm, "cls": cls[0], "band": band, "widen": widen,
                         "month": month, "seed": seed, "rows": sel.height,
                         **tails(frame=sel, widen=widen)}
                        for widen in (1.0, WIDENING)
                    )  # fmt: skip
    table = (
        pl.DataFrame(rows)
        .group_by("arm", "cls", "band", "widen", "month", maintain_order=True)
        .agg(
            pl.col("rows").sum(),
            pl.exclude("arm", "cls", "band", "widen", "month", "rows", "seed").mean(),
        )
    )
    mean = table.group_by("arm", "cls", "band", "widen", maintain_order=True).agg(
        pl.col("rows").sum(),
        pl.exclude("arm", "cls", "band", "widen", "month", "rows").mean(),
    )
    out = score495.fit495.WORK_DIR / "scores" / "tails.csv"
    mean.write_csv(out, float_precision=4)
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=200):
        print(mean.filter((pl.col("band") == "all") & (pl.col("widen") == 1.0)).sort("cls", "arm"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

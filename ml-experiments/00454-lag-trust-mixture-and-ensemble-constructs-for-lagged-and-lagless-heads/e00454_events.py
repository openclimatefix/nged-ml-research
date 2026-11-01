"""Event-weighted pooling and largest-event removal for experiment 454, at the signed-MW six.

Rows of the `event` and `harm` classes are grouped into real events with experiment 404's
`cluster_ids` (the real event touched by the row's lags). A construct's score on a row is the
mean over its cells of the cell's row score, so a row's contrast against B-mix is a mean over cells
against a mean over B-mix cells. Three poolings are printed, per class and construct:

- **row pooling** (every row equal, the mean over the pooled rows of all five months);
- **event pooling** (every event equal: the mean over events of the event's mean contrast);
- **event pooling without the largest event** (the event whose rows' contrast sums to the largest
  absolute value is removed first).

The interval resamples events with replacement (5,000 draws, 95%, unadjusted). Months with a
single event in the class are listed apart.

```bash
<PYTHONPATH as e00454_build.py shows> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00454-*/e00454_events.py
```
"""

from typing import Final

import e00404_report as report  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00454_build as build  # ty: ignore[unresolved-import]
import e00454_score as scoring  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

ARMS: Final[tuple[str, ...]] = ("S1-mix", "K2-S1", "K3-S1", "K5-S1", "K3-A", "A-ens2", "A0-inj")
CLASSES: Final[tuple[str, ...]] = ("event", "harm")
DRAWS: Final[int] = 5_000


def row_scores(*, arm: str, month: str, base: pl.DataFrame) -> np.ndarray:
    """Return the mean over cells of the row score (points of capacity) of a construct."""
    n_cells = scoring.construct_cells()[arm]
    total = np.zeros(base.height)
    for cell in range(n_cells):
        frame = scoring.load_cell(base=base, arm=arm, month=month, cell=cell)
        total += frame.select(score.pinball_points(frame=frame)).to_series().to_numpy()
    return total / n_cells


def interval(
    *, sums: np.ndarray, counts: np.ndarray, per_event: bool
) -> tuple[float, float, float]:
    """Return the pooled mean contrast and its 95% interval, resampling events."""
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(sums), size=(DRAWS, len(sums)))
    if per_event:
        point = float((sums / counts).mean())
        est = (sums / counts)[draws].mean(1)
    else:
        point = float(sums.sum() / counts.sum())
        est = sums[draws].sum(1) / counts[draws].sum(1)
    return point, float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))


def main() -> None:
    """Print the three poolings."""
    real = score.real_event_mask()
    frames = []
    for month in build.MONTHS:
        base = scoring.month_base(month=month).filter(pl.col("group") == "signed_six")
        ref = row_scores(arm="B-mix", month=month, base=base)
        full = base.with_columns(world=pl.lit(-1), month=pl.lit(month))
        cluster = report.cluster_ids(frame=full, real=real)
        data = {
            "month": np.full(base.height, month),
            "cls": base["cls"].to_numpy(),
            "cluster": cluster,
        }
        for arm in ARMS:
            data[arm] = row_scores(arm=arm, month=month, base=base) - ref
        frames.append(pl.DataFrame(data))
        print("done", month, flush=True)
    rows = pl.concat(frames).with_columns(
        event=pl.col("month") + "/" + pl.col("cluster").cast(pl.String)
    )
    out = []
    for cls in CLASSES:
        part = rows.filter(pl.col("cls") == cls)
        per_month = part.group_by("month").agg(pl.col("event").n_unique().alias("events"))
        print(cls, "events per month:", dict(per_month.sort("month").iter_rows()))
        for arm in ARMS:
            grouped = part.group_by("event").agg(
                pl.col(arm).sum().alias("sum"), pl.len().alias("count")
            )
            sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy().astype(float)
            largest = int(np.argmax(np.abs(sums)))
            keep = np.arange(len(sums)) != largest
            for label, s, c, per_event in (
                ("rows", sums, counts, False),
                ("events", sums, counts, True),
                ("events, largest removed", sums[keep], counts[keep], True),
            ):
                point, low, high = interval(sums=s, counts=c, per_event=per_event)
                out.append(
                    {"cls": cls, "arm": arm, "pooling": label, "events": len(s),
                     "diff": point, "low": low, "high": high}
                )  # fmt: skip
    table = pl.DataFrame(out)
    table.write_csv(scoring.SCORE_DIR / "event_pooling.csv", float_precision=4)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_width_chars(160)
    print(table)


if __name__ == "__main__":
    main()

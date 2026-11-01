"""Audit checks of experiment 460: median minus mean forecast per Primary, and mean-forecast error.

The mean of a forecast is the mean of its density mixture, read from the cached heads, so the
check needs no network. For each Primary, `median_minus_mean` is the mean over rows of the median
(q50) minus the mixture mean, in points of capacity; `mean_abs_error` is the mean absolute error of
the mean forecast, by row class, so it can be read beside the quantile-score contrasts.

```bash
source /tmp/e460_env.sh
uv run --no-sync python ml-experiments/00460-*/e00460_checks.py
```
"""

from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_report as report  # ty: ignore[unresolved-import]
import e00460_arms as arms  # ty: ignore[unresolved-import]
import e00460_score as scoring  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

HEADS: Final[dict[str, list[tuple[str, float]]]] = {
    "B-mix": [("A", 0.5), ("L", 0.5)],
    "S1": [("S", 1.0)],
    "S1d": [("Sd:all", 1.0)],
}


def mixture_mean(*, heads: list[tuple[tuple[torch.Tensor, ...], float]]) -> np.ndarray:
    """Return the mean (capacity fraction) of a density mixture of heads with the given weights."""
    total = 0.0
    for (log_w, mean, _std), weight in heads:
        total = total + weight * (log_w.exp() * mean).sum(1).numpy()
    return np.asarray(total)


def main() -> None:
    """Write the checks table."""
    fit.PRED_DIR = arms.WORK_DIR / "predictions"
    sites = fit.prep.primaries()["site"].to_list()
    rows = []
    for month in arms.MONTHS:
        for arm, spec in HEADS.items():
            parts = []
            for cell in arms.SEEDS:
                frame = report.load(arm=arm, month=month, seed=cell, worlds=[-1])
                mean = mixture_mean(
                    heads=[(arms.load_head(kind=k, month=month, seed=cell), w) for k, w in spec]
                )
                parts.append(frame.with_columns(mean_fc=pl.Series(mean)))
            frame = pl.concat(parts).filter(pl.col("lead_hours") <= 240)
            frame = frame.with_columns(
                site=pl.Series(np.array(sites)[frame["primary"].to_numpy()]),
                gap=(pl.col("q50") - pl.col("mean_fc")) * 100,
                abs_err=(pl.col("actual") - pl.col("mean_fc")).abs() * 100,
            )
            for key, part in frame.group_by("site", "group", "cls"):
                rows.append(
                    {"month": month, "arm": arm, "site": key[0], "group": key[1], "cls": key[2],
                     "rows": part.height, "gap": float(part["gap"].mean()),  # ty: ignore[invalid-argument-type]
                     "abs_err": float(part["abs_err"].mean())}  # ty: ignore[invalid-argument-type]
                )  # fmt: skip
    table = pl.DataFrame(rows)
    scoring.SCORE_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(scoring.SCORE_DIR / "checks_by_site.csv", float_precision=4)
    per_site = (
        table.filter(pl.col("cls") == "clean_real")
        .group_by("arm", "site", "group")
        .agg(
            ((pl.col("gap") * pl.col("rows")).sum() / pl.col("rows").sum()).alias(
                "median_minus_mean"
            ),
        )
        .sort("arm", "group", "site")
    )
    per_site.write_csv(scoring.SCORE_DIR / "median_minus_mean.csv", float_precision=3)
    error = (
        table.group_by("arm", "group", "cls")
        .agg(
            ((pl.col("abs_err") * pl.col("rows")).sum() / pl.col("rows").sum()).alias(
                "mean_abs_error"
            )
        )
        .sort("group", "cls", "arm")
    )
    error.write_csv(scoring.SCORE_DIR / "mean_forecast_error.csv", float_precision=3)
    with pl.Config(tbl_rows=80):
        print(per_site)
        print(error)


if __name__ == "__main__":
    main()

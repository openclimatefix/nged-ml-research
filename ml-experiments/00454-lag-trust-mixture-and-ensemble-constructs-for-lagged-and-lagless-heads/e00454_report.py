"""Contrasts against B-mix, minimum detectable effect, and tail tables for experiment 454.

Reads `work/scores/cells.parquet` (see `e00454_score.py`). A construct's score in a month is the
mean over its cells (seeds or seed pairs); a contrast is that score minus B-mix's on the same rows
and variant; the pooled contrast is the mean over the scored months. The interval resamples whole
months with replacement (100,000 draws) at a Bonferroni level over the headline contrasts (every
construct except B-mix, at the signed-MW six, for clean, harm, and event rows). With only five
months the interval is close to the range of the monthly contrasts, so `better` counts the months
in which the construct beats B-mix.

The minimum detectable effect is `2.8 * sd_pair / sqrt(months)`, where `sd_pair` is the standard
deviation, over months, of the difference between two single-seed B-mix cells' month scores (seeds
0 to 2, independent training worlds), averaged over the three seed pairs.

```bash
<PYTHONPATH as e00454_build.py shows> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00454-*/e00454_report.py contrasts
```
"""

import argparse
import itertools
from typing import Final

import e00454_build as build  # ty: ignore[unresolved-import]
import e00454_score as scoring  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

REFERENCE: Final[str] = "B-mix"
RESAMPLES: Final[int] = 100_000
HEADLINE_CLASSES: Final[tuple[str, ...]] = ("clean", "harm", "event")
TAILS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")


def month_scores() -> pl.DataFrame:
    """Return the month score of each construct: the mean over its cells."""
    cells = pl.read_parquet(scoring.SCORE_DIR / "cells.parquet")
    value_columns = ["crps", *[f"{k}_{t}" for k in ("cov", "pin") for t in TAILS]]
    return (
        cells.group_by("arm", "month", "variant", "basis", "group", "cls")
        .agg(pl.col("n").first(), *[pl.col(c).mean() for c in value_columns])
        .sort("arm", "month")
    )


def month_block_interval(*, diffs: np.ndarray, level: float, seed: int = 0) -> tuple[float, float]:
    """Return a two-sided interval of the mean of `diffs`, resampling whole months."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diffs), size=(RESAMPLES, len(diffs)))
    estimates = diffs[draws].mean(1)
    return float(np.quantile(estimates, (1.0 - level) / 2)), float(
        np.quantile(estimates, 1.0 - (1.0 - level) / 2)
    )


def contrasts(*, scores: pl.DataFrame, column: str = "crps") -> pl.DataFrame:
    """Return each construct's pooled `column` contrast against B-mix, per group, class, variant."""
    arms = [a for a in scores["arm"].unique().sort() if a != REFERENCE]
    n_headline = len(arms) * len(HEADLINE_CLASSES)
    level = 1.0 - 0.05 / n_headline
    ref = scores.filter(pl.col("arm") == REFERENCE).select(
        "month", "variant", "basis", "group", "cls", ref=pl.col(column), ref_n="n"
    )
    joined = scores.filter(pl.col("arm") != REFERENCE).join(
        ref, on=["month", "variant", "basis", "group", "cls"]
    )
    rows = []
    keys = ["arm", "variant", "basis", "group", "cls"]
    for key, part in joined.group_by(*keys, maintain_order=True):
        sorted_part = part.sort("month")
        assert (sorted_part["n"] == sorted_part["ref_n"]).all(), f"{key}: rows differ from B-mix"
        diff = (sorted_part[column] - sorted_part["ref"]).to_numpy()
        low, high = month_block_interval(diffs=diff, level=level)
        rows.append(
            dict(zip(keys, key, strict=True))
            | {
                "months": len(diff),
                "absolute": float(sorted_part[column].mean()),  # ty: ignore[invalid-argument-type]
                "reference": float(sorted_part["ref"].mean()),  # ty: ignore[invalid-argument-type]
                "diff": float(diff.mean()),
                "low": low,
                "high": high,
                "better": int((diff < 0).sum())
                if column == "crps" or column.startswith("pin")
                else -1,
                "bonferroni_contrasts": n_headline,
            }
        )
    return pl.DataFrame(rows)


def minimum_detectable_effect() -> pl.DataFrame:
    """Return the minimum detectable effect from B-mix's seed spread, per group and class."""
    cells = pl.read_parquet(scoring.SCORE_DIR / "cells.parquet").filter(
        (pl.col("arm") == REFERENCE) & (pl.col("variant") == "raw") & (pl.col("basis") == "metered")
    )
    rows = []
    for (group, cls), part in cells.group_by("group", "cls"):
        wide = part.pivot(on="cell", index="month", values="crps").sort("month")
        sds = [
            float(np.std((wide[str(i)] - wide[str(j)]).to_numpy(), ddof=1))
            for i, j in itertools.combinations(build.CELL_SEEDS, 2)
        ]
        rows.append(
            {"group": group, "cls": cls, "sd_pair": float(np.mean(sds)),
             "mde": 2.8 * float(np.mean(sds)) / np.sqrt(wide.height)}
        )  # fmt: skip
    return pl.DataFrame(rows).sort("group", "cls")


def main() -> None:
    """Write and print the contrast tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("contrasts",))
    parser.parse_args()
    scores = month_scores()
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    out = scoring.SCORE_DIR
    mde = minimum_detectable_effect()
    mde.write_csv(out / "mde.csv", float_precision=4)
    print(mde)
    crps = contrasts(scores=scores)
    crps.write_csv(out / "contrasts_crps.csv", float_precision=4)
    tails = pl.concat(
        [
            contrasts(scores=scores, column=f"{k}_{t}").with_columns(metric=pl.lit(f"{k}_{t}"))
            for k in ("pin", "cov")
            for t in TAILS
        ]
    )
    tails.write_csv(out / "contrasts_tails.csv", float_precision=4)
    scores.write_csv(out / "month_scores.csv", float_precision=4)
    show = crps.filter(
        (pl.col("variant") == "raw")
        & (pl.col("basis") == "metered")
        & (pl.col("group") != "report_only")
    ).sort("group", "cls", "diff")
    print(
        show.select("group", "cls", "arm", "absolute", "reference", "diff", "low", "high", "better")
    )


if __name__ == "__main__":
    main()

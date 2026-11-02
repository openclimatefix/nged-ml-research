"""Print the minimum detectable effect for experiment 480's contrast C1, before any A0 or A1 score.

The effect that a paired month-block bootstrap can resolve is read off experiment 381's stored
monthly arms: the half-width of the 99.67% interval of `solar` minus `control` at the eight MVA-only
Primaries (and at D12 alone), CRPS at days 1 to 10. An effect smaller than that half-width cannot
be told from zero.

```bash
PYTHONPATH=ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    uv run --no-sync python ml-experiments/00480-*/e00480_mde.py
```
"""

from typing import Final

import e00381_score as s381  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol import metrics

LEVEL: Final[float] = 100 * (1 - 0.01 / 3)
RESAMPLES: Final[int] = 20_000
WHERE: Final[dict[str, pl.Expr]] = {
    "MVA-only eight": pl.col("group") == "MVA-only",
    "D12": pl.col("label") == "D12",
}


def main() -> None:
    """Print the half-width of the 99.67% interval for each reference contrast."""
    series = s381.adapter.series_table()
    ctx = s381.build_context(series=series)
    arms = {a: s381.arm_frame(regime="monthly", arm=a, series=series) for a in ("control", "solar")}
    table, _, rows = s381.score_arms(arms=arms, series=series, ctx=ctx)
    for name, where in WHERE.items():
        r = metrics.month_bootstrap_contrast(
            treatment=rows["solar"],
            reference=rows["control"],
            metric="crps",
            where=where & pl.col(s381.HEADLINE),
            level_percent=LEVEL,
            n_resamples=RESAMPLES,
            slice_table=table,
        )
        half = (r["upper"] - r["lower"]) / 2
        print(
            f"{name}: solar minus control {r['difference']:+.3f} "
            f"({r['lower']:+.3f} to {r['upper']:+.3f}) at {LEVEL:.2f}%; "
            f"MDE (half-width) {half:.3f} points, n={r['n_half_hours']}"
        )


if __name__ == "__main__":
    main()

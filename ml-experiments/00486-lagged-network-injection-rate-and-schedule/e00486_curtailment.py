"""Avoidable-curtailment cost of experiment 486's arms beside B-mix, BG2, and manual_heuristic.

Experiment 454's script (`e00454_curtailment`) pointed at this experiment's forecasts: its reader
takes `<work>/predictions/<arm>/<month>_s*_w-1.parquet` (clean world, rows with no labelled event),
averages the stored cells of a month in quantile space, widens by 1.15 about the median, and
costs each series with experiment 449's per-series cost and cell bootstrap, for the series that
experiment 448's `sites_table` and experiment 449's costing select (the signed-MW six).

```bash
PYTHONPATH=<every folder under ml-experiments/, as e00486_figures.py needs> \
    uv run --no-sync python ml-experiments/00486-*/e00486_curtailment.py
```
"""

import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00454_build as build454  # ty: ignore[unresolved-import]
import e00454_curtailment as c454  # ty: ignore[unresolved-import]
import e00486_fit as fit486  # ty: ignore[unresolved-import]

NEW = ("R05", "R50", "Cur-down", "Cur-up")
ARMS = ("A0-inj", "B-mix", *NEW, *(f"{a}+lagless" for a in NEW))
COMPARISONS = (
    *((a, "A0-inj") for a in NEW),
    *((f"{a}+lagless", "B-mix") for a in NEW),
    ("A0-inj", "B-mix"),
    ("B-mix", "BG2"),
    ("B-mix", "manual_heuristic"),
)


def main() -> None:
    """Point experiment 454's curtailment script at experiment 486 and run it."""
    build454.WORK_DIR = fit486.WORK_DIR
    c454.ARMS = ARMS
    c454.COMPARISONS = COMPARISONS
    (fit486.WORK_DIR / "scores").mkdir(parents=True, exist_ok=True)
    c449.ARMS, c449.REFERENCES, c449.COMPARISONS = ARMS, c454.REFERENCES, COMPARISONS
    c454.main()


if __name__ == "__main__":
    main()

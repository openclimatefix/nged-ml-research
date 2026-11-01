"""Avoidable-curtailment cost of experiment 472's arms beside the anchors and references.

The references are LL, LL-ens3, B-mix, K2-S1, BG2, and manual_heuristic.

Reuses experiment 454's curtailment script, replacing only how a construct's forecasts are read:
the arms of this experiment live in its own prediction folder, experiment 467's arms in experiment
404's, and each arm averages its stored cells (`e00472_score.ARM_CELLS`). Quantiles are widened by
1.15 about the median, as in experiment 454. Copied from experiment 467's `e00467_curtailment.py`.

```bash
<PYTHONPATH as e00472_arm.py shows, plus the 377, 425, 435, 448, 449, and 454 folders> \
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00472-*/e00472_curtailment.py
```
"""

import e00087_common as common  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00454_build as build_mod  # ty: ignore[unresolved-import]
import e00454_curtailment as c454  # ty: ignore[unresolved-import]
import e00472_arm as arm_mod  # ty: ignore[unresolved-import]
import e00472_score as score472  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

ARMS = ("P100", "AvgAnchor", "C1e-2", "C1", "Z1e-2", "LL", "LL-ens3", "B-mix", "K2-S1")
COMPARISONS = (
    ("P100", "LL"), ("P100", "B-mix"), ("P100", "C1e-2"), ("P100", "Z1e-2"),
    ("AvgAnchor", "LL-ens3"), ("AvgAnchor", "K2-S1"), ("AvgAnchor", "B-mix"),
    ("AvgAnchor", "C1e-2"), ("P100", "BG2"), ("AvgAnchor", "BG2"),
    ("P100", "manual_heuristic"), ("AvgAnchor", "manual_heuristic"),
    ("K2-S1", "manual_heuristic"), ("B-mix", "manual_heuristic"),
)  # fmt: skip


def read_rows(
    *, arm: str, month: str, seeds: tuple[int, ...], factor: float, with_quantiles: bool
) -> pl.DataFrame | None:
    """Return the rows with no labelled event of an arm, averaged over fit seeds 0 and 1."""
    del seeds, with_quantiles
    build_mod.prediction_path = score472.prediction_path
    paths = [
        score472.prediction_path(arm=arm, month=month, cell=c) for c in score472.ARM_CELLS[arm]
    ]
    frames = [pl.read_parquet(p) for p in paths]
    first = frames[0]
    for f in frames[1:]:
        assert f["target"].equals(first["target"]), "cell files differ in row order"
    levels = c454.LEVELS
    q = np.mean([f.select(list(levels)).to_numpy().astype(np.float64) for f in frames], axis=0)
    median = q[:, [c454.MEDIAN]]
    q = np.sort(median + factor * (q - median), axis=1)
    actual = first["actual"].to_numpy().astype(np.float64)
    keep = ~first["win_real"].to_numpy() & ~first["tgt_post"].to_numpy() & np.isfinite(actual)
    grid = common.grid_times()
    out = first.select("primary", "lead_hours", "tgt_real").with_columns(
        actual=pl.Series(actual),
        power_fcst_init_time=grid.gather(first["origin"].to_numpy().astype(np.int64)),
        valid_time=grid.gather(first["target"].to_numpy().astype(np.int64)),
        **{n: pl.Series(q[:, i]) for i, n in enumerate(levels)},
    )
    out = out.filter(pl.Series(keep)).join(
        base.sites_table().select("primary", "time_series_id", "label", "group", "capacity_mw"),
        on="primary",
    )
    return out.with_columns(time_series_id=pl.col("time_series_id").cast(pl.Int32))


def main() -> None:
    """Write and print the per-cell costs and the bootstrap table."""
    c454.read_construct_rows = read_rows
    c454.ARMS = ARMS
    c454.COMPARISONS = COMPARISONS
    c449.ARMS = ARMS
    c449.REFERENCES = c454.REFERENCES
    c449.COMPARISONS = COMPARISONS
    c449.RESAMPLES = c454.RESAMPLES
    cost = c454.build()
    out = arm_mod.WORK_DIR / "scores"
    out.mkdir(parents=True, exist_ok=True)
    cost.write_parquet(out / "curtailment_cells.parquet")
    table = c449.bootstrap(cost=cost)
    table.write_csv(out / "curtailment_bootstrap.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=140):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()

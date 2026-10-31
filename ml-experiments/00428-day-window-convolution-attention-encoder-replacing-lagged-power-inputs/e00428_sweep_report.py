r"""Summarise the encroachment sweep tables of experiment 428 (no fit).

For each arm, variant, k, step, lead band, group, and widening, pools the months, seeds, Primaries,
and both signs, and prints: the median share of the step followed (from the pooled histogram of the
per-row share; +1 is the full step), the ratio of the 5%-95% band width after the step to before,
the coverage of the clean reading and of the event-level reading, the attention mass on each window
and on the "no day" token (before and after), and the largest lag-masked-control gap.

```bash
PYTHONPATH=ml-experiments/00428-* DATA_PATH_INTERNAL=... uv run --no-sync python \\
    ml-experiments/00428-*/e00428_sweep_report.py --step c050 --band "24-72 h"
```
"""

import argparse
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00428_sweep as sweep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

KEYS: Final[list[str]] = ["arm", "variant", "k", "step", "band", "group", "widened"]


def median_of(*, hist: np.ndarray) -> float:
    """Return the median share from histogram counts over `sweep.HIST_BINS` (bins 0..61)."""
    total = hist.sum()
    if total == 0:
        return float("nan")
    cumulative = np.cumsum(hist)
    i = int(np.searchsorted(cumulative, total / 2.0))
    edges = np.concatenate([[-np.inf], sweep.HIST_BINS, [np.inf]])
    low, high = edges[i], edges[i + 1]
    if not np.isfinite(low):
        return float(high)
    if not np.isfinite(high):
        return float(low)
    return float((low + high) / 2.0)


def load() -> pl.DataFrame:
    """Return every sweep table with a `group` column."""
    frames = [pl.read_parquet(f) for f in sorted(sweep.SWEEP_DIR.glob("*/*.parquet"))]
    table = pl.concat(frames, how="diagonal")
    group = {site: g for g, labels in score.GROUPS.items() for site in labels}
    return table.with_columns(pl.col("site").replace_strict(group, default=None).alias("group"))


def report(*, table: pl.DataFrame, step: str, band: str, widened: bool) -> pl.DataFrame:
    """Return the pooled summary rows for a step, a lead band, and a widening."""
    sub = table.filter((pl.col("step") == step) & (pl.col("band") == band))
    sub = sub.filter(pl.col("widened") == widened)
    records = []
    mass_columns = [c for c in sub.columns if c.startswith("mass") and c.endswith("_after_sum")]
    for key, part in sub.group_by([c for c in KEYS if c not in ("step", "band", "widened")]):
        hist = np.sum(np.array(part["hist"].to_list()), axis=0)
        n = part["n"].sum()
        record = dict(
            zip([c for c in KEYS if c not in ("step", "band", "widened")], key, strict=True)
        )
        record.update(
            n=int(n),
            median_share=median_of(hist=hist),
            mean_share=part["shift_sum"].sum() / n,
            width_ratio=part["width_after_sum"].sum() / part["width_before_sum"].sum(),
            cover_clean=part["cover_clean_sum"].sum() / n,
            cover_event=part["cover_event_sum"].sum() / n,
            masked_gap=part["masked_max_gap"].max(),
        )
        for column in mass_columns:
            name = column.removesuffix("_after_sum")
            record[f"{name}_before"] = part[f"{name}_before_sum"].sum() / n
            record[f"{name}_after"] = part[column].sum() / n
        records.append(record)
    return pl.DataFrame(records).sort("group", "arm", "variant", "k")


def main() -> None:
    """Print the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", default="c050")
    parser.add_argument("--band", default="24-72 h")
    parser.add_argument("--widened", action="store_true")
    args = parser.parse_args()
    table = load()
    out = report(table=table, step=args.step, band=args.band, widened=args.widened)
    out.write_parquet(data.WORK_DIR / f"sweep_report_{args.step}_{args.band.split()[0]}.parquet")
    with pl.Config(tbl_rows=300, tbl_cols=30, fmt_float="mixed"):
        print(out.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()

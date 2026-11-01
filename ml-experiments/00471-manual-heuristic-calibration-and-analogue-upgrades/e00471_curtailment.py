# ruff: noqa: E501
"""Avoidable-curtailment cost of the faithful heuristic and its variants, signed-MW six (no fit).

Runs experiment 448's `curtailment` once per scored month on the rows every reference holds (the
`main` pass of `e00471_score.py`), at limit fractions f = 0.9, 0.75, 0.5, then resamples the
(month, series) cells with replacement for an interval on each total and on each paired difference
from the faithful heuristic (experiment 449's cell bootstrap). A missing generator forecast counts
as not instructed (inherited). Output: `work/scores/curtailment_*.csv`.

```bash
source /tmp/e471_env.sh
uv run --no-sync python ml-experiments/00471-*/e00471_curtailment.py
```
"""

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00452_refs as refs452  # ty: ignore[unresolved-import]
import e00452_run as run  # ty: ignore[unresolved-import]
import e00471_common as common  # ty: ignore[unresolved-import]
import e00471_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

ARMS = ("manual_heuristic", "E56", "Q56", "Tau95_56", "MeanLS56", "Lvl14")
RESAMPLES = 20_000


def main() -> None:
    """Write the per-cell costs and the bootstrap table."""
    base.run = run  # ty: ignore[invalid-assignment]
    base.metrics.avoidable_curtailment_cost = c449.per_series_cost  # ty: ignore[invalid-assignment]
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    stored = {
        k: v
        for k, v in refs452.reference_frames(series=series).items()
        if k in ("BG2", "manual_heuristic")
    }
    refs = {**stored, **{v: score.variant_frame(name=v) for v in ARMS[1:]}}
    parts = []
    for month in common.MONTHS:
        _, shared, _ = base.build_month(month=month, arms=("PT1",), refs=refs, seeds=run.SEEDS5)
        cost = base.curtailment(
            months=(month,),
            refs=refs,
            keys=shared.select(*base.KEYS),
            ctx=ctx,
            series=series,
            arms=(),
        )
        parts.append(cost.with_columns(month=pl.lit(month)))
    cells = pl.concat(parts)
    out = common.WORK_DIR / "scores"
    cells.write_csv(out / "curtailment_cells.csv")
    ok = cells.filter(pl.col("status") == "ok").with_columns(
        cell=pl.col("month") + "/" + pl.col("label")
    )
    rng = np.random.default_rng(0)
    rows = []
    for fraction in sorted(ok["fraction"].unique(), reverse=True):
        wide = (
            ok.filter(pl.col("fraction") == fraction)
            .pivot(on="arm", index="cell", values="avoidable_cost_gbp")
            .sort("cell")
        )
        picks = rng.integers(0, wide.height, size=(RESAMPLES, wide.height))
        for arm in ("BG2", *ARMS):
            v = wide[arm].to_numpy()
            tot = v[picks].sum(axis=1) / 1e6
            rows.append({"fraction": fraction, "item": arm, "estimate": v.sum() / 1e6,
                         "low": np.percentile(tot, 2.5), "high": np.percentile(tot, 97.5)})  # fmt: skip
        for arm in ARMS[1:]:
            d = wide[arm].to_numpy() - wide["manual_heuristic"].to_numpy()
            tot = d[picks].sum(axis=1) / 1e6
            rows.append({"fraction": fraction, "item": f"{arm} minus manual_heuristic", "estimate": d.sum() / 1e6,
                         "low": np.percentile(tot, 2.5), "high": np.percentile(tot, 97.5)})  # fmt: skip
    table = pl.DataFrame(rows)
    table.write_csv(out / "curtailment_bootstrap.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=140):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()

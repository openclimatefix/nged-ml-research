"""Experiment 450, stage 1: share of observations above q99 after table P, per farm and arm.

Kept rows, true days 1 to 10, table P fitted on earlier months as in `e00450_step1`. Output:
`work/farm_share.json` (percent for Farm A, Farm B, Farm C per arm).

    source ml-experiments/00450-*/env.sh
    uv run --no-sync python ml-experiments/00450-*/e00450_farm_share.py
"""

import json

import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00450_score as score  # ty: ignore[unresolved-import]
import e00450_step1 as step1  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

ARMS = ("base", "w32", "w128", "d1", "d3", "e8", "e32", "k2", "k5", "a6")


def main() -> None:
    """Write the per-farm shares."""
    cases = data347.scored_frame()
    out: dict[str, list[float]] = {}
    for name in ARMS:
        q = np.sort(
            np.mean(np.stack(score.load_seed_tables(name=name, seeds=score.run450.SEEDS)), axis=0),
            axis=1,
        )
        base = cases.select("site", "month", "masked", "lead_hours", actual=pl.col("actual_norm"))
        frame = base.with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(step1.LEVELS)])
        frame = (
            frame.filter(pl.col("actual").is_finite())
            .with_columns(lead_band=step1.lead_band_expr())
            .filter(pl.col("lead_band").is_not_null() & ~pl.col("masked"))
        )
        months = sorted(frame["month"].unique())
        cells = []
        for i, month in enumerate(months[1:], start=1):
            earlier = frame.filter(pl.col("month").is_in(months[:i]) & (pl.col("lead_hours") >= 24))
            factors = step1.fit_factors(fit=earlier)
            held = step1.apply_factors(
                frame=frame.filter(pl.col("month") == month), factors=factors
            )
            cells.append(step1.hits(frame=held, arm=name, rowset="kept rows", variant="P"))
        farms = (
            pl.concat(cells)
            .group_by("site")
            .agg(pl.col("n").sum(), pl.col("hit_q99").sum())
            .sort("site")
        )
        out[name] = [
            round(100 * (1 - r["hit_q99"] / r["n"]), 2) for r in farms.iter_rows(named=True)
        ]
    path = score.WORK_DIR / "farm_share.json"
    path.write_text(json.dumps(out))
    print(out)


if __name__ == "__main__":
    main()

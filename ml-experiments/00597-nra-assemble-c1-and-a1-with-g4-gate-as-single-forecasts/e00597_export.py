# ruff: noqa: E501, D103, D107
"""Experiment 597: export the assembled forecasts C1 + G4 and A1 + G4 in experiment 593's layout.

No fit and no GPU. One Parquet file per system and fold month under
`data/ml-experiments/backtest_export_candidates_2026-11-06/` (never committed), named
`<system>_<month>.parquet` for `C1`, `C1_G4`, `A1`, and `A1_G4`. The columns are those of experiment
593's export; `gate_on` is filled in the `_G4` files: true when the G4 gate was on, or held (7 days
after it turned off), for the origin day, which is the rows whose band is the union band.

Seeds: the stored quantiles of seeds 0 and 1 are averaged quantile by quantile for each Primary,
origin time, and valid time, then sorted across the 11 levels so that none cross (as in 593).

Usage: python e00597_export.py [MONTH ...]   (default: all five fold months). Run `e00597_score.py`
for `c1` first: it stores the G4 state (`c1/mode_g4.npy`).
"""

import sys

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00593_export as ex  # ty: ignore[unresolved-import]
import e00597_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SYSTEMS = {"C1": k.C1_WORK, "A1": k.A1_WORK}


class Context(ex.Context):
    """Experiment 593's context with experiment 597's G4 state in place of experiment 583's gate."""

    def _gate(self) -> None:
        self.gated = np.load(k.WORK / "c1" / "mode_g4.npy")


def main(months: tuple[str, ...]) -> None:
    k.EXPORT.mkdir(parents=True, exist_ok=True)
    ex.OUT = k.EXPORT
    ctx = Context()
    ctx.sites.select(
        "time_series_id",
        pl.col("label").alias("primary_name"),
        pl.col("group")
        .replace_strict(
            {
                "MVA-only eight": "eight MVA Primaries",
                "signed-MW six": "six MW Primaries",
                "D15 and D18": "report-only (D15 and D18, short history)",
            }
        )
        .alias("group"),
        pl.col("capacity_mw").alias("capacity"),
        "unit",
    ).sort("time_series_id").write_csv(k.EXPORT / "substations.csv")
    for month in months:
        for arm, root in SYSTEMS.items():
            df = ex.load(root=root, tag="P5", month=month).sort(ex.KEYS)
            tw = ex.load(root=root, tag="P5-removed", month=month)
            both = df.join(
                tw.rename({q: f"M_{q}" for q in ex.QCOLS}), on=ex.KEYS, how="inner"
            ).sort(ex.KEYS)
            assert both.height == df.height, (arm, month, both.height, df.height)
            v, m = ex.sorted_q(both), ex.sorted_q(both, prefix="M_")
            prim = (
                both["time_series_id"].replace_strict(ctx.prim_of, return_dtype=pl.Int64).to_numpy()
            )
            origin = (
                (both["power_fcst_init_time"].dt.epoch("us") - ex.k578.prep.GRID_START_US)
                // ex.k578.prep.STEP_US
            ).to_numpy()
            wide = ctx.gated[prim, origin // 48]
            base = both.select(ex.KEYS)
            ex.write(tbl=ctx.table(df=base, q=v, gate=False), arm=arm, month=month)
            ex.write(
                tbl=ctx.table(df=base, q=k.assemble(v=v, m=m, wide=wide), gate=True),
                arm=f"{arm}_G4",
                month=month,
            )


if __name__ == "__main__":
    assert a.MAX_LEAD_H == 120 and c.MONTHS
    main(tuple(sys.argv[1:]) or c.MONTHS)

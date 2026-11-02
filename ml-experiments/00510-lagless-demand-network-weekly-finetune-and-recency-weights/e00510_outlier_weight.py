# ruff: noqa: E501
"""The learned outlier weight of L-Ww-O by row class (experiment 510).

For each fold and refit week, the saved seed-0 L-Ww-O network forecasts its scored week's rows, and
the outlier weight sigmoid(logit) is averaged by class on the scorer's rows: no labelled event (clean,
not cold, not Christmas week), cold winter nights, Christmas week, and rows inside labelled events.
"""

from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c
import e00510_fit as fit
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]


def main(*, device: str = "cuda") -> None:
    """Print the mean outlier weight by class."""
    clean = c.clean_stack()
    records = []
    for month in c.MONTHS:
        base = data.build_fold(refit=c.month_start(month=month), clean=clean)
        test = data.build_test(month=month, standardiser=base.standardiser)
        scored = pl.read_parquet(c.WORK_DIR / "scores" / f"rows_{month}.parquet").select(
            "primary", "origin", "target", "group", "cls", "cold", "xmas"
        )
        for week in range(len(c.REFIT_DAYS)):
            model = c.new_model(
                outlier=True, n_inputs=test.x.shape[1], n_primaries=len(base.target_mean)
            ).to(device)
            model.load_state_dict(
                torch.load(
                    c.WORK_DIR / "models" / "hl90O" / f"{month}_w{week}_s0.pt", map_location=device
                )
            )
            model.eval()
            lo, hi = fit.window_of(month=month, week=week)
            sel = np.nonzero((test.origin >= lo) & (test.origin < hi))[0]
            weights = []
            with torch.no_grad():
                for a in range(0, len(sel), 65_536):
                    s = sel[a : a + 65_536]
                    out = model(
                        torch.from_numpy(test.x[s]).to(device),
                        torch.from_numpy(test.primary[s]).to(device),
                    )
                    weights.append(torch.sigmoid(out[..., 3 * c.COMPONENTS]).cpu().numpy())
            frame = pl.DataFrame(
                {
                    "primary": test.primary[sel].astype(np.int16),
                    "origin": test.origin_idx[sel].astype(np.int32),
                    "target": test.target_idx[sel].astype(np.int32),
                    "p_out": np.concatenate(weights),
                }
            )
            records.append(
                frame.join(
                    scored.with_columns(
                        pl.col("primary").cast(pl.Int16),
                        pl.col("origin").cast(pl.Int32),
                        pl.col("target").cast(pl.Int32),
                    ),
                    on=["primary", "origin", "target"],
                    how="inner",
                ).with_columns(month=pl.lit(month))
            )
    t = pl.concat(records)
    cls = (
        pl.when(pl.col("cls") == "event").then(pl.lit("inside labelled events"))
        .when(pl.col("xmas")).then(pl.lit("Christmas week"))
        .when(pl.col("cold") & (pl.col("cls") == "clean")).then(pl.lit("cold winter nights, no labelled event"))
        .when(pl.col("cls") == "clean").then(pl.lit("other rows, no labelled event"))
        .otherwise(pl.lit("harm"))
    )  # fmt: skip
    out = (
        t.with_columns(row_class=cls)
        .group_by("row_class")
        .agg(n=pl.len(), mean_outlier_weight=pl.col("p_out").mean())
        .sort("row_class")
    )
    cold_by_month = (
        t.filter(pl.col("cold") & (pl.col("cls") == "clean"))
        .group_by("month")
        .agg(n=pl.len(), w=pl.col("p_out").mean())
        .sort("month")
    )
    other_by_month = (
        t.filter(~pl.col("cold") & (pl.col("cls") == "clean") & ~pl.col("xmas"))
        .group_by("month")
        .agg(w=pl.col("p_out").mean())
        .sort("month")
    )
    with pl.Config(tbl_rows=30):
        print(out.with_columns(pl.col(pl.Float64).round(4)))
        print(
            cold_by_month.join(other_by_month, on="month", suffix="_other").with_columns(
                pl.col(pl.Float64).round(4)
            )
        )
    out.write_csv(c.WORK_DIR / "scores" / "outlier_weight.csv")
    _: Final = None


if __name__ == "__main__":
    main()

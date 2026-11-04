"""Seam check of experiment 526 on the lag-light network.

The contrast is RDW-seam (persistence fill before the origin) minus RDW-T.

RDW-seam covers the three winter folds only. A difference near zero means the gain of the window
columns does not depend on how the window is filled where it crosses the forecast origin.
Output: `scores/seam_contrasts.csv` in this experiment's private data folder.
"""

import e00521_stats as s  # ty: ignore[unresolved-import]
import e00550_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

FOLDS = ("2025-12", "2026-01", "2026-03")


def main() -> None:
    """Write the RDW-seam minus RDW-T contrasts."""
    t = sc.table().with_columns(band=s.band_expr())
    seam = pl.concat(
        [
            pl.read_parquet(
                sc.ROOT / "work" / "scores_seam" / f"rows_{m}.parquet",
                columns=[*sc.K, *[f"RDW-seam__{x}" for x in sc.METRICS]],
            )
            for m in FOLDS
        ]
    )
    t = t.join(seam, on=sc.K, how="left", validate="1:1", maintain_order="left")
    t = t.filter(pl.col("month").is_in(FOLDS))
    rng = np.random.default_rng(0)
    out = []
    for name in ("eight", "six"):
        part = s.row_set(table=t, name=name)
        for slice_name, expr in sc.SLICES.items():
            if slice_name == "inside a labelled event":
                continue
            sliced = part.filter(expr.fill_null(value=False))
            for band in (*s.BANDS, s.ALL):
                sub = s.in_band(part=sliced, band=band)
                d = sub.select(
                    "month", "day", d=pl.col("RDW-seam__crps") - pl.col("RDW-T__crps")
                ).filter(pl.col("d").is_finite())
                if d.height < 200:
                    continue
                out.append(
                    {
                        "set": name,
                        "slice": slice_name,
                        "band": band,
                        "n": d.height,
                        **s.block_stats(diff=d, rng=rng, tests=sc.TESTS),
                    }
                )
    res = pl.DataFrame(out)
    res.write_csv(sc.ROOT / "scores" / "seam_contrasts.csv")
    with pl.Config(tbl_rows=60):
        print(
            res.filter(pl.col("band") == s.ALL)
            .select("set", "slice", "n", "difference", "low_month", "high_month", "folds_positive")
            .with_columns(pl.col(pl.Float64).round(3))
        )


if __name__ == "__main__":
    main()

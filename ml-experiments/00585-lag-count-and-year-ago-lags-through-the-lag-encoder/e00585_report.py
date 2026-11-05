# ruff: noqa: E501, D103
"""Experiment 585: tail calibration and per-Primary gains for every arm (printed and written to CSV).

Builds a row table like experiment 572's scorer (experiment 572's rows, stored forecasts, widening
1.15) with, for each arm and seed, the CRPS, the pinball loss at the tail levels, and the coverage
at the tail levels. Writes `tails.csv` (arm, row set, lead band) and `per_primary.csv` (arm, Primary,
lead band) under this experiment's scores folder.
"""

import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import e00565_design6 as d6  # ty: ignore[unresolved-import]
import e00572_score as sc  # ty: ignore[unresolved-import]
import e00585_score as mine
import numpy as np
import polars as pl
from e00435_tailcore import TAIL_LEVELS  # ty: ignore[unresolved-import]

METRICS = ("crps", *[f"{k}_{q}" for q in TAIL_LEVELS for k in ("pin", "cov")])


def build(*, arms: dict) -> pl.DataFrame:
    sites = build470.site_map()
    frames = []
    for month in d6.MONTHS:
        rows = pl.read_parquet(
            d6.W559 / "scores" / f"rows_{month}.parquet",
            columns=[*d6.KEYS, "group", "cls", "lead_hours", "main16"],
        )
        actual = score454.month_base(month=month).select(*d6.KEYS, "actual")
        rows = rows.join(actual, on=d6.KEYS, how="left", maintain_order="left", validate="1:1")
        cols = {}
        for label, (root, folder) in arms.items():
            for seed in sc.SEEDS:
                frame = build470.pool_stored(
                    folder=root / "seed_preds" / folder, month=month, sites=sites, seeds=(seed,)
                )
                fr = rows.select(*d6.KEYS, "actual").join(
                    frame.select(*d6.KEYS, *d6.LEVELS),
                    on=d6.KEYS,
                    how="left",
                    maintain_order="left",
                )
                ok = fr["q50"].is_not_null().to_numpy()
                fr = fr.with_columns(*[pl.col(q).fill_null(0.0) for q in d6.LEVELS])
                fr = score470.widen_arm(arm="M0", frame=fr, factor=score470.HEADLINE)
                m = score470.row_metrics(frame=fr)
                for k in METRICS:
                    cols[f"{label}|{seed}|{k}"] = np.where(ok, m[k], np.nan)
        frames.append(
            rows.drop("actual").hstack(
                pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()})
            )
            .with_columns(band=sc.band_of(pl.col("lead_hours")))
        )  # fmt: skip
    return pl.concat(frames)


def main() -> None:
    present = [
        k for k in mine.ARMS if (mine.WORK / "arms" / k / "work" / "seed_preds" / "V3D5").exists()
    ]
    arms = {"V3D5": sc.ARM_DIRS["V3D5"], "N5": (mine.W581, "V3D5")}
    arms |= {mine.ARMS[k]: (mine.WORK / "arms" / k / "work", "V3D5") for k in present}
    t = build(arms=arms)
    t = t.with_columns(
        main16=pl.col("main16")
        & pl.all_horizontal([pl.col(f"{a}|0|crps").is_finite() for a in arms])
    )
    out = mine.WORK / "scores"
    tails, per_primary = [], []
    sets = {"eight": s.row_set(table=t, name="eight"), "six": s.row_set(table=t, name="six")}
    for set_name, part in sets.items():
        for band in (*sc.BANDS, "all"):
            sub = part if band == "all" else part.filter(pl.col("band") == band)
            for arm in arms:
                row = {"set": set_name, "band": band, "arm": arm, "n": sub.height}
                for k in METRICS:
                    row[k] = float(np.mean([sub[f"{arm}|{x}|{k}"].mean() for x in sc.SEEDS]))
                tails.append(row)
    for band in sc.BANDS:
        sub = t.filter(pl.col("main16") & (pl.col("band") == band))
        for p in sorted(sub["primary"].unique().to_list()):
            ps = sub.filter(pl.col("primary") == p)
            row = {"primary": p, "band": band, "n": ps.height}
            for arm in arms:
                row[arm] = float(np.mean([ps[f"{arm}|{x}|crps"].mean() for x in sc.SEEDS]))
            per_primary.append(row)
    pl.DataFrame(tails).write_csv(out / "tails.csv")
    pl.DataFrame(per_primary).write_csv(out / "per_primary.csv")
    with pl.Config(tbl_rows=100, tbl_cols=30, tbl_width_chars=250):
        tt = pl.DataFrame(tails).filter(pl.col("band") == "24-120h")
        print(tt.with_columns(pl.col(pl.Float64).round(3)))
        pp = pl.DataFrame(per_primary).filter(pl.col("band") == "24-120h")
        print(pp.with_columns(pl.col(pl.Float64).round(2)))


if __name__ == "__main__":
    main()

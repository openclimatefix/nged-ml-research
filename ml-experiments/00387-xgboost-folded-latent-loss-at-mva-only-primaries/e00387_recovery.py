"""Recovery check at the eight MVA-only Primaries: how often is the recovered latent negative?

For each site, over the folded arm's forecasts at leads up to 240 hours, the share of sunny-midday
half-hours (ERA5 shortwave above 600 W/m², 10 to 14 UTC) whose member-mean median latent is
below 0, and the 5th and 50th percentiles of that latent, in points of capacity. Experiments 366,
370, and 374 found reverse flow at D12 and at none of the other seven. A latent that is negative
at the other seven would show that the start tilt, not the readings, set the sign.
"""

import sys

import e00087_common as common  # ty: ignore[unresolved-import]
import e00387_data as data
import e00387_fit as fit
import numpy as np
import polars as pl


def main() -> None:
    """Print the table."""
    spec = sys.argv[1] if len(sys.argv) > 1 else "f2_k1_m0_q2"
    series = common.series_table()
    arrays = common.load_series_arrays(series=series.filter(pl.col("site").is_in(data.MVA_ONLY)))
    rows = []
    for site in data.MVA_ONLY:
        a = arrays[site]
        grid = pl.DataFrame({"valid_time": a.valid_time, "shortwave": data.shortwave_of(arrays=a)})
        files = sorted((fit.WORK / "preds" / "real" / spec).glob(f"{site}_*.parquet"))
        p = pl.concat([pl.read_parquet(f) for f in files]).join(grid, on="valid_time")
        p = p.filter(
            ((pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours() <= 240)
            & (pl.col("shortwave") > 600)
            & pl.col("valid_time").dt.hour().is_between(10, 14)
        )
        latent = p["latent_mean"].to_numpy() * 100
        rows.append({"site": site, "n": len(latent), "share_negative": float((latent < 0).mean()),
                     "p5": float(np.percentile(latent, 5)), "median": float(np.median(latent)),
                     "mean_neg_share_members": float(p["neg_share"].mean())})
    pl.Config.set_tbl_rows(20)
    print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()

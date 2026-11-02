"""Cold-night rows of experiment 503 (the cold-night definition of experiment 389).

A cold night is a half-hour with the sun more than 5 degrees below the horizon (solar zenith above
95 degrees) and the ERA5 2 m temperature at or below 3 degrees Celsius, for the valid time. Reads
the per-row tables of `e00503_score.py` and prints, per arm, the number of rows, the mean CRPS, and
the coverage and pinball loss of q95 and q99, for each widening scored.
"""

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00503_fit as fit  # ty: ignore[unresolved-import]
import polars as pl

NIGHT_ZENITH_DEG = 95.0
MAX_TEMPERATURE_C = 3.0


def main() -> None:
    """Print the cold-night scores for each stored widening."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(200)
    _, arrays = fd._series_and_arrays()
    columns = common.feature_columns()
    zenith = columns.index("solar_zenith")
    temperature = columns.index("temperature_2m")
    flags = []
    for site, arrs in arrays.items():
        cold = (arrs.features[:, zenith] > NIGHT_ZENITH_DEG) & (
            arrs.features[:, temperature] <= MAX_TEMPERATURE_C
        )
        flags.append(
            pl.DataFrame({"valid_time": arrs.valid_time, "cold": cold}).with_columns(
                label=pl.lit(site)
            )
        )
    cold = pl.concat(flags)
    for widening in (1.0, 1.15):
        path = fit.WORK_DIR / "scores" / f"rows_w{widening}.parquet"
        if not path.exists():
            continue
        rows = (
            pl.read_parquet(path)
            .filter(pl.col("label").is_in(list(fit.SIGNED_MW_LABELS)))
            .join(cold, on=["label", "valid_time"], how="left")
            .filter(pl.col("cold").fill_null(value=False), pl.col("lead_hours") <= 240.0)
        )
        order = {a: i for i, a in enumerate(("a", "arep", "b", "f", "g", "ga", "gb", "ar1", "ar2"))}
        print(f"\n== cold nights, signed-MW six, widening {widening}, lead 0-240 h ==")
        print(
            rows.group_by("arm")
            .agg(
                n=pl.len(),
                crps=pl.col("crps").mean(),
                cov_q95=pl.col("coverage_q95").mean(),
                cov_q99=pl.col("coverage_q99").mean(),
                pin_q95=pl.col("pinball_q95").mean(),
                pin_q99=pl.col("pinball_q99").mean(),
            )
            .with_columns(o=pl.col("arm").replace_strict(order, default=99))
            .sort("o")
            .drop("o")
        )


if __name__ == "__main__":
    main()

"""Experiment 630, part 1: exceedance tables by month, lead band, hour, Primary and temperature."""

import polars as pl
from e00630_common import INDICATORS, NETWORKS, TABLES, bootstrap_ratio, day_table, load

VALUES = [*INDICATORS, "gap", "gap_share"]
SLICES: dict[str, list[str]] = {
    "month": ["month"],
    "band": ["band"],
    "hour": ["hour"],
    "primary": ["primary_name"],
    "tercile": ["tercile"],
    "cold_night": ["cold_night"],
    "group": ["group"],
    "overall": [],
}


def main() -> None:
    """Write the exceedance tables for each network and slice."""
    TABLES.mkdir(parents=True, exist_ok=True)
    for network in NETWORKS:
        df = load(network=network)
        frames = {
            "all": df,
            "event-free": df.filter(pl.col("event_free")),
            "event": df.filter(~pl.col("event_free")),
        }
        for name, keys in SLICES.items():
            parts = []
            for subset, sdf in frames.items():
                days = day_table(df=sdf, keys=keys, values=VALUES)
                parts.append(
                    bootstrap_ratio(days=days, keys=keys, values=VALUES).with_columns(
                        subset=pl.lit(subset)
                    )
                )
            res = pl.concat(parts).with_columns(network=pl.lit(network))
            res.write_csv(TABLES / f"exceedance_by_{name}_{network}.csv")
            if name in {"overall", "month"}:
                print(f"\n== {network} by {name}")
                with pl.Config(tbl_rows=40, tbl_cols=20, fmt_float="mixed"):
                    print(
                        res.select(
                            "subset",
                            *keys,
                            "n",
                            "n_days",
                            *[pl.col(v).round(4) for v in ("a90", "a95", "b10", "b05")],
                            pl.col("gap").round(3),
                            pl.col("gap_share").round(4),
                            pl.col("a90_lo").round(3),
                            pl.col("a90_hi").round(3),
                            pl.col("a95_lo").round(3),
                            pl.col("a95_hi").round(3),
                        ).sort("subset", *keys)
                    )


if __name__ == "__main__":
    main()

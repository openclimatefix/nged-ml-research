# ruff: noqa: E501, RUF005
"""Post hoc: how much of the BSP-meter gap do the metered generators and customers at BSP level explain?

For each BSP, the half-hourly gap (meter minus the sum of the Primaries beneath it) over the origin
months' valid times before 2026-07-01 is regressed on the summed power of the other metered
series (wind, PV, battery, biofuel, other generation) that the hierarchy file puts under the BSP.
Prints the means and the R squared. Generators are not named; series are grouped by type.
"""

import e00629_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.power import scan_power

prim = c.primaries()
nodes = c.group_nodes(prim=prim)
md = pl.read_parquet(c.ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet")
hier = pl.read_csv(
    c.ML_EXPERIMENTS_DATA_DIR.parent
    / "NGED"
    / "substation_hierarchy_2026-10-15"
    / "Substations.csv"
).select("substation_number", "bsp_name")
oth = md.filter(~pl.col("time_series_type").is_in(["Disaggregated Demand", "Raw Flow"])).join(
    hier, on="substation_number", how="left"
)
print(oth.group_by("bsp_name", "time_series_type").len().sort("bsp_name", "time_series_type"))
months = [(f"{m}-01") for m in c.MONTHS]
df = (
    scan_power()
    .filter(pl.col("time") >= pl.datetime(2025, 10, 1, time_zone="UTC"))
    .filter(pl.col("time") < pl.datetime(2026, 7, 1, time_zone="UTC"))
    .collect()
)
df = df.with_columns(pl.col("time").dt.strftime("%Y-%m").alias("ym")).filter(
    pl.col("ym").is_in(list(c.MONTHS))
)
ids = prim["time_series_id"].to_list()
for bsp_id, bname in ((1, "BOSTON"), (2, "SKEGNESS")):
    pri = [ids[i] for i in nodes[bsp_id]]
    s_pri = (
        df.filter(pl.col("time_series_id").is_in(pri))
        .group_by("time")
        .agg(pl.col("power").sum().alias("prim"), pl.len().alias("n"))
        .filter(pl.col("n") == len(pri))
    )
    meter = df.filter(pl.col("time_series_id") == bsp_id).select(
        "time", pl.col("power").alias("meter")
    )
    j = s_pri.join(meter, on="time").with_columns(gap=pl.col("meter") - pl.col("prim"))
    gens = oth.filter(pl.col("bsp_name").str.contains(bname))
    for typ in sorted(set(gens["time_series_type"].cast(pl.Utf8).to_list())):
        sid = gens.filter(pl.col("time_series_type") == typ)["time_series_id"].to_list()
        g = (
            df.filter(pl.col("time_series_id").is_in(sid))
            .group_by("time")
            .agg(pl.col("power").sum().alias(typ))
        )
        j = j.join(g, on="time", how="left")
    types = [t for t in j.columns if t not in ("time", "prim", "n", "meter", "gap")]
    jj = j.drop_nulls(["gap"] + types) if types else j
    X = np.column_stack([jj[t].to_numpy() for t in types] + [np.ones(jj.height)]) if types else None
    y = jj["gap"].to_numpy()
    print(f"\n{bname}: rows {jj.height}, mean gap {y.mean():.2f} MW; series types {types}")
    if X is not None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        print(
            "  coefficients (MW of gap per MW of series):",
            dict(zip([*types, "const"], np.round(beta, 3), strict=True)),
        )
        print(
            f"  R squared {1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}; mean of each series:",
            {t: round(float(jj[t].mean()), 2) for t in types},  # ty: ignore[invalid-argument-type]
        )
        print(f"  std of gap {y.std():.2f}; std of residual {np.std(y - pred):.2f}")

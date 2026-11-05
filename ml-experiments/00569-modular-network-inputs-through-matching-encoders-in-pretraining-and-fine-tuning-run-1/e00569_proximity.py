# ruff: noqa: E501, D103, ANN201
"""Step 2 and 3 of experiment 569: schools near each Primary and their rank correlation with B3a's effect.

Reads the DfE "Get Information about Schools" register (Open Government Licence v3.0, downloaded
2026-11-05 into this experiment's private folder) and each Primary's coordinates. Writes
`school_proximity.csv` (one row per Primary: open schools and pupils within 2 km and 5 km).
"""

from pathlib import Path

import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from pyproj import Transformer
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = (
    ML_EXPERIMENTS_DATA_DIR
    / "00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1"
)
REGISTER = next(Path(OUT / "dfe_schools_register").glob("edubasealldata*.csv"))
RADII_KM = (2.0, 5.0)


def main():
    reg = pl.read_csv(REGISTER, encoding="latin1", infer_schema_length=0).filter(
        pl.col("EstablishmentStatus (name)") == "Open"
    )
    print("open establishments", reg.height)
    reg = reg.with_columns(
        east=pl.col("Easting").cast(pl.Float64, strict=False),
        north=pl.col("Northing").cast(pl.Float64, strict=False),
        pupils=pl.col("NumberOfPupils").cast(pl.Float64, strict=False).fill_null(0.0),
    ).filter(pl.col("east").is_not_null() & pl.col("north").is_not_null())
    print(
        "with coordinates",
        reg.height,
        "phases",
        reg["PhaseOfEducation (name)"].value_counts().sort("count", descending=True).head(6).rows(),
    )
    tr = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    prim = prep.primaries()
    rows = []
    for r in prim.iter_rows(named=True):
        e, n = tr.transform(r["longitude"], r["latitude"])
        dist = np.hypot(reg["east"].to_numpy() - e, reg["north"].to_numpy() - n) / 1000.0
        rec = {"primary": r["primary"], "label": r["site"]}
        for km in RADII_KM:
            m = dist <= km
            rec[f"schools_{int(km)}km"] = int(m.sum())
            rec[f"pupils_{int(km)}km"] = float(reg["pupils"].to_numpy()[m].sum())
        rec["nearest_km"] = float(dist.min())
        rows.append(rec)
    out = pl.DataFrame(rows)
    out.write_csv(OUT / "school_proximity.csv")
    with pl.Config(tbl_rows=20):
        print(out)
    eff = (
        pl.read_csv(OUT / "school_per_primary.csv")
        .select(
            "label",
            "real_minus_placebo",
            "real_minus_placebo_low",
            "real_minus_placebo_high",
            "m0_crps",
        )
        .join(out, on="label")
    )
    rng = np.random.default_rng(0)
    for col in ("schools_2km", "schools_5km", "pupils_2km", "pupils_5km"):
        for target in ("real_minus_placebo",):
            a, b = eff[col].to_numpy(), eff[target].to_numpy()
            rho = pl.DataFrame({"a": a, "b": b}).select(pl.corr("a", "b", method="spearman")).item()
            boot = []
            for _ in range(5000):
                i = rng.integers(0, len(a), len(a))
                v = (
                    pl.DataFrame({"a": a[i], "b": b[i]})
                    .select(pl.corr("a", "b", method="spearman"))
                    .item()
                )
                if not np.isnan(v):
                    boot.append(v)
            print(
                col,
                target,
                "rho",
                round(rho, 3),
                "95% bootstrap over Primaries",
                np.percentile(boot, [2.5, 97.5]).round(3),
            )


main()

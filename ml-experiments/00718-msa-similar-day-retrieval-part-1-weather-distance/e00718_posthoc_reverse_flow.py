# ruff: noqa: E501
"""Experiments 718 and 719: post hoc, report-only read-outs on reverse flow, from stored forecasts (CPU only, no fit).

Usage: e00718_posthoc_reverse_flow.py PART

Prints, in points of CRPS on the headline band (seed-averaged, all 8 folds): the two counted contrasts of the part at every Primary;
the eight with D8 left out and with D8 only; the six, the eight, and D8 on half-hours where the observed power is negative
(reverse flow) against other half-hours, where the observed power carries a sign; and the monthly means of the first contrast by
group. Nothing here changes a verdict.
"""

import json
import sys

import e00713_score as s713  # ty: ignore[unresolved-import]
import e00718_contrasts as con
import numpy as np
import polars as pl
from e00713_arms import ALL_FOLDS  # ty: ignore[unresolved-import]

BAND = s713.HEADLINE_BAND_HOURS
KEYS = list(s713.KEYS)


def main() -> int:
    """Print the post hoc tables."""
    part = int(sys.argv[1])
    spec = con.COUNTED[part]
    seeds = (0, 1)
    _per_seed, averaged, _groups, star = con._load(part=part, months=ALL_FOLDS, seeds=seeds)
    treat = spec["first"][1]
    ref1 = star
    ref2 = spec["placebo_arm"]
    sites = s713.k578.sites()  # ty: ignore[unresolved-attribute]
    labels = dict(zip(sites["time_series_id"].to_list(), sites["label"].to_list(), strict=True))

    def diff(frame_t: pl.DataFrame, frame_r: pl.DataFrame) -> pl.DataFrame:
        f = lambda x: x.filter((pl.col("lead_hours") >= BAND[0]) & (pl.col("lead_hours") < BAND[1]))  # noqa: E731
        return (
            f(frame_t)
            .select(*KEYS, "crps", "actual")
            .join(f(frame_r).select(*KEYS, pl.col("crps").alias("r")), on=KEYS, how="inner")
            .with_columns(d=pl.col("crps") - pl.col("r"))
        )

    out: dict = {}
    for name, ref in ((f"{treat}-{ref1}", ref1), (f"{treat}-{ref2}", ref2)):
        d = diff(averaged[treat], averaged[ref])
        d = d.with_columns(
            label=pl.col("time_series_id").replace_strict(labels, return_dtype=pl.String)
        )
        per_p = d.group_by("label").agg(pl.col("d").mean(), n=pl.len()).sort("label")
        out[name] = {
            "per_primary": {r["label"]: round(r["d"], 4) for r in per_p.iter_rows(named=True)}
        }
        eight = set(s713.EIGHT)
        six = set(s713.SIX)
        sub = {
            "six": d.filter(pl.col("label").is_in(list(six))),
            "eight": d.filter(pl.col("label").is_in(list(eight))),
            "eight_without_D8": d.filter(pl.col("label").is_in(list(eight - {"D8"}))),
            "D8_only": d.filter(pl.col("label") == "D8"),
        }
        out[name]["groups"] = {k: round(float(v["d"].mean()), 4) for k, v in sub.items()}  # ty: ignore[invalid-argument-type]
        out[name]["monthly"] = {
            k: {
                m: round(float(x), 3)
                for m, x in v.with_columns(m=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
                .group_by("m")
                .agg(pl.col("d").mean())
                .sort("m")
                .iter_rows()
            }
            for k, v in sub.items()
        }
        out[name]["by_sign_of_observed_power"] = {
            k: {
                "negative_forecast_rows": int((v["actual"] < 0).sum()),
                "negative_distinct_valid_times": int(
                    v.filter(pl.col("actual") < 0)["valid_time"].n_unique()
                ),
                "mean_d_negative": None
                if (v["actual"] < 0).sum() == 0
                else round(float(v.filter(pl.col("actual") < 0)["d"].mean()), 4),  # ty: ignore[invalid-argument-type]
                "mean_d_other": round(float(v.filter(pl.col("actual") >= 0)["d"].mean()), 4),  # ty: ignore[invalid-argument-type]
            }
            for k, v in sub.items()
        }
        out[name]["monthly_sd"] = {
            k: round(float(np.std(list(m.values()), ddof=1)), 3)
            for k, m in out[name]["monthly"].items()
        }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

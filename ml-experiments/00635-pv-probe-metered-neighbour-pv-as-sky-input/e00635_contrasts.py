# ruff: noqa: E501
"""Experiment 635: paired arm contrasts, per-Primary medians and the validation CRPS of the uninjected twins."""

import json

import e00635_common as k5
import numpy as np
import polars as pl

rec = pl.read_csv(k5.WORK / "recoveries.csv")
rng = np.random.default_rng(635)
out: dict = {}
real = rec.filter((pl.col("world") == "farm") & (pl.col("mw") == 4.0))
for other in ("b", "c"):
    key = ["seed", "site", "condition"]
    a = real.filter(pl.col("arm") == "a").select(
        *key, ra=pl.col("recovery"), ea=pl.col("recovery_energy")
    )
    o = real.filter(pl.col("arm") == other).select(
        *key, ro=pl.col("recovery"), eo=pl.col("recovery_energy")
    )
    j = (
        a.join(o, on=key)
        .group_by(["site", "condition"])
        .agg(
            (pl.col("ro") - pl.col("ra")).mean().alias("d"),
            (pl.col("eo") - pl.col("ea")).mean().alias("de"),
        )
    )
    d, de = j["d"].to_numpy(), j["de"].to_numpy()
    boot = [np.median(rng.choice(d, len(d))) for _ in range(5000)]
    out[f"{other}_minus_a_real4mw"] = {
        "median": float(np.median(d)),
        "lo90": float(np.percentile(boot, 5)),
        "hi90": float(np.percentile(boot, 95)),
        "median_energy": float(np.median(de)),
        "n_clusters": len(d),
    }
by_site = real.group_by(["arm", "site"]).agg(pl.col("recovery").median()).sort("arm", "site")
out["by_site_real4mw"] = {
    f"{r['arm']}|{r['site']}": round(r["recovery"], 3) for r in by_site.iter_rows(named=True)
}
crps = {}
for arm in ("a", "b", "c"):
    for cond in ("control_r0", "control_r1", "control_r0_mshuf"):
        v = [json.loads(f.read_text()) for f in (k5.WORK / "logs" / arm).glob(f"{cond}_s*.json")]
        if v:
            crps[f"{arm}|{cond}"] = {
                "crps": float(np.mean([x["crps"] for x in v])),
                "crps_six": float(np.mean([x["crps_six"] for x in v])),
                "n": len(v),
            }
out["crps"] = crps
(k5.WORK / "contrasts.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

# ruff: noqa: ANN001, ANN201, D103, F405, I001
"""Audit of experiment 696, step 4: the host's CRPS change split by whether the engine runs."""

import numpy as np
import polars as pl
from scipy import stats
from e00696_audit_common import *  # noqa: F403

L = pl.read_parquet(OUT / "losses.parquet")
gt = (
    pl.read_parquet(SH / "ground_truth_primaries.parquet")
    .filter(pl.col("time_series_id") == 11)
    .select("valid_time", real=pl.col("power") * 100 / HOST_CAPACITY_MVA)
)
H = (
    L.filter(pl.col("arm").str.starts_with("W"))
    .join(gt, on="valid_time", how="left")
    .with_columns(run=(pl.col("real") - pl.col("act")) > 0.5)
)


def seed_t(df, t, r, col="crps"):
    g = df.group_by("arm", "seed").agg(pl.col(col).mean())
    a = g.filter(pl.col("arm") == t).sort("seed")[col].to_numpy()
    b = g.filter(pl.col("arm") == r).sort("seed")[col].to_numpy()
    d = a - b
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.3f} [{d.mean() - h:+.3f},{d.mean() + h:+.3f}]"


for lab, f in (("running", pl.col("run")), ("off", ~pl.col("run"))):
    print(
        "host",
        lab,
        "WST-W0",
        seed_t(H.filter(f), "WST", "W0"),
        " WS1-W0",
        seed_t(H.filter(f), "WS1", "W0"),
        " MAE WST-W0",
        seed_t(H.filter(f), "WST", "W0", "mae"),
    )
# D8: tracked engine flag from experiment 694 via 695's helper is not used; proxy: engine months
D = L.filter(pl.col("arm").is_in(["A0", "S1", "SP"]))
for lab, f in (
    ("engine months Nov-Feb", pl.col("valid_time").dt.month().is_in([11, 12, 1, 2])),
    ("fade months Mar-Jun", ~pl.col("valid_time").dt.month().is_in([11, 12, 1, 2])),
):
    print(
        "D8",
        lab,
        "S1-A0",
        seed_t(D.filter(f), "S1", "A0"),
        " S1-SP",
        seed_t(D.filter(f), "S1", "SP"),
    )

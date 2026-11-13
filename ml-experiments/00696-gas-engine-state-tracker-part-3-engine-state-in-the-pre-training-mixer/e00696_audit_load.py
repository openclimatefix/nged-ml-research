# ruff: noqa: ANN001, ANN201, B905, D103, F401, F405, PERF401, UP031
"""Audit of experiment 696, step 1: per-row losses of every arm from the raw fits.

Writes `audit/losses.parquet` (D8 and the host) and `audit/losses_all_primaries.parquet`
(S1 and A0 at all 16 Primaries).
"""

import datetime as dt
import json

import numpy as np
import polars as pl
from e00696_audit_common import *  # noqa: F403


def path(arm, m, s):
    return str(fits_path(arm, m, s))


caps = pl.read_parquet(SH / "series_capacity.parquet")
cap = dict(zip(caps["time_series_id"].cast(pl.Int64).to_list(), caps["capacity"].to_list()))
# host id: series where W0 and A0 fits differ most (independent of 695 lookup)
a = pl.read_parquet(path("A0", "2025-11", 3))
w = pl.read_parquet(path("W0", "2025-11", 3))
j = a.join(w, on=["time_series_id", "power_fcst_init_time", "valid_time"], suffix="_w")
d = (
    j.group_by("time_series_id")
    .agg((pl.col("q50") - pl.col("q50_w")).abs().mean().alias("d"))
    .sort("d", descending=True)
)
print(
    d.head(4)
)  # host = series whose real reading equals the W0 stack row off-engine (e00696_audit_host.py)
print("host id", HOST, "cap", cap[HOST], "D8 cap", cap[D8])
gt = pl.read_parquet(SH / "ground_truth_primaries.parquet").with_columns(
    pl.col("time_series_id").cast(pl.Int64)
)
d8t = gt.filter(pl.col("time_series_id") == D8).select("valid_time", "power")
# host truth: 695 stack W0 row (fractions) * cap; check months agree
vals = [np.load(E5 / "stacks" / "W0" / f"{m}.npz")["values"] for m in MONTHS]
print(
    "W0 stack row",
    int(np.load(E5 / "stacks" / "W0" / "2025-11.npz")["row"]),
    "months identical:",
    all(np.array_equal(vals[0], v, equal_nan=True) for v in vals),
)
y = vals[-1].astype(np.float64) * cap[HOST]
start = np.datetime64("2019-09-01T00:00", "us")
ht = pl.DataFrame(
    {
        "valid_time": pl.Series(
            start + np.arange(len(y)) * np.timedelta64(1800, "s")
        ).dt.replace_time_zone("UTC"),
        "power": pl.Series(y, nan_to_null=True),
    }
)
# also the real host reading (no engine) from gt, to compare
hreal = gt.filter(pl.col("time_series_id") == HOST).select("valid_time", real=pl.col("power"))
chk = ht.join(hreal, on="valid_time").filter(
    pl.col("valid_time") >= dt.datetime(2025, 11, 1, tzinfo=dt.UTC),
    pl.col("valid_time") < dt.datetime(2026, 7, 1, tzinfo=dt.UTC),
)
print(
    "host truth minus real reading (MVA): mean %.3f, share nonzero %.3f, nulls truth %d real %d"
    % (
        (chk["power"] - chk["real"]).mean(),
        ((chk["power"] - chk["real"]).abs() > 1e-6).mean(),
        chk["power"].null_count(),
        chk["real"].null_count(),
    )
)


def losses(arm, s, series, truth, c):
    fr = []
    for m in MONTHS:
        f = pl.read_parquet(path(arm, m, s)).filter(pl.col("time_series_id") == series)
        fr.append(f)
    f = pl.concat(fr).with_columns(origin=pl.col("power_fcst_init_time") + pl.duration(hours=3))
    assert (f["power_fcst_init_time"].dt.hour() == 9).all()
    lead = (pl.col("valid_time") - pl.col("origin")).dt.total_minutes() / 60
    f = f.filter(
        pl.col("origin") >= dt.datetime(2025, 11, 1, tzinfo=dt.UTC),
        pl.col("origin") < dt.datetime(2026, 7, 1, tzinfo=dt.UTC),
        pl.col("valid_time") < dt.datetime(2026, 7, 1, tzinfo=dt.UTC),
        lead >= 24,
        lead <= 240,
    )
    f = f.join(truth, on="valid_time", how="inner").filter(pl.col("power").is_not_null())
    qs = (
        np.sort(f.select(Q).to_numpy().astype(np.float64), axis=1) * 100
    )  # fits are fractions of capacity
    act = f["power"].to_numpy().astype(np.float64) * 100 / c
    e = act[:, None] - qs
    pb = np.maximum(LV * e, (LV - 1) * e)
    return f.select("origin", "valid_time").with_columns(
        arm=pl.lit(arm),
        seed=pl.lit(s),
        crps=pl.Series(2 * pb.mean(1)),
        mae=pl.Series(np.abs(act - qs[:, 5])),
        mean_minus_med=pl.Series(qs.mean(1) - qs[:, 5]),
        act=pl.Series(act),
        q50=pl.Series(qs[:, 5]),
        lead=lead,
    )


frames = []
for arm in ("A0", "S1", "SP", "S1d", "W0", "WST", "WS1"):
    host = arm.startswith("W")
    seeds = range(3) if arm == "S1d" else range(10)
    for s in seeds:
        frames.append(
            losses(
                arm,
                s,
                HOST if host else D8,
                ht if host else d8t,
                cap[HOST] if host else D8_CAPACITY_MVA,
            )
        )
L = pl.concat(frames).with_columns(month=pl.col("origin").dt.strftime("%Y-%m"))
L.write_parquet(OUT / "losses.parquet")
print(
    L.group_by("arm")
    .agg(pl.len(), pl.col("crps").mean(), pl.col("mae").mean(), pl.col("mean_minus_med").mean())
    .sort("arm")
)
# other primaries: S1 vs A0 at all 16
oth = []
for arm in ("A0", "S1"):
    for s in range(10):
        f = pl.concat([pl.read_parquet(path(arm, m, s)) for m in MONTHS]).with_columns(
            pl.col("time_series_id").cast(pl.Int64),
            origin=pl.col("power_fcst_init_time") + pl.duration(hours=3),
        )
        lead = (pl.col("valid_time") - pl.col("origin")).dt.total_minutes() / 60
        f = f.filter(
            pl.col("origin") < dt.datetime(2026, 7, 1, tzinfo=dt.UTC),
            pl.col("valid_time") < dt.datetime(2026, 7, 1, tzinfo=dt.UTC),
            lead >= 24,
            lead <= 240,
        )
        f = f.join(
            gt.select("time_series_id", "valid_time", "power"), on=["time_series_id", "valid_time"]
        ).filter(pl.col("power").is_not_null())
        c = f["time_series_id"].replace_strict(cap).to_numpy()
        qs = np.sort(f.select(Q).to_numpy().astype(np.float64), axis=1) * 100
        act = f["power"].to_numpy().astype(np.float64) * 100 / c
        e = act[:, None] - qs
        pb = np.maximum(LV * e, (LV - 1) * e)
        oth.append(
            f.select("time_series_id", "origin", "valid_time").with_columns(
                arm=pl.lit(arm), seed=pl.lit(s), crps=pl.Series(2 * pb.mean(1))
            )
        )
pl.concat(oth).with_columns(month=pl.col("origin").dt.strftime("%Y-%m")).write_parquet(
    OUT / "losses_all_primaries.parquet"
)

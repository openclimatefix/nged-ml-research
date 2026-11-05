# ruff: noqa: ANN001, D103, D205
"""Experiment 578: lag effect of the stored v3+D5 forecasts on the labelled events inside the scored
months.

Per event (any length, 15 events have at least 3 days of gap), at 24-120 h, for valid times inside
the event, three ratios to the labelled depth (signed capacity fraction):

- `lag_effect`: (v3+D5 median minus its lags-masked twin median) / depth. The twin is experiment
  572's `V3D5-removed`: the same fitted weights with both lag inputs set to the missing state, so
  this is the share of the depth that reading the lags adds.

- `vs_lag_free`: (v3+D5 median minus E2 median) / depth.

- `e2_minus_masked`: (E2 median minus the lags-masked twin) / depth, descriptive.

The counterfactual-corrected-lag forecast of the brief is not stored, so this is a stand-in. Medians
are seed means. Weather timing: every row comes from a stored origin, whose ensemble run is the one
issued at or before the origin minus 9 h; one example row is printed.
"""

import datetime as dt

import e00572_score as sc  # ty: ignore[unresolved-import]
import e00578_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]


def load(*, root, tag: str) -> pl.DataFrame:
    """Return the seed-mean q50 of an arm over every fold month and cutoff."""
    parts = []
    for m in k.MONTHS:
        for d in ("d0", "d14"):
            for s in (0, 1):
                f = root / "fits" / tag / f"{m}_{d}_s{s}.parquet"
                parts.append(pl.read_parquet(f, columns=[*KEYS, "q50"]))
    return pl.concat(parts).group_by(KEYS).agg(pl.col("q50").mean())


def main() -> None:
    sm = k.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    ev = pl.read_csv(k.WORK / "step0_events.csv")
    v = load(root=sc.W572, tag="V3D5").rename({"q50": "v"})
    r = load(root=sc.W572, tag="V3D5-removed").rename({"q50": "r"})
    e = load(root=k.d6.W, tag="D5").rename({"q50": "e"})
    j = (
        v.join(r, on=KEYS)
        .join(e, on=KEYS)
        .with_columns(
            lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
            prim=pl.col("time_series_id").replace_strict(prim_of),
        )
    )
    assert (j["valid_time"] > j["power_fcst_init_time"]).all()
    x = j.row(0, named=True)
    print(
        "example row:", x, "| weather run issued", x["power_fcst_init_time"] - dt.timedelta(hours=9)
    )
    out = []
    for q in ev.iter_rows(named=True):
        t0 = k.GRID_START + dt.timedelta(minutes=30 * q["onset_idx"])
        t1 = k.GRID_START + dt.timedelta(minutes=30 * q["end_idx"])
        depth = q["depth_share"] / 100 * np.sign(q["depth_mw"])
        d = j.filter(
            (pl.col("prim") == q["primary"])
            & pl.col("valid_time").is_between(t0, t1)
            & (pl.col("lead") >= 24)
            & (pl.col("lead") < 120)
        ).with_columns(
            origin_day=((pl.col("power_fcst_init_time") - t0).dt.total_hours() // 24).cast(
                pl.Int64
            ),
            lag_effect=(pl.col("v") - pl.col("r")) / depth,
            vs_lag_free=(pl.col("v") - pl.col("e")) / depth,
            e2_minus_masked=(pl.col("e") - pl.col("r")) / depth,
        )
        if d.height == 0:
            continue
        late = d.filter(pl.col("origin_day") >= 4)
        out.append(
            {
                "event_id": q["event_id"],
                "label": q["label"],
                "length_days": q["length_days"],
                "depth_share": q["depth_share"],
                "long_deep": q["long_deep"],
                "rows": d.height,
                "lag_effect": d["lag_effect"].mean(),
                "lag_effect_after_day4": late["lag_effect"].mean() if late.height else None,
                "vs_lag_free": d["vs_lag_free"].mean(),
                "e2_minus_masked": d["e2_minus_masked"].mean(),
            }
        )
    t = pl.DataFrame(out)
    t.write_csv(k.WORK / "follow_lag_effect.csv")
    with pl.Config(tbl_rows=30, tbl_cols=12):
        print(t)
    ld = t.filter(pl.col("long_deep"))
    print(
        "long and deep (3 events): mean lag_effect",
        ld["lag_effect"].mean(),
        "per-event",
        ld["lag_effect"].to_list(),
    )
    print(
        "all 15: mean",
        t["lag_effect"].mean(),
        "median",
        t["lag_effect"].median(),
        "max",
        t["lag_effect"].max(),
    )


main()

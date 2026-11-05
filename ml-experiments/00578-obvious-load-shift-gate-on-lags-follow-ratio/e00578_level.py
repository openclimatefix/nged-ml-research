# ruff: noqa: E501, D103
"""Experiment 578: how much of an event's level each forecast reproduces (the NRA concern).

Per event, at 24-120 h, for valid times inside the event: level share = sum(forecast - counterfactual) /
sum(observed - counterfactual), where the counterfactual is the NRA-corrected power (event removed) and the
forecast is the seed-mean median. 0 = the forecast sits at the counterfactual level, 1 = it reproduces the
event's level. Arms: v3+D5 with observed lags, v3+D5 with corrected lags, v3+D5 with masked lags, and the
lag-free D5 network (E2, five old folds). Only half-hours with a finite observation and counterfactual count.
"""

import datetime as dt

import e00578_common as k  # ty: ignore[unresolved-import]
import e00578_follow2 as f2  # ty: ignore[unresolved-import]
import polars as pl


def main() -> None:
    cl, sh = k.clean(), k.corrected_shift()
    sm = k.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    allm = (*k.MONTHS, *f2.NEW)
    arms = {
        "v3+D5 observed lags": ("V3D5", allm),
        "v3+D5 corrected lags": ("V3D5-corrected", allm),
        "v3+D5 masked lags": ("V3D5-removed", allm),
    }
    frames = {
        n: f2.load(root=f2._Root(), tag=t, months=m).rename({"q50": n})
        for n, (t, m) in arms.items()
    }
    frames["lag-free D5 (E2)"] = f2.load(root=k.d6.W, tag="D5", months=k.MONTHS).rename(
        {"q50": "lag-free D5 (E2)"}
    )
    j = frames["v3+D5 observed lags"]
    for n in list(frames)[1:]:
        j = j.join(frames[n], on=f2.KEYS, how="left")
    j = j.with_columns(
        lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
        prim=pl.col("time_series_id").replace_strict(prim_of),
        idx=((pl.col("valid_time").dt.epoch("us") - k.prep.GRID_START_US) // k.prep.STEP_US).cast(
            pl.Int64
        ),
    ).filter((pl.col("lead") >= 24) & (pl.col("lead") < 120))
    obs = cl[j["prim"].to_numpy(), j["idx"].to_numpy()]
    cf = obs + sh[j["prim"].to_numpy(), j["idx"].to_numpy()]
    j = j.with_columns(obs=pl.Series(obs), cf=pl.Series(cf)).filter(
        pl.col("obs").is_finite() & pl.col("cf").is_finite()
    )
    ev = pl.read_csv(k.WORK / "step0_events.csv").filter(pl.col("long"))
    out = []
    for q in ev.iter_rows(named=True):
        t0 = k.GRID_START + dt.timedelta(minutes=30 * q["onset_idx"])
        t1 = k.GRID_START + dt.timedelta(minutes=30 * q["end_idx"])
        d = j.filter((pl.col("prim") == q["primary"]) & pl.col("valid_time").is_between(t0, t1))
        if d.height < 200:
            continue
        row = {
            "event_id": q["event_id"],
            "label": q["label"],
            "long_deep": q["long_deep"],
            "rows": d.height,
        }
        for n in (*arms, "lag-free D5 (E2)"):
            dd = d.drop_nulls(n)
            row[n] = (
                float((dd[n] - dd["cf"]).sum() / (dd["obs"] - dd["cf"]).sum())  # ty: ignore[unsupported-operator]
                if dd.height > 100
                else None
            )
        out.append(row)
    t = pl.DataFrame(out)
    t.write_csv(k.WORK / "level_share.csv")
    with pl.Config(tbl_rows=30, tbl_cols=10, tbl_width_chars=200):
        print(t.with_columns(pl.col(pl.Float64).round(3)))
    for n in (*arms, "lag-free D5 (E2)"):
        s = t.drop_nulls(n)
        print(
            n,
            "n",
            s.height,
            "median",
            round(float(s[n].median()), 3),  # ty: ignore[invalid-argument-type]
            "mean",
            round(float(s[n].mean()), 3),  # ty: ignore[invalid-argument-type]
        )


main()

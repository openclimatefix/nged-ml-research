# ruff: noqa: E501, ANN001, D103, PERF401, ANN204
"""Experiment 578: follow ratio of the lag network with observed lags against NRA-corrected lags (and masked lags).

Follow ratio of an event = mean over valid times inside the event, forecasts issued 24 to 120 h earlier, of
(median with observed lags minus median with corrected lags) / labelled depth (signed capacity fraction).
Also with masked lags (the lags-masked twin), and for the lag-free D5 network (E2, five old folds only):
(E2 minus corrected-lag forecast) / depth, i.e. how much of the counterfactual gap E2 keeps. Bins by days
from onset to the forecast origin: 0-3, 4-7, 8-14, over 14, and after the end (the five days after).
Sanity check first: the re-forecast v3+D5 of the five old folds equals experiment 572's stored forecasts.
"""

import datetime as dt

import e00572_score as sc  # ty: ignore[unresolved-import]
import e00578_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
NEW = ("2025-07", "2025-08")


def load(*, root, tag: str, months) -> pl.DataFrame:
    parts = []
    for m in months:
        for d in ("d0", "d14"):
            for s in (0, 1):
                parts.append(
                    pl.read_parquet(
                        root / "fits" / tag / f"{m}_{d}_s{s}.parquet", columns=[*KEYS, "q50"]
                    )
                )
    return pl.concat(parts).group_by(KEYS).agg(pl.col("q50").mean())


def main() -> None:
    sm = k.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    group_of = dict(zip(sm["primary"].to_list(), sm["group"].to_list(), strict=True))
    allm = (*k.MONTHS, *NEW)
    # sanity: reproduction of experiment 572
    mine = load(root=_Root(), tag="V3D5", months=k.MONTHS)
    old = load(root=sc.W572, tag="V3D5", months=k.MONTHS)
    j = mine.join(old, on=KEYS, suffix="_old")
    diff = (j["q50"] - j["q50_old"]).abs()
    print(
        "reproduction of 572 v3+D5: rows",
        j.height,
        "of",
        old.height,
        "max abs diff",
        float(diff.max()),  # ty: ignore[invalid-argument-type]
        "mean",
        float(diff.mean()),  # ty: ignore[invalid-argument-type]
    )
    v = load(root=_Root(), tag="V3D5", months=allm).rename({"q50": "v"})
    c = load(root=_Root(), tag="V3D5-corrected", months=allm).rename({"q50": "c"})
    r = load(root=_Root(), tag="V3D5-removed", months=allm).rename({"q50": "r"})
    e = load(root=k.d6.W, tag="D5", months=k.MONTHS).rename({"q50": "e"})
    jj = (
        v.join(c, on=KEYS)
        .join(r, on=KEYS)
        .join(e, on=KEYS, how="left")
        .with_columns(
            lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
            prim=pl.col("time_series_id").replace_strict(prim_of),
        )
        .filter((pl.col("lead") >= 24) & (pl.col("lead") < 120))
    )
    cl = k.clean()
    cf = np.where(
        np.isfinite(cl), cl + k.corrected_shift(), np.nan
    )  # counterfactual power, no event
    gi = (
        (jj["valid_time"].dt.epoch("us").to_numpy() - k.prep.GRID_START_US) // k.prep.STEP_US
    ).astype(np.int64)
    pi = jj["prim"].to_numpy()
    jj = jj.with_columns(cf=pl.Series(cf[pi, gi]))
    ev = pl.read_csv(k.WORK / "step0_events.csv").filter(pl.col("long"))
    out = []
    for q in ev.iter_rows(named=True):
        t0 = k.GRID_START + dt.timedelta(minutes=30 * q["onset_idx"])
        t1 = k.GRID_START + dt.timedelta(minutes=30 * q["end_idx"])
        depth = q["depth_share"] / 100 * np.sign(q["depth_mw"])
        d = jj.filter(
            (pl.col("prim") == q["primary"]) & pl.col("valid_time").is_between(t0, t1)
        ).with_columns(
            od=((pl.col("power_fcst_init_time") - t0).dt.total_hours() // 24).cast(pl.Int64),
            follow=(pl.col("v") - pl.col("c")) / depth,
            follow_masked=(pl.col("v") - pl.col("r")) / depth,
            e2_vs_corrected=(pl.col("e") - pl.col("c")) / depth,
            level_v=(pl.col("v") - pl.col("cf")) / depth,
            level_c=(pl.col("c") - pl.col("cf")) / depth,
            level_e2=(pl.col("e") - pl.col("cf")) / depth,
        )
        if d.height < 200:
            continue
        bins = {"d0-3": (0, 4), "d4-7": (4, 8), "d8-14": (8, 15), "d15+": (15, 10**6)}
        row = {
            "event_id": q["event_id"],
            "label": q["label"],
            "group": group_of[q["primary"]],
            "days": q["length_days"],
            "depth_share": q["depth_share"],
            "long_deep": q["long_deep"],
            "rows": d.height,
            "follow": d["follow"].mean(),
            "follow_masked": d["follow_masked"].mean(),
            "e2_vs_corrected": d["e2_vs_corrected"].mean(),
            "level_v": d["level_v"].mean(),
            "level_c": d["level_c"].mean(),
            "level_e2": d["level_e2"].mean(),
        }
        for n, (lo, hi) in bins.items():
            s = d.filter((pl.col("od") >= lo) & (pl.col("od") < hi))
            row[f"follow_{n}"] = s["follow"].mean() if s.height >= 50 else None
        out.append(row)
    t = pl.DataFrame(out)
    t.write_csv(k.WORK / "follow_corrected.csv")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=250):
        print(
            t.select(
                "event_id",
                "label",
                "group",
                "days",
                "depth_share",
                "long_deep",
                "rows",
                "follow",
                "follow_masked",
                "e2_vs_corrected",
                "follow_d0-3",
                "follow_d4-7",
                "follow_d8-14",
                "follow_d15+",
            ).with_columns(pl.col(pl.Float64).round(3))
        )
    print(
        t.select("event_id", "label", "long_deep", "level_v", "level_c", "level_e2").with_columns(
            pl.col(pl.Float64).round(3)
        )
    )
    ld = t.filter(pl.col("long_deep"))
    print(
        "after day 4, long and deep, mean follow:",
        float(
            np.nanmean(
                np.concatenate(
                    [ld[f"follow_{b}"].drop_nulls().to_numpy() for b in ("d4-7", "d8-14", "d15+")]
                )
            )
        ),
    )
    for name, s in (("long and deep", t.filter(pl.col("long_deep"))), ("all long", t)):
        print(
            name,
            "n",
            s.height,
            "mean follow",
            s["follow"].mean(),
            "median",
            s["follow"].median(),
            "n>0.15",
            int((s["follow"] > 0.15).sum()),
        )


class _Root:
    """The work folder of this experiment, in the `root / 'fits' / tag` layout."""

    def __truediv__(self, other):
        return k.WORK / other


if __name__ == "__main__":
    main()

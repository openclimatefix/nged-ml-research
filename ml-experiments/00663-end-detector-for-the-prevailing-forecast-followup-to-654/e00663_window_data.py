# ruff: noqa: ANN001, D103, E501, PLR0915, ANN201
"""Experiment 663, window step 1: R-haz's inputs for the final-test window (CPU).

A copy of experiment 647's `build_real` (the `real_<month>.npz` head data) for experiment 620's
K1 export of the 2026-07 fold (origins 2026-07-01 to 2026-10-31), through experiment 620's
extended inputs. The lag-free table is experiment 620's `tables/predict/T_2026-07_s0.npz`. Nothing
here is tuned on the window. Writes `window/real_window.npz` with the keys of `real_<month>.npz`.

Usage: python e00663_window_data.py
"""

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00598_signals as s598  # ty: ignore[unresolved-import]
import e00620_extend as ext  # ty: ignore[unresolved-import]
import e00645_score as k645  # ty: ignore[unresolved-import]
import e00647_common as c647  # ty: ignore[unresolved-import]
import e00647_data as d647  # ty: ignore[unresolved-import]
import e00663_common as c663
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

PER = d647.PER
E620 = next(ML_EXPERIMENTS_DATA_DIR.glob("00620-*"))
OUT = c663.WORK / "window"


def main() -> None:
    ext.redirect()
    OUT.mkdir(parents=True, exist_ok=True)
    clean = k578.clean()
    nb = d647.world()
    z = np.load(E620 / "tables" / "predict" / "T_2026-07_s0.npz")
    table = s598.LagFreeTable(median=z["median"], origin_idx=z["origin_idx"])
    flags = k645.ara_flags(clean=clean)
    n_days = clean.shape[1] // PER
    flags = np.concatenate([flags, np.zeros((16, n_days - flags.shape[1]), bool)], axis=1)
    sub = pl.read_csv(c647.EXPORT / "substations.csv").select("time_series_id", "capacity")
    sm = k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    t0, step = k578.prep.GRID_START_US, k578.prep.STEP_US
    df = (
        pl.read_parquet(E620 / "export" / "K1_2026-07.parquet")
        .join(sub, on="time_series_id", how="left", validate="m:1")
        .with_columns(
            *(pl.col(c) / pl.col("capacity") for c in (*c647.QNAMES, "observed_power", "nra_power")),
            vi=((pl.col("valid_time").dt.epoch("us") - t0) // step).cast(pl.Int64),
            oi=((pl.col("power_fcst_init_time").dt.epoch("us") - t0) // step).cast(pl.Int64),
        )
    )
    assert df["capacity"].null_count() == 0
    prim = np.array([prim_of[int(t)] for t in df["time_series_id"].to_list()], np.int64)
    oi, vi, lead = df["oi"].to_numpy(), df["vi"].to_numpy(), df["lead_hours"].to_numpy()
    q = df.select(c647.QNAMES).to_numpy().astype(np.float32)
    y = df["observed_power"].to_numpy().astype(np.float32)
    res = {
        name: d647.origin_residual(prim=prim, oi=oi, vi=vi, lead=lead, q50=q[:, 5], obs=y, first=a, last=b, n_days=n_days)
        for name, (a, b) in d647.R0_SPECS.items()
    }
    keep = lead >= 24.0
    org_key = prim * 10_000_000 + oi
    uniq, org = np.unique(org_key, return_inverse=True)
    o_prim, o_oi = (uniq // 10_000_000).astype(np.int64), (uniq % 10_000_000).astype(np.int64)
    feat, age, lag_age = d647.origin_features(prim=o_prim, oi=o_oi, power_lag=clean, power_sig=clean, table=table, nb=nb)
    gate = np.zeros(len(uniq), bool)
    np.logical_or.at(gate, org, df["gate_on"].fill_null(False).to_numpy())
    np.savez(
        OUT / "real_window.npz",
        prim=prim[keep].astype(np.int8), oi=oi[keep].astype(np.int32), vi=vi[keep].astype(np.int32),
        lead=lead[keep].astype(np.float32), q=q[keep], y=y[keep], event_free=df["event_free"].to_numpy()[keep],
        verified=df["in_verified_event"].to_numpy()[keep], flagged=flags[prim[keep], vi[keep] // PER],
        org=org[keep].astype(np.int32), res=res["res"][keep], res_star=res["res_star"][keep],
        feat=feat, age=age, lag_age=lag_age, gate=gate, org_prim=o_prim.astype(np.int8), org_oi=o_oi.astype(np.int32),
    )  # fmt: skip
    print(f"window real: {int(keep.sum())} rows, {len(uniq)} origins", flush=True)


if __name__ == "__main__":
    main()

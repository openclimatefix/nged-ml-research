# ruff: noqa: E501
"""Experiment 647, stage 3: check that the E2 encoder copy and the E3 NRA branch start as K1 itself (GPU).

For the scored month and K1 seed: (1) the copied lag encoder's embedding equals the frozen one on
real rows of both refits; (2) the E3 NRA branch's quantiles before any training equal the stored
per-seed K1 forecast of experiment 613 (matched on Primary, origin, and valid time).

Usage: python e00647_enc_check.py MONTH SEED
"""

import sys

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00647_enc as enc  # ty: ignore[unresolved-import]
import e00647_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

month, seed = sys.argv[1], int(sys.argv[2])
d = fit.load(month=month)["real"]
idx = np.flatnonzero(d["usable"])[::7][:30_000]
rows = enc.Rows(parts=[(month, "real", idx)], stats=None)
one = (torch.zeros(64, device="cuda"), torch.ones(64, device="cuda"))
net = enc.Net(arm="E3", month=month, seed=seed, hl_stats=one, n_in=62 + 3 + 64 + 1)
net.eval()
sel = torch.arange(rows.n, device="cuda")
with torch.no_grad():
    own = net.hl(rows, sel)
    froz = enc.Net(arm="E1", month=month, seed=seed, hl_stats=one, n_in=1).frozen_hl(rows, sel)
    _out, q = net.nra(rows, sel, own)
print(
    f"{month} s{seed}: max |hl copy - hl frozen| = {float((own - froz).abs().max()):.3e} over {rows.n} rows"
)
base = enc.ROOT / "00613-nra-roster-contrasts-with-intervals-five-seeds/arms/k1_d5/work/fits/P5"
st = pl.concat([pl.read_parquet(base / f"{month}_d{dd}_s{seed}.parquet") for dd in (0, 14)])
step, t0 = c510.prep422.STEP_US, c510.prep422.GRID_START_US
tbl = (
    prep.primaries()
    .select("primary", "time_series_id")
    .with_columns(pl.col("time_series_id").cast(pl.Int32))
)
mine = (
    pl.DataFrame(
        {
            "primary": d["prim"][idx].astype(np.int64),
            "o": t0 + d["oi"][idx].astype(np.int64) * step,
            "v": t0 + d["vi"][idx].astype(np.int64) * step,
            "row": np.arange(len(idx)),
        }
    )
    .with_columns(
        pl.col("o").cast(pl.Datetime("us", "UTC")), pl.col("v").cast(pl.Datetime("us", "UTC"))
    )
    .join(tbl, on="primary", how="left")
)
j = mine.join(
    st,
    left_on=["time_series_id", "o", "v"],
    right_on=["time_series_id", "power_fcst_init_time", "valid_time"],
)
cols = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
stored = j.select(cols).to_numpy()
mine_q = q.cpu().numpy()[j["row"].to_numpy()]
print(
    f"{month} s{seed}: matched {j.height} of {len(idx)} rows; max |E3 NRA branch (untrained) - stored K1 seed forecast| = {np.abs(mine_q - stored).max():.3e} of capacity"
)

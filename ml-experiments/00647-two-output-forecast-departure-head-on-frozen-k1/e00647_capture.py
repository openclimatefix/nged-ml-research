# ruff: noqa: ANN001, ANN201, ANN202, D103, E402, E501, PLR0915, PT018
"""Experiment 647, stage 3, step 1: check K1's stored weights and cache the encoder inputs (GPU).

Runs experiment 591's A1 pipeline in inference mode (as experiment 613's `inject`) for one month and

1. (`--check`) loads the stored weights of one d0 and one d14 model for K1 seeds 0 and 2, rebuilds the
   month's test tensors, and compares the rebuilt forecasts with the stored per-seed forecasts of
   experiment 613 (real rows, and one injected-bank pass), and compares the lag-encoder output
   rebuilt from cached inputs with the network's own `lag_embedding`;
2. caches, for every row of the month's head data (the rows of `head_data/real_<m>.npz` and
   `bank_<m>.npz`, in the same order), the full network input row, the primary, the refit indicator
   (0: the d0 model's window, 1: the d14 model's), and the expected-input table rows at the day-lag
   and week-lag times. Nothing here changes a stored file of another experiment.

Usage: python e00647_capture.py MONTH [--check]
"""

import sys

month = sys.argv[1]
CHECK = "--check" in sys.argv
META = "--meta" in sys.argv
sys.argv = ["e00591_arm.py", "A1", "inject", "--months", month, "--seeds", "0"]

import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00591_arm as arm  # ty: ignore[unresolved-import]
import e00647_common as c647  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

e = arm.e
ROOT = ML_EXPERIMENTS_DATA_DIR
MODELS = {
    s: ROOT / ("00608-nra-combined-networks-per-class-and-leave-one-out" if s < 2 else "00613-nra-roster-contrasts-with-intervals-five-seeds") / "arms/k1_d5/work"
    for s in range(5)
}  # fmt: skip
STORED = ROOT / "00613-nra-roster-contrasts-with-intervals-five-seeds/arms/k1_d5/work"
OUT = c647.WORK / "cache"
OUT.mkdir(parents=True, exist_ok=True)
SL = slice(arm.LAG_FIRST, arm.LAG_FIRST + arm.LAG_WIDTH)


def weights(*, seed: int, day: int) -> str:
    return str(MODELS[seed] / "models" / "P5" / f"{month}_d{day}_s{seed}.pt")


def load_model(*, seed: int, day: int, n_primaries: int):
    m = e.new_model(n_primaries=n_primaries).to("cuda")
    m.load_state_dict(torch.load(weights(seed=seed, day=day), map_location="cuda"))
    m.eval()
    m.table = e.CTX["table_ft"]
    return m


def table_rows(*, x: np.ndarray, prim: np.ndarray) -> np.ndarray:
    """Return the expected-input table rows at the two lag times, (rows, 2, n_cols), as stored."""
    out = []
    t = e.CTX["table_ft"]
    for j in range(2):
        idx = torch.from_numpy(x[:, 43 + j].astype(np.int64)).cuda()
        row = t[torch.from_numpy(prim.astype(np.int64)).cuda(), idx.clamp(min=0)]
        row = torch.where((idx >= 0)[:, None], row, torch.zeros_like(row))
        out.append(row.cpu().numpy())
    return np.stack(out, axis=1)


def encode_cached(*, model, x, prim, rows):
    """The network's lag encoder rebuilt from cached inputs (what E1 to E3 use)."""
    emb = model.embedding(prim)
    outs = []
    for j, cols in enumerate((arm.mymodel.m560.DAY, arm.mymodel.m560.WEEK)):
        row = rows[:, j].float()
        pad = torch.zeros(len(row), model.n_b1, device=row.device)
        hw = torch.nn.functional.gelu(model.weather(torch.cat([row[:, :7], emb, pad], dim=-1)))
        hd = torch.nn.functional.gelu(model.datetime(torch.cat([row[:, 7:], emb], dim=-1)))
        parts = [
            x[:, cols],
            hw,
            hd,
            emb,
            model.lag_id(torch.full((len(x),), j, dtype=torch.long, device=x.device)),
        ]
        outs.append(model.lag[0](torch.cat(parts, dim=-1)))
    return torch.nn.functional.gelu(torch.cat(outs, dim=-1))


def main() -> None:
    clean = e.c.clean_stack()
    mr = arm._build_month_p5(month=month, clean=clean)
    test = e.build_test(month=month, standardiser=mr.base.standardiser)
    e.CTX["variants"], e.CTX["on"] = {}, False
    n_prim = len(mr.base.target_mean)
    cuts = e.k.cutoffs(month=month, cadence=e.CADENCE)
    windows = [e.k.window_of(month=month, cadence=e.CADENCE, index=i) for i in range(len(cuts))]
    days = [e.f.day_of(month=month, cut=c) for c in cuts]
    assert days == [0, 14], days
    print(month, "test x", test.x.shape, test.x.dtype, "windows", windows, flush=True)
    np.savez(
        OUT / f"{month}_meta.npz",
        target_mean=np.asarray(mr.base.target_mean),
        target_std=np.asarray(mr.base.target_std),
        windows=np.asarray(windows),
        n_knots=e.KNOTS,
        n_b1=e.N_B1,
    )
    if "--meta" in sys.argv[1:] or META:
        return
    start = e.c.month_start(month=month)
    start_idx = (e.c.us(t=start) - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
    key = (
        test.primary.astype(np.int64) * 10_000_000 + test.origin_idx
    ) * 10_000_000 + test.target_idx
    order = np.argsort(key)
    skey = key[order]

    def locate(*, prim, oi, vi):
        k = (prim.astype(np.int64) * 10_000_000 + oi.astype(np.int64)) * 10_000_000 + vi.astype(
            np.int64
        )
        pos = np.searchsorted(skey, k)
        assert (pos < len(skey)).all() and (skey[np.clip(pos, 0, len(skey) - 1)] == k).all(), (
            "head-data row missing in the rebuilt test rows"
        )
        return order[pos]

    real = np.load(c647.WORK / "head_data" / f"real_{month}.npz")
    bk = np.load(c647.WORK / "head_data" / f"bank_{month}.npz")
    rows_real = locate(prim=real["prim"], oi=real["oi"], vi=real["vi"])
    refit = lambda t_rows: (test.origin[t_rows] >= windows[0][1]).astype(np.int8)  # noqa: E731

    if CHECK:
        check(
            test=test,
            mr=mr,
            n_prim=n_prim,
            windows=windows,
            rows_real=rows_real,
            locate=locate,
            clean=clean,
            start_idx=start_idx,
            bk=bk,
        )
        return

    # real rows
    xr = test.x[rows_real].astype(np.float32)
    pr = test.primary[rows_real].astype(np.int16)
    np.savez(
        OUT / f"{month}_real.npz",
        x=xr,
        prim=pr,
        refit=refit(rows_real),
        tab=table_rows(x=xr, prim=pr).astype(np.float16),
    )
    print("real cached", xr.shape, flush=True)
    # bank rows, pass by pass (contaminated lag block)
    xs, ps, rs, ts = [], [], [], []
    for r in range(bank.N_PASSES):
        sel = np.flatnonzero(bk["pass_"] == r)
        t_rows = locate(prim=bk["prim"][sel], oi=bk["oi"][sel], vi=bk["vi"][sel])
        fixed, _ = bank.contaminate(clean=clean, month_start_idx=start_idx, r=r)
        x = test.x[t_rows].astype(np.float32).copy()
        x[:, SL] = e.block_of(
            clean=fixed,
            primary=test.primary[t_rows],
            target_idx=test.target_idx[t_rows],
            origin_idx=test.origin_idx[t_rows],
        )
        assert np.array_equal(x[:, 43:45], test.x[t_rows][:, 43:45])
        p = test.primary[t_rows].astype(np.int16)
        xs.append((sel, x))
        ps.append(p)
        rs.append(refit(t_rows))
        ts.append(table_rows(x=x, prim=p).astype(np.float16))
    n = len(bk["prim"])
    xb = np.empty((n, xs[0][1].shape[1]), np.float32)
    pb = np.empty(n, np.int16)
    rb = np.empty(n, np.int8)
    tb = np.empty((n, 2, ts[0].shape[2]), np.float16)
    for (sel, x), p, rf, t in zip(xs, ps, rs, ts, strict=True):
        xb[sel], pb[sel], rb[sel], tb[sel] = x, p, rf, t
    np.savez(OUT / f"{month}_bank.npz", x=xb, prim=pb, refit=rb, tab=tb)
    print("bank cached", xb.shape, flush=True)


def check(*, test, mr, n_prim, windows, rows_real, locate, clean, start_idx, bk) -> None:
    import dataclasses

    lines = []
    levels = list(e.common.LEVEL_COLUMNS)
    for seed, day, wi in ((0, 0, 0), (0, 14, 1), (2, 0, 0), (2, 14, 1)):
        model = load_model(seed=seed, day=day, n_primaries=n_prim)
        lo, hi = windows[wi]
        frame = e.window_frame(
            model=model, test=test, mr=mr, lo=lo, hi=hi, flag=False, device="cuda"
        )
        stored = pl.read_parquet(STORED / "fits/P5" / f"{month}_d{day}_s{seed}.parquet")
        j = frame.join(
            stored, on=["time_series_id", "power_fcst_init_time", "valid_time"], suffix="_st"
        )
        d = max(float((j[c] - j[c + "_st"]).abs().max()) for c in levels)
        lines.append(
            f"real  seed {seed} d{day}: rebuilt {frame.height}, stored {stored.height}, joined {j.height}, max |q rebuilt - q stored| = {d:.3e} of capacity"
        )
        # injected bank pass 3
        r = 3
        fixed, _ = bank.contaminate(clean=clean, month_start_idx=start_idx, r=r)
        blk = e.block_of(
            clean=fixed,
            primary=test.primary,
            target_idx=test.target_idx,
            origin_idx=test.origin_idx,
        )
        x = test.x.copy()
        x[:, SL] = blk
        fr = e.window_frame(
            model=model,
            test=dataclasses.replace(test, x=x),
            mr=mr,
            lo=lo,
            hi=hi,
            flag=False,
            device="cuda",
        )
        st = pl.read_parquet(STORED / "inject" / f"{month}_r{r:02d}_d{day}_s{seed}.parquet")
        j = fr.join(st, on=["time_series_id", "power_fcst_init_time", "valid_time"], suffix="_st")
        d = max(float((j[c] - j[c + "_st"]).abs().max()) for c in levels)
        lines.append(
            f"bank r{r:02d} seed {seed} d{day}: rebuilt {fr.height}, stored {st.height}, joined {j.height}, max diff = {d:.3e}"
        )
        # cached-input lag encoder vs the network's own
        idx = np.flatnonzero((test.origin >= lo) & (test.origin < hi))[::97][:50_000]
        xt = torch.from_numpy(test.x[idx]).cuda()
        pt = torch.from_numpy(test.primary[idx].astype(np.int64)).cuda()
        rows = torch.from_numpy(
            table_rows(x=test.x[idx], prim=test.primary[idx]).astype(np.float16)
        ).cuda()
        with torch.no_grad():
            own = torch.nn.functional.gelu(
                model.lag_embedding(x=xt, primary=pt, emb=model.embedding(pt))
            )
            mine = encode_cached(model=model, x=xt, prim=pt, rows=rows)
        lines.append(
            f"hl    seed {seed} d{day}: {len(idx)} rows, max |hl cached path - hl network| = {float((own - mine).abs().max()):.3e}"
        )
        del model
    print("\n".join(lines), flush=True)
    (OUT / f"check_{month}.txt").write_text("\n".join(lines) + "\n")


main()

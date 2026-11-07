# ruff: noqa: E501, D103, ANN001, ANN201, E731
"""Experiment 632: load-shift ride-through on experiment 581's lag-only injected bank.

For each fold month and each of the bank's 24 passes, a constant shift is added to the cleaned power
of every Primary over the days before day 14 of the month (experiment 581's `contaminate`). The shift
reaches the history power and the neighbour summaries (Seq) or the lag block (P2-pre) only; the
targets stay clean, so the clean reading is the normal-running-arrangement reference. For each arm
the forecast made with the shifted history ("injected") is compared with the forecast of the same
weights made with clean history ("twin"). Metrics follow experiment 581's `e00581_tails.py`:

- follow ratio at tau = mean over rows of (q_tau injected - q_tau twin) x sign / depth;
- safe side: the share of clean readings on the safe side of the injected forecast, y >= q05 when the
  event raised load and y <= q95 when it lowered load (nominal 0.95);
- rows are forecasts 24 to 120 h ahead; phases are by forecast origin relative to the event.

Usage: python e00632_inject.py ARM TAG MONTH...   with ARM in {seq, P2-pre, K3-pre}
"""

import sys

import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00632_data as d
import e00632_fit as fit
import e00632_fold as fold_mod
import e00632_model as m
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00632_score import KEYS, Q

DEV = "cuda"
HIST = int(__import__("os").environ.get("HIST", "336"))
EXTREME = ["q05", "q50", "q95"]
PHASES = ("before", "d0-3", "d4-end", "after0-7")
LENGTHS_SCORED = (3, 7, 15)


def phases_of(*, o_idx: np.ndarray, end_idx: int, length: np.ndarray) -> np.ndarray:
    onset = end_idx - length * bank.PER_DAY
    d_on = (o_idx - onset) / bank.PER_DAY
    d_end = (o_idx - end_idx) / bank.PER_DAY
    phase = np.full(len(o_idx), "", dtype=object)
    phase[(d_on >= -7) & (d_on < 0)] = "before"
    phase[(d_on >= 0) & (d_on < 4) & (d_end < 0)] = "d0-3"
    phase[(d_on >= 4) & (d_end < 0)] = "d4-end"
    phase[(d_end >= 0) & (d_end < 7)] = "after0-7"
    return phase


def seq_forecast(*, ctx: dict, tag: str, month: str, seeds, clean_inj: np.ndarray | None):
    """Return seed-mean quantiles (rows, 11) in MW of the Seq weights `tag` with history from `clean_inj`."""
    s, cap = ctx["s"][month], ctx["cap"]
    fold = fold_mod.fold_for_seq(s=s)
    z = d.standardised_power(power=ctx["clean"], fold=fold)
    z_in = z if clean_inj is None else d.standardised_power(power=clean_inj, fold=fold)
    nb_mean, nb_share = d.neighbour_summary(z=z_in, of=ctx["of"])
    st = np.load(fold_mod.P2_DIR / "stats" / "B1" / f"{month}.npz")
    b1g, cams = d.b1_scale(
        grid=ctx["b1_grid"], cams=ctx["b1_cams"], mean=st["mean"], std=st["std"], device=DEV
    )
    x = d.standardised_inputs(raw=ctx["raw"], fold=fold, device=DEV, b1_grid=b1g)
    w = d.Windows(
        x=x, z=z, nb_mean=nb_mean, nb_share=nb_share, cams=cams, tau=ctx["tau"],
        n_hist=HIST, n_fut=336, device=DEV, z_in=z_in,
    )  # fmt: skip
    refit = fold_mod.month_start(month=month)
    n_knots = int((refit.year - 2019) * 12 + refit.month - 9) + 1
    origins = fold_mod.test_origins(month=month)
    frames = []
    for seed in seeds:
        net = m.Seq(
            n_primaries=len(s.target_mean), n_knots=n_knots, n_hist=HIST, n_fut=336, arm="seq"
        ).to(DEV)
        net.load_state_dict(
            torch.load(d.ROOT / "models" / tag / f"{month}_s{seed}.pt", map_location=DEV)
        )
        frames.append(
            fit.predict(
                net=net, w=w, s=s, cap=cap, month=month, origins=origins, hist=HIST, fut=336
            )
        )
    return mean_sorted(frames=frames)


def mean_sorted(*, frames: list[pl.DataFrame]) -> pl.DataFrame:
    base = frames[0].sort(KEYS).select(KEYS)
    stack = np.stack([f.sort(KEYS).select(Q).to_numpy() for f in frames]).mean(axis=0)
    stack.sort(axis=1)
    return base.with_columns(*[pl.Series(q, stack[:, i]) for i, q in enumerate(Q)])


def comparator_forecasts(
    *, ctx: dict, arm: str, month: str, passes, seeds
) -> dict[int, pl.DataFrame]:
    """Return {pass: seed-mean forecast frame}; pass -1 is the clean twin. K3-pre ignores the lags."""
    s, cap = ctx["s"][month], ctx["cap"]
    rows = fold_mod.test_rows(raw=ctx["raw"], clean=ctx["clean"], s=s, month=month, device=DEV)
    b1 = fold_mod.b1_columns(month=month, rows=rows)
    fold = fold_mod.fold_for_seq(s=s)
    table = d.standardised_inputs(raw=ctx["raw"], fold=fold, device=DEV)
    ids = d.prep.primaries().sort("primary")["time_series_id"].to_numpy()[rows["primary"]]
    to_t = lambda idx: pl.from_epoch(
        pl.Series(d.prep422.GRID_START_US + idx * d.prep422.STEP_US), time_unit="us"
    ).dt.replace_time_zone("UTC")
    keys = pl.DataFrame(
        {
            "time_series_id": ids,
            "power_fcst_init_time": to_t(rows["origin_idx"]),
            "valid_time": to_t(rows["target_idx"]),
        }
    )
    out = {}
    start_idx = fold_mod.idx_of(t=fold_mod.month_start(month=month))
    for r in [-1, *passes]:
        clean_r = (
            ctx["clean"]
            if r < 0
            else bank.contaminate(clean=ctx["clean"], month_start_idx=start_idx, r=r)[0]
        )
        rr = dict(rows)
        if arm == "P2-pre" and r >= 0:
            rr["blk"] = fold_mod.lg.lag_block(
                clean=clean_r, primary=rows["primary"], target_idx=rows["target_idx"],
                origin_idx=rows["origin_idx"], stats=s.lag_stats, cap_steps=fold_mod.lg.CAP_24H,
            )  # fmt: skip
        frames = []
        for seed in seeds:
            q = fold_mod.run_comparator(
                arm=arm,
                month=month,
                seed=seed,
                rows=rr,
                b1=b1,
                table=table,
                s=s,
                capacity=cap,
                device=DEV,
            )
            frames.append(keys.with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(Q)]))
        out[r] = mean_sorted(frames=frames)
    return out


def event_metrics(
    *,
    inj: pl.DataFrame,
    twin: pl.DataFrame,
    month: str,
    r: int,
    clean: np.ndarray,
    prim_of: dict,
    arm: str,
) -> pl.DataFrame:
    start_idx = fold_mod.idx_of(t=fold_mod.month_start(month=month))
    end_idx = start_idx + bank.END_DAY * bank.PER_DAY
    j = inj.join(twin, on=KEYS, suffix="_t")
    prim = np.array([prim_of[t] for t in j["time_series_id"].to_list()])
    to_idx = lambda col: (
        (j[col].dt.epoch("us").to_numpy() - d.prep422.GRID_START_US) // d.prep422.STEP_US
    ).astype(np.int64)
    v_idx, o_idx = to_idx("valid_time"), to_idx("power_fcst_init_time")
    ev = np.array([bank.event(primary=p, r=r) for p in range(clean.shape[0])])
    depth, length, sign = ev[prim, 0], ev[prim, 1].astype(int), ev[prim, 2]
    phase = phases_of(o_idx=o_idx, end_idx=end_idx, length=length)
    lead = (v_idx - o_idx) * 0.5
    y = clean[prim, v_idx]
    cap = fold_mod.capacity_of_primary()
    df = j.with_columns(
        primary=pl.Series(prim), phase=pl.Series(phase), y=pl.Series(y), lead=pl.Series(lead),
        sgn=pl.Series(sign.astype(np.float64)), depth=pl.Series(np.abs(depth)), length_days=pl.Series(length),
    ).filter(
        (pl.col("phase") != "") & pl.col("y").is_finite() & (pl.col("lead") >= 24) & (pl.col("lead") < 120)
        & pl.col("length_days").is_in(list(LENGTHS_SCORED))
    )  # fmt: skip
    # injected quantiles are in MW; the clean power y is a share of capacity
    capr = pl.Series(cap[df["primary"].to_numpy()])
    f = lambda c: (pl.col(c) - pl.col(f"{c}_t")) / capr * pl.col("sgn") / pl.col("depth")
    df = df.with_columns(
        y_mw=pl.col("y") * capr,
        follow_q50=f("q50"),
        follow_tail=pl.when(pl.col("sgn") > 0).then(f("q05")).otherwise(f("q95")),
    ).with_columns(
        safe=pl.when(pl.col("sgn") > 0)
        .then(pl.col("y_mw") >= pl.col("q05"))
        .otherwise(pl.col("y_mw") <= pl.col("q95"))
        .cast(pl.Float64),
        safe_twin=pl.when(pl.col("sgn") > 0)
        .then(pl.col("y_mw") >= pl.col("q05_t"))
        .otherwise(pl.col("y_mw") <= pl.col("q95_t"))
        .cast(pl.Float64),
    )
    return (
        df.group_by("primary", "phase", "length_days")
        .agg(pl.len().alias("n"), pl.first("depth"), pl.first("sgn"), pl.col("follow_q50", "follow_tail", "safe", "safe_twin").mean())
        .with_columns(month=pl.lit(month), pass_r=pl.lit(r), arm=pl.lit(arm))
    )  # fmt: skip


def main() -> None:
    arm, tag, months = sys.argv[1], sys.argv[2], sys.argv[3:]
    import os

    seeds = tuple(int(x) for x in os.environ.get("SEEDS", "0 1 2").split())
    torch.cuda.set_per_process_memory_fraction(8 / 48)
    clean = d.clean_power()
    b1g, b1c = d.dense_b1()
    ctx = {
        "raw": d.dense_raw(), "clean": clean, "b1_grid": b1g, "b1_cams": b1c, "tau": d.tau_grid(),
        "of": d.neighbour_lists(), "cap": fold_mod.capacity_of_primary(),
        "s": {mo: fold_mod.fold_stats(month=mo, clean=clean) for mo in months},
    }  # fmt: skip
    prims = d.prep.primaries().sort("primary")
    prim_of = dict(zip(prims["time_series_id"].to_list(), prims["primary"].to_list(), strict=True))
    out_dir = d.ROOT / "inject"
    out_dir.mkdir(exist_ok=True)
    passes = range(bank.N_PASSES)
    for month in months:
        target = out_dir / f"{tag}_{month}.parquet"
        if target.exists():
            continue
        start_idx = fold_mod.idx_of(t=fold_mod.month_start(month=month))
        if arm == "seq":
            twin = seq_forecast(ctx=ctx, tag=tag, month=month, seeds=seeds, clean_inj=None)
            inj = {
                r: seq_forecast(
                    ctx=ctx,
                    tag=tag,
                    month=month,
                    seeds=seeds,
                    clean_inj=bank.contaminate(clean=clean, month_start_idx=start_idx, r=r)[0],
                )
                for r in passes
            }
        else:
            allf = comparator_forecasts(ctx=ctx, arm=arm, month=month, passes=passes, seeds=seeds)
            twin, inj = allf[-1], {r: allf[r] for r in passes}
        parts = [
            event_metrics(
                inj=inj[r], twin=twin, month=month, r=r, clean=clean, prim_of=prim_of, arm=arm
            )
            for r in passes
        ]
        pl.concat(parts).write_parquet(target)
        print(f"{tag} {month}: {sum(p.height for p in parts)} event rows", flush=True)


if __name__ == "__main__":
    main()

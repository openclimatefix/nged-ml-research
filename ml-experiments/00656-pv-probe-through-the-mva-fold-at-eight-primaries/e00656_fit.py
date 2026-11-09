# ruff: noqa: E501, ANN001, PLR0915
"""Experiment 656: pretrain experiment 554's network M0 on ERA5 and CAMS rows in an MVA world.

The network, rows, loss, learning rate, early stopping and seeds are experiment 621's `train` (which
is experiment 554's `pretrain`, stage `pre`). The only change is the target of the covered Primaries
(`e00656_common.WorldBuilder`). Rows of the eleven covered Primaries with a gap in any of the farms
A, B, C, or F are dropped from every condition, the twins included.

Usage: python e00656_fit.py --cutoff 2026-06 --conditions pf_twin pf_fleet_l1_r0 --seeds 0 1 2
[--device cpu --threads 4]
"""

import argparse
import json
import time
from datetime import UTC, datetime

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00621_common as k621  # ty: ignore[unresolved-import]
import e00621_fit as f621  # ty: ignore[unresolved-import]
import e00656_common as k
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

FRAME_HOURS = (4, 20)
"""UTC valid-time hours kept in the probe frame (the probe keeps rows with the sun 5.6 degrees up or more)."""
E649 = (
    k.ML_EXPERIMENTS_DATA_DIR
    / "00649-pv-capacity-table-season-year-from-stored-probes"
    / "work"
    / "capacity_table.parquet"
)
PROBED: tuple[str, ...] = (*k.SIX, "D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")


def night_and_rho(*, pre, rows, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each Primary's median summer-night reading (points) and the daytime-to-night demand ratio.

    The ratio `rho[month, half-hour]` is the median, over the six signed-MW Primaries and their days, of
    gross demand (signed power plus experiment 649's calibrated unmetered PV times the standard unit
    curve) over the Primary's summer-night reading.
    """
    t_us = k621.time_us(target_idx=pre.target_idx)
    months = pl.Series(t_us).cast(pl.Datetime("us")).dt.month().to_numpy()
    hh = ((t_us % k621.DAY_US) // 1_800_000_000).astype(int)
    night = keep & (pre.x[:, 14] <= 0.0) & (months >= 4) & (months <= 9)
    n_prim = len(rows.target_mean)
    night_pts = np.array(
        [float(np.median(pre.y[night & (pre.primary == p)])) for p in range(n_prim)]
    )
    table = pl.read_parquet(E649).filter(pl.col("probe_year") == 2025)
    cal = table.group_by("site").agg(c=pl.col("cal_p50").median())
    cal_mw = dict(zip(cal["site"].to_list(), cal["c"].to_list(), strict=True))
    cap = k621.effective_capacity_mw()
    frames = []
    for site, p in k.index_of(sites=k.SIX).items():
        sel = keep & (pre.primary == p) & (pre.x[:, 14] > 0.1)
        ghi, temp = k621.raw_weather(x=pre.x[sel], standardiser=rows.standardiser)
        u = k621.unit_curve(ghi=ghi, x=pre.x[sel], times_us=t_us[sel], temp=temp)
        gross = (pre.y[sel] * cap[p] + cal_mw[site] * u) / (night_pts[p] * cap[p])
        frames.append(pl.DataFrame({"m": months[sel], "hh": hh[sel], "r": gross}))
    med = pl.concat(frames).group_by(["m", "hh"]).agg(r=pl.col("r").median())
    rho = np.ones((13, 48))
    rho[med["m"].to_numpy(), med["hh"].to_numpy()] = med["r"].to_numpy()
    return night_pts, rho


def write_frame(*, cutoff: str, pre, rows, keep: np.ndarray) -> None:
    """Save the probe frame: one row per (Primary, target time) of the 14 probed Primaries, daylight only."""
    night_pts, rho = night_and_rho(pre=pre, rows=rows, keep=keep)
    idx14 = k.index_of(sites=PROBED)
    t_us = k621.time_us(target_idx=pre.target_idx)
    hour = (t_us // 3_600_000_000) % 24
    sel = (
        keep
        & np.isin(pre.primary, list(idx14.values()))
        & (hour >= FRAME_HOURS[0])
        & (hour <= FRAME_HOURS[1])
    )
    key = pre.primary[sel].astype(np.int64) * 10_000_000 + pre.target_idx[sel]
    _, first = np.unique(key, return_index=True)
    idx = np.nonzero(sel)[0][first]
    out = k.frame_path(cutoff=cutoff)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(
        tmp, x=pre.x[idx, :35], y=pre.y[idx].astype(np.float32), primary=pre.primary[idx],
        target_idx=pre.target_idx[idx], is_val=pre.is_val[idx], target_mean=rows.target_mean,
        target_std=rows.target_std, std_mean=rows.standardiser.mean[:7], std_std=rows.standardiser.std[:7],
        night_pts=night_pts, rho=rho,
    )  # fmt: skip
    tmp.rename(out)
    print(f"frame {cutoff}: {len(idx):,} rows", flush=True)


def main() -> None:
    """Pretrain the requested conditions and seeds for one cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--conditions", nargs="+", default=list(k.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frame-only", action="store_true")
    parser.add_argument("--max-fits", type=int, default=10_000)
    args = parser.parse_args()
    device = args.device
    torch.set_num_threads(args.threads)
    clean = c.clean_stack()
    refit_t = c.month_start(month=args.cutoff)
    rows = data.build_fold(refit=refit_t, clean=clean)
    pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
    latest = datetime.fromtimestamp(
        (pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC
    )
    assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
    world = k.WorldBuilder(
        primary=pre.primary,
        target_idx=pre.target_idx,
        x=pre.x,
        y=pre.y,
        standardiser=rows.standardiser,
        clean=clean,
    )
    keep = ~world.gap
    print(
        f"{args.cutoff}: {len(pre.y):,} rows, {int(world.gap.sum()):,} dropped for farm gaps; latest valid {latest.isoformat()}",
        flush=True,
    )
    print(
        "median sunny-midday (points):",
        {int(p): round(v, 3) for p, v in world.median_midday.items()},
        flush=True,
    )
    meta = k.WORK / "world" / f"{args.cutoff}.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(
        json.dumps(
            {
                "median_midday": {int(p): v for p, v in world.median_midday.items()},
                "n_dropped": int(world.gap.sum()),
            }
        )
    )
    write_frame(cutoff=args.cutoff, pre=pre, rows=rows, keep=keep)
    if args.frame_only:
        return
    tgt_mean, tgt_std = rows.target_mean[pre.primary], rows.target_std[pre.primary]
    x_d = torch.from_numpy(pre.x[keep]).to(device)
    p_d = torch.from_numpy(pre.primary[keep]).to(device)
    v_d = torch.from_numpy(pre.is_val[keep]).to(device)
    w_d = torch.ones(int(keep.sum()), device=device)
    fits = 0
    for name in args.conditions:
        cond = k.parse(name=name)
        todo = [
            s
            for s in args.seeds
            if not k.weights_path(cutoff=args.cutoff, condition=name, seed=s).exists()
        ]
        if not todo:
            continue
        y = world.target(cond=cond)
        z_d = torch.from_numpy(((y - tgt_mean) / tgt_std)[keep].astype(np.float32)).to(device)
        for seed in todo:
            if fits >= args.max_fits:
                return
            target = k.weights_path(cutoff=args.cutoff, condition=name, seed=seed)
            started = time.monotonic()
            torch.manual_seed(seed)
            model = m.ModularMdn(n_primaries=len(rows.target_mean), stage="pre").to(device)
            epochs, best = f621.train(
                model=model,
                x=x_d,
                z=z_d,
                w=w_d,
                p=p_d,
                is_val=v_d,
                seed=seed,
                lr=c.PRETRAIN_LR,
                device=device,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.cpu().state_dict(), tmp)
            tmp.rename(target)
            seconds = time.monotonic() - started
            fits += 1
            print(
                f"  {args.cutoff} {name} s{seed}: {epochs} epochs, val NLL {best:.4f}, {seconds:.0f} s ({device})",
                flush=True,
            )
            log = k.WORK / "logs" / args.cutoff / f"{name}_s{seed}.json"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                json.dumps(
                    {"epochs": epochs, "val_nll": best, "seconds": seconds, "device": device}
                )
            )
        del z_d
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

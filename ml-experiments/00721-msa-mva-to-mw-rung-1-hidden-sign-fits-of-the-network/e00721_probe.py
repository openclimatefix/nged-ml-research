# ruff: noqa: E501, ANN001, PLR0915
"""Experiment 721: the capacity probe of the hidden pair, per world and per fold (8 runs).

Experiment 662's probe estimates the AC capacity of the PV behind D5 and D10 from their folded
readings. For each of the two worlds (`W2`, `W0`) and each of the four folds, this runner:

1. builds experiment 662's pretraining rows for the fold (data before the fold's start only);
2. replaces the target of D5 and D10 by the world's folded reading (`W2`: the folded reading after
   the farm injection; `W0`: the folded reading with no injection), so that no signed reading of
   either Primary reaches the network's weights or the probe;
3. pretrains experiment 554's `ModularMdn(stage="pre")` on those rows (a GPU job of about 30
   seconds; the 8 jobs together take about 4 minutes);
4. runs the donor-day probe with the fold model `FA` on D5 and D10 (CPU) and writes
   `probes/<world>_<fold>.json` holding `{"D5": mw, "D10": mw}`.

The probe's reactive-power floor ratio `rho` is computed from the four visible signed Primaries only
(`core.VISIBLE_SIGNED`), because experiment 662's own `night_and_rho` pools the six signed Primaries,
which include D5 and D10. Running a world whose probe file exists is skipped, so a stopped run
resumes. The runner sequences before the `W2` and `W0` fits (run it first in the stage that needs the
capacities) and its GPU job obeys the slot rules of `run.sh`.

Usage:
    run.sh --go e00721_cli.py probe --device cuda
"""

import time
from pathlib import Path

import e00721_core as core
import e00721_fit as fit
import e00721_real as real
import numpy as np
import polars as pl

FRAME_HOURS = (4, 20)
"""UTC valid-time hours kept in the probe frame (experiment 662's)."""


def visible_night_and_rho(*, pre, y: np.ndarray, rows, keep: np.ndarray, k, k621, e649: Path):  # noqa: ANN201
    """Return each Primary's median summer-night reading and the daytime-to-night ratio `rho`.

    This is experiment 662's `night_and_rho` with the six signed-MW Primaries replaced by the four
    visible ones, so that D5 and D10 do not enter `rho`. `y` is the target after the replacement, so the
    night reading of D5 and D10 is the folded reading.
    """
    t_us = k621.time_us(target_idx=pre.target_idx)
    months = pl.Series(t_us).cast(pl.Datetime("us")).dt.month().to_numpy()
    hh = ((t_us % k621.DAY_US) // 1_800_000_000).astype(int)
    night = keep & (pre.x[:, 14] <= 0.0) & (months >= 4) & (months <= 9)
    n_prim = len(rows.target_mean)
    night_pts = np.array([float(np.median(y[night & (pre.primary == p)])) for p in range(n_prim)])
    table = pl.read_parquet(e649).filter(pl.col("probe_year") == 2025)
    cal = table.group_by("site").agg(c=pl.col("cal_p50").median())
    cal_mw = dict(zip(cal["site"].to_list(), cal["c"].to_list(), strict=True))
    cap = k621.effective_capacity_mw()
    frames = []
    for site, p in k.index_of(sites=tuple(core.VISIBLE_SIGNED)).items():
        sel = keep & (pre.primary == p) & (pre.x[:, 14] > 0.1)
        ghi, temp = k621.raw_weather(x=pre.x[sel], standardiser=rows.standardiser)
        u = k621.unit_curve(ghi=ghi, x=pre.x[sel], times_us=t_us[sel], temp=temp)
        gross = (y[sel] * cap[p] + cal_mw[site] * u) / (night_pts[p] * cap[p])
        frames.append(pl.DataFrame({"m": months[sel], "hh": hh[sel], "r": gross}))
    med = pl.concat(frames).group_by(["m", "hh"]).agg(r=pl.col("r").median())
    rho = np.ones((13, 48))
    rho[med["m"].to_numpy(), med["hh"].to_numpy()] = med["r"].to_numpy()
    return night_pts, rho


def probe_path(*, data_dir: Path, world: str, fold: str) -> Path:
    """Return the probe file of a world and fold."""
    return real.ProbeProvider(root=fit.data_root(data_dir=data_dir)).path(world=world, fold=fold)


def hidden_standardisation(
    *, y: np.ndarray, keep: np.ndarray, primary: np.ndarray, hidden: tuple[int, ...],
    target_mean: np.ndarray, target_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Return the per-Primary mean and standard deviation, with the hidden Primaries' own series.

    Each hidden Primary is standardised with the mean and standard deviation of the folded reading that
    the probe estimates, over the rows that are kept. The other Primaries keep the fold's values.
    """
    mean, std = target_mean.copy(), target_std.copy()
    for p in hidden:
        own = (primary == p) & keep
        mean[p], std[p] = float(y[own].mean()), float(y[own].std())
    return mean, std


def probe_capacity(*, fold: str, device: str) -> dict[str, float]:
    """Pretrain the fold's network on the un-injected folded readings and return `{"D5": mw, "D10": mw}`.

    The probe estimates the host's own PV, so it reads the `W0` reading (no injection) in both worlds.
    The `W2` build then adds the oracle increment once, in `share_scale_with_injection`; probing the
    injected reading would count the farm twice.
    """
    import e00428_data as data  # ty: ignore[unresolved-import]
    import e00448_run as r448  # ty: ignore[unresolved-import]
    import e00510_common as c  # ty: ignore[unresolved-import]
    import e00554_model as m  # ty: ignore[unresolved-import]
    import e00621_common as k621  # ty: ignore[unresolved-import]
    import e00621_fit as f621  # ty: ignore[unresolved-import]
    import e00662_common as k  # ty: ignore[unresolved-import]
    import e00662_fit as fit662  # ty: ignore[unresolved-import]
    import e00662_probe as probe  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    gi = real.load_grid_inputs(fold=fold, seed=0, phase="train", with_farms=True)
    w = real.build_fold_world_arrays(
        panel=gi.panel, farm_for_site=gi.farms, world="W0", cutoff_idx=None
    )
    sites = list(gi.panel.sites)
    hidden = {s: sites.index(s) for s in core.HIDDEN_PAIR}
    clean = c.clean_stack()
    refit = c.month_start(month=fold)
    rows = data.build_fold(refit=refit, clean=clean)
    pre = r448.pretrain_rows(rows=rows, refit=refit, clean=clean)
    y = pre.y.astype(np.float64).copy()
    keep = np.ones(len(y), dtype=bool)
    n87 = len(gi.panel.stamp_us)
    for site, p in hidden.items():
        sel = pre.primary == p
        idx = pre.target_idx[sel]
        assert bool((idx < n87).all()), "a pretraining row lies beyond experiment 87's grid"
        folded = w.fold_view[p, idx]
        y[sel] = folded
        keep[sel] = np.isfinite(folded)
        del site
    y = np.where(np.isfinite(y), y, 0.0)
    # standardise each Primary with the mean and standard deviation of the series it is asked to estimate
    t_mean, t_std = hidden_standardisation(
        y=y, keep=keep, primary=pre.primary, hidden=tuple(hidden.values()),
        target_mean=rows.target_mean, target_std=rows.target_std,
    )  # fmt: skip
    night_pts, rho = visible_night_and_rho(
        pre=pre, y=y, rows=rows, keep=keep, k=k, k621=k621, e649=fit662.E649
    )
    t_us = k621.time_us(target_idx=pre.target_idx)
    hour = (t_us // 3_600_000_000) % 24
    sel = (
        keep
        & np.isin(pre.primary, list(hidden.values()))
        & (hour >= FRAME_HOURS[0])
        & (hour <= FRAME_HOURS[1])
    )
    key = pre.primary[sel].astype(np.int64) * 10_000_000 + pre.target_idx[sel]
    _, first = np.unique(key, return_index=True)
    ix = np.nonzero(sel)[0][first]
    frame = {
        "x": pre.x[ix, :35], "y": y[ix].astype(np.float32), "primary": pre.primary[ix], "target_idx": pre.target_idx[ix],
        "is_val": pre.is_val[ix], "target_mean": t_mean, "target_std": t_std,
        "std_mean": rows.standardiser.mean[:7], "std_std": rows.standardiser.std[:7], "night_pts": night_pts, "rho": rho,
    }  # fmt: skip
    mean, std = t_mean[pre.primary], t_std[pre.primary]
    z = torch.from_numpy(((y - mean) / std)[keep].astype(np.float32)).to(device)
    x_d = torch.from_numpy(pre.x[keep]).to(device)
    p_d = torch.from_numpy(pre.primary[keep]).to(device)
    v_d = torch.from_numpy(pre.is_val[keep]).to(device)
    torch.manual_seed(0)
    model = m.ModularMdn(n_primaries=len(rows.target_mean), stage="pre").to(device)
    f621.train(
        model=model,
        x=x_d,
        z=z,
        w=torch.ones(int(keep.sum()), device=device),
        p=p_d,
        is_val=v_d,
        seed=0,
        lr=c.PRETRAIN_LR,
        device=device,
    )
    model = model.cpu().eval()
    cap = k621.effective_capacity_mw()
    out = {}
    for site, p in hidden.items():
        pp = probe.PrimaryProbe(frame=frame, primary=p, site=site, cap=float(cap[p]))
        pp.set_arm(arm="floor")
        reading = pp.forward(model=model, frame=frame)
        res = probe.probe_network(pp=pp, y=reading, boot=False, rng=np.random.default_rng(1000 + p))
        out[site] = float(res["FA"])
    return out


TIMING_FILE = "first_probe_timing.json"
"""File, in the probe folder, that records the first probe's wall time. The runner stops after it."""


def run_probes(
    *, data_dir: Path, device: str, folds: tuple[str, ...] = core.FOLDS
) -> dict[str, float]:
    """Run the probes of `folds` and return the seconds per fold.

    A process serves one fold month, so the CLI passes one fold. The probe reads the un-injected
    reading, so one pretraining per fold gives the file of both worlds. A fold whose two files exist is
    skipped. The first probe run is timed, writes `TIMING_FILE`, and stops the runner ("the first probe
    run is timed and the manager told before the second"); a later launch continues.
    """
    import e00721_paths
    import torch  # ty: ignore[unresolved-import]

    e00721_paths.pin_threads()
    if device.startswith("cuda"):
        fit.limit_gpu_memory(torch=torch)  # 6 GB per process, as in every fit
    timing = probe_path(data_dir=data_dir, world="W2", fold=core.FOLDS[0]).parent / TIMING_FILE
    seconds: dict[str, float] = {}
    for fold in folds:
        paths = [probe_path(data_dir=data_dir, world=w, fold=fold) for w in ("W2", "W0")]
        if all(p.exists() for p in paths):
            continue
        started = time.monotonic()
        capacity = probe_capacity(fold=fold, device=device)
        for path in paths:
            fit.atomic_write_json(path=path, body=capacity)
        seconds[fold] = time.monotonic() - started
        print(f"probe {fold}: {seconds[fold]:.0f} s (both worlds)", flush=True)
        if not timing.exists():
            fit.atomic_write_json(path=timing, body={"fold": fold, "seconds": seconds[fold]})
            print(
                "first probe timed: tell the manager before the next probe, then launch again",
                flush=True,
            )
            break
    return seconds

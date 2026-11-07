# ruff: noqa: E501, ANN001
"""Experiment 634: the donor-day regression probe of a pretrained lag-free network (CPU).

The probe holds every datetime input, and the Primary, at one probe date and half-hour, and feeds
the network the weather of many donor half-hours (the same half-hour of the day, within 21 days of
the probe date's day of the year, any other date in the pretraining rows). Donor shortwave is
rescaled by the donor's clear-sky index to the probe date's sun position. For each Primary the
probed net demand in megawatts is regressed, with an intercept per probe cell, on the physics unit
curve `u` of the donor weather at the probe date's sun position, plus the donor's temperature
(model `plain`) and, in model `cloud_tpl`, a flat cloud template `1 - clear-sky index`. The
effective AC capacity is minus the slope on `u`. A day-block bootstrap (Poisson weights per donor
date) gives the interval.

Usage: python e00634_probe.py --cutoff 2026-06 --conditions control farm_mw4_r0 --seeds 0 1 2
"""

import argparse
from datetime import UTC, date, datetime, timedelta

import e00397_models as base  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00634_common as k
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

PROBE_DATES: tuple[date, ...] = tuple(date(2025, 4, 9) + timedelta(days=14 * i) for i in range(12))
"""Twelve Wednesdays from 2025-04-09 to 2025-09-10; the dates with no base row at a cutoff are skipped, so 2025-07 keeps only the first 6."""
MINUTES: tuple[int, ...] = tuple(range(10 * 60, 15 * 60 + 1, 30))
"""Half-hour ends from 10:00 to 15:00 UTC."""
WINDOW_DAYS = 21
N_BOOT = 200
MIN_COS_ZENITH = 0.15
KT_MAX = 1.3


def load_frame(*, cutoff: str) -> dict[str, np.ndarray]:
    """Load a cutoff's probe frame."""
    with np.load(k.frame_path(cutoff=cutoff)) as z:
        return {name: z[name] for name in z.files}


def cells_and_donors(*, frame: dict[str, np.ndarray], primary: int) -> dict[str, np.ndarray]:
    """Return the probe inputs of one Primary: per-row base datetime x, donor weather, cell, donor date."""
    sel = frame["primary"] == primary
    x, idx = frame["x"][sel], frame["target_idx"][sel]
    t_us = k.time_us(target_idx=idx)
    days = t_us // k.DAY_US
    minute = ((t_us % k.DAY_US) // 60_000_000).astype(np.int64)
    doy = pl.Series(t_us).cast(pl.Datetime("us")).dt.ordinal_day().to_numpy().astype(np.int64)
    order = np.argsort(idx)
    x, idx, days, minute, doy, t_us = (a[order] for a in (x, idx, days, minute, doy, t_us))
    rows: list[dict[str, np.ndarray]] = []
    cell = 0
    for d in PROBE_DATES:
        day = int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()) // 86_400
        d_doy = d.timetuple().tm_yday
        for mm in MINUTES:
            base_pos = np.nonzero((days == day) & (minute == mm))[0]
            if len(base_pos) != 1:
                continue
            b = base_pos[0]
            if x[b, 14] < MIN_COS_ZENITH or x[b, 32] > 0:  # cos_zenith, bank_holiday
                continue
            near = np.abs(((doy - d_doy + 182) % 365) - 182) <= WINDOW_DAYS
            donors = np.nonzero(
                near & (minute == mm) & (days != day) & (x[:, 14] >= MIN_COS_ZENITH)
            )[0]
            rows.append(
                {
                    "base": np.full(len(donors), b), "donor": donors, "cell": np.full(len(donors), cell),
                }
            )  # fmt: skip
            cell += 1
    cat = {name: np.concatenate([r[name] for r in rows]) for name in rows[0]}
    return {"x": x, "t_us": t_us, "days": days, **cat}


def forward_mw(
    *,
    model: m.ModularMdn,
    frame: dict[str, np.ndarray],
    primary: int,
    probe: dict[str, np.ndarray],
    cap: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the probed net demand (MW) of every probe row, and the regressors."""
    x, b, d = probe["x"], probe["base"], probe["donor"]
    s_mean, s_std = frame["std_mean"], frame["std_std"]
    ghi_d = np.clip(x[d, 0].astype(np.float64) * s_std[0] + s_mean[0], 0.0, None)
    temp_d = x[d, 2].astype(np.float64) * s_std[2] + s_mean[2]
    clear_d = k.haurwitz_w_m2(cos_zenith=x[d, 14].astype(np.float64))
    clear_b = k.haurwitz_w_m2(cos_zenith=x[b, 14].astype(np.float64))
    kt = np.clip(ghi_d / np.maximum(clear_d, 1.0), 0.0, KT_MAX)
    ghi_in = kt * clear_b
    inp = x[d, :35].copy()
    inp[:, 14:35] = x[b, 14:35]
    inp[:, 0] = np.clip((ghi_in - s_mean[0]) / s_std[0], -5.0, 5.0).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(inp)
        xt = torch.cat(
            [xt, torch.zeros(len(xt), 2)], dim=1
        )  # the lead columns, unused at stage `pre`
        pt = torch.full((len(xt),), primary, dtype=torch.long)
        mean_z = []
        for i in range(0, len(xt), 65_536):
            out = model(xt[i : i + 65_536], pt[i : i + 65_536]).double()
            log_w, mean, _ = base.mixture_parts(out=out, components=m.COMPONENTS)
            mean_z.append((torch.softmax(log_w, dim=-1) * mean).sum(-1).numpy())
    y_points = np.concatenate(mean_z) * frame["target_std"][primary] + frame["target_mean"][primary]
    unit = k.unit_curve(ghi=ghi_in, x=x[b], times_us=probe["t_us"][b], temp=temp_d)
    return y_points * cap, {"u": unit, "temp": temp_d, "cloud": np.clip(1.0 - kt, 0.0, 1.0)}


def fit(*, y: np.ndarray, cell: np.ndarray, regressors: list[np.ndarray], w: np.ndarray) -> float:
    """Return minus the slope on the first regressor, after weighted demeaning within each cell."""
    wsum = np.bincount(cell, weights=w)
    wsum = np.where(wsum > 0, wsum, 1.0)

    def centred(v: np.ndarray) -> np.ndarray:
        return v - (np.bincount(cell, weights=w * v) / wsum)[cell]

    xc = np.stack([centred(v) for v in regressors], axis=1)
    yc = centred(y)
    gram = (xc * w[:, None]).T @ xc
    beta = np.linalg.solve(gram + 1e-9 * np.eye(len(gram)), (xc * w[:, None]).T @ yc)
    return float(-beta[0])


def probe_network(
    *, model: m.ModularMdn, frame, primary: int, probe, cap: float, rng: np.random.Generator
) -> dict:
    """Return the point estimates and the bootstrap draws of the effective AC capacity (MW)."""
    y, reg = forward_mw(model=model, frame=frame, primary=primary, probe=probe, cap=cap)
    cell = probe["cell"]
    donor_day = probe["days"][probe["donor"]]
    _, date_id = np.unique(donor_day, return_inverse=True)
    n_dates = int(date_id.max()) + 1
    out: dict = {}
    for name, regs in (
        ("plain", [reg["u"], reg["temp"]]),
        ("cloud_tpl", [reg["u"], reg["temp"], reg["cloud"]]),
    ):
        one = np.ones(len(y))
        point = fit(y=y, cell=cell, regressors=regs, w=one)
        boot = np.empty(N_BOOT)
        for i in range(N_BOOT):
            w = rng.poisson(1.0, size=n_dates).astype(np.float64)[date_id]
            boot[i] = fit(y=y, cell=cell, regressors=regs, w=w)
        out[name] = (point, boot)
    out["n_rows"] = len(y)
    out["mean_swing_mw"] = float(np.ptp(np.array([y[cell == c].mean() for c in np.unique(cell)])))
    return out


def main() -> None:
    """Probe every requested network and write one parquet of results per cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--conditions", nargs="+", default=list(k.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    frame = load_frame(cutoff=args.cutoff)
    six = k.six_indices()
    cap = k.effective_capacity_mw()
    probes = {site: cells_and_donors(frame=frame, primary=i) for site, i in six.items()}
    for site, pr in probes.items():
        print(site, "probe rows", len(pr["base"]), "cells", len(np.unique(pr["cell"])), flush=True)
    for name in args.conditions:
        out = k.WORK / "probes" / args.cutoff / f"{name}.parquet"
        if out.exists():
            continue
        records = []
        for seed in args.seeds:
            path = k.weights_path(cutoff=args.cutoff, condition=name, seed=seed)
            assert path.exists(), path
            model = m.ModularMdn(n_primaries=16, stage="pre")
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            for site, i in six.items():
                rng = np.random.default_rng(1000 * seed + i)
                res = probe_network(
                    model=model,
                    frame=frame,
                    primary=i,
                    probe=probes[site],
                    cap=float(cap[i]),
                    rng=rng,
                )
                for kind in ("plain", "cloud_tpl"):
                    point, boot = res[kind]
                    records.append(
                        {
                            "cutoff": args.cutoff, "condition": name, "seed": seed, "site": site, "model": kind,
                            "capacity_mw": point, "boot": boot.tolist(), "n_rows": res["n_rows"],
                            "swing_mw": res["mean_swing_mw"],
                        }
                    )  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(records).write_parquet(out)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

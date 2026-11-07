# ruff: noqa: E501, ANN001, ANN201
"""Experiment 635: the donor-day probe of experiment 621 on the metered-input networks (CPU).

The probe is experiment 621's (`e00621_probe`). The donor's metered column is swapped in with the
donor's weather. It is rescaled to the probe date's sun position as the shortwave is: the donor's
metered output over the clear-sky shape at the donor's sun (the Haurwitz curve) is capped at 1.5
and multiplied by the clear-sky shape at the probe date's sun.

Usage: python e00635_probe.py --arm b --conditions control_r0 farm_mw4_r0 --seeds 0 1 2
"""

import argparse

import e00397_models as base  # ty: ignore[unresolved-import]
import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00409_prep as p9  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00621_common as k  # ty: ignore[unresolved-import]
import e00621_probe as p621  # ty: ignore[unresolved-import]
import e00635_common as k5
import e00635_model as mm
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

HAURWITZ_PEAK = 1098.0
Q_MAX = 1.5


def metered_rows(*, probe, farm: str, shuffled: bool) -> np.ndarray:
    """Return the scaled metered column (NaN at gaps filled with 0) for every frame row of one Primary."""
    v = p9.series_at(
        times_us=probe["t_us"], series=p5.farm_norm_series(farm=farm),
        shuffle_salt=k5.SHUFFLE_SALT if shuffled else None,
    )  # fmt: skip
    return np.nan_to_num(v, nan=0.0) * k5.METERED_SCALE


def forward_mw(*, model, frame, primary: int, probe, cap: float, metered: np.ndarray):
    """Return the probed net demand (MW) and the regressors; as experiment 621 plus the metered column."""
    x, b, d = probe["x"], probe["base"], probe["donor"]
    s_mean, s_std = frame["std_mean"], frame["std_std"]
    ghi_d = np.clip(x[d, 0].astype(np.float64) * s_std[0] + s_mean[0], 0.0, None)
    temp_d = x[d, 2].astype(np.float64) * s_std[2] + s_mean[2]
    clear_d = k.haurwitz_w_m2(cos_zenith=x[d, 14].astype(np.float64))
    clear_b = k.haurwitz_w_m2(cos_zenith=x[b, 14].astype(np.float64))
    kt = np.clip(ghi_d / np.maximum(clear_d, 1.0), 0.0, p621.KT_MAX)
    ghi_in = kt * clear_b
    q = np.clip(
        (metered[d] / k5.METERED_SCALE) / np.maximum(clear_d / HAURWITZ_PEAK, 0.05), 0.0, Q_MAX
    )
    met_in = q * (clear_b / HAURWITZ_PEAK) * k5.METERED_SCALE
    inp = x[d, :35].copy()
    inp[:, 14:35] = x[b, 14:35]
    inp[:, 0] = np.clip((ghi_in - s_mean[0]) / s_std[0], -5.0, 5.0).astype(np.float32)
    with torch.no_grad():
        xt = torch.cat([torch.from_numpy(inp), torch.zeros(len(inp), 2)], dim=1)
        xt[:, mm.METERED_COL] = torch.from_numpy(met_in.astype(np.float32))
        pt = torch.full((len(xt),), primary, dtype=torch.long)
        mean_z = []
        for i in range(0, len(xt), 65_536):
            out = model(xt[i : i + 65_536], pt[i : i + 65_536]).double()
            log_w, mean, _ = base.mixture_parts(out=out, components=m.COMPONENTS)
            mean_z.append((torch.softmax(log_w, dim=-1) * mean).sum(-1).numpy())
    y_points = np.concatenate(mean_z) * frame["target_std"][primary] + frame["target_mean"][primary]
    unit = k.unit_curve(ghi=ghi_in, x=x[b], times_us=probe["t_us"][b], temp=temp_d)
    return y_points * cap, {"u": unit, "temp": temp_d}


def main() -> None:
    """Probe every requested network and write one parquet of results per condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=k5.ARMS)
    parser.add_argument("--conditions", nargs="+", default=list(k5.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k5.SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    frame = p621.load_frame(cutoff=k5.CUTOFF)
    six = k.six_indices()
    cap = k.effective_capacity_mw()
    probes = {s: p621.cells_and_donors(frame=frame, primary=i) for s, i in six.items()}
    for name in args.conditions:
        out = k5.WORK / "probes" / args.arm / f"{name}.parquet"
        if out.exists():
            continue
        _, rotation, shuffled = k5.split(name=name)
        assign = k5.assignment(rotation=rotation)
        records = []
        for seed in args.seeds:
            model = mm.MeteredMdn(n_primaries=16, arm=args.arm)
            model.load_state_dict(
                torch.load(
                    k5.weights_path(arm=args.arm, condition=name, seed=seed), map_location="cpu"
                )
            )
            model.eval()
            for site, i in six.items():
                pr = probes[site]
                met = metered_rows(probe=pr, farm=assign[site], shuffled=shuffled)
                y, reg = forward_mw(
                    model=model, frame=frame, primary=i, probe=pr, cap=float(cap[i]), metered=met
                )
                cell = pr["cell"]
                _, date_id = np.unique(pr["days"][pr["donor"]], return_inverse=True)
                rng = np.random.default_rng(1000 * seed + i)
                regs = [reg["u"], reg["temp"]]
                point = p621.fit(y=y, cell=cell, regressors=regs, w=np.ones(len(y)))
                boot = np.empty(p621.N_BOOT)
                for b in range(p621.N_BOOT):
                    w = rng.poisson(1.0, size=int(date_id.max()) + 1).astype(np.float64)[date_id]
                    boot[b] = p621.fit(y=y, cell=cell, regressors=regs, w=w)
                records.append(
                    {"condition": name, "seed": seed, "site": site, "model": "plain", "capacity_mw": point,
                     "boot": boot.tolist(), "n_rows": len(y), "input_farm": assign[site]}
                )  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(records).write_parquet(out)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

# ruff: noqa: E501, ANN001, D103
"""Experiment 650, step 3: build the head's rows and origin-level inputs for the extended-shift bank (CPU).

A copy of `e00647_data.build_bank` with experiment 650's shifts. A pass's rows come from step 2's
`k1_bank/rows_<month>_r<pass>.npz`. The Primary's own shift is added to its lags and to the target
(`shift`); the receivers' mirrored shift is added only to the neighbour signals, as in experiment
647. Experiment 647's real-row files (`head_data/real_<month>.npz`) are reused unchanged by the fit.

Writes `head_data/bank_<month>.npz` with the keys of experiment 647's file plus `pass_`.

Usage: python e00650_data.py MONTH
"""

import sys
import time

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00598_signals as s598  # ty: ignore[unresolved-import]
import e00647_data as d647  # ty: ignore[unresolved-import]
import e00650_common as c650
import numpy as np

PER = d647.PER
DATA = c650.WORK / "head_data"


def build_bank(*, month: str, clean: np.ndarray, table, nb: s598.Neighbours, n_days: int) -> None:
    keys = ("prim", "oi", "vi", "lead", "q", "y", "shift", "pass_", "org", "res", "res_star")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    feats, ages, lag_ages, gates, o_prims, o_ois, o_pass, keeps = [], [], [], [], [], [], [], []
    n_org = 0
    for r in c650.ALL_PASSES:
        f = c650.ROWS / f"rows_{month}_r{r:02d}.npz"
        assert f.exists(), f"missing rebuilt pass {f.name}"
        z = np.load(f)
        prim, oi, vi = (
            z["prim"].astype(np.int64),
            z["origin_idx"].astype(np.int64),
            z["idx"].astype(np.int64),
        )
        lead, obs, q = z["lead"], z["obs"], z["k1_inj"]
        assert np.array_equal(obs, clean[prim, vi].astype(np.float32), equal_nan=True)
        assert vi.max() < c650.WINDOW_START_IDX, "a valid time on or after 2026-07-01"
        own, mirror, _ = c650.shifts_for(clean=clean, month=month, r=r)
        fixed = np.where(np.isfinite(clean), clean + own, clean)
        mirrored = np.where(np.isfinite(fixed), fixed + mirror, fixed)
        shift = own[prim, vi].astype(np.float32)
        obs_c = obs + shift
        key = prim * 10_000_000 + oi
        uniq, org = np.unique(key, return_inverse=True)
        o_prim, o_oi = uniq // 10_000_000, uniq % 10_000_000
        feat, age, lag_age = d647.origin_features(prim=o_prim, oi=o_oi, power_lag=fixed, power_sig=mirrored, table=table, nb=nb)  # fmt: skip
        wide = (z["k1_inj"] != z["vin"]).any(axis=1)
        gate = np.zeros(len(uniq), bool)
        np.logical_or.at(gate, org, wide)
        for name, (a, b) in d647.R0_SPECS.items():
            parts[name].append(d647.origin_residual(prim=prim, oi=oi, vi=vi, lead=lead, q50=q[:, 5], obs=obs_c, first=a, last=b, n_days=n_days))  # fmt: skip
        keep = lead >= 24.0
        for k, v in (("prim", prim), ("oi", oi), ("vi", vi), ("lead", lead), ("q", q), ("y", obs), ("shift", shift), ("pass_", np.full(len(prim), r)), ("org", org + n_org)):  # fmt: skip
            parts[k].append(v)
        keeps.append(keep)
        feats.append(feat)
        ages.append(age)
        lag_ages.append(lag_age)
        gates.append(gate)
        o_prims.append(o_prim)
        o_ois.append(o_oi)
        o_pass.append(np.full(len(uniq), r))
        n_org += len(uniq)
        print(f"{month} r{r:02d}: {int(keep.sum())} rows", flush=True)
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    sel = np.concatenate(keeps)
    dt = {"prim": np.int8, "oi": np.int32, "vi": np.int32, "pass_": np.int16, "org": np.int32}
    out = {k: (v[sel].astype(dt[k]) if k in dt else v[sel]) for k, v in cat.items()}
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA / f"bank_{month}.npz", **out, feat=np.concatenate(feats), age=np.concatenate(ages),
        lag_age=np.concatenate(lag_ages), gate=np.concatenate(gates),
        org_prim=np.concatenate(o_prims).astype(np.int8), org_oi=np.concatenate(o_ois).astype(np.int32),
        org_pass=np.concatenate(o_pass).astype(np.int16),
    )  # fmt: skip
    print(f"{month} bank: {int(sel.sum())} rows, {n_org} origins", flush=True)


def main(month: str) -> None:
    t0 = time.time()
    clean = k578.clean()
    nb = d647.world()
    table = d647.lag_free_table(month=month)
    build_bank(month=month, clean=clean, table=table, nb=nb, n_days=clean.shape[1] // PER)
    print(f"{month} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

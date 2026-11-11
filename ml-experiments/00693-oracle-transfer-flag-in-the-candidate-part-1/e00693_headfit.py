# ruff: noqa: D103, E501, PLR0915, PLR0912, PERF401, PT018
"""Experiment 693: fit the prevailing head arms for one quarterly fold and score every row set (CPU).

Experiment 667's `p667_fit.fit_rhaz` is called unchanged. Only its `load` is wrapped, as experiment
681's `e00681_fit.install` does, and the paths point at this experiment's head data.

Arms (`ARM`):

- `A0`: the R-haz head with experiment 681's L3 lag block (14 residual lags, as 28 columns), 93 inputs.
- `A1`: `A0` plus the oracle: three columns (`in_progress`, `log1p(age in days)`, `age / 30`) appended,
  and the structural age `max(heuristic age, oracle age)`. The `gate` column stays 0.
- `A2`: `A1` with the placebo spans in place of the oracle's.
- `B`: B-haz with the heuristic age (no fit; experiment 667's fallback).
- `OB`: oracle B-haz: the age is the oracle age and `p = 0` outside a transfer (no fit).
- `CAND`: the Candidate's own quantiles, the head's input.

Folds before 2026-02 have fewer than 2 earlier banks, so `A0` to `A2` run only for folds 2026-02 and
2026-05; `B`, `OB`, and `CAND` run for every fold. A quarterly head is applied to all three months
of its fold. For each (fold, arm, row set) one parquet of per-row scores is written to
`<root>/head_scores/<arm>_<fold>_<kind>_<month>.parquet` (seed-mean forecast, seeds 0 to 2).

Usage: python e00693_headfit.py ARM FOLD [--seeds 0 1 2]
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00647_fit as f647  # ty: ignore[unresolved-import]
import e00652_score as s652  # ty: ignore[unresolved-import]
import e00693_headdata as hd
import e00693_scoring as sc
import numpy as np
import p667_data as dat  # ty: ignore[unresolved-import]
import p667_fit as base  # ty: ignore[unresolved-import]
import p667_head as head667  # ty: ignore[unresolved-import]
import polars as pl
import torch  # ty: ignore[unresolved-import]

OUT: Final = hd.ROOT / "head_fits"
SCORES: Final = hd.ROOT / "head_scores"
MONTHS: Final = (
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
"""Banks and real rows exist for these months only."""
FIRST_SCORED: Final = "2026-02"
"""The fold of the first contrasts; the earlier fold's head is B-haz."""
ARMS_HEAD: Final = ("A0", "A1", "A2")


def months_of(*, fold: str) -> list[str]:
    return [m for m in MONTHS if hd.fold_of(month=m) == fold]


def install(*, arm: str) -> None:
    """Wrap `base.load` so each loaded head-data dict gets the arm's structural age and extra columns, once."""
    original = base.load
    n_feat = f647.N_FEAT

    def load(*, month: str, kind: str) -> dict:
        z = original(month=month, kind=kind)
        if z.get("_e693"):
            return z
        lag = z["lag_true"]
        extra = [np.nan_to_num(lag), np.isnan(lag).astype(np.float32)]
        if arm in {"A1", "A2", "OB"}:
            name = "placebo" if arm == "A2" else "oracle"
            o_age = z[f"o_age_{name}"].astype(np.float32)
            o_in = z[f"o_in_{name}"].astype(np.float32)
            age = np.maximum(z["age"], o_age).astype(np.float32) if arm != "OB" else o_age
            z["age"] = age
            z["raw"][:, n_feat] = np.log1p(age)
            z["raw"][:, n_feat + 1] = age / 30.0
            z["o_in"] = o_in
            extra += [o_in[:, None], np.log1p(o_age)[:, None], (o_age / 30.0)[:, None]]
        z["raw"] = np.concatenate([z["raw"], *extra], axis=1).astype(np.float32)
        z["_e693"] = True
        return z

    base.load = load
    base.OUT = OUT / arm
    base._CACHE.clear()
    dat.HEAD_DATA = hd.HEAD_DATA


def row_table(
    *, z: dict, q: np.ndarray, p: np.ndarray, extra: dict[str, np.ndarray]
) -> pl.DataFrame:
    """Return the per-row score table of forecast `q` (shares of capacity) against the target."""
    y = np.nan_to_num(z["target"]).astype(np.float32)
    ok = np.isfinite(z["target"])
    cols = {
        "prim": z["prim"],
        "oi": z["oi"],
        "vi": z["vi"],
        "lead": z["lead"],
        "y": y,
        "p": p.astype(np.float32),
        "med": q[:, 5],
    }
    cols.update({k: v.astype(np.float32) for k, v in sc.row_scores(q=q, y=y).items()})
    cols.update(extra)
    return pl.DataFrame(cols).filter(pl.Series(ok))


def survival_of(*, z: dict, curve: tuple, oracle: bool) -> np.ndarray:
    """Return B-haz's survival probability per row; for the oracle, 0 outside a transfer."""
    age = z["age"][z["org"]].astype(np.float32)
    p = s652.cond_survival(curve=curve, age_days=age, lead_h=z["lead"])
    if oracle:
        p = np.where(z["o_in"][z["org"]] > 0.5, p, 0.0)
    return p.astype(np.float32)


def sets_of(*, fold: str) -> list[tuple[str, str, str]]:
    """Return the (kind, month key, label) of every row set a fold's head scores."""
    out = [(k, m, m) for m in months_of(fold=fold) for k in ("real", "bankheld")]
    out.append(("bankdec", fold, fold))
    return out


def write_scores(*, arm: str, fold: str, kind: str, label: str, table: pl.DataFrame) -> None:
    SCORES.mkdir(parents=True, exist_ok=True)
    out = SCORES / f"{arm}_{fold}_{kind}_{label}.parquet"
    table.write_parquet(out.with_suffix(".tmp"), compression="zstd")
    out.with_suffix(".tmp").rename(out)


def extras_of(*, z: dict, kind: str) -> dict[str, np.ndarray]:
    ex: dict[str, np.ndarray] = {}
    if kind == "real":
        ex["flagged"] = z["flagged"].astype(np.int8)
        ex["o_in"] = z["o_in_oracle"][z["org"]].astype(np.int8)
        ex["o_age"] = z["o_age_oracle"][z["org"]].astype(np.float32)
    else:
        ex["phase"] = z["phase"]
        ex["shift"] = z["shift"]
        ex["pass_"] = z["pass_"]
    return ex


def run(*, arm: str, fold: str, seeds: list[int]) -> None:
    t0 = time.time()
    install(arm=arm)
    earlier = [m for m in MONTHS if m < fold]
    rows = sets_of(fold=fold)
    if arm == "CAND":
        for kind, key, label in rows:
            z = base.load(month=key, kind=kind)
            write_scores(
                arm=arm,
                fold=fold,
                kind=kind,
                label=label,
                table=row_table(
                    z=z, q=z["q"], p=np.zeros(len(z["q"])), extra=extras_of(z=z, kind=kind)
                ),
            )
        print(f"CAND {fold}: scored in {time.time() - t0:.0f}s", flush=True)
        return
    if arm in {"B", "OB"}:
        print(base.fit_bhaz(month=fold), flush=True)
        z0 = np.load(base.OUT / f"Bhaz_{fold}.npz")
        curve = (str(z0["curve"][0]), tuple(float(v) for v in z0["params"]), 0.0)
        for kind, key, label in rows:
            z = base.load(month=key, kind=kind)
            p = survival_of(z=z, curve=curve, oracle=arm == "OB")
            q = np.sort(z["q"] + (np.nan_to_num(z["res"]) * p)[:, None], axis=1).astype(np.float32)
            write_scores(
                arm=arm,
                fold=fold,
                kind=kind,
                label=label,
                table=row_table(z=z, q=q, p=p, extra=extras_of(z=z, kind=kind)),
            )
        print(f"{arm} {fold}: done in {time.time() - t0:.0f}s", flush=True)
        return
    assert arm in ARMS_HEAD and len(earlier) >= base.MIN_BANKS, (
        f"{arm} needs at least {base.MIN_BANKS} earlier banks"
    )
    sums: dict[tuple[str, str], list[np.ndarray]] = {}
    for seed in seeds:
        print(base.fit_rhaz(month=fold, seed=seed, train_months=earlier), flush=True)
        pt = torch.load(base.OUT / f"Rhaz_{fold}_s{seed}.pt", weights_only=False)
        stats = pt["stats"]
        head = head667.HazardHead(n_in=stats[0].shape[0] + 3)
        head.load_state_dict(pt["state"])
        head.eval()
        for kind, key, label in rows:
            z = base.load(month=key, kind=kind)
            q, p = base.predict_rows(head=head, z=z, stats=stats, seed=seed, with_q=True)
            sums.setdefault((kind, label), []).append((q, p))  # ty: ignore[invalid-argument-type]
    for kind, key, label in rows:
        z = base.load(month=key, kind=kind)
        per_seed = sums[(kind, label)]
        q = np.sort(np.mean([a for a, _ in per_seed], axis=0), axis=1).astype(np.float32)
        p = np.mean([b for _, b in per_seed], axis=0)
        table = row_table(z=z, q=q, p=p, extra=extras_of(z=z, kind=kind))
        for s, (qs, _ps) in zip(seeds, per_seed, strict=True):
            ys = np.nan_to_num(z["target"]).astype(np.float32)
            ok = np.isfinite(z["target"])
            table = table.with_columns(
                pl.Series(
                    f"crps_s{s}",
                    sc.row_scores(q=np.sort(qs, axis=1), y=ys)["crps"][ok].astype(np.float32),
                )
            )
        write_scores(arm=arm, fold=fold, kind=kind, label=label, table=table)
    print(f"{arm} {fold}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("A0", "A1", "A2", "B", "OB", "CAND"))
    ap.add_argument("fold", choices=hd.FOLDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    run(arm=a.arm, fold=a.fold, seeds=a.seeds)

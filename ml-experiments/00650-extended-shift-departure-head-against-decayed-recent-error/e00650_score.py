# ruff: noqa: E501, ANN001, D103
"""Experiment 650, step 5: score the arms on the extended held-out family and on real event-free rows (CPU).

Arms: K1 (stored forecast), R0-168 (K1 shifted by its own decayed recent error, tau 168 h), and the
fitted heads `Rnew`, `Rold`, `Placebo` (seeds averaged by score, not by forecast). Row sets, per
scored month: `held` (the extended held-out passes, target = observed power plus the shift),
`fam` (training-family passes of the scored month, unseen), and `real` event-free rows (target =
observed power). Phases of a bank row: `pre` (valid time before the Primary's shift starts), `inside`, and
`after` (from the shift's end).

Equal width scaling: for each arm, month, set, and phase, every quantile is scaled about the median by
one factor chosen so that the q05-q95 band covers 0.90 of the rows (the nominal level). The same rule
applies to every arm including K1 and R0-168; the factor is fitted on the scored rows (an oracle).

R0's residual history is missing for some rows; rows without it are dropped from every arm (the counts are
written to `score/dropped.csv`).

Usage: python e00650_score.py [--months 2025-10 2025-12]
"""

import argparse

import e00650_common as c650
import e00650_fit as f650
import numpy as np
import polars as pl

TAUS = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99], np.float32)
OUT = c650.WORK / "score"
ARMS = ("K1", "R0-168", "Rnew", "Rold", "Placebo")
TAU_H = 168.0


def pin(*, y, q):
    e = y[:, None] - q
    return np.maximum(TAUS * e, (TAUS - 1.0) * e) * 100.0


def crps(*, y, q):
    return 2.0 * pin(y=y, q=q).mean(1)


def scale_to_nominal(*, y, q, target: float = 0.90):
    """Return (q scaled about the median, factor) so that the q05-q95 band covers `target` of the rows."""
    med = q[:, 5:6]
    lo, hi = q[:, 1:2], q[:, 9:10]

    def cover(f):
        return float(((y[:, None] >= med - f * (med - lo)) & (y[:, None] <= med + f * (hi - med))).mean())

    a, b = 0.2, 20.0
    for _ in range(40):
        m = 0.5 * (a + b)
        if cover(m) < target:
            a = m
        else:
            b = m
    f = 0.5 * (a + b)
    return (med + f * (q - med)).astype(np.float32), f


def stats(*, y, q, qs) -> dict:
    p = pin(y=y, q=q)
    return {
        "n": len(y), "crps": float(2 * p.mean()), "cover90": float(((y >= q[:, 1]) & (y <= q[:, 9])).mean()),
        "crps_scaled": float(crps(y=y, q=qs).mean()), "pin01": float(p[:, 0].mean()), "pin05": float(p[:, 1].mean()),
        "pin95": float(p[:, 9].mean()), "pin99": float(p[:, 10].mean()),
        "cover98": float(((y >= q[:, 0]) & (y <= q[:, 10])).mean()),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=list(f650.PAIR))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows_out, dropped, per_row = [], [], []
    for m in a.months:
        src = f650.sources(month=m, bank="new")
        events = pl.concat([pl.read_parquet(c650.ROWS / f"events_{m}_r{r:02d}.parquet") for r in c650.ALL_PASSES])
        ev = {(int(r["pass"]), int(r["primary"])): r for r in events.iter_rows(named=True)}
        for setname, z, idx_name in (("held", src["bank"], "held"), ("fam", src["bank"], "fam"), ("real", src["real"], "real")):
            fits = {arm: [np.load(c650.WORK / "fits" / f"{arm}_{m}_s{s}.npz") for s in a.seeds if (c650.WORK / "fits" / f"{arm}_{m}_s{s}.npz").exists()] for arm in ("Rnew", "Rold", "Placebo")}  # fmt: skip
            if not fits["Rnew"]:
                continue
            idx = fits["Rnew"][0][f"{idx_name}_idx"]
            for arm, fs in fits.items():
                for f in fs:
                    assert np.array_equal(f[f"{idx_name}_idx"], idx)
            if setname == "real":
                y = z["y"][idx]
                keep = z["event_free"][idx] & ~z["flagged"][idx] & ~z["verified"][idx] & np.isfinite(z["res"][idx])
                phase = np.full(len(idx), "event_free")
                meta = {k: np.full(len(idx), -1) for k in ("depth", "cell")}
                meta["cell"] = np.full(len(idx), "")
            else:
                y = (z["y"][idx] + z["shift"][idx]).astype(np.float32)
                keep = np.isfinite(z["res"][idx])
                pa, pr, vi = z["pass_"][idx].astype(int), z["prim"][idx].astype(int), z["vi"][idx].astype(int)
                on = np.array([ev[(p, q)]["onset_idx"] for p, q in zip(pa, pr, strict=True)])
                en = np.array([ev[(p, q)]["end_idx"] for p, q in zip(pa, pr, strict=True)])
                phase = np.where(vi < on, "pre", np.where(vi < en, "inside", "after"))
                meta = {
                    "depth": np.array([ev[(p, q)]["depth"] for p, q in zip(pa, pr, strict=True)]),
                    "cell": np.array([f'{ev[(p, q)]["out_of_support"]}|{ev[(p, q)]["length_days"]:.0f}d' for p, q in zip(pa, pr, strict=True)]),
                }  # fmt: skip
            dropped.append({"month": m, "set": setname, "rows": len(idx), "kept": int(keep.sum())})
            lead = z["lead"][idx]
            assert z["vi"][idx].max() < c650.WINDOW_START_IDX, "a valid time on or after 2026-07-01"
            assert {len(fs) for fs in fits.values()} == {len(fits["Rnew"])}, "arms have different seed counts"
            k1 = z["q"][idx]
            r0 = np.sort(k1 + (z["res"][idx] * np.exp(-lead / TAU_H))[:, None], axis=1).astype(np.float32)
            qs = {"K1": [k1], "R0-168": [r0], **{arm: [np.sort(f[f"{idx_name}_q"], axis=1) for f in fs] for arm, fs in fits.items()}}
            for ph in sorted(set(phase[keep])):
                sel = keep & (phase == ph)
                if sel.sum() < 100:
                    continue
                for arm in ARMS:
                    for si, q_ in enumerate(qs.get(arm, [])):
                        q = q_[sel]
                        qsc, fac = scale_to_nominal(y=y[sel], q=q)
                        rows_out.append({"month": m, "set": setname, "phase": ph, "arm": arm, "seed": si, "factor": fac, **stats(y=y[sel], q=q, qs=qsc)})  # fmt: skip
            for arm in ARMS:
                for si, q_ in enumerate(qs.get(arm, [])):
                    big = np.isin(phase, [p_ for p_ in set(phase[keep]) if (keep & (phase == p_)).sum() >= 100])
                    sel = keep & big
                    if not sel.any():
                        continue
                    qsc = np.zeros_like(q_[sel])
                    for ph in sorted(set(phase[keep])):
                        s2 = (phase == ph)[sel]
                        if s2.sum() < 100:
                            continue
                        qsc[s2], _ = scale_to_nominal(y=y[sel][s2], q=q_[sel][s2])
                    per_row.append(pl.DataFrame({
                        "month": m, "set": setname, "arm": arm, "seed": si, "phase": phase[sel],
                        "cell": meta["cell"][sel], "depth": meta["depth"][sel].astype(float),
                        "prim": z["prim"][idx][sel].astype(int), "crps": crps(y=y[sel], q=q_[sel]),
                        "crps_scaled": crps(y=y[sel], q=qsc),
                    }))  # fmt: skip
    pl.DataFrame(rows_out).write_csv(OUT / "scorecard.csv")
    pl.DataFrame(dropped).write_csv(OUT / "dropped.csv")
    pl.concat(per_row).write_parquet(OUT / "per_row_summary.parquet")
    print(pl.DataFrame(rows_out).group_by("set", "phase", "arm").agg(pl.col("crps").mean(), pl.col("crps_scaled").mean(), pl.col("cover90").mean()).sort("set", "phase", "arm"))  # fmt: skip


if __name__ == "__main__":
    main()

# ruff: noqa: D103, E501, PLR0915, PLR0912
"""Experiment 647, stage 2: score the arms on the first month pair, the detector read-out, and the feasibility gate (CPU).

Arms: K1 (the NRA output, unchanged), R0 and R0* (K1 shifted by its own decayed recent error, tau 72
and 168 h; R0* is the variant of the design answer's verified figure), the fitted heads (seed means of
the per-row scores; both seeds are listed), the widened-K1 control (K1's band rescaled about its
median to match H1's q05-q95 coverage on the set), and the oracle (a mixture with the true depth and
age on injected rows, K1 elsewhere).

Row sets: P1 and P5 (real event-free rows not flagged by experiment 637's detector); P2 (bank rows with
the valid time inside the shift, scored against the shifted target) and P2b (the same with the shift
already in the newest lag); P3 (bank rows after the synthetic end, origin before the end, scored
against the unshifted target); P4 (real rows on days that the detector flags, plus the verified D5 and
D16 pair). The bars' month-block intervals are not available on two months; the table gives point
estimates, per-month values, and both seeds.

Writes `score/scorecard.csv`, `score/detector.csv`, `score/gate.txt`, and prints the gate.

Usage: python e00647_score.py
"""

import hashlib
import sys
import time
from datetime import UTC, datetime

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00601_common as k601  # ty: ignore[unresolved-import]
import e00637_detector as d637  # ty: ignore[unresolved-import]
import e00645_score as k645  # ty: ignore[unresolved-import]
import e00647_common as c647  # ty: ignore[unresolved-import]
import e00647_data as dat  # ty: ignore[unresolved-import]
import e00647_fit as fit  # ty: ignore[unresolved-import]
import e00647_head as head  # ty: ignore[unresolved-import]
import e00647_r0 as r0  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch

PAIR = ("2025-10", "2025-12")
HEADS = ("H1", "H2", "H1p", "H2p", "AG", "AS", "H1short", "H1long")
OUT = c647.WORK / "score"
PER = 48
GROUPS = {"signed-MW six": "six MW", "MVA-only eight": "eight MVA", "D15 and D18": "report-only"}
BANDS = {"24-48 h": (24, 48), "48-120 h": (48, 120), "120-240 h": (120, 240)}
METRICS = ("crps", "mae", "low", "high", "width", "pin01", "pin05", "pin95", "pin99", "cost_0.9", "cost_0.75", "cost_0.5")
ALARM_D = 0.05
DETECT_FRACTION_NOTE = "alarm: |d| >= 0.05 of capacity at leads of 24 to 26 h"


def window_flags(*, clean_c: np.ndarray, ref_clim: np.ndarray, day_lo: int, day_hi: int) -> np.ndarray:
    """Return experiment 637's flags (16, day_hi - day_lo) for the days `day_lo` to `day_hi - 1` of the (possibly contaminated) power `clean_c`.

    The same rule as `e00645_score.ara_flags`, with the heuristic reference and the residuals formed
    for the window only (the cross-Primary median is per day). The climatology reference reads only
    data before 2025-07-01, so it is the same for any contamination from 2025-10 on.
    """
    rule = d637.load_rule(k645.RULE_PATH)
    n = (day_hi - day_lo) * PER
    prim = np.repeat(np.arange(16), n)
    idx = np.tile(np.arange(day_lo * PER, day_hi * PER), 16)
    ref_h = d637.s.manual_heuristic(clean=clean_c, prim=prim, idx=idx, lead_h=np.zeros(len(idx)))[:, 5].reshape(16, n)
    obs = clean_c[:, day_lo * PER : day_hi * PER]
    daily = []
    for ref in (ref_h, ref_clim[:, day_lo * PER : day_hi * PER]):
        with np.errstate(all="ignore"):
            daily.append(np.nanmean((obs - ref).reshape(16, -1, PER), axis=2))
    resid = d637.combine(daily=np.stack(daily).astype(np.float32))
    op = rule["operating"]
    return d637.flag_days(resid=resid, centre=np.array(rule["centre"]), mad=np.array(rule["mad"]), k=op["k"], n=op["n"])


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def row_metrics(*, y: np.ndarray, q: np.ndarray, prim: np.ndarray, lead: np.ndarray, cost_ctx: dict) -> dict[str, np.ndarray]:
    m = fit.metrics(y=y, q=q)
    out: dict[str, np.ndarray] = {k: np.asarray(m[k], np.float32) for k in ("crps", "low", "high", "width", "pin01", "pin05", "pin95", "pin99")}
    out["mae"] = (np.abs(y - q[:, 5]) * 100.0).astype(np.float32)
    ok = cost_ctx["six"][prim] & (lead >= 24.0) & (lead < 36.0)
    for f in k601.FRACTIONS:
        c = np.full(len(y), np.nan, np.float32)
        if ok.any():
            cap = cost_ctx["cap"][prim[ok]]
            c[ok] = k601.avoidable_cost(u=q[ok, 9] * cap, y=y[ok] * cap, limit=cost_ctx["limit"][f][prim[ok]]) * 48.0
        out[f"cost_{f}"] = c
    return out


def sums(*, m: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    out = {"rows": float(mask.sum())}
    for k in METRICS:
        v = m[k][mask].astype(np.float64)
        f = np.isfinite(v)
        out[f"{k}_sum"] = float(v[f].sum())
        out[f"{k}_n"] = float(f.sum())
    return out


def finish(s: dict[str, float]) -> dict[str, float]:
    r = {"rows": s["rows"]}
    for k in METRICS:
        r[k] = s[f"{k}_sum"] / s[f"{k}_n"] if s[f"{k}_n"] > 0 else float("nan")
    r["cover90"] = 1.0 - r["low"] - r["high"] if np.isfinite(r["low"]) else float("nan")
    return r


def shifted_q(*, q: np.ndarray, res: np.ndarray, lead: np.ndarray, tau: float) -> np.ndarray:
    return r0.shifted(q=q, res=np.nan_to_num(res), lead=lead, tau_h=tau).astype(np.float32)


def widen(*, q: np.ndarray, factor: float) -> np.ndarray:
    med = q[:, 5:6]
    return (med + factor * (q - med)).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    sm = k578.sites()
    group_of = np.array([GROUPS[g] for g in sm["group"].to_list()])
    cap_all = sm["capacity_mw"].to_numpy().astype(np.float64)
    clean = k578.clean()
    peak = np.nanmax(clean[:, : dat.month_start_idx(month="2025-10")], axis=1) * cap_all
    flags_real = k645.ara_flags(clean=clean)
    ref_clim = d637.climatology_reference(clean=clean[:, : clean.shape[1] // PER * PER])
    print("detector flags and climatology reference built", flush=True)
    six = group_of == "six MW"
    cost_ctx = {"six": six, "cap": cap_all, "limit": {f: f * peak for f in k601.FRACTIONS}}
    fit_dir = c647.WORK / "fits"
    rows_out: list[dict] = []
    det_out: list[dict] = []
    nra_hash: dict[str, str] = {}

    # ------------------------------------------------------------------ per month, per variant
    per: dict = {}  # (variant, month, setname, cut, arm, seed) -> sums
    for month in PAIR:
        d = fit.load(month=month)
        for kind in ("real", "bank"):
            z = d[kind]
            idx = np.flatnonzero(z["usable"])
            nra_hash[f"{kind}_{month}"] = sha(z["q"][idx])
        rl, bk = d["real"], d["bank"]
        ri, bi = np.flatnonzero(rl["usable"]), np.flatnonzero(bk["usable"])
        # real rows
        r = {k: rl[k][ri] for k in ("prim", "oi", "vi", "lead", "q", "y", "event_free", "verified", "flagged", "res", "res_star")}
        r["prim"] = r["prim"].astype(np.int64)
        grp_r = group_of[r["prim"]]
        sets_r = {
            "P1/P5 event-free": r["event_free"] & ~r["flagged"],
            "P4 flagged or verified": r["flagged"] | r["verified"],
            "P4a flagged days": r["flagged"],
            "P4b verified D5/D16 pair": r["verified"],
            "all real rows": np.ones(len(ri), bool),
        }
        # bank rows
        b = {k: bk[k][bi] for k in ("prim", "oi", "vi", "lead", "q", "y", "shift", "pass_", "res", "res_star")}
        b["prim"] = b["prim"].astype(np.int64)
        ms = dat.month_start_idx(month=month)
        end = ms + bank.END_DAY * PER
        ev = {(p, rr): bank.event(primary=p, r=rr) for p in range(16) for rr in range(bank.N_PASSES)}
        depth = np.array([abs(ev[(int(p), int(rr))][0]) for p, rr in zip(b["prim"], b["pass_"], strict=True)])
        length = np.array([ev[(int(p), int(rr))][1] for p, rr in zip(b["prim"], b["pass_"], strict=True)])
        on = end - length * PER
        inside = (b["vi"] >= on) & (b["vi"] < end)
        after = (b["vi"] >= end) & (b["oi"] < end)
        lag_in = (b["oi"] - PER) >= on
        sets_b = {"P2 inside shift": inside, "P2b inside, shift in lags": inside & lag_in, "P3 after end": after}
        y_b = (b["y"] + b["shift"]).astype(np.float32)

        # no-fit arms
        arms_q_r = {"K1": r["q"], **{f"R0_{t}": shifted_q(q=r["q"], res=r["res"], lead=r["lead"], tau=t) for t in (72, 168)},
                    **{f"R0s_{t}": shifted_q(q=r["q"], res=r["res_star"], lead=r["lead"], tau=t) for t in (72, 168)}}  # fmt: skip
        arms_q_b = {"K1": b["q"], **{f"R0_{t}": shifted_q(q=b["q"], res=b["res"], lead=b["lead"], tau=t) for t in (72, 168)},
                    **{f"R0s_{t}": shifted_q(q=b["q"], res=b["res_star"], lead=b["lead"], tau=t) for t in (72, 168)}}  # fmt: skip
        # oracle on bank rows: true depth, true age at the newest lag, K1 elsewhere
        signed = np.array([ev[(int(p), int(rr))][0] for p, rr in zip(b["prim"], b["pass_"], strict=True)])
        lag_idx = b["oi"] - PER
        active = (lag_idx >= on) & (lag_idx < end)
        age_true = np.where(active, (lag_idx - on) / PER + 1.0, 0.0).astype(np.float32)
        qt = torch.from_numpy(b["q"])
        dd = torch.from_numpy(np.where(active, signed, 0.0).astype(np.float32))
        pp = head.survival_probability(age_days=torch.from_numpy(age_true), lead_h=torch.from_numpy(b["lead"]), short_weight=0.5)
        qo = np.concatenate([head.mixture_quantiles(q=qt[a : a + 200000], d=dd[a : a + 200000], sigma=torch.full((min(200000, len(qt) - a),), 1e-3), p=pp[a : a + 200000]).numpy() for a in range(0, len(qt), 200000)])  # fmt: skip
        arms_q_b["oracle"] = np.where(active[:, None], qo, b["q"]).astype(np.float32)
        arms_q_r["oracle"] = r["q"]

        metr_r = {a: row_metrics(y=r["y"], q=q, prim=r["prim"], lead=r["lead"], cost_ctx=cost_ctx) for a, q in arms_q_r.items()}
        metr_b = {a: row_metrics(y=y_b, q=q, prim=b["prim"], lead=b["lead"], cost_ctx=cost_ctx) for a, q in arms_q_b.items()}
        # fitted arms
        fitted: dict = {}
        for f in sorted(fit_dir.glob(f"*_{month}_*_s*.npz")):
            arm, _, variant, seed = f.stem.split("_")
            z = np.load(f)
            assert np.array_equal(z["real_idx"], ri) and np.array_equal(z["bank_idx"], bi)
            mr, mb = {}, {}
            for kind, dst, y_, prim_, lead_ in (("real", mr, r["y"], r["prim"], r["lead"]), ("bank", mb, y_b, b["prim"], b["lead"])):
                m = {k: z[f"{kind}_{k}"] for k in ("crps", "low", "high", "width", "pin01", "pin05", "pin95", "pin99")}
                m["mae"] = (np.abs(y_ - z[f"{kind}_q50"]) * 100.0).astype(np.float32)
                q95 = z[f"{kind}_q95"]
                ok = six[prim_] & (lead_ >= 24.0) & (lead_ < 36.0)
                for fr in k601.FRACTIONS:
                    c = np.full(len(y_), np.nan, np.float32)
                    cap = cap_all[prim_[ok]]
                    c[ok] = k601.avoidable_cost(u=q95[ok] * cap, y=y_[ok] * cap, limit=cost_ctx["limit"][fr][prim_[ok]]) * 48.0
                    m[f"cost_{fr}"] = c
                dst.update({k: v for k, v in m.items()})
            fitted[(arm, variant, int(seed[1:]))] = (mr, mb, z)

        # widened-K1 control: match H1's lomo coverage on P2 and P4
        for variant in ("lomo", "fc"):
            for setname, kind, mask in (("P2 inside shift", "bank", sets_b["P2 inside shift"]), ("P4 flagged or verified", "real", sets_r["P4 flagged or verified"])):
                hs = [fitted[k] for k in fitted if k[0] == "H1" and k[1] == variant]
                if not hs or mask.sum() < 100:
                    continue
                cov = np.mean([1.0 - (h[1 if kind == "bank" else 0]["low"][mask].mean() + h[1 if kind == "bank" else 0]["high"][mask].mean()) for h in hs])
                qq, yy, pm, ll = (b["q"], y_b, b["prim"], b["lead"]) if kind == "bank" else (r["q"], r["y"], r["prim"], r["lead"])
                lo, hi = 1.0, 8.0
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    w = widen(q=qq[mask], factor=mid)
                    c = float(((yy[mask] >= w[:, 1]) & (yy[mask] <= w[:, 9])).mean())
                    lo, hi = (mid, hi) if c < cov else (lo, mid)
                w = widen(q=qq, factor=0.5 * (lo + hi))
                m = row_metrics(y=yy, q=w, prim=pm, lead=ll, cost_ctx=cost_ctx)
                tag = f"widenK1@{setname.split()[0]}_{variant}"
                (metr_b if kind == "bank" else metr_r)[tag] = m
                per[("factor", variant, month, tag)] = 0.5 * (lo + hi)

        det_out.append({"variant": "n/a", "month": month, "seed": -1, "signal": "bank", "what": "Primaries with bank rows", "value": float(len(np.unique(b["prim"]))), "n": len(b["prim"])})  # fmt: skip
        print(f"{month}: arms and fitted rows loaded at {time.time() - t_start:.0f}s", flush=True)
        # score tables
        def add(*, variant, setname, cutname, mask, arm, seed, mr_=None, mb_=None, kind) -> None:  # noqa: ANN001
            m = (mr_ if kind == "real" else mb_)
            per[(variant, month, setname, cutname, arm, seed)] = sums(m=m, mask=mask)

        def cuts_real(mask):  # noqa: ANN001, ANN202
            yield "all", mask
            for g in set(GROUPS.values()):
                yield f"group {g}", mask & (grp_r == g)
                for bn, (lo_, hi_) in BANDS.items():
                    yield f"group {g} | {bn}", mask & (grp_r == g) & (r["lead"] >= lo_) & (r["lead"] < hi_)
            for bn, (lo_, hi_) in BANDS.items():
                yield f"band {bn}", mask & (r["lead"] >= lo_) & (r["lead"] < hi_)

        def cuts_bank(mask):  # noqa: ANN001, ANN202
            yield "all", mask
            for ln in bank.LENGTHS_DAYS:
                yield f"length {ln} d", mask & (length == ln)
            for dp in bank.DEPTHS:
                yield f"depth {dp}", mask & np.isclose(depth, dp)
            for bn, (lo_, hi_) in BANDS.items():
                yield f"band {bn}", mask & (b["lead"] >= lo_) & (b["lead"] < hi_)

        for variant in ("lomo", "fc"):
            for arm in ("K1", "R0_72", "R0_168", "R0s_72", "R0s_168", "oracle", "widenK1@P2_" + variant, "widenK1@P4_" + variant):
                for setname, mask in sets_r.items():
                    if arm in metr_r and not arm.startswith("widenK1@P2"):
                        for cn, cm in cuts_real(mask):
                            add(variant=variant, setname=setname, cutname=cn, mask=cm, arm=arm.split("_" + variant)[0], seed=-1, mr_=metr_r[arm], kind="real")
                for setname, mask in sets_b.items():
                    if arm in metr_b and not arm.startswith("widenK1@P4"):
                        for cn, cm in cuts_bank(mask):
                            add(variant=variant, setname=setname, cutname=cn, mask=cm, arm=arm.split("_" + variant)[0], seed=-1, mb_=metr_b[arm], kind="bank")
            for (arm, v, seed), (mr, mb, _z) in fitted.items():
                if v != variant:
                    continue
                for setname, mask in sets_r.items():
                    for cn, cm in cuts_real(mask):
                        add(variant=variant, setname=setname, cutname=cn, mask=cm, arm=arm, seed=seed, mr_=mr, kind="real")
                for setname, mask in sets_b.items():
                    for cn, cm in cuts_bank(mask):
                        add(variant=variant, setname=setname, cutname=cn, mask=cm, arm=arm, seed=seed, mb_=mb, kind="bank")

        print(f"{month}: score sums done at {time.time() - t_start:.0f}s", flush=True)
        # ------------------------------------------------------------ detector read-out (bank + real)
        for (arm, v, seed), (_mr, _mb, z) in fitted.items():
            if arm != "H1":
                continue
            for kind, rows_d, prim_, oi_, lead_ in (("real", r, r["prim"], r["oi"], r["lead"]), ("bank", b, b["prim"], b["oi"], b["lead"])):
                dvals = z[f"{kind}_d"]
                near = lead_ < 26.0
                key = prim_[near] * 10_000_000 + oi_[near] + (1000 * 10_000_000 * (b["pass_"][near].astype(np.int64) if kind == "bank" else 0))
                uniq, inv = np.unique(key, return_inverse=True)
                dsum = np.bincount(inv, weights=dvals[near])
                cnt = np.bincount(inv)
                dmean = dsum / cnt
                first = np.zeros(len(uniq), np.int64)
                first[inv] = np.arange(near.sum())
                src = np.flatnonzero(near)[first]
                org = (rl if kind == "real" else bk)["org"][ri if kind == "real" else bi][src]
                age_o = (rl if kind == "real" else bk)["age"][org]
                p48 = head.survival_probability(age_days=torch.from_numpy(age_o.astype(np.float32)), lead_h=torch.full((len(org),), 48.0), short_weight=0.5).numpy()
                alarms = {"depth": np.abs(dmean) >= ALARM_D, "survival": (age_o >= 1) & (p48 >= 0.5)}
                u_prim = prim_[src]
                u_oi = oi_[src]
                if kind == "real":
                    ef_o = (r["event_free"] & ~r["flagged"])[src]
                    ver_o = r["verified"][src]
                    for sig, al in alarms.items():
                        det_out.append({"variant": v, "month": month, "seed": seed, "signal": sig, "what": "false alarms per Primary-month (event-free origins)", "value": float(al[ef_o].sum() / 16.0), "n": int(ef_o.sum())})  # fmt: skip
                        det_out.append({"variant": v, "month": month, "seed": seed, "signal": sig, "what": "alarm share on verified D5/D16 origins", "value": float(al[ver_o].mean()) if ver_o.any() else float("nan"), "n": int(ver_o.sum())})  # fmt: skip
                        if ver_o.any():
                            first_alarm = np.min(np.where(al & ver_o, u_oi, 10**9))
                            det_out.append({"variant": v, "month": month, "seed": seed, "signal": sig, "what": "first alarm origin on verified pair (UTC date)", "value": float(first_alarm) if first_alarm < 10**9 else float("nan"), "n": 0})  # fmt: skip
                else:
                    u_pass = bk["pass_"][bi][src]
                    for sig, al in alarms.items():
                        ev_rows = []
                        for rr in range(bank.N_PASSES):
                            for p_ in range(16):
                                sgn, ln, _ = bank.event(primary=p_, r=rr)
                                on_ = end - ln * PER
                                sel = (u_prim == p_) & (u_pass == rr) & (u_oi >= on_) & (u_oi < end)
                                if not sel.any():
                                    continue  # no stored forecast of this Primary in the event window
                                hit = sel & al
                                days = float((u_oi[hit].min() - on_) / PER) if hit.any() else float("nan")
                                ev_rows.append({"depth": abs(sgn), "length": ln, "detected": bool(hit.any()), "days": days})
                        edf = pl.DataFrame(ev_rows)
                        for (dp, ln), g in edf.group_by(["depth", "length"]):
                            det_out.append({"variant": v, "month": month, "seed": seed, "signal": sig, "what": f"bank recall depth {dp} length {ln}", "value": float(g["detected"].mean()), "n": g.height})  # fmt: skip
                            det_out.append({"variant": v, "month": month, "seed": seed, "signal": sig, "what": f"bank days to first alarm depth {dp} length {ln}", "value": float(g["days"].mean()) if g["detected"].any() else float("nan"), "n": int(g["detected"].sum())})  # fmt: skip
        # experiment 637's detector on the bank's contaminated power, and its flags on the real days
        if month in PAIR:
            ev_rows = []
            has_ev = {(int(p_), int(rr)) for p_, rr in zip(b["prim"], b["pass_"], strict=True)}
            day_lo, day_hi = ms // PER - 20, ms // PER + 45
            if month == PAIR[0]:  # the windowed rule must reproduce the full-grid flags on uncontaminated power
                w = window_flags(clean_c=clean, ref_clim=ref_clim, day_lo=day_lo, day_hi=day_hi)
                full = flags_real[:, day_lo + 5 : day_hi - 5]
                assert np.array_equal(w[:, 5:-5], full), "windowed detector differs from the full-grid detector"
            for rr in range(bank.N_PASSES):
                fixed, _ = bank.contaminate(clean=clean, month_start_idx=ms, r=rr)
                fl = window_flags(clean_c=fixed, ref_clim=ref_clim, day_lo=day_lo, day_hi=day_hi)
                for p_ in range(16):
                    if (p_, rr) not in has_ev:
                        continue
                    sgn, ln, _ = bank.event(primary=p_, r=rr)
                    on_d, end_d = (end - ln * PER) // PER - day_lo, end // PER - day_lo
                    days = np.flatnonzero(fl[p_, on_d : end_d + 1])
                    ev_rows.append({"depth": abs(sgn), "length": ln, "detected": len(days) > 0, "days": float(days.min()) if len(days) else float("nan")})
            edf = pl.DataFrame(ev_rows)
            for (dp, ln), g in edf.group_by(["depth", "length"]):
                det_out.append({"variant": "n/a", "month": month, "seed": -1, "signal": "experiment 637", "what": f"bank recall depth {dp} length {ln}", "value": float(g["detected"].mean()), "n": g.height})  # fmt: skip
                det_out.append({"variant": "n/a", "month": month, "seed": -1, "signal": "experiment 637", "what": f"bank days to first alarm depth {dp} length {ln}", "value": float(g["days"].mean()) if g["detected"].any() else float("nan"), "n": int(g["detected"].sum())})  # fmt: skip
            if month == "2025-12":
                on_v = (datetime(2025, 12, 29, tzinfo=UTC) - k578.GRID_START).days
                off_v = (datetime(2026, 1, 13, tzinfo=UTC) - k578.GRID_START).days
                for p_ in (3, 11):
                    days = np.flatnonzero(flags_real[p_, on_v : off_v + 1])
                    det_out.append({"variant": "n/a", "month": month, "seed": -1, "signal": "experiment 637", "what": f"verified pair Primary {p_}: first flagged day after onset (days)", "value": float(days.min()) if len(days) else float("nan"), "n": int(len(days))})  # fmt: skip
            d0 = (dat.month_start_idx(month=month)) // PER
            m_days = np.arange(d0, d0 + 31)
            efd = flags_real[:, m_days] & True
            det_out.append({"variant": "n/a", "month": month, "seed": -1, "signal": "experiment 637", "what": "flagged Primary-days in the month", "value": float(efd.sum()), "n": 16 * 31})  # fmt: skip

    # ------------------------------------------------------------------ tables
    # NRA integrity (N1, N2): the K1 rows the heads read are the stored ones
    for month in PAIR:
        full = c647.load_export(network="K1").filter(pl.col("month") == month, pl.col("lead_hours") >= 24)
        z = fit.load(month=month)["real"]
        idx = np.flatnonzero(z["usable"])
        stored = full.select(c647.QNAMES).to_numpy().astype(np.float32)
        assert stored.shape[0] == len(z["q"]), (stored.shape, len(z["q"]))
        assert sha(stored[idx]) == nra_hash[f"real_{month}"], "K1 rows changed"
        zb = np.load(c647.WORK / "k1_bank" / f"rows_{month}_r00.npz")
        bank_r0 = fit.load(month=month)["bank"]
        assert np.array_equal(zb["k1_inj"][zb["lead"] >= 24], bank_r0["q"][bank_r0["pass_"] == 0]), "bank K1 rows changed"
    print("N1/N2 integrity: the K1 rows read by every head equal the stored export and step-3 rows", flush=True)

    recs = []
    # per month and pooled
    seen = {}
    for k, s in per.items():
        if k[0] == "factor":
            continue
        variant, month, setname, cutname, arm, seed = k
        seen.setdefault((variant, setname, cutname, arm, seed), {})[month] = s
    for (variant, setname, cutname, arm, seed), bym in seen.items():
        for month, s in bym.items():
            recs.append({"variant": variant, "month": month, "set": setname, "cut": cutname, "arm": arm, "seed": seed, **finish(s)})
        if len(bym) == len(PAIR):
            tot = {kk: sum(bym[m_][kk] for m_ in bym) for kk in next(iter(bym.values()))}
            recs.append({"variant": variant, "month": "pair", "set": setname, "cut": cutname, "arm": arm, "seed": seed, **finish(tot)})
    tab = pl.DataFrame(recs)
    # seed mean: average the per-seed means of fitted arms
    metric_cols = [c for c in tab.columns if c not in ("variant", "month", "set", "cut", "arm", "seed", "rows")]
    fitted_t = tab.filter(pl.col("seed") >= 0).group_by("variant", "month", "set", "cut", "arm").agg(pl.col("rows").first(), *[pl.col(c).mean() for c in metric_cols]).with_columns(seed=pl.lit(-2, dtype=pl.Int64))
    final = pl.concat([tab, fitted_t.select(tab.columns)])
    final.write_csv(OUT / "scorecard.csv")
    pl.DataFrame(det_out).write_csv(OUT / "detector.csv")
    factors = {str(k[1:]): v for k, v in per.items() if k[0] == "factor"}
    (OUT / "widen_factors.txt").write_text("\n".join(f"{k}: {v:.3f}" for k, v in factors.items()))
    gate(final)


def gate(final: pl.DataFrame) -> None:
    f = final.filter((pl.col("variant") == "lomo") & (pl.col("month") == "pair") & (pl.col("cut") == "all") & ((pl.col("seed") == -1) | (pl.col("seed") == -2)))

    def crps(arm: str, setname: str) -> float:
        r = f.filter((pl.col("arm") == arm) & (pl.col("set") == setname))
        return float(r["crps"][0]) if r.height else float("nan")

    base = ("R0_72", "R0_168", "R0s_72", "R0s_168")
    lines = ["FIRST-PAIR FEASIBILITY GATE (leave-one-month-out, pooled 2025-10 and 2025-12, seed-mean CRPS, points of capacity)"]
    sets = ("P2 inside shift", "P4 flagged or verified")
    lines.append(f"{'arm':10s} " + " ".join(f"{s[:22]:>24s}" for s in sets))
    for arm in ("K1", *base, "H1", "H2", "H1p", "H2p", "AG", "AS", "H1short", "H1long", "oracle"):
        lines.append(f"{arm:10s} " + " ".join(f"{crps(arm, s):24.3f}" for s in sets))
    passed = []
    for h in ("H1", "H2"):
        for label, bl in (("every R0 and R0* variant", base), ("main R0 only", base[:2])):
            ok = all(crps(h, s) < min(crps(b_, s) for b_ in bl) for s in sets)
            lines.append(f"{h} below {label} on P2 and P4: {ok}")
            if bl is base:
                passed.append(ok)
    lines.append("GATE: " + ("PASS (a head beats the best of R0 and R0* on both P2 and P4)" if any(passed) else "FAIL (neither H1 nor H2 beats the best of R0 and R0* on both P2 and P4)"))
    text = "\n".join(lines)
    (OUT / "gate.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        gate(pl.read_csv(OUT / "scorecard.csv"))
    else:
        main()

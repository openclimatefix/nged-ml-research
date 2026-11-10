# ruff: noqa: D103, E501, PLR0912, PLR0915, PLC0206
"""Experiment 681, step 3: score the arms (CPU, no fits).

For each arm the prevailing quantiles of the six development folds (2026-01 to 2026-06) are read
from `fits/<arm>/` (seed mean, then sorted). The normal-running (NRA) quantiles are experiment
667's stored Candidate quantiles in the head data, the same for every arm.

Writes `score/summary.json` and `score/reaction.parquet` under the experiment's data folder and
prints the tables the write-up quotes. Reaction time uses origins at 09:00 (daily), so it resolves
to 24 hours: the first origin at or after the event's start (end) at which the mean of the
prevailing minus the NRA median over the next 24 hours of valid times (leads 3 h to below 27 h)
is at least half of the transfer in the right direction (is below half of the transfer).

Usage: python e00681_score.py ARM [ARM ...]
"""

import json
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(next(Path(__file__).resolve().parents[1].glob("00667-*/prevailing"))))

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00668_slices_primaries as sl  # ty: ignore[unresolved-import]
import numpy as np
import p667_data as dat  # ty: ignore[unresolved-import]
import polars as pl

DATA: Final = dat.ROOT.parent / "00681-extra-lagged-power-inputs-for-the-prevailing-forecast"
MONTHS: Final = ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06")
SEEDS: Final = (0, 1)
TAU: Final = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99], np.float32)
BANDS: Final = ((0.0, 24.0), (24.0, 72.0), (72.0, 168.0), (168.0, 339.0))
BAND_NAMES: Final = ("days 0-1", "days 1-3", "days 3-7", "days 7-14")
DAY: Final = 48
FALSE_ALARM: Final = 0.03
"""A departure of the prevailing median from the NRA median of 3 points of capacity, as a fraction."""
VERIFIED_MW: Final = {"D5": -1.47, "D16": 1.37}
"""Experiment 580's fitted transfer at the two Primaries (megawatts), from experiment 668."""


def crps_rows(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the per-row CRPS (twice the mean pinball over the 11 quantiles) in points of capacity."""
    e = y[:, None] - q
    return (2.0 * np.maximum(TAU * e, (TAU - 1.0) * e).mean(1) * 100.0).astype(np.float32)


def seed_mean(*, arm: str, month: str, key: str, name: str) -> np.ndarray:
    parts = [np.load(DATA / "fits" / arm / name.format(month=month, s=s))[key] for s in SEEDS]
    return np.sort(np.mean(parts, axis=0), axis=1).astype(np.float32)


def origin_departure(*, z: dict, q_prev: np.ndarray, q_nra: np.ndarray) -> np.ndarray:
    """Return, per origin, the mean of (prevailing - NRA) median over leads 3 h to below 27 h."""
    n_org = len(z["org_prim"])
    use = (z["lead"] >= 3.0) & (z["lead"] < 27.0)
    d = (q_prev[:, 5] - q_nra[:, 5]).astype(np.float64)
    s = np.bincount(z["org"][use], weights=d[use], minlength=n_org)
    c = np.bincount(z["org"][use], minlength=n_org)
    return np.where(c >= 40, s / np.maximum(c, 1), np.nan)


def reaction(
    *, o: np.ndarray, d: np.ndarray, onset: int, end: int, transfer: float
) -> tuple[float, float]:
    """Return (hours from the start to the first reacting origin, hours from the end to the first agreeing origin); NaN if none."""
    half = 0.5 * abs(transfer)
    sgn = np.sign(transfer)
    ok = np.isfinite(d)
    react = ok & (o >= onset) & (o < end) & (sgn * d >= half)
    start = float((o[react].min() - onset) / 2.0) if react.any() else np.nan
    agree = ok & (o >= end) & (np.abs(d) < half)
    stop = float((o[agree].min() - end) / 2.0) if agree.any() else np.nan
    return start, stop


def real_events() -> dict[str, list[tuple[int, int, int, float | None]]]:
    """Return the real events as (primary, start, end, transfer fraction or None) by set name."""
    sm = k578.sites()
    prim_of = dict(zip(sm["label"].to_list(), sm["primary"].to_list(), strict=True))
    cap = dict(zip(sm["label"].to_list(), sm["capacity_mw"].to_list(), strict=True))
    site = pl.read_csv(next(sl.R.glob("00521-*/site_map.csv")))
    sets = sl.event_sets(site=site)
    out: dict[str, list] = {}
    for name in ("verified transfer", "event table, over 7 days, 2 units or more"):
        rows = []
        for e in sets[name].iter_rows(named=True):
            tr = VERIFIED_MW.get(e["label"]) if name == "verified transfer" else None
            rows.append(
                (
                    prim_of[e["label"]],
                    e["start"],
                    e["end"],
                    None if tr is None else tr / cap[e["label"]],
                )
            )
        out[name] = rows
    return out


def summarise(*, arm: str) -> tuple[dict, list[dict]]:
    events = real_events()
    all_ev = [e for rows in events.values() for e in rows]
    res: dict = {"arm": arm}
    react_rows: list[dict] = []
    acc: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "crps_p",
            "crps_n",
            "mae_p",
            "mae_n",
            "cov_p",
            "cov_n",
            "lead",
            "inside",
            "free14",
            "alarm",
            "prim",
            "vi",
        )
    }
    tail_hits = {t: [] for t in (0, 1, 9, 10)}
    tail_pin = {t: [] for t in (0, 1, 9, 10)}
    bank_acc: dict[str, list[np.ndarray]] = {k: [] for k in ("crps", "state", "lead")}
    for month in MONTHS:
        z = dict(np.load(dat.HEAD_DATA / f"real_{month}.npz"))
        q_nra = np.sort(z["q"], axis=1)
        q_prev = seed_mean(arm=arm, month=month, key="real_q", name="Rhaz_{month}_s{s}.npz")
        assert len(q_prev) == len(q_nra)
        ok = np.isfinite(z["y"]) & ~z["flagged"]
        y, prim, vi, oi = (
            z["y"],
            z["prim"].astype(np.int64),
            z["vi"].astype(np.int64),
            z["oi"].astype(np.int64),
        )
        inside = np.zeros(len(y), bool)
        near = np.zeros(len(y), bool)
        for p, a, b, _tr in all_ev:
            own = prim == p
            inside |= own & (vi >= a) & (vi < b)
            near |= own & (
                (vi >= a - 14 * DAY) & (vi < b + 14 * DAY)
                | (oi >= a - 14 * DAY) & (oi < b + 14 * DAY)
            )
        acc["crps_p"].append(crps_rows(y=y, q=q_prev)[ok])
        acc["crps_n"].append(crps_rows(y=y, q=q_nra)[ok])
        acc["mae_p"].append((np.abs(q_prev[:, 5] - y) * 100.0)[ok])
        acc["mae_n"].append((np.abs(q_nra[:, 5] - y) * 100.0)[ok])
        acc["cov_p"].append(((y >= q_prev[:, 1]) & (y <= q_prev[:, 9]))[ok])
        acc["cov_n"].append(((y >= q_nra[:, 1]) & (y <= q_nra[:, 9]))[ok])
        acc["lead"].append(z["lead"][ok])
        acc["inside"].append(inside[ok])
        acc["free14"].append(~near[ok])
        acc["alarm"].append((np.abs(q_prev[:, 5] - q_nra[:, 5]) >= FALSE_ALARM)[ok])
        acc["prim"].append(prim[ok])
        acc["vi"].append(vi[ok])
        for t in tail_hits:
            e = y[ok] - q_prev[ok, t]
            tail_hits[t].append(y[ok] < q_prev[ok, t])
            tail_pin[t].append(np.maximum(TAU[t] * e, (TAU[t] - 1.0) * e) * 100.0)
        # reaction on real events
        d = origin_departure(z=z, q_prev=q_prev, q_nra=q_nra)
        o_prim, o_oi = z["org_prim"].astype(np.int64), z["org_oi"].astype(np.int64)
        for name, rows in events.items():
            for p, a, b, tr in rows:
                if (
                    tr is None
                ):  # estimate the transfer from the NRA error over the event, in the months scored
                    continue
                sel = o_prim == p
                if sel.sum() == 0 or not ((o_oi[sel].min() <= b) and (o_oi[sel].max() >= a)):
                    continue
                s, e_ = reaction(o=o_oi[sel], d=d[sel], onset=a, end=b, transfer=tr)
                react_rows.append(
                    {
                        "arm": arm,
                        "set": name,
                        "primary": p,
                        "month": month,
                        "kind": "real",
                        "start_h": s,
                        "end_h": e_,
                    }
                )
        # bank rows: injected events
        b = dict(np.load(dat.HEAD_DATA / f"bank_{month}.npz"))
        bq_nra = np.sort(b["q"], axis=1)
        bq_prev = seed_mean(arm=arm, month=month, key="bank_q", name="bank_{month}_s{s}.npz")
        ev = pl.read_parquet(dat.HEAD_DATA / f"events_{month}.parquet")
        on = np.zeros((12, 16), np.int64)
        en = np.zeros((12, 16), np.int64)
        sg = np.zeros((12, 16))
        for r in ev.iter_rows(named=True):
            on[r["pass"], r["primary"]], en[r["pass"], r["primary"]] = r["onset_idx"], r["end_idx"]
            sg[r["pass"], r["primary"]] = r["sign"] * r["depth"]
        pa, pr = b["pass_"].astype(np.int64), b["prim"].astype(np.int64)
        boi = b["oi"].astype(np.int64)
        o_on, o_en = on[pa, pr], en[pa, pr]
        has = o_en > o_on
        state = np.zeros(
            len(pa), np.int8
        )  # 0 event-free or before onset, 1 in force, 2 near the end, 3 ended in the last 24 h
        state[has & (boi >= o_on) & (boi < o_en)] = 1
        state[has & (boi >= o_en - 3 * DAY) & (boi < o_en)] = 2
        state[has & (boi >= o_en) & (boi < o_en + DAY)] = 3
        target = b["y"] + b["shift"]
        okb = np.isfinite(target)
        bank_acc["crps"].append(crps_rows(y=target, q=bq_prev)[okb])
        bank_acc["state"].append(state[okb])
        bank_acc["lead"].append(b["lead"][okb])
        db = origin_departure(z=b, q_prev=bq_prev, q_nra=bq_nra)
        bo_prim, bo_oi, bo_pass = (
            b[k].astype(np.int64) for k in ("org_prim", "org_oi", "org_pass")
        )
        for r in range(12):
            for p in range(16):
                if en[r, p] <= on[r, p]:
                    continue
                sel = (bo_prim == p) & (bo_pass == r)
                s, e_ = reaction(
                    o=bo_oi[sel],
                    d=db[sel],
                    onset=int(on[r, p]),
                    end=int(en[r, p]),
                    transfer=float(sg[r, p]),
                )
                react_rows.append(
                    {
                        "arm": arm,
                        "set": "injected",
                        "primary": p,
                        "month": month,
                        "kind": "injected",
                        "pass": r,
                        "start_h": s,
                        "end_h": e_,
                    }
                )
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    lead = cat["lead"]
    strata = {
        "all rows": np.ones(len(lead), bool),
        "inside events": cat["inside"],
        "event-free, >14 days from any event": cat["free14"],
    }
    table = {}
    for sname, m in strata.items():
        for bi, (lo, hi) in enumerate(BANDS):
            mm = m & (lead >= lo) & (lead < hi)
            table[f"{sname} | {BAND_NAMES[bi]}"] = {
                "n": int(mm.sum()),
                **{
                    f"{o}_{k}": float(cat[f"{k}_{o[0]}"][mm].mean()) if mm.any() else None
                    for o in ("prevailing", "nra")
                    for k in ("crps", "mae", "cov")
                },
            }
    free = cat["free14"]
    res["table"] = table
    res["false_alarm_share_event_free"] = float(cat["alarm"][free].mean())
    res["event_free_crps"] = float(cat["crps_p"][free].mean())
    res["all_crps"] = float(cat["crps_p"].mean())
    res["nra_all_crps"] = float(cat["crps_n"].mean())
    res["tails"] = {
        f"q{TAU[t]:.2f}": {
            "coverage_below": float(np.concatenate(tail_hits[t]).mean()),
            "pinball": float(np.concatenate(tail_pin[t]).mean()),
        }
        for t in tail_hits
    }
    bc, bs = np.concatenate(bank_acc["crps"]), np.concatenate(bank_acc["state"])
    res["bank_crps_by_state"] = {
        n: float(bc[bs == i].mean())
        for i, n in enumerate(
            ("before onset or none", "in force", "near the end", "ended in last 24 h")
        )
    }
    res["bank_state_counts"] = {
        n: int((bs == i).sum())
        for i, n in enumerate(
            ("before onset or none", "in force", "near the end", "ended in last 24 h")
        )
    }
    return res, react_rows


def main() -> None:
    arms = sys.argv[1:]
    out = DATA / "score"
    out.mkdir(exist_ok=True)
    summary, rows = {}, []
    for arm in arms:
        s, r = summarise(arm=arm)
        summary[arm] = s
        rows += r
        print(
            arm,
            "all CRPS",
            round(s["all_crps"], 4),
            "event-free",
            round(s["event_free_crps"], 4),
            "false alarms",
            round(s["false_alarm_share_event_free"], 4),
            flush=True,
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    pl.DataFrame(rows).write_parquet(out / "reaction.parquet")


if __name__ == "__main__":
    main()

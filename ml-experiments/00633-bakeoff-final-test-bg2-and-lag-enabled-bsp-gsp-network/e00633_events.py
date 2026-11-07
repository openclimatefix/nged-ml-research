# ruff: noqa: D103, E501, PLR0912, PLR0915
"""Experiment 633: BG2 on the event-free rows of experiment 637's detector, and the follow penalty (CPU).

Re-scores stored forecasts only; nothing is refitted and no forecast is regenerated. The six
systems' 11 quantiles are rebuilt exactly as `e00633_score.py` builds them (experiment 620's export
for the three networks, `preds/bg2_single_fit` for BG2, the heuristic and the climatology from the
cleaned power), and checked against `score/rows.npz`.

The detector is experiment 637's frozen day detector, copied without change from
`ml-experiments/00637-*/e00637_detector.py` (that experiment's folder is not on this branch), reading
experiment 637's `ara_rule.json`, whose SHA-256 is verified against the two hashes the page states.

Three outputs:

1. Event-free scoring: CRPS, MAE and q05-q95 coverage of the six systems on the rows that the
   detector does not flag, beside all rows.
2. Follow penalty on the flagged rows. The normal running arrangement (NRA) is *estimated* for each
   flagged Primary as the median power, by half-hour of day and weekday or weekend, over the 28 days
   before the Primary's first flagged day. Follow ratio, NRA-side coverage, and safe-side share are
   defined as in experiments 586 and 613 (`follow_q50`, `nra_side95`), with the estimated NRA in
   place of the true NRA and the observed shift (observed minus estimated NRA) in place of the
   injected depth.
3. A daily view of the D18 episode for a figure.

Usage: E00633_PHASE=predict gpu.sh e00633_events.py RULE_JSON OUT_DIR
       gpu.sh e00633_events.py figures OUT_DIR DOCS_DIR   (after the first command)
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00583_common as c  # ty: ignore[unresolved-import]
import e00633_common as k633
import e00633_score as s
import numpy as np
import polars as pl

PER_DAY: Final[int] = 48
CLIM_CUTOFF: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
DEV_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
DEV_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
WINDOW_END: Final[datetime] = datetime(2026, 11, 1, tzinfo=UTC)
RULE_HASHES: Final[tuple[str, ...]] = (
    "e5fb5cc08a67327d74dbc93ee51ba4e31dfc25bb25d8d4fab426bb32762b0566",
    "e1c8df968b2da090b7c959077e38894d621a437796058e7cac4f0b33272280e5",
)
"""The hash on the page of experiment 637, and the hash after the committed file gained a trailing newline."""
SYSTEMS: Final[tuple[str, ...]] = s.SYSTEMS
FOLLOW_SYSTEMS: Final[tuple[str, ...]] = SYSTEMS
NRA_PLAIN: Final[str] = "28-day profile"
NRA_DRIFT: Final[str] = "28-day profile plus drift of the other Primaries"
NRA_DAYS: Final[int] = 28
PHASES: Final[dict[str, tuple[int, int]]] = {
    "all flagged days": (0, 10_000),
    "days 0 to 6 of the episode": (0, 7),
    "day 7 on": (7, 10_000),
}
LEADSETS: Final[dict[str, tuple[float, float]]] = {
    "all leads": (0.0, 1e9),
    "lead 24 to 120 h": (24.0, 120.0),
}


# ------------------------------------------------------------------ experiment 637's detector (copied)


def day_of(when: datetime) -> int:
    return int((when - s.START_US_DT) / (s.STEP_DT * PER_DAY))


def _flat(*, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = clean.shape[1] // PER_DAY * PER_DAY
    return np.repeat(np.arange(clean.shape[0]), n), np.tile(np.arange(n), clean.shape[0])


def climatology_reference(*, clean: np.ndarray) -> np.ndarray:
    prim, idx = _flat(clean=clean)
    cut = int((CLIM_CUTOFF - s.START_US_DT) / s.STEP_DT)
    ref = s.climatology(clean=clean, prim=prim, idx=idx, cutoff_idx=cut)[:, 5]
    return ref.reshape(clean.shape[0], -1)


def daily_residuals(*, clean: np.ndarray, ref_clim: np.ndarray) -> np.ndarray:
    prim, idx = _flat(clean=clean)
    ref_h = s.manual_heuristic(clean=clean, prim=prim, idx=idx, lead_h=np.zeros(len(idx)))[:, 5]
    ref_h = ref_h.reshape(clean.shape[0], -1)
    obs = clean[:, : ref_h.shape[1]]
    out = []
    for ref in (ref_h, ref_clim):
        with np.errstate(all="ignore"):
            out.append(np.nanmean((obs - ref).reshape(clean.shape[0], -1, PER_DAY), axis=2))
    return np.stack(out).astype(np.float32)


def combine(*, daily: np.ndarray) -> np.ndarray:
    centred = daily - np.nanmedian(daily, axis=1, keepdims=True)
    return np.nanmean(centred, axis=0)


def flag_days(
    *, resid: np.ndarray, centre: np.ndarray, mad: np.ndarray, k: float, n: int
) -> np.ndarray:
    z = (resid - centre[:, None]) / mad[:, None]
    flag = np.nan_to_num(np.abs(z), nan=0.0) > k
    keep = np.zeros_like(flag)
    for p in range(flag.shape[0]):
        f = flag[p].copy()
        f[1:-1] |= f[:-2] & f[2:]
        i = 0
        while i < len(f):
            if f[i]:
                j = i
                while j < len(f) and f[j]:
                    j += 1
                if j - i >= n:
                    keep[p, i:j] = True
                i = j
            else:
                i += 1
    return keep


# ------------------------------------------------------------------ rebuilding the six systems' quantiles


def build() -> dict:
    """Return the common rows and the 11 quantiles (shares of capacity) of the six systems."""
    export = next(k633.ROOT.parent.glob("00620-*")) / "export"  # read only
    sub = pl.read_csv(s.SUBSTATIONS)
    cap = dict(zip(sub["time_series_id"].to_list(), sub["capacity"].to_list(), strict=True))
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    clean = c.k578.clean()
    cutoff_idx = int((k633.CUTOFF - s.START_US_DT) / s.STEP_DT)
    frames = {n: pl.read_parquet(export / f"{n}_2026-07.parquet") for n in s.NETWORKS}
    base = frames["K1"].select(*s.KEYS, "lead_hours", "observed_power", "unit")
    tid = base["time_series_id"].to_numpy()
    prim = np.array([prim_of[int(t)] for t in tid])
    capv = np.array([cap[int(t)] for t in tid])
    idx = (
        ((base["valid_time"].dt.epoch("us") - s.START_US) // s.STEP_US).to_numpy().astype(np.int64)
    )
    oidx = (
        ((base["power_fcst_init_time"].dt.epoch("us") - s.START_US) // s.STEP_US)
        .to_numpy()
        .astype(np.int64)
    )
    lead = base["lead_hours"].to_numpy().astype(np.float64)
    y = base["observed_power"].to_numpy().astype(np.float64) / capv
    q = {
        n: np.sort(frames[n].select(s.QCOLS).to_numpy().astype(np.float64) / capv[:, None], axis=1)
        for n in s.NETWORKS
    }
    q["Manual heuristic"] = s.manual_heuristic(clean=clean, prim=prim, idx=idx, lead_h=lead)
    q["Climatology"] = s.climatology(clean=clean, prim=prim, idx=idx, cutoff_idx=cutoff_idx)
    q["BG2"] = s.bg2_quantiles(base=base, cap=cap)
    ok = np.isfinite(y)
    for n in SYSTEMS:
        ok &= np.isfinite(q[n]).all(axis=1)
    sel = np.flatnonzero(ok)
    name = dict(zip(sub["time_series_id"].to_list(), sub["primary_name"].to_list(), strict=True))
    labels = {p: next(name[t] for t in prim_of if prim_of[t] == p) for p in range(16)}
    return {
        "clean": clean,
        "prim": prim[sel],
        "idx": idx[sel],
        "oidx": oidx[sel],
        "lead": lead[sel],
        "y": y[sel],
        "q": {n: v[sel].astype(np.float32) for n, v in q.items()},
        "labels": [labels[p] for p in range(16)],
    }


# ------------------------------------------------------------------ helpers


def cover(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    return ((q[:, 1] <= y) & (y <= q[:, 9])).astype(float)


def summary(*, day: np.ndarray, vals: np.ndarray) -> tuple[float, float, float]:
    mean, lo, hi, _ = s.day_boot(day=day, vals=vals)
    return mean, lo, hi


def nra_profile(*, clean: np.ndarray, p: int, first_day: int) -> np.ndarray:
    """Return the (2, 48) median power by weekday-or-weekend and UTC half-hour over the 28 days before `first_day`."""
    d0 = first_day - NRA_DAYS
    block = clean[p, d0 * PER_DAY : first_day * PER_DAY].reshape(NRA_DAYS, PER_DAY)
    days = np.arange(d0, first_day)
    weekend = ((6 + days) % 7) >= 5  # the grid starts on a Sunday
    out = np.full((2, PER_DAY), np.nan)
    for w in (0, 1):
        with np.errstate(all="ignore"):
            out[w] = np.nanmedian(block[weekend == bool(w)], axis=0)
    return out


def main(rule_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    k633.redirect()
    digest = hashlib.sha256(rule_path.read_bytes()).hexdigest()
    assert digest in RULE_HASHES, digest
    rule = json.loads(rule_path.read_text())
    print("rule SHA-256", digest, "(matches the page)", flush=True)
    d = build()
    clean, prim, idx, oidx, lead, y, q = (
        d["clean"], d["prim"], d["idx"], d["oidx"], d["lead"], d["y"], d["q"],
    )  # fmt: skip
    labels = d["labels"]
    day = oidx // PER_DAY
    vday = idx // PER_DAY
    # check against the Part 1 scoring
    z_ = np.load(k633.ROOT / "score" / "rows.npz")
    assert np.array_equal(z_["prim"], prim), "row order differs"
    assert np.array_equal(z_["idx"], idx), "row order differs"
    for n in SYSTEMS:
        crps = 2.0 * s.pinball(y=y, q=q[n].astype(np.float64)).mean(axis=1) * 100
        assert np.allclose(crps, z_[f"crps_{n}"], atol=2e-3), (n, float(np.abs(crps - z_[f"crps_{n}"]).max()))  # fmt: skip
    print("rebuilt quantiles reproduce score/rows.npz", flush=True)
    # ---- the frozen detector at the operating point
    resid = combine(daily=daily_residuals(clean=clean, ref_clim=climatology_reference(clean=clean)))
    centre, mad = np.array(rule["centre"]), np.array(rule["mad"])
    start, end = day_of(DEV_END), day_of(WINDOW_END)
    z0 = start - 30
    flag_end = int(vday.max()) + 1
    sel = rule["operating"]
    keep = flag_days(resid=resid[:, z0:flag_end], centre=centre, mad=mad, k=sel["k"], n=sel["n"])
    full = np.zeros(resid.shape, bool)
    full[:, z0:flag_end] = keep
    flagged = full[prim, vday]
    census = {
        "rows": len(flagged),
        "rows_removed": int(flagged.sum()),
        "share_rows": float(flagged.mean()),
        "primary_days": int(full[:, start:end].sum()),
    }
    print("census", census, flush=True)
    assert census["rows_removed"] == 56_699, census
    assert census["primary_days"] == 119, census
    pl.DataFrame([census]).write_csv(out / "census.csv")
    by_p = pl.DataFrame(
        [{"primary": labels[p], "flagged_days": int(full[p, start:end].sum())} for p in range(16)]
    )
    by_p.write_csv(out / "flagged_days_by_primary.csv")
    # ---- 1. event-free scoring
    free = ~flagged
    readings = {"All rows": np.ones(len(y), bool), "Event-free rows": free}
    stats = {}
    for n in SYSTEMS:
        qn = q[n].astype(np.float64)
        stats[n] = {
            "crps": z_[f"crps_{n}"].astype(float),
            "mae": np.abs(y - qn[:, 5]) * 100,
            "cover90": cover(y=y, q=qn),
        }
    rows = []
    for rname, m in readings.items():
        for n in SYSTEMS:
            rec: dict[str, object] = {"reading": rname, "system": n, "rows": int(m.sum())}
            for meas, v in stats[n].items():
                mean, lo, hi = summary(day=day[m], vals=v[m])
                rec |= {meas: mean, f"{meas}_lo": lo, f"{meas}_hi": hi}
            rows.append(rec)
    scores = pl.DataFrame(rows)
    scores.write_csv(out / "event_free_scores.csv")
    print(scores.select("reading", "system", "rows", "crps", "mae", "cover90"), flush=True)
    pairs = [(n, "Manual heuristic") for n in SYSTEMS if n != "Manual heuristic"]
    pairs += [("BG2", "K1"), ("BG2", "Candidate"), ("BG2", "K3")]
    alpha = 0.05 / len(pairs)
    print("Bonferroni count (CRPS contrasts)", len(pairs), flush=True)
    rows = []
    for rname, m in readings.items():
        for a_, b_ in pairs:
            mean, lo, hi, draws = s.day_boot(
                day=day[m], vals=stats[a_]["crps"][m] - stats[b_]["crps"][m]
            )
            blo, bhi = s.interval(draws, alpha)
            rows.append({"reading": rname, "a": a_, "b": b_, "crps_a_minus_b": mean, "lo95": lo, "hi95": hi, "lo_bonf": blo, "hi_bonf": bhi, "n_tests": len(pairs)})  # fmt: skip
    pl.DataFrame(rows).write_csv(out / "event_free_contrasts.csv")
    rows = []
    for rname, m in readings.items():
        for p in range(16):
            mp = m & (prim == p)
            if not mp.any():
                continue
            rows.append({"reading": rname, "primary": labels[p], "rows": int(mp.sum()), **{n: float(stats[n]["crps"][mp].mean()) for n in SYSTEMS}})  # fmt: skip
    per_p = pl.DataFrame(rows)
    per_p.write_csv(out / "event_free_by_primary.csv")
    wins = []
    for rname in readings:
        sub = per_p.filter(pl.col("reading") == rname)
        for n in SYSTEMS:
            if n == "Manual heuristic":
                continue
            wins.append({"reading": rname, "system": n, "primaries_scored": sub.height, "lower_crps_than_heuristic": int((sub[n] < sub["Manual heuristic"]).sum())})  # fmt: skip
    pl.DataFrame(wins).write_csv(out / "event_free_wins.csv")
    print(pl.DataFrame(wins), flush=True)
    # ---- 2. follow penalty on the flagged rows
    first = {}
    for p in range(16):
        days_ = np.flatnonzero(full[p, start:end]) + start
        if len(days_):
            first[p] = int(days_[0])
    n_days_all = clean.shape[1] // PER_DAY
    daily_mean = np.nanmean(
        clean[:, : n_days_all * PER_DAY].reshape(16, n_days_all, PER_DAY), axis=2
    )
    profiles = {p: nra_profile(clean=clean, p=p, first_day=f) for p, f in first.items()}
    drift = {}  # per flagged Primary: (n_days,) cross-Primary median change of the daily mean since the baseline
    for p, f in first.items():
        base_mean = np.nanmean(clean[:, (f - NRA_DAYS) * PER_DAY : f * PER_DAY], axis=1)
        chg = daily_mean - base_mean[:, None]
        chg[p] = np.nan
        chg[full[:, :n_days_all]] = np.nan  # leave out Primaries flagged that day
        with np.errstate(all="ignore"):
            drift[p] = np.nanmedian(chg, axis=0)
    variants: dict[str, np.ndarray] = {}
    for vname in (NRA_PLAIN, NRA_DRIFT):
        nra_rows = np.full(len(y), np.nan)
        for p in first:
            mp = flagged & (prim == p)
            wk = ((6 + vday[mp]) % 7) >= 5
            nra_rows[mp] = profiles[p][wk.astype(int), idx[mp] % PER_DAY]
            if vname == NRA_DRIFT:
                nra_rows[mp] += drift[p][vday[mp]]
        variants[vname] = nra_rows
    row_first = np.array([first.get(int(p), -1) for p in prim])
    since = vday - row_first
    nra_info = []
    rows = []
    for vname, nra_rows in variants.items():
        depth, sgn = {}, {}
        for p, f in first.items():
            mp = flagged & (prim == p)
            depth[p] = float(np.nanmean(y[mp] - nra_rows[mp]))
            sgn[p] = 1.0 if depth[p] >= 0 else -1.0
            nra_info.append(
                {
                    "nra_estimate": vname,
                    "primary": labels[p],
                    "first_flagged_day": str((s.START_US_DT + f * s.STEP_DT * PER_DAY).date()),
                    "flagged_days": int(full[p, start:end].sum()),
                    "flagged_rows": int(mp.sum()),
                    "nra_mean_pct_capacity": float(np.nanmean(nra_rows[mp])) * 100,
                    "shift_mean_pct_capacity": depth[p] * 100,
                    "shift_robust_sd": depth[p] / float(mad[p]),
                }
            )
        row_depth = np.array([depth.get(int(p), np.nan) for p in prim])
        row_sgn = np.array([sgn.get(int(p), np.nan) for p in prim])
        groups = {
            "D18": [labels.index("D18")],
            "D14": [labels.index("D14")],
            "All flagged Primaries": list(first),
            "Flagged Primaries other than D18": [p for p in first if labels[p] != "D18"],
        }
        for gname, plist in groups.items():
            gm = flagged & np.isin(prim, plist)
            for pname, (lo_d, hi_d) in PHASES.items():
                for lname, (lo_l, hi_l) in LEADSETS.items():
                    m = gm & (since >= lo_d) & (since < hi_d) & (lead >= lo_l) & (lead < hi_l)
                    if m.sum() < 48:
                        continue
                    nra = nra_rows[m]
                    sg = row_sgn[m]
                    dep = np.abs(row_depth[m])
                    follow_all = {
                        n: (q[n][m][:, 5].astype(np.float64) - nra) * sg / dep
                        for n in FOLLOW_SYSTEMS
                    }
                    for n in FOLLOW_SYSTEMS:
                        qn = q[n][m].astype(np.float64)
                        follow_v = follow_all[n]
                        cov_v = ((qn[:, 1] <= nra) & (nra <= qn[:, 9])).astype(float)
                        safe_v = np.where(sg > 0, nra >= qn[:, 1], nra <= qn[:, 9]).astype(float)
                        rec = {
                            "nra_estimate": vname,
                            "group": gname,
                            "phase": pname,
                            "leads": lname,
                            "system": n,
                            "rows": int(m.sum()),
                            "days": len(np.unique(day[m])),
                            "shift_pct_capacity": float(np.mean(sg * dep)) * 100,
                        }
                        for meas, v in (
                            ("follow", follow_v),
                            ("nra_cover", cov_v),
                            ("safe_side", safe_v),
                        ):
                            mean, lo, hi = summary(day=day[m], vals=v)
                            rec |= {meas: mean, f"{meas}_lo": lo, f"{meas}_hi": hi}
                        # the lag-free network K3 is the control: it cannot follow a load-shift through lags
                        mean, lo, hi = summary(day=day[m], vals=follow_v - follow_all["K3"])
                        rec |= {
                            "follow_minus_k3": mean,
                            "follow_minus_k3_lo": lo,
                            "follow_minus_k3_hi": hi,
                        }
                        rows.append(rec)
    pl.DataFrame(nra_info).write_csv(out / "nra_estimates.csv")
    print(pl.DataFrame(nra_info), flush=True)
    follow = pl.DataFrame(rows)
    follow.write_csv(out / "follow_penalty.csv")
    # ---- 3. daily view of D18 for the figure
    p18 = labels.index("D18")
    rows = []
    for dd in range(start, end):
        m = (prim == p18) & (vday == dd) & (lead >= 24) & (lead < 48)
        wk = int(((6 + dd) % 7) >= 5)
        rec = {
            "day": str((s.START_US_DT + dd * s.STEP_DT * PER_DAY).date()),
            "day_index": dd - start,
            "flagged": bool(full[p18, dd]),
            "observed": float(daily_mean[p18, dd]) * 100,
            "nra_estimate": float(np.mean(profiles[p18][wk])) * 100,
            "nra_drift_adjusted": float(np.mean(profiles[p18][wk]) + drift[p18][dd]) * 100,
        }
        for n in SYSTEMS:
            rec[n] = float(q[n][m][:, 5].mean()) * 100 if m.sum() >= 40 else None
        rows.append(rec)
    pl.DataFrame(rows).write_csv(out / "d18_daily.csv")
    print("done", flush=True)


COLOURS: Final[dict[str, str]] = {
    "BG2": "#FF4901",
    "K1": "#306BFF",
    "Candidate": "#10C5F7",
    "K3": "#B701FF",
    "Manual heuristic": "#17E58F",
    "Climatology": "#7F7F7F",
}
WIDTH_PX: Final[int] = 620


def _colour(*, with_legend: bool = False) -> alt.Color:
    return alt.Color(
        "system:N",
        title=None,
        scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
        legend=alt.Legend(orient="top", columns=3) if with_legend else None,
    )


def figures(events: Path, docs: Path) -> None:
    """Draw the three figures of the draft section from the CSVs that `main` wrote."""
    docs.mkdir(parents=True, exist_ok=True)
    order = list(COLOURS)
    # figure 1: CRPS on all rows and on event-free rows
    sc = pl.read_csv(events / "event_free_scores.csv")
    panels = []
    for reading, label in (
        ("All rows", "All 929,870 rows"),
        ("Event-free rows", "Event-free rows: 873,171 rows, flagged days removed"),
    ):
        d = sc.filter(pl.col("reading") == reading).to_pandas()  # ty: ignore[unresolved-attribute]
        base = alt.Chart(d).encode(
            y=alt.Y("system:N", sort=order, title=None),
            color=_colour(),
        )
        panels.append(
            (
                base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(
                        "crps_lo:Q",
                        title="CRPS (% of capacity; smaller is better)",
                        scale=alt.Scale(zero=True),
                    ),
                    x2="crps_hi:Q",
                )
                + base.mark_point(filled=True, size=70, opacity=1).encode(x="crps:Q")  # ty: ignore[unresolved-attribute]
            ).properties(
                width=WIDTH_PX,
                height=150,
                title=alt.TitleParams(label, anchor="start", fontSize=12),
            )
        )
    fig = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Event-free rows lower every system's CRPS; BG2 stays lowest. Dot: mean CRPS over 16 Primaries,",
                "forecast origins 2026-07-01 to 2026-10-31. Line: 95% interval, resampling whole origin days.",
            ],
            anchor="start",
            fontSize=13,
        )
    )
    fig.save(docs / "events_crps_event_free.svg")
    # figure 2: follow ratio and safe-side share at D18
    fp = pl.read_csv(events / "follow_penalty.csv").filter(
        (pl.col("nra_estimate") == NRA_PLAIN)
        & (pl.col("group") == "D18")
        & (pl.col("phase") == "day 7 on")
        & (pl.col("leads") == "all leads")
    )
    d = fp.to_pandas()  # ty: ignore[unresolved-attribute]
    panels = []
    for col, lo, hi, axis_title, sub in (
        (
            "follow",
            "follow_lo",
            "follow_hi",
            "Share of the estimated transfer the median follows (0 = none; lower is better)",
            "Follow ratio",
        ),
        (
            "safe_side",
            "safe_side_lo",
            "safe_side_hi",
            "Share of rows whose q05 or q95 keeps the estimated normal level on the safe side (higher is better)",
            "Safe-side share",
        ),
    ):
        base = alt.Chart(d).encode(y=alt.Y("system:N", sort=order, title=None), color=_colour())
        panels.append(
            (
                base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(f"{lo}:Q", title=axis_title, scale=alt.Scale(zero=True)), x2=f"{hi}:Q"
                )
                + base.mark_point(filled=True, size=70, opacity=1).encode(x=f"{col}:Q")  # ty: ignore[unresolved-attribute]
            ).properties(
                width=WIDTH_PX, height=150, title=alt.TitleParams(sub, anchor="start", fontSize=12)
            )
        )
    fig = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "At Primary D18, BG2's median follows the load-shift more than the lag-free K3 and the",
                "climatology, which cannot follow it. Rows: flagged days from day 7 after the first one.",
                "Normal level (estimated): median power in the 28 days before the first flagged day.",
                "Dot: estimate. Line: 95% interval from resampling whole origin days.",
            ],
            anchor="start",
            fontSize=13,
        )
    )
    fig.save(docs / "events_follow_d18.svg")
    # figure 3: daily view of D18, two panels so each holds one cluster of near-identical lines
    dd = pl.read_csv(events / "d18_daily.csv")
    flagged_days = dd.filter(pl.col("flagged"))
    rule_day = "2026-07-13"  # day 7 after the first flagged day, 2026-07-06
    assert flagged_days["day"].min() == "2026-07-06"
    names = {"observed": "Observed", "nra_estimate": "Estimated normal level"}
    shared = ["Observed", "Estimated normal level"]
    panels = []
    for series, sub in (
        (["BG2", "K3"], "BG2 and K3 (K3 reads no lagged power)"),
        (["K1", "Candidate", "Manual heuristic"], "K1, Candidate, and Manual heuristic"),
    ):
        long = (
            dd.unpivot(
                index=["day", "flagged"],
                on=["observed", "nra_estimate", *series],
                variable_name="series",
                value_name="value",
            )
            .drop_nulls()
            .with_columns(system=pl.col("series").replace(names))
        )
        domain = [*shared, *series]
        colours = ["#000000", "#555555", *[COLOURS[s_] for s_ in series]]
        dashes = [[1, 0], [5, 3], *[[1, 0] for _ in series]]
        enc_x = alt.X(
            "day:T", title="Valid day (2026)", axis=alt.Axis(format="%b", tickCount="month")
        )
        line = (
            alt.Chart(long.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.8)
            .encode(
                x=enc_x,
                y=alt.Y(
                    "value:Q", title="Daily mean (% of capacity)", scale=alt.Scale(domain=[10, 110])
                ),
                color=alt.Color(
                    "system:N",
                    title=None,
                    scale=alt.Scale(domain=domain, range=colours),
                    legend=alt.Legend(
                        orient="top", columns=3, symbolType="stroke", symbolStrokeWidth=2
                    ),
                ),
                strokeDash=alt.StrokeDash(
                    "system:N", scale=alt.Scale(domain=domain, range=dashes), legend=None
                ),
            )
        )
        strip = (
            alt.Chart(flagged_days.select("day").to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_bar(color="#999999", opacity=0.35, width=3)
            .encode(x="day:T", y=alt.value(230), y2=alt.value(220))
        )
        rule = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                pl.DataFrame(
                    {"day": [rule_day], "label": ["Day 7 after the first flagged day (13 Jul)"]}
                ).to_pandas()
            )  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#333333", strokeDash=[2, 2])
            .encode(x="day:T")
        )
        text = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                pl.DataFrame(
                    {"day": [rule_day], "label": ["Day 7 after first flagged day"]}
                ).to_pandas()
            )  # ty: ignore[unresolved-attribute]
            .mark_text(align="left", dx=4, dy=-4, fontSize=11, baseline="bottom")
            .encode(x="day:T", y=alt.value(214), text="label:N")
        )
        panels.append(
            (strip + line + rule + text).properties(
                width=WIDTH_PX,
                height=230,
                title=alt.TitleParams(sub, anchor="start", fontSize=12),
            )
        )
    fig = (
        alt.vconcat(*panels)
        .resolve_scale(color="independent", strokeDash="independent")
        .properties(
            title=alt.TitleParams(
                [
                    "Primary D18, forecasts 24 to 48 hours ahead (daily mean of the median). Observed power ran",
                    "above the estimated normal level (median power in the 28 days before the first flagged day).",
                    "Grey strip at the bottom: days the detector flags. Climatology is not drawn.",
                ],
                anchor="start",
                fontSize=13,
            )
        )
    )
    fig.save(docs / "events_d18_daily.svg")


if __name__ == "__main__":
    if sys.argv[1] == "figures":
        figures(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))

# ruff: noqa: E501, PLR0915, PLR0912, PLR0913, D205, ANN001, ANN201
"""Experiment 583: score the after-snap-back rule and the event-mode band on stored X3-syn forecasts.

Reads experiment 581's stored X3-syn forecasts (the lag network, its lags-masked twin, and the
lag-only injected bank) and writes, under `data/ml-experiments/00583-*/`:

- `real_event_phase.csv`: the verified load-shift events of experiment 580, per event, phase, lead
  band, and arm.
- `bank_event_phase.csv`, `bank_summary.csv`: the lag-only injected bank, per event and summarised.
- `event_free.csv`: the false-alarm cost on rows with no event within 7 days.
- `gates.csv`, `trigger_times.csv`: how often the rule is active, false triggers per Primary-year,
  and the time from onset to trigger and from end to release.

No fit and no GPU. Phases are by valid time, as in experiment 582. The rule's gate reads days that
are at least 24 h old at the forecast origin (09:00 UTC daily).
"""

import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

IQ = c.IQ
PHASES = sc.PHASES
BANDS = sc.BANDS
GROUPS_SCORED = ("MVA-only eight", "signed-MW six")
BANK_PARQUET = next(
    (ML_EXPERIMENTS_DATA_DIR / "00581-robust-multiweek-lag-encoder-x3").glob(
        "bank_events_x3syn_*.parquet"
    )
)
N_COUNTED = 16
"""4 arms (A1 to A4) x 2 metering groups x 2 coverage bands (q05-q95 and q01-q99)."""
RNG = np.random.default_rng(583)
N_BOOT = 20000


# ---------------------------------------------------------------- helpers


def metric_means(*, rows_metrics: np.ndarray, y_sign: float | None) -> dict[str, float]:
    """Return the mean of every row-level metric and the NRA-side coverage of the event's sign.

    `y_sign` is +1 when the event raised the Primary's load (the lag network's forecast moves up, so
    the NRA lies on the lower side and the NRA-side coverage is the share of readings at or above
    q05 and q01) and -1 when it lowered it (the share at or below q95 and q99).
    """
    m = dict(zip(c.METRICS, rows_metrics.mean(axis=0).tolist(), strict=True))
    if y_sign is not None:
        m["nra_side95"] = m["ge05"] if y_sign > 0 else m["le95"]
        m["nra_side99"] = m["ge01"] if y_sign > 0 else m["le99"]
    return m


def arm_row_metrics(
    *,
    pre: dict[str, np.ndarray],
    rows: np.ndarray,
    hold: np.ndarray,
    on: np.ndarray,
    frac: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return every arm's row-level metrics on `rows` (per-row hold, on, and placebo share)."""
    return c.combine(
        base={k: v[rows] for k, v in pre.items()}, hold=hold[rows], on=on[rows], frac=frac[rows]
    )


def day_arrays(
    *, t: c.X4Table, on: np.ndarray, hold: np.ndarray, frac: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the per-row `hold`, `on`, and placebo share of the origin day of each row in `rows`."""
    od, p = t.origin_day[rows], t.prim[rows]
    return hold[p, od], on[p, od], frac[p, od]


def placebo_fraction(*, snap: np.ndarray, origin_days: list[int]) -> np.ndarray:
    """Return (16, days): the share of `N_PLACEBO` random placements whose hold covers the day."""
    acc = np.zeros(snap.shape, np.float32)
    for _ in range(c.N_PLACEBO):
        acc += c.hold_from(snap=c.placebo_snap(snap=snap, origin_days=origin_days, rng=RNG))
    return acc / c.N_PLACEBO


def block_blocks(
    *, t: c.X4Table, rows: np.ndarray, v: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Return the building-block row metrics of `rows`; `v` replaces the stored lag network."""
    v_q = t.arms["V"][rows] if v is None else v
    blocks = c.building_blocks(v=v_q, m=t.arms["M"][rows])
    y = t.nra[rows]
    return {k: c.row_metrics(y=y, q=q) for k, q in blocks.items()}


# ---------------------------------------------------------------- real events and event-free rows


def real_events(
    *, t: c.X4Table, pre: dict[str, np.ndarray], on: np.ndarray, hold: np.ndarray, frac: np.ndarray
) -> pl.DataFrame:
    """Return the per-event, phase, band, arm metrics of the verified events (experiment 580)."""
    ev = sc.attach_depth(ev=sc.merged_events(), t=t)
    hold_r, on_r, frac_r = day_arrays(t=t, on=on, hold=hold, frac=frac, rows=np.arange(t.n))
    rows_out = []
    for e in ev.iter_rows(named=True):
        sgn = 1.0 if e["depth_share"] > 0 else -1.0
        for phase in PHASES:
            lo, hi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
            for band in (0, 1):
                sel = np.flatnonzero(
                    (t.prim == e["primary"]) & (t.idx >= lo) & (t.idx < hi) & (t.band == band)
                )
                if len(sel) < 100:
                    continue
                am = c.combine(
                    base={k: v[sel] for k, v in pre.items()},
                    hold=hold_r[sel],
                    on=on_r[sel],
                    frac=frac_r[sel],
                )
                for arm, mm in am.items():
                    rows_out.append(
                        {
                            "event": e["name"], "group": e["group"], "label": e["label"],
                            "depth_share": e["depth_share"], "length_days": e["length_days"],
                            "phase": phase, "band": BANDS[band], "arm": arm, "n_rows": len(sel),
                            "share_origin_days_event_mode": float(hold_r[sel].mean()),
                            "share_origin_days_gate_on": float(on_r[sel].mean()),
                            **metric_means(rows_metrics=mm, y_sign=sgn),
                        }
                    )  # fmt: skip
    return pl.DataFrame(rows_out)


def event_free(
    *, t: c.X4Table, pre: dict[str, np.ndarray], on: np.ndarray, hold: np.ndarray, frac: np.ndarray
) -> pl.DataFrame:
    """Return the false-alarm cost on event-free rows, per group and lead band and arm, with the
    contrast against X3-syn's lag network in points of capacity: the mean pinball loss at q05 and
    q95 (the q05-q95 band) and at q01 and q99 (the q01-q99 band), month-block bootstrap interval
    at the Bonferroni level of `N_COUNTED` counted contrasts.
    """
    hold_r, on_r, frac_r = day_arrays(t=t, on=on, hold=hold, frac=frac, rows=np.arange(t.n))
    am = c.combine(base=pre, hold=hold_r, on=on_r, frac=frac_r)
    months = sorted(set(t.month))
    out = []
    for g in (*GROUPS_SCORED, "D15 and D18"):
        for band in (0, 1):
            base_sel = t.free & (t.group_of_row == g) & (t.band == band)
            if base_sel.sum() < 500:
                continue
            for arm, mm in am.items():
                r = {"group": g, "band": BANDS[band], "arm": arm, "n_rows": int(base_sel.sum())}
                for k, j in zip(c.METRICS, range(mm.shape[1]), strict=True):
                    scale = 100.0 if k.startswith(("pin", "width", "ae")) else 1.0
                    r[k] = float(mm[base_sel, j].mean() * scale)
                for name, cols in (("loss90", ("pin05", "pin95")), ("loss98", ("pin01", "pin99"))):
                    idx = [c.METRICS.index(k) for k in cols]
                    val = mm[:, idx].mean(axis=1) * 100.0
                    ref = am["V"][:, idx].mean(axis=1) * 100.0
                    diffs = np.array(
                        [
                            val[base_sel & (t.month == m)].mean()
                            - ref[base_sel & (t.month == m)].mean()
                            for m in months
                            if (base_sel & (t.month == m)).any()
                        ]
                    )
                    r[name] = float(val[base_sel].mean())
                    r[f"{name}_minus_V"] = float(val[base_sel].mean() - ref[base_sel].mean())
                    lo, hi = bonferroni_interval(diffs=diffs)
                    r[f"{name}_minus_V_lo"], r[f"{name}_minus_V_hi"] = lo, hi
                    r[f"{name}_mde"] = (hi - lo) / 2.0
                    r[f"{name}_month_signs"] = "".join(
                        "+" if d > 0 else "-" if d < 0 else "0" for d in diffs
                    )
                out.append(r)
    return pl.DataFrame(out)


def bonferroni_interval(*, diffs: np.ndarray) -> tuple[float, float]:
    """Return the month-block bootstrap interval of the mean of `diffs` at the Bonferroni level of
    `N_COUNTED` counted contrasts (two-sided 5% divided by `N_COUNTED`).
    """
    if len(diffs) == 0:
        return (np.nan, np.nan)
    alpha = 0.05 / N_COUNTED
    idx = RNG.integers(0, len(diffs), size=(N_BOOT, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def gate_tables(
    *, t: c.X4Table, gi: dict, on: np.ndarray, snap: np.ndarray, hold: np.ndarray, frac: np.ndarray
) -> pl.DataFrame:
    """Return how often the rule is active on event-free (Primary, origin day) pairs, and false
    triggers per Primary-year. A false trigger is a day the gate turns on with no event within 3 days.
    """
    near, gap, od = gi["near"], gi["gap"], gi["origin_days"]
    rows = []
    for g in (*GROUPS_SCORED, "D15 and D18", "all 16"):
        ps = [p for p in range(16) if g == "all 16" or t.group[p] == g]
        pairs = [(p, d) for p in ps for d in od if not near[p, d]]
        if not pairs:
            continue
        pd_ = np.array(pairs)
        years = len(pairs) / 365.25
        turn_on = on & ~np.roll(on, 1, axis=1)
        rows.append(
            {
                "group": g,
                "event_free_origin_days": len(pairs),
                "false_triggers": int(turn_on[pd_[:, 0], pd_[:, 1]].sum()),
                "false_triggers_per_primary_year": float(
                    turn_on[pd_[:, 0], pd_[:, 1]].sum() / years
                ),
                "snap_backs_on_event_free_days": int(snap[pd_[:, 0], pd_[:, 1]].sum()),
                "share_days_gate_on": float(on[pd_[:, 0], pd_[:, 1]].mean()),
                "share_days_event_mode_A1": float(hold[pd_[:, 0], pd_[:, 1]].mean()),
                "share_days_event_mode_A2": float((hold | on)[pd_[:, 0], pd_[:, 1]].mean()),
                "share_days_event_mode_placebo": float(frac[pd_[:, 0], pd_[:, 1]].mean()),
            }
        )
    return pl.DataFrame(rows)


def trigger_times(*, ev: pl.DataFrame, on: np.ndarray, snap: np.ndarray, label: str) -> list[dict]:
    """Return, per event (rows with `primary`, `onset_idx`, `end_idx`), the days from onset to the
    first gate-on origin day and from the end to the first snap-back at or after the end.
    """
    out = []
    for e in ev.iter_rows(named=True):
        p = e["primary"]
        d0, d1 = e["onset_idx"] // 48, e["end_idx"] // 48
        trig = [d for d in range(d0, d1 + 1) if on[p, d]]
        snaps = [d for d in range(d0, d1 + 30) if d < snap.shape[1] and snap[p, d] and d >= d1]
        out.append(
            {
                "source": label, "primary": p, "onset_day": d0, "end_day": d1,
                "length_days": e["length_days"], "triggered": bool(trig),
                "days_onset_to_trigger": (trig[0] - d0) if trig else None,
                "snapped_back": bool(snaps),
                "days_end_to_snap_back": (snaps[0] - d1) if snaps else None,
            }
        )  # fmt: skip
    return out


# ---------------------------------------------------------------- the injected bank


def key(prim: np.ndarray, origin_idx: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Return an integer key of (Primary, origin, valid time)."""
    return (prim.astype(np.int64) * 200_000 + origin_idx) * 200_000 + idx


def load_injected(
    *,
    t: c.X4Table,
    month: str,
    r: int,
    rows: np.ndarray,
    order: np.ndarray,
    sorted_keys: np.ndarray,
) -> np.ndarray:
    """Return the seed-mean injected X3-syn quantiles of pass `r` for the table rows `rows` (NaN
    where absent).
    """
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    parts = []
    for d in ("d0", "d14"):
        seeds = []
        for s in (0, 1):
            f = c.INJECT / f"{month}_r{r:02d}_{d}_s{s}.parquet"
            if f.exists():
                seeds.append(pl.read_parquet(f, columns=[*c.KEYS, *c.QCOLS]))
        if seeds:
            parts.append(pl.concat(seeds).group_by(c.KEYS).agg([pl.col(q).mean() for q in c.QCOLS]))
    df = pl.concat(parts).with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=(
            (pl.col("valid_time").dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US
        ).cast(pl.Int64),
        oidx=(
            (pl.col("power_fcst_init_time").dt.epoch("us") - c.k578.prep.GRID_START_US)
            // c.k578.prep.STEP_US
        ).cast(pl.Int64),
    )
    kk = key(df["prim"].to_numpy(), df["oidx"].to_numpy(), df["idx"].to_numpy())
    pos = np.clip(np.searchsorted(sorted_keys, kk), 0, len(sorted_keys) - 1)
    ok = sorted_keys[pos] == kk
    slot = np.full(t.n, -1, np.int64)
    slot[order[pos[ok]]] = np.flatnonzero(ok)
    q = df.select(c.QCOLS).to_numpy().astype(np.float32)
    out = np.full((len(rows), 11), np.nan, np.float32)
    s = slot[rows]
    has = s >= 0
    out[has] = np.sort(q[s[has]], axis=1)
    return out


def score_bank(*, t: c.X4Table, gi: dict) -> tuple[pl.DataFrame, list[dict]]:
    """Return the per-event bank table and the per-event trigger times."""
    ev_all = pl.read_parquet(BANK_PARQUET)
    kk_all = key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk_all)
    sorted_keys = kk_all[order]
    months = [m for m in c.MONTHS if m in set(ev_all["month"].unique().to_list())]
    rows_out: list[dict] = []
    trig_rows: list[dict] = []
    gap_real = gi["gap"]
    for m in months:
        in_m = t.month == m
        days_m = np.unique(t.idx[in_m] // 48)
        origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
        gap_m = np.full_like(gap_real, np.nan)
        gap_m[:, days_m.min() : days_m.max() + 1] = gap_real[:, days_m.min() : days_m.max() + 1]
        for r in range(bank.N_PASSES):
            ev = ev_all.filter((pl.col("month") == m) & (pl.col("pass") == r)).filter(
                pl.col("primary").is_in([p for p in range(16) if t.group[p] in GROUPS_SCORED])
            )
            if ev.height == 0:
                continue
            lo_p = {e["primary"]: e["onset_idx"] - 7 * 48 for e in ev.iter_rows(named=True)}
            hi_p = {e["primary"]: e["end_idx"] + 14 * 48 for e in ev.iter_rows(named=True)}
            lo = np.array([lo_p.get(p, 1 << 40) for p in range(16)])[t.prim]
            hi = np.array([hi_p.get(p, -1) for p in range(16)])[t.prim]
            rows = np.flatnonzero(in_m & t.free & (t.idx % 2 == 0) & (t.idx >= lo) & (t.idx < hi))
            vin = load_injected(t=t, month=m, r=r, rows=rows, order=order, sorted_keys=sorted_keys)
            ok = np.isfinite(vin).all(axis=1)
            rows, vin = rows[ok], vin[ok]
            if len(rows) == 0:
                continue
            # the gate reads the contaminated lag: the clean gap plus the injected shift on the event days
            gap_i = gap_m.copy()
            for e in ev.iter_rows(named=True):
                gap_i[e["primary"], e["onset_idx"] // 48 : e["end_idx"] // 48] += (
                    e["sign"] * e["depth"]
                )
            on, snap = c.gate_state(
                gap=gap_i, thr=gi["thr"], origin_days=origin_days_m, release_days=1
            )
            hold = c.hold_from(snap=snap)
            frac = placebo_fraction(snap=snap, origin_days=origin_days_m)
            pre = {
                k: c.row_metrics(y=t.obs[rows], q=q)
                for k, q in c.building_blocks(v=vin, m=t.arms["M"][rows]).items()
            }
            hold_r, on_r, frac_r = (
                hold[t.prim[rows], t.origin_day[rows]],
                on[t.prim[rows], t.origin_day[rows]],
                frac[t.prim[rows], t.origin_day[rows]],
            )
            # reference for the follow ratio: the same weights with clean lags
            v0 = t.arms["V"][rows]
            for e in ev.iter_rows(named=True):
                p, sgn, depth = e["primary"], float(e["sign"]), float(e["depth"])
                trig_rows.extend(
                    trigger_times(
                        ev=pl.DataFrame([{**e, "length_days": float(e["length_days"])}]),
                        on=on,
                        snap=snap,
                        label=f"bank {m} r{r}",
                    )
                )
                for phase in PHASES:
                    wlo, whi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
                    for band in (0, 1):
                        sel = np.flatnonzero(
                            (t.prim[rows] == p)
                            & (t.idx[rows] >= wlo)
                            & (t.idx[rows] < whi)
                            & (t.band[rows] == band)
                        )
                        if len(sel) < 60:
                            continue
                        am = c.combine(
                            base={k: v[sel] for k, v in pre.items()},
                            hold=hold_r[sel],
                            on=on_r[sel],
                            frac=frac_r[sel],
                        )
                        for arm, mm in am.items():
                            rec = {
                                "month": m, "pass": r, "primary": p, "group": str(t.group[p]),
                                "sign": int(sgn), "depth": depth, "length_days": int(e["length_days"]),
                                "phase": phase, "band": BANDS[band], "arm": arm, "n_rows": len(sel),
                                "share_origin_days_event_mode": float(
                                    (hold_r[sel] if arm in {"A1", "A3"} else (hold_r[sel] | on_r[sel]) if arm == "A2" else (np.ones(len(sel), bool) if arm == "Union" else frac_r[sel] if arm == "A4" else np.zeros(len(sel), bool))).mean()
                                ),
                                **metric_means(rows_metrics=mm, y_sign=sgn),
                            }  # fmt: skip
                            rows_out.append(rec)
            print("bank", m, r, flush=True)
    return pl.DataFrame(rows_out), trig_rows


def summarise_bank(*, tab: pl.DataFrame) -> pl.DataFrame:
    """Return per (group, length class, phase, band, arm) means over events with a 95% event-bootstrap interval."""
    metrics = [*c.METRICS, "nra_side95", "nra_side99"]
    out = []
    long_ev = pl.col("length_days") >= 15
    classes = {
        "all lengths": pl.lit(True),
        "3 days": pl.col("length_days") == 3,
        "7 days": pl.col("length_days") == 7,
        "15 days": pl.col("length_days") == 15,
        "30 days": pl.col("length_days") == 30,
        "15 and 30 days": long_ev,
    }
    for gname, gexpr in (
        ("MVA-only eight", pl.col("group") == "MVA-only eight"),
        ("signed-MW six", pl.col("group") == "signed-MW six"),
        ("both groups", pl.lit(True)),
    ):
        for cname, cexpr in classes.items():
            sub = tab.filter(gexpr & cexpr)
            for (phase, band, arm), g in sub.group_by("phase", "band", "arm"):
                rec = {
                    "group": gname,
                    "length_class": cname,
                    "phase": phase,
                    "band": band,
                    "arm": arm,
                    "n_events": g.height,
                }
                for mname in metrics:
                    mean, lo, hi = sc._boot_mean(vals=g[mname].to_numpy())
                    rec[mname], rec[f"{mname}_lo"], rec[f"{mname}_hi"] = mean, lo, hi
                out.append(rec)
    return pl.DataFrame(out)


# ---------------------------------------------------------------- main


def main() -> None:
    """Score every arm and write the tables."""
    c.WORK.mkdir(parents=True, exist_ok=True)
    t = c.X4Table()
    gi = c.gate_inputs(t=t)
    od = gi["origin_days"]
    on, snap = c.gate_state(gap=gi["gap"], thr=gi["thr"], origin_days=od, release_days=1)
    hold = c.hold_from(snap=snap)
    frac = placebo_fraction(snap=snap, origin_days=od)
    pre = block_blocks(t=t, rows=np.arange(t.n))
    print(
        "rows",
        t.n,
        "event-free share",
        float(t.free.mean()),
        "snap-backs",
        int(snap.sum()),
        flush=True,
    )
    real = real_events(t=t, pre=pre, on=on, hold=hold, frac=frac)
    real.write_csv(c.WORK / "real_event_phase.csv")
    free = event_free(t=t, pre=pre, on=on, hold=hold, frac=frac)
    free.write_csv(c.WORK / "event_free.csv")
    gates = gate_tables(t=t, gi=gi, on=on, snap=snap, hold=hold, frac=frac)
    gates.write_csv(c.WORK / "gates.csv")
    print(gates)
    ev = sc.attach_depth(ev=sc.merged_events(), t=t)
    trig = trigger_times(ev=ev, on=on, snap=snap, label="verified")
    tab, trig_bank = score_bank(t=t, gi=gi)
    tab.write_csv(c.WORK / "bank_event_phase.csv")
    summarise_bank(tab=tab).write_csv(c.WORK / "bank_summary.csv")
    pl.DataFrame([*trig, *trig_bank]).write_csv(c.WORK / "trigger_times.csv")
    np.savez(
        c.WORK / "gate_state.npz",
        on=on,
        snap=snap,
        hold=hold,
        frac=frac,
        gap=gi["gap"],
        thr=gi["thr"],
    )
    print("done", tab.height, "bank rows")


if __name__ == "__main__":
    main()

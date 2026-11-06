# ruff: noqa: D107, D102, E501, PERF401, PLR0915, PLR0912, D103, ANN001
"""Experiment 598: score R, W, S, S-g, S-p, the width-matched S arms, and G4 on the table T.

No fit and no GPU. Reads the stored forecasts of arm A1 (experiment 591) and of the S arms (this
experiment), and writes tables under `data/ml-experiments/00598-*/score<suffix>/`.

Systems (every one a (rows, 11) quantile array on experiment 583's table of rows):

- `A1`: experiment 591's A1 alone, seed mean.
- `R`: A1 + G4 from experiment 597: the union band of A1 and its lags-masked twin on origin days where
  G4 (experiment 565's E2 as the expectation) is on or held, A1 elsewhere. The stored export.
- `W`: A1 with its band rescaled about its median by one constant factor per fold month, chosen so that
  its mean q95 - q05 on R's wide origin days equals R's. A width control, not a gateless baseline.
- `S`, `S-g`, `S-p`: this experiment's arms, seed mean.
- `S-rs`, `S-g-rs`, `S-p-rs`: each arm's band rescaled about its median, per fold month and seed, so that
  its mean q95 - q05 on R's wide origin days equals R's (one factor printed per fold and seed).
- `G4T`: A1 + G4 where the gate reads the table T (mean of the two seeds) instead of E2.

Outputs: `factors.csv`; `event_free.csv` (CRPS, coverage, pinball loss, width per group, band, and
system, with paired month-block contrasts); `real_event_phase.csv` (the verified December pair);
`bank_event_phase.csv` and `bank_summary.csv` (the injected bank); `contrasts.csv` (the seven counted
contrasts with their intervals and minimum detectable effects); `verdicts.csv` (P1 to P5).

Usage: python e00598_score.py [--suffix SUFFIX] [--months M ...] [--seeds S ...] [--no-bank]
"""

import argparse
from pathlib import Path

import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_score as s  # ty: ignore[unresolved-import]
import e00592_common as g592  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import e00597_score as sc597  # ty: ignore[unresolved-import]
import e00598_signals as sig  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00598-nra-gate-against-signals-as-inputs-at-equal-band-width"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
ARM_DIR = {"S": "S", "S-g": "Sg", "S-p": "Sp"}
S_ARMS = tuple(ARM_DIR)
N_COUNTED = 7
"""P1 (two phases), P2, P3, P4, and P5 (two metering groups)."""
ALPHA = 0.05 / N_COUNTED
BAND = "24-120 h"
RNG = np.random.default_rng(598)
N_BOOT = 20_000
TAIL = ("pin01", "pin05", "pin95", "pin99")
STAT_NAMES = sc597.STAT_NAMES
J = {n: i for i, n in enumerate(STAT_NAMES)}
PHASES = k597.PHASES
OUTCOME_PASS, OUTCOME_FAIL = 0.01, 0.03
WIDTH_TOL = 0.05
MARGIN = {"P1": 0.01, "P2": 0.03, "P3": 0.08, "P4": 0.02}
"""The margin of each prediction, for the untestable rule: a prediction whose minimum detectable effect exceeds its margin is reported as untestable."""


def width90(q: np.ndarray) -> np.ndarray:
    return q[:, c.IQ[0.95]] - q[:, c.IQ[0.05]]


def scale_about_median(*, q: np.ndarray, factor: float) -> np.ndarray:
    med = q[:, c.IQ[0.5] : c.IQ[0.5] + 1]
    return med + factor * (q - med)


# ---------------------------------------------------------------- loading


def load_rows(
    *, files: list[Path], t: c.X4Table, order: np.ndarray, sorted_keys: np.ndarray
) -> np.ndarray:
    """Return the (rows, 11) sorted quantiles of the stored forecast files on the table's rows (NaN where absent)."""
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    present = [f for f in files if f.exists()]
    if not present:
        return np.full((t.n, 11), np.nan, np.float32)
    df = pl.concat([pl.read_parquet(f, columns=[*c.KEYS, *c.QCOLS]) for f in present])
    df = df.with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=(
            (pl.col("valid_time").dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US
        ).cast(pl.Int64),
        oidx=(
            (pl.col("power_fcst_init_time").dt.epoch("us") - c.k578.prep.GRID_START_US)
            // c.k578.prep.STEP_US
        ).cast(pl.Int64),
    )
    kk = s.key(df["prim"].to_numpy(), df["oidx"].to_numpy(), df["idx"].to_numpy())
    pos = np.clip(np.searchsorted(sorted_keys, kk), 0, len(sorted_keys) - 1)
    ok = sorted_keys[pos] == kk
    slot = np.full(t.n, -1, np.int64)
    slot[order[pos[ok]]] = np.flatnonzero(ok)
    q = df.select(c.QCOLS).to_numpy().astype(np.float32)
    out = np.full((t.n, 11), np.nan, np.float32)
    has = slot >= 0
    out[has] = np.sort(q[slot[has]], axis=1)
    return out


class Arm:
    """One S arm: its work folder and the per-seed quantiles on the table's rows."""

    def __init__(self, *, name: str, suffix: str, months: list[str], seeds: list[int]) -> None:
        self.name = name
        self.work = ROOT / f"{ARM_DIR[name]}{suffix}" / "work"
        self.months, self.seeds = months, seeds
        self.q: dict[int, np.ndarray] = {}

    def available(self) -> bool:
        return all(
            (self.work / "fits/P5" / f"{m}_d{d}_s{s_}.parquet").exists()
            for m in self.months
            for d in (0, 14)
            for s_ in self.seeds
        )

    def load(self, *, t: c.X4Table, order: np.ndarray, sorted_keys: np.ndarray) -> None:
        for seed in self.seeds:
            files = [
                self.work / "fits/P5" / f"{m}_d{d}_s{seed}.parquet"
                for m in self.months
                for d in (0, 14)
            ]
            self.q[seed] = load_rows(files=files, t=t, order=order, sorted_keys=sorted_keys)

    def inject(
        self, *, t: c.X4Table, month: str, r: int, rows: np.ndarray, order, sorted_keys
    ) -> dict[int, np.ndarray]:
        """Return per seed the injected quantiles of pass `r` on `rows`."""
        out = {}
        for seed in self.seeds:
            files = [
                self.work / "inject" / f"{month}_r{r:02d}_d{d}_s{seed}.parquet" for d in (0, 14)
            ]
            full = load_rows(files=files, t=t, order=order, sorted_keys=sorted_keys)
            out[seed] = full[rows]
        return out


# ---------------------------------------------------------------- G4 on T


class TableGate:
    """G4 with the expectation T (the two seeds' lag-free medians averaged) in place of E2."""

    def __init__(self, *, inp: k597.Inputs, months: list[str], seeds: list[int]) -> None:
        self.inp, self.months, self.seeds = inp, months, seeds
        self.n_days = inp.n_days
        self.clean = inp.t.clean
        self.tmean: dict[str, np.ndarray] = {}
        self.months = [
            m
            for m in months
            if all((ROOT / "tables" / f"T_{m}_s{s_}.npz").exists() for s_ in seeds)
        ]
        for m in self.months:
            meds = [np.load(ROOT / "tables" / f"T_{m}_s{s_}.npz")["median"] for s_ in seeds]
            self.tmean[m] = np.nanmean(np.stack(meds), axis=0)
        t = inp.t
        self.month_of_day = {int(d): str(m) for d, m in zip(t.origin_day, t.month, strict=True)}
        gap = self.gap_real()
        self.gap = gap
        self.thr = g592.threshold(gap=gap, near=inp.gi["near"])

    def gap_for(self, *, power: np.ndarray, only_month: str | None = None) -> np.ndarray:
        out = np.full((16, self.n_days), np.nan, np.float32)
        for m in self.months:
            if only_month is not None and m != only_month:
                continue
            res = (power - self.tmean[m]).astype(np.float32)
            gm = sig.fresh_gap_by_day(res_sig=res, n_days=self.n_days)
            days = [d for d, mm in self.month_of_day.items() if mm == m]
            out[:, days] = gm[:, days]
        return out

    def gap_real(self) -> np.ndarray:
        return self.gap_for(power=self.clean)

    def wide(self, *, gap: np.ndarray, origin_days: list[int]) -> np.ndarray:
        return sig.g4_wide(
            gap=gap, thr=self.thr, origin_days=origin_days, neighbours=self.inp.nb
        )  # fmt: skip


# ---------------------------------------------------------------- the systems on real rows


class Systems:
    """Every system's quantiles on the table's rows, and the width factors."""

    def __init__(
        self, *, inp: k597.Inputs, arms: dict[str, Arm], months: list[str], seeds: list[int]
    ) -> None:
        t = inp.t
        self.inp, self.t, self.arms, self.months, self.seeds = inp, t, arms, months, seeds
        v, m = t.arms["V"], t.arms["M"]
        self.wide_r = inp.mode[t.prim, t.origin_day]
        self.q: dict[str, np.ndarray] = {"A1": v, "R": k597.assemble(v=v, m=m, wide=self.wide_r)}
        self.use = np.isin(t.month, months)
        factors = []
        w = np.full_like(v, np.nan)
        for mo in months:
            rows = self.use & (t.month == mo) & self.wide_r
            f = float(width90(self.q["R"][rows]).mean() / width90(v[rows]).mean())
            factors.append({"system": "W", "month": mo, "seed": -1, "factor": f})
            sel = self.use & (t.month == mo)
            w[sel] = scale_about_median(q=v[sel], factor=f)
        self.q["W"] = w
        self.tg = TableGate(inp=inp, months=months, seeds=seeds)
        wide_t = self.tg.wide(gap=self.tg.gap, origin_days=inp.origin_days)
        self.wide_t = wide_t[t.prim, t.origin_day]
        self.q["G4T"] = k597.assemble(v=v, m=m, wide=self.wide_t)
        for name, arm in arms.items():
            per_seed, per_seed_rs = [], []
            for seed in seeds:
                q = arm.q[seed]
                rs = np.full_like(q, np.nan)
                for mo in months:
                    sel = self.use & (t.month == mo)
                    rows = sel & self.wide_r & np.isfinite(q).all(axis=1)
                    f = float(width90(self.q["R"][rows]).mean() / width90(q[rows]).mean())
                    factors.append({"system": f"{name}-rs", "month": mo, "seed": seed, "factor": f})
                    rs[sel] = scale_about_median(q=q[sel], factor=f)
                per_seed.append(q)
                per_seed_rs.append(rs)
            self.q[name] = np.sort(np.mean(per_seed, axis=0), axis=1)
            self.q[f"{name}-rs"] = np.sort(np.mean(per_seed_rs, axis=0), axis=1)
        self.factors = pl.DataFrame(factors)
        self.names = list(self.q)
        self.common = self.use & np.all(
            [np.isfinite(q).all(axis=1) for q in self.q.values()], axis=0
        )

    def factor(self, *, system: str, month: str, seed: int) -> float:
        f = self.factors.filter(
            (pl.col("system") == system) & (pl.col("month") == month) & (pl.col("seed") == seed)
        )
        return float(f["factor"][0])


# ---------------------------------------------------------------- bootstrap helpers


def boot_interval(*, vals: np.ndarray, alpha: float = ALPHA) -> tuple[float, float, float]:
    """Return the mean of `vals` (events or months as blocks) and its percentile bootstrap interval at level `alpha`."""
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[idx].mean(axis=1)
    return (
        float(v.mean()),
        float(np.percentile(m, 100 * alpha / 2)),
        float(np.percentile(m, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------- event-free rows


def event_free(*, sy: Systems) -> pl.DataFrame:
    t = sy.t
    st = {n: sc597.row_stats(y=t.obs, q=np.nan_to_num(q)) for n, q in sy.q.items()}
    months = list(sy.months)
    out = []
    for grp in (*k597.GROUPS_SCORED, "D15 and D18"):
        gsel = t.group_of_row == grp
        for band in (0, 1):
            sel = t.free & gsel & (t.band == band) & sy.common
            if sel.sum() < 500:
                continue
            for name in sy.names:
                r = {"system": name, "group": grp, "band": s.BANDS[band], "n_rows": int(sel.sum())}
                for key in ("cover90", "width90", "crps", *TAIL):
                    col = J[key]
                    r[key] = float(st[name][sel, col].mean() * (1.0 if key == "cover90" else 100.0))
                for ref in ("A1", "R"):
                    if name == ref:
                        continue
                    for key in ("crps", "pin05", "pin95"):
                        d = np.array([
                            (st[name][sel & (t.month == m), J[key]] - st[ref][sel & (t.month == m), J[key]]).mean() * 100.0
                            for m in months if (sel & (t.month == m)).any()
                        ])  # fmt: skip
                        mean, lo, hi = boot_interval(vals=d)
                        (
                            r[f"{key}_minus_{ref}"],
                            r[f"{key}_minus_{ref}_lo"],
                            r[f"{key}_minus_{ref}_hi"],
                        ) = mean, lo, hi
                        r[f"{key}_minus_{ref}_mde"] = (hi - lo) / 2
                out.append(r)
    return pl.DataFrame(out, infer_schema_length=None)


# ---------------------------------------------------------------- the verified pair


def real_events(*, sy: Systems) -> pl.DataFrame:
    t = sy.t
    st = {n: sc597.row_stats(y=t.nra, q=np.nan_to_num(q)) for n, q in sy.q.items()}
    ev = sc.attach_depth(ev=sc.merged_events(), t=t)
    out = []
    for e in ev.iter_rows(named=True):
        sgn = 1.0 if e["depth_share"] > 0 else -1.0
        for phase in PHASES:
            lo, hi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
            for band in (0, 1):
                sel = np.flatnonzero(
                    sy.common
                    & (t.prim == e["primary"])
                    & (t.idx >= lo)
                    & (t.idx < hi)
                    & (t.band == band)
                )
                if len(sel) < 100:
                    continue
                for name in sy.names:
                    out.append({
                        "event": e["name"], "group": e["group"], "length_days": e["length_days"],
                        "phase": phase, "band": s.BANDS[band], "system": name, "n_rows": len(sel),
                        **sc597.means(rows_metrics=st[name][sel], sgn=sgn),
                    })  # fmt: skip
    return pl.DataFrame(out)


# ---------------------------------------------------------------- the injected bank


def score_bank(*, sy: Systems) -> pl.DataFrame:
    """Return one row per (month, pass, event, phase, band, system) of the injected bank.

    R's wide days come from G4 on E2 with the injected shifts (experiment 597); `G4T` reads T. The
    arms' injected quantiles are the seed means (the width-matched ones use the per-seed factors).
    """
    inp, t = sy.inp, sy.t
    gi = inp.gi
    ev_all = pl.read_parquet(s.BANK_PARQUET)
    kk_all = s.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk_all)
    sorted_keys = kk_all[order]
    months = [m for m in sy.months if m in set(ev_all["month"].unique().to_list())]
    rows_out: list[dict] = []
    real_wide_t = sy.wide_t
    for m in months:
        in_m = t.month == m
        days_m = np.unique(t.idx[in_m] // 48)
        origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
        sl = slice(days_m.min(), days_m.max() + 1)
        for r in range(bank.N_PASSES):
            ev = ev_all.filter((pl.col("month") == m) & (pl.col("pass") == r)).filter(
                pl.col("primary").is_in([p for p in range(16) if t.group[p] in k597.GROUPS_SCORED])
            )
            if ev.height == 0:
                continue
            lo_p = {e["primary"]: e["onset_idx"] - 7 * 48 for e in ev.iter_rows(named=True)}
            hi_p = {e["primary"]: e["end_idx"] + 14 * 48 for e in ev.iter_rows(named=True)}
            lo = np.array([lo_p.get(p, 1 << 40) for p in range(16)])[t.prim]
            hi = np.array([hi_p.get(p, -1) for p in range(16)])[t.prim]
            rows = np.flatnonzero(
                in_m & t.free & sy.common & (t.idx % 2 == 0) & (t.idx >= lo) & (t.idx < hi)
            )
            if len(rows) == 0:
                continue
            inj = {
                n: arm.inject(t=t, month=m, r=r, rows=rows, order=order, sorted_keys=sorted_keys)
                for n, arm in sy.arms.items()
            }
            a1_in = s.load_injected(
                t=t, month=m, r=r, rows=rows, order=order, sorted_keys=sorted_keys
            )
            ok = np.isfinite(a1_in).all(axis=1)
            inj = {n: d for n, d in inj.items() if all(np.isfinite(q).any() for q in d.values())}
            for d in inj.values():
                for q in d.values():
                    ok &= np.isfinite(q).all(axis=1)
            rows = rows[ok]
            a1_in = a1_in[ok]
            inj = {n: {s_: q[ok] for s_, q in d.items()} for n, d in inj.items()}
            if len(rows) == 0:
                continue
            # R's wide days under the injection (experiment 597), and G4 on T under the same injection
            old = np.full_like(gi["gap"], np.nan)
            old[:, sl] = gi["gap"][:, sl]
            shifts: list[tuple[int, int, int, float]] = []
            power = np.where(np.isfinite(t.clean), t.clean, np.nan).astype(np.float32)
            shift_arr = np.zeros_like(power)
            for e in ev.iter_rows(named=True):
                p, shift = e["primary"], e["sign"] * e["depth"]
                old[p, e["onset_idx"] // 48 : e["end_idx"] // 48] += shift
                shifts.append((p, e["onset_idx"], e["end_idx"], float(shift)))
                shift_arr[p, e["onset_idx"] : e["end_idx"]] += shift
                nbs = inp.nb.get(p, [])
                if nbs:
                    n = nbs[(r + p) % len(nbs)]
                    mir = float(-shift * t.capacity[p] / t.capacity[n])
                    shifts.append((n, e["onset_idx"], e["end_idx"], mir))
                    shift_arr[n, e["onset_idx"] : e["end_idx"]] += mir
            fresh = inp.fresh.gap(shifts=shifts)
            fresh[:, : origin_days_m[0] + 1] = np.nan
            fresh[:, origin_days_m[-1] + 1 :] = np.nan
            states = g592.all_gates(old_gap=old, old_thr=gi["thr"], fresh_gap=fresh, fresh_thr=inp.fresh_thr, origin_days=origin_days_m, neighbours=inp.nb)  # fmt: skip
            wide_r_inj = g592.arm_masks(states=states)["G4"][0]
            power_inj = np.where(np.isfinite(power), power + shift_arr, power)
            gap_t = sy.tg.gap_for(power=power_inj, only_month=m)
            gap_t[:, : origin_days_m[0] + 1] = np.nan
            gap_t[:, origin_days_m[-1] + 1 :] = np.nan
            wide_t_inj = sy.tg.wide(gap=gap_t, origin_days=origin_days_m)
            p_rows, od_rows = t.prim[rows], t.origin_day[rows]
            v_m = t.arms["M"][rows]
            q_inj = {"A1": a1_in}
            q_inj["R"] = k597.assemble(v=a1_in, m=v_m, wide=wide_r_inj[p_rows, od_rows])
            q_inj["G4T"] = k597.assemble(v=a1_in, m=v_m, wide=wide_t_inj[p_rows, od_rows])
            f_w = sy.factor(system="W", month=m, seed=-1)
            q_inj["W"] = scale_about_median(q=a1_in, factor=f_w)
            for n, d in inj.items():
                q_inj[n] = np.sort(np.mean([d[s_] for s_ in sy.seeds], axis=0), axis=1)
                q_inj[f"{n}-rs"] = np.sort(
                    np.mean([scale_about_median(q=d[s_], factor=sy.factor(system=f"{n}-rs", month=m, seed=s_)) for s_ in sy.seeds], axis=0),
                    axis=1,
                )  # fmt: skip
            wide_clean = {"R": sy.wide_r[rows], "G4T": real_wide_t[rows]}
            y = t.obs[rows]
            st = {n: sc597.row_stats(y=y, q=q) for n, q in q_inj.items()}
            twins = {n: sy.q[n][rows] for n in q_inj}
            for e in ev.iter_rows(named=True):
                p, sgn, depth = e["primary"], float(e["sign"]), float(e["depth"])
                for phase in PHASES:
                    wlo, whi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
                    for band in (0, 1):
                        sel = np.flatnonzero(
                            (p_rows == p)
                            & (t.idx[rows] >= wlo)
                            & (t.idx[rows] < whi)
                            & (t.band[rows] == band)
                        )
                        if len(sel) < 60:
                            continue
                        for name in q_inj:
                            wide_share = (
                                float(wide_r_inj[p_rows[sel], od_rows[sel]].mean()) if name == "R"
                                else float(wide_t_inj[p_rows[sel], od_rows[sel]].mean()) if name == "G4T" else np.nan
                            )  # fmt: skip
                            rows_out.append({
                                "month": m, "pass": r, "primary": p, "group": str(t.group[p]),
                                "sign": int(sgn), "depth": depth, "length_days": int(e["length_days"]),
                                "phase": phase, "band": s.BANDS[band], "system": name, "n_rows": len(sel),
                                "share_origin_days_wide": wide_share,
                                **sc597.means(rows_metrics=st[name][sel], sgn=sgn),
                                **sc597.follow(q=q_inj[name][sel], twin=twins[name][sel], sgn=sgn, depth=depth),
                            })  # fmt: skip
            del wide_clean
        print("bank", m, flush=True)
    tab = pl.DataFrame(rows_out)
    # keep only (month, pass) cells scored for every system, so that every system's mean is over the same events
    cells = tab.group_by("month", "pass").agg(pl.col("system").n_unique().alias("n"))
    full = cells.filter(pl.col("n") == cells["n"].max()).select("month", "pass")
    dropped = cells.height - full.height
    if dropped:
        print(
            f"dropped {dropped} (month, pass) cells where an arm's injected forecasts are absent",
            flush=True,
        )
    return tab.join(full, on=["month", "pass"], how="semi")


def summarise_bank(*, tab: pl.DataFrame) -> pl.DataFrame:
    """Per (group, length class, first-fold-pair flag, phase, band, system): means over events with 95% event-bootstrap intervals."""
    metrics = [*STAT_NAMES, "nra_side95", "nra_side99", "follow_q50", "follow_safety"]
    classes = {
        "all lengths": pl.lit(True),
        "7 days": pl.col("length_days") == 7,
        "15 and 30 days": pl.col("length_days") >= 15,
    }
    subsets = {
        "all months": pl.lit(True),
        "first fold pair": pl.col("month").is_in(["2025-10", "2025-12"]),
    }
    out = []
    for sname, sexpr in subsets.items():
        for gname, gexpr in (
            ("MVA-only eight", pl.col("group") == "MVA-only eight"),
            ("signed-MW six", pl.col("group") == "signed-MW six"),
            ("both groups", pl.lit(True)),
        ):
            for cname, cexpr in classes.items():
                sub = tab.filter(sexpr & gexpr & cexpr)
                for (phase, band, system), grp in sub.group_by("phase", "band", "system"):
                    rec = {
                        "subset": sname,
                        "group": gname,
                        "length_class": cname,
                        "phase": phase,
                        "band": band,
                        "system": system,
                        "n_events": grp.height,
                    }
                    for mname in metrics:
                        mean, lo, hi = sc._boot_mean(vals=grp[mname].to_numpy())
                        rec[mname], rec[f"{mname}_lo"], rec[f"{mname}_hi"] = mean, lo, hi
                    out.append(rec)
    return pl.DataFrame(out)


# ---------------------------------------------------------------- the seven counted contrasts and the verdicts


def paired(
    *, tab: pl.DataFrame, a_: str, b_: str, phase: str, metric: str, lengths: str = "15 and 30 days"
) -> tuple[float, float, float, int]:
    """Return mean(a - b), its Bonferroni-level interval, and the event count, paired on (month, pass, primary)."""
    f = tab.filter(
        (pl.col("phase") == phase)
        & (pl.col("band") == BAND)
        & (pl.col("length_days") >= (15 if lengths == "15 and 30 days" else 0))
    )
    x = f.filter(pl.col("system") == a_).select(
        "month", "pass", "primary", pl.col(metric).alias("a")
    )
    y = f.filter(pl.col("system") == b_).select(
        "month", "pass", "primary", pl.col(metric).alias("b")
    )
    j = x.join(y, on=["month", "pass", "primary"], how="inner")
    d = (j["a"] - j["b"]).to_numpy()
    mean, lo, hi = boot_interval(vals=d)
    return mean, lo, hi, len(d)


def contrasts(*, tab: pl.DataFrame, ef: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = []
    # P1: R - S coverage, days4-end and after0-7; also the width-matched S
    for phase in ("days4-end", "after0-7"):
        for arm in ("S", "S-rs"):
            mean, lo, hi, n = paired(tab=tab, a_="R", b_=arm, phase=phase, metric="cover90")
            rows.append(
                {
                    "test": f"P1 {phase}",
                    "contrast": f"cover90(R) - cover90({arm})",
                    "mean": mean,
                    "lo": lo,
                    "hi": hi,
                    "mde": (hi - lo) / 2,
                    "margin": MARGIN["P1"],
                    "n_events": n,
                }
            )
    mean, lo, hi, n = paired(tab=tab, a_="S", b_="W", phase="after0-7", metric="cover90")
    rows.append(
        {
            "test": "P2 after0-7",
            "contrast": "cover90(S) - cover90(W)",
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "mde": (hi - lo) / 2,
            "margin": MARGIN["P2"],
            "n_events": n,
        }
    )
    mean, lo, hi, n = paired(tab=tab, a_="S-rs", b_="W", phase="after0-7", metric="cover90")
    rows.append(
        {
            "test": "P2 after0-7 (width-matched S)",
            "contrast": "cover90(S-rs) - cover90(W)",
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "mde": (hi - lo) / 2,
            "margin": MARGIN["P2"],
            "n_events": n,
        }
    )
    for sysn in ("S", "R", "S-p", "A1"):
        mean, lo, hi, n = paired(tab=tab, a_=sysn, b_=sysn, phase="days4-end", metric="follow_q50")
        f = tab.filter(
            (pl.col("phase") == "days4-end")
            & (pl.col("band") == BAND)
            & (pl.col("length_days") >= 15)
            & (pl.col("system") == sysn)
        )
        m3, l3, h3 = boot_interval(vals=f["follow_q50"].to_numpy())
        rows.append(
            {
                "test": f"P3 follow ratio {sysn}",
                "contrast": f"follow_q50({sysn})",
                "mean": m3,
                "lo": l3,
                "hi": h3,
                "mde": (h3 - l3) / 2,
                "margin": MARGIN["P3"],
                "n_events": f.height,
            }
        )
    mean, lo, hi, n = paired(tab=tab, a_="S", b_="S-p", phase="days4-end", metric="cover90")
    rows.append(
        {
            "test": "P4 days4-end",
            "contrast": "cover90(S) - cover90(S-p)",
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "mde": (hi - lo) / 2,
            "margin": MARGIN["P4"],
            "n_events": n,
        }
    )
    for grp in k597.GROUPS_SCORED:
        r = ef.filter(
            (pl.col("group") == grp) & (pl.col("band") == BAND) & (pl.col("system") == "S")
        )
        if r.height:
            rows.append(
                {
                    "test": f"P5 {grp}",
                    "contrast": "event-free CRPS(S) - CRPS(A1), points of capacity",
                    "mean": float(r["crps_minus_A1"][0]),
                    "lo": float(r["crps_minus_A1_lo"][0]),
                    "hi": float(r["crps_minus_A1_hi"][0]),
                    "mde": float(r["crps_minus_A1_mde"][0]),
                    "margin": float(r["crps_minus_A1_mde"][0]),
                    "n_events": int(r["n_rows"][0]),
                }
            )
    tab_c = pl.DataFrame(rows)
    return tab_c, verdicts(c_=tab_c, tab=tab)


def bank_width(*, tab: pl.DataFrame, system: str) -> float:
    f = tab.filter(
        (pl.col("phase") == "days4-end")
        & (pl.col("band") == BAND)
        & (pl.col("length_days") >= 15)
        & (pl.col("system") == system)
    )
    return float(f["width90"].mean()) if f.height else float("nan")  # ty: ignore[invalid-argument-type]


def verdicts(*, c_: pl.DataFrame, tab: pl.DataFrame) -> pl.DataFrame:
    """Return P1 to P5 verdicts with the 'untestable' rule (minimum detectable effect above the margin)."""
    out = []
    row = {r["test"]: r for r in c_.iter_rows(named=True)}
    width_ratio = (
        bank_width(tab=tab, system="S") / bank_width(tab=tab, system="R") if tab.height else np.nan
    )

    def band_of(gap: float) -> str:
        return "pass" if gap < OUTCOME_PASS else "inconclusive" if gap < OUTCOME_FAIL else "fail"

    for phase in ("days4-end", "after0-7"):
        rs = None
        for r in c_.iter_rows(named=True):
            if r["test"] == f"P1 {phase}" and "S-rs" in r["contrast"]:
                rs = r
        s_raw = next(
            r
            for r in c_.iter_rows(named=True)
            if r["test"] == f"P1 {phase}" and r["contrast"].endswith("(S)")
        )
        gap_raw, gap_rs = s_raw["mean"], rs["mean"] if rs else np.nan
        equal_width = abs(width_ratio - 1.0) <= WIDTH_TOL
        verdict = band_of(gap_raw)
        if verdict == "pass" and not equal_width and band_of(gap_rs) != "pass":
            verdict = "not passed: S is wider or narrower than R by more than 5% and the width-matched S does not pass"
        if s_raw["mde"] > MARGIN["P1"]:
            verdict += "; untestable at this power"
        out.append(
            {
                "prediction": f"P1 {phase}",
                "gap_R_minus_S": gap_raw,
                "gap_R_minus_S_width_matched": gap_rs,
                "S_over_R_width_days4_end": width_ratio,
                "mde": s_raw["mde"],
                "verdict": verdict,
            }
        )
    for key in ("P2 after0-7", "P2 after0-7 (width-matched S)"):
        r = row[key]
        v = "supported" if r["mean"] >= MARGIN["P2"] and r["lo"] > 0 else "not supported"
        if r["mde"] > MARGIN["P2"]:
            v += "; untestable at this power"
        out.append(
            {
                "prediction": key,
                "estimate": r["mean"],
                "lo": r["lo"],
                "hi": r["hi"],
                "mde": r["mde"],
                "verdict": v,
            }
        )
    r = row["P3 follow ratio S"]
    v = "supported" if r["mean"] <= MARGIN["P3"] else "not supported"
    if r["mde"] > MARGIN["P3"]:
        v += "; untestable at this power"
    out.append(
        {
            "prediction": "P3 follow ratio of S",
            "estimate": r["mean"],
            "lo": r["lo"],
            "hi": r["hi"],
            "mde": r["mde"],
            "R": row["P3 follow ratio R"]["mean"],
            "S-p": row["P3 follow ratio S-p"]["mean"],
            "A1": row["P3 follow ratio A1"]["mean"],
            "verdict": v,
        }
    )
    r = row["P4 days4-end"]
    v = (
        "supported"
        if r["mean"] >= MARGIN["P4"] and r["lo"] > 0
        else "not supported (inputs carry nothing)"
    )
    if r["mde"] > MARGIN["P4"]:
        v += "; untestable at this power"
    out.append(
        {
            "prediction": "P4 S minus S-p",
            "estimate": r["mean"],
            "lo": r["lo"],
            "hi": r["hi"],
            "mde": r["mde"],
            "verdict": v,
        }
    )
    for grp in k597.GROUPS_SCORED:
        r = row.get(f"P5 {grp}")
        if r:
            v = "supported" if abs(r["mean"]) <= r["mde"] else "not supported"
            out.append(
                {
                    "prediction": f"P5 {grp}",
                    "estimate": r["mean"],
                    "lo": r["lo"],
                    "hi": r["hi"],
                    "mde": r["mde"],
                    "verdict": v,
                }
            )
    return pl.DataFrame(out, infer_schema_length=None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="")
    p.add_argument("--months", nargs="+", default=list(c.MONTHS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--no-bank", action="store_true")
    args = p.parse_args()
    out = ROOT / f"score{args.suffix}"
    out.mkdir(parents=True, exist_ok=True)
    s.N_COUNTED = N_COUNTED
    inp = k597.Inputs(arm="a1")
    chk = inp.check_gate_state()
    stored = np.load(k597.WORK / "a1" / "mode_g4.npy")
    print(
        "rebuilt G4 state equals experiment 592's:",
        chk,
        "| mode equals experiment 597's stored mode:",
        bool(np.array_equal(stored, inp.mode)),
        flush=True,
    )
    t = inp.t
    kk = s.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk)
    sorted_keys = kk[order]
    arms = {}
    for name in S_ARMS:
        arm = Arm(name=name, suffix=args.suffix, months=args.months, seeds=args.seeds)
        if arm.available():
            arm.load(t=t, order=order, sorted_keys=sorted_keys)
            arms[name] = arm
        else:
            print("arm", name, "has no complete fits under", arm.work, "- skipped", flush=True)
    sy = Systems(inp=inp, arms=arms, months=args.months, seeds=args.seeds)
    sy.factors.write_csv(out / "factors.csv")
    print(sy.factors, flush=True)
    ef = event_free(sy=sy)
    ef.write_csv(out / "event_free.csv")
    real_events(sy=sy).write_csv(out / "real_event_phase.csv")
    np.save(out / "wide_r.npy", inp.mode)
    np.save(out / "wide_t.npy", sy.tg.wide(gap=sy.tg.gap, origin_days=inp.origin_days))
    if args.no_bank:
        return
    tab = score_bank(sy=sy)
    tab.write_csv(out / "bank_event_phase.csv")
    summarise_bank(tab=tab).write_csv(out / "bank_summary.csv")
    if "S" in arms:
        c_, v_ = contrasts(tab=tab, ef=ef)
        c_.write_csv(out / "contrasts.csv")
        v_.write_csv(out / "verdicts.csv")
        print(v_)


if __name__ == "__main__":
    main()

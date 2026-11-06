# ruff: noqa: D107, D102, E501, PERF401, PLR0915, D103, ANN001, D205
"""Experiment 606: score R, W, A1, the permutation arms P3 to P7c, their placebos, and the width-matched arms.

No fit and no GPU. A copy of experiment 598's `e00598_score.py` with that experiment's contrasts and
verdicts replaced by this experiment's. It reads the stored forecasts of arm A1 (experiment 591) and of
the arms of this experiment, and writes tables under `data/ml-experiments/00610-*/score<suffix>/`.

Systems (every one a (rows, 11) quantile array on experiment 583's table of rows): `A1`; `R` (A1 plus the
G4 gate, experiment 597); `W` (A1 rescaled to R's mean width on R's wide days); each arm (seed mean);
`<arm>-rs`, each arm rescaled about its median, per fold month and seed, to R's width; `G4T`.

Usage: python e00610_score.py [--suffix SUFFIX] [--months M ...] [--seeds S ...] [--arms A ...] [--no-bank]
"""

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
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

NAME = "00610-nra-neighbour-summary-lag-encoder-inputs-iteration"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
ROOT598 = ML_EXPERIMENTS_DATA_DIR / "00598-nra-gate-against-signals-as-inputs-at-equal-band-width"
DEFAULT_ARMS = ("P3", "P4", "P5", "P6", "P7", "P7b", "P7c")
N_COUNTED = 12
"""P3 follow, P4 and P5 follow steps, P7 follow steps (2), the width-matched reach (2 phases), the placebo, and event-free CRPS of the best arm (2 groups), plus O2 (1)."""
"""P1 (two phases), P2, P3, P4, and P5 (two metering groups)."""
ALPHA = 0.05 / N_COUNTED
BAND = "24-120 h"
RNG = np.random.default_rng(606)
N_BOOT = 20_000
TAIL = ("pin01", "pin05", "pin95", "pin99")
STAT_NAMES = sc597.STAT_NAMES
J = {n: i for i, n in enumerate(STAT_NAMES)}
PHASES = k597.PHASES
OUTCOME_PASS, OUTCOME_FAIL = 0.01, 0.03
WIDTH_TOL = 0.05
MARGIN = {"P1": 0.01, "P2": 0.03, "P3": 0.08, "P4": 0.02, "P5": 0.03}
EF_WIDTH_TOL = 0.02
"""P1 needs S's event-free band width within 2% of R's."""
"""The margin of each prediction, for the untestable rule: a prediction whose minimum detectable effect exceeds its margin is reported as untestable. P5's margin is 0.03 points of capacity of event-free CRPS, fixed before any score."""


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
        self.work = ROOT / f"{name.replace('-', '')}{suffix}" / "work"
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
            if all((ROOT598 / "tables" / f"T_{m}_s{s_}.npz").exists() for s_ in seeds)
        ]
        for m in self.months:
            meds = [np.load(ROOT598 / "tables" / f"T_{m}_s{s_}.npz")["median"] for s_ in seeds]
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
    """Every system's quantiles on the table's rows, and the width factors.

    The width factors are fitted on the injected bank's rows (`factor_source="bank"`), the rows that
    P1 and P2 read: the 24-120 h rows of the origin days where R's band is wide under the injection.
    `factor_source="real"` fits them on the real rows R widens instead (used only with `--no-bank`).
    `factors`, if given, are the stored factors of an earlier scoring run, used as they are.
    """

    def __init__(
        self, *, inp: k597.Inputs, arms: dict[str, Arm], months: list[str], seeds: list[int],
        factor_source: str = "bank", factors: pl.DataFrame | None = None,
    ) -> None:  # fmt: skip
        t = inp.t
        self.inp, self.t, self.arms, self.months, self.seeds = inp, t, arms, months, seeds
        v, m = t.arms["V"], t.arms["M"]
        self.wide_r = inp.mode[t.prim, t.origin_day]
        self.q: dict[str, np.ndarray] = {"A1": v, "R": k597.assemble(v=v, m=m, wide=self.wide_r)}
        self.use = np.isin(t.month, months)
        self.tg = TableGate(inp=inp, months=months, seeds=seeds)
        wide_t = self.tg.wide(gap=self.tg.gap, origin_days=inp.origin_days)
        self.wide_t = wide_t[t.prim, t.origin_day]
        self.q["G4T"] = k597.assemble(v=v, m=m, wide=self.wide_t)
        for name, arm in arms.items():
            self.q[name] = np.sort(np.mean([arm.q[sd] for sd in seeds], axis=0), axis=1)
        self.common = self.use & np.all(
            [np.isfinite(q).all(axis=1) for q in self.q.values()], axis=0
        )
        if factors is not None:
            self.factors = factors
        elif factor_source == "bank":
            self.factors = fit_factors_bank(sy=self)
        else:
            self.factors = fit_factors_real(sy=self)
        print(self.factors, flush=True)
        w = np.full_like(v, np.nan)
        for mo in months:
            sel = self.use & (t.month == mo)
            w[sel] = scale_about_median(q=v[sel], factor=self.factor(system="W", month=mo, seed=-1))
        self.q["W"] = w
        for name, arm in arms.items():
            per_seed_rs = []
            for seed in seeds:
                rs = np.full_like(arm.q[seed], np.nan)
                for mo in months:
                    sel = self.use & (t.month == mo)
                    f = self.factor(system=f"{name}-rs", month=mo, seed=seed)
                    rs[sel] = scale_about_median(q=arm.q[seed][sel], factor=f)
                per_seed_rs.append(rs)
            self.q[f"{name}-rs"] = np.sort(np.mean(per_seed_rs, axis=0), axis=1)
        self.names = list(self.q)
        scaled = self.use & np.all([np.isfinite(q).all(axis=1) for q in self.q.values()], axis=0)
        assert np.array_equal(scaled, self.common), "a width factor removed rows"

    def factor(self, *, system: str, month: str, seed: int) -> float:
        f = self.factors.filter(
            (pl.col("system") == system) & (pl.col("month") == month) & (pl.col("seed") == seed)
        )
        assert f.height == 1, (system, month, seed)
        return float(f["factor"][0])


class FactorAcc:
    """Sums of R's band width and a system's band width over the rows a width factor is fitted on."""

    def __init__(self) -> None:
        self.sums: dict[tuple[str, str, int], list[float]] = {}

    def add(self, *, system: str, month: str, seed: int, ref: np.ndarray, own: np.ndarray) -> None:
        a = self.sums.setdefault((system, month, seed), [0.0, 0.0, 0.0])
        a[0] += float(width90(ref).sum())
        a[1] += float(width90(own).sum())
        a[2] += len(ref)

    def table(self, *, months: list[str], systems: list[tuple[str, int]]) -> pl.DataFrame:
        """Return one factor per (system, seed, month): R's summed width over the system's, with the
        row count. A month with no rows takes the factor pooled over all months (`pooled` True).
        """
        out = []
        for system, seed in systems:
            per = {mo: self.sums.get((system, mo, seed), [0.0, 0.0, 0.0]) for mo in months}
            tot = np.sum(list(per.values()), axis=0)
            assert tot[2] > 0, f"no rows to fit the width factor of {system} seed {seed}"
            for mo in months:
                n = per[mo][2]
                ref, own = (per[mo][0], per[mo][1]) if n > 0 else (tot[0], tot[1])
                out.append({
                    "system": system, "month": mo, "seed": seed, "factor": float(ref / own),
                    "n_rows": int(n), "pooled": bool(n == 0),
                })  # fmt: skip
        return pl.DataFrame(out)


def _system_seeds(*, sy: Systems) -> list[tuple[str, int]]:
    return [("W", -1), *((f"{n}-rs", sd) for n in sy.arms for sd in sy.seeds)]


def fit_factors_bank(*, sy: Systems) -> pl.DataFrame:
    """Fit W's and each S arm's width factor, per fold month (and seed), on injected-bank rows.

    The rows are the 24-120 h rows, in the bank's event windows, of the origin days where R's band is
    wide under the injection. The factors are the ratio of the mean R width to the mean width of A1
    (for W) or of the arm's seed (for the -rs arms), pooled over the 20 passes of the month.
    """
    t = sy.t
    acc = FactorAcc()
    for cell in bank_cells(sy=sy):
        sel = (t.band[cell.rows] == 1) & cell.wide_r_inj[cell.p_rows, cell.od_rows]
        if not sel.any():
            continue
        ref = cell.q_inj_r[sel]
        acc.add(system="W", month=cell.month, seed=-1, ref=ref, own=cell.a1_in[sel])
        for name, d in cell.inj.items():
            for sd, q in d.items():
                acc.add(system=f"{name}-rs", month=cell.month, seed=sd, ref=ref, own=q[sel])
    return acc.table(months=sy.months, systems=_system_seeds(sy=sy))


def fit_factors_real(*, sy: Systems) -> pl.DataFrame:
    """Fit the factors on the real rows R widens (24-120 h), the fallback when there is no bank."""
    t = sy.t
    acc = FactorAcc()
    for mo in sy.months:
        rows = sy.common & (t.month == mo) & sy.wide_r & (t.band == 1)
        if not rows.any():
            continue
        acc.add(system="W", month=mo, seed=-1, ref=sy.q["R"][rows], own=sy.q["A1"][rows])
        for name, arm in sy.arms.items():
            for sd in sy.seeds:
                acc.add(
                    system=f"{name}-rs", month=mo, seed=sd, ref=sy.q["R"][rows], own=arm.q[sd][rows]
                )
    return acc.table(months=sy.months, systems=_system_seeds(sy=sy))


# ---------------------------------------------------------------- bootstrap helpers


def boot_interval(*, vals: np.ndarray, alpha: float = ALPHA) -> tuple[float, float, float]:
    """Return the mean of `vals` and its percentile bootstrap interval at level `alpha`.

    The values are events or months, resampled as blocks.
    """
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
                r = {
                    "system": name,
                    "group": grp,
                    "band": s.BANDS[band],
                    "n_rows": int(sel.sum()),
                    "n_months": len({m for m in months if (sel & (t.month == m)).any()}),
                }
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


@dataclass
class Cell:
    """One (fold month, pass) of the injected bank: the rows, the forecasts on them, and R's wide days."""

    month: str
    r: int
    ev: pl.DataFrame
    rows: np.ndarray
    p_rows: np.ndarray
    od_rows: np.ndarray
    a1_in: np.ndarray
    inj: dict[str, dict[int, np.ndarray]]
    wide_r_inj: np.ndarray
    wide_t_inj: np.ndarray
    q_inj_r: np.ndarray
    rows597: np.ndarray
    q_r597: np.ndarray


def bank_cells(*, sy: Systems) -> Iterator[Cell]:
    """Yield each (fold month, pass) cell of the injected bank, one at a time.

    R's wide days come from G4 on E2 with the injected shifts (experiment 597); `G4T` reads T.
    `rows597` is experiment 597's own row set (injected A1 finite, no other filter), on which
    `q_r597` is R, for the reproduction check; `rows` is that set restricted to rows where every
    system is finite on the real rows and every arm's injected forecast is finite.
    """
    inp, t = sy.inp, sy.t
    gi = inp.gi
    ev_all = pl.read_parquet(s.BANK_PARQUET)
    kk_all = s.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk_all)
    sorted_keys = kk_all[order]
    months = [m for m in sy.months if m in set(ev_all["month"].unique().to_list())]
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
            rows597 = np.flatnonzero(
                in_m & t.free & (t.idx % 2 == 0) & (t.idx >= lo) & (t.idx < hi)
            )
            a1_597 = s.load_injected(
                t=t, month=m, r=r, rows=rows597, order=order, sorted_keys=sorted_keys
            )
            ok597 = np.isfinite(a1_597).all(axis=1)
            rows597, a1_597 = rows597[ok597], a1_597[ok597]
            if len(rows597) == 0:
                continue
            keep = sy.common[rows597]
            rows, a1_in = rows597[keep], a1_597[keep]
            inj = {
                n: arm.inject(t=t, month=m, r=r, rows=rows, order=order, sorted_keys=sorted_keys)
                for n, arm in sy.arms.items()
            }
            inj = {n: d for n, d in inj.items() if all(np.isfinite(q).any() for q in d.values())}
            ok = np.ones(len(rows), bool)
            for d in inj.values():
                for q in d.values():
                    ok &= np.isfinite(q).all(axis=1)
            rows, a1_in = rows[ok], a1_in[ok]
            inj = {n: {s_: q[ok] for s_, q in d.items()} for n, d in inj.items()}
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
            q_r = k597.assemble(v=a1_in, m=t.arms["M"][rows], wide=wide_r_inj[p_rows, od_rows])
            q_r597 = k597.assemble(
                v=a1_597, m=t.arms["M"][rows597],
                wide=wide_r_inj[t.prim[rows597], t.origin_day[rows597]],
            )  # fmt: skip
            yield Cell(
                month=m, r=r, ev=ev, rows=rows, p_rows=p_rows, od_rows=od_rows, a1_in=a1_in,
                inj=inj, wide_r_inj=wide_r_inj, wide_t_inj=wide_t_inj, q_inj_r=q_r,
                rows597=rows597, q_r597=q_r597,
            )  # fmt: skip


def reproduce_597_bank(*, sy: Systems, cell: Cell) -> list[dict]:
    """Return R's per-event means on experiment 597's own row set, keyed like experiment 597's table."""
    t = sy.t
    y = t.obs[cell.rows597]
    st = sc597.row_stats(y=y, q=cell.q_r597)
    out = []
    for e in cell.ev.iter_rows(named=True):
        p, sgn = e["primary"], float(e["sign"])
        for phase in PHASES:
            wlo, whi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
            for band in (0, 1):
                sel = np.flatnonzero(
                    (t.prim[cell.rows597] == p)
                    & (t.idx[cell.rows597] >= wlo)
                    & (t.idx[cell.rows597] < whi)
                    & (t.band[cell.rows597] == band)
                )
                if len(sel) < 60:
                    continue
                out.append({
                    "month": cell.month, "pass": cell.r, "primary": p, "phase": phase,
                    "band": s.BANDS[band], **sc597.means(rows_metrics=st[sel], sgn=sgn),
                })  # fmt: skip
    return out


def score_bank(*, sy: Systems) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return one row per (month, pass, event, phase, band, system) of the injected bank, and R's
    per-event means on experiment 597's own row set.

    The arms' injected quantiles are the seed means (the width-matched ones use the per-seed factors).
    """
    t = sy.t
    rows_out: list[dict] = []
    repro: list[dict] = []
    last = None
    for cell in bank_cells(sy=sy):
        if last is not None and last != cell.month:
            print("bank", last, flush=True)
        last = cell.month
        m, r, rows, ev = cell.month, cell.r, cell.rows, cell.ev
        repro.extend(reproduce_597_bank(sy=sy, cell=cell))
        if len(rows) == 0:
            continue
        p_rows, od_rows = cell.p_rows, cell.od_rows
        v_m = t.arms["M"][rows]
        q_inj = {"A1": cell.a1_in, "R": cell.q_inj_r}
        q_inj["G4T"] = k597.assemble(v=cell.a1_in, m=v_m, wide=cell.wide_t_inj[p_rows, od_rows])
        q_inj["W"] = scale_about_median(
            q=cell.a1_in, factor=sy.factor(system="W", month=m, seed=-1)
        )
        for n, d in cell.inj.items():
            q_inj[n] = np.sort(np.mean([d[s_] for s_ in sy.seeds], axis=0), axis=1)
            q_inj[f"{n}-rs"] = np.sort(
                np.mean([scale_about_median(q=d[s_], factor=sy.factor(system=f"{n}-rs", month=m, seed=s_)) for s_ in sy.seeds], axis=0),
                axis=1,
            )  # fmt: skip
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
                            float(cell.wide_r_inj[p_rows[sel], od_rows[sel]].mean()) if name == "R"
                            else float(cell.wide_t_inj[p_rows[sel], od_rows[sel]].mean()) if name == "G4T" else np.nan
                        )  # fmt: skip
                        rows_out.append({
                            "month": m, "pass": r, "primary": p, "group": str(t.group[p]),
                            "sign": int(sgn), "depth": depth, "length_days": int(e["length_days"]),
                            "phase": phase, "band": s.BANDS[band], "system": name, "n_rows": len(sel),
                            "share_origin_days_wide": wide_share,
                            **sc597.means(rows_metrics=st[name][sel], sgn=sgn),
                            **sc597.follow(q=q_inj[name][sel], twin=twins[name][sel], sgn=sgn, depth=depth),
                        })  # fmt: skip
    print("bank", last, flush=True)
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
    return tab.join(full, on=["month", "pass"], how="semi"), pl.DataFrame(repro)


REPRO_TOL = 1e-5
"""Largest allowed difference from experiment 597 (coverage and tail shares agree exactly; the band widths and CRPS differ by float32 rounding, about 1e-6), relative to the stored value where that is above 1 (width and CRPS are in points of capacity), else absolute."""


def check_reproduction(*, sy: Systems, repro_bank: pl.DataFrame) -> pl.DataFrame:
    """Compute R on experiment 597's own row sets and assert it equals experiment 597's stored table.

    Checks the injected bank (per event, phase, and band) and the real event-free rows (per group and
    band); returns the largest differences found. Raises if any exceeds `REPRO_TOL`.
    """
    t = sy.t
    out = []
    old = pl.read_csv(k597.WORK / "a1" / "bank_event_phase.csv", infer_schema_length=None).filter(
        (pl.col("system") == "G4") & pl.col("month").is_in(repro_bank["month"].unique())
    )
    keys = ["month", "pass", "primary", "phase", "band"]
    cols = [x for x in (*STAT_NAMES, "nra_side95", "nra_side99") if x in old.columns]
    j = repro_bank.join(old.select(*keys, *cols), on=keys, suffix="_597", how="inner")
    assert j.height == repro_bank.height == old.height, (j.height, repro_bank.height, old.height)
    bank_diff = max(
        float(((j[x] - j[f"{x}_597"]).abs() / j[f"{x}_597"].abs().clip(lower_bound=1.0)).max())  # ty: ignore[invalid-argument-type]
        for x in cols
    )
    out.append(
        {"table": "injected bank, per event", "rows_compared": j.height, "max_diff": bank_diff}
    )
    ef597 = pl.read_csv(k597.WORK / "a1" / "event_free.csv", infer_schema_length=None).filter(
        pl.col("system") == "G4"
    )
    st = sc597.row_stats(y=t.obs, q=sy.q["R"])
    real_diff, n_cmp = 0.0, 0
    for r in ef597.iter_rows(named=True):
        grp = r["group"]
        gsel = (
            np.isin(t.group_of_row, k597.GROUPS_SCORED) if grp == "both groups"
            else t.group_of_row == grp
        )  # fmt: skip
        band = 0 if r["band"] == s.BANDS[0] else 1
        sel = t.free & gsel & (t.band == band)
        assert int(sel.sum()) == int(r["n_rows"]), (grp, r["band"], int(sel.sum()), r["n_rows"])
        for key, scale in (
            ("cover90", 1.0),
            ("width90", 100.0),
            ("crps", 100.0),
            ("pin05", 100.0),
            ("pin95", 100.0),
        ):
            mine = float(st[sel, J[key]].mean() * scale)
            real_diff = max(real_diff, abs(mine - float(r[key])) / max(1.0, abs(float(r[key]))))
        n_cmp += 1
    out.append({"table": "real event-free rows, per group and band", "rows_compared": n_cmp, "max_diff": real_diff})  # fmt: skip
    tab = pl.DataFrame(out)
    print(tab, flush=True)
    assert (tab["max_diff"] <= REPRO_TOL).all(), f"R does not reproduce experiment 597: {tab}"
    return tab


def summarise_bank(*, tab: pl.DataFrame) -> pl.DataFrame:
    """Return means over events per (group, length class, first-fold-pair flag, phase, band, system).

    The 95% intervals come from resampling (month, Primary) clusters.

    The 20 passes of a month reuse the same real rows and forecasts, so events are not independent;
    each cluster is the mean over the passes and events of one Primary in one fold month.
    """
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
                    clusters = grp.group_by("month", "primary").agg(pl.col(metrics).mean())
                    rec["n_clusters"] = clusters.height
                    for mname in metrics:
                        mean, lo, hi = sc._boot_mean(vals=clusters[mname].to_numpy())
                        rec[mname], rec[f"{mname}_lo"], rec[f"{mname}_hi"] = mean, lo, hi
                    out.append(rec)
    return pl.DataFrame(out)


# ---------------------------------------------------------------- the seven counted contrasts and the verdicts


def cluster_means(*, j: pl.DataFrame, col: str) -> np.ndarray:
    """Return the mean of `col` per (month, Primary) cluster of the events in `j`."""
    return j.group_by("month", "primary").agg(pl.col(col).mean())[col].to_numpy()


def paired(
    *, tab: pl.DataFrame, a_: str, b_: str, phase: str, metric: str, lengths: str = "15 and 30 days"
) -> tuple[float, float, float, int, int]:
    """Return mean(a - b), its Bonferroni-level interval, the event count, and the cluster count.

    Events are paired on (month, pass, Primary); the interval resamples (month, Primary) clusters
    (each the mean over passes), because the passes of a month share the real rows and forecasts.
    """
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
    j = x.join(y, on=["month", "pass", "primary"], how="inner").with_columns(
        d=pl.col("a") - pl.col("b")
    )
    cl = cluster_means(j=j, col="d")
    mean, lo, hi = boot_interval(vals=cl)
    return mean, lo, hi, j.height, len(cl)


def _row(*, test: str, contrast: str, res: tuple, margin: float) -> dict:
    mean, lo, hi, n, n_cl = res
    return {
        "test": test, "contrast": contrast, "mean": mean, "lo": lo, "hi": hi,
        "mde": (hi - lo) / 2, "margin": margin, "n_events": n, "n_clusters": n_cl,
    }  # fmt: skip


def arm_family(*, arms: list[str]) -> list[str]:
    """Return the non-placebo arms of `arms`."""
    return [a for a in arms if not a.endswith("-p")]


def follow_cl(*, tab: pl.DataFrame, system: str) -> tuple[np.ndarray, pl.DataFrame]:
    f = tab.filter(
        (pl.col("phase") == "days4-end")
        & (pl.col("band") == BAND)
        & (pl.col("length_days") >= 15)
        & (pl.col("system") == system)
    )
    return cluster_means(j=f, col="follow_q50"), f


def paired_follow(*, tab: pl.DataFrame, a_: str, b_: str) -> tuple[float, float, float, int, int]:
    """Return mean(follow_q50(a) - follow_q50(b)) over events paired on (month, pass, Primary)."""
    return paired(tab=tab, a_=a_, b_=b_, phase="days4-end", metric="follow_q50")


def arm_summary(*, tab: pl.DataFrame, ef: pl.DataFrame, systems: list[str]) -> pl.DataFrame:
    """Return, per system, follow, coverage by phase, band width, and event-free CRPS minus A1."""
    out = []
    for system in systems:
        rec: dict = {"system": system}
        for phase in ("days4-end", "after0-7"):
            f = tab.filter(
                (pl.col("phase") == phase) & (pl.col("band") == BAND)
                & (pl.col("length_days") >= 15) & (pl.col("system") == system)
            )  # fmt: skip
            if f.height == 0:
                continue
            for metric in ("cover90", "width90", "follow_q50"):
                mean, lo, hi = boot_interval(vals=cluster_means(j=f, col=metric))
                (
                    rec[f"{metric}_{phase}"],
                    rec[f"{metric}_{phase}_lo"],
                    rec[f"{metric}_{phase}_hi"],
                ) = mean, lo, hi
            rec["n_events"] = f.height
        for grp in k597.GROUPS_SCORED:
            r = ef.filter(
                (pl.col("group") == grp) & (pl.col("band") == BAND) & (pl.col("system") == system)
            )
            if r.height and "crps_minus_A1" in r.columns and r["crps_minus_A1"][0] is not None:
                rec[f"ef_crps_minus_A1_{grp}"] = float(r["crps_minus_A1"][0])
                rec[f"ef_crps_minus_A1_{grp}_mde"] = float(r["crps_minus_A1_mde"][0])
        out.append(rec)
    return pl.DataFrame(out, infer_schema_length=None)


def contrasts(*, tab: pl.DataFrame, ef: pl.DataFrame, arms: list[str]) -> pl.DataFrame:
    """Return the counted contrasts of the plan (arms that were not run are left out)."""
    have = set(tab["system"].unique().to_list())
    rows = []

    def add(test: str, a_: str, b_: str, margin: float) -> None:
        if a_ in have and b_ in have:
            rows.append(_row(test=test, contrast=f"follow_q50({a_}) - follow_q50({b_})", res=paired_follow(tab=tab, a_=a_, b_=b_), margin=margin))  # fmt: skip

    add("P3 gap lowers follow", "P3", "A1", 0.015)
    add("P4 sustain vs gap only", "P4", "P3", 0.01)
    add("P5 weighted averages vs sustain", "P5", "P4", 0.01)
    add("P7 neighbours vs P5", "P7", "P5", 0.01)
    add("P7 own-plus-sum vs without", "P7", "P7b", 0.005)
    fam = [a for a in arm_family(arms=arms) if a in have]
    if fam:
        best = min(fam, key=lambda a: float(np.mean(follow_cl(tab=tab, system=a)[0])))
        print("best arm by follow:", best, flush=True)
        for phase in ("days4-end", "after0-7"):
            if f"{best}-rs" in have:
                res = paired(tab=tab, a_="R", b_=f"{best}-rs", phase=phase, metric="cover90")
                rows.append(_row(test=f"reach {phase}", contrast=f"cover90(R) - cover90({best}-rs)", res=res, margin=0.01))  # fmt: skip
        add(f"placebo ({best})", best, f"{best}-p", 0.02)
        for grp in k597.GROUPS_SCORED:
            r = ef.filter(
                (pl.col("group") == grp) & (pl.col("band") == BAND) & (pl.col("system") == best)
            )
            if r.height:
                res = (
                    float(r["crps_minus_A1"][0]), float(r["crps_minus_A1_lo"][0]),
                    float(r["crps_minus_A1_hi"][0]), int(r["n_rows"][0]), int(r["n_months"][0]),
                )  # fmt: skip
                rows.append(_row(test=f"event-free CRPS {grp}", contrast=f"CRPS({best}) - CRPS(A1), points of capacity (n_events is rows; n_clusters is month blocks)", res=res, margin=0.03))  # fmt: skip
    return pl.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="")
    p.add_argument("--months", nargs="+", default=list(c.MONTHS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    p.add_argument("--no-bank", action="store_true")
    args = p.parse_args()
    out = ROOT / f"score{args.suffix}"
    out.mkdir(parents=True, exist_ok=True)
    s.N_COUNTED = N_COUNTED
    inp = k597.Inputs(arm="a1")
    print("rebuilt G4 state equals experiment 592's:", inp.check_gate_state(), flush=True)
    t = inp.t
    kk = s.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk)
    sorted_keys = kk[order]
    arms = {}
    for name in args.arms:
        arm = Arm(name=name, suffix="", months=args.months, seeds=args.seeds)
        if arm.available():
            arm.load(t=t, order=order, sorted_keys=sorted_keys)
            arms[name] = arm
        else:
            print("arm", name, "has no complete fits under", arm.work, "- skipped", flush=True)
    sy = Systems(
        inp=inp, arms=arms, months=args.months, seeds=args.seeds,
        factor_source="real" if args.no_bank else "bank",
    )  # fmt: skip
    sy.factors.write_csv(out / "factors.csv")
    ef = event_free(sy=sy)
    ef.write_csv(out / "event_free.csv")
    real_events(sy=sy).write_csv(out / "real_event_phase.csv")
    if args.no_bank:
        return
    tab, repro_bank = score_bank(sy=sy)
    check_reproduction(sy=sy, repro_bank=repro_bank).write_csv(out / "reproduction_597.csv")
    tab.write_csv(out / "bank_event_phase.csv")
    summarise_bank(tab=tab).write_csv(out / "bank_summary.csv")
    arm_summary(tab=tab, ef=ef, systems=sorted(tab["system"].unique().to_list())).write_csv(
        out / "arm_summary.csv"
    )
    contrasts(tab=tab, ef=ef, arms=list(arms)).write_csv(out / "contrasts.csv")
    print("scored", sorted(arms), "->", out, flush=True)


if __name__ == "__main__":
    main()

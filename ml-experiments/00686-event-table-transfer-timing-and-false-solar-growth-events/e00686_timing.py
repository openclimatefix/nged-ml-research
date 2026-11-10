"""Check 1: do the donor and the recipient of a verified transfer start (and end) together?

    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 PYTHONPATH=<folder of 660>:. \
        uv run --no-sync python e00686_timing.py

For each verified transfer (D5/D16 in 2025-12, D3/D22 in 2024-10) and each edge (onset, end), fit
the residual of both Primaries (cleaned power minus experiment 327's arm C forecast) in a window of
+-5 days around the mean of the two nominal edges with, per Primary,
``a + b*t + A * ramp(t0, w) * shape(t)``, where ``shape = (1 - c) + c * f(t) / mean(f)`` and ``f``
is the donor's forecast. Two models: separate onsets (each Primary picks its own ``t0``) and one
shared ``t0``. The sum of squares of each Primary is minimised over ``w`` and ``c`` for every ``t0``
on a half-hour grid, so both models read the same table. The reported difference in fit is
calibrated by a day-block bootstrap under the shared model. A second part injects simultaneous and
lagged transfers into other, event-free windows of the same two Primaries and runs the identical
fit, to show what lag the method reports when the true lag is 0.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Final

import e00660_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00660_ev1_real_edges import OFFSET_2019, verified_rows  # ty: ignore[unresolved-import]

sys.path.insert(0, "ml-experiments/00686-event-table-transfer-timing-and-false-solar-growth-events")
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DAY: Final[int] = data.DAY
HALF: Final[int] = 5 * DAY
"""Half-width of the fit window in half-hours."""
SEARCH: Final[int] = 2 * DAY
"""Largest distance of a fitted onset from the window centre, in half-hours."""
WIDTHS: Final[tuple[int, ...]] = (0, 1, 2, 4)
"""Ramp widths in half-hours (the generator's range is 0 to 4)."""
SHARES: Final[tuple[float, ...]] = (0.0, 0.5, 1.0)
"""Share of the step that follows the donor's daily profile: constant MW, half, fully."""
N_BOOT: Final[int] = 300
N_INJECT: Final[int] = 60
WORK: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00686-event-table-transfer-timing-and-false-solar-growth-events"
)
DONOR: Final[dict[str, str]] = {"D5/D16": "D5", "D3/D22": "D22"}
RECIPIENT: Final[dict[str, str]] = {"D5/D16": "D16", "D3/D22": "D3"}


@dataclass
class Table:
    """Sum of squares and fitted step for every (t0, w, c) of one Primary and the best per t0."""

    sse: np.ndarray
    """Best sum of squares for each t0 offset, shape (n_shift,)."""
    amp: np.ndarray
    """Fitted step for the best (w, c) at each t0 offset."""
    best_wc: list[tuple[int, float]]


def shift_grid() -> np.ndarray:
    """Candidate onsets, as half-hour offsets from the window centre."""
    return np.arange(-SEARCH, SEARCH + 1)


def basis(*, n: int, t0: int, w: int, shape: np.ndarray, reverse: bool) -> np.ndarray:
    """Ramp from 0 to 1 over `w` half-hours after `t0`, times the profile shape."""
    k = np.arange(n) - t0
    r = np.where(k < 0, 0.0, np.where(k >= w, 1.0, (k + 1) / (w + 1)))
    if reverse:  # an end is the onset's step run backwards in time: level before, none after
        r = 1.0 - r
    return r * shape


def fit_table(*, y: np.ndarray, f: np.ndarray, reverse: bool) -> Table:
    """Least squares of ``y ~ a + b*tau + A*basis`` over the whole (t0, w, c) grid."""
    n = len(y)
    centre = n // 2
    fm = np.nanmean(f)
    tau = np.linspace(-1, 1, n)
    base = np.stack([np.ones(n), tau], axis=1)
    shifts = shift_grid()
    best_sse = np.full(len(shifts), np.inf)
    best_amp = np.zeros(len(shifts))
    best_wc: list[tuple[int, float]] = [(0, 0.0)] * len(shifts)
    for c in SHARES:
        shape = (1 - c) + c * f / fm
        for w in WIDTHS:
            b = np.stack(
                [basis(n=n, t0=centre + int(s), w=w, shape=shape, reverse=reverse) for s in shifts]
            )
            x = np.concatenate([np.broadcast_to(base, (len(shifts), n, 2)), b[..., None]], axis=2)
            xtx = np.einsum("gni,gnj->gij", x, x)
            xty = np.einsum("gni,n->gi", x, y)
            beta = np.linalg.solve(xtx + 1e-9 * np.eye(3), xty[..., None])[..., 0]
            resid = y - np.einsum("gni,gi->gn", x, beta)
            sse = (resid**2).sum(axis=1)
            better = sse < best_sse
            for i in np.flatnonzero(better):
                best_wc[i] = (w, c)
            best_amp = np.where(better, beta[:, 2], best_amp)
            best_sse = np.where(better, sse, best_sse)
    return Table(sse=best_sse, amp=best_amp, best_wc=best_wc)


def summarise(*, td: Table, tr: Table) -> dict:
    """Separate and shared onsets, the fit difference, and the implied gap in half-hours."""
    shifts = shift_grid()
    i_d, i_r = int(np.argmin(td.sse)), int(np.argmin(tr.sse))
    joint = td.sse + tr.sse
    i_sh = int(np.argmin(joint))
    return {
        "onset_donor_hh": int(shifts[i_d]),
        "onset_recipient_hh": int(shifts[i_r]),
        "gap_hh": int(shifts[i_r] - shifts[i_d]),
        "onset_shared_hh": int(shifts[i_sh]),
        "sse_separate": float(td.sse[i_d] + tr.sse[i_r]),
        "sse_shared": float(joint[i_sh]),
        "step_donor": float(td.amp[i_d]),
        "step_recipient": float(tr.amp[i_r]),
        "step_donor_shared": float(td.amp[i_sh]),
        "step_recipient_shared": float(tr.amp[i_sh]),
        "share_donor": float(td.best_wc[i_d][1]),
        "share_recipient": float(tr.best_wc[i_r][1]),
        "width_donor": int(td.best_wc[i_d][0]),
        "width_recipient": int(tr.best_wc[i_r][0]),
    }


def lr_stat(*, s: dict, n_total: int) -> float:
    """Twice the log likelihood ratio of separate against shared onsets, from the Gaussian fit."""
    sigma2 = s["sse_separate"] / n_total
    return (s["sse_shared"] - s["sse_separate"]) / sigma2


def window(*, site: str, pair_centre: int, forecasts: dict[str, np.ndarray], power: dict) -> tuple:
    """Residual and donor-profile windows around `pair_centre`, gaps filled by the window median."""
    lo = pair_centre - HALF
    r = power[site][lo : lo + 2 * HALF] - forecasts[site][lo : lo + 2 * HALF]
    ok = np.isfinite(r)
    return np.where(ok, r, np.nanmedian(r)), ok


def fit_pair(
    *, yd: np.ndarray, yr: np.ndarray, fd: np.ndarray, reverse: bool
) -> tuple[dict, Table, Table]:
    """Fit both Primaries and summarise."""
    td = fit_table(y=yd, f=fd, reverse=reverse)
    tr = fit_table(y=yr, f=fd, reverse=reverse)
    return summarise(td=td, tr=tr), td, tr


def bootstrap(
    *,
    yd: np.ndarray,
    yr: np.ndarray,
    fd: np.ndarray,
    reverse: bool,
    s: dict,
    rng: np.random.Generator,
) -> dict:
    """Day-block bootstrap under the shared model (null of the LR) and the separate model (gap)."""
    n = len(yd)
    centre = n // 2

    def fitted(site_amp: float, t0: int, w: int, c: float, y: np.ndarray) -> np.ndarray:
        tau = np.linspace(-1, 1, n)
        shape = (1 - c) + c * fd / np.nanmean(fd)
        x = np.stack(
            [np.ones(n), tau, basis(n=n, t0=t0, w=w, shape=shape, reverse=reverse)], axis=1
        )
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        return x @ beta

    def with_model(
        onset_d: int, onset_r: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fd_fit = fitted(0, centre + onset_d, s["width_donor"], s["share_donor"], yd)
        fr_fit = fitted(0, centre + onset_r, s["width_recipient"], s["share_recipient"], yr)
        return fd_fit, fr_fit, yd - fd_fit, yr - fr_fit

    nday = n // DAY
    lr_null, gap_sep = [], []
    fd0, fr0, rd0, rr0 = with_model(s["onset_shared_hh"], s["onset_shared_hh"])
    fd1, fr1, rd1, rr1 = with_model(s["onset_donor_hh"], s["onset_recipient_hh"])
    for _ in range(N_BOOT):
        pick = rng.integers(0, nday, size=nday)
        for kind in ("null", "sep"):
            fdx, frx, rdx, rrx = (fd0, fr0, rd0, rr0) if kind == "null" else (fd1, fr1, rd1, rr1)
            bd, br = fdx.copy(), frx.copy()
            bd[: nday * DAY] += rdx[: nday * DAY].reshape(nday, DAY)[pick].reshape(-1)
            br[: nday * DAY] += rrx[: nday * DAY].reshape(nday, DAY)[pick].reshape(-1)
            sb, _, _ = fit_pair(yd=bd, yr=br, fd=fd, reverse=reverse)
            if kind == "null":
                lr_null.append(lr_stat(s=sb, n_total=2 * n))
            else:
                gap_sep.append(sb["gap_hh"])
    return {"lr_null": lr_null, "gap_boot": gap_sep}


def real_fits() -> list[dict]:
    """Fit both verified transfers at the onset and the end."""
    info = data.series_info()
    rows = verified_rows()
    rng = np.random.default_rng(686)
    out = []
    power = {s: np.load(data.E327 / f"power/power_{info[s]['id']}.npy").astype(float) for s in info}
    fcst = {s: data.load_forecast(site=s) for s in info}
    for name, ev in rows.items():
        d, r = DONOR[name], RECIPIENT[name]
        for edge_name, k in (("onset", 0), ("end", 1)):
            nominal = {d: ev[d][k], r: ev[r][k]}
            centre = (nominal[d] + nominal[r]) // 2
            yd, okd = window(site=d, pair_centre=centre, forecasts=fcst, power=power)
            yr, okr = window(site=r, pair_centre=centre, forecasts=fcst, power=power)
            fd = fcst[d][centre - HALF : centre + HALF]
            reverse = edge_name == "end"
            s, _, _ = fit_pair(yd=yd, yr=yr, fd=fd, reverse=reverse)
            b = bootstrap(yd=yd, yr=yr, fd=fd, reverse=reverse, s=s, rng=rng)
            lr = lr_stat(s=s, n_total=2 * len(yd))
            null = np.array(b["lr_null"])
            gaps = np.array(b["gap_boot"])
            out.append(
                {
                    "pair": name,
                    "edge": edge_name,
                    "donor": d,
                    "recipient": r,
                    "nominal_gap_hh": int(nominal[r] - nominal[d]),
                    "centre_idx": int(centre),
                    "missing_share_donor": float(1 - okd.mean()),
                    "missing_share_recipient": float(1 - okr.mean()),
                    **s,
                    "lr": float(lr),
                    "lr_null_p95": float(np.percentile(null, 95)),
                    "p_value": float((null >= lr).mean()),
                    "gap_lo": float(np.percentile(gaps, 5)),
                    "gap_hi": float(np.percentile(gaps, 95)),
                    "gap_boot": gaps.tolist(),
                    "lr_null": null.tolist(),
                }
            )
            print(
                name,
                edge_name,
                {k: v for k, v in out[-1].items() if not k.endswith("boot") and k != "lr_null"},
                flush=True,
            )
    return out


def clean_windows(*, site_a: str, site_b: str, rng: np.random.Generator, n: int) -> list[int]:
    """Window centres before 2026-07-01 with no labelled or verified event within 8 days."""
    info = data.series_info()
    table = pl.read_parquet(data.E580 / "events_labelled_status.parquet")
    spans = [
        (r["label_onset_idx"] - OFFSET_2019, r["label_end_idx"] - OFFSET_2019)
        for r in table.iter_rows(named=True)
        if r["label"] in (site_a, site_b)
    ]
    ver = pl.read_parquet(data.E580 / "events_verified.parquet")
    spans += [
        (r[f"onset_{a}_idx"] - OFFSET_2019, r[f"end_{a}_idx"] - OFFSET_2019)
        for r in ver.iter_rows(named=True)
        for a in ("a", "b")
        if r[f"primary_{a}"] in (site_a, site_b)
    ]
    p = {s: np.load(data.E327 / f"power/power_{info[s]['id']}.npy") for s in (site_a, site_b)}
    cands = []
    for c in range(HALF + 2 * DAY, data.GRID_LENGTH - HALF - 2 * DAY, DAY):
        if any(a - 8 * DAY < c < b + 8 * DAY for a, b in spans):
            continue
        if any(np.isnan(p[s][c - HALF : c + HALF]).mean() > 0.05 for s in p):
            continue
        cands.append(c)
    return sorted(rng.choice(cands, size=min(n, len(cands)), replace=False).tolist())


def injection_check(real: list[dict]) -> list[dict]:
    """Inject simultaneous and lagged transfers into other windows; refit with the same method."""
    info = data.series_info()
    rng = np.random.default_rng(6860)
    out = []
    for name in DONOR:
        d, r = DONOR[name], RECIPIENT[name]
        power = {
            s: np.load(data.E327 / f"power/power_{info[s]['id']}.npy").astype(float) for s in (d, r)
        }
        fcst = {s: data.load_forecast(site=s) for s in (d, r)}
        onset = next(x for x in real if x["pair"] == name and x["edge"] == "onset")
        amp_d, amp_r = onset["step_donor_shared"], onset["step_recipient_shared"]
        for centre in clean_windows(site_a=d, site_b=r, rng=rng, n=N_INJECT):
            lo = centre - HALF
            yd0, _ = window(site=d, pair_centre=centre, forecasts=fcst, power=power)
            yr0, _ = window(site=r, pair_centre=centre, forecasts=fcst, power=power)
            fd = fcst[d][lo : lo + 2 * HALF]
            n = len(yd0)
            for share in (0.0, 1.0):
                shape = (1 - share) + share * fd / np.nanmean(fd)
                for lag in (0, 29):
                    t0d = n // 2
                    t0r = t0d + lag
                    yd = yd0 + amp_d * basis(n=n, t0=t0d, w=2, shape=shape, reverse=False)
                    yr = yr0 + amp_r * basis(n=n, t0=t0r, w=2, shape=shape, reverse=False)
                    s, _, _ = fit_pair(yd=yd, yr=yr, fd=fd, reverse=False)
                    out.append(
                        {
                            "pair": name,
                            "true_lag_hh": lag,
                            "true_share": share,
                            "centre_idx": int(centre),
                            "lr": float(lr_stat(s=s, n_total=2 * n)),
                            **{k: s[k] for k in ("gap_hh", "onset_donor_hh", "onset_recipient_hh")},
                        }
                    )
    return out


def main() -> None:
    """Run the real fits and the injection check; write the tables."""
    WORK.mkdir(parents=True, exist_ok=True)
    real = real_fits()
    (WORK / "timing_real.json").write_text(json.dumps(real))
    inj = injection_check(real)
    pl.DataFrame(inj).write_parquet(WORK / "timing_injected.parquet")
    summary = (
        pl.DataFrame(inj)
        .group_by("pair", "true_lag_hh", "true_share")
        .agg(
            pl.col("gap_hh").median().alias("median_gap"),
            pl.col("gap_hh").quantile(0.05).alias("gap_p05"),
            pl.col("gap_hh").quantile(0.95).alias("gap_p95"),
            (pl.col("gap_hh").abs() >= 24).mean().alias("share_gap_24plus"),
            pl.len().alias("n"),
        )
        .sort("pair", "true_lag_hh", "true_share")
    )
    with pl.Config(tbl_rows=40, tbl_cols=20):
        print(summary)


if __name__ == "__main__":
    main()

"""Evidence 1: transition width and step shape of the two verified transfers.

    DATA_PATH_INTERNAL=... PYTHONPATH=<this folder> uv run --no-sync python e00660_ev1_real_edges.py

For each of the four Primaries and each end of its event, fit the residual (cleaned power minus
the lag-free demand forecast of experiment 327) in a window of +-4 days with
``a + b*t + A*ramp(t0, w)``, where the ramp is the generator's own family (0 before ``t0``,
``(k+1)/(w+1)`` for the next ``w`` half-hours, 1 after). The best ``(t0, w)`` is read off a grid;
intervals come from resampling whole days of the fit's residual. Then fit the step shape inside the
event as ``delta(t) = A * ((1 - c) + c * f(t)/mean(f))`` where ``f`` is the donor's forecast, so
``c`` is the share of the step that follows the donor's daily profile.
"""

from __future__ import annotations

import json
from typing import Final

import e00660_data as data
import numpy as np
import polars as pl

DAY: Final[int] = data.DAY
HALF_WINDOW: Final[int] = 4 * DAY
SHIFTS: Final[range] = range(-48, 49)
WIDTHS: Final[range] = range(49)
N_BOOT: Final[int] = 200
OFFSET_2019: Final[int] = int((data.GRID_START - data.LABEL_ORIGIN).total_seconds() // 1800)
DONOR: Final[dict[str, str]] = {"D5/D16": "D5", "D3/D22": "D22"}
"""Donor of each verified transfer (the Primary whose power fell during the event)."""


def ramp_bank(*, n: int, centre: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """All ramp bases ``(G, n)`` for start offsets ``SHIFTS`` around `centre` and widths."""
    t = np.arange(n)
    rows, keys = [], []
    for s in SHIFTS:
        for w in WIDTHS:
            k = t - (centre + s)
            b = np.where(k < 0, 0.0, np.where(k >= w, 1.0, (k + 1) / (w + 1)))
            rows.append(b)
            keys.append((s, w))
    return np.array(rows), keys


def fit_bank(*, y: np.ndarray, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares of ``y ~ a + b*t + A*basis`` per basis; returns SSE and coefficients."""
    g, n = bank.shape
    tau = np.linspace(-1, 1, n)
    x = np.stack([np.ones((g, n)), np.broadcast_to(tau, (g, n)), bank], axis=2)
    xtx = np.einsum("gni,gnj->gij", x, x)
    xty = np.einsum("gni,n->gi", x, y)
    beta = np.linalg.solve(xtx + 1e-9 * np.eye(3), xty[..., None])[..., 0]
    resid = y - np.einsum("gni,gi->gn", x, beta)
    return (resid**2).sum(axis=1), beta


def fit_edge(*, y: np.ndarray, centre: int, rng: np.random.Generator) -> dict:
    """Best offset, ramp width, and step with day-block bootstrap intervals."""
    bank, keys = ramp_bank(n=len(y), centre=centre)
    sse, beta = fit_bank(y=y, bank=bank)
    best = int(np.argmin(sse))
    x_best = np.stack([np.ones(len(y)), np.linspace(-1, 1, len(y)), bank[best]], axis=1)
    fitted = x_best @ beta[best]
    resid = y - fitted
    nday = len(y) // DAY
    blocks = resid[: nday * DAY].reshape(nday, DAY)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, nday, size=nday)
        yb = fitted.copy()
        yb[: nday * DAY] += blocks[pick].reshape(-1)
        s, b = fit_bank(y=yb, bank=bank)
        j = int(np.argmin(s))
        boots.append((keys[j][0], keys[j][1], b[j, 2]))
    boots_a = np.array(boots)
    return {
        "shift_hh": keys[best][0],
        "width_hh": keys[best][1],
        "step": float(beta[best, 2]),
        "width_lo": float(np.percentile(boots_a[:, 1], 5)),
        "width_hi": float(np.percentile(boots_a[:, 1], 95)),
        "shift_lo": float(np.percentile(boots_a[:, 0], 5)),
        "shift_hi": float(np.percentile(boots_a[:, 0], 95)),
        "step_lo": float(np.percentile(boots_a[:, 2], 5)),
        "step_hi": float(np.percentile(boots_a[:, 2], 95)),
        "sse_gain_vs_step": float(
            1 - sse[best] / sse[next(i for i, k in enumerate(keys) if k == (keys[best][0], 0))]
        ),
    }


def verified_rows() -> dict[str, dict]:
    """Onsets and ends (grid indices) of each Primary in the two verified transfers."""
    table = pl.read_parquet(data.E580 / "events_verified.parquet")
    out: dict[str, dict] = {}
    for name, (a, b, _, _) in data.VERIFIED_PAIRS.items():
        row = table.filter(
            (pl.col("primary_a").is_in([a, b])) & (pl.col("primary_b").is_in([a, b]))
        ).to_dicts()
        row = next(r for r in row if str(r["onset_a"])[:4] in {"2024", "2025"})
        out[name] = {
            row["primary_a"]: (row["onset_a_idx"] - OFFSET_2019, row["end_a_idx"] - OFFSET_2019),
            row["primary_b"]: (row["onset_b_idx"] - OFFSET_2019, row["end_b_idx"] - OFFSET_2019),
            "depth": {row["primary_a"]: row["depth_a_mw"], row["primary_b"]: row["depth_b_mw"]},
        }
    return out


def step_shape(
    *, r: np.ndarray, f_donor: np.ndarray, on: int, end: int, rng: np.random.Generator
) -> dict:
    """Constant-versus-proportional share ``c`` of the step inside the event, with an interval."""
    margin = 8
    core = np.arange(on + margin, end - margin)
    pre = np.arange(on - 7 * DAY, on - 4)
    post = np.arange(end + 4, end + 7 * DAY)
    pre_level = np.nanmedian(r[pre])
    post_level = np.nanmedian(r[post])
    frac = (core - core[0]) / max(len(core) - 1, 1)
    base = pre_level + (post_level - pre_level) * frac
    delta = r[core] - base
    f = f_donor[core]
    ok = np.isfinite(delta) & np.isfinite(f)
    delta, f, core_ok = delta[ok], f[ok], core[ok]
    day = (core_ok - core[0]) // DAY

    def solve(d: np.ndarray, ff: np.ndarray) -> tuple[float, float]:
        fm = ff.mean()
        x = np.stack([np.ones_like(ff), ff - fm], axis=1)
        a, b = np.linalg.lstsq(x, d, rcond=None)[0]
        return float(a), float(np.clip(b * fm / a, -1, 2)) if a != 0 else float("nan")

    a, c = solve(delta, f)
    days = np.unique(day)
    cs, amps = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(days, size=len(days))
        idx = np.concatenate([np.flatnonzero(day == d) for d in pick])
        ab, cb = solve(delta[idx], f[idx])
        amps.append(ab)
        cs.append(cb)
    return {
        "mean_step": a,
        "c_proportional": c,
        "c_lo": float(np.nanpercentile(cs, 5)),
        "c_hi": float(np.nanpercentile(cs, 95)),
        "n_days": len(days),
    }


def run() -> dict:
    """Fit every edge and step shape; write the table of numbers."""
    rng = np.random.default_rng(660)
    info = data.series_info()
    rows = verified_rows()
    edges, shapes = [], []
    for name, ev in rows.items():
        donor = DONOR[name]
        f_all = {s: data.load_forecast(site=s) for s in ev if s != "depth"}
        for site in (s for s in ev if s != "depth"):
            power = np.load(data.E327 / f"power/power_{info[site]['id']}.npy").astype(float)
            r = power - f_all[site]
            on, end = ev[site]
            for edge_name, centre in (("onset", on), ("end", end)):
                lo = centre - HALF_WINDOW
                y = r[lo : lo + 2 * HALF_WINDOW]
                if np.isnan(y).mean() > 0.2:
                    continue
                y = np.where(np.isfinite(y), y, np.nanmedian(y))
                y = y if edge_name == "onset" else -y
                res = fit_edge(y=y, centre=HALF_WINDOW, rng=rng)
                edges.append(
                    {
                        "pair": name,
                        "primary": site,
                        "role": "donor" if site == donor else "recipient",
                        "edge": edge_name,
                        "nominal_idx": int(centre),
                        "step_mw": res["step"] * (1 if edge_name == "onset" else -1),
                        **{k: v for k, v in res.items() if k != "step"},
                    }
                )
            for prof_name, f in (("donor_profile", f_all[donor]), ("own_profile", f_all[site])):
                s = step_shape(r=r, f_donor=f, on=on, end=end, rng=rng)
                shapes.append({"pair": name, "primary": site, "profile": prof_name, **s})
    return {"edges": edges, "shapes": shapes}


if __name__ == "__main__":
    out = run()
    data.WORK.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out["edges"]).write_parquet(data.WORK / "ev1_edges.parquet")
    pl.DataFrame(out["shapes"]).write_parquet(data.WORK / "ev1_shapes.parquet")
    (data.WORK / "ev1.json").write_text(json.dumps(out, indent=1))
    with pl.Config(tbl_rows=40, tbl_cols=20, fmt_str_lengths=20):
        print(pl.DataFrame(out["edges"]))
        print(pl.DataFrame(out["shapes"]))

# ruff: noqa: ANN001, D103, E501
"""Experiment 667, prevailing rows, step 2: build the head's real and bank rows for one fold month (CPU).

A copy of experiment 647's `e00647_data.build_real` and experiment 650's `e00650_data.build_bank`,
with the Candidate's quantiles (seed mean of the fold's two seeds, as a fraction of capacity) in
place of K1's. Rows keep the leads from 3 h to below 339 h after the 09:00 network origin, which are
the contract's valid times (12:00 on the origin day to 14 days later). Bank rows come from the
injected passes written by `p667_inject.py`; only the even half-hours inside each event's window
(7 days before the onset to 14 days after the end) are kept, as in experiment 650.

Writes `head_data/real_<month>.npz`, `head_data/bank_<month>.npz`, and
`head_data/events_<month>.parquet` (the generator's events of every pass).

Usage: python p667_data.py MONTH
"""

import os
import sys
import time
from datetime import UTC, datetime
from typing import Final

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00598_signals as s598  # ty: ignore[unresolved-import]
import e00645_score as k645  # ty: ignore[unresolved-import]
import e00647_data as d647  # ty: ignore[unresolved-import]
import e00650_bank as gen  # ty: ignore[unresolved-import]
import e00650_common as c650  # ty: ignore[unresolved-import]
import e00667_export as x667  # ty: ignore[unresolved-import]
import e00667_extend as ext  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

PER: Final[int] = 48
ROOT: Final = x667.ROOT
HEAD_DATA: Final = ROOT / "prevailing" / "head_data"
LEAD_MIN_H: Final[float] = 3.0
LEAD_MAX_H: Final[float] = 339.0
PASSES: Final = tuple(int(p) for p in os.environ.get("P667_PASSES", "").split(",") if p) or tuple(
    range(gen.N_TRAIN_PASSES)
)
"""All 12 training passes of experiment 650's generator; `P667_PASSES` (comma-separated) restricts them for a test."""
GRID_US: Final[int] = int(k578.GRID_START.timestamp() * 1_000_000)
YEAR_END_IDX: Final[int] = int(
    (datetime(2026, 11, 1, tzinfo=UTC) - k578.GRID_START).total_seconds() // 1800
)
"""Events are clipped at the end of the backtest year."""


def next_month_start_idx(*, month: str) -> int:
    """Return the grid index of the first half-hour of the month after `month`, the earliest fold that may use this month's bank."""
    y, m = (int(v) for v in month.split("-"))
    return c650.month_start_idx(month=f"{y + (m == 12)}-{m % 12 + 1:02d}")


def generator_view(*, clean: np.ndarray, month: str) -> np.ndarray:
    """Return `clean` with every half-hour from the next month's start hidden, for the generator's means.

    A proportional shift scales the Primary's own power by its mean over the event. Hiding the later
    half-hours stops that mean reading observed power at or after the first fold that may use this
    month's bank.
    """
    out = clean.copy()
    out[:, next_month_start_idx(month=month) :] = np.nan
    return out


def prim_of() -> dict[int, int]:
    sm = k578.sites()
    return dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))


def seed_mean_rows(*, directory, pattern: str) -> dict[str, np.ndarray]:
    """Return (prim, oi, vi, lead, q) of the seed-mean quantiles of the parquet files `pattern` formats for s=0, 1."""
    frames = [pl.read_parquet(directory / pattern.format(s=s)) for s in x667.SEEDS]
    mean = x667.seed_mean(frames=frames)
    pm = prim_of()
    prim = np.array([pm[int(t)] for t in mean["time_series_id"].to_list()], np.int64)
    oi = ((mean["power_fcst_init_time"].dt.epoch("us") - GRID_US) // 1_800_000_000).to_numpy()
    vi = ((mean["valid_time"].dt.epoch("us") - GRID_US) // 1_800_000_000).to_numpy()
    lead = ((vi - oi) / 2.0).astype(np.float32)
    q = mean.select(x667.QCOLS).to_numpy().astype(np.float32)
    keep = (lead >= LEAD_MIN_H) & (lead < LEAD_MAX_H)
    return {"prim": prim[keep], "oi": oi[keep], "vi": vi[keep], "lead": lead[keep], "q": q[keep]}


def month_table(*, month: str) -> s598.LagFreeTable:
    z = np.load(ROOT / "tables" / "predict" / f"T_{month}_s0.npz")
    return s598.LagFreeTable(median=z["median"], origin_idx=z["origin_idx"])


def previous_month(*, month: str) -> str:
    y, m = (int(v) for v in month.split("-"))
    return f"{y - (m == 1)}-{(m - 2) % 12 + 1:02d}"


def real_rows(*, month: str) -> dict[str, np.ndarray]:
    return seed_mean_rows(
        directory=x667.fits_dir(month=month), pattern=f"{month}_d0_s{{s}}.parquet"
    )


def build_real(*, month: str, clean: np.ndarray, table, nb, flags: np.ndarray, n_days: int) -> None:
    r = real_rows(month=month)
    y = clean[r["prim"], r["vi"]].astype(np.float32)
    # R0's daily residuals read up to 8 days before the origin, which can fall in the previous month
    prev = previous_month(month=month)
    prev_dir = x667.fits_dir(month=prev)
    pool = r
    if prev_dir.exists():
        p = real_rows(month=prev)
        pool = {k: np.concatenate([p[k], r[k]]) for k in r}
    y_pool = clean[pool["prim"], pool["vi"]].astype(np.float32)
    res = {
        name: d647.origin_residual(
            prim=pool["prim"], oi=pool["oi"], vi=pool["vi"], lead=pool["lead"], q50=pool["q"][:, 5],
            obs=y_pool, first=a, last=b, n_days=n_days,
        )[-len(y):]
        for name, (a, b) in d647.R0_SPECS.items()
    }  # fmt: skip
    org_key = r["prim"] * 10_000_000 + r["oi"]
    uniq, org = np.unique(org_key, return_inverse=True)
    o_prim, o_oi = (uniq // 10_000_000).astype(np.int64), (uniq % 10_000_000).astype(np.int64)
    feat, age, lag_age = d647.origin_features(prim=o_prim, oi=o_oi, power_lag=clean, power_sig=clean, table=table, nb=nb)  # fmt: skip
    np.savez(
        HEAD_DATA / f"real_{month}.npz",
        prim=r["prim"].astype(np.int8), oi=r["oi"].astype(np.int32), vi=r["vi"].astype(np.int32),
        lead=r["lead"], q=r["q"], y=y, flagged=flags[r["prim"], r["vi"] // PER],
        verified=np.zeros(len(y), bool), org=org.astype(np.int32), res=res["res"],
        res_star=res["res_star"], feat=feat, age=age, lag_age=lag_age,
        gate=np.zeros(len(uniq), bool), org_prim=o_prim.astype(np.int8), org_oi=o_oi.astype(np.int32),
    )  # fmt: skip
    print(f"{month} real: {len(y)} rows, {len(uniq)} origins", flush=True)


def build_bank(*, month: str, clean: np.ndarray, table, nb, n_days: int) -> None:
    nb_of, cap, is_mw = c650.world()
    ms = c650.month_start_idx(month=month)
    inj = ROOT / f"P7_D50_{month}" / "work" / "inject"
    keys = ("prim", "oi", "vi", "lead", "q", "y", "shift", "pass_", "org", "res", "res_star")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    feats, ages, lag_ages, o_prims, o_ois, o_pass, events_rows = [], [], [], [], [], [], []
    n_org = 0
    for r in PASSES:
        z = seed_mean_rows(directory=inj, pattern=f"{month}_r{r:02d}_d0_s{{s}}.parquet")
        own, mirror, events = gen.build(
            clean=generator_view(clean=clean, month=month), month_idx=ms, month_start_idx=ms, r=r, nb_of=nb_of, capacity=cap,
            is_mw=is_mw, window_end_idx=YEAR_END_IDX,
        )  # fmt: skip
        events_rows += [e.row() for e in events]
        n = clean.shape[1]
        fixed = np.where(np.isfinite(clean), clean + own[:, :n], clean)
        mirrored = np.where(np.isfinite(fixed), fixed + mirror[:, :n], fixed)
        prim, oi, vi, lead, q = z["prim"], z["oi"], z["vi"], z["lead"], z["q"]
        obs = clean[prim, vi].astype(np.float32)
        shift = own[prim, vi].astype(np.float32)
        obs_c = obs + shift
        lo_p = {e.primary: e.onset_idx - 7 * PER for e in events}
        hi_p = {e.primary: e.end_idx + 14 * PER for e in events}
        lo = np.array([lo_p.get(p, 1 << 40) for p in range(16)])[prim]
        hi = np.array([hi_p.get(p, -1) for p in range(16)])[prim]
        residuals = {
            name: d647.origin_residual(
                prim=prim,
                oi=oi,
                vi=vi,
                lead=lead,
                q50=q[:, 5],
                obs=obs_c,
                first=a,
                last=b,
                n_days=n_days,
            )
            for name, (a, b) in d647.R0_SPECS.items()
        }
        keep = (vi % 2 == 0) & (vi >= lo) & (vi < hi)
        key = prim[keep] * 10_000_000 + oi[keep]
        uniq, org = np.unique(key, return_inverse=True)
        o_prim, o_oi = uniq // 10_000_000, uniq % 10_000_000
        feat, age, lag_age = d647.origin_features(prim=o_prim, oi=o_oi, power_lag=fixed, power_sig=mirrored, table=table, nb=nb)  # fmt: skip
        for k, v in (
            ("prim", prim), ("oi", oi), ("vi", vi), ("lead", lead), ("q", q), ("y", obs),
            ("shift", shift), ("pass_", np.full(len(prim), r)), ("res", residuals["res"]),
            ("res_star", residuals["res_star"]),
        ):  # fmt: skip
            parts[k].append(v[keep])
        parts["org"].append(org + n_org)
        feats.append(feat)
        ages.append(age)
        lag_ages.append(lag_age)
        o_prims.append(o_prim)
        o_ois.append(o_oi)
        o_pass.append(np.full(len(uniq), r))
        n_org += len(uniq)
        print(f"{month} r{r:02d}: {int(keep.sum())} rows", flush=True)
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    dt = {"prim": np.int8, "oi": np.int32, "vi": np.int32, "pass_": np.int16, "org": np.int32}
    out = {k: (v.astype(dt[k]) if k in dt else v) for k, v in cat.items()}
    np.savez(
        HEAD_DATA / f"bank_{month}.npz", **out, feat=np.concatenate(feats),
        age=np.concatenate(ages), lag_age=np.concatenate(lag_ages),
        gate=np.zeros(n_org, bool), org_prim=np.concatenate(o_prims).astype(np.int8),
        org_oi=np.concatenate(o_ois).astype(np.int32), org_pass=np.concatenate(o_pass).astype(np.int16),
    )  # fmt: skip
    pl.DataFrame(events_rows).write_parquet(HEAD_DATA / f"events_{month}.parquet")
    print(f"{month} bank: {len(cat['prim'])} rows, {n_org} origins", flush=True)


def main(month: str) -> None:
    t0 = time.time()
    HEAD_DATA.mkdir(parents=True, exist_ok=True)
    ext.redirect()
    clean = k578.clean()
    nb = d647.world()
    table = month_table(month=month)
    flags = k645.ara_flags(clean=clean)
    n_days = clean.shape[1] // PER
    flags = np.concatenate([flags, np.zeros((16, n_days - flags.shape[1]), bool)], axis=1)
    build_real(month=month, clean=clean, table=table, nb=nb, flags=flags, n_days=n_days)
    build_bank(month=month, clean=clean, table=table, nb=nb, n_days=n_days)
    print(f"{month} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

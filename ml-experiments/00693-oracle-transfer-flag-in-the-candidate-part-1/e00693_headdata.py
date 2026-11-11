# ruff: noqa: ANN001, D103, E501
"""Experiment 693: the prevailing head's rows for the quarterly step-8 Candidate (CPU).

A quarterly version of experiment 667's `p667_data.py`. The Candidate's quantiles come from
experiment 677's step-8 `P7+D50` networks (seed mean of seeds 0 to 2, one fit per quarterly
fold) and, for the injected banks, from `e00693_cand.py headbank`. Each month's files hold, besides
experiment 667's arrays, the experiment 681 lag block (`lag_true`, 14 residual lags per origin) and the
oracle and placebo transfer state per origin (`o_in_*`, `o_age_*`: whether a transfer is known to be
in progress, and its age in days, from `e00693_flags.origin_flags`).

Writes `<root>/head_data/real_<month>.npz`, `bank_<month>.npz` (experiment 650's training passes,
months before 2026-05), `events_<month>.parquet`, `bankheld_<month>.npz` (held-out passes 200 to
203), and `bankdec_<fold>.npz` (the experiment 660 generator over the fold's window).

Usage: python e00693_headdata.py KIND NAME   (KIND real|bank|held|dec; NAME a month, or a fold for dec)
"""

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
import e00667_extend as ext  # ty: ignore[unresolved-import]
import e00681_lags as l681  # ty: ignore[unresolved-import]
import e00693_flags as flg
import e00693_real as real693
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

PER: Final[int] = 48
ROOT: Final = ML_EXPERIMENTS_DATA_DIR / "00693-oracle-transfer-flag-in-the-candidate-part-1"
SRC677: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
)
HEAD_DATA: Final = ROOT / "head_data"
FOLDS: Final = ("2025-11", "2026-02", "2026-05")
SEEDS: Final = (0, 1, 2)
QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
GRID_US: Final[int] = int(k578.GRID_START.timestamp() * 1_000_000)
YEAR_END_IDX: Final[int] = int(
    (datetime(2026, 11, 1, tzinfo=UTC) - k578.GRID_START).total_seconds() // 1800
)
CUTOFF_IDX: Final[int] = int(
    (datetime(2026, 7, 1, tzinfo=UTC) - k578.GRID_START).total_seconds() // 1800
)
LEAD_MIN_H: Final[float] = 3.0
LEAD_MAX_H: Final[float] = 339.0
BEFORE_DAYS: Final[int] = 7
AFTER_DAYS: Final[int] = 14
HELD: Final = tuple(range(gen.HELD_OUT_FIRST, gen.HELD_OUT_FIRST + gen.N_HELD_OUT_PASSES))
DEC_PASSES: Final[int] = 48
REAL_WINDOW: Final = (
    int((datetime(2025, 3, 1, tzinfo=UTC) - k578.GRID_START).total_seconds() // 1800),
    CUTOFF_IDX,
)
"""The placebo's roll window for the real transfers: from the start of the clean D15 and D18 history to the scoring cut-off."""


def fold_of(*, month: str) -> str:
    """Return the cut-off month of the quarterly fold that contains `month`."""
    return max(f for f in FOLDS if f <= month)


def month_idx(*, month: str) -> tuple[int, int]:
    """Return the grid indices of the first half-hour of `month` and of the next month."""
    y, m = (int(v) for v in month.split("-"))
    return c650.month_start_idx(month=month), c650.month_start_idx(
        month=f"{y + (m == 12)}-{m % 12 + 1:02d}"
    )


def read_rows(*, files: list) -> dict[str, np.ndarray]:
    """Return (prim, oi, vi, lead, q) of the seed-mean quantiles of the seeds' parquet files, leads
    3 h to under 339 h."""
    cols = ["time_series_id", "power_fcst_init_time", "valid_time"]
    frames = [pl.read_parquet(f).sort(cols) for f in files]
    for f in frames[1:]:
        assert f.select(cols).equals(frames[0].select(cols)), "the seeds hold different rows"
    sm = k578.sites()
    pm = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    prim = np.array([pm[int(t)] for t in frames[0]["time_series_id"].to_list()], np.int64)
    oi = ((frames[0]["power_fcst_init_time"].dt.epoch("us") - GRID_US) // 1_800_000_000).to_numpy()
    vi = ((frames[0]["valid_time"].dt.epoch("us") - GRID_US) // 1_800_000_000).to_numpy()
    lead = ((vi - oi) / 2.0).astype(np.float32)
    q = np.sort(np.mean([f.select(QCOLS).to_numpy() for f in frames], axis=0), axis=1).astype(
        np.float32
    )
    keep = (lead >= LEAD_MIN_H) & (lead < LEAD_MAX_H)
    return {"prim": prim[keep], "oi": oi[keep], "vi": vi[keep], "lead": lead[keep], "q": q[keep]}


def fold_rows(*, fold: str) -> dict[str, np.ndarray]:
    """Return the step-8 Candidate's real-row forecasts of one quarterly fold (seeds 0 to 2)."""
    d = SRC677 / f"P7_D50_{fold}_8w_sp3" / "work" / "fits" / "P5"
    return read_rows(files=[d / f"{fold}_d0_s{s}.parquet" for s in SEEDS])


def fold_table(*, fold: str) -> s598.LagFreeTable:
    z = np.load(SRC677 / "tables_sp3" / "predict" / f"T_{fold}_s0.npz")
    return s598.LagFreeTable(median=z["median"], origin_idx=z["origin_idx"])


def runs_of(*, mask: np.ndarray) -> dict[int, list[tuple[int, int | None]]]:
    """Return per Primary the (onset, end) half-hour indices of the runs of `mask`; end is None for
    a run reaching the grid's end."""
    out: dict[int, list[tuple[int, int | None]]] = {}
    for p in range(mask.shape[0]):
        d = np.diff(np.concatenate([[0], mask[p].astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(starts):
            out[p] = [
                (int(a), None if b >= mask.shape[1] else int(b))
                for a, b in zip(starts, ends, strict=True)
            ]
    return out


def dilate(*, mask: np.ndarray, before: int, after: int) -> np.ndarray:
    """Return, per half-hour, whether `mask` is set within `before` half-hours behind it or `after`
    half-hours ahead of it."""
    c = np.concatenate([np.zeros((mask.shape[0], 1), np.int64), np.cumsum(mask, axis=1)], axis=1)
    n = mask.shape[1]
    i = np.arange(n)
    return (c[:, np.clip(i + 1 + after, 0, n)] - c[:, np.clip(i - before, 0, n)]) > 0


def ended_within(*, active: np.ndarray, days: int) -> np.ndarray:
    c = np.concatenate(
        [np.zeros((active.shape[0], 1), np.int64), np.cumsum(active, axis=1)], axis=1
    )
    lo = np.clip(np.arange(active.shape[1]) + 1 - days * PER, 0, None)
    return ((c[:, np.arange(active.shape[1]) + 1] - c[:, lo]) > 0) & ~active


def oracle_state(
    *,
    active: np.ndarray,
    window: tuple[int, int],
    placebo_key: str,
    seed: int,
    prim: np.ndarray,
    oi: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the oracle and the placebo (in_progress, age_days) of each (Primary, origin) row.

    The placebo rolls every Primary's runs inside `window` by one fixed offset, so lengths,
    overlaps, and counts stay as in the oracle.
    """
    out = {}
    for name, mask in (
        ("oracle", active),
        (
            "placebo",
            flg.roll_mask(
                mask=active,
                offset=flg.placebo_offset(
                    month=placebo_key, seed=seed, window_length=window[1] - window[0]
                ),
                window=window,
            ),
        ),
    ):
        inp, age = flg.origin_flags(onsets=runs_of(mask=mask), prim=prim, oi=oi)
        out[f"o_in_{name}"], out[f"o_age_{name}"] = inp, age
    return out


def origin_block(*, prim, oi, lag_series, sig_series, table, nb) -> dict[str, np.ndarray]:
    feat, age, lag_age = d647.origin_features(
        prim=prim, oi=oi, power_lag=lag_series, power_sig=sig_series, table=table, nb=nb
    )
    return {
        "feat": feat,
        "age": age,
        "lag_age": lag_age,
        "lag_true": l681.lag_block(series=lag_series, median=table.median, prim=prim, oi=oi),
    }


def residuals(*, rows: dict, obs: np.ndarray, n_days: int) -> dict[str, np.ndarray]:
    return {
        name: d647.origin_residual(
            prim=rows["prim"],
            oi=rows["oi"],
            vi=rows["vi"],
            lead=rows["lead"],
            q50=rows["q"][:, 5],
            obs=obs,
            first=a,
            last=b,
            n_days=n_days,
        )
        for name, (a, b) in d647.R0_SPECS.items()
    }


def build_real(*, month: str, clean, nb, flags, n_days: int) -> None:
    fold = fold_of(month=month)
    ms, me = month_idx(month=month)
    pool = fold_rows(fold=fold)
    if (
        fold != FOLDS[0]
    ):  # the residual reads up to 8 days before the month's first origin, which can lie in the previous fold
        prev = fold_rows(fold=FOLDS[FOLDS.index(fold) - 1])
        pool = {k: np.concatenate([prev[k], pool[k]]) for k in pool}
    sel = (pool["oi"] >= ms - 12 * PER) & (pool["oi"] < me)
    pool = {k: v[sel] for k, v in pool.items()}
    in_month = (pool["oi"] >= ms) & (pool["oi"] < me)
    obs_pool = clean[pool["prim"], pool["vi"]].astype(np.float32)
    res = residuals(rows=pool, obs=obs_pool, n_days=n_days)
    r = {k: v[in_month] for k, v in pool.items()}
    y = obs_pool[in_month]
    org_key = r["prim"] * 10_000_000 + r["oi"]
    uniq, org = np.unique(org_key, return_inverse=True)
    o_prim, o_oi = (uniq // 10_000_000).astype(np.int64), (uniq % 10_000_000).astype(np.int64)
    block = origin_block(
        prim=o_prim, oi=o_oi, lag_series=clean, sig_series=clean, table=fold_table(fold=fold), nb=nb
    )
    active = real693.mask_of(
        spans=real693.verified_spans(),
        labels=k578.sites()["label"].to_list(),
        length=clean.shape[1],
    )
    state = oracle_state(
        active=active, window=REAL_WINDOW, placebo_key="2025-11", seed=1, prim=o_prim, oi=o_oi
    )
    np.savez(
        HEAD_DATA / f"real_{month}.npz",
        prim=r["prim"].astype(np.int8), oi=r["oi"].astype(np.int32), vi=r["vi"].astype(np.int32), lead=r["lead"], q=r["q"], y=y,
        flagged=flags[r["prim"], r["vi"] // PER], verified=np.zeros(len(y), bool), org=org.astype(np.int32),
        res=res["res"][in_month], res_star=res["res_star"][in_month], **block, **state,  # ty: ignore[invalid-argument-type]
        gate=np.zeros(len(uniq), bool), org_prim=o_prim.astype(np.int8), org_oi=o_oi.astype(np.int32),
    )  # fmt: skip
    print(f"{month} real: {len(y):,} rows, {len(uniq):,} origins", flush=True)


def bank_part(
    *,
    rows: dict,
    clean,
    fixed,
    mirrored,
    active,
    window,
    placebo_key: str,
    seed: int,
    table,
    nb,
    n_days: int,
    keep_phase: bool,
) -> dict[str, np.ndarray]:
    """Return the head's rows of one injected pass: its rows near an event, even half-hours only,
    with origin-level inputs."""
    prim, oi, vi = rows["prim"], rows["oi"], rows["vi"]
    obs_c = fixed[prim, vi].astype(np.float32)
    res = residuals(rows=rows, obs=obs_c, n_days=n_days)
    near = dilate(
        mask=active, before=AFTER_DAYS * PER, after=BEFORE_DAYS * PER
    )  # kept from 7 days before an onset to 14 days after an end
    keep = (vi % 2 == 0) & near[prim, vi]
    key = prim[keep] * 10_000_000 + oi[keep]
    uniq, org = np.unique(key, return_inverse=True)
    o_prim, o_oi = uniq // 10_000_000, uniq % 10_000_000
    block = origin_block(
        prim=o_prim, oi=o_oi, lag_series=fixed, sig_series=mirrored, table=table, nb=nb
    )
    state = oracle_state(
        active=active, window=window, placebo_key=placebo_key, seed=seed, prim=o_prim, oi=o_oi
    )
    y = clean[prim, vi].astype(np.float32)[keep]
    out = {
        "prim": prim[keep], "oi": oi[keep], "vi": vi[keep], "lead": rows["lead"][keep], "q": rows["q"][keep], "y": y,
        "shift": np.where(np.isfinite(y), fixed[prim, vi][keep] - y, 0.0).astype(np.float32),
        "res": res["res"][keep], "res_star": res["res_star"][keep], "org": org,
        "org_prim": o_prim.astype(np.int8), "org_oi": o_oi.astype(np.int32), "gate": np.zeros(len(uniq), bool),
        **block, **state,
    }  # fmt: skip
    if keep_phase:
        lag_in = state["o_in_oracle"][org]
        tin = active[prim[keep], vi[keep]]
        after = ended_within(active=active, days=AFTER_DAYS)[prim[keep], vi[keep]]
        ph = np.zeros(len(org), np.int8)
        ph[lag_in & tin] = 1
        ph[lag_in & ~tin & after] = 2
        ph[~lag_in & ~tin & after] = 3
        out["phase"] = ph
    return out


def concat_parts(*, parts: list[dict]) -> dict[str, np.ndarray]:
    """Concatenate passes, offsetting each pass's origin index so origins stay unique."""
    out: dict[str, list] = {k: [] for k in parts[0]}
    off = 0
    for p, part in enumerate(parts):
        for k, v in part.items():
            out[k].append(v + off if k == "org" else v)
        out.setdefault("pass_", []).append(np.full(len(part["prim"]), p, np.int16))
        out.setdefault("org_pass", []).append(np.full(len(part["org_prim"]), p, np.int16))
        off += len(part["org_prim"])
    return {k: np.concatenate(v) for k, v in out.items()}


def build_train_bank(*, month: str, kind: str, clean, nb, n_days: int) -> None:
    """Experiment 650's passes for one month: the training passes (`bank`) or the held-out passes (`held`)."""
    fold = fold_of(month=month)
    nb_of, cap, is_mw = c650.world()
    ms, me = month_idx(month=month)
    view = clean.copy()
    view[:, me:] = np.nan
    passes = range(gen.N_TRAIN_PASSES) if kind == "bank" else HELD
    src = ROOT / f"P7_D50_{fold}_headbank" / "work" / "headbank"
    parts, events_rows = [], []
    n = clean.shape[1]
    window = (ms - 10 * PER, min(ms + (20 + 112) * PER, CUTOFF_IDX))
    for r in passes:
        z = dict(np.load(src / f"{'tr' if kind == 'bank' else 'held'}_{month}_r{r:03d}.npz"))
        own, mirror, events = gen.build(
            clean=view,
            month_idx=ms,
            month_start_idx=ms,
            r=r,
            nb_of=nb_of,
            capacity=cap,
            is_mw=is_mw,
            window_end_idx=YEAR_END_IDX,
        )
        events_rows += [e.row() for e in events]
        fixed = np.where(np.isfinite(clean), clean + own[:, :n], clean)
        mirrored = np.where(np.isfinite(fixed), fixed + mirror[:, :n], fixed)
        active = flg.active_from_shift(shift=own[:, :n])
        parts.append(
            bank_part(
                rows=z,
                clean=clean,
                fixed=fixed,
                mirrored=mirrored,
                active=active,
                window=window,
                placebo_key=month,
                seed=r,
                table=fold_table(fold=fold),
                nb=nb,
                n_days=n_days,
                keep_phase=True,
            )
        )
        print(f"{month} {kind} r{r:02d}: {len(parts[-1]['prim']):,} rows", flush=True)
    cat = concat_parts(parts=parts)
    dt = {"prim": np.int8, "oi": np.int32, "vi": np.int32, "org": np.int32}
    np.savez(
        HEAD_DATA / (f"bank_{month}.npz" if kind == "bank" else f"bankheld_{month}.npz"),
        **{k: (v.astype(dt[k]) if k in dt else v) for k, v in cat.items()},  # ty: ignore[invalid-argument-type]
    )
    if kind == "bank":
        pl.DataFrame(events_rows).write_parquet(HEAD_DATA / f"events_{month}.parquet")


def build_dec(*, fold: str, clean, nb, n_days: int) -> None:
    """The experiment 660 generator's passes over the fold's whole window (bank F's deciding bank)."""
    src = ROOT / f"P7_D50_{fold}_headbank" / "work" / "headbank"
    parts = []
    for r in range(DEC_PASSES):
        z = dict(np.load(src / f"dec_r{r:02d}.npz"))
        w = np.load(src / f"dec_r{r:02d}_world.npz")
        parts.append(
            bank_part(
                rows=z,
                clean=clean,
                fixed=w["fixed"],
                mirrored=w["fixed"],
                active=w["own_active"],
                window=(int(w["start"]), int(w["stop"])),
                placebo_key=fold,
                seed=r,
                table=fold_table(fold=fold),
                nb=nb,
                n_days=n_days,
                keep_phase=True,
            )
        )
        print(f"{fold} dec r{r:02d}: {len(parts[-1]['prim']):,} rows", flush=True)
    cat = concat_parts(parts=parts)
    dt = {"prim": np.int8, "oi": np.int32, "vi": np.int32, "org": np.int32}
    np.savez(
        HEAD_DATA / f"bankdec_{fold}.npz",
        **{k: (v.astype(dt[k]) if k in dt else v) for k, v in cat.items()},  # ty: ignore[invalid-argument-type]
    )


def main(kind: str, name: str) -> None:
    t0 = time.time()
    HEAD_DATA.mkdir(parents=True, exist_ok=True)
    ext.redirect()
    clean = k578.clean()
    nb = d647.world()
    n_days = clean.shape[1] // PER
    if kind == "real":
        flags = k645.ara_flags(clean=clean)
        flags = np.concatenate([flags, np.zeros((16, n_days - flags.shape[1]), bool)], axis=1)
        build_real(month=name, clean=clean, nb=nb, flags=flags, n_days=n_days)
    elif kind in {"bank", "held"}:
        build_train_bank(month=name, kind=kind, clean=clean, nb=nb, n_days=n_days)
    elif kind == "dec":
        build_dec(fold=name, clean=clean, nb=nb, n_days=n_days)
    else:
        raise SystemExit(f"unknown kind {kind}")
    print(f"{kind} {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

# ruff: noqa: D103, E501
"""Experiment 647, stage 2: build the head's rows and origin-level inputs for one month (CPU).

Writes `head_data/real_<month>.npz` and `head_data/bank_<month>.npz` in the data folder.

A row is one (Primary, origin, valid time) with a lead of 24 h or more. The stored K1 quantiles are
the rows' NRA output and are never changed. Real rows come from the stored roster export; bank rows
come from step 3's rebuilt K1 on experiment 581's injected bank (`k1_bank/rows_<month>_r<pass>.npz`).

The head's inputs are origin-level (one vector per Primary and origin, see `origin_features`):
experiment 606's `gap`, `ew`, and `nb` columns read at the newest allowed lag (origin minus 24 h) and a
week lag; the age of the current departure in days; the age of the freshest finite lag in hours; and
the G4 state. Row-level inputs (lead, K1 band widths) are added at fit time. No date or calendar
column enters.

R0 residuals are formed here too, for both row kinds (the same formula as `e00647_r0`, in numpy):
`res` is the main specification (days 2 to 8, lead 24 to 48 h), `res_star` is R0* (days 1 to 7).

Usage: python e00647_data.py MONTH
"""

import sys
import time
from datetime import UTC, datetime
from typing import Final

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00598_signals as s598  # ty: ignore[unresolved-import]
import e00606_signals as s606  # ty: ignore[unresolved-import]
import e00645_score as k645  # ty: ignore[unresolved-import]
import e00647_common as c647  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

PER: Final[int] = 48
D = ML_EXPERIMENTS_DATA_DIR
DATA: Final = c647.WORK / "head_data"
BANK_ROWS: Final = c647.WORK / "k1_bank"
T_DIRS: Final = (
    "00598-nra-gate-against-signals-as-inputs-at-equal-band-width",
    "00610-nra-neighbour-summary-lag-encoder-inputs-iteration",
    "00613-nra-roster-contrasts-with-intervals-five-seeds",
)
GROUPS: Final = ("gap", "ew", "nb")
COLUMNS: Final = s606.columns_of(groups=GROUPS)
FEATURE_NAMES: Final = tuple(s606.ALL_NAMES[i] for i in COLUMNS)
AGE_CAP: Final[int] = 120
AGE_THETA: Final[float] = 0.05
R0_SPECS: Final = {"res": (2, 8), "res_star": (1, 7)}
"""Residual name -> (first, last) valid day before the origin day; both use the day-ahead forecast (lead 24 to 48 h)."""


def world() -> s598.Neighbours:
    """Return the Primaries' electrical neighbours and capacities (copied from experiment 606's `world`)."""
    import e00400_blend as b400  # ty: ignore[unresolved-import]

    sm = k578.sites()
    labels = sm["label"].to_list()
    lab2p = {lab: i for i, lab in enumerate(labels)}
    of: dict[int, list[int]] = {i: [] for i in range(len(labels))}
    for a, v in b400.nged_neighbours().items():
        if a in lab2p:
            of[lab2p[a]] = [lab2p[x] for x in v if x in lab2p]
    return s598.Neighbours(of=of, capacity=sm["capacity_mw"].to_numpy().astype(np.float64))


def lag_free_table(*, month: str) -> s598.LagFreeTable:
    """Return the lag-free table T of seed 0 for `month`."""
    f = next(p for d_ in T_DIRS if (p := D / d_ / "tables" / f"T_{month}_s0.npz").exists())
    z = np.load(f)
    return s598.LagFreeTable(median=z["median"], origin_idx=z["origin_idx"])


def origin_residual(
    *, prim: np.ndarray, oi: np.ndarray, vi: np.ndarray, lead: np.ndarray, q50: np.ndarray,
    obs: np.ndarray, first: int, last: int, n_days: int,
) -> np.ndarray:  # fmt: skip
    """Return per row the R0 residual of its (Primary, origin day): NaN where no daily residual exists.

    The daily residual of a valid UTC day is the mean of `obs - q50` over rows with a lead of 24 to
    below 48 h; the origin's residual is the mean of the daily residuals over the days `origin_day -
    last` to `origin_day - first`.
    """
    ok = (lead >= c647.R0_LEAD_H[0]) & (lead < c647.R0_LEAD_H[1]) & np.isfinite(obs)
    day = vi[ok] // PER
    key = prim[ok].astype(np.int64) * n_days + day
    tot = np.bincount(key, weights=(obs[ok] - q50[ok]).astype(np.float64), minlength=16 * n_days)
    cnt = np.bincount(key, minlength=16 * n_days)
    tot, cnt = tot.reshape(16, n_days), cnt.reshape(16, n_days)
    daily = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    od = oi // PER
    s = np.zeros(len(prim))
    n = np.zeros(len(prim))
    for k in range(first, last + 1):
        dd = od - k
        good = dd >= 0
        v = np.where(good, daily[prim, np.clip(dd, 0, None)], np.nan)
        f = np.isfinite(v)
        s += np.where(f, v, 0.0)
        n += f
    return np.where(n >= 1, s / np.maximum(n, 1), np.nan).astype(np.float32)


def departure_age(*, daily: np.ndarray, prim: np.ndarray, a0: np.ndarray) -> np.ndarray:
    """Return the age in days of the departure at the lag index `a0`: consecutive same-signed daily mean gaps above 5% of capacity.

    Zero where the daily mean gap at `a0` is missing or at most 5%. Capped at `AGE_CAP` days.
    """
    d0 = daily[prim, a0]
    sgn = np.sign(d0)
    alive = np.isfinite(d0) & (np.abs(d0) > AGE_THETA)
    age = alive.astype(np.float32)
    for k in range(1, AGE_CAP):
        i = a0 - PER * k
        prev = np.where(i >= 0, daily[prim, np.clip(i, 0, None)], np.nan)
        alive = alive & np.isfinite(prev) & (np.sign(prev) == sgn) & (np.abs(prev) > AGE_THETA)
        age += alive
    return age


def origin_features(
    *, prim: np.ndarray, oi: np.ndarray, power_lag: np.ndarray, power_sig: np.ndarray,
    table: s598.LagFreeTable, nb: s598.Neighbours,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Return (features, age in days, freshest-lag age in hours) per (Primary, origin) row."""
    day_idx = (oi - PER).astype(np.int64)
    week_idx = (oi - 7 * PER).astype(np.int64)
    none = np.zeros(len(prim), bool)
    block = s606.signal_columns(
        prim=prim, origin_idx=oi, day_idx=day_idx, week_idx=week_idx, day_missing=none,
        week_missing=none, power_lag=power_lag, power_sig=power_sig, table=table, nb=nb,
        weather_origin=np.zeros(power_lag.shape, np.int64), columns=COLUMNS,
    )  # fmt: skip
    res = (power_lag - table.median).astype(np.float32)
    daily = s598.daily_mean_array(res=res)
    age = departure_age(daily=daily, prim=prim, a0=day_idx)
    fin = np.isfinite(power_lag)
    last = np.maximum.accumulate(np.where(fin, np.arange(power_lag.shape[1]), -1), axis=1)
    lag_age_h = ((oi - last[prim, day_idx]) / 2.0).astype(np.float32)
    return block, age, lag_age_h


def month_start_idx(*, month: str) -> int:
    t = datetime.fromisoformat(month + "-01").replace(tzinfo=UTC)
    return int((t - k578.GRID_START).total_seconds() // 1800)


def build_real(*, month: str, clean: np.ndarray, table: s598.LagFreeTable, nb: s598.Neighbours,
               flags: np.ndarray, n_days: int) -> None:  # fmt: skip
    full = c647.load_export(network="K1")
    sm = k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))

    def arrays(d: pl.DataFrame) -> tuple:
        return (
            np.array([prim_of[int(t)] for t in d["time_series_id"].to_list()], np.int64),
            d["oi"].to_numpy(), d["vi"].to_numpy(), d["lead_hours"].to_numpy(),
            d.select(c647.QNAMES).to_numpy().astype(np.float32),
            d["observed_power"].to_numpy().astype(np.float32),
        )  # fmt: skip

    # R0's daily residuals pool the rows of all five months (a valid day can sit in the previous month's export)
    pa, oa, va, la, qa, ya = arrays(full)
    res_full = {
        name: origin_residual(prim=pa, oi=oa, vi=va, lead=la, q50=qa[:, 5], obs=ya, first=a,
                              last=b, n_days=n_days)
        for name, (a, b) in R0_SPECS.items()
    }  # fmt: skip
    m_sel = (full["month"] == month).to_numpy()
    df = full.filter(pl.col("month") == month)
    prim_all, oi_all, vi_all, lead_all, q_all, y_all = arrays(df)
    res = {k: v[m_sel] for k, v in res_full.items()}
    keep = lead_all >= 24.0
    org_key = prim_all * 10_000_000 + oi_all
    uniq, org_all = np.unique(org_key, return_inverse=True)
    o_prim, o_oi = (uniq // 10_000_000).astype(np.int64), (uniq % 10_000_000).astype(np.int64)
    feat, age, lag_age = origin_features(prim=o_prim, oi=o_oi, power_lag=clean, power_sig=clean,
                                         table=table, nb=nb)  # fmt: skip
    gate = np.zeros(len(uniq), bool)
    np.logical_or.at(gate, org_all, df["gate_on"].fill_null(False).to_numpy())
    sel = keep
    np.savez(
        DATA / f"real_{month}.npz",
        prim=prim_all[sel].astype(np.int8), oi=oi_all[sel].astype(np.int32),
        vi=vi_all[sel].astype(np.int32), lead=lead_all[sel].astype(np.float32), q=q_all[sel],
        y=y_all[sel], event_free=df["event_free"].to_numpy()[sel],
        verified=df["in_verified_event"].to_numpy()[sel],
        flagged=flags[prim_all[sel], vi_all[sel] // PER], org=org_all[sel].astype(np.int32),
        res=res["res"][sel], res_star=res["res_star"][sel],
        feat=feat, age=age, lag_age=lag_age, gate=gate, org_prim=o_prim.astype(np.int8),
        org_oi=o_oi.astype(np.int32),
    )  # fmt: skip
    print(f"{month} real: {int(sel.sum())} rows, {len(uniq)} origins", flush=True)


def build_bank(*, month: str, clean: np.ndarray, table: s598.LagFreeTable, nb: s598.Neighbours,
               flags_fn, n_days: int) -> None:  # fmt: skip
    ms = month_start_idx(month=month)
    end = ms + bank.END_DAY * PER
    parts: dict[str, list[np.ndarray]] = {k: [] for k in
        ("prim", "oi", "vi", "lead", "q", "y", "shift", "pass_", "org", "res", "res_star", "vin_wide")}  # fmt: skip
    feats, ages, lag_ages, gates, o_prims, o_ois, o_pass = [], [], [], [], [], [], []
    n_org = 0
    for r in range(bank.N_PASSES):
        z = np.load(BANK_ROWS / f"rows_{month}_r{r:02d}.npz")
        prim, oi, vi = z["prim"].astype(np.int64), z["origin_idx"].astype(np.int64), z["idx"].astype(np.int64)
        lead, obs, q = z["lead"], z["obs"], z["k1_inj"]
        assert np.array_equal(obs, clean[prim, vi].astype(np.float32), equal_nan=True)
        signed = np.array([bank.event(primary=p, r=r)[0] for p in range(16)])
        length = np.array([bank.event(primary=p, r=r)[1] for p in range(16)])
        on = end - length * PER
        shift = np.where((vi >= on[prim]) & (vi < end), signed[prim], 0.0).astype(np.float32)
        obs_c = obs + shift
        fixed, _ = bank.contaminate(clean=clean, month_start_idx=ms, r=r)
        sh = np.where(np.isfinite(fixed) & np.isfinite(clean), fixed - clean, 0.0)
        choice = {p: qs[(r + p) % len(qs)] for p, qs in nb.of.items() if qs}
        mirrored = s598.mirrored_power(base=fixed, shift=sh, choice=choice, capacity=nb.capacity)
        key = prim * 10_000_000 + oi
        uniq, org = np.unique(key, return_inverse=True)
        o_prim, o_oi = uniq // 10_000_000, uniq % 10_000_000
        f, age, lag_age = origin_features(prim=o_prim, oi=o_oi, power_lag=fixed, power_sig=mirrored,
                                          table=table, nb=nb)  # fmt: skip
        wide = (z["k1_inj"] != z["vin"]).any(axis=1)
        gate = np.zeros(len(uniq), bool)
        np.logical_or.at(gate, org, wide)
        for name, (a, b) in R0_SPECS.items():
            parts[name].append(origin_residual(prim=prim, oi=oi, vi=vi, lead=lead, q50=q[:, 5],
                                               obs=obs_c, first=a, last=b, n_days=n_days))  # fmt: skip
        keep = lead >= 24.0
        for k, v in (("prim", prim), ("oi", oi), ("vi", vi), ("lead", lead), ("q", q), ("y", obs),
                     ("shift", shift), ("pass_", np.full(len(prim), r)), ("org", org + n_org)):  # fmt: skip
            parts[k].append(v)
        parts["vin_wide"].append(wide)
        parts.setdefault("keep", []).append(keep)
        feats.append(f)
        ages.append(age)
        lag_ages.append(lag_age)
        gates.append(gate)
        o_prims.append(o_prim)
        o_ois.append(o_oi)
        o_pass.append(np.full(len(uniq), r))
        n_org += len(uniq)
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    sel = cat.pop("keep")
    dt = {"prim": np.int8, "oi": np.int32, "vi": np.int32, "pass_": np.int8, "org": np.int32}
    out = {k: (v[sel].astype(dt[k]) if k in dt else v[sel]) for k, v in cat.items() if k != "vin_wide"}
    np.savez(
        DATA / f"bank_{month}.npz", **out, feat=np.concatenate(feats), age=np.concatenate(ages),
        lag_age=np.concatenate(lag_ages), gate=np.concatenate(gates),
        org_prim=np.concatenate(o_prims).astype(np.int8), org_oi=np.concatenate(o_ois).astype(np.int32),
        org_pass=np.concatenate(o_pass).astype(np.int8),
    )  # fmt: skip
    print(f"{month} bank: {int(sel.sum())} rows, {n_org} origins", flush=True)


def main(month: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    clean = k578.clean()
    nb = world()
    table = lag_free_table(month=month)
    flags = k645.ara_flags(clean=clean)
    n_days = clean.shape[1] // PER
    flags = np.concatenate([flags, np.zeros((16, n_days - flags.shape[1]), bool)], axis=1)
    build_real(month=month, clean=clean, table=table, nb=nb, flags=flags, n_days=n_days)
    build_bank(month=month, clean=clean, table=table, nb=nb, flags_fn=None, n_days=n_days)
    print(f"{month} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

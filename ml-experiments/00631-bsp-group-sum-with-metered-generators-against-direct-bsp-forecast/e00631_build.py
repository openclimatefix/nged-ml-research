# ruff: noqa: E501, PLR0912, PLR0915, RUF005
"""Build the BSP sums of Primary and metered-farm scenarios for experiment 631.

For the K1 demand network and each daily run of the development months, writes
`work/runs/<date>.npz` with, for the two BSPs (Boston, Skegness):

- `sums` (4 arms J0, J2, J3, J5; 2 variants without/with the farms; 2 BSPs; 51; 480) scenario sums
  in MW of the flow into the BSP. The with-farms variant is Primaries minus farm output; J2 is NaN
  with the farms, because the daily runs hold no member weather at the farms;
- `meter` (2, 480), `prim_obs_sum` (2, 480), `farm_obs_sum` (2, 480): the BSP meter, the observed sum
  of the Primaries beneath each BSP, and the observed farm output beneath it (MW, NaN if any is absent);
- `direct` (5 sources, 2, 480, 11): the stored direct BSP forecasts.

It also writes `work/offsets.npz`, the remainder terms (constant, and by half-hour of day) fitted
on the observed history before each test month, without and with the farms.

All 25 nodes (16 Primaries then 9 farms) share one set of Schaake dates and one copula. Arms:

- J0: independent random order for each node and half-hour.
- J2: ensemble copula coupling by member 2 m temperature (shortwave in daylight) for the Primaries.
- J3: Schaake shuffle with the ranks of 51 observed 480-step paths nearest in day of year.
- J5: Gaussian copula on probability-integral-transform scores, AR(1) in the half-hour, fitted
  leaving the test month out (no ensemble mixing).

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00631-*/env.sh
nice -n 19 systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00631-*/e00631_build.py
```
"""

import hashlib
import os
from datetime import UTC, datetime, timedelta

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00434_score as s434  # noqa: E402  # ty: ignore[unresolved-import]
import e00629_build as b629  # noqa: E402  # ty: ignore[unresolved-import]
import e00629_common as c629  # noqa: E402  # ty: ignore[unresolved-import]
import e00631_common as c  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

M = c.N_MEMBERS
S = c.N_STEPS
STEP = c.STEP
NF = len(c.FARMS)
MAX_MISSING = {"prim": 0.05, "farm": 0.25}


def farm_scales(*, farms: pl.DataFrame, power: pl.DataFrame) -> np.ndarray:
    """Return each farm's scoring capacity in MW: the ratio of cleaned power to the stored fraction."""
    out = np.zeros(NF)
    for i, (lab, sid) in enumerate(zip(farms["label"], farms["time_series_id"], strict=True)):
        cand = c.WIND_CANDIDATE if lab.startswith("W") else c.SOLAR_CANDIDATE
        x = (
            pl.concat([c.read_farm(candidate=cand, month=m) for m in c.MONTHS])
            .filter((pl.col("series") == lab) & (pl.col("actual") > 0.3))
            .select("valid_time", "actual")
            .unique(subset="valid_time")
        )
        j = x.join(
            power.filter(pl.col("time_series_id") == sid).select(
                pl.col("time").alias("valid_time"), "power"
            ),
            on="valid_time",
        )
        ratio = (j["power"] / j["actual"]).to_numpy()
        if ratio.size < 100 or np.std(ratio) > 0.01 * abs(np.median(ratio)):
            msg = f"{lab}: scoring capacity is not constant ({ratio.size} rows)"
            raise ValueError(msg)
        out[i] = float(np.median(ratio))
    return out


def night_slots(*, solar: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return, per solar farm, a (13, 48) boolean array: True where (month, half-hour) is night."""
    out = {}
    for lab in solar["series"].unique().to_list():
        sub = solar.filter(pl.col("series") == lab).select("valid_time").unique()
        sub = sub.with_columns(
            m=pl.col("valid_time").dt.month(),
            h=pl.col("valid_time").dt.hour() * 2 + pl.col("valid_time").dt.minute() // 30,
            d=pl.col("valid_time").dt.date(),
        )
        days = sub.group_by("m").agg(pl.col("d").n_unique().alias("n_days"))
        cnt = sub.group_by("m", "h").agg(pl.col("d").n_unique().alias("n"))
        night = np.ones((13, 48), dtype=bool)
        full = dict(zip(days["m"], days["n_days"], strict=True))
        for m, h, n in cnt.iter_rows():
            night[m, h] = n / full[m] < c.NIGHT_PRESENCE
        out[lab] = night
    return out


def farm_run(
    *,
    init: datetime,
    wind: pl.DataFrame,
    solar: pl.DataFrame,
    scales: np.ndarray,
    night: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the farms' stored quantiles (9, 480, 11) in MW for a 09:00 run; NaN where absent."""
    q = np.full((NF, S, 11), np.nan)
    cols = list(c629.QUANTILES)
    d0 = init.replace(hour=0, minute=0)
    for i, lab in enumerate(c.FARMS):
        if lab.startswith("W"):
            rows = wind.filter((pl.col("series") == lab) & (pl.col("init_time") == init))
            step = np.rint(rows["lead_hours"].to_numpy() * 2).astype(int) - 1
            ok = (step >= 0) & (step < S)
            q[i, step[ok]] = rows.select(cols).to_numpy()[ok] * scales[i]
        else:
            for k, day in enumerate(
                (d0, d0 - timedelta(days=1))
            ):  # same-day run first, then the previous day's
                rows = solar.filter((pl.col("series") == lab) & (pl.col("init_time") == day))
                step = ((rows["valid_time"] - init).dt.total_minutes() // 30).to_numpy().astype(
                    int
                ) - 1
                ok = (step >= 0) & (step < S)
                fresh = ok & (np.isnan(q[i, np.clip(step, 0, S - 1), 0]) if k else ok)
                q[i, step[fresh]] = rows.select(cols).to_numpy()[fresh] * scales[i]
            t = [init + (j + 1) * STEP for j in range(S)]
            is_night = np.array([night[lab][x.month, x.hour * 2 + x.minute // 30] for x in t])
            q[i, is_night & np.isnan(q[i, :, 0])] = 0.0
    return np.sort(q, axis=-1)


def farm_grid(*, ids: list[int], power: pl.DataFrame, start: datetime) -> np.ndarray:
    """Return cleaned farm power (len(ids), n) in MW on the half-hourly grid from `start`."""
    n = int((c.END_OF_DEV - start) / STEP)
    out = np.full((len(ids), n), np.nan)
    for p, sid in enumerate(ids):
        sub = power.filter(pl.col("time_series_id") == sid)
        idx = ((sub["time"] - start).dt.total_minutes() // 30).to_numpy()
        ok = (idx >= 0) & (idx < n)
        out[p, idx[ok]] = sub["power"].to_numpy()[ok]
    return out


def schaake_ranks(*, init: datetime, grid: np.ndarray, start: datetime, n_prim: int) -> np.ndarray:
    """Return Schaake ranks (25, 480, 51): the dates are the 51 nearest in day of year with enough data."""
    last = init - c629.SCHAAKE_BUFFER
    t0 = c629.SCHAAKE_FIRST.replace(hour=init.hour, minute=init.minute)
    tol = np.array(
        [MAX_MISSING["prim"]] * n_prim + [MAX_MISSING["farm"]] * (grid.shape[0] - n_prim)
    )
    cand = []
    while t0 <= last:
        i = int((t0 - start) / STEP) + 1
        path = grid[:, i : i + S]
        if path.shape[1] == S and (np.isnan(path).mean(axis=1) <= tol).all():
            doy = abs((t0.timetuple().tm_yday - init.timetuple().tm_yday + 183) % 365 - 183)
            cand.append((doy, t0, path))
        t0 += timedelta(days=1)
    if not cand:
        msg = f"{init}: no Schaake date qualifies"
        raise ValueError(msg)
    cand.sort(key=lambda x: (x[0], x[1]))
    x = np.arange(S)
    paths = []
    for k in range(M):
        out = cand[k % len(cand)][2].copy()
        for p in range(out.shape[0]):
            bad = np.isnan(out[p])
            if bad.any():
                out[p, bad] = np.interp(x[bad], x[~bad], out[p, ~bad])
        paths.append(out)
    rng = np.random.default_rng(int(hashlib.sha256(init.isoformat().encode()).hexdigest()[:8], 16))
    return s434.ranks_by_key(key=np.stack(paths).transpose(1, 2, 0), rng=rng)


def fit_offsets(
    *, hist: dict[str, np.ndarray], start: datetime, months_first_run: dict[str, datetime]
) -> dict[str, np.ndarray]:
    """Return remainder offsets per test month: const (2 variants, 2 BSPs) and by half-hour (2, 2, 48).

    The remainder is the mean of the observed meter minus the observed sum of the Primaries (variant
    0) or minus the Primaries less the farms' output (variant 1), over every half-hour from the first
    grid point to 16 days before the test month's first run.
    """
    n = hist["meter"].shape[1]
    t = np.array([start + i * STEP for i in range(n)])
    hod = np.array([x.hour * 2 + x.minute // 30 for x in t])
    out: dict[str, np.ndarray] = {}
    for month, first in months_first_run.items():
        before = t < first - timedelta(days=c.OFFSET_BUFFER_DAYS)
        const = np.full((2, 2), np.nan)
        tod = np.full((2, 2, 48), np.nan)
        counts = np.zeros((2, 2))
        for g in range(2):
            prim, farm, meter = hist["prim"][g], hist["farm"][g], hist["meter"][g]
            for v in range(2):
                comb = prim + (c.POWER_SIGN * farm if v else 0.0)
                resid = meter - comb
                ok = before & np.isfinite(resid)
                counts[v, g] = ok.sum()
                const[v, g] = resid[ok].mean()
                for h in range(48):
                    sel = ok & (hod == h)
                    tod[v, g, h] = resid[sel].mean() if sel.any() else const[v, g]
        out[f"const_{month}"] = const
        out[f"tod_{month}"] = tod
        out[f"n_{month}"] = counts
    return out


def main() -> None:
    """Write the observed-history offsets and one table per daily run."""
    prim = c629.primaries()
    pids = prim["time_series_id"].to_list()
    farms = c.farm_table()
    fids = farms["time_series_id"].to_list()
    nodes = c629.group_nodes(prim=prim)
    group_prim = {g: nodes[g] for g in c.BSP_GROUPS}
    group_farm = {g: [i for i, gg in enumerate(farms["group"]) if gg == g] for g in c.BSP_GROUPS}
    print({g: (len(group_prim[g]), len(group_farm[g])) for g in c.BSP_GROUPS}, flush=True)
    if sum(len(v) for v in group_farm.values()) != NF:
        msg = "a farm is under neither BSP"
        raise ValueError(msg)
    n_prim = len(pids)
    ids_all = pids + fids
    start = c629.SCHAAKE_FIRST
    n = int((c.END_OF_DEV - start) / STEP)
    from studies.power import scan_power

    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(ids_all + list(c.BSP_GROUPS))
            & (pl.col("time") >= start)
            & (pl.col("time") < c.END_OF_DEV)
        )
        .collect()
    )
    wind = pl.concat([c.read_farm(candidate=c.WIND_CANDIDATE, month=m) for m in c.MONTHS]).filter(
        pl.col("series").str.starts_with("W")
    )
    solar = pl.concat([c.read_farm(candidate=c.SOLAR_CANDIDATE, month=m) for m in c.MONTHS]).filter(
        pl.col("series").str.starts_with("Solar")
    )
    night = night_slots(solar=solar)
    prim_grid, _ = b629.observed_history(ids=pids)
    farm_g = farm_grid(ids=fids, power=power, start=start)
    # A missing cleaned solar reading at night is output zero, as in the forecast side (`farm_run`).
    times = [start + i * STEP for i in range(n)]
    for i, lab in enumerate(c.FARMS):
        if lab.startswith("Solar"):
            is_night = np.array([night[lab][t.month, t.hour * 2 + t.minute // 30] for t in times])
            farm_g[i, is_night & np.isnan(farm_g[i])] = 0.0
    meter_g = farm_grid(ids=list(c.BSP_GROUPS), power=power, start=start)
    grid = np.concatenate([prim_grid, farm_g])
    scales = farm_scales(farms=farms, power=power)
    print("farm scoring capacities found (private):", len(scales), flush=True)
    hist = {
        "prim": np.stack(
            [
                prim_grid[group_prim[g]].sum(0)
                * np.where(np.isnan(prim_grid[group_prim[g]]).any(0), np.nan, 1.0)
                for g in c.BSP_GROUPS
            ]
        ),
        "farm": np.stack(
            [
                farm_g[group_farm[g]].sum(0)
                * np.where(np.isnan(farm_g[group_farm[g]]).any(0), np.nan, 1.0)
                for g in c.BSP_GROUPS
            ]
        ),
        "meter": meter_g,
    }
    assert hist["meter"].shape[1] == n
    direct = {key: {m: c629.read_direct(key=key, month=m) for m in c.MONTHS} for key in c629.DIRECT}
    network = "K1"
    out_dir = c.WORK / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    qcols = list(c629.QUANTILES)
    runs: dict[datetime, dict] = {}
    for month in c.MONTHS:
        frame = c629.read_network(network=network, month=month)
        q = c629.run_arrays(frame=frame, ids=pids, columns=qcols)
        o = c629.run_arrays(frame=frame, ids=pids, columns=["observed_power"])
        for init in q:
            runs[init] = {"q": np.sort(q[init], axis=-1), "obs": o[init][..., 0], "month": month}
    inits = sorted(runs)
    print(network, len(inits), "runs", flush=True)
    first_run = {m: min(t for t in inits if runs[t]["month"] == m) for m in c.MONTHS}
    np.savez_compressed(
        c.WORK / "offsets.npz",
        **fit_offsets(hist=hist, start=start, months_first_run=first_run),  # ty: ignore[invalid-argument-type]
    )
    for init in inits:
        r = runs[init]
        fq = farm_run(init=init, wind=wind, solar=solar, scales=scales, night=night)
        r["fq"] = fq
        i0 = int((init - start) / STEP) + 1
        fobs = np.full((NF, S), np.nan)
        seg = farm_g[:, i0 : i0 + S]
        fobs[:, : seg.shape[1]] = seg
        r["fobs"] = fobs
        r["values"] = np.concatenate(
            [s434.fifty_one_values(quantiles=r["q"]), s434.fifty_one_values(quantiles=fq)]
        )
        actual = np.concatenate([r["obs"], fobs])
        z = b629.pit_scores(values=r["values"], actual=actual)
        degenerate = (r["values"][..., -1] - r["values"][..., 0]) < 1e-6
        z[degenerate] = np.nan
        r["z"] = z
    copulas = {}
    for month in c.MONTHS:
        start_m = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
        end_m = (start_m + timedelta(days=32)).replace(day=1)
        train = []
        for t in inits:
            if runs[t]["month"] == month or (start_m - timedelta(days=15) < t < start_m):
                continue
            zz = runs[t]["z"].copy()
            vt = np.array([t + (i + 1) * STEP for i in range(S)])
            zz[:, (vt >= c.END_OF_DEV) | ((vt >= start_m) & (vt < end_m))] = np.nan
            train.append(zz)
        copulas[month] = b629.fit_copula(z_runs=train)
        print(month, "copula fitted on", len(train), "runs", flush=True)
    for init in inits:
        out = out_dir / f"{init:%Y-%m-%d}.npz"
        if out.exists():
            continue
        r = runs[init]
        rng = np.random.default_rng(int(init.timestamp()) % (2**32))
        key_rng = np.random.default_rng(int(init.timestamp()) % (2**32) + 1)
        t_mem, sw_mem = s434.run_weather(init=init, ids=pids)
        day = np.median(sw_mem, axis=-1, keepdims=True) > 20.0
        ecc = s434.ranks_by_key(key=np.where(day, -sw_mem, -t_mem)[:, :S], rng=key_rng)
        sch = schaake_ranks(init=init, grid=grid, start=start, n_prim=n_prim)
        ind = rng.permuted(np.broadcast_to(np.arange(M), (n_prim + NF, S, M)), axis=-1)
        chol, phi = copulas[r["month"]]
        j5 = b629.latent_ranks(chol=chol, phi=phi, rng=rng)
        ecc25 = np.concatenate([ecc, ind[n_prim:]])  # farms: independent, masked below
        values = r["values"]
        sums = np.full((4, 2, 2, M, S), np.nan)
        farm_bad = np.isnan(values[n_prim:]).any(axis=-1)  # (9, 480)
        for a, ranks in enumerate((ind, ecc25, sch, j5)):
            path = s434.assign(values=np.nan_to_num(values), ranks=ranks.astype(np.int64))
            for gi, g in enumerate(c.BSP_GROUPS):
                s_p = path[:, group_prim[g]].sum(1)
                bad_p = np.isnan(values[group_prim[g]]).any(axis=(0, 2))
                s_p[:, bad_p] = np.nan
                sums[a, 0, gi] = s_p
                if a == 1:
                    continue  # no member weather at the farms on the daily runs
                fi = [n_prim + k for k in group_farm[g]]
                s_f = path[:, fi].sum(1)
                bad_f = farm_bad[group_farm[g]].any(axis=0)
                sums[a, 1, gi] = s_p + c.POWER_SIGN * s_f
                sums[a, 1, gi][:, bad_f] = np.nan
        prim_obs = np.stack([r["obs"][group_prim[g]].sum(0) for g in c.BSP_GROUPS])
        farm_obs = np.stack(
            [
                np.where(
                    np.isnan(r["fobs"][group_farm[g]]).any(0),
                    np.nan,
                    r["fobs"][group_farm[g]].sum(0),
                )
                for g in c.BSP_GROUPS
            ]
        )
        dir_arr = np.full((len(c629.DIRECT), 2, S, 11), np.nan)
        meter = np.full((2, S), np.nan)
        for k, key_d in enumerate(c629.DIRECT):
            sub = direct[key_d][r["month"]].filter(pl.col("power_fcst_init_time") == init)
            qd = c629.run_arrays(
                frame=sub, ids=list(c629.GROUP_IDS), columns=qcols + ["observed_power"]
            )
            arr = qd[init][:2]
            dir_arr[k] = np.sort(arr[..., :11], axis=-1)
            if key_d == "d622_lagged":
                meter = arr[..., 11]
        np.savez_compressed(
            out,
            sums=sums.astype(np.float32),
            meter=meter.astype(np.float32),
            prim_obs_sum=prim_obs.astype(np.float32),
            farm_obs_sum=farm_obs.astype(np.float32),
            direct=dir_arr.astype(np.float32),
            month=r["month"],
        )
        print(f"{init:%Y-%m-%d} written", flush=True)


if __name__ == "__main__":
    main()

# ruff: noqa: E501, C408, PLR0915
"""Experiment 482: forecast the injected scenarios with the stored networks (no fit).

For each test month of experiment 454 (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), scenario of
`e00482_scenarios`, and donor Primary, builds the shifted power world, recomputes the lag inputs
(and, for S1, the surprise at each lag) from it, and forecasts the rows of the donor and of its
mirror recipients with:

- `LL`: experiment 404's frozen cross-fitted lagless table (it reads no lag, so it is the control);
- `A0-inj`, `S1`: experiment 404's lagged networks trained on injected worlds (seeds 0 to 2);
- `B-mix`: the equal-weight density mixture of `A0-inj` and `LL` (3 cells, quantiles averaged);
- `K2-S1`: the mixture of two `S1` and two `LL` heads (3 seed-pair cells, quantiles averaged);
- `A-pool`: experiment 425's arm A, a pooled plain network never trained on injected shifts
  (seeds 0 to 4, quantiles averaged), with the shift added to its three lag columns before the
  stored standardiser is applied.

Writes `work/summary/<month>.parquet` (one row per month, scenario, donor, role, arm, row class,
variant) and, for the chart donors, `work/rows/<month>_<scenario>_<donor>.parquet` with the
quantiles of every arm. Scenario 0 (no shift) is scored too, as a self-test against the stored
forecasts.

```bash
source <paths of ml-experiments/*/ on PYTHONPATH>; export DATA_PATH_INTERNAL=<private data folder>
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00482-*/e00482_run.py --months 2026-01 --donors D5
```
"""

import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_models as models397  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00402_worlds as worlds402  # ty: ignore[unresolved-import]
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_report as report  # ty: ignore[unresolved-import]
import e00404_rows as rows_mod  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00446_worlds as worlds446  # ty: ignore[unresolved-import]
import e00454_build as build  # ty: ignore[unresolved-import]
import e00462_worlds as worlds462  # ty: ignore[unresolved-import]
import e00482_scenarios as scn
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00482-best-models-performance-analysis-injected-shifts-and-wind-references"
    / "work"
)
W425: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00425-best-mdn-architecture-assembly-with-ablations" / "work"
)
MONTHS: Final[tuple[str, ...]] = build.MONTHS
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
A_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
PAIRS: Final[tuple[tuple[int, int], ...]] = ((0, 1), (0, 2), (1, 2))
CHART_DONORS: Final[tuple[str, ...]] = ("D5", "D12", "D14")
ARMS: Final[tuple[str, ...]] = ("LL", "A0-inj", "S1", "B-mix", "K2-S1", "A-pool")
LEVELS: Final[tuple[str, ...]] = score.LEVEL_NAMES
WIDEN: Final[float] = 1.15
SETTING = fit397.setting_by_name(name="g09")
MASSES: Final[np.ndarray] = np.asarray(common.ATOM_MASSES, dtype=np.float64)
LEVEL_VALUES: Final[np.ndarray] = np.asarray(models.LEVELS, dtype=np.float64)
DEVICE: Final[str] = "cuda"


@dataclass
class Setup:
    """Everything shared between months and worlds."""

    sites: list[str]
    clean: np.ndarray
    capacity: np.ndarray
    neighbours: list[list[int]]
    nonnegative: np.ndarray
    spread: np.ndarray
    group: np.ndarray


def setup() -> Setup:
    """Return the clean power, capacities, neighbour lists, and lag spreads."""
    sites = prep.primaries()["site"].to_list()
    clean, _ = fit.world_power(seed=None)
    ctx = worlds402.load_context()
    capacity = np.array([ctx.capacity[s] for s in sites], dtype=np.float64)
    neighbours = worlds462.neighbour_lists(sites=sites)
    metered = fit.metered_indicator()
    return Setup(
        sites=sites, clean=clean, capacity=capacity, neighbours=neighbours,
        nonnegative=metered == 0.0, spread=scn.lag_spread(power=clean),
        group=report.group_of(sites=sites),
    )  # fmt: skip


def pinball_points(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the score in points of capacity (twice the mass-weighted pinball loss), per row."""
    diff = y[:, None] - q
    loss = np.maximum(LEVEL_VALUES[None, :] * diff, (LEVEL_VALUES[None, :] - 1.0) * diff)
    return 200.0 * (loss * MASSES[None, :]).sum(axis=1)


def widen(*, q: np.ndarray, factor: float) -> np.ndarray:
    """Scale every quantile's distance from the median by `factor`."""
    med = q[:, [5]]
    return np.sort(med + factor * (q - med), axis=1)


@dataclass
class MonthNets:
    """Stored networks, pools, and the lagless heads of one month."""

    pools: dict[tuple[str, int], tuple]
    nets: dict[tuple[str, int], torch.nn.Module]
    tables: dict[int, pl.DataFrame]
    lag_stats: dict[tuple[str, int], np.ndarray]
    scalers: dict[tuple[str, int], object]
    ll: dict[int, build.Head]
    keys: dict[str, np.ndarray]


def load_month_nets(*, month: str) -> MonthNets:
    """Load the A0-inj and S1 networks and the lagless heads of a month."""
    pools, nets, lag_stats, scalers = {}, {}, {}, {}
    tables = {s: pl.read_parquet(fit.lagless.table_path(month=month, seed=s)) for s in SEEDS}
    keys = None
    for arm in ("A0-inj", "S1"):
        for seed in SEEDS:
            net, saved = score.load_network(arm=arm, month=month, seed=seed, device=DEVICE)
            _tr, test, sc, _arrays, ls = fit.build_pools(
                month=month, seed=seed, device=DEVICE, stride=1, saved=saved
            )
            pools[arm, seed], nets[arm, seed], lag_stats[arm, seed], scalers[arm, seed] = (
                test,
                net,
                ls,
                sc,
            )
            k = {c: getattr(test, c).cpu().numpy() for c in build.KEY_COLUMNS}
            if keys is None:
                keys = k
            for c in build.KEY_COLUMNS:
                assert np.array_equal(keys[c], k[c]), f"{arm} s{seed}: {c} differs"
    assert keys is not None
    ll = {
        s: build.lagless_head(
            month=month, seed=s, primary=keys["primary"].astype(np.int64),
            target=keys["target"].astype(np.int64),
        )
        for s in SEEDS
    }  # fmt: skip
    return MonthNets(
        pools=pools,
        nets=nets,
        tables=tables,
        lag_stats=lag_stats,
        scalers=scalers,
        ll=ll,
        keys=keys,
    )


@torch.no_grad()
def lagged_heads(
    *, m: MonthNets, power: np.ndarray, idx: torch.Tensor, metered: torch.Tensor
) -> dict[str, dict[int, build.Head]]:
    """Return the A0-inj and S1 heads (capacity fractions) of the rows `idx` on `power`."""
    out: dict[str, dict[int, build.Head]] = {"A0-inj": {}, "S1": {}}
    for arm in ("A0-inj", "S1"):
        for seed in SEEDS:
            pool, net, sc = m.pools[arm, seed], m.nets[arm, seed], m.scalers[arm, seed]
            surprise = (
                fit.surprise_array(power=power, table=m.tables[seed])
                if arm == fit.SURPRISE_ARM
                else None
            )
            sub = fit.Pool(
                **{f.name: getattr(pool, f.name)[idx] for f in dataclasses.fields(fit.Pool)}
            )
            all_rows = torch.arange(len(sub), device=DEVICE)
            lags, lag_grid = fit.lag_features(
                pool=sub, power=torch.tensor(power, device=DEVICE), lag_stats=m.lag_stats[arm, seed],
                device=DEVICE, surprise=surprise,
            )  # fmt: skip
            parts = fit.forward(
                net=net, pool=sub, idx=all_rows, lags=lags, stack=None, lag_grid=lag_grid,
                scalers=sc, metered=metered,
            )  # fmt: skip
            log_w, mean, std = (p.double() for p in parts[:3])
            mu = torch.tensor(sc.target_mean, device=DEVICE)[sub.primary][:, None]  # ty: ignore[unresolved-attribute]
            sd = torch.tensor(sc.target_std, device=DEVICE)[sub.primary][:, None]  # ty: ignore[unresolved-attribute]
            out[arm][seed] = (log_w, mean * sd + mu, std * sd)
    return out


def mix_equal_gpu(*, heads: list[build.Head]) -> np.ndarray:
    """Return the sorted 11 quantiles (rows, 11) of the equal-weight mixture, solved on the device."""
    k = len(heads)
    log_w = torch.cat([h[0].to(DEVICE) - float(np.log(k)) for h in heads], dim=1)
    mean = torch.cat([h[1].to(DEVICE) for h in heads], dim=1)
    std = torch.cat([h[2].to(DEVICE) for h in heads], dim=1)
    out = []
    for start in range(0, len(log_w), build.BATCH):
        sl = slice(start, start + build.BATCH)
        out.append(models.mixture_quantiles(log_w=log_w[sl], mean=mean[sl], std=std[sl]))
    return torch.sort(torch.cat(out), dim=1).values.cpu().numpy().astype(np.float64)


def constructs(
    *, heads: dict[str, dict[int, build.Head]], ll: dict[int, build.Head]
) -> dict[str, np.ndarray]:
    """Return the quantiles (rows, 11) of LL, A0-inj, S1, B-mix, and K2-S1 (cells averaged)."""
    a, s = heads["A0-inj"], heads["S1"]
    cells = {
        "LL": [[ll[c]] for c in SEEDS],
        "A0-inj": [[a[c]] for c in SEEDS],
        "S1": [[s[c]] for c in SEEDS],
        "B-mix": [[a[c], ll[c]] for c in SEEDS],
        "K2-S1": [[s[i], s[j], ll[i], ll[j]] for i, j in PAIRS],
    }
    return {
        arm: np.mean([mix_equal_gpu(heads=members) for members in members_list], axis=0)
        for arm, members_list in cells.items()
    }


@dataclass
class APool:
    """Experiment 425's arm A for a month: rows aligned to the 404 pool, models, standardisers."""

    x_raw: np.ndarray
    present: np.ndarray
    models: list[torch.nn.Module]
    standardisers: list[prep.Standardiser]
    target_mean: list[np.ndarray]
    target_std: list[np.ndarray]
    lag_cols: list[int]


def load_apool(*, month: str, keys: dict[str, np.ndarray]) -> APool:
    """Return arm A's raw summary rows aligned to the 404 pool keys, and its five networks."""
    names, _mask, layout = prep.feature_names(scheme="summary")
    lag_cols = [names.index(c) for c in ("lag_origin", "lag_day", "lag_week")]
    origins_us = {int(o.timestamp() * 1_000_000) for o in prep.test_origins(month=month)}
    primaries, origins, targets, xs = [], [], [], []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = np.isin(z["origin"], list(origins_us))
        primaries.append(np.full(int(keep.sum()), row["primary"], dtype=np.int64))
        origins.append(((z["origin"][keep] - fit.GRID_START_US) // fit.STEP_US).astype(np.int64))
        targets.append(((z["valid"][keep] - fit.GRID_START_US) // fit.STEP_US).astype(np.int64))
        xs.append(z["x"][keep])
    x_all = np.concatenate(xs)
    frame = pl.DataFrame(
        {"primary": np.concatenate(primaries), "origin": np.concatenate(origins),
         "target": np.concatenate(targets), "pos": np.arange(len(x_all))}
    )  # fmt: skip
    query = pl.DataFrame({k: keys[k].astype(np.int64) for k in build.KEY_COLUMNS}).with_row_index(
        "i"
    )
    joined = query.join(frame, on=list(build.KEY_COLUMNS), how="left", maintain_order="left")
    present = joined["pos"].is_not_null().to_numpy()
    pos = joined["pos"].fill_null(0).to_numpy().astype(np.int64)
    x_raw = x_all[pos].astype(np.float64)
    nets, stds, tm, ts = [], [], [], []
    for seed in A_SEEDS:
        folder = W425 / "models" / "A" / month / f"s{seed}"
        meta = __import__("json").loads((folder / "meta.json").read_text())
        z = np.load(folder / "standardiser.npz")
        model = fit397.make_model(
            arm="plain_pooled",
            setting=SETTING,
            layout=layout,
            n_primaries=len(meta["target_mean"]),
            n_inputs=x_raw.shape[1],
        ).to(DEVICE)
        model.load_state_dict(torch.load(folder / "pooled.pt", map_location=DEVICE))
        model.eval()
        nets.append(model)
        stds.append(prep.Standardiser(mean=z["mean"], std=z["std"], mask=z["mask"]))
        tm.append(np.array(meta["target_mean"]))
        ts.append(np.array(meta["target_std"]))
    return APool(
        x_raw=x_raw,
        present=present,
        models=nets,
        standardisers=stds,
        target_mean=tm,
        target_std=ts,
        lag_cols=lag_cols,
    )


@torch.no_grad()
def apool_quantiles(
    *, ap: APool, rows: np.ndarray, primary: np.ndarray, lag_delta: np.ndarray
) -> np.ndarray:
    """Return arm A's seed-averaged quantiles (rows, 11) with `lag_delta` (rows, 3) added to the raw lags."""
    x = ap.x_raw[rows].copy()
    for i, col in enumerate(ap.lag_cols):
        d = lag_delta[:, i]
        x[:, col] = np.where(np.isfinite(x[:, col]) & np.isfinite(d), x[:, col] + d, x[:, col])
    out = []
    k = SETTING.components
    for model, st, tm, ts in zip(
        ap.models, ap.standardisers, ap.target_mean, ap.target_std, strict=True
    ):
        xb = torch.from_numpy(st.apply(x=x)).to(DEVICE)
        pb = torch.from_numpy(primary).to(DEVICE)
        res = fit397.forward(model=model, x=xb, primary=pb)
        log_w, mean, std = models397.mixture_parts(out=res.double(), components=k)
        q = models397.mixture_quantiles(log_w=log_w, mean=mean, std=std).cpu().numpy()
        out.append(np.sort(q * ts[primary][:, None] + tm[primary][:, None], axis=1))
    return np.mean(out, axis=0)


def self_test(*, month: str, m: MonthNets, base_q: dict[str, np.ndarray]) -> list[str]:
    """Compare the no-shift forecasts with experiment 454's stored cells and experiment 425's stored arm A."""
    lines = []
    for arm in ("S1", "B-mix", "K2-S1"):
        cells = sorted((build.WORK_DIR / "predictions" / arm).glob(f"{month}_s*_w-1.parquet"))
        frames = [pl.read_parquet(c) for c in cells]
        for c in build.KEY_COLUMNS:
            assert np.array_equal(frames[0][c].to_numpy(), m.keys[c]), f"{arm}: {c} differs"
        stored = np.mean(
            [f.select(list(LEVELS)).to_numpy().astype(np.float64) for f in frames], axis=0
        )
        lines.append(
            f"{month} {arm} vs experiment 454 stored cells: max |difference| {np.abs(stored - base_q[arm]).max():.2e} (capacity fraction)"
        )
    grid = common.grid_times()
    ids = prep.primaries().select("primary", "time_series_id")
    mine = (
        pl.DataFrame(
            {
                "primary": m.keys["primary"].astype(np.int64),
                "power_fcst_init_time": grid.gather(m.keys["origin"].astype(np.int64)),
                "valid_time": grid.gather(m.keys["target"].astype(np.int64)),
                **{f"m{n}": base_q["A-pool"][:, i] for i, n in enumerate(LEVELS)},
            }
        )
        .join(ids, on="primary")
        .with_columns(pl.col("time_series_id").cast(pl.Int32))
    )
    stored = (
        pl.concat(
            [
                pl.read_parquet(W425 / "seed_preds" / "A" / f"{month}_s{s}.parquet").with_columns(
                    seed=pl.lit(s)
                )
                for s in A_SEEDS
            ]
        )
        .group_by("time_series_id", "power_fcst_init_time", "valid_time")
        .agg(*[pl.col(n).mean().alias(f"s{n}") for n in LEVELS])
    )
    j = mine.join(stored, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="inner")
    diff = max(float((j[f"m{n}"] - j[f"s{n}"]).abs().max()) for n in LEVELS)  # ty: ignore[invalid-argument-type]
    lines.append(
        f"{month} A-pool vs experiment 425 stored arm A (seed average, {j.height:,} of {mine.height:,} rows): max |difference| {diff:.2e}"
    )
    return lines


def row_class(*, lag_grid: np.ndarray, a: int, b: int) -> np.ndarray:
    """Return `newest_in`, `older_in`, `before`, or `after` per row from the grid indices of the three lags.

    `newest_in`: the newest lag is inside the event. `older_in`: only the same-time-yesterday or
    same-time-last-week lag is inside. `before`: all lags precede the event. `after`: the newest lag
    follows the event and no lag is inside.
    """
    inside = (lag_grid >= a) & (lag_grid < b)
    newest = lag_grid[:, 0]
    return np.where(
        inside[:, 0], "newest_in",
        np.where(inside.any(axis=1), "older_in", np.where(newest < a, "before", "after")),
    )  # fmt: skip


def summarise(
    *, q: np.ndarray, y: np.ndarray, base_q50: np.ndarray, cls: np.ndarray, arm: str, meta: dict
) -> list[dict]:
    """Return per-class summaries for raw and widened quantiles."""
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    rows = []
    for variant, factor in (("raw", 1.0), ("w1.15", WIDEN)):
        qq = q if factor == 1.0 else widen(q=q, factor=factor)
        crps = pinball_points(q=qq, y=np.where(ok, y, 0.0))
        for c in ("newest_in", "older_in", "before", "after", "all"):
            sel = ok if c == "all" else ok & (cls == c)
            if not sel.any():
                continue
            rows.append(
                dict(
                    **meta,
                    arm=arm,
                    variant=variant,
                    cls=c,
                    n=int(sel.sum()),
                    crps=float(crps[sel].mean()),
                    mae=float(100 * np.abs(qq[sel, 5] - y[sel]).mean()),
                    cov_q01=float((y[sel] <= qq[sel, 0]).mean()),
                    cov_q05=float((y[sel] <= qq[sel, 1]).mean()),
                    cov_q95=float((y[sel] <= qq[sel, 9]).mean()),
                    cov_q99=float((y[sel] <= qq[sel, 10]).mean()),
                    width90=float(100 * (qq[sel, 9] - qq[sel, 1]).mean()),
                    dq50=float(100 * (qq[sel, 5] - base_q50[sel]).mean()),
                )
            )
    return rows


def run_month(
    *, month: str, st: Setup, donors: list[str], scenario_ids: list[int], force: bool
) -> None:
    """Forecast every scenario and donor of a month; write the summary and the chart rows."""
    out = WORK_DIR / "summary" / f"{month}.parquet"
    if out.exists() and not force:
        print(month, "exists")
        return
    started = time.time()
    m = load_month_nets(month=month)
    ap = load_apool(month=month, keys=m.keys)
    metered = torch.tensor(fit.metered_indicator(), device=DEVICE)
    primary = m.keys["primary"].astype(np.int64)
    origin = m.keys["origin"].astype(np.int64)
    target = m.keys["target"].astype(np.int64)
    lag_grid_all = rows_mod.lag_grids(target=target, origin=origin)
    pool0 = m.pools["S1", 0]
    lead = pool0.lead_hours.cpu().numpy()  # ty: ignore[unresolved-attribute]
    y_clean = pool0.y.cpu().numpy().astype(np.float64)  # ty: ignore[unresolved-attribute]
    print(
        f"{month}: loaded {len(primary):,} rows in {time.time() - started:.0f} s; A-pool rows present {ap.present.mean():.3f}",
        flush=True,
    )
    # clean-world baseline of every arm on every row
    all_idx = torch.arange(len(primary), device=DEVICE)
    base_heads = lagged_heads(m=m, power=st.clean, idx=all_idx, metered=metered)
    base_q = constructs(heads=base_heads, ll=m.ll)
    base_q["A-pool"] = apool_quantiles(
        ap=ap, rows=np.arange(len(primary)), primary=primary, lag_delta=np.zeros((len(primary), 3))
    )
    print(f"{month}: baselines done at {time.time() - started:.0f} s", flush=True)
    checks = self_test(month=month, m=m, base_q=base_q)
    (WORK_DIR / "checks").mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "checks" / f"{month}.txt").write_text("\n".join(checks) + "\n")
    print("\n".join(checks), flush=True)
    summary: list[dict] = []
    scenarios = [(0, None)] + [
        (i + 1, s) for i, s in enumerate(scn.SCENARIOS) if (i + 1) in scenario_ids
    ]
    for sid, s in scenarios:
        for donor_label in donors:
            donor = st.sites.index(donor_label)
            if s is None:
                shift, a, b, recipients = np.zeros_like(st.clean), 0, 0, []
            else:
                shift, a, b, recipients = scn.shift_array(
                    scenario=s, month=month, donor=donor, spread=st.spread, capacity=st.capacity,
                    neighbours=st.neighbours, shape=st.clean.shape,
                )  # fmt: skip
            world = worlds446.apply_shift(power=st.clean, shift=shift, nonnegative=st.nonnegative)
            roles = {donor: "donor", **dict.fromkeys(recipients, "recipient")}
            sel = np.isin(primary, list(roles))
            idx_np = np.nonzero(sel)[0]
            idx = torch.tensor(idx_np, device=DEVICE)
            heads = lagged_heads(m=m, power=world, idx=idx, metered=metered)
            sub_ll = {sd: tuple(t[idx_np] for t in h) for sd, h in m.ll.items()}
            q = constructs(heads=heads, ll=sub_ll)
            lg = lag_grid_all[idx_np]
            delta = np.where(
                (lg >= 0) & (lg < world.shape[1]),
                world[primary[idx_np][:, None], np.clip(lg, 0, world.shape[1] - 1)]
                - st.clean[primary[idx_np][:, None], np.clip(lg, 0, world.shape[1] - 1)],
                np.nan,
            )
            q["A-pool"] = apool_quantiles(
                ap=ap, rows=idx_np, primary=primary[idx_np], lag_delta=delta
            )
            y_world = (
                world[primary[idx_np], np.clip(target[idx_np], 0, world.shape[1] - 1)].astype(
                    np.float64
                )
                if s is not None and s.genuine
                else y_clean[idx_np]
            )
            cls = (
                row_class(lag_grid=lg, a=a, b=b)
                if s is not None
                else np.full(len(idx_np), "before")
            )
            for role_primary, role in roles.items():
                rsel = primary[idx_np] == role_primary
                d_points = float(100 * shift[role_primary, a] if s is not None else 0.0)
                meta = dict(
                    month=month, scenario=sid, donor=donor_label, role=role,
                    primary=st.sites[role_primary], group=str(st.group[role_primary]),
                    d_points=d_points, mirrored=bool(recipients),
                )  # fmt: skip
                for arm in ARMS:
                    ok_rows = rsel & (ap.present[idx_np] if arm == "A-pool" else True)
                    summary += summarise(
                        q=q[arm][ok_rows], y=y_world[ok_rows], base_q50=base_q[arm][idx_np][ok_rows, 5],
                        cls=cls[ok_rows], arm=arm, meta=meta,
                    )  # fmt: skip
            if donor_label in CHART_DONORS and sid > 0:
                rowdir = WORK_DIR / "rows"
                rowdir.mkdir(parents=True, exist_ok=True)
                frame = pl.DataFrame(
                    {
                        "primary": primary[idx_np],
                        "origin": origin[idx_np],
                        "target": target[idx_np],
                        "lead_hours": lead[idx_np],
                        "y_world": y_world,
                        "y_clean": y_clean[idx_np],
                        "cls": cls,
                        "a": a,
                        "b": b,
                    }
                )
                for arm in ARMS:
                    frame = frame.with_columns(
                        *[
                            pl.Series(f"{arm}_{n}", q[arm][:, i].astype(np.float32))
                            for i, n in enumerate(LEVELS)
                        ]
                    )
                frame.write_parquet(rowdir / f"{month}_s{sid}_{donor_label}.parquet")
        print(
            f"{month} scenario {sid}: {len(summary):,} summary rows, {time.time() - started:.0f} s",
            flush=True,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(summary).write_parquet(out)


def main() -> None:
    """Run the requested months, donors, and scenarios."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--donors", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    st = setup()
    donors = args.donors or st.sites
    print(
        "lag spread (capacity share):",
        dict(zip(st.sites, np.round(st.spread, 4), strict=True)),
        flush=True,
    )
    for month in args.months:
        run_month(month=month, st=st, donors=donors, scenario_ids=args.scenarios, force=args.force)


if __name__ == "__main__":
    main()

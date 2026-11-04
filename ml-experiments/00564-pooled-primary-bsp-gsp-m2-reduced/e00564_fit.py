# ruff: noqa: E501, ANN001, ANN201, PT018
"""Experiment 564: M0 trained on the 16 Primaries, 2 bulk supply points, and 2 grid supply points.

Rows, cutoffs, recency weights, early stopping, seeds, scoring windows, and the daily harness are
experiment 554's M0 (daily refit, 90-day half-life), copied here and extended to 20 series by
`e00564_pool.install()`. The network is `e00564_model.PooledModularMdn` (M0 plus a series-type input).

Arms (all `d` cadence, 90-day half-life, fresh mixer at fine-tuning with learning rate 5e-4):

- `M2-a`: the 4 extra series in pretraining (ERA5 and CAMS rows) and in fine-tuning (ENS rows).
- `M2-b`: the 4 extra series in pretraining only; fine-tuning rows are the 16 Primaries'.
- `M2-c`: the 4 extra series in fine-tuning only, starting from experiment 545's M0 pretrained weights.

Fine-tuning rows of an extra series are scaled so that the series' total weight equals the mean total
weight of a Primary. Pretraining rows have weight 1, as in M0.

Usage: python e00564_fit.py {pretrain|fit|assemble|build-pretrain-rows} --arm ARM [--stress] [--months ...] [--seeds ...]
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00415_prep as p415  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import e00554_model as m0  # ty: ignore[unresolved-import]
import e00564_model as m
import e00564_pool as pool
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

pool.install()
WORK = ML_EXPERIMENTS_DATA_DIR / "00564-pooled-primary-bsp-gsp-m2-reduced"
W = WORK / "work"
W545 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00545-modular-lag-free-demand-network-weather-and-datetime-encoders/work"
)
FRESH_SEED = 12345
"""Initialisation seed of every fresh mixer, as in experiments 545 and 554."""
FRESH_LR = 5e-4
"""Learning rate of the fresh mixer at fine-tuning (M0's)."""

# arm -> (pretrained weights, whether fine-tuning rows include the extra series)
SPEC = {
    "M2-a": ("P20", True),
    "M2-b": ("P20", False),
    "M2-c": ("M0", True),
}
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("pretrain", "fit", "assemble", "build-pretrain-rows"))
parser.add_argument("--arm", choices=tuple(SPEC))
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
if ARGS.command != "build-pretrain-rows":
    assert ARGS.arm, "--arm is required"
PRETRAINED, FINE_TUNE_EXTRA = SPEC[ARGS.arm or "M2-a"]
ARM = f.Arm(ARGS.arm or "M2-a", "d", 90.0, pretrained=PRETRAINED)
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}
_select_rows = c.select_rows

_, COLUMNS, _ = data.non_lag_layout()
assert len(COLUMNS) == m0.N_INPUTS
assert COLUMNS[m0.WEATHER_COLS] == [
    f"mean_{v}"
    for v in (
        "shortwave",
        "longwave",
        "temperature_2m",
        "dew_point_temperature_2m",
        "pressure_surface",
        "precipitation",
        "wind_speed_10m",
    )
]
assert all(n.startswith("std_") for n in COLUMNS[m0.SPREAD_COLS])
assert (
    COLUMNS[m0.DATETIME_COLS][0] == "cos_zenith" and COLUMNS[m0.DATETIME_COLS][-1] == "nearest_inside"
)
assert COLUMNS[m0.LEAD_COLS] == ["lead", "lead_sqrt"]
N_SERIES = len(pool.pooled_table())
assert N_SERIES == 20


def select_rows(*, rows: c.MonthRows, cutoff: datetime, half_life_days: float, **kwargs) -> c.Refit:
    """Experiment 510's row selection, then the arm's treatment of the 4 extra series.

    The extra series are dropped (M2-b) or have their weights scaled so that each series' total
    weight equals the mean total weight of a Primary (M2-a, M2-c). Early-stopping flags follow the
    rows. The leak counts in the report are unchanged, because the selection is made first.
    """
    refit = _select_rows(rows=rows, cutoff=cutoff, half_life_days=half_life_days, **kwargs)
    series = rows.primary[refit.index]
    extra = series >= pool.N_PRIMARIES
    if not FINE_TUNE_EXTRA:
        keep = ~extra
        return c.Refit(
            index=refit.index[keep],
            y=refit.y[keep],
            weight=refit.weight[keep],
            is_val=refit.is_val[keep],
            report={**refit.report, "extra_series_rows_dropped": int(extra.sum())},
        )
    weight = refit.weight.copy()
    totals = np.bincount(series, weights=weight, minlength=N_SERIES)
    primary_mean = float(totals[: pool.N_PRIMARIES].mean())
    for i in range(pool.N_PRIMARIES, N_SERIES):
        weight[series == i] *= primary_mean / totals[i]
    refit.report["extra_series_rows"] = int(extra.sum())
    return c.Refit(
        index=refit.index, y=refit.y, weight=weight, is_val=refit.is_val, report=refit.report
    )


def new_model(
    *, outlier: bool, n_inputs: int, n_primaries: int, stage: str | None = None
) -> torch.nn.Module:
    """Return an untrained pooled network (the signature of `e00510_common.new_model`)."""
    assert not outlier
    assert n_inputs == m0.N_INPUTS
    assert n_primaries == N_SERIES
    return m.PooledModularMdn(kinds=pool.kinds(), stage=stage or "ft")  # ty: ignore[invalid-argument-type]


def load_pretrained(*, model: torch.nn.Module, path: Path) -> None:
    """Load pretrained weights, then give the fresh mixer a seed-12345 initialisation."""
    raw = torch.load(path, map_location="cpu")
    if PRETRAINED == "M0":
        raw = m.expand_m0_state(raw=raw, n_series=N_SERIES)
    state = {a: v for a, v in raw.items() if not a.startswith("mixer_ft.")}
    missing = model.load_state_dict(state, strict=False)
    allowed = ("mixer_ft.", "kind")
    assert all(a.startswith(allowed) for a in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FRESH_SEED)
        fresh = base.mlp(
            n_in=model.mixer_ft[0].in_features,  # ty: ignore[not-subscriptable,unresolved-attribute]
            hidden=m0.MIXER_HIDDEN,
            layers=2,
            n_out=3 * m0.COMPONENTS,
            dropout=m0.DROPOUT,
        )
        base.init_head(head=fresh, components=m0.COMPONENTS)
    model.mixer_ft.load_state_dict(fresh.state_dict())  # ty: ignore[unresolved-attribute]


def train(
    *, model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, p: torch.Tensor,
    is_val: torch.Tensor, seed: int, lr: float, outlier: bool, prior: bool, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Experiment 510's loop, with the fresh mixer in its own learning-rate group at fine-tuning."""
    assert not outlier
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    fresh = list(model.mixer_ft.parameters()) if lr == c.FINE_TUNE_LR else []  # ty: ignore[unresolved-attribute]
    fresh_ids = {id(q) for q in fresh}
    groups = [{"params": [q for q in model.parameters() if id(q) not in fresh_ids], "lr": lr}]
    if fresh:
        groups.append({"params": fresh, "lr": FRESH_LR})
    optimiser = torch.optim.AdamW(groups, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            nll = base.mixture_nll(out=model(x[b], p[b]), z=z[b], components=m0.COMPONENTS)
            loss = (w[b] * nll).sum() / w[b].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        base.mixture_nll(out=model(x[i], p[i]), z=z[i], components=m0.COMPONENTS)
                        for i in val_idx.split(c.BATCH * 2)
                    ]
                ).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def pretrained_file(*, arm: f.Arm, month: str, seed: int) -> Path:
    """Return the pretrained weights an arm starts from (M0's from experiment 545, or this experiment's 20-series ones)."""
    if arm.pretrained == "M0":
        return W545 / "pretrained" / "M0" / f"{month}_s{seed}.pt"
    return W / "pretrained" / "P20" / f"{month}_s{seed}.pt"


c.new_model, c.load_pretrained, c.train, c.select_rows = (
    new_model,
    load_pretrained,
    train,
    select_rows,
)
f.pretrained_file = pretrained_file
f.W = W
f.model_file = lambda *, arm, month, day, seed, stress="": (
    W / "models" / arm.tag / f"{month}_d{day}_s{seed}{stress}.pt"
)


def pretrain(*, device: str) -> None:
    """Pretrain on ERA5 and CAMS rows of the 20 series for each month and seed that has no weights yet."""
    assert PRETRAINED == "P20", "M2-c starts from experiment 545's pretrained weights"
    clean = c.clean_stack()
    for month in ARGS.months:
        todo = [s for s in ARGS.seeds if not pretrained_file(arm=ARM, month=month, seed=s).exists()]
        if not todo:
            continue
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
        latest = datetime.fromtimestamp(
            (pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC
        )
        assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
        counts = np.bincount(pre.primary, minlength=N_SERIES)
        print(
            f"{month}: pretraining rows {len(pre.x):,}; latest valid {latest.isoformat()} (refit {refit_t.isoformat()}); rows per series {counts.tolist()}",
            flush=True,
        )
        z = (pre.y - rows.target_mean[pre.primary]) / rows.target_std[pre.primary]
        t = {
            "x": torch.from_numpy(pre.x).to(device), "z": torch.from_numpy(z.astype(np.float32)).to(device),
            "w": torch.ones(len(pre.y), device=device), "p": torch.from_numpy(pre.primary).to(device),
            "v": torch.from_numpy(pre.is_val).to(device),
        }  # fmt: skip
        for seed in todo:
            started = time.monotonic()
            torch.manual_seed(seed)
            model = new_model(
                outlier=False, n_inputs=m0.N_INPUTS, n_primaries=N_SERIES, stage="pre"
            ).to(device)
            target = pretrained_file(arm=ARM, month=month, seed=seed)
            epochs, best = train(
                model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                lr=c.PRETRAIN_LR, outlier=False, prior=False, device=device,
            )  # fmt: skip
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  pretrain P20 {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
        del t, pre, rows
        torch.cuda.empty_cache()


_build_month, _build_test = k.build_month, data.build_test


def build_month(*, month, clean):
    """Build the month's rows; assert each training origin's weather run exists and precedes it."""
    mr = _build_month(month=month, clean=clean)
    assert (mr.valid_us > mr.origin_us).all(), f"{month}: a training target is not after its origin"
    runs = np.unique(mr.origin_us)
    delay = int(common.PUBLICATION_DELAY.total_seconds() * 1e6)
    miss = [
        r
        for r in runs
        if not (
            prep.SUMMARY_DIR
            / f"init={datetime.fromtimestamp((r - delay) / 1e6, UTC):%Y-%m-%d}.parquet"
        ).exists()
    ]
    assert not miss, (
        f"{month}: {len(miss)} training origins have no weather-run summary file {common.PUBLICATION_DELAY} earlier"
    )
    miss_nodes = [
        r
        for r in runs
        if not (
            p415.NODE_SUMMARY_DIR
            / f"init={datetime.fromtimestamp((r - delay) / 1e6, UTC):%Y-%m-%d}.parquet"
        ).exists()
    ]
    assert not miss_nodes, f"{month}: {len(miss_nodes)} training origins have no node weather-run file"
    CHECK["train_rows"] += len(mr.y)
    CHECK["months"][month] = {
        "train_rows": len(mr.y),
        "train_origins": len(runs),
        "rows_per_series": np.bincount(mr.primary, minlength=N_SERIES).tolist(),
    }
    print(month, "train checks passed", flush=True)
    return mr


def build_test(*, month, standardiser):
    """Build the month's test rows and assert, for every test origin, nwp_init_time <= power_fcst_init_time.

    The check reads the Primaries' stored weather frame of each run (the frame holds every series'
    rows of the run, so the extra series share the run's `nwp_init_time` and `power_fcst_init_time`).
    The extra series' own summary file for the same run is checked for the same origin.
    """
    test = _build_test(month=month, standardiser=standardiser)
    assert (test.valid > test.origin).all(), f"{month}: a test target is not after its origin"
    pub = common.PUBLICATION_DELAY
    for o in np.unique(test.origin):
        origin = datetime.fromtimestamp(o / 1e6, UTC)
        run = origin - pub
        fr = pl.read_parquet(common.demand_frame_path(run=run))
        assert (fr["nwp_init_time"] <= fr["power_fcst_init_time"]).all()
        assert (fr["power_fcst_init_time"] == origin).all()
        assert (prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet").exists()
        node_file = p415.NODE_SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet"
        assert node_file.exists(), node_file
        node = pl.read_parquet(node_file, columns=["power_fcst_init_time"])
        assert (node["power_fcst_init_time"] == origin).all(), node_file
        CHECK["origins_weather_checked"] += 1
    CHECK["test_rows"] += len(test.x)
    extra = test.primary >= pool.N_PRIMARIES
    i = int(np.nonzero(extra)[0][0])
    CHECK["months"][month]["weather_example"] = (
        f"extra series {int(test.primary[i])}: target {datetime.fromtimestamp(test.valid[i] / 1e6, UTC)}, origin {datetime.fromtimestamp(test.origin[i] / 1e6, UTC)}, "
        f"weather run {datetime.fromtimestamp(test.origin[i] / 1e6, UTC) - pub} (nwp_init_time <= power_fcst_init_time asserted for all rows of every test origin)"
    )
    print(month, "test checks passed |", CHECK["months"][month]["weather_example"], flush=True)
    return test


k.build_month, data.build_test = build_month, build_test


def assemble() -> None:
    """Concatenate each cutoff's window forecast into one table per month and seed.

    Primary rows go to `seed_preds/<arm>/` (experiment 554's layout, for the scorer); the 4 extra
    series' rows go to `node_preds/<arm>/`.
    """
    series_ids = pool.pooled_table()["time_series_id"].to_list()
    extra_ids = series_ids[pool.N_PRIMARIES :]
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence=ARM.cadence)
        for seed in ARGS.seeds:
            parts = [
                W / "fits" / ARM.tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            assert all(p.exists() or p.with_suffix(".empty").exists() for p in parts), (month, seed)
            frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
            is_extra = pl.col("time_series_id").is_in(extra_ids)
            f.atomic_frame(
                frame=frame.filter(~is_extra),
                target=W / "seed_preds" / ARM.tag / f"{month}_s{seed}.parquet",
            )
            f.atomic_frame(
                frame=frame.filter(is_extra),
                target=W / "node_preds" / ARM.tag / f"{month}_s{seed}.parquet",
            )
            print("wrote", ARM.tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble()
    elif ARGS.command == "build-pretrain-rows":
        pool.build_extra_pretraining()
    elif ARGS.command == "pretrain":
        pretrain(device="cuda")
    else:
        f.fit_arm(
            arm=ARM,
            months=ARGS.months,
            seeds=ARGS.seeds,
            stress=ARGS.stress,
            limit=ARGS.limit,
            device="cuda",
        )
        out = (
            WORK
            / f"checks_{ARM.tag}{'_stress' if ARGS.stress else ''}_s{'-'.join(map(str, ARGS.seeds))}.json"
        )
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in CHECK.items() if a != "months"})

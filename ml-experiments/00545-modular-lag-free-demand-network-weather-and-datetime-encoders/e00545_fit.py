# ruff: noqa: E501, ANN001, ANN201, PT018
"""Experiment 545: pretrain, fine-tune, and forecast the modular lag-free networks M0 and M1.

Rows, cutoffs, recency weights, early stopping, seeds, scoring windows, and the weekly/daily
harness are experiment 511's L-1d (daily refit, 90-day half-life). Only the network, its pretraining
loop, and its fine-tuning loop differ (`e00545_model`). The origin-time weather assertions are
experiment 539's, minus the lag checks.

Usage: python e00545_fit.py {pretrain|fit|assemble} --arm M0|M1 [--stress] [--months ...] [--seeds ...]
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
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import e00545_model as m
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = (
    ML_EXPERIMENTS_DATA_DIR / "00545-modular-lag-free-demand-network-weather-and-datetime-encoders"
)
W = WORK / "work"
MIXER_LR = 5e-4
"""Learning rate of the fresh fine-tuning mixer; the encoders and auxiliary heads use `c.FINE_TUNE_LR`."""
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("pretrain", "fit", "assemble"))
parser.add_argument("--arm", choices=("M0", "M1", "M0lr"), required=True)
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
AUX = ARGS.arm == "M1"
PRETRAINED_TAG = "M0" if ARGS.arm == "M0lr" else ARGS.arm
if ARGS.arm == "M0lr":
    MIXER_LR = (
        c.FINE_TUNE_LR
    )  # post hoc control: the fresh mixer fine-tunes at the same rate as the encoders
ARM = f.Arm(ARGS.arm, "d", 90.0)
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}

_, COLUMNS, _ = data.non_lag_layout()
assert len(COLUMNS) == m.N_INPUTS
assert COLUMNS[m.WEATHER_COLS] == [
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
assert all(n.startswith("std_") for n in COLUMNS[m.SPREAD_COLS])
assert (
    COLUMNS[m.DATETIME_COLS][0] == "cos_zenith" and COLUMNS[m.DATETIME_COLS][-1] == "nearest_inside"
)
assert COLUMNS[m.LEAD_COLS] == ["lead", "lead_sqrt"]


def new_model(
    *, outlier: bool, n_inputs: int, n_primaries: int, stage: str = "ft"
) -> torch.nn.Module:
    """Return an untrained modular network (the signature of `e00510_common.new_model`)."""
    assert not outlier
    assert n_inputs == m.N_INPUTS
    return m.ModularMdn(n_primaries=n_primaries, aux=AUX, stage=stage)


def load_pretrained(*, model: torch.nn.Module, path: Path) -> None:
    """Load pretrained weights; the fine-tuning mixer is discarded and freshly initialised (seed 12345)."""
    state = {
        a: v
        for a, v in torch.load(path, map_location="cpu").items()
        if not a.startswith("mixer_ft.")
    }
    missing = model.load_state_dict(state, strict=False)
    assert all(a.startswith("mixer_ft.") for a in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12345)
        fresh = base.mlp(
            n_in=model.mixer_ft[0].in_features,  # ty: ignore[not-subscriptable,unresolved-attribute]
            hidden=m.MIXER_HIDDEN,
            layers=2,
            n_out=3 * m.COMPONENTS,
            dropout=m.DROPOUT,
        )
        base.init_head(head=fresh, components=m.COMPONENTS)
    model.mixer_ft.load_state_dict(fresh.state_dict())  # ty: ignore[unresolved-attribute]


def row_nll(
    *, model: torch.nn.Module, x: torch.Tensor, p: torch.Tensor, z: torch.Tensor, training: bool
) -> torch.Tensor:
    """Return the per-row training loss (main NLL plus, for M1 in training, the two weighted auxiliary NLLs)."""
    outs = model.forward_all(x, p)  # ty: ignore[call-non-callable]
    nll = base.mixture_nll(out=outs[0], z=z, components=m.COMPONENTS)
    if training and len(outs) == 3:
        nll = nll + m.AUX_WEIGHT * (
            base.mixture_nll(out=outs[1], z=z, components=m.COMPONENTS)
            + base.mixture_nll(out=outs[2], z=z, components=m.COMPONENTS)
        )
    return nll


def train(
    *, model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, p: torch.Tensor,
    is_val: torch.Tensor, seed: int, lr: float, outlier: bool, prior: bool, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Experiment 510's loop with the auxiliary loss in training and a main-NLL-only early stop."""
    assert not outlier
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    mixer = list(model.mixer_ft.parameters()) if model.stage == "ft" else []  # ty: ignore[unresolved-attribute]
    mixer_ids = {id(q) for q in mixer}
    groups = [{"params": [q for q in model.parameters() if id(q) not in mixer_ids], "lr": lr}]
    if mixer:
        groups.append({"params": mixer, "lr": MIXER_LR})
    optimiser = torch.optim.AdamW(groups, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            nll = row_nll(model=model, x=x[b], p=p[b], z=z[b], training=True)
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
                        row_nll(model=model, x=x[i], p=p[i], z=z[i], training=False)
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


c.new_model, c.load_pretrained, c.train = new_model, load_pretrained, train
f.pretrained_file = lambda *, arm, month, seed: (
    W / "pretrained" / PRETRAINED_TAG / f"{month}_s{seed}.pt"
)
f.W = W
f.model_file = lambda *, arm, month, day, seed, stress="": (
    W / "models" / arm.tag / f"{month}_d{day}_s{seed}{stress}.pt"
)


def pretrain(*, device: str) -> None:
    """Pretrain on ERA5 and CAMS rows (experiment 448's rows, the L0PT pretraining) for each month and seed."""
    clean = c.clean_stack()
    for month in ARGS.months:
        todo = [
            s for s in ARGS.seeds if not (W / "pretrained" / ARM.tag / f"{month}_s{s}.pt").exists()
        ]
        if not todo:
            continue
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
        assert pre.x.shape[1] == m.N_INPUTS
        latest = datetime.fromtimestamp(
            (pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC
        )
        assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
        print(
            f"{month}: pretraining rows {len(pre.x):,}; latest valid {latest.isoformat()} (refit {refit_t.isoformat()})",
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
                outlier=False, n_inputs=m.N_INPUTS, n_primaries=len(rows.target_mean), stage="pre"
            ).to(device)
            epochs, best = train(
                model=model,
                x=t["x"],
                z=t["z"],
                w=t["w"],
                p=t["p"],
                is_val=t["v"],
                seed=seed,
                lr=c.PRETRAIN_LR,
                outlier=False,
                prior=False,
                device=device,
            )
            target = W / "pretrained" / ARM.tag / f"{month}_s{seed}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  pretrain {ARM.tag} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
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
    CHECK["train_rows"] += len(mr.y)
    CHECK["months"][month] = {"train_rows": len(mr.y), "train_origins": len(runs)}
    print(month, "train checks passed", flush=True)
    return mr


def build_test(*, month, standardiser):
    """Build the month's test rows and assert, for every test origin, nwp_init_time <= power_fcst_init_time."""
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
        CHECK["origins_weather_checked"] += 1
    CHECK["test_rows"] += len(test.x)
    CHECK["months"][month]["weather_example"] = (
        f"target {datetime.fromtimestamp(test.valid[0] / 1e6, UTC)}, origin {datetime.fromtimestamp(test.origin[0] / 1e6, UTC)}, "
        f"weather run {datetime.fromtimestamp(test.origin[0] / 1e6, UTC) - pub} (nwp_init_time <= power_fcst_init_time asserted for all rows of every test origin)"
    )
    print(month, "test checks passed |", CHECK["months"][month]["weather_example"], flush=True)
    return test


k.build_month, data.build_test = build_month, build_test


def assemble() -> None:
    """Concatenate each cutoff's window forecast into one table per month and seed."""
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence="d")
        for seed in (0, 1):
            parts = [
                W / "fits" / ARM.tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            assert all(p.exists() or p.with_suffix(".empty").exists() for p in parts), (month, seed)
            frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
            f.atomic_frame(
                frame=frame, target=W / "seed_preds" / ARM.tag / f"{month}_s{seed}.parquet"
            )
            print("wrote", ARM.tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble()
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
        out = WORK / f"checks_{ARM.tag}{'_stress' if ARGS.stress else ''}.json"
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in CHECK.items() if a != "months"})

# ruff: noqa: E501, ANN001, ANN201, PT018
"""Experiment 554: replicate the modular lag-free network M0 with 5 seeds and test what raises its leak.

Rows, cutoffs, recency weights, early stopping, seeds, scoring windows, and the daily harness are
experiment 511's L-1d (daily refit, 90-day half-life). Only the network, its pretraining loop, and
its fine-tuning loop differ (`e00554_model`). The origin-time weather assertions are experiment 539's,
minus the lag checks.

Arms (all `d` cadence, 90-day half-life unless the name says otherwise):

- `L-1d`: the single multilayer network of experiment 511, seeds 2 to 4 run here (seeds 0 and 1 are 511's).
- `M0`, `M0lr`: experiment 545's arms, seeds 2 to 4 run here (seeds 0 and 1 are 545's).
- `M0-sep`: M0's separate encoders, but the pretrained mixer is kept at fine-tuning (no spread or
  lead input).
- `M0-mix`: the single network of L-1d, whose last two linear layers are re-initialised at fine-tuning.
- `M0-hl180`: M0 with a 180-day recency half-life.
- `M0-c3`: M0 refitted every third day (cutoffs on days 0, 2, 5, 8, ...).

Usage: python e00554_fit.py {pretrain|fit|assemble} --arm ARM [--stress] [--months ...] [--seeds ...]
"""

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as f510  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import e00554_model as m
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = ML_EXPERIMENTS_DATA_DIR / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out"
W = WORK / "work"
W545 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00545-modular-lag-free-demand-network-weather-and-datetime-encoders/work"
)
W511 = f.W
"""Experiment 511's work folder (its `fits/L-1d` holds seeds 0 and 1 of L-1d)."""
SEEDS_ALL = (0, 1, 2, 3, 4)
ASSEMBLED_SEEDS = {"L-1d": SEEDS_ALL, "M0": SEEDS_ALL, "M0lr": SEEDS_ALL}
"""Seeds assembled per arm; every other arm has seeds 0 to 2."""
SEEDS_REUSED = (0, 1)
"""Seeds whose fits come from the earlier experiment's folder (545 for M0 and M0lr, 511 for L-1d)."""
FRESH_SEED = 12345
"""Initialisation seed of every fresh mixer, as in experiment 545."""
CADENCE_3_OFFSETS = (0, *range(2, 31, 3))
"""Refit days of arm M0-c3: day 0, then every third day from day 2, so that days 14, 17, and 20 are refits."""

# arm -> (family, network stage at fine-tuning, pretrained weights, cadence, half-life in days, mixer lr)
SPEC = {
    "L-1d": ("plain", None, "L0PT", "d", 90.0, None),
    "M0": ("modular", "ft", "M0", "d", 90.0, 5e-4),
    "M0lr": ("modular", "ft", "M0", "d", 90.0, c.FINE_TUNE_LR),
    "M0-sep": ("modular", "pre", "M0", "d", 90.0, None),
    "M0-mix": ("plain-fresh", None, "L0PT", "d", 90.0, 5e-4),
    "M0-hl180": ("modular", "ft", "M0", "d", 180.0, 5e-4),
    "M0-c3": ("modular", "ft", "M0", "d", 90.0, 5e-4),
}
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("pretrain", "fit", "assemble"))
parser.add_argument("--arm", choices=tuple(SPEC), required=True)
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[2, 3, 4])
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
FAMILY, STAGE, PRETRAINED, CADENCE, HALF_LIFE, FRESH_LR = SPEC[ARGS.arm]
ARM = f.Arm(ARGS.arm, CADENCE, HALF_LIFE, pretrained=PRETRAINED)
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}
_new_model, _load_pretrained, _train = c.new_model, c.load_pretrained, c.train
_cutoffs = k.cutoffs

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


def cutoffs(*, month: str, cadence: str) -> list[datetime]:
    """Experiment 511's cutoffs; arm M0-c3 refits on days 0, 2, 5, 8, ... (it passes cadence `d`)."""
    if ARGS.arm != "M0-c3":
        return _cutoffs(month=month, cadence=cadence)
    start, stop = c.month_start(month=month), c.month_stop(month=month)
    days = (stop - start).days
    return [start + timedelta(days=d) for d in CADENCE_3_OFFSETS if d < days]


def fresh_parameters(*, model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Return the parameters that start fresh at fine-tuning and train at `FRESH_LR`."""
    if FRESH_LR is None:
        return []
    if FAMILY == "modular":
        return list(model.mixer_ft.parameters())  # ty: ignore[unresolved-attribute]
    return [*model.net[3].parameters(), *model.net[6].parameters()]  # ty: ignore[not-subscriptable,unresolved-attribute]


def new_model(
    *, outlier: bool, n_inputs: int, n_primaries: int, stage: str | None = None
) -> torch.nn.Module:
    """Return an untrained network of the arm's family (the signature of `e00510_common.new_model`)."""
    assert not outlier
    if FAMILY != "modular":
        return _new_model(outlier=False, n_inputs=n_inputs, n_primaries=n_primaries)
    assert n_inputs == m.N_INPUTS
    return m.ModularMdn(n_primaries=n_primaries, stage=stage or STAGE)  # ty: ignore[invalid-argument-type]


def load_pretrained(*, model: torch.nn.Module, path: Path) -> None:
    """Load pretrained weights, then give the arm's fresh mixer (if any) a seed-12345 initialisation."""
    if FAMILY == "plain":
        _load_pretrained(model=model, path=path)
        return
    if FAMILY == "plain-fresh":
        _load_pretrained(model=model, path=path)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(FRESH_SEED)
            fresh = base.mlp(
                n_in=model.net[3].in_features,  # ty: ignore[not-subscriptable,unresolved-attribute]
                hidden=model.net[3].out_features,  # ty: ignore[not-subscriptable,unresolved-attribute]
                layers=1,
                n_out=model.net[6].out_features,  # ty: ignore[not-subscriptable,unresolved-attribute]
            )
            base.init_head(head=fresh, components=c.COMPONENTS)
        model.net[3].load_state_dict(fresh[0].state_dict())  # ty: ignore[not-subscriptable,unresolved-attribute]
        model.net[6].load_state_dict(fresh[2].state_dict())  # ty: ignore[not-subscriptable,unresolved-attribute]
        return
    raw = torch.load(path, map_location="cpu")
    if STAGE == "pre":
        model.load_state_dict(raw)
        return
    state = {a: v for a, v in raw.items() if not a.startswith("mixer_ft.")}
    missing = model.load_state_dict(state, strict=False)
    assert all(a.startswith("mixer_ft.") for a in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FRESH_SEED)
        fresh = base.mlp(
            n_in=model.mixer_ft[0].in_features,  # ty: ignore[not-subscriptable,unresolved-attribute]
            hidden=m.MIXER_HIDDEN,
            layers=2,
            n_out=3 * m.COMPONENTS,
            dropout=m.DROPOUT,
        )
        base.init_head(head=fresh, components=m.COMPONENTS)
    model.mixer_ft.load_state_dict(fresh.state_dict())  # ty: ignore[unresolved-attribute]


def train(
    *, model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, p: torch.Tensor,
    is_val: torch.Tensor, seed: int, lr: float, outlier: bool, prior: bool, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Experiment 510's loop, with the fresh mixer (if any) in its own learning-rate group."""
    assert not outlier
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    fresh = fresh_parameters(model=model) if lr == c.FINE_TUNE_LR else []
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
            nll = base.mixture_nll(out=model(x[b], p[b]), z=z[b], components=m.COMPONENTS)
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
                        base.mixture_nll(out=model(x[i], p[i]), z=z[i], components=m.COMPONENTS)
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


def l0pt_file(*, month: str, seed: int) -> Path:
    """Return the single-network pretrained weights (this folder's if written here, else 448's or 510's)."""
    own = W / "pretrained" / "L0PT" / f"{month}_s{seed}.pt"
    return (
        own
        if own.exists()
        else f510.pretrained_path(v=f510.VARIANTS["hl90"], month=month, seed=seed)
    )


def pretrained_file(*, arm: f.Arm, month: str, seed: int) -> Path:
    """Return the pretrained weights an arm starts from."""
    if arm.pretrained == "L0PT":
        return l0pt_file(month=month, seed=seed)
    return W / "pretrained" / "M0" / f"{month}_s{seed}.pt"


c.new_model, c.load_pretrained, c.train = new_model, load_pretrained, train
k.cutoffs = cutoffs
f.pretrained_file = pretrained_file
f.W = W
f.model_file = lambda *, arm, month, day, seed, stress="": (
    W / "models" / arm.tag / f"{month}_d{day}_s{seed}{stress}.pt"
)


def pretrain(*, device: str) -> None:
    """Pretrain on ERA5 and CAMS rows (experiment 448's rows) for each month and seed that has no weights yet."""
    assert FAMILY != "plain-fresh", "M0-mix starts from the L0PT weights; pretrain with --arm L-1d"
    clean = c.clean_stack()
    for month in ARGS.months:
        todo = [
            s
            for s in ARGS.seeds
            if not (
                (W / "pretrained" / "M0" / f"{month}_s{s}.pt").exists()
                if PRETRAINED == "M0"
                else l0pt_file(month=month, seed=s).exists()
            )
        ]
        if not todo:
            continue
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
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
            if PRETRAINED == "M0":
                model = new_model(
                    outlier=False,
                    n_inputs=m.N_INPUTS,
                    n_primaries=len(rows.target_mean),
                    stage="pre",
                ).to(device)
                target = W / "pretrained" / "M0" / f"{month}_s{seed}.pt"
            else:
                model = _new_model(
                    outlier=False, n_inputs=pre.x.shape[1], n_primaries=len(rows.target_mean)
                ).to(device)
                target = W / "pretrained" / "L0PT" / f"{month}_s{seed}.pt"
            epochs, best = train(
                model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                lr=c.PRETRAIN_LR, outlier=False, prior=False, device=device,
            )  # fmt: skip
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  pretrain {PRETRAINED} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
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


def source_folder(*, seed: int) -> Path:
    """Return the folder holding the arm's window forecasts of a seed (the earlier experiment's for reused seeds)."""
    if seed in SEEDS_REUSED and ARGS.arm == "L-1d":
        return W511 / "fits" / "L-1d"
    if seed in SEEDS_REUSED and ARGS.arm in {"M0", "M0lr"}:
        return W545 / "fits" / ARGS.arm
    return W / "fits" / ARGS.arm


def assemble() -> None:
    """Concatenate each cutoff's window forecast into one table per month and seed (seeds 0 to 4)."""
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence=ARM.cadence)
        for seed in ASSEMBLED_SEEDS.get(ARGS.arm, (0, 1, 2)):
            folder = source_folder(seed=seed)
            parts = [
                folder / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            assert all(p.exists() or p.with_suffix(".empty").exists() for p in parts), (
                month,
                seed,
                folder,
            )
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
        assert (
            not set(ARGS.seeds) & set(SEEDS_REUSED)
            or ARGS.arm not in {"L-1d", "M0", "M0lr"}
            or ARGS.stress
        ), "seeds 0 and 1 of this arm come from the earlier experiment"
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

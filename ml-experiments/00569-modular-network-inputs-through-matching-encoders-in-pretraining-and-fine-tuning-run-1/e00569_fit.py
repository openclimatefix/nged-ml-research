# ruff: noqa: E501, ANN001, ANN201, D103, PT018
"""Experiment 569, run 1: inputs enter the modular lag-free network in pre-training AND fine-tuning, through the matching encoder.

Rows, cutoffs' training windows, recency weights (90-day half-life), early stopping, seeds, scoring
windows and the fine-tuning loop are experiment 511's L-1d through experiment 554's M0 loop (fresh
mixer at 5e-4, stage `ft`). The differences from experiment 562: (1) each fold's network is
pre-trained on ERA5 and CAMS rows (experiment 448's rows) with the extra columns, and (2) the refit
is coarse, one fit per fold at the first day of the month (`--cut-day 0`, scored for the whole
month); the leak check refits once at day 14 (`--cut-day 14`), with and without the injected shift.

Arms: `M0` (control, experiment 554's pre-trained weights, no extra columns); `B1`, `B2`, `B3a`,
`B3b`, `B4`, `B3c` and the shuffled-day placebos `B1p`, `B2p`, `B3ap`, `B3bp`, `B4p`.

Usage: python e00569_fit.py {pretrain|fit|assemble|show|columns} --arm ARM [--months ...] [--seeds ...] [--cut-day N] [--stress]
"""

import argparse
import dataclasses
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
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import e00513_data as d513  # ty: ignore[unresolved-import]
import e00515_placebo as placebo  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00562_columns as c562  # ty: ignore[unresolved-import]
import e00569_columns as cols
import e00569_model as xm
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
W = WORK / "work"
W554 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out"
    / "work"
)
FRESH_SEED = 12345
MIXER_LR = 5e-4
PRESENT_FRACTION = 0.5
ARMS = ("M0", *cols.ARMS, *[f"{a}p" for a in cols.ARMS if a != "B3c"])

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("pretrain", "fit", "assemble", "show", "columns"))
parser.add_argument("--arm", choices=ARMS, required=True)
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0])
parser.add_argument("--cut-day", type=int, default=0)
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
CONTROL = ARGS.arm == "M0"
BASE = cols.base_arm(arm=ARGS.arm) if not CONTROL else "M0"
WIDTH = 0 if CONTROL else cols.width(arm=ARGS.arm)
ROUTE = None if CONTROL else cols.ROUTE[BASE]
ARM = f.Arm(ARGS.arm, "d", 90.0, pretrained="M0")
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}
GRID_START, STEP = c.prep422.GRID_START_US, c.prep422.STEP_US
FOLDS: dict[str, c562.FoldColumns] = {}
_, COLUMNS, _ = data.non_lag_layout()
assert len(COLUMNS) == m.N_INPUTS


def proximity() -> dict[int, float]:
    """Return each Primary's schools within 2 km as a share of the largest count (B3c's gate; 0 means no school)."""
    t = pl.read_csv(WORK / "school_proximity.csv")
    top = float(t[cols.SCHOOL_RADIUS_KM].max())
    return {
        int(p): float(n) / top for p, n in zip(t["primary"], t[cols.SCHOOL_RADIUS_KM], strict=True)
    }


EX = None if CONTROL else cols.Extras(arm=ARGS.arm, proximity=proximity())


def month_code(*, month: str) -> int:
    return int(month.replace("-", ""))


def stats_file(*, month: str) -> Path:
    return W / "stats" / ARGS.arm / f"{month}.npz"


def pretrained_file(*, arm, month: str, seed: int) -> Path:
    if CONTROL:
        path = W554 / "pretrained" / "M0" / f"{month}_s{seed}.pt"
    else:
        path = W / "pretrained" / ARGS.arm / f"{month}_s{seed}.pt"
    assert path.exists(), path
    return path


def new_model(
    *, outlier: bool, n_inputs: int, n_primaries: int, stage: str | None = None
) -> torch.nn.Module:
    assert not outlier
    assert n_inputs == m.N_INPUTS + WIDTH, (n_inputs, WIDTH)
    if CONTROL:
        return m.ModularMdn(n_primaries=n_primaries, stage=stage or "ft")  # ty: ignore[invalid-argument-type]
    return xm.ExtraMdn(n_primaries=n_primaries, stage=stage or "ft", route=ROUTE, n_extra=WIDTH)  # ty: ignore[invalid-argument-type]


def load_pretrained(*, model: torch.nn.Module, path: Path) -> None:
    """Load the pre-trained weights without the pre-training mixer; the fine-tuning mixer starts fresh (seed 12345)."""
    raw = torch.load(path, map_location="cpu")
    state = {a: v for a, v in raw.items() if not a.startswith("mixer_ft.")}
    missing = model.load_state_dict(state, strict=False)
    assert all(a.startswith("mixer_ft.") for a in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FRESH_SEED)
        fresh = base.mlp(
            n_in=model.mixer_ft[0].in_features,  # ty: ignore[not-subscriptable,unresolved-attribute]
            hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT,
        )  # fmt: skip
        base.init_head(head=fresh, components=m.COMPONENTS)
    model.mixer_ft.load_state_dict(fresh.state_dict())  # ty: ignore[unresolved-attribute]


def train(
    *, model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, p: torch.Tensor,
    is_val: torch.Tensor, seed: int, lr: float, outlier: bool, prior: bool, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Experiment 554's loop: the fresh mixer has its own learning rate at fine-tuning only."""
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
        groups.append({"params": fresh, "lr": MIXER_LR})
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
            val = float(torch.cat([base.mixture_nll(out=model(x[i], p[i]), z=z[i], components=m.COMPONENTS) for i in val_idx.split(c.BATCH * 2)]).mean())  # fmt: skip
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
f.pretrained_file = pretrained_file
f.W = W
f.model_file = lambda *, arm, month, day, seed, stress="": (
    W / "models" / ARGS.arm / f"{month}_d{day}_s{seed}{stress}.pt"
)
_cutoffs = k.cutoffs


def cutoffs(*, month: str, cadence: str) -> list[datetime]:
    """One cutoff per fold: `--cut-day` days after the month's start (coarse refit)."""
    return [c.month_start(month=month) + timedelta(days=ARGS.cut_day)]


k.cutoffs = cutoffs
_extension = c._extension


def extension(*, refit, last, clean, rows) -> dict:
    """Experiment 510's extension; empty when the only cutoff is the refit itself (no rows after it)."""
    if last <= refit:
        return {
            "x": np.empty((0, m.N_INPUTS), dtype=np.float32), "y": np.empty(0, dtype=np.float32),
            "origin": np.empty(0, dtype=np.int64), "valid": np.empty(0, dtype=np.int64),
            "target_idx": np.empty(0, dtype=np.int64), "lead": np.empty(0, dtype=rows.lead.dtype),
            "primary": np.empty(0, dtype=np.int64), "is_val": np.empty(0, dtype=bool),
        }  # fmt: skip
    return _extension(refit=refit, last=last, clean=clean, rows=rows)


c._extension = extension
assert not ARGS.stress or ARGS.cut_day == 14, "the injected-shift check is defined at cut day 14"


def fit_until(*, month: str) -> int:
    return int((c.us(t=c.month_start(month=month)) - GRID_START) // STEP)


# --- pre-training -------------------------------------------------------------------------


def pretrain_columns(*, month: str, pre) -> np.ndarray:
    """Return the arm's standardised extra columns for the fold's pre-training rows, and store the fold's scale."""
    n = len(pre.y)
    fit_idx = fit_until(month=month)
    primary, oi, ti = pre.primary, pre.origin_idx, pre.target_idx
    if BASE == "B4":
        shapes = d513.ShapeTables(fit_until_idx=fit_idx)
        rng = np.random.default_rng([569, month_code(month=month)])
        present = rng.random(n) < PRESENT_FRACTION
        shape = shapes.columns(source="placebo" if EX.placebo else "price", target_idx=ti)  # ty: ignore[unresolved-attribute]
        out = np.concatenate(
            [np.where(present[:, None], shape, 0.0), present[:, None]], axis=1
        ).astype(np.float32)
        CHECK["months"].setdefault(month, {})["pretrain_present_share"] = float(present.mean())
        return out
    real = EX.raw_pretrain(primary=primary, origin_idx=oi, target_idx=ti, shift=None)  # ty: ignore[unresolved-attribute]
    EX.fit(raw=real)  # ty: ignore[unresolved-attribute]
    stats = stats_file(month=month)
    stats.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats, mean=EX.mean, std=EX.std)  # ty: ignore[unresolved-attribute]
    if not EX.placebo:  # ty: ignore[unresolved-attribute]
        CHECK["months"].setdefault(month, {})["pretrain_present_share"] = float(
            np.isfinite(real).all(axis=1).mean()
        )
        return EX.scale(raw=real)  # ty: ignore[unresolved-attribute]
    days = placebo.DayStructure(fit_until_idx=fit_idx, allowed=None)
    raw = np.full(real.shape, np.nan, dtype=np.float32)
    todo = np.ones(n, dtype=bool)
    for attempt in range(len(placebo.SEEDS)):
        shift = days.shifts(valid_idx=oi if BASE == "B3b" else ti, attempt=attempt)
        idx = np.nonzero(todo & np.isfinite(shift))[0]
        if not len(idx):
            continue
        got = EX.raw_pretrain(
            primary=primary[idx], origin_idx=oi[idx], target_idx=ti[idx], shift=shift[idx]
        )  # ty: ignore[unresolved-attribute]
        raw[idx] = got
        todo[idx] = False
    CHECK["months"].setdefault(month, {})["pretrain_placebo_rows_without_donor"] = int(todo.sum())
    # a placebo keeps the real arm's presence flag (history availability is not content)
    for i, name in enumerate(cols.COLUMNS[BASE]):
        if name.endswith("_present"):
            raw[:, i] = real[:, i]
    return EX.scale(raw=raw)  # ty: ignore[unresolved-attribute]


def pretrain(*, device: str) -> None:
    """Pre-train on ERA5 and CAMS rows (experiment 448's rows) with the arm's extra columns, per fold and seed."""
    assert not CONTROL, "arm M0 starts from experiment 554's pre-trained weights"
    clean = c.clean_stack()
    for month in ARGS.months:
        todo = [
            s for s in ARGS.seeds if not (W / "pretrained" / ARGS.arm / f"{month}_s{s}.pt").exists()
        ]
        if not todo:
            continue
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
        latest = datetime.fromtimestamp((pre.target_idx.max() * STEP + GRID_START) / 1e6, UTC)
        assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
        extra = pretrain_columns(month=month, pre=pre)
        assert extra.shape == (len(pre.y), WIDTH) and np.isfinite(extra).all()
        print(f"{month}: pretraining rows {len(pre.x):,}; latest valid {latest.isoformat()} (refit {refit_t.isoformat()}); extra columns {WIDTH}, mean {extra.mean():+.3f}, sd {extra.std():.3f}", flush=True)  # fmt: skip
        i = int(len(pre.y) // 2)
        print(f"  example pretraining row: primary {int(pre.primary[i])}, origin {datetime.fromtimestamp((pre.origin_idx[i] * STEP + GRID_START) / 1e6, UTC)}, valid {datetime.fromtimestamp((pre.target_idx[i] * STEP + GRID_START) / 1e6, UTC)} (ERA5/CAMS stand-in, as the 37 base inputs), extra {np.round(extra[i], 3).tolist()}", flush=True)  # fmt: skip
        z = (pre.y - rows.target_mean[pre.primary]) / rows.target_std[pre.primary]
        n = len(pre.y)
        x = torch.empty((n, m.N_INPUTS + WIDTH), device=device)
        for a in range(0, n, 2_000_000):
            b = min(a + 2_000_000, n)
            x[a:b, : m.N_INPUTS] = torch.from_numpy(pre.x[a:b]).to(device)
            x[a:b, m.N_INPUTS :] = torch.from_numpy(extra[a:b]).to(device)
        zt = torch.from_numpy(z.astype(np.float32)).to(device)
        wt, pt, vt = (
            torch.ones(n, device=device),
            torch.from_numpy(pre.primary).to(device),
            torch.from_numpy(pre.is_val).to(device),
        )
        del pre, rows, extra
        for seed in todo:
            started = time.monotonic()
            torch.manual_seed(seed)
            model = new_model(
                outlier=False, n_inputs=m.N_INPUTS + WIDTH, n_primaries=16, stage="pre"
            ).to(device)
            epochs, best = train(model=model, x=x, z=zt, w=wt, p=pt, is_val=vt, seed=seed, lr=c.PRETRAIN_LR, outlier=False, prior=False, device=device)  # fmt: skip
            target = W / "pretrained" / ARGS.arm / f"{month}_s{seed}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(f"  pretrain {ARGS.arm} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s", flush=True)  # fmt: skip
        del x, zt, wt, pt, vt
        torch.cuda.empty_cache()


# --- fine-tuning and scoring rows ------------------------------------------------------------


def _idx(*, us: np.ndarray) -> np.ndarray:
    return ((us - GRID_START) // STEP).astype(np.int64)


def finetune_columns(*, month: str, primary, origin_idx, target_idx, present) -> np.ndarray:
    """Return the arm's standardised extra columns for fine-tuning or scoring rows (stage 2), on the fold's pre-training scale."""
    fc = FOLDS[month]
    if BASE == "B4":
        return fc.price_columns(arm=ARGS.arm, target_idx=target_idx, present=present)
    stats = np.load(stats_file(month=month))
    EX.mean, EX.std = stats["mean"], stats["std"]  # ty: ignore[unresolved-attribute]
    table_arm = "B3a" if BASE == "B3c" else BASE
    getter = fc._raw_placebo if EX.placebo else fc._raw_real  # ty: ignore[unresolved-attribute]
    raw = getter(arm=table_arm, primary=primary, origin_idx=origin_idx, valid_idx=target_idx)
    flag = np.ones((len(raw), 1), dtype=np.float32)
    if BASE in {"B3a", "B3b", "B3c"}:
        raw = np.concatenate([raw, flag], axis=1)
    if BASE == "B3c":
        gate = EX._gate(primary=primary)[:, None]  # ty: ignore[unresolved-attribute]
        raw = np.concatenate([raw, np.nan_to_num(raw[:, :5]) * gate], axis=1)
    return EX.scale(raw=raw)  # ty: ignore[unresolved-attribute]


_build_month, _build_test = k.build_month, data.build_test


def build_month(*, month, clean):
    """Build the month's training rows, assert each origin's weather run precedes it, and append the extra columns."""
    mr = _build_month(month=month, clean=clean)
    assert (mr.valid_us > mr.origin_us).all(), f"{month}: a training target is not after its origin"
    runs = np.unique(mr.origin_us)
    delay = int(common.PUBLICATION_DELAY.total_seconds() * 1e6)
    miss = [r for r in runs if not (prep.SUMMARY_DIR / f"init={datetime.fromtimestamp((r - delay) / 1e6, UTC):%Y-%m-%d}.parquet").exists()]  # fmt: skip
    assert not miss, (
        f"{month}: {len(miss)} training origins have no weather-run summary file {common.PUBLICATION_DELAY} earlier"
    )
    CHECK["train_rows"] += len(mr.y)
    CHECK["months"].setdefault(month, {})["train_rows"] = len(mr.y)
    if CONTROL:
        print(month, "train checks passed", flush=True)
        return mr
    for other in [x for x in FOLDS if x != month]:
        del FOLDS[other]
    if month not in FOLDS:
        FOLDS[month] = c562.FoldColumns(fit_until_idx=fit_until(month=month))
    origin_idx = _idx(us=mr.origin_us)
    rng = np.random.default_rng([569, month_code(month=month), 2])
    present = rng.random(len(mr.y)) < PRESENT_FRACTION
    add = finetune_columns(
        month=month,
        primary=mr.primary,
        origin_idx=origin_idx,
        target_idx=mr.target_idx,
        present=present,
    )
    assert add.shape == (len(mr.y), WIDTH) and np.isfinite(add).all()
    i = int(np.argmax(mr.valid_us > mr.origin_us + 24 * 3_600_000_000))
    CHECK["months"][month]["train_example"] = f"origin {datetime.fromtimestamp(mr.origin_us[i] / 1e6, UTC)}, weather run {datetime.fromtimestamp(mr.origin_us[i] / 1e6, UTC) - common.PUBLICATION_DELAY}, valid {datetime.fromtimestamp(mr.valid_us[i] / 1e6, UTC)}, primary {int(mr.primary[i])}, extra {np.round(add[i], 3).tolist()}"  # fmt: skip
    print(month, "train checks passed |", CHECK["months"][month]["train_example"], flush=True)
    return dataclasses.replace(mr, x=np.concatenate([mr.x, add], axis=1))


def build_test(*, month, standardiser):
    """Build the month's test rows, assert nwp_init_time <= power_fcst_init_time for each origin, and append the extra columns (B4 scored without price)."""
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
    if CONTROL:
        return test
    add = finetune_columns(month=month, primary=test.primary, origin_idx=test.origin_idx, target_idx=test.target_idx, present=np.zeros(len(test.x), dtype=bool))  # fmt: skip
    assert add.shape == (len(test.x), WIDTH) and np.isfinite(add).all()
    CHECK["months"][month]["test_example"] = f"origin {datetime.fromtimestamp(test.origin[0] / 1e6, UTC)}, weather run {datetime.fromtimestamp(test.origin[0] / 1e6, UTC) - pub}, valid {datetime.fromtimestamp(test.valid[0] / 1e6, UTC)}, extra {np.round(add[0], 3).tolist()} (nwp_init_time <= power_fcst_init_time asserted for all rows of every test origin)"  # fmt: skip
    print(month, "test checks passed |", CHECK["months"][month]["test_example"], flush=True)
    return dataclasses.replace(test, x=np.concatenate([test.x, add], axis=1))


k.build_month, data.build_test = build_month, build_test


def assemble() -> None:
    """Concatenate each fold's coarse-refit forecast into one table per month and seed."""
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence="d")
        for seed in ARGS.seeds:
            parts = [W / "fits" / ARM.tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet" for cut in cuts]  # fmt: skip
            assert all(p.exists() or p.with_suffix(".empty").exists() for p in parts), (month, seed)
            frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
            f.atomic_frame(
                frame=frame, target=W / "seed_preds" / ARM.tag / f"{month}_s{seed}.parquet"
            )
            print("wrote", ARM.tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble()
    elif ARGS.command == "show":
        model = new_model(outlier=False, n_inputs=m.N_INPUTS + WIDTH, n_primaries=16)
        print(
            ARGS.arm,
            "extra columns",
            WIDTH,
            "route",
            ROUTE,
            "parameters",
            sum(q.numel() for q in model.parameters()),
        )
    elif ARGS.command == "pretrain":
        pretrain(device="cuda")
    else:
        started = time.monotonic()
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
            / f"checks_{ARM.tag}_s{'-'.join(map(str, ARGS.seeds))}_d{ARGS.cut_day}{'_stress' if ARGS.stress else ''}.json"
        )
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in CHECK.items() if a != "months"}, f"{time.monotonic() - started:.0f} s")  # fmt: skip

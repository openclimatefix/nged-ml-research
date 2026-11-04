# ruff: noqa: E501, ANN001, ANN201, PT018
"""Experiment 562: fine-tune and forecast the modular lag-free network M0 with extra input columns.

Rows, cutoffs, recency weights, early stopping, seeds, scoring windows, and the daily harness are
experiment 511's L-1d (daily refit, 90-day half-life), through experiment 554's M0 fine-tuning loop
(fresh mixer at 5e-4, stage `ft`). The pretrained weights are experiment 554's M0 weights, read-only.
The extra columns (`e00562_columns`) enter only at fine-tuning, through zero-initialised input units
(`e00562_model`).

Arms: `B1`, `B2`, `B3a`, `B3b`, `B4` and their shuffled-day placebos `B1p`, `B2p`, `B3ap`, `B3bp`,
`B4p`; `B4-oracle` scores arm B4's fitted networks with the real price shape present (it fits nothing:
run it after B4).

Usage: python e00562_fit.py {fit|assemble|show} --arm ARM [--stress] [--months ...] [--seeds ...]
"""

import argparse
import dataclasses
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00562_columns as cols
import e00562_model as xm
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00562-modular-lag-free-network-null-input-retry-with-placebos"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
W = WORK / "work"
W554 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out"
    / "work"
)
FRESH_SEED = 12345
"""Initialisation seed of the fresh mixer, as in experiments 545 and 554."""
MIXER_LR = 5e-4
"""Learning rate of the fresh fine-tuning mixer in arm M0 (experiments 545 and 554)."""
PRESENT_FRACTION = 0.5
"""Share of fine-tuning rows that carry the real price shape in arms B4 and B4p."""
ARMS = ("B1", "B2", "B3a", "B3b", "B4", "B1p", "B2p", "B3ap", "B3bp", "B4p", "B4-oracle")

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("fit", "assemble", "show", "checks"))
parser.add_argument("--arm", choices=ARMS, required=True)
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0])
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
BASE = cols._base_arm(arm=ARGS.arm)
ORACLE = ARGS.arm.endswith("-oracle")
STORE_AS = BASE if ORACLE else ARGS.arm
"""Arm whose fitted networks the run reads and writes (`B4-oracle` reads arm B4's)."""
WIDTH = cols.width(arm=ARGS.arm)
ARM = f.Arm(ARGS.arm, "d", 90.0, pretrained="M0")
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}

_, COLUMNS, _ = data.non_lag_layout()
assert len(COLUMNS) == m.N_INPUTS
assert (
    COLUMNS[m.WEATHER_COLS][0] == "mean_shortwave"
    and COLUMNS[m.WEATHER_COLS][-1] == "mean_wind_speed_10m"
)
assert all(n.startswith("std_") for n in COLUMNS[m.SPREAD_COLS])
assert (
    COLUMNS[m.DATETIME_COLS][0] == "cos_zenith" and COLUMNS[m.DATETIME_COLS][-1] == "nearest_inside"
)
assert COLUMNS[m.LEAD_COLS] == ["lead", "lead_sqrt"]

FOLDS: dict[str, cols.FoldColumns] = {}
GRID_START, STEP = c.prep422.GRID_START_US, c.prep422.STEP_US


def new_model(
    *, outlier: bool, n_inputs: int, n_primaries: int, stage: str | None = None
) -> torch.nn.Module:
    """Return an untrained network (the signature of `e00510_common.new_model`)."""
    assert not outlier
    assert n_inputs == m.N_INPUTS + WIDTH, (n_inputs, WIDTH)
    return xm.ExtraMdn(n_primaries=n_primaries, encoder=cols.ENCODER[BASE], n_extra=WIDTH)


def load_pretrained(*, model: torch.nn.Module, path: Path) -> None:
    """Load experiment 554's M0 weights, zero-padding the widened first layer; the mixer starts fresh."""
    xm.load_pretrained_padded(model=model, path=str(path), fresh_seed=FRESH_SEED)  # ty: ignore[invalid-argument-type]


def train(
    *, model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, p: torch.Tensor,
    is_val: torch.Tensor, seed: int, lr: float, outlier: bool, prior: bool, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Experiment 554's loop for arm M0: experiment 510's loop with the fresh mixer in its own learning-rate group."""
    assert not outlier
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    fresh = list(model.mixer_ft.parameters())  # ty: ignore[unresolved-attribute]
    fresh_ids = {id(q) for q in fresh}
    groups = [
        {"params": [q for q in model.parameters() if id(q) not in fresh_ids], "lr": lr},
        {"params": fresh, "lr": MIXER_LR},
    ]
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


def pretrained_file(*, arm: f.Arm, month: str, seed: int) -> Path:
    """Return experiment 554's pretrained M0 weights (read-only)."""
    path = W554 / "pretrained" / "M0" / f"{month}_s{seed}.pt"
    assert path.exists(), path
    return path


c.new_model, c.load_pretrained, c.train = new_model, load_pretrained, train
f.pretrained_file = pretrained_file
f.W = W
f.model_file = lambda *, arm, month, day, seed, stress="": (
    W / "models" / STORE_AS / f"{month}_d{day}_s{seed}{stress}.pt"
)


def _idx(*, us: np.ndarray) -> np.ndarray:
    return ((us - GRID_START) // STEP).astype(np.int64)


def month_code(*, month: str) -> int:
    """Return `2025-10` as 202510."""
    return int(month.replace("-", ""))


def extras(
    *, month: str, primary, origin_idx, target_idx, present: np.ndarray | None
) -> np.ndarray:
    """Return the arm's extra columns for some rows (arm B4 and B4p use `present`; the others ignore it)."""
    fc = FOLDS[month]
    if BASE == "B4":
        assert present is not None
        return fc.price_columns(arm=ARGS.arm, target_idx=target_idx, present=present)
    return fc.table_columns(
        arm=ARGS.arm, primary=primary, origin_idx=origin_idx, valid_idx=target_idx
    )


_build_month, _build_test = k.build_month, data.build_test


def build_month(*, month, clean):
    """Build the month's rows, assert each training origin's weather run exists and precedes it, and append the arm's columns."""
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
    refit = c.month_start(month=month)
    fit_until = int((c.us(t=refit) - GRID_START) // STEP)
    if month not in FOLDS:
        FOLDS[month] = cols.FoldColumns(fit_until_idx=fit_until)
    fc = FOLDS[month]
    origin_idx = _idx(us=mr.origin_us)
    train_rows = np.zeros(len(mr.y), dtype=bool)
    train_rows[: mr.n_base] = ~mr.is_val[: mr.n_base]
    fc.fit_standardiser(
        primary=mr.primary, origin_idx=origin_idx, valid_idx=mr.target_idx, train=train_rows
    )
    rng = np.random.default_rng([562, month_code(month=month)])
    present = rng.random(len(mr.y)) < PRESENT_FRACTION
    add = extras(
        month=month,
        primary=mr.primary,
        origin_idx=origin_idx,
        target_idx=mr.target_idx,
        present=present,
    )
    assert add.shape == (len(mr.y), WIDTH) and np.isfinite(add).all()
    CHECK["train_rows"] += len(mr.y)
    i = int(np.argmax(mr.valid_us > mr.origin_us + 24 * 3_600_000_000))
    CHECK["months"][month] = {
        "train_rows": len(mr.y), "train_origins": len(runs), "placebo_rows_without_donor": int(fc.unfilled),
        "extra_columns": WIDTH, "present_share": float(present.mean()) if BASE == "B4" else None,
        "train_example": f"origin {datetime.fromtimestamp(mr.origin_us[i] / 1e6, UTC)}, weather run {datetime.fromtimestamp(mr.origin_us[i] / 1e6, UTC) - common.PUBLICATION_DELAY}, valid {datetime.fromtimestamp(mr.valid_us[i] / 1e6, UTC)}, primary {int(mr.primary[i])}, extra columns {np.round(add[i], 3).tolist()}",
    }  # fmt: skip
    print(month, "train checks passed |", CHECK["months"][month]["train_example"], flush=True)
    return dataclasses.replace(mr, x=np.concatenate([mr.x, add], axis=1))


def build_test(*, month, standardiser):
    """Build the month's test rows, assert nwp_init_time <= power_fcst_init_time for each test origin, and append the arm's columns."""
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
    present = np.full(len(test.x), ORACLE, dtype=bool)
    add = extras(
        month=month,
        primary=test.primary,
        origin_idx=test.origin_idx,
        target_idx=test.target_idx,
        present=present,
    )
    assert add.shape == (len(test.x), WIDTH) and np.isfinite(add).all()
    CHECK["test_rows"] += len(test.x)
    CHECK["months"][month]["test_example"] = (
        f"origin {datetime.fromtimestamp(test.origin[0] / 1e6, UTC)}, weather run {datetime.fromtimestamp(test.origin[0] / 1e6, UTC) - pub}, "
        f"valid {datetime.fromtimestamp(test.valid[0] / 1e6, UTC)}, extra columns {np.round(add[0], 3).tolist()} "
        "(nwp_init_time <= power_fcst_init_time asserted for all rows of every test origin)"
    )
    print(month, "test checks passed |", CHECK["months"][month]["test_example"], flush=True)
    return dataclasses.replace(test, x=np.concatenate([test.x, add], axis=1))


k.build_month, data.build_test = build_month, build_test


def assemble() -> None:
    """Concatenate each cutoff's window forecast into one table per month and seed."""
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence="d")
        for seed in ARGS.seeds:
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


def checks(*, month: str) -> None:
    """Print the sanity checks of the arm on one fold: widths, placebo overlap, presence flags, and zero-start equality with M0."""
    clean = c.clean_stack()
    mr = k.build_month(month=month, clean=clean)
    test = data.build_test(month=month, standardiser=mr.base.standardiser)
    extra_train, extra_test = mr.x[:, m.N_INPUTS :], test.x[:, m.N_INPUTS :]
    print(
        f"{ARGS.arm} {month}: {len(mr.y):,} training rows, {len(test.x):,} test rows, {WIDTH} extra columns"
    )
    print(f"  training extras mean {extra_train.mean():+.3f}, sd {extra_train.std():.3f}; scoring extras mean {extra_test.mean():+.3f}, sd {extra_test.std():.3f}")  # fmt: skip
    print(f"  all-zero scoring rows: {float((extra_test == 0).all(axis=1).mean()):.3f}; all-zero training rows: {float((extra_train == 0).all(axis=1).mean()):.3f}")  # fmt: skip
    if BASE != "B4":
        real = cols.FoldColumns.table_columns(
            FOLDS[month],
            arm=BASE,
            primary=mr.primary,
            origin_idx=_idx(us=mr.origin_us),
            valid_idx=mr.target_idx,
        )
        placebo_arm = BASE + "p"
        other = FOLDS[month].table_columns(arm=placebo_arm, primary=mr.primary, origin_idx=_idx(us=mr.origin_us), valid_idx=mr.target_idx)  # fmt: skip
        same = float((real == other).all(axis=1).mean())
        print(
            f"  share of training rows whose placebo columns equal the real columns exactly: {same:.4f}"
        )
        print(f"  placebo rows with no donor: {FOLDS[month].unfilled}")
    # Zero start: the widened network equals experiment 554's M0 on the same rows before fine-tuning.
    path = pretrained_file(arm=ARM, month=month, seed=0)
    plain = m.ModularMdn(n_primaries=len(mr.base.target_mean), stage="ft")
    plain_state = {
        a: v
        for a, v in torch.load(path, map_location="cpu").items()
        if not a.startswith("mixer_ft.")
    }
    plain.load_state_dict(plain_state, strict=False)
    plain.mixer_ft.load_state_dict(
        xm.ExtraMdn(
            n_primaries=len(mr.base.target_mean), encoder=cols.ENCODER[BASE], n_extra=WIDTH
        ).mixer_ft.state_dict()
    )  # ty: ignore[unresolved-attribute]
    wide = new_model(
        outlier=False, n_inputs=m.N_INPUTS + WIDTH, n_primaries=len(mr.base.target_mean)
    )
    load_pretrained(model=wide, path=path)
    plain.mixer_ft.load_state_dict(wide.mixer_ft.state_dict())  # ty: ignore[unresolved-attribute]
    plain.eval()
    wide.eval()
    x, p = torch.from_numpy(test.x[:5000]), torch.from_numpy(test.primary[:5000])
    with torch.no_grad():
        gap = float((plain(x[:, : m.N_INPUTS], p) - wide(x, p)).abs().max())
    print(f"  largest output difference between M0 and the zero-started widened network: {gap:.2e}")
    assert gap < 1e-5


if __name__ == "__main__":
    if ARGS.command == "checks":
        checks(month=ARGS.months[0])
    elif ARGS.command == "assemble":
        assemble()
    elif ARGS.command == "show":
        model = new_model(outlier=False, n_inputs=m.N_INPUTS + WIDTH, n_primaries=16)
        print(ARGS.arm, "extra columns", WIDTH, "encoder", cols.ENCODER[BASE], "parameters", sum(q.numel() for q in model.parameters()))  # fmt: skip
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
            / f"checks_{ARM.tag}_s{'-'.join(map(str, ARGS.seeds))}{'_stress' if ARGS.stress else ''}.json"
        )
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print(
            "CHECKS PASSED",
            {a: v for a, v in CHECK.items() if a != "months"},
            f"{time.monotonic() - started:.0f} s",
        )

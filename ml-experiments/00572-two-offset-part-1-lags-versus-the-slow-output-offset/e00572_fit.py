# ruff: noqa: E501, ANN001, ANN201, PT018, D102, RUF028
"""Experiment 572: coarse-refit fits of the offset (D5) and lag (v3) arms of Part 1 (GPU).

A copy of experiment 565's `e00565_fit.py` (coarse cadence `h`: refits on days 0 and 14 of each fold;
rows, recency weights, early stopping, seeds, scoring windows, origin-time weather assertions, the
frozen-encoder D5 fine-tune) with experiment 560's lag machinery layered on (`e00560_fit.py`: capped
day and week lags, block dropout, the expected-input table, the shifted-lag stress run, and the
lags-removed and lags-aged scoring twins). Arms: E2-ft (D5 learnt in fine-tuning only), V3 (experiment
560's v3, refitted coarse from its stored pre-training), V3D5 (v3 plus the D5 offset, pre-trained
here), and V35 (v3 with lags observed at least 3.5 h before the origin, pre-trained here). E2 and M0
are experiment 565's stored arms D5 and C-M0.

Usage: python e00572_fit.py {pretrain-wt|fit|assemble} --arm ARM [--stress] [--months ...] [--seeds ...]
"""

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
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
import e00558_lags as lags  # ty: ignore[unresolved-import]
import e00559_extra as extra  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_data as d560  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00572_lags as lg
import e00572_model as mod
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00572-two-offset-part-1-lags-versus-the-slow-output-offset"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
W = WORK / "work"
W559 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty/work"
)
"""Experiment 559's work folder; only its `pretrained_drift/real` weights (arm A10) are read."""
W560 = ML_EXPERIMENTS_DATA_DIR / "00560-modular-lag-encoder-capped-lags-and-block-dropout/work"
"""Experiment 560's work folder; only its `pretrained/LAG-v3` weights (arm V3) are read."""
W554 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out/work"
)
"""Experiment 554's work folder; only its `pretrained/M0` weights are read."""
FRESH_SEED = 12345
"""Initialisation seed of every fresh mixer, as in experiments 545 and 554."""
FRESH_LR = 5e-4
"""Learning rate of the fresh mixer (M0's, and step 2's)."""
COARSE_OFFSETS = (0, 14)
"""Cadence `h`: refits on days 0 and 14, so that the injected-shift control (the day-14 fit) exists."""
CADENCE_3_OFFSETS = (0, *range(2, 31, 3))
"""Refit days of arm A7: day 0, then every third day from day 2, so that days 14, 17, and 20 are refits."""
UNIFORM_HALF_LIFE = 1e12
"""A half-life so long that every weight is 1 (step 2 has no recency weighting)."""


@dataclass(frozen=True)
class Spec:
    """How an arm's coarse fits start and train."""

    tag: str
    lags: str  # "none", "24h" (experiment 560's cap), or "3.5h"
    variant: str  # "ftonly" (E2-ft), "lag" (v3 or v3-3.5), "lagoff" (v3+D5)
    frozen: bool  # freeze the weather and datetime encoders in fine-tuning, as arm D5 does
    cadence: str = "h"
    half_life: float = 90.0
    lr: float = c.FINE_TUNE_LR
    max_epochs: int = c.MAX_EPOCHS
    start: str = "fresh"
    penalty: bool = False
    head: str = "mixture"
    average_of: str | None = None

    @property
    def columns(self) -> bool:
        """Whether the arm appends the knot-position column."""
        return self.variant in {"ftonly", "lagoff"}

    @property
    def cap_steps(self) -> int:
        """Half-hours between the newest lag and the forecast origin."""
        return lg.CAP_3_5H if self.lags == "3.5h" else lg.CAP_24H


SPECS = {
    s.tag: s
    for s in (
        Spec("E2ft", "none", "ftonly", True),
        Spec("V3", "24h", "lag", False),
        Spec("V3D5", "24h", "lagoff", True),
        Spec("V35", "3.5h", "lag", False),
    )
}
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("fit", "assemble", "pretrain-wt"))
parser.add_argument("--arm", choices=tuple(SPECS), required=True)
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
SPEC = SPECS[ARGS.arm]
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}
_cutoffs = k.cutoffs

_, COLUMNS, _ = data.non_lag_layout()
assert len(COLUMNS) == m.N_INPUTS
assert all(n.startswith("std_") for n in COLUMNS[m.SPREAD_COLS])
assert (
    COLUMNS[m.DATETIME_COLS][0] == "cos_zenith" and COLUMNS[m.DATETIME_COLS][-1] == "nearest_inside"
)
assert COLUMNS[m.LEAD_COLS] == ["lead", "lead_sqrt"]
STRESS_FIRST_DAY = 10
"""Stress fits start four days before the injected block's end (day 14) so that A5 can average 5 fits."""


def cutoffs(*, month: str, cadence: str) -> list[datetime]:
    """Experiment 511's cutoffs; arm A7 (cadence `d` so the windows tile) refits on days 0, 2, 5, 8."""
    if SPEC.cadence == "d":
        return _cutoffs(month=month, cadence=cadence)
    start, stop = c.month_start(month=month), c.month_stop(month=month)
    days = (stop - start).days
    offsets = {"c3": CADENCE_3_OFFSETS, "h": COARSE_OFFSETS}[SPEC.cadence]
    return [start + timedelta(days=d) for d in offsets if d < days]


k.cutoffs = cutoffs
CADENCE = "d"


def pretrained_dir() -> Path:
    """Return where this experiment's own pre-trained weights of the arm live (arms V3D5 and V35)."""
    return W / f"pretrained_{SPEC.tag}"


def pretrained_file(*, month: str, seed: int) -> Path:
    """Return the pre-trained weights: experiment 560's v3 (V3), experiment 554's M0 (E2ft), or this experiment's own."""
    if SPEC.tag == "V3":
        p = W560 / "pretrained" / "LAG-v3" / f"{month}_s{seed}.pt"
    elif SPEC.tag == "E2ft":
        p = W554 / "pretrained" / "M0" / f"{month}_s{seed}.pt"
    else:
        p = pretrained_dir() / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def row_nll(*, out: torch.Tensor, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Return each row's negative log-likelihood."""
    _ = p
    return base.mixture_nll(out=out, z=z, components=m.COMPONENTS)


def fresh_mixer(*, model: torch.nn.Module) -> None:
    """Give the model's `mixer_ft` a seed-12345 initialisation (experiment 554's recipe)."""
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


def load_pretrained_fresh(*, model: torch.nn.Module, path: Path) -> None:
    """Load pre-trained weights (without `mixer_ft`), then a fresh mixer (seed 12345)."""
    raw = torch.load(path, map_location="cpu")
    if SPEC.variant == "lagoff":
        mod.load_lag_pre_into_ft(model=model, state=raw)  # ty: ignore[invalid-argument-type]
    else:
        state = {a: v for a, v in raw.items() if not a.startswith("mixer_ft.")}
        missing = model.load_state_dict(state, strict=False)
        allowed = lambda a: a.startswith("mixer_ft.") or (SPEC.variant == "ftonly" and a == "knots")  # noqa: E731
        assert all(allowed(a) for a in missing.missing_keys), missing
        assert not missing.unexpected_keys, missing
    fresh_mixer(model=model)  # ty: ignore[invalid-argument-type]


def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    """Return an untrained network (fine-tuning stage unless `stage` says `pre`), with the expected-input table attached."""
    if SPEC.variant == "ftonly":
        assert stage == "ft"
        return mod.OffsetFtOnly(n_primaries=n_primaries, n_knots=KNOTS)
    model: torch.nn.Module
    if SPEC.variant == "lagoff":
        model = mod.LagOffsetMdn(n_primaries=n_primaries, stage=stage, n_knots=KNOTS)
    else:
        model = m560.LagModularMdn(n_primaries=n_primaries, mode="v3", stage=stage)
    model.table = CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


def train(  # noqa: PLR0912, PLR0915
    *, model: torch.nn.Module, x, z, w, p, is_val, seed: int, lr: float, max_epochs: int,
    fresh_lr: float | None, frozen: bool, anchor: dict | None, lam: float, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Experiment 510's loop with a separate rate for a fresh mixer, frozen encoders, and an L2 penalty to `anchor`.

    The early-stopping validation NLL has no penalty. Frozen encoders (weather and datetime) are
    held in evaluation mode so their dropout is off.
    """
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    encoder_ids = {id(q) for part in (model.weather, model.datetime) for q in part.parameters()}  # ty: ignore[unresolved-attribute]
    if frozen:
        for part in (model.weather, model.datetime):
            for q in part.parameters():  # ty: ignore[unresolved-attribute]
                q.requires_grad_(False)
    fresh = list(model.mixer_ft.parameters()) if fresh_lr is not None else []  # ty: ignore[unresolved-attribute]
    fresh_ids = {id(q) for q in fresh}
    normal = [
        q
        for q in model.parameters()
        if id(q) not in fresh_ids and not (frozen and id(q) in encoder_ids)
    ]
    groups = [{"params": normal, "lr": lr}]
    if fresh:
        groups.append({"params": fresh, "lr": fresh_lr})
    optimiser = torch.optim.AdamW(groups, weight_decay=c.WEIGHT_DECAY)
    anchors = (
        {a: v.to(device) for a, v in anchor.items()} if anchor is not None and lam > 0 else None
    )
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        if frozen:
            model.weather.eval()  # ty: ignore[unresolved-attribute]
            model.datetime.eval()  # ty: ignore[unresolved-attribute]
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            nll = row_nll(out=model(x[b], p[b]), z=z[b], p=p[b])
            loss = (w[b] * nll).sum() / w[b].sum()
            if hasattr(model, "penalty"):
                loss = loss + model.penalty()  # ty: ignore[call-non-callable]
            if anchors is not None:
                loss = loss + lam * sum(
                    ((q - anchors[a]) ** 2).sum()
                    for a, q in model.named_parameters()
                    if q.requires_grad
                )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        row_nll(out=model(x[i], p[i]), z=z[i], p=p[i])
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
    for q in model.parameters():
        q.requires_grad_(True)
    return epochs, best


_build_month, _build_test = k.build_month, data.build_test
KNOTS: int = 0
WT: extra.DriftColumns | None = None
CTX: dict = {"on": False, "extra": {}}
AGED_DAYS = (1, 2, 3)
SIBLINGS = ["removed", *[f"aged{n}" for n in AGED_DAYS]]
CAP_HOURS = SPEC.cap_steps * 0.5
N_LAG_INPUTS = m560.N_INPUTS  # 37 lag-free columns + 8 lag columns


def apply_dropout(*, x: torch.Tensor, origin_idx: np.ndarray, seed: int) -> dict:
    """Mask lags in place per experiment 558: whole origin days, all lags or the day lag only."""
    mask_all, mask_day = lags.block_dropout_masks(origin_idx=origin_idx, seed=seed)
    ma = torch.from_numpy(mask_all).to(x.device)
    md = torch.from_numpy(mask_day).to(x.device)
    amax = lg.age_max(cap_steps=SPEC.cap_steps)
    for use, c0, age in ((ma | md, 37, amax[0]), (ma, 40, amax[1])):
        x[use, c0], x[use, c0 + 1], x[use, c0 + 2] = 0.0, 1.0, float(age)
    return {
        "share_all_lags_masked": float(mask_all.mean()),
        "share_day_lag_only_masked": float(mask_day.mean()),
        "share_day_lag_missing": float((x[:, 38] > 0.5).float().mean()),
        "share_week_lag_missing": float((x[:, 41] > 0.5).float().mean()),
    }


def lag_delta(*, x_cols: dict, valid_us: np.ndarray, prim: np.ndarray) -> np.ndarray:
    """Return the standardised stress shift to add to the day and week lag values (rows x 2)."""
    shift, sign, blk = CTX["shift"], CTX["sign"], CTX["block"]
    out = np.zeros((len(valid_us), 2), dtype=np.float32)
    for j, (age_col, miss_col) in enumerate(((39, 38), (42, 41))):
        t = valid_us - (x_cols[age_col].astype(np.float64) * lags.LEAD_SCALE_HOURS * 3.6e9).astype(
            np.int64
        )
        inblock = (t >= blk[0]) & (t < blk[1]) & (x_cols[miss_col] < 0.5)
        out[:, j] = np.where(inblock, sign * shift[prim] / CTX["stats"][j, 1], 0.0)
    return out


def shifted(*, x: np.ndarray, valid_us: np.ndarray, prim: np.ndarray) -> np.ndarray:
    """Return a copy of `x` with the stress shift added to the in-block lag values."""
    cols = {n: x[:, n] for n in (38, 39, 41, 42)}
    delta = lag_delta(x_cols=cols, valid_us=valid_us, prim=prim)
    out = x.copy()
    for j, col in enumerate((37, 40)):
        out[:, col] = np.clip(out[:, col] + delta[:, j], -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
    return out


class LagGpuMonth(k.GpuMonth):
    """The month's rows on the GPU; training rows get the stress shift and block dropout."""

    def refit_tensors(self, *, refit, shift_z=None):
        t = super().refit_tensors(refit=refit, shift_z=shift_z)
        idx = refit.index
        if CTX["on"]:
            cols = {n: self.mr.x[idx, n] for n in (38, 39, 41, 42)}
            delta = lag_delta(
                x_cols=cols, valid_us=self.mr.valid_us[idx], prim=self.mr.primary[idx]
            )
            for j, col in enumerate((37, 40)):
                v = t["x"][:, col] + torch.from_numpy(delta[:, j]).to(t["x"].device)
                t["x"][:, col] = v.clamp(-prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
            CTX["n_shifted"] = int((delta != 0).any(axis=1).sum())
        origin_idx = (self.mr.origin_us[idx] - c.prep422.GRID_START_US) // c.prep422.STEP_US
        CTX["dropout"] = apply_dropout(x=t["x"], origin_idx=origin_idx, seed=CTX["seed"])
        return t


if SPEC.lags != "none":
    k.GpuMonth = LagGpuMonth


def block_of(*, clean, primary, target_idx, origin_idx, extra_days=0) -> np.ndarray:
    """Return the 8 lag columns for rows, in chunks."""
    parts = []
    for a in range(0, len(primary), 2_000_000):
        s = slice(a, a + 2_000_000)
        parts.append(
            lg.lag_block(
                clean=clean,
                primary=primary[s],
                target_idx=target_idx[s],
                origin_idx=origin_idx[s],
                stats=CTX["stats"],
                cap_steps=SPEC.cap_steps,
                extra_days=extra_days,
            )  # fmt: skip
        )
    return np.concatenate(parts)


def widen(*, x37: np.ndarray, blk: np.ndarray) -> np.ndarray:
    """Return `x37` with the 8 appended lag columns, filled in chunks to bound peak memory."""
    out = np.empty((len(x37), N_LAG_INPUTS), dtype=np.float32)
    for a in range(0, len(x37), 1_000_000):
        out[a : a + 1_000_000, :37] = x37[a : a + 1_000_000]
    out[:, 37:] = blk
    return out


def attach_tables(*, standardiser) -> None:
    """Build and upload the standardised expected-input table of the fine-tuning (ENS) source."""
    cols, origin = d560.raw_table(source="ft", cache=W560 / "expected")
    CTX["origin_ft"] = np.asarray(origin)
    CTX["table_ft"] = torch.from_numpy(
        d560.standardised_table(cols=cols, standardiser=standardiser)
    ).to("cuda")
    del cols


_window_frame, _atomic_frame = f.window_frame, f.atomic_frame


def window_frame(*, model, test, mr, lo, hi, flag, device):
    """Forecast the window; outside the stress run also forecast with the lags removed and aged."""
    if SPEC.lags == "none":
        return _window_frame(model=model, test=test, mr=mr, lo=lo, hi=hi, flag=flag, device=device)
    model.table = CTX["table_ft"]
    if CTX["on"]:
        test = dataclasses.replace(
            test, x=shifted(x=test.x, valid_us=test.valid, prim=test.primary)
        )
    frame = _window_frame(model=model, test=test, mr=mr, lo=lo, hi=hi, flag=flag, device=device)
    CTX["extra"] = {}
    if not CTX["on"] and frame is not None:
        for name, xv in CTX["variants"].items():
            CTX["extra"][name] = _window_frame(
                model=model, test=dataclasses.replace(test, x=xv), mr=mr, lo=lo, hi=hi, flag=flag, device=device
            )  # fmt: skip
    return frame


def atomic_frame(*, frame, target):
    """Write the frame, and its removed/aged twins beside it."""
    _atomic_frame(frame=frame, target=target)
    if CTX.get("extra") and target.parent.name == SPEC.tag:
        for name, fr in CTX["extra"].items():
            _atomic_frame(
                frame=fr, target=target.parent.parent / f"{SPEC.tag}-{name}" / target.name
            )


f.window_frame, f.atomic_frame = window_frame, atomic_frame


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
    if SPEC.lags != "none":
        CTX["stats"] = lg.lag_stats(rows=mr.base, clean=clean, cap_steps=SPEC.cap_steps)
        CTX["clean"], CTX["month"] = clean, month
        origin_idx = (mr.origin_us - c.prep422.GRID_START_US) // c.prep422.STEP_US
        if month == ARGS.months[0] and SPEC.cap_steps == lg.CAP_24H:
            lg.check_matches_560(
                clean=clean, primary=mr.primary, target_idx=mr.target_idx, origin_idx=origin_idx,
                lead=mr.lead, stats=CTX["stats"],
            )  # fmt: skip
        blk = block_of(
            clean=clean, primary=mr.primary, target_idx=mr.target_idx, origin_idx=origin_idx
        )
        mr.x = widen(x37=mr.x, blk=blk)
        attach_tables(standardiser=mr.base.standardiser)
        ex = lg.check_lag_ages(
            x=mr.x, origin_us=mr.origin_us, primary=mr.primary, origin_tab=CTX["origin_ft"],
            cap_hours=CAP_HOURS, label=f"{month} train",
        )  # fmt: skip
        CHECK["months"][month]["lag_example_train"] = ex
        print(ex, flush=True)
    if SPEC.columns:
        global WT  # noqa: PLW0603
        global KNOTS  # noqa: PLW0603
        WT = extra.DriftColumns(month=month, mr=mr)
        KNOTS = WT.n_knots
        mr.x = np.concatenate([mr.x, WT.month(null=False)], axis=1)
        CHECK["months"][month]["knot_example"] = WT.example
        print(month, "knot-position column appended |", WT.example, flush=True)
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
    tau = None
    if SPEC.columns:
        tau = WT.rows(  # ty: ignore[unresolved-attribute]
            kind="ens", primary=test.primary, origin_idx=test.origin_idx, valid_idx=test.target_idx, null=False,
        )  # fmt: skip
    if SPEC.lags == "none":
        return (
            dataclasses.replace(test, x=np.concatenate([test.x, tau], axis=1))
            if tau is not None
            else test
        )
    clean = CTX["clean"]
    variants = {}
    for extra_days in (0, *AGED_DAYS):
        blk = block_of(
            clean=clean, primary=test.primary, target_idx=test.target_idx, origin_idx=test.origin_idx,
            extra_days=extra_days,
        )  # fmt: skip
        variants[extra_days] = widen(x37=test.x, blk=blk)
    ex = lg.check_lag_ages(
        x=variants[0], origin_us=test.origin, primary=test.primary, origin_tab=CTX["origin_ft"],
        cap_hours=CAP_HOURS, label=f"{month} test",
    )  # fmt: skip
    for e in AGED_DAYS:
        diff = variants[0][:, 43] - variants[e][:, 43]
        assert (diff >= 48 * e).all() and (diff % 48 == 0).all(), (
            "an aged day lag is not a multiple of 24 h older"
        )
    gone = variants[0].copy()
    lg.absent(x=gone, cap_steps=SPEC.cap_steps)
    sets = {"removed": gone, **{f"aged{e}": variants[e] for e in AGED_DAYS}}
    if tau is not None:
        variants = {e: np.concatenate([v, tau], axis=1) for e, v in variants.items()}
        sets = {n: np.concatenate([v, tau], axis=1) for n, v in sets.items()}
    CTX["variants"] = sets
    i = int(np.argmax(variants[0][:, 38] < 0.5))
    CHECK["months"][month]["lag_example"] = (
        f"row {i}: origin {datetime.fromtimestamp(test.origin[i] / 1e6, UTC)}, target {datetime.fromtimestamp(test.valid[i] / 1e6, UTC)}, "
        f"day lag at {datetime.fromtimestamp((c.prep422.GRID_START_US + int(variants[0][i, 43]) * c.prep422.STEP_US) / 1e6, UTC)} (>= {CAP_HOURS} h before origin asserted for all sampled rows)"
    )
    CHECK["months"][month]["lag_example_test"] = ex
    print(
        month,
        "test lag checks passed |",
        CHECK["months"][month]["lag_example"],
        "|",
        ex,
        flush=True,
    )
    return dataclasses.replace(test, x=variants[0])


k.build_month, data.build_test = build_month, build_test


def pretrain_wt(*, device: str) -> None:
    """Pre-train the arm's network on each fold's ERA5 and CAMS rows (arms V3D5 and V35)."""
    assert SPEC.tag in {"V3D5", "V35"}, "arms V3 and E2ft use stored pre-training"
    clean = c.clean_stack()
    for month in ARGS.months:
        todo = [s for s in ARGS.seeds if not (pretrained_dir() / f"{month}_s{s}.pt").exists()]
        if not todo:
            continue
        mr = k.build_month(month=month, clean=clean)  # sets CTX["stats"] and the ENS-side lag table
        refit_t = c.month_start(month=month)
        pre = r448.pretrain_rows(rows=mr.base, refit=refit_t, clean=clean)
        assert pre.x.shape[1] == 37
        latest = datetime.fromtimestamp(
            (pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC
        )
        assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
        print(
            f"{month}: pretraining rows {len(pre.x):,}; latest valid {latest.isoformat()} (refit {refit_t.isoformat()})",
            flush=True,
        )
        blk = block_of(
            clean=clean, primary=pre.primary, target_idx=pre.target_idx, origin_idx=pre.origin_idx
        )
        cols, origin = d560.raw_table(source="pre", cache=W560 / "expected")
        table_pre = torch.from_numpy(
            d560.standardised_table(cols=cols, standardiser=mr.base.standardiser)
        ).to(device)
        del cols
        CTX["table_pre"] = table_pre
        n_in = N_LAG_INPUTS + (1 if SPEC.columns else 0)
        x_gpu = torch.empty((len(pre.x), n_in), dtype=torch.float32, device=device)
        for a in range(0, len(pre.x), 1_000_000):
            x_gpu[a : a + 1_000_000, :37] = torch.from_numpy(pre.x[a : a + 1_000_000]).to(device)
        if SPEC.columns:
            tau = WT.rows(  # ty: ignore[unresolved-attribute]
                kind="era5", primary=pre.primary, origin_idx=pre.origin_idx, valid_idx=pre.target_idx, null=False,
            )  # fmt: skip
            x_gpu[:, N_LAG_INPUTS] = torch.from_numpy(tau[:, 0]).to(device)
        stub = np.zeros((min(2_000_000, len(blk)), N_LAG_INPUTS), np.float32)
        stub[:, 37:] = blk[: len(stub)]
        ex = lg.check_lag_ages(
            x=stub, origin_us=c.prep422.GRID_START_US + pre.origin_idx[: len(stub)] * c.prep422.STEP_US,
            primary=pre.primary[: len(stub)], origin_tab=np.asarray(origin), cap_hours=CAP_HOURS, label=f"{month} pretrain",
        )  # fmt: skip
        del stub
        print(ex, flush=True)
        CHECK["months"][month]["lag_example_pretrain"] = ex
        z = (pre.y - mr.base.target_mean[pre.primary]) / mr.base.target_std[pre.primary]
        t = {
            "z": torch.from_numpy(z.astype(np.float32)).to(device), "w": torch.ones(len(pre.y), device=device),
            "p": torch.from_numpy(pre.primary).to(device), "v": torch.from_numpy(pre.is_val).to(device),
        }  # fmt: skip
        n_primaries = len(mr.base.target_mean)
        origin_idx = pre.origin_idx
        del pre, mr
        for seed in todo:
            started = time.monotonic()
            for a in range(0, len(blk), 1_000_000):
                x_gpu[a : a + 1_000_000, 37:N_LAG_INPUTS] = torch.from_numpy(
                    blk[a : a + 1_000_000]
                ).to(device)
            share = apply_dropout(x=x_gpu, origin_idx=origin_idx, seed=seed)
            torch.manual_seed(seed)
            model = new_model(n_primaries=n_primaries, stage="pre").to(device)
            epochs, best = train(
                model=model, x=x_gpu, z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                lr=c.PRETRAIN_LR, max_epochs=c.MAX_EPOCHS, fresh_lr=None, frozen=False,
                anchor=None, lam=0.0, device=device,
            )  # fmt: skip
            target = pretrained_dir() / f"{month}_s{seed}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  pretrain {SPEC.tag} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, dropout {share}",
                flush=True,
            )
        del t, x_gpu, blk
        CTX.pop("table_pre", None)
        torch.cuda.empty_cache()


def model_file(*, tag: str, month: str, day: int, seed: int, sfx: str = "") -> Path:
    """Return the saved weights of a daily fit."""
    return W / "models" / tag / f"{month}_d{day}_s{seed}{sfx}.pt"


def fine_tune(
    *, g: k.GpuMonth, month: str, cut: datetime, seed: int, device: str,
    shift: np.ndarray | None = None, sign: int = 0, block: tuple[int, int] | None = None,
) -> tuple[torch.nn.Module, dict]:  # fmt: skip
    """Return a fine-tuned model and its log for one cutoff of the arm in `SPEC`."""
    mr = g.mr
    started = time.monotonic()
    CTX.update(on=shift is not None, shift=shift, sign=sign, block=block, seed=seed, cut=cut)
    refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=SPEC.half_life)
    assert refit.report["rows_past_valid_cut"] == 0 and refit.report["rows_past_origin_cut"] == 0
    shift_z, injected = None, 0
    if shift is not None and block is not None:
        inj = (
            (mr.valid_us[refit.index] >= block[0])
            & (mr.valid_us[refit.index] < block[1])
            & ~refit.is_val
        )
        prim = mr.primary[refit.index]
        shift_z = np.where(inj, sign * shift[prim] / mr.base.target_std[prim], 0.0)
        injected = int(inj.sum())
    t = g.refit_tensors(refit=refit, shift_z=shift_z)
    feeder = time.monotonic() - started
    model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
    anchor = None
    load_pretrained_fresh(model=model, path=pretrained_file(month=month, seed=seed))
    model.to(device)
    t0 = time.monotonic()
    epochs, best = train(
        model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
        lr=SPEC.lr, max_epochs=SPEC.max_epochs, fresh_lr=FRESH_LR,
        frozen=SPEC.frozen, anchor=anchor, lam=0.0, device=device,
    )  # fmt: skip
    day = f.day_of(month=month, cut=cut)
    log = {
        "name": f"{SPEC.tag}/{month}_d{day}_s{seed}", "epochs": epochs, "val_nll": best,
        "feeder_seconds": round(feeder, 2), "train_seconds": round(time.monotonic() - t0, 2),
        "injected_rows": injected, **refit.report,
    }  # fmt: skip
    log.update(CTX.get("dropout", {}) if SPEC.lags != "none" else {})
    if CTX["on"] and SPEC.lags != "none":
        log["rows_with_shifted_lag"] = CTX.get("n_shifted")
    del t
    return model, log


def fit_arm(
    *, months: list[str], seeds: list[int], stress: bool, limit: int | None, device: str
) -> None:
    """Fine-tune and forecast the arm's own scored window at each cutoff (or its stress cutoffs)."""
    assert SPEC.average_of is None, "use the average command for arm A5"
    clean = c.clean_stack()
    for month in months:
        mr = k.build_month(month=month, clean=clean)
        test = data.build_test(month=month, standardiser=mr.base.standardiser)
        g = k.GpuMonth(mr=mr, device=device)
        cuts = k.cutoffs(month=month, cadence=CADENCE)
        start_day = c.month_start(month=month)
        if stress:
            end = start_day + timedelta(days=f.STRESS_END_DAY)
            first = start_day + timedelta(days=f.STRESS_END_DAY)
            todo = [i for i, x in enumerate(cuts) if first <= x < end + timedelta(days=7)]
            shift = c.shift_sizes(clean=clean, cutoff=end)
            block = (
                c.us(t=end - timedelta(days=f.STRESS_BLOCK_DAYS)),
                c.us(t=end) - c.CUTOFF_AGE_US,
            )
        else:
            todo = list(range(len(cuts)))[:limit]
            shift, block = None, None
        for seed in seeds:
            for i in todo:
                cut = cuts[i]
                day = f.day_of(month=month, cut=cut)
                for sign in (+1, -1) if stress else (0,):
                    sfx = f"_stress{'p' if sign > 0 else 'n'}" if stress else ""
                    out = W / "fits" / SPEC.tag / f"{month}_d{day}_s{seed}{sfx}.parquet"
                    mfile = model_file(tag=SPEC.tag, month=month, day=day, seed=seed, sfx=sfx)
                    if (out.exists() or out.with_suffix(".empty").exists()) and mfile.exists():
                        continue
                    started = time.monotonic()
                    if mfile.exists():
                        model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
                        model.load_state_dict(torch.load(mfile, map_location=device))
                        model.eval()
                        log: dict = {
                            "name": f"{SPEC.tag}/{month}_d{day}_s{seed}",
                            "reused": str(mfile),
                        }
                    else:
                        model, log = fine_tune(
                            g=g, month=month, cut=cut, seed=seed, device=device,
                            shift=shift, sign=sign, block=block,
                        )  # fmt: skip
                        mfile.parent.mkdir(parents=True, exist_ok=True)
                        tmp_model = mfile.with_suffix(".tmp")
                        torch.save(model.state_dict(), tmp_model)
                        tmp_model.rename(mfile)
                    lo, hi = k.window_of(month=month, cadence=CADENCE, index=i)
                    frame = f.window_frame(
                        model=model, test=test, mr=mr, lo=lo, hi=hi, flag=False, device=device
                    )
                    if frame is None:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.with_suffix(".empty").touch()
                    else:
                        f.atomic_frame(frame=frame, target=out)
                    log["seconds"] = round(time.monotonic() - started, 2)
                    c.write_json(
                        obj=log, path=W / "logs" / SPEC.tag / f"{month}_d{day}_s{seed}{sfx}.json"
                    )
                    print(
                        f"{log['name']}{sfx}: {log.get('epochs', 'reused')} epochs, {log['seconds']} s (feeder {log.get('feeder_seconds', 0)} s)",
                        flush=True,
                    )
                    del model
        del g, mr, test
        torch.cuda.empty_cache()


def assemble() -> None:
    """Concatenate each cutoff's window forecast into one table per month, seed, and scoring variant."""
    tags = (
        (SPEC.tag, *[f"{SPEC.tag}-{s}" for s in SIBLINGS]) if SPEC.lags != "none" else (SPEC.tag,)
    )
    for tag in tags:
        for month in ARGS.months:
            cuts = k.cutoffs(month=month, cadence=CADENCE)
            for seed in ARGS.seeds:
                parts = [
                    W / "fits" / tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                    for cut in cuts
                ]
                twin = lambda p: (p.parent.parent / SPEC.tag / p.name).with_suffix(".empty")  # noqa: E731
                assert all(
                    p.exists() or p.with_suffix(".empty").exists() or twin(p).exists()
                    for p in parts
                ), (tag, month, seed)
                frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
                f.atomic_frame(
                    frame=frame, target=W / "seed_preds" / tag / f"{month}_s{seed}.parquet"
                )
                print("wrote", tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble()
    elif ARGS.command == "pretrain-wt":
        pretrain_wt(device="cuda")
    else:
        fit_arm(
            months=ARGS.months,
            seeds=ARGS.seeds,
            stress=ARGS.stress,
            limit=ARGS.limit,
            device="cuda",
        )
    if ARGS.command == "fit":
        out = (
            WORK
            / f"checks_fit_{SPEC.tag}{'_stress' if ARGS.stress else ''}_s{'-'.join(map(str, ARGS.seeds))}.json"
        )
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in CHECK.items() if a != "months"})

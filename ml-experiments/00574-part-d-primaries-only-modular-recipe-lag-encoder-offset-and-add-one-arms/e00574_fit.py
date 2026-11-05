# ruff: noqa: E501, ANN001, ANN201, PT018, D102, RUF028, PLR0915
"""Experiment 574: coarse-refit fits of the Part D add-one arms on experiment 572's v3+D5 base (GPU).

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
import e00515_placebo as placebo  # ty: ignore[unresolved-import]
import e00558_lags as lags  # ty: ignore[unresolved-import]
import e00559_extra as extra  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00559_wt as wt559  # ty: ignore[unresolved-import]
import e00560_data as d560  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00562_columns as c562  # ty: ignore[unresolved-import]
import e00568_spread as spread_lib  # ty: ignore[unresolved-import]
import e00569_columns as cols569  # ty: ignore[unresolved-import]
import e00572_lags as lg  # ty: ignore[unresolved-import]
import e00572_model as m572  # ty: ignore[unresolved-import]
import e00574_model as mod
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
W = WORK / "work"
W559 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty/work"
)
"""Experiment 559's work folder; only its `pretrained_drift/real` weights (arm A10) are read."""
W560 = ML_EXPERIMENTS_DATA_DIR / "00560-modular-lag-encoder-capped-lags-and-block-dropout/work"
"""Experiment 560's work folder; only its `pretrained/LAG-v3` weights (arm V3) are read."""
W572 = ML_EXPERIMENTS_DATA_DIR / "00572-two-offset-part-1-lags-versus-the-slow-output-offset/work"
"""Experiment 572's work folder; its `pretrained_V3D5` weights are the pre-training of every arm that adds no pre-training input (base P0)."""
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
EMBARGO_DAYS = 3
"""Days before the cutoff whose targets arm P4E leaves out of step 2 and of the daily fine-tune (experiment 556's embargo)."""
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
    spread: bool = False  # arm P1: experiment 568's seven spread columns in the fine-tuning mixer
    sched: str = "1s"  # "1s" (experiment 559's one-step schedule), "3s" (three-step), or "3sE" (three-step plus a 3-day embargo)
    wt: bool = False  # arm P6: experiment 515's three trailing-temperature columns (W-T) beside the B1 columns
    b1: str = ""  # arm P2 ("B1") or its placebo P2-plac ("B1p"): experiment 569's thermal-lag and solar-gain columns
    pretrain_from: str = "V3D5"  # which pre-training folder (under this experiment, or 572's V3D5) the arm starts from

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
        Spec("P1", "24h", "lagoff", True, spread=True),
        Spec("P3E0", "24h", "lagoff", False),
        Spec("P5", "24h", "lagoff", True, sched="3sE", b1="B1", pretrain_from="P2"),
        Spec("P0", "24h", "lagoff", True),
        Spec("P6", "24h", "lagoff", True, sched="3sE", b1="B1", wt=True, pretrain_from="P6"),
        Spec("P4", "24h", "lagoff", True, sched="3s"),
        Spec("P4E", "24h", "lagoff", True, sched="3sE"),
        Spec("P2", "24h", "lagoff", True, b1="B1", pretrain_from="P2"),
        Spec("P2plac", "24h", "lagoff", True, b1="B1p", pretrain_from="P2plac"),
    )
}
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("fit", "assemble", "pretrain-wt", "step2", "events"))
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
LEAD_BANDS = ((0, 24), (24, 120), (120, 240))
"""Lead bands (hours) of the scored output, one printed timing example each."""
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
    return W / f"pretrained_{SPEC.pretrain_from}"


def pretrained_file(*, month: str, seed: int) -> Path:
    """Return the pre-trained weights: experiment 572's v3+D5, or this experiment's own for arms that add pre-training inputs."""
    own = SPEC.pretrain_from != "V3D5"
    p = (pretrained_dir() if own else W572 / "pretrained_V3D5") / f"{month}_s{seed}.pt"
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
        m572.load_lag_pre_into_ft(model=model, state=raw)  # ty: ignore[invalid-argument-type]
    else:
        state = {a: v for a, v in raw.items() if not a.startswith("mixer_ft.")}
        missing = model.load_state_dict(state, strict=False)
        allowed = lambda a: a.startswith("mixer_ft.") or (SPEC.variant == "ftonly" and a == "knots")  # noqa: E731
        assert all(allowed(a) for a in missing.missing_keys), missing
        assert not missing.unexpected_keys, missing
    fresh_mixer(model=model)  # ty: ignore[invalid-argument-type]


def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    """Return an untrained network, with the expected-input table attached for the lag arms."""
    if SPEC.variant == "ftonly":
        assert stage == "ft"
        return m572.OffsetFtOnly(n_primaries=n_primaries, n_knots=KNOTS)
    model: torch.nn.Module
    if SPEC.variant == "lagoff":
        model = mod.PartDMdn(
            n_primaries=n_primaries, stage=stage, n_knots=KNOTS, spread=SPEC.spread, n_b1=N_B1
        )
    else:
        model = m560.LagModularMdn(n_primaries=n_primaries, mode="v3", stage=stage)
    model.table = CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


def train(  # noqa: PLR0912
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
N_B1_COLUMNS = cols569.width(arm=SPEC.b1) if SPEC.b1 else 0
N_WT = 3 if SPEC.wt else 0
N_B1 = N_B1_COLUMNS + N_WT
"""Width of the weather-encoder extras: the B1 columns, then the three W-T columns when the arm has them."""
WTC: wt559.WtColumns | None = None
EX = cols569.Extras(arm=SPEC.b1, proximity={}) if SPEC.b1 else None
FOLDS: dict[str, c562.FoldColumns] = {}
PRETRAIN_PRESENT = "B1 has no presence flag, so every pre-training row carries its columns"
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


BLOCKS: dict[str, spread_lib.SpreadBlock] = {}


PRETRAINING: dict[str, bool] = {"on": False}
"""True while `pretrain_wt` builds a fold's rows: the B1 scale and spread block exist only after pre-training."""


def stats_file(*, month: str) -> Path:
    """Return the fold's mean and standard deviation of the B1 columns (fitted on pre-training rows)."""
    return W / "stats" / SPEC.b1 / f"{month}.npz"


def fit_until(*, month: str) -> int:
    """Return the grid index of the fold's refit time."""
    return int((c.us(t=c.month_start(month=month)) - c.prep422.GRID_START_US) // c.prep422.STEP_US)


def pretrain_b1_columns(*, month: str, pre) -> np.ndarray:
    """Return the standardised B1 columns of the fold's pre-training rows and store the fold's scale.

    Experiment 569's construction: the columns come from the dense ERA5 and CAMS weather at each
    row's own origin and valid time; the placebo takes them from a donor day of the same side of the
    refit, with the real arm's scale.
    """
    assert EX is not None
    primary, oi, ti = pre.primary, pre.origin_idx, pre.target_idx
    real = EX.raw_pretrain(primary=primary, origin_idx=oi, target_idx=ti, shift=None)
    EX.fit(raw=real)
    stats = stats_file(month=month)
    stats.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats, mean=EX.mean, std=EX.std)
    if not EX.placebo:
        CHECK["months"].setdefault(month, {})["pretrain_b1_complete_share"] = float(
            np.isfinite(real).all(axis=1).mean()
        )
        return EX.scale(raw=real)
    days = placebo.DayStructure(fit_until_idx=fit_until(month=month), allowed=None)
    raw = np.full(real.shape, np.nan, dtype=np.float32)
    todo = np.ones(len(primary), dtype=bool)
    for attempt in range(len(placebo.SEEDS)):
        shift = days.shifts(valid_idx=ti, attempt=attempt)
        idx = np.nonzero(todo & np.isfinite(shift))[0]
        if not len(idx):
            continue
        raw[idx] = EX.raw_pretrain(
            primary=primary[idx], origin_idx=oi[idx], target_idx=ti[idx], shift=shift[idx]
        )
        todo[idx] = False
    CHECK["months"].setdefault(month, {})["pretrain_placebo_rows_without_donor"] = int(todo.sum())
    return EX.scale(raw=raw)


def finetune_b1_columns(*, month: str, primary, origin_idx, target_idx) -> np.ndarray:
    """Return the standardised B1 columns of fine-tuning or scoring rows, on the fold's pre-training scale.

    Each row's columns are experiment 437's stored values for the ensemble run keyed by the row's
    origin (the run issued at or before the origin minus 9 h); a row with no stored key raises.
    """
    assert EX is not None
    if month not in FOLDS:
        FOLDS.clear()
        FOLDS[month] = c562.FoldColumns(fit_until_idx=fit_until(month=month))
    fc = FOLDS[month]
    stats = np.load(stats_file(month=month))
    EX.mean, EX.std = stats["mean"], stats["std"]
    getter = fc._raw_placebo if EX.placebo else fc._raw_real
    raw = getter(arm="B1", primary=primary, origin_idx=origin_idx, valid_idx=target_idx)
    return EX.scale(raw=raw)


def origin_idx_of(*, us: np.ndarray) -> np.ndarray:
    """Return the grid index of origin times given in microseconds."""
    return ((us - c.prep422.GRID_START_US) // c.prep422.STEP_US).astype(np.int64)


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
    if SPEC.wt:
        global WTC  # noqa: PLW0603
        WTC = wt559.WtColumns(month=month, mr=mr)
        CHECK["months"][month]["wt_example"] = WTC.example
        print(month, "W-T columns built |", WTC.example, flush=True)
    if SPEC.b1 and not PRETRAINING["on"]:
        add = finetune_b1_columns(
            month=month,
            primary=mr.primary,
            origin_idx=origin_idx_of(us=mr.origin_us),
            target_idx=mr.target_idx,
        )
        if SPEC.wt:
            add = np.concatenate([add, WTC.month(null=False)], axis=1)
        assert add.shape == (len(mr.y), N_B1) and np.isfinite(add).all()
        assert mr.x.shape[1] == mod.b1_cols(n_b1=N_B1).start, mr.x.shape
        mr.x = np.concatenate([mr.x, add], axis=1)
        i = int(np.argmax(mr.valid_us > mr.origin_us + 24 * 3_600_000_000))
        CHECK["months"][month]["b1_example_train"] = (
            f"origin {datetime.fromtimestamp(mr.origin_us[i] / 1e6, UTC)}, weather run {datetime.fromtimestamp(mr.origin_us[i] / 1e6, UTC) - common.PUBLICATION_DELAY}, "
            f"valid {datetime.fromtimestamp(mr.valid_us[i] / 1e6, UTC)}, Primary {int(mr.primary[i])}, B1 {np.round(add[i], 3).tolist()}"
        )
        print(
            month, "B1 columns appended |", CHECK["months"][month]["b1_example_train"], flush=True
        )
    if SPEC.spread and not PRETRAINING["on"]:
        sb = spread_lib.SpreadBlock(month=month, mr=mr, placebo=False)
        block = sb.block(
            x37=mr.x[:, :37], standardiser=mr.base.standardiser, primary=mr.primary,
            origin_idx=origin_idx_of(us=mr.origin_us), valid_idx=mr.target_idx, describe=True,
        )  # fmt: skip
        assert mr.x.shape[1] == mod.spread_cols(n_b1=N_B1).start, mr.x.shape
        mr.x = np.concatenate([mr.x, block], axis=1)
        BLOCKS[month] = sb
        CHECK["months"][month]["spread_example_train"] = sb.example
        print(month, "spread block appended |", sb.example, flush=True)
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
    lead_h = (test.valid - test.origin) / 3.6e9
    for lo_h, hi_h in LEAD_BANDS:
        i = int(np.argmax((lead_h >= lo_h) & (lead_h < hi_h)))
        assert lo_h <= lead_h[i] < hi_h, f"{month}: no test row in lead band {lo_h}-{hi_h} h"
        CHECK["months"][month][f"band_{lo_h}_{hi_h}_example"] = (
            f"lead {lead_h[i]:.1f} h: origin {datetime.fromtimestamp(test.origin[i] / 1e6, UTC)}, "
            f"weather run {datetime.fromtimestamp(test.origin[i] / 1e6, UTC) - pub} (origin minus {pub}), "
            f"valid {datetime.fromtimestamp(test.valid[i] / 1e6, UTC)}"
        )
    print(
        month,
        "lead-band examples |",
        [v for a_, v in CHECK["months"][month].items() if a_.startswith("band_")],
        flush=True,
    )
    tau = None
    if SPEC.columns:
        tau = WT.rows(  # ty: ignore[unresolved-attribute]
            kind="ens", primary=test.primary, origin_idx=test.origin_idx, valid_idx=test.target_idx, null=False,
        )  # fmt: skip
        if SPEC.b1:
            add = finetune_b1_columns(
                month=month,
                primary=test.primary,
                origin_idx=test.origin_idx,
                target_idx=test.target_idx,
            )
            if SPEC.wt:
                add = np.concatenate(
                    [
                        add,
                        WTC.rows(
                            kind="ens",
                            primary=test.primary,
                            origin_idx=test.origin_idx,
                            valid_idx=test.target_idx,
                            null=False,
                        ),
                    ],
                    axis=1,
                )
            assert add.shape == (len(test.x), N_B1) and np.isfinite(add).all()
            tau = np.concatenate([tau, add], axis=1)
            CHECK["months"][month]["b1_example_test"] = (
                f"origin {datetime.fromtimestamp(test.origin[0] / 1e6, UTC)}, weather run {datetime.fromtimestamp(test.origin[0] / 1e6, UTC) - pub}, "
                f"valid {datetime.fromtimestamp(test.valid[0] / 1e6, UTC)}, B1 {np.round(add[0], 3).tolist()}"
            )
            print(month, "test B1 columns |", CHECK["months"][month]["b1_example_test"], flush=True)
        if SPEC.spread:
            sb = BLOCKS[month]
            block = sb.block(
                x37=test.x[:, :37], standardiser=standardiser, primary=test.primary,
                origin_idx=test.origin_idx, valid_idx=test.target_idx, describe=True,
            )  # fmt: skip
            tau = np.concatenate([tau, block], axis=1)
            CHECK["months"][month]["spread_example_test"] = sb.example
            print(month, "test spread block |", sb.example, flush=True)
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
    assert SPEC.pretrain_from != "V3D5", "this arm uses the stored pre-training of experiment 572"
    clean = c.clean_stack()
    for month in ARGS.months:
        todo = [s for s in ARGS.seeds if not (pretrained_dir() / f"{month}_s{s}.pt").exists()]
        if not todo:
            continue
        PRETRAINING["on"] = True
        mr = k.build_month(month=month, clean=clean)  # sets CTX["stats"] and the ENS-side lag table
        PRETRAINING["on"] = False
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
        n_in = N_LAG_INPUTS + (1 if SPEC.columns else 0) + N_B1
        x_gpu = torch.empty((len(pre.x), n_in), dtype=torch.float32, device=device)
        for a in range(0, len(pre.x), 1_000_000):
            x_gpu[a : a + 1_000_000, :37] = torch.from_numpy(pre.x[a : a + 1_000_000]).to(device)
        if SPEC.columns:
            tau = WT.rows(  # ty: ignore[unresolved-attribute]
                kind="era5", primary=pre.primary, origin_idx=pre.origin_idx, valid_idx=pre.target_idx, null=False,
            )  # fmt: skip
            x_gpu[:, N_LAG_INPUTS] = torch.from_numpy(tau[:, 0]).to(device)
        if SPEC.b1:
            extra_b1 = pretrain_b1_columns(month=month, pre=pre)
            if SPEC.wt:
                extra_b1 = np.concatenate(
                    [
                        extra_b1,
                        WTC.rows(
                            kind="era5",
                            primary=pre.primary,
                            origin_idx=pre.origin_idx,
                            valid_idx=pre.target_idx,
                            null=False,
                        ),
                    ],
                    axis=1,
                )
            assert extra_b1.shape == (len(pre.x), N_B1) and np.isfinite(extra_b1).all()
            x_gpu[:, mod.b1_cols(n_b1=N_B1)] = torch.from_numpy(extra_b1).to(device)
            j = len(extra_b1) // 2
            print(
                f"  B1 pre-training example: Primary {int(pre.primary[j])}, origin {datetime.fromtimestamp((pre.origin_idx[j] * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC)}, valid {datetime.fromtimestamp((pre.target_idx[j] * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC)} (ERA5/CAMS stand-in), B1 {np.round(extra_b1[j], 3).tolist()}",
                flush=True,
            )
            del extra_b1
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


def step2_file(*, month: str, seed: int) -> Path:
    """Return the saved step-2 weights (encoders, lag encoder, offset, and a mixer trained without recency weighting)."""
    return W / "step2" / SPEC.tag / f"{month}_s{seed}.pt"


def embargoed(*, mr, refit: c.Refit, cut: datetime) -> c.Refit:
    """Return `refit` without the targets in the newest `EMBARGO_DAYS` days before the cutoff (arm P4E)."""
    keep = mr.valid_us[refit.index] <= c.us(t=cut) - EMBARGO_DAYS * c.DAY_US
    return c.Refit(
        refit.index[keep], refit.y[keep], refit.weight[keep], refit.is_val[keep], refit.report
    )


def model_file(*, tag: str, month: str, day: int, seed: int, sfx: str = "") -> Path:
    """Return the saved weights of a daily fit."""
    return W / "models" / tag / f"{month}_d{day}_s{seed}{sfx}.pt"


def fine_tune(
    *, g: k.GpuMonth, month: str, cut: datetime, seed: int, device: str,
    shift: np.ndarray | None = None, sign: int = 0, block: tuple[int, int] | None = None,
    correction=None,
) -> tuple[torch.nn.Module, dict]:  # fmt: skip
    """Return a fine-tuned model and its log for one cutoff of the arm in `SPEC`."""
    mr = g.mr
    started = time.monotonic()
    CTX.update(on=shift is not None, shift=shift, sign=sign, block=block, seed=seed, cut=cut)
    refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=SPEC.half_life)
    assert refit.report["rows_past_valid_cut"] == 0 and refit.report["rows_past_origin_cut"] == 0
    if SPEC.sched == "3sE":
        refit = embargoed(mr=mr, refit=refit, cut=cut)
    if KNOTS:
        # A knot sits at each month start, and a row after the newest knot reads it (flat extrapolation), so no row's position may exceed the cutoff's own.
        cut_tau = float(WT.tau(valid_us=np.array([c.us(t=cut)]))[0])  # ty: ignore[unresolved-attribute]
        assert WT.tau(valid_us=mr.valid_us[refit.index]).max() <= cut_tau + 1e-6, (
            "a training row lies after the refit cut"
        )  # ty: ignore[unresolved-attribute]
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
    if (
        correction is not None
    ):  # the real-event harness: a per-row target correction (capacity fraction) for the selected rows
        shift_z = correction(refit) / mr.base.target_std[mr.primary[refit.index]]
    t = g.refit_tensors(refit=refit, shift_z=shift_z)
    feeder = time.monotonic() - started
    model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
    anchor = None
    if SPEC.sched == "1s":
        load_pretrained_fresh(model=model, path=pretrained_file(month=month, seed=seed))
        fresh_lr = FRESH_LR
    else:
        model.load_state_dict(torch.load(step2_file(month=month, seed=seed), map_location="cpu"))
        fresh_lr = None  # fine-tuning from the step-2 weights keeps the mixer at the ordinary rate (experiment 559's A1)
    model.to(device)
    t0 = time.monotonic()
    epochs, best = train(
        model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
        lr=SPEC.lr, max_epochs=SPEC.max_epochs, fresh_lr=fresh_lr,
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


def step2(*, months: list[str], seeds: list[int], device: str) -> None:
    """Train and save step 2 of the three-step schedule at each fold's first cutoff (arms P4 and P4E).

    Experiment 559's recipe on the lag network of experiment 566: the pre-trained encoders, lag
    encoder, and offset knots, a fresh mixer, no recency weighting (every weight is 1), early
    stopping on the usual weeks, and rows with targets up to the cutoff minus 3.5 hours. P4E drops
    the newest 3 days of targets as well.
    """
    assert SPEC.sched in {"3s", "3sE"}
    clean = c.clean_stack()
    for month in months:
        todo = [s for s in seeds if not step2_file(month=month, seed=s).exists()]
        if not todo:
            continue
        mr = k.build_month(month=month, clean=clean)
        g = k.GpuMonth(mr=mr, device=device)
        cut = c.month_start(month=month)
        refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=UNIFORM_HALF_LIFE)
        assert (
            refit.report["rows_past_valid_cut"] == 0 and refit.report["rows_past_origin_cut"] == 0
        )
        if SPEC.sched == "3sE":
            refit = embargoed(mr=mr, refit=refit, cut=cut)
        for seed in todo:
            started = time.monotonic()
            CTX.update(on=False, shift=None, sign=0, block=None, seed=seed, cut=cut)
            t = g.refit_tensors(refit=refit)
            assert float(t["w"].min()) > 0.999, "step 2 must have no recency weighting"
            model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
            load_pretrained_fresh(model=model, path=pretrained_file(month=month, seed=seed))
            model.to(device)
            epochs, best = train(
                model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                lr=SPEC.lr, max_epochs=SPEC.max_epochs, fresh_lr=FRESH_LR, frozen=SPEC.frozen,
                anchor=None, lam=0.0, device=device,
            )  # fmt: skip
            target = step2_file(month=month, seed=seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
            latest = datetime.fromtimestamp(mr.valid_us[refit.index].max() / 1e6, UTC)
            assert latest <= cut - c.regimes.REFIT_CUTOFF_AGE, (latest, cut)
            log = {
                **refit.report,
                "epochs": epochs,
                "val_nll": best,
                "rows": len(refit.index),
                "latest_valid": str(latest),
            }
            c.write_json(obj=log, path=W / "logs" / "step2" / SPEC.tag / f"{month}_s{seed}.json")
            print(
                f"step 2 ({SPEC.tag}) {month} s{seed}: {len(refit.index):,} rows, {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s; latest valid {latest} (cutoff {cut})",
                flush=True,
            )
            del t, model
        del g, mr
        torch.cuda.empty_cache()


NRA_SHARES = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails/work/nra_correction_share.npz"
)
"""Experiment 435's NRA correction (share of capacity, 16 x grid) at depth factor 1.0."""
MIN_AGE_DAYS = 1
WINDOW_DAYS = 7


def stored_model_file(*, month: str, day: int, seed: int) -> Path:
    """Return the arm's fine-tuned model of a coarse cutoff (arm P0 is experiment 572's v3+D5)."""
    if SPEC.tag == "P0":
        return W572 / "models" / "V3D5" / f"{month}_d{day}_s{seed}.pt"
    return model_file(tag=SPEC.tag, month=month, day=day, seed=seed)


def median_of(*, model, x: np.ndarray, primary: np.ndarray, mr, device: str) -> np.ndarray:
    """Return the model's median forecast (capacity fraction) of test rows."""
    mu = torch.from_numpy(mr.base.target_mean).to(device)
    sd = torch.from_numpy(mr.base.target_std).to(device)
    out = []
    for a in range(0, len(x), 262_144):
        xb = torch.from_numpy(x[a : a + 262_144]).to(device)
        pb = torch.from_numpy(primary[a : a + 262_144]).to(device)
        q = torch.sort(c.serving_quantiles(model=model, x=xb, p=pb, outlier=False), dim=1).values[
            :, 5
        ]
        out.append((q * sd[pb] + mu[pb]).float().cpu().numpy())
    return np.concatenate(out)


def events_command(*, months: list[str], seed: int, limit: int | None, device: str) -> None:
    """Experiment 556's real-event harness on this arm: refit with one labelled event's targets corrected, per event.

    The cutoff is the first coarse cutoff at least `MIN_AGE_DAYS` days after the event's end (the
    event ends before the cut minus 3.5 h, asserted). The correction is experiment 435's labelled
    step (depth factor 1.0), subtracted from the training targets of that one event only; the lag
    inputs stay as observed. The absorbed share is the event's sign times (metered median minus
    corrected median) over the 7 days after the cutoff at the event's Primary, divided by the
    labelled depth. Step 2 is not refitted per event (`in_step2` marks events that end before the
    fold's first cutoff, which sit in step 2 of both fits).
    """
    events = c.labelled_events()
    shares = np.load(NRA_SHARES)["f1.0"]
    grid, step = c.prep422.GRID_START_US, c.prep422.STEP_US
    out = W / "scores" / f"real_events_{SPEC.tag}.parquet"
    rows_out: list[dict] = []
    clean = c.clean_stack()
    count = 0
    for month in months:
        mr = k.build_month(month=month, clean=clean)
        test = data.build_test(month=month, standardiser=mr.base.standardiser)
        g = k.GpuMonth(mr=mr, device=device)
        cuts = k.cutoffs(month=month, cadence=CADENCE)
        first_cut = c.month_start(month=month)
        wanted: dict[datetime, list[tuple[int, int, int]]] = {}
        for p, spans in events.items():
            for onset, end in spans:
                end_us = grid + end * step
                if float(np.abs(shares[p, onset : end + 1]).max()) <= 0:
                    continue
                ok = [x for x in cuts if c.us(t=x) >= end_us + MIN_AGE_DAYS * c.DAY_US]
                if ok and c.us(t=ok[0]) - end_us <= 30 * c.DAY_US:
                    wanted.setdefault(ok[0], []).append((p, onset, end))
        print(
            f"{month}: {sum(len(v) for v in wanted.values())} events at {len(wanted)} cutoffs",
            flush=True,
        )
        for cut, evs in sorted(wanted.items()):
            day = f.day_of(month=month, cut=cut)
            win = np.nonzero(
                (test.origin >= c.us(t=cut)) & (test.origin < c.us(t=cut) + WINDOW_DAYS * c.DAY_US)
            )[0]
            if len(win) == 0:
                continue
            model_m = new_model(n_primaries=len(mr.base.target_mean)).to(device)
            model_m.load_state_dict(
                torch.load(stored_model_file(month=month, day=day, seed=seed), map_location=device)
            )
            model_m.eval()
            model_m.table = CTX["table_ft"]  # ty: ignore[invalid-assignment]
            q_m = median_of(
                model=model_m, x=test.x[win], primary=test.primary[win], mr=mr, device=device
            ).astype(np.float64)
            del model_m
            prim_rows = test.primary[win]
            for p, onset, end in evs:
                end_us = grid + end * step
                assert end_us <= c.us(t=cut) - c.CUTOFF_AGE_US, (
                    "an event ends after the refit cut minus 3.5 h"
                )

                def correction(refit, p=p, onset=onset, end=end, mr=mr) -> np.ndarray:
                    prim, tgt = mr.primary[refit.index], mr.target_idx[refit.index]
                    corr = -shares[prim, np.clip(tgt, 0, shares.shape[1] - 1)]
                    return np.where((prim == p) & (tgt >= onset) & (tgt <= end), corr, 0.0).astype(
                        np.float32
                    )

                model_c, _ = fine_tune(
                    g=g, month=month, cut=cut, seed=seed, device=device, correction=correction
                )
                model_c.table = CTX["table_ft"]  # ty: ignore[invalid-assignment]
                diff = q_m - median_of(
                    model=model_c, x=test.x[win], primary=prim_rows, mr=mr, device=device
                ).astype(np.float64)
                sign = 1 if float(shares[p, onset : end + 1].sum()) > 0 else -1
                depth = float(np.abs(shares[p, onset : end + 1]).max())
                sel = prim_rows == p
                rows_out.append({
                    "arm": SPEC.tag, "month": month, "seed": seed, "cut": str(cut.date()), "primary": p,
                    "onset": onset, "end": end, "depth_points": 100 * depth,
                    "age_days": (c.us(t=cut) - end_us) / c.DAY_US, "in_step2": bool(end_us <= c.us(t=first_cut) - c.CUTOFF_AGE_US),
                    "absorbed_share": float(sign * diff[sel].mean() / depth) if sel.any() else None,
                })  # fmt: skip
                del model_c
                count += 1
                print(
                    "event",
                    count,
                    month,
                    "Primary",
                    p,
                    "share",
                    rows_out[-1]["absorbed_share"],
                    flush=True,
                )
                if limit and count >= limit:
                    break
            if limit and count >= limit:
                break
        del g, mr, test
        torch.cuda.empty_cache()
        if limit and count >= limit:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows_out, schema_overrides={"absorbed_share": pl.Float64}).write_parquet(out)
    print("real events", len(rows_out), flush=True)


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
    elif ARGS.command == "events":
        events_command(months=ARGS.months, seed=ARGS.seeds[0], limit=ARGS.limit, device="cuda")
    elif ARGS.command == "step2":
        step2(months=ARGS.months, seeds=ARGS.seeds, device="cuda")
    elif ARGS.command == "pretrain-wt":
        pretrain_wt(device="cuda")
    else:
        if (
            ARGS.stress
            and SPEC.lags != "none"
            and list((W / "models" / SPEC.tag).glob("*_stress*.pt"))
        ):
            raise SystemExit(
                f"stress models of {SPEC.tag} exist; a resumed stress run would score reused models without the shift. Delete them and their fits first."
            )
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

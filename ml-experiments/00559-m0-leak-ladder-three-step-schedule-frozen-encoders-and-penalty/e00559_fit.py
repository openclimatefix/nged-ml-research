# ruff: noqa: E501, ANN001, ANN201, PT018
"""Experiment 559: the leak ladder for the modular lag-free demand network M0.

Rows, cutoffs, recency weights, early stopping, seeds, scoring windows, and the daily harness are
experiment 511's (daily refit, 90-day half-life). The network is experiment 554's `ModularMdn`. The
origin-time weather assertions are experiment 554's, copied.

Training schedule of arms A1 to A5 and A8 (maintainer's three steps):

1. ERA5/CAMS pretraining (experiment 554's saved `pretrained/M0` weights, read only).
2. `step2`: a fresh mixer plus the encoders trained on every ensemble row up to the fold's first
   cutoff minus 3.5 h, uniform weights (no recency weighting). The weights are saved.
3. Daily fine-tune from the step-2 weights, 90-day half-life.

Arms (see `ARMS`): A1 plain step 3; A2 encoders frozen; A3 L2 penalty to the step-2 weights; A4 a lower
rate and at most 5 epochs; A5 the average of the last 5 daily fine-tunes of A1 (inference only); A6 and
A7 are experiment 554's M0-hl180 and M0-c3 (fresh mixer at every daily fit, no step 2).

Usage: python e00559_fit.py {step2|choose-lambda|fit|average|assemble} --arm ARM [--stress] [--months ...] [--seeds ...]
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
import e00559_extra as extra
import e00559_folded as folded
import e00559_model as m
import e00559_model_drift as mdr
import e00559_model_extra as mex
import e00559_model_wt as mwt
import e00559_wt as wt
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
W = WORK / "work"
W554 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out/work"
)
"""Experiment 554's work folder; only its `pretrained/M0` weights are read."""
FRESH_SEED = 12345
"""Initialisation seed of every fresh mixer, as in experiments 545 and 554."""
FRESH_LR = 5e-4
"""Learning rate of the fresh mixer (M0's, and step 2's)."""
CADENCE_3_OFFSETS = (0, *range(2, 31, 3))
"""Refit days of arm A7: day 0, then every third day from day 2, so that days 14, 17, and 20 are refits."""
UNIFORM_HALF_LIFE = 1e12
"""A half-life so long that every weight is 1 (step 2 has no recency weighting)."""
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0)
LAMBDA_TOLERANCE = 0.002
LAMBDA_FOLD = "2025-10"
A4_LR = 5e-5
A4_MAX_EPOCHS = 5
AVERAGE_OVER = 5


@dataclass(frozen=True)
class Spec:
    """How an arm's daily fits start and train."""

    tag: str
    start: str  # "step2" or "fresh" (pretrained weights, fresh mixer)
    cadence: str = "d"
    half_life: float = 90.0
    frozen: bool = False
    penalty: bool = False
    lr: float = c.FINE_TUNE_LR
    max_epochs: int = c.MAX_EPOCHS
    average_of: str | None = None  # the arm whose daily weights are averaged (A5)
    source: str = "wt"  # which extra columns: "wt" (experiment 515's W-T) or "uv" (wind components)
    columns: str | None = (
        None  # "real" or "null": add experiment 515's W-T columns to the weather encoder (A9)
    )
    head: str = "mixture"  # "folded": folded-normal head for the MVA-only Primaries (A8)


SPECS = {
    s.tag: s
    for s in (
        Spec("A1", "step2"),
        Spec("A2", "step2", frozen=True),
        Spec("A3", "step2", penalty=True),
        Spec("A4", "step2", lr=A4_LR, max_epochs=A4_MAX_EPOCHS),
        Spec("A5", "step2", average_of="A1"),
        Spec("A6", "fresh", half_life=180.0),
        Spec("A7", "fresh", cadence="c3"),
        Spec("A8", "step2", head="folded"),
        Spec("A9", "fresh", columns="real"),
        Spec("A9-null", "fresh", columns="null"),
        Spec("A10", "fresh", frozen=True, columns="real", source="drift"),
        Spec("A11", "fresh", columns="real", source="uv"),
        Spec("A11-null", "fresh", columns="null", source="uv"),
    )
}
parser = argparse.ArgumentParser()
parser.add_argument(
    "command", choices=("step2", "choose-lambda", "fit", "average", "assemble", "pretrain-wt")
)
parser.add_argument("--arm", choices=tuple(SPECS), default="A1")
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
    """Experiment 511's cutoffs; arm A7 (which passes cadence `d` so the windows tile) refits on days 0, 2, 5, 8, ..."""
    if SPEC.cadence != "c3":
        return _cutoffs(month=month, cadence=cadence)
    start, stop = c.month_start(month=month), c.month_stop(month=month)
    days = (stop - start).days
    return [start + timedelta(days=d) for d in CADENCE_3_OFFSETS if d < days]


k.cutoffs = cutoffs
CADENCE = "d"


def pretrained_file(*, month: str, seed: int) -> Path:
    """Return experiment 554's pretrained M0 weights (read only), or this experiment's own for the W-T arms."""
    if SPEC.columns:
        own = W / f"pretrained_{SPEC.source}" / SPEC.columns / f"{month}_s{seed}.pt"
        assert own.exists(), own
        return own
    p = W554 / "pretrained" / "M0" / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def step2_file(*, month: str, seed: int) -> Path:
    """Return the saved step-2 weights (arm A8 has its own, trained with the folded head's loss)."""
    return W / ("step2_a8" if SPEC.head == "folded" else "step2") / f"{month}_s{seed}.pt"


FOLDED: dict[str, torch.Tensor] | None = None
"""Per-Primary `offset` (target mean over target std) and `mask` (MVA-only eight) for arm A8, set by `set_folded`."""
MVA_ONLY_GROUP = "MVA-only eight"


def set_folded(*, mr: c.MonthRows, device: str) -> None:
    """Set the folded head's per-Primary tables from the month's target statistics (arm A8 only)."""
    global FOLDED  # noqa: PLW0603
    if SPEC.head != "folded":
        return
    sites = pl.read_csv(
        ML_EXPERIMENTS_DATA_DIR
        / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
    )
    group = dict(zip(sites["primary"].to_list(), sites["group"].to_list(), strict=True))
    n = len(mr.base.target_mean)
    mask = np.array([group[i] == MVA_ONLY_GROUP for i in range(n)])
    offset = (mr.base.target_mean / mr.base.target_std).astype(np.float32)
    FOLDED = {
        "offset": torch.from_numpy(offset).to(device),
        "mask": torch.from_numpy(mask).to(device),
    }
    print("folded head on", int(mask.sum()), "of", n, "Primaries", flush=True)


def row_nll(*, out: torch.Tensor, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Return each row's negative log-likelihood (folded for MVA-only Primaries under arm A8)."""
    if FOLDED is None:
        return base.mixture_nll(out=out, z=z, components=m.COMPONENTS)
    return folded.folded_nll(
        out=out, z=z, offset=FOLDED["offset"][p], folded=FOLDED["mask"][p], components=m.COMPONENTS
    )


def serving_quantiles(
    *, model: torch.nn.Module, x: torch.Tensor, p: torch.Tensor, outlier: bool
) -> torch.Tensor:
    """Return the 11 standardised quantiles; folded rows use the folded CDF (replaces experiment 510's)."""
    assert not outlier
    with torch.no_grad():
        log_w, mean, std = base.mixture_parts(out=model(x, p).double(), components=m.COMPONENTS)
        q = base.mixture_quantiles(log_w=log_w, mean=mean, std=std)
        if FOLDED is not None:
            rows = FOLDED["mask"][p]
            if bool(rows.any()):
                q[rows] = folded.folded_quantiles(
                    log_w=log_w[rows],
                    mean=mean[rows],
                    std=std[rows],
                    offset=FOLDED["offset"][p][rows].double(),
                )
        return q


c.serving_quantiles = serving_quantiles


def fresh_mixer(*, model: m.ModularMdn) -> None:
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


def load_pretrained_fresh(*, model: m.ModularMdn, path: Path) -> None:
    """Load pretrained weights (without `mixer_ft`), then a fresh mixer."""
    raw = torch.load(path, map_location="cpu")
    if SPEC.source == "drift":
        mdr.load_pre_into_ft(model=model, state=raw)
        fresh_mixer(model=model)
        return
    state = {a: v for a, v in raw.items() if not a.startswith("mixer_ft.")}
    missing = model.load_state_dict(state, strict=False)
    assert all(a.startswith("mixer_ft.") for a in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing
    fresh_mixer(model=model)


def new_model(*, n_primaries: int, stage: str = "ft") -> m.ModularMdn:
    """Return an untrained network (fine-tuning stage unless `stage` says `pre`)."""
    if SPEC.columns:
        if SPEC.source == "drift":
            return mdr.ModularMdnDrift(n_primaries=n_primaries, stage=stage, n_knots=KNOTS)
        if SPEC.source == "uv":
            return mex.ModularMdnExtra(
                n_primaries=n_primaries, stage=stage, n_weather=4, n_spread=4
            )
        return mwt.ModularMdnWt(n_primaries=n_primaries, stage=stage)
    return m.ModularMdn(n_primaries=n_primaries, stage=stage)


def train(  # noqa: PLR0912, PLR0915
    *, model: m.ModularMdn, x, z, w, p, is_val, seed: int, lr: float, max_epochs: int,
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
    encoder_ids = {id(q) for part in (model.weather, model.datetime) for q in part.parameters()}
    if frozen:
        for part in (model.weather, model.datetime):
            for q in part.parameters():
                q.requires_grad_(False)
    fresh = list(model.mixer_ft.parameters()) if fresh_lr is not None else []
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
            model.weather.eval()
            model.datetime.eval()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            nll = row_nll(out=model(x[b], p[b]), z=z[b], p=p[b])
            loss = (w[b] * nll).sum() / w[b].sum()
            if hasattr(model, "penalty"):
                loss = loss + model.penalty()
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
WT: wt.WtColumns | extra.ExtraColumns | extra.DriftColumns | None = None
KNOTS: int = 0
UV: extra.UvSource | None = None


def uv_source() -> extra.UvSource:
    """Return the wind-component source, loaded once per process."""
    global UV  # noqa: PLW0603
    if UV is None:
        UV = extra.UvSource()
    return UV


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
    if SPEC.columns:
        global WT  # noqa: PLW0603
        global KNOTS  # noqa: PLW0603
        if SPEC.source == "drift":
            WT = extra.DriftColumns(month=month, mr=mr)
            KNOTS = WT.n_knots
        elif SPEC.source == "uv":
            WT = extra.ExtraColumns(month=month, mr=mr, source=uv_source())
        else:
            WT = wt.WtColumns(month=month, mr=mr)
        mr.x = np.concatenate([mr.x, WT.month(null=SPEC.columns == "null")], axis=1)
        CHECK["months"][month]["wt_example"] = WT.example
        CHECK["months"][month]["placebo_rows_without_donor"] = WT.null_rows_without_donor
        print(month, f"{SPEC.source} columns appended |", WT.example, flush=True)
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
    if SPEC.columns:
        extra = WT.rows(
            kind="ens", primary=test.primary, origin_idx=test.origin_idx, valid_idx=test.target_idx,
            null=SPEC.columns == "null",
        )  # fmt: skip
        test = dataclasses.replace(test, x=np.concatenate([test.x, extra], axis=1))
    return test


k.build_month, data.build_test = build_month, build_test


def refit_rows(*, g: k.GpuMonth, cut: datetime, half_life: float, shift_z=None):
    """Return the cutoff's selected rows (with the leak report) and their GPU tensors."""
    refit = c.select_rows(rows=g.mr, cutoff=cut, half_life_days=half_life)
    assert refit.report["rows_past_valid_cut"] == 0 and refit.report["rows_past_origin_cut"] == 0
    return refit, g.refit_tensors(refit=refit, shift_z=shift_z)


def step2(*, device: str) -> None:
    """Train and save the step-2 weights of each month and seed: encoders and a fresh mixer on all rows up to the first cutoff."""
    clean = c.clean_stack()
    for month in ARGS.months:
        todo = [s for s in ARGS.seeds if not step2_file(month=month, seed=s).exists()]
        if not todo:
            continue
        mr = k.build_month(month=month, clean=clean)
        g = k.GpuMonth(mr=mr, device=device)
        set_folded(mr=mr, device=device)
        cut = c.month_start(month=month)
        refit, t = refit_rows(g=g, cut=cut, half_life=UNIFORM_HALF_LIFE)
        assert float(t["w"].min()) > 0.999, "step 2 must have no recency weighting"
        print(f"{month}: step 2 rows {len(refit.index):,}; {refit.report}", flush=True)
        for seed in todo:
            started = time.monotonic()
            model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
            load_pretrained_fresh(model=model, path=pretrained_file(month=month, seed=seed))
            model.to(device)
            epochs, best = train(
                model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                lr=c.FINE_TUNE_LR, max_epochs=c.MAX_EPOCHS, fresh_lr=FRESH_LR, frozen=False,
                anchor=None, lam=0.0, device=device,
            )  # fmt: skip
            target = step2_file(month=month, seed=seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  step 2 {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
            c.write_json(
                obj={"epochs": epochs, "val_nll": best, **refit.report},
                path=W / "logs" / "step2" / f"{month}_s{seed}.json",
            )
        del t, g, mr
        torch.cuda.empty_cache()


def pretrain_wt(*, device: str) -> None:
    """Pretrain the W-T arm's network (real or placebo columns) on each fold's ERA5 and CAMS rows."""
    assert SPEC.columns
    clean = c.clean_stack()
    for month in ARGS.months:
        todo = [
            s
            for s in ARGS.seeds
            if not (W / f"pretrained_{SPEC.source}" / SPEC.columns / f"{month}_s{s}.pt").exists()
        ]
        if not todo:
            continue
        mr = k.build_month(month=month, clean=clean)
        refit_t = c.month_start(month=month)
        pre = r448.pretrain_rows(rows=mr.base, refit=refit_t, clean=clean)
        latest = datetime.fromtimestamp(
            (pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC
        )
        assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
        extra = WT.rows(
            kind="era5", primary=pre.primary, origin_idx=pre.origin_idx, valid_idx=pre.target_idx,
            null=SPEC.columns == "null",
        )  # fmt: skip
        x = np.concatenate([pre.x, extra], axis=1)
        z = (pre.y - mr.base.target_mean[pre.primary]) / mr.base.target_std[pre.primary]
        t = {
            "x": torch.from_numpy(x).to(device), "z": torch.from_numpy(z.astype(np.float32)).to(device),
            "w": torch.ones(len(pre.y), device=device), "p": torch.from_numpy(pre.primary).to(device),
            "v": torch.from_numpy(pre.is_val).to(device),
        }  # fmt: skip
        print(
            f"{month}: pretraining rows {len(x):,}; latest valid {latest.isoformat()} (refit {refit_t.isoformat()})",
            flush=True,
        )
        for seed in todo:
            started = time.monotonic()
            torch.manual_seed(seed)
            model = new_model(n_primaries=len(mr.base.target_mean), stage="pre").to(device)
            epochs, best = train(
                model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                lr=c.PRETRAIN_LR, max_epochs=c.MAX_EPOCHS, fresh_lr=None, frozen=False,
                anchor=None, lam=0.0, device=device,
            )  # fmt: skip
            target = W / f"pretrained_{SPEC.source}" / SPEC.columns / f"{month}_s{seed}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  pretrain {SPEC.tag} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
        del t, pre, mr
        torch.cuda.empty_cache()


def lambda_value() -> float:
    """Return the L2 strength chosen on fold 1 (stored by `choose-lambda`)."""
    return float(json.loads((W / "lambda.json").read_text())["lambda"])


def choose_lambda(*, device: str) -> None:
    """Pick the L2 strength on fold 1's training days only.

    For each strength, fine-tune the fold's step-2 weights at the fold's first cutoff (seed 0) and
    record the early-stopping weeks' NLL. The pick is the largest strength whose NLL is within
    `LAMBDA_TOLERANCE` of the best. No test row is read.
    """
    clean = c.clean_stack()
    mr = k.build_month(month=LAMBDA_FOLD, clean=clean)
    g = k.GpuMonth(mr=mr, device=device)
    _, t = refit_rows(g=g, cut=c.month_start(month=LAMBDA_FOLD), half_life=90.0)
    anchor = torch.load(step2_file(month=LAMBDA_FOLD, seed=0), map_location="cpu")
    scores = {}
    for lam in LAMBDAS:
        model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
        model.load_state_dict(anchor)
        epochs, best = train(
            model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=0,
            lr=c.FINE_TUNE_LR, max_epochs=c.MAX_EPOCHS, fresh_lr=None, frozen=False,
            anchor=anchor, lam=lam, device=device,
        )  # fmt: skip
        scores[lam] = best
        print(f"lambda {lam}: {epochs} epochs, early-stopping NLL {best:.5f}", flush=True)
    top = min(scores.values())
    pick = max(lam for lam, v in scores.items() if v <= top + LAMBDA_TOLERANCE)
    (W / "lambda.json").write_text(json.dumps({"lambda": pick, "scores": scores}, indent=1))
    print("chosen lambda", pick, flush=True)


def model_file(*, tag: str, month: str, day: int, seed: int, sfx: str = "") -> Path:
    """Return the saved weights of a daily fit."""
    return W / "models" / tag / f"{month}_d{day}_s{seed}{sfx}.pt"


def fine_tune(
    *, g: k.GpuMonth, month: str, cut: datetime, seed: int, device: str,
    shift: np.ndarray | None = None, sign: int = 0, block: tuple[int, int] | None = None,
) -> tuple[m.ModularMdn, dict]:  # fmt: skip
    """Return a fine-tuned model and its log for one cutoff of the arm in `SPEC`."""
    mr = g.mr
    started = time.monotonic()
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
    if SPEC.start == "step2":
        anchor = torch.load(step2_file(month=month, seed=seed), map_location="cpu")
        model.load_state_dict(anchor)
    else:
        load_pretrained_fresh(model=model, path=pretrained_file(month=month, seed=seed))
    model.to(device)
    t0 = time.monotonic()
    epochs, best = train(
        model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
        lr=SPEC.lr, max_epochs=SPEC.max_epochs, fresh_lr=FRESH_LR if SPEC.start == "fresh" else None,
        frozen=SPEC.frozen, anchor=anchor, lam=lambda_value() if SPEC.penalty else 0.0, device=device,
    )  # fmt: skip
    day = f.day_of(month=month, cut=cut)
    log = {
        "name": f"{SPEC.tag}/{month}_d{day}_s{seed}", "epochs": epochs, "val_nll": best,
        "feeder_seconds": round(feeder, 2), "train_seconds": round(time.monotonic() - t0, 2),
        "injected_rows": injected, **refit.report,
    }  # fmt: skip
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
        set_folded(mr=mr, device=device)
        cuts = k.cutoffs(month=month, cadence=CADENCE)
        start_day = c.month_start(month=month)
        if stress:
            end = start_day + timedelta(days=f.STRESS_END_DAY)
            first = start_day + timedelta(
                days=STRESS_FIRST_DAY if SPEC.tag == "A1" else f.STRESS_END_DAY
            )
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


def average_arm(*, months: list[str], seeds: list[int], stress: bool, device: str) -> None:
    """Arm A5: the window forecast of cutoff `d` from the mean weights of A1's fits on days d-4 to d (clamped at day 0)."""
    assert SPEC.average_of == "A1"
    clean = c.clean_stack()
    for month in months:
        mr = k.build_month(month=month, clean=clean)
        test = data.build_test(month=month, standardiser=mr.base.standardiser)
        cuts = k.cutoffs(month=month, cadence=CADENCE)
        start_day = c.month_start(month=month)
        if stress:
            end = start_day + timedelta(days=f.STRESS_END_DAY)
            todo = [i for i, x in enumerate(cuts) if end <= x < end + timedelta(days=7)]
        else:
            todo = list(range(len(cuts)))
        for seed in seeds:
            for i in todo:
                day = f.day_of(month=month, cut=cuts[i])
                for sign in (+1, -1) if stress else (0,):
                    sfx = f"_stress{'p' if sign > 0 else 'n'}" if stress else ""
                    out = W / "fits" / SPEC.tag / f"{month}_d{day}_s{seed}{sfx}.parquet"
                    if out.exists() or out.with_suffix(".empty").exists():
                        continue
                    days = range(max(0, day - AVERAGE_OVER + 1), day + 1)
                    files = [
                        model_file(tag="A1", month=month, day=d, seed=seed, sfx=sfx) for d in days
                    ]
                    if not all(p.exists() for p in files):
                        print("skip (A1 fits missing)", month, day, seed, sfx, flush=True)
                        continue
                    states = [torch.load(p, map_location=device) for p in files]
                    mean = {
                        a: torch.stack([s[a].float() for s in states])
                        .mean(0)
                        .to(states[0][a].dtype)
                        for a in states[0]
                    }
                    model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
                    model.load_state_dict(mean)
                    model.eval()
                    lo, hi = k.window_of(month=month, cadence=CADENCE, index=i)
                    frame = f.window_frame(
                        model=model, test=test, mr=mr, lo=lo, hi=hi, flag=False, device=device
                    )
                    if frame is None:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.with_suffix(".empty").touch()
                    else:
                        f.atomic_frame(frame=frame, target=out)
                    print(f"A5/{month}_d{day}_s{seed}{sfx}: mean of {len(files)} fits", flush=True)
        del mr, test
        torch.cuda.empty_cache()


def assemble() -> None:
    """Concatenate each cutoff's window forecast into one table per month and seed."""
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence=CADENCE)
        for seed in ARGS.seeds:
            parts = [
                W / "fits" / SPEC.tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            assert all(p.exists() or p.with_suffix(".empty").exists() for p in parts), (month, seed)
            frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
            f.atomic_frame(
                frame=frame, target=W / "seed_preds" / SPEC.tag / f"{month}_s{seed}.parquet"
            )
            print("wrote", SPEC.tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble()
    elif ARGS.command == "step2":
        step2(device="cuda")
    elif ARGS.command == "choose-lambda":
        choose_lambda(device="cuda")
    elif ARGS.command == "pretrain-wt":
        pretrain_wt(device="cuda")
    elif ARGS.command == "average":
        average_arm(months=ARGS.months, seeds=ARGS.seeds, stress=ARGS.stress, device="cuda")
    else:
        fit_arm(
            months=ARGS.months,
            seeds=ARGS.seeds,
            stress=ARGS.stress,
            limit=ARGS.limit,
            device="cuda",
        )
    if ARGS.command in {"fit", "step2"}:
        out = (
            WORK
            / f"checks_{ARGS.command}_{SPEC.tag}{'_stress' if ARGS.stress else ''}_s{'-'.join(map(str, ARGS.seeds))}.json"
        )
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in CHECK.items() if a != "months"})

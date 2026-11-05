# ruff: noqa: E501, ANN001, ANN201, PT018
"""Experiment 571: coarse-refit fits of the drift arms in a known-answer world (GPU).

A reduced copy of experiment 565's `e00565_fit.py` (daily harness, rows, recency weights, early
stopping, seeds, scoring windows, origin-time weather assertions), with these changes: the world named
by the `WORLD` environment variable is injected into the cleaned power before any row is built
(`e00571_world`); the arms are the base network, the oracle, D5, A10, and four hybrids; and the
state of each world lives under its own folder. Cadence `h` refits on days 0 and 14 of each fold.

Usage: `WORLD=Wc python e00571_fit.py {pretrain|fit|assemble} --arm ARM [--stress]
[--months ...] [--seeds ...]`
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
import e00559_extra as extra  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00559_model_drift as mdr  # ty: ignore[unresolved-import]
import e00565_model_variants as var  # ty: ignore[unresolved-import]
import e00571_models as mine
import e00571_world as world
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

WORLD = world.world_from_env()
WORK = world.WORK
W = WORK / "worlds" / WORLD / "work"
FRESH_SEED = 12345
"""Initialisation seed of every fresh mixer, as in experiments 545, 554, and 565."""
FRESH_LR = 5e-4
COARSE_OFFSETS = (0, 14)
"""Cadence `h`: refits on days 0 and 14 (the day-14 fit is the injected-shift control)."""


@dataclass(frozen=True)
class Spec:
    """How an arm's fits start and train."""

    tag: str
    source: str  # "none" (the base network), "oracle" (the true injected input), or "drift" (knot position)
    variant: str  # network: "base", "oracle", "a10", "out" (D5), "hybrid" (joint), "film", "orth"
    frozen: bool = False
    staged: bool = False
    half_life: float = 90.0
    lr: float = c.FINE_TUNE_LR
    max_epochs: int = c.MAX_EPOCHS
    columns: str = "real"


SPECS = {
    s.tag: s
    for s in (
        Spec("M0w", "none", "base"),
        Spec("ORA", "oracle", "oracle"),
        Spec("A10", "drift", "a10", frozen=True),
        Spec("D5", "drift", "out", frozen=True),
        Spec("HJ", "drift", "hybrid", frozen=True),
        Spec("HS", "drift", "hybrid", frozen=True, staged=True),
        Spec("HF", "drift", "film", frozen=True),
        Spec("HO", "drift", "orth", frozen=True),
    )
}
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("pretrain", "fit", "assemble"))
parser.add_argument("--arm", choices=tuple(SPECS), required=True)
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
SPEC = SPECS[ARGS.arm]
CHECK: dict = {
    "world": WORLD, "train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}
}  # fmt: skip
_cutoffs = k.cutoffs
world.install(WORLD)

_, COLUMNS, _ = data.non_lag_layout()
assert len(COLUMNS) == m.N_INPUTS
assert all(n.startswith("std_") for n in COLUMNS[m.SPREAD_COLS])
assert not any("lag" in n for n in COLUMNS), "the base network must be lag-free"
assert COLUMNS[m.LEAD_COLS] == ["lead", "lead_sqrt"]


def cutoffs(*, month: str, cadence: str) -> list[datetime]:
    """Refit times of the coarse cadence: days 0 and 14 of the month."""
    start, stop = c.month_start(month=month), c.month_stop(month=month)
    days = (stop - start).days
    return [start + timedelta(days=d) for d in COARSE_OFFSETS if d < days]


k.cutoffs = cutoffs
CADENCE = "d"


def pretrained_dir(*, tag: str | None = None) -> Path:
    """Return where an arm's pretrained weights live."""
    return W / f"pretrained_{tag or SPEC.tag}" / SPEC.columns


def pretrained_file(*, month: str, seed: int, tag: str | None = None) -> Path:
    """Return an arm's pretrained weights of a fold and seed."""
    p = pretrained_dir(tag=tag) / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def row_nll(*, out: torch.Tensor, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Return each row's negative log-likelihood."""
    return base.mixture_nll(out=out, z=z, components=m.COMPONENTS)


def serving_quantiles(
    *, model: torch.nn.Module, x: torch.Tensor, p: torch.Tensor, outlier: bool
) -> torch.Tensor:
    """Return the 11 standardised quantiles of the main mixture."""
    assert not outlier
    with torch.no_grad():
        log_w, mean, std = base.mixture_parts(out=model(x, p).double(), components=m.COMPONENTS)
        return base.mixture_quantiles(log_w=log_w, mean=mean, std=std)


c.serving_quantiles = serving_quantiles


class NoColumns:
    """No extra column (the base network)."""

    n_knots = 0
    null_rows_without_donor = 0
    example = "no extra column"

    def __init__(self, *, month: str, mr: c.MonthRows) -> None:
        """Remember the month's row count."""
        self.n = len(mr.valid_us)

    def month(self, *, null: bool) -> np.ndarray:
        """Return an empty block for every month row."""
        return np.zeros((self.n, 0), dtype=np.float32)

    def rows(self, *, kind, primary, origin_idx, valid_idx, null) -> np.ndarray:
        """Return an empty block for another row set."""
        return np.zeros((len(primary), 0), dtype=np.float32)


class OracleColumns(NoColumns):
    """The true injected component at the target time, over the Primary's target standard deviation."""

    def __init__(self, *, month: str, mr: c.MonthRows) -> None:
        """Load the world's truth (all Primaries by grid)."""
        super().__init__(month=month, mr=mr)
        self.truth = world.truth_full(world=WORLD, n_primaries=len(mr.base.target_mean))
        self.std = mr.base.target_std.astype(np.float32)
        self.primary, self.target_idx = mr.primary, mr.target_idx
        i = len(self.primary) // 2
        self.example = f"target index {self.target_idx[i]}, Primary {self.primary[i]}: oracle input {self._u(self.primary[i : i + 1], self.target_idx[i : i + 1])[0]:.4f}"

    def _u(self, primary: np.ndarray, valid_idx: np.ndarray) -> np.ndarray:
        return (self.truth[primary, valid_idx] / self.std[primary]).astype(np.float32)

    def month(self, *, null: bool) -> np.ndarray:
        """Return the oracle input of every month row."""
        return self._u(self.primary, self.target_idx)[:, None]

    def rows(self, *, kind, primary, origin_idx, valid_idx, null) -> np.ndarray:
        """Return the oracle input of another row set."""
        return self._u(np.asarray(primary), np.asarray(valid_idx, dtype=np.int64))[:, None]


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
    if isinstance(model, mine.FilmHybrid | mine.OracleMdn) or SPEC.variant == "base":
        mine.load_pre_into_ft(model=model, state=raw)
    else:
        var.load_pre_into_ft(model=model, state=raw)  # ty: ignore[invalid-argument-type]
    fresh_mixer(model=model)


KNOTS: int = 0


def new_model(*, n_primaries: int, stage: str = "ft") -> m.ModularMdn:
    """Return an untrained network (fine-tuning stage unless `stage` says `pre`)."""
    match SPEC.variant:
        case "base":
            return m.ModularMdn(n_primaries=n_primaries, stage=stage)
        case "oracle":
            return mine.OracleMdn(n_primaries=n_primaries, stage=stage)
        case "a10":
            return mdr.ModularMdnDrift(n_primaries=n_primaries, stage=stage, n_knots=KNOTS)
        case "out":
            return var.OutputOffsetDrift(n_primaries=n_primaries, stage=stage, n_knots=KNOTS)
        case "hybrid":
            return var.HybridDrift(n_primaries=n_primaries, stage=stage, n_knots=KNOTS)
        case "film" | "orth":
            return mine.FilmHybrid(
                n_primaries=n_primaries, stage=stage, n_knots=KNOTS, orth=SPEC.variant == "orth"
            )
    raise ValueError(SPEC.variant)


def init_staged_pretraining(*, model: var.HybridDrift, month: str, seed: int) -> None:
    """Start the staged hybrid's pre-training from D5's: copy its encoders, Primary embedding, and offset knots.

    The offset knots are then frozen; the hybrid's wider mixer and the embedding knots are fitted.
    """
    raw = torch.load(pretrained_file(month=month, seed=seed, tag="D5"), map_location="cpu")
    own = model.state_dict()
    keep = {
        a: v
        for a, v in raw.items()
        if a.split(".")[0] in {"embedding", "weather", "datetime"} and own[a].shape == v.shape
    }
    model.load_state_dict(keep, strict=False)
    with torch.no_grad():
        model.off_knots.copy_(raw["knots"][..., 0])
    model.off_knots.requires_grad_(False)


def train(  # noqa: PLR0912, PLR0915
    *, model: m.ModularMdn, x, z, w, p, is_val, seed: int, lr: float, max_epochs: int,
    fresh_lr: float | None, frozen: bool, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Experiment 565's loop: a separate rate for a fresh mixer, frozen encoders, and the network's own penalty.

    The early-stopping validation NLL has no penalty. Frozen encoders (weather and datetime) are held
    in evaluation mode so their dropout is off. Parameters with `requires_grad` already False stay out
    of the optimiser's updates (the staged hybrid's offset, the FiLM map in fine-tuning).
    """
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    if hasattr(model, "set_probe"):
        model.set_probe(x=x[train_idx], p=p[train_idx], seed=seed)
    encoder_ids = {id(q) for part in (model.weather, model.datetime) for q in part.parameters()}
    was_frozen = {id(q) for q in model.parameters() if not q.requires_grad}
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
                loss = loss + model.penalty()  # ty: ignore[call-non-callable]
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
        if id(q) not in was_frozen:
            q.requires_grad_(True)
    return epochs, best


_build_month, _build_test = k.build_month, data.build_test
WT: NoColumns | extra.DriftColumns | None = None


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
    global WT, KNOTS  # noqa: PLW0603
    if SPEC.source == "drift":
        WT = extra.DriftColumns(month=month, mr=mr)
        KNOTS = WT.n_knots
    elif SPEC.source == "oracle":
        WT = OracleColumns(month=month, mr=mr)
    else:
        WT = NoColumns(month=month, mr=mr)
    mr.x = np.concatenate([mr.x, WT.month(null=False)], axis=1)
    CHECK["months"][month]["extra_example"] = WT.example
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
    extra_cols = WT.rows(  # ty: ignore[unresolved-attribute]
        kind="ens", primary=test.primary, origin_idx=test.origin_idx, valid_idx=test.target_idx,
        null=False,
    )  # fmt: skip
    return dataclasses.replace(test, x=np.concatenate([test.x, extra_cols], axis=1))


k.build_month, data.build_test = build_month, build_test


def refit_rows(*, g: k.GpuMonth, cut: datetime, half_life: float, shift_z=None):
    """Return the cutoff's selected rows (with the leak report) and their GPU tensors."""
    refit = c.select_rows(rows=g.mr, cutoff=cut, half_life_days=half_life)
    assert refit.report["rows_past_valid_cut"] == 0 and refit.report["rows_past_origin_cut"] == 0
    return refit, g.refit_tensors(refit=refit, shift_z=shift_z)


def pretrain(*, device: str) -> None:
    """Pretrain the arm's network on each fold's ERA5 and CAMS rows (the world's injected target)."""
    clean = c.clean_stack()
    for month in ARGS.months:
        todo = [s for s in ARGS.seeds if not (pretrained_dir() / f"{month}_s{s}.pt").exists()]
        if not todo:
            continue
        mr = k.build_month(month=month, clean=clean)
        refit_t = c.month_start(month=month)
        pre = r448.pretrain_rows(rows=mr.base, refit=refit_t, clean=clean)
        latest = datetime.fromtimestamp(
            (pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC
        )
        assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
        extra_cols = WT.rows(  # ty: ignore[unresolved-attribute]
            kind="era5", primary=pre.primary, origin_idx=pre.origin_idx, valid_idx=pre.target_idx,
            null=False,
        )  # fmt: skip
        x = np.concatenate([pre.x, extra_cols], axis=1)
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
            if SPEC.staged:
                init_staged_pretraining(model=model, month=month, seed=seed)  # ty: ignore[invalid-argument-type]
            epochs, best = train(
                model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                lr=c.PRETRAIN_LR, max_epochs=c.MAX_EPOCHS, fresh_lr=None, frozen=False, device=device,
            )  # fmt: skip
            target = pretrained_dir() / f"{month}_s{seed}.pt"
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


def model_file(*, tag: str, month: str, day: int, seed: int, sfx: str = "") -> Path:
    """Return the saved weights of a fit."""
    return W / "models" / tag / f"{month}_d{day}_s{seed}{sfx}.pt"


def freeze_staged_offset(
    *, model: var.HybridDrift, month: str, day: int, seed: int, sfx: str
) -> None:
    """Set the staged hybrid's newest offset knot to D5's fitted value for the same fit, and freeze it."""
    d5 = torch.load(
        model_file(tag="D5", month=month, day=day, seed=seed, sfx=sfx), map_location="cpu"
    )
    with torch.no_grad():
        model.off_new.copy_(d5["knot_new"][:, 0])  # ty: ignore[call-non-callable]
    model.off_new.requires_grad_(False)


def fine_tune(
    *, g: k.GpuMonth, month: str, cut: datetime, seed: int, device: str,
    shift: np.ndarray | None = None, sign: int = 0, block: tuple[int, int] | None = None,
    sfx: str = "",
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
    day = f.day_of(month=month, cut=cut)
    load_pretrained_fresh(model=model, path=pretrained_file(month=month, seed=seed))
    if SPEC.staged:
        freeze_staged_offset(model=model, month=month, day=day, seed=seed, sfx=sfx)  # ty: ignore[invalid-argument-type]
    model.to(device)
    t0 = time.monotonic()
    epochs, best = train(
        model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
        lr=SPEC.lr, max_epochs=SPEC.max_epochs, fresh_lr=FRESH_LR, frozen=SPEC.frozen, device=device,
    )  # fmt: skip
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
                            shift=shift, sign=sign, block=block, sfx=sfx,
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
    """Concatenate each cutoff's window forecast into one table per month and seed."""
    for month in ARGS.months:
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
            print("wrote", WORLD, SPEC.tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble()
    elif ARGS.command == "pretrain":
        pretrain(device="cuda")
    else:
        fit_arm(
            months=ARGS.months,
            seeds=ARGS.seeds,
            stress=ARGS.stress,
            limit=ARGS.limit,
            device="cuda",
        )
    if ARGS.command in {"fit", "pretrain"}:
        out = (
            W
            / f"checks_{ARGS.command}_{SPEC.tag}{'_stress' if ARGS.stress else ''}_s{'-'.join(map(str, ARGS.seeds))}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in CHECK.items() if a != "months"})

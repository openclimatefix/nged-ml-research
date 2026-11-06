# ruff: noqa: E501, ANN001, ANN201, PT018
"""Experiment 611, Design 1: the lag-free network with the O2 offset, pooled per group (GPU).

A trimmed copy of experiment 577's `e00577_fit.py`: only arm O2 remains, with the penalty weight
3000 that experiment 577 picked. The harness is pointed at one group of 2 series by
`e00611_pool.install`. The harness is experiment 565's coarse one: refits on days 0 and 14 of each of
the five folds, ERA5 and CAMS pretraining rows, frozen encoders, a fresh mixer, early stopping, and
the origin-time weather assertions.

Usage: python e00611_fit.py {pretrain|fit|assemble} --group {bsp|gsp} [--months ...]
       [--seeds ...] [--limit N]
"""

import argparse
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
import e00577_model_variants as mv  # ty: ignore[unresolved-import]
import e00577_widen as widen  # ty: ignore[unresolved-import]
import e00611_pool as pool
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

FRESH_SEED = 12345
FRESH_LR = 5e-4
COARSE_OFFSETS = (0, 14)


@dataclass(frozen=True)
class Spec:
    """One arm: its network, penalty weight, and where its weights come from."""

    tag: str
    variant: str  # out, tv1, tv2, gate, outvar, tv1var, tv2var
    lam: float | None = None  # sweep value; None reads the pick for `group`
    group: str | None = None  # key into picks.json: O1, O2, or O3
    pre_from: str | None = None  # another arm's pretrained weights ("D5" is experiment 565's)
    models_from: str | None = None  # re-forecast this arm's saved fine-tuned models ("D5" is 565's)
    post: str = "none"  # none, var, q1, q3, qconst: how a forecast adjusts the mixture
    prior_sd: float | None = None  # sweep value of the posterior prior; None reads the pick


SPECS = {"O2": Spec("O2", "tv2", lam=3000.0)}
VARIANTS = {"tv2": mv.TvDailyOffset}
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("pretrain", "fit", "assemble"))
parser.add_argument("--group", choices=tuple(pool.GROUPS), required=True)
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
SPEC = SPECS["O2"]
W = pool.install(group=ARGS.group)
WORK = W.parent.parent
W.mkdir(parents=True, exist_ok=True)
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}
_cutoffs = k.cutoffs

_, COLUMNS, _ = data.non_lag_layout()
assert len(COLUMNS) == m.N_INPUTS, "the base network has no lag column"
assert all(n.startswith("std_") for n in COLUMNS[m.SPREAD_COLS])


def picks() -> dict:
    """Return no picks: arm O2 has none."""
    return {}


def lam_value() -> float:
    """Return the arm's penalty weight."""
    return float(SPEC.lam)  # ty: ignore[invalid-argument-type]


def prior_value() -> None:
    """Return no posterior prior: arm O2 has none."""
    return


def cutoffs(*, month: str, cadence: str) -> list[datetime]:
    """Refits on days 0 and 14 of the month."""
    start, stop = c.month_start(month=month), c.month_stop(month=month)
    days = (stop - start).days
    _ = cadence
    return [start + timedelta(days=d) for d in COARSE_OFFSETS if d < days]


k.cutoffs = cutoffs
CADENCE = "d"


def pre_tag() -> str:
    """Return the arm whose pretrained weights this arm reads."""
    return SPEC.pre_from or SPEC.tag


def pretrained_dir() -> Path:
    """Return where the arm's pretrained weights live."""
    return W / f"pretrained_{pre_tag()}" / "real"


def models_dir(*, tag: str | None = None) -> Path:
    """Return the folder of saved fine-tuned models of an arm (experiment 565's for D5)."""
    t = tag or SPEC.models_from or SPEC.tag
    return W / "models" / t


def set_context(
    *, model: torch.nn.Module, month: str, day: int, seed: int, mr: c.MonthRows
) -> None:
    """Attach the forecast-time context that the widening hook reads."""
    model.cut_offset_days = float(day)  # ty: ignore[invalid-assignment]
    model.target_std = torch.from_numpy(mr.base.target_std.astype(np.float64)).to("cuda")  # ty: ignore[invalid-assignment]
    widen.CTX.update(month=month, day=day, seed=seed)
    if SPEC.tag == "Q5shuf":
        n = len(mr.base.target_mean)
        model.perm = torch.from_numpy(np.roll(np.arange(n), 1)).to("cuda")  # ty: ignore[invalid-assignment]


def adjusted(*, model, x, p, out):
    """Return the mixture parts after the arm's widening."""
    log_w, mean, std = base.mixture_parts(out=out.double(), components=m.COMPONENTS)
    return widen.apply(
        post=SPEC.post, model=model, x=x, p=p, log_w=log_w, mean=mean, std=std, picks=picks()
    )


def row_nll(*, model, x, p, z):
    """Return each row's negative log-likelihood; evaluation includes the posterior variance."""
    out = model(x, p)
    if hasattr(model, "extra_variance") and not model.training:
        log_w, mean, std = adjusted(model=model, x=x, p=p, out=out)
        log_pdf = -0.5 * ((z[..., None] - mean) / std) ** 2 - torch.log(std)
        return -torch.logsumexp(log_w + log_pdf, dim=-1) + 0.5 * float(np.log(2 * np.pi))
    return base.mixture_nll(out=out, z=z, components=m.COMPONENTS)


def serving_quantiles(*, model, x, p, outlier):
    """Return the 11 standardised quantiles of the (possibly widened) mixture."""
    assert not outlier
    with torch.no_grad():
        log_w, mean, std = adjusted(model=model, x=x, p=p, out=model(x, p))
        return base.mixture_quantiles(log_w=log_w, mean=mean, std=std)


c.serving_quantiles = serving_quantiles


def fresh_mixer(*, model: m.ModularMdn) -> None:
    """Give the model's `mixer_ft` a seed-12345 initialisation (experiment 554's recipe)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FRESH_SEED)
        fresh = base.mlp(
            n_in=model.mixer_ft[0].in_features,  # ty: ignore[not-subscriptable,unresolved-attribute]
            hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT,
        )  # fmt: skip
        base.init_head(head=fresh, components=m.COMPONENTS)
    model.mixer_ft.load_state_dict(fresh.state_dict())  # ty: ignore[unresolved-attribute]


def load_pretrained_fresh(*, model, path: Path) -> None:
    """Load pretrained weights, then a fresh mixer."""
    raw = torch.load(path, map_location="cpu")
    mv.load_pre_into_ft(model=model, state=raw)
    fresh_mixer(model=model)


KNOTS = 0


def new_model(*, n_primaries: int, stage: str = "ft"):
    """Return an untrained network of the arm's variant."""
    return VARIANTS[SPEC.variant](n_primaries=n_primaries, stage=stage, n_knots=KNOTS)


def train(
    *, model, x, z, w, p, is_val, seed: int, lr: float, fresh_lr: float | None, frozen: bool, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Run experiment 565's loop with the offsets' penalty scaled by the row count."""
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    lam = lam_value()
    model.configure(
        **({"lam": lam} if lam is not None else {"lam": 0.0}),
        n_rows=float(len(train_idx)),
        prior_sd=prior_value(),
    )
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
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        if frozen:
            model.weather.eval()
            model.datetime.eval()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            nll = row_nll(model=model, x=x[b], p=p[b], z=z[b])
            loss = (w[b] * nll).sum() / w[b].sum() + model.penalty()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        row_nll(model=model, x=x[i], p=p[i], z=z[i])
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
WT: extra.DriftColumns | None = None


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
    global WT, KNOTS  # noqa: PLW0603
    WT = extra.DriftColumns(month=month, mr=mr)
    KNOTS = WT.n_knots
    mr.x = np.concatenate([mr.x, WT.month(null=False)], axis=1)
    CHECK["months"][month]["knot_position_example"] = WT.example
    print(month, "train checks passed |", WT.example, flush=True)
    return mr


def build_test(*, month, standardiser):
    """Build the month's test rows; assert nwp_init_time <= power_fcst_init_time for every test origin."""
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
        f"weather run {datetime.fromtimestamp(test.origin[0] / 1e6, UTC) - pub}"
    )
    print(month, "test checks passed |", CHECK["months"][month]["weather_example"], flush=True)
    extra_cols = WT.rows(  # ty: ignore[unresolved-attribute]
        kind="ens",
        primary=test.primary,
        origin_idx=test.origin_idx,
        valid_idx=test.target_idx,
        null=False,
    )  # ty: ignore[unresolved-attribute]
    import dataclasses

    return dataclasses.replace(test, x=np.concatenate([test.x, extra_cols], axis=1))


k.build_month, data.build_test = build_month, build_test


def atomic_save(*, obj, target: Path) -> None:
    """Save a state dict atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(target)


def pretrain(*, device: str) -> None:
    """Pretrain the arm's network on each fold's ERA5 and CAMS rows; log each early-stopping NLL."""
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
            kind="era5",
            primary=pre.primary,
            origin_idx=pre.origin_idx,
            valid_idx=pre.target_idx,
            null=False,
        )  # ty: ignore[unresolved-attribute]
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
            epochs, best = train(
                model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                lr=c.PRETRAIN_LR, fresh_lr=None, frozen=False, device=device,
            )  # fmt: skip
            atomic_save(obj=model.state_dict(), target=pretrained_dir() / f"{month}_s{seed}.pt")
            c.write_json(
                obj={"epochs": epochs, "val_nll": best},
                path=W / "logs" / "pretrain" / f"{SPEC.tag}_{month}_s{seed}.json",
            )
            print(
                f"  pretrain {SPEC.tag} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
        del t, pre, mr
        torch.cuda.empty_cache()


def fine_tune(*, g: k.GpuMonth, month: str, cut: datetime, seed: int, device: str):
    """Return a fine-tuned model and its log for one cutoff of the arm."""
    mr = g.mr
    started = time.monotonic()
    refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=90.0)
    assert refit.report["rows_past_valid_cut"] == 0 and refit.report["rows_past_origin_cut"] == 0
    t = g.refit_tensors(refit=refit, shift_z=None)
    model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
    load_pretrained_fresh(model=model, path=pretrained_dir() / f"{month}_s{seed}.pt")
    model.to(device)
    t0 = time.monotonic()
    epochs, best = train(
        model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
        lr=c.FINE_TUNE_LR, fresh_lr=FRESH_LR, frozen=True, device=device,
    )  # fmt: skip
    day = f.day_of(month=month, cut=cut)
    log = {
        "name": f"{SPEC.tag}/{month}_d{day}_s{seed}", "epochs": epochs, "val_nll": best,
        "train_seconds": round(time.monotonic() - t0, 2), "seconds_to_tensors": round(t0 - started, 2), **refit.report,
    }  # fmt: skip
    del t
    return model, log


def fit_arm(*, device: str) -> None:
    """Fine-tune (or re-load) and forecast the arm's scored window at each cutoff."""
    clean = c.clean_stack()
    for month in ARGS.months:
        mr = k.build_month(month=month, clean=clean)
        test = data.build_test(month=month, standardiser=mr.base.standardiser)
        g = k.GpuMonth(mr=mr, device=device)
        cuts = k.cutoffs(month=month, cadence=CADENCE)
        for seed in ARGS.seeds:
            for i in list(range(len(cuts)))[: ARGS.limit]:
                cut = cuts[i]
                day = f.day_of(month=month, cut=cut)
                out = W / "fits" / SPEC.tag / f"{month}_d{day}_s{seed}.parquet"
                posthoc = SPEC.models_from is not None
                mfile = models_dir() / f"{month}_d{day}_s{seed}.pt"
                if (out.exists() or out.with_suffix(".empty").exists()) and (
                    posthoc or mfile.exists()
                ):
                    continue
                started = time.monotonic()
                if mfile.exists():
                    model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
                    model.load_state_dict(torch.load(mfile, map_location=device))
                    model.eval()
                    log: dict = {"name": f"{SPEC.tag}/{month}_d{day}_s{seed}", "reused": str(mfile)}
                else:
                    assert not posthoc, f"saved model missing: {mfile}"
                    model, log = fine_tune(g=g, month=month, cut=cut, seed=seed, device=device)
                    atomic_save(obj=model.state_dict(), target=mfile)
                set_context(model=model, month=month, day=day, seed=seed, mr=mr)
                lo, hi = k.window_of(month=month, cadence=CADENCE, index=i)
                assert lo >= c.us(t=cut), "a scored window starts before its refit cutoff"
                assert (cut - c.month_start(month=month)).days == day
                frame = f.window_frame(
                    model=model, test=test, mr=mr, lo=lo, hi=hi, flag=False, device=device
                )
                if frame is None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.with_suffix(".empty").touch()
                else:
                    assert frame["power_fcst_init_time"].min() >= cut, (
                        "a forecast origin precedes the refit cut"
                    )
                    f.atomic_frame(frame=frame, target=out)
                    ex = frame.row(0, named=True)
                    print(
                        f"  example row: origin {ex['power_fcst_init_time']} valid {ex['valid_time']} cut {cut} (newest knot fitted at the cut)",
                        flush=True,
                    )
                log["seconds"] = round(time.monotonic() - started, 2)
                c.write_json(obj=log, path=W / "logs" / SPEC.tag / f"{month}_d{day}_s{seed}.json")
                print(
                    f"{log['name']}: {log.get('epochs', 'reused')} epochs, val NLL {log.get('val_nll')}, {log['seconds']} s",
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
            ids = pool.group_table()["time_series_id"].to_list()
            assert sorted(frame["time_series_id"].unique().to_list()) == sorted(ids), (month, seed)
            n_origins = frame.group_by("time_series_id").agg(
                pl.col("power_fcst_init_time").n_unique()
            )
            assert n_origins["power_fcst_init_time"].n_unique() == 1, (month, seed, n_origins)
            lead = (frame["valid_time"] - frame["power_fcst_init_time"]).dt.total_hours()
            assert lead.max() <= 240 and lead.min() >= 0  # ty: ignore[unsupported-operator], (month, seed)
            f.atomic_frame(
                frame=frame, target=W / "seed_preds" / SPEC.tag / f"{month}_s{seed}.parquet"
            )
            print("wrote", SPEC.tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble()
    elif ARGS.command == "pretrain":
        pretrain(device="cuda")
    elif ARGS.command == "fit":
        fit_arm(device="cuda")
    if ARGS.command in {"fit", "pretrain"}:
        out = WORK / f"checks_{ARGS.command}_{ARGS.group}_s{'-'.join(map(str, ARGS.seeds))}.json"
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in CHECK.items() if a != "months"})

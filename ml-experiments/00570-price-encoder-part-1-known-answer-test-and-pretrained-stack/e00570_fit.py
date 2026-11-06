# ruff: noqa: E501, ANN001, ANN201, PT018
"""Experiment 570: pretrain and fine-tune the modular lag-free network with a learnt price encoder.

Rows, recency weights, early stopping, seeds, scoring windows, and the fine-tuning harness are
experiment 511's, through experiment 554's M0 fine-tuning loop (fresh mixer at 5e-4, stage `ft`), except
that the cadence is monthly: one fine-tune per fold, at the fold's first day. The price series is
truncated at the refit time for every training row; oracle scoring reads the untruncated copy.

Arm names are `FAMILY-KIND[-oracle]`:

- `M0`: experiment 554's pretrained weights, no price anywhere (new monthly-refit control).
- `A6-KIND`: price encoder with the oracle price in pretraining only; weather encoder, datetime encoder,
  and embedding go to a fresh price-free mixer (the M0 fine-tuning recipe).
- `A1-KIND`: price encoder kept in fine-tuning, present in a seeded 50% of fine-tuning rows, null token
  at scoring (`-oracle` scores the same fitted networks with the price present).
- `A3-KIND`: as A1 but the encoder is fresh at fine-tuning (pretrained from M0).

KIND is `res` (calendar-residualised price), `raw`, `p1` (shuffled day), `p2` (typical shape), `noise`,
or `const`.

Usage: python e00570_fit.py {pretrain|fit|assemble|show} --arm ARM [--months ...] [--seeds ...]
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
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00570_model as xm
import e00570_price as price
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00570-price-encoder-part-1-known-answer-test-and-pretrained-stack"
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
KIND_SERIES = {"res": "res", "raw": "real", "p1": "real", "p2": "typical", "noise": "noise", "const": "const"}
FAMILIES = ("M0", "A6", "A1", "A3")

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("pretrain", "fit", "assemble", "show", "known"))
parser.add_argument("--arm", required=True)
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0])
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--rules", nargs="+", default=["R1", "R2", "R3"])
parser.add_argument("--variants", nargs="+", default=["all", "train"])
parser.add_argument("--with-p1", action="store_true")
ARGS = parser.parse_args()
ORACLE = ARGS.arm.endswith("-oracle")
STORE_AS = ARGS.arm.removesuffix("-oracle")
FAMILY, _, KIND = STORE_AS.partition("-")
assert ARGS.command == "known" or FAMILY in FAMILIES and (FAMILY == "M0") == (KIND == ""), ARGS.arm
assert ARGS.command == "known" or FAMILY == "M0" or KIND in KIND_SERIES, ARGS.arm
USE_PRICE_IN_FT = FAMILY in {"A1", "A3"}
PRETRAIN_ARM = f"A6-{KIND}" if FAMILY in {"A6", "A1"} else "M0"
ARM = f.Arm(ARGS.arm, "m", 90.0, pretrained="M0")
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}
GRID_START, STEP = c.prep422.GRID_START_US, c.prep422.STEP_US
N_STEPS = 2600 * price.STEPS_PER_DAY
"""Grid length: 2019-09-01 for 2600 days (to 2026-11-14)."""
DEVICE = "cuda"
SERIES: dict[tuple[str, str], tuple[price.Bundle, np.ndarray]] = {}
FOLDS_UNFILLED: dict[str, int] = {}

_, COLUMNS, _ = data.non_lag_layout()
assert len(COLUMNS) == m.N_INPUTS
assert COLUMNS[m.WEATHER_COLS][0] == "mean_shortwave" and COLUMNS[m.WEATHER_COLS][-1] == "mean_wind_speed_10m"
assert COLUMNS[m.LEAD_COLS] == ["lead", "lead_sqrt"]


def fit_until_idx(*, month: str) -> int:
    """Return the grid index of the fold's refit time (first day of the month)."""
    return int((c.us(t=c.month_start(month=month)) - GRID_START) // STEP)


def bundle_for(*, month: str) -> tuple[price.Bundle, np.ndarray]:
    """Return the month's price bundle (truncated copy, then full copy) and the donor shift per grid half-hour."""
    key = (month, KIND)
    if key not in SERIES:
        until = fit_until_idx(month=month)
        full = price.build_series(kind=KIND_SERIES[KIND], grid_start_us=GRID_START, n_steps=N_STEPS, fit_until=until, seed=570)
        cut = full.copy()
        cut[:, until:] = 0.0
        cut[3, until:] = 0.0 if KIND not in {"noise", "const"} else 1.0
        if KIND in {"noise", "const"}:
            cut = full.copy()  # no information about the future in a noise or constant series
        bundle = price.Bundle(parts=[cut, full], device=DEVICE)
        shift = np.zeros(N_STEPS, dtype=np.int64)
        if KIND == "p1":
            day_shift = price.donor_shift(n_steps=N_STEPS, grid_start_us=GRID_START, seed=570 + int(month.replace("-", "")))
            shift = np.repeat(day_shift, price.STEPS_PER_DAY)
        SERIES[key] = (bundle, shift)
    return SERIES[key]


def extras(*, month: str, target_idx: np.ndarray, present_wanted: np.ndarray, scoring: bool) -> np.ndarray:
    """Return the two extra columns (bundle index, presence) for rows with target grid indices.

    Training rows index the truncated half of the bundle; scoring rows index the full half
    (offset by N_STEPS). The presence flag is the wanted flag times the mask at the (donor) half-hour.
    """
    if FAMILY == "M0":
        return np.zeros((len(target_idx), 2), dtype=np.float32)
    bundle, shift = bundle_for(month=month)
    donor = target_idx + shift[target_idx]
    assert (donor <= target_idx).all()
    idx = donor + (N_STEPS if scoring else 0)
    mask = bundle.chans[3].cpu().numpy()[idx] > 0
    present = (present_wanted & mask).astype(np.float32)
    return np.stack([idx.astype(np.float32), present], axis=1)


def new_model(*, outlier: bool, n_inputs: int, n_primaries: int, stage: str | None = None) -> torch.nn.Module:
    """Return an untrained network (the signature of `e00510_common.new_model`)."""
    assert not outlier
    assert n_inputs == m.N_INPUTS + xm.N_EXTRA, n_inputs
    stage = stage or "ft"
    if USE_PRICE_IN_FT or (stage == "pre" and FAMILY != "M0"):
        model = xm.PriceMdn(n_primaries=n_primaries, stage=stage)
    else:
        model = xm.PlainWithExtras(n_primaries=n_primaries, stage=stage)
    return model


def attach_bundle(*, model: torch.nn.Module, month: str) -> None:
    """Give a price network its month's bundle."""
    if isinstance(model, xm.PriceMdn) and model.bundle is None:
        model.bundle = bundle_for(month=month)[0]


CURRENT_MONTH: list[str] = [""]


def load_pretrained(*, model: torch.nn.Module, path: Path) -> None:
    """Load the arm's pretrained weights; the fine-tuning mixer starts fresh with seed 12345."""
    xm.load_pretrained(model=model, path=str(path), fresh_seed=FRESH_SEED, keep_price=FAMILY == "A1")
    attach_bundle(model=model, month=CURRENT_MONTH[0])


def train(
    *, model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, p: torch.Tensor,
    is_val: torch.Tensor, seed: int, lr: float, outlier: bool, prior: bool, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Experiment 554's loop for arm M0, plus the price encoder's regime-bank penalty."""
    assert not outlier
    attach_bundle(model=model, month=CURRENT_MONTH[0])
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    has_price = isinstance(model, xm.PriceMdn)
    fresh: list = []
    if lr == c.FINE_TUNE_LR:
        fresh = list(model.mixer_ft.parameters())  # ty: ignore[unresolved-attribute]
        if FAMILY == "A3":
            fresh += list(model.price.parameters())  # ty: ignore[unresolved-attribute]
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
            if has_price:
                loss = loss + model.price.aux_loss()  # ty: ignore[unresolved-attribute]
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
    """Return the pretrained weights an arm starts from (this experiment's M0 or A6-KIND pretraining)."""
    CURRENT_MONTH[0] = month
    if FAMILY == "A3" or PRETRAIN_ARM == "M0":
        return W / "pretrained" / "M0" / f"{month}_s{seed}.pt"
    return W / "pretrained" / PRETRAIN_ARM / f"{month}_s{seed}.pt"


def cutoffs(*, month: str, cadence: str) -> list[datetime]:
    """Cadence `m` is one refit per fold, at the first day of the month; other cadences are experiment 511's."""
    if cadence == "m":
        return [c.month_start(month=month)]
    return _cutoffs(month=month, cadence=cadence)


_cutoffs, _window_frame = k.cutoffs, f.window_frame


def window_frame(*args, **kwargs):
    """Attach the month's price bundle to a reloaded network before it forecasts."""
    model = kwargs.get("model")
    if model is not None:
        attach_bundle(model=model, month=CURRENT_MONTH[0])
    return _window_frame(*args, **kwargs)


f.window_frame = window_frame
c.new_model, c.load_pretrained, c.train = new_model, load_pretrained, train
k.cutoffs = cutoffs
f.pretrained_file = pretrained_file
f.W = W
f.model_file = lambda *, arm, month, day, seed, stress="": (
    W / "models" / STORE_AS / f"{month}_d{day}_s{seed}{stress}.pt"
)


def pretrain(*, device: str) -> None:
    """Pretrain on ERA5 and CAMS rows (experiment 448's rows) with the oracle price, for each month and seed lacking weights."""
    assert ARGS.arm == "M0" or FAMILY == "A6", "only M0 and A6-KIND are pretrained here (A1 shares A6's; A3 shares M0's)"
    clean = c.clean_stack()
    for month in ARGS.months:
        CURRENT_MONTH[0] = month
        todo = [s for s in ARGS.seeds if not (W / "pretrained" / ARGS.arm / f"{month}_s{s}.pt").exists()]
        if not todo:
            continue
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
        last = int(pre.target_idx.max())
        until = fit_until_idx(month=month)
        assert last < until, (last, until)
        # Standing rule: the price window of a pretraining row never reaches the refit time (truncated half).
        add = extras(month=month, target_idx=pre.target_idx.astype(np.int64), present_wanted=np.ones(len(pre.y), bool), scoring=False)
        print(
            f"{month}: pretraining rows {len(pre.x):,}; latest valid {datetime.fromtimestamp((last * STEP + GRID_START) / 1e6, UTC).isoformat()} (refit {refit_t.isoformat()}); price present in {add[:, 1].mean():.3f} of rows; example row 0: target index {int(pre.target_idx[0])}, price index {int(add[0, 0])}, present {int(add[0, 1])}",
            flush=True,
        )
        x2 = np.concatenate([pre.x, add], axis=1)
        z = (pre.y - rows.target_mean[pre.primary]) / rows.target_std[pre.primary]
        t = {
            "x": torch.from_numpy(x2).to(device), "z": torch.from_numpy(z.astype(np.float32)).to(device),
            "w": torch.ones(len(pre.y), device=device), "p": torch.from_numpy(pre.primary).to(device),
            "v": torch.from_numpy(pre.is_val).to(device),
        }  # fmt: skip
        for seed in todo:
            started = time.monotonic()
            torch.manual_seed(seed)
            model = new_model(outlier=False, n_inputs=m.N_INPUTS + xm.N_EXTRA, n_primaries=len(rows.target_mean), stage="pre").to(device)
            epochs, best = train(
                model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                lr=c.PRETRAIN_LR, outlier=False, prior=False, device=device,
            )  # fmt: skip
            target = W / "pretrained" / ARGS.arm / f"{month}_s{seed}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(f"  pretrain {ARGS.arm} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s", flush=True)
        del t, pre, rows
        torch.cuda.empty_cache()


def known(*, device: str) -> None:
    """Known-answer test: inject a price-driven generator into the six signed-MW Primaries and ask what the encoder recovers.

    For each month and rule, pretrain M0 (no price), and price networks fed the real price, a shuffled-day
    price (P1), and the typical shape (P2), on rows whose target has the generator subtracted. The recovered
    share is 1 - E[(g + d)^2] / E[g^2] on held-out rows, with d the price network's mean forecast minus the
    M0 network's, both in MW. For rule R1 a second run injects into the training rows only and scores the
    clean held-out rows (K4). Results are appended to `known_<month>.json`.
    """
    import e00570_inject as inject  # noqa: PLC0415
    from e00397_models import mixture_parts  # noqa: PLC0415,I001

    clean = c.clean_stack()
    caps = pl.DataFrame(__import__("e00470_build").site_map())
    for month in ARGS.months:
        out_path = WORK / f"known_{month}_s{ARGS.seeds[0]}.json"
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
        CURRENT_MONTH[0] = month
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
        until = fit_until_idx(month=month)
        raw_price = price.load_raw(grid_start_us=GRID_START, n_steps=N_STEPS)[0].copy()
        raw_price[until:] = np.nan
        groups = pl.read_parquet(next(W554.glob("scores/rows_2025-10.parquet")), columns=["primary", "group"]).unique()
        six = set(groups.filter(pl.col("group") == "signed_six")["primary"].to_list())
        cap = dict(zip(caps["primary"].to_list(), caps["capacity_mw"].to_list(), strict=True))
        tidx = pre.target_idx.astype(np.int64)
        prim = pre.primary.astype(np.int64)
        print(f"{month}: mean |y| per signed-MW Primary vs capacity (units check):", {int(p): (round(float(np.abs(pre.y[prim == p]).mean()), 2), cap[p]) for p in sorted(six)}, flush=True)
        for rule in ARGS.rules:
            g = np.zeros(len(pre.y))
            for p in six:
                series = inject.RULES[rule](price=raw_price, capacity_mw=0.02 * cap[p])
                sel = prim == p
                g[sel] = series[tidx[sel]]
            for variant in [v for v in ARGS.variants if rule == "R1" or v == "all"]:
                key = f"{rule}-{variant}"
                if key in results:
                    continue
                y_in = pre.y - g
                if variant == "train":
                    y_in = np.where(pre.is_val, pre.y, y_in)
                held = np.nonzero(pre.is_val)[0][:: max(1, pre.is_val.sum() // 200_000)]
                res = {"injected_share_of_rows_nonzero": float((g != 0).mean()), "mean_g_mw": float(g.mean())}
                means = {}
                kinds = (("none", "real", "p1", "p2") if ARGS.with_p1 else ("none", "real", "p2")) if variant == "all" else ("none", "real")
                for kind in kinds:
                    z = (y_in - rows.target_mean[pre.primary]) / rows.target_std[pre.primary]
                    zc = (pre.y - rows.target_mean[pre.primary]) / rows.target_std[pre.primary]
                    z_fit = z if variant == "all" else z
                    if kind == "none":
                        model = new_model_known(n_primaries=len(rows.target_mean), price_net=False).to(device)
                        add = np.zeros((len(pre.y), 2), dtype=np.float32)
                    else:
                        ser = price.build_series(kind=KIND_SERIES["raw" if kind == "real" else kind], grid_start_us=GRID_START, n_steps=N_STEPS, fit_until=until, seed=570)
                        cut = ser.copy()
                        cut[:, until:] = 0.0
                        cut[3, until:] = 0.0
                        shift = np.repeat(price.donor_shift(n_steps=N_STEPS, grid_start_us=GRID_START, seed=570 + int(month.replace("-", ""))), price.STEPS_PER_DAY) if kind == "p1" else np.zeros(N_STEPS, dtype=np.int64)
                        bundle = price.Bundle(parts=[cut, ser], device=device)
                        idx = tidx + shift[tidx]
                        present = (cut[3][idx] > 0).astype(np.float32)
                        add = np.stack([idx.astype(np.float32), present], axis=1)
                        model = new_model_known(n_primaries=len(rows.target_mean), price_net=True).to(device)
                        model.bundle = bundle
                    x2 = np.concatenate([pre.x, add], axis=1)
                    t = {"x": torch.from_numpy(x2).to(device), "z": torch.from_numpy(z_fit.astype(np.float32)).to(device),
                         "w": torch.ones(len(pre.y), device=device), "p": torch.from_numpy(pre.primary).to(device), "v": torch.from_numpy(pre.is_val).to(device)}  # fmt: skip
                    if variant == "train":  # validation rows are clean in the contamination run
                        t["z"][torch.from_numpy(pre.is_val).to(device)] = torch.from_numpy(zc.astype(np.float32)).to(device)[torch.from_numpy(pre.is_val).to(device)]
                    torch.manual_seed(ARGS.seeds[0])
                    epochs, best = train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=ARGS.seeds[0], lr=c.PRETRAIN_LR, outlier=False, prior=False, device=device)  # fmt: skip
                    with torch.no_grad():
                        hi = torch.from_numpy(held).to(device)
                        parts = mixture_parts(out=model(t["x"][hi], t["p"][hi]), components=m.COMPONENTS)
                        mu = (parts[0].exp() * parts[1]).sum(-1).cpu().numpy() * rows.target_std[pre.primary][held]
                    means[kind] = mu
                    res[f"val_nll_{kind}"] = best
                    print(f"  {month} {key} {kind}: {epochs} epochs, val NLL {best:.4f}", flush=True)
                    del model, t
                    torch.cuda.empty_cache()
                gh = g[held]
                on_six = np.isin(prim[held], list(six))
                for kind in kinds[1:]:
                    d = (means[kind] - means["none"])[on_six]
                    res[f"recovered_{kind}"] = float(1 - ((gh[on_six] + d) ** 2).mean() / (gh[on_six] ** 2).mean()) if variant == "all" else None
                    if rule == "R1" and variant == "all" and kind == "real":
                        pr = raw_price[tidx[held]][on_six]
                        dem = -d  # demand falls where the generator runs
                        bins = np.arange(0, 400, 5)
                        cap_mean = np.array([0.02 * cap[p] for p in prim[held][on_six]])
                        frac = dem / cap_mean
                        curve = [(float(b), float(np.nanmean(frac[(pr >= b) & (pr < b + 5)]))) for b in bins if ((pr >= b) & (pr < b + 5)).sum() > 50]
                        res["threshold_curve_price_frac_of_capacity"] = curve
                        high = frac[pr > 150]
                        res["gain_over_capacity_price_above_150"] = float(np.nanmean(high))
                        half = [b for b, v in curve if v >= 0.5]
                        res["learned_threshold_half_output"] = half[0] if half else None
                results[key] = res
                out_path.write_text(json.dumps(results, indent=1))
                print(f"  {month} {key}: " + json.dumps({k: v for k, v in res.items() if not k.startswith("threshold_curve")}), flush=True)
        del pre, rows
        torch.cuda.empty_cache()


def new_model_known(*, n_primaries: int, price_net: bool) -> torch.nn.Module:
    """Return a stage `pre` network with (`price_net`) or without the price encoder."""
    if price_net:
        return xm.PriceMdn(n_primaries=n_primaries, stage="pre")
    return xm.PlainWithExtras(n_primaries=n_primaries, stage="pre")


_build_month, _build_test = k.build_month, data.build_test


def build_month(*, month, clean):
    """Build the month's rows, assert each origin's weather run precedes it, and append the price columns."""
    CURRENT_MONTH[0] = month
    mr = _build_month(month=month, clean=clean)
    assert (mr.valid_us > mr.origin_us).all(), f"{month}: a training target is not after its origin"
    runs = np.unique(mr.origin_us)
    delay = int(common.PUBLICATION_DELAY.total_seconds() * 1e6)
    miss = [
        r for r in runs
        if not (prep.SUMMARY_DIR / f"init={datetime.fromtimestamp((r - delay) / 1e6, UTC):%Y-%m-%d}.parquet").exists()
    ]  # fmt: skip
    assert not miss, f"{month}: {len(miss)} training origins have no weather-run summary file {common.PUBLICATION_DELAY} earlier"
    rng = np.random.default_rng([570, int(month.replace("-", ""))])
    wanted = rng.random(len(mr.y)) < PRESENT_FRACTION if USE_PRICE_IN_FT else np.ones(len(mr.y), bool)
    add = extras(month=month, target_idx=mr.target_idx.astype(np.int64), present_wanted=wanted, scoring=False)
    i = int(np.argmax(mr.valid_us > mr.origin_us + 24 * 3_600_000_000))
    CHECK["train_rows"] += len(mr.y)
    CHECK["months"][month] = {
        "train_rows": len(mr.y), "train_origins": len(runs), "price_present_share": float(add[:, 1].mean()),
        "train_example": f"origin {datetime.fromtimestamp(mr.origin_us[i] / 1e6, UTC)}, weather run {datetime.fromtimestamp(mr.origin_us[i] / 1e6, UTC) - common.PUBLICATION_DELAY}, valid {datetime.fromtimestamp(mr.valid_us[i] / 1e6, UTC)}, price index {int(add[i, 0])} (truncated half, so no price from the refit time on), present {int(add[i, 1])}",
    }  # fmt: skip
    print(month, "train checks passed |", CHECK["months"][month]["train_example"], flush=True)
    return dataclasses.replace(mr, x=np.concatenate([mr.x, add], axis=1))


def build_test(*, month, standardiser):
    """Build the month's test rows, assert nwp_init_time <= power_fcst_init_time, and append the price columns."""
    CURRENT_MONTH[0] = month
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
    wanted = np.full(len(test.x), ORACLE, dtype=bool)
    add = extras(month=month, target_idx=test.target_idx.astype(np.int64), present_wanted=wanted, scoring=True)
    CHECK["test_rows"] += len(test.x)
    CHECK["months"][month]["test_example"] = (
        f"origin {datetime.fromtimestamp(test.origin[0] / 1e6, UTC)}, weather run {datetime.fromtimestamp(test.origin[0] / 1e6, UTC) - pub}, valid {datetime.fromtimestamp(test.valid[0] / 1e6, UTC)}, "
        f"price index {int(add[0, 0])} (full half), present {int(add[0, 1])} (oracle scoring {ORACLE}; nwp_init_time <= power_fcst_init_time asserted for all rows of every test origin)"
    )
    print(month, "test checks passed |", CHECK["months"][month]["test_example"], flush=True)
    return dataclasses.replace(test, x=np.concatenate([test.x, add], axis=1))


k.build_month, data.build_test = build_month, build_test


def assemble() -> None:
    """Concatenate each cutoff's window forecast into one table per month and seed."""
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence="m")
        for seed in ARGS.seeds:
            parts = [W / "fits" / ARM.tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet" for cut in cuts]
            assert all(p.exists() or p.with_suffix(".empty").exists() for p in parts), (month, seed)
            frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
            f.atomic_frame(frame=frame, target=W / "seed_preds" / ARM.tag / f"{month}_s{seed}.parquet")
            print("wrote", ARM.tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble()
    elif ARGS.command == "show":
        model = new_model(outlier=False, n_inputs=m.N_INPUTS + xm.N_EXTRA, n_primaries=16)
        print(ARGS.arm, "parameters", sum(q.numel() for q in model.parameters()))
    elif ARGS.command == "pretrain":
        pretrain(device=DEVICE)
    elif ARGS.command == "known":
        known(device=DEVICE)
    else:
        if ORACLE:
            missing = [(mo, sd) for mo in ARGS.months for sd in ARGS.seeds if not list((W / "models" / STORE_AS).glob(f"{mo}_d*_s{sd}.pt"))]
            assert not missing, f"{ARGS.arm} fits nothing; run arm {STORE_AS} first: {missing}"
        started = time.monotonic()
        f.fit_arm(arm=ARM, months=ARGS.months, seeds=ARGS.seeds, stress=False, limit=ARGS.limit, device=DEVICE)
        out = WORK / f"checks_{ARM.tag}_s{'-'.join(map(str, ARGS.seeds))}.json"
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in CHECK.items() if a != "months"}, f"{time.monotonic() - started:.0f} s")

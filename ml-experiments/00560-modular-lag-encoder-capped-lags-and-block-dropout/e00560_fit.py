# ruff: noqa: E501, ANN001, ANN003, ANN201, D102, PT018, RUF028
"""Experiment 560: pretrain, fine-tune, and forecast the modular networks with a lag encoder.

Rows, cutoffs, recency weights, early stopping, seeds, scoring windows, the daily refit, and the
network's weather and datetime encoders and mixer are experiment 545's M0 (daily refit, 90-day
half-life). New here: the capped lags and block dropout of experiment 558 (`e00558_lags`), a lag
encoder (`e00560_model`), pretraining with those lags, and the expected-input table (`e00560_data`).
Every fit is also scored with the lags removed and with the lags aged by 1, 2, and 3 extra days
(inference only); those forecasts go to sibling folders `<arm>-removed` and `<arm>-aged<k>`.

Usage: python e00560_fit.py {pretrain|fit|assemble} --arm v1|v2|v3|v4|b_shared|b_unshared [--stress]
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
import e00545_model as m545  # ty: ignore[unresolved-import]
import e00558_lags as lags  # ty: ignore[unresolved-import]
import e00560_data as d
import e00560_model as m
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = ML_EXPERIMENTS_DATA_DIR / "00560-modular-lag-encoder-capped-lags-and-block-dropout"
W = WORK / "work"
MIXER_LR = 5e-4
"""Learning rate of the fresh fine-tuning mixer, as in experiment 545."""
AGED_DAYS = (1, 2, 3)
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("pretrain", "fit", "assemble"))
parser.add_argument("--arm", choices=m.MODES, required=True)
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
MODE = ARGS.arm
ARM = f.Arm(f"LAG-{MODE}", "d", 90.0)
SIBLINGS = ["removed", *[f"aged{n}" for n in AGED_DAYS]]
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}
CTX: dict = {"on": False, "extra": {}}

_, COLUMNS, _ = data.non_lag_layout()
assert len(COLUMNS) == m.N_BASE
assert (
    COLUMNS[m545.WEATHER_COLS][0] == "mean_shortwave"
    and COLUMNS[m545.DATETIME_COLS][0] == "cos_zenith"
)


def new_model(
    *, outlier: bool, n_inputs: int, n_primaries: int, stage: str = "ft"
) -> torch.nn.Module:
    """Return an untrained network with the expected-input table of the current month attached."""
    assert not outlier
    assert n_inputs == m.N_INPUTS
    model = m.LagModularMdn(n_primaries=n_primaries, mode=MODE, stage=stage)
    model.table = CTX.get(f"table_{stage}")
    return model


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
            hidden=m545.MIXER_HIDDEN, layers=2, n_out=3 * m545.COMPONENTS, dropout=m545.DROPOUT,
        )  # fmt: skip
        base.init_head(head=fresh, components=m545.COMPONENTS)
    model.mixer_ft.load_state_dict(fresh.state_dict())  # ty: ignore[unresolved-attribute]


def row_nll(*, model, x, p, z) -> torch.Tensor:
    """Return the per-row mixture negative log-likelihood."""
    return base.mixture_nll(out=model(x, p), z=z, components=m545.COMPONENTS)


def train(
    *, model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, p: torch.Tensor,
    is_val: torch.Tensor, seed: int, lr: float, outlier: bool, prior: bool, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Experiment 510's loop with experiment 545's separate fresh-mixer learning rate."""
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
            nll = row_nll(model=model, x=x[b], p=p[b], z=z[b])
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
    return epochs, best


c.new_model, c.load_pretrained, c.train = new_model, load_pretrained, train
f.pretrained_file = lambda *, arm, month, seed: W / "pretrained" / arm.tag / f"{month}_s{seed}.pt"
f.W = W
f.model_file = lambda *, arm, month, day, seed, stress="": (
    W / "models" / arm.tag / f"{month}_d{day}_s{seed}{stress}.pt"
)


# --- block dropout and the shifted-lag stress run -----------------------------------------------


def apply_dropout(*, x: torch.Tensor, origin_idx: np.ndarray, seed: int) -> dict:
    """Mask lags in place per experiment 558: whole origin days, all lags or the day lag only."""
    mask_all, mask_day = lags.block_dropout_masks(origin_idx=origin_idx, seed=seed)
    ma = torch.from_numpy(mask_all).to(x.device)
    md = torch.from_numpy(mask_day).to(x.device)
    for use, c0, age in ((ma | md, 37, d.AGE_MAX_DAY), (ma, 40, d.AGE_MAX_WEEK)):
        x[use, c0], x[use, c0 + 1], x[use, c0 + 2] = 0.0, 1.0, float(age)
    return {
        "share_all_lags_masked": float(mask_all.mean()),
        "share_day_lag_only_masked": float(mask_day.mean()),
        "share_day_lag_missing": float((x[:, 38] > 0.5).float().mean()),
        "share_week_lag_missing": float((x[:, 41] > 0.5).float().mean()),
    }


def lag_delta(*, x_cols: dict, valid_us: np.ndarray, prim: np.ndarray, ctx: dict) -> np.ndarray:
    """Return the standardised shift to add to the day and week lag values (rows x 2)."""
    shift, sign, block = ctx["shift"], ctx["sign"], ctx["block"]
    out = np.zeros((len(valid_us), 2), dtype=np.float32)
    for j, (age_col, miss_col) in enumerate(((39, 38), (42, 41))):
        t = valid_us - (x_cols[age_col].astype(np.float64) * lags.LEAD_SCALE_HOURS * 3.6e9).astype(
            np.int64
        )
        inblock = (t >= block[0]) & (t < block[1]) & (x_cols[miss_col] < 0.5)
        out[:, j] = np.where(inblock, sign * shift[prim] / CTX["stats"][j, 1], 0.0)
    return out


def shifted(*, x: np.ndarray, valid_us: np.ndarray, prim: np.ndarray) -> np.ndarray:
    """Return a copy of `x` with the stress shift added to the in-block lag values."""
    cols = {n: x[:, n] for n in (38, 39, 41, 42)}
    delta = lag_delta(x_cols=cols, valid_us=valid_us, prim=prim, ctx=CTX)
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
                x_cols=cols, valid_us=self.mr.valid_us[idx], prim=self.mr.primary[idx], ctx=CTX
            )
            for j, col in enumerate((37, 40)):
                v = t["x"][:, col] + torch.from_numpy(delta[:, j]).to(t["x"].device)
                t["x"][:, col] = v.clamp(-prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
            CTX["n_shifted"] = int((delta != 0).any(axis=1).sum())
        origin_idx = (self.mr.origin_us[idx] - c.prep422.GRID_START_US) // c.prep422.STEP_US
        CTX["dropout"] = apply_dropout(x=t["x"], origin_idx=origin_idx, seed=CTX["seed"])
        return t


k.GpuMonth = LagGpuMonth


# --- rows ----------------------------------------------------------------------------------------

_build_month, _build_test = k.build_month, data.build_test


def widen(*, x37: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Return `x37` with the 8 appended columns, filled chunk by chunk to bound peak memory."""
    out = np.empty((len(x37), m.N_INPUTS), dtype=np.float32)
    for a in range(0, len(x37), 1_000_000):
        out[a : a + 1_000_000, : m.N_BASE] = x37[a : a + 1_000_000]
    out[:, m.N_BASE :] = block
    return out


def block_of(*, clean, primary, target_idx, origin_idx, lead, extra_days=0) -> np.ndarray:
    """Return the 8 lag columns for rows, in chunks."""
    parts = []
    for a in range(0, len(primary), 2_000_000):
        s = slice(a, a + 2_000_000)
        parts.append(
            d.lag_block(
                clean=clean,
                primary=primary[s],
                target_idx=target_idx[s],
                origin_idx=origin_idx[s],
                lead=lead[s],
                stats=CTX["stats"],
                extra_days=extra_days,
            )  # fmt: skip
        )
    return np.concatenate(parts)


def attach_tables(*, standardiser) -> None:
    """Build and upload the standardised expected-input tables of the fine-tuning and pretraining sources."""
    for src in ("ft", "pre"):
        cols, origin = d.raw_table(source=src, cache=W / "expected")
        CTX[f"origin_{src}"] = np.asarray(origin)
        CTX[f"table_{src}"] = torch.from_numpy(
            d.standardised_table(cols=cols, standardiser=standardiser)
        ).to("cuda")
        del cols


def build_month(*, month, clean):
    """Build the month's rows with the lag columns; assert weather runs, lag times, and print examples."""
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
    CTX["stats"] = d.lag_stats(rows=mr.base, clean=clean)
    CTX["clean"], CTX["month"] = clean, month
    origin_idx = (mr.origin_us - c.prep422.GRID_START_US) // c.prep422.STEP_US
    blk = block_of(
        clean=clean,
        primary=mr.primary,
        target_idx=mr.target_idx,
        origin_idx=origin_idx,
        lead=mr.lead,
    )
    mr.x = widen(x37=mr.x, block=blk)
    attach_tables(standardiser=mr.base.standardiser)
    ex = d.check_expected(
        origin_tab=CTX["origin_ft"],
        primary=mr.primary,
        x=mr.x,
        origin_us=mr.origin_us,
        label=f"{month} train",
    )
    ages = np.asarray(mr.x[:200_000, 39]) * lags.LEAD_SCALE_HOURS
    assert (ages >= 24.0).all(), "a day lag younger than 24 h"
    CHECK["train_rows"] += len(mr.y)
    CHECK["months"][month] = {
        "train_rows": len(mr.y),
        "train_origins": len(runs),
        "expected_example_train": ex,
    }
    print(month, "train checks passed |", ex, flush=True)
    return mr


def build_test(*, month, standardiser):
    """Build the month's test rows with the lag columns, the weather-run assertions, and the inference-only variants."""
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
    clean = CTX["clean"]
    variants = {}
    for extra in (0, *AGED_DAYS):
        blk = block_of(
            clean=clean,
            primary=test.primary,
            target_idx=test.target_idx,
            origin_idx=test.origin_idx,
            lead=test.lead,
            extra_days=extra,
        )
        variants[extra] = widen(x37=test.x, block=blk)
    ex = d.check_expected(
        origin_tab=CTX["origin_ft"],
        primary=test.primary,
        x=variants[0],
        origin_us=test.origin,
        label=f"{month} test",
    )
    aged_ok = all(
        ((variants[0][:, 43] - variants[e][:, 43]) >= 48 * e).all()
        and ((variants[0][:, 43] - variants[e][:, 43]) % 48 == 0).all()
        for e in AGED_DAYS
    )
    assert aged_ok, "an aged day lag is not a multiple of 24 h older"
    gone = variants[0].copy()
    d.absent(x=gone)
    CTX["variants"] = {"removed": gone, **{f"aged{e}": variants[e] for e in AGED_DAYS}}
    CHECK["test_rows"] += len(test.x)
    CHECK["months"][month]["expected_example_test"] = ex
    CHECK["months"][month]["weather_example"] = (
        f"target {datetime.fromtimestamp(test.valid[0] / 1e6, UTC)}, origin {datetime.fromtimestamp(test.origin[0] / 1e6, UTC)}, "
        f"weather run {datetime.fromtimestamp(test.origin[0] / 1e6, UTC) - pub} (nwp_init_time <= power_fcst_init_time asserted for all rows of every test origin)"
    )
    i = int(np.argmax(variants[0][:, 38] < 0.5))
    CHECK["months"][month]["lag_example"] = (
        f"row {i}: origin {datetime.fromtimestamp(test.origin[i] / 1e6, UTC)}, target {datetime.fromtimestamp(test.valid[i] / 1e6, UTC)}, "
        f"day lag at {datetime.fromtimestamp((c.prep422.GRID_START_US + variants[0][i, 43] * c.prep422.STEP_US) / 1e6, UTC)} (>= 24 h before origin asserted for all rows), "
        f"week lag at {datetime.fromtimestamp((c.prep422.GRID_START_US + variants[0][i, 44] * c.prep422.STEP_US) / 1e6, UTC)}"
    )
    print(
        month,
        "test checks passed |",
        CHECK["months"][month]["lag_example"],
        "|",
        CHECK["months"][month]["weather_example"],
        "|",
        ex,
        flush=True,
    )
    return dataclasses.replace(test, x=variants[0])


k.build_month, data.build_test = build_month, build_test
_fine_tune, _window_frame = f.fine_tune, f.window_frame


def fine_tune(**kw):
    """Fine-tune one cutoff, with the stress context set for the shifted-lag run."""
    CTX.update(
        on=kw.get("shift") is not None,
        shift=kw.get("shift"),
        sign=kw.get("sign"),
        block=kw.get("block"),
        seed=kw["seed"],
        cut=kw["cut"],
    )
    model, log = _fine_tune(**kw)
    log.update(CTX.get("dropout", {}))
    if CTX["on"]:
        log["rows_with_shifted_lag"] = CTX.get("n_shifted")
    return model, log


def window_frame(*, model, test, mr, lo, hi, flag, device):
    """Forecast the window; outside the stress run also forecast with the lags removed and aged."""
    model.table = CTX["table_ft"]
    x = test.x
    if CTX["on"]:
        x = shifted(x=test.x, valid_us=test.valid, prim=test.primary)
        test = dataclasses.replace(test, x=x)
    frame = _window_frame(model=model, test=test, mr=mr, lo=lo, hi=hi, flag=flag, device=device)
    CTX["extra"] = {}
    if not CTX["on"] and frame is not None:
        for name, xv in CTX["variants"].items():
            CTX["extra"][name] = _window_frame(
                model=model,
                test=dataclasses.replace(test, x=xv),
                mr=mr,
                lo=lo,
                hi=hi,
                flag=flag,
                device=device,
            )
    return frame


_atomic_frame = f.atomic_frame


def atomic_frame(*, frame, target):
    """Write the frame, and its removed/aged twins beside it."""
    _atomic_frame(frame=frame, target=target)
    if CTX.get("extra") and target.parent.name == ARM.tag:
        for name, fr in CTX["extra"].items():
            _atomic_frame(frame=fr, target=target.parent.parent / f"{ARM.tag}-{name}" / target.name)


f.fine_tune, f.window_frame, f.atomic_frame = fine_tune, window_frame, atomic_frame


# --- pretraining ---------------------------------------------------------------------------------


def pretrain(*, device: str) -> None:
    """Pretrain on ERA5 and CAMS rows with the capped lags and block dropout, per month and seed."""
    clean = c.clean_stack()
    for month in ARGS.months:
        todo = [
            s for s in ARGS.seeds if not (W / "pretrained" / ARM.tag / f"{month}_s{s}.pt").exists()
        ]
        if not todo:
            continue
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=clean)
        CTX["stats"] = d.lag_stats(rows=rows, clean=clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
        assert pre.x.shape[1] == m.N_BASE
        latest = datetime.fromtimestamp(
            (pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC
        )
        assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
        print(
            f"{month}: pretraining rows {len(pre.x):,}; latest valid {latest.isoformat()} (refit {refit_t.isoformat()})",
            flush=True,
        )
        blk = block_of(
            clean=clean,
            primary=pre.primary,
            target_idx=pre.target_idx,
            origin_idx=pre.origin_idx,
            lead=pre.lead,
        )
        cols, origin = d.raw_table(source="pre", cache=W / "expected")
        CTX["table_pre"] = torch.from_numpy(
            d.standardised_table(cols=cols, standardiser=rows.standardiser)
        ).to(device)
        del cols
        pre_x_gpu = torch.empty((len(pre.x), m.N_INPUTS), dtype=torch.float32, device=device)
        for a in range(0, len(pre.x), 1_000_000):
            pre_x_gpu[a : a + 1_000_000, : m.N_BASE] = torch.from_numpy(
                pre.x[a : a + 1_000_000]
            ).to(device)
        m_chk = 2_000_000
        stub = np.zeros((min(m_chk, len(blk)), m.N_INPUTS), np.float32)
        stub[:, m.N_BASE :] = blk[:m_chk]
        ex = d.check_expected(
            origin_tab=np.asarray(origin),
            primary=pre.primary[:m_chk],
            x=stub,
            origin_us=c.prep422.GRID_START_US + pre.origin_idx[:m_chk] * c.prep422.STEP_US,
            label=f"{month} pretrain",
        )
        del stub
        print(ex, flush=True)
        z = (pre.y - rows.target_mean[pre.primary]) / rows.target_std[pre.primary]
        t = {
            "z": torch.from_numpy(z.astype(np.float32)).to(device), "w": torch.ones(len(pre.y), device=device),
            "p": torch.from_numpy(pre.primary).to(device), "v": torch.from_numpy(pre.is_val).to(device),
        }  # fmt: skip
        n_primaries = len(rows.target_mean)
        origin_idx = pre.origin_idx
        del pre, rows
        for seed in todo:
            started = time.monotonic()
            for a in range(0, len(blk), 1_000_000):
                pre_x_gpu[a : a + 1_000_000, m.N_BASE :] = torch.from_numpy(
                    blk[a : a + 1_000_000]
                ).to(device)
            share = apply_dropout(x=pre_x_gpu, origin_idx=origin_idx, seed=seed)
            torch.manual_seed(seed)
            model = new_model(
                outlier=False, n_inputs=m.N_INPUTS, n_primaries=n_primaries, stage="pre"
            ).to(device)
            epochs, best = train(
                model=model,
                x=pre_x_gpu,
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
            f_tmp = target.with_suffix(".tmp")
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f_tmp)
            f_tmp.rename(target)
            print(
                f"  pretrain {ARM.tag} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, dropout {share}",
                flush=True,
            )
        del t, pre_x_gpu, blk
        CTX.pop("table_pre", None)
        torch.cuda.empty_cache()


def assemble() -> None:
    """Concatenate each cutoff's window forecast into one table per month, seed, and scoring variant."""
    for tag in (ARM.tag, *[f"{ARM.tag}-{s}" for s in SIBLINGS]):
        for month in c.MONTHS:
            cuts = k.cutoffs(month=month, cadence="d")
            for seed in (0, 1):
                parts = [
                    W / "fits" / tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                    for cut in cuts
                ]
                twin = lambda p: (p.parent.parent / ARM.tag / p.name).with_suffix(".empty")  # noqa: E731
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

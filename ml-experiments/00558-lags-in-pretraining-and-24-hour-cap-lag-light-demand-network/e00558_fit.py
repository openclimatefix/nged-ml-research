# ruff: noqa: ANN001, ANN003, ANN201, D101, D102, D103, E501, E731
"""Experiment 558, arm R-DW-24 (experiment 544's fit script with the 24 h cap on the lags).

The newest lag is dropped. The day and week lags are the latest same-half-hour and same-weekday
values observed at least 24 h before the origin (e00558_lags). In every training origin day, block
dropout masks all lags in 30% of days, and only the day lag in 15% of those; masked inputs are
value 0, missing flag 1, age at the training maximum. Lags are added at fine-tuning only (539).
Scored with the readings present (R-DW-24) and with every row's readings absent (R-DW-24-absent).

Usage: python e00558_fit.py {fit|assemble} --arm R-DW-24 [--stress]
"""

import argparse
import dataclasses
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import e00558_lags as lags  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

WORK = Path(
    os.environ["DATA_PATH_INTERNAL"]
    + "/ml-experiments/00558-lags-in-pretraining-and-24-hour-cap-lag-light-demand-network"
)
W = WORK / "work"
EXTRA_IDX = {"R-DW-24": [1, 2, 4, 5, 7, 8]}
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("fit", "assemble"))
parser.add_argument("--arm", choices=list(EXTRA_IDX), required=True)
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
ARM = f.Arm(ARGS.arm, "d", 90.0)

NAMES, MASK, LAYOUT = prep.feature_names(scheme="summary")
LAG = LAYOUT.columns_of(groups=["lags"])
assert len(LAG) == 9
EXTRA = [LAG[i] for i in EXTRA_IDX[ARGS.arm]]
KEEP0, COLS0, MASK0 = data.non_lag_layout()
N0 = len(KEEP0)
assert N0 == 37
NEW_NAMES = [NAMES[i] for i in EXTRA]
POS = {n: N0 + j for j, n in enumerate(NEW_NAMES)}
VALUE = [n for n in NEW_NAMES if not n.endswith(("_missing", "_age"))]
CTX: dict = {"on": False}
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}


def layout():
    keep = list(KEEP0) + EXTRA
    return keep, [*COLS0, *NEW_NAMES], MASK[np.array(keep)]


data.non_lag_layout = layout


def load_pretrained(*, model, path):
    """Load the 37-input pretrained weights; new inputs get zero first-layer weights."""
    state = torch.load(path, map_location="cpu")
    if any(key.startswith("net.net.") or key == "net.embedding.weight" for key in state):
        state = {key[4:]: v for key, v in state.items()}
    w = state["net.0.weight"]
    assert w.shape[1] == N0 + 8, w.shape
    new = torch.zeros(w.shape[0], w.shape[1] + len(EXTRA), dtype=w.dtype)
    new[:, :N0] = w[:, :N0]
    new[:, N0 + len(EXTRA) :] = w[:, N0:]
    state["net.0.weight"] = new
    model.load_state_dict(state)


c.load_pretrained = load_pretrained
f.W = W
f.model_file = lambda *, arm, month, day, seed, stress="": (
    W / "models" / arm.tag / f"{month}_d{day}_s{seed}{stress}.pt"
)


def lag_time_us(*, x, valid_us, name):
    age = name.replace("_missing", "").replace("lag_", "")
    col = POS[f"lag_{age}_age"]
    return valid_us - (x[:, col].astype(np.float64) * prep.LEAD_SCALE_HOURS * 3.6e9).astype(
        np.int64
    )


def check_rows(*, x, valid_us, origin_us, lead, label):
    """Check row by row that the ages, lag times and weather runs precede the origin.

    Ages equal the independent recomputation, every lag was observed at or before the origin minus
    the data delay, and the weather run is at or before the origin.
    """
    cap_day, cap_week = lags.capped_lag_ages_hours(lead_hours=lead.astype(np.float64))
    ages = np.stack([np.zeros_like(cap_day), cap_day, cap_week], axis=1)
    delay_us = int(lags.CAP_STEPS * 30 * 60 * 1e6)
    example = None
    for j, tag in enumerate(("origin", "day", "week")):
        nm = f"lag_{tag}_age"
        if nm not in POS:
            continue
        stored = x[:, POS[nm]].astype(np.float64) * prep.LEAD_SCALE_HOURS
        assert np.allclose(stored, ages[:, j], atol=1e-3), (
            f"{label} {tag} age differs from independent recomputation"
        )
        t_lag = valid_us - (ages[:, j] * 3.6e9).astype(np.int64)
        assert (t_lag <= origin_us - delay_us).all(), (
            f"{label} {tag} lag observed after origin minus 24 h"
        )
        if example is None:
            i = int(np.argmax(ages[:, j] == ages[:, j].min()))
            example = (
                f"{tag} lag of row {i}: origin {datetime.fromtimestamp(origin_us[i] / 1e6, UTC)}, target {datetime.fromtimestamp(valid_us[i] / 1e6, UTC)}, "
                f"lag observed {datetime.fromtimestamp(t_lag[i] / 1e6, UTC)} (<= origin - 24 h)"
            )
    assert (valid_us > origin_us).all(), f"{label}: target not after origin"
    return example


def put_capped(*, x, clean, primary, target_idx, origin_idx, lead, standardiser):
    """Overwrite the day and week lag columns of x with the capped lags (standardised like 539's)."""
    vd, vw = lags.capped_lag_values(
        clean=clean, primary=primary, target_idx=target_idx, origin_idx=origin_idx
    )
    ad, aw = lags.capped_lag_ages_hours(lead_hours=lead.astype(np.float64))
    for slot, v, a in (("lag_day", vd, ad), ("lag_week", vw, aw)):
        miss = ~np.isfinite(v)
        z = (v - standardiser.mean[POS[slot]]) / standardiser.std[POS[slot]]
        x[:, POS[slot]] = np.clip(np.where(miss, 0.0, z), -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
        x[:, POS[f"{slot}_missing"]] = miss.astype(np.float32)
        x[:, POS[f"{slot}_age"]] = (a / prep.LEAD_SCALE_HOURS).astype(np.float32)
    return {
        "day_missing_share": float(np.isnan(vd).mean()),
        "week_missing_share": float(np.isnan(vw).mean()),
    }


def verify_capped_values(*, x, clean, primary, target_idx, origin_idx, standardiser, label):
    """Recompute 2,000 random rows' capped lags with an explicit loop and compare with the columns."""
    rng = np.random.default_rng(2)
    sel = rng.choice(len(primary), min(2000, len(primary)), replace=False)
    out = ""
    for slot, period in (("lag_day", 48), ("lag_week", 336)):
        for i in sel:
            k_ = 0
            while origin_idx[i] - (target_idx[i] - period * k_) < 48:
                k_ += 1
            j = target_idx[i] - period * k_
            raw = clean[primary[i], j] if j >= 0 else np.nan
            if np.isfinite(raw):
                z = (raw - standardiser.mean[POS[slot]]) / standardiser.std[POS[slot]]
                want = float(np.clip(z, -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD))
                assert abs(x[i, POS[slot]] - want) < 1e-4, f"{label} {slot} differs at row {i}"
                assert x[i, POS[f"{slot}_missing"]] == 0.0
            else:
                assert x[i, POS[f"{slot}_missing"]] == 1.0, f"{label} {slot} flag at row {i}"
        i = int(sel[0])
        out += f"{slot}: lag index {j} for target {target_idx[i]}, origin {origin_idx[i]}; "
    return f"{label}: capped lags match an independent loop on {len(sel)} rows; {out}"


_build_month, _build_test = k.build_month, data.build_test


def build_month(*, month, clean):
    mr = _build_month(month=month, clean=clean)
    CTX["clean"] = clean
    origin_idx = (mr.origin_us - c.prep422.GRID_START_US) // c.prep422.STEP_US
    share = put_capped(
        x=mr.x,
        clean=clean,
        primary=mr.primary,
        target_idx=mr.target_idx,
        origin_idx=origin_idx,
        lead=mr.lead,
        standardiser=mr.base.standardiser,
    )
    ex = check_rows(
        x=mr.x, valid_us=mr.valid_us, origin_us=mr.origin_us, lead=mr.lead, label=f"{month} train"
    )
    ex += " | " + verify_capped_values(
        x=mr.x,
        clean=clean,
        primary=mr.primary,
        target_idx=mr.target_idx,
        origin_idx=origin_idx,
        standardiser=mr.base.standardiser,
        label=f"{month} train",
    )
    for key in [k_ for k_ in CTX if isinstance(k_, tuple)]:
        del CTX[key]
    CTX["age_max"] = {n: float(mr.x[:, POS[n]].max()) for n in POS if n.endswith("_age")}
    CTX["origin_idx_mr"], CTX["month"] = origin_idx, month
    runs = np.unique(mr.origin_us)
    miss = [
        r
        for r in runs
        if not (
            prep.SUMMARY_DIR
            / f"init={datetime.fromtimestamp((r - int(common.PUBLICATION_DELAY.total_seconds() * 1e6)) / 1e6, UTC):%Y-%m-%d}.parquet"
        ).exists()
    ]
    assert not miss, (
        f"{month}: {len(miss)} training origins have no weather-run summary file 9 h earlier"
    )
    CHECK["train_rows"] += len(mr.y)
    CHECK["months"][month] = {
        "train_rows": len(mr.y),
        "train_origins": len(runs),
        "lag_example_train": ex,
        "capped_lag_missing_shares": share,
    }
    print(month, "train checks passed;", ex, flush=True)
    return mr


def build_test(*, month, standardiser):
    test = _build_test(month=month, standardiser=standardiser)
    put_capped(
        x=test.x,
        clean=CTX["clean"],
        primary=test.primary,
        target_idx=test.target_idx,
        origin_idx=test.origin_idx,
        lead=test.lead,
        standardiser=standardiser,
    )
    ex = check_rows(
        x=test.x, valid_us=test.valid, origin_us=test.origin, lead=test.lead, label=f"{month} test"
    )
    ex += " | " + verify_capped_values(
        x=test.x,
        clean=CTX["clean"],
        primary=test.primary,
        target_idx=test.target_idx,
        origin_idx=test.origin_idx,
        standardiser=standardiser,
        label=f"{month} test",
    )
    pub = common.PUBLICATION_DELAY
    for o in np.unique(test.origin):
        origin = datetime.fromtimestamp(o / 1e6, UTC)
        run = origin - pub
        fr = pl.read_parquet(common.demand_frame_path(run=run))
        assert (fr["nwp_init_time"] <= fr["power_fcst_init_time"]).all()
        assert (fr["power_fcst_init_time"] == origin).all()
        assert (prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet").exists()
        CHECK["origins_weather_checked"] += 1
    i = 0
    CHECK["test_rows"] += len(test.x)
    CHECK["months"][month]["lag_example_test"] = ex
    CHECK["months"][month]["weather_example"] = (
        f"target {datetime.fromtimestamp(test.valid[i] / 1e6, UTC)}, origin {datetime.fromtimestamp(test.origin[i] / 1e6, UTC)}, "
        f"weather run {datetime.fromtimestamp(test.origin[i] / 1e6, UTC) - pub} (nwp_init_time <= power_fcst_init_time asserted for all rows of every test origin)"
    )
    print(
        month, "test checks passed;", ex, "|", CHECK["months"][month]["weather_example"], flush=True
    )
    return test


k.build_month, data.build_test = build_month, build_test


def lag_delta(*, x_cols, valid_us, prim, std, ctx):
    """Return the standardised shift to add to each lag value column (rows x len(VALUE))."""
    shift, sign, block = ctx["shift"], ctx["sign"], ctx["block"]
    out = np.zeros((len(valid_us), len(VALUE)), dtype=np.float32)
    for j, nm in enumerate(VALUE):
        tag = nm.replace("lag_", "")
        step = 1_800_000_000  # 30 min in microseconds; rounding removes float32 age error
        t = (
            valid_us
            - np.rint(
                x_cols[f"lag_{tag}_age"].astype(np.float64) * prep.LEAD_SCALE_HOURS * 3.6e9 / step
            ).astype(np.int64)
            * step
        )
        inblock = (t >= block[0]) & (t < block[1]) & (x_cols[f"lag_{tag}_missing"] < 0.5)
        out[:, j] = np.where(inblock, sign * shift[prim] / std[POS[nm]], 0.0)
    return out


class LagGpuMonth(k.GpuMonth):
    def refit_tensors(self, *, refit, shift_z=None):
        t = super().refit_tensors(refit=refit, shift_z=shift_z)
        if CTX["on"]:
            idx = refit.index
            names = [n for n in POS if n.endswith(("_age", "_missing"))]
            cols = {n: self.mr.x[idx, POS[n]] for n in names}
            d = lag_delta(
                x_cols=cols,
                valid_us=self.mr.valid_us[idx],
                prim=self.mr.primary[idx],
                std=self.mr.base.standardiser.std,
                ctx=CTX,
            )
            for j, nm in enumerate(VALUE):
                v = t["x"][:, POS[nm]] + torch.from_numpy(d[:, j]).to(t["x"].device)
                t["x"][:, POS[nm]] = v.clamp(-prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
            CTX["n_shifted"] = int((d != 0).any(axis=1).sum())
        idx = refit.index
        mask_all, mask_day = lags.block_dropout_masks(
            origin_idx=CTX["origin_idx_mr"][idx], seed=CTX["seed"]
        )
        for n in NEW_NAMES:
            value = (
                1.0
                if n.endswith("_missing")
                else (CTX["age_max"][n] if n.endswith("_age") else 0.0)
            )
            rows = mask_all | (mask_day if n.startswith("lag_day") else False)
            t["x"][torch.from_numpy(rows).to(t["x"].device), POS[n]] = value
        CTX["n_dropped"] = int((mask_all | mask_day).sum())
        CTX["n_all_masked"], CTX["n_day_masked"] = int(mask_all.sum()), int(mask_day.sum())
        return t


k.GpuMonth = LagGpuMonth
_fine_tune, _window_frame = f.fine_tune, f.window_frame


def fine_tune(**kw):
    CTX.update(
        on=kw.get("shift") is not None,
        shift=kw.get("shift"),
        sign=kw.get("sign"),
        block=kw.get("block"),
        seed=kw["seed"],
        cut=kw["cut"],
    )
    model, log = _fine_tune(**kw)
    if CTX["on"]:
        log["rows_with_shifted_lag"] = CTX.get("n_shifted")
    log["rows_with_lags_dropped"] = CTX.get("n_dropped")
    log["rows_all_lags_masked"] = CTX.get("n_all_masked")
    log["rows_day_lag_only_masked"] = CTX.get("n_day_masked")
    return model, log


def window_frame(*, model, test, mr, lo, hi, flag, device):
    if CTX["on"]:
        names = [n for n in POS if n.endswith(("_age", "_missing"))]
        cols = {n: test.x[:, POS[n]] for n in names}
        d = lag_delta(
            x_cols=cols,
            valid_us=test.valid,
            prim=test.primary,
            std=mr.base.standardiser.std,
            ctx=CTX,
        )
        x = test.x.copy()
        for j, nm in enumerate(VALUE):
            x[:, POS[nm]] = np.clip(
                x[:, POS[nm]] + d[:, j], -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD
            )
        test = dataclasses.replace(test, x=x)
    frame = _window_frame(model=model, test=test, mr=mr, lo=lo, hi=hi, flag=flag, device=device)
    CTX["absent_frame"] = None
    if ARGS.arm == "R-DW-24" and not CTX["on"] and frame is not None:
        x = test.x.copy()
        for n in NEW_NAMES:
            x[:, POS[n]] = (
                1.0
                if n.endswith("_missing")
                else (CTX["age_max"][n] if n.endswith("_age") else 0.0)
            )
        CTX["absent_frame"] = _window_frame(
            model=model,
            test=dataclasses.replace(test, x=x),
            mr=mr,
            lo=lo,
            hi=hi,
            flag=flag,
            device=device,
        )
    return frame


_atomic_frame = f.atomic_frame


def atomic_frame(*, frame, target):
    _atomic_frame(frame=frame, target=target)
    if (
        ARGS.arm == "R-DW-24"
        and CTX.get("absent_frame") is not None
        and target.parent.name == "R-DW-24"
    ):
        _atomic_frame(
            frame=CTX["absent_frame"], target=target.parent.parent / "R-DW-24-absent" / target.name
        )


f.fine_tune, f.window_frame, f.atomic_frame = fine_tune, window_frame, atomic_frame


def assemble():
    for tag in (ARM.tag, "R-DW-24-absent") if ARM.tag == "R-DW-24" else (ARM.tag,):
        assemble_one(tag=tag)


def assemble_one(*, tag):
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence="d")
        for seed in (0, 1):
            parts = [
                W / "fits" / tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            # R-DW-24-absent shares R-DW-24's windows, so an empty R-DW-24 window is empty here too
            twin = lambda p: (p.parent.parent / "R-DW-24" / p.name).with_suffix(".empty")
            assert all(
                p.exists()
                or p.with_suffix(".empty").exists()
                or (tag == "R-DW-24-absent" and twin(p).exists())
                for p in parts
            ), (month, seed)
            frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
            f.atomic_frame(frame=frame, target=W / "seed_preds" / tag / f"{month}_s{seed}.parquet")
            print("wrote", tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble()
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
        print("CHECKS PASSED", {k_: v for k_, v in CHECK.items() if k_ != "months"})

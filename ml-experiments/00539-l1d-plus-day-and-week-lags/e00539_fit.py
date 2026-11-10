# ruff: noqa: ANN001, ANN003, ANN201, D101, D102, D103, E501
"""Experiment 538: 511's daily-refit L-1d recipe plus recent-level inputs (R1: 3 inputs; R9: all 9 lag inputs).

Rows, cutoffs, recency weights, early stopping, seeds, and scoring windows are 511's. The lag columns
(value, missing flag, age) are the ones in experiment 397's raw summary matrices, which are the columns
experiment 425 (A-pool) reads, standardised the same way. They are appended after the 37 lag-free
columns, so the first network layer keeps the pretrained weights of ERA5/CAMS pretraining (37 inputs
and the 8-wide Primary embedding) and the new inputs start with zero first-layer weights. The new
inputs are therefore added at fine-tuning only; pretraining is the unchanged lag-free L0PT.

Usage: python e00539_fit.py {fit|assemble} --arm R1|R9 [--stress]
"""

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = Path(str(ML_EXPERIMENTS_DATA_DIR / "00539-l1d-plus-day-and-week-lags"))
W = WORK / "work"
EXTRA_IDX = {"R-DW": [1, 2, 4, 5, 7, 8]}
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
CHECK = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}


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
    """Row by row: ages equal the independent recomputation, every lag observed <= origin - data delay, weather run <= origin."""
    ages = prep.lag_ages_hours(lead_hours=lead.astype(np.float64))
    delay_us = int(common.NEWEST_AGE * 30 * 60 * 1e6)
    pub_us = int(common.PUBLICATION_DELAY.total_seconds() * 1e6)
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
            f"{label} {tag} lag observed after origin minus data delay"
        )
        if example is None:
            i = int(np.argmax(ages[:, j] == ages[:, j].min()))
            example = (
                f"{tag} lag of row {i}: origin {datetime.fromtimestamp(origin_us[i] / 1e6, UTC)}, target {datetime.fromtimestamp(valid_us[i] / 1e6, UTC)}, "
                f"lag observed {datetime.fromtimestamp(t_lag[i] / 1e6, UTC)} (<= origin - {common.NEWEST_AGE * 0.5} h)"
            )
    assert (valid_us > origin_us).all(), f"{label}: target not after origin"
    assert ((origin_us - pub_us) <= origin_us).all()
    return example


_build_month, _build_test = k.build_month, data.build_test


def build_month(*, month, clean):
    mr = _build_month(month=month, clean=clean)
    ex = check_rows(
        x=mr.x, valid_us=mr.valid_us, origin_us=mr.origin_us, lead=mr.lead, label=f"{month} train"
    )
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
    CHECK["train_rows"] += len(mr.y)  # ty: ignore[unsupported-operator]
    CHECK["months"][month] = {  # ty: ignore[invalid-assignment]
        "train_rows": len(mr.y),
        "train_origins": len(runs),
        "lag_example_train": ex,
    }
    # sanity: standardised newest lag against the cleaned power at origin minus the data delay
    if "lag_origin" in POS:
        sd, mean = (
            mr.base.standardiser.std[POS["lag_origin"]],
            mr.base.standardiser.mean[POS["lag_origin"]],
        )
        sel = np.random.default_rng(0).choice(len(mr.y), 200_000, replace=False)
        raw = mr.x[sel, POS["lag_origin"]] * sd + mean
        idx = mr.target_idx[sel] - np.rint(2 * mr.lead[sel]).astype(np.int64) - common.NEWEST_AGE
        ref = clean[mr.primary[sel], idx]
        ok = (
            (mr.x[sel, POS["lag_origin_missing"]] < 0.5)
            & np.isfinite(ref)
            & (np.abs(mr.x[sel, POS["lag_origin"]]) < 4.9)
        )
        CHECK["months"][month]["newest_lag_vs_cleaned_power"] = {  # ty: ignore[not-subscriptable]
            "rows": int(ok.sum()),
            "within_1e-3": float((np.abs(raw[ok] - ref[ok]) < 1e-3).mean()),
        }
    print(month, "train checks passed;", ex, flush=True)
    return mr


def build_test(*, month, standardiser):
    test = _build_test(month=month, standardiser=standardiser)
    ex = check_rows(
        x=test.x, valid_us=test.valid, origin_us=test.origin, lead=test.lead, label=f"{month} test"
    )
    pub = common.PUBLICATION_DELAY
    for o in np.unique(test.origin):
        origin = datetime.fromtimestamp(o / 1e6, UTC)
        run = origin - pub
        fr = pl.read_parquet(common.demand_frame_path(run=run))
        assert (fr["nwp_init_time"] <= fr["power_fcst_init_time"]).all()
        assert (fr["power_fcst_init_time"] == origin).all()
        assert (prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet").exists()
        CHECK["origins_weather_checked"] += 1  # ty: ignore[unsupported-operator]
    i = 0
    CHECK["test_rows"] += len(test.x)  # ty: ignore[unsupported-operator]
    CHECK["months"][month]["lag_example_test"] = ex  # ty: ignore[not-subscriptable]
    CHECK["months"][month]["weather_example"] = (  # ty: ignore[not-subscriptable]
        f"target {datetime.fromtimestamp(test.valid[i] / 1e6, UTC)}, origin {datetime.fromtimestamp(test.origin[i] / 1e6, UTC)}, "
        f"weather run {datetime.fromtimestamp(test.origin[i] / 1e6, UTC) - pub} (nwp_init_time <= power_fcst_init_time asserted for all rows of every test origin)"
    )
    print(
        month,
        "test checks passed;",
        ex,
        "|",
        CHECK["months"][month]["weather_example"],  # ty: ignore[not-subscriptable]
        flush=True,  # ty: ignore[not-subscriptable]
    )
    return test


k.build_month, data.build_test = build_month, build_test


def lag_delta(*, x_cols, valid_us, prim, std, ctx):
    """Return the standardised shift to add to each lag value column (rows x len(VALUE))."""
    shift, sign, block = ctx["shift"], ctx["sign"], ctx["block"]
    out = np.zeros((len(valid_us), len(VALUE)), dtype=np.float32)
    for j, nm in enumerate(VALUE):
        tag = nm.replace("lag_", "")
        t = valid_us - (
            x_cols[f"lag_{tag}_age"].astype(np.float64) * prep.LEAD_SCALE_HOURS * 3.6e9
        ).astype(np.int64)
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
        return t


k.GpuMonth = LagGpuMonth
_fine_tune, _window_frame = f.fine_tune, f.window_frame


def fine_tune(**kw):
    CTX.update(
        on=kw.get("shift") is not None,
        shift=kw.get("shift"),
        sign=kw.get("sign"),
        block=kw.get("block"),
    )
    model, log = _fine_tune(**kw)
    if CTX["on"]:
        log["rows_with_shifted_lag"] = CTX.get("n_shifted")
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
    return _window_frame(model=model, test=test, mr=mr, lo=lo, hi=hi, flag=flag, device=device)


f.fine_tune, f.window_frame = fine_tune, window_frame


def assemble():
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence="d")
        for seed in (0, 1):
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

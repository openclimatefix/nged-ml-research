# ruff: noqa: ANN001, ANN003, ANN201, D101, D102, D103, E501, E731, RET504, B007
"""Experiment 544 (copy of experiment 539's fit script with R-MED and R-DROP): 538: 511's daily-refit L-1d recipe plus recent-level inputs (R1: 3 inputs; R9: all 9 lag inputs).

Rows, cutoffs, recency weights, early stopping, seeds, and scoring windows are 511's. The lag columns
(value, missing flag, age) are the ones in experiment 397's raw summary matrices, which are the columns
experiment 425 (A-pool) reads, standardised the same way. They are appended after the 37 lag-free
columns, so the first network layer keeps the pretrained weights of ERA5/CAMS pretraining (37 inputs
and the 8-wide Primary embedding) and the new inputs start with zero first-layer weights. The new
inputs are therefore added at fine-tuning only; pretraining is the unchanged lag-free L0PT.

R-MED replaces the day-ago and week-ago values by the median of the same half-hour on the 7 and on the
28 most recent days at least a day old (flag = fewer than 3 or 10 members observed; age = the
newest member's age). R-DROP keeps 539's six inputs and, in 30% of rows, sets them to "absent"
(value 0, missing flag 1, age at the training maximum) during fine-tuning; it is scored with the
readings present (arm R-DROP) and with every row's readings absent (arm R-DROP-absent).

Usage: python e00544_fit.py {fit|assemble} --arm R-MED|R-DROP [--stress]
"""

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import e00087_common as common
import e00397_prep as prep
import e00428_data as data
import e00510_common as c
import e00511_common as k
import e00511_fit as f
import numpy as np
import polars as pl
import torch

WORK = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/00544-lag-light-demand-network-robust-same-slot-median-and-lag-dropout"
)
W = WORK / "work"
EXTRA_IDX = {"R-MED": [1, 2, 4, 5, 7, 8], "R-DROP": [1, 2, 4, 5, 7, 8]}
DROP_FRACTION = 0.3
MED_SHORT_DAYS, MED_LONG_DAYS = 7, 28
MED_MIN = {MED_SHORT_DAYS: 3, MED_LONG_DAYS: 10}
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


STEPS = common.STEPS_PER_DAY
SLOTS = {"lag_day": MED_SHORT_DAYS, "lag_week": MED_LONG_DAYS}


def median_table(*, clean, days):
    """Return (primaries, grid): the median of clean[:, g - 48 j] for j < days (NaN with too few observed members).

    Every member index is at most g, so a lookup at an index no later than the newest usable observation
    uses only usable observations.
    """
    n_primary, n_grid = clean.shape
    members = np.full((days, n_primary, n_grid), np.inf, dtype=np.float32)
    for j in range(days):
        shifted = clean[:, : n_grid - STEPS * j].astype(np.float32)
        members[j, :, STEPS * j :] = np.where(np.isfinite(shifted), shifted, np.inf)
    members.sort(axis=0)
    count = np.isfinite(members).sum(axis=0)
    lo = np.take_along_axis(members, np.clip((count - 1) // 2, 0, days - 1)[None], axis=0)[0]
    hi = np.take_along_axis(members, np.clip(count // 2, 0, days - 1)[None], axis=0)[0]
    out = np.where(count >= MED_MIN[days], (lo + hi) / 2.0, np.nan).astype(np.float32)
    return out


def tables(*, clean):
    return {slot: median_table(clean=clean, days=d) for slot, d in SLOTS.items()}


def first_index(*, target_idx, origin_idx):
    """Grid index of the latest same half-hour at or before the newest usable observation."""
    newest = origin_idx - common.NEWEST_AGE
    gap = target_idx - newest
    first = target_idx - STEPS * np.ceil(gap / STEPS).astype(np.int64)
    assert (first <= newest).all(), "a median member is younger than the newest usable observation"
    assert (first >= 0).all()
    return first


def median_columns(*, tab, prim, first, standardiser):
    """Standardised median value and missing flag per slot for rows (prim, first)."""
    out = {}
    for slot in SLOTS:
        raw = tab[slot][prim, first]
        miss = ~np.isfinite(raw)
        z = (raw - standardiser.mean[POS[slot]]) / standardiser.std[POS[slot]]
        z = np.clip(np.where(miss, 0.0, z), -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
        out[slot] = (z.astype(np.float32), miss.astype(np.float32), raw)
    return out


def put_medians(*, x, cols, age_day):
    for slot, (z, miss, _) in cols.items():
        x[:, POS[slot]] = z
        x[:, POS[f"{slot}_missing"]] = miss
        x[:, POS[f"{slot}_age"]] = age_day


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


def verify_medians(*, prim, first, cols, clean, label):
    """Recompute 2,000 random rows' medians with an explicit loop, and print one example row."""
    rng = np.random.default_rng(1)
    sel = rng.choice(len(prim), min(2000, len(prim)), replace=False)
    example = ""
    for slot, days in SLOTS.items():
        raw = cols[slot][2]
        for i in sel:
            members = np.array(
                [
                    clean[prim[i], first[i] - STEPS * j]
                    for j in range(days)
                    if first[i] - STEPS * j >= 0
                ]
            )
            ok = np.isfinite(members)
            want = np.median(members[ok]) if ok.sum() >= MED_MIN[days] else np.nan
            assert (np.isnan(want) and np.isnan(raw[i])) or abs(want - raw[i]) < 1e-5, (
                f"{label} {slot} median differs at row {i}"
            )
        i = int(sel[0])
        example += (
            f"{slot}: median of {days} same-half-hour values ending at grid index {first[i]} (newest usable <= origin - 3.5 h asserted for all rows)="
            f"{raw[i]:.4f}; "
        )
    return f"{label} medians match an independent loop on {len(sel)} rows; {example}"


_build_month, _build_test = k.build_month, data.build_test


def build_month(*, month, clean):
    mr = _build_month(month=month, clean=clean)
    ex = check_rows(
        x=mr.x, valid_us=mr.valid_us, origin_us=mr.origin_us, lead=mr.lead, label=f"{month} train"
    )
    for key in [k_ for k_ in CTX if isinstance(k_, tuple)]:
        del CTX[key]
    if ARGS.arm == "R-MED":
        CTX["clean"], CTX["tab"] = clean, tables(clean=clean)
        origin_idx = (mr.origin_us - c.prep422.GRID_START_US) // c.prep422.STEP_US
        first = first_index(target_idx=mr.target_idx, origin_idx=origin_idx)
        CTX["first_mr"], CTX["month"] = first, month
        age_day = mr.x[:, POS["lag_day_age"]].copy()
        cols = median_columns(
            tab=CTX["tab"], prim=mr.primary, first=first, standardiser=mr.base.standardiser
        )
        put_medians(x=mr.x, cols=cols, age_day=age_day)
        ex = (
            ex
            + " | "
            + verify_medians(
                prim=mr.primary, first=first, cols=cols, clean=clean, label=f"{month} train"
            )
        )
    if ARGS.arm == "R-DROP":
        CTX["age_max"] = {n: float(mr.x[:, POS[n]].max()) for n in POS if n.endswith("_age")}
        CTX["month"] = month
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
        CHECK["months"][month]["newest_lag_vs_cleaned_power"] = {
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
    if ARGS.arm == "R-MED":
        first = first_index(target_idx=test.target_idx, origin_idx=test.origin_idx)
        cols = median_columns(
            tab=CTX["tab"], prim=test.primary, first=first, standardiser=standardiser
        )
        put_medians(x=test.x, cols=cols, age_day=test.x[:, POS["lag_day_age"]].copy())
        ex = (
            ex
            + " | "
            + verify_medians(
                prim=test.primary, first=first, cols=cols, clean=CTX["clean"], label=f"{month} test"
            )
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
        t = valid_us - (
            x_cols[f"lag_{tag}_age"].astype(np.float64) * prep.LEAD_SCALE_HOURS * 3.6e9
        ).astype(np.int64)
        inblock = (t >= block[0]) & (t < block[1]) & (x_cols[f"lag_{tag}_missing"] < 0.5)
        out[:, j] = np.where(inblock, sign * shift[prim] / std[POS[nm]], 0.0)
    return out


def stressed_tab(*, sign):
    """Median tables of the cleaned power with the injected shift added inside the stress block."""
    key = ("stab", CTX["month"], sign)
    if key not in CTX:
        shift = CTX["shift"]
        lo = int((CTX["block"][0] - c.prep422.GRID_START_US) // c.prep422.STEP_US)
        hi = int(-(-(CTX["block"][1] - c.prep422.GRID_START_US) // c.prep422.STEP_US))
        shifted = CTX["clean"].copy()
        shifted[:, lo:hi] += (sign * np.asarray(shift))[:, None]
        CTX[key] = tables(clean=shifted)
    return CTX[key]


def stressed_medians(*, g, prim, first, standardiser=None):
    std = standardiser if standardiser is not None else g.mr.base.standardiser
    return median_columns(
        tab=stressed_tab(sign=CTX["sign"]), prim=prim, first=first, standardiser=std
    )


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
            if ARGS.arm == "R-MED":
                z = stressed_medians(g=self, prim=self.mr.primary[idx], first=CTX["first_mr"][idx])
                for slot, (zz, mm, _) in z.items():
                    t["x"][:, POS[slot]] = torch.from_numpy(zz).to(t["x"].device)
        if ARGS.arm == "R-DROP":
            rng = np.random.default_rng([CTX["seed"], int(CTX["cut"].timestamp())])
            drop = torch.from_numpy(rng.random(len(refit.index)) < DROP_FRACTION).to(t["x"].device)
            for n in NEW_NAMES:
                value = (
                    1.0
                    if n.endswith("_missing")
                    else (CTX["age_max"][n] if n.endswith("_age") else 0.0)
                )
                t["x"][drop, POS[n]] = value
            CTX["n_dropped"] = int(drop.sum())
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
    if ARGS.arm == "R-DROP":
        log["rows_with_lags_dropped"] = CTX.get("n_dropped")
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
        if ARGS.arm == "R-MED":
            first = first_index(target_idx=test.target_idx, origin_idx=test.origin_idx)
            for slot, (zz, _, _) in stressed_medians(
                g=None, prim=test.primary, first=first, standardiser=mr.base.standardiser
            ).items():
                x[:, POS[slot]] = zz
        test = dataclasses.replace(test, x=x)
    frame = _window_frame(model=model, test=test, mr=mr, lo=lo, hi=hi, flag=flag, device=device)
    CTX["absent_frame"] = None
    if ARGS.arm == "R-DROP" and not CTX["on"] and frame is not None:
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
        ARGS.arm == "R-DROP"
        and CTX.get("absent_frame") is not None
        and target.parent.name == "R-DROP"
    ):
        _atomic_frame(
            frame=CTX["absent_frame"], target=target.parent.parent / "R-DROP-absent" / target.name
        )


f.fine_tune, f.window_frame, f.atomic_frame = fine_tune, window_frame, atomic_frame


def assemble():
    for tag in (ARM.tag, "R-DROP-absent") if ARM.tag == "R-DROP" else (ARM.tag,):
        assemble_one(tag=tag)


def assemble_one(*, tag):
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence="d")
        for seed in (0, 1):
            parts = [
                W / "fits" / tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            # R-DROP-absent shares R-DROP's windows, so an empty R-DROP window is empty here too
            twin = lambda p: (p.parent.parent / "R-DROP" / p.name).with_suffix(".empty")
            assert all(
                p.exists()
                or p.with_suffix(".empty").exists()
                or (tag == "R-DROP-absent" and twin(p).exists())
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

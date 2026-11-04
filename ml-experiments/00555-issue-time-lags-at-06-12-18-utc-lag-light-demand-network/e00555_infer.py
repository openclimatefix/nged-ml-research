# ruff: noqa: ANN001, ANN201, ANN202, D103, E501, PLR0912, PLR0915, PT018
"""Score the stored lag-light networks at the 00, 06, 12, and 18 UTC issue times (inference only).

Each origin day's ECMWF 00 UTC run is usable from the 09:00 UTC origin. For an issue time `I`
(12 and 18 UTC the same day; 00 and 06 UTC the next day) the stored network is loaded unchanged and
scored on that run's test rows with the inputs that depend on the issue time rebuilt:

- the lead and its square root, counted from `I`;
- the day-ago and week-ago readings (value, missing flag, age), and for R9 the newest reading too,
  retaken from the latest same-clock half-hour at or before `I` minus the 3.5 h data delay.

The weather columns, the calendar columns, and the Primary embedding are unchanged. Rows whose valid
time is not after `I` are dropped. The tag `09` keeps the stored table untouched and must reproduce
the stored forecasts (the production origin, for reference).

Usage: python e00555_infer.py [--months ...] [--limit-days N] [--arms R-DW R-DROP R9]
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
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

DD = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
OUT = DD / "00555-issue-time-lags-at-06-12-18-utc-lag-light-demand-network"
W = Path(os.environ.get("E555_WORK", OUT / "work"))
ROOT = {
    "R-DW": DD / "00539-l1d-plus-day-and-week-lags/work",
    "R-DROP": DD / "00544-lag-light-demand-network-robust-same-slot-median-and-lag-dropout/work",
    "R9": DD / "00538-l1d-plus-recent-level-and-lags/work",
}
LAG_INDICES = {"R-DW": (1, 2, 4, 5, 7, 8), "R-DROP": (1, 2, 4, 5, 7, 8), "R9": tuple(range(9))}
ISSUES = {"09": 0, "12": 6, "18": 18, "00": 30, "06": 42}
"""Issue hour (UTC) -> half-hours after the 09:00 UTC origin of the run's day."""
parser = argparse.ArgumentParser()
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--limit-days", type=int, default=None)
parser.add_argument("--arms", nargs="+", default=list(ROOT))
ARGS = parser.parse_args()

NAMES, MASK, LAYOUT = prep.feature_names(scheme="summary")
LAG = LAYOUT.columns_of(groups=["lags"])
assert len(LAG) == 9
KEEP0, COLS0, MASK0 = data.non_lag_layout()
N0 = len(KEEP0)
assert N0 == 37
LEAD_COL, LEAD_SQRT_COL = COLS0.index("lead"), COLS0.index("lead_sqrt")
GRID, STEP = c.prep422.GRID_START_US, c.prep422.STEP_US
NEWEST = common.NEWEST_AGE
PERIOD = {"day": 48, "week": 336}
CUR: dict = {}
CHECK: dict = {"issues": {}, "reproduction": {}}


def set_layout(*, arm: str) -> None:
    extra = [LAG[i] for i in LAG_INDICES[arm]]
    names = [NAMES[i] for i in extra]
    CUR.update(extra=extra, names=names, pos={n: N0 + j for j, n in enumerate(names)})

    def layout():
        keep = list(KEEP0) + extra
        return keep, [*COLS0, *names], MASK[np.array(keep)]

    data.non_lag_layout = layout


def lag_sources(*, target_idx, newest_idx):
    """Return the grid index of each lag's source (newest, day-ago, week-ago) at or before the newest reading."""
    out = {"origin": newest_idx}
    for tag, period in PERIOD.items():
        out[tag] = target_idx - period * np.ceil((target_idx - newest_idx) / period).astype(
            np.int64
        )
    return out


def issue_x(*, test, step, std, clean, label):
    """Return the test table rebuilt for an issue `step` half-hours after the origin."""
    pos = CUR["pos"]
    issue_idx = test.origin_idx + step
    newest = issue_idx - NEWEST
    x = test.x.copy()
    lead_h = (test.target_idx - issue_idx) / 2.0
    assert np.allclose(test.lead - step / 2.0, lead_h, atol=1e-6), (
        f"{label}: lead differs from the grid"
    )
    x[:, LEAD_COL] = (lead_h / prep.LEAD_SCALE_HOURS).astype(np.float32)
    x[:, LEAD_SQRT_COL] = np.sqrt(np.clip(lead_h / prep.LEAD_SCALE_HOURS, 0, None)).astype(
        np.float32
    )
    src = lag_sources(target_idx=test.target_idx, newest_idx=newest)
    # independent recomputation of the lag ages from the lead (experiment 397's formula)
    ages = prep.lag_ages_hours(lead_hours=lead_h)
    for j, tag in enumerate(("origin", "day", "week")):
        name = f"lag_{tag}"
        if name not in pos:
            continue
        assert (src[tag] <= newest).all(), (
            f"{label}: {tag} lag source younger than issue minus data delay"
        )
        if tag in PERIOD:
            assert ((test.target_idx - src[tag]) % PERIOD[tag] == 0).all(), (
                f"{label}: {tag} source is not the same clock slot"
            )
            assert (src[tag] + PERIOD[tag] > newest).all(), (
                f"{label}: {tag} source is not the latest such slot"
            )
        src_us = GRID + src[tag] * STEP
        assert (src_us <= GRID + issue_idx * STEP - int(NEWEST * 30 * 60 * 1e6)).all(), (
            f"{label}: {tag} source later than issue minus data delay (clock check)"
        )
        assert (src[tag] <= issue_idx - NEWEST).all()
        gap_hours = (test.target_idx - src[tag]) / 2.0
        assert np.allclose(gap_hours, ages[:, j], atol=1e-6), (
            f"{label}: {tag} age differs from independent recomputation"
        )
        ref = np.where(src[tag] >= 0, clean[test.primary, np.maximum(src[tag], 0)], np.nan)
        miss = ~np.isfinite(ref)
        col = pos[name]
        z = np.clip((ref - std.mean[col]) / std.std[col], -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
        x[:, col] = np.where(miss, 0.0, z).astype(np.float32)
        x[:, pos[f"{name}_missing"]] = miss.astype(np.float32)
        x[:, pos[f"{name}_age"]] = (gap_hours / prep.LEAD_SCALE_HOURS).astype(np.float32)
    keep = lead_h > 0
    i = int(np.nonzero(keep)[0][len(np.nonzero(keep)[0]) // 2])
    when = lambda idx: datetime.fromtimestamp((GRID + int(idx) * STEP) / 1e6, UTC)  # noqa: E731
    example = (
        f"issue {when(issue_idx[i])}, weather run {when(test.origin_idx[i]) - common.PUBLICATION_DELAY} "
        f"(usable {when(test.origin_idx[i])}), target {when(test.target_idx[i])}, "
        + ", ".join(
            f"{t} lag from {when(src[t][i])}"
            for t in ("origin", "day", "week")
            if f"lag_{t}" in pos
        )
    )
    assert (test.origin_idx <= issue_idx).all(), (
        f"{label}: weather run not usable at the issue time"
    )
    return x, keep, example


_build_test = data.build_test


def build_test(*, month, standardiser):
    test = _build_test(month=month, standardiser=standardiser)
    pub = common.PUBLICATION_DELAY
    for o in np.unique(test.origin):
        origin = datetime.fromtimestamp(o / 1e6, UTC)
        run = origin - pub
        fr = pl.read_parquet(common.demand_frame_path(run=run))
        assert (fr["nwp_init_time"] <= fr["power_fcst_init_time"]).all()
        assert (fr["power_fcst_init_time"] == origin).all()
        assert (prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet").exists()
    return test


data.build_test = build_test


def main():
    clean = c.clean_stack()
    device = "cuda"
    d15_d18 = [
        int(r["primary"])
        for r in prep.primaries().iter_rows(named=True)
        if r["site"] in ("D15", "D18")
    ]
    for arm in ARGS.arms:
        set_layout(arm=arm)
        pos = CUR["pos"]
        for month in ARGS.months:
            mr = k.build_month(month=month, clean=clean)
            std = mr.base.standardiser
            test = data.build_test(month=month, standardiser=std)
            n_in = test.x.shape[1]
            assert n_in == mr.x.shape[1]
            tables = {}
            for tag, step in ISSUES.items():
                label = f"{arm} {month} {tag}"
                if step == 0:
                    z, keep, ex = test.x, np.ones(len(test.x), bool), "stored table"
                    # the rebuilt table at the 09:00 UTC origin must match the stored table (apart from D15/D18 and clipped rows)
                    zz, _, _ = issue_x(test=test, step=0, std=std, clean=clean, label=label)
                    ok = ~np.isin(test.primary, d15_d18)
                    for col in (LEAD_COL, LEAD_SQRT_COL):
                        assert np.abs(zz[:, col] - test.x[:, col]).max() < 1e-5, (
                            f"{label}: rebuilt lead column differs from the stored table"
                        )
                    other = [
                        j
                        for j in range(n_in)
                        if j not in set(pos.values()) | {LEAD_COL, LEAD_SQRT_COL}
                    ]
                    assert (zz[:, other] == test.x[:, other]).all()
                    for name in pos:
                        if name.endswith("_age"):
                            assert np.abs(zz[:, pos[name]] - test.x[:, pos[name]]).max() < 1e-4, (
                                f"{label}: rebuilt {name} differs from the stored table"
                            )
                        elif name.endswith("_missing"):
                            assert (zz[ok, pos[name]] == test.x[ok, pos[name]]).all(), (
                                f"{label}: rebuilt {name} differs from the stored table"
                            )
                        else:
                            both = (
                                ok
                                & (np.abs(test.x[:, pos[name]]) < prep.INPUT_CLIP_SD - 1e-6)
                                & (np.abs(zz[:, pos[name]]) < prep.INPUT_CLIP_SD - 1e-6)
                            )
                            assert (
                                np.abs(zz[both, pos[name]] - test.x[both, pos[name]]).max() < 2e-3
                            ), f"{label}: rebuilt {name} differs from the stored table"
                    CHECK["reproduction"][f"{arm} {month} rebuild at 09"] = True
                else:
                    z, keep, ex = issue_x(test=test, step=step, std=std, clean=clean, label=label)
                CHECK["issues"].setdefault(
                    label, {"rows_kept": int(keep.sum()), "rows": len(keep), "example": ex}
                )
                tables[tag] = (z, keep)
                print(label, "checks passed;", ex, flush=True)
            cuts = k.cutoffs(month=month, cadence="d")
            for seed in (0, 1):
                frames = {tag: [] for tag in ISSUES}
                for i, cut in enumerate(cuts[: ARGS.limit_days]):
                    day = f.day_of(month=month, cut=cut)
                    mfile = ROOT[arm] / "models" / arm / f"{month}_d{day}_s{seed}.pt"
                    lo, hi = k.window_of(month=month, cadence="d", index=i)
                    window = (test.origin >= lo) & (test.origin < hi)
                    if not window.any():
                        continue
                    model = c.new_model(
                        outlier=False, n_inputs=n_in, n_primaries=len(mr.base.target_mean)
                    ).to(device)
                    model.load_state_dict(torch.load(mfile, map_location=device))
                    model.eval()
                    for tag, (z, keep) in tables.items():
                        t = dataclasses.replace(test, x=z)
                        frames[tag].append(
                            c.forecast_frame(
                                model=model,
                                outlier=False,
                                test=t,
                                select=window & keep,
                                rows=mr.base,
                                widen=1.0,
                                device=device,
                            )
                        )
                    del model
                for tag, parts in frames.items():
                    frame = pl.concat(parts)
                    f.atomic_frame(
                        frame=frame,
                        target=W / "seed_preds" / f"{arm}-i{tag}" / f"{month}_s{seed}.parquet",
                    )
                    if tag == "09" and ARGS.limit_days is None:
                        stored = pl.read_parquet(
                            ROOT[arm] / "seed_preds" / arm / f"{month}_s{seed}.parquet"
                        )
                        keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
                        j = frame.join(stored, on=keys, suffix="_s", validate="1:1")
                        qcols = [q for q in frame.columns if q not in keys]
                        dq = max(float((j[q] - j[f"{q}_s"]).abs().max()) for q in qcols)
                        CHECK["reproduction"][f"{arm} {month} s{seed}"] = {
                            "rows": j.height,
                            "stored_rows": stored.height,
                            "max_abs": dq,
                        }
                        assert j.height == stored.height == frame.height and dq < 1e-4, (
                            arm,
                            month,
                            seed,
                            j.height,
                            stored.height,
                            dq,
                        )
                print(month, arm, "seed", seed, "done", flush=True)
            del mr, test, tables
            torch.cuda.empty_cache()
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "checks_infer.json").write_text(json.dumps(CHECK, indent=1, default=str))
    print(
        "ISSUE-TIME CHECKS PASSED;",
        len(CHECK["issues"]),
        "arm-month-issue tables asserted row by row",
    )


if __name__ == "__main__":
    main()

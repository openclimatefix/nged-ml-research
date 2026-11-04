# ruff: noqa: ANN001, ANN201, ANN202, D103, E501, PT018
"""Part 3: lag removal and late reading with the stored R-DW (539) and R-DROP (544) weights (inference only, weights fixed).

For each month, day, and seed the stored network is loaded unchanged and scored on its own origin window
with the six lag inputs (day-ago and week-ago value, missing flag, age) altered in the test rows only:

- ``base``: readings present (must reproduce the stored forecasts; checked);
- ``none``: every reading removed (value 0, flag 1, age at the training maximum), as R-DROP's dropout does;
- ``noday`` / ``noweek``: only the day-ago / week-ago reading removed;
- ``age1`` / ``age2`` / ``age3``: the readings arrive 1, 2, or 3 days late, so the newest usable reading is
  that much older and every lag is re-taken from the latest same-clock half-hour at or before it
  (value, flag, and age all recomputed from the cleaned power);
- ``rand25``: in 25% of rows (fixed seed 551) both readings are removed.

Usage: python e00551_removal.py [--months ...] [--limit-days N]
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
OUT = DD / "00551-lag-light-release-safety-lag-timestamp-audit-holidays-and-lag-removal"
W = OUT / "work"
SRC = {
    "R-DW": DD / "00539-l1d-plus-day-and-week-lags/work",
    "R-DROP": DD / "00544-lag-light-demand-network-robust-same-slot-median-and-lag-dropout/work",
}
SCENARIOS = ("base", "none", "noday", "noweek", "age1", "age2", "age3", "rand25")
parser = argparse.ArgumentParser()
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--limit-days", type=int, default=None)
ARGS = parser.parse_args()

NAMES, MASK, LAYOUT = prep.feature_names(scheme="summary")
LAG = LAYOUT.columns_of(groups=["lags"])
EXTRA = [LAG[i] for i in (1, 2, 4, 5, 7, 8)]
KEEP0, COLS0, MASK0 = data.non_lag_layout()
N0 = len(KEEP0)
assert N0 == 37
NEW_NAMES = [NAMES[i] for i in EXTRA]
POS = {n: N0 + j for j, n in enumerate(NEW_NAMES)}
assert list(POS) == [
    "lag_day",
    "lag_week",
    "lag_day_missing",
    "lag_week_missing",
    "lag_day_age",
    "lag_week_age",
]
GRID, STEP = c.prep422.GRID_START_US, c.prep422.STEP_US
NEWEST = common.NEWEST_AGE
PERIOD = {"day": 48, "week": 336}
CHECK: dict = {"months": {}}


def layout():
    keep = list(KEEP0) + EXTRA
    return keep, [*COLS0, *NEW_NAMES], MASK[np.array(keep)]


data.non_lag_layout = layout


def aged_columns(*, test, std, clean, extra_days):
    """Recompute the six lag columns when the newest usable reading is `extra_days` older."""
    newest = test.origin_idx - NEWEST - 48 * extra_days
    out = {}
    for tag, period in PERIOD.items():
        src = test.target_idx - period * np.ceil((test.target_idx - newest) / period).astype(
            np.int64
        )
        ref = np.where(src >= 0, clean[test.primary, np.maximum(src, 0)], np.nan)
        miss = ~np.isfinite(ref)
        col = POS[f"lag_{tag}"]
        z = np.clip((ref - std.mean[col]) / std.std[col], -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
        out[tag] = (
            np.where(miss, 0.0, z).astype(np.float32),
            miss.astype(np.float32),
            ((test.target_idx - src) / 2.0 / prep.LEAD_SCALE_HOURS).astype(np.float32),
            src,
        )
    return out


def scenario_x(*, test, scn, age_max, std, clean):
    x = test.x.copy()

    def absent(rows, tags):
        for tag in tags:
            x[rows, POS[f"lag_{tag}"]] = 0.0
            x[rows, POS[f"lag_{tag}_missing"]] = 1.0
            x[rows, POS[f"lag_{tag}_age"]] = age_max[f"lag_{tag}_age"]

    if scn == "base":
        return x
    if scn == "none":
        absent(slice(None), ("day", "week"))
    elif scn == "noday":
        absent(slice(None), ("day",))
    elif scn == "noweek":
        absent(slice(None), ("week",))
    elif scn == "rand25":
        rng = np.random.default_rng(551)
        rows = rng.random(len(x)) < 0.25
        absent(rows, ("day", "week"))
        CHECK.setdefault("rand25_fraction_by_call", []).append(float(rows.mean()))
    elif scn.startswith("age"):
        days = int(scn[3:])
        cols = aged_columns(test=test, std=std, clean=clean, extra_days=days)
        for tag, (z, miss, age, src) in cols.items():
            (
                x[:, POS[f"lag_{tag}"]],
                x[:, POS[f"lag_{tag}_missing"]],
                x[:, POS[f"lag_{tag}_age"]],
            ) = z, miss, age
            assert (src <= test.origin_idx - NEWEST - 48 * days).all(), (
                "aged lag younger than the late-reading limit"
            )
    else:
        raise ValueError(scn)
    return x


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
    i = 0
    CHECK["months"].setdefault(month, {})["weather_example"] = (
        f"target {datetime.fromtimestamp(test.valid[i] / 1e6, UTC)}, origin {datetime.fromtimestamp(test.origin[i] / 1e6, UTC)}, "
        f"weather run {datetime.fromtimestamp(test.origin[i] / 1e6, UTC) - pub} (nwp_init_time <= power_fcst_init_time asserted for all rows of every test origin)"
    )
    print(month, CHECK["months"][month]["weather_example"], flush=True)
    return test


data.build_test = build_test


def main():
    clean = c.clean_stack()
    device = "cuda"
    for month in ARGS.months:
        mr = k.build_month(month=month, clean=clean)
        std = mr.base.standardiser
        test = data.build_test(month=month, standardiser=std)
        age_max = {n: float(mr.x[:, POS[n]].max()) for n in POS if n.endswith("_age")}
        CHECK["months"][month]["age_max"] = age_max
        xs = {
            scn: scenario_x(test=test, scn=scn, age_max=age_max, std=std, clean=clean)
            for scn in SCENARIOS
        }
        # the aged recomputation at zero extra days must reproduce the built table (apart from D15/D18 readings cut from the cleaned power)
        z0 = aged_columns(test=test, std=std, clean=clean, extra_days=0)
        ok = ~np.isin(
            test.primary,
            [
                int(r["primary"])
                for r in prep.primaries().iter_rows(named=True)
                if r["site"] in ("D15", "D18")
            ],
        )
        for tag, (z, miss, age, _) in z0.items():
            assert np.allclose(age, test.x[:, POS[f"lag_{tag}_age"]], atol=1e-4)
            assert (miss[ok] == test.x[ok, POS[f"lag_{tag}_missing"]]).all()
            both = ok & (miss < 0.5) & (np.abs(z) < prep.INPUT_CLIP_SD - 1e-6)
            assert np.abs(z[both] - test.x[both, POS[f"lag_{tag}"]]).max() < 2e-3
        CHECK["months"][month]["recompute_matches_table"] = True
        cuts = k.cutoffs(month=month, cadence="d")
        for arm, root in SRC.items():
            for seed in (0, 1):
                frames = {scn: [] for scn in SCENARIOS}
                for i, cut in enumerate(cuts[: ARGS.limit_days]):
                    day = f.day_of(month=month, cut=cut)
                    mfile = root / "models" / arm / f"{month}_d{day}_s{seed}.pt"
                    lo, hi = k.window_of(month=month, cadence="d", index=i)
                    sel = (test.origin >= lo) & (test.origin < hi)
                    if not sel.any():
                        continue
                    model = c.new_model(
                        outlier=False, n_inputs=mr.x.shape[1], n_primaries=len(mr.base.target_mean)
                    ).to(device)
                    model.load_state_dict(torch.load(mfile, map_location=device))
                    model.eval()
                    for scn in SCENARIOS:
                        t = dataclasses.replace(test, x=xs[scn])
                        frames[scn].append(
                            c.forecast_frame(
                                model=model,
                                outlier=False,
                                test=t,
                                select=sel,
                                rows=mr.base,
                                widen=1.0,
                                device=device,
                            )
                        )
                    del model
                for scn, parts in frames.items():
                    frame = pl.concat(parts)
                    target = W / "seed_preds" / f"{arm}-{scn}" / f"{month}_s{seed}.parquet"
                    f.atomic_frame(frame=frame, target=target)
                    if ARGS.limit_days is None and (
                        scn == "base" or (scn == "none" and arm == "R-DROP")
                    ):
                        stored_name = arm if scn == "base" else "R-DROP-absent"
                        stored = pl.read_parquet(
                            root / "seed_preds" / stored_name / f"{month}_s{seed}.parquet"
                        )
                        keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
                        j = frame.join(stored, on=keys, suffix="_s", validate="1:1")
                        qcols = [q for q in frame.columns if q not in keys]
                        dq = max(float((j[q] - j[f"{q}_s"]).abs().max()) for q in qcols)
                        CHECK["months"][month].setdefault(
                            "reproduces_stored_max_abs_all_quantiles", {}
                        )[f"{arm}-{scn}_s{seed} vs {stored_name}"] = {
                            "rows": j.height,
                            "max_abs": dq,
                        }
                        CHECK["n_reproduction_checks"] = CHECK.get("n_reproduction_checks", 0) + 1
                        assert j.height == stored.height and dq < 1e-4, (
                            arm,
                            scn,
                            month,
                            seed,
                            j.height,
                            stored.height,
                            dq,
                        )
                print(month, arm, "seed", seed, "done", flush=True)
        del mr, test, xs
        torch.cuda.empty_cache()
        (OUT / "checks_removal.json").write_text(json.dumps(CHECK, indent=1, default=str))
    assert ARGS.limit_days is not None or CHECK.get("n_reproduction_checks", 0) == 6 * len(
        ARGS.months
    )
    print(
        "REMOVAL CHECKS PASSED;",
        CHECK.get("n_reproduction_checks", 0),
        "stored-forecast reproduction checks ran",
    )


if __name__ == "__main__":
    main()

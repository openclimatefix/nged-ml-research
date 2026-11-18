# ruff: noqa: D103, E501, PLR0912, PLR0915, PT018, E731, RUF059
"""Experiment 708: write the production export bundle of one refit of K1, table T, and the Candidate.

Usage: `python e00708_export.py --month MONTH --dump DUMP.npz --out BUNDLE_DIR --source-root
P7_D50_MONTH --chain-work CHAIN --scratch SCRATCH` (`e00708_capture.py` writes DUMP.npz). The
exporter runs on CPU. It reads the stored weights, tables, and forecasts of the arm read-only,
rebuilds every standardiser from the captured fold state, asserts that the standardisers, the input
matrix, table T, and the quantiles reproduce the stored files, and writes the bundle that
`reference_check.py` reads. It extends experiment 705's bundle format 1 with `components/k1/`,
`components/candidate/`, `fixtures/k1/`, and `fixtures/candidate/`. See the bundle's `README.md`.
"""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00437_features as f437  # ty: ignore[unresolved-import]
import e00558_lags as lags558  # ty: ignore[unresolved-import]
import e00560_data as d560  # ty: ignore[unresolved-import]
import e00572_lags as lg  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00667_extend as ext  # ty: ignore[unresolved-import]
import e00708_reference_check as ref  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from contracts.power_schemas import PowerForecast

HERE: Final = Path(__file__).resolve().parent
QCOLS: Final = ref.LEVEL_NAMES
SEEDS_ALL: Final = (0, 1)
GRID_START_US: Final = int(common.GRID_START.timestamp() * 1_000_000)
STEP_US: Final = 1_800_000_000
B1_NAMES: Final = (*f437.BUNDLES["T"], *f437.BUNDLES["S"])
LAG_NAMES: Final = (
    "day_lag_value", "day_lag_missing", "day_lag_age",
    "week_lag_value", "week_lag_missing", "week_lag_age",
    "day_lag_grid_index", "week_lag_grid_index", "knot_position",
)  # fmt: skip
WEATHER_VARIABLES: Final = prep.WEATHER
CAMS_COLUMNS: Final = ("cams_prev_day_mean", "cams_prev_day_csi")
HISTORY_DAYS: Final = 180
"""Length of the power and table T history in each fixture; the capture builds the dense table over the same length."""
HISTORY_DAYS_CANDIDATES: Final = (30, 60, 90, 120, 150, HISTORY_DAYS)
"""Window lengths whose truncation error is reported; `HISTORY_DAYS` must be under `TRUNCATION_LIMIT`."""
TRUNCATION_LIMIT: Final = 2e-6
"""Largest tolerated difference between a signal column built from the window and from the whole history (columns carry a factor of 10)."""
MAX_LEAD: Final = 48
P7_GROUPS: Final = ("gap", "own:gap,ew1,ew3,ew7", "exp", "nb:gap,ew1,ew3,ew7")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(*, obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False, allow_nan=False) + "\n")


def git_state() -> dict:
    root = HERE.parents[1]
    run = lambda *a: subprocess.run(
        ["git", "-C", str(root), *a], capture_output=True, text=True, check=True
    ).stdout.strip()
    tracked = run("status", "--porcelain", "--untracked-files=no")
    untracked = [
        line[3:]
        for line in run("status", "--porcelain", "--untracked-files=all").splitlines()
        if line.startswith("??")
    ]
    here = run("status", "--porcelain", "--untracked-files=all", "--", str(HERE))
    return {
        "dirty_files_under_this_experiment": len(here.splitlines()),
        "commit": run("rev-parse", "HEAD"),
        "dirty_tracked_files": len(tracked.splitlines()),
        "untracked_files": len(untracked),
        "untracked_files_under_ml_experiments_or_src": sum(
            p.startswith(("ml-experiments/", "src/")) for p in untracked
        ),
        "dirty": bool(tracked or untracked),
    }


def column_table(*, names37: list[str]) -> list[dict]:
    groups = (
        [(n, "weather_mean") for n in names37[:7]]
        + [(n, "weather_std") for n in names37[7:14]]
        + [(n, "sun") for n in names37[14:17]]
        + [(n, "calendar") for n in names37[17:29]]
        + [(n, "holiday") for n in names37[29:35]]
        + [(n, "lead") for n in names37[35:37]]
    )
    assert len(groups) == 37
    out = [
        {"index": i, "name": n, "group": g, "origin": "37-column lag-free matrix"}
        for i, (n, g) in enumerate(groups)
    ]
    lag_groups = ["day_lag"] * 3 + ["week_lag"] * 3 + ["lag_grid_index"] * 2 + ["knot_position"]
    out += [
        {"index": 37 + i, "name": n, "group": g, "origin": "lag block / knot position"}
        for i, (n, g) in enumerate(zip(LAG_NAMES, lag_groups, strict=True))
    ]
    out += [
        {"index": 46 + i, "name": n, "group": "b1", "origin": "experiment 437 table"}
        for i, n in enumerate(B1_NAMES)
    ]
    assert len(out) == 60
    return out


def primaries_table() -> list[dict]:
    series = prep.primaries()
    site_map = k578.sites().rename({"label": "d_label"})
    merged = series.join(
        site_map.select("time_series_id", "capacity_mw", "group"),
        on="time_series_id",
        how="left",
        suffix="_map",
    )
    out = []
    for r in merged.sort("primary").iter_rows(named=True):
        assert abs(float(r["effective_capacity_mw"]) - float(r["capacity_mw"])) < 1e-5, r
        out.append(
            {
                "primary_index": int(r["primary"]),
                "time_series_id": int(r["time_series_id"]),
                "d_label": r["site"],
                "unit": r["units"],
                "h3_res_5": int(r["h3_res_5"]),
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "effective_capacity_mw": float(r["effective_capacity_mw"]),
                "capacity_mw": float(r["capacity_mw"]),
                "group": r["group_map"],
            }
        )
    return out


def weights_index(*, folder: Path, seeds: list[int], source: dict[int, Path], label: str) -> int:
    """Copy the seeds' weights into `folder`, write the index, and return the number of offset knots."""
    index = {}
    n_knots = None
    (folder / "weights").mkdir(parents=True)
    for seed in seeds:
        dst = folder / "weights" / f"seed{seed}.pt"
        shutil.copyfile(source[seed], dst)
        state = torch.load(dst, map_location="cpu", weights_only=True)
        index[f"seed{seed}"] = {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "numel": v.numel(),
                "used_at_inference": not k.startswith(("mixer_pre", "head.")),
            }
            for k, v in state.items()
        }
        n_knots = state["knots_old"].shape[1] + 1
    write_json(
        obj={
            "class": label,
            "unused_at_inference": "mixer_pre.*, head.* (auxiliary depth head)",
            "files": index,
        },
        path=folder / "weights" / "weights_index.json",
    )
    assert n_knots is not None
    return n_knots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--chain-work", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS_ALL))
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="write a bundle although files of this experiment are uncommitted",
    )
    args = parser.parse_args()
    out = Path(args.out).resolve()
    assert out.parent.parent.parent.name.startswith("00708-"), (
        f"--out must be <00708 folder>/bundles/<arm>/<month>: {out}"
    )
    if out.exists():
        shutil.rmtree(out)
    k1_dir, cand_dir = out / "components" / "k1", out / "components" / "candidate"
    for sub in (k1_dir, cand_dir, out / "fixtures" / "k1", out / "fixtures" / "candidate"):
        sub.mkdir(parents=True)
    git = git_state()
    if git["dirty_files_under_this_experiment"] and not args.allow_dirty:
        msg = f"{git['dirty_files_under_this_experiment']} files of experiment 708 are uncommitted: commit them, or pass --allow-dirty for a test bundle"
        raise SystemExit(msg)
    z = np.load(args.dump)
    source = Path(args.source_root)
    chain = Path(args.chain_work)
    month = args.month
    seeds = list(args.seeds)
    refit = datetime.fromisoformat(month + "-01").replace(tzinfo=UTC)
    primaries = primaries_table()
    cap = np.array([p["capacity_mw"] for p in primaries])
    checks: dict = {}

    # --- standardisers --------------------------------------------------------------------------
    keep, names37, mask = data.non_lag_layout()
    assert len(names37) == 37
    std_mean, std_std = z["std_mean"], z["std_std"]
    assert np.array_equal(mask, z["std_mask"])
    assert (std_std >= 1e-6).all()
    stats_path = chain / "stats" / "B1" / f"{month}.npz"
    b1_file = np.load(stats_path)
    assert np.array_equal(b1_file["mean"], z["b1_mean"]) and np.array_equal(
        b1_file["std"], z["b1_std"]
    ), "captured B1 statistics differ from the stored file"
    lag_stats = z["lag_stats"]
    standardisers = {
        "x37": {
            "names": names37,
            "mean": [float(v) for v in std_mean],
            "std": [float(v) for v in std_std],
            "standardised": [bool(v) for v in mask],
            "dtype": "float32",
            "rule": "(raw - mean) / std in float32 (the means and stds are float32 numbers), clip to +-5, NaN to 0, cast to float32; columns not standardised have mean 0 and std 1 but are still clipped",
            "source": "e00428_data.build_fold: prep.Standardiser.fit on the fold's training rows (NaN-aware, population std, std floored at 1e-6)",
        },
        "lag": {
            "day": {"mean": float(lag_stats[0, 0]), "std": float(lag_stats[0, 1])},
            "week": {"mean": float(lag_stats[1, 0]), "std": float(lag_stats[1, 1])},
            "rule": "z = clip((power / effective_capacity - mean) / std, -5, 5); missing lag: value 0, flag 1, age code = missing_age_code",
            "source": "e00572_lags.lag_stats on the fold's training rows; float64",
        },
        "b1": {
            "names": list(B1_NAMES),
            "mean": [float(v) for v in z["b1_mean"]],
            "std": [float(v) for v in z["b1_std"]],
            "rule": "(raw - mean) / std in float32, non-finite to 0, no clip",
            "source": "e00569_columns.Extras.scale; statistics fitted on the fold's pre-training rows and stored in stats/B1/<month>.npz of the chain",
            "stats_file": str(stats_path),
        },
        "target": {
            "primary_index": list(range(16)),
            "time_series_id": [p["time_series_id"] for p in primaries],
            "mean": [float(v) for v in z["target_mean"]],
            "std": [float(v) for v in z["target_std"]],
            "rule": "network target = (power / effective_capacity - mean) / std per Primary; forecast fraction = z * std + mean",
            "std_floor": 1e-3,
            "source": "e00428_data.build_fold: mean and std of the training rows' power / effective capacity",
        },
        "expected_table": {
            "rule": "float32 arithmetic; the 28 expected-weather columns use the x37 mean and std at x37_positions, clipped to +-5, NaN to 0, stored as float16",
        },
        "shared_by": "K1 and the Candidate read the same standardisers: K1's step-2 fit and the Candidate's fine-tune use one fold standardiser",
    }

    # --- input matrix: reproduce the 60 base columns from the raw ones ---------------------------------
    x = z["x_base"]
    raw37 = z["raw37"]
    rebuilt = np.nan_to_num(np.clip((raw37 - std_mean) / std_std, -5.0, 5.0), nan=0.0).astype(
        np.float32
    )
    assert np.array_equal(rebuilt, x[:, :37]), (
        "x37 does not reproduce from the raw matrix and the standardiser"
    )
    primary, origin, valid, target_idx = z["primary"], z["origin"], z["valid"], z["target_idx"]
    clean = z["clean"]
    amax = lg.age_max(cap_steps=lg.CAP_24H)
    lag_raw = np.full((len(x), 2), np.nan, dtype=np.float32)
    lag_idx = np.stack([x[:, 43], x[:, 44]], axis=1).astype(np.int64)
    rebuilt_lag = np.zeros((len(x), 8), dtype=np.float32)
    for j in range(2):
        idx = lag_idx[:, j]
        raw = np.where(idx >= 0, clean[primary, np.clip(idx, 0, None)], np.nan)
        lag_raw[:, j] = raw
        miss = ~np.isfinite(raw)
        zz = np.clip((raw - lag_stats[j, 0]) / lag_stats[j, 1], -5.0, 5.0)
        rebuilt_lag[:, 3 * j] = np.where(miss, 0.0, zz)
        rebuilt_lag[:, 3 * j + 1] = miss
        rebuilt_lag[:, 3 * j + 2] = np.where(miss, amax[j], (target_idx - idx) * 0.5 / 336.0)
        rebuilt_lag[:, 6 + j] = idx
    assert np.array_equal(rebuilt_lag, x[:, 37:45]), (
        "lag columns do not reproduce from the cleaned power"
    )
    tau = x[:, 45]
    t = valid.astype("datetime64[us]")
    start = t.astype("datetime64[M]")
    frac = (t - start.astype("datetime64[us]")) / (
        (start + np.timedelta64(1, "M")).astype("datetime64[us]") - start.astype("datetime64[us]")
    )
    tau_again = ((start - np.datetime64("2019-09", "M")).astype(np.float64) + frac).astype(
        np.float32
    )
    assert np.array_equal(tau, tau_again), "knot position does not reproduce"
    b1_std = ((z["b1_raw"] - z["b1_mean"]) / z["b1_std"]).astype(np.float32)
    b1_std = np.where(np.isfinite(b1_std), b1_std, 0.0).astype(np.float32)
    assert np.array_equal(b1_std, x[:, 46:60]), "B1 columns do not reproduce"
    checks["input_matrix_base"] = (
        "all 60 base columns reproduce exactly (x37, lags, knot position, B1) from raw columns, standardisers, and cleaned power"
    )

    # --- weights ----------------------------------------------------------------------------------
    n_knots = weights_index(
        folder=cand_dir, seeds=seeds,
        source={s: source / "work" / "models" / "P5" / f"{month}_d0_s{s}.pt" for s in seeds},
        label="e00613_nbmodel.PermLagMdn(n_extra=60, no bottleneck, no label head) over e00591_model.AuxLagMdn(kind='shared', n_b1=14), stage 'ft'; state dict saved with torch.save(model.state_dict())",
    )  # fmt: skip
    n_knots_k1 = weights_index(
        folder=k1_dir, seeds=seeds,
        source={s: Path(args.chain_work) / "step2" / "P5" / f"{month}_s{s}.pt" for s in seeds},
        label="e00591_model.AuxLagMdn(kind='shared', n_b1=14), stage 'ft'; K1's step-2 weights; state dict saved with torch.save(model.state_dict())",
    )  # fmt: skip
    assert n_knots == int(z["knots"]) == n_knots_k1, (n_knots, int(z["knots"]), n_knots_k1)
    knot_dates = [
        f"{(2019 * 12 + 8 + i) // 12}-{(2019 * 12 + 8 + i) % 12 + 1:02d}-01" for i in range(n_knots)
    ]
    assert knot_dates[-1] == f"{month}-01", knot_dates[-1]
    knots_obj = {
        "n_knots": n_knots,
        "first_knot_month": "2019-09",
        "knot_dates": knot_dates,
        "rule": "knot k is the first of month 2019-09 + k months; knots_old holds knots 0 .. n-2 (pre-trained, frozen), knot_new holds knot n-1 (trained at this refit); the position of a valid time is months since 2019-09 plus the fraction of its month, clamped to [0, n-1]; linear interpolation per Primary; the offset is added to the three component means in z units",
    }
    write_json(obj=knots_obj, path=k1_dir / "offset_knots.json")
    write_json(obj=knots_obj, path=cand_dir / "offset_knots.json")

    # --- expected-weather rows --------------------------------------------------------------------
    need, table_f16, table_origin = z["table_need"], z["table_std_f16"], z["table_origin"]
    cols_raw = np.load(
        ext.OUT_ROOT / "inputs" / "w560" / "expected" / "expected_ft_cols.npy", mmap_mode="r"
    )
    origin_tab = np.load(
        ext.OUT_ROOT / "inputs" / "w560" / "expected" / "expected_ft_origin.npy", mmap_mode="r"
    )
    raw_rows = np.array(cols_raw[need[:, 0], need[:, 1]], dtype=np.float32)
    assert np.array_equal(np.asarray(origin_tab[need[:, 0], need[:, 1]]), table_origin)
    pos = d560.EXPECTED_37
    mean_e, std_e = std_mean[pos], std_std[pos]
    again16 = np.nan_to_num(np.clip((raw_rows - mean_e) / std_e, -5.0, 5.0), nan=0.0).astype(
        np.float16
    )
    assert np.array_equal(again16, table_f16), (
        "expected-weather rows do not reproduce from raw rows and the standardiser"
    )
    exp_names = [names37[p] for p in pos]
    grid_time = pl.Series(
        "grid_time", (GRID_START_US + need[:, 1].astype(np.int64) * STEP_US)
    ).cast(pl.Datetime("us", "UTC"))
    run_origin = pl.Series("run_origin_time", table_origin).cast(pl.Datetime("us", "UTC"))
    expected_frame = pl.DataFrame({"primary_index": need[:, 0].astype(np.int8), "grid_index": need[:, 1].astype(np.int64)}).with_columns(
        grid_time, run_origin,
        *[pl.Series(f"raw__{n}", raw_rows[:, i]) for i, n in enumerate(exp_names)],
        *[pl.Series(f"std_f16__{n}", table_f16[:, i].astype(np.float32)) for i, n in enumerate(exp_names)],
    )  # fmt: skip
    gt = need[:, 1].astype(np.int64) * STEP_US + GRID_START_US
    assert (table_origin[table_origin > 0] <= gt[table_origin > 0]).all(), (
        "expected-weather run issued after the half-hour"
    )
    expected_frame.write_parquet(k1_dir / "expected_weather_rows.parquet", compression="zstd")
    present = {
        (int(p), int(i)): bool(o > 0)
        for p, i, o in zip(need[:, 0], need[:, 1], table_origin, strict=True)
    }

    # --- stored forecasts and tables ---------------------------------------------------------------
    stored_q = {}
    for seed in seeds:
        st = pl.read_parquet(
            source / "work" / "fits" / "P5" / f"{month}_d0_s{seed}.parquet"
        ).with_columns(
            o=pl.col("power_fcst_init_time").dt.epoch("us"), v=pl.col("valid_time").dt.epoch("us")
        )
        mine = pl.DataFrame(
            {
                "time_series_id": z[f"key_series_s{seed}"],
                "o": z[f"key_origin_s{seed}"],
                "v": z[f"key_valid_s{seed}"],
            },
        ).hstack(pl.DataFrame(z[f"q_all_s{seed}"], schema=list(QCOLS)))
        j = mine.join(st, on=["time_series_id", "o", "v"], suffix="_stored", how="inner")
        assert j.height == mine.height == st.height, (j.height, mine.height, st.height)
        diff = max(
            float(
                np.abs(
                    j[c].to_numpy().astype(np.float64)
                    - j[c + "_stored"].to_numpy().astype(np.float64)
                ).max()
            )
            for c in QCOLS
        )
        checks[f"rerun_vs_stored_forecast_seed{seed}"] = {
            "rows": j.height,
            "max_abs_diff_fraction": diff,
        }
        assert diff == 0.0, f"seed {seed}: the capture differs from the stored forecast by {diff}"
        stored_q[seed] = st
    tables = {}
    for seed in seeds:
        mine_t = np.load(Path(args.scratch) / "tables" / "predict" / f"T_{month}_s{seed}.npz")
        stored_t = np.load(source.parent / "tables" / "predict" / f"T_{month}_s{seed}.npz")
        same_median = np.array_equal(mine_t["median"], stored_t["median"], equal_nan=True)
        same_origin = np.array_equal(mine_t["origin_idx"], stored_t["origin_idx"])
        finite = np.isfinite(mine_t["median"]) & np.isfinite(stored_t["median"])
        checks[f"table_t_rerun_vs_stored_seed{seed}"] = {
            "median_identical": bool(same_median),
            "source_origin_identical": bool(same_origin),
            "max_abs_diff": float(
                np.abs(mine_t["median"][finite] - stored_t["median"][finite]).max()
            ),
            "filled_entries": int(np.isfinite(mine_t["median"]).sum()),
            "grid_columns": int(mine_t["median"].shape[1]),
        }
        assert same_median and same_origin, (
            f"seed {seed}: the rerun of table T differs from the stored table"
        )
        tables[seed] = mine_t

    # --- dense table T, and the signal-window length ----------------------------------------------------
    signal_names = [str(n) for n in z["signal_names"]]
    n_extra = len(signal_names)
    nb_of: dict[int, list[int]] = {p: [] for p in range(16)}
    for p, q in z["nb_of_flat"]:
        nb_of[int(p)].append(int(q))
    assert np.array_equal(z["nb_capacity"], cap), (
        "neighbour capacities differ from the capacities of the Primaries"
    )
    neighbours = {
        "of": {str(p): qs for p, qs in nb_of.items()},
        "capacity_mw": [float(c) for c in cap],
    }
    cfg_probe = {"signals": {"ew_min_weight": 0.2}}
    origins_us = np.unique(origin)
    dense_median = {s: z[f"dense_t_median_s{s}"] for s in seeds}
    dense_origin = {s: z[f"dense_t_origin_idx_s{s}"] for s in seeds}
    extras = {s: z[f"extras_dense_s{s}"] for s in seeds}
    for s in seeds:
        assert z[f"x_s{s}"].shape[1] == 60 + n_extra
    meta = json.loads(Path(args.dump).with_suffix(".json").read_text())
    checks["capture"] = {
        k: meta[k]
        for k in (
            "sparse_replication",
            "dense_vs_research_table",
            "dense_months",
            "dense_origin_range_idx",
            "synthetic_case",
            "synthetic_flag_counts",
        )
    }
    assert meta["history_days"] == HISTORY_DAYS

    def window_signals(*, sel: np.ndarray, seed: int, days: int) -> np.ndarray:
        last = int(z["origin_idx"][sel][0]) - 48
        first = last - days * 48 + 1
        return ref.signal_block(
            primary=primary[sel], day_idx=lag_idx[sel, 0], week_idx=lag_idx[sel, 1],
            day_miss=x[sel][:, 38] > 0.5, week_miss=x[sel][:, 41] > 0.5,
            power=clean[:, first : last + 1], table_median=dense_median[seed][:, first : last + 1],
            first_idx=first, present=present, neighbours=neighbours, names=signal_names,
            config=cfg_probe,
        )  # fmt: skip

    truncation = {}
    for days in HISTORY_DAYS_CANDIDATES:
        err = 0.0
        for o_us in origins_us:
            sel = np.nonzero(origin == o_us)[0]
            for seed in seeds:
                diff = window_signals(sel=sel, seed=seed, days=days) - extras[seed][sel]
                err = max(err, float(np.abs(diff).max()))
        truncation[str(days)] = err
        print("history days", days, "max abs error of the signal columns", err, flush=True)
    history_days = HISTORY_DAYS
    assert truncation[str(history_days)] <= TRUNCATION_LIMIT, truncation
    checks["signal_history_truncation_max_abs_error"] = truncation
    checks["signal_history_days_used"] = history_days

    names60 = [c["name"] for c in column_table(names37=names37)]
    # --- K1 history rows: every daily origin of the window, shared by the fixtures ------------------------
    d_k = out / "fixtures" / "k1"
    d_prim = z["dense_primary"]
    d_x = z["dense_x"]
    assert d_x.shape[1] == 60
    k1_frame = pl.DataFrame(
        {
            "primary_index": d_prim.astype(np.int8),
            "origin_grid_index": z["dense_origin_idx"],
            "target_grid_index": z["dense_target_idx"],
        }
    ).with_columns(
        origin_time=(GRID_START_US + pl.col("origin_grid_index") * STEP_US).cast(
            pl.Datetime("us", "UTC")
        ),
        valid_time=(GRID_START_US + pl.col("target_grid_index") * STEP_US).cast(
            pl.Datetime("us", "UTC")
        ),
    )
    k1_frame = k1_frame.hstack(
        pl.DataFrame({f"std__{n}": d_x[:, i] for i, n in enumerate(names60)})
    )
    for s in seeds:
        qh = z[f"dense_k1_q_s{s}"]
        k1_frame = k1_frame.hstack(
            pl.DataFrame({f"k1_{lv}_s{s}": qh[:, i] for i, lv in enumerate(("q05", "q50", "q95"))})
        )
    k1_frame.write_parquet(d_k / "history_rows.parquet", compression="zstd")
    checks["k1_history_rows"] = {
        "rows": k1_frame.height,
        "daily_origins": len(np.unique(z["dense_origin_idx"])),
        "first_origin_utc": f"{k1_frame['origin_time'].min():%Y-%m-%d %H:%M}",
        "last_origin_utc": f"{k1_frame['origin_time'].max():%Y-%m-%d %H:%M}",
        "inside_the_fit_period": "every origin before the refit month is inside K1's training period (in-sample, except in held-out weeks)",
    }

    # --- fixtures -----------------------------------------------------------------------------------
    ts_of = np.array([p["time_series_id"] for p in primaries])
    contract_schema = {
        "time_series_id": str(PowerForecast.dtypes["time_series_id"]),
        "valid_time": str(PowerForecast.dtypes["valid_time"]),
        "nwp_init_time": str(PowerForecast.dtypes["nwp_init_time"]),
        "power_fcst_init_time": str(PowerForecast.dtypes["power_fcst_init_time"]),
        **{f"{lv}_mw": "Float32" for lv in QCOLS},
        **{f"{lv}_s{seed}_mw": "Float32" for seed in seeds for lv in QCOLS},
    }
    v_sel = z["v_sel"]
    t_blank = z["t_blank"]
    weather_blank = z["weather_blank"]

    def write_candidate(
        *, name: str, sel: np.ndarray, x_rows: np.ndarray, clean_arr: np.ndarray,
        extras_rows: dict[int, np.ndarray], raw_out: dict[int, np.ndarray],
        q_rows: dict[int, np.ndarray], blanks: bool,
    ) -> dict:  # fmt: skip
        """Write one candidate fixture folder (a fixture origin, or the synthetic case) and return its summary."""
        o_us = int(origin[sel][0])
        o_dt = datetime.fromtimestamp(o_us / 1e6, UTC)
        d_c = out / "fixtures" / "candidate" / f"origin={name}"
        d_c.mkdir(parents=True)
        o_idx = int(z["origin_idx"][sel][0])
        prim_rows = primary[sel]
        run = o_dt - common.PUBLICATION_DELAY
        stored_summary = pl.read_parquet(
            ext.E607 / "ens_summary" / f"init={run:%Y-%m-%d}.parquet"
        ).sort("time_series_id", "valid_time")
        key = pl.DataFrame(
            {"time_series_id": ts_of[prim_rows], "valid_time": valid[sel]}
        ).with_columns(pl.col("valid_time").cast(pl.Datetime("us", "UTC")))
        joined = key.join(
            stored_summary, on=["time_series_id", "valid_time"], how="left", maintain_order="left"
        )
        assert joined["mean_shortwave"].is_not_null().all()
        expected14 = np.empty((joined.height, 14), dtype=np.float64)
        for k, kind in enumerate(("mean", "std")):
            for j, w in enumerate(WEATHER_VARIABLES):
                v = joined[f"{kind}_{w}"].to_numpy().astype(np.float64)
                expected14[:, 7 * k + j] = (
                    np.log1p(np.clip(v, 0, None)) if w == "precipitation" else v
                )
        ens_diff = float(np.abs(expected14 - raw37[sel][:, :14].astype(np.float64)).max())
        stored_summary.write_parquet(d_c / "ens_summary.parquet", compression="zstd")
        # the lag columns reproduce from the (possibly modified) power
        lag_i = np.stack([x_rows[:, 43], x_rows[:, 44]], axis=1).astype(np.int64)
        lag_vals = np.full((len(sel), 2), np.nan, dtype=np.float32)
        rebuilt_lag_rows = np.zeros((len(sel), 8), dtype=np.float32)
        for j in range(2):
            idx = lag_i[:, j]
            raw = np.where(idx >= 0, clean_arr[prim_rows, np.clip(idx, 0, None)], np.nan)
            lag_vals[:, j] = raw
            miss = ~np.isfinite(raw)
            zz = np.clip((raw - lag_stats[j, 0]) / lag_stats[j, 1], -5.0, 5.0)
            rebuilt_lag_rows[:, 3 * j] = np.where(miss, 0.0, zz)
            rebuilt_lag_rows[:, 3 * j + 1] = miss
            rebuilt_lag_rows[:, 3 * j + 2] = np.where(
                miss, amax[j], (target_idx[sel] - idx) * 0.5 / 336.0
            )
            rebuilt_lag_rows[:, 6 + j] = idx
        assert np.array_equal(rebuilt_lag_rows, x_rows[:, 37:45]), f"{name}: lag columns differ"
        raw60 = np.concatenate(
            [raw37[sel], lag_vals[:, [0]], np.zeros((len(sel), 2), np.float32), lag_vals[:, [1]],
             np.zeros((len(sel), 2), np.float32), x_rows[:, 43:46], z["b1_raw"][sel]],
            axis=1,
        )  # fmt: skip
        raw60[:, 38], raw60[:, 39] = x_rows[:, 38], x_rows[:, 39]
        raw60[:, 41], raw60[:, 42] = x_rows[:, 41], x_rows[:, 42]
        keys = pl.DataFrame(
            {
                "primary_index": prim_rows.astype(np.int8),
                "time_series_id": ts_of[prim_rows].astype(np.int32),
                "origin_time": origin[sel],
                "valid_time": valid[sel],
                "lead_hours": z["lead"][sel].astype(np.float64),
                "target_grid_index": target_idx[sel],
                "origin_grid_index": z["origin_idx"][sel],
            }
        ).with_columns(
            pl.col("origin_time").cast(pl.Datetime("us", "UTC")),
            pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        )
        matrix = (
            keys.hstack(pl.DataFrame({f"raw__{n}": raw60[:, i] for i, n in enumerate(names60)}))
            .hstack(pl.DataFrame({f"std__{n}": x_rows[:, i] for i, n in enumerate(names60)}))
            .hstack(
                pl.DataFrame(
                    {
                        f"extra_s{s}__{n}": extras_rows[s][:, i]
                        for s in seeds
                        for i, n in enumerate(signal_names)
                    }
                )
            )
        )
        matrix.write_parquet(d_c / "input_matrix.parquet", compression="zstd")
        # lag power and expected-weather rows of the 336-half-hour lag window
        window = np.arange(o_idx - 48 - 335, o_idx - 48 + 1)
        used = np.unique(lag_i)
        assert set(used[used >= 0]) <= set(window.tolist()), (
            "a lag lies outside the 336-half-hour window"
        )
        grid = pl.DataFrame(
            {
                "primary_index": np.repeat(np.arange(16, dtype=np.int8), len(window)),
                "grid_index": np.tile(window, 16),
            }
        )
        power = clean_arr[
            grid["primary_index"].to_numpy().astype(np.int64), grid["grid_index"].to_numpy()
        ]
        grid = grid.with_columns(
            pl.Series("power_fraction", power),
            pl.Series("grid_time", GRID_START_US + grid["grid_index"].to_numpy() * STEP_US).cast(
                pl.Datetime("us", "UTC")
            ),
        ).join(
            expected_frame.drop("grid_time"),
            on=["primary_index", "grid_index"],
            how="left",
            validate="1:1",
        )
        assert grid["run_origin_time"].is_not_null().all()
        grid.write_parquet(d_c / "lag_and_expected_rows.parquet", compression="zstd")
        # power and the dense table T over the history window
        last = o_idx - 48
        first = last - history_days * 48 + 1
        hist_grid = np.arange(first, last + 1)
        t_med = {s: dense_median[s].copy() for s in seeds}
        t_org = {s: dense_origin[s].copy() for s in seeds}
        if blanks:
            for s in seeds:
                for p, a, b in t_blank:
                    t_med[s][p, a:b] = np.nan
                    t_org[s][p, a:b] = -1
        history = pl.DataFrame(
            {
                "primary_index": np.repeat(np.arange(16, dtype=np.int8), len(hist_grid)),
                "grid_index": np.tile(hist_grid, 16),
            }
        ).with_columns(
            pl.Series("power_fraction", clean_arr[:, first : last + 1].reshape(-1)),
            *[
                pl.Series(f"t_dense_median_s{s}", t_med[s][:, first : last + 1].reshape(-1))
                for s in seeds
            ],
            *[
                pl.Series(
                    f"t_dense_origin_grid_index_s{s}", t_org[s][:, first : last + 1].reshape(-1)
                )
                for s in seeds
            ],
        )
        history.write_parquet(d_c / "signal_history.parquet", compression="zstd")
        if blanks:
            pl.DataFrame(
                {
                    "primary_index": [int(p) for p, _, _ in t_blank],
                    "first_grid_index": [int(a) for _, a, _ in t_blank],
                    "end_grid_index": [int(b) for _, _, b in t_blank],
                }
            ).write_parquet(d_c / "t_blank.parquet")
            blank_keys = [(int(p), i) for p, a, b in weather_blank for i in range(int(a), int(b))]
            pl.DataFrame(
                {
                    "primary_index": [p for p, _ in blank_keys],
                    "grid_index": [i for _, i in blank_keys],
                    "run_origin_time": [datetime(1970, 1, 1, tzinfo=UTC)] * len(blank_keys),
                    **{f"std_f16__{n}": np.zeros(len(blank_keys), np.float32) for n in exp_names},
                }
            ).with_columns(pl.col("run_origin_time").cast(pl.Datetime("us", "UTC"))).write_parquet(
                d_c / "expected_rows_override.parquet"
            )
        # outputs: native per seed, seed mean, contract
        out_native = pl.DataFrame(
            {"primary_index": prim_rows.astype(np.int8), "valid_time": valid[sel]}
        ).with_columns(pl.col("valid_time").cast(pl.Datetime("us", "UTC")))
        per_seed = [q_rows[s] for s in seeds]
        for seed in seeds:
            out_native = out_native.hstack(
                pl.DataFrame({f"{lv}_s{seed}": q_rows[seed][:, i] for i, lv in enumerate(QCOLS)})
            ).hstack(pl.DataFrame({f"raw_out_{k}_s{seed}": raw_out[seed][:, k] for k in range(9)}))
        mean_q = np.sort(np.mean(np.stack(per_seed), axis=0, dtype=np.float32), axis=1)
        out_native = out_native.hstack(
            pl.DataFrame({f"{lv}_mean": mean_q[:, i] for i, lv in enumerate(QCOLS)})
        )
        out_native.write_parquet(d_c / "outputs_native.parquet", compression="zstd")
        contract_origin = o_us + 3 * 3_600_000_000
        cmask = (valid[sel] >= contract_origin) & (
            valid[sel] < contract_origin + 14 * 86_400_000_000
        )
        assert int(cmask.sum()) == 16 * 672
        mw = [(q.astype(np.float64) * cap[prim_rows][:, None]).astype(np.float32) for q in per_seed]
        mw_mean = np.sort(np.mean(np.stack(mw), axis=0, dtype=np.float32), axis=1)
        contract = (
            pl.DataFrame(
                {
                    "time_series_id": ts_of[prim_rows][cmask],
                    "valid_time": valid[sel][cmask],
                    "nwp_init_time": (contract_origin // 86_400_000_000) * 86_400_000_000,
                    "power_fcst_init_time": contract_origin,
                }
            )
            .with_columns(
                pl.col("time_series_id").cast(PowerForecast.dtypes["time_series_id"]),
                pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
                pl.col("nwp_init_time").cast(pl.Datetime("us", "UTC")),
                pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
            )
            .hstack(pl.DataFrame({f"{lv}_mw": mw_mean[cmask][:, i] for i, lv in enumerate(QCOLS)}))
            .hstack(
                pl.DataFrame(
                    {
                        f"{lv}_s{seed}_mw": mw[k][cmask][:, i]
                        for k, seed in enumerate(seeds)
                        for i, lv in enumerate(QCOLS)
                    }
                )
            )
        )
        assert {c: str(t) for c, t in contract.schema.items()} == contract_schema, contract.schema
        contract.write_parquet(d_c / "outputs_contract.parquet", compression="zstd")
        return {
            "origin": name,
            "weekday": o_dt.strftime("%A"),
            "rows": len(sel),
            "week_lag_missing_rows": int((x_rows[:, 41] > 0.5).sum()),
            "day_lag_missing_rows": int((x_rows[:, 38] > 0.5).sum()),
            "ens_summary_vs_network_raw_columns_max_abs_diff": ens_diff,
            "dense_table_t_empty_share_in_history_window": float(
                np.mean([np.isnan(t_med[s][:, first : last + 1]).mean() for s in seeds])
            ),
            "research_table_t_empty_share_in_history_window": float(
                np.mean([np.isnan(tables[s]["median"][:, first : last + 1]).mean() for s in seeds])
            ),
        }

    fixture_origins = []
    for o_us in origins_us:
        sel = np.nonzero(origin == o_us)[0]
        name = f"{datetime.fromtimestamp(int(o_us) / 1e6, UTC):%Y-%m-%d}"
        fixture_origins.append(name)
        info = write_candidate(
            name=name, sel=sel, x_rows=x[sel], clean_arr=clean, blanks=False,
            extras_rows={s: extras[s][sel] for s in seeds},
            raw_out={s: z[f"raw_out_dense_s{s}"][sel] for s in seeds},
            q_rows={s: z[f"q_fix_dense_s{s}"][sel] for s in seeds},
        )  # fmt: skip
        # what the Candidate does with the research table (the table it trained on) instead of the dense one
        sparse_q = [z[f"q_fix_s{s}"][sel] for s in seeds]
        dense_q = [z[f"q_fix_dense_s{s}"][sel] for s in seeds]
        sparse_mean = np.sort(np.mean(np.stack(sparse_q), axis=0, dtype=np.float32), axis=1)
        dense_mean = np.sort(np.mean(np.stack(dense_q), axis=0, dtype=np.float32), axis=1)
        info["sparse_vs_dense_table_t"] = {
            "per_seed_max_abs_diff_fraction_of_capacity": [
                float(np.abs(a - b).max()) for a, b in zip(sparse_q, dense_q, strict=True)
            ],
            "per_seed_mean_abs_diff_fraction_of_capacity": [
                float(np.abs(a - b).mean()) for a, b in zip(sparse_q, dense_q, strict=True)
            ],
            "seed_mean_max_abs_diff_fraction_of_capacity": float(
                np.abs(sparse_mean - dense_mean).max()
            ),
            "seed_mean_mean_abs_diff_fraction_of_capacity": float(
                np.abs(sparse_mean - dense_mean).mean()
            ),
            "signal_columns_max_abs_diff": [
                float(np.abs(extras[s][sel] - z[f"x_s{s}"][sel][:, 60:]).max()) for s in seeds
            ],
        }
        sp = pl.DataFrame({"primary_index": primary[sel].astype(np.int8), "valid_time": valid[sel]})
        sp = sp.with_columns(pl.col("valid_time").cast(pl.Datetime("us", "UTC")))
        for s, q in zip(seeds, sparse_q, strict=True):
            sp = sp.hstack(pl.DataFrame({f"{lv}_s{s}": q[:, i] for i, lv in enumerate(QCOLS)}))
        sp = sp.hstack(
            pl.DataFrame({f"{lv}_mean": sparse_mean[:, i] for i, lv in enumerate(QCOLS)})
        )
        sp.write_parquet(
            out
            / "fixtures"
            / "candidate"
            / f"origin={name}"
            / "outputs_with_research_table_t.parquet",
            compression="zstd",
        )
        checks[f"fixture_{name}"] = info
        print(info, flush=True)
    synthetic_name = f"{fixture_origins[1]}_synthetic_missing_inputs"
    clean_mod = z["v_clean"]
    info = write_candidate(
        name=synthetic_name, sel=v_sel, x_rows=z["v_x"], clean_arr=clean_mod, blanks=True,
        extras_rows={s: z[f"v_extras_s{s}"] for s in seeds},
        raw_out={s: z[f"v_raw_out_s{s}"] for s in seeds},
        q_rows={s: z[f"v_q_s{s}"] for s in seeds},
    )  # fmt: skip
    info["case"] = meta["synthetic_case"]
    info["flag_counts_seed0"] = meta["synthetic_flag_counts"]
    checks[f"fixture_{synthetic_name}"] = info
    print(info, flush=True)

    # --- config ---------------------------------------------------------------------------------------
    f = f437
    column_slices = {
        "weather": [0, 7], "spread": [7, 14], "datetime": [14, 35], "lead": [35, 37],
        "day_lag": [37, 40], "week_lag": [40, 43], "b1": [46, 60],
    }  # fmt: skip

    def network_config(*, name: str, extra: int) -> dict:
        return {
            "arm": "drop",
            "n_primaries": 16,
            "input_clip_sd": prep.INPUT_CLIP_SD,
            "weather_variables": list(WEATHER_VARIABLES),
            "precipitation_transform": "log1p(clip(value, 0)) on the 51-member mean and on the 51-member population std",
            "member_std_ddof": 0,
            "member_count": 51,
            "lead_scale_hours": 336.0,
            "ens_summary_recipe": "as in experiment 705's bundle (`config.json`, `components.step5.ens_summary_recipe`); the fixtures here hold the member summaries that the research run read, not the native members",
            "lags": {
                "cap_steps": lg.CAP_24H,
                "week_steps": lg.WEEK_STEPS,
                "rule": "day lag = target - 48 * ceil((target - (origin - 48)) / 48); week lag = target - 336 * ceil((target - (origin - 48)) / 336); grid steps of 30 min; the origin is the native 09:00 origin; lag value = cleaned power / effective capacity at the lag's grid index",
                "clip_sd": prep.INPUT_CLIP_SD,
                "age_scale_hours": lags558.LEAD_SCALE_HOURS,
                "age_code": "(target_index - lag_index) * 0.5 / 336",
                "missing_age_code": {"day": float(amax[0]), "week": float(amax[1])},
                "missing_age_rule": "24 * ceil(375 / 24) / 336 and 168 * ceil(375 / 168) / 336 with 375 = 351 + 24",
                "std_floor": 1e-6,
                "no_dropout_at_inference": True,
            },
            "column_slices": column_slices,
            "n_base_columns": 60,
            "n_extra": extra,
            "lag_grid_index_columns": [43, 44],
            "knot_position_column": 45,
            "n_b1": 14,
            "mixture_components": 3,
            "log_std_clamp": [-3.0, 3.0],
            "quantile_levels": list(base.LEVELS),
            "bisection_range": base.Z_LIMIT,
            "bisection_steps": base.BISECTION_STEPS,
            "quantile_dtype": "forward float32; read-out float64; output float32",
            "expected_weather": {
                "x37_positions": list(pos),
                "dtype": "float16",
                "rows_per_origin_and_primary": 336,
                "rule": "for each Primary and half-hour g in (origin - 192 h, origin - 24 h], take the ENS summary row of the run with the smallest lead for valid time g (the freshest run issued before g, one 09:00 native origin per day): its 7 weather means (precipitation log1p) and 21 datetime columns; recompute the 21 datetime columns from g; standardise with the x37 mean and std at x37_positions, clip +-5, round to float16; the lag encoder reads the row at each lag's own grid index, with the 14 B1 slots set to 0. No run issued after g is used. A half-hour with no run has run_origin_time at or before 1970-01-01 and a zero row",
                "rows_in_bundle": len(need),
            },
            "b1_constants": {
                "cold_snap_threshold_c": f.COLD_SNAP_THRESHOLD_C,
                "cold_snap_cap_days": f.COLD_SNAP_CAP_DAYS,
                "lookback_steps": f.LOOKBACK_STEPS,
                "min_clear_sky_w_m2": f.MIN_CLEAR_SKY_W_M2,
                "max_clear_sky_index": f.MAX_CSI,
                "trailing_windows_steps": [6, 12, 48],
                "cams_columns": list(CAMS_COLUMNS),
                "cams_columns_source": "arm drop of experiment 704: both columns are 0.0 in every stage (pre-training, fine-tune, K1, and the Candidate); the standardiser has mean 0 and std 1e-6 for them, so production must write 0.0 and never a CAMS value",
            },
            "offset_knots": {"first_knot_month": "2019-09", "n_knots": n_knots},
            "training_provenance": {
                "recency_half_life_days": 90,
                "fine_tune_embargo_days": 7,
                "half_weight_days_after_embargo": 7,
                "step2_embargo_days": 3,
                "training_window_days": prep.WINDOW_DAYS,
                "target_cutoff": "valid time <= refit - 3.5 h",
                "held_out": "1 week in 12 (84-day cycle), purge 15 days before and 9 after",
                "seeds": seeds,
                "synthetic_load_shifts": True,
                "component": name,
            },
        }

    candidate_config = network_config(name="candidate", extra=n_extra) | {
        "arm_name": "P7+D50 (experiment 667), arm B of experiment 703",
        "signals": {
            "groups": list(P7_GROUPS),
            "scale": 10.0,
            "clip": 10.0,
            "ew_half_lives_days": [1, 3, 7],
            "ew_rule": "weight of a reading k half-hours old is 0.5 ** (k / (48 * half_life_days)); missing readings carry no weight; the average is NaN where the weights under it are under ew_min_weight of a full gap-free history",
            "ew_min_weight": 0.2,
            "gap": "cleaned power (fraction of effective capacity) minus table T's lags-masked median, at a grid index; NaN where either is missing",
            "neighbour_rule": "per Primary, over all its electrical neighbours q: value of q's quantity at the Primary's lag time times capacity_q / capacity_p; summaries sum, mean, min, max over neighbours with a finite value, plus own value plus the sum; count of neighbours with a finite day-lag gap; none flag where the count is 0",
            "row_rule": "signal columns of a row read gaps only at the row's two lag times and earlier; masked or absent lags give 0 (flags 1); the expected-missing flags are 1 where the lag exists and the expected-weather table has no run for the lag's half-hour",
            "no_training_noise_or_dropout_at_inference": True,
            "history_window_days": history_days,
            "history_window_rule": "the check builds every average from the window [origin - 48 half-hours - history_window_days days, origin - 48 half-hours] only; the research code builds them from the whole history, and the difference is the truncation error in export_checks.signal_history_truncation_max_abs_error",
        },
        "table_t": {
            "max_lead_half_hours": MAX_LEAD,
            "rule": "T[primary, g] = the median (q50) of K1's lags-masked forecast with the largest origin index o for which 0 < g - o <= max_lead_half_hours, over every K1 forecast ever issued; lags-masked = day and week lag columns set to value 0, flag 1, largest age code, lag grid-index columns unchanged",
            "k1_quantile_columns": ["q05", "q50", "q95"],
            "read_by_candidate": "only entries with g <= origin - 48 half-hours, whose source origin is therefore at least 24 h before the Candidate's origin",
            "research_construction": "e00598_signals.build_lag_free_table over the fold's training rows, the fold month's test rows, and the previous month's test rows. That table has whole-day holes (purge windows around held-out weeks), and the Candidate was TRAINED on it",
            "dense_table_in_this_bundle": "the fixtures use the dense table that a daily store would hold: this refit's K1, lags masked, run at every daily 09:00 origin of the history window (fixtures/k1/history_rows.parquet). The Candidate's forecasts with the research table instead are in each fixture's outputs_with_research_table_t.parquet; export_checks.fixture_<origin>.sparse_vs_dense_table_t gives the difference, which is the train/serve skew a port inherits",
        },
    }
    k1_config = network_config(name="k1", extra=0) | {
        "arm_name": "K1: step-2 network of experiment 613's A1 chain (experiment 667's chain), arm B of experiment 703",
        "lags_masked_read_out": {
            "k1_quantile_columns": ["q05", "q50", "q95"],
            "positions_among_11_levels": list(ref.K1_QUANTILE_COLUMNS),
            "rule": "forward with both lags masked, sort the 11 quantiles, scale by the Primary's target std and mean; keep q05, q50, q95",
        },
    }
    config = {
        "shared": {
            "native_origin_hour_utc": 9,
            "publication_delay_hours": common.PUBLICATION_DELAY / timedelta(hours=1),
            "weather_run": "ECMWF ENS 00Z run of the origin day; native origin = run init + 9 h",
            "contract_origin_hour_utc": 12,
            "contract_relabel": "origin + 3 h; keep valid_time in [contract origin, contract origin + 14 days); 672 half-hours; no clamp, no calibration",
            "horizon_days": 14,
            "contract_half_hours": 672,
            "contract_schema": contract_schema,
            "contract_columns": "time_series_id, valid_time, nwp_init_time, power_fcst_init_time, then q01_mw .. q99_mw (seed mean, sorted) and q01_s<seed>_mw .. per seed; one quantile forecast per series and half-hour, no ensemble_member column",
            "max_lead_hours_trained_and_stored": 339.0,
            "max_lead_hours_ens_summaries": 351.0,
            "grid_start_utc": common.GRID_START.isoformat(),
            "grid_step_minutes": 30,
            "timestamp_convention": "period-ending stamps; the solar position and the calendar columns use the stamp itself or stamp - 15 min as documented in columns.json",
            "calendar_timezone": "Europe/London",
            "holidays": {
                "calendar": "holidays.country_holidays('GB', subdiv='ENG')",
                "years": [common.HOLIDAY_YEARS.start, common.HOLIDAY_YEARS.stop - 1],
                "nearest_holiday_clip_days": common.HOLIDAY_CLIP_DAYS,
                "christmas_clip_days": common.CHRISTMAS_CLIP_DAYS,
                "covid_first_day": common.LOCKDOWN_START.date().isoformat(),
                "covid_last_day": common.LOCKDOWN_END.date().isoformat(),
            },
            "solar_position": "pvlib.solarposition.get_solarposition default method, apparent zenith and azimuth at stamp - 15 min, at the Primary's latitude and longitude",
        },
        "components": {"k1": k1_config, "candidate": candidate_config},
        "state_between_daily_runs": {
            "summary": "production must keep four stores: the cleaned power, K1's lags-masked medians, the expected-weather rows, and (to rebuild them after a gap) nothing else. Each reaches back history_window_days days before the Candidate's origin, except the expected-weather rows, which reach back 8 days.",
            "history_window_days": history_days,
            "cleaned_power": "per Primary and half-hour, as a fraction of effective capacity, missing half-hours kept missing; the lags read the 336 half-hours ending 24 h before the origin (the oldest lag is up to 8 days before the origin) and the signal columns read the whole window",
            "k1_forecasts_to_store_per_origin": "q50 of K1's lags-masked forecast for the 48 half-hours after each daily 09:00 origin, per Primary and seed (16 x 48 x 2 numbers per day), with the origin",
            "expected_weather_rows": "per Primary and half-hour g from 8 days before the origin to 24 h before it: the 28 expected-weather columns of the freshest ENS run issued before g (smallest lead for valid time g), with that run's origin time. Both networks' lag encoders read the row at each lag's own half-hour, and the Candidate's expected-missing flag is 1 where the table has no run for that half-hour. Rows are built once, when the half-hour passes, and never change afterwards",
            "k1_must_run_every_day": "K1 runs with its lags masked at every daily origin, on the same input columns as the Candidate (the 60 base columns), even though only its first 48 half-hours are stored",
            "freshness": "the Candidate at origin o reads table T only at half-hours up to o - 48 half-hours, so it needs K1 forecasts from origins at least 2 days old; yesterday's K1 forecast is stored but not read until the day after tomorrow",
            "cold_start": "after a refit or a gap in the store, rebuild the missing days by running K1 over the days' origins with that day's inputs, as fixtures/k1/history_rows.parquet does; entries with no forecast are NaN and the Candidate then sees missing-gap flags",
            "training_saw_a_different_table": "the Candidate was trained on the research table, which has whole-day holes; a daily store is dense. The Candidate's quantiles differ between the two tables by the amounts in export_checks.fixture_<origin>.sparse_vs_dense_table_t",
            "refit_changes_k1": "table T entries come from whichever K1 version issued them; the research rebuilds the whole window with the current K1 (in-sample for the training days), and so does the dense table in this bundle",
        },
    }
    write_json(obj=config, path=out / "config.json")
    write_json(
        obj={"primaries": primaries, "order": "primary_index = row of site_map.csv, D2 .. D22"},
        path=out / "primaries.json",
    )
    columns = column_table(names37=names37)
    write_json(obj=columns, path=k1_dir / "columns.json")
    write_json(obj=columns, path=cand_dir / "columns.json")
    write_json(obj=standardisers, path=k1_dir / "standardisers.json")
    write_json(obj=standardisers, path=cand_dir / "standardisers.json")
    write_json(
        obj={
            "names": signal_names,
            "start_column": 60,
            "order": "columns 60 .. 60 + n_extra - 1 of the Candidate's input matrix, appended after the 60 base columns, in this order (sorted by the research index of each name)",
            "n_extra": n_extra,
        },
        path=cand_dir / "signal_columns.json",
    )
    write_json(obj=neighbours, path=cand_dir / "neighbours.json")
    shutil.copyfile(HERE / "e00708_reference_check.py", out / "reference_check.py")
    shutil.copyfile(HERE / "bundle_readme.md", out / "README.md")

    # --- manifest ---------------------------------------------------------------------------------------
    pins = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "polars", "pvlib", "holidays", "ml_core")
    }
    direct = json.loads(
        importlib.metadata.distribution("ml_core").read_text("direct_url.json") or "{}"
    )
    manifest = {
        "bundle_format_version": "1",
        "network_name": "candidate",
        "arm": "drop",
        "refit_month": month,
        "refit_time_utc": refit.isoformat(),
        "training_cutoff_utc": (refit - timedelta(hours=3.5)).isoformat(),
        "training_cutoff_rule": f"training targets with valid time <= refit time - 3.5 h and origin <= refit time; the {month} refit uses data before {month}-01 00:00 UTC",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "research_git": git,
        "allow_dirty": bool(args.allow_dirty),
        "pinned_production_package_commit": direct["vcs_info"]["commit_id"],
        "library_versions": pins,
        "source_chain_paths": {
            "candidate_models_and_forecasts": str(source),
            "k1_pretraining_and_step2_weights": args.chain_work,
            "b1_statistics": str(stats_path),
            "table_t_stored": str(source.parent / "tables" / "predict"),
        },
        "seeds": seeds,
        "components": {
            "k1": {
                "kind": "demand_network_k1_step2",
                "seeds": seeds,
                "files": {
                    "columns": "components/k1/columns.json",
                    "standardisers": "components/k1/standardisers.json",
                    "offset_knots": "components/k1/offset_knots.json",
                    "expected_weather_rows": "components/k1/expected_weather_rows.parquet",
                    "weights": [f"components/k1/weights/seed{s}.pt" for s in seeds],
                    "weights_index": "components/k1/weights/weights_index.json",
                    "fixtures": "fixtures/k1/",
                },
            },
            "candidate": {
                "kind": "demand_network_candidate_p7_d50",
                "seeds": seeds,
                "reads_components": ["k1"],
                "files": {
                    "columns": "components/candidate/columns.json",
                    "standardisers": "components/candidate/standardisers.json (identical to K1's)",
                    "offset_knots": "components/candidate/offset_knots.json",
                    "expected_weather_rows": "components/k1/expected_weather_rows.parquet (shared with K1)",
                    "signal_columns": "components/candidate/signal_columns.json",
                    "neighbours": "components/candidate/neighbours.json",
                    "weights": [f"components/candidate/weights/seed{s}.pt" for s in seeds],
                    "weights_index": "components/candidate/weights/weights_index.json",
                    "fixtures": "fixtures/candidate/",
                },
            },
        },
        "shared_files": {
            "config": "config.json",
            "primaries": "primaries.json",
            "readme": "README.md",
            "reference_check": "reference_check.py",
        },
        "fixture_origins": fixture_origins,
        "export_checks": checks,
        "code_sha256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))}
        | {"e00704_arm.py": sha256(next(HERE.parent.glob("00704-*")) / "e00704_arm.py")},
    }
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest["sha256"] = {str(p.relative_to(out)): sha256(p) for p in files}
    write_json(obj=manifest, path=out / "manifest.json")
    print("bundle written", out, len(files) + 1, "files", flush=True)


if __name__ == "__main__":
    main()

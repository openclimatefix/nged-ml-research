# ruff: noqa: D103, E501, PLR0915, PT018, E731, RUF059, B007
"""Experiment 705: write the production export bundle of one refit of the step-5 network.

Usage: `python e00705_export.py --arm ARM --month MONTH --dump DUMP.npz --out BUNDLE_DIR` with
`--source-root`, `--chain-work`, `--stats-work` as for `e00705_capture.py` (which writes DUMP.npz).
The exporter runs on CPU. It reads the stored weights and forecasts of the arm read-only, rebuilds
every standardiser from the captured fold state, asserts that the standardisers, the input matrix
and the quantiles reproduce the stored files, and writes the bundle that `reference_check.py`
reads. See the bundle's `README.md` for the format.
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

import e00048_build_frames as build  # ty: ignore[unresolved-import]
import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00437_features as f437  # ty: ignore[unresolved-import]
import e00558_lags as lags558  # ty: ignore[unresolved-import]
import e00560_data as d560  # ty: ignore[unresolved-import]
import e00572_lags as lg  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00677_extend as ext  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

HERE: Final = Path(__file__).resolve().parent
QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
SEEDS_ALL: Final = (0, 1, 2)
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
EXPECTED_NAMES_OFFSET: Final = 0


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
    return {
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


def ens_fixture(
    *,
    run: datetime,
    targets: pl.DataFrame,
    rows_x: np.ndarray,
    series_of_row: np.ndarray,
    valid_of_row: np.ndarray,
    out_dir: Path,
) -> dict:
    """Write the ENS fixtures of one run and return the comparison results."""
    cells = targets["h3_res_5"].unique().to_list()
    native = (
        pl.scan_parquet(build.partition_path(run=run) / "*.parquet")
        .select(build.SOURCE_COLUMNS)
        .filter(pl.col("h3_index").is_in(cells))
        .join(
            targets.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
            how="inner",
        )
        .drop("h3_index")
        .sort("time_series_id", "ensemble_member", "valid_time")
        .collect()
    )
    keep = [
        "time_series_id",
        "ensemble_member",
        "valid_time",
        *build.ENS_COLUMNS.values(),
        "wind_u_10m",
        "wind_v_10m",
    ]
    native.select(keep).write_parquet(out_dir / "ens_native.parquet", compression="zstd")
    half, init = build._half_hourly_rows(run=run, targets=targets)
    half = half.sort("time_series_id", "ensemble_member", "valid_time")
    half.select(
        "time_series_id",
        "ensemble_member",
        "valid_time",
        "lead_hours",
        *WEATHER_VARIABLES,
        "wind_u_10m",
        "wind_v_10m",
    ).write_parquet(out_dir / "ens_halfhourly.parquet", compression="zstd")
    summary = (
        prep.summarise(frame=half)
        .sort("time_series_id", "valid_time")
        .cast(
            dict.fromkeys(
                [f"{k}_{w}" for k in ("mean", "std") for w in WEATHER_VARIABLES], pl.Float32
            )
        )
    )
    stored = pl.read_parquet(ext.E607 / "ens_summary" / f"init={run:%Y-%m-%d}.parquet").sort(
        "time_series_id", "valid_time"
    )
    assert stored.columns == summary.columns, (stored.columns, summary.columns)
    assert stored.height == summary.height, (stored.height, summary.height)
    assert stored["valid_time"].equals(summary["valid_time"]) and stored["time_series_id"].equals(
        summary["time_series_id"]
    )
    diff = max(
        float(
            np.abs(
                stored[c].to_numpy().astype(np.float64) - summary[c].to_numpy().astype(np.float64)
            ).max()
        )
        for c in stored.columns
        if c.startswith(("mean_", "std_"))
    )
    summary.write_parquet(out_dir / "ens_summary.parquet", compression="zstd")
    # the network's raw columns 0 to 13 are these summaries (precipitation through log1p of the value clipped at zero)
    key = pl.DataFrame({"time_series_id": series_of_row, "valid_time": valid_of_row}).with_columns(
        pl.col("valid_time").cast(pl.Datetime("us", "UTC"))
    )
    joined = key.join(
        summary, on=["time_series_id", "valid_time"], how="left", maintain_order="left"
    )
    assert joined["mean_shortwave"].is_not_null().all()
    expected = np.empty((joined.height, 14), dtype=np.float64)
    for k, kind in enumerate(("mean", "std")):
        for j, w in enumerate(WEATHER_VARIABLES):
            v = joined[f"{kind}_{w}"].to_numpy().astype(np.float64)
            expected[:, 7 * k + j] = np.log1p(np.clip(v, 0, None)) if w == "precipitation" else v
    x_diff = float(np.abs(expected - rows_x[:, :14].astype(np.float64)).max())
    return {
        "summary_vs_stored_max_abs_diff": diff,
        "summary_rows": summary.height,
        "network_raw_columns_vs_summary_max_abs_diff": x_diff,
        "native_rows": native.height,
        "halfhourly_rows": half.height,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("replay", "proxy", "drop"))
    parser.add_argument("--month", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--chain-work", required=True)
    parser.add_argument("--stats-work", required=True)
    parser.add_argument(
        "--backtest", default=None, help="stored contract rows (per seed) to compare with"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS_ALL))
    args = parser.parse_args()
    out = Path(args.out).resolve()
    assert out.parent.parent.parent.name.startswith("00705-"), (
        f"--out must be <00705 folder>/bundles/<arm>/<month>: {out}"
    )
    if out.exists():
        shutil.rmtree(out)
    step5 = out / "components" / "step5"
    for sub in (step5 / "weights", out / "fixtures" / "step5"):
        sub.mkdir(parents=True)
    z = np.load(args.dump)
    source = Path(args.source_root) / "arms" / "k1_5w_sp1" / "work"
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
    stats_path = Path(args.stats_work) / "stats" / "B1" / f"{month}.npz"
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
    }

    # --- input matrix: reproduce the standardised columns from the raw ones --------------------------
    x = z["x"]
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
    checks["input_matrix"] = (
        "all 60 columns reproduce exactly (x37, lags, knot position, B1) from raw columns, standardisers, and cleaned power"
    )

    # --- weights ----------------------------------------------------------------------------------
    index = {}
    n_knots = None
    for seed in seeds:
        src = source / "models" / "P5" / f"{month}_d0_s{seed}.pt"
        dst = step5 / "weights" / f"seed{seed}.pt"
        shutil.copyfile(src, dst)
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
    assert n_knots == int(z["knots"]), (n_knots, int(z["knots"]))
    knot_dates = [
        f"{(2019 * 12 + 8 + i) // 12}-{(2019 * 12 + 8 + i) % 12 + 1:02d}-01" for i in range(n_knots)
    ]
    assert knot_dates[-1] == f"{month}-01", knot_dates[-1]
    write_json(
        obj={
            "class": "e00591_model.AuxLagMdn(kind='shared', n_b1=14, detector=False), stage 'ft'; state dict saved with torch.save(model.state_dict())",
            "unused_at_inference": "mixer_pre.*, head.* (auxiliary depth head)",
            "files": index,
        },
        path=step5 / "weights" / "weights_index.json",
    )
    write_json(
        obj={
            "n_knots": n_knots,
            "first_knot_month": "2019-09",
            "knot_dates": knot_dates,
            "rule": "knot k is the first of month 2019-09 + k months; knots_old holds knots 0 .. n-2 (pre-trained, frozen), knot_new holds knot n-1 (trained at this refit); the position of a valid time is months since 2019-09 plus the fraction of its month, clamped to [0, n-1]; linear interpolation per Primary; the offset is added to the three component means in z units",
        },
        path=step5 / "offset_knots.json",
    )

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
    # the run behind every row was issued at or before the half-hour it describes (no lookahead)
    gt = need[:, 1].astype(np.int64) * STEP_US + GRID_START_US
    assert (table_origin[table_origin > 0] <= gt[table_origin > 0]).all(), (
        "expected-weather run issued after the half-hour"
    )
    expected_frame.write_parquet(step5 / "expected_weather_rows.parquet", compression="zstd")

    # --- stored forecasts, per seed, native fractions ------------------------------------------------
    stored_q = {}
    for seed in seeds:
        st = pl.read_parquet(source / "fits" / "P5" / f"{month}_d0_s{seed}.parquet").with_columns(
            o=pl.col("power_fcst_init_time").dt.epoch("us"), v=pl.col("valid_time").dt.epoch("us")
        )
        mine = pl.DataFrame(
            {
                "time_series_id": z[f"key_series_s{seed}_main"],
                "o": z[f"key_origin_s{seed}_main"],
                "v": z[f"key_valid_s{seed}_main"],
            },
        ).hstack(pl.DataFrame(z[f"q_all_s{seed}_main"], schema=list(QCOLS)))
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

    # --- fixtures -----------------------------------------------------------------------------------
    origins_us = np.unique(origin)
    targets = prep.primaries().select("time_series_id", "h3_res_5", "latitude", "longitude")
    ts_of = np.array([p["time_series_id"] for p in primaries])
    backtest = pl.read_parquet(args.backtest) if args.backtest else None
    fixture_origins = []
    for o_us in origins_us:
        sel = np.nonzero(origin == o_us)[0]
        o_dt = datetime.fromtimestamp(int(o_us) / 1e6, UTC)
        name = f"{o_dt:%Y-%m-%d}"
        fixture_origins.append(name)
        d = out / "fixtures" / "step5" / f"origin={name}"
        d.mkdir(parents=True)
        run = o_dt - common.PUBLICATION_DELAY
        ens = ens_fixture(
            run=run,
            targets=targets,
            rows_x=raw37[sel],
            series_of_row=ts_of[primary[sel]],
            valid_of_row=valid[sel],
            out_dir=d,
        )
        # input matrix: keys, raw columns, standardised columns
        names60 = [c["name"] for c in column_table(names37=names37)]
        raw60 = np.concatenate(
            [
                raw37[sel],
                lag_raw[sel][:, [0]],
                np.zeros((len(sel), 2), np.float32),
                lag_raw[sel][:, [1]],
                np.zeros((len(sel), 2), np.float32),
                x[sel][:, 43:46],
                z["b1_raw"][sel],
            ],
            axis=1,
        )
        raw60[:, 38], raw60[:, 39] = x[sel][:, 38], x[sel][:, 39]
        raw60[:, 41], raw60[:, 42] = x[sel][:, 41], x[sel][:, 42]
        assert raw60.shape[1] == 60
        keys = pl.DataFrame(
            {
                "primary_index": primary[sel].astype(np.int8),
                "time_series_id": ts_of[primary[sel]].astype(np.int32),
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
        matrix = keys.hstack(
            pl.DataFrame({f"raw__{n}": raw60[:, i] for i, n in enumerate(names60)})
        ).hstack(pl.DataFrame({f"std__{n}": x[sel][:, i] for i, n in enumerate(names60)}))
        matrix.write_parquet(d / "input_matrix.parquet", compression="zstd")
        # expected-weather rows and lag power of this origin
        o_idx = int(z["origin_idx"][sel][0])
        window = np.arange(o_idx - 48 - 335, o_idx - 48 + 1)
        used = np.unique(np.concatenate([lag_idx[sel, 0], lag_idx[sel, 1]]))
        assert set(used[used >= 0]) <= set(window.tolist()), (
            "a lag lies outside the 336-half-hour window"
        )
        grid = pl.DataFrame(
            {
                "primary_index": np.repeat(np.arange(16, dtype=np.int8), len(window)),
                "grid_index": np.tile(window, 16),
            }
        )
        power = clean[
            grid["primary_index"].to_numpy().astype(np.int64), grid["grid_index"].to_numpy()
        ]
        grid = grid.with_columns(
            pl.Series("power_fraction", power),
            pl.Series("grid_time", GRID_START_US + grid["grid_index"].to_numpy() * STEP_US).cast(
                pl.Datetime("us", "UTC")
            ),
        )
        grid = grid.join(
            expected_frame.drop("grid_time"),
            on=["primary_index", "grid_index"],
            how="left",
            validate="1:1",
        )
        assert grid["run_origin_time"].is_not_null().all()
        grid.write_parquet(d / "lag_and_expected_rows.parquet", compression="zstd")
        # outputs: native per seed, seed mean, contract
        cols = {"primary_index": primary[sel].astype(np.int8), "valid_time": valid[sel]}
        out_native = pl.DataFrame(cols).with_columns(
            pl.col("valid_time").cast(pl.Datetime("us", "UTC"))
        )
        per_seed = []
        for seed in seeds:
            q = z[f"q_fix_s{seed}_main"][sel]
            per_seed.append(q)
            out_native = out_native.hstack(
                pl.DataFrame({f"{lv}_s{seed}": q[:, i] for i, lv in enumerate(QCOLS)})
            ).hstack(
                pl.DataFrame(
                    {
                        f"raw_out_{k}_s{seed}": z[f"raw_out_s{seed}_main"][sel][:, k]
                        for k in range(9)
                    }
                )
            )
        mean_q = np.sort(np.mean(np.stack(per_seed), axis=0, dtype=np.float32), axis=1)
        out_native = out_native.hstack(
            pl.DataFrame({f"{lv}_mean": mean_q[:, i] for i, lv in enumerate(QCOLS)})
        )
        out_native.write_parquet(d / "outputs_native.parquet", compression="zstd")
        removed = pl.DataFrame(cols).with_columns(
            pl.col("valid_time").cast(pl.Datetime("us", "UTC"))
        )
        for seed in seeds:
            q = z[f"q_fix_s{seed}_removed"][sel]
            removed = removed.hstack(
                pl.DataFrame({f"{lv}_s{seed}": q[:, i] for i, lv in enumerate(QCOLS)})
            )
        removed.write_parquet(d / "outputs_lags_removed.parquet", compression="zstd")
        # contract: 12:00 relabel x capacity, per seed in float32 MW, then mean and sort
        contract_origin = o_us + 3 * 3_600_000_000
        mask = (valid[sel] >= contract_origin) & (
            valid[sel] < contract_origin + 14 * 86_400_000_000
        )
        assert int(mask.sum()) == 16 * 672
        mw = [
            (q.astype(np.float64) * cap[primary[sel]][:, None]).astype(np.float32) for q in per_seed
        ]
        mw_mean = np.sort(np.mean(np.stack(mw), axis=0, dtype=np.float32), axis=1)
        contract = (
            pl.DataFrame(
                {
                    "time_series_id": ts_of[primary[sel]][mask].astype(np.int8),
                    "valid_time": valid[sel][mask],
                    "nwp_init_time": (contract_origin // 86_400_000_000) * 86_400_000_000,
                    "power_fcst_init_time": contract_origin,
                }
            )
            .with_columns(
                pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
                pl.col("nwp_init_time").cast(pl.Datetime("us", "UTC")),
                pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
            )
            .hstack(pl.DataFrame({f"{lv}_mw": mw_mean[mask][:, i] for i, lv in enumerate(QCOLS)}))
            .hstack(
                pl.DataFrame(
                    {
                        f"{lv}_s{seed}_mw": mw[k][mask][:, i]
                        for k, seed in enumerate(seeds)
                        for i, lv in enumerate(QCOLS)
                    }
                )
            )
        )
        contract.write_parquet(d / "outputs_contract.parquet", compression="zstd")
        compare = None
        if backtest is not None:
            sd = backtest.filter(
                pl.col("power_fcst_init_time") == datetime.fromtimestamp(contract_origin / 1e6, UTC)
            )
            diffs = []
            for k, seed in enumerate(seeds):
                got = contract.select(
                    "time_series_id", "valid_time", *[f"{lv}_s{seed}_mw" for lv in QCOLS]
                ).with_columns(pl.col("time_series_id").cast(pl.Int8))
                ref = sd.filter(pl.col("seed") == seed).select(
                    pl.col("time_series_id").cast(pl.Int8), "valid_time", *QCOLS
                )
                jj = got.join(ref, on=["time_series_id", "valid_time"], how="inner")
                assert jj.height == contract.height, (jj.height, contract.height)
                diffs.append(
                    max(
                        float(
                            np.abs(
                                jj[f"{lv}_s{seed}_mw"].to_numpy().astype(np.float64)
                                - jj[lv].to_numpy().astype(np.float64)
                            ).max()
                        )
                        for lv in QCOLS
                    )
                )
            compare = {"per_seed_contract_vs_stored_backtest_max_abs_diff_mw": diffs}
            assert max(diffs) == 0.0, diffs
        info = {
            "origin": name,
            "weekday": o_dt.strftime("%A"),
            "rows": len(sel),
            "week_lag_missing_rows": int((x[sel][:, 41] > 0.5).sum()),
            "day_lag_missing_rows": int((x[sel][:, 38] > 0.5).sum()),
            "ens": ens,
            "backtest": compare,
        }
        checks[f"fixture_{name}"] = info
        print(info, flush=True)

    # --- config ---------------------------------------------------------------------------------------
    f = f437
    config = {
        "shared": {
            "network_name": "step5",
            "native_origin_hour_utc": 9,
            "publication_delay_hours": common.PUBLICATION_DELAY / timedelta(hours=1),
            "weather_run": "ECMWF ENS 00Z run of the origin day; native origin = run init + 9 h",
            "contract_origin_hour_utc": 12,
            "contract_relabel": "origin + 3 h; keep valid_time in [contract origin, contract origin + 14 days); 672 half-hours; no clamp, no calibration",
            "horizon_days": 14,
            "contract_half_hours": 672,
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
        "components": {
            "step5": {
                "arm": args.arm,
                "n_primaries": 16,
                "input_clip_sd": prep.INPUT_CLIP_SD,
                "weather_variables": list(WEATHER_VARIABLES),
                "ens_summary_recipe": {
                    "source_names": {
                        "temperature_2m": "temperature_2m",
                        "dew_point_temperature_2m": "dew_point_temperature_2m",
                        "shortwave": "downward_short_wave_radiation_flux_surface",
                        "longwave": "downward_long_wave_radiation_flux_surface",
                        "pressure_surface": "pressure_surface",
                        "pressure_msl": "pressure_reduced_to_mean_sea_level",
                        "precipitation": "precipitation_surface",
                        "wind_speed_10m": "sqrt(wind_u_10m ** 2 + wind_v_10m ** 2)",
                    },
                    "steps": [
                        "select each Primary's H3 resolution-5 cell from the run (all 51 members, native 3-hourly then 6-hourly steps)",
                        "interpolate each member alone with ml_core.features._nwp._upsample_nwp_to_half_hourly, passing solar_coordinates = the Primaries' latitude and longitude (the shortwave interpolation needs them)",
                        "rename the archive columns to the network's names with source_names",
                        "derive wind_speed_10m from the interpolated u and v of each member, before any averaging",
                        "take the mean and the population std (ddof=0) over the 51 members per (Primary, valid half-hour)",
                    ],
                    "wrong_order_error": "averaging u and v before the speed gives up to 4.2 m/s of error (finding 1)",
                },
                "precipitation_transform": "log1p(clip(value, 0)) on the 51-member mean and on the 51-member population std",
                "member_std_ddof": 0,
                "member_count": 51,
                "lead_scale_hours": 336.0,
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
                "column_slices": {
                    "weather": [0, 7],
                    "spread": [7, 14],
                    "datetime": [14, 35],
                    "lead": [35, 37],
                    "day_lag": [37, 40],
                    "week_lag": [40, 43],
                    "b1": [46, 60],
                },
                "n_columns": 60,
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
                    "rule": "for each Primary and half-hour g in (origin - 192 h, origin - 24 h], take the ENS summary row of the run with the smallest lead for valid time g (the freshest run issued before g, one 09:00 native origin per day): its 7 weather means (precipitation log1p) and 21 datetime columns; recompute the 21 datetime columns from g; standardise with the x37 mean and std at x37_positions, clip +-5, round to float16; the lag encoder reads the row at each lag's own grid index, with the 14 B1 slots set to 0. No run issued after g is used",
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
                    "cams_columns_source": f"arm {args.arm}",
                },
                "offset_knots": {"first_knot_month": "2019-09", "n_knots": n_knots},
                "variant_origin": fixture_origins[-1],
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
                },
            }
        },
    }
    write_json(obj=config, path=out / "config.json")
    write_json(
        obj={"primaries": primaries, "order": "primary_index = row of site_map.csv, D2 .. D22"},
        path=out / "primaries.json",
    )
    write_json(obj=column_table(names37=names37), path=step5 / "columns.json")
    write_json(obj=standardisers, path=step5 / "standardisers.json")
    shutil.copyfile(HERE / "reference_check.py", out / "reference_check.py")
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
        "network_name": "step5",
        "arm": args.arm,
        "refit_month": month,
        "refit_time_utc": refit.isoformat(),
        "training_cutoff_utc": (refit - timedelta(hours=3.5)).isoformat(),
        "training_cutoff_rule": "training targets with valid time <= refit time - 3.5 h and origin <= refit time; the 2026-04 refit uses data before 2026-04-01 00:00 UTC",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "research_git": git_state(),
        "pinned_production_package_commit": direct["vcs_info"]["commit_id"],
        "library_versions": pins,
        "source_chain_paths": {
            "models_and_forecasts": str(source),
            "pretraining_and_step2_weights": args.chain_work,
            "b1_statistics": str(stats_path),
        },
        "seeds": seeds,
        "components": {
            "step5": {
                "kind": "demand_network_step5",
                "seeds": seeds,
                "files": {
                    "columns": "components/step5/columns.json",
                    "standardisers": "components/step5/standardisers.json",
                    "offset_knots": "components/step5/offset_knots.json",
                    "expected_weather_rows": "components/step5/expected_weather_rows.parquet",
                    "weights": [f"components/step5/weights/seed{s}.pt" for s in seeds],
                    "weights_index": "components/step5/weights/weights_index.json",
                    "fixtures": "fixtures/step5/",
                },
            }
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

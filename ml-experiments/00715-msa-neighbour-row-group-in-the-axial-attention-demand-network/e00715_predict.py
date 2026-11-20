# ruff: noqa: E501, PT018
"""Experiment 715: forecast one fold with a fitted F4 network (prediction phase).

Usage: e00715_predict.py ARM MONTH SEED [--device cuda] [--mask-history] [--mask-neighbours] [--accept-code-change OLD_HASH ...]
       e00715_predict.py --kn13 MONTH SEED [--device cuda]

Run it through `run.sh`, in a separate process from the fit: it sets `E00667_PHASE=predict`, which opens
the fold month's scored window. It loads the stage 3 state, asserts that the fold's standardiser equals
the training phase's, forecasts every (Primary, scored origin, lead day), drops valid times on or after
2026-07-01, and writes `work/fits/<arm>/<month>_s<seed>.parquet` atomically. The columns are experiment
713's. **Every arm, `F4-W` and `F4-WR` included, reads the cleaned power as host history and as neighbour
power at prediction on the real panel** (mode `predict_real`), and the script asserts it. A state whose
stage 3 log carries another code hash is refused unless `--accept-code-change` names the old hash.
`--mask-history` forecasts with every host history power token masked, under the arm name `<arm>-nohist`;
`--mask-neighbours` masks every neighbour slot, under `<arm>-nonbr` (design section 7.4).

`--kn13` is check KN13: F4's `predict` run on the stored stage 3 state of `M*` for the fold, against
experiment 713's stored forecast file: the same keys, and every quantile of every row within 1e-6 of
capacity. It needs no fit.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

os.environ["E00667_PHASE"] = "predict"

import e00713_arms as a713  # ty: ignore[unresolved-import]
import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr713  # ty: ignore[unresolved-import]
import e00715_arms as arms
import e00715_batch as fb
import e00715_model as f4
import e00715_neighbours as nb
import e00715_train as tr
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.experiment_store import HOLDOUT_START
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
QCOLS = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
PREDICT_BATCH = 512
KN13_TOLERANCE = 1e-6


def load_network(
    *,
    arm: str,
    month: str,
    seed: int,
    device: str,
    data_dir: Path,
    accept: Sequence[str],
    record: dict,
) -> tuple[mdl.MsaNet, bool, arms.ArmSpec | None]:
    """Load the stage 3 state of `arm` (one of F4's arms, or one of experiment 713's, `M*` resolved).

    Returns the network, whether it has the neighbour path, and the arm's specification (None for an
    arm of experiment 713). An arm of experiment 713 loads into `F4Net` and is called without the
    neighbour tensors, which gives the output of `MsaNet` exactly.
    """
    if arm in arms.ARMS:
        spec = arms.resolve(arm=arm)
        state = arms.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
        stage_log = arms.stage_log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
        assert state.exists(), state
        assert stage_log.exists(), f"{state} has no stage log: refit it"
        arms.check_code(
            earlier=json.loads(stage_log.read_text()), accept=accept, record=record, what=state.name
        )
        model: mdl.MsaNet = (
            f4.F4Net(switches=spec.switches)
            if spec.neighbour_path
            else mdl.MsaNet(switches=spec.switches)
        )
        model.load_state_dict(torch.load(state, map_location=device))
        return model.to(device).eval(), spec.neighbour_path, spec
    spec713 = a713.resolve(arm=arm, data_dir=data_dir)
    state = a713.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
    assert state.exists(), state
    net = f4.F4Net(switches=spec713.switches)
    net.load_mstar_state(torch.load(state, map_location=device))
    return net.to(device).eval(), False, None


def forecast_frame(
    *, model: mdl.MsaNet, neighbours: bool, spec: arms.ArmSpec | None, data: inp.FoldData,
    source: fb.F4Source, seed: int, mask_history: bool, mask_neighbours: bool, grid_start_us: int,
    time_series_ids: np.ndarray,
) -> tuple[pl.DataFrame, dict]:  # fmt: skip
    """Return the forecast frame of all scored origins, 16 Primaries, and 14 lead days, and the visibility counts.

    A copy of `e00713_predict.forecast_frame` with the neighbour tensors. The source must read the
    cleaned power as history and as neighbour power (asserted).
    """
    assert data.test_origin_idx is not None
    assert source.label == "cleaned", (
        f"the real-panel prediction reads the {source.label} array, not the cleaned power"
    )
    assert torch.equal(source.host.power_in, source.host.power_z), (
        "the host's history is not the cleaned power"
    )
    assert source.neighbour_power is source.host.power_z or torch.equal(
        source.neighbour_power, source.host.power_z
    )
    c_all = data.test_origin_idx + lay.NATIVE_OFFSET_STEPS
    assert data.source.ens is not None
    fb.check_origin_days(ens=data.source.ens, c_idx=c_all, what="the scored origins")
    prim, cc, lead = np.meshgrid(np.arange(16), c_all, np.arange(14), indexing="ij")
    prim, cc, lead = prim.ravel(), cc.ravel(), lead.ravel()
    samples = tr.F4Samples(primary=prim, c_idx=cc, is_val=np.zeros(len(prim), bool))  # ty: ignore[unknown-argument]
    if spec is not None and spec.slots == "shuffled":
        pool = nb.shuffle_pool(table=source.table, pool=tr713.block_pool(clean=data.clean))
        samples.nbr_ages = tr713.placebo_ages_for(primary=prim, c_idx=cc, pool=pool)
    parts = []
    visible, filled = 0, 0
    model.eval()
    src = data.source
    with torch.no_grad():
        for start in range(0, len(prim), PREDICT_BATCH):
            sel = np.arange(start, min(start + PREDICT_BATCH, len(prim)))
            ones = np.ones(len(sel), bool)
            drop = (ones, np.zeros(len(sel), bool)) if mask_history else None
            b = tr.batch_for(
                source=source, samples=samples, index=sel, lead=lead[sel], ensemble=True, cut=10**9,
                dropout=drop, mask_all_slots=mask_neighbours,
            )  # fmt: skip
            assert bool(
                src.ens.filled[
                    b.host.inputs.primary[:, None],
                    (b.host.c_idx // 48 - src.ens.first_day)[:, None],
                    (48 * b.host.inputs.lead_day[:, None] + src.steps),
                ].all()
            ), "a scored row has no ensemble inputs"
            v, f = fb.visible_share(batch=b)
            visible, filled = visible + v, filled + f
            out = tr.run_model(model=model, batch=b, neighbours=neighbours)
            m = mdl.standardised_mixture_mean(out=out)
            std = torch.from_numpy(data.target_std).to(m.device)[b.host.inputs.primary]
            mean = torch.from_numpy(data.target_mean).to(m.device)[b.host.inputs.primary]
            m = (m * std[:, None] + mean[:, None]).cpu().numpy()
            q = mdl.standardised_quantiles(out=out) * std[:, None, None] + mean[:, None, None]
            q = torch.sort(q, dim=-1).values.cpu().numpy()
            actual = (
                data.source.power_z[b.host.inputs.primary[:, None], b.host.target_idx].cpu().numpy()
            )
            parts.append((sel, q, b.host.target_idx.cpu().numpy(), actual, m))
    rows = []
    for sel, q, tidx, actual, m in parts:
        pr = np.repeat(prim[sel], 48)
        c = np.repeat(cc[sel], 48)
        flat_t = tidx.reshape(-1)
        mean = data.target_mean[pr]
        std = data.target_std[pr]
        rows.append(
            pl.DataFrame(
                {
                    "time_series_id": time_series_ids[pr].astype(np.int32),
                    "origin_us": grid_start_us + (c - lay.NATIVE_OFFSET_STEPS) * lay.HALF_HOUR_US,
                    "valid_us": grid_start_us + flat_t * lay.HALF_HOUR_US,
                    **{
                        name: q[:, :, i].reshape(-1).astype(np.float32)
                        for i, name in enumerate(QCOLS)
                    },
                    "mean": m.reshape(-1).astype(np.float32),
                    "actual": (actual.reshape(-1) * std + mean).astype(np.float32),
                }
            )
        )
    frame = pl.concat(rows)
    utc = pl.Datetime("us", "UTC")
    frame = (
        frame.with_columns(
            power_fcst_init_time=pl.col("origin_us").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            valid_time=pl.col("valid_us").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
        )
        .with_columns(
            nwp_init_time=pl.col("power_fcst_init_time").dt.truncate("1d"),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0,
            seed=pl.lit(seed, dtype=pl.Int64),
        )
        .filter(pl.col("valid_time") < pl.lit(HOLDOUT_START).cast(utc))
        .drop("origin_us", "valid_us")
        .select(
            "time_series_id", "power_fcst_init_time", "valid_time", *QCOLS, "seed", "mean", "actual",
            "lead_hours", "nwp_init_time",
        )
        .sort("time_series_id", "power_fcst_init_time", "valid_time")
    )  # fmt: skip
    return frame, {"visible_neighbour_tokens": visible, "tokens_of_filled_slots": filled}


def predict(
    *, arm: str, month: str, seed: int, device: str, mask_history: bool, mask_neighbours: bool, data_dir: Path,
    accept: Sequence[str] = (), out_path: Path | None = None,
) -> Path:  # fmt: skip
    """Forecast one fold and write the per-seed file (or `out_path`, for check KN13)."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]

    name = arm + ("-nohist" if mask_history else "") + ("-nonbr" if mask_neighbours else "")
    out = out_path or arms.fit_file(data_dir=data_dir, arm=name, month=month, seed=seed)
    if out.exists():
        print(f"{name} {month} s{seed}: done, skipped", flush=True)
        return out
    predict_log: dict = {"arm": name, "month": month, "seed": seed, **arms.code_record()}
    model, neighbours, spec = load_network(
        arm=arm, month=month, seed=seed, device=device, data_dir=data_dir, accept=accept, record=predict_log
    )  # fmt: skip
    arms.install_own_cache()
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    saved = a713.root(data_dir=data_dir) / "work" / "standardiser" / f"{month}.sha256"
    assert saved.exists() and saved.read_text() == data.fingerprint, (
        "standardiser differs from the training phase"
    )
    table = nb.load_table()
    mode = spec.slots if spec is not None else "none"
    source = arms.f4_source(data=data, table=table, mode=mode, rewire_seed=seed)
    sites = k578.sites().sort("primary")
    frame, counts = forecast_frame(
        model=model, neighbours=neighbours, spec=spec, data=data, source=source, seed=seed,
        mask_history=mask_history, mask_neighbours=mask_neighbours, grid_start_us=prep422.GRID_START_US,
        time_series_ids=sites["time_series_id"].to_numpy(),
    )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    frame.write_parquet(tmp)
    tmp.rename(out)
    if out_path is None:
        log_path = arms.log_file(
            data_dir=data_dir, arm=name, month=month, seed=seed, kind="predict"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_log = log_path.with_name(log_path.name + f".{os.getpid()}.tmp")
        share = counts["visible_neighbour_tokens"] / max(counts["tokens_of_filled_slots"], 1)
        tmp_log.write_text(
            json.dumps(
                {**predict_log, "rows": frame.height, **counts, "visible_share": share},
                indent=1,
                default=str,
            )
        )
        tmp_log.rename(log_path)
    print(f"{name} {month} s{seed}: {frame.height} rows", flush=True)
    return out


def compare_forecasts(*, new: pl.DataFrame, stored: pl.DataFrame) -> dict:
    """Return the key comparison and the largest quantile difference of two forecast frames (check KN13)."""
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    a, b = new.select(keys).sort(keys), stored.select(keys).sort(keys)
    same_keys = a.equals(b)
    if not same_keys:
        return {
            "same_keys": False,
            "n_new": a.height,
            "n_stored": b.height,
            "max_abs_difference": float("nan"),
        }
    joined = new.select(keys, *QCOLS).join(stored.select(keys, *QCOLS), on=keys, suffix="_s")
    worst = max(float((joined[q] - joined[f"{q}_s"]).abs().max()) for q in QCOLS)  # ty: ignore[invalid-argument-type]
    return {
        "same_keys": True,
        "n_rows": joined.height,
        "max_abs_difference": worst,
        "pass": worst <= KN13_TOLERANCE,
    }


def kn13(*, month: str, seed: int, device: str, data_dir: Path) -> dict:
    """KN13: F4's prediction path on the stored stage 3 state of `M*` against experiment 713's stored forecast file."""
    star = a713.m_star(data_dir=data_dir)
    stored_path = a713.fit_file(data_dir=data_dir, arm=star, month=month, seed=seed)
    state = a713.model_file(data_dir=data_dir, arm=star, month=month, seed=seed, stage=3)
    if not (stored_path.exists() and state.exists()):
        return {"skipped": f"experiment 713 has not stored {state.name} and {stored_path.name} yet"}
    tmp = arms.root(data_dir=data_dir) / "work" / "kn13" / f"{star}_{month}_s{seed}.parquet"
    tmp.unlink(missing_ok=True)
    predict(
        arm=star,
        month=month,
        seed=seed,
        device=device,
        mask_history=False,
        mask_neighbours=False,
        data_dir=data_dir,
        out_path=tmp,
    )
    return compare_forecasts(new=pl.read_parquet(tmp), stored=pl.read_parquet(stored_path))


def main() -> int:
    """Parse the command line and predict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", nargs="?")
    parser.add_argument("month", nargs="?")
    parser.add_argument("seed", type=int, nargs="?")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-history", action="store_true")
    parser.add_argument("--mask-neighbours", action="store_true")
    parser.add_argument("--kn13", action="store_true")
    parser.add_argument("--accept-code-change", action="append", default=[], metavar="OLD_HASH")
    args = parser.parse_args()
    if args.kn13:
        # with --kn13 the positional arguments are MONTH SEED
        month, seed = args.arm, int(args.month)
        result = kn13(month=month, seed=seed, device=args.device, data_dir=ML_EXPERIMENTS_DATA_DIR)
        print(json.dumps(result, indent=1))
        return 0 if result.get("pass", True) else 1
    predict(
        arm=args.arm, month=args.month, seed=args.seed, device=args.device, mask_history=args.mask_history,
        mask_neighbours=args.mask_neighbours, data_dir=ML_EXPERIMENTS_DATA_DIR, accept=args.accept_code_change,
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())

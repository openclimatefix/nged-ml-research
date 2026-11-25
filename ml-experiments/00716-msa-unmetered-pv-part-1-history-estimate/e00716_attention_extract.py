# ruff: noqa: E501, PT018
"""Experiment 716 and 717: attention maps and read-out weights of `M*`, `H`, and `HT` on the same samples.

Usage: run.sh e00716_attention_extract.py MONTH SEED [--arms M* H HT] [--device cuda] [--accept-code-change OLD_HASH ...]
       run.sh e00716_attention_extract.py 2026-05 0

Run it through `run.sh` (one thread for every numerical library, a 12 GB cap with no swap through
`systemd-run --scope -p MemorySwapMax=0`, the venv's Python called directly). Set `DATA_PATH_INTERNAL`
first. On a GPU the script caps the caching allocator at 6 GB. One process serves one fold and one
seed: it builds the fold's arrays once, then loads the stage 3 state of each arm in `--arms`.

Arm `M*` is experiment 713's selected arm, loaded as arm `Z` of this experiment's network from
`M*`'s stage 3 state (as check KF7 does). The other arms are this experiment's own: `H` and the
placebos of part 1, and `HT`, `HT-S`, and `T` of part 2. Every arm reads the same samples, and the
network of an arm without a PV path ignores the PV tensors, so one set of PV tables serves `M*` and the
first PV arm.

Nothing is refitted and no state is written: the forward pass with `return_attention=True` computes the
forecast by the same operations as the prediction phase. Each arm's forecast on the cases is compared
with the stored forecast file of the arm, and the script stops when the difference exceeds
`MAX_STORED_DIFFERENCE` of capacity.

The script refuses to overwrite a stored output: it stops before loading any data when the file of any
arm in `--arms` exists. Delete the file by hand to extract it again.

**Cases.** Three hosts, `CASE_HOSTS` (D11, D19, D21), each with two lead-day-1 cases chosen from the
fold's scored origins and the CAMS shortwave (weather column 0 of the dense table) at the host. A
`clear_day` is the origin with the highest mean shortwave over the target day. A `dull_after_sunny` is
the origin with the lowest target-day mean among the origins whose history rows are in the sunniest
third, and the script stops when that target-day mean is not below the fold's lower tercile. The history
mean covers the 14 history rows of the network (ages 2 to 15, `lay.history_grid_index`), so every value
of it ends at least 21 hours before the native origin. The target-day mean is the realised sky after the
origin: it picks the case and is fed to no input. No other Primary appears in a case.

**Stored per arm and case** (`<host>/<case>/...` in `work/attention/<arm>_<month>_s<seed>.npz` of the
folder that owns the arm): `across` (the last block's weights from each target-row token to the target
row and the 14 history rows, per head: 48 x 4 x 15; slot 0 is the target row, and slot k from 1 to 14 is the history
row of age k+1, which starts k+1 days before the forecast origin), `within_top` (the last block's within-day weights of the target row,
4 x 48 x 48), `alpha` (the read-out weights, 14 x 48), `hist_idx`, `target_idx`, `meta` (Primary index,
contract-origin index, zero-based lead day), the forecast `q` (48 x 11 quantiles, in units of capacity),
`mean`, and `actual`, and the history `hist_power` (14 x 48, capacity units, NaN where hidden) and, for an
arm with a history path, `hist_pv` (14 x 48, the estimate as a share of capacity, so it can be drawn
beside `hist_power`), and `history_mean` and `target_mean` (the two shortwave means that picked the
case). **Stored in the summary** (every Primary, every scored
origin, lead days 1 and 7): `summary_lead<k>/mean_weight` and `summary_lead<k>/same_weekday_weight`, each
16 x 15, and `summary_lead<k>/count`, 16. The file also holds `max_difference_to_stored`, the largest
difference between the arm's forecast on the cases and the stored forecast file.

Each file has a sidecar `.json` with the code record of the run. The script writes nothing else.
Plots of the stored forecasts (the time series of the page) read the stored parquet files, and plots of
`across` and `alpha` read the files of this script.

All arms in one call must have the same PV input, because one set of PV tables serves them all: run
`HT` in a call of its own (`--arms HT`), and run `M*` with `H` or with any one PV arm.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

os.environ.setdefault("E00667_PHASE", "predict")

import e00713_arms as arms713  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00716_arms as arms
import e00716_batch as pvb
import e00716_model as pvm
import e00716_train as pvt
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)

CASE_HOSTS: Final[tuple[str, ...]] = ("D11", "D19", "D21")
"""The three Primaries that the page may show as forecast examples; D8, D12, and D22 never appear."""
CASE_KINDS: Final[tuple[str, ...]] = ("clear_day", "dull_after_sunny")
SUNNY_QUANTILE: Final = 2 / 3
"""A history window is sunny when its mean shortwave is at or above this quantile of the fold's origins."""
SUMMARY_LEADS: Final[tuple[int, ...]] = (0, 6)
"""Lead days 1 and 7, as zero-based indices."""
N_PRIMARIES: Final = 16
N_SLOTS: Final = 15
"""The target row and the 14 history rows."""
SUMMARY_BATCH: Final = 256
MAX_STORED_DIFFERENCE: Final = 1e-4
"""The largest accepted difference to the stored forecast, in units of capacity."""
GPU_CAP_GB: Final = 6
SHORTWAVE_COLUMN: Final = 0
M_STAR: Final = "M*"
QUANTILE_COLUMNS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)


def attention_file(*, data_dir: Path, arm: str, month: str, seed: int) -> Path:
    """Return the output file of `arm`, in the `work/attention` folder of the experiment that owns it."""
    owner = "Z" if arm == M_STAR else arm
    name = "Mstar" if arm == M_STAR else arm
    return (
        arms.root(data_dir=data_dir, arm=owner)
        / "work"
        / "attention"
        / f"{name}_{month}_s{seed}.npz"
    )


def case_key(*, host: str, kind: str) -> str:
    """Return the prefix of a case's arrays in the output file."""
    return f"{host}/{kind}"


class Case(NamedTuple):
    """One case: where it is, and the two shortwave means that picked it."""

    primary: int
    contract_idx: int
    lead_day: int
    history_mean: float
    target_mean: float


def history_means(*, shortwave: np.ndarray, c_all: np.ndarray) -> np.ndarray:
    """Return the mean shortwave of the 14 history rows of each origin in `c_all` (shortwave is one Primary's (G,) row)."""
    return shortwave[lay.history_grid_index(contract_idx=c_all)].mean(axis=(1, 2))


def pick_cases(
    *, shortwave: np.ndarray, primaries: dict[str, int], c_all: np.ndarray
) -> dict[str, Case]:
    """Return each case, keyed by `<host>/<kind>`.

    Args:
        shortwave: (16, G) CAMS shortwave of the dense weather array, on the half-hour grid.
        primaries: The Primary index of each host in `CASE_HOSTS`.
        c_all: The contract-origin indices of the scored origins.

    Raises:
        AssertionError: When no origin in the sunniest third has a target day below the fold's lower tercile.
    """
    day = lay.STEPS_PER_DAY
    assert int(c_all.max()) + day <= shortwave.shape[1], "a target day ends after the grid"
    cases: dict[str, Case] = {}
    for host in CASE_HOSTS:
        p = primaries[host]
        target = np.array([shortwave[p, c : c + day].mean() for c in c_all])
        history = history_means(shortwave=shortwave[p], c_all=c_all)
        sunny = history >= np.quantile(history, SUNNY_QUANTILE)
        clear = int(np.argmax(target))
        dull = int(np.argmin(np.where(sunny, target, np.inf)))
        assert target[dull] < np.quantile(target, 1 - SUNNY_QUANTILE), (
            f"{host}: no origin in the sunniest third has a target day below the lower tercile"
        )
        for kind, k in (("clear_day", clear), ("dull_after_sunny", dull)):
            cases[case_key(host=host, kind=kind)] = Case(
                primary=p,
                contract_idx=int(c_all[k]),
                lead_day=0,
                history_mean=float(history[k]),
                target_mean=float(target[k]),
            )
    return cases


def run_model(
    *, model: pvm.PvNet, source: pvb.PvSource, primary: np.ndarray, c_idx: np.ndarray, lead: int
) -> tuple[mdl.MsaOutput, pvb.PvAssembled]:
    """Run `model` with `return_attention=True` on the samples (`primary`, `c_idx`) at one lead day."""
    samples = tr.SampleSet(primary=primary, c_idx=c_idx, is_val=np.zeros(len(primary), dtype=bool))
    batch = pvt.batch_for(
        source=source,
        samples=samples,
        index=np.arange(len(primary)),
        lead=np.full(len(primary), lead),
        cut=10**9,
    )
    with torch.no_grad():
        return model(batch.inputs, return_attention=True), batch


def last_across(*, res: mdl.MsaOutput) -> torch.Tensor:
    """Return the last block's across-day weights (B, 48, heads, 15)."""
    assert res.attention is not None
    last = max(int(k.split("_")[1]) for k in res.attention if k.startswith("across_"))
    return res.attention[f"across_{last}"]


def case_arrays(
    *,
    model: pvm.PvNet,
    source: pvb.PvSource,
    case: Case,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the stored arrays of one case (see the module docstring)."""
    p, c, lead = case.primary, case.contract_idx, case.lead_day
    res, batch = run_model(
        model=model, source=source, primary=np.array([p]), c_idx=np.array([c]), lead=lead
    )
    assert res.attention is not None
    last = max(int(k.split("_")[1]) for k in res.attention)
    scale, shift = float(target_std[p]), float(target_mean[p])
    quantiles = torch.sort(mdl.standardised_quantiles(out=res.out), dim=-1).values * scale + shift
    mean = mdl.standardised_mixture_mean(out=res.out) * scale + shift
    inputs = batch.inputs
    hist_power = torch.where(
        inputs.hist_visible[0], inputs.hist_power[0] * scale + shift, torch.nan
    )
    out = {
        "across": last_across(res=res)[0].cpu().numpy(),
        "alpha": res.alpha[0].cpu().numpy(),
        "hist_idx": batch.hist_idx[0, :, 0].cpu().numpy(),
        "target_idx": batch.target_idx[0].cpu().numpy(),
        "meta": np.array([p, c, lead]),
        "q": quantiles[0].cpu().numpy(),
        "mean": mean[0].cpu().numpy(),
        "actual": (source.power_z[p, batch.target_idx[0]] * scale + shift).cpu().numpy(),
        "hist_power": hist_power.cpu().numpy(),
        "history_mean": np.array(case.history_mean),
        "target_mean": np.array(case.target_mean),
    }
    if f"within_{last}" in res.attention:
        out["within_top"] = res.attention[f"within_{last}"][0, 0].cpu().numpy()
    if model.pv.history:
        out["hist_pv"] = (inputs.hist_pv[0] * scale).cpu().numpy()
    return out


def summary_arrays(
    *, model: pvm.PvNet, source: pvb.PvSource, c_all: np.ndarray
) -> dict[str, np.ndarray]:
    """Return, per Primary and per history slot, the mean across-day weight and the part on rows with the target's weekday.

    Slot 0 is the target row, which has no weekday split. A history row has the target's weekday when
    its age in whole days from the target row's first half-hour is a multiple of 7.
    """
    out: dict[str, np.ndarray] = {}
    for lead in SUMMARY_LEADS:
        sums = np.zeros((N_PRIMARIES, N_SLOTS))
        same = np.zeros((N_PRIMARIES, N_SLOTS))
        count = np.zeros(N_PRIMARIES)
        for p in range(N_PRIMARIES):
            for start in range(0, len(c_all), SUMMARY_BATCH):
                c = c_all[start : start + SUMMARY_BATCH]
                res, batch = run_model(
                    model=model, source=source, primary=np.full(len(c), p), c_idx=c, lead=lead
                )
                weight = last_across(res=res).mean(dim=(1, 2)).cpu().numpy()  # (B, 15)
                age_days = (batch.target_idx[:, 0:1] - batch.hist_idx[:, :, 0]) // lay.STEPS_PER_DAY
                weekday = (age_days % 7 == 0).cpu().numpy()  # (B, 14)
                sums[p] += weight.sum(axis=0)
                same[p, 1:] += (weight[:, 1:] * weekday).sum(axis=0)
                count[p] += len(c)
        out[f"summary_lead{lead + 1}/mean_weight"] = sums / count[:, None]
        out[f"summary_lead{lead + 1}/same_weekday_weight"] = same / count[:, None]
        out[f"summary_lead{lead + 1}/count"] = count
    return out


def max_difference(
    *,
    stored: pl.DataFrame,
    time_series_id: int,
    origin_us: int,
    valid_us: np.ndarray,
    quantiles: np.ndarray,
    mean: np.ndarray,
) -> float:
    """Return the largest absolute difference between one case's forecast and the stored forecast file's rows, over the 11 quantiles and the mean.

    Every one of the 48 valid times must be in the stored file, and every value must be finite.
    """
    assert np.isfinite(quantiles).all() and np.isfinite(mean).all(), (
        "a forecast value is not finite"
    )
    mine = pl.DataFrame(
        {
            "valid_time": pl.from_epoch(pl.Series(valid_us), time_unit="us").dt.replace_time_zone(
                "UTC"
            ),
            **{name: quantiles[:, i].astype(np.float32) for i, name in enumerate(QUANTILE_COLUMNS)},
            "mean": mean.astype(np.float32),
        }
    )
    rows = stored.filter(
        (pl.col("time_series_id") == time_series_id)
        & (
            pl.col("power_fcst_init_time")
            == pl.lit(origin_us, dtype=pl.Int64).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
        )
    )
    joined = rows.join(mine, on="valid_time", suffix="_mine")
    assert joined.height == len(valid_us), (
        f"{joined.height} of {len(valid_us)} valid times are in the stored forecast"
    )
    worst = max(
        float(np.abs(joined[c].to_numpy() - joined[f"{c}_mine"].to_numpy()).max())
        for c in (*QUANTILE_COLUMNS, "mean")
    )
    assert np.isfinite(worst), "the stored forecast holds a value that is not finite"
    return worst


def limit_gpu_memory(*, device: str) -> None:
    """Cap the caching allocator of `device` at `GPU_CAP_GB` (the CUDA context sits outside the cap)."""
    if not device.startswith("cuda"):
        return
    index = torch.device(device).index or 0
    total = torch.cuda.get_device_properties(index).total_memory
    torch.cuda.set_per_process_memory_fraction(fraction=GPU_CAP_GB * 2**30 / total, device=index)


def stored_forecast_file(*, data_dir: Path, arm: str, month: str, seed: int) -> Path:
    """Return the stored per-seed forecast file of `arm` (`M*` lives in experiment 713's folder)."""
    if arm == M_STAR:
        star = arms713.m_star(data_dir=data_dir)
        return arms713.fit_file(data_dir=data_dir, arm=star, month=month, seed=seed)
    return arms.fit_file(data_dir=data_dir, arm=arm, month=month, seed=seed)


def load_arm_model(
    *,
    arm: str,
    month: str,
    seed: int,
    device: str,
    data_dir: Path,
    accept: Sequence[str],
    record: dict,
) -> pvm.PvNet:
    """Load the stage 3 state of `arm` into its network, after the code-hash check of the prediction phase."""
    if arm == M_STAR:
        star = arms713.m_star(data_dir=data_dir)
        state = arms713.model_file(data_dir=data_dir, arm=star, month=month, seed=seed, stage=3)
        net = pvm.PvNet(arm="Z")
    else:
        import e00716_fit as fit

        state = arms.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
        stage_log = fit.stage_log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
        assert stage_log.exists(), f"{state} has no stage log: refit it"
        arms.check_code(
            earlier=json.loads(stage_log.read_text()), accept=accept, record=record, what=state.name
        )
        net = pvm.PvNet(arm=arms.ARMS[arm].net, seed=seed)  # ty: ignore[invalid-argument-type]
    assert state.exists(), state
    net = net.to(device)
    net.load_state_dict(torch.load(state, map_location=device))
    return net.eval()


def extract(  # noqa: PLR0915
    *,
    month: str,
    seed: int,
    arm_names: Sequence[str],
    device: str,
    data_dir: Path,
    accept: Sequence[str],
) -> list[Path]:
    """Extract the attention maps of every arm in `arm_names` for one fold and one seed."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    import e00716_fit as fit

    assert len(set(arm_names)) == len(arm_names), "an arm is named twice"
    unknown = [a for a in arm_names if a != M_STAR and a not in arms.ARMS]
    assert not unknown, f"unknown arms {unknown}"
    paths = {a: attention_file(data_dir=data_dir, arm=a, month=month, seed=seed) for a in arm_names}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        msg = f"refusing to overwrite stored output: {existing}"
        raise FileExistsError(msg)
    pv_arms = [a for a in arm_names if a != M_STAR]
    table_arm = pv_arms[0] if pv_arms else "Z"
    assert all(
        (arms.ARMS[a].history, arms.ARMS[a].target)
        == (arms.ARMS[table_arm].history, arms.ARMS[table_arm].target)
        for a in pv_arms
    ), "the arms have different PV inputs: run them in separate calls"
    record: dict = {"month": month, "seed": seed, "arms": list(arm_names), **arms.code_record()}
    arms.install_own_cache()
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device, keep_rows=True)
    for arm in arm_names:
        owner = "Z" if arm == M_STAR else arm
        saved = (
            arms.root(data_dir=data_dir, arm=owner) / "work" / "standardiser" / f"{month}.sha256"
        )
        assert saved.exists() and saved.read_text() == data.fingerprint, (
            f"{arm}: the standardiser differs from the training phase"
        )
    assert data.test_origin_idx is not None
    c_all = data.test_origin_idx + lay.NATIVE_OFFSET_STEPS
    sites = k578.sites()
    primaries = {
        host: int(sites.filter(pl.col("label") == host)["primary"][0]) for host in CASE_HOSTS
    }
    ids = sites.sort("primary")["time_series_id"].to_numpy()
    shortwave = data.source.dense_weather[:, :, SHORTWAVE_COLUMN].float().cpu().numpy()
    cases = pick_cases(shortwave=shortwave, primaries=primaries, c_all=c_all)
    source = pvb.PvSource(
        source=data.source,
        tables=fit.build_pv_tables(pipe=pipe, data=data, arm=table_arm, device=device),
    )
    written: list[Path] = []
    for arm in arm_names:
        model = load_arm_model(
            arm=arm,
            month=month,
            seed=seed,
            device=device,
            data_dir=data_dir,
            accept=accept,
            record=record,
        )
        stored = pl.read_parquet(
            stored_forecast_file(data_dir=data_dir, arm=arm, month=month, seed=seed)
        )
        out: dict[str, np.ndarray] = {}
        worst = 0.0
        for key, case in cases.items():
            arrays = case_arrays(
                model=model,
                source=source,
                case=case,
                target_mean=data.target_mean,
                target_std=data.target_std,
            )
            for name, value in arrays.items():
                out[f"{key}/{name}"] = value
            p, c = case.primary, case.contract_idx
            worst = max(
                worst,
                max_difference(
                    stored=stored,
                    time_series_id=int(ids[p]),
                    origin_us=prep422.GRID_START_US
                    + (c - lay.NATIVE_OFFSET_STEPS) * lay.HALF_HOUR_US,
                    valid_us=prep422.GRID_START_US + arrays["target_idx"] * lay.HALF_HOUR_US,
                    quantiles=arrays["q"],
                    mean=arrays["mean"],
                ),
            )
        assert worst <= MAX_STORED_DIFFERENCE, (
            f"{arm}: the forecast on the cases differs from the stored forecast by {worst:.2e} of capacity"
        )
        out["max_difference_to_stored"] = np.array(worst)
        out.update(summary_arrays(model=model, source=source, c_all=c_all))
        path = paths[arm]
        path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = path.with_suffix(".json")
        sidecar_tmp = sidecar.with_name(sidecar.name + f".{os.getpid()}.tmp")
        sidecar_tmp.write_text(
            json.dumps(
                {**record, "arm": arm, "max_difference_to_stored": worst}, indent=1, default=str
            )
        )
        sidecar_tmp.rename(sidecar)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **out)  # ty: ignore[invalid-argument-type]
        tmp.rename(path)
        print(
            f"{arm} {month} s{seed}: stored {path}, largest difference to the stored forecast {worst:.2e}",
            flush=True,
        )
        written.append(path)
    return written


def main() -> int:
    """Parse the command line and extract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--arms", nargs="+", default=[M_STAR, "H"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--accept-code-change", action="append", default=[], metavar="OLD_HASH")
    args = parser.parse_args()
    limit_gpu_memory(device=args.device)
    extract(
        month=args.month,
        seed=args.seed,
        arm_names=args.arms,
        device=args.device,
        data_dir=ML_EXPERIMENTS_DATA_DIR,
        accept=args.accept_code_change,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

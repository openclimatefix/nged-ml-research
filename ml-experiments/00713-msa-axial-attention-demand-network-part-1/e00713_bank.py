# ruff: noqa: E501
"""Experiment 713: prediction-only read-outs of fitted MSA networks (design 4.3, 6.5).

Usage:
    e00713_bank.py swap a|b ARM MONTH SEED     # history weather from the ensemble table
    e00713_bank.py shifts ARM MONTH SEED       # share of 581's injected shifts that the median follows

Nothing is refitted. The all-history-masked read-out is `e00713_predict.py --mask-history`.

**Swaps (a) and (b)** replace the 7 weather columns of every history token with the Candidate's
expected-input table `e00560_data.raw_table(source="ft")`, the ensemble run issued at or before the
token's time. Swap (a) keeps the 14 B1 columns as ERA5 and CAMS give them. Swap (b) sets them to zero,
which is how the Candidate reads its lag times. The forecasts are written under the arm names
`<arm>-swapa` and `<arm>-swapb`.

**The injected shifts of experiment 581** (`e00581_bank.contaminate`, passes 0 to 11) add a level
shift to every Primary, ending 14 days after the month's start. The follow share is experiment 581's
`follow_q50` (`e00581_tails._aggregate`): the median forecast on the shifted history minus the median
forecast of the same network on the clean history (581's "uninjected twin"), times the sign of the
shift, divided by its depth. It is averaged over origins from 4 days after the onset to the end of the
event (581's phase `d4-end`) and over lead days 1 to 5 (581's 24-120 h band), then over the 16 Primaries.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["E00667_PHASE"] = "predict"

import e00713_arms as arms
import e00713_inputs as inp
import e00713_layout as lay
import e00713_model as mdl
import e00713_predict as pred
import e00713_train as tr
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
SHIFT_PASSES = range(12)


def load_model(
    *, spec: arms.ArmSpec, arm: str, month: str, seed: int, device: str, data_dir: Path
) -> mdl.MsaNet:
    """Load the stage 3 state of a fitted arm."""
    model = mdl.MsaNet(switches=spec.switches).to(device)
    state = arms.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
    model.load_state_dict(torch.load(state, map_location=device))
    return model.eval()


def swapped_source(*, pipe: inp.Pipeline, data: inp.FoldData, keep_b1: bool, device: str):  # noqa: ANN201
    """Return the fold's source with the history weather columns taken from the ensemble table."""
    import e00560_data as d560  # ty: ignore[unresolved-import]

    e = pipe.e
    cols, _origin = d560.raw_table(source="ft", cache=e.W560 / "expected")
    # the pipeline attached the table standardised with the fold's own standardiser
    table = e.CTX["table_ft"].cpu()[..., :7]
    del cols
    src = data.source
    weather = src.dense_weather.clone()
    weather[..., :7] = table.to(weather.dtype).to(weather.device)
    if not keep_b1:
        weather[..., 7:] = 0
    return type(src)(
        power_z=src.power_z, dense_weather=weather, dense_datetime=src.dense_datetime,
        day_type=src.day_type, ens=src.ens, power_in=src.power_in,
    )  # fmt: skip


def cmd_swap(*, variant: str, arm: str, month: str, seed: int, device: str) -> None:
    """Forecast a fold with swapped history weather and write the per-seed file."""
    data_dir = ML_EXPERIMENTS_DATA_DIR
    name = f"{arm}-swap{variant}"
    out = arms.fit_file(data_dir=data_dir, arm=name, month=month, seed=seed)
    if out.exists():
        return
    spec = arms.resolve(arm=arm, data_dir=data_dir)
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    data.source = swapped_source(pipe=pipe, data=data, keep_b1=variant == "a", device=device)  # ty: ignore[invalid-assignment]
    world = tr.World(
        source=data.source, cutoff_idx=data.cutoff_idx, clean=data.clean, target_mean=data.target_mean,
        target_std=data.target_std, month=month, placebo_pool=tr.block_pool(clean=data.clean),
    )  # fmt: skip
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]

    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    frame = pred.forecast_frame(
        model=load_model(spec=spec, arm=arm, month=month, seed=seed, device=device, data_dir=data_dir),
        data=data, spec=spec, world=world, seed=seed, mask_history=False,
        grid_start_us=prep422.GRID_START_US, time_series_ids=ids,
    )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    frame.write_parquet(tmp)
    tmp.rename(out)
    print(name, month, seed, frame.height, "rows", flush=True)


def median_at(
    *, model: mdl.MsaNet, source: Any, primary: np.ndarray, c_idx: np.ndarray, lead: int, data: inp.FoldData,
) -> np.ndarray:  # fmt: skip
    """Return the median (capacity shares) of every (Primary, origin) at one lead day, shape (N, 48)."""
    samples = tr.SampleSet(primary=primary, c_idx=c_idx, is_val=np.zeros(len(primary), bool))
    out = []
    with torch.no_grad():
        for start in range(0, len(primary), 512):
            sel = np.arange(start, min(start + 512, len(primary)))
            b = tr.batch_for(
                source=source, samples=samples, index=sel, lead=np.full(len(sel), lead),
                ensemble=True, cut=10**9, dropout=None,
            )  # fmt: skip
            q = mdl.standardised_quantiles(out=model(b.inputs).out)[..., 5].cpu().numpy()
            out.append(
                q * data.target_std[primary[sel]][:, None] + data.target_mean[primary[sel]][:, None]
            )
    return np.concatenate(out)


def cmd_shifts(*, arm: str, month: str, seed: int, device: str) -> None:
    """Write the follow share of each of experiment 581's 12 injected passes for one fitted network."""
    import e00581_bank as bank  # ty: ignore[unresolved-import]

    data_dir = ML_EXPERIMENTS_DATA_DIR
    path = arms.root(data_dir=data_dir) / "work" / "bank581" / arm / f"{month}_s{seed}.json"
    if path.exists():
        return
    spec = arms.resolve(arm=arm, data_dir=data_dir)
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    model = load_model(spec=spec, arm=arm, month=month, seed=seed, device=device, data_dir=data_dir)
    month_idx = data.cutoff_idx
    result: dict[str, float] = {}
    for r in SHIFT_PASSES:
        shifted, rows = bank.contaminate(clean=data.clean, month_start_idx=month_idx, r=r)
        z = (shifted - data.target_mean[:, None]) / data.target_std[:, None]
        src = data.source.with_power_in(power_in=torch.from_numpy(z.astype(np.float32)).to(device))
        shares = []
        for p in range(16):
            row = rows[p]
            first_day = (row["onset_idx"] + 4 * lay.STEPS_PER_DAY) // 48
            last_day = row["end_idx"] // 48
            days = np.arange(first_day, last_day)
            if not len(days):
                continue
            c = days * 48 + 24
            prim = np.full(len(c), p)
            diffs = []
            for lead in range(
                1, 6
            ):  # lead indices 1 to 5: 24 h to 144 h after the 12:00 UTC origin (experiment 581: 24 h to 120 h after 09:00 UTC); report-only
                a = median_at(model=model, source=src, primary=prim, c_idx=c, lead=lead, data=data)
                b = median_at(
                    model=model, source=data.source, primary=prim, c_idx=c, lead=lead, data=data
                )
                diffs.append((a - b).mean() / (row["depth"] * row["sign"]))
            shares.append(float(np.mean(diffs)))
        result[f"r{r:02d}"] = float(np.mean(shares))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.rename(path)
    print(arm, month, seed, "follow share", float(np.mean(list(result.values()))), flush=True)


def main() -> int:
    """Dispatch the sub-command."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("swap")
    s.add_argument("variant", choices=("a", "b"))
    h = sub.add_parser("shifts")
    for p in (s, h):
        p.add_argument("arm")
        p.add_argument("month")
        p.add_argument("seed", type=int)
        p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "swap":
        cmd_swap(
            variant=args.variant, arm=args.arm, month=args.month, seed=args.seed, device=args.device
        )
    else:
        cmd_shifts(arm=args.arm, month=args.month, seed=args.seed, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())

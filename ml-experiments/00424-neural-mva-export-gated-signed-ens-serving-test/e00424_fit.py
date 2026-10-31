# ruff: noqa: E501
"""Fit the arms of experiment 424 and forecast the test months with ENS mean and spread inputs.

Arms (all on experiment 420's rows, rotations, settings, and months; monthly refit):

- `P0`: experiment 420's P0 (every folded Primary's gate forced shut, the plain neural network read
  as signed), fitted on ENS inputs.
- `A`: experiment 420's arm A (the Primary-level export gate), fitted on ENS inputs.
- `A-applied`: the CAMS-and-ERA5-trained network of experiment 420's arm A (seed 0), applied to ENS
  inputs with no refit (diagnostic, zero fits). The ENS mean temperature replaces ERA5 temperature
  and the ENS unit shape replaces the CAMS unit shape; the spread columns do not exist in that
  network.

Each forecast file holds the hidden pair's signed forecast and the MVA-only eight's reading
forecast, with the ENS unit shape and clearness index the network saw (`u_ens`, `k_ens`) and the
realised CAMS values (`u_gate`, `k`) that define the scoring slices, as in experiment 420.

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
export PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:ml-experiments/00405-mdn-explicit-pv-term-learned-capacity-known-answer:ml-experiments/00412-mdn-signed-mw-from-mva-meters-known-answer:ml-experiments/00417-hidden-sign-mva-export-gate-zero-crossing-labels:ml-experiments/00420-neural-mva-export-gated-signed-final:ml-experiments/00424-neural-mva-export-gated-signed-ens-serving-test
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue -p CPUQuota=100% \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00424-*/e00424_fit.py run --stage 1
```
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Final

import e00397_prep as base  # ty: ignore[unresolved-import]
import e00412_fit as f412  # ty: ignore[unresolved-import]
import e00417_fit as f417  # ty: ignore[unresolved-import]
import e00417_inputs as i417  # ty: ignore[unresolved-import]
import e00420_fit as f420  # ty: ignore[unresolved-import]
import e00420_models as models  # ty: ignore[unresolved-import]
import e00424_inputs as ens  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "neural-mva-export-gated-signed-ens-serving-test"
OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00424-{SLUG}" / "work"
TEST_MONTHS: Final[tuple[str, ...]] = f420.TEST_MONTHS
SETTINGS: Final[dict[str, i417.Setting]] = {name: f420.SETTINGS[name] for name in ("L2", "L0")}
"""Level 2.0 (the main setting) and the no-injection placebo. Level 1.0 is not run."""
ARMS: Final[dict[str, f420.Arm]] = {"P0": f420.ARMS["P0"], "A": f420.ARMS["A"]}
MAX_LEAD_HOURS: Final[float] = f420.MAX_LEAD_HOURS
SCORE_CHUNK_ROWS: Final[int] = f420.SCORE_CHUNK_ROWS
LEVEL_COLUMNS: Final[tuple[str, ...]] = f420.LEVEL_COLUMNS
MIN_AVAILABLE_GIB: Final[float] = f420.MIN_AVAILABLE_GIB
APPLIED: Final[str] = "A-applied"
CAMS_MODELS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00420-neural-mva-export-gated-signed-final" / "work" / "models" / "A"
)
"""Experiment 420's saved arm A networks, read by `A-applied`."""
Job = f420.Job


def stage1_jobs() -> list[Job]:
    """Return stage 1: rotation r1, P0 and A at level 2.0 and the placebo (seed 0), A seed 1 at level 2.0."""
    jobs = []
    for month in TEST_MONTHS:
        for setting in ("L2", "L0"):
            jobs += [Job(a, "r1", setting, month, 0) for a in ("P0", "A")]
        jobs.append(Job("A", "r1", "L2", month, 1))
    return jobs


def full_jobs() -> list[Job]:
    """Return all 105 fits of the full design (the 35 stage-1 fits are a subset)."""
    jobs = []
    for rotation in i417.ROTATIONS:
        for month in TEST_MONTHS:
            for setting in SETTINGS:
                jobs += [Job(a, rotation, setting, month, 0) for a in ("P0", "A")]
            jobs.append(Job("A", rotation, "L2", month, 1))
    return jobs


def applied_jobs(*, rotations: tuple[str, ...]) -> list[Job]:
    """Return the forecast-only jobs of the applied-only diagnostic (experiment 420's A, seed 0)."""
    return [
        Job(APPLIED, rotation, setting, month, 0)
        for rotation in rotations
        for month in TEST_MONTHS
        for setting in SETTINGS
    ]


def target_of(job: Job) -> Path:
    """Return the hidden-pair forecast file of a job."""
    return OUT_DIR / "fits" / job.arm / f"{job.stem}.parquet"


def forecast(
    *,
    model: models.SignedMdnAv,
    prep: f420.Prepared,
    arm: f420.Arm,
    setting: i417.Setting,
    month: str,
    device: str,
    cams_scaler: i417.TemperatureScaler | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the hidden pair's signed forecast and the MVA-only eight's reading forecast on ENS inputs.

    With `cams_scaler` set, the network is experiment 420's CAMS-and-ERA5 network: its one
    temperature column is the ENS mean temperature standardised with the ERA5 scaler.
    """
    fold = prep.fold
    origins = {int(o.timestamp() * 1_000_000) for o in base.test_origins(month=month)}
    names = base.primaries()["site"].to_list()
    pair_pieces, eight_pieces = [], []
    for site in [*fold.pair, *i417.MVA_ONLY]:
        hidden = site in fold.pair
        arr = i417.site_arrays(site=site, fold=True, setting=setting)
        sel = np.isin(arr.origin, list(origins)) & (arr.lead <= MAX_LEAD_HOURS) & np.isfinite(arr.y)
        valid = arr.valid[sel]
        e = ens.ens_arrays(site=site)
        if len(e["u"]) != len(arr.valid):
            msg = f"{site}: extras and summary rows differ"
            raise ValueError(msg)
        u_ens, k_ens, e_in = e["u"][sel], e["k"][sel], e["inputs"][sel]
        u_cams, k_cams = i417.cams_for_rows(site=site, valid_us=valid)
        if cams_scaler is None:
            x = ens.model_inputs(
                x_raw=arr.x[sel],
                ens_inputs=e_in,
                standardiser=fold.standardiser,
                scaler=fold.scaler,  # ty: ignore[invalid-argument-type]
                cols=fold.cols,
            )
        else:
            t = cams_scaler.apply(t=e_in[:, 0])
            x = np.concatenate(
                [fold.standardiser.apply(x=arr.x[sel])[:, fold.cols], t[:, None]], axis=1
            )
        u_in = np.nan_to_num(u_ens, nan=0.0)
        p = names.index(site)
        opened = f420.row_open(
            arm=arm,
            fold=fold,
            gate_map=prep.gates,
            primary=np.full(len(valid), p),
            gate_u=u_in,
            valid_us=valid,
        )
        p_neg, qs = [], []
        with torch.no_grad():
            for s in range(0, len(x), SCORE_CHUNK_ROWS):
                xb = torch.from_numpy(x[s : s + SCORE_CHUNK_ROWS]).to(device)
                ub = torch.from_numpy(u_in[s : s + SCORE_CHUNK_ROWS]).to(device)
                ob = torch.from_numpy(opened[s : s + SCORE_CHUNK_ROWS]).to(device)
                pb = torch.full((len(xb),), p, dtype=torch.int64, device=device)
                log_w, mean, std = (a.double() for a in model.parts(xb, pb, ub))
                if hidden:
                    p_neg.append(
                        models.p_negative(log_w=log_w, mean=mean, std=std, open_mask=ob)
                        .cpu()
                        .numpy()
                    )
                    qs.append(
                        models.signed_quantiles(log_w=log_w, mean=mean, std=std, open_mask=ob)
                        .cpu()
                        .numpy()
                    )
                else:
                    qs.append(
                        models.reading_quantiles(log_w=log_w, mean=mean, std=std, open_mask=ob)
                        .cpu()
                        .numpy()
                    )
        q = np.sort(np.concatenate(qs), axis=1)
        columns = {
            "site": [site] * int(sel.sum()),
            "origin": arr.origin[sel],
            "valid": valid,
            "lead": arr.lead[sel],
            "u_gate": np.nan_to_num(u_cams),
            "k": np.nan_to_num(k_cams),
            "u_ens": u_in,
            "k_ens": np.nan_to_num(k_ens, nan=0.0),
            "reading_points": arr.y[sel] * 100.0,
            "truth_points": arr.y_truth[sel] * 100.0,
            "row_open": opened,
        }
        if hidden:
            columns["p_negative"] = np.concatenate(p_neg).astype(np.float32)
        frame = pl.DataFrame(columns).hstack(
            pl.DataFrame(q.astype(np.float32), schema=list(LEVEL_COLUMNS), orient="row")
        )
        (pair_pieces if hidden else eight_pieces).append(frame)
    return pl.concat(pair_pieces), pl.concat(eight_pieces)


def write_outputs(
    *, job: Job, pair_frame: pl.DataFrame, eight_frame: pl.DataFrame, info: dict
) -> None:
    """Write a job's two forecast files, then its JSON (written last: its presence marks a finished job)."""
    target = target_of(job)
    target.parent.mkdir(parents=True, exist_ok=True)
    for path, frame in ((target, pair_frame), (target.with_suffix(".eight.parquet"), eight_frame)):
        tmp = path.with_name(path.name + ".tmp")
        frame.write_parquet(tmp)
        tmp.rename(path)
    tmp_json = target.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(info))
    tmp_json.rename(target.with_suffix(".json"))


def run_job(*, job: Job, fold: i417.Fold, device: str) -> None:
    """Fit one job on a shared ENS fold, forecast, and write its files atomically."""
    arm = ARMS[job.arm]
    started = time.monotonic()
    refit = f412.refit_time(month=job.month)
    prep = f420.prepare(arm=arm, fold=fold)
    print(f"  gates: {json.dumps(prep.info['gates'])}", flush=True)
    model = f420.make_model(arm=arm, fold=fold, refit=refit, seed=job.seed)
    epochs, val, shares = f417.train(model=model, prep=prep, seed=job.seed, device=device)  # ty: ignore[invalid-argument-type]
    pair_frame, eight_frame = forecast(
        model=model,
        prep=prep,
        arm=arm,
        setting=SETTINGS[job.setting],
        month=job.month,
        device=device,
    )
    seconds = time.monotonic() - started
    state = OUT_DIR / "models" / job.arm / f"{job.stem}.pt"
    state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), state)
    info = {
        "epochs": epochs,
        "val_nll": val,
        "seconds": seconds,
        "prior_gradient_share": shares,
        "capacity_points": model.capacity().detach().cpu().numpy().tolist(),
        "dropped": fold.n_dropped,
        **prep.info,
    }
    write_outputs(job=job, pair_frame=pair_frame, eight_frame=eight_frame, info=info)
    print(
        f"{job.arm} {job.stem}: {epochs} epochs, val {val:.4f}, {seconds:.0f} s, "
        f"dropped {sum(fold.n_dropped.values())}",
        flush=True,
    )


def run_applied_job(*, job: Job, device: str) -> None:
    """Forecast one month with experiment 420's saved arm A network (CAMS fold, ENS inputs)."""
    arm = f420.ARMS["A"]
    refit = f412.refit_time(month=job.month)
    cams_fold = i417.build_fold(
        refit=refit, pair=i417.ROTATIONS[job.rotation], setting=SETTINGS[job.setting]
    )
    prep = f420.prepare(arm=arm, fold=cams_fold)
    model = f420.make_model(arm=arm, fold=cams_fold, refit=refit, seed=job.seed)
    state_file = CAMS_MODELS / f"{job.stem}.pt"
    model.load_state_dict(torch.load(state_file, map_location=device))
    model.to(device).eval()
    pair_frame, eight_frame = forecast(
        model=model,
        prep=prep,
        arm=arm,
        setting=SETTINGS[job.setting],
        month=job.month,
        device=device,
        cams_scaler=cams_fold.scaler,  # ty: ignore[invalid-argument-type]
    )
    info = {
        "gates": prep.info["gates"],
        "applied_from": str(state_file),
        "dropped": cams_fold.n_dropped,
    }
    write_outputs(job=job, pair_frame=pair_frame, eight_frame=eight_frame, info=info)
    print(f"{job.arm} {job.stem}: applied", flush=True)


def stop_if_low_memory() -> None:
    """Exit if the `available` column of `free -g` is under the stop line."""
    free = f420.available_gib()
    if free < MIN_AVAILABLE_GIB:
        print(f"STOP: {free:.1f} GiB available, below {MIN_AVAILABLE_GIB:.0f}", flush=True)
        sys.exit(3)


def run(*, jobs: list[Job], device: str) -> None:
    """Run the jobs not yet finished, one fold at a time, and stop at the memory line."""
    todo = [j for j in jobs if not target_of(j).with_suffix(".json").exists()]
    print(f"{len(todo)} of {len(jobs)} jobs to run", flush=True)
    groups: dict[tuple[str, str, str], list[Job]] = {}
    for job in todo:
        if job.arm != APPLIED:
            groups.setdefault((job.rotation, job.setting, job.month), []).append(job)
    for (rotation, setting, month), group in groups.items():
        stop_if_low_memory()
        fold = ens.build_fold(
            refit=f412.refit_time(month=month),
            pair=i417.ROTATIONS[rotation],
            setting=SETTINGS[setting],
        )
        for job in group:
            stop_if_low_memory()
            run_job(job=job, fold=fold, device=device)
        del fold
    for job in todo:
        if job.arm == APPLIED:
            stop_if_low_memory()
            run_applied_job(job=job, device=device)


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "list"))
    parser.add_argument("--stage", choices=("1", "full"), default="1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--applied", action="store_true", help="add the forecast-only A-applied jobs"
    )
    args = parser.parse_args()
    jobs = stage1_jobs() if args.stage == "1" else full_jobs()
    if args.applied:
        jobs += applied_jobs(rotations=("r1",) if args.stage == "1" else tuple(i417.ROTATIONS))
    if args.action == "list":
        counts: dict[str, int] = {}
        for j in jobs:
            counts[j.arm] = counts.get(j.arm, 0) + 1
        print(len(jobs), "jobs", counts)
        return
    run(jobs=jobs, device=args.device)


if __name__ == "__main__":
    main()

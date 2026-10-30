# ruff: noqa: E501
"""Fit the arms of experiment 420 and forecast the test months.

Arms (each fitted on experiment 417's CAMS-and-ERA5 inputs, monthly refit):

- `P0`: experiment 417's network with every folded Primary's gate forced shut (`X >= 0` at every
  row), the plain neural network read as signed.
- `A`: experiment 417's arm A (the Primary-level export gate).
- `AV`: `A` plus the shared PV-proportional variance term.
- `A-shuffle`: `A` with `u` permuted across days within year, month, and half-hour (the gate keeps
  the real `u`).
- `ref`: experiment 417's reference, with no Primary gate (reported only).

A fold is built once per rotation, setting, and test month and shared across the arms and seeds.
Each fit writes the hidden pair's signed forecast and the MVA-only eight's reading forecast.

```bash
. env.sh   # PYTHONPATH and DATA_PATH_INTERNAL; see the plan's Compute section
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue -p CPUQuota=100% \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00420-*/e00420_fit.py run --stage 1
```
"""

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00397_prep as base  # ty: ignore[unresolved-import]
import e00412_fit as f412  # ty: ignore[unresolved-import]
import e00417_fit as f417  # ty: ignore[unresolved-import]
import e00417_gates as gates  # ty: ignore[unresolved-import]
import e00417_inputs as inputs  # ty: ignore[unresolved-import]
import e00420_models as models
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "neural-mva-export-gated-signed-final"
OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00420-{SLUG}" / "work"
TEST_MONTHS: Final[tuple[str, ...]] = (
    "2025-07",
    "2025-08",
    "2025-09",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
MAX_LEAD_HOURS: Final[float] = f412.MAX_LEAD_HOURS
SCORE_CHUNK_ROWS: Final[int] = f412.SCORE_CHUNK_ROWS
LEVEL_COLUMNS: Final[tuple[str, ...]] = f412.LEVEL_COLUMNS
MIN_AVAILABLE_GIB: Final[float] = 12.0
"""The stop line: the `available` column of `free -g`."""
SETTINGS: Final[dict[str, inputs.Setting]] = {
    "L2": inputs.Setting(2.0, 2.0),
    "L1": inputs.Setting(1.0, 2.0),
    "L0": inputs.Setting(0.0, 2.0),
}


@dataclass(frozen=True)
class Arm:
    """What an arm changes."""

    mode: str = "primary"
    """`none` (no Primary gate), `shut` (every folded gate shut), or `primary` (design A)."""
    variance: bool = False
    shuffle_u: bool = False


ARMS: Final[dict[str, Arm]] = {
    "P0": Arm(mode="shut"),
    "A": Arm(),
    "AV": Arm(variance=True),
    "A-shuffle": Arm(shuffle_u=True),
    "ref": Arm(mode="none"),
}


@dataclass(frozen=True)
class Job:
    """One fit: an arm on a rotation, a setting, a test month, and a seed."""

    arm: str
    rotation: str
    setting: str
    month: str
    seed: int

    @property
    def stem(self) -> str:
        """Return the file stem of the job's outputs."""
        return f"{SETTINGS[self.setting].tag}_{self.rotation}_{self.month}_s{self.seed}"


def stage1_jobs() -> list[Job]:
    """Return stage 1: rotation r1, seed 0, P0, A, and AV at level 2.0 and the placebo, A-shuffle.

    Seed 1 of A at level 2.0 is added so that the minimum detectable effect can be written down
    before any arm-against-arm score is read. Those fits belong to the full design.
    """
    jobs = []
    for month in TEST_MONTHS:
        for setting in ("L2", "L0"):
            jobs += [Job(a, "r1", setting, month, 0) for a in ("P0", "A", "AV")]
        jobs += [Job("A-shuffle", "r1", "L2", month, 0), Job("A", "r1", "L2", month, 1)]
    return jobs


def full_jobs() -> list[Job]:
    """Return all 378 fits of the full design (the 56 stage-1 fits are a subset)."""
    jobs = []
    for rotation in inputs.ROTATIONS:
        for month in TEST_MONTHS:
            for setting in SETTINGS:
                for seed in (0, 1):
                    jobs += [Job(a, rotation, setting, month, seed) for a in ("A", "AV")]
                jobs.append(Job("P0", rotation, setting, month, 0))
                if setting in ("L2", "L0"):
                    jobs.append(Job("ref", rotation, setting, month, 0))
                if setting == "L2":
                    jobs.append(Job("A-shuffle", rotation, setting, month, 0))
    return jobs


def available_gib() -> float:
    """Return the `available` column of `free -g`, in GiB."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return float("inf")


def target_of(job: Job) -> Path:
    """Return the hidden-pair forecast file of a job."""
    return OUT_DIR / "fits" / job.arm / f"{job.stem}.parquet"


def make_model(*, arm: Arm, fold: inputs.Fold, refit, seed: int) -> models.SignedMdnAv:  # noqa: ANN001
    """Return an untrained network (the variance scale is frozen at a tiny value when the arm has none)."""
    torch.manual_seed(seed)
    centre = f412.prior_centre_points(refit=refit)
    n = len(fold.target_mean)
    model = models.SignedMdnAv(
        n_inputs=fold.x.shape[1],
        n_primaries=n,
        target_mean=fold.target_mean,
        target_std=fold.target_std,
        folded=fold.folded,
        capacity_start=np.full(n, centre),
        prior_centre_points=centre,
    )
    if not arm.variance:
        model.raw_s.requires_grad_(False)
        model.raw_s.fill_(-30.0)  # softplus(-30) is about 1e-13: no variance term
    return model


@dataclass
class Prepared:
    """A fold with the arm's masks, in the shape `e00417_fit.train` reads."""

    fold: inputs.Fold
    open_row: np.ndarray
    label: np.ndarray
    gates: dict
    info: dict


def prepare(*, arm: Arm, fold: inputs.Fold) -> Prepared:
    """Compute an arm's Primary gates and training-row mask from the fold's training rows."""
    gate_map = gates.primary_gates(fold=fold) if arm.mode != "none" else {}
    open_row = row_open(
        arm=arm,
        fold=fold,
        gate_map=gate_map,
        primary=fold.primary,
        gate_u=fold.gate_u,
        valid_us=fold.valid,
    )
    info = {
        "gates": {
            s: {"open": g.open, "spearman": g.spearman, "p1_points": g.p1_points, "n": g.n_rows}
            for s, g in gate_map.items()
        }
    }
    return Prepared(
        fold=fold,
        open_row=open_row,
        label=np.zeros(len(fold.y), dtype=np.int8),
        gates=gate_map,
        info=info,
    )


def row_open(
    *,
    arm: Arm,
    fold: inputs.Fold,
    gate_map: dict,
    primary: np.ndarray,
    gate_u: np.ndarray,
    valid_us: np.ndarray,
) -> np.ndarray:
    """Return where the folded density applies: nowhere for `shut`, else experiment 417's rule."""
    if arm.mode == "shut":
        return np.zeros(len(primary), dtype=bool)
    return gates.open_mask(
        fold=fold,
        primary=primary,
        gate_u=gate_u,
        valid_us=valid_us,
        mode=arm.mode,
        gates=gate_map,
        thresholds=None,
    )


def shuffled_fold(*, fold: inputs.Fold) -> inputs.Fold:
    """Return the fold with `u` replaced by the shuffled sky (the gate keeps the real `u`)."""
    names = base.primaries()["site"].to_list()
    u = np.zeros_like(fold.u)
    for p, site in enumerate(names):
        sel = fold.primary == p
        if sel.any():
            u[sel] = np.nan_to_num(inputs.shuffled_u(site=site, valid_us=fold.valid[sel]))
    return dataclasses.replace(fold, u=u)


def forecast(
    *,
    model: models.SignedMdnAv,
    prep: Prepared,
    arm: Arm,
    setting: inputs.Setting,
    month: str,
    device: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the hidden pair's signed forecast and the MVA-only eight's reading forecast."""
    fold = prep.fold
    origins = {int(o.timestamp() * 1_000_000) for o in base.test_origins(month=month)}
    names = base.primaries()["site"].to_list()
    pair_pieces, eight_pieces = [], []
    sites = [*fold.pair, *(() if arm.shuffle_u else inputs.MVA_ONLY)]
    for site in sites:
        hidden = site in fold.pair
        arr = inputs.site_arrays(site=site, fold=True, setting=setting)
        sel = np.isin(arr.origin, list(origins)) & (arr.lead <= MAX_LEAD_HOURS) & np.isfinite(arr.y)
        valid = arr.valid[sel]
        u_gate, k = inputs.cams_for_rows(site=site, valid_us=valid)
        u_in = inputs.shuffled_u(site=site, valid_us=valid) if arm.shuffle_u else u_gate
        x = inputs.model_inputs(
            x_raw=arr.x[sel],
            valid_us=valid,
            site=site,
            standardiser=fold.standardiser,
            scaler=fold.scaler,
            cols=fold.cols,
        )
        p = names.index(site)
        opened = row_open(
            arm=arm,
            fold=fold,
            gate_map=prep.gates,
            primary=np.full(len(valid), p),
            gate_u=np.nan_to_num(u_gate),
            valid_us=valid,
        )
        p_neg, qs = [], []
        with torch.no_grad():
            for s in range(0, len(x), SCORE_CHUNK_ROWS):
                xb = torch.from_numpy(x[s : s + SCORE_CHUNK_ROWS]).to(device)
                ub = torch.from_numpy(np.nan_to_num(u_in[s : s + SCORE_CHUNK_ROWS])).to(device)
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
            "u_gate": np.nan_to_num(u_gate),
            "k": np.nan_to_num(k),
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
    eight = pl.concat(eight_pieces) if eight_pieces else pl.DataFrame()
    return pl.concat(pair_pieces), eight


def run_job(*, job: Job, fold: inputs.Fold, device: str) -> None:
    """Fit one job on a shared fold, forecast, and write its files atomically."""
    arm = ARMS[job.arm]
    started = time.monotonic()
    refit = f412.refit_time(month=job.month)
    use_fold = shuffled_fold(fold=fold) if arm.shuffle_u else fold
    prep = prepare(arm=arm, fold=use_fold)
    print(f"  gates: {json.dumps(prep.info['gates'])}", flush=True)
    model = make_model(arm=arm, fold=use_fold, refit=refit, seed=job.seed)
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
    target = target_of(job)
    target.parent.mkdir(parents=True, exist_ok=True)
    for path, frame in ((target, pair_frame), (target.with_suffix(".eight.parquet"), eight_frame)):
        if len(frame):
            tmp = path.with_name(path.name + ".tmp")
            frame.write_parquet(tmp)
            tmp.rename(path)
    state = OUT_DIR / "models" / job.arm / f"{job.stem}.pt"
    state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), state)
    info = {
        "epochs": epochs,
        "val_nll": val,
        "seconds": seconds,
        "prior_gradient_share": shares,
        "capacity_points": model.capacity().detach().cpu().numpy().tolist(),
        "variance_scale": float(model.variance_scale().detach()),
        "dropped": fold.n_dropped,
        **prep.info,
    }
    tmp_json = target.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(info))
    tmp_json.rename(target.with_suffix(".json"))  # written last: its presence marks a finished job
    print(
        f"{job.arm} {job.stem}: {epochs} epochs, val {val:.4f}, {seconds:.0f} s, "
        f"s {info['variance_scale']:.3f}, dropped {sum(fold.n_dropped.values())}",
        flush=True,
    )


def run(*, jobs: list[Job], device: str) -> None:
    """Run the jobs not yet finished, one fold at a time, and stop at the memory line."""
    todo = [j for j in jobs if not target_of(j).with_suffix(".json").exists()]
    print(f"{len(todo)} of {len(jobs)} fits to run", flush=True)
    groups: dict[tuple[str, str, str], list[Job]] = {}
    for job in todo:
        groups.setdefault((job.rotation, job.setting, job.month), []).append(job)
    for (rotation, setting, month), group in groups.items():
        free = available_gib()
        if free < MIN_AVAILABLE_GIB:
            print(f"STOP: {free:.1f} GiB available, below {MIN_AVAILABLE_GIB:.0f}", flush=True)
            sys.exit(3)
        refit = f412.refit_time(month=month)
        fold = inputs.build_fold(
            refit=refit, pair=inputs.ROTATIONS[rotation], setting=SETTINGS[setting]
        )
        for job in group:
            free = available_gib()
            if free < MIN_AVAILABLE_GIB:
                print(f"STOP: {free:.1f} GiB available, below {MIN_AVAILABLE_GIB:.0f}", flush=True)
                sys.exit(3)
            run_job(job=job, fold=fold, device=device)
        del fold


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "list"))
    parser.add_argument("--stage", choices=("1", "full"), default="1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    jobs = stage1_jobs() if args.stage == "1" else full_jobs()
    if args.action == "list":
        print(len(jobs), "fits")
        counts: dict[str, int] = {}
        for j in jobs:
            counts[j.arm] = counts.get(j.arm, 0) + 1
        print(counts)
        return
    run(jobs=jobs, device=args.device)


if __name__ == "__main__":
    main()

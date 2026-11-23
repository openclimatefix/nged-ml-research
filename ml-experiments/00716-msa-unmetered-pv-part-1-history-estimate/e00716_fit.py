# ruff: noqa: E501, PT018, PLR0915, E731
"""Experiment 716: fit one PV arm (one arm, one fold, one seed) in the training phase (design section 6, WP4).

Usage: e00716_fit.py ARM MONTH SEED [--device cuda] [--accept-code-change OLD_HASH ...]
       [--accept-gate-change OLD_HASH] [--accept-713-hash OLD_HASH ...]

Run it through `run.sh`. It sets `E00667_PHASE=train`. It refuses to run without
`work/known_answers_passed.json` (written by `e00716_known_answer.py`) and without experiment 713's
gate record (which may be waived, and the waiver is logged). Every arm starts stages 2 and 3 from the
stored stage 1 state of `M*` of the same fold and seed, and refuses a state whose experiment 713 code
hash differs from the current one unless `--accept-713-hash` names the old hash. A stage that ends at
the 60-pass cap is rerun from its starting state with a cap of 120. Each stage's state is saved
atomically, and a stage whose file exists is skipped. A resume, a finished-job skip, and the gate are
refused when the code hash differs, as in experiment 713.
"""

import argparse
import copy
import json
import os
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("E00667_PHASE", "train")

import e00713_arms as arms713  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00716_arms as arms
import e00716_batch as pvb
import e00716_model as pvm
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
CAP_FIRST, CAP_RERUN = 60, 120


@dataclass
class FitInputs:
    """What a fit reads, besides the stored states.

    Attributes:
        world: The training world; its `source` is a `PvSource`.
        dropped: The per-fold counts of dropped half-hours.
        fingerprint: The hash of the fold's standardiser.
        encoders: `frozen`, `trained`, or `random`, as experiment 713's `M*` resolves.
    """

    world: tr.World
    dropped: dict
    fingerprint: str
    encoders: str


Loader = Callable[[str, str, int, str], FitInputs]
"""A loader of (arm, month, seed, device) returning the fit's inputs."""


def write_json_atomic(*, obj: dict, path: Path) -> None:
    """Write a JSON file under a temporary name that carries the process id, then rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.rename(path)


def stage_log_file(*, data_dir: Path, arm: str, month: str, seed: int, stage: int) -> Path:
    """Return the log of one stage."""
    path = arms.log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit")
    return path.with_name(f"{path.stem}_stage{stage}.json")


def new_model(
    *,
    arm: str,
    seed: int,
    month: str,
    device: str,
    data_dir: Path,
    encoders: str,
    log: dict,
    accept_713: Sequence[str],
) -> pvm.PvNet:
    """Build the network of `arm`, load the stored stage 1 state of `M*`, and freeze the encoders as `M*` does."""
    spec = arms.ARMS[arm]
    state_path = arms.m_star_stage1_state(data_dir=data_dir, month=month, seed=seed)
    log_path = arms.m_star_stage1_log(data_dir=data_dir, month=month, seed=seed)
    assert state_path.exists() and log_path.exists(), (
        f"the stage 1 state of M* is missing: {state_path}"
    )
    arms.check_713_warm_start(
        stage1_log=json.loads(log_path.read_text()), accept_713=accept_713, record=log,
        what=f"M* stage 1 {month} s{seed}",
    )  # fmt: skip
    model = pvm.PvNet(arm=spec.net, seed=seed)  # ty: ignore[invalid-argument-type]
    model.load_m_star(state=torch.load(state_path, map_location="cpu"))
    if encoders == "frozen":
        model.freeze_encoders()
    return model.to(device)


def fit(
    *, arm: str, month: str, seed: int, device: str, data_dir: Path, loader: Loader,
    accept: Sequence[str] = (), accept_gate: str | None = None, accept_713: Sequence[str] = (),
    require_gate: bool = True,
) -> dict:  # fmt: skip
    """Fit stages 2 and 3 and return the log.

    `loader` builds the fold's inputs (`default_loader` reads the real data; the checks pass a synthetic one).
    """
    log: dict = {"arm": arm, "month": month, "seed": seed}
    if require_gate:
        gate = arms.gate_file(data_dir=data_dir)
        assert gate.exists(), f"{gate} is missing: run e00716_known_answer.py first"
        arms.check_code(
            earlier=json.loads(gate.read_text()).get("code", {}), accept=[accept_gate] if accept_gate else [],
            record=log, what="the known-answer gate file", kind="gate",
        )  # fmt: skip
        log["e00713_gate"] = arms.gate713_summary(data_dir=data_dir)
        print("experiment 713's gate:", json.dumps(log["e00713_gate"], default=str), flush=True)
    final = arms.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
    if final.exists():
        return skip_finished(
            arm=arm, month=month, seed=seed, data_dir=data_dir, accept=accept, log=log
        )
    started = time.monotonic()
    before = arms.watched_mtimes(data_dir=data_dir)
    inputs = loader(arm, month, seed, device)
    check_fingerprint(inputs=inputs, data_dir=data_dir, month=month, arm=arm)
    model = new_model(
        arm=arm, seed=seed, month=month, device=device, data_dir=data_dir, encoders=inputs.encoders,
        log=log, accept_713=accept_713,
    )  # fmt: skip
    log.update(
        stages=[], dropped=inputs.dropped, recipe=arms713.recipe_hashes(), **arms.code_record()
    )
    for stage in (2, 3):
        path = arms.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=stage)
        stage_log = stage_log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=stage)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            assert stage_log.exists(), f"{path} has no stage log: refit the stage"
            earlier = json.loads(stage_log.read_text())
            arms.check_code(earlier=earlier, accept=accept, record=log, what=f"{path.name}")
            log["stages"].append(earlier)
            continue
        samples = tr.make_samples(world=inputs.world, stage=stage, placebo=False)
        scale: dict[str, float] = {}
        if stage == 2 and model.pv.any_path:
            cut = tr.target_end(world=inputs.world, stage=2)
            from e00716_train import calibration_batches

            scale = model.calibrate_scale(
                batches=calibration_batches(
                    world=inputs.world, samples=samples, source=inputs.world.source, cut=cut
                )
            )  # ty: ignore[invalid-argument-type]
        start_state = copy.deepcopy(model.state_dict())
        record = train_with_rerun(
            model=model,
            world=inputs.world,
            stage=stage,
            samples=samples,
            seed=seed,
            encoders=inputs.encoders,
            start_state=start_state,
        )
        entry = {
            "stage": stage,
            "pairs": len(samples),
            "train_pairs": int((~samples.is_val).sum()),
            "first": vars(record[0]).copy(),
            "rerun": None if record[1] is None else vars(record[1]).copy(),
        }
        final_record = record[1] or record[0]
        entry.update(
            passes=final_record.passes, best_val=final_record.best_val, cap_hit=final_record.cap_hit,
            scale=scale, **arms.code_record(), recipe=arms713.recipe_hashes(),
        )  # fmt: skip
        log["stages"].append(entry)
        write_json_atomic(
            obj=entry, path=stage_log
        )  # written before the state file, so a state file always has its log
        tr.save_atomic(state=model.state_dict(), path=path)
        print(
            f"{arm} {month} s{seed} stage {stage}: {final_record.passes} passes, val {final_record.best_val:.4f}",
            flush=True,
        )
    assert arms.watched_mtimes(data_dir=data_dir) == before, (
        "a file of experiment 713's data changed during the fit"
    )
    log["seconds"] = round(time.monotonic() - started, 1)
    log["peak_host_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if device.startswith("cuda"):
        log["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
    print(
        f"{arm} {month} s{seed}: peak host {log['peak_host_mb']:.0f} MB"
        + (f", peak GPU {log['peak_gpu_mb']:.0f} MB" if "peak_gpu_mb" in log else ""),
        flush=True,
    )
    write_json_atomic(
        obj=log, path=arms.log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit")
    )
    return log


def train_with_rerun(
    *, model: pvm.PvNet, world: tr.World, stage: int, samples: tr.SampleSet, seed: int, encoders: str, start_state: dict,
) -> tuple[tr.StageLog, tr.StageLog | None]:  # fmt: skip
    """Train a stage at a cap of 60 passes; a stage that ends at the cap is rerun from its starting state at 120."""
    import e00716_train as pvt

    first = pvt.train_stage(
        model=model,
        world=world,
        stage=stage,
        samples=samples,
        seed=seed,
        encoders=encoders,
        max_passes=CAP_FIRST,
    )
    if not first.cap_hit:
        return first, None
    model.load_state_dict(start_state)
    again = pvt.train_stage(
        model=model,
        world=world,
        stage=stage,
        samples=samples,
        seed=seed,
        encoders=encoders,
        max_passes=CAP_RERUN,
    )
    return first, again


def skip_finished(
    *, arm: str, month: str, seed: int, data_dir: Path, accept: Sequence[str], log: dict
) -> dict:
    """Handle a job whose stage 3 state exists: check the code, rebuild a missing final log, and return it."""
    final_log = arms.log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit")
    stages = []
    for stage in (2, 3):
        path = stage_log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=stage)
        assert path.exists(), f"{path} is missing: refit the job"
        earlier = json.loads(path.read_text())
        arms.check_code(
            earlier=earlier, accept=accept, record=log, what=f"{arm} {month} s{seed} stage {stage}"
        )
        stages.append(earlier)
    if final_log.exists():
        existing = json.loads(final_log.read_text())
        new = [
            o for o in log.get("code_overrides", []) if o not in existing.get("code_overrides", [])
        ]
        if new:
            existing.setdefault("code_overrides", []).extend(new)
            write_json_atomic(obj=existing, path=final_log)
        print(f"{arm} {month} s{seed}: done, skipped", flush=True)
        return existing
    rebuilt = {
        **log,
        "stages": stages,
        "rebuilt_from_stage_logs": True,
        "code_hash": stages[-1].get("code_hash"),
        "git_commit": stages[-1].get("git_commit"),
    }
    write_json_atomic(obj=rebuilt, path=final_log)
    print(f"{arm} {month} s{seed}: done, final log rebuilt from the stage logs", flush=True)
    return rebuilt


def check_fingerprint(*, inputs: FitInputs, data_dir: Path, month: str, arm: str) -> None:
    """Write the fold's standardiser hash, or assert it is unchanged (the predict phase compares it)."""
    path = arms.root(data_dir=data_dir, arm=arm) / "work" / "standardiser" / f"{month}.sha256"
    if path.exists():
        assert path.read_text() == inputs.fingerprint, "the standardiser changed between two fits"
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(inputs.fingerprint)
    tmp.rename(path)


# ------------------------------------------------------------------ the real loader


def build_pv_tables(  # noqa: PLR0912
    *,
    pipe,  # noqa: ANN001
    data,  # noqa: ANN001
    arm: str,
    device: str,
    readout: str | None = None,
) -> pvb.PvTables:
    """Build the PV arrays of `arm` for one fold from the real data (`FoldData` built with `keep_rows=True`).

    The history estimate is built from the dense raw table, the as-of capacity of the fold, and
    `target_std`; the placebo `H-S` shifts the sky by 28 days and `H-X` reads the partner's estimate. The
    target row is built from the float32 Candidate rows, with the origin's own run (`real`) or the run 28
    days earlier (`shifted`).
    """
    import e00397_prep as prep397  # ty: ignore[unresolved-import]
    import e00716_pv as pv
    import numpy as np
    import polars as pl  # noqa: F401

    spec = arms.ARMS[arm]
    sites = [r["site"] for r in prep397.primaries().iter_rows(named=True)]
    power_cap = np.array(
        [float(r["effective_capacity_mw"]) for r in prep397.primaries().iter_rows(named=True)]
    )
    capacity_fold = pipe.month
    if readout == "lookahead" and pipe.month < "2026-03":
        capacity_fold = "2026-03"  # the six read the 2026-03 capacity of experiment 649; the eight are unchanged (their cutoff is already 2026-03)
    caps = pv.capacity_table(fold=capacity_fold, sites=sites)
    for line in pv.lookahead_statement(caps=caps):
        print("capacity:", line, flush=True)
    cap_mw = np.array([c.mw for c in caps])
    std = np.asarray(data.target_std, dtype=np.float64)
    raw = pipe.d632.dense_raw()
    n_grid = raw.shape[1]
    doy = pv.day_of_year(grid_start_us=pv.GRID_START_US, n_grid=n_grid)
    if spec.history == "none":
        hist_p = np.zeros((16, n_grid), dtype=np.float32)
        hist_ok = np.zeros((16, n_grid), dtype=bool)
    else:
        if readout == "cams":
            temp = pv.monthly_mean_temperature(raw=raw, cutoff_idx=data.cutoff_idx)
            est = pv.dense_share(
                raw=raw,
                doy=doy,
                capacities_mw=cap_mw,
                power_capacity_mw=power_cap,
                temperature_override=temp,
            )
        elif spec.history == "shifted":
            ghi, temp, _, _ = pv.shifted_sky(raw=raw)
            est = pv.dense_share(
                raw=raw,
                doy=doy,
                capacities_mw=cap_mw,
                power_capacity_mw=power_cap,
                ghi_override=ghi,
                temperature_override=temp,
            )
        else:
            est = pv.dense_share(
                raw=raw, doy=doy, capacities_mw=cap_mw, power_capacity_mw=power_cap
            )
        if spec.history == "partner":
            hist_p, hist_ok = pv.partner_p(
                share=est.share,
                ok=est.ok,
                partner=pv.host_partner_index(sites=sites),
                target_std=std,
            )
        else:
            hist_p, hist_ok = pv.p_dense(estimate=est, target_std=std)
    mr, test = data.rows
    standardiser = mr.base.standardiser
    first_day, n_days = data.source.ens.first_day, data.source.ens.filled.shape[1]
    from e00422_prep import GRID_START_US, STEP_US  # ty: ignore[unresolved-import]

    origin_idx = ((mr.origin_us - GRID_START_US) // STEP_US).astype(np.int64)
    raw_t = pv.target_raw_from_rows(
        x=mr.x, primary=mr.primary, origin_idx=origin_idx, target_idx=mr.target_idx, origin_us=mr.origin_us,
        mean=standardiser.mean, std=standardiser.std, first_day=first_day, n_days=n_days,
    )  # fmt: skip
    if test is not None:
        scored = pv.target_raw_from_rows(
            x=test.x, primary=test.primary, origin_idx=test.origin_idx, target_idx=test.target_idx,
            origin_us=test.origin, mean=standardiser.mean, std=standardiser.std, first_day=first_day, n_days=n_days,
        )  # fmt: skip
        raw_t = pv.merge_target_raw(base=raw_t, other=scored)
    donor = None
    exception_days: frozenset[int] = frozenset()
    run_us = raw_t.run_us
    if spec.target == "shifted":
        filled_days = np.nonzero(raw_t.filled.any(axis=(0, 2)))[0]
        first_ens = int(first_day + filled_days.min())
        abs_day = first_day + np.arange(n_days)
        donor_abs, early = pv.donor_origin_day(origin_day=abs_day, first_ens_day=first_ens)
        donor = donor_abs - first_day
        exception_days = frozenset(int(d) for d in abs_day[early & raw_t.filled.any(axis=(0, 2))])
        run_us = np.where(
            (donor[None, :] >= 0) & (donor[None, :] < n_days),
            raw_t.run_us[:, np.clip(donor, 0, n_days - 1)],
            0,
        )
    share_scale = np.where(cap_mw > 0, cap_mw / power_cap, 0.0)
    if spec.target == "none":
        top = pv.TargetEstimate(
            p0=np.zeros((16, n_days, pv.N_ENS_STEPS, 2), np.float32),
            filled=np.zeros((16, n_days, pv.N_ENS_STEPS), bool),
        )
    else:
        top = pv.target_estimate(
            raw=raw_t,
            doy_of_day=pv.target_doy(first_day=first_day, n_days=n_days),
            share_scale=share_scale,
            target_std=std,
            donor_day=donor,
        )
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return pvb.PvTables(
        hist_p=to(hist_p), hist_ok=to(hist_ok), top_p0=to(top.p0), top_filled=to(top.filled), first_day=first_day,
        run_us=to(run_us.astype(np.int64)), exception_days=exception_days,
    )  # fmt: skip


def default_loader(arm: str, month: str, seed: int, device: str) -> FitInputs:
    """Read the real fold: 713's pipeline and arrays, then this arm's PV arrays."""
    import e00713_inputs as inp  # ty: ignore[unresolved-import]

    arms.install_own_cache()
    pipe = inp.load_pipeline(
        month=month, phase="train", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device, keep_rows=True)
    tables = build_pv_tables(pipe=pipe, data=data, arm=arm, device=device)
    source = pvb.PvSource(source=data.source, tables=tables)
    world = tr.World(
        source=source, cutoff_idx=data.cutoff_idx, clean=data.clean, target_mean=data.target_mean,
        target_std=data.target_std, month=month, placebo_pool=None,
    )  # fmt: skip
    star = arms713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    return FitInputs(
        world=world,
        dropped=data.dropped_per_fold,
        fingerprint=data.fingerprint,
        encoders=arms713.SCREEN_ARMS[star][1],
    )


def build_pv_source_for_check(
    *, month: str, arm: str = "HT", phase: str = "train"
) -> tuple[pvb.PvSource, tr.World, np.ndarray | None]:
    """Return the real fold's `PvSource` of `arm` on the CPU, 713's `World` around it, and the scored origins, for KF2 on real arrays (`e00716_check_batch.py real MONTH`).

    The scored origins (`test_origin_idx`) exist only in the `predict` phase, whose process must have
    set `E00667_PHASE=predict` before importing this module; the `train` phase returns `None`.
    """
    import e00713_inputs as inp  # ty: ignore[unresolved-import]

    arms.install_own_cache()
    assert os.environ["E00667_PHASE"] == phase, "set E00667_PHASE before importing the pipeline"
    pipe = inp.load_pipeline(month=month, phase=phase, chain_root=inp.chain_root_of_seed(seed=0))
    data = inp.build_fold_data(pipe=pipe, device="cpu", keep_rows=True)
    source = pvb.PvSource(
        source=data.source, tables=build_pv_tables(pipe=pipe, data=data, arm=arm, device="cpu")
    )
    world = tr.World(
        source=source, cutoff_idx=data.cutoff_idx, clean=data.clean, target_mean=data.target_mean,
        target_std=data.target_std, month=month, placebo_pool=None,
    )  # fmt: skip
    return source, world, data.test_origin_idx


def main() -> int:
    """Parse the command line and fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--accept-code-change", action="append", default=[], metavar="OLD_HASH")
    parser.add_argument("--accept-gate-change", default=None, metavar="OLD_HASH")
    parser.add_argument("--accept-713-hash", action="append", default=[], metavar="OLD_HASH")
    args = parser.parse_args()
    fit(
        arm=args.arm, month=args.month, seed=args.seed, device=args.device, data_dir=ML_EXPERIMENTS_DATA_DIR,
        loader=default_loader, accept=args.accept_code_change, accept_gate=args.accept_gate_change,
        accept_713=args.accept_713_hash,
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())

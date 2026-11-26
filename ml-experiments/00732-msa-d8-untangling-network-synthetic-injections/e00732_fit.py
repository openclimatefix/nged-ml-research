# ruff: noqa: E501
"""Experiment 732: one fit, the stored predictions of a fit, the timing gate, and the chain runner (torch).

A fit stores its record (`fits/<name>.json`, with the per-epoch history of every loss term), its best
weights (`networks/<name>.pt`), and its predictions on every set a scoring needs
(`predictions/<name>/*.npz`). Every array is flat on the valid half-hours in experiment 729's `Rows`
order (the overlay sets are `(P, 48)` arrays cut to the valid half-hours).
"""

import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import e00732_paths  # noqa: F401

# isort: split
import e00729_banks as banks729
import e00729_train as train729
import e00730_banks as banks730
import e00732_arms as arms
import e00732_core as core
import e00732_leakage as leak
import e00732_nets as nets
import e00732_overlays as ov
import e00732_stages as stages
import e00732_train as train
import e00732_truth as truth
import numpy as np
import torch

Log = stages.Log


@dataclass
class Sets:
    """The prepared sets of one fit.

    Attributes:
        train: The training bank with its transfer overlay.
        index: The training bank's pair index.
        val_plain: The validation bank, plain.
        val_transfer: The validation bank with its transfer overlay.
        val_index: The validation bank's pair index.
    """

    train: list[arms.Prepared]
    index: ov.PairIndex
    val_plain: list[arms.Prepared]
    val_transfer: list[arms.Prepared]
    val_index: ov.PairIndex


def load_sets(*, root730: Path, env: stages.Env, seed: int) -> Sets:
    """Build the prepared training and validation sets of one seed (from experiment 730's banks, read only).

    The neighbour table and the training medians come from `env`, and every set reads them.
    """
    tr = banks729.load_bank(path=banks730.bank_path(root=root730, kind="train", seed=seed))
    va = banks729.load_bank(path=banks730.bank_path(root=root730, kind="validation"))
    common = {"holidays": env.holidays, "stats": env.stats, "medians": env.medians}
    index, val_index = (
        ov.build_pair_index(segments=tr.segments),
        ov.build_pair_index(segments=va.segments),
    )
    leak.assert_pair_index_from_own_bank(index=index, segments=tr.segments)
    leak.assert_pair_index_from_own_bank(index=val_index, segments=va.segments)
    return Sets(
        train=arms.prepare(overlays=ov.overlay_segments(segments=tr.segments, kind="train", seed=seed, neighbours=env.neighbours), **common),
        index=index,
        val_plain=arms.prepare_plain(segments=va.segments, neighbours=env.neighbours, **common),
        val_transfer=arms.prepare(overlays=ov.overlay_segments(segments=va.segments, kind="validation", neighbours=env.neighbours), **common),
        val_index=val_index,
    )  # fmt: skip


def assert_fit_inputs(*, spec: core.ArmSpec732, sets: Sets, seed: int) -> None:
    """Run the plan's input leakage assertions on one training batch of the arm (verdict row 1 on a failure)."""
    leak.assert_arm_weights(arm=spec.name)
    leak.assert_no_month_or_day_of_year()
    rng = np.random.default_rng(seed)
    picks = arms.sample_picks(prepared=sets.train, n=64, rng=rng) + stages.solar_world_picks(
        prepared=sets.train, limit=32
    )
    b = arms.make_batch(
        prepared=sets.train,
        picks=picks,
        spec=spec,
        rng=rng,
        index=sets.index if spec.inject else None,
    )
    stages.assert_batch_channels(batch=b, spec=spec)
    leak.assert_inputs_not_targets(windows=b.x, targets=b.t, delta=b.delta)
    if b.xi is not None:
        leak.assert_inputs_not_targets(windows=b.xi, targets=b.ti, delta=b.delta)  # ty: ignore[invalid-argument-type]
    if (b.keeps[:, arms.K["sign"]] > 0.5).sum() == 0:
        msg = "the sign term reads no half-hour of a training batch"
        raise leak.CorrectnessFailureError(msg)


def assert_net_checks(*, net: nets.Net, spec: core.ArmSpec732, x: torch.Tensor) -> None:
    """Check, on the fitted network, that the sign logits ignore the extra heads.

    For U-SG, the check also confirms that the extra losses do not reach the backbone.
    """
    if net.extra is None:
        return
    net.eval()
    with torch.no_grad():
        before = net(x)[:, 0].detach().cpu().numpy()
        saved = {k: v.clone() for k, v in net.extra.state_dict().items()}
        for p in net.extra.parameters():
            p.add_(1.0)
        after = net(x)[:, 0].detach().cpu().numpy()
        net.extra.load_state_dict(saved)
    leak.assert_sign_ignores_the_extra_heads(sign_before=before, sign_after=after)
    if spec.stop_gradient:
        net.zero_grad()
        net(x)[:, 1:].sum().backward()
        grads = [
            float(p.grad.abs().sum()) if p.grad is not None else 0.0
            for p in net.backbone.parameters()
        ]
        net.zero_grad()
        leak.assert_stop_gradient_blocks_backbone(backbone_grad_sums=grads)


def store_set(
    *, folder: Path, name: str, pred: train.Predictions, tr: truth.Truth, member: str = "plain"
) -> None:
    """Store one member of a prediction set, flat on the valid half-hours."""
    arrays = getattr(pred, member)
    flat = {k: tr.flat(values=v) for k, v in arrays.items()}
    if tr.keys is not None:
        flat |= dict(zip(truth.ROW_KEY_NAMES, tr.flat_keys(), strict=True))
    train729.save_predictions(arrays=flat, path=folder / f"{name}.npz")


def check_stored_rows(*, root: Path, root730: Path, name: str, env: stages.Env) -> None:
    """Check that every test set a fit stored holds the row keys of the test bank's rows.

    The check runs on `test`, `test_transfer`, `test_nobsp`, every injection set, and `test_raised`.

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        name: The fit's name (its folder under `predictions`).
        env: The stages' environment (the holidays the rows are built with).

    Raises:
        CorrectnessFailureError: When a stored file lacks the keys or they differ (verdict row 1).
    """
    import e00729_score as score729
    import e00732_report as report

    te = banks729.load_bank(path=banks730.bank_path(root=root730, kind="test"))
    rows = score729.rows_from_segments(segments=te.segments, holidays=env.holidays)
    keys = report.row_keys_of(rows=rows)
    folder = root / "predictions" / name
    files = sorted(folder.glob("test*.npz"))
    if not files:
        msg = f"{name}: no test set was stored"
        raise leak.CorrectnessFailureError(msg)
    for path in files:
        report.check_stored_keys(
            arrays=[report.read_set(folder=folder, name=path.stem)],
            keys=keys,
            what=f"{name} {path.stem}",
        )


def store_predictions(
    *,
    root: Path,
    root730: Path,
    name: str,
    net: nets.Net,
    spec: core.ArmSpec732,
    device: str,
    env: stages.Env,
    sets: Sets,
) -> None:
    """Store a fit's outputs on the validation, test, transfer, injection, raised-sky, shifted, and real sets."""
    folder = root / "predictions" / name
    te = banks729.load_bank(path=banks730.bank_path(root=root730, kind="test"))
    sh = banks729.load_bank(path=banks730.bank_path(root=root730, kind="shifted"))
    common = {"holidays": env.holidays, "stats": env.stats, "medians": env.medians}
    test_plain = arms.prepare_plain(segments=te.segments, neighbours=env.neighbours, **common)
    test_transfer = arms.prepare(
        overlays=ov.overlay_segments(segments=te.segments, kind="test", neighbours=env.neighbours),
        **common,
    )
    test_index = ov.build_pair_index(segments=te.segments, sources=core.TEST_SIBLING_SOURCES)
    shifted = arms.prepare_plain(segments=sh.segments, neighbours=env.neighbours, **common)
    for name_, prep in (("validation", sets.val_plain), ("test", test_plain), ("shifted", shifted)):
        pred = train.predict(net=net, prepared=prep, device=device, spec=spec)
        store_set(folder=folder, name=name_, pred=pred, tr=truth.truth(prepared=prep))
    pred = train.predict(net=net, prepared=test_transfer, device=device, spec=spec)
    store_set(
        folder=folder, name="test_transfer", pred=pred, tr=truth.truth(prepared=test_transfer)
    )
    if not spec.bsp_zeroed:
        pred = train.predict(net=net, prepared=test_plain, device=device, hide_bsp=True, spec=spec)
        store_set(folder=folder, name="test_nobsp", pred=pred, tr=truth.truth(prepared=test_plain))
    if spec.heads:
        for hide in (False, True):
            for a in core.TEST_INJECTION_POINTS:
                pred = train.predict(
                    net=net, prepared=test_plain, device=device, index=test_index, amount=a,
                    hide_bsp=hide, spec=spec,
                )  # fmt: skip
                store_set(
                    folder=folder,
                    name=f"test_inj_{a:g}{'_nobsp' if hide else ''}",
                    pred=pred,
                    tr=truth.truth(prepared=test_plain, index=test_index, amount=a),
                    member="inj",
                )
        pred = train.predict(
            net=net, prepared=test_plain, device=device, eps=core.TEST_EPS, spec=spec
        )
        store_set(
            folder=folder,
            name="test_raised",
            pred=pred,
            tr=truth.truth(prepared=test_plain, eps=core.TEST_EPS),
            member="raised",
        )
    if env.real_sets is not None:
        for set_name, segments in env.real_sets().items():
            prep = arms.prepare_plain(segments=segments, neighbours=env.real_neighbours, **common)
            pred = train.predict(net=net, prepared=prep, device=device, spec=spec)
            store_set(
                folder=folder, name=f"real_{set_name}", pred=pred, tr=truth.truth(prepared=prep)
            )


def fit_one(
    *,
    root: Path,
    root730: Path,
    env: stages.Env,
    arm: str,
    seed: int,
    device: str,
    log: Log,
    max_epochs: int = core.MAX_EPOCHS,
) -> dict:
    """Fit one network, save it, store its predictions, and write its record (a fit whose record exists is skipped)."""
    stages.guard_run(root=root, root730=root730, stage="fit")
    spec = core.ARMS[arm]
    name = core.job_name(arm=arm, arch=spec.arch, seed=seed)
    record_path = stages.job_path(root=root, arm=arm, arch=spec.arch, seed=seed)
    if record_path.exists():
        return stages.read_json(path=record_path)
    if device == "cuda":
        train729.limit_gpu_memory()
    sets = load_sets(root730=root730, env=env, seed=seed)
    assert_fit_inputs(spec=spec, sets=sets, seed=seed)
    net, record = train.fit(
        spec=spec, seed=seed, train=sets.train, index=sets.index, val_plain=sets.val_plain, val_transfer=sets.val_transfer,
        device=device, val_index=sets.val_index, max_epochs=max_epochs, out_path=root / "networks" / f"{name}.pt", log=log,
    )  # fmt: skip
    probe = torch.from_numpy(
        np.random.default_rng(seed)
        .normal(size=(2, arms.n_channels(spec=spec), arms.STEPS))
        .astype(np.float32)
    ).to(device)
    assert_net_checks(net=net.to(device), spec=spec, x=probe)
    store_predictions(
        root=root, root730=root730, name=name, net=net, spec=spec, device=device, env=env, sets=sets
    )
    check_stored_rows(root=root, root730=root730, name=name, env=env)
    payload = {**record.as_dict(), "peak_rss_gb": train.peak_rss_gb()}
    stages.write_json(path=record_path, payload=payload)
    return payload


def run_jobs(
    *,
    root: Path,
    jobs: Sequence[tuple[str, str, int]],
    run_one: Callable[[str, str, int], dict],
    log: Log,
    rerun: bool = False,
) -> dict[str, int]:
    """Run fits one at a time; a fit that raises is recorded as failed and the run goes on.

    A failed fit leaves `<name>.failed.json` and is not run again unless `rerun` is true. A
    `CorrectnessFailureError` (verdict row 1), a `StoppedError` (verdict row 0), and a `GuardError`
    (a gate that is open) are never hidden as a failed fit: each propagates.

    Args:
        root: The data folder.
        jobs: The (arm, architecture, seed) of each fit.
        run_one: Runs one fit.
        log: Receives progress lines.
        rerun: Whether to run a fit again that has a `.failed.json`.

    Returns:
        The counts of fits run, skipped, skipped as already failed, and failed.
    """
    counts = {"run": 0, "skipped": 0, "skipped_failed": 0, "failed": 0}
    for arm, arch, seed in jobs:
        record = stages.job_path(root=root, arm=arm, arch=arch, seed=seed)
        failed_path = record.with_suffix(".failed.json")
        if record.exists():
            counts["skipped"] += 1
            continue
        if failed_path.exists() and not rerun:
            counts["skipped_failed"] += 1
            continue
        try:
            run_one(arm, arch, seed)
            failed_path.unlink(missing_ok=True)
            counts["run"] += 1
        except leak.CorrectnessFailureError, stages.StoppedError, stages.GuardError:
            raise
        except Exception as exc:  # noqa: BLE001
            counts["failed"] += 1
            stages.write_json(
                path=failed_path,
                payload={
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            log(f"fit {record.stem} failed: {type(exc).__name__}: {exc}")
    return counts


def stage_timing(
    *,
    root: Path,
    root730: Path,
    env: stages.Env,
    device: str,
    manager_approved: bool,
    log: Log,
    epochs: int = core.TIMING_EPOCHS,
    cores: str = "?",
) -> dict:
    """Run the timing gate: a few epochs of each timed arm, then project every fit, and write `scores/timing.json`.

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        env: The stages' environment.
        device: "cuda" or "cpu".
        manager_approved: Whether the manager approved a projection over the stop line.
        log: Receives progress lines.
        epochs: The epochs timed for each arm.
        cores: The slot's cores, for the printed lines.

    Returns:
        The timing report with its pass flag.

    Raises:
        StoppedError: When the projection is over the stop line and the manager has not approved, or
            the memory is over its cap.
    """
    stages.guard_run(root=root, root730=root730, stage="timing")
    log(fits_starting_line(family="timing", hours=core.TIMING_GATE_HOURS, cores=cores))
    if device == "cuda":
        train729.limit_gpu_memory()
    sets = load_sets(root730=root730, env=env, seed=0)
    secs, building, peak_gpu = {}, {}, 0.0
    for arm in core.TIMING_ARMS:
        spec = core.ARMS[arm]
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        secs[arm] = train.measure_seconds_per_epoch(
            spec=spec,
            train=sets.train,
            index=sets.index,
            val_plain=sets.val_plain,
            val_transfer=sets.val_transfer,
            device=device,
            epochs=epochs,
        )
        if device == "cuda":
            peak_gpu = max(peak_gpu, torch.cuda.max_memory_allocated() / 1e9)
        building[arm] = train.time_pair_building(spec=spec, train=sets.train, index=sets.index)
    report = core.timing_gate_report(
        seconds_per_epoch=secs,
        peak_gpu_gb=peak_gpu,
        peak_rss_gb=train.peak_rss_gb(),
        pair_seconds_per_epoch=building["U*"],
    )
    report["window_building_seconds_per_epoch"] = building
    passes = core.timing_gate_passes(report=report, manager_approved=manager_approved)
    stages.write_json(path=root / "scores" / "timing.json", payload={**report, "pass": passes})
    log(
        f"timing gate: projection {report['hours_at_cap']:.2f} h at the cap, dropped {report['dropped_arms']}, pass {passes}"
    )
    log(last_fit_ended_line(family="timing", counts={"timed": len(secs)}))
    if not passes:
        msg = "the timing gate failed: the projection or a memory peak is over its line (verdict row 0)"
        raise stages.StoppedError(msg)
    return {**report, "pass": passes}


def fits_starting_line(*, family: str, hours: float, cores: str) -> str:
    """Return the line sent to the manager before each family of fits (the timing gate, S+ and U*.

    U-N, U-NC, U-noBSP, U-NCw, U-SG, U*-L).
    """
    return f"fits starting: experiment 732, family {family}, projection {hours:.2f} GPU-process-hours at the cap, core {cores}, {datetime.now(UTC):%FT%TZ}"


def last_fit_ended_line(*, family: str, counts: dict[str, int]) -> str:
    """Return the line printed after the last fit of a family."""
    return f"last fit ended: experiment 732, family {family}, {counts}, {datetime.now(UTC):%FT%TZ}"


def write_status(*, root: Path, status: dict[str, Any]) -> None:
    """Write `chain_status.json` (atomic) with the stage reached and its outcome.

    For a stop, the file also holds the reason (names and counts only).
    """
    stages.write_json(
        path=root / "chain_status.json",
        payload={**status, "time": datetime.now(UTC).isoformat(timespec="seconds")},
    )


CHAIN_FAILED_EXIT: Final = 5
"""The exit status of a chain with a failed fit (3 is a stop and 4 a correctness failure; `run.sh` uses 1 for a refused slot and 64 for bad usage, and a launcher retries on 1)."""


def arm_failed(*, root: Path, arm: str) -> bool:
    """Return whether an arm has a seed whose last attempt failed: a `.failed.json` and no fit record.

    A `.failed.json` left from an earlier attempt that a later run completed is stale and is ignored.
    """
    for seed in core.SEEDS:
        record = stages.job_path(root=root, arm=arm, arch=core.ARMS[arm].arch, seed=seed)
        if record.with_suffix(".failed.json").exists() and not record.exists():
            return True
    return False


def chain_exit_code(*, status: Mapping[str, Any]) -> int:
    """Return the process exit status of a chain run: 0 only when every fit succeeded."""
    return 0 if status.get("state") == "complete" else CHAIN_FAILED_EXIT


def stage_chain(
    *,
    root: Path,
    root730: Path,
    run_one: Callable[[str, str, int], dict],
    log: Log,
    cores: str = "?",
    rerun: bool = False,
) -> dict:
    """Run the fits family by family, stopping at the first stop or failure.

    The function reports each stop and writes a status file.

    The arms that the timing gate dropped (`dropped_arms` of `scores/timing.json`) are not fitted.

    Every family prints a "fits starting" line and a "last fit ended" line. A stop (row 0), a correctness
    failure (row 1), or a fit that raised and a counted reading needs (row 2) ends the chain with
    `chain_status.json` holding `state` "stopped", "invalid", or "failed". A report-only family that
    failed (U-NCw, U-SG, U*-L) does not end the chain, but the final state is then "failed" (with
    `needed` false), never "complete", so that a launcher cannot read the status as a success.

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        run_one: Runs one fit given its arm, architecture, and seed.
        log: Receives progress lines.
        cores: The slot's cores, for the printed lines.
        rerun: Whether to run a fit again that has a `.failed.json`.

    Returns:
        The status record.
    """
    stages.guard_run(root=root, root730=root730, stage="chain")
    timing = stages.read_json(path=root / "scores" / "timing.json")
    dropped = set(timing.get("dropped_arms", []))
    done: list[str] = []
    for family, all_arms in core.FIT_FAMILIES:
        arm_names = tuple(a for a in all_arms if a not in dropped)
        if not arm_names:
            continue
        jobs = [(a, core.ARMS[a].arch, s) for a in arm_names for s in core.SEEDS]
        write_status(root=root, status={"state": "running", "family": family, "done": done})
        log(fits_starting_line(family=family, hours=float(timing["hours_at_cap"]), cores=cores))
        try:
            counts = run_jobs(root=root, jobs=jobs, run_one=run_one, log=log, rerun=rerun)
        except leak.CorrectnessFailureError as exc:
            stages.record_invalid(root=root, rule=str(exc))
            status = {"state": "invalid", "family": family, "reason": str(exc), "done": done}
            write_status(root=root, status=status)
            raise
        except (stages.StoppedError, stages.GuardError) as exc:
            write_status(
                root=root,
                status={"state": "stopped", "family": family, "reason": str(exc), "done": done},
            )
            raise
        log(last_fit_ended_line(family=family, counts=counts))
        needed_failed = [
            a for a in arm_names if a in core.NEEDED_ARMS and arm_failed(root=root, arm=a)
        ]
        done.append(family)
        if needed_failed:
            status = {"state": "failed", "family": family, "arms": needed_failed, "done": done}
            write_status(root=root, status=status)
            return status
    failed_arms = [
        a
        for _family, all_arms in core.FIT_FAMILIES
        for a in all_arms
        if a not in dropped and arm_failed(root=root, arm=a)
    ]
    if failed_arms:
        status = {"state": "failed", "arms": failed_arms, "needed": False, "done": done}
    else:
        status = {"state": "complete", "done": done}
    write_status(root=root, status=status)
    return status

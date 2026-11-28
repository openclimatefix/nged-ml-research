"""Stages of experiment 737, one command per stage.

    e00737_run.py gates                      the known-answer tests, the attribute check, G2, G3,
                                             G4, and the scored-quarter table (no fit)
    e00737_run.py review-mark                write the code hash into review_stage1.ok (after the
                                             review of that code; a person or the manager runs it)
    e00737_run.py freeze-code                write code.sha256 (needs a review marker of this code)
    e00737_run.py tune-ridge --worker i --of n  30 synthetic draws at each ridge scale, sharded
    e00737_run.py tune-ridge-collect          the ridge scale into ridge.json
    e00737_run.py tune-growth --worker i --of n the 120 tuning worlds at each growth standard
                                             deviation, at the chosen ridge scale, sharded by
                                             Primary
    e00737_run.py tune-collect                the growth standard deviation and the ridge scale
                                             into tuned.json, and its SHA-256 into tuned.sha256
    e00737_run.py g1-reference                one fit of the real record that G1 and G1b read
    e00737_run.py g1 --worker i --of n        the 100 + 100 synthetic fits of G1 and G1b, sharded
    e00737_run.py g1-collect                  G1 and G1b into gates.json
    e00737_run.py unit                        the timed first unit; prints the CPU-hour projection
    e00737_run.py fit --worker i --of n       the bulk: every world, sharded by Primary
    e00737_run.py smoke                       G2 on every world file, and every world is present
    e00737_run.py score
    e00737_run.py report                      prints and stores report_output.txt
    e00737_run.py figures --part forward|fleets|posterior|real   private figures

`run.sh` sets the environment and the memory-capped scope. Every output goes under `OUT`. Every
stage prints `OUT` first. The stages after `gates` refuse to start without `review_stage1.ok`
holding the current code hash and `code.sha256` holding the same hash. The tuning stages also need
`gates.json` to record a pass of G2, G3, and G4. The stages after the tuning (G1 and G1b, the first
unit, and the bulk) need `tuned.json`, and the stages from the first unit on also need G1 and G1b.
A stage that fails writes `OUT/failed/stage_<stage>_worker_<i>.json` and exits with code 3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from e00737_common import (
    EM_MAX_ITER,
    EXIT_FAILED,
    EXPECTED_INJECTED_WORLDS,
    EXPECTED_TUNING_WORLDS,
    EXPECTED_UNINJECTED_FITS,
    G1_DRAWS,
    GROWTH_SD_GRID,
    MAIN_SEEDS,
    OUT,
    PAST_ONLY_CUTS,
    PRIMARIES,
    RIDGE_DRAWS,
    RIDGE_SCALE_GRID,
    RIDGE_SEED,
    RIDGE_STEP_GROWTH_SD,
    Tuned,
    atomic_json,
    atomic_npz,
    sha256_of,
    time_log,
)
from e00737_fit import (
    PrimaryModel,
    build_model,
    fit_all_arms,
    fit_arm,
    fits_to_arrays,
    random_walk_prior,
)
from e00737_forward import FleetParams
from e00737_inputs import PrimaryInputs, build_fit_data, load_primary, placebo_sky, unit_term
from e00737_phases import (
    CODE_SHA,
    GATES_BEFORE_TUNING,
    MARKER_REVIEW,
    STOP_CPU_HOURS,
    TUNED_SHA_NAME,
    assert_blas_pinned,
    code_hash,
    read_record,
    require_budget,
    require_code_frozen,
    require_finished,
    require_gates,
    require_review,
    require_ridge_scale,
    require_tuned,
    require_unit,
    run_shard,
    shard,
    write_stage_failure,
)
from e00737_worlds import (
    World,
    WorldSpec,
    build_world,
    check_injection,
    draw_fleet,
    tuning_specs,
    world_arrays,
    world_specs,
)

STAGES: tuple[str, ...] = (
    "gates",
    "review-mark",
    "freeze-code",
    "tune-ridge",
    "tune-ridge-collect",
    "tune-growth",
    "tune-collect",
    "g1-reference",
    "g1",
    "g1-collect",
    "unit",
    "fit",
    "smoke",
    "score",
    "report",
    "figures",
)
UNIT_PRIMARY: str = PRIMARIES[0]
UNIT_WORLD: str = "main_2.5_ramp_737101"
PLAN_ESTIMATE_CPU_HOURS: float = 17.5
"""The plan's estimate, printed beside the projection."""
TUNING_WORLDS_PER_PRIMARY: int = 20


@lru_cache(maxsize=1)
def get_model(*, site: str) -> PrimaryModel:
    """Build (and keep) one Primary's model inputs.

    Args:
        site: The Primary's label.

    Returns:
        The model inputs.
    """
    return build_model(inputs=load_primary(site=site))


def oracle_term(*, model: PrimaryModel, fleet: FleetParams | None) -> np.ndarray | None:
    """Return arm O's unit solar term at the fit rows, or None to reuse arm A.

    Args:
        model: The Primary's model inputs.
        fleet: The injected fleet (None for a world with none, or a matched fleet).

    Returns:
        The term, or None.
    """
    from e00737_forward import PRIOR_FLEET

    if fleet is None or fleet == PRIOR_FLEET:
        return None
    return unit_term(inputs=model.inputs, fleet=fleet)[model.data.rows]


def real_record_extras(*, inputs: PrimaryInputs, model: PrimaryModel) -> dict[str, np.ndarray]:
    """Return the Primary-level arrays that the scoring reads from the control file.

    Args:
        inputs: The Primary's inputs.
        model: The Primary's model inputs.

    Returns:
        The scored quarters, usable shares, and the counts of dropped half-hours.
    """
    return {
        "scored": inputs.scored,
        "usable_share": inputs.usable_share,
        "dropped": np.array(
            [int((~np.isfinite(inputs.power)).sum()), int((~inputs.reliable).sum())]
        ),
        "n_rows": np.array([len(model.data.rows)]),
    }


def task_control(*, site: str, store: Path, growth_sd: float, ridge_scale: float) -> None:
    """Fit every arm to the real record of one Primary.

    Args:
        site: The Primary's label.
        store: The folder that holds the `fits` and `worlds` folders.
        growth_sd: The tuned growth standard deviation.
        ridge_scale: The tuned multiplier of the baseline's ridge precision.
    """
    start = time.time()
    model = get_model(site=site)
    fits = fit_all_arms(
        model=model,
        power=model.inputs.power,
        growth_sd=growth_sd,
        ridge_scale=ridge_scale,
        oracle_term=None,
    )
    atomic_npz(
        arrays={
            **fits_to_arrays(fits=fits),
            **real_record_extras(inputs=model.inputs, model=model),
        },
        path=store / "fits" / site / "control.npz",
    )
    time_log(stage="fit", task=f"control_{site}", start=start, out=store)


def task_ocontrol(
    *, site: str, seed: int, store: Path, growth_sd: float, ridge_scale: float
) -> None:
    """Fit arm A with a seed's fleet term to the real record, the control of arm O.

    Args:
        site: The Primary's label.
        seed: The seed whose mismatched fleet gives the term.
        store: The output folder.
        growth_sd: The tuned growth standard deviation.
        ridge_scale: The tuned multiplier of the baseline's ridge precision.
    """
    start = time.time()
    model = get_model(site=site)
    fleet = draw_fleet(seed=seed, primary_index=PRIMARIES.index(site), matched=False)
    term = unit_term(inputs=model.inputs, fleet=fleet)[model.data.rows]
    fit = fit_arm(
        model=model,
        power=model.inputs.power,
        prior=random_walk_prior(growth_sd=growth_sd),
        ridge_scale=ridge_scale,
        solar=term,
    )
    atomic_npz(
        arrays={
            "A_cap": fit.capacity,
            "A_sd": fit.sd,
            "A_info": np.array([fit.n_iter, fit.converged]),
        },
        path=store / "fits" / site / f"ocontrol_{seed}.npz",
    )
    time_log(stage="fit", task=f"ocontrol_{site}_{seed}", start=start, out=store)


def task_placebo(*, site: str, store: Path, growth_sd: float, ridge_scale: float) -> None:
    """Fit arm A to the real record with the placebo sky.

    Args:
        site: The Primary's label.
        store: The output folder.
        growth_sd: The tuned growth standard deviation.
        ridge_scale: The tuned multiplier of the baseline's ridge precision.
    """
    import dataclasses

    from e00737_forward import PRIOR_FLEET

    start = time.time()
    real = load_primary(site=site)
    sky, reliable = placebo_sky(inputs=real)
    twin = dataclasses.replace(real, sky=sky, reliable=reliable)
    data = build_fit_data(inputs=twin)
    model = PrimaryModel(
        inputs=twin,
        data=data,
        terms={PRIOR_FLEET.dc_ac_ratio: unit_term(inputs=twin, fleet=PRIOR_FLEET)[data.rows]},
        q_basis=np.empty((0, 4)),
    )
    fit = fit_arm(
        model=model,
        power=real.power,
        prior=random_walk_prior(growth_sd=growth_sd),
        ridge_scale=ridge_scale,
    )
    atomic_npz(
        arrays={
            "A_cap": fit.capacity,
            "A_sd": fit.sd,
            "A_info": np.array([fit.n_iter, fit.converged]),
        },
        path=store / "fits" / site / "placebo.npz",
    )
    time_log(stage="fit", task=f"placebo_{site}", start=start, out=store)


def task_world(
    *, site: str, world_name: str, store: Path, growth_sd: float, ridge_scale: float
) -> None:
    """Build one injected world, check G2 on it, write it, and fit every arm.

    Args:
        site: The Primary's label.
        world_name: The world's name.
        store: The output folder.
        growth_sd: The tuned growth standard deviation.
        ridge_scale: The tuned multiplier of the baseline's ridge precision.

    Raises:
        RuntimeError: When gate G2 fails for the built world.
    """
    start = time.time()
    model = get_model(site=site)
    spec = next(s for s in world_specs(primary=site) if s.name == world_name)
    world = build_world(inputs=model.inputs, spec=spec)
    check = check_injection(inputs=model.inputs, world=world)
    if not check["passed"]:
        msg = f"G2 failed for world {site}/{world_name}: {check}"
        raise RuntimeError(msg)
    atomic_npz(arrays=world_arrays(world=world), path=store / "worlds" / site / f"{world_name}.npz")
    fits = fit_all_arms(
        model=model,
        power=world.power,
        growth_sd=growth_sd,
        ridge_scale=ridge_scale,
        oracle_term=oracle_term(model=model, fleet=world.fleet),
    )
    atomic_npz(arrays=fits_to_arrays(fits=fits), path=store / "fits" / site / f"{world_name}.npz")
    time_log(stage="fit", task=f"world_{site}_{world_name}", start=start, out=store)


def task_past(
    *, site: str, quarter: int, store: Path, growth_sd: float, ridge_scale: float
) -> None:
    """Fit arm A to the real record cut at the end of a quarter (the past-only sample).

    Args:
        site: The Primary's label.
        quarter: The zero-based quarter whose end is the cut.
        store: The output folder.
        growth_sd: The tuned growth standard deviation.
        ridge_scale: The tuned multiplier of the baseline's ridge precision.
    """
    start = time.time()
    model = get_model(site=site)
    power = np.where(model.inputs.quarter <= quarter, model.inputs.power, np.nan)
    fit = fit_arm(
        model=model,
        power=power,
        prior=random_walk_prior(growth_sd=growth_sd),
        ridge_scale=ridge_scale,
    )
    atomic_npz(
        arrays={
            "A_cap": fit.capacity,
            "A_sd": fit.sd,
            "A_info": np.array([fit.n_iter, fit.converged]),
        },
        path=store / "past" / site / f"cut{quarter}.npz",
    )
    time_log(stage="fit", task=f"past_{site}_{quarter}", start=start, out=store)


@lru_cache(maxsize=1)
def ridge_reference() -> dict[str, np.ndarray]:
    """Fit arm A to the first Primary's real record at the unscaled ridge, once for each process.

    The synthetic series of the ridge draws take their baseline and noise scales from this fit.
    Each worker process pays one cold fit for it (about 24 seconds), which the first-unit
    projection does not count: at most 6 workers add about 0.04 CPU-hours.

    Returns:
        The baseline and scales, as `g1_reference` returns them.
    """
    from e00737_gates import g1_model, g1_reference

    return g1_reference(
        model=g1_model(), tuned=Tuned(growth_sd=RIDGE_STEP_GROWTH_SD, ridge_scale=1.0)
    )


def task_tune_ridge(*, draw: int, store: Path) -> None:
    """Fit one synthetic series at each ridge scale and store the capacity errors.

    Args:
        draw: The draw's number.
        store: The output folder.
    """
    from e00737_gates import g1_model, ridge_draw

    start = time.time()
    errors = ridge_draw(model=g1_model(), reference=ridge_reference(), draw=draw)
    atomic_npz(arrays={"error": errors}, path=store / "tuning" / "ridge" / f"draw_{draw:03d}.npz")
    time_log(stage="tune", task=f"tune_ridge_{draw:03d}", start=start, out=store)


def task_tune_control(*, site: str, store: Path, ridge_scale: float) -> None:
    """Fit arm A to the real record at each growth standard deviation of the grid.

    Args:
        site: The Primary's label.
        store: The output folder.
        ridge_scale: The chosen ridge scale.
    """
    start = time.time()
    model = get_model(site=site)
    caps = growth_fits(model=model, power=model.inputs.power, ridge_scale=ridge_scale)
    atomic_npz(
        arrays={"cap": np.array(caps), "scored": model.inputs.scored},
        path=store / "tuning" / site / "growth_control.npz",
    )
    time_log(stage="tune", task=f"tune_growth_control_{site}", start=start, out=store)


def task_tune_world(*, site: str, world_name: str, store: Path, ridge_scale: float) -> None:
    """Fit arm A to one tuning world at each growth standard deviation of the grid.

    Args:
        site: The Primary's label.
        world_name: The tuning world's name.
        store: The output folder.
        ridge_scale: The chosen ridge scale.
    """
    start = time.time()
    model = get_model(site=site)
    spec = next(s for s in tuning_specs(primary=site) if s.name == world_name)
    world = build_world(inputs=model.inputs, spec=spec)
    caps = growth_fits(model=model, power=world.power, ridge_scale=ridge_scale)
    atomic_npz(
        arrays={"cap": np.array(caps), "truth": world.truth},
        path=store / "tuning" / site / f"growth_{world_name}.npz",
    )
    time_log(stage="tune", task=f"tune_growth_{site}_{world_name}", start=start, out=store)


def growth_fits(*, model: PrimaryModel, power: np.ndarray, ridge_scale: float) -> list[np.ndarray]:
    """Fit arm A at each growth standard deviation of the grid and return the capacity paths.

    Args:
        model: The Primary's model inputs.
        power: The power to fit (a tuning world's, or the real record's).
        ridge_scale: The chosen ridge scale.

    Returns:
        One capacity path for each growth standard deviation, in the grid's order.
    """
    return [
        fit_arm(
            model=model,
            power=power,
            prior=random_walk_prior(growth_sd=sd),
            ridge_scale=ridge_scale,
        ).capacity
        for sd in GROWTH_SD_GRID
    ]


def primaries_of(*, worker: int, of: int) -> list[str]:
    """Return the Primaries one worker handles: every `of`-th Primary from `worker`.

    Args:
        worker: This worker's index.
        of: The number of workers.

    Returns:
        The Primaries' labels.
    """
    return [p for k, p in enumerate(PRIMARIES) if k % of == worker]


def fit_tasks(*, primaries: list[str], store: Path, tuned: Tuned) -> list[tuple[str, dict]]:
    """List the bulk-fit tasks of some Primaries, in the order that satisfies their dependencies.

    Args:
        primaries: The Primaries' labels.
        store: The output folder.
        tuned: The tuned growth standard deviation and ridge scale.

    Returns:
        Pairs of a task name and its keyword arguments (the function is chosen by the prefix).
    """
    common = {"store": store, "growth_sd": tuned.growth_sd, "ridge_scale": tuned.ridge_scale}
    tasks: list[tuple[str, dict]] = []
    for site in primaries:
        tasks.append((f"control_{site}", {"kind": "control", "site": site, **common}))
        tasks += [
            (f"ocontrol_{site}_{seed}", {"kind": "ocontrol", "site": site, "seed": seed, **common})
            for seed in MAIN_SEEDS
        ]
        tasks.append((f"placebo_{site}", {"kind": "placebo", "site": site, **common}))
        tasks += [
            (
                f"world_{site}_{s.name}",
                {"kind": "world", "site": site, "world_name": s.name, **common},
            )
            for s in world_specs(primary=site)
            if s.kind not in ("placebo", "control")
        ]
        tasks += [
            (f"past_{site}_{q}", {"kind": "past", "site": site, "quarter": q, **common})
            for q in PAST_ONLY_CUTS
        ]
    return tasks


def run_fit_task(*, kind: str, **kwargs: Any) -> None:
    """Run one bulk-fit task by its kind.

    Args:
        kind: `control`, `ocontrol`, `placebo`, `world`, or `past`.
        **kwargs: The task's keyword arguments.
    """
    {
        "control": task_control,
        "ocontrol": task_ocontrol,
        "placebo": task_placebo,
        "world": task_world,
        "past": task_past,
    }[kind](**kwargs)


def tune_tasks(
    *, primaries: list[str], store: Path, part: str, ridge_scale: float | None, worker: int, of: int
) -> list[tuple[str, dict]]:
    """List one worker's tuning tasks for one part of the tuning.

    Args:
        primaries: The Primaries' labels (the growth part's tasks are sharded by Primary).
        store: The output folder.
        part: `ridge` (the synthetic draws) or `growth` (the tuning worlds).
        ridge_scale: The chosen ridge scale (None for part `ridge`).
        worker: This worker's index (the ridge part's draws are sharded by draw).
        of: The number of workers.

    Returns:
        Pairs of a task name and its keyword arguments.
    """
    if part == "ridge":
        draws = shard(items=list(range(RIDGE_DRAWS)), worker=worker, of=of)
        return [
            (f"tune_ridge_{d:03d}", {"kind": "ridge", "draw": d, "store": store}) for d in draws
        ]
    common = {"store": store, "ridge_scale": ridge_scale}
    tasks: list[tuple[str, dict]] = []
    for site in primaries:
        tasks.append((f"tune_growth_control_{site}", {"kind": "control", "site": site, **common}))
        tasks += [
            (
                f"tune_growth_{site}_{s.name}",
                {"kind": "world", "site": site, "world_name": s.name, **common},
            )
            for s in tuning_specs(primary=site)
        ]
    return tasks


def run_tune_task(*, kind: str, **kwargs: Any) -> None:
    """Run one tuning task by its kind.

    Args:
        kind: `ridge`, `control`, or `world`.
        **kwargs: The task's keyword arguments.
    """
    {"ridge": task_tune_ridge, "control": task_tune_control, "world": task_tune_world}[kind](
        **kwargs
    )


def tune_ridge_collect() -> dict[str, Any]:
    """Choose the ridge scale with the lowest median largest capacity error on synthetic draws.

    The error of a draw is the largest absolute gap, over the 18 quarters, between the fitted and
    the injected capacity. A tie goes to the larger ridge scale.

    Returns:
        The record written to `ridge.json`.

    Raises:
        RuntimeError: When a draw task has no done marker of the current code, a failure record
            exists, or a draw file is missing.
    """
    from e00737_score import load_npz

    require_finished(names=[f"tune_ridge_{d:03d}" for d in range(RIDGE_DRAWS)])
    folder = OUT / "tuning" / "ridge"
    files = [folder / f"draw_{d:03d}.npz" for d in range(RIDGE_DRAWS)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        msg = f"{len(missing)} ridge draw files are missing, such as {missing[0]}"
        raise RuntimeError(msg)
    errors = np.array([load_npz(path=f)["error"] for f in files])
    medians = np.median(errors, axis=0)
    best = int(np.argmin(medians))  # the first minimum, and the grid runs from the larger scale
    record = {
        "ridge_scale": RIDGE_SCALE_GRID[best],
        "grid": list(RIDGE_SCALE_GRID),
        "growth_sd_held_at": RIDGE_STEP_GROWTH_SD,
        "median_max_capacity_error_mw": [float(m) for m in medians],
        "n_draws": len(files),
        "seeds": [RIDGE_SEED, RIDGE_SEED + RIDGE_DRAWS - 1],
        "code_hash": code_hash(),
    }
    atomic_json(obj=record, path=OUT / "ridge.json")
    return record


def tune_collect() -> dict[str, Any]:
    """Choose the growth standard deviation with the lowest median path error, at the ridge scale.

    The stage also writes `tuned.sha256`, which holds the SHA-256 of `tuned.json`, and every stage
    that reads the chosen values checks it.

    Returns:
        The record written to `tuned.json`, which holds the chosen ridge scale too.

    Raises:
        RuntimeError: When a tuning task has no done marker of the current code, a failure record
            exists, or the number of tuning worlds is not `EXPECTED_TUNING_WORLDS`.
    """
    from e00737_score import load_npz, path_error

    ridge = read_record(name="ridge.json")
    require_finished(
        names=[
            name
            for name, _ in tune_tasks(
                primaries=list(PRIMARIES),
                store=OUT,
                part="growth",
                ridge_scale=ridge["ridge_scale"],
                worker=0,
                of=1,
            )
        ]
    )
    errors: list[list[float]] = [[] for _ in GROWTH_SD_GRID]
    for site in PRIMARIES:
        folder = OUT / "tuning" / site
        control = load_npz(path=folder / "growth_control.npz")
        scored = control["scored"].astype(bool)
        for spec in tuning_specs(primary=site):
            arrays = load_npz(path=folder / f"growth_{spec.name}.npz")
            for k in range(len(GROWTH_SD_GRID)):
                increment = arrays["cap"][k] - control["cap"][k]
                errors[k].append(
                    path_error(truth=arrays["truth"], estimate=increment, scored=scored)
                )
    n_worlds = len(errors[0])
    if n_worlds != EXPECTED_TUNING_WORLDS:
        msg = f"{n_worlds} tuning worlds were found, expected {EXPECTED_TUNING_WORLDS}"
        raise RuntimeError(msg)
    medians = [float(np.nanmedian(e)) for e in errors]
    record = {
        "growth_sd": GROWTH_SD_GRID[int(np.argmin(medians))],
        "ridge_scale": ridge["ridge_scale"],
        "grid": list(GROWTH_SD_GRID),
        "median_path_error": medians,
        "ridge_grid": ridge["grid"],
        "ridge_median_max_capacity_error_mw": ridge["median_max_capacity_error_mw"],
        "n_worlds": n_worlds,
        "code_hash": code_hash(),
    }
    atomic_json(obj=record, path=OUT / "tuned.json")
    (OUT / TUNED_SHA_NAME).write_text(
        f"{sha256_of(path=OUT / 'tuned.json')}  tuned.json code_hash={code_hash()}\n"
    )
    return record


def project_cpu_hours(*, timing: dict[str, float]) -> dict[str, float]:
    """Project the CPU-hours of the whole run from the first unit's measured times.

    Args:
        timing: Seconds of a Primary's inputs load (`load`), of the control task (`control`), of
            a world task (`world`), and of one fit of arm A (`fit_a`).

    Returns:
        The projected CPU-hours of each stage and the total (key `total`).
    """
    n_primaries = len(PRIMARIES)
    fit_a = timing["fit_a"]
    seconds = {
        "gates and figures": 0.5 * 3600.0,
        "g1 and g1b (200 fits plus the reference)": (2 * G1_DRAWS + 1) * fit_a,
        "tune (90 ridge fits, then 126 records at 3 growth sds)": (
            RIDGE_DRAWS * len(RIDGE_SCALE_GRID)
            + (EXPECTED_TUNING_WORLDS + n_primaries) * len(GROWTH_SD_GRID)
        )
        * fit_a,
        "fit: injected worlds": EXPECTED_INJECTED_WORLDS * timing["world"],
        "fit: controls": n_primaries * timing["control"],
        "fit: o-controls, placebos, past-only": (
            n_primaries * (len(MAIN_SEEDS) + 1 + len(PAST_ONLY_CUTS)) * fit_a
        ),
        "fit: loading inputs": 2 * n_primaries * timing["load"],
        "score, report, real-record figures": 0.2 * 3600.0,
    }
    hours = {k: v / 3600.0 for k, v in seconds.items()}
    hours["total"] = sum(hours.values())
    return hours


def unit_convergence(*, store: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Read the convergence flag and iteration count of every arm of the first unit's fits.

    Args:
        store: The unit's output folder.

    Returns:
        For the control and the world, each arm's `converged` and `n_iter`.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for label, name in (("control", "control"), ("world", UNIT_WORLD)):
        with np.load(store / "fits" / UNIT_PRIMARY / f"{name}.npz") as f:
            out[label] = {
                key[: -len("_info")]: {"converged": bool(f[key][1]), "n_iter": int(f[key][0])}
                for key in sorted(f.files)
                if key.endswith("_info")
            }
    return out


def worst_case_timing(*, timing: dict[str, float], iterations: int) -> dict[str, float]:
    """Scale the measured fit times to 30 iterations of every fit (the worst case).

    Args:
        timing: The measured timing of `project_cpu_hours`.
        iterations: The iterations that the control's arm A fit used.

    Returns:
        The timing with the fit times multiplied by `EM_MAX_ITER` over `iterations` (the load time
        is unchanged).
    """
    factor = max(1.0, EM_MAX_ITER / max(1, iterations))
    return {k: v if k == "load" else v * factor for k, v in timing.items()}


def stage_unit() -> None:
    """Run the first unit in a subfolder, time it, and print the projection for the manager.

    Raises:
        RuntimeError: When any fit of the first unit did not converge (the record is still written).
    """
    tuned = require_tuned()
    store = OUT / "unit"
    t0 = time.perf_counter()
    get_model(site=UNIT_PRIMARY)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    task_control(
        site=UNIT_PRIMARY,
        store=store,
        growth_sd=tuned.growth_sd,
        ridge_scale=tuned.ridge_scale,
    )
    control_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    task_world(
        site=UNIT_PRIMARY,
        world_name=UNIT_WORLD,
        store=store,
        growth_sd=tuned.growth_sd,
        ridge_scale=tuned.ridge_scale,
    )
    world_s = time.perf_counter() - t0
    with np.load(store / "fits" / UNIT_PRIMARY / "control.npz") as f:
        fit_a = float(f["A_info"][7])
        iterations = {arm: int(f[f"{arm}_info"][0]) for arm in ("A", "F", "Qt", "O", "R")}
    timing = {"load": load_s, "control": control_s, "world": world_s, "fit_a": fit_a}
    hours = project_cpu_hours(timing=timing)
    worst = project_cpu_hours(timing=worst_case_timing(timing=timing, iterations=iterations["A"]))
    convergence = unit_convergence(store=store)
    not_converged = [
        f"{label}/{arm}"
        for label, arms in convergence.items()
        for arm, info in arms.items()
        if not info["converged"]
    ]
    record = {
        "timing_seconds": timing,
        "iterations_of_the_control_fits": iterations,
        "convergence_of_the_first_fits": convergence,
        "not_converged": not_converged,
        "projected_cpu_hours": hours,
        "projected_total_cpu_hours": hours["total"],
        "worst_case_30_iterations_cpu_hours": worst,
        "stop_cpu_hours": STOP_CPU_HOURS,
        "code_hash": code_hash(),
    }
    atomic_json(obj=record, path=OUT / "unit_timing.json")
    print(f"first unit: Primary {UNIT_PRIMARY}, the control task and the world {UNIT_WORLD}")
    for key, value in timing.items():
        print(f"  {key}: {value:.1f} s")
    print(f"  iterations of the control fits: {iterations}")
    print(f"  convergence of the first fits (flag, iterations): {convergence}")
    print("projected CPU-hours at one thread per worker (measured | every fit at 30 iterations):")
    for key, value in hours.items():
        print(f"  {key}: {value:.2f} | {worst[key]:.2f}")
    print(
        f"the plan estimated {PLAN_ESTIMATE_CPU_HOURS} CPU-hours and stops at {STOP_CPU_HOURS:.0f}"
    )
    time_log(stage="unit", task="first_unit", start=time.time() - (load_s + control_s + world_s))
    if not_converged:
        msg = f"fits of the first unit did not converge: {not_converged} (see unit_timing.json)"
        raise RuntimeError(msg)


def world_from_files(
    *, inputs: PrimaryInputs, spec: WorldSpec, arrays: dict[str, np.ndarray]
) -> World:
    """Rebuild a `World` from its stored file, for the smoke check.

    Args:
        inputs: The Primary's inputs.
        spec: The world's recipe.
        arrays: The world file's arrays.

    Returns:
        The world (its power is the real power minus the stored injection plus the addition).
    """
    values = arrays["fleet"]
    fleet = (
        FleetParams(
            mix=(float(values[0]), float(values[1]), float(values[2])),
            tilt_deg=float(values[3]),
            dc_ac_ratio=float(values[4]),
        )
        if values.any()
        else None
    )
    return World(
        spec=spec,
        power=inputs.power - arrays["injected"] + arrays["addition"],
        injected=arrays["injected"],
        addition=arrays["addition"],
        truth=arrays["truth"],
        fleet=fleet,
        windows=[(int(a), int(b)) for a, b in arrays["windows"]],
    )


def stage_smoke(*, store: Path | None = None) -> dict[str, Any]:
    """Check gate G2 on every world file, that every world is present, and none is all zero.

    Args:
        store: The folder to check (`OUT` by default).

    Returns:
        The record (also written to `smoke.json` under the folder).

    Raises:
        RuntimeError: When a world file or fit file is missing, a world is all zero, or G2 fails.
    """
    from e00737_score import load_npz

    folder = store if store is not None else OUT
    injected = missing = bad = 0
    problems: list[str] = []
    for site in PRIMARIES:
        inputs = load_primary(site=site)
        for spec in world_specs(primary=site):
            fit_file = folder / "fits" / site / f"{spec.name}.npz"
            if spec.kind in ("placebo", "control"):
                fit_file = folder / "fits" / site / f"{spec.kind}.npz"
            if not fit_file.exists():
                missing += 1
                problems.append(f"missing fit file {site}/{fit_file.name}")
                continue
            if spec.kind in ("placebo", "control"):
                continue
            path = folder / "worlds" / site / f"{spec.name}.npz"
            if not path.exists():
                missing += 1
                problems.append(f"missing world file {site}/{spec.name}")
                continue
            injected += 1
            arrays = load_npz(path=path)
            if not arrays["injected"].any():
                bad += 1
                problems.append(f"all-zero injection {site}/{spec.name}")
                continue
            check = check_injection(
                inputs=inputs, world=world_from_files(inputs=inputs, spec=spec, arrays=arrays)
            )
            if not check["passed"]:
                bad += 1
                problems.append(f"G2 failed {site}/{spec.name}")
    record = {
        "injected_worlds_found": injected,
        "expected_injected": EXPECTED_INJECTED_WORLDS,
        "uninjected_fits_expected": EXPECTED_UNINJECTED_FITS,
        "missing": missing,
        "bad": bad,
        "problems": problems,
    }
    atomic_json(obj=record, path=folder / "smoke.json")
    if missing or bad or injected != EXPECTED_INJECTED_WORLDS:
        msg = f"smoke check failed: {record}"
        raise RuntimeError(msg)
    return record


def stage_gates() -> None:
    """Run the gates that need no fit and merge them into `gates.json`."""
    from e00737_gates import stage0_gates

    start = time.time()
    out = stage0_gates()
    existing = OUT / "gates.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if old.get("code_hash") == code_hash():
            for gate in ("G1", "G1b"):
                if gate in old:
                    out[gate] = old[gate]
    out["code_hash"] = code_hash()
    atomic_json(obj=out, path=existing)
    time_log(stage="gates", task="gates", start=start)
    print(json.dumps(out, indent=1, default=float))
    failed = [
        k
        for k in ("attributes", "known_answers", "quarters", "G2", "G3", "G4")
        if not out[k]["passed"]
    ]
    if failed:
        sys.exit(f"gates failed: {failed}")


def stage_g1_collect() -> None:
    """Add G1 and G1b to `gates.json`, which the current code must have written (no restamp)."""
    from e00737_gates import collect_g1

    path = OUT / "gates.json"
    out = read_record(name="gates.json")  # refuses when the record is missing or of other code
    record = collect_g1(tuned=require_tuned())
    out.update(record)
    atomic_json(obj=out, path=path)
    print(json.dumps(record, indent=1, default=float))
    if not all(v["passed"] for v in record.values()):
        sys.exit("gate G1 or G1b failed")


def run_g1_task(*, growth_sd: float, ridge_scale: float, draw: int, heavy: bool) -> None:
    """Run one G1 or G1b draw as a timed task.

    Args:
        growth_sd: The tuned growth standard deviation.
        ridge_scale: The tuned multiplier of the baseline's ridge precision.
        draw: The draw's number.
        heavy: Whether this is a G1b draw.
    """
    from e00737_gates import run_g1_draw

    start = time.time()
    run_g1_draw(tuned=Tuned(growth_sd=growth_sd, ridge_scale=ridge_scale), draw=draw, heavy=heavy)
    time_log(stage="g1", task=f"g1{'b' if heavy else ''}_{draw}", start=start)


def review_mark() -> None:
    """Write the hash of the current code into the review marker.

    A person or the manager runs this after the review of exactly this code.
    """
    marker = OUT / MARKER_REVIEW
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{code_hash()}  reviewed code\n")
    print(marker.read_text(), end="")


def freeze_code() -> None:
    """Write `code.sha256`, which needs a review marker that holds the current code hash."""
    require_review(marker=MARKER_REVIEW)
    CODE_SHA.parent.mkdir(parents=True, exist_ok=True)
    CODE_SHA.write_text(f"{code_hash()}  code\n")
    print(CODE_SHA.read_text(), end="")


def parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: The arguments (the process's by default).

    Returns:
        The parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--part", default="forward", choices=("forward", "fleets", "posterior", "real"))
    return ap.parse_args(argv)


def require_fit_stage() -> None:
    """Stop unless the review marker, the frozen code, and the gates allow a stage that fits."""
    require_review(marker=MARKER_REVIEW)
    require_code_frozen()
    require_gates()


def stage_g1_reference_or_draws(*, args: argparse.Namespace) -> None:
    """Run the G1 reference fit (`g1-reference`) or this worker's synthetic-recovery draws (`g1`).

    Args:
        args: The parsed arguments.
    """
    require_review(marker=MARKER_REVIEW)
    require_code_frozen()
    tuned = require_tuned()
    if args.stage == "g1-reference":
        from e00737_gates import write_g1_reference

        write_g1_reference(tuned=tuned)
        return
    tasks = [
        (
            f"g1{'b' if heavy else ''}_{d:03d}",
            {
                "growth_sd": tuned.growth_sd,
                "ridge_scale": tuned.ridge_scale,
                "draw": d,
                "heavy": heavy,
            },
        )
        for d in range(G1_DRAWS)
        for heavy in (False, True)
    ]
    run_shard(names_and_kwargs=shard(items=tasks, worker=args.worker, of=args.of), fn=run_g1_task)


def stage_fit(*, args: argparse.Namespace) -> None:
    """Fit this worker's Primaries (the bulk stage).

    Args:
        args: The parsed arguments.
    """
    require_fit_stage()
    require_budget(check_projection=True)
    unconverged = require_unit().get("not_converged")
    if unconverged:
        msg = f"refusing to fit: the first unit's fits did not converge: {unconverged}"
        raise RuntimeError(msg)
    tuned = require_tuned()
    tasks = fit_tasks(
        primaries=primaries_of(worker=args.worker, of=args.of), store=OUT, tuned=tuned
    )
    run_shard(names_and_kwargs=tasks, fn=run_fit_task)


def require_tune_stage() -> None:
    """Stop unless the review marker, the frozen code, and G2, G3, and G4 allow a tuning stage."""
    require_review(marker=MARKER_REVIEW)
    require_code_frozen()
    require_gates(required=GATES_BEFORE_TUNING)


def stage_tune(*, args: argparse.Namespace) -> None:
    """Fit this worker's share of the ridge draws (`tune-ridge`) or tuning worlds (`tune-growth`).

    Args:
        args: The parsed arguments.
    """
    require_tune_stage()
    part = "ridge" if args.stage == "tune-ridge" else "growth"
    tasks = tune_tasks(
        primaries=primaries_of(worker=args.worker, of=args.of),
        store=OUT,
        part=part,
        ridge_scale=None if part == "ridge" else require_ridge_scale(),
        worker=args.worker,
        of=args.of,
    )
    run_shard(names_and_kwargs=tasks, fn=run_tune_task)


def stage_score() -> None:
    """Score the stored fits."""
    require_fit_stage()
    require_unit()
    tuned = require_tuned()
    require_finished(
        names=[name for name, _ in fit_tasks(primaries=list(PRIMARIES), store=OUT, tuned=tuned)]
    )
    import e00737_score as score

    score.run_score()


def stage_report() -> None:
    """Print the report from the stored scores."""
    require_code_frozen()
    import e00737_score as score

    score.report()


def stage_figures(*, args: argparse.Namespace) -> None:
    """Draw the private figures.

    Args:
        args: The parsed arguments.
    """
    require_code_frozen()
    import e00737_figures as figs

    figs.make_figures(part=args.part)


def stage_checked(*, fn: Callable[[], Any]) -> None:
    """Run a stage that needs the review marker and the frozen code, and no gates record.

    Args:
        fn: The stage.
    """
    require_review(marker=MARKER_REVIEW)
    require_code_frozen()
    fn()


def dispatch(*, args: argparse.Namespace) -> None:
    """Run the stage that the arguments name.

    Args:
        args: The parsed arguments.
    """
    stage = args.stage
    simple: dict[str, Callable[[], Any]] = {
        "gates": stage_gates,
        "review-mark": review_mark,
        "freeze-code": freeze_code,
        "tune-ridge-collect": lambda: (
            require_tune_stage(),
            print(json.dumps(tune_ridge_collect(), indent=1)),
        ),
        "tune-collect": lambda: (
            require_tune_stage(),
            print(json.dumps(tune_collect(), indent=1)),
        ),
        "unit": lambda: (require_fit_stage(), stage_unit()),
        "score": stage_score,
        "report": stage_report,
    }
    with_args: dict[str, Callable[..., None]] = {
        "g1-reference": stage_g1_reference_or_draws,
        "g1": stage_g1_reference_or_draws,
        "tune-ridge": stage_tune,
        "tune-growth": stage_tune,
        "fit": stage_fit,
        "figures": stage_figures,
    }
    if stage in simple:
        simple[stage]()
    elif stage in with_args:
        with_args[stage](args=args)
    elif stage == "g1-collect":
        stage_checked(fn=stage_g1_collect)
    elif stage == "smoke":
        stage_checked(fn=lambda: print(json.dumps(stage_smoke(), indent=1)))
    else:
        msg = f"unknown stage {stage!r}"
        raise SystemExit(msg)


def main(*, argv: list[str] | None = None) -> None:
    """Parse the command line and run one stage, recording a failure and exiting with code 3.

    Args:
        argv: The arguments (the process's by default).
    """
    assert_blas_pinned()
    args = parse_args(argv=argv)
    print(f"output folder: {OUT}", flush=True)
    try:
        dispatch(args=args)
    except Exception as exc:  # noqa: BLE001 - the failure record is the point of this boundary
        write_stage_failure(stage=args.stage, worker=args.worker, exc=exc)
        print(f"FAILED stage {args.stage}: {exc}", file=sys.stderr, flush=True)
        sys.exit(EXIT_FAILED)


if __name__ == "__main__":
    main()

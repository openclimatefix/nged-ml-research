"""Run experiment 657: decode the injected worlds and the real record for each solar-term arm.

    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 \
        PYTHONPATH=ml-experiments/00653-edge-state-event-table-penalised-segmentation:. \
        uv run --no-sync python e00657_run.py <stage> [n_worlds]

Stages: `inject` (the default spec of experiment 653, same seeds), `real`. Experiment 653's
decoder, penalties, generator, and frozen noise units are used unchanged. The injected events are
added to the plain residual by experiment 653's generator and the arm's solar term is added to the
same world, which is the same as adding the events after the solar term (the generator only
adds to the residual). The slow level is refitted per arm and per world inside `decode_world`.
"""

from __future__ import annotations

import json
import sys
from typing import Any, cast

import numpy as np
import polars as pl
from e00653_common import GRID_LENGTH, SCORED_FROM  # ty: ignore[unresolved-import]
from e00653_core import decode_world, events_frame, match  # ty: ignore[unresolved-import]
from e00653_inject import GenParams, inject  # ty: ignore[unresolved-import]
from e00653_run import N_WORLDS, load_params  # ty: ignore[unresolved-import]
from e00657_solar import ARM_NAMES, WORK, SolarInputs, arm_residual, build  # ty: ignore[unresolved-import]

SPEC: GenParams = GenParams()
SPEC_NAME = "default"
KINDS = ("transfer", "negatives")


def check_frozen(*, inp: SolarInputs) -> None:
    """Assert that the recomputed noise units equal the frozen ones stored by experiment 653."""
    stored = json.loads(
        (
            WORK.parent / "00653-edge-state-event-table-penalised-segmentation/noise_units.json"
        ).read_text()
    )["sigma_block_mw"]
    for site, value in inp.frozen.sigma.items():
        if np.isfinite(value) or stored[site] is not None:
            assert abs(value - stored[site]) < 1e-9, (site, value, stored[site])


def run_inject(*, n_worlds: int = N_WORLDS) -> None:
    """Score the default-spec injected worlds for every arm; save the scored truth tables."""
    inp = build()
    check_frozen(inp=inp)
    params = load_params()
    WORK.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for k in range(n_worlds):
        world, truth = inject(
            panel=inp.panel,
            residual=inp.res,
            sigma=inp.frozen.sigma,
            rng=np.random.default_rng(1000 + 17 * k + len(SPEC_NAME)),
            lo=SCORED_FROM,
            hi=GRID_LENGTH,
            gen=SPEC,
            kinds=KINDS,
        )
        for arm in ARM_NAMES:
            arm_world = arm_residual(inp=inp, arm=arm, base=world)
            events, _ = decode_world(
                panel=inp.panel, residual=arm_world, frozen=inp.frozen, arm="masked", params=params
            )
            rows.extend(
                {**r, "spec": SPEC_NAME, "world": k, "arm": arm}
                for r in match(truth=truth, events=events)
            )
            print("inject", k, arm, len(events), flush=True)
    pl.DataFrame(rows).write_parquet(WORK / "injected_truth_scored.parquet")


def run_real() -> None:
    """Decode the real record for each arm; save the event tables and slow levels."""
    inp = build()
    check_frozen(inp=inp)
    params = load_params()
    WORK.mkdir(parents=True, exist_ok=True)
    for arm in ARM_NAMES:
        residual = arm_residual(inp=inp, arm=arm)
        events, slows = decode_world(
            panel=inp.panel, residual=residual, frozen=inp.frozen, arm="masked", params=params
        )
        events_frame(events=events).write_parquet(WORK / f"real_events_{arm}.parquet")
        np.savez(WORK / f"real_slow_{arm}.npz", **cast("dict[str, Any]", slows))
        print("real", arm, len(events), "events", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "inject":
        run_inject(n_worlds=int(sys.argv[2]) if len(sys.argv) > 2 else N_WORLDS)
    else:
        run_real()

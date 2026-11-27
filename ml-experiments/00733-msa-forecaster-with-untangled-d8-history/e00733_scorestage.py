# ruff: noqa: E501
"""Experiment 733: the `score` stage, which reads every stored prediction and writes the verdict.

`run_score` reads the real-fold contrasts (`e00733_scoring`) and the synthetic contrasts
(`e00733_scoring_synth`), applies the plan's guards and verdict table (`e00733_score.assemble`), and
writes `scores/verdict.json` and `scores/readings.json` under `root`. On `rows="validation"` it reads
the validation sets only and puts F0r in the place of M*, so no test month is read: the smoke test
and the scratch runs use it, and every number is labelled interim.
"""

import argparse
from pathlib import Path
from typing import Any

import e00733_core as core
import e00733_fit as fit
import e00733_score as sc
import e00733_scoring as scoring
import e00733_scoring_synth as synth
import e00733_stages as st

REAL_NAMES = ("C1", "C2", "R1", "R2", "R3", "R4", "R12")
EXTRA_READINGS = ("R8", "R9", "R10", "R11")


def counted_fit_failed(*, root: Path) -> bool:
    """Return whether a fit that C1 or C2 needs raised and was not rerun (verdict row 2).

    The fits that C1 and C2 read are FU (C1, C2) and FU-S (C2); F0r gates C1, so its failure is
    counted too. A failed Y0 or YU fit is not row 2: C3 and C4 are not in the verdict table, so the
    sign-forecast word and the history word read `void` instead. A failure of a report-only arm
    (FS or YT) changes nothing.

    Args:
        root: This experiment's data folder.

    Returns:
        Whether any F0r, FU, or FU-S job has a failure record and no fit record.
    """
    return any(
        fit.arm_failed(
            root=root, arm=arm, folds=core.ARMS[arm].folds or (None,), seeds=core.ARMS[arm].seeds
        )
        for arm in ("F0r", "FU", "FU-S")
    )


def run_score(
    *, root: Path, rows: str = "test", n_resamples_real: int | None = None, n_resamples_synthetic: int | None = None
) -> dict[str, Any]:  # fmt: skip
    """Score every stored prediction under `root` and write `scores/verdict.json` and `readings.json`.

    Args:
        root: This experiment's data folder (inputs and outputs).
        rows: "test" for the official score, "validation" for the smoke test and scratch runs.
        n_resamples_real: Replaces the bootstrap resamples of the real folds (tests only).
        n_resamples_synthetic: Replaces the resamples of the synthetic world (tests only).

    Returns:
        The verdict.

    Raises:
        IncompletePlanError: When C1 or C2 lacks stored forecasts for every allowed plan, and the
            chain did not stop, no fit failed, and no correctness check failed. A reading that is
            missing or not finite after a complete chain is verdict row 1 in the table itself.
    """
    state = st.chain_state(root=root)
    failed = counted_fit_failed(root=root)
    stopped, invalid = state == "stopped", st.invalid_recorded(root=root)
    real = scoring.real_readings(
        root=root, rows=rows, n_resamples=n_resamples_real,
        tolerate_missing=stopped or invalid or failed,
    )  # fmt: skip
    syn = synth.synthetic_readings(root=root, rows=rows, n_resamples=n_resamples_synthetic)
    contrasts: dict[str, Any] = {name: real.contrasts.get(name) for name in REAL_NAMES}
    contrasts.update(syn["contrasts"])
    c1 = contrasts["C1"]
    guards = None
    if c1 is not None and real.guards_inputs is not None:
        gi = real.guards_inputs
        guards = sc.guards_hold(c1=c1, mae_point=gi["mae"], q01=gi["q01"], q99=gi["q99"])
    verdict, readings = sc.assemble(
        stopped=stopped, invalid=invalid, failed=failed, c=contrasts, guards=guards, sign_inputs=syn["sign_inputs"], dropped_arms=st.dropped_arms(root=root),
    )  # fmt: skip
    readings.update({"R8": real.r8, "R9": real.r9, "R10": real.r10, "R11": syn["r11"]})
    readings["plans"] = real.plans
    readings["mae_of_median"] = {
        "C1": None if real.guards_inputs is None else real.guards_inputs["mae"]
    }
    readings["log_loss"] = syn["log_loss"]
    readings["n_scored_synthetic_half_hours"] = syn["n_scored"]
    interim = rows != "test" or any(p is not None and p["interim"] for p in real.plans.values())
    verdict["rows"] = rows
    verdict["interim"] = interim
    sc.write_results(root=root, verdict=verdict, readings=readings)
    return verdict


def stage_score(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Run the `score` stage: the official test score, or `--validation-only` on a scratch root.

    Args:
        args: The parsed arguments (`--validation-only`).
        root: This experiment's data folder.

    Returns:
        The verdict.

    Raises:
        RuntimeError: When `--validation-only` is given without a scratch data root.
    """
    validation = bool(getattr(args, "validation_only", False))
    if validation and getattr(args, "data_root", None) is None:
        msg = "--validation-only is for a scratch data root"
        raise RuntimeError(msg)
    verdict = run_score(root=root, rows="validation" if validation else "test")
    print(f"verdict row {verdict['row']}: {verdict['word']} (interim: {verdict['interim']})")
    return verdict

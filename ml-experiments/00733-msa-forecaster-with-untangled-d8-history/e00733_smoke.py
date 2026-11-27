# ruff: noqa: E501
"""Experiment 733: the real-environment smoke test (the orchestration; no torch import here).

The smoke test runs one pass of every arm (F0r, FU, FU-S, FS, Y0, YU, YT) on the real device and
the real
files, writing forecasts and states to a scratch folder outside the data folder, then reads each
stored
output with the scoring code, runs the extra real-file checks, and finally runs every later stage
(score,
report, figures, loss curves, attention maps) once on validation rows only. It never writes the
first-test-score
marker, and it asserts that the marker is absent afterwards. The record `scores/smoke.json` (a
small JSON file
in the data folder) holds one line per step and a single `pass`, which `e00733_stages.guard_run`
reads.

Every step is a callable of `SmokeSteps`, so the real adapters are passed by `e00733_cli` and a
test passes
stubs. A step that raises is recorded as failed with its message and traceback, and the later steps
still run,
so one smoke run shows every failure at once.
"""

import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import e00733_core as core
import e00733_paths as paths
import e00733_stages as st

SMOKE_ARMS: Final = tuple(a for a in core.FIT_ORDER if a != "FU-screen")
"""Every fitted arm, one pass each."""
LATER_STAGES: Final = ("score", "report", "figures", "loss_curves", "attention", "series")
"""The stages after the fits; each runs once on validation rows only."""


class SmokeError(RuntimeError):
    """The smoke test cannot start (a scratch folder inside the data folder.

    Or a test score already taken).
    """


@dataclass
class SmokeSteps:
    """What the smoke test calls.

    Attributes:
        fit_real: One pass of a real-fold arm: (arm, fold, seed) -> record.
        fit_synthetic: One pass of a synthetic arm: (arm, seed) -> record.
        predict: Predicts the fit into the scratch folder and returns the stored file: (arm, fold
            or None, seed) -> path.
        read_stored: Reads a stored file with the scoring code and returns the row count: (arm,
            path) -> n.
        checks: Extra real-file checks by name; each returns a detail string and raises on failure
            (the one-origin re-run, the
            banks' digests, the synthetic samples of one test segment, the channel coverage per
            fold, experiment 729's smoke script).
        later: The later stages, each called with `validation_only=True`.
        log: Receives one line per step.
        gpu_info: Returns `{"gpu_name", "peak_gpu_gb"}` of this process after the steps (the real
            GPU run gives it; the CPU run has none).
    """

    fit_real: Callable[[str, str, int], dict]
    fit_synthetic: Callable[[str, int], dict]
    predict: Callable[[str, str | None, int], Path]
    read_stored: Callable[[str, Path], int]
    checks: Mapping[str, Callable[[], str]] = field(default_factory=dict)
    later: Mapping[str, Callable[..., object]] = field(default_factory=dict)
    log: Callable[[str], None] = print
    gpu_info: Callable[[], Mapping[str, Any]] | None = None


def assert_scratch_outside(*, scratch: Path, data_root: Path) -> None:
    """Refuse a scratch folder that is, or lies inside.

    The data folder of this or any earlier experiment.

    Args:
        scratch: The scratch folder.
        data_root: This experiment's data folder.

    Raises:
        SmokeError: When the scratch folder is inside `DATA_PATH_INTERNAL`'s `ml-experiments`
            folder.
    """
    inside = paths.data_root().parent.resolve()
    for folder in (data_root.resolve(), inside):
        s = scratch.resolve()
        if s == folder or folder in s.parents:
            msg = "the scratch folder lies inside the data folder: use a scratch root outside it"
            raise SmokeError(msg)


def _step(
    *, record: dict[str, Any], name: str, fn: Callable[[], str], log: Callable[[str], None]
) -> bool:
    start = time.monotonic()
    try:
        detail = fn()
        record[name] = {"ok": True, "detail": detail, "seconds": round(time.monotonic() - start, 1)}
    except (st.StoppedError, st.CorrectnessFailureError, Exception) as exc:  # noqa: BLE001
        record[name] = {
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "seconds": round(time.monotonic() - start, 1),
        }
    log(f"smoke {name}: {'ok' if record[name]['ok'] else 'FAIL'}: {record[name]['detail']}")
    return bool(record[name]["ok"])


def run_smoke(*, root: Path, scratch: Path, steps: SmokeSteps, device: str) -> dict[str, Any]:
    """Run the smoke test and write `scores/smoke.json` (`scores/smoke_cpu.json` on the CPU).

    Args:
        root: This experiment's data folder (the record is written here; nothing else is).
        scratch: The scratch folder for states and forecasts (outside the data folder).
        steps: The steps.
        device: "cuda" or "cpu" (recorded; only a "cuda" record opens the chain, and it carries the
            GPU's name and the peak GPU memory of the steps).

    Returns:
        The record, with `pass` true only when every step passed.

    Raises:
        SmokeError: When the scratch folder is inside the data folder or the first test score was
            already taken.
    """
    assert_scratch_outside(scratch=scratch, data_root=root)
    if st.marker_exists(root=root):
        msg = "the first test score was already taken, so a smoke test would be a choice made after the test"
        raise SmokeError(msg)
    scratch.mkdir(parents=True, exist_ok=True)
    steps_record: dict[str, Any] = {}
    stored: dict[str, Path] = {}
    for arm in SMOKE_ARMS:
        spec = core.ARMS[arm]
        fold = spec.folds[0] if spec.where == "real" else None
        seed = 0

        def fit_one(
            arm: str = arm, fold: str | None = fold, spec: Any = spec, seed: int = seed
        ) -> str:
            rec = (
                steps.fit_real(arm, fold, seed)  # ty: ignore[invalid-argument-type]
                if spec.where == "real"
                else steps.fit_synthetic(arm, seed)
            )  # ty: ignore[invalid-argument-type]
            return f"one pass, stored pass {rec.get('stored_pass', [s['stored_pass'] for s in rec.get('stages', [])])}"

        def predict_one(arm: str = arm, fold: str | None = fold, seed: int = seed) -> str:
            path = steps.predict(arm, fold, seed)
            stored[arm] = path
            return f"stored {path.name}"

        def read_one(arm: str = arm) -> str:
            n = steps.read_stored(arm, stored[arm])
            if n <= 0:
                msg = "the scoring code read no row from the stored output"
                raise SmokeError(msg)
            return f"{n} rows read by the scoring code"

        if _step(record=steps_record, name=f"fit {arm}", fn=fit_one, log=steps.log) and _step(
            record=steps_record, name=f"predict {arm}", fn=predict_one, log=steps.log
        ):
            _step(record=steps_record, name=f"read {arm}", fn=read_one, log=steps.log)
    for name, fn in steps.checks.items():
        _step(record=steps_record, name=f"check {name}", fn=fn, log=steps.log)
    for name in LATER_STAGES:
        if name not in steps.later:
            steps_record[f"later {name}"] = {
                "ok": False,
                "detail": "no function given for this stage",
            }
            steps.log(f"smoke later {name}: FAIL: no function given")
            continue
        _step(
            record=steps_record,
            name=f"later {name}",
            fn=lambda n=name: str(
                steps.later[n](validation_only=True) or "ran on validation rows only"
            ),
            log=steps.log,
        )
    marker_absent = not st.marker_exists(root=root)
    steps_record["no test marker"] = {
        "ok": marker_absent,
        "detail": "the first-test-score marker is absent" if marker_absent else "the marker exists",
    }
    record = {
        "pass": all(v["ok"] for v in steps_record.values()), "code_version": st.CODE_VERSION, "device": device,
        "scratch_outside_data_folder": True, "steps": steps_record,
    }  # fmt: skip
    if device == "cuda":
        info = dict(steps.gpu_info()) if steps.gpu_info is not None else {}
        record["gpu_name"] = str(info.get("gpu_name", ""))
        record["peak_gpu_gb"] = float(info.get("peak_gpu_gb", 0.0))
    st.write_json(path=root / st.smoke_file(device=device), payload=record)
    return record

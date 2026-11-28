"""Constants, paths, and small file helpers shared by the scripts of experiment 736.

Nothing here reads private data. Every file the experiment writes goes under `OUT`, which is
`data/ml-experiments/00736-switching-offset-model-vs-untangler-on-d8/` unless the environment
variable `E00736_OUT` names another folder (the smoke test uses a scratch folder).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

FOLDER: Final[str] = "00736-switching-offset-model-vs-untangler-on-d8"


def _output_folder() -> Path:
    """Return the experiment's output folder, refusing an override that is not a scratch folder.

    Returns:
        `E00736_OUT` when set (and `E00736_SCRATCH=1` or a `scratch` path component allows it),
        otherwise the experiment's folder under the shared data folder.

    Raises:
        RuntimeError: When `E00736_OUT` is set to a path that is not marked as a scratch folder.
    """
    override = os.environ.get("E00736_OUT")
    if override is None:
        return ML_EXPERIMENTS_DATA_DIR / FOLDER
    path = Path(override)
    if os.environ.get("E00736_SCRATCH") != "1" and "scratch" not in path.parts:
        msg = f"E00736_OUT={override} is not a scratch folder (set E00736_SCRATCH=1 to allow it)"
        raise RuntimeError(msg)
    return path


OUT: Final[Path] = _output_folder()
"""The only folder this experiment writes to."""
DATA_732: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00732-msa-d8-untangling-network-synthetic-injections"
)
"""Experiment 732's data folder, read only."""
DATA_734: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00734-switching-offset-model-part-1"
"""Experiment 734's data folder; only `params.json` and `uninjected/` are read, both read only."""
RECAL: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "briefs" / "msa_architecture" / "e_untangler_recal"
"""The recalibration report's folder (not committed), read only."""
DIAG: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "briefs" / "msa_architecture" / "e_untangler_diag"
"""The diagnostics' folder (not committed), read only."""

POINT_MW: Final[float] = 10.379 / 100.0
"""Megawatts per point of the folded reading (10.379 MVA over 100 points)."""
HALF_HOURS_PER_DAY: Final[int] = 48
BLOCK: Final[int] = 12
"""Half-hours per 6-hour block, which is also the start tolerance of the match rule."""
TOLERANCE_HH: Final[int] = 12
"""Start tolerance of the match rule, 6 hours in half-hours."""
MERGE_GAP_HH: Final[int] = 4
"""Calls of one sign closer than this many half-hours (2 hours) merge into one call."""
PLACEBO_SHIFT_HH: Final[int] = 7 * HALF_HOURS_PER_DAY
"""The placebo moves every injected start this many half-hours later (7 days)."""
INDEX_OFFSET: Final[int] = 1
"""The offset model's panel index of a bank half-hour is the bank half-hour plus this."""
STEP_UNITS_COUNTED: Final[float] = 4.0
"""Smallest transfer size of the counted cell, in step noise units."""
COUNTED_HOURS: Final[tuple[float, float]] = (24.0, 96.0)
"""Duration of the counted cell in hours, both ends included."""
MIN_COUNTED: Final[int] = 20
"""Fewest transfers a cell needs to be read."""
N_DRAWS: Final[int] = 8
"""Worlds, one per draw of experiment 732's test bank."""
N_MONTHS: Final[int] = 6
"""Test months that lie inside experiment 653's record."""
OVERLAY_DIGEST_732: Final[str] = "4c57546a9278dc0d0a6e3b2803d880b3205ae441dea7725681ab168000c30c6f"
"""Digest of experiment 732's regenerated test overlays (gate G4)."""
FROZEN_THRESHOLDS_SHA256: Final[str] = (
    "49e81b90bc600dd27097efea831fbb209c5747a72d2419264d7758686325cc95"
)
"""SHA-256 of the recalibration report's `frozen_thresholds.json`."""
PARAMS_734_SHA256: Final[str] = "d7b3c57cb43c21fcb9569b66cadc5d4b50eacf1ad98395fc67a94cef37896120"
"""SHA-256 of experiment 734's `params.json`, the tuned values that the decode reads unchanged."""
EXIT_FAILED: Final[int] = 3
"""Exit code of a failed task."""
ARMS: Final[tuple[str, ...]] = ("H-own", "H-nbr")
"""Offset-model arms that are decoded."""


def atomic_json(*, obj: object, path: Path) -> None:
    """Write a JSON file under a temporary name and rename it.

    Args:
        obj: The JSON-serialisable object.
        path: The final path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.rename(path)


def sha256_of(*, path: Path) -> str:
    """Return the hex SHA-256 of a file.

    Args:
        path: The file.

    Returns:
        The digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def time_log(*, stage: str, task: str, start: float) -> None:
    """Append one line (stage, task, device, start, end) to the per-phase time log.

    Args:
        stage: The phase, such as `decode`.
        task: The task's name.
        start: The task's start, from `time.time()`.
    """
    stamp = "%FT%T"
    begin, end = (time.strftime(stamp, time.gmtime(v)) for v in (start, time.time()))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "time_log.csv").open("a") as f:
        f.write(f"{stage},{task},cpu,{begin},{end}\n")

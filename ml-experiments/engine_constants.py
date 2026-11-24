"""Read the gas engine's private constants from one file that is never committed.

A script that gates rows on the D8 engine's connection date, or that divides a record by the
engine's recorded unit size, needs those values to run. NGED has asked that the values stay out of
the repository, so they live in `engine_constants.json` under the private data folder
(`DATA_PATH_INTERNAL`), at `ml-experiments/private/engine_constants.json`. The file is a flat
JSON object. Dates are `YYYY-MM-DD` strings and sizes are numbers.

The environment variable `ENGINE_CONSTANTS_PATH` names a different file, which a test uses to
supply invented numbers. Every error names the file and the key but never the value found.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

PATH_ENV_VAR: Final[str] = "ENGINE_CONSTANTS_PATH"
DEFAULT_PATH: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "private" / "engine_constants.json"


class EngineConstantsError(RuntimeError):
    """Raised when the private constants file is missing, unreadable, or lacks a usable key."""


def constants_path() -> Path:
    """Return the file the constants are read from."""
    override = os.environ.get(PATH_ENV_VAR)
    return Path(override) if override else DEFAULT_PATH


def _read(*, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EngineConstantsError(
            f"The private engine-constants file {path} does not exist. Create it as a flat JSON "
            f"object (see ml-experiments/engine_constants.py), or set {PATH_ENV_VAR} to a file."
        )
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        raise EngineConstantsError(
            f"The private engine-constants file {path} is not JSON."
        ) from None
    if not isinstance(loaded, dict):
        raise EngineConstantsError(
            f"The private engine-constants file {path} is not a JSON object."
        )
    return loaded


def _value(*, key: str, path: Path | None) -> tuple[Any, Path]:
    used = path or constants_path()
    constants = _read(path=used)
    if key not in constants:
        raise EngineConstantsError(f"The key {key!r} is missing from the private file {used}.")
    return constants[key], used


def engine_float(*, key: str, path: Path | None = None) -> float:
    """Return the number stored under `key`, in the unit the key's name gives."""
    value, used = _value(key=key, path=path)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EngineConstantsError(
            f"The key {key!r} in {used} must be a number, not {type(value).__name__}."
        )
    return float(value)


def engine_string(*, key: str, path: Path | None = None) -> str:
    """Return the non-empty string stored under `key`."""
    value, used = _value(key=key, path=path)
    if not isinstance(value, str) or not value:
        raise EngineConstantsError(
            f"The key {key!r} in {used} must be a non-empty string, not {type(value).__name__}."
        )
    return value


def engine_datetime(*, key: str, path: Path | None = None) -> datetime:
    """Return midnight UTC on the `YYYY-MM-DD` date stored under `key`."""
    value, used = _value(key=key, path=path)
    if not isinstance(value, str):
        raise EngineConstantsError(
            f"The key {key!r} in {used} must be a YYYY-MM-DD string, not {type(value).__name__}."
        )
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        raise EngineConstantsError(f"The key {key!r} in {used} is not a YYYY-MM-DD date.") from None

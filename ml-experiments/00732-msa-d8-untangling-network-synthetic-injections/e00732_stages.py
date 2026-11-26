# ruff: noqa: E501
"""Experiment 732: the gates and the stages that need no torch (the banks check, neighbours.

Limits, overlays, tensor checks, guard).

Every stage takes the data folder `root` of this experiment and experiment 730's data folder
`root730`, which is read and never written. A stage runs `guard_run` first, so a stage on the outputs
of another run, before a gate, or after a correctness failure refuses to start. A gate that fails
raises `StoppedError` (verdict row 0) or `CorrectnessFailureError` (verdict row 1), and the command
line turns either into a status file and a stop. No stage prints a date, a series identifier, or a
value of a real solar era.
"""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00732_paths  # noqa: F401

# isort: split
import e00729_arms as arms729
import e00729_banks as banks729
import e00729_score as score729
import e00730_banks as banks730
import e00730_masks as masks
import e00730_score as score730
import e00732_arms as arms
import e00732_baselines as baselines
import e00732_core as core
import e00732_leakage as leak
import e00732_neighbours as nb
import e00732_overlays as ov
import e00732_score as score
import numpy as np

Log = Callable[[str], None]
StoppedError = banks730.StoppedError


class GuardError(RuntimeError):
    """Raised by `guard_run` when a gate is open or the folder holds the outputs of another run."""


RUN_MARKER: Final = "run_marker.json"
CODE_VERSION: Final = 2
"""The version of this experiment's code that the run marker records. A change that alters a stored output raises it."""
INVALID_FILE: Final = "scores/invalid.json"
BANKS_CHECK: Final = "banks_check.json"
NEIGHBOUR_CHECK: Final = "neighbours/check.json"
MEDIANS_FILE: Final = "neighbours/medians.json"
TENSOR_FILE: Final = "scores/tensor_checks.json"
GATE_ATTEMPTS: Final = "scores/gate_attempts.json"
MAX_GATE_VERSIONS: Final = 2
"""The gates run under at most two code versions: the first, and one after a change."""
TEST_MARKER: Final = "test_started"
BT_RECORD: Final = "scores/bt_fit.json"
GATED_STAGES: Final = ("timing", "fit", "chain")
"""The stages that start a fit: they need the neighbour table, the limits, the overlays, the tensor checks, the gallery check, and no recorded failure."""
FIT_STAGES: Final = ("fit", "chain")
"""The stages that also need the timing gate."""
CONTROL_FOLDERS: Final = {"S": ("S", "unet"), "S-L": ("S", "bilstm"), "SD*": ("SD-R", "unet")}
STORED_SEEDS: Final = (0, 1, 2)


@dataclass
class Env:
    """What the stages read from the real files (or, in the tests, from an invented calendar).

    Attributes:
        holidays: Bank-holiday dates.
        stats: The standardisation constants of the temperature and wind speed.
        neighbours: D3, D5, and D14 on D8's base calendar, in points (D5 and D14 signed), as
            `neighbours.read_table` returns them.
        medians: The training median of each folded neighbour (the scale of its channels).
        real_sets: A function that returns the real-day pseudo-segments by set name (None without real data).
        real_neighbours: The three neighbours' series on the calendar of the real-day pseudo-segments
            (D8's whole day table, not only the base days), in points, or None for zero channels.
    """

    holidays: np.ndarray
    stats: arms729.InputStats
    neighbours: Mapping[str, np.ndarray]
    medians: Mapping[str, float]
    real_sets: Callable[[], dict[str, list]] | None = None
    real_neighbours: Mapping[str, np.ndarray] | None = None


@dataclass
class BaseDays:
    """D8's base calendar as the neighbour and limits stages read it.

    Attributes:
        dates: The base days, sorted, `datetime64[D]`.
        reading: D8's reading in points, `(n, 48)`, NaN where dropped.
        bsp: The BSP meter in points, `(n, 48)`, NaN where the meter has no data.
        holidays: Bank-holiday dates.
        split: The split of each base day ("training", "validation", or "test").
    """

    dates: np.ndarray
    reading: np.ndarray
    bsp: np.ndarray
    holidays: np.ndarray
    split: np.ndarray

    @property
    def train_rows(self) -> np.ndarray:
        """Boolean mask of the training base days."""
        return np.asarray(self.split) == "training"

    @property
    def weekend(self) -> np.ndarray:
        """Whether each base day is a weekend day or a bank holiday."""
        return baselines.weekend_flags(dates=self.dates, holidays=self.holidays)


# --------------------------------------------------------------------------- files


def write_json(*, path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON file atomically."""
    banks730.write_json(path=path, payload=payload)


def read_json(*, path: Path) -> dict:
    """Read a JSON file."""
    return banks730.read_json(path=path)


def write_once_json(*, path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON file that may never be written twice (a second write raises `FileExistsError`)."""
    core.core730.write_once_json(path=path, payload=payload)


def file_digest(*, path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_dir(*, root: Path) -> Path:
    """Return the folder of the fit records."""
    return root / "fits"


def job_path(*, root: Path, arm: str, arch: str, seed: int) -> Path:
    """Return the record file of one fit."""
    return fit_dir(root=root) / f"{core.job_name(arm=arm, arch=arch, seed=seed)}.json"


def record_invalid(*, root: Path, rule: str) -> None:
    """Write `scores/invalid.json` once (verdict row 1); a later call leaves the first record."""
    path = root / INVALID_FILE
    if not path.exists():
        write_json(path=path, payload={"rule": rule})


READING_STAGES: Final = ("score", "real", "report", "figures", "series")
"""The stages that read the test outputs; none may run while the chain is fitting."""
CHAIN_STATUS: Final = "chain_status.json"


def marker_exists(*, root: Path) -> bool:
    """Return whether the first test score has been taken (the marker exists)."""
    return (root / "scores" / TEST_MARKER).exists()


def chain_running(*, root: Path) -> bool:
    """Return whether `chain_status.json` says that the chain is running."""
    path = root / CHAIN_STATUS
    return path.exists() and read_json(path=path).get("state") == "running"


def choice_events(*, root: Path) -> list[str]:
    """Return the choices of the run and the first test score, in order of file time.

    The events are `limits` (`limits.json`), `fits` (the newest stored prediction file), `bt` (the
    record of the B-T fit, `scores/bt_fit.json`), and `test_score` (the marker `scores/test_started`,
    made now when it does not exist yet). A choice that was rewritten after the first test score
    sorts after `test_score`, and `leak.assert_no_test_choice` fails. A choice that does not exist is
    left out, and `assert_no_test_choice` fails on that too.

    Args:
        root: The data folder.

    Returns:
        The event names, earliest first.
    """
    marker = root / "scores" / TEST_MARKER
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("")
    stamps: dict[str, int] = {"test_score": marker.stat().st_mtime_ns}
    if (root / "limits.json").exists():
        stamps["limits"] = (root / "limits.json").stat().st_mtime_ns
    fits = prediction_files(root=root)
    if fits:
        stamps["fits"] = max(p.stat().st_mtime_ns for p in fits)
    bt = root / BT_RECORD
    if bt.exists():
        stamps["bt"] = bt.stat().st_mtime_ns
    return sorted(stamps, key=lambda k: (stamps[k], k == "test_score"))


def prediction_files(*, root: Path) -> list[Path]:
    """Return the stored prediction files of this experiment's fits."""
    return sorted((root / "predictions").glob("*/*.npz"))


def invalid_recorded(*, root: Path) -> bool:
    """Return whether a correctness check has failed in this run."""
    return (root / INVALID_FILE).exists()


# --------------------------------------------------------------------------- the run marker and the guard


def run_marker_payload(*, root730: Path) -> dict:
    """Return what the run marker holds: the experiment, the row set, and the code version.

    The marker also holds the id of experiment 730's banks.
    """
    return {
        "experiment": "00732",
        "row_set_id": core.ROW_SET_ID,
        "code_version": CODE_VERSION,
        "banks_digest": banks730.banks_digest_id(root=root730),
    }


def gallery_passed(*, root: Path, root730: Path) -> bool:
    """Return whether `PRIVATE/gallery/panel_check.json` records every panel as passed.

    The record must be for the banks in use.
    """
    path = root / "PRIVATE" / "gallery" / "panel_check.json"
    if not path.exists():
        return False
    record = read_json(path=path)
    if record.get("banks_digest") != banks730.banks_digest_id(root=root730):
        return False
    panels = record.get("panels", {})
    return (
        bool(panels)
        and set(core.PANEL_REQUIREMENTS) <= set(panels)
        and all(bool(v) for v in panels.values())
    )


def _passed(*, path: Path) -> bool:
    """Return whether a JSON file exists and holds a true `pass`."""
    return path.exists() and bool(read_json(path=path).get("pass"))


def overlays_passed(*, root: Path) -> bool:
    """Return whether `overlays/summary.json` records a pass."""
    return _passed(path=root / "overlays" / "summary.json")


def neighbours_passed(*, root: Path) -> bool:
    """Return whether the neighbour table exists and `neighbours/check.json` records a pass."""
    return (root / nb.TABLE_NAME).exists() and _passed(path=root / NEIGHBOUR_CHECK)


def tensor_passed(*, root: Path) -> bool:
    """Return whether `scores/tensor_checks.json` records a pass."""
    return _passed(path=root / TENSOR_FILE)


def timing_passed(*, root: Path) -> bool:
    """Return whether `scores/timing.json` records a pass."""
    return _passed(path=root / "scores" / "timing.json")


def fits_exist(*, root: Path) -> bool:
    """Return whether any fit, network, or stored prediction exists."""
    return any(
        (root / n).exists() and any((root / n).iterdir())
        for n in ("fits", "networks", "predictions")
    )


def gate_attempts(*, root: Path) -> dict:
    """Return the gate record: the code versions that have run the gate and the attempts used."""
    path = root / GATE_ATTEMPTS
    return read_json(path=path) if path.exists() else {"versions": [], "attempts": 0}


def begin_gate_attempt(*, root: Path) -> None:
    """Count one attempt of the gates (the overlays and tensor stages) under the current code version.

    The run has one attempt, and one more after a change of `CODE_VERSION`. A run of the same code
    version is not a new attempt, because the gates are deterministic.

    Args:
        root: The data folder.

    Raises:
        StoppedError: When a third code version asks for a gate attempt (verdict row 0).
    """
    record = gate_attempts(root=root)
    if CODE_VERSION in record["versions"]:
        return
    if len(record["versions"]) >= MAX_GATE_VERSIONS:
        msg = "the gates have used their one attempt after a change of the code version (verdict row 0)"
        raise StoppedError(msg)
    record["versions"].append(CODE_VERSION)
    record["attempts"] = int(record["attempts"]) + 1
    write_json(path=root / GATE_ATTEMPTS, payload=record)


def _guard_banks(*, root: Path, expected: dict, current: dict | None) -> None:
    """Apply the `banks` stage's rules of `guard_run`."""
    if current is None or current == expected:
        return
    only_version = {k: v for k, v in current.items() if k != "code_version"} == {
        k: v for k, v in expected.items() if k != "code_version"
    }
    if fits_exist(root=root) or not only_version:
        msg = "the run marker differs from the code and the banks in use: move the outputs to superseded_run_N yourself"
        raise GuardError(msg)
    if len(gate_attempts(root=root)["versions"]) >= MAX_GATE_VERSIONS:
        msg = "the gates have used their one attempt after a change of the code version (verdict row 0)"
        raise GuardError(msg)


def guard_run(*, root: Path, root730: Path, stage: str) -> None:
    """Refuse to run a stage on the outputs of another run, or without the gates before a fit.

    Args:
        root: The data folder of this experiment.
        root730: The data folder of experiment 730 (read only).
        stage: The stage name.

    Raises:
        GuardError: When a gate is open, a correctness check has failed, or the folder holds
            outputs of another run.
    """
    expected = run_marker_payload(root730=root730)
    marker = root / RUN_MARKER
    current = read_json(path=marker) if marker.exists() else None
    if stage == "banks":
        _guard_banks(root=root, expected=expected, current=current)
        return
    if current != expected:
        msg = "no run marker matches the banks and the code in use: run `banks` first"
        raise GuardError(msg)
    needs = {
        "limits": (("neighbours", neighbours_passed(root=root)),),
        "overlays": (("limits.json", (root / "limits.json").exists()),),
        "tensor": (("limits.json", (root / "limits.json").exists()),),
        "gallery": (
            ("neighbours", neighbours_passed(root=root)),
            ("limits.json", (root / "limits.json").exists()),
        ),
    }
    for what, ok in needs.get(stage, ()):
        if not ok:
            msg = f"`{stage}` needs {what} first"
            raise GuardError(msg)
    if stage in GATED_STAGES and marker_exists(root=root):
        msg = f"`{stage}` starts a fit, but the first test score was already taken: a fit now would be a choice made after the test (verdict row 1)"
        raise GuardError(msg)
    if stage in GATED_STAGES:
        if invalid_recorded(root=root):
            msg = f"`{stage}` starts a fit, but {INVALID_FILE} records a failed correctness check (verdict row 1): the run is over"
            raise GuardError(msg)
        for what, ok in (
            (
                "neighbours/check.json with a pass and the neighbour table",
                neighbours_passed(root=root),
            ),
            ("limits.json, written before any fit", (root / "limits.json").exists()),
            ("overlays/summary.json with a pass", overlays_passed(root=root)),
            (f"{TENSOR_FILE} with a pass", tensor_passed(root=root)),
            (
                "PRIVATE/gallery/panel_check.json with every panel passed (the gallery gate of the plan)",
                gallery_passed(root=root, root730=root730),
            ),
        ):
            if not ok:
                msg = f"`{stage}` needs {what}"
                raise GuardError(msg)
    if stage in FIT_STAGES and not timing_passed(root=root):
        msg = f"`{stage}` needs scores/timing.json with a pass: run `timing` first"
        raise GuardError(msg)
    if (
        stage in ("score", "real", "report", "figures", "series")
        and not (root / "limits.json").exists()
    ):
        msg = f"`{stage}` needs limits.json"
        raise GuardError(msg)
    if stage in READING_STAGES and chain_running(root=root):
        msg = f"`{stage}` reads the test outputs, but the chain is running (`chain_status.json` says running): wait for the last fit"
        raise GuardError(msg)


# --------------------------------------------------------------------------- the banks of experiment 730


def job_name_730(*, arm: str, arch: str, seed: int) -> str:
    """Return the folder name of one of experiment 730's fits.

    The name for `SD*` is `SDstar`, because no stored control holds an asterisk.
    """
    return f"{arm.replace('*', 'star')}_{arch}_s{seed}"


def stored_predictions(
    *, root730: Path, arm: str, arch: str, seeds: Sequence[int] = STORED_SEEDS
) -> list[Path]:
    """Return the `test.npz` files of one stored arm of experiment 730, in seed order."""
    return [
        root730 / "predictions" / job_name_730(arm=arm, arch=arch, seed=s) / "test.npz"
        for s in seeds
    ]


def load_stored(
    *,
    root730: Path,
    arm: str,
    arch: str,
    seeds: Sequence[int] = STORED_SEEDS,
    file: str = "test",
) -> list[dict[str, np.ndarray]]:
    """Read one stored arm's arrays of every seed (read only)."""
    out = []
    for s in seeds:
        path = root730 / "predictions" / job_name_730(arm=arm, arch=arch, seed=s) / f"{file}.npz"
        with np.load(path) as z:
            out.append({k: z[k] for k in z.files})
    return out


def seed_mean_p(*, arrays: Sequence[Mapping[str, np.ndarray]]) -> np.ndarray:
    """Return the seed-mean P(X<0) of a stored arm."""
    return score730.seed_mean(p_by_seed=[a["p"].astype(np.float64) for a in arrays])


def assert_bank_months_disjoint(*, root730: Path, test_segments: Sequence[Any]) -> None:
    """Check that no month of the test bank is a month of a training bank or of the validation bank.

    Args:
        root730: The data folder of experiment 730 (read only).
        test_segments: The test bank's segments.

    Raises:
        CorrectnessFailureError: When a month is shared, or a month list is empty.
    """
    used: dict[str, set[str]] = {"validation": set()}
    va = banks729.load_bank(path=banks730.bank_path(root=root730, kind="validation"))
    used["validation"] = {str(seg.month) for seg in va.segments}
    train: set[str] = set()
    for path in sorted((root730 / "banks").glob("train_s*.pkl")):
        train |= {str(seg.month) for seg in banks729.load_bank(path=path).segments}
    used["training"] = train
    leak.assert_months_disjoint(used=used, test_months=[str(seg.month) for seg in test_segments])


def stage_banks(*, root: Path, root730: Path, holidays: np.ndarray, log: Log) -> dict:
    """Verify experiment 730's banks and stored files, record their digests once, and write the run marker.

    The stage reads experiment 730's data folder only. It stops at verdict row 0 when a bank digest, the
    stored margin, or a stored prediction file differs from the digest recorded at the first run, and
    fails at verdict row 1 (through `score.stored_prediction_problems`) when a stored array has the
    wrong row count, the row-key digest differs, or the Brier canary of S does not reproduce.

    Args:
        root: The data folder of this experiment.
        root730: The data folder of experiment 730 (read only).
        holidays: Bank-holiday dates (the rows are built with them).
        log: Receives progress lines.

    Returns:
        The record of `banks_check.json`.

    Raises:
        StoppedError: When a digest differs (verdict row 0).
        CorrectnessFailureError: When a row count, the row-key digest, or the canary fails (verdict row 1).
    """
    guard_run(root=root, root730=root730, stage="banks")
    banks730.verify_banks_digest(root=root730)
    test = banks729.load_bank(path=banks730.bank_path(root=root730, kind="test"))
    assert_bank_months_disjoint(root730=root730, test_segments=test.segments)
    leak.assert_pair_index_from_own_bank(
        index=ov.build_pair_index(segments=test.segments, sources=core.TEST_SIBLING_SOURCES),
        segments=test.segments,
    )
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    path = root / BANKS_CHECK
    first = read_json(path=path) if path.exists() else None
    stored = {
        name: load_stored(root730=root730, arm=arm, arch=arch)
        for name, (arm, arch) in core.STORED_730.items()
    }
    problems = score.stored_prediction_problems(
        arrays=stored,
        rows=rows,
        recorded_digest=score.row_key_digest(rows=rows)
        if first is None
        else first["row_key_digest"],
    )
    if problems:
        msg = "; ".join(problems) + " (verdict row 1)"
        raise leak.CorrectnessFailureError(msg)
    files = {
        f"{name}_s{k}": file_digest(path=p)
        for name, (arm, arch) in core.STORED_730.items()
        for k, p in enumerate(stored_predictions(root730=root730, arm=arm, arch=arch))
    }
    margin = read_json(path=root730 / "margin.json")
    if abs(float(margin["margin"]) - core.MARGIN_730) > core.MARGIN_730_FILE_TOLERANCE:
        msg = "experiment 730's margin.json does not hold the margin of the plan (verdict row 0)"
        raise StoppedError(msg)
    record = {
        "n_rows": rows.n, "row_key_digest": score.row_key_digest(rows=rows), "brier_canary_matches": True,
        "predictions": files, "margin_json": file_digest(path=root730 / "margin.json"), "banks_digest": banks730.banks_digest_id(root=root730),
    }  # fmt: skip
    if first is not None:
        if (
            first["predictions"] != record["predictions"]
            or first["margin_json"] != record["margin_json"]
        ):
            msg = "experiment 730's stored predictions or margin.json differ from their recorded digests (verdict row 0)"
            raise StoppedError(msg)
    else:
        write_once_json(path=path, payload=record)
    write_json(path=root / RUN_MARKER, payload=run_marker_payload(root730=root730))
    log(f"banks verified: {rows.n} rows, {len(files)} stored files, canary reproduced")
    return record


# --------------------------------------------------------------------------- the neighbour table


def split_coverage(
    *, base: BaseDays, arrays: Mapping[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Return each neighbour's coverage of D8's present half-hours in each split."""
    out: dict[str, dict[str, float]] = {}
    for split in ("training", "validation", "test"):
        rows = np.asarray(base.split) == split
        if rows.any() and np.isfinite(base.reading[rows]).any():
            out[split] = nb.coverage(
                table={n: arrays[n][rows] for n in nb.NEIGHBOURS}, reading=base.reading[rows]
            )
    return out


def stage_neighbours(
    *, root: Path, root730: Path, base: BaseDays,
    build: Callable[[], dict[str, np.ndarray]], log: Log,
) -> dict:  # fmt: skip
    """Place D3, D5, and D14 on D8's base calendar, write the table once, and run its gates.

    The table is written once, before the gates, so that a failed gate leaves the table to inspect.
    The gates are the coverage of at least 90% for every neighbour and the timestamp check (the
    correlation of the half-hour changes at lag 0 is the largest of lags -1, 0, and +1). The training
    medians of the folded neighbours go to `neighbours/medians.json`, written once.

    Args:
        root: The data folder of this experiment.
        root730: The data folder of experiment 730 (read only).
        base: D8's base calendar.
        build: Reads the three series from the stored power files (`neighbours.build_arrays`).
        log: Receives progress lines.

    Returns:
        The record of `neighbours/check.json`.

    Raises:
        StoppedError: When a gate fails (verdict row 0).
    """
    guard_run(root=root, root730=root730, stage="neighbours")
    path = root / nb.TABLE_NAME
    if path.exists():
        arrays = nb.read_table(root=root, dates=base.dates)
    else:
        arrays = build()
        nb.write_table(root=root, dates=base.dates, arrays=arrays)
        arrays = nb.read_table(root=root, dates=base.dates)
    shares = nb.coverage(table=arrays, reading=base.reading)
    correlations: dict[str, dict[int, float]] = {}
    problems = nb.coverage_problems(shares=shares)
    for n in nb.NEIGHBOURS:
        try:
            correlations[n] = nb.lag_correlations(reading=base.reading, neighbour=arrays[n])
        except nb.NeighbourError as exc:
            problems.append(f"{n}: the timestamp check could not be read ({exc})")
    problems += nb.alignment_problems(correlations=correlations)
    medians = {
        n: nb.training_median(
            folded=nb.fold(signed=arrays[n], floor=0.0), train_rows=base.train_rows
        )
        for n in nb.NEIGHBOURS
    }
    medians_path = root / MEDIANS_FILE
    if medians_path.exists() and read_json(path=medians_path) != medians:
        msg = "the neighbours' training medians differ from the ones recorded at the first run (verdict row 0)"
        raise StoppedError(msg)
    if not medians_path.exists():
        write_once_json(path=medians_path, payload=medians)
    record = {
        "coverage": shares, "coverage_by_split": split_coverage(base=base, arrays=arrays),
        "lag_correlations": {n: {str(k): v for k, v in c.items()} for n, c in correlations.items()},
        "problems": problems, "pass": not problems,
    }  # fmt: skip
    write_json(path=root / NEIGHBOUR_CHECK, payload=record)
    log("neighbours: coverage " + ", ".join(f"{n} {v:.3f}" for n, v in shares.items()))
    if problems:
        msg = "; ".join(problems) + " (verdict row 0)"
        raise StoppedError(msg)
    return record


def read_neighbour_inputs(
    *, root: Path, base: BaseDays
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Read the stored neighbour table and the training medians (the inputs of every later stage)."""
    arrays = nb.read_table(root=root, dates=base.dates)
    medians = {k: float(v) for k, v in read_json(path=root / MEDIANS_FILE).items()}
    return arrays, medians


# --------------------------------------------------------------------------- limits


def noise_units(*, base: BaseDays, arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Return the four noise units in points, from the training base days alone."""
    scales = baselines.compute_scales(
        reading=base.reading, bsp=base.bsp, neighbours=arrays, dates=base.dates,
        weekend=base.weekend, train_rows=base.train_rows,
    )  # fmt: skip
    return {n: float(scales.units[n]) for n in core.NOISE_UNIT_NAMES}


def stage_limits(
    *, root: Path, root730: Path, holidays: np.ndarray, base: BaseDays,
    arrays: Mapping[str, np.ndarray], log: Log,
) -> dict:  # fmt: skip
    """Write `limits.json` once from the test bank's truth and the training base days alone.

    The function stops when the pooled limit is not experiment 730's, and when an existing
    `limits.json` differs from the limits recomputed from the test bank.
    """
    guard_run(root=root, root730=root730, stage="limits")
    test = banks729.load_bank(path=banks730.bank_path(root=root730, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    limits = core.band_limits(x=rows.x, r=rows.r)
    problems = core.limit_problems(limits=limits)
    if problems:
        msg = "; ".join(problems) + " (verdict row 0)"
        raise StoppedError(msg)
    path = root / "limits.json"
    if path.exists():
        stored = read_json(path=path)
        recomputed = {
            k: {"share": v.share, "limit": v.limit, "n_rows": v.n_rows} for k, v in limits.items()
        }
        if stored["bands"] != recomputed or stored["n_rows"] != rows.n:
            msg = (
                "limits.json differs from the limits recomputed from the test bank (verdict row 0)"
            )
            raise StoppedError(msg)
        return stored
    record = core.write_limits_file(
        path=path, limits=limits, s=float(np.mean(rows.x < 0.0)), n_rows=rows.n,
        noise_units=noise_units(base=base, arrays=arrays),
    )  # fmt: skip
    log("limits written: " + ", ".join(f"{k} {v.limit}" for k, v in limits.items()))
    return record


# --------------------------------------------------------------------------- overlays


def share_json(*, shares: Mapping[str, ov.Share]) -> dict[str, dict[str, float]]:
    """Return component shares as plain dictionaries."""
    return {k: {"value": float(v.value), "n": int(v.n)} for k, v in shares.items()}


def failed_shares(*, shares: Mapping[str, ov.Share], where: str) -> list[str]:
    """Return a line for each share that has no cases or is below `MIN_OUTPUT_SHARE`."""
    return [
        f"{where}: {k} reads {v.value:.4f} on {v.n} cases"
        for k, v in shares.items()
        if not ov.share_passes(share=v)
    ]


def reverse_share(*, segments: Sequence[Any], holidays: np.ndarray) -> float:
    """Return the share of scored half-hours with a negative truth (the sign-supervised half-hours.

    Hidden spans aside).
    """
    return float(
        np.mean(score729.rows_from_segments(segments=list(segments), holidays=holidays).x < 0.0)
    )


def overlay_stage_bank(
    *,
    name: str,
    segments: Sequence[Any],
    kind: str,
    seed: int | None,
    holidays: np.ndarray,
    neighbours: Mapping[str, np.ndarray],
    test: bool,
) -> tuple[dict, list[str]]:
    """Build one bank's overlays and return its record and the problems found (component shares, counts, digests)."""
    overlays = ov.overlay_segments(segments=segments, kind=kind, seed=seed, neighbours=neighbours)
    checks = ov.transfer_checks(overlays=overlays, originals=segments)
    problems = failed_shares(shares=checks, where=name)
    index = ov.build_pair_index(segments=segments)
    unpaired = index.share_unpaired()
    problems += [
        f"{name}: {p} lacks sibling pairs for more than 5% of its windows"
        for p in ov.unpaired_problems(share_unpaired=unpaired)
    ]
    record: dict[str, Any] = {
        "digest": ov.overlay_digest(overlays=overlays), "n_transfers": int(sum(len(o.transfers) for o in overlays)),
        "share_unpaired_by_month_position": [float(unpaired[m]) for m in sorted(unpaired)],
        "reverse_share_with_transfers": reverse_share(segments=[o.seg for o in overlays], holidays=holidays),
        "reverse_share_stored": reverse_share(segments=segments, holidays=holidays),
        "transfer_checks": share_json(shares=checks),
    }  # fmt: skip
    n_bad, n_up = ov.unreliable_counts(segments=segments)
    record["unreliable"] = {"n": n_bad, "of": n_up, "share": n_bad / n_up if n_up else 0.0}
    record["unreliable_digest"] = hashlib.sha256(
        b"".join(ov.unreliable_solar(seg=s).tobytes() for s in segments)
    ).hexdigest()
    by_mask = sum(int(ov.unreliable_solar(seg=s).sum()) for s in segments)
    by_loop = sum(ov.unreliable_by_loop(seg=s) for s in segments)
    record["unreliable"]["all_half_hours_by_mask"] = by_mask
    record["unreliable"]["all_half_hours_by_loop"] = by_loop
    if by_mask != by_loop:
        problems.append(
            f"{name}: the unreliable-solar mask holds {by_mask} half-hours and an independent count holds {by_loop}"
        )
    if n_up and n_bad / n_up > core.MAX_UNRELIABLE_SHARE:
        problems.append(
            f"{name}: {n_bad} of {n_up} S1 sun-up half-hours of solar worlds are unreliable solar targets (more than 7%)"
        )
    if test:
        windows = {}
        index = ov.build_pair_index(segments=segments, sources=core.TEST_SIBLING_SOURCES)
        for a in core.TEST_INJECTION_POINTS:
            pair = ov.pair_checks(segments=segments, index=index, amount=a)
            problems += failed_shares(shares=pair, where=f"{name} pairs at {a:g}")
            record[f"pair_checks_{a:g}"] = share_json(shares=pair)
            windows[a] = sum(
                1
                for (i, d) in index.siblings
                if ov.fixed_sibling(index=index, segment=i, day=d) is not None
            )
        raised = ov.raised_checks(segments=segments)
        problems += failed_shares(shares={"raised": raised}, where=name)
        record["raised"] = share_json(shares={"raised": raised})["raised"]
        reach = ov.reachability(overlays=overlays, index=index, n_windows_by_amount=windows)
        record["reachability"] = {
            k: {"count": v[0], "bar": v[1], "pass": v[2]} for k, v in reach.items()
        }
        record["reachability_short"] = sorted(k for k, v in reach.items() if not v[2])
    return record, problems


def stage_overlays(
    *,
    root: Path,
    root730: Path,
    holidays: np.ndarray,
    neighbours: Mapping[str, np.ndarray],
    log: Log,
    train_seeds: Sequence[int] = range(5),
) -> dict:
    """Build every overlay on the CPU, run the component checks, and write `overlays/summary.json`.

    The stage stops at verdict row 0 when a component check fails or has no cases (the mirror check
    included), a month lacks sibling pairs for more than 5% of its windows, or the training share of
    reverse flow with the transfers differs from the stored share by more than 10% of the stored
    share. A reachability shortfall only voids a word of the verdict (the
    plan says the line does not stop), so the summary lists it under `reachability_short` of the test
    bank and `stage_score` reads the counts again.

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        holidays: Bank-holiday dates.
        neighbours: The neighbour table on D8's base calendar.
        log: Receives progress lines.
        train_seeds: The training seeds to build.

    Returns:
        The summary written to `overlays/summary.json`.

    Raises:
        StoppedError: With every problem listed (counts and names only).
    """
    guard_run(root=root, root730=root730, stage="overlays")
    begin_gate_attempt(root=root)
    leak.assert_seed_spaces_disjoint()
    leak.assert_no_month_or_day_of_year()
    records: dict[str, Any] = {}
    problems: list[str] = []
    jobs = [(f"train_s{s}", "train", s, False) for s in train_seeds] + [
        ("validation", "validation", None, False),
        ("test", "test", None, True),
    ]
    for name, kind, seed, is_test in jobs:
        bank = banks729.load_bank(path=banks730.bank_path(root=root730, kind=kind, seed=seed))
        leak.assert_overlay_depends_only_on_its_segment(
            segments=bank.segments[:6], kind=kind, seed=seed, neighbours=neighbours
        )
        record, found = overlay_stage_bank(
            name=name, segments=bank.segments, kind=kind, seed=seed, holidays=holidays,
            neighbours=neighbours, test=is_test,
        )  # fmt: skip
        records[name], problems = record, problems + found
        log(f"overlay {name}: {record['n_transfers']} transfers, {len(found)} problems")
    stored = float(np.mean([records[f"train_s{s}"]["reverse_share_stored"] for s in train_seeds]))
    withs = float(
        np.mean([records[f"train_s{s}"]["reverse_share_with_transfers"] for s in train_seeds])
    )
    if not stored > 0.0 or abs(withs - stored) > core.TRAIN_REVERSE_SHARE_TOLERANCE * stored:
        problems.append(
            "the training share of reverse flow with the transfers differs from the stored share by more than 10%"
        )
    summary = {
        "banks": records,
        "reverse_share": {"stored": stored, "with_transfers": withs},
        "problems": problems,
        "pass": not problems,
    }
    path = root / "overlays" / "summary.json"
    digests = {k: v["digest"] for k, v in records.items()} | {
        f"{k}/unreliable": v["unreliable_digest"] for k, v in records.items()
    }
    if (
        path.exists()
        and read_json(path=path).get("pass")
        and read_json(path=path)["digests"] != digests
    ):
        msg = "an overlay digest differs from the one recorded at the first run"
        raise leak.CorrectnessFailureError(msg)
    write_json(path=path, payload={**summary, "digests": digests})
    if problems:
        msg = f"{len(problems)} overlay problems (verdict row 0): " + " | ".join(problems[:6])
        raise StoppedError(msg)
    return summary


# --------------------------------------------------------------------------- checks on the input tensors


def mirror_edge_picks(*, prepared: Sequence[arms.Prepared], limit: int) -> list[tuple[int, int]]:
    """Return windows whose scored day holds an edge of a mirrored transfer."""
    picks: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i, p in enumerate(prepared):
        days = {int(d) for d in p.days}
        for tr in p.transfers:
            if not tr.mirrored:
                continue
            for hh in (tr.start_half_hour, tr.end_half_hour):
                key = (i, hh // 48)
                if hh // 48 in days and key not in seen:
                    seen.add(key)
                    picks.append(key)
        if len(picks) >= limit:
            break
    return picks[:limit]


def solar_world_picks(*, prepared: Sequence[arms.Prepared], limit: int) -> list[tuple[int, int]]:
    """Return windows of solar-world segments (the days of the solar and injection checks)."""
    picks = [
        (i, int(d))
        for i, p in enumerate(prepared)
        if p.world in ov.SOLAR_WORLDS
        for d in p.days[:3]
    ]
    return picks[:limit]


def bsp_present_solar_picks(
    *, prepared: Sequence[arms.Prepared], limit: int
) -> list[tuple[int, int]]:
    """Return windows of solar-world segments whose scored day has BSP data and solar output.

    The check that the injected member's BSP channel moves needs such windows, and most segments
    have no BSP meter.

    Args:
        prepared: The prepared segments.
        limit: The most windows to return.

    Returns:
        `(segment index, dense day)` pairs.
    """
    picks: list[tuple[int, int]] = []
    for i, p in enumerate(prepared):
        if p.world not in ov.SOLAR_WORLDS:
            continue
        for d in p.days:
            a = int(d) * 48
            if np.isfinite(p.bsp[a : a + 48]).all() and (p.base.targets[3, a : a + 48] > 0).any():
                picks.append((i, int(d)))
                break
        if len(picks) >= limit:
            break
    return picks


def assert_batch_channels(*, batch: arms.Batch, spec: core.ArmSpec732) -> None:
    """Check a batch's neighbour channels and its injected member with the leakage assertions.

    A neighbour value must be non-negative and a neighbour mask 0 or 1, and w+ must differ from w
    only in the reading, the BSP step, and the sign-aligned changes.

    Args:
        batch: A batch built by `arms.make_batch` or `arms.make_eval_batch`.
        spec: The arm the batch was built for.

    Raises:
        CorrectnessFailureError: When one of those fails (verdict row 1).
    """
    if spec.neighbours:
        leak.assert_neighbours_unsigned_and_masks_binary(batch=batch)
    if spec.inject:
        leak.assert_injection_changes_only_allowed_channels(batch=batch)


def tensor_check_specs() -> dict[str, core.ArmSpec732]:
    """Return the arms whose batches the tensor checks run on: U*, U-N, and U-noBSP."""
    return {name: core.ARMS[name] for name in ("U*", "U-N", "U-noBSP")}


def stage_tensor_checks(
    *, root: Path, root730: Path, env: Env, log: Log, n_edges: int = 80, n_solar: int = 80,
    seed: int = 0,
) -> dict:  # fmt: skip
    """Run the checks on the input tensors on the stored training bank of seed 0.

    The function also checks the validation bank, before any fit.

    The training batches come from `arms.make_batch` (hidden spans on half the windows), and the
    validation batches from `arms.make_eval_batch` with fixed hidden spans (`masks.fixed_hidden_for_picks`),
    a fixed injected amount, and the fixed raised sky. Each of U*, U-N, and U-noBSP is checked. A
    check that finds no case fails.

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        env: The stages' environment.
        log: Receives progress lines.
        n_edges: Windows at mirrored transfer edges per batch.
        n_solar: Windows of solar worlds per batch.
        seed: The seed of the training batches.

    Returns:
        The record written to `scores/tensor_checks.json`.

    Raises:
        StoppedError: With every failed check (verdict row 0).
    """
    guard_run(root=root, root730=root730, stage="tensor")
    begin_gate_attempt(root=root)
    tr = banks729.load_bank(path=banks730.bank_path(root=root730, kind="train", seed=0))
    va = banks729.load_bank(path=banks730.bank_path(root=root730, kind="validation"))
    train = arms.prepare(
        overlays=ov.overlay_segments(segments=tr.segments, kind="train", seed=0, neighbours=env.neighbours),
        holidays=env.holidays, stats=env.stats, medians=env.medians,
    )  # fmt: skip
    val = arms.prepare(
        overlays=ov.overlay_segments(segments=va.segments, kind="validation", neighbours=env.neighbours),
        holidays=env.holidays, stats=env.stats, medians=env.medians,
    )  # fmt: skip
    index_tr, index_va = (
        ov.build_pair_index(segments=tr.segments),
        ov.build_pair_index(segments=va.segments),
    )
    record: dict[str, Any] = {}
    problems: list[str] = []
    picks_tr = (
        mirror_edge_picks(prepared=train, limit=n_edges)
        + solar_world_picks(prepared=train, limit=n_solar)
        + bsp_present_solar_picks(prepared=train, limit=n_solar)
    )
    picks_va = (
        mirror_edge_picks(prepared=val, limit=n_edges)
        + solar_world_picks(prepared=val, limit=n_solar)
        + bsp_present_solar_picks(prepared=val, limit=n_solar)
    )
    hidden = masks.fixed_hidden_for_picks(
        style=core.STYLE, picks=picks_va, offsets=[p.offsets for p in val],
        segment_ids=[p.source_index for p in val],
    )  # fmt: skip
    for name, spec in tensor_check_specs().items():
        rng = np.random.default_rng(seed)
        batches = {
            "train": (
                arms.make_batch(prepared=train, picks=picks_tr, spec=spec, rng=rng, index=index_tr if spec.inject else None),
                train,
            ),
            "validation": (
                arms.make_eval_batch(
                    prepared=val, picks=picks_va, spec=spec, index=index_va, amount=core.TEST_INJECTION_POINTS[1],
                    eps=core.TEST_EPS, hidden=hidden,
                ),
                val,
            ),
        }  # fmt: skip
        for where, (batch, prepared) in batches.items():
            assert_batch_channels(batch=batch, spec=spec)
            checks = arms.tensor_checks(batch=batch, prepared=prepared, spec=spec)
            found = arms.tensor_problems(checks=checks)
            record[f"{name}/{where}"] = share_json(shares=checks)
            problems += [f"{name} on the {where} bank: tensor check {c} failed" for c in found]
            log(f"tensor checks {name} {where}: {len(checks)} checks, {len(found)} failed")
    summary = {"checks": record, "problems": problems, "pass": not problems}
    write_json(path=root / TENSOR_FILE, payload=summary)
    if problems:
        msg = f"{len(problems)} tensor-check failures (verdict row 0): " + " | ".join(problems[:6])
        raise StoppedError(msg)
    return summary

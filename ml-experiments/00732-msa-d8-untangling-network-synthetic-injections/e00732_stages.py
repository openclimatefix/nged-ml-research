# ruff: noqa: E501
"""Experiment 732: the gates and the stages that need no torch (the banks check, limits, overlays, guard).

Every stage takes the data folder `root` of this experiment and experiment 730's data folder
`root730`, which is read and never written. A stage runs `guard_run` first, so a stage on the outputs
of another run, before a gate, or after a correctness failure refuses to start. A gate that fails
raises `StoppedError` (verdict row 0) or `CorrectnessFailureError` (verdict row 1), and the command
line turns either into a status file and a stop. No stage prints a date, a series identifier, or a
value of a real solar era.
"""

import hashlib
import json
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
import e00730_score as score730
import e00732_arms as arms
import e00732_core as core
import e00732_leakage as leak
import e00732_overlays as ov
import numpy as np

Log = Callable[[str], None]
StoppedError = banks730.StoppedError
RUN_MARKER: Final = "run_marker.json"
INVALID_FILE: Final = "scores/invalid.json"
BANKS_CHECK: Final = "banks_check.json"
TEST_MARKER: Final = "test_started"
GATED_STAGES: Final = ("timing", "fit", "chain")
"""The stages that start a fit: they need the limits, the overlays, the gallery check, and no recorded failure."""
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
        real_sets: A function that returns the real-day pseudo-segments by set name (None without real data).
    """

    holidays: np.ndarray
    stats: arms729.InputStats
    real_sets: Callable[[], dict[str, list]] | None = None


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


def invalid_recorded(*, root: Path) -> bool:
    """Return whether a correctness check has failed in this run."""
    return (root / INVALID_FILE).exists()


# --------------------------------------------------------------------------- the run marker and the guard


def run_marker_payload(*, root730: Path) -> dict:
    """Return what the run marker holds: the experiment, the row set, and the id of experiment 730's banks."""
    return {
        "experiment": "00732",
        "row_set_id": core.ROW_SET_ID,
        "banks_digest": banks730.banks_digest_id(root=root730),
    }


def gallery_passed(*, root: Path, root730: Path) -> bool:
    """Return whether `PRIVATE/gallery/panel_check.json` records every panel of the written list as passed on the banks in use."""
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


def overlays_passed(*, root: Path) -> bool:
    """Return whether `overlays/summary.json` records a pass."""
    path = root / "overlays" / "summary.json"
    return path.exists() and bool(read_json(path=path).get("pass"))


def timing_passed(*, root: Path) -> bool:
    """Return whether `scores/timing.json` records a pass."""
    path = root / "scores" / "timing.json"
    return path.exists() and bool(read_json(path=path).get("pass"))


def guard_run(*, root: Path, root730: Path, stage: str) -> None:
    """Refuse to run a stage on the outputs of another run, or without the gates before a fit.

    Args:
        root: The data folder of this experiment.
        root730: The data folder of experiment 730 (read only).
        stage: The stage name.

    Raises:
        RuntimeError: When a gate is open, a correctness check has failed, or the folder holds outputs of another run.
    """
    expected = run_marker_payload(root730=root730)
    marker = root / RUN_MARKER
    current = read_json(path=marker) if marker.exists() else None
    if stage == "banks":
        if current is not None and current != expected:
            msg = "the run marker differs from the code and the banks in use: move the outputs to superseded_run_N yourself"
            raise RuntimeError(msg)
        return
    if current != expected:
        msg = "no run marker matches the banks in use: run `banks` first"
        raise RuntimeError(msg)
    if stage in GATED_STAGES:
        if invalid_recorded(root=root):
            msg = f"`{stage}` starts a fit, but {INVALID_FILE} records a failed correctness check (verdict row 1): the run is over"
            raise RuntimeError(msg)
        if not (root / "limits.json").exists():
            msg = f"`{stage}` needs limits.json, written before any fit"
            raise RuntimeError(msg)
        if not overlays_passed(root=root):
            msg = f"`{stage}` needs overlays/summary.json with a pass"
            raise RuntimeError(msg)
        if not gallery_passed(root=root, root730=root730):
            msg = f"`{stage}` needs PRIVATE/gallery/panel_check.json with every panel passed (the gallery gate of the plan)"
            raise RuntimeError(msg)
    if stage in FIT_STAGES and not timing_passed(root=root):
        msg = f"`{stage}` needs scores/timing.json with a pass: run `timing` first"
        raise RuntimeError(msg)


# --------------------------------------------------------------------------- the banks of experiment 730


def job_name_730(*, arm: str, arch: str, seed: int) -> str:
    """Return the folder name of one of experiment 730's fits (`SD*` would be `SDstar`; no stored control holds an asterisk)."""
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
    name: str,
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


def row_key_digest(*, rows: score729.Rows) -> str:
    """Return the SHA-256 of the row keys (segment, day, and half-hour) of a bank's scored rows."""
    h = hashlib.sha256()
    for a in (rows.seg, rows.date, rows.hh):
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return h.hexdigest()


def seed_mean_p(*, arrays: Sequence[Mapping[str, np.ndarray]]) -> np.ndarray:
    """Return the seed-mean P(X<0) of a stored arm."""
    return score730.seed_mean(p_by_seed=[a["p"].astype(np.float64) for a in arrays])


def stage_banks(*, root: Path, root730: Path, holidays: np.ndarray, log: Log) -> dict:
    """Verify experiment 730's banks and stored files, record their digests once, and write the run marker.

    The stage reads experiment 730's data folder only. It stops at verdict row 0 when a bank digest, the
    stored margin, or a stored prediction file differs from the digest recorded at the first run, and
    fails at verdict row 1 when a stored array has the wrong row count, the row-key digest differs, or
    the Brier canary of S does not reproduce.

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
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    if rows.n != core.N_TEST_ROWS:
        msg = f"the test bank holds {rows.n} scored half-hours, not {core.N_TEST_ROWS}"
        raise leak.CorrectnessFailureError(msg)
    for name, (arm, arch) in CONTROL_FOLDERS.items():
        for k, arrays in enumerate(load_stored(root730=root730, name=name, arm=arm, arch=arch)):
            if any(a.shape[0] != rows.n for a in arrays.values()):
                msg = f"a stored array of {name} seed {k} does not have {rows.n} rows"
                raise leak.CorrectnessFailureError(msg)
    canary = score730.mean_brier(
        p=seed_mean_p(arrays=load_stored(root730=root730, name="S", arm="S", arch="unet")), x=rows.x
    )
    if round(canary, core.BRIER_CANARY_DECIMALS) != round(
        core.BRIER_CANARY, core.BRIER_CANARY_DECIMALS
    ):
        msg = "the Brier canary of S does not reproduce experiment 730's printed value"
        raise leak.CorrectnessFailureError(msg)
    files = {
        f"{name}_s{k}": file_digest(path=p)
        for name, (arm, arch) in CONTROL_FOLDERS.items()
        for k, p in enumerate(stored_predictions(root730=root730, arm=arm, arch=arch))
    }
    margin = read_json(path=root730 / "margin.json")
    if abs(float(margin["margin"]) - core.MARGIN_730) > 5e-6:
        msg = "experiment 730's margin.json does not hold the margin of the plan (verdict row 0)"
        raise StoppedError(msg)
    record = {
        "n_rows": rows.n, "row_key_digest": row_key_digest(rows=rows), "brier_canary_matches": True,
        "predictions": files, "margin_json": file_digest(path=root730 / "margin.json"), "banks_digest": banks730.banks_digest_id(root=root730),
    }  # fmt: skip
    path = root / BANKS_CHECK
    if path.exists():
        first = read_json(path=path)
        if first["row_key_digest"] != record["row_key_digest"]:
            msg = "the row-key digest differs from the one recorded at the first run"
            raise leak.CorrectnessFailureError(msg)
        if (
            first["predictions"] != record["predictions"]
            or first["margin_json"] != record["margin_json"]
        ):
            msg = "experiment 730's stored predictions or margin.json differ from their recorded digests (verdict row 0)"
            raise StoppedError(msg)
    else:
        write_once_json(path=path, payload=record)
    marker = root / RUN_MARKER
    if not marker.exists():
        write_json(path=marker, payload=run_marker_payload(root730=root730))
    log(f"banks verified: {rows.n} rows, {len(files)} stored files, canary reproduced")
    return record


# --------------------------------------------------------------------------- limits


def stage_limits(*, root: Path, root730: Path, holidays: np.ndarray, log: Log) -> dict:
    """Write `limits.json` once from the test bank's truth alone, and stop when the pooled limit is not experiment 730's."""
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
        return read_json(path=path)
    record = core.write_limits_file(
        path=path, limits=limits, s=float(np.mean(rows.x < 0.0)), n_rows=rows.n
    )
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
    """Return the share of scored half-hours with a negative truth (the sign-supervised half-hours, hidden spans aside)."""
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
    test: bool,
) -> tuple[dict, list[str]]:
    """Build one bank's overlays and return its record and the problems found (component shares, counts, digests)."""
    overlays = ov.overlay_segments(segments=segments, kind=kind, seed=seed)
    problems = failed_shares(
        shares=ov.transfer_checks(overlays=overlays, originals=segments), where=name
    )
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
    }  # fmt: skip
    if test:
        windows = {}
        for a in core.TEST_INJECTION_POINTS:
            checks = ov.pair_checks(segments=segments, index=index, amount=a)
            problems += failed_shares(shares=checks, where=f"{name} pairs at {a:g}")
            record[f"pair_checks_{a:g}"] = share_json(shares=checks)
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
        problems += [
            f"{name}: reachability {k} holds {v[0]} against {v[1]}"
            for k, v in reach.items()
            if not v[2]
        ]
    record["transfer_checks"] = share_json(
        shares=ov.transfer_checks(overlays=overlays, originals=segments)
    )
    return record, problems


def stage_overlays(
    *,
    root: Path,
    root730: Path,
    holidays: np.ndarray,
    log: Log,
    train_seeds: Sequence[int] = range(5),
) -> dict:
    """Build every overlay on the CPU, run the component checks, and write `overlays/summary.json`.

    The stage stops at verdict row 0 when a component check fails or has no cases, a month lacks
    sibling pairs for more than 5% of its windows, a reachability count is short, or the training share
    of reverse flow with the transfers differs from the stored share by more than 10% of the stored share.

    Raises:
        StoppedError: With every problem listed (counts and names only).
    """
    guard_run(root=root, root730=root730, stage="overlays")
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
            segments=bank.segments[:6], kind=kind, seed=seed
        )
        record, found = overlay_stage_bank(
            name=name, segments=bank.segments, kind=kind, seed=seed, holidays=holidays, test=is_test
        )
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
    digests = {k: v["digest"] for k, v in records.items()}
    if path.exists() and read_json(path=path)["digests"] != digests:
        msg = "an overlay digest differs from the one recorded at the first run"
        raise leak.CorrectnessFailureError(msg)
    write_json(path=path, payload={**summary, "digests": digests})
    if problems:
        msg = f"{len(problems)} overlay problems (verdict row 0): " + " | ".join(problems[:6])
        raise StoppedError(msg)
    return summary

# ruff: noqa: E501
"""Experiment 733: the sign forecasts of the synthetic world, joined to the truth (no torch).

The reliability figure and the series stage read the same arrays: YU's, Y0's, and B-hist's
probability of reverse flow on every sample of the test bank (the validation bank on validation
rows), with the true sign. A segment's solar source (S1 or S2) is not stored in the world file, so
`segment_sources` recovers it from the bank and fails when the bank and the world disagree.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00733_paths as paths
import e00733_scoring_synth as ss
import e00733_synth as sy
import e00733_world as world
import numpy as np

SIGN_ARMS: Final = ("YU", "Y0")
"""The synthetic arms whose probability of reverse flow is drawn (YT is a report-only arm)."""


class SignDataError(RuntimeError):
    """The stored forecasts or the bank cannot be joined to the world's samples."""


@dataclass
class SignData:
    """The scored samples of the synthetic world with every forecast of the sign.

    Attributes:
        segments: The world's segments.
        sources: Each segment's solar source ("S1", "S2", or "none").
        index: The samples (segment, origin day, lead index), one per target day.
        truth: `(N, 48)` the indicator of X < 0, NaN where no sign exists.
        probabilities: `(N, 48)` per forecast: "YU", "Y0", "B-hist", "B-clim".
        keep: `(N, 48)` whether the half-hour is scored (lead days 2 to 10 with a sign).
        seeds: The seeds held for each arm.
    """

    segments: list[Any]
    sources: list[str]
    index: sy.SampleIndex
    truth: np.ndarray
    probabilities: dict[str, np.ndarray]
    keep: np.ndarray
    seeds: dict[str, tuple[int, ...]]


def segment_sources(*, root: Path, kind: str, segments: list[Any]) -> list[str]:
    """Return the solar source of each world segment, read from experiment 730's bank.

    A world segment is a bank segment with a covered origin, in the bank's order. The source of
    each kept bank segment is checked against the world by comparing the injected solar output.

    Args:
        root: This experiment's data folder.
        kind: "test" or "validation".
        segments: The world's segments of that bank.

    Returns:
        One of "S1", "S2", or "none" per world segment.

    Raises:
        SignDataError: When the bank's kept segments do not match the world's segments.
    """
    import e00729_banks as b729  # ty: ignore[unresolved-import]
    import e00730_banks as b730  # ty: ignore[unresolved-import]

    del root
    bank = b729.load_bank(path=b730.bank_path(root=paths.root_730(), kind=kind))
    run_days = _run_days()
    kept = []
    for seg in bank.segments:
        dates = np.asarray(seg.dates).astype("datetime64[D]").astype(np.int64)
        if (np.asarray(seg.cal_index) >= 0)[np.isin(dates, run_days)].any():
            kept.append(seg)
    if len(kept) != len(segments):
        msg = f"the {kind} bank keeps {len(kept)} covered segments, the world holds {len(segments)}"
        raise SignDataError(msg)
    for i, (b, w) in enumerate(zip(kept, segments, strict=True)):
        if not np.array_equal(np.asarray(b.solar, dtype=np.float32).reshape(-1), w.solar_points):
            msg = f"segment {i} of the {kind} world differs from the bank's segment in its solar output"
            raise SignDataError(msg)
    return [str(b.source) for b in kept]


def _run_days() -> np.ndarray:
    """Return the epoch days that have a 00Z ensemble run."""
    run_days_dt = sy.summary_run_days(summary_dirs=paths.ens_summary_dirs())
    return np.asarray(
        sorted({int(np.datetime64(d, "D").astype("int64")) for d in run_days_dt}), dtype=np.int64
    )


def load_sign_data(*, root: Path, rows: str) -> SignData:
    """Read every arm's forecast of the sign on the scored samples.

    Args:
        root: This experiment's data folder.
        rows: "test" or "validation".

    Returns:
        The joined data.

    Raises:
        SignDataError: When YU or Y0 holds no archive, or an archive holds another sample count.
    """
    import e00733_adapters as adapters

    kind = "validation" if rows == "validation" else "test"
    segments, _ = world.read_world(root=root, kind=kind)
    index = sy.enumerate_samples(segments=segments)
    seeds, arms = {}, {}
    for arm in SIGN_ARMS:
        seeds[arm] = ss.seeds_held(root=root, arm=arm, rows=rows)
        got = ss.read_arm(root=root, arm=arm, seeds=seeds[arm], rows=rows) if seeds[arm] else None
        if got is None:
            msg = (
                f"no stored forecast of {arm} on the {rows} rows, so no sign forecast can be drawn"
            )
            raise SignDataError(msg)
        if len(got["seg"]) != len(index):
            msg = f"the archive of {arm} holds {len(got['seg'])} samples, the bank {len(index)}"
            raise SignDataError(msg)
        arms[arm] = got
    meta = world.read_meta(root=root)["constants"]
    base = ss.baseline_probabilities(
        segments=segments, index=index, clim=ss.clim_from_limits(root=root),
        d8_index=adapters.d8_index(), target_std=float(meta["std"]),
    )  # fmt: skip
    truth = arms["YU"]["sign_truth"]
    keep = ss.scored_mask(lead_index=index.lead_index)[:, None] & np.isfinite(truth)
    if not keep.any():
        msg = "no half-hour of the synthetic bank is scored"
        raise SignDataError(msg)
    probabilities = {
        "YU": arms["YU"]["p_neg"], "Y0": arms["Y0"]["p_neg"],
        "B-hist": base["b_hist"], "B-clim": base["b_clim"],
    }  # fmt: skip
    return SignData(
        segments=segments, sources=segment_sources(root=root, kind=kind, segments=segments),
        index=index, truth=truth, probabilities=probabilities, keep=keep, seeds=seeds,
    )  # fmt: skip

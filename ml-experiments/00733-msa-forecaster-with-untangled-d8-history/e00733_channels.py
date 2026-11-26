# ruff: noqa: E501
"""Experiment 733: the untangled channels of D8 on the forecaster's history grid.

This module applies the rule of "Which untangled signals enter", reads experiment 732's stored
real-day outputs (or experiment 730's, by the rule), averages them over seeds, places them on the
grid of experiments 422, 713, and 716, and builds the placebo that reads each channel 28 days
earlier. A channel the rule drops is zero in every token. The table keeps the solar output in
points (1% of D8's effective capacity); the fold's standardisation turns it into the input `v` in
`e00733_inputs`, because the standard deviation of the target differs between folds.

No function prints a date or a value of a real untangled channel.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import e00733_core as core
import e00733_paths as paths  # noqa: F401
import numpy as np

# isort: split
from e00730_leakage import CorrectnessFailureError  # verdict row 1 (one class for every guard)

STORED_SET: Final = "real_eras"
"""The stored real-day set that holds every era day of D8 (the days the forecaster's histories read)."""
JOB_732: Final = "Ustar_unet_s{seed}"
JOB_730: Final = "S_unet_s{seed}"
GRID_START_EPOCH_DAY: Final = (date.fromisoformat(core.GRID_START_DATE) - date(1970, 1, 1)).days
"""The epoch day of the grid's first day."""
POINTS_PER_CAPACITY: Final = 100.0
"""One point is 1% of D8's effective capacity."""


class StoppedError(RuntimeError):
    """The line stops at verdict row 0: a stop reason of the plan holds."""


# --------------------------------------------------------------------------- the rule


@dataclass(frozen=True)
class Reading732:
    """What the rule reads of experiment 732's stored score files.

    Attributes:
        row: Experiment 732's verdict row.
        solar_word: Its solar word ("reasonable", "partly", "not shown", or "void").
        transfer_word: Its transfer word.
        checks: Whether each real-day check passed (`h2`, `k2`, `p1`, `p3`); None when unread.
        chain_complete: Whether experiment 732's chain status is "complete".
    """

    row: int
    solar_word: str
    transfer_word: str
    checks: Mapping[str, bool | None]
    chain_complete: bool


@dataclass(frozen=True)
class ChannelChoice:
    """The rule's choice, written to `channels.json` before any fit.

    Attributes:
        p_neg_source: "U*" (experiment 732) or "S" (experiment 730).
        usable: The channels that enter, in `core.CHANNEL_NAMES` order.
        dropped: The channels set to zero.
        reasons: The reason of each channel's status, in words with no value.
        fu_equals_fs: P(X<0) is the only usable channel, so FU and FS are one arm.
    """

    p_neg_source: str
    usable: tuple[str, ...]
    dropped: tuple[str, ...]
    reasons: Mapping[str, str]
    fu_equals_fs: bool

    def as_dict(self) -> dict[str, object]:
        """Return the JSON content."""
        return {
            "p_neg_source": self.p_neg_source, "usable": list(self.usable),
            "dropped": list(self.dropped), "reasons": dict(self.reasons),
            "fu_equals_fs": self.fu_equals_fs,
        }  # fmt: skip


def _passed(record: object) -> bool | None:
    """Return whether a stored check record passed: its `pass` flag, or its `status` word."""
    if not isinstance(record, dict):
        return None
    if "pass" in record:
        return bool(record["pass"])
    if "status" in record:
        return record["status"] == "pass"
    return None


def read_reading_732(*, root: Path) -> Reading732 | None:
    """Return the rule's reading of experiment 732's stored files, or None when `verdict.json` is missing.

    The checks come from `scores/real.json`: `h2`, `k2`, `p1` (its `all` part), and `p3`. A check
    that the file does not hold is None, and the rule treats None as not passed.

    Args:
        root: Experiment 732's data folder (read only).

    Returns:
        The reading, or None when the verdict has not been written.
    """
    verdict_path = root / "scores" / "verdict.json"
    if not verdict_path.exists():
        return None
    verdict = json.loads(verdict_path.read_text())
    real_path = root / "scores" / "real.json"
    real = json.loads(real_path.read_text()) if real_path.exists() else {}
    status_path = root / "chain_status.json"
    complete = (
        status_path.exists() and json.loads(status_path.read_text()).get("state") == "complete"
    )
    p1 = real.get("p1")
    words = verdict.get("words", {})
    return Reading732(
        row=int(verdict["row"]),
        solar_word=str(words.get("solar", "void")),
        transfer_word=str(words.get("transfer", "void")),
        checks={
            "h2": _passed(real.get("h2")), "k2": _passed(real.get("k2")),
            "p1": _passed(p1.get("all") if isinstance(p1, dict) else None),
            "p3": _passed(real.get("p3")),
        },
        chain_complete=complete,
    )  # fmt: skip


def decide(*, reading: Reading732 | None) -> ChannelChoice:
    """Return the rule's choice from experiment 732's reading.

    P(X<0) comes from U* when experiment 732's verdict row is 4 or 5 and its checks H2 and K2 pass,
    and from experiment 730's S otherwise. The solar output enters when the P(X<0) source is U*, the
    solar word is "reasonable" or "partly", and P1 and P3 pass. The transfer-step probability enters
    when the P(X<0) source is U* and the transfer word is "works" or "works only in Brier score".

    Args:
        reading: The reading of experiment 732's files.

    Returns:
        The choice.

    Raises:
        StoppedError: When experiment 732 has not finished (verdict row 0).
    """
    if reading is None or not reading.chain_complete:
        msg = "experiment 732 has not finished, so the rule cannot read its verdict (verdict row 0)"
        raise StoppedError(msg)
    reasons: dict[str, str] = {}
    checks = reading.checks
    p_neg_ok = reading.row in core.P_NEG_USABLE_ROWS_732 and all(
        checks.get(c) is True for c in core.P_NEG_REQUIRED_CHECKS
    )
    if p_neg_ok:
        reasons["p_neg"] = "experiment 732's verdict row and real-day checks H2 and K2 allow U*"
    else:
        failing = [c for c in core.P_NEG_REQUIRED_CHECKS if checks.get(c) is not True]
        reasons["p_neg"] = (
            f"experiment 730's S is the source: verdict row of experiment 732 usable "
            f"{reading.row in core.P_NEG_USABLE_ROWS_732}, checks not passed {failing}"
        )
    solar_ok = (
        p_neg_ok
        and reading.solar_word in core.SOLAR_USABLE_WORDS
        and all(checks.get(c) is True for c in core.SOLAR_REQUIRED_CHECKS)
    )
    transfer_ok = p_neg_ok and reading.transfer_word in core.TRANSFER_USABLE_WORDS
    reasons["solar"] = (
        "usable"
        if solar_ok
        else "dropped: the P(X<0) source is S, the solar word, or P1 or P3 fails"
    )
    reasons["step_p"] = (
        "usable" if transfer_ok else "dropped: the P(X<0) source is S or the transfer word fails"
    )
    usable = tuple(
        n for n, ok in zip(core.CHANNEL_NAMES, (True, solar_ok, transfer_ok), strict=True) if ok
    )
    return ChannelChoice(
        p_neg_source=core.SOURCE_732 if p_neg_ok else core.SOURCE_730,
        usable=usable,
        dropped=tuple(n for n in core.CHANNEL_NAMES if n not in usable),
        reasons=reasons,
        fu_equals_fs=usable == ("p_neg",),
    )


def write_choice(*, path: Path, choice: ChannelChoice) -> None:
    """Write `channels.json` once; refuse to overwrite a different choice."""
    payload = json.dumps(choice.as_dict(), indent=1, sort_keys=True)
    if path.exists():
        if path.read_text() != payload:
            msg = "channels.json exists with a different choice; the rule is written once, before any fit"
            raise CorrectnessFailureError(msg)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def read_choice(*, path: Path) -> ChannelChoice:
    """Read `channels.json`."""
    d = json.loads(path.read_text())
    return ChannelChoice(
        p_neg_source=d["p_neg_source"], usable=tuple(d["usable"]), dropped=tuple(d["dropped"]),
        reasons=d["reasons"], fu_equals_fs=bool(d["fu_equals_fs"]),
    )  # fmt: skip


# --------------------------------------------------------------------------- stored outputs


@dataclass(frozen=True)
class StoredOutputs:
    """The seed-averaged real-day outputs on the stored rows.

    Attributes:
        day: The epoch day (days since 1970-01-01) of each row.
        hh: The half-hour (0 to 47, the slot that starts at `30 hh` minutes) of each row.
        values: `(n, 3)` float32: P(X<0), the solar output in points, and the step probability.
    """

    day: np.ndarray
    hh: np.ndarray
    values: np.ndarray


def grid_index(*, day: np.ndarray, hh: np.ndarray) -> np.ndarray:
    """Return the grid index of half-hours.

    The stamp is the period end, so slot `hh` is `hh + 1` steps after midnight.
    """
    d = np.asarray(day, dtype=np.int64) - GRID_START_EPOCH_DAY
    return d * core.STEPS_PER_DAY + np.asarray(hh, dtype=np.int64) + 1


def _read_npz(*, path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def mean_over_seeds(*, arrays: Sequence[Mapping[str, np.ndarray]], key: str) -> np.ndarray:
    """Return the mean of one stored output over seeds, in float64.

    Args:
        arrays: The stored arrays of each seed.
        key: The name of the stored output.

    Returns:
        The mean over seeds.

    Raises:
        CorrectnessFailureError: When no seed is given or a seed lacks the key or has another length.
    """
    if not arrays:
        msg = "no seed was read, so no seed average exists"
        raise CorrectnessFailureError(msg)
    if any(key not in a or len(a[key]) != len(arrays[0][key]) for a in arrays):
        msg = f"a seed lacks the stored output '{key}' or holds another number of rows"
        raise CorrectnessFailureError(msg)
    return np.mean([np.asarray(a[key], dtype=np.float64) for a in arrays], axis=0)


def load_outputs(
    *,
    choice: ChannelChoice,
    root_732: Path,
    root_730: Path,
    seeds: Sequence[int] = core.SOURCE_SEEDS,
    eras_keys: tuple[np.ndarray, np.ndarray] | None = None,
) -> StoredOutputs:
    """Read the stored real-day outputs of the rule's source.

    Averaged over seeds, with dropped channels at zero.

    Args:
        choice: The rule's choice.
        root_732: Experiment 732's data folder.
        root_730: Experiment 730's data folder.
        seeds: The seeds to average.
        eras_keys: The (epoch day, half-hour) of the era rows, from experiment 729's pseudo-segments,
            which fixes the order of experiment 730's rows (its files hold no row keys).

    Returns:
        The outputs. The solar and step columns are zero when the rule drops them.

    Raises:
        CorrectnessFailureError: When a stored file is missing or empty, when the three seeds
            hold other rows, or when a usable channel is all zero (a silent component).
    """
    from_732 = choice.p_neg_source == core.SOURCE_732
    folder, template = (root_732, JOB_732) if from_732 else (root_730, JOB_730)
    files = [folder / "predictions" / template.format(seed=s) / f"{STORED_SET}.npz" for s in seeds]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        msg = (
            f"{len(missing)} stored real-day files of the source {choice.p_neg_source} are missing"
        )
        raise CorrectnessFailureError(msg)
    arrays = [_read_npz(path=p) for p in files]
    p_neg = mean_over_seeds(arrays=arrays, key="p")
    n = len(p_neg)
    keys_732 = root_732 / "predictions" / JOB_732.format(seed=seeds[0]) / f"{STORED_SET}.npz"
    if from_732:
        day, hh = arrays[0]["row_date"].astype(np.int64), arrays[0]["row_hh"].astype(np.int64)
        for a in arrays[1:]:
            if not (
                np.array_equal(a["row_date"], arrays[0]["row_date"])
                and np.array_equal(a["row_hh"], arrays[0]["row_hh"])
            ):
                msg = "two seeds of the source hold other (date, half-hour) rows"
                raise CorrectnessFailureError(msg)
    elif eras_keys is not None:
        day, hh = (np.asarray(k, dtype=np.int64) for k in eras_keys)
        if len(day) != n:
            msg = "experiment 730's stored rows and the era rows of experiment 729 differ in number"
            raise CorrectnessFailureError(msg)
    elif keys_732.exists():
        other = _read_npz(path=keys_732)
        day, hh = other["row_date"].astype(np.int64), other["row_hh"].astype(np.int64)
        if len(day) != n:
            msg = "experiment 730's stored rows and experiment 732's differ in number"
            raise CorrectnessFailureError(msg)
    else:
        msg = "experiment 730's stored rows hold no keys, and no era keys were given"
        raise CorrectnessFailureError(msg)
    solar = mean_over_seeds(arrays=arrays, key="pv") if "solar" in choice.usable else np.zeros(n)
    step = (
        mean_over_seeds(arrays=arrays, key="step_p") if "step_p" in choice.usable else np.zeros(n)
    )
    values = np.stack([p_neg, solar, step], axis=1).astype(np.float32)
    assert_channels_not_silent(values=values, usable=choice.usable)
    return StoredOutputs(day=day, hh=hh, values=values)


def assert_channels_not_silent(*, values: np.ndarray, usable: Sequence[str]) -> None:
    """Fail when a usable channel is zero everywhere, non-finite, or out of range (a silent or broken component).

    The check counts the rows where each usable channel is above a small threshold and fails on an
    empty count, so it cannot pass on an empty counter.

    Args:
        values: The channel table `(n, 3)`.
        usable: The channels that the rule lets in.

    Raises:
        CorrectnessFailureError: When a usable channel is silent, non-finite, or out of its range.
    """
    if values.shape[0] == 0:
        msg = "the channel table holds no rows, so no channel can be checked"
        raise CorrectnessFailureError(msg)
    if not np.isfinite(values).all():
        msg = "a stored untangled channel holds a non-finite value"
        raise CorrectnessFailureError(msg)
    for name in core.CHANNEL_NAMES:
        col = values[:, core.CHANNEL_NAMES.index(name)]
        if name not in usable:
            if (col != 0.0).any():
                msg = f"the dropped channel {name} is not zero"
                raise CorrectnessFailureError(msg)
            continue
        if int((col > 1e-3).sum()) == 0:
            msg = f"the usable channel {name} is silent: no row is above 0.001"
            raise CorrectnessFailureError(msg)
        if name != "solar" and (col.min() < 0.0 or col.max() > 1.0):
            msg = f"the channel {name} leaves 0 to 1"
            raise CorrectnessFailureError(msg)
        if name == "solar" and col.min() < 0.0:
            msg = "the solar channel is negative"
            raise CorrectnessFailureError(msg)


def eras_row_keys(
    *, segments: Sequence[object], calendar_dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (epoch day, half-hour) of every scored half-hour of experiment 729's real pseudo-segments.

    The order is experiment 729's: segment by segment, day by day, then the half-hours whose reading
    is valid and whose truth is finite, which is the order of the stored rows.

    Args:
        segments: The pseudo-segments of the "eras" set of experiment 729.
        calendar_dates: The dates of the day table, `datetime64[D]`.

    Returns:
        The epoch day and the half-hour of each row.
    """
    days, hhs = [], []
    epoch = np.asarray(calendar_dates).astype("datetime64[D]").astype(np.int64)
    for seg in segments:
        for d in np.flatnonzero(seg.cal_index >= 0):  # ty: ignore[unresolved-attribute]
            ok = seg.valid[d] & np.isfinite(seg.x[d])  # ty: ignore[unresolved-attribute]
            days.append(np.full(int(ok.sum()), epoch[int(seg.cal_index[d])], dtype=np.int64))  # ty: ignore[unresolved-attribute]
            hhs.append(np.arange(core.STEPS_PER_DAY, dtype=np.int64)[ok])
    return np.concatenate(days), np.concatenate(hhs)


# --------------------------------------------------------------------------- the table on the grid


@dataclass(frozen=True)
class ChannelTable:
    """The untangled channels at the grid indices that hold a stored output.

    Attributes:
        grid_index: Sorted, unique grid indices `(n,)`.
        values: `(n, 3)` float32: P(X<0), the solar output in points, and the step probability.
    """

    grid_index: np.ndarray
    values: np.ndarray

    def dense(self, *, n_grid: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the values `(n_grid, 3)` and the presence flag `(n_grid,)` on the whole grid."""
        if len(self.grid_index) and (self.grid_index.min() < 0 or self.grid_index.max() >= n_grid):
            msg = "a stored half-hour lies outside the grid"
            raise CorrectnessFailureError(msg)
        out = np.zeros((n_grid, core.N_CHANNELS), dtype=np.float32)
        ok = np.zeros(n_grid, dtype=bool)
        out[self.grid_index] = self.values
        ok[self.grid_index] = True
        return out, ok


def build_table(*, outputs: StoredOutputs) -> ChannelTable:
    """Return the table of the stored outputs, sorted by grid index.

    Args:
        outputs: The seed-averaged stored outputs.

    Returns:
        The table.

    Raises:
        CorrectnessFailureError: When two rows share a grid index.
    """
    gi = grid_index(day=outputs.day, hh=outputs.hh)
    order = np.argsort(gi, kind="stable")
    gi, values = gi[order], outputs.values[order]
    if len(np.unique(gi)) != len(gi):
        msg = "two stored rows share a grid index"
        raise CorrectnessFailureError(msg)
    return ChannelTable(grid_index=gi, values=values)


def write_table(*, table: ChannelTable, path: Path) -> None:
    """Write the table as parquet (`grid_index` and the three channels), atomically."""
    import polars as pl

    frame = pl.DataFrame(
        {
            "grid_index": table.grid_index,
            **{n: table.values[:, i] for i, n in enumerate(core.CHANNEL_NAMES)},
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def read_table(*, path: Path) -> ChannelTable:
    """Read the table written by `write_table`."""
    import polars as pl

    frame = pl.read_parquet(path)
    return ChannelTable(
        grid_index=frame["grid_index"].to_numpy().astype(np.int64),
        values=np.stack([frame[n].to_numpy() for n in core.CHANNEL_NAMES], axis=1).astype(
            np.float32
        ),
    )


def select_channels(
    *, values: np.ndarray, ok: np.ndarray, names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Zero every channel not in `names` and switch the flag off where no named channel remains.

    A dropped channel is zero in every token. When `names` is empty the flag is off everywhere.
    """
    keep = np.array([n in names for n in core.CHANNEL_NAMES])
    out = np.where(keep[None, :], values, 0.0).astype(np.float32)
    return out, ok & bool(keep.any())


# --------------------------------------------------------------------------- the placebo


def placebo(
    *,
    values: np.ndarray,
    ok: np.ndarray,
    shift_steps: int = core.PLACEBO_SHIFT_DAYS * core.STEPS_PER_DAY,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the table read `shift_steps` earlier.

    The value at grid index `g` is the stored value at `g - shift_steps`.

    A grid index with no stored value `shift_steps` earlier has its flag off (an absent flag), so the
    placebo never fills a token from a half-hour after the history token.

    Args:
        values: The stored values `(n_grid, 3)`.
        ok: The presence flag `(n_grid,)`.
        shift_steps: The shift in half-hours (at least 28 days).

    Returns:
        The shifted values and flags.

    Raises:
        CorrectnessFailureError: When the shift is under 28 days.
    """
    if shift_steps < core.PLACEBO_SHIFT_DAYS * core.STEPS_PER_DAY:
        msg = (
            "the placebo's values come from half-hours less than 28 days before the token they fill"
        )
        raise CorrectnessFailureError(msg)
    out = np.zeros_like(values)
    flag = np.zeros_like(ok)
    out[shift_steps:] = values[:-shift_steps]
    flag[shift_steps:] = ok[:-shift_steps]
    return out, flag


def absent_share(*, flag: np.ndarray, visible: np.ndarray) -> float:
    """Return the share of visible half-hours whose flag is off (the absent flags a C2 reading prints)."""
    n = int(visible.sum())
    if n == 0:
        msg = "no visible half-hour, so the absent share cannot be counted"
        raise CorrectnessFailureError(msg)
    return float((visible & ~flag).sum() / n)


# --------------------------------------------------------------------------- coverage


def visible_history_mask(*, origin_days: np.ndarray, reading_ok: np.ndarray) -> np.ndarray:
    """Return, per origin day, the `(n_origin, 14, 48)` mask of visible history half-hours that hold a D8 reading.

    A half-hour is visible when its grid index is at most `c - 54` with `c = 48 D + 24` for the
    origin's UTC day index D on the grid. A history row of age `a` holds the 48 stamps `c - 48 a + j`.

    Args:
        origin_days: The grid day index of each origin (the grid index of 00:00 UTC divided by 48).
        reading_ok: Whether D8's cleaned reading exists at each grid index `(n_grid,)`.

    Returns:
        The mask.
    """
    c = np.asarray(origin_days, dtype=np.int64) * core.STEPS_PER_DAY + 24
    ages = np.arange(core.MIN_AGE, core.MAX_AGE + 1)
    gi = (
        c[:, None, None]
        - core.STEPS_PER_DAY * ages[None, :, None]
        + np.arange(core.STEPS_PER_DAY)[None, None, :]
    )
    in_grid = (gi >= 0) & (gi < len(reading_ok))
    visible = gi <= (c - core.NEWEST_VISIBLE_BEFORE_CONTRACT)[:, None, None]
    present = np.zeros(gi.shape, dtype=bool)
    present[in_grid] = reading_ok[gi[in_grid]]
    return visible & present


def coverage(*, origin_days: np.ndarray, reading_ok: np.ndarray, channel_ok: np.ndarray) -> float:
    """Return the share of visible history half-hours with a D8 reading that hold an untangled channel.

    Args:
        origin_days: The grid day index of each origin.
        reading_ok: Whether D8's cleaned reading exists at each grid index.
        channel_ok: Whether an untangled channel exists at each grid index.

    Returns:
        The share.

    Raises:
        CorrectnessFailureError: When no visible half-hour holds a reading (an empty denominator).
    """
    visible = visible_history_mask(origin_days=origin_days, reading_ok=reading_ok)
    n = int(visible.sum())
    if n == 0:
        msg = "no visible history half-hour holds a reading, so the coverage cannot be counted"
        raise CorrectnessFailureError(msg)
    c = np.asarray(origin_days, dtype=np.int64) * core.STEPS_PER_DAY + 24
    ages = np.arange(core.MIN_AGE, core.MAX_AGE + 1)
    gi = (
        c[:, None, None]
        - core.STEPS_PER_DAY * ages[None, :, None]
        + np.arange(core.STEPS_PER_DAY)[None, None, :]
    )
    have = np.zeros(gi.shape, dtype=bool)
    inside = (gi >= 0) & (gi < len(channel_ok))
    have[inside] = channel_ok[gi[inside]]
    return float((visible & have).sum() / n)


def assert_coverage(*, coverages: Mapping[str, float]) -> None:
    """Stop the line (verdict row 0) when any fold's coverage is under `COVERAGE_MIN` or not a number.

    Args:
        coverages: The coverage share of each fold.

    Raises:
        StoppedError: When the mapping is empty, a share is not finite, or a share is under 0.90.
    """
    if not coverages:
        msg = "no coverage was counted"
        raise StoppedError(msg)
    low = [k for k, v in coverages.items() if not math.isfinite(v) or v < core.COVERAGE_MIN]
    if low:
        msg = f"the channel table covers under {core.COVERAGE_MIN:.0%} of the visible history half-hours in {len(low)} folds (verdict row 0)"
        raise StoppedError(msg)

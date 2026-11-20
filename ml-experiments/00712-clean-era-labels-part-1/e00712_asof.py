"""Experiment 712, stage 1: the check schedule and the as-of label table of every arm.

**This module makes no call.** It turns a candidate's answers into the per-day signal table of
experiment 709 (`e00709_signal.Signal`), so that `era_block`, `hold_mask`, and `row_weight_factor`
read the new arms exactly as they read 709's arms. The table of day `d` is the state at the end of
day `d`, and the origin of day `D` reads the entry of day `D - 1`, so a label made on check day `d`
reaches the network at the origin of day `d + 1` with no new mechanism (design section 2.4).

**The schedule.** A candidate is checked on its alarm day, 7 days later, and 21 days later (the
real record), and on the day its closure becomes known (`Candidate.closure_day`). Two consecutive
checks that give the same class at confidence 0.8 or more cancel the scheduled checks that remain,
and the closure check still runs while the state is `era` or `unknown`. A scheduled check that falls
on or after the closure day is dropped, because the closure check replaces it. `next_check` reads
only the answers already made and the facts known on the check day.

**The state of a candidate.** The latest answer sets the state: `era`, `switch`, or `noise` at
confidence 0.8 or more gives `era`, `switch`, or `none`, and anything else gives `unknown`. A
candidate classed `era` stays `era` through a later answer that is `undecided` or under 0.8, and
only a later `switch` or `noise` at 0.8 or more overturns it (design section 2.4, "the language
model, not the detector, decides whether an era ends"). Pass `sticky_era=False` for the literal
"latest answer" reading.

**The columns.** The site state of a day is the state of the open candidate with the latest alarm,
else `era` for the newest standing era, else `none`. The change vector, anchor, depth, age,
`withdrawn`, `era_id`, `onset_day`, and `confirm_day` follow `e00709_classify.compose_site_signal`,
with `first_era` of an episode row the first day the candidate is classed `era`; its `end` is -1
while the candidate is `era` or `unknown`, and the detector's end day otherwise, raised to the day
a check overturned an `era` so that `hold_mask` sees the withdrawal inside the span.
"""

import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00709_bocpd as bocpd  # ty: ignore[unresolved-import]
import e00709_features as features  # ty: ignore[unresolved-import]
import e00709_statdetect as statdetect  # ty: ignore[unresolved-import]
import e00709_stats as stats  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00712_prompt import (
    DEFAULT_PUBLICATION_LAG_DAYS,
    REAL_OFFSETS,
    SCHEDULED_KINDS,
    Candidate,
    Check,
    RegisterRow,
)

ABSENT: Final[int] = -1
STATE_NONE: Final[int] = 0
STATE_UNKNOWN: Final[int] = 1
STATE_ERA: Final[int] = 2
STATE_SWITCH: Final[int] = 3
"""The state codes of experiment 709 (`e00709_hmm`); `ABSENT` marks days before an alarm."""
CONFIDENCE_RULE: Final[float] = 0.8
"""A class at or above this confidence sets the state; anything else gives `unknown`."""
SETTLING_DAYS: Final[int] = 28
"""The change vector updates for this many days after the first `era` day, then freezes."""
AGE_CAP_DAYS: Final[int] = 365
WATCH_DAYS_AFTER_CONNECTION: Final[int] = 120
WATCH_THRESHOLD: Final[float] = 3.0
"""Threshold of the year-on-year detector inside a register watch window (design section 3.2)."""
TRUTH_MARGIN_DAYS: Final[int] = 30
SWITCH_MARGIN_AFTER_DAYS: Final[int] = 14
MIN_WINDOW_MW: Final[float] = 1.0
STORED_DIR: Final[Path] = features.DATA_DIR / "signals"
MODELS_PATH: Final[Path] = features.DATA_DIR / "stage0" / "models" / "models.json"
ERA_TABLE_PATH: Final[Path] = features.DATA_DIR / "era_table.parquet"
KNOWN_EVENTS_PATH: Final[Path] = features.DATA_DIR / "known_events.parquet"
ARMS: Final[tuple[str, ...]] = ("det", "llm", "ecr", "llm-ecr", "ceiling")
NO_CALL_ARMS: Final[tuple[str, ...]] = ("det", "ecr", "ceiling")
COLUMN_NAMES: Final[tuple[str, ...]] = (
    "state", "p_era", "age", "level", "anchor", "depth", "withdrawn", "era_id", "onset_day",
    "confirm_day",
    *(f"c{i}" for i in range(features.N_FEATURES)),
    *(f"b{i}" for i in range(4)),
)  # fmt: skip
"""The per-day columns of experiment 709's signal; a test asserts they equal `e00709_signal`'s."""
INT_COLUMNS: Final[frozenset[str]] = frozenset(
    {"state", "withdrawn", "era_id", "onset_day", "confirm_day"}
)
EPISODE_SCHEMA: Final[dict[str, Any]] = {
    "site": pl.Int64,
    "onset": pl.Int64,
    "alarm": pl.Int64,
    "end": pl.Int64,
    "first_era": pl.Int64,
}


@dataclass(frozen=True)
class Resolved:
    """One finished check of one candidate.

    Attributes:
        kind: `alarm`, `plus7`, `plus21`, or `closure`.
        day: The check day index.
        label: `era`, `switch`, `noise`, `undecided`, or `None` for a check with no valid reply.
        confidence: The reply's confidence, 0 where there is no valid reply.
        onset_offset_days: The reply's onset estimate (stored, never read by the network).
    """

    kind: str
    day: int
    label: str | None
    confidence: float
    onset_offset_days: int | None = None


@dataclass(frozen=True)
class DetectorBank:
    """What the as-of table reads from the detectors.

    Attributes:
        n_days: Number of days of the grid.
        spread: Robust spread of the six features of each Primary, shape (sites, 6).
        delta: For each (Primary, detector), the running change of the six scaled features since
            the episode's onset, shape (days, 6), NaN while no episode is open.
        vectors: For each (Primary, detector, alarm day), the day the 12-statistic change vector
            can first be read and the vector, which draws the era ids.
    """

    n_days: int
    spread: np.ndarray
    delta: dict[tuple[int, str], np.ndarray]
    vectors: dict[tuple[int, str, int], tuple[int, np.ndarray]]


@dataclass(frozen=True)
class AsOfTable:
    """The per-day signal of one arm, in the layout of `e00709_signal.Signal`.

    Attributes:
        arm: The arm's name.
        columns: Each of `COLUMN_NAMES`, shape (sites, days).
        episodes: One row per candidate: `site`, `onset`, `alarm`, `end`, and `first_era`.
    """

    arm: str
    columns: dict[str, np.ndarray]
    episodes: pl.DataFrame


# ---------------------------------------------------------------- candidates


def derive_ended_by(*, rows: list[dict[str, int]]) -> list[str]:
    """Return how each episode of one detector at one Primary closed.

    Episodes closed by a later alarm of the same detector end on the day before that alarm, and an
    episode closed by a return cannot be followed by an alarm on the next day (an alarm needs three
    consecutive days over the threshold after a return).

    Args:
        rows: The detector's episodes at one Primary, each with `alarm` and `end`.

    Returns:
        `return`, `superseded`, or `open` for each episode.
    """
    alarms = {row["alarm"] for row in rows}
    return [
        "open" if row["end"] < 0 else "superseded" if row["end"] + 1 in alarms else "return"
        for row in rows
    ]


def candidates_from_tables(
    *, bocpd_rows: pl.DataFrame, stat_rows: pl.DataFrame, stat3_rows: pl.DataFrame | None = None
) -> tuple[list[Candidate], list[Candidate]]:
    """Return the union candidates, and the lower-threshold candidates, from episode tables.

    Args:
        bocpd_rows: The change-point detector's episodes (`site`, `onset`, `alarm`, `end`).
        stat_rows: The year-on-year detector's episodes at its threshold of 4.0.
        stat3_rows: The same detector's episodes at its threshold of 3.0, or `None`.

    Returns:
        The union candidates ordered by Primary and alarm day (a statistics alarm inside a
        change-point episode joins that candidate), and the `stat3` candidates, which are listed
        separately and not merged.

    Raises:
        ValueError: If an episode ends before its alarm.
    """

    def build(*, table: pl.DataFrame, detector: str) -> list[dict[str, Any]]:
        rows = table.sort("site", "alarm").to_dicts()
        ended = {}
        for site in {r["site"] for r in rows}:
            mine = [r for r in rows if r["site"] == site]
            ended.update(
                {
                    (site, r["alarm"]): e
                    for r, e in zip(mine, derive_ended_by(rows=mine), strict=True)
                }
            )
        for number, row in enumerate(rows):
            if 0 <= row["end"] < row["alarm"]:
                msg = f"{detector} episode {number} has an end before its alarm"
                raise ValueError(msg)
            row |= {
                "detector": detector,
                "number": number,
                "ended_by": ended[(row["site"], row["alarm"])],
            }
        return rows

    first, second = (
        build(table=bocpd_rows, detector="bocpd"),
        build(table=stat_rows, detector="stat"),
    )
    joined: set[tuple[int, int]] = set()
    found: list[Candidate] = []
    for row in first:
        end = row["end"] if row["end"] >= 0 else 10**9
        absorbed = [
            s["alarm"]
            for s in second
            if s["site"] == row["site"] and row["alarm"] <= s["alarm"] <= end
        ]
        joined.update((row["site"], a) for a in absorbed)
        found.append(_candidate(row=row, stat_alarm=min(absorbed) if absorbed else None))
    found += [
        _candidate(row=s, stat_alarm=None) for s in second if (s["site"], s["alarm"]) not in joined
    ]
    third = (
        [_candidate(row=r, stat_alarm=None) for r in build(table=stat3_rows, detector="stat3")]
        if stat3_rows is not None
        else []
    )
    return sorted(found, key=lambda c: (c.site, c.alarm)), sorted(
        third, key=lambda c: (c.site, c.alarm)
    )


def _candidate(*, row: dict[str, Any], stat_alarm: int | None) -> Candidate:
    """Build a candidate from one episode row."""
    return Candidate(
        site=row["site"],
        episode_id=f"{row['detector']}{row['number']}",
        detector=row["detector"],
        onset=row["onset"],
        alarm=row["alarm"],
        end=row["end"],
        ended_by=row["ended_by"],
        stat_alarm=stat_alarm,
    )


def load_stored_candidates() -> list[Candidate]:
    """Return the union candidates of the stored 709 detector tables (no detector is re-run).

    Returns:
        The candidates, as `candidates_from_tables` builds them.
    """
    union, _ = candidates_from_tables(
        bocpd_rows=pl.read_parquet(STORED_DIR / "all_episodes.parquet"),
        stat_rows=pl.read_parquet(STORED_DIR / "stat_episodes.parquet"),
    )
    return union


# ---------------------------------------------------------------- the detectors, re-run for vectors


def build_detector_bank(*, stat3_sites: Sequence[int] = ()) -> tuple[DetectorBank, pl.DataFrame]:
    """Re-run both detectors on the real panel to read the change vectors of every candidate.

    The detectors are the frozen detectors of experiment 709 at its frozen thresholds
    (`stage0/models/models.json`). The year-on-year detector also runs at a threshold of 3.0 at
    `stat3_sites`, for the register watch windows.

    Args:
        stat3_sites: Primaries that need the lower-threshold run.

    Returns:
        The bank, and the episodes of every run with `site`, `detector`, `onset`, `alarm`, `end`,
        and `ended_by`, for comparison with the stored tables.
    """
    thresholds = json.loads(MODELS_PATH.read_text())
    panel = features.load_panel()
    all_features = features.compute_panel_features(panel=panel)
    n_days = panel.n_days
    spread = np.stack([np.nan_to_num(f.spread) for f in all_features])
    delta: dict[tuple[int, str], np.ndarray] = {}
    vectors: dict[tuple[int, str, int], tuple[int, np.ndarray]] = {}
    episode_rows: list[dict[str, Any]] = []
    for site, feat in enumerate(all_features):
        if feat.ready_day >= n_days:
            continue
        runs: list[tuple[str, bocpd.DetectorOutput, statdetect.StatOutput | None]] = []
        runs.append(
            (
                "bocpd",
                bocpd.detect(
                    scaled=feat.scaled,
                    spread=feat.spread,
                    start_day=feat.ready_day,
                    threshold=float(thresholds["threshold"]),
                ),
                None,
            )
        )
        windows = stats.window_statistics(power_site=panel.power[site])
        stat_out = statdetect.detect_stat(
            windows=windows, scaled=feat.scaled, threshold=float(thresholds["stat_threshold"])
        )
        runs.append(("stat", stat_out.detector, stat_out))
        if site in stat3_sites:
            low = statdetect.detect_stat(
                windows=windows, scaled=feat.scaled, threshold=WATCH_THRESHOLD
            )
            runs.append(("stat3", low.detector, low))
        for name, detector, stat_run in runs:
            delta[(site, name)] = detector.delta_scaled
            source = stat_run if stat_run is not None else stat_out
            for episode in detector.episodes:
                read_day, vector = (
                    stat_run.vectors[episode.episode]
                    if stat_run is not None
                    else statdetect._change_vector(
                        stat=windows.stat, scale=source.scale, episode=episode
                    )
                )
                vectors[(site, name, episode.alarm_day)] = (read_day, vector)
                episode_rows.append(
                    {
                        "site": site,
                        "detector": name,
                        "onset": episode.onset_day,
                        "alarm": episode.alarm_day,
                        "end": episode.end_day,
                        "ended_by": episode.ended_by,
                    }
                )
    bank = DetectorBank(n_days=n_days, spread=spread, delta=delta, vectors=vectors)
    return bank, pl.DataFrame(episode_rows)


def compare_with_stored(*, rerun: pl.DataFrame) -> None:
    """Check that the re-run detectors reproduce the stored episode tables and the derived closure.

    Args:
        rerun: The episodes that `build_detector_bank` returned.

    Raises:
        ValueError: If a stored episode is missing from the re-run, or the way an episode closed
            differs from `derive_ended_by`.
    """
    for detector, name in (("bocpd", "all"), ("stat", "stat")):
        stored = pl.read_parquet(STORED_DIR / f"{name}_episodes.parquet").select(
            "site", "onset", "alarm", "end"
        )
        mine = rerun.filter(pl.col("detector") == detector).select("site", "onset", "alarm", "end")
        if stored.sort("site", "alarm").to_dicts() != mine.sort("site", "alarm").to_dicts():
            msg = f"the re-run {detector} episodes differ from {name}_episodes.parquet"
            raise ValueError(msg)
    for detector in rerun["detector"].unique().to_list():
        for site in rerun["site"].unique().to_list():
            rows = rerun.filter((pl.col("detector") == detector) & (pl.col("site") == site))
            rows = rows.sort("alarm")
            derived = derive_ended_by(rows=rows.to_dicts())
            if derived != rows["ended_by"].to_list():
                msg = f"derive_ended_by disagrees with the detector for {detector} at site {site}"
                raise ValueError(msg)


def frozen_change(
    *, bank: DetectorBank, candidate: Candidate, class_day: int, day: int
) -> np.ndarray:
    """Return the six scaled changes at `day`: live until 28 days after the first `era` day.

    Args:
        bank: The detector bank.
        candidate: The candidate.
        class_day: The first day the candidate is classed `era`, or -1.
        day: The day.

    Returns:
        The change vector, as `e00709_classify._frozen_change` reads `delta_scaled`.
    """
    last_open = candidate.end if candidate.end >= 0 else bank.n_days - 1
    freeze = min(class_day + SETTLING_DAYS, last_open) if class_day >= 0 else last_open
    return np.nan_to_num(bank.delta[(candidate.site, candidate.detector)][min(day, freeze)])


# ---------------------------------------------------------------- answers, states, and the schedule


def class_of(*, label: str | None, confidence: float) -> int:
    """Map an answer to a state code by the confidence rule.

    Args:
        label: `era`, `switch`, `noise`, `undecided`, or `None`.
        confidence: The reply's confidence.

    Returns:
        `STATE_ERA`, `STATE_SWITCH`, or `STATE_NONE` (noise) for a class at confidence 0.8 or more,
        and `STATE_UNKNOWN` otherwise.
    """
    if label is None or confidence < CONFIDENCE_RULE:
        return STATE_UNKNOWN
    return {"era": STATE_ERA, "switch": STATE_SWITCH, "noise": STATE_NONE}.get(label, STATE_UNKNOWN)


def class_timeline(
    *, candidate: Candidate, resolved: Sequence[Resolved], n_days: int, sticky_era: bool = True
) -> np.ndarray:
    """Return the candidate's own state on every day, `ABSENT` before its alarm.

    Args:
        candidate: The candidate.
        resolved: Its finished checks.
        n_days: Number of days.
        sticky_era: Whether an `era` survives a later answer that is `undecided` or under 0.8.

    Returns:
        An int array of length `n_days`.
    """
    out = np.full(n_days, ABSENT, dtype=np.int8)
    answers = sorted(resolved, key=lambda r: r.day)
    current, i = STATE_UNKNOWN, 0
    for day in range(candidate.alarm, n_days):
        while i < len(answers) and answers[i].day <= day:
            new = class_of(label=answers[i].label, confidence=answers[i].confidence)
            current = (
                STATE_ERA if (sticky_era and current == STATE_ERA and new == STATE_UNKNOWN) else new
            )
            i += 1
        out[day] = current
    return out


def state_on_day(*, candidate: Candidate, resolved: Sequence[Resolved], day: int) -> int:
    """Return the candidate's own state on `day` from the checks made on or before it."""
    made = [r for r in resolved if r.day <= day]
    return int(class_timeline(candidate=candidate, resolved=made, n_days=day + 1)[day])


def scheduled_checks(*, candidate: Candidate, offsets: Sequence[int], last_day: int) -> list[Check]:
    """List the scheduled checks that fall inside the data and before the closure day.

    Args:
        candidate: The candidate.
        offsets: Offsets from the alarm day of the alarm check and the two later scheduled checks.
        last_day: Last day index with data.

    Returns:
        The scheduled checks in day order. A scheduled check after the alarm check is dropped when
        it falls on or after the closure day.
    """
    closure = candidate.closure_day
    out = []
    for kind, offset in zip(SCHEDULED_KINDS, offsets, strict=True):
        day = candidate.alarm + offset
        if day > last_day or (kind != "alarm" and closure is not None and day >= closure):
            continue
        out.append(Check(candidate=candidate, kind=kind, day=day))
    return out


def all_possible_checks(
    *, candidate: Candidate, offsets: Sequence[int], last_day: int
) -> list[Check]:
    """List every check the schedule could ask for, before the early-stop and closure rules."""
    out = scheduled_checks(candidate=candidate, offsets=offsets, last_day=last_day)
    closure = candidate.closure_day
    if closure is not None and closure <= last_day:
        out.append(Check(candidate=candidate, kind="closure", day=closure))
    return sorted(out, key=lambda c: c.day)


def _confident_class(*, resolved: Resolved) -> str | None:
    """Return the class of a confident era, switch, or noise answer, else `None`."""
    ok = resolved.label in ("era", "switch", "noise") and resolved.confidence >= CONFIDENCE_RULE
    return resolved.label if ok else None


def next_check(
    *,
    candidate: Candidate,
    resolved: Sequence[Resolved],
    offsets: Sequence[int] = REAL_OFFSETS,
    last_day: int,
) -> Check | None:
    """Return the next check to make, from the answers made so far, or `None` if none remains.

    Args:
        candidate: The candidate.
        resolved: Its finished checks.
        offsets: Offsets from the alarm day of the three scheduled checks.
        last_day: Last day index with data.

    Returns:
        The next scheduled check, else the closure check, else `None`. Two consecutive confident
        answers of the same class cancel the scheduled checks after the alarm check. The closure
        check runs only if the state on the closure day is `era` or `unknown`.
    """
    done = {r.kind for r in resolved}
    ordered = sorted(resolved, key=lambda r: r.day)
    stop_early = len(ordered) >= 2 and (
        _confident_class(resolved=ordered[-1]) is not None
        and _confident_class(resolved=ordered[-1]) == _confident_class(resolved=ordered[-2])
    )
    for check in scheduled_checks(candidate=candidate, offsets=offsets, last_day=last_day):
        if check.kind in done or (stop_early and check.kind != "alarm"):
            continue
        return check
    closure = candidate.closure_day
    if closure is None or closure > last_day or "closure" in done:
        return None
    if state_on_day(candidate=candidate, resolved=ordered, day=closure) in (
        STATE_ERA,
        STATE_UNKNOWN,
    ):
        return Check(candidate=candidate, kind="closure", day=closure)
    return None


def visible_candidate(*, candidate: Candidate, day: int) -> Candidate:
    """Return the candidate as it is known at the end of `day`: an end not yet known is open."""
    closure = candidate.closure_day
    if closure is not None and closure <= day:
        return candidate
    return Candidate(
        site=candidate.site,
        episode_id=candidate.episode_id,
        detector=candidate.detector,
        onset=candidate.onset,
        alarm=candidate.alarm,
        end=-1,
        ended_by="open",
        stat_alarm=candidate.stat_alarm,
    )


# ---------------------------------------------------------------- the as-of table


def _empty_columns(*, n_sites: int, n_days: int) -> dict[str, np.ndarray]:
    """Return the columns of a signal with no label: state `none` everywhere."""
    out: dict[str, np.ndarray] = {}
    for name in COLUMN_NAMES:
        if name in INT_COLUMNS:
            fill = -1 if name in {"onset_day", "confirm_day"} else 0
            out[name] = np.full((n_sites, n_days), fill, dtype=np.int64)
        else:
            out[name] = np.zeros((n_sites, n_days), dtype=np.float64)
    return out


def _write_day(
    *,
    cols: dict[str, np.ndarray],
    site: int,
    day: int,
    state: int,
    p_era: float,
    age: int,
    vector: np.ndarray,
    weights: np.ndarray,
    spread: np.ndarray,
    onset_day: int,
    confirm_day: int,
) -> None:
    """Write the columns of one (Primary, day) the way `compose_site_signal` does."""
    share = float(weights @ vector)
    cols["state"][site, day] = state
    cols["p_era"][site, day] = p_era
    cols["age"][site, day] = min(age, AGE_CAP_DAYS)
    cols["anchor"][site, day] = share if state == STATE_ERA else 0.0
    cols["depth"][site, day] = 0.0 if state == STATE_ERA else share
    cols["level"][site, day] = share
    cols["onset_day"][site, day] = onset_day
    cols["confirm_day"][site, day] = confirm_day
    for i in range(features.N_FEATURES):
        cols[f"c{i}"][site, day] = vector[i]
    for i in range(4):
        cols[f"b{i}"][site, day] = vector[i] * spread[i]


def compose_site(
    *,
    cols: dict[str, np.ndarray],
    site: int,
    candidates: Sequence[Candidate],
    timelines: Sequence[np.ndarray],
    bank: DetectorBank,
) -> list[dict[str, int]]:
    """Write one Primary's rows of the signal and return its episode rows.

    Args:
        cols: The signal's columns, written in place.
        site: The Primary.
        candidates: The Primary's candidates.
        timelines: The state of each candidate on each day (`class_timeline`).
        bank: The detector bank.

    Returns:
        One episode row per candidate (`onset`, `alarm`, `end`, `first_era`, without the site).
    """
    n_days = bank.n_days
    spread = bank.spread[site]
    weights = features.BAND_WEIGHTS * spread
    class_day = [
        int(np.flatnonzero(t == STATE_ERA)[0]) if (t == STATE_ERA).any() else -1 for t in timelines
    ]
    persistent: list[int] = []
    stacks: list[tuple[int, ...]] = []
    overturned: dict[int, int] = {}
    for day in range(n_days):
        for i, line in enumerate(timelines):
            now = int(line[day])
            before = int(line[day - 1]) if day > 0 else ABSENT
            if now == STATE_ERA and i not in persistent:
                persistent.append(i)
            elif before == STATE_ERA and now in (STATE_SWITCH, STATE_NONE):
                if i in persistent:
                    persistent.remove(i)
                cols["withdrawn"][site, day] = 1
                overturned[i] = day
        active = [
            i for i, c in enumerate(candidates) if c.alarm <= day and (c.end < 0 or day <= c.end)
        ]
        own = max(active, key=lambda i: (candidates[i].alarm, i)) if active else None
        own_state = int(timelines[own][day]) if own is not None else STATE_NONE
        if own is not None and own_state in (STATE_UNKNOWN, STATE_SWITCH):
            c = candidates[own]
            _write_day(
                cols=cols, site=site, day=day, state=own_state,
                p_era=0.5 if own_state == STATE_UNKNOWN else 0.0,
                age=day - c.onset + 1,
                vector=frozen_change(bank=bank, candidate=c, class_day=class_day[own], day=day),
                weights=weights, spread=spread, onset_day=c.onset, confirm_day=c.alarm,
            )  # fmt: skip
        elif persistent:
            top = persistent[-1]
            c = candidates[top]
            _write_day(
                cols=cols, site=site, day=day, state=STATE_ERA, p_era=1.0,
                age=day - c.onset + 1,
                vector=frozen_change(bank=bank, candidate=c, class_day=class_day[top], day=day),
                weights=weights, spread=spread, onset_day=c.onset, confirm_day=class_day[top],
            )  # fmt: skip
        stacks.append(tuple(persistent))
    cols["era_id"][site] = _era_ids(
        site=site, candidates=candidates, stacks=stacks, bank=bank, n_days=n_days
    )
    rows = []
    for i, c in enumerate(candidates):
        end = c.end
        if end >= 0 and i in overturned:
            end = max(end, overturned[i])
        rows.append({"onset": c.onset, "alarm": c.alarm, "end": end, "first_era": class_day[i]})
    return rows


def _era_ids(
    *,
    site: int,
    candidates: Sequence[Candidate],
    stacks: list[tuple[int, ...]],
    bank: DetectorBank,
    n_days: int,
) -> np.ndarray:
    """Return the era id of each day: an id is given when an era's vector can first be read."""
    out = np.zeros(n_days, dtype=np.int64)
    first_seen: dict[int, int] = {}
    for day, stack in enumerate(stacks):
        for i in stack:
            first_seen.setdefault(i, day)
    confirmed_on: dict[int, int] = {}
    vectors: dict[int, np.ndarray] = {}
    for i, day in first_seen.items():
        c = candidates[i]
        found = bank.vectors.get((site, c.detector, c.alarm))
        if found is not None and found[0] < n_days:
            confirmed_on[i] = max(day, found[0])
            vectors[i] = found[1]
    ids = statdetect.assign_era_ids(
        confirmed=[(i, day, vectors[i]) for i, day in confirmed_on.items()]
    )
    for day, stack in enumerate(stacks):
        for i in reversed(stack):
            if i in confirmed_on and confirmed_on[i] <= day:
                out[day] = ids[i]
                break
    return out


def build_table(
    *,
    arm: str,
    candidates: Sequence[Candidate],
    answers: dict[str, list[Resolved]],
    bank: DetectorBank,
    n_sites: int,
    sticky_era: bool = True,
) -> AsOfTable:
    """Build the as-of label table of one arm from its candidates and their answers.

    Args:
        arm: The arm's name.
        candidates: The arm's candidates.
        answers: The finished checks of each candidate, by `Candidate.key`; a candidate with no
            entry has no valid reply and stays `unknown` from its alarm day.
        bank: The detector bank.
        n_sites: Number of Primaries.
        sticky_era: See `class_timeline`.

    Returns:
        The table.
    """
    n_days = bank.n_days
    cols = _empty_columns(n_sites=n_sites, n_days=n_days)
    episode_rows: list[dict[str, int]] = []
    for site in range(n_sites):
        mine = sorted(
            (c for c in candidates if c.site == site), key=lambda c: (c.alarm, c.episode_id)
        )
        if not mine:
            continue
        timelines = [
            class_timeline(
                candidate=c, resolved=answers.get(c.key, []), n_days=n_days, sticky_era=sticky_era
            )
            for c in mine
        ]
        rows = compose_site(cols=cols, site=site, candidates=mine, timelines=timelines, bank=bank)
        episode_rows += [{"site": site, **row} for row in rows]
    return AsOfTable(
        arm=arm, columns=cols, episodes=pl.DataFrame(episode_rows, schema=EPISODE_SCHEMA)
    )


# ---------------------------------------------------------------- the arms that need no call


def det_answers(*, candidates: Sequence[Candidate]) -> dict[str, list[Resolved]]:
    """Return arm `det`'s answers: every candidate is `era` from its alarm day, at confidence 1."""
    return {
        c.key: [Resolved(kind="alarm", day=c.alarm, label="era", confidence=1.0)]
        for c in candidates
    }


VERIFIED_SOURCE: Final[str] = "experiment 580 verified"
LABELLED_SOURCE: Final[str] = "experiment 41 labelled set"
KNOWN_SOURCES: Final[tuple[str, ...]] = (VERIFIED_SOURCE, LABELLED_SOURCE)


@dataclass(frozen=True)
class KnownAnswer:
    """What the era table and the known events say about one candidate.

    Attributes:
        kind: `era`, `switch`, `unresolved` (a dated row of the era table with no era index), or
            `none` when no known answer names the candidate.
        source: The event source of a `switch`, or the era table for the other kinds.
        via: `site` or `other_site` for a `switch`: which of the event's two Primaries matched.
        reference_day: The day index the candidate's onset is compared with: the telemetry date
            (else the register date) of an era, or the onset of a switching event.
        register_day: The register date of an era, if any.
        telemetry_day: The telemetry date of an era, if any.
    """

    kind: str
    source: str | None = None
    via: str | None = None
    reference_day: int | None = None
    register_day: int | None = None
    telemetry_day: int | None = None


def _event_days(*, row: dict[str, Any]) -> tuple[int, int]:
    """Return the first and last day index of a known switching event."""
    first = int((row["onset_utc"] - features.GRID_START).total_seconds() // 86400)
    last = int((row["end_utc"] - features.GRID_START).total_seconds() // 86400)
    return first, last


def known_answers(
    *,
    candidates: Sequence[Candidate],
    era_table: pl.DataFrame,
    known_events: pl.DataFrame,
    sources: Sequence[str] = KNOWN_SOURCES,
    use_other_site: bool = True,
    margin: int = TRUTH_MARGIN_DAYS,
    margin_after: int = SWITCH_MARGIN_AFTER_DAYS,
) -> dict[str, KnownAnswer]:
    """Return what the known answers say about each candidate.

    A candidate is an `era` if its estimated onset lies within `margin` days of the telemetry or
    register date of a dated era of the same Primary. It is `unresolved` if the same holds for an
    era-table row with no era index. It is a `switch` if its estimated onset lies from `margin`
    days before the onset of a known switching event to `margin_after` days after its end, where
    the event's `site` (and, with `use_other_site`, its `other_site`) is the candidate's Primary.
    The order of precedence is era, unresolved, switch. These margins are this script's reading of
    design section 3.1 ("where the era table or the verified events name one").

    Args:
        candidates: The candidates.
        era_table: `era_table.parquet`.
        known_events: `known_events.parquet`.
        sources: The event sources that count.
        use_other_site: Whether an event also names its `other_site`.
        margin: Days before and after an era date, and days before a switching event.
        margin_after: Days after the end of a switching event.

    Returns:
        The known answer by `Candidate.key`.
    """
    eras: dict[int, list[KnownAnswer]] = {}
    for row in era_table.iter_rows(named=True):
        when = row["telemetry_date"] or row["register_date"]
        if when is None or row["primary"] not in features.PRIMARIES:
            continue
        site = features.PRIMARIES.index(row["primary"])
        eras.setdefault(site, []).append(
            KnownAnswer(
                kind="era" if row["era_index"] is not None else "unresolved",
                source="era table",
                reference_day=features.day_index(day=when),
                register_day=features.day_index(day=row["register_date"])
                if row["register_date"]
                else None,
                telemetry_day=features.day_index(day=row["telemetry_date"])
                if row["telemetry_date"]
                else None,
            )
        )
    events: dict[int, list[tuple[int, int, str, str]]] = {}
    for row in known_events.filter(pl.col("source").is_in(list(sources))).iter_rows(named=True):
        first, last = _event_days(row=row)
        for via in ("site", "other_site") if use_other_site else ("site",):
            name = row.get(via)
            if name in features.PRIMARIES:
                events.setdefault(features.PRIMARIES.index(name), []).append(
                    (first, last, row["source"], via)
                )
    out: dict[str, KnownAnswer] = {}
    for c in candidates:
        near = [e for e in eras.get(c.site, []) if abs(c.onset - (e.reference_day or 0)) <= margin]
        best = next((e for e in near if e.kind == "era"), None) or (near[0] if near else None)
        hits = [
            e for e in events.get(c.site, []) if e[0] - margin <= c.onset <= e[1] + margin_after
        ]
        if best is not None:
            out[c.key] = best
        elif hits:
            first, _, source, via = min(hits, key=lambda e: abs(c.onset - e[0]))
            out[c.key] = KnownAnswer(kind="switch", source=source, via=via, reference_day=first)
        else:
            out[c.key] = KnownAnswer(kind="none")
    return out


def true_classes(
    *,
    candidates: Sequence[Candidate],
    era_table: pl.DataFrame,
    known_events: pl.DataFrame,
    use_other_site: bool = True,
) -> dict[str, str]:
    """Return the true class of each candidate that a known answer names, else `noise`.

    Arm `ceiling` reads the era table and the events of experiment 580 that were verified (not the
    labelled set of experiment 41). With `use_other_site`, an event also names its second Primary.

    Args:
        candidates: The candidates.
        era_table: `era_table.parquet`.
        known_events: `known_events.parquet`.
        use_other_site: See `known_answers`.

    Returns:
        `era`, `switch`, or `noise` by `Candidate.key`; an `unresolved` row counts as `noise`.
    """
    known = known_answers(
        candidates=candidates,
        era_table=era_table,
        known_events=known_events,
        sources=(VERIFIED_SOURCE,),
        use_other_site=use_other_site,
    )
    return {key: k.kind if k.kind in ("era", "switch") else "noise" for key, k in known.items()}


def ceiling_answers(
    *, candidates: Sequence[Candidate], truth: dict[str, str]
) -> dict[str, list[Resolved]]:
    """Return arm `ceiling`'s answers: the true class, shown on the alarm day, at confidence 1."""
    return {
        c.key: [Resolved(kind="alarm", day=c.alarm, label=truth[c.key], confidence=1.0)]
        for c in candidates
    }


# ---------------------------------------------------------------- the register arms


def _technology_class(*, text: str | None) -> str:
    """Map the register's technology text to solar, wind, storage, engine, or other."""
    lowered = (text or "").lower()
    for word, name in (
        ("solar", "solar"),
        ("wind", "wind"),
        ("storage", "storage"),
        ("battery", "storage"),
        ("engine", "engine"),
        ("fossil", "engine"),
    ):
        if word in lowered:
            return name
    return "other"


def register_rows_from_era_table(*, era_table: pl.DataFrame) -> tuple[RegisterRow, ...]:
    """Return the dated register rows of the era table as register rows.

    The real record holds three dated rows of 1 MW or more (design section 3.2). The register file
    and its coordinate match are not read here. A row whose source says it is listed under another
    Primary is filed under that Primary with the distance band `under 1 km` and a watch window
    there, and it is also shown at its own Primary as `listed under a neighbouring substation`
    with no window (design section 3.2, the engine of D8 and D14).

    Args:
        era_table: `era_table.parquet`.

    Returns:
        The rows.
    """
    rows: list[RegisterRow] = []
    for row in era_table.iter_rows(named=True):
        if row["register_date"] is None or row["primary"] not in features.PRIMARIES:
            continue
        size = float(row["size_mw_register"] or 0.0)
        day = features.day_index(day=row["register_date"])
        tech = _technology_class(text=row["technology"])
        home = features.PRIMARIES.index(row["primary"])
        match = re.search(r"listed under (D\d+)", str(row["source"]))
        if match and match.group(1) in features.PRIMARIES:
            filed = features.PRIMARIES.index(match.group(1))
            rows.append(
                RegisterRow(
                    site=filed,
                    connected_day=day,
                    size_mw=size,
                    technology=tech,
                    distance_band="under 1 km",
                    opens_window=size >= MIN_WINDOW_MW,
                )
            )
            rows.append(
                RegisterRow(
                    site=home,
                    connected_day=day,
                    size_mw=size,
                    technology=tech,
                    distance_band="listed under a neighbouring substation",
                    opens_window=False,
                )
            )
        else:
            rows.append(
                RegisterRow(
                    site=home,
                    connected_day=day,
                    size_mw=size,
                    technology=tech,
                    distance_band="not matched by coordinates",
                    opens_window=size >= MIN_WINDOW_MW,
                )
            )
    return tuple(rows)


def watch_windows(
    *, rows: Sequence[RegisterRow], lag: int = DEFAULT_PUBLICATION_LAG_DAYS
) -> list[tuple[int, int, int]]:
    """Return the watch windows as (Primary, first day, last day).

    A window opens on the day a row of 1 MW or more becomes visible and closes 120 days after its
    connection date. A row that opens no window is skipped.
    """
    return [
        (r.site, r.connected_day + lag, r.connected_day + WATCH_DAYS_AFTER_CONNECTION)
        for r in rows
        if r.opens_window
    ]


def ecr_candidates(
    *, windows: Sequence[tuple[int, int, int]], low_threshold: Sequence[Candidate]
) -> list[Candidate]:
    """Return the candidates of arm `ecr`: the first lower-threshold alarm in each watch window.

    Args:
        windows: The watch windows.
        low_threshold: The `stat3` candidates.

    Returns:
        The first candidate of each window with an alarm in it.
    """
    chosen: dict[str, Candidate] = {}
    for site, first, last in windows:
        inside = [c for c in low_threshold if c.site == site and first <= c.alarm <= last]
        if inside:
            winner = min(inside, key=lambda c: c.alarm)
            chosen[winner.key] = winner
    return sorted(chosen.values(), key=lambda c: (c.site, c.alarm))


def llm_ecr_candidates(
    *,
    union: Sequence[Candidate],
    windows: Sequence[tuple[int, int, int]],
    low_threshold: Sequence[Candidate],
) -> list[Candidate]:
    """Return the candidates of arm `llm-ecr`: the union plus lower-threshold alarms in a window.

    A lower-threshold alarm that matches a union candidate's alarm day at the same Primary adds
    nothing.
    """
    have = {(c.site, c.alarm) for c in union}
    extra = [
        c
        for c in low_threshold
        if (c.site, c.alarm) not in have
        and any(site == c.site and first <= c.alarm <= last for site, first, last in windows)
    ]
    return sorted([*union, *extra], key=lambda c: (c.site, c.alarm))


# ---------------------------------------------------------------- disk


def ensure_709_importable() -> None:
    """Put the folders of experiments 86, 598, 660, and 687 on `sys.path` for `e00709_signal`."""
    root = Path(__file__).resolve().parents[1]
    for pattern in ("00086-*", "00598-*", "00660-*", "00687-*"):
        for folder in root.glob(pattern):
            if str(folder) not in sys.path:
                sys.path.append(str(folder))


def to_signal(*, table: AsOfTable) -> Any:
    """Wrap a table as a `e00709_signal.Signal` (causal: every arm reads an as-of table)."""
    ensure_709_importable()
    import e00709_signal as signal  # ty: ignore[unresolved-import]

    return signal.Signal(arm=table.arm, causal=True, columns=table.columns, episodes=table.episodes)


def write_table(*, table: AsOfTable, folder: Path) -> Path:
    """Write a table to `<folder>/<arm>.parquet` and `<arm>_episodes.parquet` (709's format).

    Args:
        table: The table.
        folder: The destination folder.

    Returns:
        The path of the signal file.
    """
    ensure_709_importable()
    import e00709_signal as signal  # ty: ignore[unresolved-import]

    return signal.write_signal(signal=to_signal(table=table), folder=folder)

"""Experiment 712, cost trial: label the detector candidates of one Primary once, to measure cost.

**This script measures what a language-model labelling call costs, and its outputs are never labels
and never scored.** It builds the pinned-style prompt of the design (sections 2.2 and 2.3 of
`briefs/clean_era_labels/design.md`) for every check of every candidate of one real-record
Primary, sends each prompt to `claude-sonnet-5-5` once, and records tokens, reported cost, wall
time, the billing route (`apiKeySource`), and any rate-limit event. A summary step extrapolates the
measured tokens and cost to 16 Primaries, to the injected bank, and to 2,500 substations.

**The prompt shows numbers about one unnamed substation, with every time as an offset from the
check day.** It holds no name, date, year, month number, weekday mark, bank-holiday flag, or
megawatt value. Every table is built from a copy of the inputs cut at the end of the check day, and
the script asserts the cut and that perturbing every later reading changes no prompt.

**Usage.** `--list-candidates` prints the candidate table. `--dry-run` builds and stores every
prompt and makes no call. Without either flag the script sends one billing probe, then the check
prompts, at most 2 at a time, and stops at once on a rate-limit message with a non-zero exit.
`--summary` reads the trial log and prints the totals and the extrapolation.

**Open items before the prompt is pinned (not fixed in this trial).** Calendar keys remain in the
prompt: the weekday (the weekday and weekend columns, the daily means, and the hourly profiles
show it, because the climatology has no weekend flag), the step where the climatology changes on
the 1st of each month, the Easter and Christmas dips, and the "n/a" of a partial last block in
Table 3. The year probe of design section 2.2.1 should oversample Easter, Christmas, and May bank
holiday windows, and a causal day-of-year climatology with a weekend flag should be considered.

**Safety.** With no flag the script only lists candidates. `--probe-only` sends the billing
probe. `--run --expect-key-source=<value>` sends the probe, aborts if the reported `apiKeySource`
differs, and otherwise sends the checks. A second spending run needs `--resume`.

**Not implemented against the design.** The 13 evidence columns of `e00709_classify.EVIDENCE_NAMES`
need `stage0.process_hosts` and are replaced by the detector, the days since the alarm, and the
episode state. The last 6 days of weather come from no source (the ensemble summaries are not read),
so those cells read "not available". The early-stop rule is not applied, so a candidate gets every
scheduled check.
"""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import e00709_features as features  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.capacity_labeller import Prompt
from studies.capacity_llm import (
    CALL_TIMEOUT_SECONDS,
    MODEL_ID,
    PER_CALL_BUDGET_USD,
    BudgetExceeded,
    BudgetGuard,
    CallStore,
    ClaudeCliModel,
    ModelError,
    ModelReply,
    estimate_input_tokens,
    parse_stream,
    stream_message,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

JsonObject = dict[str, Any]
"""A JSON object read from the command-line tool's output or from the trial log."""

EXPERIMENT_ID: Final[int] = 712
SLUG: Final[str] = "clean-era-labels-part-1"
TRIAL_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "trial"
"""Private output folder: prompts, the call store, and the trial log. Never committed."""
PROMPT_DIR: Final[Path] = TRIAL_DIR / "prompts"
CALL_DIR: Final[Path] = TRIAL_DIR / "calls"
LOG_PATH: Final[Path] = TRIAL_DIR / "trial_log.jsonl"
STOP_PATH: Final[Path] = TRIAL_DIR / "stopped.json"
SOURCE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00709-era-signal-in-the-candidate-part-1" / "signals"
)

DEFAULT_PRIMARY: Final[str] = "D10"
"""The Primary of the trial; never D8 or D5, which are known answers (design section 2.3.1)."""
FORBIDDEN_PRIMARIES: Final[frozenset[str]] = frozenset({"D8", "D5"})

MAX_CALLS: Final[int] = 40
MAX_COST_USD: Final[float] = 5.0
MAX_INPUT_TOKENS: Final[int] = 1_000_000
MAX_WORKERS: Final[int] = 2
PRINT_EVERY: Final[int] = 5
MAX_ATTEMPTS: Final[int] = 2
"""A reply that fails the schema is asked for once more; each attempt is a call."""
RETRIES_RESERVED: Final[int] = 4
"""Calls kept free in the call limit for the one retry of an invalid reply."""

SCHEDULED_OFFSETS: Final[tuple[int, ...]] = (0, 7, 21)
"""Check days as offsets from the alarm day (design section 2.4); the closure check is added."""
WEEKS: Final[int] = 17
DAILY_DAYS: Final[int] = 28
PROFILE_WEEKS_BEFORE: Final[int] = 8
MAX_NEIGHBOURS: Final[int] = 3
ERA5_LAG_DAYS: Final[int] = 6
"""ERA5 is used only for days up to this many days before the check day (so the last 6 days
(days -5 to 0) read "not available")."""
FEATURE_LABELS: Final[tuple[str, ...]] = ("night", "morning", "midday", "evening", "p05", "p95")
HALF_HOURS: Final[int] = features.HALF_HOURS_PER_DAY

PRIMARY_YEARS: Final[float] = 115.0
CANDIDATES_PER_PRIMARY_YEAR: Final[float] = 0.8
PLANNED_REAL_CALLS: Final[int] = 245
"""The design's planned calls for the 92 real-record candidates (section 6.3)."""
BANK_CANDIDATES: Final[int] = 182
BANK_MAX_CALLS: Final[int] = 364
SUBSTATIONS: Final[int] = 2_500
RECORD_YEARS_PER_SUBSTATION: Final[float] = PRIMARY_YEARS / 16

RATE_LIMIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"rate.?limit|usage limit|too many requests|429|quota|overloaded|limit reached", re.IGNORECASE
)

SYSTEM_PROMPT: Final[str] = """\
You label one detected shift in the half-hourly power of one electricity substation.
All times are offsets from the check day (day 0, week 0). You see nothing after day 0.
Classes:
  era     a change in the substation's own net demand that will persist, such as a generator
          or a solar farm connecting, or a large customer arriving or leaving.
  switch  load moved to or from a neighbouring substation by a temporary reconfiguration of
          the electricity network. The level is expected to return.
  noise   no real shift: weather, a holiday, a seasonal pattern, a metering fault, or an
          excursion that has already ended.
Evidence to weigh: whether a neighbour shows the opposite change and the sum is conserved
(switch); whether the change follows sunshine and is zero at night (era, solar); whether the
change is flat across the day (era, engine, or switch); how long the change has lasted; and,
where register rows are given, whether a row of matching size and type lies near the onset.
A register row is evidence, not proof: a registered capacity can be several times the change
seen at the substation, and a connection date can precede the change by weeks.
If the evidence does not yet separate the classes, answer undecided.
Reply with one JSON object and nothing else."""
"""The draft system prompt of design section 2.3, verbatim."""

SCHEMA_TEXT: Final[str] = """\
Reply with one JSON object with exactly these keys:
  "label": "era", "switch", "noise", or "undecided"
  "onset_offset_days": integer, 0 or below, your estimate of the first day of the shift
  "confidence": number from 0 to 1, your probability that label is right
  "neighbour_mirror": "yes", "no", or "unmetered"
  "shape": "solar", "flat", "demand-like", or "irregular"
  "register_row": integer or null (no register rows are given in this prompt)
  "reason": string of at most 60 words"""

PROBE_SYSTEM: Final[str] = "Reply with the single word OK."
PROBE_TEXT: Final[str] = "Billing probe. Reply OK."

LABELS: Final[frozenset[str]] = frozenset({"era", "switch", "noise", "undecided"})
MIRRORS: Final[frozenset[str]] = frozenset({"yes", "no", "unmetered"})
SHAPES: Final[frozenset[str]] = frozenset({"solar", "flat", "demand-like", "irregular"})
SCHEMA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "label",
        "onset_offset_days",
        "confidence",
        "neighbour_mirror",
        "shape",
        "register_row",
        "reason",
    }
)


class RateLimited(RuntimeError):
    """The command-line tool reported a rate limit, so the whole run must stop."""


@dataclass(frozen=True)
class Candidate:
    """One detector candidate shift.

    Attributes:
        site: Index of the Primary in `features.PRIMARIES`.
        episode_id: Detector name and the episode's position among that detector's episodes.
        detector: The detector that raised the alarm.
        onset: Estimated onset day index.
        alarm: Alarm day index.
        end: End day index of the detector's episode.
        stat_alarm: Day of the first year-on-year statistics alarm inside this change-point
            episode, or `None`. It is shown in a prompt only on or after that day.
    """

    site: int
    episode_id: str
    detector: str
    onset: int
    alarm: int
    end: int
    stat_alarm: int | None


@dataclass(frozen=True)
class Check:
    """One scheduled check of one candidate.

    Attributes:
        candidate: The candidate.
        kind: `alarm`, `plus7`, `plus21`, or `closure`.
        day: Check day index.
    """

    candidate: Candidate
    kind: str
    day: int

    @property
    def offset(self) -> int:
        """The check day as an offset from the alarm day, the only time label stored."""
        return self.day - self.candidate.alarm

    @property
    def call_id(self) -> str:
        """A date-free identifier of the check."""
        return f"{features.PRIMARIES[self.candidate.site]}_{self.candidate.episode_id}_{self.kind}"


@dataclass(frozen=True)
class Inputs:
    """The real record the prompts are built from.

    Attributes:
        panel: The 16 Primaries' power as a share of capacity.
        shortwave_midday: ERA5 midday shortwave per site and day.
        temperature: ERA5 daily mean temperature per site and day.
        last_day: Last day index with any reading at any Primary.
    """

    panel: features.Panel
    shortwave_midday: np.ndarray
    temperature: np.ndarray
    last_day: int


@dataclass(frozen=True)
class RawRun:
    """What one run of the command-line tool produced.

    Attributes:
        stdout: The `stream-json` output.
        stderr: The standard error.
        returncode: The exit status.
        seconds: Wall time.
        directory: The empty directory the tool ran in.
    """

    stdout: str
    stderr: str
    returncode: int
    seconds: float
    directory: str = ""


def load_candidates() -> list[Candidate]:
    """Return the union of the two stored 709 detectors, as design section 2.1 merges them.

    Returns:
        Candidates ordered by site and alarm day. A statistics alarm inside an open change-point
        episode (alarm day from the change-point alarm to its end day) joins that candidate and is
        not listed separately.
    """
    bocpd = pl.read_parquet(SOURCE_DIR / "all_episodes.parquet").sort("site", "alarm")
    stat = pl.read_parquet(SOURCE_DIR / "stat_episodes.parquet").sort("site", "alarm")
    found: list[Candidate] = []
    joined_stat: set[tuple[int, int]] = set()
    for number, row in enumerate(bocpd.iter_rows(named=True)):
        absorbed = stat.filter(
            (pl.col("site") == row["site"])
            & (pl.col("alarm") >= row["alarm"])
            & (pl.col("alarm") <= row["end"])
        )
        joined_stat.update((row["site"], alarm) for alarm in absorbed["alarm"].to_list())
        found.append(
            Candidate(
                site=row["site"],
                episode_id=f"bocpd{number}",
                detector="bocpd",
                onset=row["onset"],
                alarm=row["alarm"],
                end=row["end"],
                stat_alarm=min(absorbed["alarm"].to_list()) if absorbed.height else None,
            )
        )
    for number, row in enumerate(stat.iter_rows(named=True)):
        if (row["site"], row["alarm"]) in joined_stat:
            continue
        found.append(
            Candidate(
                site=row["site"],
                episode_id=f"stat{number}",
                detector="stat",
                onset=row["onset"],
                alarm=row["alarm"],
                end=row["end"],
                stat_alarm=None,
            )
        )
    for candidate in found:
        if candidate.end < candidate.alarm:
            msg = f"{candidate.episode_id} has an end before its alarm"
            raise ValueError(msg)
    return sorted(found, key=lambda c: (c.site, c.alarm))


def load_inputs() -> Inputs:
    """Read the real panel and the ERA5 daily weather.

    Returns:
        The inputs.
    """
    panel = features.load_panel()
    weather = features.load_weather(panel=panel)
    finite_days = np.isfinite(panel.power).any(axis=0).reshape(-1, HALF_HOURS).any(axis=1)
    return Inputs(
        panel=panel,
        shortwave_midday=weather.shortwave_midday,
        temperature=weather.temperature,
        last_day=int(np.flatnonzero(finite_days).max()),
    )


def checks_of(*, candidate: Candidate, last_day: int) -> list[Check]:
    """List the checks of a candidate that fall inside the available data.

    Args:
        candidate: The candidate.
        last_day: Last day index with data.

    Returns:
        The alarm-day check, the checks 7 and 21 days later, and the closure check, in day order.
        The closure check runs on the day after the end day, when both ways a detector closes an
        episode are known (design section 2.4), and only if that day has data. Checks that fall on
        the same day appear once.
    """
    wanted = [
        Check(candidate=candidate, kind=kind, day=candidate.alarm + offset)
        for kind, offset in zip(("alarm", "plus7", "plus21"), SCHEDULED_OFFSETS, strict=True)
    ]
    wanted.append(Check(candidate=candidate, kind="closure", day=candidate.end + 1))
    seen: set[int] = set()
    out: list[Check] = []
    for check in sorted(wanted, key=lambda c: c.day):
        if check.day > last_day or check.day in seen:
            continue
        seen.add(check.day)
        out.append(check)
    return out


def _fmt(*, value: float, places: int = 1) -> str:
    """Format a number, or `n/a` for NaN."""
    return "n/a" if not math.isfinite(value) else f"{value:.{places}f}"


def _weekday_flags(*, first_day: int, count: int) -> np.ndarray:
    """Return True for each of `count` days from `first_day` that is a Saturday or Sunday."""
    return np.array([features.day_of(index=first_day + k).weekday() >= 5 for k in range(count)])


def _mean(*, values: np.ndarray) -> float:
    """Mean of the finite values, NaN if there are none."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _block_means(*, series: np.ndarray, start: int, stop: int) -> tuple[float, float]:
    """Return the weekday mean and the weekend mean of `series[start:stop]` (days)."""
    flags = _weekday_flags(first_day=start, count=stop - start)
    part = series[start:stop]
    return _mean(values=part[~flags]), _mean(values=part[flags])


def capacity_scale(*, power: np.ndarray) -> np.ndarray:
    """Return each site's capacity scale known at the end of `power`: its 99th percentile.

    The panel holds power as a share of an effective capacity set from the whole record. Dividing
    by the 99th percentile of the absolute share up to the check day gives a scale that uses no
    later reading.

    Args:
        power: Power over the panel's capacity, shape (sites, half-hours), cut at the check day.

    Returns:
        One factor per site; 1 where a site has no reading.
    """
    out = np.ones(power.shape[0])
    for row in range(power.shape[0]):
        finite = np.abs(power[row][np.isfinite(power[row])])
        if finite.size:
            out[row] = max(float(np.percentile(finite, 99)), 1e-6)
    return out


def build_prompt_text(*, inputs: Inputs, check: Check) -> str:
    """Build the case text of one check from readings up to the end of the check day.

    Every array is cut to `check.day + 1` days before any table is computed, so no table can read
    a later reading.

    Args:
        inputs: The real record.
        check: The check.

    Returns:
        The user text, ending in the output schema.
    """
    candidate, day0 = check.candidate, check.day
    n_days = day0 + 1
    if day0 < DAILY_DAYS - 1:
        msg = "the check day is too early for the 28-day tables"
        raise ValueError(msg)
    raw_power = inputs.panel.power[:, : n_days * HALF_HOURS]
    if raw_power.shape[1] != n_days * HALF_HOURS:
        msg = "the input must end at the check day"
        raise ValueError(msg)
    scale = capacity_scale(power=raw_power)
    power = raw_power / scale[:, None]
    site = candidate.site
    own_features = features.compute_site_features(power_site=power[site]).raw
    onset_offset = candidate.onset - day0
    lines: list[str] = []
    lines.append("All values are percent of the substation's capacity unless stated otherwise.")
    lines.append("Week -k is the 7-day block that ends 7k days before day 0.")
    lines.extend(_weekly_features(raw=own_features, day0=day0, onset_offset=onset_offset))
    lines.extend(_daily_features(raw=own_features, day0=day0))
    lines.extend(_profiles(power_site=power[site], candidate=candidate, day0=day0))
    lines.extend(_neighbours(inputs=inputs, power=power, scale=scale, site=site, day0=day0))
    lines.extend(_weather(inputs=inputs, site=site, day0=day0))
    lines.extend(_evidence(check=check))
    lines.append("")
    lines.append(SCHEMA_TEXT)
    return "\n".join(lines) + "\n"


def _weekly_features(*, raw: np.ndarray, day0: int, onset_offset: int) -> list[str]:
    """Table 1: weekday and weekend means of the six features for each of the 17 weeks."""
    header = "week  " + "  ".join(f"{name}(wd/we)".ljust(15) for name in FEATURE_LABELS)
    out = ["", "TABLE 1. Weekly detector features (residual against a climatology):", header]
    for k in range(WEEKS):
        stop = day0 + 1 - 7 * k
        start = stop - 7
        cells = []
        for f in range(raw.shape[1]):
            if start < 0:
                cells.append("n/a".ljust(15))
                continue
            wd, we = _block_means(series=raw[:, f] * 100.0, start=start, stop=stop)
            cells.append(f"{_fmt(value=wd, places=2)}/{_fmt(value=we, places=2)}".ljust(15))
        mark = (
            "  <- detector's estimated onset" if start - day0 <= onset_offset < stop - day0 else ""
        )
        out.append(f"{-k:>4}  " + "  ".join(cells) + mark)
    return out


def _daily_features(*, raw: np.ndarray, day0: int) -> list[str]:
    """Table 2: the six features for each of the last 28 days."""
    out = ["", "TABLE 2. Daily detector features, last 28 days:"]
    out.append("day   " + "  ".join(name.ljust(8) for name in FEATURE_LABELS))
    for offset in range(-(DAILY_DAYS - 1), 1):
        index = day0 + offset
        cells = [_fmt(value=raw[index, f] * 100.0, places=2).ljust(8) for f in range(raw.shape[1])]
        out.append(f"{offset:>4}  " + "  ".join(cells))
    return out


def _hourly(*, power_site: np.ndarray, first: int, last: int, weekend: bool) -> str:
    """Mean power by UTC hour over the weekday or weekend days of `first..last`, as one line."""
    flags = _weekday_flags(first_day=first, count=last - first + 1)
    days = [first + k for k in range(last - first + 1) if flags[k] == weekend]
    if not days:
        return "n/a"
    grid = np.stack([power_site[d * HALF_HOURS : (d + 1) * HALF_HOURS] for d in days]) * 100.0
    hours = grid.reshape(len(days), 24, 2)
    with np.errstate(all="ignore"):
        finite = np.where(np.isfinite(hours), hours, np.nan)
        counts = np.isfinite(finite).sum(axis=(0, 2))
        sums = np.nansum(finite, axis=(0, 2))
    means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return " ".join(_fmt(value=float(m), places=1) for m in means)


def _profiles(*, power_site: np.ndarray, candidate: Candidate, day0: int) -> list[str]:
    """Table 3: mean power by hour, weekdays and weekend days apart, around the onset."""
    out = [
        "",
        "TABLE 3. Mean power by hour (24 values, hour 0 to 23), percent of capacity.",
        "Blocks before the estimated onset are the 8 weeks that end the day before it;",
        "blocks since the onset are consecutive 7-day blocks from the onset day to day 0.",
    ]
    onset = candidate.onset
    blocks: list[tuple[str, int, int]] = [
        (f"before onset, week -{k}", onset - 7 * k, onset - 7 * k + 6)
        for k in range(PROFILE_WEEKS_BEFORE, 0, -1)
    ]
    post = 0
    first = onset
    while first <= day0:
        post += 1
        blocks.append((f"since onset, week +{post}", first, min(first + 6, day0)))
        first += 7
    for name, first, last in blocks:
        if first < 0:
            continue
        weekdays = _hourly(power_site=power_site, first=first, last=last, weekend=False)
        weekends = _hourly(power_site=power_site, first=first, last=last, weekend=True)
        out.append(f"{name}, weekdays: {weekdays}")
        out.append(f"{name}, weekend days: {weekends}")
    return out


def _night_and_mean(*, power_site: np.ndarray, n_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the daily mean power and the night-band mean power, as percent of capacity."""
    grid = power_site[: n_days * HALF_HOURS].reshape(n_days, HALF_HOURS) * 100.0
    night = features.BAND_HALF_HOURS[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        daily = np.where(np.isfinite(grid).sum(axis=1) >= 36, np.nanmean(grid, axis=1), np.nan)
        night_mean = np.where(
            np.isfinite(grid[:, night]).sum(axis=1) >= len(night) / 2,
            np.nanmean(grid[:, night], axis=1),
            np.nan,
        )
    return daily, night_mean


def _neighbours(
    *, inputs: Inputs, power: np.ndarray, scale: np.ndarray, site: int, day0: int
) -> list[str]:
    """Table 4: weekly and daily mean and night power of up to 3 neighbours, and the sum."""
    members = inputs.panel.neighbours[site][:MAX_NEIGHBOURS]
    out = ["", "TABLE 4. Neighbouring substations (mean power and night-band power):"]
    if not members:
        out.append("no metered neighbour")
        return out
    n_days = day0 + 1
    capacity = inputs.panel.capacity_mw * scale
    all_members = [site, *members]
    daily = {m: _night_and_mean(power_site=power[m], n_days=n_days) for m in all_members}
    weights = capacity[all_members] / capacity[all_members].sum()
    summed = sum(w * daily[m][0] for w, m in zip(weights, all_members, strict=True))
    labels = {site: "own", **{m: f"N{i + 1}" for i, m in enumerate(members)}}
    out.append("Weekly, weekday mean / night mean (week -16 to week 0):")
    for m in all_members:
        cells = []
        for k in range(WEEKS - 1, -1, -1):
            stop = day0 + 1 - 7 * k
            if stop - 7 < 0:
                cells.append("n/a")
                continue
            wd, _ = _block_means(series=daily[m][0], start=stop - 7, stop=stop)
            wn, _ = _block_means(series=daily[m][1], start=stop - 7, stop=stop)
            cells.append(f"{_fmt(value=wd)}/{_fmt(value=wn)}")
        out.append(f"{labels[m]}: " + " ".join(cells))
    cells = []
    for k in range(WEEKS - 1, -1, -1):
        stop = day0 + 1 - 7 * k
        wd = (
            _block_means(series=np.asarray(summed), start=stop - 7, stop=stop)[0]
            if stop - 7 >= 0
            else float("nan")
        )
        cells.append(_fmt(value=wd))
    out.append("capacity-weighted mean of own and neighbours, weekday mean: " + " ".join(cells))
    out.append("Daily, mean / night mean (day -27 to day 0):")
    for m in all_members:
        cells = [
            f"{_fmt(value=float(daily[m][0][day0 + o]))}/{_fmt(value=float(daily[m][1][day0 + o]))}"
            for o in range(-(DAILY_DAYS - 1), 1)
        ]
        out.append(f"{labels[m]}: " + " ".join(cells))
    cells = [_fmt(value=float(summed[day0 + o])) for o in range(-(DAILY_DAYS - 1), 1)]
    out.append("capacity-weighted mean of own and neighbours: " + " ".join(cells))
    return out


def _weather(*, inputs: Inputs, site: int, day0: int) -> list[str]:
    """Table 5: ERA5 midday shortwave (W/m2) and temperature (deg C), weekly and daily."""
    usable = day0 - ERA5_LAG_DAYS
    sw = inputs.shortwave_midday[site, : day0 + 1].copy()
    temp = inputs.temperature[site, : day0 + 1].copy()
    sw[usable + 1 :] = np.nan
    temp[usable + 1 :] = np.nan
    out = ["", "TABLE 5. Weather (reanalysis; the last 6 days are not available):"]
    cells_sw, cells_t = [], []
    for k in range(WEEKS - 1, -1, -1):
        stop = day0 + 1 - 7 * k
        start = stop - 7
        cells_sw.append("n/a" if start < 0 else _fmt(value=_mean(values=sw[start:stop]), places=0))
        cells_t.append("n/a" if start < 0 else _fmt(value=_mean(values=temp[start:stop])))
    out.append("Weekly mean midday shortwave, W/m2 (week -16 to week 0): " + " ".join(cells_sw))
    out.append("Weekly mean temperature, deg C (week -16 to week 0): " + " ".join(cells_t))
    days = range(-(DAILY_DAYS - 1), 1)
    out.append(
        "Daily midday shortwave, W/m2 (day -27 to day 0): "
        + " ".join(_fmt(value=float(sw[day0 + o]), places=0) for o in days)
    )
    out.append(
        "Daily temperature, deg C (day -27 to day 0): "
        + " ".join(_fmt(value=float(temp[day0 + o])) for o in days)
    )
    return out


def _evidence(*, check: Check) -> list[str]:
    """Table 6: the detector, the days since the alarm, and the episode state on the check day."""
    candidate, day0 = check.candidate, check.day
    detector = "change-point detector" if candidate.detector == "bocpd" else "year-on-year detector"
    out = ["", "TABLE 6. Detector evidence:"]
    out.append(f"alarm raised by: {detector}")
    if candidate.stat_alarm is not None and candidate.stat_alarm <= day0:
        out.append("the year-on-year detector also alarmed inside this episode")
    out.append(f"days since the alarm: {day0 - candidate.alarm}")
    out.append(f"estimated onset: day {candidate.onset - day0}")
    closed_known = candidate.end + 1 <= day0
    state = f"closed, ended on day {candidate.end - day0}" if closed_known else "open"
    out.append(f"detector episode on day 0: {state}")
    return out


def prompt_of(*, inputs: Inputs, check: Check) -> Prompt:
    """Build the prompt of one check.

    Args:
        inputs: The real record.
        check: The check.

    Returns:
        The prompt, with no image.
    """
    text = build_prompt_text(inputs=inputs, check=check)
    if re.search(r"\bD\d+\b", text) or re.search(r"\b20[12]\d\b", text):
        msg = "the prompt holds a Primary name or a year"
        raise ValueError(msg)
    return Prompt(system=SYSTEM_PROMPT, text=text, images=())


def perturbed(*, inputs: Inputs, day: int) -> Inputs:
    """Return the inputs with every reading after the end of `day` replaced by a large value."""
    power = inputs.panel.power.copy()
    power[:, (day + 1) * HALF_HOURS :] = 9.99
    shortwave = inputs.shortwave_midday.copy()
    temperature = inputs.temperature.copy()
    shortwave[:, day + 1 :] = 9999.0
    temperature[:, day + 1 :] = 99.0
    return Inputs(
        panel=inputs.panel.with_power(power=power),
        shortwave_midday=shortwave,
        temperature=temperature,
        last_day=inputs.last_day,
    )


def future_hidden(*, candidate: Candidate, day: int) -> Candidate:
    """Return the candidate with every fact that is not yet known on `day` changed.

    A statistics alarm after `day` moves far into the future, and an end day that is not yet known
    on `day` (it is known from the day after it) moves far into the future too.

    Args:
        candidate: The candidate.
        day: The check day.

    Returns:
        The changed candidate.
    """
    stat_alarm = candidate.stat_alarm
    if stat_alarm is not None and stat_alarm > day:
        stat_alarm = day + 100
    end = candidate.end if candidate.end + 1 <= day else day + 500
    return replace(candidate, stat_alarm=stat_alarm, end=end)


def assert_cut_changes_nothing(*, inputs: Inputs, check: Check, prompt: Prompt) -> None:
    """Check that no reading and no candidate field from after the check day reaches the prompt.

    The prompt must stay byte for byte unchanged when every later reading is overwritten and when
    every candidate fact not yet known on the check day is changed. The own features computed on
    the full record must equal those computed on the record cut at the check day, both divided by
    the capacity scale known on the check day.

    Args:
        inputs: The real record.
        check: The check.
        prompt: The prompt built from the real record.

    Raises:
        ValueError: If a later reading or a later candidate fact changes the prompt.
    """
    other = prompt_of(inputs=perturbed(inputs=inputs, day=check.day), check=check)
    if other.text != prompt.text:
        msg = f"a later reading changed the prompt of {check.call_id}"
        raise ValueError(msg)
    hidden = replace(check, candidate=future_hidden(candidate=check.candidate, day=check.day))
    if prompt_of(inputs=inputs, check=hidden).text != prompt.text:
        msg = f"a later candidate fact changed the prompt of {check.call_id}"
        raise ValueError(msg)
    stat_alarm = check.candidate.stat_alarm
    shows_stat = "also alarmed" in prompt.text
    if shows_stat != (stat_alarm is not None and stat_alarm <= check.day):
        msg = f"the statistics alarm line of {check.call_id} does not follow the alarm day"
        raise ValueError(msg)
    site = check.candidate.site
    n_half_hours = (check.day + 1) * HALF_HOURS
    scale = capacity_scale(power=inputs.panel.power[:, :n_half_hours])[site]
    full = features.compute_site_features(power_site=inputs.panel.power[site] / scale).raw
    cut = features.compute_site_features(power_site=inputs.panel.power[site, :n_half_hours] / scale)
    if not np.array_equal(full[: check.day + 1], cut.raw, equal_nan=True):
        msg = "the features read data after the check day"
        raise ValueError(msg)


def sha256(text: str) -> str:
    """Return the hexadecimal SHA-256 of a string."""
    return hashlib.sha256(text.encode()).hexdigest()


def write_prompts(*, prompts: dict[str, Prompt]) -> None:
    """Store every prompt as a text file in the call store folder."""
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    for call_id, prompt in prompts.items():
        (PROMPT_DIR / f"{call_id}.txt").write_text(prompt.text)
    (PROMPT_DIR / "system_prompt.txt").write_text(SYSTEM_PROMPT)


def parse_label(*, text: str) -> JsonObject:
    """Validate a reply against the output schema of design section 2.3.

    Args:
        text: The reply text.

    Returns:
        The parsed object.

    Raises:
        ValueError: If the text is not one JSON object that satisfies the schema.
    """
    body = text.strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body)
    obj = json.loads(body)
    if not isinstance(obj, dict) or set(obj) != SCHEMA_KEYS:
        msg = f"keys {sorted(obj) if isinstance(obj, dict) else type(obj)} do not match the schema"
        raise ValueError(msg)
    checks = (
        obj["label"] in LABELS,
        isinstance(obj["onset_offset_days"], int) and obj["onset_offset_days"] <= 0,
        isinstance(obj["confidence"], int | float) and 0 <= obj["confidence"] <= 1,
        obj["neighbour_mirror"] in MIRRORS,
        obj["shape"] in SHAPES,
        obj["register_row"] is None or isinstance(obj["register_row"], int),
        isinstance(obj["reason"], str) and len(obj["reason"].split()) <= 60,
    )
    if not all(checks):
        msg = f"a field breaks the schema: {obj}"
        raise ValueError(msg)
    return obj


def run_claude(*, prompt: Prompt) -> RawRun:
    """Run the command-line tool once, keeping its output even when it exits with an error.

    Args:
        prompt: The prompt.

    Returns:
        The raw run.

    Raises:
        ModelError: On a timeout.
    """
    model = ClaudeCliModel(model_id=MODEL_ID)
    environment = {**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="labeller_empty_") as empty:
        try:
            done = subprocess.run(
                model.command(prompt=prompt),
                input=stream_message(prompt=prompt),
                capture_output=True,
                text=True,
                cwd=empty,
                env=environment,
                timeout=CALL_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            msg = f"the model call timed out after {CALL_TIMEOUT_SECONDS} s"
            raise ModelError(msg) from error
    return RawRun(
        stdout=done.stdout,
        stderr=done.stderr,
        returncode=done.returncode,
        seconds=time.monotonic() - started,
        directory=empty,
    )


def stream_events(*, stdout: str) -> list[JsonObject]:
    """Parse the `stream-json` lines that are valid JSON objects."""
    events: list[JsonObject] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def rate_limit_payloads(*, events: Sequence[JsonObject]) -> list[JsonObject]:
    """Return the raw payload of every `rate_limit_event`."""
    return [e for e in events if e.get("type") == "rate_limit_event"]


def rate_limit_hit(*, events: Sequence[JsonObject], run: RawRun) -> str | None:
    """Return a description of a rate-limit message in a run, or `None`.

    A `rate_limit_event` whose status is anything but `allowed`, an error result, or an error
    text that matches the rate-limit pattern counts.

    Args:
        events: The parsed events.
        run: The raw run.

    Returns:
        A short description, or `None`.
    """
    for event in rate_limit_payloads(events=events):
        info = event.get("rate_limit_info")
        status = info.get("status") if isinstance(info, dict) else None
        if status != "allowed":
            kind = info.get("rateLimitType") if isinstance(info, dict) else None
            reset = info.get("resetsAt") if isinstance(info, dict) else None
            return f"rate_limit_event status {status!r} type {kind!r} resetsAt {reset!r}"
    error_text = run.stderr
    for event in events:
        if event.get("type") == "result" and event.get("is_error"):
            error_text += " " + str(event.get("result", ""))
    failed = run.returncode != 0 or any(
        e.get("type") == "result" and e.get("is_error") for e in events
    )
    if failed and RATE_LIMIT_PATTERN.search(error_text):
        return f"error text: {error_text[:300]}"
    return None


def read_log() -> list[JsonObject]:
    """Return every entry of the trial log, or an empty list if there is none."""
    if not LOG_PATH.exists():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]


class Trial:
    """Runs the calls, with the guard, the call store, the stop flag, and the log.

    A stop cannot interrupt a call that the other worker already has in flight, which can last up
    to `CALL_TIMEOUT_SECONDS`; it only prevents every later call.
    """

    def __init__(self, *, resume: bool) -> None:
        """Create the store, the guard, and the stop flag.

        Args:
            resume: Whether to continue from the existing log. The guard then starts from the
                log's totals under the explicit limits, and finished calls are skipped.
        """
        self.store = CallStore(directory=CALL_DIR, log_path=LOG_PATH)
        self.guard = BudgetGuard.from_log(log_path=LOG_PATH) if resume else BudgetGuard()
        self.guard.ceiling_usd = MAX_COST_USD
        self.guard.max_calls = MAX_CALLS
        self.guard.max_input_tokens = MAX_INPUT_TOKENS
        earlier = read_log() if resume else []
        self.attempts: dict[str, int] = {}
        self.finished: set[str] = set()
        for entry in earlier:
            call_id = str(entry["call_id"])
            self.attempts[call_id] = self.attempts.get(call_id, 0) + 1
            if entry.get("status") == "ok" and (
                entry["tag"] == "billing-probe" or entry.get("parsed_label") is not None
            ):
                self.finished.add(call_id)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.done = len(earlier)
        self.output_tokens = sum(int(e["output_tokens"]) for e in earlier)
        sources = [e["apiKeySource"] for e in earlier if e.get("apiKeySource") is not None]
        self.api_key_source = str(sources[-1]) if sources else "unknown"
        self.cli_version = ""

    def _log(self, *, entry: JsonObject) -> None:
        """Append one entry to the log under the lock."""
        with self.lock:
            self.store.append_log(entry=entry)

    def call(self, *, call_id: str, tag: str, offset: int | None, prompt: Prompt) -> None:  # noqa: PLR0915 - one linear sequence of logged steps
        """Make one call, with one retry on an invalid reply, and log it.

        Any error sets the stop flag before it propagates, so the other worker takes no new call.

        Args:
            call_id: A date-free identifier.
            tag: `billing-probe` or `trial`.
            offset: The check day as an offset from the alarm day, or `None` for the probe.
            prompt: The prompt.

        Raises:
            RateLimited: On a rate-limit message.
            BudgetExceeded: If the guard refuses.
            ModelError: If the call fails or times out.
        """
        if call_id in self.finished:
            return
        for attempt in range(self.attempts.get(call_id, 0), MAX_ATTEMPTS):
            if self.stop.is_set():
                return
            entry: JsonObject = {
                "call_id": call_id,
                "attempt": attempt,
                "tag": tag,
                "check_offset_days": offset,
                "model": MODEL_ID,
                "claude_code_version": self.cli_version,
                "prompt_sha256": sha256(text=prompt.text),
                "prompt": prompt.text,
                "estimated_input_tokens": estimate_input_tokens(prompt=prompt),
            }
            try:
                with self.lock:
                    self.guard.check(prompt=prompt)
                run = run_claude(prompt=prompt)
            except BudgetExceeded as error:
                self._trip(reason=f"budget: {error}")
                raise
            except ModelError as error:
                billed = self._cap_cost(prompt=prompt)
                self._finish(entry=entry, reply=billed, events=[])
                entry |= {"status": "failed", "error": str(error)[:500], "wall_seconds": None}
                self._log(entry=entry)
                self._trip(reason=f"timeout or launch failure: {error}")
                raise
            events = stream_events(stdout=run.stdout)
            limit = rate_limit_hit(events=events, run=run)
            init = next((e for e in events if e.get("type") == "system" and "tools" in e), {})
            result = next((e for e in events if e.get("type") == "result"), {})
            entry |= {
                "apiKeySource": init.get("apiKeySource"),
                "init_event": init,
                "result_event": {k: v for k, v in result.items() if k != "result"},
                "rate_limit_events": rate_limit_payloads(events=events),
                "wall_seconds": round(run.seconds, 2),
            }
            try:
                reply = parse_stream(
                    output=run.stdout,
                    requested_model=MODEL_ID,
                    expected_cwd=Path(run.directory),
                )
            except ModelError as error:
                billed = error.reply or self._cap_cost(prompt=prompt)
                self._finish(entry=entry, reply=billed, events=events)
                entry |= {"status": "failed", "error": str(error)[:500], "raw_output": run.stdout}
                self._log(entry=entry)
                self._trip(reason=limit or f"model error: {error}")
                if limit:
                    raise RateLimited(limit) from error
                raise
            self._finish(entry=entry, reply=reply, events=events)
            entry |= {"status": "ok", "raw_output": reply.text}
            if limit:
                self._log(entry=entry)
                self._trip(reason=limit)
                raise RateLimited(limit)
            if tag == "billing-probe":
                entry["parsed_label"] = None
                self._log(entry=entry)
                return
            # The parsed label only checks that the reply fits the schema. No code reads it for
            # any other purpose, and the trial's replies are never labels and never scored.
            try:
                entry["parsed_label"] = parse_label(text=reply.text)
            except (ValueError, json.JSONDecodeError) as error:
                entry |= {"parsed_label": None, "parse_error": str(error)[:300]}
                self._log(entry=entry)
                continue
            self._log(entry=entry)
            return

    @staticmethod
    def _cap_cost(*, prompt: Prompt) -> ModelReply:
        """Return the placeholder reply of a call whose bill is unknown, costed at the cap."""
        return ModelReply(
            text="",
            input_tokens=estimate_input_tokens(prompt=prompt),
            output_tokens=0,
            cost_usd=PER_CALL_BUDGET_USD,
            model=MODEL_ID,
        )

    def _finish(
        self,
        *,
        entry: JsonObject,
        reply: ModelReply,
        events: Sequence[JsonObject],
    ) -> None:
        """Add the token counts and the cost to a log entry and to the running totals."""
        result = next((e for e in events if e.get("type") == "result"), {})
        usage: JsonObject = result.get("usage") or {}
        entry |= {
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
            "uncached_input_tokens": usage.get("input_tokens"),
            "cache_creation_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_tokens": usage.get("cache_read_input_tokens"),
            "cost_usd": reply.cost_usd,
            "reported_total_cost_usd": result.get("total_cost_usd"),
            "cost_is_placeholder": not events,
        }
        with self.lock:
            self.guard.record(reply=reply)
            self.done += 1
            self.output_tokens += reply.output_tokens
            if entry.get("apiKeySource") is not None:
                self.api_key_source = str(entry["apiKeySource"])
            if self.done % PRINT_EVERY == 0:
                print(self.totals_line(), flush=True)

    def totals_line(self) -> str:
        """One line of running totals."""
        g = self.guard
        return (
            f"running totals: calls={g.calls} input_tokens={g.input_tokens} "
            f"output_tokens={self.output_tokens} reported_cost_usd={g.spent_usd:.3f}"
        )

    def _trip(self, *, reason: str) -> None:
        """Set the stop flag and write what was done, for every kind of stop."""
        self.stop.set()
        STOP_PATH.write_text(json.dumps({"reason": reason, "totals": self.totals_line()}, indent=2))


def cli_version() -> str:
    """Return the output of `claude --version`, or an empty string if it cannot be read."""
    try:
        done = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=60, check=False
        )
    except OSError, subprocess.TimeoutExpired:
        return ""
    return done.stdout.strip()


def run_trial(
    *,
    checks: list[Check],
    prompts: dict[str, Prompt],
    resume: bool,
    expect_key_source: str | None,
) -> int:
    """Send the billing probe, check its route, then every check prompt, 2 at a time at most.

    Args:
        checks: The checks, in order; empty for a probe-only run.
        prompts: The prompt of each check by call id.
        resume: Whether to continue from the existing log.
        expect_key_source: The `apiKeySource` the probe must report before any check is sent, or
            `None` to stop after the probe.

    Returns:
        The exit status: 0 if all calls ran, 2 on a rate limit, 3 on any other stop, 4 if the
        probe's key source differs from the expected one.
    """
    if resume and STOP_PATH.exists():
        STOP_PATH.rename(STOP_PATH.with_name(f"stopped_before_resume_{int(time.time())}.json"))
    trial = Trial(resume=resume)
    trial.cli_version = cli_version()
    probe = Prompt(system=PROBE_SYSTEM, text=PROBE_TEXT, images=())
    try:
        trial.call(call_id="billing_probe", tag="billing-probe", offset=None, prompt=probe)
    except RateLimited as error:
        print(f"RATE LIMIT on the billing probe: {error}", file=sys.stderr)
        return 2
    except (BudgetExceeded, ModelError) as error:
        print(f"STOP on the billing probe: {error}", file=sys.stderr)
        return 3
    print(f"billing probe: apiKeySource = {trial.api_key_source!r}", flush=True)
    if expect_key_source is None:
        print("probe only: no check call made")
        return 0
    if trial.api_key_source != expect_key_source:
        print(
            f"ABORT: expected apiKeySource {expect_key_source!r}, the probe reports "
            f"{trial.api_key_source!r}; no check call made",
            file=sys.stderr,
        )
        return 4
    status = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(
                trial.call,
                call_id=check.call_id,
                tag="trial",
                offset=check.offset,
                prompt=prompts[check.call_id],
            )
            for check in checks
        ]
        for future in futures:
            try:
                future.result()
            except RateLimited as error:
                print(f"RATE LIMIT, stopping: {error}", file=sys.stderr)
                status = 2
            except (BudgetExceeded, ModelError) as error:
                print(f"STOP: {error}", file=sys.stderr)
                status = status or 3
    print(trial.totals_line())
    if status:
        print(f"stopped early, exit status {status}; see {STOP_PATH}", file=sys.stderr)
    return status


def _median(*, values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def print_summary() -> None:  # noqa: PLR0915 - one printed report
    """Print totals, per-call medians, and the extrapolation from the trial log."""
    if not LOG_PATH.exists():
        print("no trial log yet")
        return
    seen: set[tuple[str, int]] = set()
    entries: list[JsonObject] = []
    for entry in read_log():
        key = (str(entry["call_id"]), int(entry["attempt"]))
        if key not in seen:
            seen.add(key)
            entries.append(entry)
    ok = [e for e in entries if e.get("status") == "ok"]
    trial = [e for e in ok if e["tag"] == "trial"]
    all_input = sum(e["input_tokens"] for e in entries)
    all_output = sum(e["output_tokens"] for e in entries)
    all_cost = sum(e["cost_usd"] for e in entries)
    wall = sum(e["wall_seconds"] for e in entries)
    print(f"calls logged (probe, retries, failures included): {len(entries)}")
    print(f"  of which ok trial calls: {len(trial)}")
    failures = sum(1 for e in entries if e.get("status") == "failed")
    parse_failures = sum(1 for e in trial if e.get("parsed_label") is None)
    placeholders = sum(1 for e in entries if e.get("cost_is_placeholder"))
    print(f"  failed calls: {failures} (costed at the placeholder cap: {placeholders}); replies")
    print(f"  that fail the schema: {parse_failures}. Labels are not read or printed here.")
    print(f"input tokens total: {all_input:,}   output tokens total: {all_output:,}")
    print(f"reported cost total: US${all_cost:.3f}   summed wall time: {wall:.0f} s")
    sources = sorted({str(e.get("apiKeySource")) for e in entries})
    print(f"apiKeySource values seen: {sources}")
    if not trial:
        return
    med_in = _median(values=[e["input_tokens"] for e in trial])
    med_out = _median(values=[e["output_tokens"] for e in trial])
    med_wall = _median(values=[e["wall_seconds"] for e in trial])
    mean_in = statistics.fmean(e["input_tokens"] for e in trial)
    mean_out = statistics.fmean(e["output_tokens"] for e in trial)
    mean_cost = statistics.fmean(e["cost_usd"] for e in trial)
    est_ratio = statistics.fmean(e["input_tokens"] / e["estimated_input_tokens"] for e in trial)
    print(
        f"per trial call, median: input {med_in:,.0f}, output {med_out:,.0f}, "
        f"wall {med_wall:.1f} s; mean cost US${mean_cost:.4f}; "
        f"tokens counted / estimated at 1.4 characters per token: {est_ratio:.2f}"
    )
    labelled = {e["call_id"].rsplit("_", 1)[0] for e in trial}
    calls_per_candidate = len(trial) / max(len(labelled), 1)
    print(f"trial calls per candidate (no early stop, retries included): {calls_per_candidate:.2f}")
    api_ms = [
        e["result_event"]["duration_api_ms"]
        for e in trial
        if isinstance(e.get("result_event"), dict) and "duration_api_ms" in e["result_event"]
    ]
    if api_ms:
        print(f"median duration_api_ms per trial call: {_median(values=api_ms):,.0f}")
    print_plan_quota(entries=entries)
    print(
        "\nExtrapolation. Assumptions: every future call costs the trial mean (input "
        f"{mean_in:,.0f}, output {mean_out:,.0f} tokens, US${mean_cost:.4f});"
    )
    print("candidates per Primary-year 0.8 (design 6.3); 16 Primaries hold 115 Primary-years;")
    print(
        "2,500 substations are assumed to have the same 7.2 years of record and the same candidate"
    )
    print("rate, with the same number of neighbours and the same weather source; bank prompts are")
    print("assumed to cost the same as real-record prompts. Early stopping would lower the counts.")
    print("The trial prompt lacks the 13 evidence columns, the ensemble weather cells, and the")
    print("register rows of arm llm-ecr, so all three would add tokens.")
    real_candidates = CANDIDATES_PER_PRIMARY_YEAR * PRIMARY_YEARS
    backfill = SUBSTATIONS * RECORD_YEARS_PER_SUBSTATION * CANDIDATES_PER_PRIMARY_YEAR
    per_year = SUBSTATIONS * CANDIDATES_PER_PRIMARY_YEAR
    scenarios = [
        ("16 Primaries, trial schedule", real_candidates * calls_per_candidate),
        ("16 Primaries, design plan (245 calls)", float(PLANNED_REAL_CALLS)),
        (
            f"bank block ({BANK_CANDIDATES} candidates, at most {BANK_MAX_CALLS} calls)",
            float(BANK_MAX_CALLS),
        ),
        ("2,500 substations, one-off backfill, trial schedule", backfill * calls_per_candidate),
        (
            "2,500 substations, per year of operation, trial schedule",
            per_year * calls_per_candidate,
        ),
    ]
    print(f"{'scenario':<62}{'calls':>9}{'input Mtok':>12}{'output Mtok':>13}{'cost US$':>11}")
    for name, calls in scenarios:
        print(
            f"{name:<62}{calls:>9,.0f}{calls * mean_in / 1e6:>12.2f}"
            f"{calls * mean_out / 1e6:>13.2f}{calls * mean_cost:>11.2f}"
        )


def _utilization(*, info: JsonObject, kind: str) -> object:
    """Return the utilisation of one rate-limit window.

    The value sits under `unifiedWindows.<kind>.utilization`, with `utilization` directly in
    `rate_limit_info` as a fallback.
    """
    windows = info.get("unifiedWindows")
    if isinstance(windows, dict):
        window = windows.get(kind)
        if isinstance(window, dict) and "utilization" in window:
            return window["utilization"]
    return info.get("utilization")


def print_plan_quota(*, entries: Sequence[JsonObject]) -> None:
    """Print what the logged rate-limit events show about plan-quota use, and what they cannot."""
    print("plan quota: the command-line tool reports no plan-quota use per call.")
    print("  Under a subscription (apiKeySource = none) total_cost_usd is an API-equivalent price,")
    print("  not a charge.")
    print(
        "  Any utilization field is an account-wide snapshot that every concurrent session moves,"
    )
    print(
        "  so this trial's share can be attributed only if other sessions are idle during the run."
    )
    payloads = [p for e in entries for p in e.get("rate_limit_events", [])]
    infos = [p["rate_limit_info"] for p in payloads if isinstance(p.get("rate_limit_info"), dict)]
    if not infos:
        print("  No rate_limit_event appeared in any call, so no counter is observable.")
        return
    by_type: dict[str, list[JsonObject]] = {}
    for info in infos:
        by_type.setdefault(str(info.get("rateLimitType")), []).append(info)
    for kind, group in by_type.items():
        first, last = group[0], group[-1]
        first_use = _utilization(info=first, kind=kind)
        print(
            f"  {kind}: {len(group)} events; utilization first {first_use!r}, "
            f"last {_utilization(info=last, kind=kind)!r}; resetsAt {last.get('resetsAt')!r}; "
            f"status last {last.get('status')!r}"
        )
    seen = sorted(
        {
            (kind, value)
            for kind, group in by_type.items()
            for value in (_utilization(info=i, kind=kind) for i in group)
            if value is not None
        },
        key=str,
    )
    print(f"  Distinct (window, utilization) readings across all calls: {seen}")
    print(
        "  Each utilization is ACCOUNT-WIDE, shared by every session on the account, so the share "
        "of this trial is below the resolution of the field and cannot be attributed."
    )
    print(
        f"  isUsingOverage values: {sorted({str(i.get('isUsingOverage')) for i in infos})}; "
        f"overageStatus values: {sorted({str(i.get('overageStatus')) for i in infos})}"
    )
    if not any(_utilization(info=i, kind=str(i.get("rateLimitType"))) is not None for i in infos):
        print("  The payloads hold a status and reset time only: no usage counter is observable.")


def print_candidate_table(*, candidates: list[Candidate], chosen: int) -> None:
    """Print the union candidates of the chosen Primary with their dates."""
    print(f"Candidates of {features.PRIMARIES[chosen]} (private; dates never reach a prompt):")
    print("Primary  alarm date   detector  episode id  joined")
    for c in candidates:
        if c.site == chosen:
            print(
                f"{features.PRIMARIES[c.site]:<8} {features.day_of(index=c.alarm)}  "
                f"{c.detector:<8}  {c.episode_id:<10}  {c.stat_alarm is not None}"
            )


def print_counts_by_primary(*, candidates: list[Candidate]) -> None:
    """Print how many union candidates each Primary has."""
    counts = pl.Series([features.PRIMARIES[c.site] for c in candidates]).value_counts()
    line = ", ".join(f"{p}={n}" for p, n in sorted(counts.rows(), key=lambda r: r[0]))
    print(f"Union candidates by Primary: {line}")


def main() -> int:  # noqa: PLR0911 - one exit status per refusal
    """List the candidates, build prompts, probe the billing route, or run the trial.

    With no flag the script only lists the candidates. Spending needs `--run` with
    `--expect-key-source`, or `--probe-only`.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--primary", default=DEFAULT_PRIMARY, help="Primary code, e.g. D10")
    parser.add_argument("--list-candidates", action="store_true", help="print candidates (default)")
    parser.add_argument("--dry-run", action="store_true", help="build prompts, make no call")
    parser.add_argument("--probe-only", action="store_true", help="send the billing probe only")
    parser.add_argument("--run", action="store_true", help="probe, then send the check prompts")
    parser.add_argument("--expect-key-source", help="apiKeySource the probe must report for --run")
    parser.add_argument("--resume", action="store_true", help="continue from the existing log")
    parser.add_argument("--summary", action="store_true", help="summarise the trial log and exit")
    args = parser.parse_args()
    if args.summary:
        print_summary()
        return 0
    if args.primary in FORBIDDEN_PRIMARIES:
        print(f"{args.primary} is a known answer; the design forbids calls on it", file=sys.stderr)
        return 1
    if args.run and not args.expect_key_source:
        print(
            "--run needs --expect-key-source (run --probe-only first to learn it)", file=sys.stderr
        )
        return 1
    spends = args.run or args.probe_only
    if spends and not args.resume and (LOG_PATH.exists() or STOP_PATH.exists()):
        print(
            f"{LOG_PATH} or {STOP_PATH} exists; pass --resume to continue, which skips finished "
            "calls and counts earlier spending, or move the folder away",
            file=sys.stderr,
        )
        return 1
    site = features.PRIMARIES.index(args.primary)
    candidates = load_candidates()
    print_counts_by_primary(candidates=candidates)
    print_candidate_table(candidates=candidates, chosen=site)
    if args.probe_only:
        TRIAL_DIR.mkdir(parents=True, exist_ok=True)
        return run_trial(checks=[], prompts={}, resume=args.resume, expect_key_source=None)
    if not (args.dry_run or args.run):
        return 0
    inputs = load_inputs()
    checks = [
        check
        for candidate in candidates
        if candidate.site == site
        for check in checks_of(candidate=candidate, last_day=inputs.last_day)
    ]
    prompts: dict[str, Prompt] = {}
    for check in checks:
        prompt = prompt_of(inputs=inputs, check=check)
        assert_cut_changes_nothing(inputs=inputs, check=check, prompt=prompt)
        prompts[check.call_id] = prompt
        print(
            f"{check.call_id:<22} offset {check.offset:>4}  "
            f"estimated input tokens {estimate_input_tokens(prompt=prompt):>6,}  "
            f"characters {len(prompt.system) + len(prompt.text):>7,}"
        )
    print(f"planned: 1 billing probe + {len(checks)} check calls; cut checks passed")
    if len(checks) + 1 > MAX_CALLS - RETRIES_RESERVED:
        print("too many calls for the trial limit", file=sys.stderr)
        return 1
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    write_prompts(prompts=prompts)
    if args.dry_run:
        print(f"dry run: prompts written to {PROMPT_DIR}; no call made")
        return 0
    return run_trial(
        checks=checks,
        prompts=prompts,
        resume=args.resume,
        expect_key_source=args.expect_key_source,
    )


if __name__ == "__main__":
    sys.exit(main())

"""Experiment 712, stage 1: the candidates, the checks, and the prompt builder of the labelling run.

**This module builds prompts and makes no call.** It is a copy of the prompt builder of
`e00712_trial.py` (the cost trial, a published script that stays unedited), changed in the ways the
code review of the trial (`briefs/clean_era_labels/trial_code_review.md`) asked for the labelling
run: the prompt holds no calendar key, the capacity scale uses only readings up to the check day,
the closure day follows the way the detector's episode closed, and the register rows of arm
`llm-ecr` can be added as a seventh table.

**The prompt shows numbers about one unnamed substation, with every time as an offset from the
check day.** It holds no name, date, year, month, weekday, bank-holiday flag, or megawatt value of
the substation's demand. Every table is built from a copy of the inputs cut at the end of the check
day, and `assert_cut_changes_nothing` asserts that perturbing every later reading, every candidate
fact not yet known, and every register row not yet visible changes no prompt.

**Calendar keys.** Every table is built to carry no weekday, no day of the month, no clock change,
and no month boundary. Weekly tables give one mean over all 7 days of each block. The daily tables
give the mean of the 7 days that end on each row, so the weekly cycle cancels. The six residual
features are computed against one climatology for the whole prompt, the one of the calendar month of
day 0, so the residual has no step on the 1st of a month. The hourly profiles use Europe/London
clock hours, so the demand ramps keep their clock position when the clock changes. **Calendar keys
that remain:** the weather values (ERA5 midday shortwave and temperature, weekly and daily) show the
season to within about a month, and the seasonal drift of the residual against a single climatology
does too. Easter, Christmas, and the May bank holidays still dip the power, smeared over 7 days in
the daily rows and visible in the weekly rows. A block since the onset that is not a whole number of
weeks has an uneven mix of weekdays. The year probe of design section 2.2.1 tests what remains.

**Not implemented against the design.** The 13 evidence columns of `e00709_classify.EVIDENCE_NAMES`
need `stage0.process_hosts` and are replaced by the detector, the days since the alarm, the
estimated onset, and the episode state. The last 6 days of weather come from no source (the ensemble
summaries are not read), so those cells read "not available".
"""

import hashlib
import math
import re
import warnings
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import lru_cache
from typing import Final
from zoneinfo import ZoneInfo

import e00709_features as features  # ty: ignore[unresolved-import]
import numpy as np
from studies.capacity_labeller import Prompt

SCHEDULED_KINDS: Final[tuple[str, ...]] = ("alarm", "plus7", "plus21")
"""The scheduled checks, in order; the offsets from the alarm day are an argument of the run."""
REAL_OFFSETS: Final[tuple[int, ...]] = (0, 7, 21)
"""Offsets of the scheduled checks from the alarm day on the real record (design section 2.4)."""
WEEKS: Final[int] = 17
WINDOW_DAYS: Final[int] = 7 * WEEKS - 1
"""Days before day 0 that Table 1 reaches back to."""
TRAILING_MIN_DAYS: Final[int] = 4
LONDON: Final[ZoneInfo] = ZoneInfo("Europe/London")
DAILY_DAYS: Final[int] = 28
PROFILE_WEEKS_BEFORE: Final[int] = 8
MAX_NEIGHBOURS: Final[int] = 3
ERA5_LAG_DAYS: Final[int] = 6
"""ERA5 is used for days at least this many days before the check day, so the last 6 days (days -5
to 0) read "not available"."""
FEATURE_LABELS: Final[tuple[str, ...]] = ("night", "morning", "midday", "evening", "p05", "p95")
HALF_HOURS: Final[int] = features.HALF_HOURS_PER_DAY
DEFAULT_PUBLICATION_LAG_DAYS: Final[int] = 14
"""A register row is visible this many days after its connection date (design section 3.2)."""
REGISTER_BEFORE_DAYS: Final[int] = 180
REGISTER_AFTER_DAYS: Final[int] = 30

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

SCHEMA_HEAD: Final[str] = """\
Reply with one JSON object with exactly these keys:
  "label": "era", "switch", "noise", or "undecided"
  "onset_offset_days": integer, 0 or below, your estimate of the first day of the shift
  "confidence": number from 0 to 1, your probability that label is right
  "neighbour_mirror": "yes", "no", or "unmetered"
  "shape": "solar", "flat", "demand-like", or "irregular\""""
SCHEMA_NO_REGISTER: Final[str] = (
    SCHEMA_HEAD
    + """
  "register_row": null (no register rows are given in this prompt)
  "reason": string of at most 60 words"""
)
SCHEMA_WITH_REGISTER: Final[str] = (
    SCHEMA_HEAD
    + """
  "register_row": the number of the register row you relied on, or null
  "reason": string of at most 60 words"""
)

BANNED_WORDS: Final[tuple[str, ...]] = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december", "holiday", "christmas", "easter",
)  # fmt: skip
"""Words that would name a calendar position; `May` is left out because it is also a verb."""
FORBIDDEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bD\d+\b|\b(?:19|20)\d\d\b|\b(?:" + "|".join(BANNED_WORDS) + r")\b", re.IGNORECASE
)


@dataclass(frozen=True)
class Candidate:
    """One detector candidate shift.

    Attributes:
        site: Index of the Primary in `features.PRIMARIES`.
        episode_id: Detector name and the episode's position among that detector's episodes.
        detector: `bocpd` (the change-point detector), `stat` (the year-on-year detector at its
            threshold of 4.0), or `stat3` (the same detector at the lower threshold of 3.0).
        onset: Estimated onset day index, fixed on the alarm day.
        alarm: Alarm day index.
        end: End day index of the detector's episode, or -1 while the episode is open.
        ended_by: `return` (the level came back), `superseded` (a later alarm of the same
            detector), or `open`.
        stat_alarm: Day of the first year-on-year statistics alarm inside this change-point
            episode, or `None`. It is shown in a prompt only on or after that day.
    """

    site: int
    episode_id: str
    detector: str
    onset: int
    alarm: int
    end: int
    ended_by: str
    stat_alarm: int | None

    @property
    def closure_day(self) -> int | None:
        """The first day on which the episode's end is known, or `None` while it is open.

        A return is detected from readings up to the end day itself, so it is known at the end of
        the end day. A supersession sets the end to the day before the later alarm, so it is known
        one day after the end day.
        """
        if self.ended_by == "open" or self.end < 0:
            return None
        return self.end if self.ended_by == "return" else self.end + 1

    @property
    def key(self) -> str:
        """A date-free identifier of the candidate."""
        return f"{features.PRIMARIES[self.site]}_{self.episode_id}"


@dataclass(frozen=True)
class Check:
    """One check of one candidate.

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

    def call_id(self, *, arm: str) -> str:
        """Return a date-free identifier of the check for one arm."""
        return f"{arm}:{self.candidate.key}_{self.kind}"


@dataclass(frozen=True)
class RegisterRow:
    """One Embedded Capacity Register row, with only the fields a prompt may show.

    Attributes:
        site: Index of the Primary the row is filed under.
        connected_day: Day index of the connection date.
        size_mw: Registered export capacity in MW.
        technology: `solar`, `wind`, `storage`, `engine`, or `other`.
        distance_band: `under 1 km`, `1 to 3 km`, `over 3 km`, `listed under a neighbouring
            substation`, or `not matched by coordinates`.
        opens_window: Whether the row opens a watch window at `site` (design section 3.2).
    """

    site: int
    connected_day: int
    size_mw: float
    technology: str
    distance_band: str
    opens_window: bool

    def visible_on(self, *, day: int, lag: int = DEFAULT_PUBLICATION_LAG_DAYS) -> bool:
        """Return whether the row is published by the end of `day` under a publication lag."""
        return self.connected_day + lag <= day


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


def sha256(*, text: str) -> str:
    """Return the hexadecimal SHA-256 of a string."""
    return hashlib.sha256(text.encode()).hexdigest()


def _fmt(*, value: float, places: int = 1) -> str:
    """Format a number, or `n/a` for NaN."""
    return "n/a" if not math.isfinite(value) else f"{value:.{places}f}"


def _mean(*, values: np.ndarray) -> float:
    """Mean of the finite values, NaN if there are none."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _trailing(*, series: np.ndarray, day: int) -> float:
    """Mean of the 7 days that end on `day`, NaN with fewer than 4 finite days."""
    part = series[max(day - 6, 0) : day + 1]
    return _mean(values=part) if np.isfinite(part).sum() >= TRAILING_MIN_DAYS else float("nan")


@lru_cache(maxsize=1)
def local_hour_grid() -> np.ndarray:
    """Return the Europe/London clock hour of every half-hour of the grid.

    The grid starts at 00:00 UTC on its first day. The mapping uses the date only inside this
    function, and a prompt shows the hour and never the date.
    """
    return np.array(
        [
            (features.GRID_START + timedelta(minutes=30 * i)).astimezone(LONDON).hour
            for i in range(features.N_HALF_HOURS)
        ],
        dtype=np.int8,
    )


def prompt_features(*, power_site: np.ndarray, day0: int) -> np.ndarray:
    """Return the six daily features of the days before `day0`, against one climatology.

    The climatology is the one of the calendar month of day 0, built as experiment 709 builds it:
    the mean at each half-hour over the days of that month of the year inside the 730 days that end
    28 days before that month starts. It is applied to every day of the prompt, so the residual has
    no step where a calendar month changes. Only readings before the start of day 0's month and the
    readings of the prompt window are read.

    Args:
        power_site: One Primary's power over capacity, from the start of the grid.
        day0: The check day index.

    Returns:
        An array (day0 + 1, 6) of raw features in capacity share, NaN outside the prompt window or
        where no climatology exists.
    """
    n_days = day0 + 1
    grid = power_site[: n_days * HALF_HOURS].reshape(n_days, HALF_HOURS)
    out = np.full((n_days, features.N_FEATURES), np.nan)
    today = features.day_of(index=day0)
    month_start = features.day_index(day=today.replace(day=1))
    high = month_start - features.CLIMATOLOGY_LAG_DAYS
    if high <= 0:
        return out
    low = max(high - features.CLIMATOLOGY_WINDOW_DAYS, 0)
    valid = np.isfinite(grid[low:high]).sum(axis=1) >= features.MIN_READINGS_PER_DAY
    same_month = valid & np.array(
        [features.day_of(index=i).month == today.month for i in range(low, high)]
    )
    if valid.sum() < features.MIN_CLIMATOLOGY_DAYS or same_month.sum() < features.MIN_DAYS_IN_MONTH:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        climatology = np.nanmean(grid[low:high][same_month], axis=0)
    first = max(day0 - WINDOW_DAYS, 0)
    out[first:n_days] = features.daily_features(residual=grid[first:n_days] - climatology[None, :])
    return out


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


def build_prompt_text(
    *, inputs: Inputs, check: Check, register: tuple[RegisterRow, ...] | None = None
) -> str:
    """Build the case text of one check from readings up to the end of the check day.

    Every array is cut to `check.day + 1` days before any table is computed, so no table can read
    a later reading.

    Args:
        inputs: The real record.
        check: The check.
        register: The register rows of arm `llm-ecr`, or `None` for arm `llm` (no seventh table).

    Returns:
        The user text, ending in the output schema.

    Raises:
        ValueError: If the check day is too early for the 28-day tables or the input is too short.
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
    own_features = prompt_features(power_site=power[site], day0=day0)
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
    if register is not None:
        lines.extend(
            _register_table(
                rows=register,
                candidate=candidate,
                day0=day0,
                capacity_mw=_capacity(inputs=inputs, scale=scale, site=site),
            )
        )
    lines.append("")
    lines.append(SCHEMA_NO_REGISTER if register is None else SCHEMA_WITH_REGISTER)
    return "\n".join(lines) + "\n"


def _capacity(*, inputs: Inputs, scale: np.ndarray, site: int) -> float:
    """Return the substation's capacity in MW on the scale known at the check day."""
    return float(inputs.panel.capacity_mw[site] * scale[site])


def _weekly_features(*, raw: np.ndarray, day0: int, onset_offset: int) -> list[str]:
    """Table 1: the mean of each of the six features over each of the 17 weeks."""
    header = "week  " + "  ".join(name.ljust(8) for name in FEATURE_LABELS)
    out = ["", "TABLE 1. Weekly detector features (residual against a climatology):", header]
    for k in range(WEEKS):
        stop = day0 + 1 - 7 * k
        start = stop - 7
        cells = [
            "n/a".ljust(8)
            if start < 0
            else _fmt(value=_mean(values=raw[start:stop, f] * 100.0), places=2).ljust(8)
            for f in range(raw.shape[1])
        ]
        mark = (
            "  <- detector's estimated onset" if start - day0 <= onset_offset < stop - day0 else ""
        )
        out.append(f"{-k:>4}  " + "  ".join(cells) + mark)
    return out


def _daily_features(*, raw: np.ndarray, day0: int) -> list[str]:
    """Table 2: the six features for each of the last 28 days, as 7-day trailing means."""
    out = [
        "",
        "TABLE 2. Detector features, last 28 days (each row: mean of the 7 days ending on it):",
    ]
    out.append("day   " + "  ".join(name.ljust(8) for name in FEATURE_LABELS))
    for offset in range(-(DAILY_DAYS - 1), 1):
        cells = [
            _fmt(value=_trailing(series=raw[:, f] * 100.0, day=day0 + offset), places=2).ljust(8)
            for f in range(raw.shape[1])
        ]
        out.append(f"{offset:>4}  " + "  ".join(cells))
    return out


def _hourly(*, power_site: np.ndarray, first: int, last: int) -> str:
    """Mean power by Europe/London clock hour over the days `first..last`, as one line."""
    window = slice(first * HALF_HOURS, (last + 1) * HALF_HOURS)
    values = power_site[window] * 100.0
    hours = local_hour_grid()[window]
    cells = [_fmt(value=_mean(values=values[hours == hour]), places=1) for hour in range(24)]
    return " ".join(cells)


def profile_blocks(*, onset: int, day0: int) -> list[tuple[str, int, int]]:
    """Return the blocks of Table 3 as (name, first day, last day).

    The 8 blocks before the onset are full 7-day blocks that end the day before it. The blocks
    since the onset are 7-day blocks counted back from day 0, and the earliest absorbs the days
    left over, so no block is partial. A span since the onset shorter than 7 days is one block
    over all of its days.

    Args:
        onset: The detector's estimated onset day index.
        day0: The check day index.

    Returns:
        The blocks, in time order.
    """
    blocks: list[tuple[str, int, int]] = [
        (f"before onset, week -{k}", onset - 7 * k, onset - 7 * k + 6)
        for k in range(PROFILE_WEEKS_BEFORE, 0, -1)
    ]
    span = day0 - onset + 1
    if span < 7:
        blocks.append((f"since onset, all {span} days", onset, day0))
        return blocks
    count = span // 7
    for j in range(count - 1, -1, -1):
        first = onset if j == count - 1 else day0 - 7 * j - 6
        last = day0 - 7 * j
        blocks.append((f"since onset, block {count - j} of {count}", first, last))
    return blocks


def _profiles(*, power_site: np.ndarray, candidate: Candidate, day0: int) -> list[str]:
    """Table 3: mean power by Europe/London clock hour around the onset."""
    out = [
        "",
        "TABLE 3. Mean power by clock hour (24 values, hour 0 to 23, local clock time),",
        "percent of capacity. Blocks before the estimated onset are the 8 weeks that end the day",
        "before it; blocks since the onset are 7-day blocks counted back from day 0 (the earliest",
        "block absorbs the days left over; a span under 7 days is one block over all of its days).",
    ]
    for name, first, last in profile_blocks(onset=candidate.onset, day0=day0):
        if first < 0:
            continue
        out.append(f"{name}: {_hourly(power_site=power_site, first=first, last=last)}")
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
    out.append("Weekly, mean / night mean over all 7 days (week -16 to week 0):")
    for m in all_members:
        cells = []
        for k in range(WEEKS - 1, -1, -1):
            stop = day0 + 1 - 7 * k
            if stop - 7 < 0:
                cells.append("n/a")
                continue
            wm = _mean(values=daily[m][0][stop - 7 : stop])
            wn = _mean(values=daily[m][1][stop - 7 : stop])
            cells.append(f"{_fmt(value=wm)}/{_fmt(value=wn)}")
        out.append(f"{labels[m]}: " + " ".join(cells))
    cells = []
    for k in range(WEEKS - 1, -1, -1):
        stop = day0 + 1 - 7 * k
        wm = _mean(values=np.asarray(summed)[stop - 7 : stop]) if stop - 7 >= 0 else float("nan")
        cells.append(_fmt(value=wm))
    out.append("capacity-weighted mean of own and neighbours, all-days mean: " + " ".join(cells))
    out.append(
        "Daily, mean / night mean, each the mean of the 7 days ending on it (day -27 to day 0):"
    )
    for m in all_members:
        cells = [
            f"{_fmt(value=_trailing(series=daily[m][0], day=day0 + o))}/"
            f"{_fmt(value=_trailing(series=daily[m][1], day=day0 + o))}"
            for o in range(-(DAILY_DAYS - 1), 1)
        ]
        out.append(f"{labels[m]}: " + " ".join(cells))
    cells = [
        _fmt(value=_trailing(series=np.asarray(summed), day=day0 + o))
        for o in range(-(DAILY_DAYS - 1), 1)
    ]
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
    closed = candidate.closure_day is not None and candidate.closure_day <= day0
    state = f"closed, ended on day {candidate.end - day0}" if closed else "open"
    out.append(f"detector episode on day 0: {state}")
    return out


def shown_register_rows(
    *, rows: tuple[RegisterRow, ...], candidate: Candidate, day0: int, lag: int
) -> list[RegisterRow]:
    """Return the register rows a check may show, in a fixed order.

    A row is shown when it is filed under the candidate's Primary, its connection date lies from
    180 days before to 30 days after the candidate's estimated onset, and it is published by the
    end of the check day under the publication lag.
    """
    shown = [
        r
        for r in rows
        if r.site == candidate.site
        and candidate.onset - REGISTER_BEFORE_DAYS <= r.connected_day <= candidate.onset + 30
        and r.visible_on(day=day0, lag=lag)
    ]
    return sorted(shown, key=lambda r: (r.connected_day, r.size_mw, r.technology))


def _register_table(
    *, rows: tuple[RegisterRow, ...], candidate: Candidate, day0: int, capacity_mw: float
) -> list[str]:
    """Table 7: the visible register rows near the onset, with no name, address, or coordinate."""
    out = ["", "TABLE 7. Embedded Capacity Register rows near the estimated onset:"]
    shown = shown_register_rows(
        rows=rows, candidate=candidate, day0=day0, lag=DEFAULT_PUBLICATION_LAG_DAYS
    )
    if not shown:
        out.append("no register row")
        return out
    for number, row in enumerate(shown, start=1):
        share = 100.0 * row.size_mw / capacity_mw
        out.append(
            f"row {number}: {row.distance_band}, {row.size_mw:.2g} MW "
            f"({share:.1f}% of the substation's capacity), {row.technology}, "
            f"connected day {row.connected_day - day0}"
        )
    return out


def prompt_of(
    *, inputs: Inputs, check: Check, register: tuple[RegisterRow, ...] | None = None
) -> Prompt:
    """Build the prompt of one check.

    Args:
        inputs: The real record.
        check: The check.
        register: The register rows of arm `llm-ecr`, or `None` for arm `llm`.

    Returns:
        The prompt, with no image.

    Raises:
        ValueError: If the text holds a Primary name, a year, a calendar word, or (without a
            register) a megawatt figure.
    """
    text = build_prompt_text(inputs=inputs, check=check, register=register)
    if FORBIDDEN_PATTERN.search(text):
        msg = "the prompt holds a Primary name, a year, or a calendar word"
        raise ValueError(msg)
    if register is None and re.search(r"\bMW\b", text):
        msg = "the prompt holds a megawatt figure"
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

    A statistics alarm after `day` moves far into the future, and an end that is not yet known on
    `day` (see `Candidate.closure_day`) moves far into the future too and is marked open.

    Args:
        candidate: The candidate.
        day: The check day.

    Returns:
        The changed candidate.
    """
    stat_alarm = candidate.stat_alarm
    if stat_alarm is not None and stat_alarm > day:
        stat_alarm = day + 100
    closure = candidate.closure_day
    if closure is not None and closure <= day:
        return replace(candidate, stat_alarm=stat_alarm)
    return replace(candidate, stat_alarm=stat_alarm, end=day + 500, ended_by="open")


def perturbed_register(*, rows: tuple[RegisterRow, ...], day: int) -> tuple[RegisterRow, ...]:
    """Return the register rows with every row not yet published on `day` changed."""
    return tuple(
        r
        if r.visible_on(day=day)
        else replace(r, size_mw=r.size_mw * 7.0, technology="other", connected_day=day + 400)
        for r in rows
    )


def assert_cut_changes_nothing(
    *,
    inputs: Inputs,
    check: Check,
    prompt: Prompt,
    register: tuple[RegisterRow, ...] | None = None,
) -> None:
    """Check that no reading, candidate fact, or register row from after the check day is used.

    The prompt must stay byte for byte unchanged when every later reading is overwritten, when
    every candidate fact not yet known on the check day is changed, and when every register row
    not yet published is changed. The own features computed on the full record must equal those
    computed on the record cut at the check day, both divided by the scale known on the check day.

    Args:
        inputs: The real record.
        check: The check.
        prompt: The prompt built from the real record.
        register: The register rows of arm `llm-ecr`, or `None`.

    Raises:
        ValueError: If a later reading, candidate fact, or register row changes the prompt.
    """
    name = check.call_id(arm="cut")
    other = prompt_of(
        inputs=perturbed(inputs=inputs, day=check.day), check=check, register=register
    )
    if other.text != prompt.text:
        msg = f"a later reading changed the prompt of {name}"
        raise ValueError(msg)
    hidden = replace(check, candidate=future_hidden(candidate=check.candidate, day=check.day))
    if prompt_of(inputs=inputs, check=hidden, register=register).text != prompt.text:
        msg = f"a later candidate fact changed the prompt of {name}"
        raise ValueError(msg)
    if register is not None:
        moved = perturbed_register(rows=register, day=check.day)
        if prompt_of(inputs=inputs, check=check, register=moved).text != prompt.text:
            msg = f"a register row published after the check day changed the prompt of {name}"
            raise ValueError(msg)
    stat_alarm = check.candidate.stat_alarm
    shows_stat = "also alarmed" in prompt.text
    if shows_stat != (stat_alarm is not None and stat_alarm <= check.day):
        msg = f"the statistics alarm line of {name} does not follow the alarm day"
        raise ValueError(msg)
    site = check.candidate.site
    n_half_hours = (check.day + 1) * HALF_HOURS
    scale = capacity_scale(power=inputs.panel.power[:, :n_half_hours])[site]
    full = prompt_features(power_site=inputs.panel.power[site] / scale, day0=check.day)
    cut = prompt_features(
        power_site=inputs.panel.power[site, :n_half_hours] / scale, day0=check.day
    )
    if not np.array_equal(full, cut, equal_nan=True):
        msg = "the features read data after the check day"
        raise ValueError(msg)

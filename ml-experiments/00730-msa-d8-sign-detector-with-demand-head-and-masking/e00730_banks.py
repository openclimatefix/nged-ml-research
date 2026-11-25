"""Experiment 730: the bank build and the realism gate with the data root as a parameter.

Experiment 729's `stage_banks` and `stage_gate` write to experiment 729's data folder through
`e00729_paths.data_root()`, so this module never calls them. It calls experiment 729's lower-level
functions (`build_bank`, `bank_digest`, `save_bank`, `load_bank`, and the gate's functions that take
arrays) and passes the root explicitly. The module holds the run marker and its guard, the component
checks (a simulated component that is silent passes every score-based check, so each component is
checked for output where output must exist), the fallback counts, and the limits file. Every
function returns counts and shares only. None prints a month, a date, an engine size, or a series
identifier.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00730_paths  # noqa: F401

# isort: split
import e00729_banks as banks729
import e00729_core as core729
import e00729_data as data729
import e00729_leakage as leak729
import e00729_score as score729
import e00729_sim as sim729
import e00729_stages as stages729
import e00730_core as core
import e00730_sky as sky
import numpy as np

Log = Callable[[str], None]

RUN_MARKER: Final = "run_marker.json"
"""Written by the `banks` stage; names the run, the sky rule, and a hash of the ranges."""
GATE_ATTEMPTS: Final = stages729.GATE_ATTEMPTS
SUPERSEDED_NAMES: Final = (
    "banks", "fits", "networks", "predictions", "scores", "margin.json", "limits.json",
    "verdict.json", "selection.json", "style.json", "figures", "gallery", "pretrained",
    "timing_scratch",
)  # fmt: skip
"""The outputs of another run. A run that starts without a marker must find none of them."""
GATED_STAGES: Final = (
    "gallery", "pretrain", "timing", "screen", "choose", "margin", "styles", "later", "refit729",
    "fit", "score", "figures", "series", "real",
)  # fmt: skip
"""The stages that read fits or scores, or start one: each needs a passed gate on the banks."""
GALLERY_GATED_STAGES: Final = (
    "pretrain", "timing", "screen", "choose", "margin", "styles", "later", "refit729", "fit",
)  # fmt: skip
"""The stages that start a fit or a pre-training run: each also needs a passed gallery check."""
NO_RUN_MARKER_STAGES: Final = ("banks",)
INVALID_FILE: Final = "scores/invalid.json"
"""Write-once record that a correctness check failed (verdict row 1); it holds the rule only."""


def record_invalid(*, root: Path, rule: str) -> None:
    """Write `scores/invalid.json` once; a later call leaves the first record in place.

    Args:
        root: The data folder of this experiment.
        rule: The rule that failed (the text of the check, no score).
    """
    path = root / INVALID_FILE
    if not path.exists():
        write_json(path=path, payload={"rule": rule})


def invalid_recorded(*, root: Path) -> bool:
    """Return whether a correctness check has failed in this run."""
    return (root / INVALID_FILE).exists()


class StoppedError(RuntimeError):
    """The line stops at verdict row 0: no verdict, and the manager is told."""


# --------------------------------------------------------------------------- the fixed code


def assert_fixed_code() -> None:
    """Stop at verdict row 0 unless experiment 729's merged code is the fixed code.

    Raises:
        StoppedError: When `e00729_core.RUN_ID` is not "2" or `e00729_data.sky_present` is missing.
    """
    if core729.RUN_ID != core.REQUIRED_RUN_ID:
        msg = "experiment 729's RUN_ID is not the fixed one (verdict row 0)"
        raise StoppedError(msg)
    if not callable(getattr(data729, "sky_present", None)):
        msg = "experiment 729's sky rule `sky_present` does not exist (verdict row 0)"
        raise StoppedError(msg)


RANGES_OVERRIDE_FILE: Final = "ranges_override.json"
"""The one permitted re-tune: new simulator ranges, read from this experiment's data folder."""
_ORIGINAL_RANGES: Final = dict(core729.RANGES)


def apply_ranges_override(*, root: Path) -> None:
    """Set experiment 729's `RANGES` to its own values plus the override file of `root`, if any.

    Experiment 729's file is not edited. The run marker hashes `RANGES` as it is after this call,
    so an override changes the marker, and every stage that calls `guard_run` applies it first.

    Args:
        root: The experiment's data folder, which may hold the override file.

    Raises:
        ValueError: When the file names a range that experiment 729 does not have.
    """
    core729.RANGES.clear()
    core729.RANGES.update(_ORIGINAL_RANGES)
    path = root / RANGES_OVERRIDE_FILE
    if not path.exists():
        return
    for name, value in json.loads(path.read_text()).items():
        if name not in _ORIGINAL_RANGES:
            msg = f"the range override names an unknown range ({name})"
            raise ValueError(msg)
        core729.RANGES[name] = tuple(value)


RETUNED_RANGES: Final = {
    "spells_per_30_days": (0.0, 1.0),
    "spell_days": (2, 3),
    "no_run_share": (0.05, 0.5),
    "solar_capacity_points": (20.0, 250.0),
}
"""The ranges of the one permitted re-tune, chosen offline by `e00730_gate_calibration.py`.

The registered values are 0 to 2 spells per 30 days, spells of 2 to 19 days, 0.2 to 0.7 of days
with no engine run, and a solar capacity of 5 to 150 points. Every other range is unchanged.
"""


def write_retune_override(*, root: Path) -> Path:
    """Write the override file of the one permitted re-tune and return its path.

    The re-tune needs exactly one gate attempt, which failed its share checks and passed its BSP
    checks, and no earlier override.

    Args:
        root: The experiment's data folder.

    Returns:
        The path of the override file.

    Raises:
        StoppedError: When the gate has not failed exactly once on its share checks alone.
        RuntimeError: When an override file exists already, or fits or `margin.json` exist.
    """
    path = root / RANGES_OVERRIDE_FILE
    if path.exists():
        msg = "the ranges override exists already: the one re-tune is used"
        raise RuntimeError(msg)
    fitted = [
        n
        for n in ("fits", "networks", "predictions")
        if (root / n).exists() and any((root / n).iterdir())
    ]
    if fitted or (root / "margin.json").exists():
        msg = "fits or margin.json exist: the re-tune must come before any fit"
        raise RuntimeError(msg)
    attempts = sorted((root / "scores").glob("gate_attempt*.json"))
    if len(attempts) != 1:
        msg = f"{len(attempts)} gate attempts exist and the re-tune needs exactly 1 (verdict row 0)"
        raise StoppedError(msg)
    attempt = read_json(path=attempts[0])
    if attempt["pass"] or attempt["bsp_failed"]:
        msg = "the re-tune needs attempt 1 to fail its share checks and pass its BSP checks"
        raise StoppedError(msg)
    write_json(path=path, payload=RETUNED_RANGES)
    return path


def run_marker_payload() -> dict:
    """Return what the run marker must hold for the code as it is now."""
    return {
        "experiment": "00730",
        **stages729.run_marker_payload(),
        "sky_gap_fill": sky.SKY_FILL_VERSION,
    }


# --------------------------------------------------------------------------- files and the guard


def write_json(*, path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON file atomically."""
    data729.atomic_write_text(path=path, text=json.dumps(payload, indent=1, default=float))


def read_json(*, path: Path) -> dict:
    """Read a JSON file."""
    return json.loads(path.read_text())


def bank_path(*, root: Path, kind: str, seed: int | None = None) -> Path:
    """Return the file of a bank under `root`."""
    name = f"{kind}_s{seed}.pkl" if seed is not None else f"{kind}.pkl"
    return root / "banks" / name


def gate_passed(*, root: Path, expected: Mapping[str, object]) -> bool:
    """Return whether `scores/gate.json` records a pass on a marker equal to `expected`."""
    path = root / "scores" / "gate.json"
    if not path.exists():
        return False
    gate = read_json(path=path)
    return bool(gate.get("pass")) and gate.get("marker") == expected


def gallery_passed(*, root: Path) -> bool:
    """Return whether `PRIVATE/gallery/panel_check.json` records every panel as passed."""
    path = root / "PRIVATE" / "gallery" / "panel_check.json"
    if not path.exists():
        return False
    record = read_json(path=path)
    panels = record.get("panels", {})
    needed = set(core.PANEL_REQUIREMENTS)
    return bool(panels) and needed <= set(panels) and all(bool(v) for v in panels.values())


def guard_run(*, root: Path, stage: str) -> None:
    """Refuse to run a stage on the outputs of another run, or without the gates before a fit.

    The `banks` stage starts a run: it needs no output of an earlier run in `root`, or a marker that
    equals the code's, or (the one permitted re-tune) a marker from changed ranges after exactly one
    failed gate attempt and before any fit. Every other stage needs a marker that equals the code's.
    A stage in `GATED_STAGES` also needs a passed realism gate on the banks now in use, and a stage
    in `GALLERY_GATED_STAGES` (any stage that starts a fit or a pre-training run) also needs the
    gallery check `PRIVATE/gallery/panel_check.json` to record every panel as passed.

    Args:
        root: The data folder of this experiment.
        stage: The stage name.

    Raises:
        RuntimeError: When a gate is open or the data folder holds outputs of another run.
    """
    apply_ranges_override(root=root)
    marker = root / RUN_MARKER
    expected = run_marker_payload()
    current = read_json(path=marker) if marker.exists() else None
    if stage == "banks":
        _guard_banks(root=root, expected=expected, current=current)
        return
    if current != expected:
        msg = (
            "no run marker matches the code (the banks were built by another run, other ranges, "
            f"or not yet): run `banks` first, after moving earlier outputs out of {root.name}"
        )
        raise RuntimeError(msg)
    if stage in GALLERY_GATED_STAGES and invalid_recorded(root=root):
        msg = (
            f"`{stage}` starts a fit, but {INVALID_FILE} records a failed correctness check "
            "(verdict row 1): the run is over"
        )
        raise RuntimeError(msg)
    if stage in GATED_STAGES and not gate_passed(root=root, expected=expected):
        msg = (
            f"`{stage}` needs scores/gate.json with a pass on the banks now in use: run `gate`, "
            "which has not passed (verdict row 0 applies after a second failure)"
        )
        raise RuntimeError(msg)
    if stage in GALLERY_GATED_STAGES and not gallery_passed(root=root):
        msg = (
            f"`{stage}` starts a fit and needs PRIVATE/gallery/panel_check.json with every panel "
            "passed (the gallery gate of the plan)"
        )
        raise RuntimeError(msg)


SKY_FIX_FOLDER: Final = "superseded_run_2"
"""Where the banks, `limits.json`, and `scores` of the run before the sky-gap fix are moved."""


def _differs_only_in_sky_fill(
    *, expected: Mapping[str, object], current: Mapping[str, object]
) -> bool:
    """Return whether the two markers differ, and only in the `sky_gap_fill` entry."""
    rest = {k: v for k, v in expected.items() if k != "sky_gap_fill"}
    return expected != current and rest == {k: v for k, v in current.items() if k != "sky_gap_fill"}


def sky_fix_rebuild_blocked(*, root: Path) -> str | None:
    """Return why `banks` may not rebuild after the sky-gap fix, or None when it may.

    The rebuild needs the earlier run's banks, `limits.json`, and `scores` moved to
    `superseded_run_2` by hand, and nothing else of the earlier run in `root` (the gallery's
    `figures` are overwritten). The ranges override stays, and no new re-tune is allowed.
    """
    if not (root / SKY_FIX_FOLDER).is_dir() or not any((root / SKY_FIX_FOLDER).iterdir()):
        return f"move the banks, limits.json, and scores to {SKY_FIX_FOLDER} first"
    left = [n for n in (*SUPERSEDED_NAMES, "limits.json") if n != "figures" and (root / n).exists()]
    if left:
        return f"{', '.join(left)} must be moved to {SKY_FIX_FOLDER}"
    return None


def gate_attempts_allowed(*, root: Path) -> int:
    """Return how many gate attempts the banks now in use may have.

    A run has two attempts, with one re-tune between them. A run that starts after the re-tune
    was spent (the ranges override exists and `superseded_run_2` exists) has one attempt and no
    re-tune.
    """
    spent = (root / RANGES_OVERRIDE_FILE).exists() and (root / SKY_FIX_FOLDER).exists()
    return 1 if spent else GATE_ATTEMPTS


def _guard_banks(*, root: Path, expected: dict, current: dict | None) -> None:
    """Apply the `banks` stage's rules of `guard_run`."""
    if current == expected:
        return
    if current is None:
        stale = [n for n in SUPERSEDED_NAMES if (root / n).exists()]
        if stale == ["banks"] and not (root / "banks" / "summary.json").exists():
            msg = (
                "banks/ holds an unfinished build (no summary.json, so a crash or a stop during "
                f"`banks`): move banks/ to {root.name}/unfinished_banks yourself, then run again"
            )
            raise RuntimeError(msg)
        if stale:
            msg = (
                f"the data folder holds outputs of an earlier run ({', '.join(stale)}) and no run "
                f"marker; move them to {root.name}/superseded_run_1 yourself, then run `banks`"
            )
            raise RuntimeError(msg)
        return
    fitted = [
        n
        for n in ("fits", "networks", "predictions")
        if (root / n).exists() and any((root / n).iterdir())
    ]
    if fitted or (root / "margin.json").exists():
        msg = (
            "the run marker differs from the code and fits or margin.json exist: "
            "move the outputs to superseded_run_N yourself"
        )
        raise RuntimeError(msg)
    if _differs_only_in_sky_fill(expected=expected, current=current):
        why = sky_fix_rebuild_blocked(root=root)
        if why is not None:
            msg = f"`banks` may not rebuild after the sky-gap fix: {why}"
            raise RuntimeError(msg)
        return
    why = stages729.retune_allowed(root=root, expected=expected, current=current)
    if why is not None:
        msg = f"`banks` may not rebuild under a changed marker: {why} (verdict row 0 applies)"
        raise RuntimeError(msg)
    limits = root / "limits.json"
    if limits.exists():
        limits.replace(root / "limits_before_retune.json")


def verify_banks_digest(*, root: Path) -> None:
    """Stop at verdict row 0 unless every stored bank matches the digest of `banks/summary.json`.

    Args:
        root: The data folder of this experiment.

    Raises:
        StoppedError: When a bank's recomputed digest differs from the summary's.
    """
    digests = read_json(path=root / "banks" / "summary.json")["digests"]
    for key, digest in digests.items():
        kind, _, seed = key.partition("_s")
        path = bank_path(root=root, kind=kind, seed=int(seed) if seed else None)
        bank = banks729.load_bank(path=path)
        if bank.digest != digest:
            msg = "a stored bank's digest differs from the summary (verdict row 0)"
            raise StoppedError(msg)


# --------------------------------------------------------------------------- component checks


def season_of(*, date: np.datetime64) -> str:
    """Return the meteorological season (winter, spring, summer, autumn) of a day's UTC month."""
    month = int(str(date)[5:7])
    return core.COMPONENT_SEASONS[(month % 12) // 3]


@dataclass
class ShareCounter:
    """Counts of cases and of cases with output, by key.

    Attributes:
        hits: The cases with output, by key.
        total: All cases, by key.
    """

    hits: dict[tuple, int]
    total: dict[tuple, int]

    @classmethod
    def new(cls) -> ShareCounter:
        """Return an empty counter."""
        return cls(defaultdict(int), defaultdict(int))

    def add(self, *, key: tuple, hit: bool) -> None:
        """Count one case."""
        self.total[key] += 1
        self.hits[key] += int(hit)

    def shares(self) -> dict[str, float]:
        """Return each key's share, with the key written as a `/`-joined string."""
        return {"/".join(k): self.hits[k] / self.total[k] for k in sorted(self.total)}

    def cases(self) -> dict[str, int]:
        """Return the number of cases behind each key's share."""
        return {"/".join(k): self.total[k] for k in sorted(self.total)}


SUN_UP_COSZ: Final = 0.1
"""A half-hour is sun-up when its clear-sky index exists and the zenith cosine exceeds this."""
MIN_SUN_UP_HALF_HOURS: Final = 8
"""A clear base day has at least this many sun-up half-hours."""
CLEAR_DAY_SHARE: Final = 0.9
"""A clear base day has a clear half-hour on at least this share of its sun-up half-hours."""
MIN_CASES: Final = 20
"""A component key with fewer cases than this (days or half-hours) fails its check."""


def clear_sun_up(*, csi: np.ndarray, cosz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the clear sun-up half-hours and the clear base days of `(days, 48)` arrays.

    The clear-sky index is NaN wherever the clear-sky irradiance is at most 5 W/m2, so a sun-up
    half-hour is one on which the index exists and `cosz` exceeds `SUN_UP_COSZ`. A clear half-hour
    has an index of at least `CLEAR_SUN_UP_CSI`. A clear base day has at least
    `MIN_SUN_UP_HALF_HOURS` sun-up half-hours and a clear one on at least `CLEAR_DAY_SHARE` of them.
    """
    sun_up = np.isfinite(csi) & np.isfinite(cosz) & (cosz > SUN_UP_COSZ)
    clear = sun_up & (np.nan_to_num(csi, nan=0.0) >= core.CLEAR_SUN_UP_CSI)
    n_up = sun_up.sum(axis=1)
    clear_day = (n_up >= MIN_SUN_UP_HALF_HOURS) & (clear.sum(axis=1) >= CLEAR_DAY_SHARE * n_up)
    return clear, clear_day


def segment_season(*, seg: sim729.Segment) -> str:
    """Return the season of a segment's first base day."""
    first = int(np.flatnonzero(seg.cal_index >= 0)[0])
    return season_of(date=seg.dates[first])


METERED_ZERO_MAX_SHARE: Final = 0.05
"""The most S1 clear base days of a bank kind that may be excluded as metered at zero.

In the real record 7 of 678 clear farm days (1.0%) have a farm that reports data but is metered
at zero or missing whenever the sun is up (an outage or a zero-filled day), and the rate reaches
4.6% in summer. The bound is 5 times the pooled real rate, so that a rule that silences more S1
days than the real record does fails the check."""


def metered_zero_days(*, cal: sim729.Calendar) -> np.ndarray:
    """Return the calendar days on which some reporting farm is metered at zero while the sun is up.

    Experiment 729's S1 draw copies the metered shape of the first farm with data and zeroes the
    half-hours with the sun down, so on such a day the simulated solar output is zero although the
    sky is clear. A day on which the meter shows only a small value at night counts too.

    Args:
        cal: The calendar of base days, with the farms' metered power.

    Returns:
        A mask over the calendar's days.
    """
    sun_up = cal.farms.zenith < core729.SUN_UP_ZENITH_DEG
    daytime = np.where(sun_up & np.isfinite(cal.farms.power), cal.farms.power, -1.0)
    return ((daytime.max(axis=2) <= 0.0) & cal.farms.power_ok).any(axis=0)


def pooled_engine_cases(*, counts: Mapping[str, int]) -> dict[str, int]:
    """Return the base days with engine output of each world and bank kind, pooled over seasons.

    The bar `MIN_CASES` applies to the key "subtract" pooled over seasons, because a split of a few
    months holds few days with engine output in some seasons. The keys "weekday" and "quiet" are
    pools already and carry no case bar.

    Args:
        counts: The cases of each engine key.

    Returns:
        The cases of each `world/kind`.
    """
    pooled: dict[str, int] = defaultdict(int)
    for key, n in counts.items():
        name, world, kind, _ = key.split("/")
        if name == "subtract":
            pooled[f"{world}/{kind}"] += n
    return dict(pooled)


def solar_output_shares(
    *, bank: banks729.Bank, cal: sim729.Calendar
) -> tuple[ShareCounter, dict[str, list[int]]]:
    """Count, per world type, source, and season, the clear base days with solar output.

    An S1 day with no output on a calendar day that `metered_zero_days` marks is not counted in
    the share. It is counted apart, and `component_checks` bounds its share of the S1 clear base
    days by `METERED_ZERO_MAX_SHARE`.

    A day counts when it is a clear base day (see `clear_sun_up`) of a solar-world segment, is not
    flagged as a fallback day, and the day's solar output is above zero on at least one clear sun-up
    half-hour. The keys "solar_subtract" count each base day with solar output and are a hit when
    the truth then lies below the base demand, so a change that stops the simulator subtracting the
    solar output fails the check even if the engine key still passes.
    """
    counter = ShareCounter.new()
    zero_day = metered_zero_days(cal=cal)
    excluded: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for seg in bank.segments:
        if seg.world not in ("solar", "both"):
            continue
        clear, clear_day = clear_sun_up(csi=seg.csi, cosz=seg.cosz)
        days = np.flatnonzero((seg.cal_index >= 0) & clear_day & ~seg.s1_fallback)
        for d in days:
            hit = bool((seg.solar[d][clear[d]] > 0.0).any())
            if seg.source == "S1":
                is_zero = (not hit) and bool(zero_day[seg.cal_index[d]])
                excluded[f"{bank.kind}/S1"][0] += int(is_zero)
                excluded[f"{bank.kind}/S1"][1] += 1
                if is_zero:
                    continue
            season = season_of(date=seg.dates[d])
            counter.add(key=(seg.world, bank.kind, seg.source, season), hit=hit)
            ok = seg.valid[d] & (seg.solar[d] > MIN_STEP_POINTS)
            if ok.any():
                subtracted = bool((seg.x[d][ok] < seg.base[d][ok] - 0.5 * seg.solar[d][ok]).all())
                counter.add(
                    key=("solar_subtract", seg.world, bank.kind, seg.source, season), hit=subtracted
                )
    return counter, dict(excluded)


def expected_solar_keys(*, bank: banks729.Bank) -> set[str]:
    """Return the solar keys a bank must hold: each (world, source, season) with a clear base day.

    A season in which no segment holds a clear base day (a month of the real record with no clear
    sky) has no case to count, so it is not expected. A bank with solar-world segments and no clear
    base day at all expects the key "no_clear_base_day", which is never counted, so that the check
    fails instead of passing on nothing.

    Args:
        bank: A bank of segments.

    Returns:
        The expected keys.
    """
    keys = set()
    solar_segments = 0
    for seg in bank.segments:
        if seg.world not in ("solar", "both"):
            continue
        solar_segments += 1
        _, clear_day = clear_sun_up(csi=seg.csi, cosz=seg.cosz)
        if ((seg.cal_index >= 0) & clear_day & ~seg.s1_fallback).any():
            keys.add("/".join((seg.world, bank.kind, seg.source, segment_season(seg=seg))))
    if solar_segments and not keys:
        return {"no_clear_base_day"}
    return keys | {f"solar_subtract/{k}" for k in keys}


def pooled_solar_cases(*, counts: Mapping[str, int]) -> dict[str, int]:
    """Return the clear base days of each bank kind and source, pooled over worlds and seasons.

    The bar `MIN_CASES` applies to this pool, because a split of a few months holds too few clear
    base days for a bar on each season (the key `world/kind/source/season`).

    Args:
        counts: The cases of each solar key.

    Returns:
        The cases of each `kind/source`.
    """
    pooled: dict[str, int] = defaultdict(int)
    for key, n in counts.items():
        if key.startswith("solar_subtract/"):
            continue
        _, kind, source, _ = key.split("/")
        pooled[f"{kind}/{source}"] += n
    return dict(pooled)


SIGMA_FLOOR: Final = 4.0
"""A pooled engine key fails when its output days fall this many standard deviations short."""
MIN_EXPECTED_HITS: Final = 3.0
"""A pooled engine key with at least this many expected output days fails when it has none."""


def engine_day_chance(*, params: Mapping[str, float], quiet: np.ndarray) -> np.ndarray:
    """Return a lower bound of each day's chance of engine output, from the drawn parameters.

    Experiment 729's simulator skips a day with chance `no_run_share`, and otherwise starts the
    evening, morning, and midday blocks independently with chance `prob * factor`, where `factor`
    is the drawn `weekend_factor` on a quiet day and 1 otherwise. A day has output when at least
    one block starts. Multi-day spells and a block that starts on the day before add output, so the
    chance is a lower bound.

    Args:
        params: The segment's drawn parameters.
        quiet: True on a weekend or bank holiday, `(days,)`.

    Returns:
        The chance of at least one half-hour of output on each day, `(days,)`.
    """
    factor = np.where(quiet, params["weekend_factor"], 1.0)
    none_start = np.ones(quiet.shape)
    for name in ("evening_prob", "morning_prob", "midday_prob"):
        none_start = none_start * (1.0 - params[name] * factor)
    return (1.0 - params["no_run_share"]) * (1.0 - none_start)


@dataclass
class PooledDays:
    """Engine output days pooled over segments, by key.

    Attributes:
        hits: The days with output.
        expected: The sum of each day's lower-bound chance of output.
        variance: The sum of each day's binomial variance.
    """

    hits: dict[tuple, int]
    expected: dict[tuple, float]
    variance: dict[tuple, float]

    @classmethod
    def new(cls) -> PooledDays:
        """Return an empty pool."""
        return cls(defaultdict(int), defaultdict(float), defaultdict(float))

    def add(self, *, key: tuple, hit: np.ndarray, chance: np.ndarray) -> None:
        """Pool the days of one segment."""
        self.hits[key] += int(hit.sum())
        self.expected[key] += float(chance.sum())
        self.variance[key] += float((chance * (1.0 - chance)).sum())

    def fails(self, *, key: tuple) -> bool:
        """Return whether the key's output days fall short of the lower bound."""
        expected = self.expected[key]
        low = expected - SIGMA_FLOOR * float(np.sqrt(self.variance[key]))
        return self.hits[key] < low or (self.hits[key] == 0 and expected >= MIN_EXPECTED_HITS)


def engine_output_shares(*, bank: banks729.Bank, cal: sim729.Calendar) -> ShareCounter:
    """Count engine-world days with output against the drawn run chances, and the subtraction.

    Experiment 729's engine simulator returns the summed output only and marks no running
    half-hour, so output cannot be compared with a schedule. The keys "weekday" and "quiet" pool
    the base days of every engine-world segment of a world, bank, and season, with quiet days the
    weekends and `cal.holidays`. The pooled count of days with output must reach the sum of each
    day's lower-bound chance (`engine_day_chance`) less four standard deviations, and a pool
    that expects at least 3 days must not be empty. Each such key counts one case, a hit when it
    passes. The check reads "any output on the day", so a silenced block (the evening only) still
    passes it; the key "subtract" is the second line of defence: it counts each base day with
    engine output, and is a hit when the truth then lies below the base demand.

    Args:
        bank: A bank of segments.
        cal: The calendar, whose `holidays` are quiet days.

    Returns:
        The counter of keys "weekday", "quiet", and "subtract".
    """
    counter = ShareCounter.new()
    pool = PooledDays.new()
    for seg in bank.segments:
        if seg.world not in ("engine", "both"):
            continue
        season = segment_season(seg=seg)
        days = np.flatnonzero(seg.cal_index >= 0)
        dates = seg.dates[days].astype("datetime64[D]")
        weekday = (dates.astype(int) + 3) % 7
        quiet = (weekday >= 5) | np.isin(dates, cal.holidays.astype("datetime64[D]"))
        chance = engine_day_chance(params=seg.params, quiet=quiet)
        hit = np.array([(seg.engine[d] > 0.0).any() for d in days], dtype=bool)
        for name, on in (("weekday", ~quiet), ("quiet", quiet)):
            pool.add(key=(name, seg.world, bank.kind, season), hit=hit[on], chance=chance[on])
        for d in days:
            running = seg.engine[d] > 0.0
            ok = seg.valid[d] & running
            if not ok.any():
                continue
            subtracted = bool((seg.x[d][ok] < seg.base[d][ok] - 0.5 * seg.engine[d][ok]).all())
            counter.add(key=("subtract", seg.world, bank.kind, season), hit=subtracted)
    for key in sorted(pool.hits):
        counter.add(key=key, hit=not pool.fails(key=key))
    return counter


def expected_engine_keys(*, bank: banks729.Bank) -> set[str]:
    """Return the engine keys that a bank must hold, by world and season of its segments."""
    keys = set()
    for s in bank.segments:
        if s.world in ("engine", "both"):
            season = segment_season(seg=s)
            for name in ("weekday", "quiet", "subtract"):
                keys.add(f"{name}/{s.world}/{bank.kind}/{season}")
    return keys


def bsp_shares(*, bank: banks729.Bank, cal: sim729.Calendar) -> ShareCounter:
    """Count the base days of BSP-month segments whose synthetic BSP follows the injection.

    A day counts when it has a half-hour with `beta * (engine + solar)` above `MIN_STEP_POINTS` and
    the real meter has data there. It is a hit when the synthetic BSP differs from the real meter at
    every such half-hour.
    """
    counter = ShareCounter.new()
    for seg in bank.segments:
        if not seg.has_bsp or seg.world == "none":
            continue
        beta = seg.params["beta"]
        for d in np.flatnonzero(seg.cal_index >= 0):
            real = cal.bsp[seg.cal_index[d]]
            moved = beta * (seg.engine[d] + seg.solar[d]) > MIN_STEP_POINTS
            where = moved & np.isfinite(real)
            if not where.any():
                continue
            differs = np.abs(seg.bsp[d][where].astype(np.float64) - real[where]) > 0.0
            counter.add(key=("bsp", seg.world, bank.kind), hit=bool(differs.all()))
    return counter


def expected_bsp_keys(*, bank: banks729.Bank) -> set[str]:
    """Return the BSP keys that a bank must hold: each injection world with BSP-month segments."""
    return {f"bsp/{s.world}/{bank.kind}" for s in bank.segments if s.has_bsp and s.world != "none"}


MIN_STEP_POINTS: Final = 1e-3
"""An injection step smaller than this many points is below the float32 resolution of the meter."""


def none_and_fold_shares(*, bank: banks729.Bank) -> dict[str, float]:
    """Return the share of no-injection truths equal to the base, and the fold's share.

    Args:
        bank: A bank of segments.

    Returns:
        `none_truth_equals_base` over valid half-hours of no-injection segments (1.0 where there are
        none) and `reading_at_least_floor`, over valid half-hours of every segment, of the reading
        being at least the segment's lowest reactive floor.
    """
    equal = total = 0
    above = above_total = 0
    for seg in bank.segments:
        on = (seg.cal_index >= 0)[:, None] & seg.valid
        if seg.world == "none":
            equal += int((seg.x[on] == seg.base[on]).sum())
            total += int(on.sum())
        floor = seg.params["q_mva"] / core729.CAPACITY_MVA * core729.POINTS * seg.params["q_factor"]
        above += int((seg.r[on] >= floor - 1e-3).sum())
        above_total += int(on.sum())
    return {
        "none_truth_equals_base": equal / total if total else 1.0,
        "reading_at_least_floor": above / above_total if above_total else 1.0,
    }


def fallback_counts(*, bank: banks729.Bank) -> dict[str, float]:
    """Count the days that leave their drawn solar source, and the days with no satellite sky.

    A flagged day of a solar-world segment took source S2 because no farm had metered output (S1
    only), or has zero solar output because no farm had a satellite sky. The second kind is the one
    that the line must keep under `MAX_NO_SKY_SHARE`. Both counts are printed per bank.

    Args:
        bank: A bank of segments.

    Returns:
        The solar-world base-day count, the flagged share, the share of days that carry zero output
        although the sun is up (no sky), and whether that share is within the bar.
    """
    days = flagged = zero_sky = 0
    for seg in bank.segments:
        if seg.world not in ("solar", "both"):
            continue
        for d in np.flatnonzero(seg.cal_index >= 0):
            days += 1
            flagged += int(seg.s1_fallback[d])
            sun = np.isfinite(seg.cosz[d]) & (seg.cosz[d] > 0.0)
            zero_sky += int(seg.s1_fallback[d] and sun.any() and not (seg.solar[d] > 0.0).any())
    share = zero_sky / days if days else 0.0
    return {
        "solar_world_base_days": float(days),
        "flagged_share": flagged / days if days else 0.0,
        "no_sky_zero_output_share": share,
        "no_sky_within_bar": share <= core.MAX_NO_SKY_SHARE,
    }


MIDDAY_WINDOW: Final = slice(20, 28)
"""Half-hours 10:00 to 14:00 UTC, where the sun is above 10 degrees of elevation all year in GB."""


def zeroed_sunup_share(*, bank: banks729.Bank) -> float:
    """Return the share of mid-day half-hours with zero solar output on source-S2 solar days.

    A solar day counts when the segment drew source S2, the day is a base day that kept its drawn
    source, and some half-hour of the day has output. A zero at mid-day on such a day is a sky gap
    that was filled with zero (experiment 729's `farm_set`), because the sun is up.

    Args:
        bank: Any bank.

    Returns:
        The share, or NaN when the bank holds no such day.
    """
    zero = total = 0
    for s in bank.segments:
        if s.world not in ("solar", "both") or s.source != "S2":
            continue
        for d in np.flatnonzero(s.cal_index >= 0):
            if s.s1_fallback[d] or s.solar[d].max() <= 0.0:
                continue
            window = s.solar[d, MIDDAY_WINDOW]
            zero += int((window <= 1e-9).sum())
            total += window.size
    return zero / total if total else float("nan")


def component_checks(*, banks_by_name: Mapping[str, banks729.Bank], cal: sim729.Calendar) -> dict:
    """Run every component check on the evaluation banks and the training bank of seed 0.

    Each share must be at least `MIN_OUTPUT_SHARE`, or `EXACT_SHARE` for the no-injection truth and
    the fold. The no-sky share of every bank must be at most `MAX_NO_SKY_SHARE`.

    Args:
        banks_by_name: The banks by name (`validation`, `test`, `shifted`, `train_s0`, and so on).
        cal: The calendar of base days (the real BSP meter).

    Returns:
        The shares of every check, the keys that fell short, and `pass`.
    """
    shares: dict[str, dict[str, float]] = {}
    cases: dict[str, dict[str, int]] = {}
    short: list[str] = []
    for name, bank in banks_by_name.items():
        solar_counter, excluded = solar_output_shares(bank=bank, cal=cal)
        shares[f"{name}/solar_metered_zero"] = {
            k: n_zero / n_all for k, (n_zero, n_all) in excluded.items() if n_all
        }
        short += [
            f"{name}/solar/metered_zero_share/{k}"
            for k, (n_zero, n_all) in excluded.items()
            if n_all and n_zero / n_all > METERED_ZERO_MAX_SHARE
        ]
        for label, counter, expected in (
            ("solar", solar_counter, expected_solar_keys(bank=bank)),
            ("engine", engine_output_shares(bank=bank, cal=cal), expected_engine_keys(bank=bank)),
            ("bsp", bsp_shares(bank=bank, cal=cal), expected_bsp_keys(bank=bank)),
        ):
            result = counter.shares()
            counts = counter.cases()
            shares[f"{name}/{label}"] = result
            cases[f"{name}/{label}"] = counts
            short += [f"{name}/{label}/{k}" for k, v in result.items() if v < core.MIN_OUTPUT_SHARE]
            short += [f"{name}/{label}/{k}/missing" for k in sorted(expected - set(counts))]
            if label == "solar":
                pooled = pooled_solar_cases(counts=counts)
                short += [
                    f"{name}/solar/pooled/{k}/too_few" for k, n in pooled.items() if n < MIN_CASES
                ]
            elif label == "engine":
                pooled = pooled_engine_cases(counts=counts)
                short += [
                    f"{name}/engine/pooled/{k}/too_few" for k, n in pooled.items() if n < MIN_CASES
                ]
            else:
                short += [
                    f"{name}/{label}/{k}/too_few"
                    for k, n in counts.items()
                    if n < MIN_CASES and not k.startswith(("weekday", "quiet"))
                ]
        zeroed = zeroed_sunup_share(bank=bank)
        shares[f"{name}/solar_zeroed_sunup"] = {"S2": zeroed}
        if np.isfinite(zeroed) and zeroed > sky.MAX_ZEROED_SHARE:
            short.append(f"{name}/solar/zeroed_sunup_share/S2")
        exact = none_and_fold_shares(bank=bank)
        shares[f"{name}/exact"] = exact
        short += [f"{name}/exact/{k}" for k, v in exact.items() if v < core.EXACT_SHARE]
        fall = fallback_counts(bank=bank)
        shares[f"{name}/fallback"] = fall
        if not fall["no_sky_within_bar"]:
            short.append(f"{name}/fallback/no_sky_zero_output_share")
    return {"shares": shares, "cases": cases, "short": short, "pass": not short}


# --------------------------------------------------------------------------- the bank build


@dataclass
class BankInputs:
    """What the bank build reads, as arrays.

    Attributes:
        cal: The calendar of base days.
        months: The months of each split ("training", "validation", "test").
        use: For each split and month, the mask over `cal` of days a segment may use.
    """

    cal: sim729.Calendar
    months: Mapping[str, Sequence[str]]
    use: Mapping[str, Mapping[str, np.ndarray]]


def bank_leakage_checks(
    *, built: Mapping[str, banks729.Bank], cal: sim729.Calendar, train_seeds: Sequence[int]
) -> None:
    """Run experiment 729's bank checks on the built banks.

    Args:
        built: The built banks by kind.
        cal: The simulator calendar.
        train_seeds: The training bank seeds, whose spaces the evaluation banks must not use.

    Raises:
        CorrectnessFailureError: When a check fails (verdict row 1).
    """
    evaluation = [built[k] for k in ("validation", "test", "shifted")]
    leak729.assert_seed_spaces(train_seeds=train_seeds, evaluation=evaluation)
    leak729.assert_banks_independent(evaluation=evaluation)
    leak729.assert_parameter_ranges(segments=[s for b in built.values() for s in b.segments])
    folder = Path(stages729.__file__).parent
    leak729.assert_simulator_source_clean(paths=[folder / n for n in stages729.SIMULATOR_SOURCES])
    for bank in evaluation:
        for seg in bank.segments:
            leak729.assert_bsp_construction(seg=seg, cal=cal)
            leak729.assert_s1_same_day(seg=seg, cal=cal)


def build_and_save_banks(
    *, root: Path, inputs: BankInputs, train_seeds: Sequence[int], log: Log
) -> dict[str, banks729.Bank]:
    """Build the validation, test, shifted-mix, and training banks and save them under `root/banks`.

    Args:
        root: The data folder of this experiment (the only folder written).
        inputs: The calendar, months, and day masks.
        train_seeds: The training seeds (0 to 4).
        log: Receives progress lines.

    Returns:
        The banks by name.
    """
    built: dict[str, banks729.Bank] = {}
    for kind, split in (("validation", "validation"), ("test", "test"), ("shifted", "test")):
        built[kind] = banks729.build_bank(
            cal=inputs.cal, kind=kind, seed=None, months=inputs.months[split],  # ty: ignore[invalid-argument-type]
            month_use=inputs.use[split],  # ty: ignore[invalid-argument-type]
        )  # fmt: skip
        banks729.save_bank(bank=built[kind], path=bank_path(root=root, kind=kind))
        log(f"built the {kind} bank: {len(built[kind].segments)} segments")
    for seed in train_seeds:
        bank = banks729.build_bank(
            cal=inputs.cal, kind="train", seed=seed, months=inputs.months["training"],
            month_use=inputs.use["training"],
        )  # fmt: skip
        banks729.save_bank(bank=bank, path=bank_path(root=root, kind="train", seed=seed))
        built[f"train_s{seed}"] = bank
        log(f"built the training bank of seed {seed}: {len(bank.segments)} segments")
    return built


def limits_of_test_bank(*, bank: banks729.Bank, holidays: np.ndarray) -> core.Limits:
    """Return f, L, and the calibration bar from the test bank's truth (no model output)."""
    rows = score729.rows_from_segments(segments=bank.segments, holidays=holidays)
    return core.compute_limits(x=rows.x, r=rows.r)


def stage_banks(*, root: Path, inputs: BankInputs, train_seeds: Sequence[int], log: Log) -> dict:
    """Build every bank, check every component, write `limits.json` and the run marker.

    The stage writes the summary before it checks, so a stop leaves the numbers on disk, and it
    writes the run marker last, so a stopped build leaves no marker and every later stage refuses
    to run.

    Args:
        root: The data folder of this experiment.
        inputs: The calendar, months, and day masks.
        train_seeds: The training seeds to build (0 to 4).
        log: Receives progress lines.

    Returns:
        The summary written to `banks/summary.json`.

    Raises:
        StoppedError: When the code is not the fixed code, a component check falls short, the
            reachability counts are short, or L lies outside `LIMIT_RANGE` (verdict row 0).
    """
    assert_fixed_code()
    guard_run(root=root, stage="banks")
    built = build_and_save_banks(root=root, inputs=inputs, train_seeds=train_seeds, log=log)
    banks729_list = [built[k] for k in ("validation", "test", "shifted")]
    check = component_checks(banks_by_name=built, cal=inputs.cal)
    leakage_pass = True
    try:
        bank_leakage_checks(built=built, cal=inputs.cal, train_seeds=train_seeds)
    except leak729.CorrectnessFailureError:
        leakage_pass = False
    rows = score729.rows_from_segments(
        segments=built["test"].segments, holidays=inputs.cal.holidays
    )
    reach = score729.reachability(rows=rows)
    limits = core.compute_limits(x=rows.x, r=rows.r)
    summary = {
        "digests": {k: b.digest for k, b in built.items()},
        "n_segments": {k: len(b.segments) for k, b in built.items()},
        "injected_share": {b.kind: banks729.injected_share(bank=b) for b in banks729_list},
        "reachability": reach,
        "limits": limits.__dict__,
        "limit_in_range": core.limit_in_range(limit=limits.harm_limit),
        "components": check,
        "leakage_pass": leakage_pass,
        "seed_spaces_disjoint": banks729.seed_spaces_disjoint(train_seeds=train_seeds),
    }
    write_json(path=root / "banks" / "summary.json", payload=summary)
    if not leakage_pass:
        msg = "a bank leakage check failed (verdict row 1)"
        raise leak729.CorrectnessFailureError(msg)
    if not check["pass"]:
        msg = f"a component check fell short on {len(check['short'])} shares (verdict row 0)"
        raise StoppedError(msg)
    if not reach["pass"]:
        msg = "the reachability counts are short (verdict row 0)"
        raise StoppedError(msg)
    if not summary["limit_in_range"]:
        msg = "L lies outside its range (verdict row 0)"
        raise StoppedError(msg)
    core.write_limits_file(path=root / "limits.json", limits=limits)
    write_json(path=root / RUN_MARKER, payload=run_marker_payload())
    log(f"banks built; components pass; L {limits.harm_limit}, f {limits.floor}")
    return summary


# --------------------------------------------------------------------------- the realism gate


@dataclass
class GateInputs:
    """What the realism gate reads, as arrays.

    Attributes:
        real_days_by_era: The real readings `(days, 48)` of the net era and of the solar era.
        d_eng: The metered engine record's half-hour steps.
        d_bsp: The real BSP meter's steps on the record's days.
        placebo_slopes: The slopes with the record shifted by whole days (circular).
    """

    real_days_by_era: Mapping[str, np.ndarray]
    d_eng: np.ndarray
    d_bsp: np.ndarray
    placebo_slopes: np.ndarray


def gate_from_arrays(
    *, train_bank: banks729.Bank, test_bank: banks729.Bank, gate_inputs: GateInputs, log: Log
) -> dict:
    """Run experiment 729's realism gate on arrays and return the record (aggregates only).

    The folded shares compare the engine-only worlds with the net era and the engine-and-solar
    worlds with the solar era, each within 0.5 to 2.0 times the real share. The BSP half
    regresses the half-hour BSP steps on the engine steps, size matched in quintile bins.

    Args:
        train_bank: The training bank of seed 0.
        test_bank: The test bank.
        gate_inputs: The real arrays.
        log: Receives progress lines.

    Returns:
        The gate record with `pass`, `void`, and the run marker.
    """
    out: dict = {"shares": {}, "marker": run_marker_payload()}
    passes = []
    for label, era, worlds in (("net", "net", ("engine",)), ("solar", "solar", ("both",))):
        real_day, real_hh = banks729.near_zero_shares(r=gate_inputs.real_days_by_era[era])
        syn_day, syn_hh = banks729.bank_shares_by_world(bank=train_bank, worlds=worlds)
        ok = core729.gate_ratio_ok(synthetic=syn_day, real=real_day) and core729.gate_ratio_ok(
            synthetic=syn_hh, real=real_hh
        )
        passes.append(ok)
        out["shares"][label] = {
            "real_day": real_day, "real_half_hour": real_hh, "synthetic_day": syn_day,
            "synthetic_half_hour": syn_hh, "pass": ok,
        }  # fmt: skip
        log(f"gate {label}: pass {ok}")
    fit = banks729.step_fit(d_bsp=gate_inputs.d_bsp, d_eng=gate_inputs.d_eng)
    ci = (fit.slope - 1.96 * fit.se, fit.slope + 1.96 * fit.se)
    edges = banks729.real_bin_edges(d_eng=gate_inputs.d_eng)
    shares = banks729.real_bin_shares(d_eng=gate_inputs.d_eng, edges=edges)
    synthetic, length = stages729.synthetic_stretches(
        bank=test_bank, record_length=len(gate_inputs.d_eng), edges=edges, shares=shares
    )
    real_fits = banks729.chunk_fits(d_bsp=gate_inputs.d_bsp, d_eng=gate_inputs.d_eng, length=length)
    real_ref = banks729.StepFit(
        fit.slope,
        float(np.nanmedian([f.se for f in real_fits])) if real_fits else float("nan"),
        float(np.nanmedian([f.resid_sd for f in real_fits])) if real_fits else float("nan"),
        int(np.median([f.n for f in real_fits])) if real_fits else 1,
    )
    gate = banks729.bsp_gate(real=real_ref, real_ci=ci, synthetic=synthetic)
    placebo = gate_inputs.placebo_slopes
    spread = float(np.std(placebo)) or float("nan")
    out["bsp"] = {
        **gate,
        "real_slope_z_vs_placebo": float((fit.slope - np.mean(placebo)) / spread),
        "n_synthetic_stretches": gate["n_finite_fits"],
        "n_synthetic_stretches_all": len(synthetic),
        "n_real_stretches": len(real_fits),
        "stretch_half_hours": length,
    }  # fmt: skip
    passes.append(bool(gate["pass"]))
    out["void"] = bool(gate["void"])
    out["pass"] = all(passes)
    out["bsp_failed"] = not bool(gate["pass"])
    log(f"gate BSP steps: pass {gate['pass']}, void {gate['void']}")
    return out


def stage_gate(*, root: Path, gate_inputs: GateInputs, log: Log) -> dict:
    """Load the banks of `root`, run the gate, and write `gate.json` and `gate_attempt<n>.json`.

    The gate has two attempts: the first run and the one re-tune of the ranges. A BSP failure on
    attempt 1, a failure on attempt 2, or a pass that was already recorded raises `StoppedError`
    (the first two are verdict row 0) or `RuntimeError` (a repeated gate).

    Args:
        root: The data folder of this experiment.
        gate_inputs: The real arrays.
        log: Receives progress lines.

    Returns:
        The gate record.

    Raises:
        StoppedError: When attempt 2 fails, or attempt 1 fails a BSP check.
    """
    guard_run(root=root, stage="gate")
    scores = root / "scores"
    earlier = sorted(scores.glob("gate_attempt*.json"))
    if earlier and read_json(path=earlier[-1])["pass"]:
        msg = "the gate already passed; it is not run again"
        raise RuntimeError(msg)
    allowed = gate_attempts_allowed(root=root)
    if len(earlier) >= allowed:
        msg = f"the gate has used its {allowed} attempts"
        raise RuntimeError(msg)
    attempt = len(earlier) + 1
    verify_banks_digest(root=root)
    train = banks729.load_bank(path=bank_path(root=root, kind="train", seed=0))
    test = banks729.load_bank(path=bank_path(root=root, kind="test"))
    out = gate_from_arrays(train_bank=train, test_bank=test, gate_inputs=gate_inputs, log=log)
    out["attempt"] = attempt
    write_json(path=scores / "gate.json", payload=out)
    write_json(path=scores / f"gate_attempt{attempt}.json", payload=out)
    if not out["pass"] and (attempt >= allowed or out["bsp_failed"]):
        msg = "the realism gate failed after its one re-tune, or attempt 1 failed a BSP check"
        raise StoppedError(msg)
    return out


def ranges_digest() -> str:
    """Return the hash of the registered ranges that the run marker holds."""
    return hashlib.sha256(repr(sorted(core729.RANGES.items())).encode()).hexdigest()[:16]

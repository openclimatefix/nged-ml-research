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
import e00729_score as score729
import e00729_sim as sim729
import e00729_stages as stages729
import e00730_core as core
import numpy as np

Log = Callable[[str], None]

RUN_MARKER: Final = "run_marker.json"
"""Written by the `banks` stage; names the run, the sky rule, and a hash of the ranges."""
GATE_ATTEMPTS: Final = stages729.GATE_ATTEMPTS
SUPERSEDED_NAMES: Final = (
    "banks", "fits", "networks", "predictions", "scores", "margin.json", "limits.json",
    "verdict.json", "selection.json", "figures", "gallery", "pretrained",
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


def run_marker_payload() -> dict:
    """Return what the run marker must hold for the code as it is now."""
    return {"experiment": "00730", **stages729.run_marker_payload()}


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


def _guard_banks(*, root: Path, expected: dict, current: dict | None) -> None:
    """Apply the `banks` stage's rules of `guard_run`."""
    if current == expected:
        return
    if current is None:
        stale = stages729.existing_outputs(root=root)
        stale = [n for n in stale if n in SUPERSEDED_NAMES]
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
    if fitted or (root / "margin.json").exists() or (root / "limits.json").exists():
        msg = (
            "the run marker differs from the code and fits, margin.json, or limits.json exist: "
            "move the outputs to superseded_run_N yourself"
        )
        raise RuntimeError(msg)
    why = stages729.retune_allowed(root=root, expected=expected, current=current)
    if why is not None:
        msg = f"`banks` may not rebuild under a changed marker: {why} (verdict row 0 applies)"
        raise RuntimeError(msg)


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


def clear_sun_up(*, csi: np.ndarray, cosz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the clear sun-up half-hours and the clear base days of a `(days, 48)` pair of arrays.

    A clear sun-up half-hour has a clear-sky index of at least `CLEAR_SUN_UP_CSI`. A clear base day
    has at least one sun-up half-hour, and every one of them is clear.
    """
    sun_up = np.isfinite(cosz) & (cosz > 0.0)
    clear = sun_up & np.isfinite(csi) & (csi >= core.CLEAR_SUN_UP_CSI)
    clear_day = sun_up.any(axis=1) & (clear.sum(axis=1) == sun_up.sum(axis=1))
    return clear, clear_day


def solar_output_shares(*, bank: banks729.Bank) -> ShareCounter:
    """Count, per world type, source, and season, the clear base days with solar output.

    A day counts when it is a clear base day (see `clear_sun_up`) of a solar-world segment, is not
    flagged as a fallback day, and the day's solar output is above zero on at least one clear sun-up
    half-hour.
    """
    counter = ShareCounter.new()
    for seg in bank.segments:
        if seg.world not in ("solar", "both"):
            continue
        clear, clear_day = clear_sun_up(csi=seg.csi, cosz=seg.cosz)
        days = np.flatnonzero((seg.cal_index >= 0) & clear_day & ~seg.s1_fallback)
        for d in days:
            hit = bool((seg.solar[d][clear[d]] > 0.0).any())
            key = (seg.world, bank.kind, seg.source, season_of(date=seg.dates[d]))
            counter.add(key=key, hit=hit)
    return counter


def engine_output_shares(*, bank: banks729.Bank) -> ShareCounter:
    """Count the segments and base days of engine worlds with output, by world, bank, and season.

    The engine simulator of experiment 729 returns the summed output only and marks no running
    half-hour, so output cannot be compared with a marker. Two proxies stand in. The key
    "segment" counts each engine-world segment once, with a hit when its engine output is above zero
    on at least one half-hour. The key "day" counts each base day of an engine-world segment on
    which
    the engine output is above zero on at least one half-hour, against the share of such days
    that the truth then subtracts: the truth must fall below the base by the engine output.
    """
    counter = ShareCounter.new()
    for seg in bank.segments:
        if seg.world not in ("engine", "both"):
            continue
        counter.add(key=("segment", seg.world, bank.kind), hit=bool((seg.engine > 0.0).any()))
        for d in np.flatnonzero(seg.cal_index >= 0):
            if not (seg.engine[d] > 0.0).any():
                continue
            running = seg.engine[d] > 0.0
            ok = seg.valid[d] & running
            subtracted = bool((seg.x[d][ok] < seg.base[d][ok] - 0.5 * seg.engine[d][ok]).all())
            counter.add(
                key=("day", seg.world, bank.kind, season_of(date=seg.dates[d])), hit=subtracted
            )
    return counter


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
    short: list[str] = []
    for name, bank in banks_by_name.items():
        for label, counter in (
            ("solar", solar_output_shares(bank=bank)),
            ("engine", engine_output_shares(bank=bank)),
            ("bsp", bsp_shares(bank=bank, cal=cal)),
        ):
            result = counter.shares()
            shares[f"{name}/{label}"] = result
            short += [f"{name}/{label}/{k}" for k, v in result.items() if v < core.MIN_OUTPUT_SHARE]
        exact = none_and_fold_shares(bank=bank)
        shares[f"{name}/exact"] = exact
        short += [f"{name}/exact/{k}" for k, v in exact.items() if v < core.EXACT_SHARE]
        fall = fallback_counts(bank=bank)
        shares[f"{name}/fallback"] = fall
        if not fall["no_sky_within_bar"]:
            short.append(f"{name}/fallback/no_sky_zero_output_share")
    return {"shares": shares, "short": short, "pass": not short}


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
        "seed_spaces_disjoint": banks729.seed_spaces_disjoint(train_seeds=train_seeds),
    }
    write_json(path=root / "banks" / "summary.json", payload=summary)
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
    if len(earlier) >= GATE_ATTEMPTS:
        msg = f"the gate has used its {GATE_ATTEMPTS} attempts"
        raise RuntimeError(msg)
    attempt = len(earlier) + 1
    verify_banks_digest(root=root)
    train = banks729.load_bank(path=bank_path(root=root, kind="train", seed=0))
    test = banks729.load_bank(path=bank_path(root=root, kind="test"))
    out = gate_from_arrays(train_bank=train, test_bank=test, gate_inputs=gate_inputs, log=log)
    out["attempt"] = attempt
    write_json(path=scores / "gate.json", payload=out)
    write_json(path=scores / f"gate_attempt{attempt}.json", payload=out)
    if not out["pass"] and (attempt >= GATE_ATTEMPTS or out["bsp_failed"]):
        msg = "the realism gate failed after its one re-tune, or attempt 1 failed a BSP check"
        raise StoppedError(msg)
    return out


def ranges_digest() -> str:
    """Return the hash of the registered ranges that the run marker holds."""
    return hashlib.sha256(repr(sorted(core729.RANGES.items())).encode()).hexdigest()[:16]

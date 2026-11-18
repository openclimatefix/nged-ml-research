# ruff: noqa: E501
"""Experiment 706, stage 0 and 1: the known-answer tests and the placebos (CPU only).

Run it from the repository root with the environment of `CLAUDE.md` set and the folders of
experiments 668, 683, and 688 on `PYTHONPATH` (the event spans come from experiment 688):

    PP=$(ls -d ml-experiments/00668-* ml-experiments/00683-* ml-experiments/00688-* | paste -sd:)
    PYTHONPATH=$PP uv run --no-sync python ml-experiments/00706-*/e00706_known_answer.py --n-jobs 4

`--test` picks one test (`counts`, `readability`, `perturb`, `ka1a`, `ka1b`, `ka2`, `pl`), or `all`.
`--month YYYY-MM` restricts the origins that the placebo scores read (a smoke test).

- `counts`: the number of bank events with depth of at least 10% of capacity and at least 21 days
  (the design says 337 of 1,152 in the six folds) and with no metered receiver (165).
- `readability`: the standard deviation over origins of the 14-day window mean of the residual on
  event-free half-hours. Every injected size must be at least 5 times it.
- `perturb`: changing power after 6 hours before the origin, or any weather run after the origin
  day, moves no coefficient and no correction of arms G and D; changing the 06:00 UTC reading moves
  it. The same holds for the levels of P-cp (window ending at 03:00 UTC, 9 hours before the 12:00 origin), RR, and M28.
- `ka1a` (gate): a 10-point additive step in truth with the forecast held fixed. Arm G with factor 1.0
  must remove 100% plus or minus 5% at 14 days and 50% plus or minus 5% at 7 days after the onset.
  A failed gate stops the experiment after one fix pass.
- `ka1b` (reported, no gate): the same removal on the re-run bank of experiment 667, with arm G on
  the smallest-lead residual of the bank's re-run forecasts.
- `ka2`: a PV-shaped injection and a cold-night injection into real half-hours, with the weather that
  happened, read by an estimator that sees ensemble weather. The forecast is not re-run.
- `pl`: PL1 (arm G with the window of another Primary of the same metering group that is no
  electrical neighbour) and PL2 (arm D with the drivers permuted in whole days inside each
  origin's own window, solved per origin).
"""

import argparse
import csv
import dataclasses
from pathlib import Path
from typing import Final

import e00706_common as c  # ty: ignore[unresolved-import]
import e00706_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol.metrics import MEDIAN_INDEX
from studies.sources import ML_EXPERIMENTS_DATA_DIR

BOTH: Final[tuple[int, int]] = (c.PV_DRIVER, c.COLD_DRIVER)
STEP_POINTS: Final[float] = 30.0
"""The additive step of KA1a, in points of capacity (the bank's events start at 10% of capacity)."""
PV_POINTS: Final[float] = 20.0
"""KA2's PV-shaped injection: this many points of capacity at an irradiance of 1,000 W/m2."""
COLD_POINTS_PER_K: Final[float] = 3.0
"""KA2's cold injection: this many points per kelvin below 5 degrees Celsius."""
TEST_DAYS: Final[tuple[int, int]] = (14, 7)
"""Days after the onset at which KA1 reads the removal."""
MIN_SIZE_OVER_NOISE: Final[float] = 5.0
HEAD_DATA: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00667-*")) / "prevailing" / "head_data"
BANK_MONTHS: Final[tuple[str, ...]] = (
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
MIN_DEPTH: Final[float] = 0.10
MIN_LENGTH_DAYS: Final[float] = 21.0
NGED_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "substation_hierarchy_2026-10-15"
GROUPS: Final[tuple[str, ...]] = (c.GROUP_SIGNED, c.GROUP_MVA, c.GROUP_SHORT)
"""The metering groups of `demand-v2`; a placebo partner stays inside one."""


def all_corrections(
    *, inputs: c.Inputs, residuals: c.Residuals, spec: c.ArmSpec
) -> tuple[np.ndarray, list[c.WindowFit]]:
    """Return the corrections of every series, (K, N_ORIGINS, N_H), and the fits."""
    pairs = [
        c.series_corrections(inputs=inputs, residuals=residuals, series=series, spec=spec)
        for series in range(len(inputs.ids))
    ]
    return np.stack([pair[0] for pair in pairs]), [pair[1] for pair in pairs]


def scored_rows(*, inputs: c.Inputs, inside: np.ndarray, first_origin: int) -> np.ndarray:
    """Return the event-free scored rows, (K, N_ORIGINS, N_H): lead up to 240 h, power present."""
    truth = c.target_truth(inputs=inputs)
    origins = np.arange(c.N_ORIGINS) >= first_origin
    event_free = ~c.stamp_to_target(per_stamp=inside)
    return (
        np.isfinite(truth)
        & event_free
        & origins[None, :, None]
        & c.lead_mask()[None, None, :]
        & np.isfinite(inputs.q_nra[..., MEDIAN_INDEX])
    )


# --------------------------------------------------------------------------- counts and readability


def counts() -> None:
    """Print the bank's event counts that the design quotes (337 and 165)."""
    events = pl.concat(
        [
            pl.read_parquet(HEAD_DATA / f"events_{month}.parquet").with_columns(month=pl.lit(month))
            for month in BANK_MONTHS
        ]
    )
    qualifying = events.filter(
        pl.col("depth") >= MIN_DEPTH, pl.col("length_days") >= MIN_LENGTH_DAYS
    )
    print(
        f"COUNTS bank events in the six folds {events.height:,} (design 1,152); with depth >= "
        f"{MIN_DEPTH:.0%} of capacity and >= {MIN_LENGTH_DAYS:.0f} days {qualifying.height} (design 337); "
        f"with no metered receiver {events.filter(pl.col('n_receivers') == 0).height} (design 165)"
    )
    print(
        f"  of the qualifying events, with no metered receiver "
        f"{qualifying.filter(pl.col('n_receivers') == 0).height}"
    )


def readability(*, inputs: c.Inputs, residuals: c.Residuals, inside: np.ndarray) -> None:
    """Print the spread of the 14-day window mean of the residual on event-free half-hours."""
    clean = np.where(inside, np.nan, residuals.smallest)
    origins = np.arange(c.FIRST_ORIGIN_DEC, c.N_ORIGINS)
    spreads = []
    for series, label in enumerate(inputs.labels):
        fit = c.fit_window(
            residual=clean[series][None, :],
            design=np.zeros((c.N_STAMPS, 0)),
            penalty=np.zeros(0),
            window_half_hours=14 * c.PER_DAY,
            lag_half_hours=c.LAG_6H,
        )
        means = fit.coef[origins, 0, 0][fit.covered[origins, 0]]
        spreads.append(float(means.std()))
        print(f"READABILITY {label}: SD of the 14-day window mean {spreads[-1]:.3f} points")
    spread = np.array(spreads)
    for group, mask in c.group_masks(inputs=inputs).items():
        worst = float(spread[mask].max())
        print(
            f"READABILITY {group}: largest SD {worst:.3f}; a {STEP_POINTS:.0f}-point step is "
            f"{STEP_POINTS / worst:.1f} times it (need {MIN_SIZE_OVER_NOISE:.0f}); PV injection "
            f"{PV_POINTS:.0f} points at full sun"
        )


# --------------------------------------------------------------------------- perturbation test


def perturb(*, inputs: c.Inputs) -> bool:
    """Return whether the look-ahead perturbation test passes, and print each check."""
    ok = True
    spec_g, spec_d = c.ArmSpec(name="g"), c.ArmSpec(name="d", drivers=BOTH)
    base_residuals = c.build_residuals(inputs=inputs)
    for series in (0, 3):
        for origin in (40, 100, 160, 220):
            last = c.ORIGIN_STAMP + c.PER_DAY * origin - c.LAG_6H
            reference = {
                spec.name: c.series_corrections(
                    inputs=inputs, residuals=base_residuals, series=series, spec=spec
                )
                for spec in (spec_g, spec_d)
            }
            truth_late = inputs.truth.copy()
            truth_late[series, last + 1 :] += 7.0
            truth_edge = inputs.truth.copy()
            truth_edge[series, last] = 50.0
            weather_late = inputs.weather.copy()
            weather_late[:, origin + 1 :] += 1.0
            weather_own = inputs.weather.copy()
            weather_own[:, origin] += 1.0
            cases = {
                "power after t-6h": (dataclasses.replace(inputs, truth=truth_late), False, False),
                "06:00 power": (dataclasses.replace(inputs, truth=truth_edge), True, True),
                "later weather runs": (
                    dataclasses.replace(inputs, weather=weather_late),
                    False,
                    False,
                ),
                "own weather run": (dataclasses.replace(inputs, weather=weather_own), False, True),
            }
            for name, (changed, coefficient_moves, correction_moves) in cases.items():
                residuals = c.build_residuals(inputs=changed)
                for spec in (spec_g, spec_d):
                    correction, fit = c.series_corrections(
                        inputs=changed, residuals=residuals, series=series, spec=spec
                    )
                    old_correction, old_fit = reference[spec.name]
                    coefficient_moved = not np.allclose(
                        fit.coef[origin], old_fit.coef[origin], atol=1e-6
                    )
                    correction_moved = not np.allclose(
                        correction[origin], old_correction[origin], atol=1e-6
                    )
                    if spec.name == "g" and name == "own weather run":
                        continue
                    if name == "own weather run" and not np.any(old_fit.coef[origin, :, 1:]):
                        continue  # both slopes are zero by rule, so weather cannot move it
                    passed = (
                        coefficient_moved == coefficient_moves
                        and correction_moved == correction_moves
                    )
                    ok &= passed
                    print(
                        f"PERTURB series {series} origin {origin} {name} arm {spec.name}: "
                        f"coefficient moved {coefficient_moved}, correction moved {correction_moved} "
                        f"-> {'ok' if passed else 'FAIL'}"
                    )
    ok &= perturb_levels(inputs=inputs)
    print(f"PERTURB overall {'PASS' if ok else 'FAIL'}")
    return ok


def level_of(*, name: str, residuals: c.Residuals, series: int, origin: int) -> float:
    """Return the level that the reference arm `name` adds at `origin` for one series."""
    if name == "P-cp":
        levels = c.changepoint_levels(
            residual=residuals.smallest[series], lag_half_hours=c.LAG_3H_CAP
        )
        return float(levels[origin])
    if name == "RR":
        return float(score.recipe_r_correction(residuals=residuals)[series, origin].sum())
    return float(score.m28_correction(residuals=residuals)[series, origin, 0])


def perturb_levels(*, inputs: c.Inputs) -> bool:
    """Return whether the perturbation test passes for the levels of P-cp, RR, and M28.

    P-cp reads power up to 03:00 UTC, 9 hours before the 12:00 origin; RR and M28 read it up to 6 hours before. None
    of the three reads weather. Power after the last half-hour a level reads must move nothing, and
    shifting the last day up to and including that half-hour must move the level.
    """
    ok = True
    base_residuals = c.build_residuals(inputs=inputs)
    for name, lag in (("P-cp", c.LAG_3H_CAP), ("RR", c.LAG_6H), ("M28", c.LAG_6H)):
        for series in (0, 3):
            for origin in (40, 100, 160, 220):
                last = c.ORIGIN_STAMP + c.PER_DAY * origin - lag
                reference = level_of(
                    name=name, residuals=base_residuals, series=series, origin=origin
                )
                truth_late = inputs.truth.copy()
                truth_late[series, last + 1 :] += 7.0
                truth_edge = inputs.truth.copy()
                truth_edge[series, last - c.PER_DAY + 1 : last + 1] += 20.0
                truth_edge_down = inputs.truth.copy()
                truth_edge_down[series, last - c.PER_DAY + 1 : last + 1] -= 20.0
                weather_late = inputs.weather.copy()
                weather_late[:, origin:] += 1.0
                cases = {
                    "power after the last half-hour read": (
                        dataclasses.replace(inputs, truth=truth_late),
                        False,
                    ),
                    "last day up to the last half-hour": (
                        dataclasses.replace(inputs, truth=truth_edge),
                        True,
                    ),
                    "weather runs from the origin day on": (
                        dataclasses.replace(inputs, weather=weather_late),
                        False,
                    ),
                }
                for case, (changed, should_move) in cases.items():
                    value = level_of(
                        name=name,
                        residuals=c.build_residuals(inputs=changed),
                        series=series,
                        origin=origin,
                    )
                    moved = not np.isclose(value, reference, atol=1e-6)
                    if should_move and not moved:
                        # A median can stay put under a one-sided shift, so try the other sign.
                        value_down = level_of(
                            name=name,
                            residuals=c.build_residuals(
                                inputs=dataclasses.replace(inputs, truth=truth_edge_down)
                            ),
                            series=series,
                            origin=origin,
                        )
                        moved = not np.isclose(value_down, reference, atol=1e-6)
                    passed = moved == should_move
                    ok &= passed
                    print(
                        f"PERTURB series {series} origin {origin} {case} level {name}: moved "
                        f"{moved} -> {'ok' if passed else 'FAIL'}"
                    )
    return ok


# --------------------------------------------------------------------------- KA1a


def ka1a(*, inputs: c.Inputs) -> bool:
    """Return whether KA1a passes: factor-1.0 arm G removes 100% at 14 days and 50% at 7 days."""
    spec = c.ArmSpec(name="ka1a", factor=1.0)
    base_residuals = c.build_residuals(inputs=inputs)
    base, base_fits = all_corrections(inputs=inputs, residuals=base_residuals, spec=spec)
    removal: dict[int, list[float]] = {14: [], 7: []}
    rows = slice(1, c.MAX_LEAD_HALF_HOURS + 1)
    for onset in range(c.FIRST_ORIGIN_DEC, c.N_ORIGINS - 15, 5):
        truth = inputs.truth.copy()
        truth[:, c.ORIGIN_STAMP + c.PER_DAY * onset :] += STEP_POINTS
        stepped = dataclasses.replace(inputs, truth=truth)
        corrections, fits = all_corrections(
            inputs=stepped, residuals=c.build_residuals(inputs=stepped), spec=spec
        )
        for days in TEST_DAYS:
            origin = onset + days
            for series in range(len(inputs.ids)):
                if fits[series].covered[origin].all() and base_fits[series].covered[origin].all():
                    gain = corrections[series, origin, rows] - base[series, origin, rows]
                    removal[days].append(float(gain.mean()) / STEP_POINTS)
    ok = True
    for days, target in ((14, 1.0), (7, 0.5)):
        values = np.array(removal[days])
        passed = abs(values.mean() - target) <= 0.05
        ok &= passed
        step_half_hours = days * c.PER_DAY - c.LAG_6H + 1
        exact = step_half_hours / (14 * c.PER_DAY)
        print(
            f"KA1a at {days} days after onset: removal {values.mean():.3f} (range {values.min():.3f} "
            f"to {values.max():.3f}, {len(values)} series-onsets); exact expected {exact:.4f} "
            f"({step_half_hours} of the window's {14 * c.PER_DAY} half-hours are after the onset); "
            f"need {target:.2f} +- 0.05 -> {'PASS' if passed else 'FAIL'}"
        )
    print(f"KA1a overall {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- KA1b


def ka1b() -> None:
    """Print the removal of arm G (factor 1.0) on the re-run bank, with no gate."""
    removal: dict[int, list[float]] = {14: [], 7: []}
    skipped = {14: 0, 7: 0}
    for month in BANK_MONTHS:
        events = pl.read_parquet(HEAD_DATA / f"events_{month}.parquet").filter(
            pl.col("depth") >= MIN_DEPTH, pl.col("length_days") >= MIN_LENGTH_DAYS
        )
        bank = np.load(HEAD_DATA / f"bank_{month}.npz")
        before_holdout = bank["vi"] < c.LIB_HOLDOUT_INDEX
        keys = {
            name: bank[name][before_holdout]
            for name in ("prim", "oi", "vi", "lead", "y", "shift", "pass_")
        }
        median = bank["q"][before_holdout][:, MEDIAN_INDEX]
        residual = (keys["y"] + keys["shift"] - median) * 100.0
        for event in events.iter_rows(named=True):
            mine = (keys["pass_"] == event["pass"]) & (keys["prim"] == event["primary"])
            if not mine.any():
                continue
            oi, vi, lead = keys["oi"][mine], keys["vi"][mine], keys["lead"][mine]
            smallest = (lead >= 3.0) & (lead < 27.0)
            for days in TEST_DAYS:
                origin12 = event["onset_idx"] + days * c.PER_DAY
                chosen = oi[oi + 6 >= origin12]
                if len(chosen) == 0:
                    skipped[days] += 1
                    continue
                at = int(chosen.min()) + 6
                window = smallest & (vi > at - c.LAG_6H - 14 * c.PER_DAY) & (vi <= at - c.LAG_6H)
                target = (vi >= at) & (vi < at + c.PER_DAY) & (oi == at - 6)
                if window.sum() < 0.5 * 14 * c.PER_DAY / 2 or not target.any():
                    skipped[days] += 1
                    continue
                left = float(np.nanmean(residual[mine][target]))
                if not np.isfinite(left) or abs(left) < 1.0:
                    skipped[days] += 1
                    continue
                removal[days].append(float(np.nanmean(residual[mine][window])) / left)
    for days, values in removal.items():
        if values:
            array = np.array(values)
            print(
                f"KA1b at {days} days after onset (reported, no gate): median removal "
                f"{np.median(array):.3f}, mean {array.mean():.3f}, {len(array)} events, "
                f"{skipped[days]} skipped (no bank rows at that origin, thin window, or error < 1 point)"
            )
        else:
            print(f"KA1b at {days} days: no usable events ({skipped[days]} skipped)")


# --------------------------------------------------------------------------- KA2


def ka2(*, inputs: c.Inputs, inside: np.ndarray) -> None:
    """Print how much of a PV-shaped and a cold-night injection arms D and G remove."""
    era5 = c.era5_grid(ids=inputs.ids)
    onset_origin = c.FIRST_ORIGIN_DEC
    after = (np.arange(c.N_STAMPS) >= c.ORIGIN_STAMP + c.PER_DAY * onset_origin)[None, :]
    injections = {
        "PV-shaped": -PV_POINTS * era5[..., 0] / 1000.0 * after,
        "cold-night": COLD_POINTS_PER_K
        * np.maximum(0.0, c.COLD_THRESHOLD_C - era5[..., 1])
        * after,
    }
    spec_g = c.ArmSpec(name="ka2-g", factor=1.0)
    spec_d = c.ArmSpec(name="ka2-d", drivers=BOTH, factor=1.0)
    base_residuals = c.build_residuals(inputs=inputs)
    base_g, _ = all_corrections(inputs=inputs, residuals=base_residuals, spec=spec_g)
    base_d, base_fits = all_corrections(inputs=inputs, residuals=base_residuals, spec=spec_d)
    print_slope_errors(inputs=inputs, fits=base_fits)
    first = onset_origin + 15
    rows = scored_rows(inputs=inputs, inside=inside, first_origin=first)
    shortwave = c.stamp_to_target(per_stamp=era5[..., 0])
    temperature = c.stamp_to_target(per_stamp=era5[..., 1])
    sets = {
        "midday (shortwave >= 300 W/m2)": shortwave >= 300.0,
        "night (shortwave < 1 W/m2)": shortwave < 1.0,
        "cold night (<= 3 C)": (shortwave < 1.0) & (temperature <= 3.0),
    }
    for name, injection in injections.items():
        injected = dataclasses.replace(inputs, truth=inputs.truth + injection.astype(np.float64))
        residuals = c.build_residuals(inputs=injected)
        corr_g, _ = all_corrections(inputs=injected, residuals=residuals, spec=spec_g)
        corr_d, _ = all_corrections(inputs=injected, residuals=residuals, spec=spec_d)
        injected_target = c.stamp_to_target(per_stamp=injection.astype(np.float32))
        for group, series_mask in c.group_masks(inputs=inputs).items():
            selected = rows & series_mask[:, None, None]
            for set_name, flag in sets.items():
                chosen = selected & flag
                if not chosen.any():
                    continue
                size = float(injected_target[chosen].mean())
                gain_d = float((corr_d - base_d)[chosen].mean())
                gain_g = float((corr_g - base_g)[chosen].mean())
                share_d = gain_d / size if abs(size) > 1e-2 else float("nan")
                print(
                    f"KA2 {name} injection, {group}, {set_name}: injected {size:+.3f} points; "
                    f"arm D removes {share_d:.3f} (left {size - gain_d:+.3f}); arm G leaves "
                    f"{size - gain_g:+.3f} (removes {gain_g / size if abs(size) > 1e-2 else float('nan'):.3f})"
                )
        if name == "PV-shaped":
            for group, series_mask in c.group_masks(inputs=inputs).items():
                print_ka2_gate(
                    group=group,
                    selected=rows & series_mask[:, None, None],
                    sets=sets,
                    injected_target=injected_target,
                    gain_d=corr_d - base_d,
                    gain_g=corr_g - base_g,
                )
    print("KA2 note: the injection is added to truth only; the stored forecast is not re-run.")


def print_ka2_gate(
    *,
    group: str,
    selected: np.ndarray,
    sets: dict[str, np.ndarray],
    injected_target: np.ndarray,
    gain_d: np.ndarray,
    gain_g: np.ndarray,
) -> None:
    """Print KA2's gate on the PV-shaped injection for one metering group (the gate reads the pooled 14)."""
    verdict_set = selected & sets["midday (shortwave >= 300 W/m2)"]
    night_set = selected & sets["night (shortwave < 1 W/m2)"]
    if not verdict_set.any() or not night_set.any():
        print(f"KA2 GATE (PV-shaped, {group}): no rows")
        return
    size = float(injected_target[verdict_set].mean())
    share = float(gain_d[verdict_set].mean()) / size
    night_error = float(gain_d[night_set].mean())
    night_g = float(gain_g[night_set].mean())
    print(
        f"KA2 GATE (PV-shaped, {group}): arm D removes {share:.3f} of the midday error "
        f"(need >= 0.5 at factor 1.0) -> {'PASS' if share >= 0.5 else 'FAIL'}; night error of arm D "
        f"{-night_error:+.3f} points (need within 1) -> {'PASS' if abs(night_error) <= 1.0 else 'FAIL'}; "
        f"night error of arm G {-night_g:+.3f} points (sign opposite to the injection: "
        f"{'yes' if night_g * size > 0 else 'NO'}, a sanity print)"
    )


def print_slope_errors(*, inputs: c.Inputs, fits: list[c.WindowFit]) -> None:
    """Print the standard error of the two ridge slopes per month, against the injected slope."""
    months = c.origin_months()
    injected = {
        "x_pv": PV_POINTS * c.SHORTWAVE_SCALE / 1000.0,
        "x_cold": COLD_POINTS_PER_K * c.COLD_SCALE_K,
    }
    series_mask = c.group_masks(inputs=inputs)[c.POOLED]
    for month in sorted(set(months[c.FIRST_ORIGIN_DEC :])):
        origins = np.flatnonzero(months == month)
        for index, name in ((1, "x_pv"), (2, "x_cold")):
            errors = np.stack([fits[k].se[origins, :, index] for k in np.flatnonzero(series_mask)])
            if np.isfinite(errors).any():
                error = float(np.nanmean(errors))
                print(
                    f"KA2 readability {month} {name}: mean slope standard error {error:.3f} points per unit; "
                    f"injected slope {injected[name]:.1f}; ratio {injected[name] / error:.1f} "
                    f"(need {MIN_SIZE_OVER_NOISE:.0f})"
                )
            else:
                print(f"KA2 readability {month} {name}: slope zero by rule in every window")


# --------------------------------------------------------------------------- placebos


def neighbours_of(*, inputs: c.Inputs) -> dict[str, set[str]]:
    """Return each Primary's metered electrical neighbours from NGED's interconnection list.

    The logic of `nged_neighbours` in `e00400_blend.py` of experiment 400, copied so that this script
    does not import that file's chain of dependencies.
    """
    label_of = dict(zip(inputs.ids.tolist(), inputs.labels, strict=True))
    with (NGED_DIR / "Monitor Direction.csv").open() as handle:
        monitors = list(csv.DictReader(handle))
    sites_of_number: dict[str, set[str]] = {}
    for row in monitors:
        if int(row["TimeSeriesID"]) in label_of:
            sites_of_number.setdefault(row["SubstationNumber"], set()).add(
                label_of[int(row["TimeSeriesID"])]
            )
    with (NGED_DIR / "Primary Substation Interconnections.csv").open() as handle:
        pairs = [(r["PrimarySubstation1"], r["PrimarySubstation2"]) for r in csv.DictReader(handle)]
    metered: dict[str, set[str]] = {label: set() for label in inputs.labels}
    for first, second in pairs:
        for here, there in ((first, second), (second, first)):
            for site in sites_of_number.get(here, ()):
                if there in sites_of_number:
                    metered[site] |= sites_of_number[there]
    return {label: values - {label} for label, values in metered.items()}


def derangement(*, inputs: c.Inputs) -> np.ndarray:
    """Return, per series, a partner of the same group that is no neighbour; -1 where none exists.

    The partner map is a permutation inside each group, found by backtracking in the site map's
    order, so it is fixed. A group where no such permutation exists gets -1 for every member.
    """
    neighbours = neighbours_of(inputs=inputs)
    partner = np.full(len(inputs.ids), -1, dtype=np.int64)
    for group in GROUPS:
        members = [i for i, g in enumerate(inputs.groups) if g == group]
        chosen: list[int] = []

        def place(
            *, position: int, members: list[int] = members, chosen: list[int] = chosen
        ) -> bool:
            if position == len(members):
                return True
            here = members[position]
            for candidate in members:
                if (
                    candidate != here
                    and candidate not in chosen
                    and inputs.labels[candidate] not in neighbours[inputs.labels[here]]
                ):
                    chosen.append(candidate)
                    if place(position=position + 1):
                        return True
                    chosen.pop()
            return False

        if place(position=0):
            partner[members] = chosen
    return partner


def mean_by_group(*, inputs: c.Inputs, values: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    """Return the mean of `values` over `rows` for each group and for the 14 verdict series."""
    return {
        group: float(values[rows & mask[:, None, None]].mean())
        for group, mask in c.group_masks(inputs=inputs).items()
    }


def print_nesting_check(*, fits_g: list[c.WindowFit], fits_d: list[c.WindowFit]) -> None:
    """Print the largest gap between D's and G's intercepts where both of D's slopes are zero by rule.

    With no slope, D's regression is G's window mean, so the intercepts must agree.
    """
    gaps = []
    for fit_g, fit_d in zip(fits_g, fits_d, strict=True):
        both_zero = np.isnan(fit_d.se[..., 1:]).all(axis=-1) & fit_d.covered
        if both_zero.any():
            gaps.append(np.abs(fit_d.coef[..., 0] - fit_g.coef[..., 0])[both_zero].max())
    if gaps:
        print(
            f"NESTING max |D intercept - G intercept| over windows where both slopes are zero by "
            f"rule: {max(gaps):.2e} points"
        )
    else:
        print("NESTING no window has both D slopes zero by rule")


def placebos(
    *, inputs: c.Inputs, residuals: c.Residuals, inside: np.ndarray, month: str | None
) -> None:
    """Print the event-free CRPS of arm G, arm D, and the two placebos, against the stored rows."""
    spec_g = c.ArmSpec(name="g")
    spec_d = c.ArmSpec(name="d", drivers=BOTH)
    spec_permuted = c.ArmSpec(name="pl2", drivers=BOTH, permute_days=True)
    corr_g, fits_g = all_corrections(inputs=inputs, residuals=residuals, spec=spec_g)
    corr_d, fits_d = all_corrections(inputs=inputs, residuals=residuals, spec=spec_d)
    corr_permuted, _ = all_corrections(inputs=inputs, residuals=residuals, spec=spec_permuted)
    print_nesting_check(fits_g=fits_g, fits_d=fits_d)
    partner = derangement(inputs=inputs)
    print(
        "PL1 partners (series -> window of):",
        {inputs.labels[i]: (inputs.labels[p] if p >= 0 else None) for i, p in enumerate(partner)},
    )
    corr_pl1 = np.where((partner >= 0)[:, None, None], corr_g[np.maximum(partner, 0)], np.nan)
    rows = scored_rows(inputs=inputs, inside=inside, first_origin=c.FIRST_ORIGIN_DEC)
    if month is not None:
        rows &= np.isin(np.arange(c.N_ORIGINS), c.origin_of_month(month=month))[None, :, None]
    truth = c.target_truth(inputs=inputs)
    crps = {}
    for name, correction in (
        ("S-NRA", np.zeros_like(corr_g)),
        ("G", corr_g),
        ("PL1", corr_pl1),
        ("D", corr_d),
        ("PL2", corr_permuted),
    ):
        quantiles = c.apply_correction(
            inputs=inputs, quantiles=inputs.q_nra, correction=correction.astype(np.float32)
        )
        crps[name] = c.loss_arrays(truth=truth, quantiles=quantiles)["crps"]
    common = rows
    for values in crps.values():
        common = common & np.isfinite(values)
    table = {
        name: mean_by_group(inputs=inputs, values=values, rows=common)
        for name, values in crps.items()
    }
    for group in c.group_masks(inputs=inputs):
        base = table["S-NRA"][group]
        print(
            f"PLACEBO {group}: CRPS S-NRA {base:.3f}; G {table['G'][group] - base:+.3f}; PL1 "
            f"{table['PL1'][group] - base:+.3f}; D {table['D'][group] - base:+.3f}; PL2 "
            f"{table['PL2'][group] - base:+.3f} (differences from S-NRA, points)"
        )
        gain_g = base - table["G"][group]
        gain_pl1 = base - table["PL1"][group]
        verdict = (
            "PASS"
            if gain_pl1 < gain_g / 3.0
            else "FAIL (report the gain as a common bias of the Candidate)"
        )
        print(f"  PL1 gain {gain_pl1:+.3f} vs a third of G's gain {gain_g / 3.0:+.3f} -> {verdict}")
        selection = common & c.group_masks(inputs=inputs)[group][:, None, None]
        if month is not None or not selection.any():
            print("  PL1 minus S-NRA interval: n/a (one month or no rows)")
        else:
            lower, upper = score.interval_job(
                ids=inputs.ids,
                rows=np.nonzero(selection),
                values={"PL1": crps["PL1"][selection], "S-NRA": crps["S-NRA"][selection]},
                treatment="PL1",
                reference="S-NRA",
                level=score.REPORTED_LEVEL,
                n_resamples=score.REPORTED_RESAMPLES,
            )
            print(
                f"  PL1 minus S-NRA: {table['PL1'][group] - base:+.3f} points, "
                f"{score.REPORTED_LEVEL:.0f}% month-block interval [{lower:+.3f}, {upper:+.3f}] "
                "(a gain would show as an interval below 0)"
            )
        excess = table["G"][group] - table["PL2"][group]
        print(
            f"  PL2: D with permuted drivers minus G {-excess:+.3f} points "
            f"(gains nothing if not below -0.01) -> {'PASS' if -excess >= -0.01 else 'FAIL (caps C2 at inconclusive)'}"
        )


def main() -> None:
    """Run the chosen known-answer tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default="all")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--month", default=None)
    arguments = parser.parse_args()
    wanted = arguments.test
    if wanted in ("all", "counts"):
        counts()
    inputs = c.load_inputs(load_prevailing=False, n_jobs=arguments.n_jobs)
    print(f"LOADED peak memory {c.peak_memory_gb():.2f} GB")
    residuals = c.build_residuals(inputs=inputs)
    inside, _, _ = c.load_event_masks(inputs=inputs, kind="primary")
    if wanted in ("all", "readability"):
        readability(inputs=inputs, residuals=residuals, inside=inside)
    if wanted in ("all", "perturb"):
        perturb(inputs=inputs)
    if wanted in ("all", "ka1a"):
        ka1a(inputs=inputs)
    if wanted in ("all", "ka1b"):
        ka1b()
    if wanted in ("all", "ka2"):
        ka2(inputs=inputs, inside=inside)
    if wanted in ("all", "pl"):
        placebos(inputs=inputs, residuals=residuals, inside=inside, month=arguments.month)
    print(f"PEAK MEMORY {c.peak_memory_gb():.2f} GB")


if __name__ == "__main__":
    main()

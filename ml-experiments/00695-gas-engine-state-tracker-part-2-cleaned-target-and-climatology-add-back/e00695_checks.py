# ruff: noqa: E501
r"""Experiment 695's leakage tests, identity control, and gate code, on small synthetic inputs.

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00695-*/e00695_checks.py            # synthetic checks (no GPU, no data)
    uv run --no-sync python ml-experiments/00695-*/e00695_checks.py --real 2026-02   # leakage test on D8's real series

The synthetic checks write nothing. The real-series leakage test and the gates (`run_g1`,
`sun_check`) read stored data, and the plan fixes them to run before any network is fitted. This
module defines the gates and does not run them on import. `main` writes `checks.json` with the
key `leakage_passed` when every leakage test passed, which `e00695_score.py` reads.
"""

import argparse
import json
from dataclasses import replace
from typing import Final

import e00695_addback as addback
import e00695_tracker as tracker
import numpy as np
import polars as pl
from e00695_common import DATA_DIR, FOLD_MONTHS, READING_DELAY, fold_refit, grid_index
from e00695_score import VerdictInputs, verdict
from scipy.stats import norm

CHECKS_PATH: Final = DATA_DIR / "checks.json"
G1_MIN_F1: Final = 0.60
G1_MIN_PRECISION: Final = 0.50
G1_MAX_FALSE_SHARE: Final = 0.03
SUN_GAP_PERCENTAGE_POINTS: Final = 10.0


def synthetic_series(*, n: int = 2400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a demand series with 0 to 2 units of 2 MVA switching on and off, and the true unit counts."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    level = 6 + 1.5 * np.sin(2 * np.pi * t / 48) + np.cumsum(rng.normal(0, 0.03, n))
    k = np.zeros(n, dtype=int)
    current = 0
    for i in range(n):
        if rng.random() < 0.02:
            current = int(rng.integers(0, 3))
        k[i] = current
    y = np.sqrt((level - k * 2.0) ** 2 + 0.14**2) + rng.normal(0, 0.04, n)
    y[500:510] = np.nan
    return y, k


def check_cut_at_train_end() -> float:
    """Return the largest change in a training-time state when every reading from the cut is missing.

    A state at a half-hour before the refit must not read a reading from `refit - 7` on. The test
    hides those readings and expects every state before the refit unchanged.
    """
    y, _ = synthetic_series()
    lvl = tracker.spread_levels(y=y)
    fit = tracker.fit_era(y=y, lvl=lvl)
    cut = 1500
    full = tracker.track_segment(y=y, fit=fit, refit=cut)
    hidden = y.copy()
    hidden[cut - READING_DELAY :] = np.nan
    other = tracker.track_segment(y=hidden, fit=fit, refit=cut)
    return float(
        max(
            np.abs(full.pk[:cut] - other.pk[:cut]).max(),
            np.abs(full.lmean[:cut] - other.lmean[:cut]).max(),
        )
    )


def check_origin_cut() -> float:
    """Return the largest change in an input state at least 24 hours old without later readings.

    Readings after the origin are hidden.

    The climatology reads states up to 48 half-hours before the native origin, and a reading takes 7
    half-hours to arrive. Hiding every reading from `origin - 7` on must leave those states unchanged.
    """
    y, _ = synthetic_series()
    lvl = tracker.spread_levels(y=y)
    fit = tracker.fit_era(y=y, lvl=lvl)
    cut, origin = 1500, 1800
    full = tracker.track_segment(y=y, fit=fit, refit=cut)
    hidden = y.copy()
    hidden[origin - READING_DELAY :] = np.nan
    other = tracker.track_segment(y=hidden, fit=fit, refit=cut)
    window = slice(cut, origin - 48 + 1)
    return float(np.abs(full.pk[window] - other.pk[window]).max())


def check_climatology_causal() -> float:
    """Return the largest change in an origin's climatology weights when states after `origin - 48` change."""
    rng = np.random.default_rng(1)
    n = 6000
    mass = rng.dirichlet(np.ones(3), size=n)
    calendar = addback.Calendar(cell=np.arange(n) % 96, n_grid=n)
    origin = 5000
    base = addback.climatology_table(mass=mass, calendar=calendar, native_origin=origin)
    changed = mass.copy()
    changed[origin - 47 :] = rng.dirichlet(np.ones(3), size=n - (origin - 47))
    other = addback.climatology_table(mass=changed, calendar=calendar, native_origin=origin)
    return float(np.abs(base - other).max())


def check_identity() -> dict[str, float]:
    """Return how far the pipeline moves when cleaning is off (k = 0, p0 = 1).

    The cleaned series must equal the reading, and the add-back must return the forecast it was
    given, both exactly or to the add-back's interpolation tolerance.
    """
    y, _ = synthetic_series()
    n = len(y)
    zero = tracker.identity_states(y=y)
    fit = tracker.fit_era(y=y, lvl=tracker.spread_levels(y=y))
    cleaned = tracker.cleaned_series(y=y, states=zero, fit=fit)
    reading_gap = float(np.nanmax(np.abs(cleaned - y)))
    quantiles = np.array([norm.ppf(addback.LEVELS, 6.0, 0.8)])
    weights = np.array([[1.0, 0.0, 0.0, 0.0]])
    routed = addback.mixture_quantiles(
        clean_q=quantiles, outputs=np.array([0.0, 2.0, 4.0, 6.0]), weights=weights
    )
    # a forecast frame through `add_back` with an all-off mass
    calendar = addback.build_calendar(n_grid=n + 10)
    mass = np.zeros((n + 10, 4))
    mass[:, 0] = 1.0
    frame = pl.DataFrame(
        {
            "power_fcst_init_time": [addback.PIPELINE_GRID_START] * 2,
            "valid_time": [addback.PIPELINE_GRID_START] * 2,
        }
    ).with_columns(
        pl.col("valid_time").dt.offset_by("5d"), pl.col("power_fcst_init_time").dt.offset_by("4d")
    )
    columns = {
        name: [float(quantiles[0, i])] * 2 for i, name in enumerate(addback.QUANTILE_COLUMNS)
    }
    frame = frame.with_columns([pl.Series(name, values) for name, values in columns.items()])
    out = addback.add_back(
        forecast=frame,
        mass=mass,
        outputs=np.array([0.0, 2.0, 4.0, 6.0]),
        calendar=calendar,
        variant="climatology",
    )
    forecast_gap = float(
        np.abs(
            out.select(list(addback.QUANTILE_COLUMNS)).to_numpy()
            - frame.select(list(addback.QUANTILE_COLUMNS)).to_numpy()
        ).max()
    )
    return {
        "cleaned_minus_reading": reading_gap,
        "route_identity": float(np.abs(routed - quantiles).max()),
        "add_back_identity": forecast_gap,
    }


def check_addback_against_simulation() -> float:
    """Return the largest gap between the mixture's quantiles and a Monte Carlo draw (MVA)."""
    quantiles = np.array([norm.ppf(addback.LEVELS, 6.0, 0.8)])
    weights = np.array([[0.5, 0.5]])
    out = addback.mixture_quantiles(
        clean_q=quantiles, outputs=np.array([0.0, 2.0]), weights=weights
    )
    rng = np.random.default_rng(0)
    demand = rng.normal(6.0, 0.8, 1_000_000)
    running = rng.random(1_000_000) < 0.5
    reading = np.where(running, np.sqrt((demand - 2.0) ** 2 + 0.14**2), demand)
    return float(np.abs(out[0] - np.quantile(reading, addback.LEVELS)).max())


def check_tracker_against_694() -> float:
    """Return the largest gap between the full smoother here and experiment 694's backward pass."""
    from e00694_trackers import _grid_backward, _grid_forward  # ty: ignore[unresolved-import]

    y, _ = synthetic_series()
    lvl = tracker.spread_levels(y=y)
    fit = tracker.fit_era(y=y, lvl=lvl)
    ours = tracker.smooth_full(y=y, lvl=lvl, fit=fit)
    means, occ, transition = tracker._model(fit=fit)
    _, alpha, scales = _grid_forward(y=y, lvl=lvl, means=means, a=transition, keep=True, occ=occ)
    theirs, _ = _grid_backward(
        y=y, lvl=lvl, means=means, a=transition, alpha=alpha, scales=scales, occ=occ
    )
    return float(np.abs(ours.pk[:, : fit.k_max + 1] - theirs).max())


def check_verdict_table() -> None:
    """Assert the verdict function takes the plan's branches in the plan's order."""
    base = VerdictInputs(
        leakage_failed=False,
        g0_failed=False,
        g1_failed=False,
        seeds_disagree=False,
        c1_passes=True,
        c1_estimate=-0.5,
        threshold=0.4,
        c4_passes=True,
        mae_rises=False,
        other_primary_worse=False,
        c2_below=True,
        c3_below=True,
    )
    cases = [
        (replace(base, leakage_failed=True, g0_failed=True), (1, "invalid")),
        (replace(base, g0_failed=True, g1_failed=True), (2, "failed")),
        (replace(base, g1_failed=True), (3, "inconclusive")),
        (replace(base, seeds_disagree=True), (3, "inconclusive")),
        (base, (4, "supported")),
        (replace(base, c4_passes=False), (5, "partly supported")),
        (replace(base, mae_rises=True), (5, "partly supported")),
        (replace(base, other_primary_worse=True), (5, "partly supported")),
        (replace(base, c1_estimate=-0.2), (6, "small gain")),
        (replace(base, c1_passes=False, c3_below=False), (7, "partly supported")),
        (replace(base, c1_passes=False, c2_below=False), (7, "partly supported")),
        (replace(base, c1_passes=False, c2_below=False, c3_below=False), (8, "not supported")),
        (replace(base, c1_passes=False), (8, "not supported")),
    ]
    for inputs, expected in cases:
        got = verdict(v=inputs)
        assert got == expected, (inputs, got, expected)


def g1_gate(*, flag: np.ndarray, truth_on: np.ndarray) -> dict:
    """Return the G1 figures: half-hour F1 and precision for "any unit running", and the false-flag share.

    Args:
        flag: Whether the causal tracker's most probable unit count is 1 or more, per half-hour.
        truth_on: Whether the injected engine is running, per half-hour.

    Returns:
        F1, precision, recall, the share of un-injected half-hours flagged, and whether G1 passes.
    """
    tp = float((flag & truth_on).sum())
    fp = float((flag & ~truth_on).sum())
    fn = float((~flag & truth_on).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    false_share = fp / max(float((~truth_on).sum()), 1.0)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "false_share": false_share,
        "passes": bool(
            f1 >= G1_MIN_F1 and precision >= G1_MIN_PRECISION and false_share <= G1_MAX_FALSE_SHARE
        ),
    }


def sun_check(
    *, flag: np.ndarray, cell: np.ndarray, sunniness: np.ndarray, days: np.ndarray
) -> dict:
    """Return the flagged-share gap between the sunniest and dullest third of days, within daylight cells.

    Args:
        flag: Whether a unit is flagged, per half-hour.
        cell: The half-hour of the day of each element (0 to 47).
        sunniness: A per-day sun measure, indexed like `days`.
        days: The day number of each half-hour, an index into `sunniness`.

    Returns:
        The gap in percentage points, averaged over the cells with at least one flagged half-hour,
        and whether it exceeds the pre-registered 10 points.
    """
    low, high = np.quantile(sunniness, [1 / 3, 2 / 3])
    dull = sunniness[days] <= low
    sunny = sunniness[days] >= high
    gaps = []
    for c in np.unique(cell):
        in_cell = cell == c
        if flag[in_cell].any() and (in_cell & dull).any() and (in_cell & sunny).any():
            gaps.append(flag[in_cell & sunny].mean() - flag[in_cell & dull].mean())
    gap = 100.0 * float(np.mean(gaps)) if gaps else 0.0
    return {"gap_points": gap, "exceeds": bool(gap > SUN_GAP_PERCENTAGE_POINTS)}


def run_g1() -> dict:
    """Run G1 on the host's causal states, per fold, from `states/W1` (not run on import)."""
    import e00695_clean as clean
    from e00695_engine import drive_series

    setup = clean.load_setup(arm="W1")
    g, _ = drive_series(n_grid=len(setup.y))
    start, stop = grid_index(when=clean.ERA2_START), grid_index(when=clean.HOLDOUT)
    flags, truths = [], []
    for month in clean.FOLD_MONTHS:
        first = grid_index(when=fold_refit(month=month))
        last = min(first + 31 * 48, stop)
        data = np.load(DATA_DIR / "states" / "W1" / f"{month}.npz")
        pk = data["pk"][first:last].astype(np.float64)
        observed = np.isfinite(setup.y[first:last])
        flags.append((np.argmax(pk, axis=1) >= 1)[observed])
        truths.append((g[first:last] > 0.0)[observed])
    assert start < stop
    return g1_gate(flag=np.concatenate(flags), truth_on=np.concatenate(truths))


def check_real_stack_cut(*, month: str) -> float:
    """Return the largest change in a fold's training-time states when readings from the cut are perturbed.

    Tracks D8 for the fold twice: once from the series, and once with every finite
    reading from `refit - 7` half-hours on raised by 3 MVA (a missing reading stays missing, so the
    series keeps its length and the tracker takes the same code path). The state marginals
    and level means of the half-hours before the refit must not change. The cleaned value of a
    half-hour with no unit running is the reading itself, so the test compares the states. These
    are the states that the training rows of all three stages read (pre-training, the fresh mixer,
    and the fine-tune).
    """
    import e00695_clean as clean

    setup = clean.load_setup(arm="A1")
    refit = grid_index(when=fold_refit(month=month))
    _, full, _, _ = clean.fold_states(setup=setup, month=month, hindsight=False, mode="folded")
    perturbed = setup.y.copy()
    perturbed[refit - READING_DELAY :] += 3.0
    _, other, _, _ = clean.fold_states(
        setup=replace(setup, y=perturbed), month=month, hindsight=False, mode="folded"
    )
    first = grid_index(when=clean.ERA1_START)
    window = slice(first, refit)
    return float(
        max(
            np.abs(full.pk[window] - other.pk[window]).max(),
            np.abs(full.lmean[window] - other.lmean[window])[np.isfinite(full.lmean[window])].max(),
        )
    )


def main() -> None:
    """Run the synthetic checks, and the real-series cut test if asked, and write `checks.json`.

    `leakage_passed` is true only when the synthetic leakage tests and the real-series cut test
    for every refit month all gave zero. The real-series results accumulate in `checks.json`
    across runs, so `--real` can be given one month at a time.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", nargs="+", metavar="MONTH", default=[], help="months, or `all`")
    parser.add_argument("--write", action="store_true", help="write checks.json")
    parser.add_argument(
        "--g1",
        action="store_true",
        help="also run gate G1 on the host's states (needs stacks and states)",
    )
    args = parser.parse_args()
    months = list(FOLD_MONTHS) if args.real == ["all"] else args.real
    results: dict = {
        "cut_at_train_end": check_cut_at_train_end(),
        "origin_cut": check_origin_cut(),
        "climatology_causal": check_climatology_causal(),
        "identity": check_identity(),
        "addback_against_simulation_mva": check_addback_against_simulation(),
        "tracker_against_694": check_tracker_against_694(),
    }
    check_verdict_table()
    results["verdict_table"] = "ok"
    previous = json.loads(CHECKS_PATH.read_text()) if CHECKS_PATH.exists() else {}
    real: dict[str, float] = dict(previous.get("real_stack_cut", {}))
    for month in months:
        real[month] = check_real_stack_cut(month=month)
    results["real_stack_cut"] = real
    assert results["identity"]["cleaned_minus_reading"] == 0.0
    assert results["identity"]["add_back_identity"] < 1e-6
    assert results["addback_against_simulation_mva"] < 0.1
    assert results["tracker_against_694"] < 1e-9
    synthetic = [results["cut_at_train_end"], results["origin_cut"], results["climatology_causal"]]
    results["leakage_passed"] = bool(
        max(synthetic) == 0.0 and set(real) >= set(FOLD_MONTHS) and max(real.values()) == 0.0
    )
    if args.g1:
        results["g1"] = run_g1()
        results["g1_passed"] = results["g1"]["passes"]
    print(json.dumps(results, indent=1))
    if args.write:
        CHECKS_PATH.write_text(json.dumps(previous | results, indent=1))


if __name__ == "__main__":
    main()

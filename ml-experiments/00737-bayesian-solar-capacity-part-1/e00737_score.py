"""Scoring of experiment 737: path error, recovered share, coverage, sign tests, bars, and verdict.

The unit of evidence is the Primary. Each Primary's median difference of path error over its main
worlds is one paired observation, so a contrast has six observations, and an exact one-sided sign
test at 0.025 is applied to them. The functions here are pure (they read no file) except
`run_score` and `report`, which read the world and fit files that the fit stage wrote and write the
private score tables.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from e00737_common import (
    ARMS,
    B1_BAND,
    B2_BAND,
    B3_MAX_RATIO,
    B4_MAX_DROP,
    B5_MAX_DROP,
    B6_MAX,
    B7_MAX_MW,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    C1_THRESHOLD,
    C2_THRESHOLD,
    CAPACITY_TABLE_649,
    G5_MIN_CONVERGED,
    GRID_LENGTH,
    INTERVAL_Z,
    N_QUARTERS,
    OUT,
    PAST_ONLY_CUTS,
    PRIMARIES,
    SIGN_TEST_ALPHA,
    SPECIAL_SEED,
    atomic_json,
    grid_end_times,
    quarter_index,
)
from e00737_worlds import world_specs

MIN_REAL_CAPACITY_MW: Final[float] = 0.05
"""A real-record mean capacity at or under this makes the placebo ratio unreadable."""
BAR_NAMES: Final[tuple[str, ...]] = ("B1", "B2", "B3", "B4", "B5", "B6", "B7")
C2_STATES: Final[tuple[str, ...]] = ("M", "S", "N", "R")
VERDICTS: Final[tuple[str, ...]] = (
    "supported",
    "partly supported",
    "small gain",
    "not supported",
    "refuted",
    "inconclusive",
    "invalid",
)


def path_error(*, truth: np.ndarray, estimate: np.ndarray, scored: np.ndarray) -> float:
    """Return the mean absolute gap over the scored quarters, as a share of the mean truth.

    A quarter whose estimate is NaN is left out of both means.

    Args:
        truth: The injected capacity of each quarter.
        estimate: The fitted increment of each quarter.
        scored: Which quarters are scored.

    Returns:
        The path error (NaN when the mean truth is zero or no quarter has an estimate).
    """
    keep = scored & np.isfinite(estimate)
    if not keep.any() or truth[keep].mean() <= 0:
        return float("nan")
    return float(np.abs(estimate[keep] - truth[keep]).mean() / truth[keep].mean())


def recovered_share(*, truth: np.ndarray, estimate: np.ndarray, scored: np.ndarray) -> float:
    """Return the sum of the fitted increments over the sum of the injected capacities.

    Args:
        truth: The injected capacity of each quarter.
        estimate: The fitted increment of each quarter.
        scored: Which quarters are scored.

    Returns:
        The recovered share (NaN when the injected capacities sum to zero).
    """
    keep = scored & np.isfinite(estimate)
    if not keep.any() or truth[keep].sum() <= 0:
        return float("nan")
    return float(estimate[keep].sum() / truth[keep].sum())


def interval_hits(
    *, truth: np.ndarray, estimate: np.ndarray, sd: np.ndarray, scored: np.ndarray
) -> tuple[int, int]:
    """Count the scored quarters whose 80% interval covers the injected capacity.

    Args:
        truth: The injected capacity of each quarter.
        estimate: The fitted increment of each quarter.
        sd: The standard deviation of the injected fit's capacity (inflated).
        scored: Which quarters are scored.

    Returns:
        The number of covered quarters and the number of quarters counted.
    """
    keep = scored & np.isfinite(estimate) & np.isfinite(sd)
    hit = np.abs(estimate - truth) <= INTERVAL_Z * sd
    return int(hit[keep].sum()), int(keep.sum())


def exact_sign_p(*, positives: int, n: int) -> float:
    """Return the exact one-sided sign-test p-value, P(at least `positives` of `n` at 1/2).

    Args:
        positives: The number of observations in the predicted direction.
        n: The number of observations.

    Returns:
        The p-value.
    """
    return sum(math.comb(n, k) for k in range(positives, n + 1)) / 2**n


def sign_test_counts(*, medians: np.ndarray) -> tuple[int, float, bool]:
    """Apply the exact sign test to the Primaries' median differences.

    Args:
        medians: One median difference per Primary (positive means arm A is better).

    Returns:
        The number of positive medians, the p-value, and whether the test passes at 0.025.
    """
    positives = int((medians > 0).sum())
    p = exact_sign_p(positives=positives, n=len(medians))
    return positives, p, p <= SIGN_TEST_ALPHA


def c1_status(*, medians: np.ndarray) -> str:
    """Return `pass`, `reversed`, or `fail` for contrast C1 (F minus A).

    It passes when the sign test passes and the median of the six medians is at least 0.10. It is
    reversed when F beats A at 6 of 6 Primaries.

    Args:
        medians: One median of (F minus A) per Primary.

    Returns:
        The status.
    """
    if (medians < 0).all():
        return "reversed"
    _, _, counts = sign_test_counts(medians=medians)
    return "pass" if counts and float(np.median(medians)) >= C1_THRESHOLD else "fail"


def c2_state(*, medians: np.ndarray) -> str:
    """Return the state `M`, `S`, `N`, or `R` of contrast C2 (Qt minus A).

    Args:
        medians: One median of (Qt minus A) per Primary.

    Returns:
        `R` when Qt beats A at 6 of 6 Primaries; `M` when the test counts and the median is at
        least 0.05; `S` when it counts and the median is above 0 and under 0.05; else `N`.
    """
    if (medians < 0).all():
        return "R"
    _, _, counts = sign_test_counts(medians=medians)
    median = float(np.median(medians))
    if counts and median >= C2_THRESHOLD:
        return "M"
    if counts and median > 0:
        return "S"
    return "N"


def count_bar(*, passes: int, unreadable: int, need: int) -> bool | None:
    """Return whether a bar that needs `need` Primaries is met, or None when it cannot be read.

    Args:
        passes: Primaries that meet the bar.
        unreadable: Primaries whose value cannot be read.
        need: Primaries needed.

    Returns:
        True when enough pass, False when enough cannot, else None.
    """
    if passes >= need:
        return True
    if passes + unreadable < need:
        return False
    return None


def bars_state(*, bars: dict[str, bool | None]) -> str:
    """Return `fail` when any bar fails, `unreadable` when any cannot be read, else `pass`.

    Args:
        bars: Each bar's outcome.

    Returns:
        The state.
    """
    if any(v is False for v in bars.values()):
        return "fail"
    if any(v is None for v in bars.values()):
        return "unreadable"
    return "pass"


def verdict_row(
    *,
    gates_ok: bool,
    lookahead: bool,
    c1: str,
    c2: str,
    bars: str,
    c2_ci_upper: float,
    n_converged_primaries: int,
) -> tuple[int, str]:
    """Map the inputs to a row of the verdict table and its verdict; the first matching row applies.

    Args:
        gates_ok: Whether G1, G1b, and G2 to G5 all pass.
        lookahead: Whether a look-ahead was found in the inputs.
        c1: `pass`, `reversed`, or `fail`.
        c2: `M`, `S`, `N`, or `R`.
        bars: `pass`, `fail`, or `unreadable` (from `bars_state`).
        c2_ci_upper: The upper end of C2's world-resampled 95% interval.
        n_converged_primaries: Primaries with a converged fit.

    Returns:
        The row number (1 to 7) and the verdict.
    """
    row, verdict = 7, "inconclusive"
    if not gates_ok or lookahead:
        row, verdict = 1, "invalid"
    elif c1 == "reversed" or c2 == "R":
        row, verdict = 2, "refuted"
    elif n_converged_primaries < 5 or c1 != "pass" or bars == "unreadable":
        pass
    elif c2 == "N":
        if c2_ci_upper < C2_THRESHOLD:
            row, verdict = 6, "not supported"
    elif c2 == "M":
        row, verdict = (3, "supported") if bars == "pass" else (4, "partly supported")
    else:
        row, verdict = (5, "small gain") if bars == "pass" else (6, "not supported")
    return row, verdict


def median_of_medians_interval(
    *, values: list[np.ndarray], draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Return the 95% world-resampled interval of the median over Primaries of the medians.

    Each draw resamples the worlds of every Primary with replacement, takes each Primary's median,
    and then the median over the Primaries.

    Args:
        values: Each Primary's per-world differences.
        draws: The number of draws.
        seed: The random seed.

    Returns:
        The 2.5th and 97.5th percentiles.
    """
    rng = np.random.default_rng(seed)
    per_primary = []
    for v in values:
        clean = v[np.isfinite(v)]
        index = rng.integers(0, len(clean), size=(draws, len(clean)))
        per_primary.append(np.median(clean[index], axis=1))
    stat = np.median(np.stack(per_primary), axis=0)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def world_metrics(
    *,
    spec_name: str,
    primary: str,
    kind: str,
    size: float,
    shape: str,
    seed: int,
    truth: np.ndarray,
    scored: np.ndarray,
    fits: dict[str, np.ndarray],
    controls: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    """Return one metric row per arm for an injected world.

    The increment of an arm is its capacity in the injected world minus its capacity in its
    control (arm O has its own control in the same fleet's term). The interval takes its standard
    deviation from the injected fit alone.

    Args:
        spec_name: The world's name.
        primary: The Primary's label.
        kind: The world type.
        size: The fleet's final capacity.
        shape: The growth shape.
        seed: The seed.
        truth: The injected capacity of each quarter.
        scored: Which quarters are scored.
        fits: The world's fit arrays (`<arm>_cap`, `<arm>_sd`).
        controls: For each arm, the control arrays (`cap` and `sd`) to subtract.

    Returns:
        Rows with `E`, `share`, `hits`, and `cells`.
    """
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        cap = fits[f"{arm}_cap"]
        sd = fits[f"{arm}_sd"]
        inc = cap - controls[arm]["cap"]
        hits, cells = interval_hits(truth=truth, estimate=inc, sd=sd, scored=scored)
        rows.append(
            {
                "world": spec_name,
                "primary": primary,
                "kind": kind,
                "size": size,
                "shape": shape,
                "seed": seed,
                "arm": arm,
                "E": path_error(truth=truth, estimate=inc, scored=scored),
                "share": recovered_share(truth=truth, estimate=inc, scored=scored),
                "hits": hits,
                "cells": cells,
            }
        )
    return rows


def load_npz(*, path: Path) -> dict[str, np.ndarray]:
    """Load every array of an `.npz` file.

    Args:
        path: The file.

    Returns:
        The arrays by name.
    """
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def collect_metrics() -> tuple[pl.DataFrame, dict[str, dict]]:
    """Read every world and fit file and build the metric table of the injected worlds.

    Returns:
        The table (one row per world and arm), and per-Primary extras: the scored quarters, the
        control and placebo capacities, and the transfer windows' quarters.
    """
    rows: list[dict[str, object]] = []
    extras: dict[str, dict] = {}
    for primary in PRIMARIES:
        fit_dir = OUT / "fits" / primary
        control = load_npz(path=fit_dir / "control.npz")
        scored = control["scored"].astype(bool)
        base_controls = {
            arm: {"cap": control[f"{arm}_cap"], "sd": control[f"{arm}_sd"]} for arm in ARMS
        }
        ocontrols = {
            int(p.stem.split("_")[1]): load_npz(path=p)
            for p in sorted(fit_dir.glob("ocontrol_*.npz"))
        }
        extras[primary] = {
            "scored": scored,
            "control": control,
            "placebo": load_npz(path=fit_dir / "placebo.npz"),
            "transfer_quarters": None,
        }
        for spec in world_specs(primary=primary):
            if spec.kind in ("placebo", "control"):
                continue
            world = load_npz(path=OUT / "worlds" / primary / f"{spec.name}.npz")
            fits = load_npz(path=fit_dir / f"{spec.name}.npz")
            controls = dict(base_controls)
            if spec.kind != "matched":
                controls["O"] = {
                    "cap": ocontrols[spec.seed]["A_cap"],
                    "sd": ocontrols[spec.seed]["A_sd"],
                }
            rows += world_metrics(
                spec_name=spec.name,
                primary=primary,
                kind=spec.kind,
                size=spec.size_mw,
                shape=spec.shape,
                seed=spec.seed,
                truth=world["truth"],
                scored=scored,
                fits=fits,
                controls=controls,
            )
            if spec.kind == "transfer":
                idx = np.concatenate([np.arange(a, b) for a, b in world["windows"]])
                quarters = quarter_index(end_time=grid_end_times()[idx])
                extras[primary]["transfer_quarters"] = np.unique(quarters)
                extras[primary]["transfer_fit"] = fits
    return pl.DataFrame(rows), extras


def per_primary_median(*, frame: pl.DataFrame, column: str) -> dict[str, float]:
    """Return the median of a column for each Primary, in `PRIMARIES` order.

    Args:
        frame: A table with a `primary` column.
        column: The column.

    Returns:
        The medians.
    """
    grouped = frame.group_by("primary").agg(pl.col(column).median().alias("m"))
    lookup = dict(zip(grouped["primary"].to_list(), grouped["m"].to_list(), strict=True))
    return {p: float(lookup.get(p, float("nan"))) for p in PRIMARIES}


def contrast(*, metrics: pl.DataFrame, rival: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return a contrast's per-Primary medians and per-world differences (rival minus arm A).

    Args:
        metrics: The metric table.
        rival: The rival arm (`F` or `Qt`).

    Returns:
        The six medians, and each Primary's per-world differences of path error.
    """
    main = metrics.filter(pl.col("kind") == "main")
    wide = main.filter(pl.col("arm").is_in(["A", rival])).pivot(
        on="arm", index=["primary", "world"], values="E"
    )
    diff = wide.with_columns(d=pl.col(rival) - pl.col("A"))
    per = [
        diff.filter(pl.col("primary") == p)["d"].drop_nans().drop_nulls().to_numpy()
        for p in PRIMARIES
    ]
    return np.array([float(np.median(v)) for v in per]), per


def evaluate_bars(*, metrics: pl.DataFrame, extras: dict[str, dict]) -> dict[str, dict]:
    """Evaluate the seven bars on arm A.

    Args:
        metrics: The metric table.
        extras: The per-Primary extras of `collect_metrics`.

    Returns:
        For each bar, a record with `ok` (True, False, or None) and the values behind it.
    """
    a = metrics.filter(pl.col("arm") == "A")
    out: dict[str, dict] = {}
    main = a.filter(pl.col("kind") == "main")
    b1 = per_primary_median(frame=main.filter(pl.col("size").is_in([2.5, 5.0])), column="share")
    passes = sum(B1_BAND[0] <= v <= B1_BAND[1] for v in b1.values() if np.isfinite(v))
    out["B1"] = {"ok": count_bar(passes=passes, unreadable=0, need=5), "median_share": b1}
    coverage = float(np.sum(main["hits"].to_numpy())) / float(np.sum(main["cells"].to_numpy()))
    out["B2"] = {"ok": B2_BAND[0] <= coverage <= B2_BAND[1], "coverage": coverage}
    ratios, unreadable = {}, 0
    for p in PRIMARIES:
        scored = extras[p]["scored"]
        real = np.clip(extras[p]["control"]["A_cap"], 0.0, None)[scored].mean()
        placebo = np.clip(extras[p]["placebo"]["A_cap"], 0.0, None)[scored].mean()
        if real <= MIN_REAL_CAPACITY_MW:
            unreadable += 1
            ratios[p] = float("nan")
        else:
            ratios[p] = float(placebo / real)
    passes = sum(v <= B3_MAX_RATIO for v in ratios.values() if np.isfinite(v))
    out["B3"] = {"ok": count_bar(passes=passes, unreadable=unreadable, need=5), "ratio": ratios}
    like = main.filter(
        (pl.col("shape") == "ramp")
        & pl.col("size").is_in([2.5, 5.0])
        & pl.col("seed").is_in([737101, 737102])
    )
    matched = a.filter(pl.col("kind") == "matched")
    mis_median = float(np.median(like["share"].to_numpy()))
    match_median = float(np.median(matched["share"].to_numpy()))
    out["B4"] = {
        "ok": mis_median >= match_median - B4_MAX_DROP,
        "mismatched_median": mis_median,
        "matched_median": match_median,
    }
    farm_median = float(np.median(a.filter(pl.col("kind") == "farm")["share"].to_numpy()))
    ramp_median = float(
        np.median(
            main.filter((pl.col("size") == 2.5) & (pl.col("shape") == "ramp"))["share"].to_numpy()
        )
    )
    out["B5"] = {
        "ok": farm_median >= ramp_median - B5_MAX_DROP,
        "farm_median": farm_median,
        "ramp_median": ramp_median,
    }
    shifts, moves = {}, {}
    for p in PRIMARIES:
        base = main.filter(
            (pl.col("primary") == p)
            & (pl.col("size") == 2.5)
            & (pl.col("shape") == "ramp")
            & (pl.col("seed") == SPECIAL_SEED)
        )["share"][0]
        cloud = a.filter((pl.col("primary") == p) & (pl.col("kind") == "cloud"))["share"][0]
        shifts[p] = float(abs(cloud - base))
        scored = extras[p]["scored"]
        quarters = extras[p]["transfer_quarters"]
        fit = extras[p]["transfer_fit"]
        main_fit = load_npz(path=OUT / "fits" / p / f"main_2.5_ramp_{SPECIAL_SEED}.npz")
        gap = np.abs(fit["A_cap"] - main_fit["A_cap"])
        pick = np.zeros(N_QUARTERS, dtype=bool)
        pick[quarters] = True
        moves[p] = float(gap[pick & scored].max())
    out["B6"] = {"ok": float(np.median(list(shifts.values()))) <= B6_MAX, "shift": shifts}
    out["B7"] = {"ok": float(np.median(list(moves.values()))) <= B7_MAX_MW, "move_mw": moves}
    return out


def convergence_table(*, extras: dict[str, dict]) -> dict[str, float]:
    """Return the share of cold-start fits (arms A, F, Qt, O, R) that stopped on the tolerance.

    Args:
        extras: The per-Primary extras of `collect_metrics`.

    Returns:
        The share converged, and the number of fits, and the number of fits whose capacities all
        sit at or below zero.
    """
    flags, all_zero = [], 0
    for path in sorted((OUT / "fits").glob("*/*.npz")):
        arrays = load_npz(path=path)
        for arm in ("A", "F", "Qt", "O", "R"):
            key = f"{arm}_info"
            if key in arrays:
                flags.append(bool(arrays[key][1]))
                if arm == "A" and not (arrays[f"{arm}_cap"] > 0).any():
                    all_zero += 1
    return {
        "share_converged": float(np.mean(flags)) if flags else float("nan"),
        "n_fits": float(len(flags)),
        "n_all_at_bound": float(all_zero),
    }


def g5_record(*, extras: dict[str, dict]) -> dict[str, object]:
    """Gate G5: 99% of the cold fits converge, and no fit ends with every capacity at its bound.

    Args:
        extras: The per-Primary extras.

    Returns:
        The record with `passed`.
    """
    table = convergence_table(extras=extras)
    return {
        "passed": bool(
            table["share_converged"] >= G5_MIN_CONVERGED and table["n_all_at_bound"] == 0
        ),
        **table,
    }


def normal_cdf(*, x: float) -> float:
    """Return the standard normal cumulative distribution at `x`.

    Args:
        x: The point.

    Returns:
        The probability.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def real_record_aggregates(*, extras: dict[str, dict]) -> dict[str, object]:
    """Return the aggregates of the real-record fits (arm A and its comparisons).

    Every aggregate is a ratio, a share, or a count. The private tables keep every Primary's value,
    and the figure's array (`real_path.npz`) holds each Primary's capacity as a multiple of its own
    mean. The probability that capacity rises uses a normal approximation that treats the
    quarters' posterior errors as independent, so it is approximate.

    Args:
        extras: The per-Primary extras of `collect_metrics`.

    Returns:
        The aggregates and their per-Primary values.
    """
    ratios, rising, r_over_a, wide, per_primary, paths = {}, {}, {}, [], {}, []
    table = pl.read_parquet(CAPACITY_TABLE_649)
    overlap, readable = 0, 0
    for p in PRIMARIES:
        control = extras[p]["control"]
        scored = extras[p]["scored"]
        raw, sd = control["A_cap"], control["A_sd"]
        cap = np.clip(raw, 0.0, None)
        idx = np.flatnonzero(scored)
        first, last = idx[:4], idx[-4:]
        mean_first, mean_last = cap[first].mean(), cap[last].mean()
        ratios[p] = float(mean_last / mean_first) if mean_first > 0 else float("nan")
        diff = raw[last].mean() - raw[first].mean()
        spread = math.sqrt(float((sd[first] ** 2).sum() + (sd[last] ** 2).sum())) / 4.0
        rising[p] = normal_cdf(x=diff / spread)
        r_cap = np.clip(control["R_cap"], 0.0, None)[scored].mean()
        r_over_a[p] = float(r_cap / cap[scored].mean()) if cap[scored].mean() > 0 else float("nan")
        half = 0.5 * float(np.median(cap[scored]))
        wide += (2.0 * INTERVAL_Z * sd[scored] > half).tolist()
        season = [13, 14]
        season_mean = float(raw[season].mean())
        season_sd = math.sqrt(float((sd[season] ** 2).sum())) / 2.0
        ref = table.filter((pl.col("site") == p) & (pl.col("probe_year") == 2025)).sort("cutoff")
        if ref.height:
            readable += 1
            lo, hi = float(ref["cal_p10"][-1]), float(ref["cal_p90"][-1])
            overlap += int(
                season_mean - INTERVAL_Z * season_sd <= hi
                and season_mean + INTERVAL_Z * season_sd >= lo
            )
        own_mean = cap[scored].mean()
        paths.append(np.where(scored, cap / own_mean if own_mean > 0 else np.nan, np.nan))
        per_primary[p] = {
            "scored_quarters": int(scored.sum()),
            "dropped_no_power_and_unreliable": control["dropped"].tolist(),
        }
    stack = np.array(paths)
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "scores" / "real_path.npz", paths=stack)
    return {
        "median_last_over_first_year": float(np.nanmedian(list(ratios.values()))),
        "share_rising_with_prob_0.9": float(np.mean([v >= 0.9 for v in rising.values()])),
        "median_r_over_a": float(np.nanmedian(list(r_over_a.values()))),
        "overlap_with_649": {"count": overlap, "of": readable},
        "share_of_quarters_interval_wider_than_half_median": float(np.mean(wide)),
        "per_primary": per_primary,
        "ratios": ratios,
        "rising": rising,
        "r_over_a": r_over_a,
    }


def past_only_gap(*, extras: dict[str, dict]) -> dict[str, float]:
    """Return how far a past-only fit's last quarter is from the whole-record fit's.

    For each Primary and each cut, the gap is the absolute difference of arm A's capacity in the
    cut's last quarter, between the fit that read only up to that quarter and the whole-record
    fit, over the Primary's mean capacity.

    Args:
        extras: The per-Primary extras of `collect_metrics`.

    Returns:
        The median and the largest gap over the 24 fits (NaN where a mean is not positive).
    """
    gaps = []
    for p in PRIMARIES:
        control = extras[p]["control"]
        scored = extras[p]["scored"]
        mean = float(np.clip(control["A_cap"], 0.0, None)[scored].mean())
        for q in PAST_ONLY_CUTS:
            path = OUT / "past" / p / f"cut{q}.npz"
            if mean > 0 and path.exists() and scored[q]:
                cut = load_npz(path=path)
                gaps.append(abs(float(cut["A_cap"][q] - control["A_cap"][q])) / mean)
    return {
        "median_gap": float(np.median(gaps)) if gaps else float("nan"),
        "max_gap": float(np.max(gaps)) if gaps else float("nan"),
        "n": float(len(gaps)),
    }


def run_score() -> None:
    """Score every world and write the private score tables and `scores.json`."""
    metrics, extras = collect_metrics()
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    metrics.write_parquet(OUT / "scores" / "world_metrics.parquet")
    c1_medians, c1_values = contrast(metrics=metrics, rival="F")
    c2_medians, c2_values = contrast(metrics=metrics, rival="Qt")
    c2_lo, c2_hi = median_of_medians_interval(values=c2_values)
    c1_lo, c1_hi = median_of_medians_interval(values=c1_values)
    bars = evaluate_bars(metrics=metrics, extras=extras)
    g5 = g5_record(extras=extras)
    gates = json.loads((OUT / "gates.json").read_text())
    gates_ok = all(gates[g]["passed"] is True for g in ("G1", "G1b", "G2", "G3", "G4")) and bool(
        g5["passed"]
    )
    n_converged = sum(
        bool(load_npz(path=OUT / "fits" / p / "control.npz")["A_info"][1]) for p in PRIMARIES
    )
    c1 = c1_status(medians=c1_medians)
    c2 = c2_state(medians=c2_medians)
    state = bars_state(bars={k: v["ok"] for k, v in bars.items()})
    row, verdict = verdict_row(
        gates_ok=gates_ok,
        lookahead=False,
        c1=c1,
        c2=c2,
        bars=state,
        c2_ci_upper=c2_hi,
        n_converged_primaries=n_converged,
    )
    atomic_json(
        obj={
            "C1": {"medians": c1_medians.tolist(), "status": c1, "ci": [c1_lo, c1_hi]},
            "C2": {"medians": c2_medians.tolist(), "state": c2, "ci": [c2_lo, c2_hi]},
            "bars": {k: dict(v) for k, v in bars.items()},
            "G5": g5,
            "real": real_record_aggregates(extras=extras),
            "past_only": past_only_gap(extras=extras),
            "verdict_row": row,
            "verdict": verdict,
            "n_converged_primaries": n_converged,
            "grid_length": GRID_LENGTH,
        },
        path=OUT / "scores" / "scores.json",
    )
    print(f"verdict row {row}: {verdict}")


def report() -> None:
    """Print the report from the stored scores and write it to `report_output.txt`."""
    scores = json.loads((OUT / "scores" / "scores.json").read_text())
    metrics = pl.read_parquet(OUT / "scores" / "world_metrics.parquet")
    lines = [
        (
            f"verdict row {scores['verdict_row']}: {scores['verdict']} "
            "(a synthetic fleet, not the real capacity)"
        ),
        "",
        "C1 (F minus A), median path error difference per Primary: "
        + ", ".join(f"{v:+.3f}" for v in scores["C1"]["medians"])
        + f"; status {scores['C1']['status']}; 95% interval {scores['C1']['ci']}",
        "C2 (Qt minus A), median path error difference per Primary: "
        + ", ".join(f"{v:+.3f}" for v in scores["C2"]["medians"])
        + f"; state {scores['C2']['state']}; 95% interval {scores['C2']['ci']}",
        "",
    ]
    for name in BAR_NAMES:
        record = scores["bars"][name]
        values = {k: v for k, v in record.items() if k != "ok"}
        lines.append(f"{name}: {record['ok']}  {values}")
    lines += [
        "",
        f"G5: {scores['G5']}",
        "",
        f"Real record (private): {scores['real']}",
        f"Past-only sample: {scores['past_only']}",
        "",
    ]
    table = (
        metrics.filter(pl.col("kind") == "main")
        .group_by("arm")
        .agg(
            pl.col("E").median().alias("median_E"),
            pl.col("share").median().alias("median_share"),
            (pl.col("hits").sum() / pl.col("cells").sum()).alias("coverage"),
        )
        .sort("arm")
    )
    lines.append("Main worlds by arm (private):")
    lines.append(str(table))
    by_shape = (
        metrics.filter((pl.col("kind") == "main") & pl.col("arm").is_in(["A", "Qt", "F"]))
        .pivot(on="arm", index=["primary", "world", "shape"], values="E")
        .with_columns(c2=pl.col("Qt") - pl.col("A"))
        .group_by("shape")
        .agg(pl.col("c2").median())
    )
    lines.append("C2 by shape (pooled median):")
    lines.append(str(by_shape))
    text = "\n".join(lines)
    (OUT / "report_output.txt").write_text(text + "\n")
    print(text)

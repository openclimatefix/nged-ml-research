# ruff: noqa: E501
"""Experiment 639: score the test windows, print the minimum detectable effect and every contrast.

CRPS is 2 times the integral of the pinball loss over the quantile levels (trapezoid rule on the 11
levels), in percent of effective capacity. A contrast is arm A's CRPS minus arm B's, averaged over seeds
row by row, so a positive value favours B. Intervals are day-block bootstraps (the unit is a cutoff and
UTC date) at the Bonferroni level over `N_CONTRASTS`.

Usage: python e00639_analyse.py mde        # the null contrast of the free network only
       python e00639_analyse.py contrasts  # every contrast, after all arms are fitted
"""

import argparse
import itertools
import json
from pathlib import Path
from statistics import NormalDist

import e00639_common as k
import numpy as np

N_CONTRASTS = 4
N_RESAMPLES = 2000
ALPHA = 0.05


def crps(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the per-row CRPS (points) from sorted quantiles (rows, 11)."""
    q = np.sort(q, axis=1)
    tau = np.asarray(k.LEVELS)[None, :]
    diff = actual[:, None] - q
    pinball = np.where(diff >= 0, diff * tau, -diff * (1.0 - tau))
    return 2.0 * np.trapezoid(pinball, np.asarray(k.LEVELS), axis=1)


def load(*, world: str) -> dict[str, np.ndarray]:
    """Return the six Primaries' test rows of all cutoffs, stacked, with the day key and masks."""
    parts = []
    for cutoff in k.CUTOFFS:
        z = np.load(k.test_rows_path(cutoff=cutoff, world=world))
        six = z["inc"] > 0
        parts.append(
            {
                "cutoff": np.full(int(six.sum()), k.CUTOFFS.index(cutoff)), "six_mask": six,
                **{name: z[name][six] for name in ("primary", "y", "kt", "hour", "t_us")},
            }
        )  # fmt: skip
    out = {
        name: np.concatenate([p[name] for p in parts])
        for name in ("cutoff", "primary", "y", "kt", "hour", "t_us")
    }
    out["day"] = out["cutoff"] * 100_000 + out["t_us"] // k.k634.DAY_US % 100_000
    out["sunny"] = (
        (out["kt"] >= k.SUNNY_KT)
        & (out["hour"] >= k.SUNNY_HOURS[0])
        & (out["hour"] <= k.SUNNY_HOURS[1])
    )
    out["_six"] = np.concatenate([p["six_mask"] for p in parts])  # for slicing predictions
    return out


def arm_crps(
    *, world: str, arm: str, rows: dict[str, np.ndarray], seeds: tuple[int, ...] = k.SEEDS
) -> np.ndarray:
    """Return the CRPS (percent of capacity) of every row for each seed: shape (seeds, rows)."""
    out = []
    for seed in seeds:
        scores = []
        for cutoff in k.CUTOFFS:
            z = np.load(k.test_rows_path(cutoff=cutoff, world=world))
            six = z["inc"] > 0
            q = np.load(k.prediction_path(cutoff=cutoff, world=world, arm=arm, seed=seed))["q"][six]
            scores.append(
                100.0 * crps(actual=z["y"][six].astype(np.float64), q=q.astype(np.float64))
            )
        out.append(np.concatenate(scores))
    return np.stack(out)


def bootstrap(
    *, diff: np.ndarray, day: np.ndarray, level: float, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) of the mean of `diff` with days resampled as blocks."""
    days, inverse = np.unique(day, return_inverse=True)
    sums = np.bincount(inverse, weights=diff)
    counts = np.bincount(inverse).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(days), size=(N_RESAMPLES, len(days)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(diff.sum() / counts.sum()), float(lo), float(hi)


def level_bonferroni() -> float:
    """Return the two-sided interval level after a Bonferroni correction over the planned contrasts."""
    return 1.0 - ALPHA / N_CONTRASTS


def mde(*, world: str = "step") -> float:
    """Return the mean half-width of the null contrast: every split of the free network's seeds into 3 and 2."""
    rows = load(world=world)
    scores = arm_crps(world=world, arm="free", rows=rows)
    mask = rows["sunny"]
    widths = []
    for chosen in itertools.combinations(range(len(k.SEEDS)), 3):
        rest = [s for s in range(len(k.SEEDS)) if s not in chosen]
        diff = scores[list(chosen)].mean(axis=0) - scores[rest].mean(axis=0)
        _, lo, hi = bootstrap(diff=diff[mask], day=rows["day"][mask], level=level_bonferroni())
        widths.append((hi - lo) / 2)
    return float(np.mean(widths))


def contrast(*, world_a: str, arm_a: str, world_b: str, arm_b: str, subset: str) -> dict:
    """Return arm A minus arm B on a subset ("sunny" or "all"), with the interval and the per-seed spread."""
    rows = load(world=world_a)
    a = arm_crps(world=world_a, arm=arm_a, rows=rows)
    b = arm_crps(world=world_b, arm=arm_b, rows=rows)
    mask = rows["sunny"] if subset == "sunny" else np.ones(len(rows["y"]), dtype=bool)
    diff = (a - b).mean(axis=0)
    mean, lo, hi = bootstrap(diff=diff[mask], day=rows["day"][mask], level=level_bonferroni())
    per_seed = (a - b)[:, mask].mean(axis=1)
    return {
        "world": world_a, "a": arm_a, "b": arm_b, "subset": subset, "n_rows": int(mask.sum()),
        "diff": mean, "lo": lo, "hi": hi, "per_seed_sd": float(per_seed.std(ddof=1)), "z": NormalDist().inv_cdf(0.975),
    }  # fmt: skip


def checks(*, out_dir: Path) -> None:
    """Print the per-cutoff and per-Primary contrasts, the bias of the free network, and the implied step energy."""
    result: dict = {}
    for world in k.WORLDS:
        rows = load(world=world)
        free = arm_crps(world=world, arm="free", rows=rows)
        cal = arm_crps(world=world, arm="cal", rows=rows)
        diff = (free - cal).mean(axis=0)
        sunny = rows["sunny"]
        for ci, cutoff in enumerate(k.CUTOFFS):
            m = sunny & (rows["cutoff"] == ci)
            result[f"{world}_cutoff_{cutoff}"] = float(diff[m].mean())
        for p in np.unique(rows["primary"]):
            m = sunny & (rows["primary"] == p)
            result[f"{world}_primary_{int(p)}"] = float(diff[m].mean())
        # The free network shifted by its own mean error in each cutoff and Primary (a hindsight diagnostic).
        shifted, gap_rows = [], []
        for ci, cutoff in enumerate(k.CUTOFFS):
            z = np.load(k.test_rows_path(cutoff=cutoff, world=world))
            six = z["inc"] > 0
            y = z["y"][six].astype(np.float64)
            sun = rows["sunny"][rows["cutoff"] == ci]
            prim = rows["primary"][rows["cutoff"] == ci]
            per_seed = []
            for seed in k.SEEDS:
                q = np.load(k.prediction_path(cutoff=cutoff, world=world, arm="free", seed=seed))[
                    "q"
                ][six].astype(np.float64)
                q = q.copy()
                for p in np.unique(prim):
                    m = sun & (prim == p)
                    q[prim == p] -= (q[m, 5] - y[m]).mean()
                per_seed.append(100 * crps(actual=y, q=q))
            shifted.append(np.mean(per_seed, axis=0))
            gap_rows.append(sun)
        shifted_all = np.concatenate(shifted)
        result[f"{world}_free_bias_corrected_minus_cal_sunny"] = float(
            (shifted_all - cal.mean(axis=0))[sunny].mean()
        )
        # Median error and the approximate mean forecast from the 11 quantiles.
        for arm in k.ARMS_BY_WORLD[world]:
            med, mean_gap = [], []
            for cutoff in k.CUTOFFS:
                z = np.load(k.test_rows_path(cutoff=cutoff, world=world))
                six = z["inc"] > 0
                q = np.mean(
                    [
                        np.load(k.prediction_path(cutoff=cutoff, world=world, arm=arm, seed=s))[
                            "q"
                        ][six]
                        for s in k.SEEDS
                    ],
                    axis=0,
                )
                approx_mean = np.trapezoid(q, np.asarray(k.LEVELS), axis=1) / (
                    k.LEVELS[-1] - k.LEVELS[0]
                )
                med.append(100 * np.abs(q[:, 5] - z["y"][six]))
                mean_gap.append(100 * (q[:, 5] - approx_mean))
            med_all, gap_all = np.concatenate(med), np.concatenate(mean_gap)
            result[f"{world}_{arm}_median_abs_error_sunny"] = float(med_all[sunny].mean())
            result[f"{world}_{arm}_median_minus_mean_sunny"] = float(gap_all[sunny].mean())
    (out_dir / "checks.json").write_text(json.dumps(result, indent=1))
    for name, value in result.items():
        print(f"{name:55s} {value:+.4f}")


def main() -> None:
    """Print the minimum detectable effect, or every contrast."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("mde", "contrasts", "checks"))
    args = parser.parse_args()
    out_dir = k.WORK / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "checks":
        checks(out_dir=out_dir)
        return
    if args.command == "mde":
        values = {w: mde(world=w) for w in k.WORLDS}
        (out_dir / "mde.json").write_text(json.dumps(values, indent=1))
        print(
            "MDE (half-width of the null contrast, CRPS points, sunny middays):", values, flush=True
        )
        return
    results = []
    for world in k.WORLDS:
        for arm in k.ARMS_BY_WORLD[world]:
            if arm == "cal":
                continue
            results.extend(
                contrast(world_a=world, arm_a=arm, world_b=world, arm_b="cal", subset=subset)
                for subset in ("sunny", "all")
            )
    (out_dir / "contrasts.json").write_text(json.dumps(results, indent=1))
    for r in results:
        print(
            f"{r['world']:5s} {r['a']:7s} minus cal {r['subset']:5s} n={r['n_rows']:6d} "
            f"{r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] seed sd {r['per_seed_sd']:.4f}"
        )


if __name__ == "__main__":
    main()

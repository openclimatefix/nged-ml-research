"""Experiment 618, step 2a: known-answer test of the capacity estimator, with placebos.

Base series: each testable farm's daily index divided by its own fitted piecewise-constant level
(penalty 12), so real level steps are removed and the real day-to-day noise stays. Capacity steps
of known size are injected into the base series. Placebos permute the order of the days (a
no-change series) and add a smooth fall (a drift series).

```bash
export DATA_PATH_INTERNAL=...
PYTHONPATH=ml-experiments/00618-* uv run --no-sync python \
  ml-experiments/00618-*/e00618_known_answer.py
```
"""

from typing import Final

import e00618_common as c
import numpy as np
import polars as pl

BASE_PENALTY: Final[float] = 12.0
SIZES: Final[tuple[float, ...]] = (0.04, 0.08, 0.12, 0.20)
N_TUNE: Final[int] = 20
N_EVAL: Final[int] = 40
MATCH_DAYS: Final[int] = 5
SIZE_TOLERANCE: Final[float] = 0.25
BLOCK: Final[int] = 7
DRIFT_PER_YEAR: Final[float] = 0.10 / 3.0
INFLATE_GRID: Final[tuple[float, ...]] = (1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5)
FALSE_STEP_BAR_PER_YEAR: Final[float] = 1.0
DRIFT_STEP_BAR_PER_YEAR: Final[float] = 3.0


def load_farms() -> dict[str, dict]:
    """Return per farm: raw daily values, base series, years since the first reading, span."""
    rows = c.load_rows()
    out = {}
    for farm in c.FARMS:
        daily = c.daily_index(rows=rows, farm=farm)
        first = rows.filter(pl.col("site") == farm)["time"].min().date()  # ty: ignore[unresolved-attribute]
        years = np.array([(d - first).days / 365.25 for d in daily["day"]])
        v = daily["value"].to_numpy()
        _, level, _ = c.fit_levels(values=v, penalty=BASE_PENALTY)
        out[farm] = {
            "day": daily["day"].to_list(),
            "values": v,
            "base": v / level,
            "years": years,
            "span": float(years[-1] - years[0]),
        }
    return out


def permute(*, x: np.ndarray, rng: np.random.Generator, block: int) -> np.ndarray:
    """Return `x` with its days shuffled singly (`block` 1) or in consecutive blocks."""
    if block == 1:
        return rng.permutation(x)
    starts = np.arange(0, len(x), block)
    pieces = [x[s : s + block] for s in starts]
    order = rng.permutation(len(pieces))
    return np.concatenate([pieces[i] for i in order])


def placebo_table(*, farms: dict, seeds: range, tag: str) -> pl.DataFrame:
    """Count steps found in permuted series, per farm, kind, and penalty."""
    recs = []
    for farm in c.TESTABLE:
        f = farms[farm]
        for kind, series, block in (
            ("real_days_shuffled", f["values"], 1),
            ("real_blocks_shuffled", f["values"], BLOCK),
            ("base_days_shuffled", f["base"], 1),
            ("base_blocks_shuffled", f["base"], BLOCK),
        ):
            for seed in seeds:
                rng = np.random.default_rng(seed)
                y = permute(x=series, rng=rng, block=block)
                for pen in c.PENALTY_GRID:
                    steps, _, _ = c.fit_levels(values=y, penalty=pen)
                    recs.append(
                        {
                            "farm": farm,
                            "kind": kind,
                            "seed": seed,
                            "penalty": pen,
                            "steps": len(steps),
                            "per_year": len(steps) / f["span"],
                            "set": tag,
                        }
                    )
    return pl.DataFrame(recs)


def choose_penalty(*, tuning: pl.DataFrame) -> float:
    """Return the smallest penalty whose mean false steps a year is within the bar.

    The mean is taken over the tuning permutations, and the worst farm and the worst of the four
    placebo kinds (days or 7-day blocks shuffled, real or step-free series) must meet the bar. The
    brief names the day shuffle; the block shuffle keeps short-range weather persistence, so it is
    the stricter and more realistic placebo, and it was added before any real fit.
    """
    for pen in c.PENALTY_GRID:
        worst = (
            tuning.filter(pl.col("penalty") == pen)
            .group_by("farm", "kind")
            .agg(pl.col("per_year").mean())["per_year"]
            .max()
        )
        if worst <= FALSE_STEP_BAR_PER_YEAR:  # ty: ignore[unsupported-operator]
            return pen
    return c.PENALTY_GRID[-1]


def inject_runs(*, farms: dict, seeds: range, n_starts: int, tag: str) -> pl.DataFrame:
    """Inject one permanent capacity step per run and record what the estimator finds."""
    recs = []
    for farm in c.TESTABLE:
        f = farms[farm]
        n = len(f["base"])
        rng = np.random.default_rng(1000 + c.FARMS.index(farm) + (0 if tag == "tune" else 500))
        starts = rng.integers(15, n - 15, size=n_starts)
        for size in SIZES:
            for sign in (1.0, -1.0):
                delta = sign * size
                for s in starts:
                    truth = np.where(np.arange(n) >= s, 1.0 + delta, 1.0)
                    y = f["base"] * truth
                    for pen in c.PENALTY_GRID:
                        steps, level, sigma = c.fit_levels(values=y, penalty=pen)
                        match = [j for j in steps if abs(j - s) <= MATCH_DAYS]
                        est = np.nan
                        if match:
                            j = min(match, key=lambda m: abs(m - s))
                            before = level[j - 1]
                            est = level[j] / before - 1.0
                        rec = {
                            "farm": farm,
                            "set": tag,
                            "delta": delta,
                            "start": int(s),
                            "penalty": pen,
                            "n_steps": len(steps),
                            "detected": bool(match) and np.sign(est) == np.sign(delta),
                            "est": est,
                        }
                        for infl in INFLATE_GRID:
                            half = c.segment_halfwidth(steps=steps, n=n, sigma=sigma, inflate=infl)
                            rec[f"cover_{infl}"] = float(np.mean(np.abs(level - truth) <= half))
                        recs.append(rec)
    return pl.DataFrame(recs)


def drift_runs(*, farms: dict, seeds: range, penalty: float) -> pl.DataFrame:
    """Permuted base series times a linear fall of 10% in three years."""
    recs = []
    for farm in c.TESTABLE:
        f = farms[farm]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            y = permute(x=f["base"], rng=rng, block=1) * (
                1.0 - DRIFT_PER_YEAR * (f["years"] - f["years"][0])
            )
            steps, level, _ = c.fit_levels(values=y, penalty=penalty)
            truth_end = 1.0 - DRIFT_PER_YEAR * (f["years"][-1] - f["years"][0])
            recs.append(
                {
                    "farm": farm,
                    "seed": seed,
                    "steps_per_year": len(steps) / f["span"],
                    "final_error": level[-1] / truth_end - 1.0,
                }
            )
    return pl.DataFrame(recs)


def main() -> None:
    """Run tuning, evaluation, and drift placebos; write CSVs and print the pass-bar table."""
    farms = load_farms()
    c.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tuning = placebo_table(farms=farms, seeds=range(N_TUNE), tag="tune")
    penalty = choose_penalty(tuning=tuning)
    print("chosen penalty (in sigma^2):", penalty)
    evaluation = placebo_table(farms=farms, seeds=range(100, 100 + N_EVAL), tag="eval")
    pl.concat([tuning, evaluation]).write_csv(c.DATA_DIR / "placebo_steps.csv")
    ev = (
        evaluation.filter(pl.col("penalty") == penalty)
        .group_by("farm", "kind")
        .agg(pl.col("per_year").mean(), pl.col("per_year").max().alias("worst"))
        .sort("farm", "kind")
    )
    print(ev)
    tune_inj = inject_runs(farms=farms, seeds=range(1), n_starts=15, tag="tune")
    eval_inj = inject_runs(farms=farms, seeds=range(1), n_starts=40, tag="eval")
    pl.concat([tune_inj, eval_inj]).write_csv(c.DATA_DIR / "injected_runs.csv")
    cov_cols = [f"cover_{i}" for i in INFLATE_GRID]
    tune_cov = tune_inj.filter(pl.col("penalty") == penalty).select(cov_cols).mean().to_dicts()[0]
    inflate = min(INFLATE_GRID, key=lambda i: abs(tune_cov[f"cover_{i}"] - 0.90))
    print("tuning coverage by inflation:", tune_cov, "chosen:", inflate)
    eval_cov = eval_inj.filter(pl.col("penalty") == penalty).select(cov_cols).mean().to_dicts()[0]
    print("eval coverage by inflation:", eval_cov)
    (c.DATA_DIR / "chosen_inflation.txt").write_text(str(inflate))
    drift = drift_runs(farms=farms, seeds=range(100, 120), penalty=penalty)
    drift.write_csv(c.DATA_DIR / "drift_runs.csv")
    print(drift.group_by("farm").agg(pl.col("steps_per_year").mean(), pl.col("final_error").mean()))
    recall = (
        eval_inj.filter(pl.col("penalty") == penalty)
        .group_by("farm", pl.col("delta").abs().alias("size"))
        .agg(recall=pl.col("detected").mean())
        .sort("farm", "size")
        .pivot(on="size", index="farm", values="recall")
    )
    print(recall)
    (c.DATA_DIR / "chosen_penalty.txt").write_text(str(penalty))


if __name__ == "__main__":
    main()

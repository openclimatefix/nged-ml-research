"""Experiment 617: does knowing the turbine count help? Known-answer recall at W1, count fixed.

The count comes from the private register match in `DATA_DIR/register_candidates_private.parquet`
(nearest row of W1). It is never printed. Compares recall with the count fixed and inferred.
"""

from dataclasses import replace

import e00617_hmm as h
import e00617_known_answer as k
import numpy as np
import polars as pl

STEP_UP = 0.001


def main() -> None:
    """Print recall with the count fixed and inferred."""
    reg = pl.read_parquet(h.DATA_DIR / "register_candidates_private.parquet").filter(
        pl.col("site") == "W1"
    )
    n_fixed = float(reg["n_turbines"][0])
    base = h.remove_fault_days(h.daily_series("W1"))
    rows = []
    for m in (2, 3):
        for dur in (14, 28, 60):
            for rep in range(3):
                rng = np.random.default_rng(10 * rep + m + dur)
                starts = k._starts(rng, base.n_days, dur, k.EVENTS_PER_RUN)
                factor = np.ones(base.n_days)
                for s0 in starts:
                    factor[s0 : s0 + dur] = 1 - m / n_fixed
                value = base.value * factor[base.day]
                shaped = replace(base, value=value, ceiling=float(np.quantile(value, 0.99)))
                for arm, grid in (
                    ("count fixed", np.array([n_fixed])),
                    ("count inferred", h.N_GRID),
                ):
                    fits = h.fit_grid(shaped, step_up=STEP_UP, n_grid=grid)
                    summ = h.deficit_summary(fits, h.log_n_weights(fits))
                    unit = base.ceiling / n_fixed
                    for s0 in starts:
                        core = slice(s0 + dur // 4, s0 + dur - dur // 4)
                        pre = slice(max(s0 - k.BEFORE_DAYS, 0), s0)
                        step = np.median(summ["mean"][core]) - np.median(summ["mean"][pre])
                        rows.append(
                            {
                                "arm": arm,
                                "m": m,
                                "duration": dur,
                                "detected": step >= 0.5 * m * unit,
                            }
                        )
    out = pl.DataFrame(rows)
    print(
        out.group_by("arm", "m", "duration")
        .agg(recall=pl.col("detected").mean())
        .sort("arm", "m", "duration")
    )
    print(out.group_by("arm").agg(recall=pl.col("detected").mean()))


if __name__ == "__main__":
    main()

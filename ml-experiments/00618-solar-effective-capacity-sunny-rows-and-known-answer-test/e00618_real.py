"""Experiment 618, step 2b: fit the real daily indices of Solar A to F at the tuned penalty.

Reads the penalty and the interval inflation chosen by `e00618_known_answer.py`. Writes
`real_levels.csv` (one row per sunny day), `real_steps.csv`, and `real_comparison.csv` (mean level
in 2021 to 2023 against 2025 to 2026, the contrast experiment 453 saw by eye).

```bash
PYTHONPATH=ml-experiments/00618-* uv run --no-sync python ml-experiments/00618-*/e00618_real.py
```
"""

import e00618_common as c
import e00618_known_answer as ka
import numpy as np
import polars as pl

N_BOOT: int = 2000
BOOT_BLOCK: int = 7


def block_bootstrap_ratio(*, early: np.ndarray, late: np.ndarray, seed: int) -> tuple[float, float]:
    """Return a 90% interval for mean(early)/mean(late) by resampling blocks of days."""
    rng = np.random.default_rng(seed)

    def draw(x: np.ndarray) -> float:
        n_blocks = int(np.ceil(len(x) / BOOT_BLOCK))
        starts = rng.integers(0, max(len(x) - BOOT_BLOCK, 1), size=n_blocks)
        return float(np.mean(np.concatenate([x[s : s + BOOT_BLOCK] for s in starts])))

    r = [draw(early) / draw(late) for _ in range(N_BOOT)]
    return float(np.quantile(r, 0.05)), float(np.quantile(r, 0.95))


def main() -> None:
    """Fit the six farms and write the tables."""
    penalty = float((c.DATA_DIR / "chosen_penalty.txt").read_text())
    inflate = float((c.DATA_DIR / "chosen_inflation.txt").read_text())
    farms = ka.load_farms()
    levels, steps_out, comp = [], [], []
    for farm in c.FARMS:
        f = farms[farm]
        v = f["values"]
        steps, level, sigma = c.fit_levels(values=v, penalty=penalty)
        half = c.segment_halfwidth(steps=steps, n=len(v), sigma=sigma, inflate=inflate)
        years = f["years"]
        days = f["day"]
        levels.append(
            pl.DataFrame(
                {
                    "farm": farm,
                    "year": [d.year for d in days],
                    "years_since_first": years,
                    "value": v,
                    "level": level,
                    "lo": level - half,
                    "hi": level + half,
                }
            )
        )
        for j in steps:
            steps_out.append(
                {
                    "farm": farm,
                    "years_since_first": float(years[j]),
                    "change": float(level[j] / level[j - 1] - 1.0),
                    "days_before": j,
                    "days_after": len(v) - j,
                }
            )
        yr = np.array([d.year for d in days])
        early = v[(yr >= 2021) & (yr <= 2023)]
        late = v[(yr >= 2025) & (yr <= 2026)]
        lo, hi = (np.nan, np.nan)
        if len(early) > 20 and len(late) > 20:
            lo, hi = block_bootstrap_ratio(early=early, late=late, seed=7)
        comp.append(
            {
                "farm": farm,
                "days_total": len(v),
                "n_steps": len(steps),
                "days_2021_23": len(early),
                "days_2025_26": len(late),
                "level_2021_23_over_2025_26": float(np.mean(early) / np.mean(late))
                if len(early) and len(late)
                else np.nan,
                "ratio_lo": lo,
                "ratio_hi": hi,
                "sigma_daily": sigma,
            }
        )
    pl.concat(levels).write_csv(c.DATA_DIR / "real_levels.csv")
    pl.DataFrame(steps_out).write_csv(c.DATA_DIR / "real_steps.csv")
    out = pl.DataFrame(comp)
    out.write_csv(c.DATA_DIR / "real_comparison.csv")
    print(out)
    print(pl.DataFrame(steps_out))


if __name__ == "__main__":
    main()

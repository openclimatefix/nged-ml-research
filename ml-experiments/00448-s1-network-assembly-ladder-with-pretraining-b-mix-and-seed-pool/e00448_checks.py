# ruff: noqa: E501
"""Mean-forecast checks for experiment 448 (audit trigger 3).

The stored forecasts hold 11 quantiles, not the mixture parameters, so the mean forecast is
approximated by integrating the quantile function: each quantile carries the probability mass
between the midpoints to its neighbours (the outer two carry the tails down to 0 and up to 1). The
approximation is the same for every arm, so it ranks arms, but the tails make its level
approximate; `validate` compares it with the exact mixture mean of S1 seed 0 on one month. The
script reads the raw (unwidened) quantiles and keeps only rows with a finite actual. For each
arm, group, and Primary the script prints, in points of capacity (one point is 1%), on clean real-world rows, days 1 to 10, averaged over seeds 0 to 4: the median minus the
approximate mean, the mean-forecast error (bias and mean absolute error), and the median's
absolute error.

```bash
source <paths>; PYTHONPATH=$PP DATA_PATH_INTERNAL=... \
  uv run --no-sync python ml-experiments/00448-*/e00448_checks.py
```
"""

from typing import Final

import e00428_score as score  # ty: ignore[unresolved-import]
import e00448_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

ARMS: Final[tuple[str, ...]] = (
    "S1", "S1PT", "S1-Bmix", "S1-pool5", "S1PT-pool5", "R-inj", "L0", "L0PT",
)  # fmt: skip
LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99)
QUANTILE_COLUMNS: Final[tuple[str, ...]] = tuple(f"q{round(100 * q):02d}" for q in LEVELS)
POINTS: Final[float] = 100.0
MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)


def weights() -> np.ndarray:
    """Return the probability mass carried by each of the 11 quantiles."""
    edges = np.concatenate([[0.0], (np.array(LEVELS)[1:] + np.array(LEVELS)[:-1]) / 2.0, [1.0]])
    return np.diff(edges)


def arm_rows(*, arm: str) -> pl.DataFrame:
    """Return per-row checks of one arm on clean real-world rows, averaged over seeds."""
    frames = []
    w = weights()
    for month in MONTHS:
        seed_frames = []
        for seed in SEEDS:
            f = pl.read_parquet(run.WORK_DIR / "predictions" / arm / f"{month}_s{seed}_w0.parquet")
            q = f.select(list(QUANTILE_COLUMNS)).to_numpy()
            q = np.sort(q, axis=1)
            mean = q @ w
            seed_frames.append(
                f.select("primary", "lead", "actual", "win_real", "tgt_post").with_columns(
                    median=pl.Series(q[:, 5]), mean=pl.Series(mean)
                )
            )
        base = seed_frames[0]
        merged = base.with_columns(
            median=pl.Series(np.mean([s["median"].to_numpy() for s in seed_frames], axis=0)),
            mean=pl.Series(np.mean([s["mean"].to_numpy() for s in seed_frames], axis=0)),
        )
        frames.append(merged.with_columns(month=pl.lit(month)))
    table = pl.concat(frames)
    return table.filter(
        ~pl.col("win_real")
        & ~pl.col("tgt_post")
        & (pl.col("lead") <= 240.0)
        & pl.col("actual").is_finite()
    )


def validate(*, device: str) -> None:
    """Compare the quantile-integral mean with the exact mixture mean of S1 seed 0, 2026-01."""
    import e00428_data as data  # ty: ignore[unresolved-import]
    import e00428_fit as fit  # ty: ignore[unresolved-import]
    import e00428_predict as pr  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    run.install()
    month, seed = "2026-01", 0
    power_np, _ = data.load_stack(include_eval=True)
    power = torch.from_numpy(power_np).to(device)
    loaded = run.load_from(
        folder=run.locate(arm="S1", month=month, seed=seed), arm_name="S1", device=device
    )
    test = data.build_test(month=month, standardiser=loaded.standardiser)
    inputs = fit.Inputs(
        power=power, x=test.x, primary=test.primary, origin_idx=test.origin_idx,
        target_idx=test.target_idx, power_mean=loaded.power_mean, power_std=loaded.power_std,
        device=device, surprise=fit.surprise_for(arm=loaded.arm, month=month, seed=seed, device=device),
    )  # fmt: skip
    rows = torch.arange(0, min(20_000, len(test.x)), device=device)
    world = torch.zeros(len(rows), dtype=torch.long, device=device)
    res = pr.forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world)
    primary = inputs.primary[rows]
    mu = torch.from_numpy(loaded.target_mean).to(device)[primary][:, None]
    sd = torch.from_numpy(loaded.target_std).to(device)[primary][:, None]
    exact = (res["log_w"].exp() * (res["mean"] * sd + mu)).sum(dim=1).cpu().numpy()
    approx = res["q"].cpu().numpy() @ weights()
    gap = (approx - exact) * POINTS
    print(
        f"validate: approximate minus exact mixture mean, points: mean {gap.mean():+.3f}, "
        f"mean absolute {np.abs(gap).mean():.3f}, 99th percentile {np.percentile(np.abs(gap), 99):.3f}"
    )


def main() -> None:
    """Print the checks per arm and group, and the median minus mean per Primary."""
    run.install()
    validate(device="cuda")
    rows = []
    for arm in ARMS:
        if not (run.WORK_DIR / "predictions" / arm).exists():
            continue
        t = arm_rows(arm=arm)
        group = score.group_of(primary=t["primary"].to_numpy())
        for name, mask in group.items():
            sel = t.filter(pl.Series(mask))
            err = (sel["mean"] - sel["actual"]).to_numpy() * POINTS
            med = (sel["median"] - sel["actual"]).to_numpy() * POINTS
            gap = (sel["median"] - sel["mean"]).to_numpy() * POINTS
            rows.append(
                {
                    "arm": arm, "group": name, "n": sel.height,
                    "median_minus_mean": float(gap.mean()), "mean_bias": float(err.mean()),
                    "mean_mae": float(np.abs(err).mean()), "median_mae": float(np.abs(med).mean()),
                }
            )  # fmt: skip
        per = (
            t.group_by("primary")
            .agg((pl.col("median") - pl.col("mean")).mean().alias("gap"))
            .sort("primary")
        )
        print(
            arm,
            "median minus mean per Primary (points):",
            [round(v * POINTS, 3) for v in per["gap"]],
        )
    out = pl.DataFrame(rows)
    out.write_csv(run.WORK_DIR / "figure_tables" / "mean_forecast_checks.csv")
    with pl.Config(tbl_rows=100, tbl_cols=12, tbl_width_chars=200):
        print(out.sort("group", "arm"))


if __name__ == "__main__":
    main()

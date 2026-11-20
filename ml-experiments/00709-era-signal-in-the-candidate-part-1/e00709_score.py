r"""Score experiment 709's stored forecasts on the real rows (interim table).

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00709-*/e00709_score.py --folds 2025-11 2026-01 \
        --seeds 0 1 --arms none aug oracle placebo [--resamples N]

Reads the prediction sets `<arm>-<month>-s<seed>` of experiment 709 (quantiles in MW at the
contract's 12:00 UTC origin), divides them by each Primary's effective capacity, and scores them
against the cleaned half-hourly power (`clean_stack`, the series the neural network trains on, with
flagged half-hours set to NaN). The score is the quantile score (CRPS), in points of effective
capacity. For each arm it prints the mean CRPS over all rows, at D8, and on the other
Primaries, and the paired difference against the reference arm with a bootstrap interval that
resamples whole origin months. Every number is interim: the screen uses 2 seeds. Nothing here
fits a neural network.
"""

import argparse
import pickle
from pathlib import Path
from typing import Final

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00709_features as features  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.eval_protocol.metrics import quantile_score
from studies.experiment_store import losses_for_bootstrap, read_predictions

ROW_SET_ID: Final = "experiment_677_monthly_refit_rows_valid_2025_11_to_2026_06"
E_ROW_SET_ID: Final = "experiment_709_phase_e_monthly_refit_rows_valid_2025_04_to_2025_06"
"""Phase E arms are named `e-<arm>`; their sets carry this row set id, and they are scored apart."""
QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
D8_LABEL: Final = "D8"
SCORES_DIR: Final[Path] = features.DATA_DIR / "scores"


def series_to_site() -> dict[int, int]:
    """Return the Primary index of each `time_series_id` (the site map's `primary` column)."""
    sites = k578.sites()
    assert tuple(sites["label"].to_list()) == features.PRIMARIES, (
        "site map and panel differ in order"
    )
    assert sites["primary"].to_list() == list(range(len(features.PRIMARIES))), (
        "primary != row index"
    )
    return {int(t): int(p) for t, p in zip(sites["time_series_id"], sites["primary"], strict=True)}


def row_losses(
    *, forecast: pl.DataFrame, truth: np.ndarray, capacity_mw: np.ndarray, to_site: dict[int, int]
) -> pl.DataFrame:
    """Return each forecast row's CRPS in points of capacity, on the rows with a finite reading."""
    site = np.array([to_site[int(t)] for t in forecast["time_series_id"].to_list()])
    start = np.datetime64(features.GRID_START.replace(tzinfo=None), "us")
    valid = forecast["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    idx = ((valid - start) // np.timedelta64(1800, "s")).astype(np.int64)
    inside = (idx >= 0) & (idx < truth.shape[1])
    idx = np.clip(idx, 0, truth.shape[1] - 1)
    actual = np.where(inside, truth[site, idx], np.nan) * 100.0
    capacity = capacity_mw[site]
    q = forecast.select(list(QCOLS)).to_numpy().astype(np.float64) / capacity[:, None] * 100.0
    finite = np.isfinite(actual)
    crps = np.full(len(site), np.nan)
    crps[finite] = quantile_score(actual=actual[finite], quantiles=q[finite])
    return (
        forecast.select("time_series_id", "power_fcst_init_time", "valid_time")
        .with_columns(pl.Series("crps", crps), pl.Series("primary", site))
        .filter(pl.col("crps").is_not_nan())
    )


POST_SHIFT_DAYS: Final = 14
"""Origins within this many days after an alarm of the stat detector count as after a shift."""


def post_shift_expression() -> pl.Expr:
    """Return a row filter for origins within `POST_SHIFT_DAYS` days after a stat detector alarm.

    The stat detector's alarms are the same for every arm, so every arm is scored on the same rows
    and the paired bootstrap stays valid. The slice says nothing about which shifts are eras: the
    detector also alarms on switching events.

    Returns:
        A Polars boolean expression over `primary` and `origin_day`.
    """
    episodes = pl.read_parquet(features.DATA_DIR / "signals" / "stat_episodes.parquet")
    expr = pl.lit(False)
    for site, alarm in zip(episodes["site"].to_list(), episodes["alarm"].to_list(), strict=True):
        expr = expr | (
            (pl.col("primary") == site)
            & (pl.col("origin_day") >= alarm)
            & (pl.col("origin_day") <= alarm + POST_SHIFT_DAYS)
        )
    return expr


def touched_expression(*, folds: list[str], seeds: list[int]) -> pl.Expr:
    """Return a row filter for the Primaries that the curriculum world touched in all seeds.

    Only Primaries touched in every seed count, so each seed is scored on the same rows.

    Args:
        folds: Fold months scored.
        seeds: Seeds scored.

    Returns:
        A Polars boolean expression over `fold` and `primary`.
    """
    expr = pl.lit(False)
    for month in folds:
        common: set[int] | None = None
        for seed in seeds:
            path = features.DATA_DIR / "worlds" / f"curriculum_{month}_s{seed}.pkl"
            with path.open("rb") as handle:
                parts = pickle.load(handle)
            touched = {int(p) for p in parts["touched"]}
            common = touched if common is None else common & touched
        expr = expr | (pl.col("fold") == month) & pl.col("primary").is_in(sorted(common or ()))
    return expr


def read_arm(
    *,
    arm: str,
    folds: list[str],
    seeds: list[int],
    truth: np.ndarray,
    capacity_mw: np.ndarray,
    to_site: dict[int, int],
) -> pl.DataFrame:
    """Return one arm's per-row losses over every fold and seed."""
    parts = []
    for month in folds:
        for seed in seeds:
            name = f"{arm}-{month}-s{seed}"
            frame = read_predictions(
                experiment_id=features.EXPERIMENT_ID,
                name=name,
                row_set_id=E_ROW_SET_ID if arm.startswith("e-") else ROW_SET_ID,
            )
            parts.append(
                row_losses(
                    forecast=frame, truth=truth, capacity_mw=capacity_mw, to_site=to_site
                ).with_columns(pl.lit(seed).alias("seed"), pl.lit(month).alias("fold"))
            )
    return pl.concat(parts)


def main() -> None:
    """Score the arms and print the interim table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--reference", default="none")
    parser.add_argument("--level", type=float, default=95.0)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--tag", default="real_rows", help="the output file is interim_<tag>.csv")
    args = parser.parse_args()
    panel = features.load_panel()
    to_site = series_to_site()
    sites = k578.sites()
    assert np.allclose(sites["capacity_mw"].to_numpy(), panel.capacity_mw, rtol=1e-4), (
        "capacities differ"
    )
    truth = k578.clean()[:, : panel.power.shape[1]].astype(
        np.float64
    )  # cleaned share, same start as the panel
    # every set must exist before any is read
    missing = [
        f"{a}-{m}-s{sd}"
        for a in dict.fromkeys([args.reference, *args.arms])
        for m in args.folds
        for sd in args.seeds
        if not (features.DATA_DIR / "predictions" / f"{a}-{m}-s{sd}").exists()
    ]
    if missing:
        msg = f"missing prediction sets: {missing}"
        raise FileNotFoundError(msg)
    d8 = features.PRIMARIES.index(D8_LABEL)
    arms = list(dict.fromkeys([args.reference, *args.arms]))
    losses = {
        a: read_arm(
            arm=a,
            folds=args.folds,
            seeds=args.seeds,
            truth=truth,
            capacity_mw=panel.capacity_mw,
            to_site=to_site,
        )
        for a in arms
    }
    # every arm holds the same rows: the bootstrap helper checks this and raises when they differ
    both = losses_for_bootstrap(arm_losses=losses)
    both = both.with_columns(
        origin_day=(
            (
                pl.col("power_fcst_init_time").dt.replace_time_zone(None)
                - pl.lit(features.GRID_START.replace(tzinfo=None))
            ).dt.total_days()
        )
    )
    post_shift = post_shift_expression()
    touched = touched_expression(folds=args.folds, seeds=args.seeds)
    scopes = {
        "all rows": pl.lit(True),
        "D8": pl.col("primary") == d8,
        "other Primaries": pl.col("primary") != d8,
        f"weeks after a stat alarm (<= {POST_SHIFT_DAYS} days), all Primaries": post_shift,
        f"weeks after a stat alarm (<= {POST_SHIFT_DAYS} days), D8": post_shift
        & (pl.col("primary") == d8),
        "other rows": ~post_shift,
        "touched Primaries (curriculum world)": touched,
    }
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    print(
        f"INTERIM: seeds {args.seeds}, folds {args.folds} (an interval over fewer than 4 "
        "fold months "
        "is indicative only); CRPS in points of effective capacity; "
        "difference = arm minus reference\n"
    )
    for scope, expr in scopes.items():
        scoped = both.filter(expr)
        ref_mean = float(scoped.filter(pl.col("arm") == args.reference)["crps"].mean())  # ty: ignore[invalid-argument-type]
        for arm in arms:
            mean = float(scoped.filter(pl.col("arm") == arm)["crps"].mean())  # ty: ignore[invalid-argument-type]
            low = high = float("nan")
            if arm != args.reference:
                low, high = bootstrap_difference_at_level(
                    losses=scoped,
                    treatment=arm,
                    reference=args.reference,
                    metric="crps",
                    level=args.level,
                    n_resamples=args.resamples,
                )
            out.append(
                {
                    "scope": scope,
                    "arm": arm,
                    "crps": mean,
                    "diff": mean - ref_mean,
                    "lower": low,
                    "upper": high,
                }
            )
            print(
                f"{scope:16s} {arm:12s} crps {mean:7.4f}  diff {mean - ref_mean:+.4f} "
                f"[{low:+.4f}, {high:+.4f}]"
            )
    pl.DataFrame(out).write_csv(SCORES_DIR / f"interim_{args.tag}.csv")


if __name__ == "__main__":
    main()

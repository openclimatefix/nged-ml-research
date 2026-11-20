"""Score experiment 709's injected-bank forecasts on injected eras and switching events (interim).

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00709-*/e00709_bank_score.py --folds 2025-11 2026-01 \
        2026-04 2026-06 --seeds 0 1 --arms oracle placebo all rule soft hmm stat [--resamples N]

Each bank pass (`bank/<arm>_<fold>_s<seed>/passNN.parquet`) holds the forecasts of the touched
Primaries and their neighbours on a panel into which two eras and two switching events were
injected. `truth_share` is the injected power (what the meter reads) and `clean_share` the power
without any injection (what the normal running arrangement would have read). The quantile score
(CRPS, points of capacity) is taken on two slices:

- era events: the host's rows whose origin lies 0 to 7 or 7 to 28 days after the onset, scored
  against `truth_share`, because an era persists into the forecast;
- switching events: the host's rows whose origin lies between the onset and 7 days after the end,
  scored against `clean_share`, because the normal running arrangement forecast must exclude the
  switch.

One unit is one injected event, and its loss is the mean CRPS over its rows, averaged over the
seeds. The interval is a percentile bootstrap over events, so it rests on the number of events
printed. Every number is interim.
"""

import argparse
from pathlib import Path
from typing import Final

import e00709_features as features  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol.metrics import quantile_score

QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
BANK_DIR: Final[Path] = features.DATA_DIR / "bank"
SCORES_DIR: Final[Path] = features.DATA_DIR / "scores"


def event_losses(*, arm: str, month: str, seed: int) -> pl.DataFrame:
    """Return one row per (pass, event, slice) with the mean CRPS of the arm's rows in the slice.

    Args:
        arm: Arm name as stored under `bank/`.
        month: Fold month.
        seed: Seed.

    Returns:
        Columns `fold`, `pass`, `event`, `slice`, `seed`, `crps`, `n_rows`.
    """
    rows: list[dict] = []
    folder = BANK_DIR / f"{arm.replace('-', '_')}_{month}_s{seed}"
    for forecast_path in sorted(folder.glob("pass[0-9][0-9].parquet")):
        events = pl.read_parquet(forecast_path.with_name(forecast_path.stem + "_truth.parquet"))
        forecast = pl.read_parquet(forecast_path)
        origin_day = (
            (
                forecast["power_fcst_init_time"].dt.replace_time_zone(None)
                - pl.Series([features.GRID_START.replace(tzinfo=None)] * forecast.height)
            )
            .dt.total_days()
            .to_numpy()
        )
        site = forecast["time_series_id"].to_numpy()
        q = forecast.select(list(QCOLS)).to_numpy().astype(np.float64)
        for index, event in enumerate(events.iter_rows(named=True)):
            host_rows = site == event["host"]
            is_era = event["truth"] == "era"
            onset = event["onset_day"]
            if is_era:
                slices = {"era 0-7 d": (onset, onset + 7), "era 7-28 d": (onset + 7, onset + 28)}
                truth_column = "truth_share"
            else:
                slices = {"switch to end+7 d": (onset, event["end_day"] + 7)}
                truth_column = "clean_share"
            truth = forecast[truth_column].to_numpy().astype(np.float64)
            for name, (lo, hi) in slices.items():
                sel = host_rows & (origin_day >= lo) & (origin_day < hi) & np.isfinite(truth)
                if not sel.any():
                    continue
                score = quantile_score(actual=truth[sel] * 100.0, quantiles=q[sel] * 100.0)
                rows.append(
                    {
                        "fold": month,
                        "pass": forecast_path.stem,
                        "event": index,
                        "slice": name,
                        "seed": seed,
                        "crps": float(np.mean(score)),
                        "n_rows": int(sel.sum()),
                    }
                )
    assert rows, f"no bank rows for {folder}"
    return pl.DataFrame(rows)


def main() -> None:
    """Print the interim table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--reference", default="none")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--level", type=float, default=99.44)
    args = parser.parse_args()
    arms = list(dict.fromkeys([args.reference, *args.arms]))
    frames = []
    for arm in arms:
        for month in args.folds:
            for seed in args.seeds:
                frames.append(
                    event_losses(arm=arm, month=month, seed=seed).with_columns(pl.lit(arm).alias("arm"))
                )
    losses = pl.concat(frames).group_by("arm", "fold", "pass", "event", "slice").agg(
        pl.col("crps").mean(), pl.col("seed").n_unique().alias("n_seeds")
    )
    assert losses["n_seeds"].min() == len(args.seeds), "an event is missing a seed"
    rng = np.random.default_rng(0)
    alpha = (100.0 - args.level) / 2.0
    out = []
    print(
        f"INTERIM: folds {args.folds}, seeds {args.seeds}; CRPS in points of capacity; difference "
        f"= arm minus {args.reference}; percentile bootstrap over injected events at "
        f"{args.level}%\n"
    )
    for slice_name in sorted(losses["slice"].unique().to_list()):
        sub = losses.filter(pl.col("slice") == slice_name)
        wide = sub.pivot(on="arm", index=["fold", "pass", "event"], values="crps").drop_nulls()
        n = wide.height
        print(f"{slice_name}: {n} events")
        ref = wide[args.reference].to_numpy()
        for arm in arms:
            mean = float(wide[arm].mean())  # ty: ignore[invalid-argument-type]
            diff = wide[arm].to_numpy() - ref
            if arm == args.reference:
                print(f"  {arm:12s} crps {mean:7.3f}")
                continue
            draws = rng.integers(0, n, size=(args.resamples, n))
            means = diff[draws].mean(axis=1)
            low, high = np.percentile(means, [alpha, 100.0 - alpha])
            print(f"  {arm:12s} crps {mean:7.3f}  diff {diff.mean():+.3f} [{low:+.3f}, {high:+.3f}]")
            out.append(
                {"slice": slice_name, "arm": arm, "events": n, "crps": mean,
                 "diff": float(diff.mean()), "lower": low, "upper": high}
            )  # fmt: skip
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out).write_csv(SCORES_DIR / "interim_bank_rows.csv")


if __name__ == "__main__":
    main()

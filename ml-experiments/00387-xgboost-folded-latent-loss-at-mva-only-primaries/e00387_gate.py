"""Score the known-answer gate of experiment 387: recovered signed power against the true signed power.

For each spec of a synthetic mode (see `e00387_data`), pooled over the sites and per site, in
points of effective capacity and over leads up to 240 hours (days 1 to 10):

- signed MAE of `latent_mean` against the true signed power;
- reverse flow: true power below -1 point; balanced accuracy of "latent below 0" against it, the
  recall on reverse-flow half-hours, and the false-reverse rate on the others;
- baselines: predict the plain median read as signed, and the plain median with its sign set negative when ERA5 shortwave
  exceeds a threshold fitted on the training window's true signs (optimistic: ERA5 at the valid time).

    PYTHONPATH=... uv run --no-sync python ml-experiments/00387-*/e00387_gate.py --mode syn:none:1.0 \
        --specs plain f0 f2:k1:m1:q0 --sites D5 D9 D10
"""

import argparse
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00387_data as data
import e00387_fit as fit
import numpy as np
import polars as pl

REVERSE_POINTS: Final[float] = -1.0
MAX_LEAD_HOURS: Final[float] = 240.0


def read_preds(*, root: Path, mode: str, spec: str, site: str) -> pl.DataFrame:
    """Return all refits of one spec at one site, or an empty frame."""
    folder = root / mode.replace(":", "_") / spec.replace(":", "_")
    files = sorted(folder.glob(f"{site}_*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()


def balanced(*, truth_reverse: np.ndarray, predicted_reverse: np.ndarray) -> tuple[float, float, float]:
    """Return balanced accuracy, recall on reverse flow, and the false-reverse rate."""
    recall = float(predicted_reverse[truth_reverse].mean()) if truth_reverse.any() else float("nan")
    false = float(predicted_reverse[~truth_reverse].mean())
    return (recall + (1 - false)) / 2, recall, false


def threshold_rule(*, truth: np.ndarray, shortwave: np.ndarray, train: np.ndarray) -> float:
    """Return the shortwave threshold with the best balanced accuracy on the training window."""
    best, best_score = 0.0, -1.0
    reverse = truth[train] < REVERSE_POINTS / 100
    for t in np.arange(100.0, 1000.0, 25.0):
        score = balanced(truth_reverse=reverse, predicted_reverse=shortwave[train] > t)[0]
        if score > best_score:
            best, best_score = float(t), score
    return best


def score_site(*, root: Path, mode: str, specs: list[str], site: str) -> list[dict[str, object]]:
    """Return one result row per spec (and per baseline) at one site."""
    series = common.series_table().filter(pl.col("site") == site)
    row = next(series.iter_rows(named=True))
    arrays = common.load_series_arrays(series=series)[site]
    synthetic = fit.synthetic_for(mode=mode, arrays=arrays)
    assert synthetic is not None
    grid = pl.DataFrame(
        {
            "valid_time": arrays.valid_time,
            "truth": synthetic.truth * 100,
            "observed": synthetic.arrays.power * 100,
            "era5_shortwave": data.shortwave_of(arrays=arrays),
        }
    )
    train = (arrays.valid_time < data.TRAIN_END).to_numpy()
    threshold = threshold_rule(
        truth=synthetic.truth, shortwave=data.shortwave_of(arrays=arrays), train=train
    )
    out: list[dict[str, object]] = []
    for spec in specs:
        preds = read_preds(root=root, mode=mode, spec=spec, site=site)
        if preds.is_empty():
            continue
        joined = (
            preds.join(grid, on="valid_time", how="left")
            .with_columns(
                lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours()
            )
            .filter(pl.col("lead") <= MAX_LEAD_HOURS)
            .filter(pl.col("truth").is_finite())
        )
        truth = joined["truth"].to_numpy()
        latent = joined["latent_mean"].to_numpy() * 100
        rev = truth < REVERSE_POINTS
        plain = read_preds(root=root, mode=mode, spec="plain", site=site).select(
            "power_fcst_init_time", "valid_time", plain_median=pl.col("latent_mean")
        )
        joined = joined.join(plain, on=["power_fcst_init_time", "valid_time"], how="left")
        plain_signed = joined["plain_median"].to_numpy() * 100
        cases = {
            spec: latent,
            "baseline_plain_as_signed": plain_signed,
            f"baseline_rule_T{threshold:.0f}": np.where(
                joined["era5_shortwave"].to_numpy() > threshold, -1.0, 1.0
            )
            * plain_signed,
        }
        for name, estimate in cases.items():
            ba, recall, false = balanced(truth_reverse=rev, predicted_reverse=estimate < 0)
            out.append(
                {
                    "site": site, "spec": name, "n": len(truth), "n_reverse": int(rev.sum()),
                    "mae_points": float(np.abs(estimate - truth).mean()),
                    "mae_reverse_points": float(np.abs(estimate - truth)[rev].mean()) if rev.any() else float("nan"),
                    "balanced_accuracy": ba, "recall": recall, "false_reverse": false,
                    "negative_share_of_training_rows": float("nan"),
                }
            )
    del row
    return out


def main() -> None:
    """Print the gate table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--specs", nargs="+", required=True)
    parser.add_argument("--sites", nargs="+", default=list(data.SIGNED_SITES))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = fit.WORK / ("preds_smoke" if args.smoke else "preds")
    rows = [r for s in args.sites for r in score_site(root=root, mode=args.mode, specs=args.specs, site=s)]
    table = pl.DataFrame(rows).drop("negative_share_of_training_rows")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(40)
    print(table.with_columns(pl.col(pl.Float64).round(3)))
    table = table.filter(~pl.col("spec").str.starts_with("baseline_rule") | True)
    pooled = (
        table.group_by("spec")
        .agg(
            pl.col("mae_points").mean(),
            pl.col("mae_reverse_points").mean(),
            pl.col("balanced_accuracy").mean(),
            pl.col("recall").mean(),
            pl.col("false_reverse").mean(),
            pl.col("n_reverse").sum(),
        )
        .sort("spec")
    )
    print(pooled.with_columns(pl.col(pl.Float64).round(3)))
    table.write_parquet(fit.WORK / f"gate_{args.mode.replace(':', '_')}_{'smoke' if args.smoke else 'full'}.parquet")


if __name__ == "__main__":
    main()

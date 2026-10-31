"""Score experiment 408's known-answer test and placebos: the signed median against known truth.

Modes (see `e00408_common`): `inj:<level>` injects a metered farm's output into a signed-MW
Primary and discards the sign; `abs` reads the real signed readings as if they were apparent power.
For each signed-MW Primary, over the scored refits, leads up to 240 hours, and daylight half-hours
(CAMS unit shape above 0.01), in points of effective capacity:

- reverse flow is a true signed power below -1 point;
- the arm's estimate is the signed median of `A2_70`; the plain model (`A0`) read as signed is its
  median with a positive sign; the shortwave rule gives the plain median a negative sign when ERA5
  shortwave at the valid time exceeds a threshold fitted on the training window's true signs;
- reported: balanced accuracy, recall on reverse flow, false-reverse rate on the other half-hours,
  and signed mean absolute error against the truth. In mode `abs`, also the false-reverse rate on
  sunny summer middays (CAMS `k` at least 0.7).

Pass at level 2.0: pooled balanced accuracy at least 0.85 and every Primary at least 0.70;
false-reverse rate at most 5%; balanced accuracy at least 0.03 above the shortwave rule; signed
mean absolute error no worse than the plain model read as signed.

```bash
E408_MODE=inj:2.0 PYTHONPATH=... uv run --no-sync python ml-experiments/00408-*/e00408_gate.py \
    --mode inj:2.0 --arm A2_70
```
"""

import argparse
import os
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00408_common as c
import e00408_fit as fit
import numpy as np
import polars as pl

REVERSE_POINTS: Final[float] = -1.0
MAX_LEAD_HOURS: Final[float] = 240.0
DAYLIGHT_UNIT_SHAPE: Final[float] = 0.01
SHORTWAVE_THRESHOLDS: Final[np.ndarray] = np.arange(100.0, 1000.0, 25.0)
PASS_POOLED_BA: Final[float] = 0.85
PASS_SITE_BA: Final[float] = 0.70
PASS_FALSE_REVERSE: Final[float] = 0.05
PASS_MARGIN_OVER_RULE: Final[float] = 0.03
MEDIAN_COLUMN: Final[str] = common.LEVEL_COLUMNS[common.MEDIAN]


def read_arm(*, root: Path, folder: str, arm: str, site: str) -> pl.DataFrame:
    """Return all refits of one arm at one site, or an empty frame."""
    files = sorted((root / folder / arm).glob(f"{site}_*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()


def balanced(*, truth_reverse: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    """Return balanced accuracy, recall on reverse flow, and the false-reverse rate."""
    recall = float(predicted[truth_reverse].mean()) if truth_reverse.any() else float("nan")
    false = float(predicted[~truth_reverse].mean())
    return (recall + (1 - false)) / 2, recall, false


def shortwave_threshold(*, truth: np.ndarray, shortwave: np.ndarray, train: np.ndarray) -> float:
    """Return the shortwave threshold with the best balanced accuracy on the training window."""
    reverse = truth[train] < REVERSE_POINTS
    scores = [
        balanced(truth_reverse=reverse, predicted=shortwave[train] > t)[0]
        for t in SHORTWAVE_THRESHOLDS
    ]
    return float(SHORTWAVE_THRESHOLDS[int(np.nanargmax(scores))])


def score_site(*, root: Path, mode: str, arm: str, site: str, series: pl.DataFrame) -> list[dict]:
    """Return one result row per estimate (the arm, the plain model, the shortwave rule)."""
    _, arrays = fd._series_and_arrays()
    truth_full = fd._CACHE["truth"][site] * 100.0
    row = next(series.filter(pl.col("site") == site).iter_rows(named=True))
    cams = c.cams_arrays(row=row)
    grid = common.grid_times()
    hour, month = c.start_hour_and_month(grid=grid)
    shortwave = arrays[site].features[:, common.feature_columns().index("shortwave")].astype(float)
    train = (
        (grid < c.TRAIN_END).to_numpy()
        & np.isfinite(truth_full)
        & (np.nan_to_num(cams.u) > DAYLIGHT_UNIT_SHAPE)
    )
    threshold = shortwave_threshold(truth=truth_full, shortwave=shortwave, train=train)
    signed = read_arm(root=root, folder="signed", arm=arm, site=site)
    plain = read_arm(root=root, folder="preds", arm="A0", site=site)
    if signed.is_empty() or plain.is_empty():
        return []
    joined = (
        signed.select("power_fcst_init_time", "valid_time", signed_median=pl.col(MEDIAN_COLUMN))
        .join(
            plain.select("power_fcst_init_time", "valid_time", plain_median=pl.col(MEDIAN_COLUMN)),
            on=["power_fcst_init_time", "valid_time"],
            how="inner",
        )
        .with_columns(lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours())
    )
    index = np.asarray(common.grid_index(times=joined["valid_time"]))
    truth = truth_full[index]
    keep = (
        (joined["lead"].to_numpy() <= MAX_LEAD_HOURS)
        & np.isfinite(truth)
        & (np.nan_to_num(cams.u[index]) > DAYLIGHT_UNIT_SHAPE)
    )
    index, truth = index[keep], truth[keep]
    estimates = {
        arm: joined["signed_median"].to_numpy()[keep] * 100.0,
        "plain_as_signed": joined["plain_median"].to_numpy()[keep] * 100.0,
        f"shortwave_rule_T{threshold:.0f}": np.where(shortwave[index] > threshold, -1.0, 1.0)
        * joined["plain_median"].to_numpy()[keep]
        * 100.0,
    }
    reverse = truth < REVERSE_POINTS
    sunny_midday = (
        (np.nan_to_num(cams.k[index]) >= c.GATE_K_MIN)
        & (hour[index] >= c.MIDDAY_START_HOURS[0])
        & (hour[index] < c.MIDDAY_START_HOURS[1])
        & (month[index] >= c.SUMMER_MONTHS[0])
        & (month[index] <= c.SUMMER_MONTHS[1])
    )
    out = []
    for name, estimate in estimates.items():
        ba, recall, false = balanced(truth_reverse=reverse, predicted=estimate < 0)
        out.append(
            {
                "site": site,
                "estimate": name,
                "n": len(truth),
                "n_reverse": int(reverse.sum()),
                "balanced_accuracy": ba,
                "recall": recall,
                "false_reverse": false,
                "false_reverse_sunny_midday": float((estimate[sunny_midday & ~reverse] < 0).mean())
                if (sunny_midday & ~reverse).any()
                else float("nan"),
                "signed_mae_points": float(np.abs(estimate - truth).mean()),
            }
        )
    return out


def pooled(*, frame: pl.DataFrame, column: str) -> float:
    """Return the mean of `column` over all half-hours of the sites.

    Each site is weighted by the half-hours it contributes: reverse-flow half-hours for recall,
    the other half-hours for the false-reverse rate.
    """
    weight = pl.col("n_reverse") if column == "recall" else pl.col("n") - pl.col("n_reverse")
    return float(frame.select((pl.col(column) * weight).sum() / weight.sum()).item())


def pooled_ba(*, frame: pl.DataFrame) -> float:
    """Return the pooled balanced accuracy: pooled recall and pooled false-reverse rate."""
    return (
        pooled(frame=frame, column="recall") + 1 - pooled(frame=frame, column="false_reverse")
    ) / 2


def verdict(*, table: pl.DataFrame, arm: str) -> dict[str, object]:
    """Return the pass criteria of the known-answer test, evaluated on the pooled table."""
    mine = table.filter(pl.col("estimate") == arm)
    plain = table.filter(pl.col("estimate") == "plain_as_signed")
    rule = table.filter(pl.col("estimate").str.starts_with("shortwave_rule"))
    checks = {
        "pooled_ba": pooled_ba(frame=mine),
        "min_site_ba": float(mine["balanced_accuracy"].min()),  # ty: ignore[invalid-argument-type]
        "false_reverse": pooled(frame=mine, column="false_reverse"),
        "margin_over_rule": pooled_ba(frame=mine) - pooled_ba(frame=rule),
        "signed_mae": float(mine["signed_mae_points"].mean()),  # ty: ignore[invalid-argument-type]
        "plain_signed_mae": float(plain["signed_mae_points"].mean()),  # ty: ignore[invalid-argument-type]
    }
    checks["pass"] = bool(
        checks["pooled_ba"] >= PASS_POOLED_BA
        and checks["min_site_ba"] >= PASS_SITE_BA
        and checks["false_reverse"] <= PASS_FALSE_REVERSE
        and checks["margin_over_rule"] >= PASS_MARGIN_OVER_RULE
        and checks["signed_mae"] <= checks["plain_signed_mae"]
    )
    return checks


def main() -> None:
    """Print the known-answer table and the verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--arm", default="A2_70")
    parser.add_argument("--sites", nargs="+", default=list(c.SIGNED_SITES))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get(c.MODE_ENV) != args.mode:
        msg = f"set {c.MODE_ENV}={args.mode} before launching"
        raise SystemExit(msg)
    root = fit.mode_dir(mode=args.mode, smoke=args.smoke)
    series, _ = fd._series_and_arrays()
    rows = [
        r
        for s in args.sites
        for r in score_site(root=root, mode=args.mode, arm=args.arm, site=s, series=series)
    ]
    table = pl.DataFrame(rows)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(40)
    print(table.with_columns(pl.col(pl.Float64).round(3)))
    result = verdict(table=table, arm=args.arm)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in result.items()})
    table.write_parquet(root / f"gate_{args.arm}.parquet")


if __name__ == "__main__":
    main()

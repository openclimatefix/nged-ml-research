# ruff: noqa: E501
"""Score experiment 418's arms against experiment 397's pooled plain network on the screen folds.

`folds` prints the fold-by-fold CRPS and tails, the gain of each arm over the reference, and
whether each arm passes the pre-registered advance rule. The reference is experiment 397's stored
`plain_pooled` forecasts (summary scheme, three seeds); the arms average their own seeds in
quantile space, as experiment 397 does.

```bash
source /tmp/e418_env.sh  # PYTHONPATH as in e00418_fit.py, plus experiments 377 and 381
uv run --no-sync python ml-experiments/00418-*/e00418_score.py folds
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_score as s397  # ty: ignore[unresolved-import]
import e00418_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

REFERENCE: Final[str] = "ref_e397_pooled"
MIN_GAIN: Final[float] = 0.03
EXPECTED_SEEDS: Final[int] = 3
SIX: Final[str] = "signed-MW six"
EIGHT: Final[str] = "MVA-only eight"
NOMINAL: Final[dict[str, float]] = {"coverage_q95": 0.95, "coverage_q99": 0.99}


def average_dir(*, folder: Path, months: list[str], seeds: list[int] | None = None) -> pl.DataFrame:
    """Return the seed-averaged forecast (fraction of capacity) of a folder, months stacked."""
    keys = list(common.KEYS)
    pieces = []
    for month in months:
        files = sorted(
            p
            for p in folder.glob(f"{month}_s*.parquet")
            if not p.stem.endswith("_gate")
            and (seeds is None or int(p.stem.split("_s")[-1]) in seeds)
        )
        if not files:
            continue
        if seeds is None and len(files) != EXPECTED_SEEDS:
            msg = f"{folder.name} {month}: {len(files)} seed files, expected {EXPECTED_SEEDS}"
            raise ValueError(msg)
        stack = pl.concat([pl.read_parquet(f) for f in files])
        pieces.append(
            stack.group_by(keys)
            .agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS])
            .sort(keys)
            .with_columns(pl.col(*common.LEVEL_COLUMNS).cast(pl.Float32))
        )
    out = pl.concat(pieces)
    values = np.sort(out.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return out.with_columns(
        *[pl.Series(c, values[:, i]) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def arm_frames(
    *, folder: str, months: list[str], arms: list[str], series: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    """Return the seed-averaged forecasts of named arms in the shape the scorer expects."""
    out = {}
    for arm in arms:
        name = fit.arm_name(arm=arm)
        out[name] = adapter.forecast_frame(
            stored=average_dir(folder=fit.OUT_DIR / folder / name, months=months),
            series=series,
            unit="fraction",
        )
    return out


def score_table(
    *, frames: dict[str, pl.DataFrame], months: list[str], series: pl.DataFrame, ctx: object
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return per-fold CRPS and tail metrics of every frame, as (crps table, tails table)."""
    crps, tails = [], []
    for month in months:
        in_month = {k: s397.only_month(frame=v, month=month) for k, v in frames.items()}
        _, scored, _ = s381.score_arms(arms=in_month, series=series, ctx=ctx)
        for name, frame in scored.items():
            for group, value in s397.group_means(
                scored=frame, metric="crps", slice_name=s397.HEADLINE
            ).items():
                crps.append({"fold": month, "arm": name, "group": group, "crps": value})
            for metric in NOMINAL:
                for group, value in s397.group_means(
                    scored=frame, metric=metric, slice_name=s397.HEADLINE
                ).items():
                    tails.append(
                        {
                            "fold": month,
                            "arm": name,
                            "group": group,
                            "metric": metric,
                            "value": value,
                        }
                    )
    return pl.DataFrame(crps), pl.DataFrame(tails)


def advance_rule(
    *, crps: pl.DataFrame, tails: pl.DataFrame, arm: str, se: float
) -> dict[str, object]:
    """Evaluate the pre-registered advance rule of an arm against the reference."""

    def by_fold(group: str, name: str) -> pl.DataFrame:
        return crps.filter((pl.col("group") == group) & (pl.col("arm") == name)).sort("fold")

    six_gain = by_fold(SIX, arm)["crps"].to_numpy() - by_fold(SIX, REFERENCE)["crps"].to_numpy()
    eight_gain = (
        by_fold(EIGHT, arm)["crps"].to_numpy() - by_fold(EIGHT, REFERENCE)["crps"].to_numpy()
    )

    def distance(name: str) -> float:
        rows = (
            tails.filter((pl.col("group") == SIX) & (pl.col("arm") == name))
            .group_by("metric")
            .agg(pl.col("value").mean())
        )
        return float(
            sum(abs(r["value"] - NOMINAL[r["metric"]]) for r in rows.iter_rows(named=True))
        )

    return {
        "folds_beating_reference": int((six_gain < 0).sum()),
        "mean_gain_six": float(six_gain.mean()),
        "mean_gain_eight": float(eight_gain.mean()),
        "tail_distance_arm": distance(arm),
        "tail_distance_reference": distance(REFERENCE),
        "passes": bool(
            (six_gain < 0).sum() >= 3
            and six_gain.mean() <= -max(MIN_GAIN, 2 * se)
            and eight_gain.mean() <= 0.02
            and distance(arm) <= distance(REFERENCE)
        ),
    }


def reference_se(*, months: list[str], series: pl.DataFrame, ctx: object) -> float:
    """Return the standard error of the reference's four-fold mean (signed-MW six) over its seeds."""
    means = []
    for seed in (0, 1, 2):
        frame = adapter.forecast_frame(
            stored=average_dir(
                folder=s397.SEED_DIR / "summary" / "plain_pooled", months=months, seeds=[seed]
            ),
            series=series,
            unit="fraction",
        )
        table, _ = score_table(frames={"r": frame}, months=months, series=series, ctx=ctx)
        means.append(table.filter(pl.col("group") == SIX)["crps"].mean())
    return float(np.std(np.asarray(means), ddof=1) / np.sqrt(len(means)))


def folds(*, months: list[str], arms: list[str]) -> None:
    """Print the fold-by-fold tables and the advance rule."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    refs = s397.references(series=series)
    reference = adapter.forecast_frame(
        stored=average_dir(folder=s397.SEED_DIR / "summary" / "plain_pooled", months=months),
        series=series,
        unit="fraction",
    )
    mine = arm_frames(folder="seed_preds", months=months, arms=arms, series=series)
    frames = {REFERENCE: reference, **mine, **refs}
    crps, tails = score_table(frames=frames, months=months, series=series, ctx=ctx)
    se = reference_se(months=months, series=series, ctx=ctx)
    with pl.Config(tbl_rows=200, tbl_cols=20):
        for group in (SIX, EIGHT):
            print(f"\n== CRPS, days 1 to 10, {group}, points ==")
            print(
                crps.filter(pl.col("group") == group)
                .pivot(on="fold", index="arm", values="crps")
                .with_columns(mean=pl.mean_horizontal(pl.exclude("arm")))
                .sort("mean")
            )
        print("\n== tails, days 1 to 10 (nominal 0.95, 0.99) ==")
        print(
            tails.group_by("arm", "group", "metric")
            .agg(pl.col("value").mean())
            .pivot(on="metric", index=["arm", "group"], values="value")
            .sort("group", "arm")
        )
    print(f"\nreference standard error of the four-fold mean over seeds (signed-MW six): {se:.4f}")
    for name in mine:
        print(name, advance_rule(crps=crps, tails=tails, arm=name, se=se))


def main() -> None:
    """Run a mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("folds",))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--arms", nargs="+", default=["P1", "P2", "P3"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        fit.OUT_DIR = fit.prep.WORK_DIR / "smoke"
    folds(months=args.months, arms=args.arms)


if __name__ == "__main__":
    main()

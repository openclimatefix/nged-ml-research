"""Score the forecasts of experiment 404 by group and row class, with event-cluster intervals.

Row classes. On the injected scoring worlds: `harm_inj` (target outside every injected event, a day
window inside one), `tgt_inj` (target inside an injected event), and `clean_inj` (neither). On the
clean world (world -1, real data): `harm_real` (target outside every padded real event and the 96
half-hours after it, a day window inside one), `tgt_real` (target inside a padded real event), and
`clean_real` (neither). Groups are the signed-MW six, the MVA-only eight, and the report-only
Primaries D15 and D18. The score is the CRPS-like mass-weighted pinball loss in points of capacity.
A contrast is the difference of two arms' mean score on the same rows, with a 95% interval from
resampling whole events (real events for `harm_real`; a world, Primary, and origin day for the
injected classes).

```bash
PYTHONPATH=<as e00404_fit.py shows> \
    uv run --no-sync python ml-experiments/00404-*/e00404_report.py table --month 2026-06 --seed 0 \
    --arms A0 A0-inj B0 B1 B2 B5 B-mix
```
"""

import argparse
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_rows as rows_mod  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SIGNED_SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
REPORT_ONLY: Final[tuple[str, ...]] = ("D15", "D18")
BOOTSTRAP: Final[int] = 1000


def group_of(*, sites: list[str]) -> np.ndarray:
    """Return the group name of each Primary index."""
    return np.array(
        [
            "report_only"
            if s in REPORT_ONLY
            else "mva_eight"
            if s in fit.MVA_ONLY
            else "signed_six"
            for s in sites
        ]
    )


def load(*, arm: str, month: str, seed: int, worlds: list[int]) -> pl.DataFrame:
    """Return an arm's forecasts over the given worlds, scored and classified."""
    sites = fit.prep.primaries()["site"].to_list()
    groups = group_of(sites=sites)
    parts = []
    for world in worlds:
        frame = pl.read_parquet(score.prediction_path(arm=arm, month=month, seed=seed, world=world))
        parts.append(frame.with_columns(world=pl.lit(world)))
    frame = pl.concat(parts)
    frame = frame.with_columns(
        crps=score.pinball_points(frame=frame),
        inside90=((pl.col("actual") >= pl.col("q05")) & (pl.col("actual") <= pl.col("q95"))).cast(
            pl.Float64
        ),
        above95=(pl.col("actual") <= pl.col("q95")).cast(pl.Float64),
        group=pl.Series(groups[frame["primary"].to_numpy()]),
    )
    injected = pl.col("world") >= 0
    return frame.with_columns(
        cls=pl.when(injected & pl.col("tgt_inj"))
        .then(pl.lit("tgt_inj"))
        .when(injected & pl.col("win_inj"))
        .then(pl.lit("harm_inj"))
        .when(injected)
        .then(pl.lit("clean_inj"))
        .when(pl.col("tgt_real") | pl.col("tgt_post"))
        .then(pl.lit("tgt_real"))
        .when(pl.col("win_real"))
        .then(pl.lit("harm_real"))
        .otherwise(pl.lit("clean_real"))
    )


def cluster_ids(*, frame: pl.DataFrame, real: np.ndarray) -> np.ndarray:
    """Return an event cluster id per row: a real event for real classes, else world/Primary/day."""
    day = (frame["origin"].to_numpy().astype(np.int64)) // 48
    primary = frame["primary"].to_numpy().astype(np.int64)
    world = frame["world"].to_numpy().astype(np.int64)
    ids = (world + 1) * 10_000_000 + primary * 100_000 + (day - day.min())
    lag = rows_mod.lag_grids(
        target=frame["target"].to_numpy().astype(np.int64),
        origin=frame["origin"].to_numpy().astype(np.int64),
    )
    is_real = world < 0
    for p in np.unique(primary[is_real]):
        run = np.cumsum(np.diff(np.concatenate([[0], real[p].astype(np.int8)])) == 1)
        sel = np.flatnonzero(is_real & (primary == p))
        inside = real[p][np.clip(lag[sel], 0, real.shape[1] - 1)]
        first = np.where(inside.any(1), inside.argmax(1), 0)
        grid = np.clip(lag[sel, first], 0, real.shape[1] - 1)
        ids[sel] = -1 - (p * 10_000 + run[grid])
    return ids


def contrast(
    *, a: pl.DataFrame, b: pl.DataFrame, cluster: np.ndarray, seed: int = 0
) -> tuple[float, float, float]:
    """Return mean(a.crps - b.crps) and its 95% event-cluster bootstrap interval (same rows)."""
    diff = a["crps"].to_numpy() - b["crps"].to_numpy()
    codes, inverse = np.unique(cluster, return_inverse=True)
    sums = np.bincount(inverse, weights=diff, minlength=len(codes))
    counts = np.bincount(inverse, minlength=len(codes)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(codes), size=(BOOTSTRAP, len(codes)))
    estimates = sums[draws].sum(1) / counts[draws].sum(1)
    return (
        float(diff.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def table(*, month: str, seed: int, arms: list[str], worlds: list[int]) -> None:
    """Print mean score and 90% band coverage by arm, group, and row class, then contrasts."""
    real = score.real_event_mask()
    frames = {arm: load(arm=arm, month=month, seed=seed, worlds=worlds) for arm in arms}
    pl.Config.set_tbl_rows(100)
    summary = pl.concat(
        [
            f.group_by("group", "cls")
            .agg(
                pl.len().alias("rows"),
                pl.col("crps").mean().alias("crps"),
                pl.col("inside90").mean().alias("inside90"),
                pl.col("above95").mean().alias("below_q95"),
            )
            .with_columns(arm=pl.lit(arm))
            for arm, f in frames.items()
        ]
    )
    print(summary.sort("group", "cls", "arm"))
    reference = "A0-inj" if "A0-inj" in frames else arms[0]
    clusters = cluster_ids(frame=frames[reference], real=real)
    for arm in arms:
        if arm == reference:
            continue
        for group in ("signed_six", "mva_eight"):
            for cls in ("harm_inj", "tgt_inj", "clean_inj", "harm_real", "tgt_real", "clean_real"):
                mask = (
                    (frames[reference]["group"] == group) & (frames[reference]["cls"] == cls)
                ).to_numpy()
                if mask.sum() == 0:
                    continue
                point, low, high = contrast(
                    a=frames[arm].filter(pl.Series(mask)),
                    b=frames[reference].filter(pl.Series(mask)),
                    cluster=clusters[mask],
                )
                print(
                    f"{arm} minus {reference} | {group} | {cls}: "
                    f"{point:+.3f} [{low:+.3f}, {high:+.3f}]"
                )


def main() -> None:
    """Print the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("table",))
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--worlds", type=int, nargs="*", default=[*score.SCORING_SEEDS, -1])
    args = parser.parse_args()
    table(month=args.month, seed=args.seed, arms=args.arms, worlds=args.worlds)


if __name__ == "__main__":
    main()

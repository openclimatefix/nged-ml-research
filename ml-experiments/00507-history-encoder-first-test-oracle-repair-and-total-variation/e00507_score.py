"""Experiment 507: contrasts, minimum detectable effects, and the rule (a) reading.

Reads the per-month score tables written by `e00507_arms.py` (mean CRPS and tail scores per arm,
month, seed, scoring world, group, and row class). The injected "step rows" are the union of
`harm_inj` and `tgt_inj` on the old worlds, pooled with the rows' counts as weights and averaged
over scoring worlds 200 to 202. Each arm's score in a month is the mean over its 3 seeds. Intervals
are month-block bootstrap percentiles enumerated exactly: with 4 scored months there are 256
ordered resamples of the 4 monthly contrasts.

```bash
ml-experiments/00507-*/run.sh ml-experiments/00507-*/e00507_score.py old
```
"""

import argparse
import itertools
from typing import Final

import e00507_arms as arms  # ty: ignore[unresolved-import]
import e00507_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

BONFERRONI: Final[int] = 8
LEVELS: Final[tuple[float, float]] = (0.95, 1 - 0.05 / BONFERRONI)
STEP_CLASSES: Final[tuple[str, str]] = ("harm_inj", "tgt_inj")
METRICS: Final[tuple[str, ...]] = ("crps", "pin_q01", "pin_q05", "pin_q95", "pin_q99", "cov_q99")


def load(*, kind: str) -> pl.DataFrame:
    """Return all months' score tables for `kind` (`old` or `new`)."""
    frames = [pl.read_parquet(arms.SCORE_DIR / f"{kind}_{m}.parquet") for m in c.SCREEN_MONTHS]
    return pl.concat(frames)


def pooled(*, scores: pl.DataFrame, classes: tuple[str, ...], band: str = "all") -> pl.DataFrame:
    """Return the n-weighted mean of each metric over `classes`, per arm label, month, seed, group.

    Averages over the scoring worlds that hold the classes (worlds with a positive id).
    """
    part = scores.filter(pl.col("cls").is_in(classes) & (pl.col("band") == band))
    weighted = part.with_columns([(pl.col(m) * pl.col("n")).alias(m) for m in METRICS])
    keys = ["label", "month", "seed", "group", "world"]
    one = weighted.group_by(keys).agg(
        pl.col("n").sum(), *[pl.col(m).sum() / pl.col("n").sum() for m in METRICS]
    )
    return one.group_by("label", "month", "seed", "group").agg(
        pl.col("n").mean(), *[pl.col(m).mean() for m in METRICS]
    )


def month_means(*, table: pl.DataFrame, label: str, group: str, metric: str) -> np.ndarray:
    """Return the label's per-month mean over seeds (months in screen order)."""
    part = table.filter((pl.col("label") == label) & (pl.col("group") == group))
    by_month = part.group_by("month").agg(pl.col(metric).mean())
    lookup = dict(zip(by_month["month"], by_month[metric], strict=True))
    return np.array([lookup[m] for m in c.SCREEN_MONTHS], dtype=np.float64)


def seed_series(*, table: pl.DataFrame, label: str, group: str, metric: str) -> np.ndarray:
    """Return a (seeds, months) array of the label's metric."""
    part = table.filter((pl.col("label") == label) & (pl.col("group") == group))
    seeds = sorted(part["seed"].unique().to_list())
    out = []
    for s in seeds:
        rows = part.filter(pl.col("seed") == s)
        lookup = dict(zip(rows["month"], rows[metric], strict=True))
        out.append([lookup[m] for m in c.SCREEN_MONTHS])
    return np.array(out, dtype=np.float64)


def minimum_detectable_effect(*, series: np.ndarray) -> float:
    """Return experiment 454's formula: 2.8 times the mean paired seed sd, over sqrt(months)."""
    sds = [
        float(np.std(series[i] - series[j], ddof=1))
        for i, j in itertools.combinations(range(len(series)), 2)
    ]
    return 2.8 * float(np.mean(sds)) / np.sqrt(series.shape[1])


def block_interval(*, diffs: np.ndarray, level: float) -> tuple[float, float]:
    """Return the percentile interval of the mean over all ordered resamples of the months."""
    n = len(diffs)
    means = np.array([diffs[list(ix)].mean() for ix in itertools.product(range(n), repeat=n)])
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def contrast(
    *, table: pl.DataFrame, arm: str, ref: str, group: str, metric: str = "crps"
) -> dict[str, float | str]:
    """Return arm minus ref: the mean, both intervals, monthly values, and absolute scores."""
    a = month_means(table=table, label=arm, group=group, metric=metric)
    r = month_means(table=table, label=ref, group=group, metric=metric)
    d = a - r
    lo95, hi95 = block_interval(diffs=d, level=LEVELS[0])
    loB, hiB = block_interval(diffs=d, level=LEVELS[1])
    return {"arm": arm, "ref": ref, "group": group, "metric": metric, "abs_arm": a.mean(),
            "abs_ref": r.mean(), "diff": d.mean(), "lo95": lo95, "hi95": hi95, "loB": loB,
            "hiB": hiB, "months_arm_better": int((d < 0).sum()),
            "monthly": ", ".join(f"{x:+.3f}" for x in d)}  # fmt: skip


def old_report() -> None:
    """Print the old-world tables and the rule (a) reading."""
    scores = load(kind="old")
    out_rows = []
    for group in ("six", "eight"):
        step = pooled(scores=scores, classes=STEP_CLASSES)
        tgt = pooled(scores=scores, classes=("tgt_inj",))
        harm = pooled(scores=scores, classes=("harm_inj",))
        clean_inj = pooled(scores=scores, classes=("clean_inj",))
        series = seed_series(table=step, label="plain", group=group, metric="crps")
        mde = minimum_detectable_effect(series=series)
        print(f"\n== {group}: injected-row minimum detectable effect (3-seed spread of plain): "
              f"{mde:.3f} points\n   plain per seed and month:\n{np.round(series, 3)}")  # fmt: skip
        for name, table in (("step rows", step), ("harm_inj", harm), ("tgt_inj", tgt),
                            ("clean_inj", clean_inj)):  # fmt: skip
            for arm in ("oracle", "oracle_reprediction", "zero"):
                row = contrast(table=table, arm=arm, ref="plain", group=group)
                out_rows.append({"rows": name, "mde": mde, **row})
                print(f"{name:10s} {arm:20s} minus plain: {row['diff']:+.3f} "
                      f"95% [{row['lo95']:+.3f}, {row['hi95']:+.3f}] "
                      f"Bonf [{row['loB']:+.3f}, {row['hiB']:+.3f}] "
                      f"abs {row['abs_arm']:.3f} vs {row['abs_ref']:.3f}; "
                      f"months {row['monthly']}")  # fmt: skip
        real = pooled(scores=scores.filter(pl.col("world") == -1), classes=("clean",))
        real_ev = pooled(scores=scores.filter(pl.col("world") == -1), classes=("event",))
        for name, table in (
            ("no labelled event (world -1)", real),
            ("near a labelled event", real_ev),
        ):
            for arm in ("oracle", "zero"):
                for metric in ("crps", "pin_q99", "cov_q99"):
                    row = contrast(table=table, arm=arm, ref="plain", group=group, metric=metric)
                    out_rows.append({"rows": name, "mde": mde, **row})
                    print(f"{name:30s} {arm:8s} {metric:8s} minus plain: {row['diff']:+.4f} "
                          f"95% [{row['lo95']:+.4f}, {row['hi95']:+.4f}] abs {row['abs_arm']:.4f} "
                          f"vs {row['abs_ref']:.4f}")  # fmt: skip
    arms.SCORE_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out_rows).write_csv(arms.SCORE_DIR / "old_contrasts.csv", float_precision=4)


def mixed(*, temp: pl.DataFrame, gen: pl.DataFrame, genuine: float | None) -> pl.DataFrame:
    """Return step rows as a mix of `temp` and `gen` rows.

    With `genuine` None the mix is the rows' own counts. Otherwise the genuine (`gen`) rows get
    the weight `genuine` and the temporary (`temp`) rows `1 - genuine`.
    """
    keys = ["label", "month", "seed", "group"]
    joined = temp.join(gen, on=keys, suffix="_g")
    own = pl.col("n_g") / (pl.col("n") + pl.col("n_g"))
    w = own if genuine is None else pl.lit(genuine)
    return joined.select(
        *keys,
        pl.col("n") + pl.col("n_g"),
        *[(pl.col(m) * (1 - w) + pl.col(f"{m}_g") * w).alias(m) for m in METRICS],
    )


def new_report() -> None:
    """Print the new-generator tables and the rule (b) reading."""
    scores = load(kind="new")
    scores = scores.filter(pl.col("world") > 0)
    temp = pooled(scores=scores, classes=("temp",))
    gen = pooled(scores=scores, classes=("gen",))
    none = pooled(scores=scores, classes=("none",))
    allrows = pooled(scores=scores, classes=("temp", "gen", "none"))
    out_rows = []
    for group in ("six", "eight"):
        for genuine, tag in ((None, "generator mix"), (0.6, "reweighted to 60% genuine")):
            step = mixed(temp=temp, gen=gen, genuine=genuine)
            series = seed_series(table=step, label="plain", group=group, metric="crps")
            mde = minimum_detectable_effect(series=series)
            print(f"\n== {group}, step rows at the {tag}: MDE {mde:.3f} points")
            res = {}
            for arm in ("oracle", "oracle_const", "tv", "zero"):
                row = contrast(table=step, arm=arm, ref="plain", group=group)
                res[arm] = row
                out_rows.append({"rows": f"step ({tag})", "mde": mde, **row})
                print(f"   {arm:13s} minus plain: {row['diff']:+.3f} 95% [{row['lo95']:+.3f}, "
                      f"{row['hi95']:+.3f}] Bonf [{row['loB']:+.3f}, {row['hiB']:+.3f}] "
                      f"months {row['monthly']}")  # fmt: skip
            # Rule (b): TV keeps half the oracle gain when (tv - plain) - 0.5 (oracle - plain) <= 0.
            a = {n: month_means(table=step, label=n, group=group, metric="crps") for n in
                 ("plain", "oracle", "tv")}  # fmt: skip
            d = (a["tv"] - a["plain"]) - 0.5 * (a["oracle"] - a["plain"])
            lo, hi = block_interval(diffs=d, level=LEVELS[0])
            print(f"   rule (b): TV loss minus the halfway point to the oracle: {d.mean():+.3f} "
                  f"95% [{lo:+.3f}, {hi:+.3f}] (negative: TV keeps more than half); months "
                  f"{', '.join(f'{x:+.3f}' for x in d)}")  # fmt: skip
            out_rows.append({"rows": f"rule b ({tag})", "mde": mde, "arm": "tv_vs_half_oracle",
                             "group": group, "diff": d.mean(), "lo95": lo, "hi95": hi})  # fmt: skip
    print()
    for group in ("six", "eight"):
        for name, table in (("no shift touching a lag", none), ("all windows", allrows),
                            ("temporary-touched", temp), ("genuine-touched", gen)):  # fmt: skip
            for arm in ("oracle", "tv", "zero"):
                for metric in ("crps", "pin_q99", "cov_q99"):
                    row = contrast(table=table, arm=arm, ref="plain", group=group, metric=metric)
                    out_rows.append({"rows": name, "mde": float("nan"), **row})
                    print(f"{group:5s} {name:24s} {arm:7s} {metric:8s} minus plain: "
                          f"{row['diff']:+.4f} 95% [{row['lo95']:+.4f}, {row['hi95']:+.4f}] "
                          f"abs {row['abs_arm']:.4f} vs {row['abs_ref']:.4f}")  # fmt: skip
    arms.SCORE_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out_rows, infer_schema_length=None).write_csv(
        arms.SCORE_DIR / "new_contrasts.csv", float_precision=4
    )


def main() -> None:
    """Run one reading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("old", "new"))
    args = parser.parse_args()
    if args.step == "old":
        old_report()
    else:
        new_report()


if __name__ == "__main__":
    main()

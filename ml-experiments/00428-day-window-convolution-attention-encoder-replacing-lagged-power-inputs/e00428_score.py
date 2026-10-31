# ruff: noqa: E501
"""Score the stored forecasts of experiment 428 and read the sweep (no fit).

**Scores.** For each arm the three seeds' per-row quantile scores (`studies.eval_protocol.metrics.
quantile_score`, in points of effective capacity) are averaged over seeds, row by row, and then
over rows. Row sets (leads up to 240 hours, finite actual):

- `clean`: real series, no observed half-hour of the three windows inside the padded real-event
  mask, target outside the padded mask and the 96 half-hours after an event.
- `real_event`: real series, target inside the padded real-event mask.
- `real_harm`: real series, target outside the mask and its post-event period, a window inside.
- `inj_clean` and `inj_harm`: the three held-out injected worlds; harm rows have the target outside
  every real event and an injected event inside the windows.

Groups: signed-MW six, MVA-only eight, D15 and D18 (report-only).

**Contrasts** C1 to C6 of the plan are paired row by row. The interval is a bootstrap over blocks of
one fold and one origin day (2,000 resamples, 95% level), labelled interim: four folds give wide
intervals. A fold counts as won when the contrast's fold mean is below zero by more than the minimum
detectable effect, which is computed from R-inj's seed spread as the plan states.

```bash
source <paths>; PYTHONPATH=$PP:ml-experiments/00428-* DATA_PATH_INTERNAL=... \
  uv run --no-sync python ml-experiments/00428-*/e00428_score.py --months 2026-06 2026-01 2025-10 2026-03
```
"""

import argparse
import math
from collections.abc import Callable
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.eval_protocol.slices import MVA_ONLY_LABELS, SIGNED_MW_LABELS

ARMS: Final[tuple[str, ...]] = ("R", "R-inj", "L0", "Bmix", "E1", "E2", "E3")
WORLD_SETS: Final[dict[str, tuple[int, ...]]] = {
    "real": (0,),
    "inj": (4, 5, 6),
}
ROW_SETS: Final[dict[str, tuple[str, Callable[[dict[str, np.ndarray]], np.ndarray]]]] = {
    "clean": ("real", lambda f: ~f["win_real"] & ~f["tgt_post"]),
    "real_event": ("real", lambda f: f["tgt_real"]),
    "real_harm": ("real", lambda f: ~f["tgt_post"] & f["win_real"]),
    "inj_clean": ("inj", lambda f: ~f["win_inj"] & ~f["win_real"] & ~f["tgt_post"]),
    "inj_harm": ("inj", lambda f: f["win_inj"] & ~f["tgt_post"]),
}
"""Row sets: the world set they are read from, and the flag rule."""
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "signed-MW six": SIGNED_MW_LABELS,
    "MVA-only eight": MVA_ONLY_LABELS,
    "D15 and D18": ("D15", "D18"),
}
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 E1 - R-inj": ("E1", "R-inj"),
    "C2 E2 - E1": ("E2", "E1"),
    "C3 E3 - E2": ("E3", "E2"),
    "C4 E3 - R-inj": ("E3", "R-inj"),
    "C5 E3 - Bmix": ("E3", "Bmix"),
    "C6 R-inj - R": ("R-inj", "R"),
}
TAILS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
LEVEL_OF: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
RESAMPLES: Final[int] = 2_000
POINTS: Final[float] = 100.0
"""Quantile scores are fractions of capacity; one point is 1%."""


def read_arm(*, arm: str, month: str, world: int, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return the seed-averaged quantiles, row flags, and keys of an arm, month, and world."""
    frames = [
        pl.read_parquet(pr.PRED_DIR / arm / f"{month}_s{seed}_w{world}.parquet") for seed in seeds
    ]
    keys = frames[0].select("primary", "origin_us", "valid_us", "lead", "actual", *_flags())
    for f in frames[1:]:
        assert f["valid_us"].equals(frames[0]["valid_us"]), "seed files differ in row order"
    quantiles = {
        name: np.mean([f[name].to_numpy() for f in frames], axis=0) for name in pr.LEVEL_NAMES
    }
    per_seed = [
        metrics.quantile_score(
            actual=f["actual"].to_numpy(),
            quantiles=np.sort(f.select(list(pr.LEVEL_NAMES)).to_numpy(), axis=1),
        )
        for f in frames
    ]
    return keys.with_columns(
        [pl.Series(k, v) for k, v in quantiles.items()],
        score=pl.Series(np.mean(per_seed, axis=0)),
        **{f"score_s{s}": pl.Series(v) for s, v in zip(seeds, per_seed, strict=True)},
    )


def _flags() -> list[str]:
    return ["win_inj", "win_real", "tgt_real", "tgt_post"]


def row_mask(
    *, frame: pl.DataFrame, rule: Callable[[dict[str, np.ndarray]], np.ndarray]
) -> np.ndarray:
    """Return the rows a row-set rule selects that also have a finite actual."""
    flags = {n: frame[n].to_numpy().astype(bool) for n in _flags()}
    return rule(flags) & np.isfinite(frame["actual"].to_numpy())


def group_of(*, primary: np.ndarray) -> dict[str, np.ndarray]:
    """Return a boolean mask per group over rows."""
    sites = np.array(data.sites())[primary]
    return {g: np.isin(sites, labels) for g, labels in GROUPS.items()}


def collect(*, months: list[str], seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return one long table: per arm, month, world set, row set, group, origin day, block sums."""
    records = []
    for month in months:
        for set_name, worlds in WORLD_SETS.items():
            for arm in ARMS:
                for world in worlds:
                    frame = read_arm(arm=arm, month=month, world=world, seeds=seeds)
                    day = (frame["origin_us"].to_numpy() // (86_400 * 1_000_000)).astype(int)
                    groups = group_of(primary=frame["primary"].to_numpy())
                    actual = frame["actual"].to_numpy()
                    for row_set, (which, rule) in ROW_SETS.items():
                        if which != set_name:
                            continue
                        keep = row_mask(frame=frame, rule=rule)
                        for group, in_group in groups.items():
                            sel = keep & in_group
                            for d in np.unique(day[sel]):
                                m = sel & (day == d)
                                record = {
                                    "arm": arm, "month": month, "world": world, "row_set": row_set,
                                    "group": group, "day": int(d), "n": int(m.sum()),
                                    "score": float(frame["score"].to_numpy()[m].sum()),
                                }  # fmt: skip
                                for s in seeds:
                                    record[f"score_s{s}"] = float(
                                        frame[f"score_s{s}"].to_numpy()[m].sum()
                                    )
                                for tail in TAILS:
                                    q = frame[tail].to_numpy()[m]
                                    a = actual[m]
                                    level = LEVEL_OF[tail]
                                    err = a - q
                                    record[f"pin_{tail}"] = float(
                                        np.maximum(level * err, (level - 1) * err).sum()
                                    )
                                    record[f"cov_{tail}"] = float((a <= q).sum())
                                records.append(record)
    return pl.DataFrame(records)


def summarise(*, table: pl.DataFrame, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return absolute scores: CRPS and tail pinball and coverage by arm, row set, and group."""
    sums = table.group_by("arm", "row_set", "group").agg(
        pl.col("n").sum(), pl.col("score").sum(),
        *[pl.col(f"pin_{t}").sum() for t in TAILS], *[pl.col(f"cov_{t}").sum() for t in TAILS],
    )  # fmt: skip
    return sums.with_columns(
        crps=pl.col("score") / pl.col("n") * POINTS,
        **{f"pinball_{t}": pl.col(f"pin_{t}") / pl.col("n") * POINTS for t in TAILS},
        **{f"coverage_{t}": pl.col(f"cov_{t}") / pl.col("n") for t in TAILS},
    ).drop("score", *[f"pin_{t}" for t in TAILS], *[f"cov_{t}" for t in TAILS])


def contrast_table(*, table: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Return each contrast's mean difference (points) with a block-bootstrap 95% interval."""
    keys = ["row_set", "group"]
    records = []
    for name, (a, b) in CONTRASTS.items():
        left = table.filter(pl.col("arm") == a).select(*keys, "month", "world", "day", "n", "score")
        right = table.filter(pl.col("arm") == b).select(
            *keys, "month", "world", "day", pl.col("score").alias("score_b")
        )
        joined = left.join(right, on=[*keys, "month", "world", "day"], how="inner")
        for (row_set, group), part in joined.group_by(keys):
            blocks = part.group_by("month", "day").agg(
                pl.col("n").sum(), (pl.col("score") - pl.col("score_b")).sum().alias("diff")
            )
            n, diff = blocks["n"].to_numpy().astype(float), blocks["diff"].to_numpy()
            mean = diff.sum() / n.sum() * POINTS
            draws = rng.integers(0, len(n), (RESAMPLES, len(n)))
            boot = diff[draws].sum(1) / n[draws].sum(1) * POINTS
            by_month = part.group_by("month").agg(
                pl.col("n").sum(), (pl.col("score") - pl.col("score_b")).sum().alias("diff")
            )
            records.append(
                {
                    "contrast": name, "row_set": row_set, "group": group, "n": int(n.sum()),
                    "difference": mean, "low": float(np.percentile(boot, 2.5)),
                    "high": float(np.percentile(boot, 97.5)),
                    "fold_differences": {
                        r["month"]: r["diff"] / r["n"] * POINTS for r in by_month.iter_rows(named=True)
                    },
                }
            )  # fmt: skip
    return pl.DataFrame(records)


def minimum_detectable_effect(*, table: pl.DataFrame, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return the plan's minimum detectable effect from R-inj's seed spread, per row set and group."""
    records = []
    sub = table.filter(pl.col("arm") == "R-inj")
    for (row_set, group), part in sub.group_by("row_set", "group"):
        per_fold = part.group_by("month").agg(
            pl.col("n").sum(), *[pl.col(f"score_s{s}").sum() for s in seeds]
        )
        values = np.array(
            [
                [r[f"score_s{s}"] / r["n"] * POINTS for s in seeds]
                for r in per_fold.iter_rows(named=True)
            ]
        )
        folds, n_seeds = values.shape
        residual = values - values.mean(axis=1, keepdims=True)
        sd = math.sqrt((residual**2).sum() / (folds * (n_seeds - 1)))
        records.append(
            {"row_set": row_set, "group": group, "seed_sd": sd,
             "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(n_seeds)}
        )  # fmt: skip
    return pl.DataFrame(records)


def main() -> None:
    """Print the absolute scores, the contrasts, and the minimum detectable effect."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit.SEEDS))
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    table = collect(months=args.months, seeds=seeds)
    out = data.WORK_DIR / "scores"
    out.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out / "blocks.parquet")
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60):
        absolute = summarise(table=table, seeds=seeds)
        absolute.write_parquet(out / "absolute.parquet")
        print(absolute.sort("row_set", "group", "arm"))
        mde = minimum_detectable_effect(table=table, seeds=seeds)
        mde.write_parquet(out / "mde.parquet")
        print(mde.sort("row_set", "group"))
        contrasts = contrast_table(table=table, rng=np.random.default_rng(0))
        contrasts.write_parquet(out / "contrasts.parquet")
        print(contrasts.sort("contrast", "row_set", "group").drop("fold_differences"))


if __name__ == "__main__":
    main()

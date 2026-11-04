# ruff: noqa: E501, PERF401
"""Score experiment 552: CRPS in points of capacity by arm, issue hour, lead band, group, and fold.

CRPS is twice the mean pinball loss over the 11 quantile levels (experiment 448's `row_scores`
convention), times 100. Rows without a finite cleaned power value are dropped identically for
every arm. Contrasts are the pre-registered ones plus the full grid (descriptive). Intervals
resample the four monthly folds (a month-block bootstrap with 4 blocks: all 256 resamples are
enumerated, so the interval is near the minimum-to-maximum of the folds).
"""

import itertools
from pathlib import Path
from typing import Final

import e00448_refs as refs  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00552_data as d  # ty: ignore[unresolved-import]
import e00552_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

OUT: Final[Path] = d.WORK_DIR / "scores"
MONTHS: Final[tuple[str, ...]] = ("2026-03", "2026-04", "2026-05", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1)
BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-6 h", 0, 6), ("6-12 h", 6, 12), ("12-24 h", 12, 24), ("24-48 h", 24, 48), ("48-74 h", 48, 78.01),
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array(c.common.LEVELS)
BONFERRONI: Final[int] = 5
"""Pre-registered contrasts: P1 at 3 issue hours, P2 at 1, P3 pooled over issue hours."""


def per_row() -> pl.DataFrame:
    """Return one row per (arm, seed, fold, forecast row) with its CRPS in points."""
    clean = c.clean_stack()
    group = refs.sites_table().sort("primary")["group"].to_list()
    frames = []
    for arm in run.ARMS:
        for month in MONTHS:
            for seed in SEEDS:
                f = pl.read_parquet(run.PRED_DIR / arm / f"{month}_s{seed}.parquet")
                p = f["primary"].to_numpy()
                y = clean[p, d.target_idx(f["valid"].to_numpy())]
                q = f.select(list(c.common.LEVEL_COLUMNS)).to_numpy()
                err = y[:, None] - q
                pin = np.maximum(LEVELS * err, (LEVELS - 1.0) * err)
                crps = refs.POINTS * 2.0 * pin.mean(1)
                frames.append(
                    f.select("primary", "issue", "valid")
                    .with_columns(
                        pl.Series("crps", crps),
                        pl.Series("ok", np.isfinite(y)),
                        arm=pl.lit(arm),
                        month=pl.lit(month),
                        seed=pl.lit(seed),
                    )
                    .with_columns(pl.Series("group", [group[i] for i in p]))
                )
    df = pl.concat(frames)
    key = df.filter(pl.col("arm") == "N0", pl.col("seed") == 0)
    ok = key.filter(pl.col("ok")).select("primary", "issue", "valid", "month")
    df = df.join(ok, on=["primary", "issue", "valid", "month"], how="inner")
    sizes = df.group_by("arm", "seed", "month").len()
    assert sizes.group_by("month").agg(pl.col("len").n_unique())["len"].max() == 1, (
        "arms score different rows"
    )
    lead = (pl.col("valid") - pl.col("issue")) / d.H
    hour = (pl.col("issue") % d.DAY) // d.H
    band = (
        pl.when(lead < 6)
        .then(pl.lit("0-6 h"))
        .when(lead < 12)
        .then(pl.lit("6-12 h"))
        .when(lead < 24)
        .then(pl.lit("12-24 h"))
        .when(lead < 48)
        .then(pl.lit("24-48 h"))
        .otherwise(pl.lit("48-74 h"))
    )
    return df.with_columns(issue_hour=hour.cast(pl.Int32), band=band, lead_h=lead)


def bootstrap(*, per_fold: np.ndarray) -> tuple[float, float]:
    """Return the 0.5% and 99.5% points of the mean over every one of the 256 resamples of 4 folds."""
    means = [
        per_fold[list(idx)].mean()
        for idx in itertools.product(range(len(per_fold)), repeat=len(per_fold))
    ]
    return float(np.percentile(means, 0.5)), float(np.percentile(means, 99.5))


def contrast(
    df: pl.DataFrame, *, a: str, b: str, hours: list[int], bands: list[str], group: str
) -> dict:
    """Return the CRPS difference a - b for one cell (negative: a is better).

    The result also holds the per-fold values and the seed MDE.
    """
    sub = df.filter(
        pl.col("issue_hour").is_in(hours), pl.col("band").is_in(bands), pl.col("group") == group
    )
    cell = sub.group_by("arm", "month", "seed").agg(pl.col("crps").mean())
    wide = cell.pivot(on="arm", index=["month", "seed"], values="crps").with_columns(
        (pl.col(a) - pl.col(b)).alias("diff")
    )
    per_fold = wide.group_by("month").agg(pl.col("diff").mean()).sort("month")["diff"].to_numpy()
    seed_mean = wide.group_by("seed").agg(pl.col("diff").mean())["diff"].to_numpy()
    lo, hi = bootstrap(per_fold=per_fold)
    return {
        "a": a, "b": b, "hours": ",".join(map(str, hours)), "bands": ",".join(bands), "group": group,
        "n_rows": sub.filter((pl.col("arm") == a) & (pl.col("seed") == 0)).height,
        "diff": float(per_fold.mean()), "lo": lo, "hi": hi,
        "signs": "".join("-" if v < 0 else "+" for v in per_fold), "per_fold": ";".join(f"{v:.3f}" for v in per_fold),
        "mde": float(2.8 * np.std(seed_mean, ddof=1)) if len(seed_mean) > 1 else float("nan"),
    }  # fmt: skip


def main() -> None:
    """Write the absolute scores and the contrasts."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = per_row()
    absolute = (
        df.group_by("arm", "group", "issue_hour", "band", "month")
        .agg(pl.col("crps").mean(), pl.len())
        .group_by("arm", "group", "issue_hour", "band")
        .agg(pl.col("crps").mean(), (pl.col("len").sum() // len(SEEDS)).alias("rows"))
        .sort("arm", "group", "issue_hour", "band")
    )
    absolute.write_csv(OUT / "absolute.csv")
    eight, six = "MVA-only eight", "signed-MW six"
    low, high = ["0-6 h", "6-12 h", "12-24 h"], ["24-48 h", "48-74 h"]
    rows = []
    for grp in (eight, six):
        for h in (6, 12, 18):
            rows.append(
                {
                    "id": "P1" if grp == eight else "P1-six",
                    **contrast(df, a="N1", b="N0", hours=[h], bands=low, group=grp),
                }
            )
        rows.append(
            {
                "id": "P2" if grp == eight else "P2-six",
                **contrast(df, a="N1", b="N0", hours=[0], bands=low, group=grp),
            }
        )
        rows.append(
            {
                "id": "P3" if grp == eight else "P3-six",
                **contrast(df, a="N2", b="N1", hours=[0, 6, 12, 18], bands=high, group=grp),
            }
        )
    for grp in (eight, six):
        for h in (0, 6, 12, 18):
            for band, _, _ in BANDS:
                for a, b in (("N1", "N0"), ("N2", "N1"), ("N2", "N0")):
                    rows.append(
                        {"id": "grid", **contrast(df, a=a, b=b, hours=[h], bands=[band], group=grp)}
                    )
    pl.DataFrame(rows).write_csv(OUT / "contrasts.csv")
    print(
        pl.DataFrame(rows)
        .filter(pl.col("id") != "grid")
        .drop("per_fold", "bands")
        .with_columns(pl.col(pl.Float64).round(3))
    )
    print(
        absolute.filter(pl.col("group") == eight)
        .with_columns(pl.col("crps").round(2))
        .pivot(on="arm", index=["issue_hour", "band"], values="crps")
    )


if __name__ == "__main__":
    main()

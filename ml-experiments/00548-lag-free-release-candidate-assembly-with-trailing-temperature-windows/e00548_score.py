# ruff: noqa: E501
"""Score arm E (experiment 548) against L-1d, A-pool, R-DW, and the rebuilt K2-S1 on experiment 533's rows.

Reads the per-row scores written by `e00548_rows.py`, experiment 539's R-DW rows, and experiment 541's
rebuilt K2-S1 rows, and uses experiment 521's loader, lead bands, row sets, and month-block bootstrap.
Intervals are 95% month-block bootstraps with 5 blocks (near the minimum to maximum of the folds),
widened for the Bonferroni count `TESTS`. Output: `scores/absolute.csv`, `scores/contrasts.csv`,
`scores/slices.csv`, and `scores/checks.json`, under this experiment's private data folder.
"""

import json
import math
from pathlib import Path
from typing import Final

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD: Final[Path] = ML_EXPERIMENTS_DATA_DIR
ROOT: Final[Path] = (
    DD / "00548-lag-free-release-candidate-assembly-with-trailing-temperature-windows"
)
E524: Final[Path] = (
    DD / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator" / "work" / "scores"
)
E539: Final[Path] = DD / "00539-l1d-plus-day-and-week-lags" / "work" / "scores"
E541: Final[Path] = DD / "00541-k2-s1-rebuilt-on-origin-time-weather" / "work"
K: Final[list[str]] = ["primary", "origin", "target"]
METRICS: Final[tuple[str, ...]] = ("crps", "pin_q95", "pin_q99", "cov_q95", "cov_q99")
ARMS: Final[tuple[str, ...]] = (
    "E", "R-DW", "A-pool", "L-1d", "L-Ww", "K2-S1-origin-trained", "K2-S1", "manual_heuristic",
)  # fmt: skip
CONTRASTS: Final[tuple[tuple[str, str, bool], ...]] = (
    ("E", "A-pool", True),
    ("E", "L-1d", True),
    ("E", "R-DW", True),
    ("E", "K2-S1-origin-trained", False),
    ("E", "K2-S1", False),
    ("E", "manual_heuristic", False),
    ("R-DW", "A-pool", False),
    ("L-1d", "A-pool", False),
)
TESTS: Final[int] = 42
"""3 counted contrasts x 7 lead bands x 2 row sets."""
SLICES: Final[dict[str, pl.Expr]] = {
    "all rows": pl.col("main16"),
    "cold winter nights": pl.col("cold"),
    "inside a labelled event": pl.col("tgt_real"),
}


def table() -> pl.DataFrame:
    """Return experiment 521's table joined with the per-row scores of A-pool, R-DW, E, and the rebuilt K2-S1."""
    base = s.load()
    extra = pl.concat(
        [
            pl.read_parquet(
                E524 / f"rows_{m}.parquet",
                columns=[
                    *K,
                    *[f"{a}__{x}" for a in ("A-pool", "manual_heuristic") for x in METRICS],
                ],
            )
            for m in s.MONTHS
        ]
    )
    assert base.select(K).equals(extra.select(K))
    drop = [c for c in extra.columns[3:] if c in base.columns]
    out = pl.concat([base.drop(drop), extra.drop(K)], how="horizontal")
    n0 = out.height
    for folder, arm_in in (
        (E539, "R-DW"),
        (ROOT / "work" / "scores", "E"),
    ):
        part = pl.concat(
            [
                pl.read_parquet(
                    folder / f"rows_{m}.parquet",
                    columns=[*K, *[f"{arm_in}__{x}" for x in METRICS]],
                )
                for m in s.MONTHS
            ]
        )
        out = out.join(part, on=K, how="left", validate="1:1", maintain_order="left")
    k2 = pl.concat(
        [
            pl.read_parquet(
                E541 / f"new_{m}.parquet",
                columns=[*K, *[f"K2-S1-origin-trained__{x}" for x in METRICS]],
            )
            for m in s.MONTHS
        ]
    )
    out = out.join(k2, on=K, how="left", validate="1:1", maintain_order="left")
    assert out.height == n0
    return out


def mde(*, sub: pl.DataFrame, cols: list[str]) -> float:
    """Return the minimum detectable effect from the spread of the seeds' per-fold means."""
    mm = (
        sub.group_by("month")
        .agg([pl.col(c).mean() for c in cols])
        .sort("month")
        .select(cols)
        .to_numpy()
    )
    f, k = mm.shape
    r = mm - mm.mean(axis=1, keepdims=True)
    return 1.96 * math.sqrt(2) * math.sqrt(float((r**2).sum()) / (f * (k - 1))) / math.sqrt(k)


def main() -> None:
    """Write the absolute scores, the contrasts, and the slices."""
    t = table().with_columns(band=s.band_expr())
    need = [f"{a}__crps" for a in ARMS if a != "manual_heuristic" and f"{a}__crps" in t.columns]
    fin = pl.all_horizontal([pl.col(c).is_finite() for c in need])
    t = t.with_columns(main16=pl.col("main16") & fin)
    checks = {
        "rows": t.height,
        "rows with every arm finite": int(t["main16"].sum()),
        "columns used": need,
    }
    (ROOT / "scores").mkdir(exist_ok=True, parents=True)
    (ROOT / "scores" / "checks.json").write_text(json.dumps(checks, indent=1))
    print(checks)
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    seedcols = ["L-1d-seed0__crps", "L-1d-seed1__crps"]
    for name in ("eight", "six", "report"):
        part = s.row_set(table=t, name=name)
        for slice_name, expr in SLICES.items():
            sliced = part.filter(expr.fill_null(value=False))
            for band in (*s.BANDS, s.ALL):
                sub = s.in_band(part=sliced, band=band)
                if sub.height < 200:
                    continue
                m = mde(sub=sub, cols=seedcols)
                for arm in ARMS:
                    for metric in METRICS:
                        col = f"{arm}__{metric}"
                        if col in sub.columns:
                            absolute.append({"set": name, "slice": slice_name, "band": band, "arm": arm, "metric": metric, "n": sub.height, "value": float(sub.filter(pl.col(col).is_finite())[col].mean())})  # fmt: skip
                for a, b, counted in CONTRASTS:
                    for metric in ("crps", "pin_q95", "pin_q99"):
                        ca, cb = f"{a}__{metric}", f"{b}__{metric}"
                        if ca not in sub.columns or cb not in sub.columns:
                            continue
                        d = sub.select("month", "day", d=pl.col(ca) - pl.col(cb)).filter(
                            pl.col("d").is_finite()
                        )
                        if d.height < 200:
                            continue
                        st = s.block_stats(diff=d, rng=rng, tests=TESTS)
                        contrasts.append({"set": name, "slice": slice_name, "band": band, "metric": metric, "contrast": f"{a} - {b}", "counted": counted and band != s.ALL and name != "report" and slice_name == "all rows" and metric == "crps", "tests": TESTS, "mde": m, **st})  # fmt: skip
    pl.DataFrame(absolute).write_csv(ROOT / "scores" / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(ROOT / "scores" / "contrasts.csv")


if __name__ == "__main__":
    main()

# ruff: noqa: E501, PERF401
"""Cut experiment 511's per-row scores by lead time and test where K2-S1 beats the daily lag-free network.

Inputs (stored, read-only; no fit and no GPU):

- experiment 511's `work/scores/rows_{month}.parquet`: per-row CRPS and tail scores of K2-S1 (cell c0
  alone, see `e00521_refs.py`), L-1d, and L-Ww;
- this experiment's `work/refs_{month}.parquet`: the same scores for K2-S1's three cells,
  experiment 454's monthly lag-free arms (`LL`, `LL-ens3`, `LL-ens5`), and L-1d's two seeds;
- this experiment's `work/flags_{month}.parquet`: experiment 454's `lag_real` and `win_real` flags.

A difference "A - B" is arm A's mean CRPS minus arm B's, in points of effective capacity, so a
positive number means B is better. "Gap" in this experiment is L-1d minus K2-S1: positive means
K2-S1 is better. Row sets: `eight` (the MVA-only Primaries), `six` (the signed-MW Primaries, rows
with no flagged class, as experiment 470's `six_clean`), `six_all` (every row of the six), and
`report` (D15 and D18). Lead bands are in hours. Intervals are percentile intervals of a block
bootstrap: blocks are origin days within a month (as experiment 511 used), and, as a conservative
check, whole months (5 blocks, so near the minimum-to-maximum range of the five folds). The Bonferroni
level uses 28 counted tests: 7 lead bands x 2 groups x 2 contrasts.

```bash
uv run --no-sync python ml-experiments/00521-*/e00521_stats.py
```
"""

import json
import math
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00521_refs import DATA, E511, MONTHS, WORK

SCORES: Final[Path] = DATA / "scores"
KEYS: Final[list[str]] = ["primary", "origin", "target"]
BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-6h": (0.0, 6.0),
    "6-12h": (6.0, 12.0),
    "12-24h": (12.0, 24.0),
    "24-48h": (24.0, 48.0),
    "48-72h": (48.0, 72.0),
    "72-120h": (72.0, 120.0),
    "120-240h": (120.0, 240.0),
}
"""Lead bands in hours, lower bound inclusive, upper bound exclusive (the last includes 240)."""
ALL: Final[str] = "all"
TESTS: Final[int] = 28
RESAMPLES: Final[int] = 20_000
TAILS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
FROM_511: Final[tuple[str, ...]] = ("K2-S1", "L-1d", "L-Ww")
FROM_REFS: Final[tuple[str, ...]] = (
    "K2-S1-3cell", "K2-S1-c1", "K2-S1-c2", "LL", "LL-ens3", "LL-ens5", "L-1d-seed0", "L-1d-seed1",
)  # fmt: skip
PRIMARY_LABELS: Final[dict[int, str]] = {}
"""Filled by `load()` from the site map: Primary index -> label (D2, D3, ...)."""
SETS: Final[tuple[str, ...]] = ("eight", "six", "six_all", "report")
CONTRASTS: Final[tuple[tuple[str, str, bool], ...]] = (
    ("L-1d", "K2-S1", True),
    ("L-1d", "LL-ens5", True),
    ("L-1d", "K2-S1-3cell", False),
    ("L-1d", "LL-ens3", False),
    ("L-1d", "LL", False),
    ("L-1d", "L-Ww", False),
    ("L-Ww", "K2-S1", False),
    ("K2-S1", "LL-ens5", False),
)
"""(A, B, counted): A minus B. The first two are the 28 counted tests; the rest are report-only."""


def load() -> pl.DataFrame:
    """Return the five folds' rows with every arm's scores and the flags, on experiment 511's rows."""
    sites = pl.read_csv(DATA / "site_map.csv")
    PRIMARY_LABELS.update(
        dict(zip(sites["primary"].to_list(), sites["label"].to_list(), strict=True))
    )
    metrics = ("crps", *[f"{k}_{t}" for k in ("pin", "cov") for t in TAILS])
    wanted = [f"{a}__{m}" for a in FROM_511 for m in metrics]
    frames = []
    for month in MONTHS:
        rows = pl.read_parquet(
            E511 / "scores" / f"rows_{month}.parquet",
            columns=[
                *KEYS, *wanted, "group", "cls", "lead_hours", "tgt_real", "cold", "xmas", "main16",
            ],
        )  # fmt: skip
        refs = pl.read_parquet(WORK / f"refs_{month}.parquet").drop(KEYS)
        flags = pl.read_parquet(WORK / f"flags_{month}.parquet").drop("tgt_real")
        merged = pl.concat([rows, refs], how="horizontal").join(
            flags, on=KEYS, how="left", maintain_order="left"
        )
        frames.append(
            merged.with_columns(
                month=pl.lit(month),
                day=(pl.col("origin") // 48).cast(pl.Int32),
                half_hour=(pl.col("target") % 48).cast(pl.Int32),
            )
        )
    return pl.concat(frames)


def band_expr() -> pl.Expr:
    """Return the lead band name of `lead_hours`."""
    expr = pl.lit(None, dtype=pl.String)
    for i, (name, (low, high)) in enumerate(BANDS.items()):
        last = i == len(BANDS) - 1
        lead = pl.col("lead_hours")
        inside = (lead >= low) & ((lead <= high) if last else (lead < high))
        expr = pl.when(inside).then(pl.lit(name)).otherwise(expr)
    return expr


def row_set(*, table: pl.DataFrame, name: str) -> pl.DataFrame:
    """Return the rows of a row set (all arms present)."""
    part = table.filter(pl.col("main16"))
    if name == "eight":
        return part.filter(pl.col("group") == "mva_eight")
    if name == "six":
        return part.filter((pl.col("group") == "signed_six") & (pl.col("cls") == "clean"))
    if name == "six_all":
        return part.filter(pl.col("group") == "signed_six")
    return part.filter(pl.col("group") == "report_only")


def in_band(*, part: pl.DataFrame, band: str) -> pl.DataFrame:
    """Return the rows of a lead band, or all rows for `all`."""
    return part if band == ALL else part.filter(pl.col("band") == band)


def percentiles(*, boot: np.ndarray, tests: int = TESTS) -> dict[str, float]:
    """Return the 95% and the Bonferroni percentile interval of bootstrap draws."""
    alpha = 100.0 * 0.05 / tests / 2.0
    return {
        "low": float(np.percentile(boot, 2.5)), "high": float(np.percentile(boot, 97.5)),
        "low_bonferroni": float(np.percentile(boot, alpha)),
        "high_bonferroni": float(np.percentile(boot, 100.0 - alpha)),
    }  # fmt: skip


def block_stats(*, diff: pl.DataFrame, rng: np.random.Generator, tests: int = TESTS) -> dict:
    """Return the mean of `d`, its day-block and month-block intervals, and the per-fold means."""
    out: dict = {"n": diff.height, "difference": float(diff["d"].mean())}
    for label, keys in (("day", ["month", "day"]), ("month", ["month"])):
        blocks = diff.group_by(keys).agg(pl.len().alias("n"), pl.col("d").sum().alias("s"))
        n, s = blocks["n"].to_numpy().astype(float), blocks["s"].to_numpy()
        draws = rng.integers(0, len(n), (RESAMPLES, len(n)))
        boot = s[draws].sum(axis=1) / n[draws].sum(axis=1)
        out |= {f"{k}_{label}": v for k, v in percentiles(boot=boot, tests=tests).items()}
        out[f"blocks_{label}"] = len(n)
    by_fold = diff.group_by("month").agg(pl.col("d").mean()).sort("month")
    out["by_fold"] = json.dumps(
        {r["month"]: round(r["d"], 4) for r in by_fold.iter_rows(named=True)}
    )
    out["folds_positive"] = int((by_fold["d"] > 0).sum())
    out["folds"] = by_fold.height
    return out


def contrast_table(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return every contrast in every row set and lead band, with intervals."""
    rng = np.random.default_rng(0)
    records = []
    for name in SETS:
        part = row_set(table=table, name=name)
        for a, b, counted in CONTRASTS:
            for metric in ("crps", "pin_q99") if counted else ("crps",):
                tests = TESTS * (2 if metric == "pin_q99" else 1)
                for band in (*BANDS, ALL):
                    sub = in_band(part=part, band=band).select(
                        "month", "day", d=pl.col(f"{a}__{metric}") - pl.col(f"{b}__{metric}")
                    )
                    sub = sub.filter(pl.col("d").is_finite())
                    records.append(
                        {"set": name, "contrast": f"{a} - {b}", "metric": metric, "band": band,
                         "counted": counted and band != ALL and name in ("eight", "six"),
                         "tests": tests, **block_stats(diff=sub, rng=rng, tests=tests)}
                    )  # fmt: skip
    return pl.DataFrame(records)


def absolute_table(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's mean CRPS and tail scores per row set and lead band."""
    arms = (*FROM_511, *FROM_REFS)
    cols = ("crps", *[f"{k}_{t}" for k in ("pin", "cov") for t in TAILS])
    records = []
    for name in SETS:
        part = row_set(table=table, name=name)
        for band in (*BANDS, ALL):
            sub = in_band(part=part, band=band)
            for arm in arms:
                records.append(
                    {"set": name, "band": band, "arm": arm, "n": sub.height}
                    | {m: float(sub[f"{arm}__{m}"].mean()) for m in cols}
                )
    return pl.DataFrame(records)


def primary_table(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the gap L-1d minus K2-S1 per Primary, with its share of the row set's summed gap."""
    records = []
    for name in ("eight", "six", "six_all", "report"):
        part = row_set(table=table, name=name).with_columns(
            d=pl.col("L-1d__crps") - pl.col("K2-S1__crps")
        )
        for band in (*BANDS, ALL):
            sub = in_band(part=part, band=band)
            total = float(sub["d"].sum())
            by = sub.group_by("primary").agg(pl.len().alias("n"), pl.col("d").sum().alias("s"))
            for r in by.sort("primary").iter_rows(named=True):
                records.append(
                    {"set": name, "band": band, "primary": PRIMARY_LABELS[r["primary"]],
                     "n": r["n"], "gap": r["s"] / r["n"], "share_of_pooled_gap": r["s"] / total,
                     "pooled_gap": total / sub.height}
                )  # fmt: skip
    return pl.DataFrame(records)


def slice_table(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the gap L-1d minus K2-S1 and L-1d minus LL-ens5 on slices (report-only, uncorrected)."""
    rng = np.random.default_rng(1)
    slices: dict[str, pl.Expr] = {
        "all rows": pl.lit(True),
        "no labelled event (tgt_real false)": ~pl.col("tgt_real"),
        "inside a labelled event (tgt_real true)": pl.col("tgt_real"),
        "lag_real true": pl.col("lag_real"),
        "lag_real false and win_real false": ~pl.col("lag_real") & ~pl.col("win_real"),
        "cold winter nights": pl.col("cold"),
        "Christmas week": pl.col("xmas"),
    }
    records = []
    for name in ("eight", "six_all", "report"):
        part = row_set(table=table, name=name)
        for slice_name, expr in slices.items():
            for a, b in (("L-1d", "K2-S1"), ("L-1d", "LL-ens5")):
                for band in (*BANDS, ALL):
                    sub = in_band(part=part, band=band).filter(expr.fill_null(value=False))
                    diff = sub.select("month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps"))
                    diff = diff.filter(pl.col("d").is_finite())
                    if diff.height < 200:
                        continue
                    records.append(
                        {"set": name, "slice": slice_name, "contrast": f"{a} - {b}", "band": band,
                         **block_stats(diff=diff, rng=rng)}
                    )  # fmt: skip
    return pl.DataFrame(records)


def half_hour_table(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the gap L-1d minus K2-S1 by UTC half-hour of the valid time and coarse lead band."""
    coarse = (
        pl.when(pl.col("lead_hours") < 24)
        .then(pl.lit("0-24h"))
        .when(pl.col("lead_hours") < 120)
        .then(pl.lit("24-120h"))
        .otherwise(pl.lit("120-240h"))
    )
    records = []
    for name in ("eight", "six"):
        part = row_set(table=table, name=name).with_columns(
            coarse=coarse, d=pl.col("L-1d__crps") - pl.col("K2-S1__crps")
        )
        by = part.group_by("coarse", "half_hour").agg(pl.len().alias("n"), pl.col("d").mean())
        records.append(by.with_columns(set=pl.lit(name)))
    return pl.concat(records).sort("set", "coarse", "half_hour")


def mde_table(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the minimum detectable effect per set and band from the seed or cell spread.

    The spread is the standard deviation, over the five folds, of the arm's seeds (L-1d: two seeds;
    K2-S1: three cells), pooled as in experiment 470: `1.96 * sqrt(2) * sd / sqrt(k)`.
    """
    sources = {
        "L-1d (2 seeds)": ["L-1d-seed0__crps", "L-1d-seed1__crps"],
        "K2-S1 (3 cells)": ["K2-S1__crps", "K2-S1-c1__crps", "K2-S1-c2__crps"],
    }
    records = []
    for name in ("eight", "six", "report"):
        part = row_set(table=table, name=name)
        for band in (*BANDS, ALL):
            sub = in_band(part=part, band=band)
            for label, cols in sources.items():
                means = sub.group_by("month").agg([pl.col(c).mean() for c in cols]).sort("month")
                values = means.select(cols).to_numpy()
                folds, k = values.shape
                resid = values - values.mean(axis=1, keepdims=True)
                sd = math.sqrt(float((resid**2).sum()) / (folds * (k - 1)))
                records.append(
                    {"set": name, "band": band, "source": label, "seed_sd": sd,
                     "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(k)}
                )  # fmt: skip
    return pl.DataFrame(records)


def checks_table(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the reproduction checks: row counts, and the pooled gaps experiment 511 reported."""
    records = []
    for name in ("eight", "six", "six_all", "report"):
        part = row_set(table=table, name=name)
        records.append({"check": f"rows in {name}", "value": float(part.height)})
        gap = float((part["L-1d__crps"] - part["K2-S1__crps"]).mean())
        records.append({"check": f"pooled gap L-1d minus K2-S1, {name}", "value": gap})
        no_event = part.filter(~pl.col("tgt_real"))
        gap_ne = float((no_event["L-1d__crps"] - no_event["K2-S1__crps"]).mean())
        records.append({"check": f"pooled gap, {name}, tgt_real false", "value": gap_ne})
        gap3 = float((part["L-1d__crps"] - part["K2-S1-3cell__crps"]).mean())
        records.append({"check": f"pooled gap vs 3-cell K2-S1, {name}", "value": gap3})
    for name in ("eight", "six"):
        part = row_set(table=table, name=name)
        total = float((part["L-1d__crps"] - part["K2-S1__crps"]).sum())
        for band in BANDS:
            sub = in_band(part=part, band=band)
            share = float((sub["L-1d__crps"] - sub["K2-S1__crps"]).sum()) / total
            records.append({"check": f"share of pooled summed gap, {name} {band}", "value": share})
            records.append(
                {
                    "check": f"rows in {name} {band}",
                    "value": float(in_band(part=part, band=band).height),
                }
            )
    return pl.DataFrame(records)


def main() -> None:
    """Write every table to this experiment's `scores` folder and print the headline profile."""
    SCORES.mkdir(parents=True, exist_ok=True)
    table = load().with_columns(band=band_expr())
    mde = mde_table(table=table)
    mde.write_csv(SCORES / "mde.csv")
    print("minimum detectable effect (printed before any contrast):")
    with pl.Config(tbl_rows=100, tbl_width_chars=200):
        print(mde.filter(pl.col("set") != "report").with_columns(pl.col(pl.Float64).round(3)))
    checks = checks_table(table=table)
    checks.write_csv(SCORES / "checks.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=200):
        print(checks.with_columns(pl.col(pl.Float64).round(4)))
    absolute_table(table=table).write_csv(SCORES / "absolute.csv")
    contrasts = contrast_table(table=table)
    contrasts.write_csv(SCORES / "contrasts.csv")
    print(
        f"counted tests: {int(contrasts['counted'].sum())} (28 CRPS + 28 more for q99, pre-registered 56)"
    )
    primary_table(table=table).write_csv(SCORES / "per_primary.csv")
    slice_table(table=table).write_csv(SCORES / "slices.csv")
    half_hour_table(table=table).write_csv(SCORES / "half_hour.csv")
    show = contrasts.filter(
        (pl.col("contrast") == "L-1d - K2-S1") & (pl.col("metric") == "crps")
    ).drop("by_fold")
    with pl.Config(tbl_rows=100, tbl_width_chars=250):
        print(show.select(pl.exclude("n")).with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()

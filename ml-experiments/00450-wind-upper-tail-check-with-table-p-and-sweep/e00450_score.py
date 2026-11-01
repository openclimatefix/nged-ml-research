"""Experiment 450, step 2 scoring: scores, contrasts against the base, and tails after table P.

Reads the fitted arms' quantile tables (`parts/<arm>_h<seed>.npy`; the base arm falls back to
experiment 429's `member_plain` when it has not been refitted, which is bit-identical for seeds 0 to
4). Score: quantile score over 11 levels in points of capacity, kept rows (outside experiment 210's
idle-band mask), true days 1 to 10 (lead 24 to 240 hours); also reported on all rows. Each arm's
score is the mean over its seeds of the row-level scores; a contrast is the paired row-level
difference to the base, with a month-block bootstrap interval (2,000 resamples of the 12 months)
at the stated level. Tails: coverage and pinball loss at q01, q05, q95, q99 raw and after table P
(`e00450_step1`), pooled and per farm.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00450-*/env.sh
uv run --no-sync python ml-experiments/00450-*/e00450_score.py <arm> ... [--level 0.99]
```
Output: `work/score_<tag>.json` for each call's arm list, and a printed table.
"""

import json
import sys
from pathlib import Path
from typing import Final

import e00303_score as score303  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00429_run as run429  # ty: ignore[unresolved-import]
import e00450_run as run450  # ty: ignore[unresolved-import]
import e00450_step1 as step1  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

WORK_DIR: Final[Path] = run450.DATA_DIR / "work"
RESAMPLES: Final[int] = 2000
MARGIN: Final[float] = 0.10
"""The agreed wind margin in points of capacity."""
TAIL_HARM: Final[float] = 0.02
UPPER_TAIL_LIMIT: Final[float] = 0.015


def load_seed_tables(*, name: str, seeds: tuple[int, ...]) -> list[np.ndarray]:
    """Return one (rows, 11) table per seed, base falling back to experiment 429's member_plain."""
    tag = name.replace(":", "_")
    tables = []
    for seed in seeds:
        path = run450.PARTS_DIR / f"{tag}_h{seed}.npy"
        if not path.exists() and name == "base":
            path = run429.PARTS_DIR / f"member_plain_h{seed}.npy"
        tables.append(np.load(path))
    return tables


def row_scores(*, tables: list[np.ndarray], actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return seed-mean row scores (qs, p01, p99) and the per-seed mean of qs over the rows."""
    metrics = score303.row_metrics(quantiles=np.stack(tables), actual=actual)
    return {k: metrics[k].mean(axis=0) for k in ("qs", "p01", "p99")} | {
        "qs_by_seed": metrics["qs"].mean(axis=1)
    }


def month_block_interval(
    *, diff: np.ndarray, month: np.ndarray, level: float
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) of a row-level difference with a month-block bootstrap."""
    months = np.unique(month)
    total = np.array([diff[month == m].sum() for m in months])
    count = np.array([(month == m).sum() for m in months], dtype=float)
    draws = np.random.default_rng(0).integers(0, len(months), size=(RESAMPLES, len(months)))
    boot = total[draws].sum(axis=1) / count[draws].sum(axis=1)
    low, high = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(total.sum() / count.sum()), float(low), float(high)


def tails_after_p(*, name: str, tables: list[np.ndarray]) -> pl.DataFrame:
    """Return per-farm and pooled share above q99 and coverage at q01, q05 for raw and P."""
    cases = data347.scored_frame()
    q = np.sort(np.mean(np.stack(tables), axis=0), axis=1)
    base = cases.select("site", "month", "masked", "lead_hours", actual=pl.col("actual_norm"))
    frame = base.with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(step1.LEVELS)])
    frame = (
        frame.filter(pl.col("actual").is_finite())
        .with_columns(lead_band=step1.lead_band_expr())
        .filter(pl.col("lead_band").is_not_null())
    )
    months = sorted(frame["month"].unique())
    out: list[pl.DataFrame] = []
    for rowset, keep in (("all rows", pl.lit(value=True)), ("kept rows", ~pl.col("masked"))):
        sub = frame.filter(keep)
        for i, month in enumerate(months[1:], start=1):
            earlier = sub.filter(pl.col("month").is_in(months[:i]))
            held = sub.filter(pl.col("month") == month)
            factors = step1.fit_factors(fit=earlier)
            for variant, f in (
                ("raw", held),
                ("P", step1.apply_factors(frame=held, factors=factors)),
            ):
                out.append(step1.hits(frame=f, arm=name, rowset=rowset, variant=variant))
    cells = pl.concat(out)
    pooled = cells.group_by("arm", "rowset", "variant").agg(
        pl.col("n").sum(), pl.col("^hit_.*$").sum()
    )
    return pooled.with_columns(
        above_q99=1.0 - pl.col("hit_q99") / pl.col("n"),
        cov_q01=pl.col("hit_q01") / pl.col("n"),
        cov_q05=pl.col("hit_q05") / pl.col("n"),
    ).select("arm", "rowset", "variant", "n", "above_q99", "cov_q01", "cov_q05")


def main() -> None:
    """Score the arms named on the command line against the base."""
    args = sys.argv[1:]
    seeds, level = run450.SEEDS, 0.99
    if "--level" in args:
        at = args.index("--level")
        level = float(args[at + 1])
        args = args[:at] + args[at + 2 :]
    if "--seeds" in args:
        at = args.index("--seeds")
        seeds = tuple(int(s) for s in args[at + 1 :])
        args = args[:at]
    cases = data347.scored_frame()
    true_1_10 = cases["lead_hours"].is_between(24, 240).to_numpy()
    kept = ~cases["masked"].to_numpy()
    month = cases["month"].to_numpy()
    actual = cases["actual_norm"].to_numpy()
    results: dict[str, dict] = {}
    base_rows: dict[str, dict[str, np.ndarray]] = {}
    for rowset, sel in (("kept rows", true_1_10 & kept), ("all rows", true_1_10)):
        tables = [t[sel] for t in load_seed_tables(name="base", seeds=run450.SEEDS)]
        base_rows[rowset] = row_scores(tables=tables, actual=actual[sel])
    for name in ["base", *[a for a in args if a != "base"]]:
        entry: dict = {}
        full = load_seed_tables(name=name, seeds=seeds if name != "base" else run450.SEEDS)
        for rowset, sel in (("kept rows", true_1_10 & kept), ("all rows", true_1_10)):
            rows = row_scores(tables=[t[sel] for t in full], actual=actual[sel])
            ref = base_rows[rowset]
            cell: dict = {
                "qs": float(rows["qs"].mean()),
                "seed_sd": float(rows["qs_by_seed"].std(ddof=1)) if len(full) > 1 else None,
                "p01": float(rows["p01"].mean()),
                "p99": float(rows["p99"].mean()),
            }
            for metric in ("qs", "p01", "p99"):
                d, lo, hi = month_block_interval(
                    diff=rows[metric] - ref[metric], month=month[sel], level=level
                )
                cell[f"d_{metric}"] = [d, lo, hi]
            entry[rowset] = cell
        entry["tails"] = tails_after_p(name=name, tables=full).to_dicts()
        results[name] = entry
        kept_cell = entry["kept rows"]
        p_row = next(
            t for t in entry["tails"] if t["rowset"] == "kept rows" and t["variant"] == "P"
        )
        d = kept_cell["d_qs"]
        print(
            f"{name:12s} qs {kept_cell['qs']:.3f} (all rows {entry['all rows']['qs']:.3f}) "
            f"d vs base {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] "
            f"dp01 {kept_cell['d_p01'][0]:+.4f} dp99 {kept_cell['d_p99'][0]:+.4f} "
            f"above q99 after P {p_row['above_q99']:.4f}",
            flush=True,
        )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(a.replace(":", "_") for a in args) or "base"
    (WORK_DIR / f"score_{tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

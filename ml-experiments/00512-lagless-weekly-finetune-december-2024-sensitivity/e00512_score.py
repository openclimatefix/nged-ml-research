"""Score experiment 512's arms (L-M, L-W, L-Ww on the fold 2024-12) and write `scores/slices.csv`.

Seeds 0 and 1 are averaged by quantile and widened by 1.15 (experiment 470's scorer functions). The
actuals are the cleaned power over capacity; rows with an unflagged actual are scored. No K2-S1,
A-pool, or event labels exist for this month, so the page reports each arm's CRPS in points of
effective capacity, split by group (the eight and the six), week of the month, and Christmas week
(valid 22 to 31 December), with origin-day block-bootstrap intervals (no multiplicity correction:
descriptive).

```bash
source <paths>; uv run --no-sync python ml-experiments/00512-*/e00512_score.py
```
"""

from pathlib import Path
from typing import Final

import e00404_report as report  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00512-lagless-weekly-finetune-december-2024-sensitivity" / "work"
)
MONTH: Final[str] = "2024-12"
ARMS: Final[tuple[str, ...]] = ("L-M", "L-W", "L-Ww")
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (("L-W", "L-M"), ("L-Ww", "L-W"))
HEADLINE: Final[float] = 1.15
GROUPS: Final[dict[str, str]] = {"eight": "mva_eight", "six": "signed_six"}


def arm_rows(*, arm: str, sites: pl.DataFrame, actual: np.ndarray) -> pl.DataFrame:
    """Return an arm's per-row CRPS (points of capacity) with the keys and actuals."""
    pooled = build470.pool_stored(
        folder=WORK_DIR / "seed_preds" / arm, month=MONTH, sites=sites, seeds=(0, 1)
    )
    primary = pooled["primary"].to_numpy().astype(np.int64)
    target = pooled["target"].to_numpy().astype(np.int64)
    frame = pooled.with_columns(actual=pl.Series(actual[primary, target].astype(np.float64)))
    frame = score470.widen_arm(arm=arm, frame=frame, factor=HEADLINE)
    ok = frame["actual"].is_finite().to_numpy()
    crps = score470.row_metrics(frame=frame.with_columns(pl.col("actual").fill_nan(0.0)))["crps"]
    return frame.select("primary", "origin", "target", "valid_time").with_columns(
        crps=pl.Series(np.where(ok, crps, np.nan))
    )


def boot(*, d: pl.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    """Return the mean of `d['d']` and its origin-day block-bootstrap 95% interval."""
    by_day = d.group_by("day").agg(s=pl.col("d").sum(), n=pl.len())
    s, n = by_day["s"].to_numpy(), by_day["n"].to_numpy()
    draws = np.empty(2_000)
    for i in range(len(draws)):
        pick = rng.integers(0, len(s), len(s))
        draws[i] = s[pick].sum() / n[pick].sum()
    return (
        float(s.sum() / n.sum()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def main() -> None:
    """Write and print the slice table."""
    sites = build470.site_map()
    actual = c.clean_stack()
    names = report.group_of(sites=[str(s) for s in c.sites()])
    tables = {a: arm_rows(arm=a, sites=sites, actual=actual) for a in ARMS}
    wide = tables["L-M"].select("primary", "origin", "target", "valid_time", **{"L-M": "crps"})
    for a in ARMS[1:]:
        wide = wide.join(
            tables[a].select("primary", "origin", "target", **{a: "crps"}),
            on=["primary", "origin", "target"],
        )
    wide = wide.filter(pl.all_horizontal(pl.col(a).is_finite() for a in ARMS))
    wide = wide.with_columns(
        group=pl.Series(names[wide["primary"].to_numpy()]),
        day=(pl.col("origin") // 48).cast(pl.Int32),
    )
    day = pl.col("valid_time").dt.day()
    first_day = int(wide["day"].min())  # ty: ignore[invalid-argument-type]
    slices = {
        "all": wide,
        "christmas": wide.filter((pl.col("valid_time").dt.month() == 12) & day.is_between(22, 31)),
        **{
            f"origin_days_{lo}_{hi}": wide.filter(
                pl.col("day").is_between(lo - 1 + first_day, hi - 1 + first_day)
            )
            for lo, hi in ((1, 7), (8, 14), (15, 21), (22, 28), (29, 31))
        },
    }
    rng = np.random.default_rng(0)
    records = []
    for sname, frame in slices.items():
        for gname, group in GROUPS.items():
            part = frame.filter(pl.col("group") == group)
            if part.height == 0:
                continue
            records.extend(
                {
                    "slice": sname,
                    "group": gname,
                    "kind": "mean",
                    "arm": a,
                    "n": part.height,
                    "value": part[a].mean(),
                }
                for a in ARMS
            )
            for a, b in CONTRASTS:
                est, lo, hi = boot(d=part.select("day", d=pl.col(a) - pl.col(b)), rng=rng)
                records.append(
                    {
                        "slice": sname,
                        "group": gname,
                        "kind": f"{a} - {b}",
                        "arm": "",
                        "n": part.height,
                        "value": est,
                        "low": lo,
                        "high": hi,
                    }
                )
    table = pl.DataFrame(records, infer_schema_length=None)
    (WORK_DIR / "scores").mkdir(parents=True, exist_ok=True)
    table.write_csv(WORK_DIR / "scores" / "slices.csv")
    with pl.Config(tbl_rows=200, tbl_width_chars=160):
        print(table.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()

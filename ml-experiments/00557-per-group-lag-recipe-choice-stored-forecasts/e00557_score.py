# ruff: noqa: E501, PLR0912
"""Score three options for the lag recipe across the 14 Primaries from stored per-row scores.

Option (a): the lag-light network of experiment 553 (arm RDWT-D15) at all 14 Primaries. Option (b):
the lag-free recipe of experiment 548 (arm E) at all 14. Option (c): RDWT-D15 at the eight MVA-only
Primaries and E at the six signed-MW Primaries, a row selection of the two stored arms. D15 and D18
(the report set) are scored for (a) and (b) only. Reuses experiment 553's table, slices, and minimum
detectable effect, and experiment 521's lead bands and month-block bootstrap (95%, 5 blocks, near
the minimum to maximum of the folds), widened for the Bonferroni count `TESTS`. No fit is run.
Output under this experiment's private data folder, `scores/`: `absolute.csv`, `contrasts.csv`, and
`checks.json`.
"""

import json
from pathlib import Path
from typing import Final

import e00521_stats as s  # ty: ignore[unresolved-import]
import e00553_score as e553  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00557-per-group-lag-recipe-choice-stored-forecasts"
METRICS: Final[tuple[str, ...]] = ("crps", "pin_q95", "pin_q99")
TESTS: Final[int] = 14
"""Lag-light minus lag-free, 7 lead bands, at the eight and at the six."""
OPTIONS: Final[dict[str, str]] = {
    "a_lag_light_everywhere": "RDWT-D15",
    "b_lag_free_everywhere": "E",
}
REFERENCES: Final[tuple[str, ...]] = ("A-pool", "L-1d", "manual_heuristic")
COUNTED: Final[tuple[tuple[str, str], ...]] = (("RDWT-D15", "E"),)
HEADLINE: Final[str] = e553.HEADLINE


def seedcols_names() -> tuple[str, ...]:
    """Return the arm names of the two L-1d seeds, which give the minimum detectable effect."""
    return ("L-1d-seed0", "L-1d-seed1")


def main() -> None:
    """Write the absolute scores and contrasts of the three options."""
    t = e553.table().with_columns(band=s.band_expr())
    for metric in METRICS:
        t = t.with_columns(
            pl.when(pl.col("group") == "mva_eight")
            .then(pl.col(f"RDWT-D15__{metric}"))
            .when(pl.col("group") == "signed_six")
            .then(pl.col(f"E__{metric}"))
            .otherwise(None)
            .alias(f"c_per_group__{metric}")
        )
    arms = ["RDWT-D15", "E", "c_per_group", *REFERENCES]
    # Keep only rows where every arm used below has a finite CRPS (as experiment 553 does).
    need = [f"{a}__crps" for a in ("RDWT-D15", "E", "A-pool", "L-1d", *seedcols_names())]
    t = t.with_columns(
        main16=pl.col("main16") & pl.all_horizontal(pl.col(c).is_finite() for c in need)
    )
    eight, six = t.filter(pl.col("group") == "mva_eight"), t.filter(pl.col("group") == "signed_six")
    assert eight["c_per_group__crps"].null_count() == 0
    assert six["c_per_group__crps"].null_count() == 0
    checks = {
        "rows": t.height,
        "rows with every arm finite": int(t["main16"].sum()),
        "tests": TESTS,
    }
    ROOT.mkdir(exist_ok=True, parents=True)
    (ROOT / "scores").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    seedcols = ["L-1d-seed0__crps", "L-1d-seed1__crps"]
    for name in ("eight", "six", "report", "fourteen"):
        part = (
            pl.concat([s.row_set(table=t, name="eight"), s.row_set(table=t, name="six")])
            if name == "fourteen"
            else s.row_set(table=t, name=name)
        )
        for slice_name, expr in e553.slices().items():
            sliced = part.filter(expr.fill_null(value=False))
            for band in (*s.BANDS, s.ALL, HEADLINE):
                sub = (
                    sliced.filter(pl.col("headline"))
                    if band == HEADLINE
                    else s.in_band(part=sliced, band=band)
                )
                if sub.height < 200:
                    continue
                m = e553.mde(sub=sub, cols=seedcols)
                for arm in arms:
                    for metric in METRICS:
                        col = f"{arm}__{metric}"
                        if col in sub.columns and not (name == "report" and arm == "c_per_group"):
                            v = sub.filter(pl.col(col).is_finite())[col]
                            absolute.append({"set": name, "slice": slice_name, "band": band, "arm": arm, "metric": metric, "n": v.len(), "value": float(v.mean())})  # fmt: skip
                if name == "report":
                    pairs = [("RDWT-D15", "E")]
                elif name == "fourteen":
                    pairs = [("RDWT-D15", "c_per_group"), ("E", "c_per_group"), ("RDWT-D15", "E")]
                else:
                    pairs = [
                        ("RDWT-D15", "E"),
                        ("RDWT-D15", "A-pool"),
                        ("E", "A-pool"),
                        ("c_per_group", "A-pool"),
                    ]
                for a, b in pairs:
                    for metric in METRICS:
                        d = sub.select("month", "day", d=pl.col(f"{a}__{metric}") - pl.col(f"{b}__{metric}")).filter(pl.col("d").is_finite())  # fmt: skip
                        if d.height < 200:
                            continue
                        st = s.block_stats(diff=d, rng=rng, tests=TESTS)
                        counted = (a, b) in COUNTED and name in ("eight", "six") and band in s.BANDS and slice_name == "all rows" and metric == "crps"  # fmt: skip
                        contrasts.append({"set": name, "slice": slice_name, "band": band, "metric": metric, "contrast": f"{a} - {b}", "counted": counted, "tests": TESTS, "mde": m, **st})  # fmt: skip
    pl.DataFrame(absolute).write_csv(ROOT / "scores" / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(ROOT / "scores" / "contrasts.csv")
    checks["counted contrasts"] = sum(1 for c in contrasts if c["counted"])
    assert checks["counted contrasts"] == TESTS
    (ROOT / "scores" / "checks.json").write_text(json.dumps(checks, indent=1))
    print(checks)


if __name__ == "__main__":
    main()

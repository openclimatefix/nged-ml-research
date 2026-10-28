# ruff: noqa: E501
"""Score experiment 373's arms from `work/meta_wide_D8.parquet` and write `work/scores.json`.

Run it with the folders of experiments 252, 264, and 354 on `PYTHONPATH`.
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from typing import Final

import e00354_lib as lib  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = ML_EXPERIMENTS_DATA_DIR / "00373-d8-meta-model-weights-worker-forecasts" / "work"
QUANTILES: Final = lib.QUANTILES
ARM_PREFIX: Final[dict[str, str]] = {
    "meta": "meta__", "meta_nolag": "meta_nolag__", "meta_nocal": "meta_nocal__",
    "single": "single__", "b50": "b50__", "gam": "gam__", "bg2": "bg2_",
    "W": "only_W__", "M": "only_M__", "AN": "only_AN__", "PT": "only_PT__", "ST": "only_ST__",
    "climatology": "climatology_", "manual": "manual_",
}  # fmt: skip
CONTRASTS: Final[list[tuple[str, str]]] = [
    ("meta", "single"), ("meta", "b50"), ("meta", "bg2"),
    ("meta", "meta_nolag"), ("meta", "meta_nocal"), ("meta", "gam"), ("single", "bg2"), ("b50", "bg2"),
]  # fmt: skip
SLICES: Final = (
    "all",
    "lead_0_24",
    "lead_24_72",
    "lead_72_168",
    "event_day",
    "other_day",
    "h1",
    "h2",
)
FIRST_HALF: Final = ("2025-10", "2025-11", "2025-12", "2026-01")


def mean(series: pl.Series) -> float:
    """Return the mean of a numeric series as a float."""
    return float(series.mean())  # ty: ignore[invalid-argument-type]


def main() -> None:
    """Write absolute scores, contrasts, tails, and checks."""
    wide = pl.read_parquet(WORK / "meta_wide_D8.parquet")
    actual = wide["actual"].to_numpy().astype(np.float64)
    lead = wide["lead_hours"]
    wide = wide.with_columns(
        *[
            pl.Series(f"s_{a}", lib.row_scores(actual=actual, frame=wide, prefix=p))
            for a, p in ARM_PREFIX.items()
        ],
        all=pl.lit(True),
        lead_0_24=lead <= 24,
        lead_24_72=(lead > 24) & (lead <= 72),
        lead_72_168=lead > 72,
        other_day=~pl.col("event_day"),
        h1=pl.col("month").is_in(list(FIRST_HALF)),
        h2=~pl.col("month").is_in(list(FIRST_HALF)),
    )
    wide = wide.with_columns(
        *[(pl.col(f"s_{x}") - pl.col(f"s_{y}")).alias(f"c_{x}__{y}") for x, y in CONTRASTS],
        *[
            (pl.col(f"{p}q50") - actual_err).abs().alias(f"mae_{a}")
            for a, p in ARM_PREFIX.items()
            for actual_err in [pl.col("actual")]
        ],
    )
    out: dict = {"rows": wide.height, "months": wide["month"].n_unique()}
    out["absolute"] = {
        flag: {a: mean(wide.filter(pl.col(flag))[f"s_{a}"]) for a in ARM_PREFIX} for flag in SLICES
    }
    out["event_rows"] = int(wide["event_day"].sum())
    out["contrast"] = {
        f"c_{x}__{y}": {
            flag: lib.contrast_both(wide=wide, column=f"c_{x}__{y}", flag=flag, site="D8")
            for flag in SLICES
        }
        for x, y in CONTRASTS
    }
    out["mae_median"] = {a: mean(wide[f"mae_{a}"]) for a in ARM_PREFIX}
    out["corr_median"] = {
        a: float(np.corrcoef(actual, wide[f"{p}q50"].to_numpy().astype(np.float64))[0, 1])
        for a, p in ARM_PREFIX.items()
    }
    out["tails"] = {a: lib.coverage_and_tails(frame=wide, prefix=p) for a, p in ARM_PREFIX.items()}
    out["pinball_tails"] = {
        a: {
            q: float(
                np.mean(
                    np.maximum(
                        lv * (actual - wide[f"{p}{q}"].to_numpy()),
                        (lv - 1) * (actual - wide[f"{p}{q}"].to_numpy()),
                    )
                )
            )
            for q, lv in (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99))
        }
        for a, p in ARM_PREFIX.items()
    }
    out["monthly"] = {
        r["month"]: {a: r[f"s_{a}"] for a in ARM_PREFIX}
        for r in wide.group_by("month")
        .agg(*[pl.col(f"s_{a}").mean() for a in ARM_PREFIX])
        .sort("month")
        .to_dicts()
    }
    out["checks"] = {
        "median_minus_blend_median_mean": {
            a: mean(wide[f"{p}q50"] - 0.5 * (wide["gam__q50"] + wide["bg2_q50"]))
            for a, p in ARM_PREFIX.items()
        },
        "corr_median_error_meta_vs_b50": float(
            np.corrcoef(
                actual - wide["meta__q50"].to_numpy(), actual - wide["b50__q50"].to_numpy()
            )[0, 1]
        ),
    }
    (WORK / "scores.json").write_text(json.dumps(out, indent=2, default=float))
    print("rows", out["rows"], "event rows", out["event_rows"])
    for flag in (
        "all",
        "h1",
        "h2",
        "lead_0_24",
        "lead_24_72",
        "lead_72_168",
        "event_day",
        "other_day",
    ):
        print(flag, {a: round(v, 3) for a, v in out["absolute"][flag].items()})
    for name, per in out["contrast"].items():
        print(
            name,
            {
                f: [round(x, 3) for x in per[f]["ci95"]]
                for f in ("all", "h1", "h2", "event_day", "other_day")
            },
        )
    print("mae", {a: round(v, 3) for a, v in out["mae_median"].items()})
    print("corr", {a: round(v, 4) for a, v in out["corr_median"].items()})
    for a in ("meta", "single", "b50", "bg2"):
        print(
            "tails",
            a,
            {k: round(v, 4) for k, v in out["tails"][a].items()},
            {k: round(v, 3) for k, v in out["pinball_tails"][a].items()},
        )
    print("checks", out["checks"])


if __name__ == "__main__":
    main()

"""Part B of experiment 395: winter, top-quartile, and cold-night re-score of stored monthly arms.

Reads experiment 381's `monthly/<arm>` forecasts and experiment 389's `S0M_raw` and `S1M_raw` (all
monthly refits, no 1.15 widening) and the GAM stack, and prints each arm minus `monthly/control`
by slice. Nothing is refitted. Usage: `... e00395_part_b.py` (build, then print).
"""
# ruff: noqa: E501

import sys
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = (
    DATA / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts" / "work"
)
E389: Final[Path] = (
    DATA / "00389-xgboost-tail-calibration-and-ensemble-spread-columns" / "work" / "arms"
)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
HEADLINE: Final[str] = "lead_days_1_10"
SIX: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
EIGHT: Final[list[str]] = ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
MONTHLY: Final[tuple[str, ...]] = (
    "control",
    "e7",
    "solar",
    "sunny",
    "hl1",
    "hl025",
    "win12",
    "leadw",
)


def frames(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the monthly arms, 389's S0M and S1M, and the stack, all raw, in MW."""
    out = {f"m_{a}": s381.arm_frame(regime="monthly", arm=a, series=series) for a in MONTHLY}
    for name in ("S0M_raw", "S1M_raw"):
        stored = pl.read_parquet(E389 / f"{name}.parquet").filter(
            pl.col("power_fcst_init_time") >= base.SCORED_FIRST
        )
        out[name] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    out["stack"] = base.stack_frames(series=series)["stack"]
    return out


def build() -> None:
    """Score the arms on their shared half-hours and write per-row tables."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    arms = frames(series=series)
    table, _, _ = s381.score_arms(arms=arms, series=series, ctx=ctx)
    table.write_parquet(WORK / "b_slice_table.parquet")
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        rows = metrics.scored_rows(
            forecast=kept, actuals=ctx["actuals"], series=series,
            thresholds=ctx["thresholds"], slice_table=table,
        )  # fmt: skip
        rows.write_parquet(WORK / f"b_rows_{name}.parquet")


def report() -> None:
    """Print the Part B tables."""
    pl.Config.set_tbl_rows(100)
    table = pl.read_parquet(WORK / "b_slice_table.parquet")
    names = [f"m_{a}" for a in MONTHLY] + ["S0M_raw", "S1M_raw", "stack"]
    arms = {
        n: pl.read_parquet(WORK / f"b_rows_{n}.parquet").join(
            table, on=KEYS, how="left", validate="1:1"
        )
        for n in names
    }
    top = pl.col("actual_points") >= pl.col("actual_points").quantile(0.75).over("time_series_id")
    slices = {
        "all days 1-10": pl.col(HEADLINE),
        "winter": pl.col(HEADLINE) & pl.col("winter"),
        "top quartile": pl.col(HEADLINE) & top,
        "cold nights": pl.col(HEADLINE) & pl.col("cold_nights"),
        "other (not winter)": pl.col(HEADLINE) & ~pl.col("winter"),
    }
    groups = {"six": SIX, "eight": EIGHT, "D11": ["D11"], "D14": ["D14"]}
    print("== B1: CRPS by arm, slice and group ==")
    for sname, cond in slices.items():
        for gname, labs in groups.items():
            vals = {
                n: round(a.filter(pl.col("label").is_in(labs)).filter(cond)["crps"].mean(), 3)  # ty: ignore[no-matching-overload]
                for n, a in arms.items()
            }
            print(sname, gname, vals)
    print("\n== B2: arm minus monthly/control (CRPS), 95% month bootstrap ==")
    for n in names:
        if n == "m_control":
            continue
        for sname, cond in slices.items():
            for gname in ("six", "eight", "D11", "D14"):
                where = pl.col("label").is_in(groups[gname]) & cond
                r = metrics.month_bootstrap_contrast(
                    treatment=arms[n], reference=arms["m_control"], metric="crps",
                    where=where, slice_table=table,
                )  # fmt: skip
                print(
                    f"{n:9s} {sname:19s} {gname:6s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f})"
                )
    print("\n== B3: share of the D11 and D14 gap to the stack closed vs control ==")
    for lab in ("D11", "D14"):
        c = arms["m_control"].filter(pl.col("label") == lab, pl.col(HEADLINE))["crps"].mean()
        s = arms["stack"].filter(pl.col("label") == lab, pl.col(HEADLINE))["crps"].mean()
        print(
            lab,
            f"control {c:.3f} stack {s:.3f}",
            {
                n: f"{(c - arms[n].filter(pl.col('label') == lab, pl.col(HEADLINE))['crps'].mean()) / (c - s):.0%}"  # ty: ignore[unsupported-operator]
                for n in names
                if n not in ("m_control", "stack")
            },
        )
    print("\n== B4: coverage at q01,q05,q95,q99 and cold-night q95, six ==")
    for n in names:
        a = arms[n].filter(pl.col("label").is_in(SIX), pl.col(HEADLINE))
        print(
            n,
            {
                c: round(a[c].mean(), 3)  # ty: ignore[no-matching-overload]
                for c in ("coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99")
            },
            "cold q95",
            round(a.filter(pl.col("cold_nights"))["coverage_q95"].mean(), 3),  # ty: ignore[no-matching-overload]
            "winter q95",
            round(a.filter(pl.col("winter"))["coverage_q95"].mean(), 3),  # ty: ignore[no-matching-overload]
        )
    print(
        "\nrows per slice (six):",
        {
            s: arms["m_control"].filter(pl.col("label").is_in(SIX)).filter(c).height
            for s, c in slices.items()
        },
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    {"build": build, "report": report}[sys.argv[1]]()

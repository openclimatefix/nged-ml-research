# ruff: noqa: E501
r"""Phase A: the short look. Reprints every figure of the plan's "What the short look found".

    PYTHONPATH=src:ml-experiments/00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back \
        uv run --no-sync python ml-experiments/00695-*/e00695_look.py

Every forecast figure is on the scored rows only: D8, leads 24 to 240 hours from the 12:00 UTC
origin, origins 2025-11-01 to 2026-06-30, valid times before 2026-07-01, with a finite metered
reading. A half-hour counts once per forecast that covers it. Fits no model. It prints no Gas A
series, only counts.
"""

import pickle

import numpy as np
import polars as pl
from e00695_common import (
    D8_CAPACITY_MVA,
    D8_SERIES,
    DATA_DIR,
    E677_PROTOCOL_ROWS,
    E694_STATE_PATH,
    ENGINE_MONTHS,
    FADE_MONTHS,
    QUANTILE_COLUMNS,
    SHARED_DIR,
    d8_truth,
    e694_flags,
    month_label,
    scored_window,
)
from studies.eval_protocol.metrics import MEDIAN_INDEX, quantile_score

LINES: list[str] = []


def say(text: str) -> None:
    """Print a line and keep it for `printed_look.txt`."""
    print(text)
    LINES.append(text)


def candidate_rows() -> pl.DataFrame:
    """Return the stored Candidate's normal-running rows at D8 on the decision window."""
    forecast = (
        pl.scan_parquet(SHARED_DIR / "primaries.parquet")
        .filter(pl.col("time_series_id") == D8_SERIES, pl.col("NRA_or_prevailing") == "NRA")
        .collect()
    )
    return scored_window(
        frame=forecast.join(d8_truth(), on="valid_time", how="left", validate="m:1")
    )


def score(*, rows: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's CRPS and absolute median error, in points of effective capacity."""
    scale = 100.0 / D8_CAPACITY_MVA
    actual = rows["power"].to_numpy().astype(np.float64) * scale
    quantiles = rows.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64) * scale
    return (
        quantile_score(actual=actual, quantiles=quantiles),
        np.abs(actual - quantiles[:, MEDIAN_INDEX]),
    )


def look_candidate() -> None:
    """Print the stored Candidate's score and its median error by tracked state."""
    rows = candidate_rows()
    finite = rows.filter(pl.col("power").is_not_null())
    say(
        f"Stored Candidate (NRA rows of primaries.parquet), D8, decision window: {rows.height} rows, "
        f"{rows.height - finite.height} with no metered reading, {finite.height} scored"
    )
    crps, mae = score(rows=finite)
    say(
        f"  CRPS {crps.mean():.2f} points of capacity, normalised MAE of the median {mae.mean():.2f}"
    )
    flags = e694_flags()
    joined = finite.join(flags, on="valid_time", how="left", validate="m:1").with_columns(
        month=month_label(valid_time=pl.col("valid_time")),
        error_points=(pl.col("q50") - pl.col("power")) * 100.0 / D8_CAPACITY_MVA,
    )
    say(
        f"  rows with no hindsight state: {joined['flag'].null_count()} (all are dropped from the tables)"
    )
    joined = joined.filter(pl.col("flag").is_not_null())
    engine = joined.filter(pl.col("month").is_in(ENGINE_MONTHS))
    for name, on in (("tracked-off", False), ("tracked-on", True)):
        part = engine.filter(pl.col("flag") == on)
        say(
            f"  {', '.join((ENGINE_MONTHS[0], ENGINE_MONTHS[-1]))} {name}: {part.height} rows, "
            f"mean (median forecast minus reading) {part['error_points'].mean():+.1f} points"
        )
    say("  share of scored rows with the smoothed flag on, and of all half-hours with a reading:")
    all_hh = flags.filter(pl.col("reading").is_not_null()).with_columns(
        month=month_label(valid_time=pl.col("valid_time"))
    )
    for group, months in (("engine months", ENGINE_MONTHS), ("fade months", FADE_MONTHS)):
        scored_share, all_share, filt_share = [], [], []
        for month in months:
            scored_share.append(float(joined.filter(pl.col("month") == month)["flag"].mean()))  # ty: ignore[invalid-argument-type]
            month_hh = all_hh.filter(pl.col("month") == month)
            all_share.append(float(month_hh["flag"].mean()))  # ty: ignore[invalid-argument-type]
            filt_share.append(float(month_hh["filtered"].mean()))  # ty: ignore[invalid-argument-type]
        say(
            f"  {group}: scored rows {min(scored_share):.0%} to {max(scored_share):.0%}; "
            f"all half-hours {min(all_share):.0%} to {max(all_share):.0%}; "
            f"filtered flag on all half-hours {min(filt_share):.0%} to {max(filt_share):.0%}"
        )
        say(
            "    by month, scored rows: "
            + ", ".join(f"{m} {s:.0%}" for m, s in zip(months, scored_share, strict=True))
        )


def look_step9a() -> None:
    """Print the stored step 9a and the stored sister arms on experiment 677's shared rows."""
    frame = (
        pl.scan_parquet(E677_PROTOCOL_ROWS)
        .select(
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            "label",
            "arm",
            "crps",
            "mae_median",
        )
        .collect()
    )
    window = scored_window(frame=frame)
    d8 = window.filter(pl.col("label") == "D8")
    table = d8.group_by("arm").agg(
        rows=pl.len(), crps=pl.col("crps").mean(), mae=pl.col("mae_median").mean()
    )
    say("Experiment 677's shared rows at D8, decision window (monthly refits):")
    for arm in (
        "Step 5w monthly",
        "Step 5wo monthly",
        "Step 8w monthly",
        "Experiment 667 (monthly refits)",
        "Climatology",
    ):
        row = table.filter(pl.col("arm") == arm).row(0, named=True)
        say(
            f"  {arm}: {row['rows']} rows, CRPS {row['crps']:.3f}, normalised MAE of the median {row['mae']:.2f}"
        )
    a0 = float(table.filter(pl.col("arm") == "Step 5w monthly")["crps"][0])
    say(
        f"  5.5% of step 9a's D8 CRPS: {0.055 * a0:.3f} points; 0.4 / 8 = {0.4 / 8:.3f}; 0.4 / 7.18 = {0.4 / 7.18:.3f}"
    )
    sister = (
        window.filter(pl.col("arm").is_in(["Step 5w monthly", "Step 5wo monthly"]))
        .group_by("label", "arm")
        .agg(crps=pl.col("crps").mean())
        .pivot(on="arm", index="label", values="crps")
        .with_columns(difference=(pl.col("Step 5w monthly") - pl.col("Step 5wo monthly")).abs())
        .sort("label")
    )
    say(
        "  absolute CRPS difference between step 9a (5w) and its sister without injected shifts (5wo), by Primary:"
    )
    say(
        "    "
        + ", ".join(f"{r['label']} {r['difference']:.3f}" for r in sister.iter_rows(named=True))
    )
    monthly = (
        d8.filter(pl.col("arm").is_in(["Step 5w monthly", "Step 5wo monthly"]))
        .with_columns(month=month_label(valid_time=pl.col("power_fcst_init_time")))
        .group_by("month", "arm")
        .agg(crps=pl.col("crps").mean())
        .pivot(on="arm", index="month", values="crps")
        .with_columns(difference=pl.col("Step 5w monthly") - pl.col("Step 5wo monthly"))
        .sort("month")
    )
    say(
        "  D8, monthly CRPS difference 5w minus 5wo: "
        + ", ".join(f"{r['month']} {r['difference']:+.3f}" for r in monthly.iter_rows(named=True))
        + f"; standard deviation {float(monthly['difference'].to_numpy().std(ddof=1)):.3f}"
    )


def look_arithmetic() -> None:
    """Print the capacity arithmetic and the engine record's running share."""
    say(
        f"Three units of 1.75 MW on {D8_CAPACITY_MVA} MVA: {3 * 1.75 / D8_CAPACITY_MVA * 100:.1f} points of capacity"
    )
    say(
        f"Gas A running share: {137 + 133 + 80} of {1704 + 137 + 133 + 80} half-hours = {(137 + 133 + 80) / 2054:.1%}"
    )
    for era in pickle.loads(E694_STATE_PATH.read_bytes()):
        say(
            f"Experiment 694 hindsight M2 fit, era {era['label']}: K {era['M2']['k']}, unit {era['M2']['u']} MW"
        )


def main() -> None:
    """Print every figure of the plan's look."""
    look_candidate()
    look_step9a()
    look_arithmetic()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "printed_look.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()

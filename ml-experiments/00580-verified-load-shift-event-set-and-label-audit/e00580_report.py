# ruff: noqa: D103, E501
"""Experiment 580, step 7: print the numbers and tables the write-up quotes (nothing else is quoted)."""

import e00580_verify as v  # ty: ignore[unresolved-import]
import polars as pl


def day(i: int) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime(2019, 9, 1, tzinfo=UTC) + timedelta(minutes=30 * int(i))).strftime("%Y-%m-%d")


def main() -> None:
    s = pl.read_parquet(v.WORK / "events_labelled_status.parquet")
    tr = pl.read_parquet(v.WORK / "events_verified.parquet")
    bins = [
        ("at least 7 days", pl.col("label_length_days") >= 7),
        (
            "1 to under 7 days",
            (pl.col("label_length_days") >= 1) & (pl.col("label_length_days") < 7),
        ),
        (
            "6 hours to under 1 day",
            (pl.col("label_length_days") >= 0.25) & (pl.col("label_length_days") < 1),
        ),
        ("under 6 hours", pl.col("label_length_days") < 0.25),
        ("all", pl.lit(True)),
    ]
    classes = [
        "verified transfer",
        "wrong dates or sign",
        "unpaired step (possible level change)",
        "rejected",
        "too short to check",
    ]
    print(
        "| Label length | Events | Verified transfer | Telemetry shows a different transfer (wrong dates or sign) | Step at one Primary only | Rejected | Too short to check |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, m in bins:
        d = s.filter(m)
        print(
            f"| {name} | {d.height} | "
            + " | ".join(str(d.filter(pl.col("final_class") == c).height) for c in classes)
            + " |"
        )
    print()
    print("| Event | Primary | Label | Labelled dates | Fitted dates | Fitted depth | Class |")
    print("|---|---|---|---|---|---|---|")
    for r in s.filter(pl.col("label_length_days") >= 7).sort("event_id").iter_rows(named=True):
        fd = f"{r['fit_depth_mw']:+.2f} MW" if r["fit_depth_mw"] == r["fit_depth_mw"] else "no data"
        fits = f"{r['fit_onset'][:10]} to {r['fit_end'][:10]}" if r["fit_onset"] else "no data"
        cls = r["final_class"].replace(" (possible level change)", "")
        print(
            f"| {r['event_id']} | {r['label']} | {r['label_depth_mw']:+.2f} MW | {day(r['label_onset_idx'])} to {day(r['label_end_idx'])} | {fits} | {fd} | {cls} |"
        )
    print()
    print(
        "| Transfer | Primaries | Fitted dates | Length (days) | Fitted depth at each | Hand check |"
    )
    print("|---|---|---|---|---|---|")
    for r in tr.filter(pl.col("in_label_period")).iter_rows(named=True):
        print(
            f"| {r['transfer_id']} | {r['primary_a']} and {r['primary_b']} | {r['onset_a'][:10]} to {r['end_a'][:10]} | {r['length_days']:.0f} | {r['depth_a_mw']:+.2f} and {r['depth_b_mw']:+.2f} MW | {r['hand_check']} |"
        )
    print()
    print(
        "transfers",
        tr.height,
        "in label period",
        tr.filter(pl.col("in_label_period")).height,
        "labelled",
        tr.filter(pl.col("source") == "labelled").height,
        "p99 both",
        int(tr["passes_p99_at_both"].sum()),
    )
    long = s.filter(pl.col("label_length_days") >= 7)
    print(
        "median label/fit depth, long events with a fit of the labelled sign:",
        long.filter(pl.col("label_depth_over_fit") > 0)["label_depth_over_fit"].median(),
        "n",
        long.filter(pl.col("label_depth_over_fit") > 0).height,
    )
    print(
        "noise p95 range (3-day, share of capacity):",
        float(s.filter(pl.col("window_half_hours") == 144)["noise_p95_share"].min()),  # ty: ignore[invalid-argument-type]
        float(s.filter(pl.col("window_half_hours") == 144)["noise_p95_share"].max()),  # ty: ignore[invalid-argument-type]
    )
    print(
        "noise p99 range:",
        float(s.filter(pl.col("window_half_hours") == 144)["noise_p99_share"].min()),  # ty: ignore[invalid-argument-type]
        float(s.filter(pl.col("window_half_hours") == 144)["noise_p99_share"].max()),  # ty: ignore[invalid-argument-type]
    )
    print(pl.read_csv(v.WORK / "recompute_verified_pair.csv"))
    print(
        pl.read_parquet(v.WORK / "injected_event_bank.parquet")
        .group_by("month")
        .len()
        .sort("month")
        .rows()
    )


if __name__ == "__main__":
    main()

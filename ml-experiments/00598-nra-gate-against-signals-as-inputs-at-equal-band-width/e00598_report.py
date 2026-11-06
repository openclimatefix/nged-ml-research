# ruff: noqa: E501, D103, ANN001
"""Experiment 598: print every number the write-up quotes, and test the pre-registered predictions.

Reads the tables of `e00598_score.py` and `e00597_score.py` and writes `report.txt` beside them.

Usage: python e00598_report.py [--suffix SUFFIX]
"""

import argparse
import contextlib
import io

import e00598_score as score  # ty: ignore[unresolved-import]
import polars as pl

pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_tbl_width_chars(240)
BAND = score.BAND
LONG = "15 and 30 days"
SYSTEM_ORDER = ["A1", "R", "W", "G4T", "S", "S-rs", "S-g", "S-g-rs", "S-p", "S-p-rs"]
V597 = score.k597.WORK / "a1"


def read(out, name: str) -> pl.DataFrame:
    return pl.read_csv(out / name, infer_schema_length=None)


def order(df: pl.DataFrame) -> pl.DataFrame:
    rank = {s: i for i, s in enumerate(SYSTEM_ORDER)}
    return (
        df.with_columns(_r=pl.col("system").replace_strict(rank, default=99))
        .sort("_r", maintain_order=True)
        .drop("_r")
    )


def run(out) -> None:
    factors = read(out, "factors.csv")
    print("== Width factors (one per fold month and seed; W has no seed)")
    print(
        factors.pivot(on="month", index=["system", "seed"], values="factor").sort("system", "seed")
    )
    ef = read(out, "event_free.csv")
    bank = read(out, "bank_event_phase.csv")
    summ = read(out, "bank_summary.csv")

    print(
        "\n== Reproduction of experiment 597's stored A1 + G4 (R) on the injected bank, cover90 by event"
    )
    old = pl.read_csv(V597 / "bank_event_phase.csv", infer_schema_length=None).filter(
        pl.col("system") == "G4"
    )
    new = bank.filter(pl.col("system") == "R")
    j = new.select(
        "month", "pass", "primary", "phase", "band", pl.col("cover90").alias("new")
    ).join(
        old.select("month", "pass", "primary", "phase", "band", pl.col("cover90").alias("old")),
        on=["month", "pass", "primary", "phase", "band"],
    )
    print(
        f"events compared {j.height}, largest |difference| in cover90 {float((j['new'] - j['old']).abs().max()):.6f}"  # ty: ignore[invalid-argument-type]
    )

    for subset in ("all months", "first fold pair"):
        for grp in ("both groups", "MVA-only eight", "signed-MW six"):
            b = summ.filter(
                (pl.col("subset") == subset)
                & (pl.col("group") == grp)
                & (pl.col("length_class") == LONG)
                & (pl.col("band") == BAND)
            )
            print(f"\n== Injected bank: {LONG}, {grp}, {BAND}, {subset}, by phase and system")
            print(order(b).sort("phase", maintain_order=True).select("system", "phase", "n_events", "cover90", "cover90_lo", "cover90_hi", "nra_side95", "nra_side95_lo", "nra_side95_hi", "cover98", "nra_side99", "follow_q50", "follow_q50_lo", "follow_q50_hi", "width90", *score.TAIL, "crps"))  # fmt: skip
    b = summ.filter(
        (pl.col("subset") == "all months")
        & (pl.col("group") == "both groups")
        & (pl.col("length_class") == "all lengths")
        & (pl.col("band") == BAND)
    )
    print("\n== Injected bank: all event lengths (3, 7, 15, 30 days), both groups, 24-120 h")
    print(
        order(b).select(
            "system", "phase", "n_events", "cover90", "nra_side95", "follow_q50", "width90"
        )
    )
    b = summ.filter(
        (pl.col("subset") == "all months")
        & (pl.col("group") == "both groups")
        & (pl.col("length_class") == LONG)
        & (pl.col("band") == "0-24 h")
    )
    print("\n== Injected bank: 15 and 30 day events, both groups, 0-24 h")
    print(
        order(b).select(
            "system", "phase", "n_events", "cover90", "nra_side95", "follow_q50", "width90"
        )
    )

    print(
        "\n== Event-free rows: CRPS, coverage, width, and tail pinball (points of capacity) per metering group and band"
    )
    print(
        order(ef)
        .select("system", "group", "band", "n_rows", "cover90", "width90", "crps", *score.TAIL)
        .sort("group", "band", maintain_order=True)
    )
    print(
        "\n== Event-free contrasts of CRPS (points of capacity), month-block interval at the Bonferroni level of",
        score.N_COUNTED,
        "tests",
    )
    cols = [x for x in ef.columns if x.startswith(("crps_minus", "pin05_minus", "pin95_minus"))]
    print(
        ef.filter(pl.col("system").is_in(["S", "S-g", "S-p", "R", "W", "G4T"]))
        .select("system", "group", "band", *cols)
        .sort("group", "band", "system")
    )

    print("\n== Real December pair (descriptive), per phase")
    rp = read(out, "real_event_phase.csv")
    print(
        order(rp.filter(pl.col("band") == BAND))
        .select("event", "phase", "system", "n_rows", "cover90", "nra_side95", "width90", "crps")
        .sort("event", "phase", maintain_order=True)
    )

    print(
        f"\n== Counted contrasts (Bonferroni level 0.05 / {score.N_COUNTED} = {score.ALPHA:.4f}; MDE = half the interval width)"
    )
    try:
        print(read(out, "contrasts.csv"))
        print(
            "\n== Predictions P1 to P5 (P1 outcome bands on the gap R - S: under 0.01 passes, 0.01 to 0.03 inconclusive, 0.03 or more fails)"
        )
        print(read(out, "verdicts.csv"))
    except FileNotFoundError:
        print("no contrasts: arm S has no bank output yet")
    print(
        "\n== Stop-rule input: gap R - S in day-4-to-end coverage, first fold pair (2025-10 and 2025-12), 24-120 h, 15 and 30 day events"
    )
    fp = summ.filter(
        (pl.col("subset") == "first fold pair")
        & (pl.col("group") == "both groups")
        & (pl.col("length_class") == LONG)
        & (pl.col("band") == BAND)
        & (pl.col("phase") == "days4-end")
    )
    d = {r["system"]: r["cover90"] for r in fp.iter_rows(named=True)}
    if "R" in d and "S" in d:
        gap = d["R"] - d["S"]
        print(
            f"R {d['R']:.4f}, S {d['S']:.4f}, gap {gap:.4f} -> {'pass' if gap < score.OUTCOME_PASS else 'inconclusive' if gap < score.OUTCOME_FAIL else 'fail: run S-g only next'}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="")
    args = p.parse_args()
    out = score.ROOT / f"score{args.suffix}"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run(out)
    text = buf.getvalue()
    (out / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

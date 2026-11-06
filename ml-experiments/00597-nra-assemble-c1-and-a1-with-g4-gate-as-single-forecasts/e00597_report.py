# ruff: noqa: E501, D103
"""Experiment 597: print every number the write-up quotes, and test the pre-registered predictions.

Reads the tables of `e00597_score.py` for arms `c1` and `a1` and writes `report.txt` (and prints it).
"""

import contextlib
import io

import e00583_score as s  # ty: ignore[unresolved-import]
import e00597_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_tbl_width_chars(220)
s.N_COUNTED = 6
BAND = "24-120 h"
LONG = "15 and 30 days"
SYS = {"c1": "C1", "a1": "A1"}


def bank(arm: str) -> pl.DataFrame:
    return pl.read_csv(k.WORK / arm / "bank_summary.csv")


def pick(
    *, b: pl.DataFrame, group: str = "both groups", cls: str = LONG, band: str = BAND
) -> pl.DataFrame:
    return b.filter(
        (pl.col("group") == group) & (pl.col("length_class") == cls) & (pl.col("band") == band)
    )


def main() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run()
    text = buf.getvalue()
    (k.WORK / "report.txt").write_text(text)
    print(text)


def run() -> None:
    old = pl.read_csv(k.V592 / "bank_summary.csv")
    new_c1 = bank("c1")
    f_old = pick(b=old).filter((pl.col("arm") == "G4") & (pl.col("phase") == "days4-end"))
    f_new = pick(b=new_c1).filter((pl.col("system") == "G4") & (pl.col("phase") == "days4-end"))
    diff = abs(float(f_new["cover90"][0]) - float(f_old["cover90"][0]))
    print(
        "== Reproduction of experiment 592 (C1 + G4, injected 15 and 30 day events, day 4 to end, 24-120 h, both groups)"
    )
    print(
        f"this experiment {f_new['cover90'][0]:.5f} ({f_new['cover90_lo'][0]:.4f} to {f_new['cover90_hi'][0]:.4f}), experiment 592 {f_old['cover90'][0]:.5f}, |difference| {diff:.6f}, tolerance 0.002, events {f_new['n_events'][0]}"
    )
    for arm in ("c1", "a1"):
        print(
            arm,
            "gate state equals experiment 592's stored state:",
            (k.WORK / arm / "gate_check.json").read_text(),
        )

    for arm in ("c1", "a1"):
        b = bank(arm)
        print(
            f"\n== {SYS[arm]}: injected bank, 15 and 30 day events, both groups, {BAND}, by phase"
        )
        print(
            pick(b=b).sort("system", "phase")
            .select("system", "phase", "n_events", "cover90", "cover90_lo", "cover90_hi", "nra_side95", "nra_side95_lo", "nra_side95_hi", "cover98", "nra_side99", "follow_q50", "follow_q50_lo", "follow_q50_hi", "follow_safety", "pin01", "pin05", "pin95", "pin99", "width90", "crps")
        )  # fmt: skip
        for grp in k.GROUPS_SCORED:
            print(f"\n-- {SYS[arm]}, {k.GROUP_NAME[grp]}")
            print(
                pick(b=b, group=grp).sort("system", "phase")
                .select("system", "phase", "n_events", "cover90", "cover90_lo", "cover90_hi", "nra_side95", "follow_q50", "follow_safety", "pin05", "pin95", "width90")
            )  # fmt: skip
        print(f"\n-- {SYS[arm]}, 0-24 h band, both groups")
        print(
            pick(b=b, band="0-24 h").sort("system", "phase")
            .select("system", "phase", "n_events", "cover90", "nra_side95", "follow_q50", "pin05", "pin95", "width90")
        )  # fmt: skip
        print(f"\n-- {SYS[arm]}, by depth (day 4 to end and 0-7 days after), both groups, {BAND}")
        for d in ("0.15", "0.30", "0.50"):
            print(
                pick(b=b, cls=f"{LONG}, depth {d}").filter(pl.col("phase").is_in(["days4-end", "after0-7"])).sort("system", "phase")
                .select(pl.lit(d).alias("depth"), "system", "phase", "n_events", "cover90", "nra_side95", "follow_q50", "follow_safety")
            )  # fmt: skip

        print(f"\n== {SYS[arm]}: false triggers per Primary-year on event-free rows")
        print(
            pl.read_csv(k.WORK / arm / "false_triggers.csv").filter(
                pl.col("group").is_in(
                    ["all 16", "MVA-only eight", "signed-MW six", "all with a neighbour"]
                )
            )
        )
        print(f"\n== {SYS[arm]}: gate onset and release, injected 15 and 30 day events (G4)")
        tr = pl.read_csv(k.WORK / arm / "trigger_times.csv").filter(
            (pl.col("gate") == "G4")
            & (pl.col("source") != "verified")
            & (pl.col("length_days") >= 15)
        )
        print(
            tr.select(
                pl.len().alias("events"),
                pl.col("triggered").mean().alias("detection_rate"),
                pl.col("days_onset_to_trigger").median().alias("median_onset_delay_days"),
                pl.col("days_end_to_snap_back").median().alias("median_release_delay_days"),
            )
        )

        print(
            f"\n== {SYS[arm]}: event-free rows, CRPS (points of capacity), coverage of the observed power, pinball loss"
        )
        ef = pl.read_csv(k.WORK / arm / "event_free.csv")
        print(
            ef.select(
                "system",
                "group",
                "band",
                "n_rows",
                "share_rows_wide",
                "crps",
                "cover90",
                "pin05",
                "pin95",
                "width90",
                "crps_minus_plain",
                "crps_lo",
                "crps_hi",
                "pin05_minus_plain",
                "pin05_lo",
                "pin05_hi",
                "pin05_mde",
                "pin95_minus_plain",
                "pin95_lo",
                "pin95_hi",
                "pin95_mde",
            ).sort("band", "group", "system")
        )

        print(f"\n== {SYS[arm]}: verified December pair (descriptive, 2 Primary-events), {BAND}")
        rp = pl.read_csv(k.WORK / arm / "real_event_phase.csv").filter(
            pl.col("event").str.contains("2025-12-29") & (pl.col("band") == BAND)
        )
        print(
            rp.select(
                "event",
                "phase",
                "system",
                "n_rows",
                "cover90",
                "nra_side95",
                "share_origin_days_wide",
                "width90",
                "crps",
            ).sort("event", "phase", "system")
        )
        old_r = pl.read_csv(k.V592 / "real_event_phase.csv").filter(
            (pl.col("gate") == "G4")
            & pl.col("event").str.contains("2025-12-29")
            & (pl.col("band") == BAND)
        )
        j = rp.filter(pl.col("system") == "G4").join(old_r, on=["event", "phase"], suffix="_592")
        print(
            "largest |difference| from experiment 592's G4 coverage, C1 only:",
            float((j["cover90"] - j["cover90_592"]).abs().max())
            if arm == "c1"
            else "n/a (A1 is not in 592)",
        )

    print(
        "\n== Cross-arm contrast: A1 + G4 minus C1 + G4, event-free CRPS (points of capacity), month-block interval at Bonferroni level, counted tests 6"
    )
    z1, z2 = (
        np.load(k.WORK / "c1" / "event_free_rows.npz"),
        np.load(k.WORK / "a1" / "event_free_rows.npz"),
    )
    d1 = pl.DataFrame({n: z1[n] for n in z1.files}).with_columns(
        pl.col("group").cast(pl.String), pl.col("month").cast(pl.String)
    )
    d2 = pl.DataFrame({n: z2[n] for n in z2.files}).with_columns(
        pl.col("group").cast(pl.String), pl.col("month").cast(pl.String)
    )
    jn = d1.join(d2, on=["prim", "origin_idx", "idx"], suffix="_a1")
    print("rows in C1 table", d1.height, "A1 table", d2.height, "shared", jn.height)
    for grp in k.GROUPS_SCORED:
        for band, name in ((1, BAND), (0, "0-24 h")):
            sub = jn.filter(pl.col("free") & (pl.col("group") == grp) & (pl.col("band") == band))
            months = sorted(sub["month"].unique().to_list())
            dm = np.array(
                [
                    (
                        sub.filter(pl.col("month") == m)["crps_g4_a1"]
                        - sub.filter(pl.col("month") == m)["crps_g4"]
                    ).mean()
                    * 100
                    for m in months
                ]
            )
            lo, hi = s.bonferroni_interval(diffs=dm)
            print(
                f"{k.GROUP_NAME[grp]}, {name}: {dm.mean():+.4f} ({lo:+.4f} to {hi:+.4f}), minimum detectable effect {(hi - lo) / 2:.4f}, n={sub.height}, counted={band == 1}"
            )
    print(
        "Counted tests: 6 = (C1 + G4 minus C1, A1 + G4 minus A1) x (q05, q95 pinball, both groups pooled) + (A1 + G4 minus C1 + G4) CRPS x 2 metering groups, all at 24-120 h."
    )

    print("\n== Pre-registered checks")
    for arm in ("c1", "a1"):
        b = bank(arm)
        g4 = pick(b=b).filter(pl.col("system") == "G4")
        ca = g4.filter(pl.col("phase") == "after0-7")
        c4 = g4.filter(pl.col("phase") == "days4-end")
        print(
            f"{SYS[arm]} + G4: day 4 to end {c4['cover90'][0]:.3f} ({c4['cover90_lo'][0]:.3f} to {c4['cover90_hi'][0]:.3f}); 0-7 days after {ca['cover90'][0]:.3f} ({ca['cover90_lo'][0]:.3f} to {ca['cover90_hi'][0]:.3f}); safe side after {ca['nra_side95'][0]:.3f}; follow day 4 to end {c4['follow_q50'][0]:.3f}"
        )


if __name__ == "__main__":
    main()

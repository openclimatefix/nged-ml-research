# ruff: noqa: E501
"""Experiment 674, step 3: print every number the write-up quotes.

PYTHONPATH=ml-experiments/00087-*:ml-experiments/00506-*:ml-experiments/00508-*:ml-experiments/00674-* \
        uv run --no-sync python ml-experiments/00674-*/e00674_report.py | tee ml-experiments/00674-*/printed_report.txt
"""

from pathlib import Path

import e00506_data as data  # ty: ignore[unresolved-import]
import e00674_features as feat  # ty: ignore[unresolved-import]
import polars as pl

HINT = {
    "D8": "Yes",
    "D12": "Maybe",
    "D22": "Maybe",
    "D2": "No",
    "D3": "No",
    "D4": "No",
    "D19": "No",
    "D21": "No",
}
ARMS = ("C", "B", "A", "A_flat", "A_steep")


def show(df: pl.DataFrame) -> None:
    """Print a frame as compact pipe-separated lines."""
    print(" | ".join(df.columns))
    for row in df.iter_rows():
        print(" | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row))


pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)


def pct(col: str) -> list[pl.Expr]:
    """Return est, lo, hi of a metric as rounded columns."""
    return [pl.col(f"{col}_{k}").round(3) for k in ("est", "lo", "hi")]


def known() -> None:
    """Print the known-answer tables."""
    m = pl.read_parquet(feat.WORK / "known_answer_metrics.parquet")
    for title, flt in (
        ("ALL levels, subset all", (pl.col("subset") == "all") & (pl.col("level") == "all")),
        ("rbin 1.5-3 by stratum", (pl.col("rbin") == "1.5-3") & (pl.col("level") == "all")),
        (
            "pooled_below_3 by subset",
            (pl.col("rbin") == "pooled_below_3") & (pl.col("level") == "all"),
        ),
        (
            "pooled_below_3, level 0 (real reverse flow)",
            (pl.col("rbin") == "pooled_below_3")
            & (pl.col("level") == "0.0")
            & (pl.col("subset") == "all"),
        ),
        (
            "pooled_below_3, by level, subset all",
            (pl.col("rbin") == "pooled_below_3")
            & (pl.col("subset") == "all")
            & (pl.col("level") != "all"),
        ),
    ):
        print(f"\n=== {title}")
        sub = m.filter(flt, pl.col("arm").is_in(ARMS)).sort("arm", "rbin", "subset", "level")
        show(
            sub.select(
                "arm",
                "rbin",
                "subset",
                "level",
                *pct("recall"),
                *pct("false_reverse"),
                pl.col("false_reverse_all_est").round(3),
                pl.col("unknown_est").round(3),
                pl.col("not_scorable_est").round(3),
                pl.col("recall_den"),
                pl.col("false_reverse_den"),
            )
        )
    days = pl.read_parquet(feat.WORK / "known_answer_days.parquet")
    print("\n=== real reverse-flow days at level 0, D5+D10 (+D9), recall by arm")
    real = days.filter(pl.col("level") == 0.0)
    for arm in ARMS:
        c = real.filter(pl.col("crossing"))
        print(
            arm,
            "crossing",
            c.height,
            "flagged",
            c.filter(pl.col(f"cls_{arm}") == "reverse").height,
            c.group_by(f"cls_{arm}").len().to_dicts(),
        )
    print("\n=== P-check per arm at rbin 1.5-3 pooled (point estimates)")
    p = m.filter(
        (pl.col("rbin") == "1.5-3") & (pl.col("subset") == "all") & (pl.col("level") == "all")
    )
    show(
        p.select(
            "arm",
            "recall_est",
            "false_reverse_est",
            "recall_lo",
            "recall_hi",
            "false_reverse_lo",
            "false_reverse_hi",
        )
    )
    print("\n=== fallbacks used in leave-one-out thresholds")
    import json

    th = json.loads((feat.WORK / "thresholds_leave_one_out.json").read_text())
    n_fb = sum(
        1
        for s in th.values()
        for arm in s.values()
        for st in arm.values()
        for v in st.values()
        if v[3]
    )
    print("cells with fallback:", n_fb)
    print("\n=== counts per world: days, crossing days, by site and level (all April-September)")
    show(
        days.group_by("site", "level")
        .agg(
            n=pl.len(),
            crossing=pl.col("crossing").sum(),
            scorable=(pl.col("cls_C") != "not_scorable").sum(),
        )
        .sort("site", "level")
    )


def eight() -> None:
    """Print the eight's label counts, hint agreement, and the counter-examples."""
    for tag in ("gate", "nogate"):
        d = pl.read_parquet(feat.WORK / f"eight_days_{tag}.parquet")
        print(f"\n=== EIGHT ({tag}): April-September label counts per site and arm")
        for arm in ARMS:
            t = (
                d.group_by("site", f"cls_{arm}")
                .len()
                .pivot(on=f"cls_{arm}", index="site", values="len")
                .fill_null(0)
                .sort("site")
            )
            print(f"-- arm {arm}")
            show(t)
        sunny = d.filter(pl.col("month").is_in([4, 5, 6, 7, 8]), pl.col("k_mid") >= feat.SUNNY_K)
        print(
            f"-- sunny April-August days ({tag}): share of days labelled reverse / unknown, per site and arm"
        )
        rows = []
        for site in sorted(HINT):
            s = sunny.filter(pl.col("site") == site)
            row = {"site": site, "hint": HINT[site], "n_sunny": s.height}
            for arm in ARMS:
                row[f"{arm}_rev"] = round(
                    s.filter(pl.col(f"cls_{arm}") == "reverse").height / max(s.height, 1), 3
                )
            row["unknown"] = round(
                s.filter(pl.col("cls_C") == "unknown").height / max(s.height, 1), 3
            )
            row["not_scorable"] = round(
                s.filter(pl.col("cls_C") == "not_scorable").height / max(s.height, 1), 3
            )
            known_only = s.filter(pl.col("cls_A") != "unknown")
            row["A_rev_of_known"] = round(
                known_only.filter(pl.col("cls_A") == "reverse").height / max(known_only.height, 1),
                3,
            )
            rows.append(row)
        show(pl.DataFrame(rows))
        print(f"-- flagged days by stratum ({tag}), arms C and A")
        for arm in ("C", "A"):
            print(arm)
            show(
                d.filter(pl.col(f"cls_{arm}") == "reverse")
                .group_by("site", "stratum")
                .len()
                .sort("site", "stratum")
                .pivot(on="stratum", index="site", values="len")
                .fill_null(0)
            )
    d = pl.read_parquet(feat.WORK / "eight_days_gate.parquet")
    print("\n=== D8 engine overlap (experiment 508's 71 dip days)")
    eng = pl.read_parquet(
        Path(
            str(feat.WORK).replace(
                "00674-mva-sign-labels-soft-and-hard-pv-prior-day-labeller",
                "00508-mva-shoulder-hours-and-d8-d22-dips-solar-or-load-shift",
            )
        )
        / "partB_engine_d8.parquet"
    )
    d8 = d.filter(pl.col("site") == "D8").join(
        eng.select("date", "engine_share", "coincident"), on="date", how="left"
    )
    on_dip = d8.filter(pl.col("engine_share").is_not_null())
    print("dip days in the April-September table:", on_dip.height)
    for arm in ARMS:
        print(
            arm,
            on_dip.group_by(f"cls_{arm}").len().to_dicts(),
            "| engine-flag days labelled reverse:",
            on_dip.filter(pl.col("coincident"), pl.col(f"cls_{arm}") == "reverse").height,
        )
    print("\n=== D22 dip days of experiment 508")
    shape = pl.read_parquet(
        Path(
            str(feat.WORK).replace(
                "00674-mva-sign-labels-soft-and-hard-pv-prior-day-labeller",
                "00508-mva-shoulder-hours-and-d8-d22-dips-solar-or-load-shift",
            )
        )
        / "partB_shape.parquet"
    )
    d22 = d.filter(pl.col("site") == "D22").join(
        shape.filter((pl.col("site") == "D22") & (pl.col("kind") == "dip")).select("date").unique(),
        on="date",
        how="inner",
    )
    print("D22 dip days in table:", d22.height)
    for arm in ARMS:
        print(arm, d22.group_by(f"cls_{arm}").len().to_dicts())
    print("\n=== D12 winter dips (all months, thresholds are April-September)")
    a = pl.read_parquet(feat.WORK / "eight_days_allmonths.parquet").filter(pl.col("site") == "D12")
    dips = a.filter(pl.col("month").is_in([10, 11, 12, 1, 2]), pl.col("mid_min") < 6.9)
    print("dips (mid_min < 6.9 points):", dips.height)
    for arm in ARMS:
        print(arm, dips.group_by(f"cls_{arm}").len().to_dicts())
    print(
        "\n=== D8 flagged-day sky split (arm C vs A) already printed above; D12 ratio median",
        d.filter(pl.col("site") == "D12")["r"].median(),
    )
    print(
        "\nnight import (points) per site:",
        d.group_by("site").agg(pl.col("night_import").first()).sort("site").to_dicts(),
    )


if __name__ == "__main__":
    print("thresholds frozen before the eight were built; data cut", feat.CUT_DATE)
    known()
    eight()
    _ = data

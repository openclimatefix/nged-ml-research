# ruff: noqa: ANN201, D103, E501, PERF401
"""Scores of the issue-time forecasts against L-1d, with lead counted from the issue time.

Pre-registered counted contrasts (Bonferroni count 32 = 4 issue times x 4 lead bands x 2 row sets):
R-DW minus L-1d at the 12, 18, 00, and 06 UTC issue times, in the lead bands 0-6, 6-12, 12-24, and
24-48 h, at the eight and the six. Every other contrast is report-only. The pre-registered P1 is the
0-12 h pooled contrast at 12 UTC at the eight.
"""

import json
import math
import os
from pathlib import Path

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

DD = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
OUT = DD / "00555-issue-time-lags-at-06-12-18-utc-lag-light-demand-network"
K = ["primary", "origin", "target"]
STEPS = {"09": 0, "12": 6, "18": 18, "00": 30, "06": 42}
"""Issue hour -> half-hours after the 09:00 UTC origin (hour 09 is the production origin, reference only)."""
ARMS = ("R-DW", "R-DROP", "R9")
COUNTED_BANDS = ("0-6h", "6-12h", "12-24h", "24-48h")
TESTS = 32
POOLED = {"0-12h": (0.0, 12.0)}


def table(*, tag: str) -> pl.DataFrame:
    base = s.load().select(
        *K, "group", "cls", "lead_hours", "main16", "month", "day", "cold",
        "L-1d__crps", "L-1d-seed0__crps", "L-1d-seed1__crps",
    )  # fmt: skip
    mine = pl.concat(
        [
            pl.read_parquet(
                OUT / f"work_i{tag}/scores/rows_{m}.parquet",
                columns=[*K, *[f"{a}__crps" for a in ARMS], "A-pool__crps"],
            )
            for m in s.MONTHS
        ]
    )
    n0 = base.height
    t = base.join(mine, on=K, how="left", validate="1:1", maintain_order="left")
    assert t.height == n0
    need = [f"{a}__crps" for a in ARMS]
    finite = pl.all_horizontal([pl.col(c).is_finite().fill_null(False) for c in need])
    step_hours = STEPS[tag] / 2.0
    t = t.with_columns(
        lead_origin=pl.col("lead_hours"),
        lead_hours=pl.col("lead_hours") - step_hours,
        main16=pl.col("main16") & finite,
    )
    kept = t.filter(pl.col("main16"))
    # every kept row has its valid time after the issue time, and every dropped-by-issue row has no forecast
    assert (kept["lead_hours"] > 0).all(), tag
    assert (t.filter(pl.col("lead_hours") <= 0)["main16"] == False).all(), tag  # noqa: E712
    return t.with_columns(
        band=s.band_expr(), pooled=(pl.col("lead_hours") >= 0) & (pl.col("lead_hours") < 12)
    )


def mde(part: pl.DataFrame) -> float:
    seeds = ["L-1d-seed0__crps", "L-1d-seed1__crps"]
    m = (
        part.group_by("month")
        .agg([pl.col(c).mean() for c in seeds])
        .sort("month")
        .select(seeds)
        .to_numpy()
    )
    f, k = m.shape
    r = m - m.mean(axis=1, keepdims=True)
    return 1.96 * math.sqrt(2) * math.sqrt(float((r**2).sum()) / (f * (k - 1))) / math.sqrt(k)


def main():
    rng = np.random.default_rng(0)
    absolute, contrasts, checks = [], [], {}
    for tag in STEPS:
        t = table(tag=tag)
        checks[tag] = {
            "rows_kept": int(t["main16"].sum()),
            "min_lead_hours": float(t.filter(pl.col("main16"))["lead_hours"].min()),  # ty: ignore[invalid-argument-type]
            "folds": sorted(t.filter(pl.col("main16"))["month"].unique().to_list()),
        }
        for name in ("eight", "six", "report"):
            part = s.row_set(table=t, name=name)
            views = {b: s.in_band(part=part, band=b) for b in (*s.BANDS, s.ALL)}
            views["0-12h"] = part.filter(pl.col("pooled"))
            for band, sub in views.items():
                if sub.height == 0:
                    continue
                m = mde(sub)
                for col in ("L-1d", *ARMS):
                    absolute.append(
                        {
                            "issue": tag,
                            "set": name,
                            "band": band,
                            "arm": col,
                            "n": sub.height,
                            "crps": float(sub[f"{col}__crps"].mean()),
                        }
                    )  # ty: ignore[invalid-argument-type]
                pairs = [(a, "L-1d") for a in ARMS] + [("R9", "R-DW"), ("R-DROP", "R-DW")]
                for a, b in pairs:
                    d = sub.select("month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps"))
                    st = s.block_stats(diff=d, rng=rng, tests=TESTS)
                    counted = (
                        (a, b) == ("R-DW", "L-1d")
                        and tag != "09"
                        and band in COUNTED_BANDS
                        and name in ("eight", "six")
                    )
                    contrasts.append(
                        {
                            "issue": tag,
                            "set": name,
                            "band": band,
                            "contrast": f"{a} - {b}",
                            "counted": counted,
                            "tests": TESTS,
                            "mde": m,
                            **st,
                        }
                    )
    pl.DataFrame(absolute).write_csv(OUT / "absolute.csv")
    c = pl.DataFrame(contrasts)
    c.write_csv(OUT / "contrasts.csv")
    checks["counted_tests"] = int(c["counted"].sum())
    assert checks["counted_tests"] == TESTS, checks["counted_tests"]
    (OUT / "checks_score.json").write_text(json.dumps(checks, indent=1))
    with pl.Config(tbl_rows=300, tbl_width_chars=250):
        show = c.filter(pl.col("contrast") == "R-DW - L-1d").filter(pl.col("set") != "report")
        print(
            show.select(
                "issue", "set", "band",
                pl.col("difference").round(3), pl.col("low_month").round(3), pl.col("high_month").round(3),
                pl.col("mde").round(3), "folds_positive", "folds", "counted",
            ).sort("set", "band", "issue")
        )  # fmt: skip
    print("counted tests:", checks["counted_tests"])


main()

# ruff: noqa: D103, E501
"""Experiment 693: figures (CPU).

uv run --no-sync python ml-experiments/00693-*/e00693_figures.py OUT_DIR

Writes SVG figures from the stored head scores and the contrasts CSV: the planned contrasts, the CRPS
of each arm by phase on bank F, six real-transfer time series (day-ahead forecasts stitched), and a
day-ahead and a 14-day forecast from one origin. Substations may be named and dated; no generator
appears. The stored scores hold each forecast's median only, so no figure draws the quantile bands.
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00693_headdata as hd
import e00693_real as real693
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

SCORES: Final = hd.ROOT / "head_scores"
COL: Final = {
    "Observed": "#000000",
    "Candidate alone": "#306BFF",
    "A0 (no flag)": "#FF4901",
    "A1 (oracle)": "#B701FF",
    "A2 (placebo)": "#17E58F",
    "B-haz": "#10C5F7",
    "Oracle B-haz": "#FC9700",
}
ARM_OF: Final = {
    "Candidate alone": "CAND",
    "A0 (no flag)": "A0",
    "A1 (oracle)": "A1",
    "A2 (placebo)": "A2",
    "B-haz": "B",
    "Oracle B-haz": "OB",
}
PER: Final[int] = 48
LABELS: Final = k578.sites()["label"].to_list()


def stamp(idx: pl.Expr) -> pl.Expr:
    """Return the UTC timestamp of the half-hour at grid index `idx`."""
    return pl.from_epoch(
        idx.cast(pl.Int64) * 1800 + int(k578.GRID_START.timestamp()), time_unit="s"
    )


def idx_of(*, y: int, m: int, d: int, h: int = 0) -> int:
    return int((datetime(y, m, d, h, tzinfo=UTC) - k578.GRID_START).total_seconds() // 1800)


def real_scores(*, arm: str, month: str) -> pl.DataFrame:
    return pl.read_parquet(SCORES / f"{arm}_{hd.fold_of(month=month)}_real_{month}.parquet")


def emit(chart: alt.TopLevelMixin, path: Path) -> None:
    """Write the chart as SVG, and as PNG in $E693_PNG_DIR when that is set (to look at the render)."""
    chart.save(path)
    if png := os.environ.get("E693_PNG_DIR"):
        chart.save(Path(png) / path.with_suffix(".png").name, scale_factor=1.0)


def contrasts(*, out: Path) -> None:
    c = pl.read_csv(hd.ROOT / "scores" / "head_contrasts.csv")
    keep = {
        ("bankdec", "C2 A1-A2 P1"): "C2: oracle minus placebo, target inside a transfer (bank F)",
        ("bankdec", "C3 A1-A2 P2"): "C3: oracle minus placebo, target after the end (bank F)",
        ("real verified", "C4 A1-A0"): "C4: oracle minus no flag, event-free real rows",
        ("bankdec", "C5 A2-A0 P1"): "C5: placebo minus no flag, inside a transfer (bank F)",
    }
    rows = [
        {
            "label": v,
            **{
                k: r[k]
                for k in ("mean", "95% low", "95% high", "Bonferroni low", "Bonferroni high")
            },
        }
        for (s, n), v in keep.items()
        for r in c.filter((pl.col("set") == s) & (pl.col("contrast") == n)).iter_rows(named=True)
    ]
    df = pl.DataFrame(rows).to_pandas()
    order = list(df["label"])
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=420))
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#888888", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    wide = base.mark_rule(color="#FF4901", strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "Bonferroni low:Q",
            title="CRPS difference (points of capacity; negative is better for the first-named arm)",
        ),
        x2="Bonferroni high:Q",
    )
    narrow = base.mark_rule(color="#306BFF", strokeWidth=5).encode(x="95% low:Q", x2="95% high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=90, color="#000000").encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
    chart = (rule + wide + narrow + dot).properties(
        width=CONTENT_WIDTH_PX - 330,
        height=170,
        title=alt.TitleParams(
            [
                "Planned contrasts, February to June 2026, quarterly heads.",
                "Dot: estimate. Blue bar: 95% interval. Orange line: Bonferroni interval for 3 tests. Dashed rule: no difference.",
            ],
            fontSize=12,
            anchor="start",
        ),
    )
    emit(chart, out / "contrasts.svg")


def by_phase(*, out: Path) -> None:
    rows = []
    for arm, name in (
        ("CAND", "Candidate alone"),
        ("B", "B-haz"),
        ("OB", "Oracle B-haz"),
        ("A0", "A0 (no flag)"),
        ("A1", "A1 (oracle)"),
        ("A2", "A2 (placebo)"),
    ):
        d = pl.concat(
            [
                pl.read_parquet(f)
                for fold in ("2026-02", "2026-05")
                for f in sorted(SCORES.glob(f"{arm}_{fold}_bankdec_*.parquet"))
            ]
        ).filter((pl.col("vi") >= 112560) & (pl.col("vi") < hd.CUTOFF_IDX))
        for ph, g in d.group_by("phase"):
            rows.append(
                {
                    "arm": name,
                    "phase": {
                        0: "other",
                        1: "P1 inside",
                        2: "P2 after end",
                        3: "P3 clean, after end",
                    }[ph[0]],
                    "crps": float(g["crps"].mean()),  # ty: ignore[invalid-argument-type]
                }
            )
    df = pl.DataFrame(rows).to_pandas()
    chart = (
        alt.Chart(df)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "arm:N",
                sort=[
                    "Candidate alone",
                    "B-haz",
                    "Oracle B-haz",
                    "A0 (no flag)",
                    "A1 (oracle)",
                    "A2 (placebo)",
                ],
                title=None,
                axis=alt.Axis(labelAngle=-30),
            ),
            y=alt.Y("crps:Q", title="CRPS (points of capacity; smaller is better)"),
            color=alt.Color(
                "arm:N",
                legend=None,
                scale=alt.Scale(
                    domain=[
                        "Candidate alone",
                        "B-haz",
                        "Oracle B-haz",
                        "A0 (no flag)",
                        "A1 (oracle)",
                        "A2 (placebo)",
                    ],
                    range=["#306BFF", "#10C5F7", "#17E58F", "#FF4901", "#B701FF", "#FC9700"],
                ),
            ),
            row=alt.Row("phase:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .properties(
            width=CONTENT_WIDTH_PX - 120,
            height=110,
            title=alt.TitleParams(
                [
                    "Mean CRPS on injected transfers (bank F, deciding bank), by phase, February to June 2026."
                ],
                fontSize=12,
                anchor="start",
            ),
        )
    )
    emit(chart, out / "crps_by_phase.svg")


def transfer_panel(
    *, label: str, start: tuple[int, int, int], end: tuple[int, int, int], out: Path
) -> None:
    prim = LABELS.index(label)
    s, e = idx_of(y=start[0], m=start[1], d=start[2]), idx_of(y=end[0], m=end[1], d=end[2])
    lo, hi = s - 7 * PER, e + 7 * PER
    frames, obs = [], None
    for arm_name, arm in ARM_OF.items():
        for month in ("2025-12", "2026-01", "2026-05", "2026-06"):
            try:
                d = real_scores(arm=arm, month=month)
            except FileNotFoundError:
                continue
            d = d.filter(
                (pl.col("prim") == prim)
                & (pl.col("lead") >= 24.0)
                & (pl.col("lead") < 48.0)
                & (pl.col("vi") >= lo)
                & (pl.col("vi") < hi)
            )
            if d.is_empty():
                continue
            frames.append(
                d.select("vi", pl.col("med").alias("value")).with_columns(series=pl.lit(arm_name))
            )
            if obs is None or arm == "CAND":
                obs = d.select("vi", pl.col("y").alias("value")).with_columns(
                    series=pl.lit("Observed")
                )
    df = (
        pl.concat([f for f in (*frames, obs) if f is not None])
        .with_columns(t=stamp(pl.col("vi")))
        .to_pandas()
    )
    band = (
        alt.Chart(pl.select(a=stamp(pl.lit(s)), b=stamp(pl.lit(e))).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rect(opacity=0.12, color="#FC9700")
        .encode(x="a:T", x2="b:T")
    )
    lines = (
        alt.Chart(df)  # ty: ignore[unresolved-attribute]
        .mark_line(clip=True, aria=False)
        .encode(
            x=alt.X("t:T", title=None),
            y=alt.Y("value:Q", title="Power (share of capacity)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(COL), range=list(COL.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    emit(
        (band + lines).properties(
            width=CONTENT_WIDTH_PX - 90,
            height=150,
            title=alt.TitleParams(
                [
                    f"{label}: day-ahead forecasts (24 to 48 hours ahead) around a logged transfer. Shaded: the transfer."
                ],
                fontSize=12,
                anchor="start",
            ),
        ),
        out / f"transfer_{label}_{start[0]}{start[1]:02d}.svg",
    )


def fan(*, label: str, origin: tuple[int, int, int], out: Path, tag: str) -> None:
    prim = LABELS.index(label)
    oi = idx_of(y=origin[0], m=origin[1], d=origin[2], h=9)
    month = f"{origin[0]}-{origin[1]:02d}"
    frames = []
    for arm_name, arm in ARM_OF.items():
        d = real_scores(arm=arm, month=month).filter(
            (pl.col("prim") == prim) & (pl.col("oi") == oi)
        )
        frames.append(
            d.select("vi", "lead", pl.col("med").alias("value")).with_columns(
                series=pl.lit(arm_name)
            )
        )
        if arm == "CAND":
            frames.append(
                d.select("vi", "lead", pl.col("y").alias("value")).with_columns(
                    series=pl.lit("Observed")
                )
            )
    df = pl.concat(frames).with_columns(t=stamp(pl.col("vi"))).to_pandas()
    emit(
        alt.Chart(df)  # ty: ignore[unresolved-attribute]
        .mark_line(clip=True, aria=False)
        .encode(
            x=alt.X("lead:Q", title="Hours after the 09:00 origin"),
            y=alt.Y("value:Q", title="Power (share of capacity)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(COL), range=list(COL.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=CONTENT_WIDTH_PX - 90,
            height=170,
            title=alt.TitleParams(
                [
                    f"{label}: forecast medians from one 09:00 origin out to 14 days ({tag}). The quantile bands are not stored."
                ],
                fontSize=12,
                anchor="start",
            ),
        ),
        out / f"fan_{label}_{tag}.svg",
    )


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    contrasts(out=out)
    by_phase(out=out)
    for label, st, en in (
        ("D5", (2025, 12, 29), (2026, 1, 13)),
        ("D16", (2025, 12, 29), (2026, 1, 13)),
        ("D10", (2026, 5, 11), (2026, 5, 21)),
        ("D11", (2026, 5, 12), (2026, 5, 22)),
        ("D12", (2026, 5, 11), (2026, 5, 21)),
        ("D16", (2026, 5, 4), (2026, 5, 22)),
    ):
        transfer_panel(label=label, start=st, end=en, out=out)
    fan(label="D5", origin=(2026, 3, 10), out=out, tag="winter")
    fan(label="D5", origin=(2026, 6, 10), out=out, tag="summer")
    _ = real693  # the spans behind the shaded bands are experiment 580's verified transfers


if __name__ == "__main__":
    main(Path(sys.argv[1]))

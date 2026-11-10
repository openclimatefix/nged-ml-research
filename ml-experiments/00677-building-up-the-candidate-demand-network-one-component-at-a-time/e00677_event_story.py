# ruff: noqa: B007, D103, D415, E501
"""Experiment 677: do steps 6 to 8 ride through load-shift events better than step 5? (private CSV and SVG)

Usage: python e00677_event_story.py OUT_DIR

Three analyses on the quarterly ladder (seeds 0 to 4 for steps 5 to 8):

1. Event-aligned error on the real logged events. The events are the 10 logged shifts that last more
   than
   7 days and involve 2 units or more, plus the verified D5 and D16 transfer. For each step, the
   mean absolute
   error of the median is taken by whole days since the event start and by whole days since the
   event end,
   with a 95% interval from resampling events.
2. Follow of an injected shift by shift depth, from the held-out bank (`e00677_bank_score.py`'s
   files). The
   ignored share is one minus the follow.
3. Mean absolute error on event-free rows: rows more than 14 days from every logged event of the
   same series
   (all lengths), with the contrast against step 5 and a 7-day block bootstrap over origin days.

Writes `scores/event_story_*.csv` and the SVG figures under OUT_DIR.
"""

import sys
from pathlib import Path

import altair as alt
import e00668_slices_primaries as s668  # ty: ignore[unresolved-import]
import e00677_bank_score as bk
import e00677_export as exp
import e00677_score as sc
import numpy as np
import polars as pl

STEPS = ("3b", "5w", "6w", "7w", "8w")
NAME = {
    "3b": "3b: date-time + weather (pre-trained)",
    "5w": "5: + 24 h and 1-week lags",
    "6w": "6: + gap columns",
    "7w": "7: + weighted averages of the gap",
    "8w": "8: + neighbour columns (Candidate)",
}
COLOUR = {"3b": "#17E58F", "5w": "#0B215F", "6w": "#B701FF", "7w": "#FC9700", "8w": "#009C75"}
DAY = s668.DAY
N_BOOT = 2000


def step_rows(step: str) -> pl.DataFrame:
    tag = "s01234" if step[0] in "5678" else "s012"
    f = sc.ROOT / "scores" / f"step{step}_{exp.VERSION[step]}_sp3_{tag}" / "rows.parquet"
    return (
        pl.read_parquet(
            f, columns=["time_series_id", "valid_idx", "origin_idx", "lead_h", "ok", "Model_mae"]
        )
        .filter(pl.col("ok"))
        .drop("ok")
    )


def all_rows() -> pl.DataFrame:
    site = pl.read_csv(next(s668.R.glob("00521-*/site_map.csv")))
    out = None
    for s in STEPS:
        r = step_rows(s).rename({"Model_mae": f"mae_{s}"})
        out = (
            r
            if out is None
            else out.join(
                r.drop("lead_h"), on=["time_series_id", "valid_idx", "origin_idx"], how="inner"
            )
        )
    return out.join(site.select("time_series_id", "label"), on="time_series_id")  # ty: ignore[unresolved-attribute]


def events() -> tuple[pl.DataFrame, pl.DataFrame]:
    site = pl.read_csv(next(s668.R.glob("00521-*/site_map.csv")))
    sets = s668.event_sets(site=site)
    chosen = pl.concat(
        [sets["event table, over 7 days, 2 units or more"], sets["verified transfer"]]
    ).with_row_index("event")
    table = pl.read_parquet(next(s668.R.glob("00653-*/real_events_masked.parquet"))).filter(
        pl.col("start") >= s668.SCORED_FROM_653
    )
    table = table.with_columns(
        label=pl.col("primary").replace_strict(
            dict(zip(site["label"].to_list(), site["label"].to_list(), strict=True)), default=None
        )
        if table["primary"].dtype == pl.String
        else pl.col("primary").replace_strict(
            dict(zip(site["primary"].to_list(), site["label"].to_list(), strict=True))
        ),
        start=pl.col("start") + s668.OFFSET_653,
        end=pl.col("end") + s668.OFFSET_653,
    ).select("label", "start", "end")
    return chosen, table


def aligned(*, rows: pl.DataFrame, ev: pl.DataFrame, anchor: str, lo: int, hi: int) -> pl.DataFrame:
    """Return, per event, step, and whole day since `anchor` ("start" or "end"), the mean absolute error."""
    parts = []
    for e in ev.iter_rows(named=True):
        sub = rows.filter(
            (pl.col("label") == e["label"])
            & (pl.col("valid_idx") >= e[anchor] + lo * DAY)
            & (pl.col("valid_idx") < e[anchor] + hi * DAY)
        )
        if sub.height == 0:
            continue
        sub = sub.with_columns(
            rel=((pl.col("valid_idx") - e[anchor]) // DAY).cast(pl.Int32), event=pl.lit(e["event"])
        )
        parts.append(sub.group_by("event", "rel").agg([pl.col(f"mae_{s}").mean() for s in STEPS]))
    return pl.concat(parts)


def pooled(*, per_event: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(677)
    events_ = per_event["event"].unique().to_list()
    recs = []
    for rel in sorted(per_event["rel"].unique().to_list()):
        sub = per_event.filter(pl.col("rel") == rel)
        for s in STEPS:
            by = dict(zip(sub["event"].to_list(), sub[f"mae_{s}"].to_list(), strict=True))
            vals = np.array([by[e] for e in events_ if e in by])
            if len(vals) < 3:
                continue
            draws = [rng.choice(vals, size=len(vals)).mean() for _ in range(N_BOOT)]
            recs.append(
                {
                    "rel": rel,
                    "step": s,
                    "mae": float(vals.mean()),
                    "lo": float(np.quantile(draws, 0.025)),
                    "hi": float(np.quantile(draws, 0.975)),
                    "events": len(vals),
                }
            )
    return pl.DataFrame(recs)


def line_chart(
    *, frame: pl.DataFrame, title: str, xtitle: str, shade: tuple[float, float]
) -> alt.LayerChart:
    pdf = frame.with_columns(system=pl.col("step").replace_strict(NAME)).to_pandas()
    domain = [NAME[s] for s in STEPS]
    colour = alt.Color(
        "system:N",
        title=None,
        scale=alt.Scale(domain=domain, range=[COLOUR[s] for s in STEPS]),
        legend=alt.Legend(
            orient="top", columns=1, symbolType="stroke", symbolStrokeWidth=3, labelLimit=500
        ),
    )
    import pandas as pd  # noqa: TID251  (Altair takes pandas frames)

    rect = (
        alt.Chart(pd.DataFrame({"a": [shade[0]], "b": [shade[1]], "t": ["median event length"]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color="#D9D9D9", opacity=0.6)
        .encode(x="a:Q", x2="b:Q")
    )
    text = (
        alt.Chart(pd.DataFrame({"a": [shade[0]], "t": ["median event length"]}))  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", baseline="top", dx=4, dy=2, fontSize=11, color="#555555")
        .encode(x="a:Q", y=alt.value(0), text="t:N")
    )
    x = alt.X("rel:Q", title=xtitle)
    line = (
        alt.Chart(pdf)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=x,
            y=alt.Y(
                "mae:Q",
                title="Mean absolute error of the median (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=colour,
        )
    )
    return alt.layer(rect, text, line).properties(  # ty: ignore[invalid-return-type]
        width=620, height=300, title=alt.TitleParams(title, anchor="start", fontSize=12)
    )


def event_free(*, rows: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    near = np.zeros(rows.height, dtype=bool)
    label, v = rows["label"].to_numpy(), rows["valid_idx"].to_numpy()
    for e in table.iter_rows(named=True):
        near |= (label == e["label"]) & (v >= e["start"] - 14 * DAY) & (v < e["end"] + 14 * DAY)
    free = rows.filter(pl.Series(~near))
    day = (free["origin_idx"].to_numpy() // DAY).astype(np.int64)
    days, inv = np.unique(day, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(6)
    blocks = np.arange(len(days)) // 7
    nb = blocks.max() + 1
    recs = []
    base = free["mae_5w"].to_numpy().astype(np.float64)
    for s in STEPS:
        v_ = free[f"mae_{s}"].to_numpy().astype(np.float64)
        for tag, vals in (("mae", v_), ("minus step 5", v_ - base)):
            sums, cnt = np.bincount(inv, weights=vals), counts
            bs, bc = (
                np.bincount(blocks, weights=sums[: len(blocks)], minlength=nb),
                np.bincount(blocks, weights=cnt, minlength=nb),
            )
            draws = np.array(
                [
                    bs[i].sum() / bc[i].sum()
                    for i in (rng.integers(0, nb, size=nb) for _ in range(N_BOOT))
                ]
            )
            recs.append(
                {
                    "step": s,
                    "measure": tag,
                    "mean": float(vals.mean()),
                    "lo": float(np.quantile(draws, 0.025)),
                    "hi": float(np.quantile(draws, 0.975)),
                    "rows": free.height,
                }
            )
    return pl.DataFrame(recs)


def follow_by_depth() -> pl.DataFrame:
    rows = []
    for step in ("5w", "6w", "7w", "8w", "5wo", "6wo", "7wo", "8wo"):
        for month in bk.FOLDS:
            for f in sorted(
                (bk.ROOT / f"{bk.ARM[step[0]]}_{month}_{step}_sp3" / "work" / "bank").glob(
                    "*.parquet"
                )
            ):
                seed = int(f.stem.rsplit("_s", 1)[1])
                d = pl.read_parquet(f, columns=["median_base", "median_bank", "lag_shift"]).filter(
                    pl.col("lag_shift") != 0
                )
                d = d.with_columns(
                    depth=(pl.col("lag_shift").abs() * 20).round() / 20,
                    change=pl.col("median_bank") - pl.col("median_base"),
                )
                g = d.group_by("depth").agg(
                    num=(pl.col("change") * pl.col("lag_shift")).sum(),
                    den=(pl.col("lag_shift") ** 2).sum(),
                    rows=pl.len(),
                )
                rows.append(
                    g.with_columns(step=pl.lit(step), fold=pl.lit(month), seed=pl.lit(seed))
                )
    allg = pl.concat(rows)
    by = allg.group_by("step", "depth", "seed").agg(
        follow=pl.col("num").sum() / pl.col("den").sum(), rows=pl.col("rows").sum()
    )
    return (
        by.group_by("step", "depth")
        .agg(
            follow=pl.col("follow").mean(),
            sd_over_seeds=pl.col("follow").std(),
            seeds=pl.len(),
            rows_per_seed=pl.col("rows").mean(),
        )
        .sort("step", "depth")
    )


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = all_rows()
    ev, table = events()
    print(
        "events", ev.height, "all logged events in the scored year table", table.height, flush=True
    )
    for anchor, lo, hi, shade_end in (("start", -7, 21, None), ("end", -14, 14, None)):
        per = aligned(rows=rows, ev=ev, anchor=anchor, lo=lo, hi=hi)
        pool = pooled(per_event=per)
        pool.write_csv(sc.ROOT / "scores" / f"event_story_aligned_{anchor}.csv")
        length = float(np.median((ev["end"] - ev["start"]).to_numpy()) / DAY)
        shade = (0.0, length) if anchor == "start" else (-length, 0.0)
        line_chart(
            frame=pool,
            title=[  # ty: ignore[invalid-argument-type]
                f"Error of the median forecast by whole days since the event {anchor}; 10 to {ev.height} logged events pooled (two began before the backtest).",
                "Mean over events and seeds 0 to 4. The steps nearly coincide.",
            ],
            xtitle=f"Days since the event {anchor} (shading: the median event length, {length:.0f} days)",
            shade=shade,
        ).save(out / f"event_aligned_{anchor}.svg")
    ev_t = ev.tail(2)
    per = aligned(rows=rows, ev=ev_t, anchor="start", lo=-14, hi=28)
    # two events only: no interval is meaningful; draw the mean without a band.
    pool = per.group_by("rel").agg([pl.col(f"mae_{s}").mean() for s in STEPS]).sort("rel")
    pool.write_csv(sc.ROOT / "scores" / "event_story_transfer_by_day.csv")
    free = event_free(rows=rows, table=table)
    free.write_csv(sc.ROOT / "scores" / "event_story_event_free.csv")
    print(free)
    fd = follow_by_depth()
    fd.write_csv(sc.ROOT / "scores" / "event_story_follow_by_depth.csv")
    print(fd)

    d = (
        fd.with_columns(depth=(pl.col("depth") * 100).round().cast(pl.Int32) / 100.0)
        .filter(pl.col("depth").is_in([0.15, 0.3, 0.5]))
        .with_columns(
            lo=pl.col("follow") - pl.col("sd_over_seeds"),
            hi=pl.col("follow") + pl.col("sd_over_seeds"),
            injected=pl.when(pl.col("step").str.ends_with("wo"))
            .then(pl.lit("trained without injected events"))
            .otherwise(pl.lit("trained with injected events")),
            number=pl.col("step").str.slice(0, 1),
        )
        .to_pandas()
    )
    names = {
        "5": "5: + 24 h and 1-week lags",
        "6": "6: + gap columns",
        "7": "7: + weighted averages of the gap",
        "8": "8: + neighbour columns (Candidate)",
    }
    d["system"] = d["number"].map(names)
    colour = alt.Color(
        "system:N",
        title=None,
        scale=alt.Scale(
            domain=list(names.values()), range=["#0B215F", "#B701FF", "#FC9700", "#009C75"]
        ),
        legend=alt.Legend(orient="top", columns=1, symbolType="stroke", symbolStrokeWidth=3),
    )
    x = alt.X(
        "depth:Q",
        title="Depth of the injected shift in the 1-day lag (share of the Primary's capacity)",
        axis=alt.Axis(values=[0.15, 0.3, 0.5], format=".2f"),
        scale=alt.Scale(domain=[0.1, 0.55]),
    )
    panels = []
    for inj in ("trained with injected events", "trained without injected events"):
        sub = d[d.injected == inj]
        panels.append(
            (
                alt.Chart(sub)  # ty: ignore[unresolved-attribute]
                .mark_line(strokeWidth=1.2, point=True)
                .encode(
                    x=x,
                    y=alt.Y(
                        "follow:Q",
                        title="Share of the shift copied (0 = ignored; smaller is better)",
                        scale=alt.Scale(domain=[-0.02, 0.18]),
                    ),
                    color=colour,
                )
                + alt.Chart(sub)  # ty: ignore[unresolved-attribute]
                .mark_rule(strokeWidth=1)
                .encode(x=x, y="lo:Q", y2="hi:Q", color=colour)
            ).properties(
                width=620, height=300, title=f"Held-out bank of injected shifts, networks {inj}"
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Dot: mean of seeds 0 to 2. Line: plus and minus one standard deviation over seeds. Shift injected into the lag inputs only.",
            anchor="start",
            fontSize=11,
        )
    ).save(out / "bank_follow_by_depth.svg")


if __name__ == "__main__":
    main()

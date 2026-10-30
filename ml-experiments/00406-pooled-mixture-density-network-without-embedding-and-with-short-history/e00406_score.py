# ruff: noqa: E501
"""Score experiment 406's pooling controls against each other and against experiment 397.

Reads the per-seed forecasts that `e00406_fit.py` wrote (and, for the contrasts against the uncut
forecasts, experiment 397's), averages the seeds in quantile space, restricts every arm to the
half-hours all arms share, and prints:

- the absolute CRPS at days 1 to 10 per group, per arm, at every history length;
- the contrasts of `PLAN` (month-block bootstrap of `studies.eval_protocol`, 95%, 20,000 resamples);
- how the pooled gain over the per-Primary network varies with the history length, per group and
  per Primary.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
uv run --no-sync python ml-experiments/00406-*/e00406_score.py
```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT_406: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00406-pooled-mixture-density-network-without-embedding-and-with-short-history"
)
SEED_406: Final[Path] = ROOT_406 / "work" / "seed_preds"
SEED_397: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00397-plain-and-attention-mixture-density-network-for-primary-demand"
    / "work"
    / "seed_preds"
    / "summary"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
HISTORIES: Final[tuple[int, ...]] = (3, 6, 12)
N_SEEDS: Final[int] = 3
HEADLINE: Final[str] = base.HEADLINE_BAND
SIGNED: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
SHORT_HISTORY: Final[tuple[str, ...]] = ("D15", "D18")
SUBSETS: Final[dict[str, tuple[str, ...]]] = {
    "signed-MW six": SIGNED,
    "MVA-only eight": MVA_ONLY,
    "D15 and D18": SHORT_HISTORY,
}
KEYS: Final[list[str]] = list(common.KEYS)
ARM_DIRS: Final[dict[str, Path]] = {
    "per_primary": SEED_406 / "per_primary",
    "pooled_emb": SEED_406 / "pooled_emb",
    "pooled_noemb": SEED_406 / "pooled_noemb",
    **{f"per_primary_h{h}": SEED_406 / f"per_primary_h{h}" for h in HISTORIES},
    **{f"pooled_emb_h{h}": SEED_406 / f"pooled_emb_h{h}" for h in HISTORIES},
    "uncut_per_primary": SEED_397 / "plain_own",
    "uncut_pooled_emb": SEED_397 / "plain_pooled",
}
PLAN: Final[dict[str, tuple[str, str]]] = {
    "C1 no embedding minus per-Primary": ("pooled_noemb", "per_primary"),
    "C2 no embedding minus embedding": ("pooled_noemb", "pooled_emb"),
    "C3 embedding minus per-Primary": ("pooled_emb", "per_primary"),
    **{
        f"C4 embedding minus per-Primary, {h} months": (f"pooled_emb_h{h}", f"per_primary_h{h}")
        for h in HISTORIES
    },
    "C5 cut minus uncut, per-Primary": ("per_primary", "uncut_per_primary"),
    "C5 cut minus uncut, pooled embedding": ("pooled_emb", "uncut_pooled_emb"),
}
OUT_DIR: Final[Path] = ROOT_406
FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00406-pooled-mixture-density-network-without-embedding-and-with-short-history"
)


def seed_average(*, arm_dir: Path) -> pl.DataFrame:
    """Return an arm's forecast (fraction of capacity) averaged over seeds, all five months."""
    pieces = []
    for month in MONTHS:
        files = [
            f for f in sorted(arm_dir.glob(f"{month}_s*.parquet")) if not f.stem.endswith("_gate")
        ]
        if len(files) != N_SEEDS:
            msg = f"{arm_dir}: {len(files)} seeds for {month}, expected {N_SEEDS}"
            raise ValueError(msg)
        stacked = pl.concat([pl.read_parquet(f) for f in files])
        pieces.append(stacked.group_by(KEYS).agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS]))
    frame = pl.concat(pieces).sort(KEYS)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def contrast(*, name: str, rows: dict, table: pl.DataFrame, pair: tuple[str, str]) -> None:
    """Print one contrast per group."""
    for group, labels in SUBSETS.items():
        where = pl.col("label").is_in(list(labels)) & pl.col(HEADLINE)
        r = metrics.month_bootstrap_contrast(
            treatment=rows[pair[0]],
            reference=rows[pair[1]],
            metric="crps",
            where=where,
            level_percent=95.0,
            n_resamples=20_000,
            slice_table=table,
        )
        print(
            f"{name:48s} {group:15s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) n={r['n_half_hours']}"
        )


def history_months(*, series: pl.DataFrame, ctx: dict) -> dict[str, float]:
    """Return each Primary's months of readings before 2025-10-01."""
    cut = datetime(2025, 10, 1, tzinfo=UTC)
    label_of = dict(zip(series["time_series_id"].to_list(), series["label"].to_list(), strict=True))
    return {
        label_of[int(sid)]: max((cut - first).days / 30.44, 0.0)
        for sid, first in ctx["first_readings"].items()
    }


def main() -> None:
    """Print the absolute table, the contrasts, and the gain against history length."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    available = {
        n: d
        for n, d in ARM_DIRS.items()
        if len(list(d.glob("*_s*.parquet"))) == N_SEEDS * len(MONTHS)
    }
    print("arms found:", ", ".join(available))
    frames = {
        n: adapter.forecast_frame(stored=seed_average(arm_dir=d), series=series, unit="fraction")
        for n, d in available.items()
    }
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    for n, t in scored.items():
        labels = set(t["label"].unique().to_list())
        if len(labels) != len(series.filter(pl.col("substation_type") == "Primary")):
            msg = f"arm {n} scores only {sorted(labels)}: a Primary has no forecast"
            raise ValueError(msg)
    crps = pl.concat(
        [
            t.filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE).select(
                "label", pl.lit(n).alias("arm"), "value"
            )
            for n, t in scored.items()
        ]
    ).pivot(on="arm", index="label", values="value")
    crps = crps.sort(pl.col("label").str.slice(1).cast(pl.Int32))
    with pl.Config(tbl_rows=40, tbl_cols=30):
        print("\n== CRPS days 1 to 10 per Primary (points of capacity) ==")
        print(crps)
    print("\n== group means (equal weight per Primary) ==")
    for group, labels in SUBSETS.items():
        sub = crps.filter(pl.col("label").is_in(list(labels)))
        print(group, {a: round(float(sub[a].mean()), 3) for a in available})
    print("\n== contrasts (treatment minus reference, month-block bootstrap, interim) ==")
    for name, pair in PLAN.items():
        if pair[0] in rows and pair[1] in rows:
            contrast(name=name, rows=rows, table=table, pair=pair)
    gain_by_history = {}
    for h in (0, *HISTORIES):
        suffix = "" if h == 0 else f"_h{h}"
        a, b = f"pooled_emb{suffix}", f"per_primary{suffix}"
        if a in rows and b in rows:
            gain_by_history[h] = crps[a] - crps[b]
    if gain_by_history:
        hist = history_months(series=series, ctx=ctx)
        print(
            "\n== pooled minus per-Primary per Primary and history length (negative favours pooling) =="
        )
        out = pl.DataFrame(
            {"label": crps["label"], **{f"h{h or 'full'}": g for h, g in gain_by_history.items()}}
        )
        with pl.Config(tbl_rows=40):
            print(out)
        countable = out.filter(~pl.col("label").is_in(list(SHORT_HISTORY)))
        months = np.array([hist[x] for x in countable["label"]])
        for h in gain_by_history:
            r = np.corrcoef(countable[f"h{h or 'full'}"].to_numpy(), months)[0, 1]
            print(
                f"correlation of the gain at history {h or 'full'} with months of history: {r:+.2f}"
            )
    draw_figure(rows=rows, table=table)
    draw_examples(frames=frames, ctx=ctx, series=series)


EXAMPLES: Final[dict[str, str]] = {"D10": "signed-MW Primary", "D4": "MVA-only Primary"}
EXAMPLE_ORIGINS: Final[tuple[str, ...]] = ("2026-01-12", "2026-06-15")


def draw_examples(*, frames: dict, ctx: dict, series: pl.DataFrame) -> None:
    """Draw a day-ahead and a 14-day forecast of the pooled network against the actuals."""
    import altair as alt
    from plotting import ocf_theme as theme

    forecast = frames["pooled_emb"]
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    for origin_text in EXAMPLE_ORIGINS:
        origin = datetime.fromisoformat(origin_text).replace(tzinfo=UTC)
        for label, kind in EXAMPLES.items():
            one = forecast.filter(
                pl.col("time_series_id") == ids[label], pl.col("power_fcst_init_time") == origin
            ).join(
                ctx["actuals"].filter(pl.col("time_series_id") == ids[label]),
                on=["time_series_id", "valid_time"],
                how="left",
            )
            one = one.with_columns(
                day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0
            ).sort("valid_time")
            for horizon, window in (("day-ahead", (1.0, 2.0)), ("14 days", (0.0, 14.0))):
                sub = one.filter(pl.col("day").is_between(*window)).to_pandas()
                x = alt.X(
                    "day:Q",
                    title="Days after the forecast was made",
                    scale=alt.Scale(domain=list(window), nice=False),
                )
                base = alt.Chart(sub).encode(x=x)  # ty: ignore[unresolved-attribute]
                wide = base.mark_area(opacity=0.2, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
                )
                narrow = base.mark_area(opacity=0.35, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                    y="q25:Q", y2="q75:Q"
                )
                median = base.mark_line(color=theme.DATA_BLUE, strokeWidth=2).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
                actual = base.mark_line(color="black", strokeWidth=1.5).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
                chart = (wide + narrow + median + actual).properties(
                    width=590,
                    height=200,
                    title=alt.TitleParams(
                        [
                            f"{label} ({kind}), {horizon} forecast made on {origin_text}",
                            "Blue line: median. Bands: 50% and 90%. Black line: what happened.",
                        ],
                        anchor="start",
                    ),
                )
                name = f"figure_example_{label}_{origin_text}_{horizon.replace(' ', '_').replace('-', '_')}"
                chart.save(FIGURE_DIR / f"{name}.svg")


def draw_figure(*, rows: dict, table: pl.DataFrame) -> None:
    """Draw the pooled-minus-per-Primary contrast against the history length, per group."""
    import altair as alt
    from plotting import ocf_theme as theme

    data = []
    for h in (3, 6, 12, 0):
        suffix = "" if h == 0 else f"_h{h}"
        if f"pooled_emb{suffix}" not in rows or f"per_primary{suffix}" not in rows:
            continue
        for group, labels in SUBSETS.items():
            r = metrics.month_bootstrap_contrast(
                treatment=rows[f"pooled_emb{suffix}"],
                reference=rows[f"per_primary{suffix}"],
                metric="crps",
                where=pl.col("label").is_in(list(labels)) & pl.col(HEADLINE),
                level_percent=95.0,
                n_resamples=20_000,
                slice_table=table,
            )
            data.append(
                {
                    "history": "Full 24 months" if h == 0 else f"{h} months",
                    "group": group,
                    "difference": r["difference"],
                    "lower": r["lower"],
                    "upper": r["upper"],
                }
            )
    frame = pl.DataFrame(data).to_pandas()
    order = ["3 months", "6 months", "12 months", "Full 24 months"]
    colour = alt.Color(
        "group:N",
        scale=alt.Scale(
            domain=list(SUBSETS), range=[theme.DATA_BLUE, theme.BRAND_ORANGE, theme.DATA_PURPLE]
        ),
        legend=alt.Legend(title=None, orient="top", direction="horizontal"),
    )
    y = alt.Y("history:N", sort=order, title=None)
    offset = alt.YOffset("group:N", sort=list(SUBSETS))
    bars = (
        alt.Chart(frame)
        .mark_rule(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            yOffset=offset,
            x=alt.X(
                "lower:Q",
                title="Pooled minus per-Primary network, CRPS at days 1 to 10 (points of capacity; below zero favours the pooled network)",
            ),
            x2="upper:Q",
            color=colour,
        )
    )
    dots = (
        alt.Chart(frame)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y, yOffset=offset, x="difference:Q", color=colour
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
        .mark_rule(strokeDash=[4, 4], color="grey")
        .encode(  # ty: ignore[unresolved-attribute]
            x="x:Q"
        )
    )
    chart = (bars + dots + zero).properties(
        width=560,
        height=260,
        title=alt.TitleParams(
            [
                "Pooling gain against the length of each Primary's training history",
                "Dot: estimate. Line: 95% interval from resampling whole months. Dashed rule: no difference.",
            ],
            anchor="start",
        ),
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(FIGURE_DIR / "figure_1_gain_by_history.svg")
    chart.save(OUT_DIR / "figure_1_gain_by_history.png", scale_factor=2)


if __name__ == "__main__":
    main()

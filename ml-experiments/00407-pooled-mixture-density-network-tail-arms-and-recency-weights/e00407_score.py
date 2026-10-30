# ruff: noqa: E501
"""Score experiment 407's tail arms and recency-weight arms against the widened pooled network.

Reads the per-seed forecasts `e00407_fit.py` wrote, averages the seeds in quantile space, restricts
every arm to the half-hours all arms share, and prints:

- CRPS at days 1 to 10 per group, for every arm raw, widened by the global factor 1.15 (`_w115`,
  as in experiments 389 and 397), and widened per lead band on held-out rows (`_bw`);
- coverage and pinball loss at q01, q05, q95, and q99 per group, and the same on sunny half-hours
  where the reference's median forecast is in the lowest quarter for its Primary ("sunny
  low-demand");
- the registered contrasts against the reference `ctrl_w115`, with the month-block bootstrap of
  `studies.eval_protocol` at 95% (20,000 resamples);
- example forecasts of the reference against what happened.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
uv run --no-sync python ml-experiments/00407-*/e00407_score.py
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

ROOT_407: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00407-pooled-mixture-density-network-tail-arms-and-recency-weights"
)
SEED_407: Final[Path] = ROOT_407 / "work" / "seed_preds"
SEED_406: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00406-pooled-mixture-density-network-without-embedding-and-with-short-history"
    / "work"
    / "seed_preds"
)
SEED_397: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00397-plain-and-attention-mixture-density-network-for-primary-demand"
    / "work"
    / "seed_preds"
    / "summary"
)
FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00407-pooled-mixture-density-network-tail-arms-and-recency-weights"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
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
OUTER_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
FLAGGED_KEYS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00398-load-shift-event-rescore-without-flagged-zero-actuals"
    / "flagged_keys.parquet"
)
GLOBAL_FACTOR: Final[float] = 1.15
NEW_ARMS: Final[tuple[str, ...]] = ("ctrl", "k5", "skew", "qhead", "hl365", "hl91", "a0")
REFERENCE: Final[str] = "ctrl_w115"
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "T1 5 components (widened 1.15) minus reference": ("k5_w115", REFERENCE),
    "T2 split-normal, raw, minus reference": ("skew", REFERENCE),
    "T2 split-normal, widened 1.15, minus reference": ("skew_w115", REFERENCE),
    "T3 per-lead-band widening of ctrl minus reference": ("ctrl_bw", REFERENCE),
    "T4 pinball quantile head, raw, minus reference": ("qhead", REFERENCE),
    "T4 pinball quantile head, widened 1.15, minus reference": ("qhead_w115", REFERENCE),
    "H1 half-life 1 year (widened 1.15) minus reference": ("hl365_w115", REFERENCE),
    "H2 half-life 0.25 years (widened 1.15) minus reference": ("hl91_w115", REFERENCE),
    "S ctrl minus the pooled plain network of experiment 406 (sanity, about 0)": (
        "ctrl",
        "pooled_plain_cut_406",
    ),
    "C6 A0 with the cut minus A0 without it (raw)": ("a0", "uncut_a0"),
    "C7 A0 with the cut minus per-Primary network with the cut (raw)": (
        "a0",
        "per_primary_cut",
    ),
}
EXAMPLES: Final[dict[str, str]] = {"D10": "signed-MW Primary", "D4": "MVA-only Primary"}
EXAMPLE_ORIGINS: Final[tuple[str, ...]] = ("2026-01-12", "2026-06-15")


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


def widen_global(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the quantiles with q01, q05, q95, and q99 widened by 1.15 about the median."""
    return frame.with_columns(
        *[pl.col("q50") + GLOBAL_FACTOR * (pl.col(c) - pl.col("q50")) for c in OUTER_LEVELS]
    )


def is_complete(*, arm_dir: Path) -> bool:
    """Return whether a folder holds every month and seed."""
    files = [f for f in arm_dir.glob("*_s*.parquet") if not f.stem.endswith("_gate")]
    return len(files) == N_SEEDS * len(MONTHS)


def load_frames(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the scored arms (MW, one row per key): the new arms, their widenings, references."""
    dirs: dict[str, Path] = {}
    for arm in NEW_ARMS:
        dirs[arm] = SEED_407 / arm
        dirs[f"{arm}_bw"] = SEED_407 / f"{arm}_bw"
    dirs["uncut_a0"] = SEED_397 / "A0"
    dirs["uncut_pooled_plain"] = SEED_397 / "plain_pooled"
    dirs["per_primary_cut"] = SEED_406 / "per_primary"
    dirs["pooled_plain_cut_406"] = SEED_406 / "pooled_emb"
    found = {n: d for n, d in dirs.items() if is_complete(arm_dir=d)}
    print("arms found:", ", ".join(found))
    stored = {n: seed_average(arm_dir=d) for n, d in found.items()}
    for arm in NEW_ARMS:
        if arm in stored:
            stored[f"{arm}_w115"] = widen_global(frame=stored[arm])
    return {
        n: adapter.forecast_frame(stored=f, series=series, unit="fraction")
        for n, f in stored.items()
    }


EXTRA_SLICES: Final[tuple[str, ...]] = (
    "cold_nights",
    "winter_nights",
    "sunny_unit_shape",
    "economy_7_step",
)


def extra_slices(*, scored: dict[str, pl.DataFrame]) -> None:
    """Print group means of CRPS on the cold-night and other slices, and the curtailment cost."""
    one = next(iter(scored.values()))
    print("\n== metrics available:", sorted(one["metric"].unique().to_list()))
    print("== slices available:", sorted(one["slice"].unique().to_list()))
    for slice_name in EXTRA_SLICES:
        for group, members in SUBSETS.items():
            means = {}
            for arm, t in scored.items():
                sub = t.filter(
                    pl.col("metric") == "crps",
                    pl.col("slice") == slice_name,
                    pl.col("label").is_in(list(members)),
                )
                if sub.height:
                    means[arm] = round(float(np.nanmean(sub["value"].to_numpy())), 3)
            if means:
                print(f"CRPS on {slice_name}, {group}: {means}")
    for metric in sorted(one["metric"].unique().to_list()):
        if "curtail" not in metric:
            continue
        for group, members in SUBSETS.items():
            means = {}
            for arm, t in scored.items():
                sub = t.filter(
                    pl.col("metric") == metric,
                    pl.col("slice") == HEADLINE,
                    pl.col("label").is_in(list(members)),
                )
                if sub.height:
                    means[arm] = round(float(np.nanmean(sub["value"].to_numpy())), 3)
            print(f"{metric} (days 1 to 10), {group}: {means}")


def sunny_low_demand(*, table: pl.DataFrame, reference: pl.DataFrame) -> pl.DataFrame:
    """Return the keys of sunny half-hours whose reference median is in the lowest quarter.

    The quarter is taken per Primary over its sunny half-hours (the flag `sunny_unit_shape`), and
    the cut uses the reference's median, never the actual, so the rows do not depend on the outcome.
    """
    sunny = table.filter(pl.col("sunny_unit_shape")).select(
        "time_series_id", "power_fcst_init_time", "valid_time"
    )
    joined = sunny.join(
        reference.select("time_series_id", "power_fcst_init_time", "valid_time", "q50"),
        on=["time_series_id", "power_fcst_init_time", "valid_time"],
    )
    cut = joined.group_by("time_series_id").agg(pl.col("q50").quantile(0.25).alias("cut"))
    return (
        joined.join(cut, on="time_series_id")
        .filter(pl.col("q50") <= pl.col("cut"))
        .select("time_series_id", "power_fcst_init_time", "valid_time")
    )


def tail_table(*, rows: dict[str, pl.DataFrame], low: pl.DataFrame) -> None:
    """Print coverage and pinball loss at the four outer levels, overall and sunny low-demand."""
    for slice_name, restrict in (("days 1 to 10", None), ("sunny low-demand", low)):
        out = []
        for arm, frame in rows.items():
            part = frame.filter(pl.col(HEADLINE))
            if restrict is not None:
                part = part.join(
                    restrict, on=["time_series_id", "power_fcst_init_time", "valid_time"]
                )
            label_col = "label"
            for group, members in SUBSETS.items():
                sub = part.filter(pl.col(label_col).is_in(list(members)))
                per_primary = sub.group_by(label_col).agg(
                    *[pl.col(f"coverage_{lv}").mean() for lv in OUTER_LEVELS],
                    *[pl.col(f"pinball_{lv}").mean() for lv in OUTER_LEVELS],
                )
                out.append(
                    {"arm": arm, "group": group, "n_half_hours": sub.height}
                    | {
                        c: round(float(np.nanmean(per_primary[c].to_numpy())), 3)
                        for c in per_primary.columns
                        if c != label_col
                    }
                )
        with pl.Config(tbl_rows=200, tbl_cols=20):
            print(
                f"\n== tails, {slice_name} (mean over Primaries; coverage nominal q01 0.01, q05 0.05, q95 0.95, q99 0.99) =="
            )
            print(pl.DataFrame(out).sort("group", "arm"))


def main() -> None:
    """Print the absolute table, the tails, the contrasts, and draw the example charts."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(FLAGGED_KEYS).select("time_series_id", "valid_time")
    before = ctx["actuals"].height
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    print(f"scoring actuals dropped for a flagged exact zero: {before - ctx['actuals'].height}")
    frames = load_frames(series=series)
    if REFERENCE not in frames:
        msg = f"the reference {REFERENCE} is missing"
        raise ValueError(msg)
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    crps = pl.concat(
        [
            t.filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE).select(
                "label", pl.lit(n).alias("arm"), "value"
            )
            for n, t in scored.items()
        ]
    ).pivot(on="arm", index="label", values="value")
    with pl.Config(tbl_rows=40, tbl_cols=40):
        print("\n== CRPS days 1 to 10 per Primary (points of capacity) ==")
        print(crps.sort(pl.col("label").str.slice(1).cast(pl.Int32)))
    print("\n== group means of CRPS at days 1 to 10 (equal weight per Primary) ==")
    for group, members in SUBSETS.items():
        sub = crps.filter(pl.col("label").is_in(list(members)))
        print(group, {a: round(float(sub[a].mean()), 3) for a in frames})
    extra_slices(scored=scored)
    low = sunny_low_demand(table=table, reference=frames[REFERENCE])
    print(f"\nsunny low-demand half-hours: {low.height}")
    tail_table(rows=rows, low=low)
    print("\n== contrasts (treatment minus reference, month-block bootstrap, interim) ==")
    for name, (treatment, reference) in CONTRASTS.items():
        if treatment not in rows or reference not in rows:
            continue
        for group, members in SUBSETS.items():
            r = metrics.month_bootstrap_contrast(
                treatment=rows[treatment],
                reference=rows[reference],
                metric="crps",
                where=pl.col("label").is_in(list(members)) & pl.col(HEADLINE),
                level_percent=95.0,
                n_resamples=20_000,
                slice_table=table,
            )
            print(
                f"{name:62s} {group:15s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) n={r['n_half_hours']}"
            )
    draw_examples(frame=frames[REFERENCE], ctx=ctx, series=series)


def draw_examples(*, frame: pl.DataFrame, ctx: dict, series: pl.DataFrame) -> None:
    """Draw a day-ahead and a 14-day forecast of the reference against the actuals."""
    import altair as alt
    from plotting import ocf_theme as theme

    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for origin_text in EXAMPLE_ORIGINS:
        origin = datetime.fromisoformat(origin_text).replace(tzinfo=UTC)
        for label, kind in EXAMPLES.items():
            one = (
                frame.filter(
                    pl.col("time_series_id") == ids[label],
                    pl.col("power_fcst_init_time") == origin,
                )
                .join(
                    ctx["actuals"].filter(pl.col("time_series_id") == ids[label]),
                    on=["time_series_id", "valid_time"],
                    how="left",
                )
                .with_columns(day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0)
                .sort("valid_time")
            )
            for horizon, window in (("day-ahead", (1.0, 2.0)), ("14 days", (0.0, 14.0))):
                sub = one.filter(pl.col("day").is_between(*window)).to_pandas()
                x = alt.X(
                    "day:Q",
                    title="Days after the forecast was made",
                    scale=alt.Scale(domain=list(window), nice=False),
                )
                base_chart = alt.Chart(sub).encode(x=x)  # ty: ignore[unresolved-attribute]
                wide = base_chart.mark_area(opacity=0.2, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
                )
                narrow = base_chart.mark_area(opacity=0.35, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                    y="q25:Q", y2="q75:Q"
                )
                median = base_chart.mark_line(color=theme.DATA_BLUE, strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
                    y="q50:Q"
                )
                actual = base_chart.mark_line(color="black", strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
                    y="actual:Q"
                )
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
                slug = horizon.replace(" ", "_").replace("-", "_")
                chart.save(FIGURE_DIR / f"figure_example_{label}_{origin_text}_{slug}.svg")


if __name__ == "__main__":
    main()

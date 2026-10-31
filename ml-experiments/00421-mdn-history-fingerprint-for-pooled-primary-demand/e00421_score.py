# ruff: noqa: E501
"""Score the history-fingerprint arms of experiment 421.

Reads the per-seed forecasts that `e00421_fit.py` wrote and experiment 406's `pooled_emb` and
`pooled_noemb`, averages the seeds in quantile space, restricts every arm to the half-hours all arms
share, and prints the absolute CRPS per group, the planned contrasts (month-block bootstrap of
`studies.eval_protocol`, Bonferroni level, 20,000 resamples), the probe accuracies, and the
held-out-Primary stage.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
uv run --no-sync python ml-experiments/00421-*/e00421_score.py
```
"""

import json
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

ROOT_421: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00421-mdn-history-fingerprint-for-pooled-primary-demand"
)
SEED_421: Final[Path] = ROOT_421 / "work" / "seed_preds"
SEED_406: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00406-pooled-mixture-density-network-without-embedding-and-with-short-history"
    / "work"
    / "seed_preds"
)
FLAGGED_KEYS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00398-load-shift-event-rescore-without-flagged-zero-actuals"
    / "flagged_keys.parquet"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
N_GROUPS: Final[int] = 4
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
MAIN_ARMS: Final[tuple[str, ...]] = ("E3g", "N3g", "NH", "NHs", "EH", "NH0")
HELD_OUT_ARMS: Final[tuple[str, ...]] = ("N3g", "NH", "E3g", "EH")
BONFERRONI_LEVEL: Final[float] = 100.0 - 5.0 / (9 * 2)
"""Nine planned contrasts read in two groups."""
PLAN: Final[dict[str, tuple[str, str]]] = {
    "C1 E3g minus E406": ("E3g", "E406"),
    "C2 N3g minus E3g": ("N3g", "E3g"),
    "C3 NH minus N3g": ("NH", "N3g"),
    "C4 EH minus E3g": ("EH", "E3g"),
    "C5 NH minus EH": ("NH", "EH"),
    "C6 NH0 minus NH": ("NH0", "NH"),
    "C9 NH minus NHs (shuffled history)": ("NH", "NHs"),
    "info: N3g minus experiment 406 no embedding": ("N3g", "pooled_noemb406"),
}
HELD_OUT_PLAN: Final[dict[str, tuple[str, str]]] = {
    "C7 NH minus E3g (mean embedding)": ("NH_ho", "E3g_ho"),
    "descriptive: NH minus E3g (metering-group mean embedding)": ("NH_ho", "E3g_ho_gm"),
    "descriptive: NH minus EH (mean embedding)": ("NH_ho", "EH_ho"),
    "C8 NH minus N3g": ("NH_ho", "N3g_ho"),
}
OUT_DIR: Final[Path] = ROOT_421
FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00421-mdn-history-fingerprint-for-pooled-primary-demand"
)


def seed_dirs(*, name: str) -> tuple[list[Path], int]:
    """Return the folders of an arm and its number of seeds (held-out arms have 4 quarters)."""
    if name == "E406":
        return [SEED_406 / "pooled_emb"], 3
    if name == "pooled_noemb406":
        return [SEED_406 / "pooled_noemb"], 3
    if name.endswith(("_ho", "_ho_gm")):
        stem, _, suffix = name.partition("_ho")
        return [SEED_421 / f"{stem}_ho_g{g}{suffix}" for g in range(N_GROUPS)], 2
    return [SEED_421 / name], 3


def is_complete(*, name: str) -> bool:
    """Return whether every folder of an arm holds every month and seed."""
    dirs, n_seeds = seed_dirs(name=name)
    return all(len(list(d.glob("*_s*.parquet"))) == n_seeds * len(MONTHS) for d in dirs)


def seed_average(*, name: str) -> pl.DataFrame:
    """Return an arm's forecast (fraction of capacity) averaged over seeds, all five months."""
    dirs, n_seeds = seed_dirs(name=name)
    pieces = []
    for arm_dir in dirs:
        for month in MONTHS:
            files = sorted(arm_dir.glob(f"{month}_s*.parquet"))
            if len(files) != n_seeds:
                msg = f"{arm_dir}: {len(files)} seeds for {month}, expected {n_seeds}"
                raise ValueError(msg)
            stacked = pl.concat([pl.read_parquet(f) for f in files])
            pieces.append(
                stacked.group_by(KEYS).agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS])
            )
    frame = pl.concat(pieces).sort(KEYS)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def contrast(*, name: str, rows: dict, table: pl.DataFrame, pair: tuple[str, str]) -> None:
    """Print one contrast per group at the Bonferroni level."""
    for group, labels in SUBSETS.items():
        where = pl.col("label").is_in(list(labels)) & pl.col(HEADLINE)
        r = metrics.month_bootstrap_contrast(
            treatment=rows[pair[0]],
            reference=rows[pair[1]],
            metric="crps",
            where=where,
            level_percent=BONFERRONI_LEVEL,
            n_resamples=20_000,
            slice_table=table,
        )
        print(
            f"{name:52s} {group:15s} {r['difference']:+.3f} "
            f"({r['lower']:+.3f} to {r['upper']:+.3f}) n={r['n_half_hours']}"
        )


def print_probes() -> None:
    """Print the mean probe accuracies of each encoder arm over folds and seeds."""
    for arm in ("NH", "EH", "NH0"):
        records = [
            json.loads(f.read_text())["probe"]
            for f in sorted((ROOT_421 / "work/info" / arm).glob("*.json"))
        ]
        if not records:
            continue
        keys = sorted({k for r in records for k in r})
        print(arm, {k: round(float(np.mean([r[k] for r in records if k in r])), 3) for k in keys})


def run_score(
    *, names: list[str], plan: dict[str, tuple[str, str]], series: pl.DataFrame, ctx: dict
) -> dict:
    """Score the named arms and print the absolute table and the contrasts; return the frames."""
    frames = {
        n: adapter.forecast_frame(stored=seed_average(name=n), series=series, unit="fraction")
        for n in names
    }
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
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
        print(group, {a: round(float(sub[a].mean()), 3) for a in names if a in sub.columns})
    print("\n== contrasts (treatment minus reference, month-block bootstrap, interim) ==")
    for name, pair in plan.items():
        if pair[0] in rows and pair[1] in rows:
            contrast(name=name, rows=rows, table=table, pair=pair)
    return frames


def main() -> None:
    """Print the main-stage table and contrasts, the probes, then the held-out stage."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(FLAGGED_KEYS).select("time_series_id", "valid_time")
    before = ctx["actuals"].height
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    print(f"scoring actuals dropped for a flagged exact zero: {before - ctx['actuals'].height}")
    main_names = [n for n in (*MAIN_ARMS, "E406", "pooled_noemb406") if is_complete(name=n)]
    print("main-stage arms found:", ", ".join(main_names))
    frames = run_score(names=main_names, plan=PLAN, series=series, ctx=ctx)
    print("\n== probe accuracies (mean over folds and seeds) ==")
    print_probes()
    held = [f"{a}_ho" for a in HELD_OUT_ARMS]
    held += [f"{a}_ho_gm" for a in HELD_OUT_ARMS if a in ("E3g", "EH")]
    held_names = [n for n in held if is_complete(name=n)]
    if held_names:
        print("\nheld-out arms found:", ", ".join(held_names))
        run_score(names=held_names, plan=HELD_OUT_PLAN, series=series, ctx=ctx)
    draw_examples(frames=frames, ctx=ctx, series=series)


EXAMPLES: Final[dict[str, str]] = {"D10": "signed-MW Primary", "D4": "MVA-only Primary"}
EXAMPLE_ORIGINS: Final[tuple[str, ...]] = ("2026-01-12", "2026-06-15")


def draw_examples(*, frames: dict, ctx: dict, series: pl.DataFrame) -> None:
    """Draw a day-ahead and a 14-day forecast of the NH arm against the actuals."""
    import altair as alt
    from plotting import ocf_theme as theme

    forecast = frames["NH"]
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    for origin_text in EXAMPLE_ORIGINS:
        origin = datetime.fromisoformat(origin_text).replace(hour=9, tzinfo=UTC)
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


if __name__ == "__main__":
    main()

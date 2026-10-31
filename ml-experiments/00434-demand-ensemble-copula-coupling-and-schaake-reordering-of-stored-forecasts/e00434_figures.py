# ruff: noqa: E501
"""Draw the figures of experiment 434 from the files `e00434_score.py` wrote.

1. `figure_1_variogram_contrasts.svg`: each ordering's variogram score relative to the independent
   draws, per block and lead window, with month-block intervals and the 5% margin;
2. `figure_2_example_*.svg`: a day-ahead and a 14-day forecast of one Primary with its bands;
3. `figure_3_paths.svg`: 12 of the 51 sample paths of one run under three orderings.

```bash
DATA_PATH_INTERNAL=... PYTHONPATH=<folders of experiments 87, 377, 381, 434> \
  uv run --no-sync python ml-experiments/00434-*/e00434_figures.py
```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00434_report as rep  # ty: ignore[unresolved-import]
import e00434_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Path = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / sc.SLUG
WIDTH: Final[int] = 600
ARM_COLOURS: Final[dict[str, str]] = {
    "ECC-T": theme.BRAND_ORANGE,
    "ECC-SW": theme.DATA_PURPLE,
    "SCH": theme.DATA_BLUE,
    "CO": "#1A1A1A",
}
EXAMPLE_PRIMARY: Final[str] = "D5"
EXAMPLE_ORIGINS: Final[dict[str, str]] = {"winter": "2026-01-12", "summer": "2026-06-15"}
BLOCK_NAMES: Final[dict[str, str]] = {
    "spatial": "Across Primaries",
    "temporal": "Across leads",
    "joint": "Both",
}
WINDOW_NAMES: Final[dict[str, str]] = {"days_1_10": "days 1 to 10", "h12_36": "12 to 36 hours"}


def figure_contrasts() -> None:
    """Draw the variogram score of each ordering relative to the independent draws."""
    folds = pl.read_parquet(rep.FIGURE_TABLE_DIR / "fold_means.parquet").filter(
        pl.col("slice") == "all"
    )
    rng = np.random.default_rng(434)
    rows = []
    for block in BLOCK_NAMES:
        for window in WINDOW_NAMES:
            sub = folds.filter((pl.col("block") == block) & (pl.col("window") == window))
            for arm in ARM_COLOURS:
                point, lo, hi = rep.relative_contrast(
                    folds=sub, treat=arm, ref="IND", score="variogram", rng=rng
                )
                rows.append(
                    {
                        "row": f"{BLOCK_NAMES[block]}, {WINDOW_NAMES[window]}",
                        "arm": arm,
                        "change": point * 100,
                        "lo": lo * 100,
                        "hi": hi * 100,
                    }
                )
    data = pl.DataFrame(rows).to_pandas()  # ty: ignore[unresolved-attribute]
    order = [f"{b}, {w}" for b in BLOCK_NAMES.values() for w in WINDOW_NAMES.values()]
    base = alt.Chart(data).encode(
        y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("arm:N", sort=list(ARM_COLOURS)),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
            legend=alt.Legend(orient="top", title=None, columns=4),
        ),
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lo:Q",
            title="Variogram score relative to independent draws (%); lower is more coherent",
        ),
        x2="hi:Q",
    )
    points = base.mark_point(filled=True, size=50).encode(x="change:Q")  # ty: ignore[unresolved-attribute]
    margin = (
        alt.Chart(pl.DataFrame({"x": [-5.0, 5.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#8A8A8A")
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color="#1A1A1A").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    chart = (zero + margin + bars + points).properties(
        width=WIDTH, height=300,
        title=alt.TitleParams(["Ordering by temperature: less coherent across Primaries,", "more coherent across leads. Points: mean over 12 folds.", "Lines: 98.75% month-block intervals. Dashed: the 5% margin."], anchor="start"),
    )  # fmt: skip
    chart.save(DOCS_DIR / "figure_1_variogram_contrasts.svg")


def _run(*, origin: datetime) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return the example Primary's orderings, quantiles (points), actuals, and Schaake dates."""
    series = adapter.series_table()
    ctx = sc.cleaned_context(series=series)
    grid = sc.Grid(series=series, actuals=ctx["actuals"])
    stored = sc.stored_arm_a(month=f"{origin:%Y-%m}")
    ids = series["time_series_id"].to_list()
    order = {sid: i for i, sid in enumerate(ids)}
    sub = (
        stored.filter(pl.col("power_fcst_init_time") == origin)
        .with_columns(p=pl.col("time_series_id").replace_strict(order, return_dtype=pl.Int64))
        .sort("p", "valid_time")
    )
    quantiles = (
        sub.select(sc.LEVELS).to_numpy().reshape(len(ids), sc.N_STEPS, len(sc.LEVELS)) * 100.0
    )
    values = sc.fifty_one_values(quantiles=quantiles)
    t_member, sw_member = sc.run_weather(init=origin, ids=ids)
    import hashlib

    rng = np.random.default_rng(
        int(hashlib.sha256(origin.isoformat().encode()).hexdigest()[:8], 16)
    )
    pool, n_dates = sc.schaake_pool(init=origin, grid=grid)
    identity = np.broadcast_to(np.arange(sc.N_MEMBERS), values.shape)
    ecc_t = sc.assign(values=values, ranks=sc.ranks_by_key(key=-t_member, rng=rng))
    sc.ranks_by_key(key=-sw_member, rng=rng)
    sch = sc.assign(values=values, ranks=sc.ranks_by_key(key=pool.transpose(1, 2, 0), rng=rng))
    ind = sc.assign(values=values, ranks=rng.permuted(identity, axis=-1))
    i0 = grid.index(t=origin) + 1
    actual = grid.values[:, i0 : i0 + sc.N_STEPS]
    p = series["label"].to_list().index(EXAMPLE_PRIMARY)
    paths = {
        "Independent draws": ind[:, p],
        "By temperature rank (ECC-T)": ecc_t[:, p],
        "Schaake shuffle": sch[:, p],
    }
    return paths, quantiles[p], actual[p], np.array(series["capacity_mw"].to_list())[p], n_dates


def figure_examples_and_paths() -> None:
    """Draw the example forecasts with bands, and the sample paths of the winter run."""
    for season, text in EXAMPLE_ORIGINS.items():
        origin = datetime.fromisoformat(text).replace(hour=9, tzinfo=UTC)
        paths, q, actual, _, _ = _run(origin=origin)
        day = (np.arange(1, sc.N_STEPS + 1) * 0.5) / 24.0
        frame = pl.DataFrame(
            {"day": day, "actual": actual, **{c: q[:, i] for i, c in enumerate(sc.LEVELS)}}
        ).to_pandas()  # ty: ignore[unresolved-attribute]
        for horizon, window in (("day-ahead", (1.0, 2.0)), ("14 days", (0.0, 14.0))):
            sub = frame[(frame["day"] >= window[0]) & (frame["day"] <= window[1])]
            x = alt.X(
                "day:Q",
                title="Days after the forecast was made",
                scale=alt.Scale(domain=list(window), nice=False),
            )
            base = alt.Chart(sub).encode(x=x)
            wide = base.mark_area(opacity=0.2, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("q05:Q", title="Demand (% of capacity)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
            )
            narrow = base.mark_area(opacity=0.35, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
                y="q25:Q", y2="q75:Q"
            )
            median = base.mark_line(color=theme.BRAND_ORANGE, strokeWidth=2).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
            obs = base.mark_line(color="black", strokeWidth=1.5).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
            chart = (wide + narrow + median + obs).properties(
                width=WIDTH, height=200,
                title=alt.TitleParams([f"{EXAMPLE_PRIMARY} {horizon} forecast made in {season} (arm A of experiment 425, widened 1.15).", "Orange line: median. Bands: 50% and 90%. Black line: what happened."], anchor="start"),
            )  # fmt: skip
            chart.save(DOCS_DIR / f"figure_2_example_{season}_{horizon.replace(' ', '_')}.svg")
        if season != "winter":
            continue
        panels = []
        obs_frame = pl.DataFrame({"day": day[::2], "demand": actual[::2]}).to_pandas()  # ty: ignore[unresolved-attribute]
        for name, path in paths.items():
            rows = []
            for j in range(12):
                rows += [
                    {"day": d, "demand": v, "path": j}
                    for d, v in zip(day[::2], path[j, ::2], strict=True)
                ]
            data = pl.DataFrame(rows).to_pandas()  # ty: ignore[unresolved-attribute]
            lines = alt.Chart(data).mark_line(opacity=0.45, strokeWidth=0.8, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("day:Q", title="Days after the forecast was made", scale=alt.Scale(domain=[0, 14], nice=False)),
                y=alt.Y("demand:Q", title="Demand (% of capacity)", scale=alt.Scale(zero=False)),
                detail="path:N",
            )  # fmt: skip
            truth = (
                alt.Chart(obs_frame)  # ty: ignore[unresolved-attribute]
                .mark_line(color="black", strokeWidth=1.5)
                .encode(x="day:Q", y="demand:Q")
            )
            panels.append(
                alt.layer(lines, truth).properties(
                    width=WIDTH,
                    height=130,
                    title=alt.TitleParams(name, anchor="start", fontSize=12),
                )
            )
        alt.vconcat(*panels).properties(
            title=alt.TitleParams(
                [
                    f"12 of the 51 sample paths for {EXAMPLE_PRIMARY}, forecast made in winter.",
                    "Blue: sample paths. Black: what happened.",
                ],
                anchor="start",
            )
        ).save(DOCS_DIR / "figure_3_paths.svg")


if __name__ == "__main__":
    figure_contrasts()
    figure_examples_and_paths()

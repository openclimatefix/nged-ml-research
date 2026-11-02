# ruff: noqa: E501
r"""Draw the figures of experiment 462 from the stored scores, sweeps, and forecasts (no fit).

Scores are read from `work/scores/w1_15` (seeds 0 and 1, 1.15 widening). Figures:

- Figure 1: S3-amb minus S3-rw-amb, CRPS, by row set at both metering groups, with the minimum
  detectable effect.
- Figure 2: the share of a paired step that the median follows, by arm, number of lags stepped,
  and mirror, beside the Bayes-optimal shares of the training worlds.
- Figure 3: U1n-2, S3-amb, and S3-rw-amb on a clean origin and on a level change with no mirror.
- Figure 4: the same three arms on six real labelled events made the day before each event starts.

The month-block interval of the sweep gap (no mirror minus mirror) is also written to
`work/figure_tables/sweep_gap.csv`.

```bash
source /tmp/pp462.sh; export DATA_PATH_INTERNAL=<private data folder>
PYTHONPATH=$PP uv run --no-sync python ml-experiments/00462-*/e00462_figures.py
```
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00446_figures as f446  # ty: ignore[unresolved-import]
import e00462_run as run
import e00462_score as score462
import numpy as np
import polars as pl
from plotting import ocf_theme as theme  # noqa: F401

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00462-lag-trust-mirrored-genuine-level-change-worlds"
)
SCORES: Final[Path] = run.WORK_DIR / "scores"
TABLES: Final[Path] = run.WORK_DIR / "figure_tables"
SIGNED: Final[str] = f446.SIGNED
MVA: Final[str] = f446.MVA
WIDTH: Final[int] = f446.WIDTH
POINTS: Final[float] = 100.0
SEEDS: Final[tuple[int, ...]] = (0, 1)
ARMS: Final[list[str]] = ["U1n-2", "S3-amb", "S3-rw-amb"]
COLOURS: Final[list[str]] = ["#FF4901", "#17E58F", "#B701FF"]
ROW_NAMES: Final[dict[str, str]] = {
    "gen_nomirror": "Level change, no mirror",
    "gen_mirror_donor": "Level change, mirrored",
    "gen_recipient": "Received mirror",
    "inj_harm": "Injected load-shift harm rows",
    "clean": "Rows with no labelled event",
    "real_harm": "Real harm rows",
}
BAYES: Final[dict[tuple[int, bool], float]] = {
    (1, False): 0.081, (1, True): 0.028, (2, False): 0.102,
    (2, True): 0.035, (3, False): 0.118, (3, True): 0.043,
}  # fmt: skip
SWEEP_ARMS: Final[list[str]] = ["S1", "U1n-2", "S3-amb", "S3-rw-amb"]


def install() -> None:
    """Point experiment 446's figure helpers at this experiment's folders and arms."""
    score462.install()
    pr.PRED_DIR = run.PRED_DIR
    f446.FIGURE_DIR = DOCS_DIR
    f446.SEEDS = SEEDS
    f446.score = score462.sc
    f446.run = run


def figure_contrasts() -> None:
    """Figure 1: S3-amb minus S3-rw-amb by row set and group, with the minimum detectable effect."""
    c = pl.read_parquet(SCORES / "w1_15" / "contrasts.parquet").drop("fold_differences")
    mde = pl.read_parquet(SCORES / "w1_15" / "mde_pair.parquet").select("row_set", "group", "mde")
    c = (
        c.filter(
            pl.col("metric") == "crps",
            pl.col("contrast") == "S3-amb - S3-rw-amb",
            pl.col("group").is_in([SIGNED, MVA]),
            pl.col("row_set").is_in(list(ROW_NAMES)),
        )
        .join(mde, on=["row_set", "group"], how="left")
        .with_columns(row=pl.col("row_set").replace_strict(ROW_NAMES), neg_mde=-pl.col("mde"))
    )
    order = list(ROW_NAMES.values())
    y = alt.Y("row:N", sort=order, title=None)

    def panel(group: str) -> alt.LayerChart:
        sub = c.filter(pl.col("group") == group).to_pandas()
        d = alt.Chart(sub)
        return f446.titled(
            chart=alt.layer(  # ty: ignore[invalid-argument-type]
                d.mark_rect(opacity=0.12, color="#888", height=14).encode(  # ty: ignore[unresolved-attribute]
                    y=y, x="neg_mde:Q", x2="mde:Q"
                ),  # ty: ignore[unresolved-attribute]
                d.mark_rule(strokeWidth=1.5, color="#17E58F").encode(  # ty: ignore[unresolved-attribute]
                    y=y,
                    x=alt.X(
                        "low:Q",
                        title="S3-amb minus S3-rw-amb, CRPS (points; negative favours S3-amb)",
                    ),
                    x2="high:Q",
                ),  # ty: ignore[unresolved-attribute]
                d.mark_point(filled=True, size=50, color="#17E58F").encode(y=y, x="difference:Q"),  # ty: ignore[unresolved-attribute]
                alt.Chart(pl.DataFrame({"z": [0.0]}).to_pandas())
                .mark_rule(color="#333", strokeDash=[3, 3])
                .encode(x="z:Q"),  # ty: ignore[unresolved-attribute]
            ).properties(width=WIDTH - 330, height=170),
            title=group,
        )

    chart = (
        alt.vconcat(panel(SIGNED), panel(MVA))
        .resolve_scale(x="independent")
        .properties(
            title=alt.TitleParams(
                [
                    "Does seeing the neighbours help? S3-amb minus S3-rw-amb, CRPS (interim)",
                    "Dot: estimate. Line: 95% interval from resampling test-month origin days. Grey: plus or minus the minimum detectable effect.",
                ],
                anchor="start",
            )
        )
    )
    f446.save(chart=chart, name="figure_1_pair_contrasts.svg")


def sweep_gap() -> pl.DataFrame:
    """Return the share followed per arm, k, and mirror, with a block bootstrap of the gap."""
    blocks = pl.read_parquet(SCORES / "sweep_blocks.parquet")
    rng = np.random.default_rng(0)
    rows = []
    for arm in SWEEP_ARMS:
        for k in (1, 2, 3):
            part = blocks.filter(pl.col("arm") == arm, pl.col("k") == k)
            keys = part.select("month", "seed").unique().sort("month", "seed")
            n_blocks = len(keys)
            per = {
                m: part.filter(pl.col("mirror") == m).sort("month", "seed")
                for m in part["mirror"].unique().to_list()
            }
            for mirror, p in per.items():
                share = float(p["share_sum"].sum() / p["n"].sum())  # ty: ignore[unsupported-operator]
                row = {
                    "arm": arm,
                    "k": k,
                    "mirror": mirror,
                    "share": share,
                    "gap": None,
                    "gap_low": None,
                    "gap_high": None,
                }
                if mirror and False in per:
                    s_m = p["share_sum"].to_numpy() / p["n"].to_numpy()
                    s_0 = per[False]["share_sum"].to_numpy() / per[False]["n"].to_numpy()
                    gaps = []
                    for _ in range(2000):
                        pick = rng.integers(0, n_blocks, n_blocks)
                        gaps.append(float(s_0[pick].mean() - s_m[pick].mean()))
                    row.update(
                        gap=float(s_0.mean() - s_m.mean()),
                        gap_low=float(np.quantile(gaps, 0.025)),
                        gap_high=float(np.quantile(gaps, 0.975)),
                    )
                rows.append(row)
    table = pl.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.write_csv(TABLES / "sweep_gap.csv")
    return table


def figure_sweep() -> None:
    """Figure 2: share of a paired step followed, by arm, number of lags stepped, and mirror."""
    t = sweep_gap().with_columns(
        label=pl.when(pl.col("mirror")).then(pl.lit("With a mirror")).otherwise(pl.lit("No mirror"))
    )
    bayes = pl.DataFrame([{"k": k, "mirror": m, "bayes": s} for (k, m), s in BAYES.items()])
    t = t.join(bayes, on=["k", "mirror"], how="left")
    x = alt.X("k:O", title="Lags stepped", axis=alt.Axis(labelAngle=0))
    colour = alt.Color(
        "label:N",
        scale=alt.Scale(domain=["No mirror", "With a mirror"], range=["#306BFF", "#FF4901"]),
        legend=alt.Legend(title=None, orient="top"),
    )
    base = alt.Chart(t.filter(pl.col("arm").is_in(["S3-amb", "S3-rw-amb", "U1n-2"])).to_pandas())
    bars = base.mark_bar(opacity=0.8).encode(  # ty: ignore[unresolved-attribute]
        x=x,
        xOffset="label:N",
        y=alt.Y("share:Q", title="Share of the step the median follows"),
        color=colour,
    )
    ref = base.mark_tick(color="#111", thickness=2, size=22).encode(  # ty: ignore[unresolved-attribute]
        x=x, xOffset="label:N", y="bayes:Q"
    )  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(bars, ref)
        .properties(width=170, height=170)
        .facet(column=alt.Column("arm:N", sort=["S3-amb", "S3-rw-amb", "U1n-2"], title=None))
        .properties(
            title=alt.TitleParams(
                [
                    "Share of a 5% step in the lags that the median forecast follows (interim)",
                    "Black tick: the share that a forecaster knowing the training worlds exactly would follow. U1n-2 sees no neighbour.",
                ],
                anchor="start",
            )
        )
    )
    f446.save(chart=chart, name="figure_2_sweep_shares.svg")


def jump_origin(*, month: str) -> tuple[int, str, int]:
    """Return the world, signed-MW Primary, and origin of the largest unmirrored level change."""
    best: tuple[float, int, str, int] | None = None
    for world in run.GEN_WORLDS:
        f = f446.example_origin(arm="S3-amb", month=month, world=world)
        g = (
            f.filter(
                pl.col("site").is_in(score462.sc.SIGNED_MW_LABELS),
                pl.col("shift_lag"),
                pl.col("level_kind") == 1,
            )
            .group_by("site", "origin_us")
            .agg(
                pl.col("days_since").min().alias("days"),
                (pl.col("actual") - pl.col("actual_clean")).abs().mean().alias("deviation"),
            )
            .filter(pl.col("days") < 1.0)
            .sort("deviation", descending=True)
        )
        if g.is_empty():
            continue
        row = g.row(0, named=True)
        if best is None or row["deviation"] > best[0]:
            best = (float(row["deviation"]), world, row["site"], int(row["origin_us"]))
    assert best is not None, "no origin with an unmirrored level change at the newest lag"
    return best[1], best[2], best[3]


def example_frame(*, month: str, world: int, site: str, origin_us: int) -> pl.DataFrame:
    """Return the three arms' rows for one origin, in points of capacity."""
    parts = [
        f446.example_origin(arm=arm, month=month, world=world).filter(
            pl.col("site") == site, pl.col("origin_us") == origin_us
        )
        for arm in ARMS
    ]
    return pl.concat(parts, how="diagonal").with_columns(
        hours=pl.col("lead").cast(pl.Float64),
        **{q: pl.col(q) * POINTS for q in ("q01", "q05", "q50", "q95", "q99", "actual")},
    )


def figure_examples(*, month: str = "2026-01") -> None:
    """Figure 3: the three arms on a clean origin and on an unmirrored level change."""
    base = f446.example_origin(arm="S3-amb", month=month, world=0)
    clean_site, clean_origin = f446.choose_clean_origin(f=base)
    world, jump_site, jump_origin_us = jump_origin(month=month)
    cases = [
        ("Clean origin", 0, clean_site, clean_origin),
        ("Level change, no mirror", world, jump_site, jump_origin_us),
    ]
    panels = []
    for title, w, site, origin in cases:
        d = example_frame(month=month, world=w, site=site, origin_us=origin)
        for i, arm in enumerate(ARMS):
            pdata = d.filter(pl.col("arm") == arm).sort("hours").to_pandas()
            x = alt.X(
                "hours:Q",
                title="Hours after the forecast origin" if i == len(ARMS) - 1 else None,
                scale=alt.Scale(domain=[0, 240]),
                axis=alt.Axis(values=[0, 48, 96, 144, 192, 240]),
            )
            outer = (
                alt.Chart(pdata)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.12, color=COLOURS[i])
                .encode(
                    x=x,
                    y=alt.Y("q01:Q", title="Power (points)", scale=alt.Scale(zero=False)),
                    y2="q99:Q",
                )
            )  # ty: ignore[unresolved-attribute]
            inner = (
                alt.Chart(pdata)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.28, color=COLOURS[i])
                .encode(x=x, y="q05:Q", y2="q95:Q")
            )  # ty: ignore[unresolved-attribute]
            median = alt.Chart(pdata).mark_line(color=COLOURS[i]).encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
            truth = (
                alt.Chart(pdata)  # ty: ignore[unresolved-attribute]
                .mark_line(color="#333", strokeDash=[2, 2])
                .encode(x=x, y="actual:Q")
            )  # ty: ignore[unresolved-attribute]
            panels.append(
                (outer + inner + median + truth).properties(
                    width=WIDTH - 90,
                    height=110,
                    title=alt.TitleParams(f"{title}: {arm}", anchor="start", fontSize=11),
                )
            )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Forecasts of U1n-2, S3-amb, and S3-rw-amb from one origin (10-day horizon, interim)",
                "Line: median. Dark band: 5% to 95%. Light band: 1% to 99%. Dashed: the reading, which steps at the level change.",
            ],
            anchor="start",
        )
    )
    f446.save(chart=chart, name="figure_3_example_forecasts.svg")


def figure_events() -> None:
    """Figure 4: six real labelled events, three per metering group, forecast the day before."""
    import e00087_common as common  # ty: ignore[unresolved-import]

    times = common.grid_times()
    power, _ = data.load_stack(include_eval=False)
    sites = data.sites()
    month_of = {m: datetime.fromisoformat(f"{m}-01").replace(tzinfo=UTC) for m in f446.MONTHS}
    picked: list[dict] = []
    per_group = {SIGNED: 0, MVA: 0}
    for e in f446.real_events(count=14):
        if per_group[e["group"]] >= 3:
            continue
        month = next(
            (m for m, t in month_of.items() if t <= times[e["start"]] < t + timedelta(days=31)),
            None,
        )
        if month is None:
            continue
        e["month"] = month
        picked.append(e)
        per_group[e["group"]] += 1
    panels = []
    for i, e in enumerate(picked, start=1):
        frames = []
        for arm in ARMS:
            f = score462.sc.read_arm(
                arm=arm,
                month=e["month"],
                world=0,
                seeds=SEEDS,
                factor=score462.sc.score428.WIDENING,
            )
            assert f is not None
            frames.append(
                f.filter(pl.col("primary") == sites.index(e["site"])).with_columns(arm=pl.lit(arm))
            )
        f = pl.concat(frames, how="diagonal").with_columns(
            time=pl.from_epoch("valid_us", time_unit="us")
        )
        t0 = times[e["start"]] - timedelta(days=3)
        t1 = times[min(e["end"], len(times) - 1)] + timedelta(days=2)
        want = int((times[e["start"]] - timedelta(days=1)).timestamp() * 1_000_000)
        origins = np.array(sorted(f["origin_us"].unique().to_list()))
        if not len(origins):
            continue
        origin = int(origins[np.argmin(np.abs(origins - want))])
        d = f.filter(pl.col("origin_us") == origin, pl.col("lead") <= 120).with_columns(
            q05=pl.col("q05") * POINTS, q50=pl.col("q50") * POINTS, q95=pl.col("q95") * POINTS
        )
        grid = (
            pl.DataFrame({"time": times})
            .with_columns(actual=pl.Series(power[0][sites.index(e["site"])] * POINTS))
            .filter(pl.col("time").is_between(t0, t1))
            .drop_nulls()
            .drop_nans()
        )
        colour = f446.arm_colour(arms=ARMS, colours=COLOURS, legend=i == 1)
        x = alt.X("time:T", title=None, scale=alt.Scale(domain=[t0, t1]))
        pdata = d.to_pandas()
        shade = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                pl.DataFrame(
                    {"a": [times[e["start"]]], "b": [times[min(e["end"], len(times) - 1)]]}
                ).to_pandas()
            )
            .mark_rect(opacity=0.12, color="#888")
            .encode(x="a:T", x2="b:T")
        )  # ty: ignore[unresolved-attribute]
        band = (
            alt.Chart(pdata)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.14)
            .encode(
                x=x,
                y=alt.Y("q05:Q", title="Power (points)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
                color=colour,
            )
        )  # ty: ignore[unresolved-attribute]
        median = alt.Chart(pdata).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
        truth = alt.Chart(grid.to_pandas()).mark_line(color="#333").encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            (shade + band + median + truth).properties(
                width=WIDTH - 90,
                height=130,
                title=f"Event {i}: Primary {e['site']} ({e['group']}), forecast from the day before the event starts",
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Labelled load-shift events: U1n-2, S3-amb, and S3-rw-amb forecasts made the day before",
                "Shaded: the padded event. Black: the cleaned reading. Events are labelled by hand, and some real events are not logged.",
            ],
            anchor="start",
        )
    )
    f446.save(chart=chart, name="figure_4_real_events.svg")


def main() -> None:
    """Draw every figure."""
    install()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts()
    figure_sweep()
    figure_examples()
    figure_events()


if __name__ == "__main__":
    main()

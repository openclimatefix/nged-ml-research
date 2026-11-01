# ruff: noqa: E501, C408
"""Experiment 482: tables and charts of the injected-scenario forecasts.

Reads `work/summary/*.parquet` and `work/rows/*.parquet` written by `e00482_run.py`. Writes, in the
experiment's private `report/` folder: `follow_share.csv` (how far the median moved with the shift
on rows whose newest lag is inside the event), `crps_by_scenario.csv` (score on rows with a lag
inside the event, with a cluster bootstrap over test month and donor), `coverage_by_scenario.csv`,
and the charts. Donors only (the scored Primary is the one shifted); recipients are in the
summary file for the mirror scenarios but are not scored here.

```bash
source <paths of ml-experiments/*/ on PYTHONPATH>; export DATA_PATH_INTERNAL=<private data folder>
uv run --no-sync python ml-experiments/00482-*/e00482_report.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00482_scenarios as scn
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00482-best-models-performance-analysis-injected-shifts-and-wind-references"
)
OUT: Final[Path] = ROOT / "report"
ARMS: Final[tuple[str, ...]] = ("LL", "S1", "B-mix", "K2-S1", "A-pool")
LABEL: Final[dict[str, str]] = {
    "LL": "Lagless network", "S1": "S1 (lagged)", "B-mix": "B-mix", "K2-S1": "K2-S1",
    "A-pool": "A-pool (never saw a shift)",
}  # fmt: skip
COLOURS: Final[dict[str, str]] = {
    "Lagless network": ocf.DATA_SKY, "S1 (lagged)": ocf.DATA_PURPLE, "B-mix": ocf.DATA_GREEN,
    "K2-S1": ocf.BRAND_ORANGE, "A-pool (never saw a shift)": ocf.DATA_MAGENTA,
}  # fmt: skip
LAG_CLASSES: Final[tuple[str, ...]] = ("newest_in", "older_in")
SITES: Final[list[str]] = prep.primaries()["site"].to_list()
DRAWS: Final[int] = 4000
WIDTH: Final[int] = 640
NAMES: Final[dict[int, str]] = {i + 1: s.name for i, s in enumerate(scn.SCENARIOS)}


def donors() -> pl.DataFrame:
    """Return the donor rows of every scenario, raw and widened."""
    files = sorted((ROOT / "work" / "summary").glob("*.parquet"))
    d = pl.concat([pl.read_parquet(f) for f in files])
    return d.filter(
        (pl.col("role") == "donor") & (pl.col("scenario") > 0) & pl.col("arm").is_in(ARMS)
    )


def pooled(*, d: pl.DataFrame, classes: tuple[str, ...], by: list[str]) -> pl.DataFrame:
    """Return the row-weighted means of the score columns over the chosen classes."""
    sel = d.filter(pl.col("cls").is_in(classes))
    cols = ["crps", "mae", "cov_q01", "cov_q05", "cov_q95", "cov_q99", "width90", "dq50"]
    return sel.group_by(*by).agg(
        n=pl.col("n").sum(),
        **{c: (pl.col(c) * pl.col("n")).sum() / pl.col("n").sum() for c in cols},
    )


def follow_share(*, d: pl.DataFrame) -> pl.DataFrame:
    """Return the share of the shift the median followed, on rows whose newest lag is inside."""
    sel = d.filter(
        (pl.col("cls") == "newest_in") & (pl.col("variant") == "raw") & (pl.col("d_points") != 0)
    )
    per = sel.with_columns(share=pl.col("dq50") / pl.col("d_points"))
    return (
        per.group_by("scenario", "arm", "group")
        .agg(
            n=pl.col("n").sum(),
            share=(pl.col("share") * pl.col("n")).sum() / pl.col("n").sum(),
        )
        .sort("scenario", "group", "arm")
    )


def cluster_contrast(
    *, d: pl.DataFrame, scenario: int, arm: str, ref: str, rng: np.random.Generator, level: float
) -> dict:
    """Return `arm` minus `ref` CRPS on rows with a lag inside, cluster bootstrap over (month, donor)."""
    sel = d.filter(
        (pl.col("scenario") == scenario)
        & (pl.col("variant") == "raw")
        & pl.col("cls").is_in(LAG_CLASSES)
    )
    w = sel.group_by("month", "donor", "arm").agg(
        n=pl.col("n").sum(), s=(pl.col("crps") * pl.col("n")).sum()
    )
    a = w.filter(pl.col("arm") == arm).sort("month", "donor")
    b = w.filter(pl.col("arm") == ref).sort("month", "donor")
    n = a["n"].to_numpy().astype(float)
    diff = a["s"].to_numpy() - b["s"].to_numpy()
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = rng.integers(0, len(n), size=len(n))
        draws[i] = diff[pick].sum() / n[pick].sum()
    lo, hi = np.percentile(draws, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return dict(
        scenario=scenario,
        arm=arm,
        ref=ref,
        contrast=float(diff.sum() / n.sum()),
        low=float(lo),
        high=float(hi),
        n=int(n.sum()),
        clusters=len(n),
    )


def main() -> None:
    """Write the tables and the summary charts."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "png").mkdir(exist_ok=True)
    d = donors()
    d.write_parquet(OUT / "donor_summary.parquet")
    fs = follow_share(d=d)
    fs.write_csv(OUT / "follow_share.csv")
    raw = d.filter(pl.col("variant") == "raw")
    cr = pooled(d=raw, classes=LAG_CLASSES, by=["scenario", "arm"]).sort("scenario", "arm")
    cr.write_csv(OUT / "crps_by_scenario.csv")
    by_group = pooled(d=raw, classes=LAG_CLASSES, by=["scenario", "arm", "group"]).sort(
        "scenario", "group", "arm"
    )
    by_group.write_csv(OUT / "crps_by_scenario_and_group.csv")
    by_class = pooled(
        d=raw, classes=("newest_in", "older_in", "before", "after"), by=["scenario", "arm", "cls"]
    ).sort("scenario", "cls", "arm")
    by_class.write_csv(OUT / "crps_by_scenario_and_class.csv")
    wide = d.filter(pl.col("variant") == "w1.15")
    pooled(d=wide, classes=LAG_CLASSES, by=["scenario", "arm"]).sort("scenario", "arm").write_csv(
        OUT / "crps_by_scenario_widened.csv"
    )
    rng = np.random.default_rng(0)
    pairs = [
        ("K2-S1", "LL"),
        ("K2-S1", "B-mix"),
        ("K2-S1", "A-pool"),
        ("S1", "LL"),
        ("B-mix", "LL"),
    ]
    level = 1 - 0.05 / (len(pairs) * len(scn.SCENARIOS))
    rows = [
        cluster_contrast(d=d, scenario=s, arm=a, ref=r, rng=rng, level=level)
        for s in range(1, 9)
        for a, r in pairs
    ]
    contrasts = pl.DataFrame(rows)
    contrasts.write_csv(OUT / "crps_contrasts.csv")
    print(f"Bonferroni count {len(pairs) * len(scn.SCENARIOS)}, level {level:.5f}")
    # headline chart: CRPS contrast with LL per scenario
    long = cr.with_columns(
        label=pl.col("arm").replace(LABEL),
        name=pl.col("scenario").replace_strict(NAMES, return_dtype=pl.String),
    )
    chart = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            y=alt.Y("name:N", title=None, sort=list(NAMES.values())),
            x=alt.X(
                "crps:Q",
                title="CRPS on rows with a lag inside the event (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "label:N",
                title=None,
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(orient="top", columns=3),
            ),
        )
        .properties(
            width=WIDTH - 250,
            height=230,
            title=alt.TitleParams(
                "Score of each forecast on rows whose lags fall inside an injected event, by scenario",
                anchor="start",
            ),
        )
    )
    chart.save(OUT / "crps_by_scenario.svg")
    chart.save(OUT / "png" / "crps_by_scenario.png", scale_factor=1.5)
    # follow-share chart
    f2 = (
        fs.group_by("scenario", "arm")
        .agg(share=(pl.col("share") * pl.col("n")).sum() / pl.col("n").sum())
        .with_columns(
            label=pl.col("arm").replace(LABEL),
            name=pl.col("scenario").replace_strict(NAMES, return_dtype=pl.String),
        )
    )
    c2 = (
        alt.Chart(f2)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            y=alt.Y("name:N", title=None, sort=list(NAMES.values())),
            x=alt.X(
                "share:Q",
                title="Share of the shift that the median moved (rows whose newest lag is inside the event)",
            ),
            color=alt.Color(
                "label:N",
                title=None,
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(orient="top", columns=3),
            ),
        )
        .properties(
            width=WIDTH - 250,
            height=230,
            title=alt.TitleParams(
                "How far each median followed the injected shift", anchor="start"
            ),
        )
    )
    c2.save(OUT / "follow_share.svg")
    c2.save(OUT / "png" / "follow_share.png", scale_factor=1.5)
    fan_charts()
    print(cr)
    print(
        fs.group_by("scenario", "arm")
        .agg(share=(pl.col("share") * pl.col("n")).sum() / pl.col("n").sum())
        .sort("scenario", "arm")
    )
    print(contrasts)


def fan_charts() -> None:
    """Draw day-ahead fans of the chart donors for every stored scenario world."""
    for f in sorted((ROOT / "work" / "rows").glob("*.parquet")):
        month, sid, donor = f.stem.split("_")
        r = pl.read_parquet(f)
        a, b = int(r["a"][0]), int(r["b"][0])
        d = (
            r.filter((pl.col("lead_hours") >= 12) & (pl.col("lead_hours") < 36))
            .unique(subset=["primary", "target"], keep="first")
            .sort("target")
        )
        # keep the donor's rows: the donor is the Primary whose actual window is shifted; rows hold donor and recipients
        d = d.filter(pl.col("primary") == SITES.index(donor))
        d = d.filter(pl.col("target").is_between(a - 3 * 48, b + 4 * 48)).with_columns(
            day=(pl.col("target") - a) / 48.0
        )
        if d.height == 0:
            continue
        lines = [
            d.select("day", v=(pl.col("y_world") * 100).cast(pl.Float64), series=pl.lit("Actual"))
        ]
        if ((d["y_world"] - d["y_clean"]).abs() > 1e-9).any():
            lines.append(
                d.select(
                    "day",
                    v=(pl.col("y_clean") * 100).cast(pl.Float64),
                    series=pl.lit("Actual without the level change"),
                )
            )
        lines.extend(
            d.select(
                "day", v=(pl.col(f"{arm}_q50") * 100).cast(pl.Float64), series=pl.lit(LABEL[arm])
            )
            for arm in ARMS
        )
        long = pl.concat(lines)
        colours = {"Actual": ocf.BLACK_1, "Actual without the level change": ocf.GREY_1, **COLOURS}
        x = alt.X("day:Q", title="Days since the event began (shaded: the event)")
        shade = (
            alt.Chart(pl.DataFrame({"a": [0.0], "b": [(b - a) / 48.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rect(color=ocf.GREY_3, opacity=0.35, clip=True)
            .encode(x="a:Q", x2="b:Q")
        )  # ty: ignore[unresolved-attribute]
        band = (
            alt.Chart(d.select("day", lo=pl.col("K2-S1_q05") * 100, hi=pl.col("K2-S1_q95") * 100))  # ty: ignore[unresolved-attribute]
            .mark_area(color=ocf.BRAND_ORANGE, opacity=0.18)
            .encode(x=x, y=alt.Y("lo:Q", title="Power (% of capacity)"), y2="hi:Q")
        )  # ty: ignore[unresolved-attribute]
        line = alt.Chart(long).mark_line(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=x, y="v:Q",
            color=alt.Color("series:N", scale=alt.Scale(domain=list(colours), range=list(colours.values())), legend=alt.Legend(title=None, orient="top", columns=3)),
            strokeWidth=alt.condition(alt.datum.series == "Actual", alt.value(2.0), alt.value(1.2)),
        )  # fmt: skip
        name = NAMES[int(sid[1:])]
        chart = (shade + band + line).properties(
            width=WIDTH - 110,
            height=190,
            title=alt.TitleParams(
                [
                    f"Primary {donor}, {month} fit, scenario {name}.",
                    "Day-ahead forecasts (12 to 36 hours ahead); orange band: K2-S1 90% interval.",
                ],
                anchor="start",
            ),
        )
        chart.save(OUT / f"fan_{month}_{sid}_{donor}.svg")
        chart.save(OUT / "png" / f"fan_{month}_{sid}_{donor}.png", scale_factor=1.5)


if __name__ == "__main__":
    main()

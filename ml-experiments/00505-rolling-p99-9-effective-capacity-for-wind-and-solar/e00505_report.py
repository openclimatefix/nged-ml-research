"""Experiment 505: draw the figures and print the capacity and score tables for the write-up.

Reads `work/cells.parquet` from `e00505_score.py`, the capacity JSON files of experiments 461 and
505, and experiment 453's `series.parquet`. Writes SVG figures to the experiment's docs folder and
the printed tables to `work/report505.txt`. Only Farm A to C and Solar A to F are named.
The bootstrap intervals of the full analysis are in `work/report.txt`; figure 2 redraws
the per-series intervals with a month-block resample of the same cells.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00429-*/env.sh   # plus the 00461 and 00505 folders
uv run --no-sync python ml-experiments/00505-*/e00505_report.py
```
"""

import json
import subprocess
from pathlib import Path
from typing import Final

import altair as alt
import e00505_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00505-rolling-p99-9-effective-capacity-for-wind-and-solar"
)
WORK: Final[Path] = common.DATA_DIR / "work"
WIND: Final[list[str]] = ["Farm A", "Farm B", "Farm C"]
SOLAR: Final[list[str]] = ["Solar A", "Solar B", "Solar C", "Solar D", "Solar E", "Solar F"]
SERIES: Final[list[str]] = WIND + SOLAR
LINE: Final[dict[str, str]] = {
    "p999": "all earlier power", "p730": "last 730 days", "p365": "last 365 days",
}
RULES: Final[list[str]] = ["old", "p999", "p730", "p365"]
LABEL: Final[dict[str, str]] = {
    "old": "P-old",
    "p999": "P-all",
    "p730": "P-730",
    "p365": "P-365",
}
COLOUR: Final[dict[str, str]] = {
    "old": ocf.DATA_BLUE,
    "p999": ocf.DATA_SKY,
    "p730": ocf.BRAND_ORANGE,
    "p365": ocf.DATA_PURPLE,
}
DRAWS: Final[int] = 2000
WIDTH: Final[int] = 680
PLOT_W: Final[int] = WIDTH - 110
TEXT_W: Final[int] = 100
SOLAR_REFITS: Final[list[str]] = ["2025-09", "2025-12", "2026-03", "2026-06"]
OUT: Final[list[str]] = []


def say(*parts: object) -> None:
    """Print a line and keep it for `work/report505.txt`."""
    line = " ".join(str(p) for p in parts)
    print(line)
    OUT.append(line)


def arm_of(*, site: str) -> str:
    """Return the headline arm of a series' technology."""
    return "member_plain" if site.startswith("Farm") else "a2_member"


def sub(*, cells: pl.DataFrame, site: str, rule: str) -> pl.DataFrame:
    """Return a series' rule's table-P cells, sorted by month."""
    return cells.filter(
        (pl.col("site") == site) & (pl.col("arm") == arm_of(site=site))
        & (pl.col("rule") == rule) & (pl.col("variant") == "P")
    ).sort("month")  # fmt: skip


def capacity_table() -> None:
    """Print the capacity ratios (new over old) and window counts per series and refit."""
    say("(a) capacity table: ratio new/old (n_valid, n_above, * = fell back to P-all)")
    say("series | refit | P-all ratio (n_history) | P-730 ratio | P-365 ratio | P-730 n_valid, n_above | P-365 n_valid, n_above")  # fmt: skip
    all_w = json.loads((common.EXP461_DIR / "wind_capacity.json").read_text())
    all_s = json.loads((common.EXP461_DIR / "solar_capacity.json").read_text())
    roll = {
        (tech, w): json.loads((common.DATA_DIR / f"{tech}_capacity_{w}.json").read_text())
        for tech in ("wind", "solar")
        for w in common.WINDOWS
    }

    def row(*, name: str, refit: str, a: dict, r730: dict, r365: dict) -> None:
        def f(r: dict) -> str:
            return f"{r['ratio']:.3f}{'*' if r['fallback'] else ''}"

        say(
            f"{name} | {refit} | {a['ratio']:.3f} ({a['n_history']}) | {f(r730)} | {f(r365)} | "
            f"{r730['n_valid']}, {r730['n_above']} | {r365['n_valid']}, {r365['n_above']}"
        )

    for key, name in common.WIND_LABEL.items():
        row(name=name, refit="one cutoff", a=all_w[key],
            r730=roll["wind", "p730"][key], r365=roll["wind", "p365"][key])  # fmt: skip
    for refit in SOLAR_REFITS:
        for letter in "ABCDEF":
            row(name=f"Solar {letter}", refit=refit, a=all_s[refit][letter],
                r730=roll["solar", "p730"][refit][letter],
                r365=roll["solar", "p365"][refit][letter])  # fmt: skip


def score_change_tables(*, cells: pl.DataFrame) -> pl.DataFrame:
    """Print the change in mean score (MW and %) of each rolling arm; return month-block rows."""
    rng = np.random.default_rng(0)
    say("")
    say("(b) change in mean quantile score after table P: rolling arm minus reference")
    say("series | arm | ref | change MW | change % | 95% month-block interval (MW)")
    rows = []
    for site in SERIES:
        tables = {r: sub(cells=cells, site=site, rule=r) for r in RULES}
        n = tables["old"]["n"].to_numpy().astype(float)
        for arm in ("p999", "p730", "p365"):
            for ref in ("old", "p999"):
                if arm == ref or (arm == "p999" and ref == "p999"):
                    continue
                s_new = tables[arm]["score_mw_sum"].to_numpy()
                s_ref = tables[ref]["score_mw_sum"].to_numpy()
                nn = tables[arm]["n"].to_numpy().astype(float)
                point = s_new.sum() / nn.sum() - s_ref.sum() / n.sum()
                draws = np.empty(DRAWS)
                for d in range(DRAWS):
                    pick = rng.integers(0, len(n), size=len(n))
                    draws[d] = s_new[pick].sum() / nn[pick].sum() - s_ref[pick].sum() / n[pick].sum()
                lo, hi = np.percentile(draws, [2.5, 97.5])
                pct = point / (s_ref.sum() / n.sum()) * 100
                say(f"{site} | {arm} | {ref} | {point:+.4f} | {pct:+.2f} | [{lo:+.4f}, {hi:+.4f}]")
                rows.append({"series": site, "arm": arm, "ref": ref, "change": float(point),
                             "low": float(lo), "high": float(hi)})  # fmt: skip
    return pl.DataFrame(rows)


def share_table(*, cells: pl.DataFrame) -> pl.DataFrame:
    """Print the share of actuals above q99 after table P, per series and rule."""
    say("")
    say("(c) share of actuals above q99 after table P (%)")
    say("series | " + " | ".join(RULES))
    rows = []
    for site in SERIES:
        vals = []
        for rule in RULES:
            t = sub(cells=cells, site=site, rule=rule)
            share = float((t["n"] - t["hit_q99"]).sum() / t["n"].sum() * 100)
            vals.append(f"{share:.2f}")
            rows.append({"series": site, "rule": LABEL[rule], "share": share})
        say(f"{site} | " + " | ".join(vals))
    return pl.DataFrame(rows)


def panels(*, make, wind_h: int, solar_h: int, title: list[str]) -> alt.VConcatChart:
    """Stack a wind panel over a solar panel, with a shared title."""
    return alt.vconcat(
        make(WIND).properties(height=wind_h, width=PLOT_W),
        make(SOLAR).properties(height=solar_h, width=PLOT_W),
    ).properties(title=title)


def colour_enc(*, rules: list[str]) -> alt.Color:
    """Return the colour encoding of a rule column, with the key above the plot."""
    return alt.Color(
        "rule:N", title=None,
        scale=alt.Scale(domain=[LABEL[r] for r in rules], range=[COLOUR[r] for r in rules]),
        legend=alt.Legend(orient="top", columns=2, symbolType="circle"),
    )  # fmt: skip


def fig_share(*, share: pl.DataFrame) -> None:
    """Figure 1: share above q99 after table P per series and rule."""
    colour = colour_enc(rules=RULES)

    def make(order: list[str]) -> alt.LayerChart:
        data = share.filter(pl.col("series").is_in(order))
        base = alt.Chart(data).encode(y=alt.Y("series:N", sort=order, title=None))
        dots = base.mark_circle(size=70, opacity=1).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("share:Q", title="Share of actuals above q99 (%; smaller is better)"),
            color=colour, yOffset=alt.YOffset("rule:N"),
        )  # fmt: skip
        limit = alt.Chart(pl.DataFrame({"x": [1.5]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        return dots + limit

    chart = panels(
        make=make, wind_h=110, solar_h=230,
        title="Share of actuals above q99, by capacity rule",
    )  # fmt: skip
    chart.save(DOCS / "figure_1_share_above_q99.svg")


def fig_score(*, change: pl.DataFrame) -> None:
    """Figure 2: score change in MW against P-old, per series and rolling or P-all rule."""
    data = change.filter(pl.col("ref") == "old").with_columns(
        rule=pl.col("arm").replace_strict(LABEL)
    )
    rules = ["p999", "p730", "p365"]
    colour = colour_enc(rules=rules)

    def make(order: list[str]) -> alt.LayerChart:
        d = data.filter(pl.col("series").is_in(order))
        base = alt.Chart(d).encode(y=alt.Y("series:N", sort=order, title=None))
        bar = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Quantile score minus P-old's (MW; negative is better)"),
            x2="high:Q", color=colour, yOffset=alt.YOffset("rule:N"),
        )  # fmt: skip
        dot = base.mark_circle(size=60, opacity=1).encode(  # ty: ignore[unresolved-attribute]
            x="change:Q", color=colour, yOffset=alt.YOffset("rule:N")
        )
        zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule().encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        return bar + dot + zero

    chart = panels(
        make=make, wind_h=100, solar_h=210,
        title="Score change against P-old (MW)",
    )  # fmt: skip
    chart.save(DOCS / "figure_2_score_change.svg")


def trailing(*, times: np.ndarray, power: np.ndarray, at: np.ndarray, days: int | None) -> list:
    """Return the p99.9 of power in [t - days, t) at each t in `at` (all earlier power if None)."""
    out = []
    for t in at:
        lo = 0 if days is None else np.searchsorted(times, t - np.timedelta64(days, "D"))
        hi = np.searchsorted(times, t)
        out.append(float(np.quantile(power[lo:hi], 0.999)) if hi - lo > 1000 else None)
    return out


def fig_over_time() -> None:
    """Figure 3: trailing p99.9 and monthly maximum power as a fraction of the old capacity."""
    series = pl.read_parquet(common.SERIES_PATH).sort("time")
    frames = []
    for site in SERIES:
        s = series.filter(pl.col("site") == site)
        old = float(s["cap"][0])
        times = s["time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[ns]")
        power = s["power"].to_numpy().astype(float)
        weeks = np.arange(times[0] + np.timedelta64(400, "D"), times[-1], np.timedelta64(7, "D"))
        for key, days, label in (("p999", None, "all earlier power"), ("p730", 730, "last 730 days"),
                                 ("p365", 365, "last 365 days")):  # fmt: skip
            vals = trailing(times=times, power=power, at=weeks, days=days)
            frames.append(pl.DataFrame({
                "series": site, "time": weeks.astype("datetime64[us]"),
                "value": [v / old if v is not None else None for v in vals],
                "line": LINE[key],
            }))  # fmt: skip
        last_month = s["time"].max().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly = (
            s.filter(pl.col("time") < last_month)
            .with_columns(month=pl.col("time").dt.truncate("1mo").dt.replace_time_zone(None))
            .group_by("month").agg(pl.col("power").max() / old)
        )
        frames.append(
            monthly.rename({"month": "time", "power": "value"})
            .with_columns(series=pl.lit(site), line=pl.lit("monthly maximum power"))
            .select("series", "time", "value", "line")
        )
    data = pl.concat(frames, how="vertical_relaxed").drop_nulls("value")
    domain = ["all earlier power", "last 730 days", "last 365 days", "monthly maximum power"]
    colour = alt.Color(
        "line:N", title=None,
        scale=alt.Scale(domain=domain,
                        range=[ocf.DATA_SKY, ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.GREY_3]),
        legend=alt.Legend(orient="top", columns=4, symbolType="stroke"),
    )  # fmt: skip
    chart = (
        alt.Chart(data)
        .mark_line(clip=True)  # ty: ignore[unresolved-attribute]
        .encode(
            x=alt.X("time:T", title="Year", axis=alt.Axis(format="%Y", values=[{"year": y} for y in (2020, 2022, 2024, 2026)])),
            y=alt.Y("value:Q", title="Share of old capacity", scale=alt.Scale(zero=False)),
            color=colour,
            strokeWidth=alt.condition(
                alt.datum.line == "monthly maximum power", alt.value(0.7), alt.value(2.2)
            ),
        )
        .properties(width=(PLOT_W - 40) // 3, height=110)
        .facet(facet=alt.Facet("series:N", sort=SERIES, title=None), columns=3)
        .properties(title="Trailing p99.9 and monthly maximum")
    )  # fmt: skip
    chart.save(DOCS / "figure_3_trailing_p999.svg")


def pick_run(*, rows: pl.DataFrame, site: str, month: str) -> pl.DataFrame:
    """Return one full run (true days 1 to 10) of a series in a month, sorted by lead."""
    part = rows.filter((pl.col("site") == site) & (pl.col("month") == month))
    counts = part.group_by("init").agg(pl.len().alias("n")).sort("n", descending=True)
    full = counts.filter(pl.col("n") == counts["n"].max()).sort("init")["init"]
    return part.filter(pl.col("init") == full[len(full) // 2]).sort("lead_hours")


def fan(
    *, run: pl.DataFrame, ref: pl.DataFrame, lo_h: float, hi_h: float, title: str, height: int
) -> alt.LayerChart:
    """Return a fan chart: P-730 median, 50% and 90% bands, actual, and the P-all median."""
    part = run.filter(pl.col("lead_hours").is_between(lo_h, hi_h))
    other = ref.filter(pl.col("lead_hours").is_between(lo_h, hi_h))
    base = alt.Chart(part).encode(
        x=alt.X("lead_hours:Q", scale=alt.Scale(domain=[lo_h, hi_h], nice=False),
                title="Hours after the run was issued")
    )  # fmt: skip

    def band(low: str, high: str, opacity: float) -> alt.Chart:
        return base.mark_area(color=COLOUR["p730"], opacity=opacity, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(f"{low}:Q", title="Power (MW)"), y2=f"{high}:Q"
        )

    return (
        band("q05", "q95", 0.2)
        + band("q25", "q75", 0.3)
        + base.mark_line(color=COLOUR["p730"], aria=False).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(other).mark_line(color=COLOUR["old"], strokeDash=[4, 3], strokeWidth=1, aria=False)  # ty: ignore[unresolved-attribute]
        .encode(x="lead_hours:Q", y="q50:Q")
        + base.mark_line(color="black", aria=False).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=PLOT_W, height=height, title=title)


def fig_examples() -> None:
    """Figures 4 and 5: day-ahead and ten-day runs for Farm A and Solar A, winter and summer."""
    spec = [("Farm A", "member_plain", "2026-01", "winter"), ("Farm A", "member_plain", "2025-07", "summer"),
            ("Solar A", "a2_member", "2025-12", "winter"), ("Solar A", "a2_member", "2026-06", "summer")]  # fmt: skip
    day, horizon = [], []
    for site, arm, month, season in spec:
        run = pick_run(rows=pl.read_parquet(WORK / f"rows_{arm}_p730.parquet"), site=site, month=month)
        full = pl.read_parquet(WORK / f"rows_{arm}_p999.parquet")
        ref = full.filter((pl.col("site") == site) & (pl.col("init") == run["init"][0]))
        for out, lo, hi, kind in ((day, 24, 48, "day ahead"), (horizon, 24, 240, "days 1 to 10")):
            out.append(fan(run=run, ref=ref, lo_h=lo, hi_h=hi, height=110,
                           title=f"{site}, one {season} run: {kind}"))  # fmt: skip
    note = "Black: actual. Orange: P-730. Dashed: P-all median"
    for name, charts in (("figure_4_day_ahead.svg", day), ("figure_5_ten_day_horizon.svg", horizon)):
        alt.vconcat(*charts).properties(title=note).save(DOCS / name)


def main() -> None:
    """Print the tables and draw every figure."""
    DOCS.mkdir(parents=True, exist_ok=True)
    cells = pl.read_parquet(WORK / "cells.parquet")
    capacity_table()
    change = score_change_tables(cells=cells)
    share = share_table(cells=cells)
    (WORK / "report505.txt").write_text("\n".join(OUT) + "\n")
    fig_share(share=share)
    fig_score(change=change)
    fig_over_time()
    fig_examples()
    for svg in sorted(DOCS.glob("figure_*.svg")):
        subprocess.run(
            ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(svg)],
            check=True,
        )


if __name__ == "__main__":
    main()

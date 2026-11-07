"""Draw example forecasts for experiment 626: arms A and D for one Primary, winter and summer.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00626-*/env.sh
uv run --no-sync python ml-experiments/00626-*/e00626_examples.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00626_common as c  # ty: ignore[unresolved-import]
import e00626_score as sc  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / c.SLUG
EXAMPLES: Final[dict[str, str]] = {"winter": "2026-01-13", "summer": "2026-06-16"}
NODE: Final[int] = 1
"""Primary D1 (the second Primary in table order)."""
COLOURS: Final[dict[str, str]] = {
    "A unchanged": theme.DATA_BLUE,
    "D two factors": theme.BRAND_ORANGE,
}


def main() -> None:
    """Write one figure per season, with a day-ahead and a 14-day panel."""
    factors = pl.read_parquet(c.WORK_626 / "factors.parquet")
    for season, date in EXAMPLES.items():
        run = c.load_run(path=c.WORK_436 / "runs" / f"{date}.npz")
        fac = sc.factor_arrays(factors=factors, month=date[:7])
        values = run["values"]
        med = values[..., c.MID : c.MID + 1]
        d = c.clip_physical(
            x=c.rescaled(x=values, med=med, up=fac["d_up"], down=fac["d_down"]), kind=c.KIND
        )
        times = c.valid_times(init=run["init"])
        rows = []
        for name, v in (("A unchanged", values), ("D two factors", d)):
            lo, hi = c.weibull(values=v, level=0.05)[NODE], c.weibull(values=v, level=0.95)[NODE]
            for i in range(480):
                rows.append(  # noqa: PERF401
                    {
                        "time": times[i],
                        "arm": name,
                        "lo": float(lo[i]),
                        "hi": float(hi[i]),
                        "median": float(v[NODE, i, c.MID]),
                        "lead_h": (i + 1) * 0.5,
                    }
                )
        df = pl.DataFrame(rows)
        actual = pl.DataFrame(
            {
                "time": times[:480],
                "actual": run["actual"][NODE, :480],
                "lead_h": [(i + 1) * 0.5 for i in range(480)],
            }
        )
        panels = []
        for title, limit in (("Day ahead", 48.0), ("Ten days", 240.0)):
            sub, act = (
                df.filter(pl.col("lead_h") <= limit),
                actual.filter(pl.col("lead_h") <= limit),
            )
            d_band = sub.filter(pl.col("arm") == "D two factors")
            a_edges = (
                sub.filter(pl.col("arm") == "A unchanged")
                .select("time", "lo", "hi")
                .unpivot(index="time", variable_name="edge", value_name="power")
                .with_columns(series=pl.lit("A unchanged: q05 and q95"))
            )
            median = sub.filter(pl.col("arm") == "A unchanged").select(
                "time",
                pl.col("median").alias("power"),
                pl.lit("Median (same in both arms)").alias("series"),
            )
            observed = act.drop_nulls().select(
                "time", pl.col("actual").alias("power"), pl.lit("Actual").alias("series")
            )
            lines = pl.concat(
                [
                    a_edges.select("time", "power", "series", pl.col("edge")),
                    median.with_columns(edge=pl.lit("m")),
                    observed.with_columns(edge=pl.lit("o")),
                ]
            ).with_columns(key=pl.col("series") + pl.col("edge"))
            domain = ["A unchanged: q05 and q95", "Median (same in both arms)", "Actual"]
            colours = [theme.DATA_BLUE, "#7A7A7A", "#000000"]
            band = (
                alt.Chart(d_band.to_pandas())
                .mark_area(opacity=0.3, color=theme.BRAND_ORANGE)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X("time:T", title="Valid time (UTC)"),
                    y=alt.Y(
                        "lo:Q", title="Power (points of capacity)", scale=alt.Scale(zero=False)
                    ),
                    y2="hi:Q",
                )
            )
            drawn = (
                alt.Chart(lines.to_pandas())
                .mark_line(strokeWidth=1.5)
                .encode(  # ty: ignore[unresolved-attribute]
                    x="time:T",
                    y=alt.Y("power:Q", scale=alt.Scale(zero=False)),
                    detail="key:N",
                    color=alt.Color(
                        "series:N",
                        title="Lines",
                        scale=alt.Scale(domain=domain, range=colours),
                    ),
                )
            )
            panels.append(
                alt.layer(band, drawn).properties(
                    width=560,
                    height=200,
                    title=alt.TitleParams(
                        f"{title}, {season}, Primary D1 (orange band: arm D, q05 to q95)",
                        anchor="start",
                    ),
                )
            )
        alt.vconcat(*panels).save(DOCS_DIR / f"figure_2_example_{season}.svg")


if __name__ == "__main__":
    main()

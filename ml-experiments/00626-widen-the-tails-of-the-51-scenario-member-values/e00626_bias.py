# ruff: noqa: E501
"""Where does demand's bias come from? Exceedance of the direct quantile forecast, by month and band.

Experiment 436's 51 values are the stored 11 quantiles of experiment 425's arm A (seeds averaged,
q01, q05, q95, and q99 widened by 1.15) interpolated linearly in the level and multiplied by 100.
The levels 0.10 and 0.90 are not widened, so q10 and q90 can be read back from the 51 values
exactly (the two values beside each level lie on the same segment). This script scores those
quantiles, and the stored table's own q10 and q90 for one month as a direct check, by month, lead
band, and Primary, and prints the mean error (actual minus median) in points of capacity.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00626-*/env.sh
uv run --no-sync python ml-experiments/00626-*/e00626_bias.py
```
"""

from pathlib import Path

import altair as alt
import e00434_score as s434  # ty: ignore[unresolved-import]
import e00626_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

PRIMARIES = 16


def exact_q(*, v: np.ndarray, level: float) -> np.ndarray:
    """Return q10 or q90 read back from the 51 values (..., 51) by extending the local segment."""
    pos = level * 52 - 1
    i = 3 if level < 0.5 else 46
    return v[..., i] + (v[..., i + 1] - v[..., i]) * (pos - i)


def draw_figure(*, df: pl.DataFrame) -> None:
    """Draw the share of Primary actuals above q90 and below q10, by month and by lead band."""
    docs = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / c.SLUG
    panels = []
    for key, title in (("month", "By month"), ("band", "By lead band")):
        g = df.group_by(key).agg(pl.col("n", "below_q10", "above_q90").sum()).sort(key)
        if key == "band":
            order = ["h0_12", "h12_36", "h36_120", "h120_240"]
            g = g.with_columns(
                pl.col("band").replace_strict(order, ["0-12 h", "12-36 h", "36-120 h", "120-240 h"])
            )
            sort = ["0-12 h", "12-36 h", "36-120 h", "120-240 h"]
        else:
            sort = g[key].to_list()
        long = pl.concat(
            [
                g.select(
                    key,
                    pl.lit("Above q90 (target 10%)").alias("side"),
                    (pl.col("above_q90") / pl.col("n") * 100).alias("rate"),
                ),
                g.select(
                    key,
                    pl.lit("Below q10 (target 10%)").alias("side"),
                    (pl.col("below_q10") / pl.col("n") * 100).alias("rate"),
                ),
            ]
        )
        bars = (
            alt.Chart(long.to_pandas())
            .mark_bar()
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(f"{key}:N", sort=sort, title=None, axis=alt.Axis(labelAngle=-45)),
                xOffset="side:N",
                y=alt.Y("rate:Q", title="Share of actuals (%)", scale=alt.Scale(domain=[0, 25])),
                color=alt.Color(
                    "side:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Above q90 (target 10%)", "Below q10 (target 10%)"],
                        range=[theme.BRAND_ORANGE, theme.DATA_BLUE],
                    ),
                ),
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"y": [10.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#555555")
            .encode(y="y:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(bars, rule).properties(
                width=380, height=200, title=alt.TitleParams(title, anchor="start")
            )
        )
    chart = alt.hconcat(*panels).properties(
        title=alt.TitleParams(
            "Demand forecast: the q90 is exceeded too often, the q10 is not",
            subtitle=[
                "The 16 Primaries, all 44 runs, leads to 240 hours, experiment 425 arm A quantiles. Dashed line: 10%, the rate of a calibrated quantile."
            ],
            anchor="start",
        )
    )
    chart.save(docs / "figure_3_demand_quantile_bias.svg")


def main() -> None:
    """Print the exceedance tables."""
    band = c.band_of_step()
    names = list(c.BANDS)
    rows = []
    for file in sorted((c.WORK_436 / "runs").glob("*.npz")):
        run = c.load_run(path=file)
        v, a = run["values"][:PRIMARIES], run["actual"][:PRIMARIES]
        ok = ~np.isnan(a) & (v[..., -1] > v[..., 0]) & (band >= 0)[None, :]
        q10, q90 = exact_q(v=v, level=0.10), exact_q(v=v, level=0.90)
        med = v[..., c.MID]
        for b, bn in enumerate(names):
            sel = ok & (band == b)[None, :]
            for p in range(PRIMARIES):
                s = sel[p]
                rows.append({"month": f"{run['init']:%Y-%m}", "band": bn, "primary": f"D{p}", "n": int(s.sum()),
                             "below_q10": int((a[p][s] < q10[p][s]).sum()), "above_q90": int((a[p][s] > q90[p][s]).sum()),
                             "err_sum": float((a[p][s] - med[p][s]).sum()), "width_sum": float((q90[p][s] - q10[p][s]).sum())})  # fmt: skip
    df = pl.DataFrame(rows)
    pl.Config.set_tbl_rows(100)
    for key in ("month", "band", "primary"):
        g = (
            df.group_by(key)
            .agg(pl.col("n", "below_q10", "above_q90", "err_sum", "width_sum").sum())
            .sort(key)
        )
        print(g.with_columns(below_q10=(pl.col("below_q10") / pl.col("n")).round(3), above_q90=(pl.col("above_q90") / pl.col("n")).round(3),
                             mean_err=(pl.col("err_sum") / pl.col("n")).round(2), mean_q10_q90_width=(pl.col("width_sum") / pl.col("n")).round(2)).drop("err_sum", "width_sum"))  # fmt: skip
    draw_figure(df=df)
    # direct check against the stored table (month 2026-01), all demand rows, lead 12 to 240 h
    month = "2026-01"
    st = s434.stored_arm_a(month=month)
    runs = {
        f"{c.load_run(path=f)['init']:%Y-%m-%d}": f
        for f in (c.WORK_436 / "runs").glob(f"{month}-*.npz")
    }
    ids = sorted(st["time_series_id"].unique().to_list())
    print("stored ids:", len(ids), "runs:", len(runs))
    for date, f in sorted(runs.items())[:2]:
        run = c.load_run(path=f)
        sub = st.filter(pl.col("power_fcst_init_time") == run["init"]).sort(
            "time_series_id", "valid_time"
        )
        for k, name in ((0.9, "q90"), (0.1, "q10")):
            stored = sub[name].to_numpy().reshape(-1, 702)[:PRIMARIES] * 100
            mine = exact_q(v=run["values"][:PRIMARIES], level=k)
            print(
                date,
                name,
                "largest |stored - read back from the 51 values|, Primaries sorted at each half-hour (points of capacity):",
                float(np.abs(np.sort(stored, axis=0) - np.sort(mine, axis=0)).max()),
            )


if __name__ == "__main__":
    main()

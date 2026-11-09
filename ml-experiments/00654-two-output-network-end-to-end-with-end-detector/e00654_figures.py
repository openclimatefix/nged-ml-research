# ruff: noqa: D103, E501
"""Experiment 654: by-lead-day injected-shift table and the headline figure (CPU)."""

import altair as alt
import e00654_common as c654  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf  # ty: ignore[unresolved-import]
import polars as pl

SC = c654.WORK / "score"
OUT = c654.WORK / "figures"
OUT.mkdir(exist_ok=True)
ARMS = ["K1", "R0-168", "Bhaz", "E0", "C0", "J1", "J2"]
LABEL = {
    "K1": "K1",
    "R0-168": "R0 (168 h)",
    "Bhaz": "B-haz",
    "E0": "E0",
    "C0": "C0",
    "J1": "J1",
    "J2": "J2 prevailing",
}
COLOUR = {
    "K1": ocf.BLACK_1,
    "R0 (168 h)": ocf.DATA_GREEN,
    "B-haz": ocf.DATA_SKY,
    "E0": ocf.DATA_BLUE,
    "C0": ocf.DATA_PURPLE,
    "J1": ocf.BRAND_ORANGE_LIGHT,
    "J2 prevailing": ocf.BRAND_ORANGE,
}
STRATA = {
    "far": "Shift in force, more than 10 days left",
    "near": "Shift in force, 24 h or less left",
    "ended24": "Shift ended in the last 24 h",
    "free": "No shift in force",
}


def table() -> pl.DataFrame:
    d = pl.read_parquet(SC / "agg.parquet").filter(
        (pl.col("set") == "fam") & pl.col("arm").is_in(ARMS) & pl.col("stratum").is_in(list(STRATA))
    )
    m = d.group_by("stratum", "arm", "lead_day", "month").agg(
        pl.col("crps_s").sum() / pl.col("sw").sum(), pl.col("sw").sum().alias("w")
    )
    rows = []
    rng = np.random.default_rng(654)
    for (st, arm, ld), g in m.group_by("stratum", "arm", "lead_day"):
        v, w = g["crps_s"].to_numpy(), g["w"].to_numpy()
        pick = rng.integers(0, len(v), size=(2000, len(v)))
        draws = (v[pick] * w[pick]).sum(axis=1) / w[pick].sum(axis=1)
        rows.append(
            {
                "stratum": st,
                "arm": LABEL[arm],
                "lead_day": ld,
                "crps_s": float((v * w).sum() / w.sum()),
                "lo": float(np.quantile(draws, 0.025)),
                "hi": float(np.quantile(draws, 0.975)),
            }
        )
    out = pl.DataFrame(rows).sort("stratum", "arm", "lead_day")
    out.write_csv(SC / "by_lead_day.csv")
    return out


def main() -> None:
    t = table()
    order = [LABEL[a] for a in ARMS]
    panels = []
    for st in ("far", "ended24"):
        base = alt.Chart(t.filter(pl.col("stratum") == st))
        enc = {
            "x": alt.X(
                "lead_day:Q",
                title="Lead day (1 = 24 to 48 h ahead)",
                axis=alt.Axis(values=list(range(1, 10))),
            ),
            "color": alt.Color(
                "arm:N",
                sort=order,
                scale=alt.Scale(domain=order, range=[COLOUR[k] for k in order]),
                legend=alt.Legend(title=None, orient="top", columns=4),
            ),
        }
        band = base.mark_area(opacity=0.12, aria=False).encode(
            y=alt.Y("lo:Q", title="Scaled CRPS (points of capacity; smaller is better)"),
            y2="hi:Q",
            **enc,
        )
        line = base.mark_line(point=True, aria=False).encode(y="crps_s:Q", **enc)
        panels.append(
            (band + line).properties(
                width=560,
                height=230,
                title=alt.TitleParams(STRATA[st], anchor="start", fontSize=12),
            )
        )
    chart = (
        alt.vconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=alt.TitleParams(
                "Forecast of observed power on injected shifts, by lead day",
                subtitle=[
                    "Injected shifts on real bases (training family), 5 development months, 2 seeds. Line: mean. Band: 95% interval",
                    "from resampling whole months. K1 is the normal-running forecast, which carries no shift.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(OUT / "headline.svg")
    chart.save(OUT / "headline.png", scale_factor=2.0)


main()

# ruff: noqa: ANN201, D103, E501, PERF401
"""Print every number experiment 570's page quotes, and draw its two figures."""

import json
import re
from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00570-price-encoder-part-1-known-answer-test-and-pretrained-stack"
D = ML_EXPERIMENTS_DATA_DIR / NAME
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
LABEL = {
    "A6-res - M0": "A6 price encoder in pretraining, then dropped (A6-res) minus plain M0",
    "A6-noise - M0": "A6 encoder fed noise minus M0 (control)",
    "A6-const - M0": "A6 encoder fed a constant minus M0 (control)",
    "A6-res - A6-noise": "A6-res minus its noise control",
    "A6-res - A6-p2": "A6-res minus its typical-shape control",
    "A1-res - A1-noise": "A1-res (encoder kept, null price at forecast) minus noise control",
    "A3-res - A3-noise": "A3-res (encoder added at fine-tuning) minus noise control",
    "A1-res-oracle - A1-res": "A1-res scored with the true price minus with null price",
}


def pretrain_nll() -> pl.DataFrame:
    rows = []
    for log in (D / "logs").glob("pre_*.log"):
        for m in re.finditer(
            r"pretrain (\S+) (\d{4}-\d{2}) s(\d): (\d+) epochs, val NLL ([\d.]+)", log.read_text()
        ):
            rows.append(
                {
                    "arm": m[1],
                    "month": m[2],
                    "seed": int(m[3]),
                    "epochs": int(m[4]),
                    "nll": float(m[5]),
                }
            )
    return (
        pl.DataFrame(rows)
        .unique(["arm", "month", "seed"], keep="last")
        .filter(pl.col("month") != "")
    )


def main():
    c = pl.read_csv(D / "work/scores/contrasts.csv")
    a = pl.read_csv(D / "work/scores/absolute.csv")
    pooled = c.filter(pl.col("band") == "24-120h")
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    print("ABSOLUTE CRPS 24-120h")
    print(
        a.filter(pl.col("band") == "24-120h")
        .pivot(on="set", index="arm", values="crps")
        .select("arm", "eight", "six", "eight_without_D8", "D8_alone", "report")
        .with_columns(pl.all().exclude("arm").round(3))
    )
    print("CONTRASTS 24-120h (positive = worse)")
    print(
        pooled.filter(pl.col("set").is_in(["eight", "six", "eight_without_D8", "D8_alone"]))
        .select(
            "set",
            "contrast",
            "difference",
            "low_month",
            "high_month",
            "low_bonferroni_month",
            "high_bonferroni_month",
            "mde",
            "folds_positive",
            "by_fold",
        )
        .with_columns(pl.col(pl.Float64).round(3))
        .sort("set", "contrast")
    )
    nll = pretrain_nll()
    piv = (
        nll.group_by("arm")
        .agg(
            pl.col("nll").mean().round(4).alias("mean_val_nll"),
            pl.col("epochs").mean().round(1).alias("mean_epochs"),
            pl.len().alias("pretrainings"),
        )
        .sort("arm")
    )
    print(
        "PRETRAIN VALIDATION NLL (mean over 5 folds x 2 seeds; lower is better; oracle price present)"
    )
    print(piv)
    m0 = nll.filter(pl.col("arm") == "M0").group_by("month").agg(pl.col("nll").mean().alias("m0"))
    diff = (
        nll.filter(pl.col("arm") != "M0")
        .group_by("arm", "month")
        .agg(pl.col("nll").mean())
        .join(m0, on="month")
        .with_columns(d=pl.col("nll") - pl.col("m0"))
    )
    print("PER-FOLD DIFFERENCE IN PRETRAIN VAL NLL FROM M0 (negative = lower than M0)")
    print(
        diff.pivot(on="month", index="arm", values="d")
        .with_columns(pl.all().exclude("arm").round(4))
        .sort("arm")
    )
    known = {}
    for f in sorted(D.glob("known_*_s0.json")):
        for key, v in json.loads(f.read_text()).items():
            known[(f.name[6:13], key)] = v
    rows = [
        {
            "month": m,
            "run": k,
            **{a: b for a, b in v.items() if a != "threshold_curve_price_frac_of_capacity"},
        }
        for (m, k), v in known.items()
    ]
    kd = pl.DataFrame(rows, infer_schema_length=None).sort("run", "month")
    print("KNOWN-ANSWER RESULTS")
    print(kd.with_columns(pl.col(pl.Float64).round(3)))
    # Figure 1: contrasts with intervals
    part = pooled.filter(
        pl.col("set").is_in(["eight", "six"]) & pl.col("contrast").is_in(list(LABEL))
    )
    part = part.with_columns(
        arm=pl.col("contrast").replace(LABEL),
        group=pl.col("set").replace(
            {"eight": "Eight MVA Primaries (MDE 0.113)", "six": "Six MW Primaries (MDE 0.128)"}
        ),
    ).to_pandas()
    panels = []
    for g in ("Eight MVA Primaries (MDE 0.113)", "Six MW Primaries (MDE 0.128)"):
        p = part[part["group"] == g]
        base = alt.Chart(p).encode(
            y=alt.Y("arm:N", sort=list(LABEL.values()), title=None, axis=alt.Axis(labelLimit=420))
        )
        bar = base.mark_rule(color=theme.DATA_BLUE, strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low_month:Q",
                title="Arm minus comparison (CRPS points of capacity; negative = better than comparison)",
                scale=alt.Scale(domain=[-0.5, 1.0]),
            ),
            x2="high_month:Q",
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(color=theme.BRAND_ORANGE, filled=True, size=60, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x="difference:Q"
        )  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black")
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=380, height=170, title=alt.TitleParams(g, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Price encoder: fine-tuned CRPS differences, 24 to 120 h leads, 5 folds, 2 seeds.",
                "Dot: estimate. Line: 95% interval from resampling whole months (5 blocks, near min-to-max).",
                "Price: Contains BMRS data © Elexon Limited copyright and database right 2026.",
            ],
            anchor="start",
            fontSize=12,
            subtitleFontSize=11,
        )
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS / "figure_1_contrasts.svg")
    # Figure 2: known-answer recovered share
    kk = (
        kd.filter(pl.col("run").str.ends_with("-all"))
        .select("month", "run", "recovered_real", "recovered_p2")
        .unpivot(index=["month", "run"], variable_name="price", value_name="recovered")
        .with_columns(
            price=pl.col("price").replace(
                {"recovered_real": "True price", "recovered_p2": "Typical-shape price"}
            ),
            rule=pl.col("run")
            .str.slice(0, 2)
            .replace(
                {
                    "R1": "R1 threshold above 90 GBP/MWh",
                    "R2": "R2 six dearest half-hours",
                    "R3": "R3 battery: cheapest 4 h charge, dearest 4 h discharge",
                }
            ),
        )
        .to_pandas()
    )
    ch2 = (
        alt.Chart(kk)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, aria=False)
        .encode(
            x=alt.X(
                "recovered:Q",
                title="Share of injected generator output recovered (1 = all; pass mark 0.6)",
                scale=alt.Scale(domain=[-0.3, 1]),
            ),
            y=alt.Y("rule:N", title=None, axis=alt.Axis(labelLimit=420)),
            color=alt.Color(
                "price:N",
                scale=alt.Scale(
                    domain=["True price", "Typical-shape price"],
                    range=[theme.BRAND_ORANGE, theme.DATA_BLUE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=420,
            height=130,
            title=alt.TitleParams(
                [
                    "Known-answer test: each dot is one fold. Injected generator capacity is 2% of",
                    "effective capacity at the six MW Primaries. Price: Contains BMRS data © Elexon Limited copyright and database right 2026.",
                ],
                anchor="start",
                fontSize=12,
            ),
        )
    )
    ch2.save(DOCS / "figure_2_known_answer.svg")


main()

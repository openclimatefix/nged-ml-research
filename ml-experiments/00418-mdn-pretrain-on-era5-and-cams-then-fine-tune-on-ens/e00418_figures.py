# ruff: noqa: E501
"""Draw the figures of experiment 418: scores by lead band, contrasts with intervals, gain by fold.

`data` scores the arms (12 monthly folds, plus the low-data check) and stores small tables;
`draw` reads them and writes the SVG figures beside the write-up.
"""

import argparse
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_score as s397  # ty: ignore[unresolved-import]
import e00418_contrasts as con  # ty: ignore[unresolved-import]
import e00418_fit as fit  # ty: ignore[unresolved-import]
import e00418_score as score  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme
from studies.eval_protocol import metrics

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens"
)
TABLES: Final[Path] = fit.prep.WORK_DIR / "work" / "figure_tables"
BANDS: Final[dict[str, str]] = {
    "lead_day_1": "Day 1",
    "lead_days_1_10": "Days 1 to 10",
    "lead_days_8_10": "Days 8 to 10",
    "lead_days_11_14": "Days 11 to 14",
}
NAMES: Final[dict[str, str]] = {
    "ens-only": "ENS only (reference)",
    "P1": "Pretrain, tune all weights",
    "P2": "Pretrain, tune input and output layers",
    "P3": "Pretrain with noise, tune all weights",
    "bg2": "BG2 (weekly XGBoost)",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
ORDER: Final[list[str]] = list(NAMES.values())
COLOURS: Final[list[str]] = [
    "#1A1A1A",
    theme.BRAND_ORANGE,
    theme.DATA_BLUE,
    theme.DATA_PURPLE,
    "#8A8A8A",
    "#B5B5B5",
    "#D5D5D5",
]
ARMS: Final[list[str]] = ["ens-only", "P1", "P2", "P3"]


def data() -> None:
    """Score everything once and store the tables."""
    TABLES.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    frames = score.arm_frames(
        folder="seed_preds", months=list(con.ALL_MONTHS), arms=ARMS, series=series
    )
    names = {fit.arm_name(arm=a): a for a in ARMS}
    frames.update(
        {
            k: v
            for k, v in s397.references(series=series).items()
            if k in ("bg2", "manual_heuristic", "climatology")
        }
    )
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    out = []
    for name, frame in scored.items():
        arm = names.get(name, name)
        for group, labels in con.LABELS.items():
            for band in BANDS:
                for metric in ("crps",):
                    sel = frame.filter(
                        pl.col("metric") == metric,
                        pl.col("slice") == band,
                        pl.col("label").is_in(list(labels)),
                    )
                    out.append(
                        {
                            "arm": arm,
                            "group": group,
                            "band": band,
                            "crps": float(sel["value"].mean()),
                        }
                    )
    pl.DataFrame(out).write_parquet(TABLES / "bands.parquet")
    inverse = {a: n for n, a in names.items()}
    contrasts = []
    for label, (t, r) in con.CONTRASTS.items():
        for group, labels in con.LABELS.items():
            res = metrics.month_bootstrap_contrast(
                treatment=rows[inverse[t]], reference=rows[inverse[r]], metric="crps",
                where=pl.col("label").is_in(list(labels)) & pl.col(s397.HEADLINE),
                level_percent=95.0, n_resamples=20_000, slice_table=table,
            )  # fmt: skip
            contrasts.append(
                {
                    "contrast": label,
                    "group": group,
                    "diff": res["difference"],
                    "lo": res["lower"],
                    "hi": res["upper"],
                }
            )
    pl.DataFrame(contrasts).write_parquet(TABLES / "contrasts.parquet")
    by_fold = []
    for month in con.ALL_MONTHS:
        sub = {
            k: s397.only_month(frame=v, month=month)
            for k, v in frames.items()
            if k in inverse.values()
        }
        _, sc, _ = s381.score_arms(arms=sub, series=series, ctx=ctx)
        for name, frame in sc.items():
            for group in ("signed-MW six", "MVA-only eight"):
                sel = frame.filter(
                    pl.col("metric") == "crps",
                    pl.col("slice") == s397.HEADLINE,
                    pl.col("label").is_in(list(con.LABELS[group])),
                )
                by_fold.append(
                    {
                        "fold": month,
                        "arm": names[name],
                        "group": group,
                        "crps": float(sel["value"].mean()),
                    }
                )
    pl.DataFrame(by_fold).write_parquet(TABLES / "by_fold.parquet")
    low = []
    for label, folder in (
        ("First 6 months of ENS only", "lowdata"),
        ("Full 24-month window", "seed_preds"),
    ):
        fr = score.arm_frames(
            folder=folder, months=list(fit.SCREEN_MONTHS), arms=["ens-only", "P1"], series=series
        )
        tb, _, rw = s381.score_arms(arms=fr, series=series, ctx=ctx)
        for group, labels in list(con.LABELS.items())[:2]:
            res = metrics.month_bootstrap_contrast(
                treatment=rw[fit.arm_name(arm="P1")], reference=rw[fit.arm_name(arm="ens-only")], metric="crps",
                where=pl.col("label").is_in(list(labels)) & pl.col(s397.HEADLINE),
                level_percent=95.0, n_resamples=20_000, slice_table=tb,
            )  # fmt: skip
            low.append(
                {
                    "history": label,
                    "group": group,
                    "diff": res["difference"],
                    "lo": res["lower"],
                    "hi": res["upper"],
                }
            )
    pl.DataFrame(low).write_parquet(TABLES / "lowdata.parquet")


def draw() -> None:
    """Draw the four figures."""
    bands = pl.read_parquet(TABLES / "bands.parquet").filter(
        pl.col("group").is_in(["signed-MW six", "MVA-only eight"]), pl.col("arm").is_in(list(NAMES))
    )
    bands = bands.with_columns(
        arm_name=pl.col("arm").replace_strict(NAMES), band_name=pl.col("band").replace_strict(BANDS)
    ).to_pandas()
    colour = alt.Color(
        "arm_name:N",
        sort=ORDER,
        scale=alt.Scale(domain=ORDER, range=COLOURS),
        legend=alt.Legend(orient="top", columns=2, title=None, labelLimit=400, symbolLimit=10),
    )
    f1 = (
        alt.Chart(bands)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("band_name:N", sort=list(BANDS.values()), title=None),
            x=alt.X(
                "crps:Q",
                title="CRPS (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=colour,
            yOffset=alt.YOffset("arm_name:N", sort=ORDER),
            row=alt.Row("group:N", title=None, sort=["signed-MW six", "MVA-only eight"]),
        )
        .properties(
            width=560,
            height=150,
            title=[
                "Pretraining lowers CRPS at days 1 to 10 and matches ENS only beyond;",
                "twelve monthly test folds, equal weight per Primary. Smaller is better.",
            ],
        )
    )
    f1.save(DOCS_DIR / "figure_1_crps_by_lead.svg")
    contrasts = (
        pl.read_parquet(TABLES / "contrasts.parquet")
        .filter(pl.col("group").is_in(["signed-MW six", "MVA-only eight"]))
        .to_pandas()
    )
    base = alt.Chart(contrasts).encode(
        y=alt.Y("contrast:N", sort=list(con.CONTRASTS), title=None),
        color=alt.value(theme.DATA_BLUE),
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
        .mark_rule(strokeDash=[4, 3], color="#888888")
        .encode(x="x:Q")
    )
    f2 = (
        (
            base.mark_rule().encode(
                x=alt.X(
                    "lo:Q",
                    title="Difference in CRPS (points of capacity; more negative favours the first arm)",
                ),
                x2="hi:Q",
            )
            + base.mark_point(filled=True, size=70).encode(x="diff:Q")
            + rule
        )
        .properties(width=560, height=130)
        .facet(row=alt.Row("group:N", title=None, sort=["signed-MW six", "MVA-only eight"]))
        .properties(
            title=[
                "Planned contrasts. Dot: estimate. Line: 95% interval from resampling whole months.",
                "Dashed line: no difference.",
            ]
        )
    )
    f2.save(DOCS_DIR / "figure_2_contrasts.svg")
    folds = pl.read_parquet(TABLES / "by_fold.parquet").filter(pl.col("group") == "signed-MW six")
    ref = folds.filter(pl.col("arm") == "ens-only").select("fold", ref=pl.col("crps"))
    gain = (
        folds.filter(pl.col("arm").is_in(["P1", "P2", "P3"]))
        .join(ref, on="fold")
        .with_columns(
            gain=pl.col("crps") - pl.col("ref"), arm_name=pl.col("arm").replace_strict(NAMES)
        )
        .to_pandas()
    )
    f3 = (
        alt.Chart(gain)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("fold:O", title="Test month (refit on the first day)"),
            y=alt.Y("gain:Q", title="CRPS minus ENS only (points; negative is better)"),
            color=alt.Color(
                "arm_name:N",
                sort=ORDER[1:4],
                scale=alt.Scale(domain=ORDER[1:4], range=COLOURS[1:4]),
                legend=alt.Legend(orient="top", columns=1, title=None, labelLimit=400),
            ),
        )
        .properties(
            width=560,
            height=240,
            title=[
                "Pretrained arms beat the ENS-only network in most months at the six",
                "signed-direction Primaries; the dashed line is no difference",
            ],
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())
        .mark_rule(strokeDash=[4, 3], color="#888888")
        .encode(y="y:Q")
    )
    (f3 + zero).save(DOCS_DIR / "figure_3_gain_by_month.svg")
    low = pl.read_parquet(TABLES / "lowdata.parquet").to_pandas()
    lb = alt.Chart(low).encode(
        y=alt.Y(
            "history:N", sort=["First 6 months of ENS only", "Full 24-month window"], title=None
        ),
        color=alt.value(theme.BRAND_ORANGE),
    )
    f4 = (
        (
            lb.mark_rule().encode(
                x=alt.X(
                    "lo:Q",
                    title="Pretrain-and-tune-all minus ENS only (CRPS points; negative is better)",
                ),
                x2="hi:Q",
            )
            + lb.mark_point(filled=True, size=70).encode(x="diff:Q")
            + rule
        )
        .properties(width=560, height=90)
        .facet(row=alt.Row("group:N", title=None, sort=["signed-MW six", "MVA-only eight"]))
        .properties(
            title=[
                "Pretraining helps most when the ENS history is short.",
                "Dot: estimate. Line: 95% interval from resampling whole months. Four test months.",
            ]
        )
    )
    f4.save(DOCS_DIR / "figure_4_low_data.svg")


def main() -> None:
    """Run a mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("data", "draw"))
    args = parser.parse_args()
    if args.mode == "data":
        data()
    else:
        draw()


if __name__ == "__main__":
    main()

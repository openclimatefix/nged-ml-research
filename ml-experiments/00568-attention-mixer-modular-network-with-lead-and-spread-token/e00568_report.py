# ruff: noqa: E501, PTH207
"""Print every number the experiment 568 page quotes, and draw its figures.

Reads the tables `e00568_score.py` and `e00568_extra.py` stored, the attention-weight files, and the
real-event table `e00568_fit.py events` wrote.
"""

import glob
import json
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00568-attention-mixer-modular-network-with-lead-and-spread-token"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
SCORES: Final[Path] = WORK / "scores"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
BANDS: Final = ("0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h")
ARMS: Final = {
    "T0": "#808080",
    "T1": theme.BRAND_ORANGE,
    "T2": theme.DATA_SKY,
    "T3": theme.DATA_GREEN,
    "T4": theme.DATA_BLUE,
    "T5": "#a05195",
}
LABELS: Final = {
    "T0": "T0 MLP mixer",
    "T1": "T1 attention, full",
    "T2": "T2 no spread token",
    "T3": "T3 temperature spread only",
    "T4": "T4 MLP + spread columns",
    "T5": "T5 shuffled-spread placebo",
}
SETS: Final = {"eight": "Eight MVA Primaries", "six": "Six MW Primaries"}
N_BOOT: Final = 2000


def tables() -> None:
    """Print absolute CRPS and the contrasts."""
    a = pl.read_csv(SCORES / "absolute.csv")
    c = pl.read_csv(SCORES / "contrasts.csv")
    with pl.Config(tbl_rows=200, tbl_cols=30, fmt_str_lengths=40):
        for s in ("eight", "eight_no_d8", "six", "report"):
            print(f"--- absolute CRPS, set {s}")
            print(
                a.filter(pl.col("set") == s)
                .pivot(on="arm", index="band", values="crps")
                .with_columns(pl.col(pl.Float64).round(3))
            )
        print("--- contrasts at 24-120 h")
        print(
            c.filter(pl.col("band") == "24-120h").select(
                "set", "contrast", "counted", "difference", "low_bonferroni_month", "high_bonferroni_month", "folds_positive", "mde"
            ).with_columns(pl.col(pl.Float64).round(3))
        )  # fmt: skip
        print("--- stress leak (points of capacity, one fit at day 14, mean over folds)")
        print(pl.read_csv(SCORES / "stress_leak.csv").with_columns(pl.col(pl.Float64).round(2)))
        print("--- slices")
        print(pl.read_csv(SCORES / "slices.csv").with_columns(pl.col(pl.Float64).round(3)))


def real_events() -> None:
    """Print the absorbed share of real labelled events for the daily-refit arms."""
    path = SCORES.parent / "work/scores/real_events.parquet"
    if not path.exists():
        return
    t = pl.read_parquet(path).filter(pl.col("absorbed_share").is_not_null())
    rng = np.random.default_rng(0)
    for arm, d in t.group_by("arm", maintain_order=True):
        for sel, part in (("all", d), ("outside step 2", d.filter(~pl.col("in_step2")))):
            x = part["absorbed_share"].to_numpy()
            if len(x) < 5:
                continue
            means = [rng.choice(x, size=len(x)).mean() for _ in range(N_BOOT)]
            print(arm, sel, "n", len(x), "mean", round(float(x.mean()), 4),
                  "interval", round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4),
                  "without largest", round(float((x.sum() - x.max()) / (len(x) - 1)), 4))  # fmt: skip
    wide = t.pivot(
        on="arm", index=["month", "primary", "onset"], values="absorbed_share"
    ).drop_nulls()
    if {"T0", "T4"} <= set(wide.columns):
        dd = (wide["T4"] - wide["T0"]).to_numpy()
        means = [rng.choice(dd, size=len(dd)).mean() for _ in range(N_BOOT)]
        print("T4 minus T0 on", len(dd), "shared events: mean", round(float(dd.mean()), 4),
              "interval", round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4))  # fmt: skip


def attention_table() -> pl.DataFrame:
    """Return the mean query-to-token attention weight by arm, token, and lead band (mean of folds and seeds)."""
    rows = []
    for arm in ("T1", "T2", "T3", "T5"):
        acc: dict = {}
        for f in sorted(glob.glob(str(WORK / "work/attention" / arm / "*.json"))):
            j = json.loads(Path(f).read_text())
            for band, v in j["by_band"].items():
                if v:
                    acc.setdefault(band, []).append(v)
        for band, v in acc.items():
            for tok, w in zip(j["tokens"], np.mean(v, axis=0), strict=True):
                rows.append({"arm": arm, "band": band, "token": tok, "weight": float(w)})
    return pl.DataFrame(rows)


def fig_bands() -> None:
    """Figure 1: CRPS by lead band, one panel per group."""
    a = pl.read_csv(SCORES / "absolute.csv").filter(pl.col("arm").is_in(list(ARMS)))
    panels = []
    for key, label in SETS.items():
        d = a.filter((pl.col("set") == key) & pl.col("band").is_in(BANDS)).with_columns(
            arm=pl.col("arm").replace(LABELS)
        )
        panels.append(
            alt.Chart(d.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X("band:N", sort=list(BANDS), title="Lead band"),
                y=alt.Y("crps:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=list(LABELS.values()), range=list(ARMS.values())),
                    title="Arm",
                ),
            )
            .properties(width=300, height=220, title=label)
        )
    alt.hconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "One refit per fold, scored over the whole month; five folds, two seeds.",
                "Lower is better.",
            ],
            anchor="start",
        )
    ).save(DOCS / "figure_1_crps_by_lead_band.svg")


def fig_leak() -> None:
    """Figure 2: CRPS at 24-120 h against the injected-shift leak, one point per arm and group."""
    a = pl.read_csv(SCORES / "absolute.csv").filter(
        (pl.col("band") == "24-120h")
        & pl.col("set").is_in(list(SETS))
        & pl.col("arm").is_in(list(ARMS))
    )
    leak = pl.read_csv(SCORES / "stress_leak.csv").rename({"arm": "arm", "leak": "leak"})
    d = a.join(leak.select("arm", "leak"), on="arm").with_columns(
        group=pl.col("set").replace(SETS), label=pl.col("arm").replace(LABELS)
    )
    base = alt.Chart(d.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("leak:Q", title="Injected-shift leak (points of capacity, one fit at day 14)"),
        y=alt.Y(
            "crps:Q", title="CRPS at 24-120 h (points of capacity)", scale=alt.Scale(zero=False)
        ),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=list(LABELS.values()), range=list(ARMS.values())),
            title="Arm",
        ),
    )
    (base.mark_point(filled=True, size=90) + base.mark_text(dx=8, dy=-6, fontSize=10).encode(text="arm:N")).properties(  # ty: ignore[unresolved-attribute]
        width=300, height=220
    ).facet(column=alt.Column("group:N", title=None), data=d.to_pandas()).resolve_scale(y="independent").save(DOCS / "figure_2_crps_against_leak.svg")  # fmt: skip


def fig_attention() -> None:
    """Figure 3: attention weight from the query token to each token of T1, by lead band."""
    t = attention_table().filter(pl.col("arm") == "T1")
    order = ["0-6 h", "6-24 h", "24-48 h", "48-72 h", "72-120 h", "120-240 h", "240-336 h"]
    alt.Chart(t.to_pandas()).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("band:N", sort=order, title="Lead band"),
        y=alt.Y("weight:Q", title="Mean attention weight", scale=alt.Scale(zero=False)),
        color=alt.Color("token:N", scale=alt.Scale(domain=["weather", "datetime", "primary", "spread"], range=[theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_GREEN, "#808080"]), title="Token"),
    ).properties(width=420, height=220, title=alt.TitleParams(["Attention weights from the query token of T1 (last layer, mean of heads, folds, and seeds).", "A diagnostic, not an explanation."], anchor="start")).save(DOCS / "figure_3_attention_weights.svg")  # fmt: skip


def main() -> None:
    """Print the tables and draw the figures."""
    DOCS.mkdir(parents=True, exist_ok=True)
    tables()
    real_events()
    with pl.Config(tbl_rows=100):
        print("--- attention weights")
        print(
            attention_table()
            .pivot(on="token", index=["arm", "band"], values="weight")
            .with_columns(pl.col(pl.Float64).round(3))
        )
    fig_bands()
    fig_leak()
    fig_attention()


if __name__ == "__main__":
    main()

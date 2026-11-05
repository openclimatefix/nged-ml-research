# ruff: noqa: E501
"""Print every number the experiment 566 page quotes, and draw its first two figures.

Reads the tables `e00566_score.py` stored and the per-event table `e00566_fit.py events` wrote.
"""

from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = (
    "00566-combination-block-lag-encoder-with-three-step-schedule-and-embargo-real-event-leak"
)
SCORES: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "scores"
EVENTS: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work/scores/real_events.parquet"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
BANDS: Final = ("0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h")
ARMS: Final = {
    "C0": "#808080",
    "C1": theme.DATA_BLUE,
    "C2": theme.DATA_GREEN,
    "C3": theme.DATA_SKY,
    "C4": theme.BRAND_ORANGE,
}
LABELS: Final = {
    "C0": "C0 lag-free",
    "C1": "C1 lag encoder",
    "C2": "C2 + three steps",
    "C3": "C3 + 3-day embargo",
    "C4": "C4 + both",
}
SETS: Final = {"eight": "MVA-only eight", "six": "Signed-MW six"}
WIDTH: Final = 300
FLAG: Final = 0.03
"""Flag level fixed in the plan before any number was read: twice the lag-free network's 0.015."""
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
        print("--- counted contrasts")
        print(
            c.filter(pl.col("counted")).select(
                "set", "contrast", "difference", "low_bonferroni_month", "high_bonferroni_month",
                "folds_positive", "mde",
            )
        )  # fmt: skip
        print("--- contrasts at 24-120 h, all sets")
        print(
            c.filter(pl.col("band") == "24-120h").select(
                "set", "contrast", "difference", "low_month", "high_month", "folds_positive", "mde"
            )
        )  # fmt: skip


def boot(*, x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Return a 95% interval of the mean, resampling events."""
    means = [rng.choice(x, size=len(x)).mean() for _ in range(N_BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def leak_rows() -> pl.DataFrame:
    """Return the mean absorbed share per arm, cut age, and event selection, with intervals."""
    t = pl.read_parquet(EVENTS).filter(pl.col("absorbed_share").is_not_null())
    rng = np.random.default_rng(0)
    out = []
    for (arm, age), d in t.group_by("arm", "min_age", maintain_order=True):
        for sel, part in (
            ("all events", d),
            ("events outside step 2", d.filter(~pl.col("in_step2"))),
            ("events up to 3 days old", d.filter(pl.col("age_days") <= 3)),
        ):
            x = part["absorbed_share"].to_numpy()
            if len(x) < 5:
                continue
            lo, hi = boot(x=x, rng=rng)
            out.append({
                "arm": arm, "cut_age_days": age, "events": sel, "n": len(x), "mean": x.mean(),
                "median": float(np.median(x)), "low": lo, "high": hi,
                "mean_without_largest": (x.sum() - x.max()) / (len(x) - 1), "largest": x.max(),
                "n_rows_corrected_zero": int((part["n_rows_corrected"] == 0).sum()),
            })  # fmt: skip
    return pl.DataFrame(out)


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
            .properties(width=WIDTH, height=220, title=label)
        )
    alt.hconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "One refit per fold, scored over the whole month; five folds, two seeds.",
                "Two seeds, five folds, month-old fits.",
            ],
            anchor="start",
        )
    ).save(DOCS / "figure_1_crps_by_lead_band.svg")


def fig_leak() -> None:
    """Figure 2: mean absorbed share per arm with an event-bootstrap interval."""
    d = leak_rows().filter(pl.col("cut_age_days") == 1.0).with_columns(
        share=pl.col("mean") * 100, lo=pl.col("low") * 100, hi=pl.col("high") * 100,
        arm=pl.col("arm").replace(LABELS),
    )  # fmt: skip
    base = alt.Chart(d.to_pandas()).encode(
        y=alt.Y("arm:N", sort=list(LABELS.values()), title=None),
        color=alt.Color(
            "events:N",
            title="Events",
            scale=alt.Scale(range=[theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_GREEN]),
        ),
        yOffset="events:N",
    )
    pts = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("share:Q", title="Share of the event's depth that reaches the forecast (%)")
    )
    bars = base.mark_rule().encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    flag = (
        alt.Chart(alt.Data(values=[{"x": 100 * FLAG}])).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    alt.layer(bars, pts, flag).properties(
        width=480, height=240,
        title=alt.TitleParams(
            ["Real labelled events, refit one day after the event ends.", "Dashed line: flag level 3%. Bars: 95% interval resampling events."],
            anchor="start",
        ),
    ).save(DOCS / "figure_2_real_event_leak.svg")  # fmt: skip


def main() -> None:
    """Print the tables and draw figures 1 and 2."""
    DOCS.mkdir(parents=True, exist_ok=True)
    tables()
    with pl.Config(tbl_rows=100, tbl_cols=20):
        print("--- real-event leak")
        print(leak_rows().with_columns(pl.col(pl.Float64).round(4)))
    lag = pl.read_parquet(EVENTS.parent / "lag_following.parquet")
    rng = np.random.default_rng(1)
    print(
        "--- lag-following share of the labelled depth (inference only, lags in the event corrected)"
    )
    for arm, d in lag.group_by("arm", maintain_order=True):
        x = d["lag_following_share"].to_numpy()
        lo, hi = boot(x=x, rng=rng)
        print(
            arm,
            "events",
            len(x),
            "mean",
            round(float(x.mean()), 4),
            "interval",
            round(lo, 4),
            round(hi, 4),
            "median",
            round(float(np.median(x)), 4),
            "largest",
            round(float(x.max()), 4),
        )
    fig_bands()
    fig_leak()


if __name__ == "__main__":
    main()

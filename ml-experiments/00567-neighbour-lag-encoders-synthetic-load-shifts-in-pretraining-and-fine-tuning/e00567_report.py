# ruff: noqa: E501
"""Print the numbers the experiment 567 page quotes beyond `e00567_score.py`'s tables, and draw its figures."""

from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00567-neighbour-lag-encoders-synthetic-load-shifts-in-pretraining-and-fine-tuning"
SCORES: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "scores"
EVENTS: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work/scores/real_events.parquet"
EVENTS_566: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00566-combination-block-lag-encoder-with-three-step-schedule-and-embargo-real-event-leak"
    / "work/scores/real_events.parquet"
)
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
BANDS: Final = ("0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h")
LABELS: Final = {
    "N0-r7": "N0 own lags only",
    "N1-r7": "N1 neighbour lags",
    "N1placebo-r7": "N1 placebo neighbours",
    "N1aux-r7-w0.1": "N1-aux flag head",
}
COLOURS: Final = ["#808080", theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_GREEN]
GROUPS: Final = {"mva_eight": "The eight MVA Primaries", "signed_six": "The six MW Primaries"}
FLAG: Final = 0.03
N_BOOT: Final = 2000


def fig_bands() -> None:
    """Figure 1: CRPS by lead band, one panel per group, at the 7% rate."""
    a = pl.read_csv(SCORES / "absolute.csv").filter(
        (pl.col("slice") == "all") & pl.col("arm").is_in(list(LABELS)) & pl.col("band").is_in(BANDS)
    ).with_columns(arm=pl.col("arm").replace(LABELS))  # fmt: skip
    panels = []
    for key, label in GROUPS.items():
        panels.append(
            alt.Chart(a.filter(pl.col("group") == key).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X("band:N", sort=list(BANDS), title="Lead band"),
                y=alt.Y("crps:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=list(LABELS.values()), range=COLOURS),
                    title=None,
                    legend=alt.Legend(orient="top", columns=2),
                ),
            )
            .properties(width=560, height=170, title=alt.TitleParams(label, anchor="start"))
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "CRPS by lead band, smaller is better. Synthetic load shifts touch 7% of Primary-days.",
                "One refit per fold, scored over whole months; five folds, two seeds.",
            ],
            anchor="start",
        )
    ).save(DOCS / "figure_1_crps_by_lead_band.svg")


def fig_rates() -> None:
    """Figure 2: N1 minus N0 at each synthetic rate, 24-120 h at the eight MVA Primaries."""
    c = pl.read_csv(SCORES / "contrasts.csv").filter(
        (pl.col("band") == "24-120h") & (pl.col("group") == "mva_eight")
    )
    h = pl.read_csv(SCORES / "heldout_contrasts.csv").filter(
        (pl.col("band") == "24-120h") & (pl.col("group") == "mva_eight")
    )
    rows = []
    for rate in ("r3", "r7", "r21"):
        for sl, label in (
            ("no_event", "Rows with no labelled event"),
            ("target_in_event_visible", "Real events with a visible recipient"),
        ):
            r = c.filter(
                (pl.col("slice") == sl) & (pl.col("contrast") == f"N1-{rate} - N0-{rate}")
            ).row(0, named=True)
            lo = r["low_bonferroni_month"] if sl == "no_event" else r["low_bonferroni"]
            hi = r["high_bonferroni_month"] if sl == "no_event" else r["high_bonferroni"]
            rows.append({"rate": rate, "slice": label, "diff": r["difference"], "lo": lo, "hi": hi})
        for var, label in (
            ("hx", "Held-out synthetic events, exact"),
            ("hi", "Held-out synthetic events, 60-100%"),
        ):
            r = h.filter(
                (pl.col("variant") == var) & (pl.col("contrast") == f"N1-{rate} - N0-{rate}")
            ).row(0, named=True)
            rows.append(
                {
                    "rate": rate,
                    "slice": label,
                    "diff": r["difference"],
                    "lo": r["low_bonferroni"],
                    "hi": r["high_bonferroni"],
                }
            )
    d = pl.DataFrame(rows).with_columns(
        rate=pl.col("rate").replace({"r3": "3.5%", "r7": "7%", "r21": "21%"})
    )
    print(d)
    order = [
        "Rows with no labelled event",
        "Real events with a visible recipient",
        "Held-out synthetic events, exact",
        "Held-out synthetic events, 60-100%",
    ]
    base = alt.Chart(d.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("slice:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("rate:N", sort=["3.5%", "7%", "21%"]),
        color=alt.Color(
            "rate:N", sort=["3.5%", "7%", "21%"], title="Share of Primary-days touched",
            scale=alt.Scale(domain=["3.5%", "7%", "21%"], range=[theme.DATA_SKY, theme.DATA_BLUE, theme.BRAND_ORANGE]),
            legend=alt.Legend(orient="top"),
        ),
    )  # fmt: skip
    pts = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("diff:Q", title="N1 minus N0 (points of capacity)")
    )
    bars = base.mark_rule().encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(alt.Data(values=[{"x": 0}]))
        .mark_rule(strokeDash=[4, 3], color="#444")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    alt.layer(bars, pts, zero).properties(
        width=430, height=230,
        title=alt.TitleParams(
            ["24-120 h CRPS at the eight MVA Primaries; below zero favours N1.", "Dot: estimate. Line: 95% interval, Bonferroni count 10. Dashed line: N1 equals N0."],
            anchor="start",
        ),
    ).save(DOCS / "figure_2_neighbour_encoder_minus_own_lags.svg")  # fmt: skip


def boot(*, x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Return a 95% interval of the mean, resampling events."""
    means = [rng.choice(x, size=len(x)).mean() for _ in range(N_BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fig_leak() -> None:
    """Figure 3: mean absorbed share per arm with an event-bootstrap interval."""
    ours = pl.read_parquet(EVENTS).filter(pl.col("absorbed_share").is_not_null())
    theirs = pl.read_parquet(EVENTS_566).filter(
        (pl.col("min_age") == 1.0)
        & pl.col("arm").is_in(["C0", "C1"])
        & pl.col("absorbed_share").is_not_null()
    )
    names = {
        "C0": "C0 lag-free (566)",
        "C1": "C1 own lags (566)",
        "N1-r7": "N1 neighbour lags",
        "N1aux-r7-w0.1": "N1-aux flag head",
    }
    rng = np.random.default_rng(0)
    rows = []
    for arm, d in pl.concat(
        [ours.select("arm", "absorbed_share"), theirs.select("arm", "absorbed_share")]
    ).group_by("arm", maintain_order=True):
        x = d["absorbed_share"].to_numpy()
        lo, hi = boot(x=x, rng=rng)
        rows.append(
            {
                "arm": names[arm[0]],
                "share": 100 * x.mean(),
                "lo": 100 * lo,
                "hi": 100 * hi,
                "n": len(x),
                "mean_without_largest": 100 * (x.sum() - x.max()) / (len(x) - 1),
            }
        )
        print(rows[-1])
    d = pl.DataFrame(rows)
    base = alt.Chart(d.to_pandas()).encode(y=alt.Y("arm:N", sort=list(names.values()), title=None))  # ty: ignore[unresolved-attribute]
    pts = base.mark_point(filled=True, size=70, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("share:Q", title="Share of the event's depth that reaches the forecast (%)")
    )
    bars = base.mark_rule(color=theme.BRAND_ORANGE).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    flag = (
        alt.Chart(alt.Data(values=[{"x": 100 * FLAG}])).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    alt.layer(bars, pts, flag).properties(
        width=430, height=130,
        title=alt.TitleParams(
            ["74 real labelled events, refit one day after the event ends.", "Dashed line: flag level 3%. Bars: 95% interval resampling events."],
            anchor="start",
        ),
    ).save(DOCS / "figure_3_real_event_leak.svg")  # fmt: skip


def main() -> None:
    """Draw the figures."""
    DOCS.mkdir(parents=True, exist_ok=True)
    fig_bands()
    fig_rates()
    fig_leak()


if __name__ == "__main__":
    main()

"""Experiment 714: three figures for the write-up, drawn from the saved results.

    E=ml-experiments
    export PYTHONPATH=$E/00660-shared-generator-of-injected-load-transfer-events:\
$E/00709-era-signal-in-the-candidate-part-1:\
$E/00714-msa-era-identifiability-audit-and-frozen-probe
    .venv/bin/python ml-experiments/00714-*/e00714_posthoc_figures.py

- `auroc_in_force_vs_ended.svg`: the second generator's host-row AUROC at 7, 14, and 28 days for
  all temporary moves, for the moves still in force at the data cut, and for the moves that had
  already ended (the last reading is post hoc and is computed here from the stored out-of-fold
  probabilities, with the same bootstrap as the planned contrasts and fewer draws).
- `aligned_vs_original_features.svg`: the pre-registered host AUROC beside the sign-aligned host
  features and the sign-aligned features plus three extra cues. The aligned values are post hoc
  numbers that the Opus audit computed (`briefs/msa_architecture/f1_audit.md`, section 1, S2) with
  its own private scripts, so they are constants here and carry no interval.
- `raw_cues_by_kind.svg`: the AUROC of each raw cue alone for each permanent kind against all
  temporary moves at 14 days, from `results/raw_cues_by_kind.csv`.

Each SVG is written under `docs/ml-experiments/<id>-<slug>/` and optimised with
`npx svgo@4 --multipass --precision=1 --final-newline`.
"""

import e00714_threads  # noqa: F401  (must run before numpy)

# isort: split

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00714_common as common
import e00714_figures as figs
import e00714_model as model
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

WIDTH: Final[int] = figs.WIDTH
DRAWS: Final[int] = 2000
CONFIDENCE: Final[float] = model.CONFIDENCE
HORIZONS: Final[tuple[int, ...]] = (7, 14, 28)
ALIGNED: Final[dict[str, tuple[float, float, float]]] = {
    "Pre-registered host features": (0.630, 0.637, 0.618),
    "Sign-aligned host features": (0.652, 0.664, 0.667),
    "Sign-aligned plus three extra cues": (0.636, 0.664, 0.670),
}
"""Post hoc AUROC at 7, 14, and 28 days on the second generator, from the Opus audit (S2)."""
CUE_LABEL: Final[dict[str, str]] = {
    "steps_per_day": "Steps per day",
    "high_fraction": "Share of half-hours at the high level",
    "flatness": "Flatness of the change",
    "mirror_share": "Neighbour mirror share",
    "mirror_z": "Neighbour mirror z-score",
}
CUE_COLOUR: Final[tuple[str, ...]] = (
    theme.DATA_BLUE, theme.DATA_GREEN, theme.BRAND_ORANGE, theme.DATA_PURPLE, theme.DATA_SKY,
)  # fmt: skip
KIND_LABEL: Final[dict[str, str]] = {
    "flat": "Flat step",
    "profile_prop": "Follows daily profile",
    "daylight": "Daylight only",
    "quantised": "Quantised step",
    "intermittent": "Intermittent (secondary world)",
}


def masked_auroc_interval(
    *, label: np.ndarray, score: np.ndarray, mask: np.ndarray, draws: list[np.ndarray]
) -> dict[str, float]:
    """Return the AUROC on the masked rows with its bootstrap interval over the same draws."""
    values = []
    for d in draws:
        keep = d[mask[d]]
        values.append(model.auroc(label=label[keep], score=score[keep]))
    low, high = model.interval(values=np.array(values), confidence=CONFIDENCE)
    return {"auroc": model.auroc(label=label[mask], score=score[mask]), "low": low, "high": high}


def in_force_rows(*, results: Path) -> pl.DataFrame:
    """Return the AUROC of all, in-force, and ended rows by horizon, printing each number."""
    stored = json.loads((results / "primary_contrasts.json").read_text())
    rows = []
    for k in HORIZONS:
        frame = (
            pl.read_parquet(common.DATA_DIR / "features" / "g2.parquet")
            .filter(pl.col("k") == k)
            .sort("pass_id", "event")
        )
        preds = pl.read_parquet(results / f"oof_g2_k{k}.parquet")
        assert (preds["pass_id"] == frame["pass_id"]).all()
        assert (preds["label"] == frame["label"]).all()
        label, score = frame["label"].to_numpy(), preds["host"].to_numpy()
        in_force = frame["in_force"].to_numpy()
        permanent = label == 1
        draws = model.bootstrap_draws(
            months=frame["month"].to_numpy(), passes=frame["pass_id"].to_numpy(),
            n_draws=DRAWS, seed=714_900 + k,
        )  # fmt: skip
        planned = stored["auroc"][f"g2/k{k}/host"]
        readings = {
            "All temporary moves": {
                "auroc": planned["auroc"],
                "low": planned["low"],
                "high": planned["high"],
            },
            "Moves still in force": masked_auroc_interval(
                label=label, score=score, mask=permanent | in_force, draws=draws
            ),
            "Moves already ended": masked_auroc_interval(
                label=label, score=score, mask=permanent | ~in_force, draws=draws
            ),
        }
        share_ended = float(1.0 - in_force[label == 0].mean())
        print(f"k={k}: share of temporary moves ended {share_ended:.3f}")
        for name, value in readings.items():
            print(f"  {name}: {value['auroc']:.3f} [{value['low']:.3f}, {value['high']:.3f}]")
            rows.append({"days": k, "reading": name, **value})
    return pl.DataFrame(rows)


def in_force_figure(*, results: Path, out: Path) -> None:
    """Draw the host-row AUROC for all, in-force, and ended temporary moves."""
    frame = in_force_rows(results=results)
    names = ["All temporary moves", "Moves still in force", "Moves already ended"]
    colours = [theme.BLACK_1, theme.BRAND_ORANGE, theme.DATA_BLUE]
    base = alt.Chart(frame).encode(
        x=alt.X("days:O", title="Days after the onset", axis=alt.Axis(labelAngle=0)),
        xOffset="reading:N",
        color=alt.Color(
            "reading:N", title=None,
            scale=alt.Scale(domain=names, range=colours),
            legend=alt.Legend(orient="top"),
        ),
    )  # fmt: skip # ty: ignore[unresolved-attribute]
    bars = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("low:Q", title="AUROC", scale=alt.Scale(domain=[0.3, 0.9])), y2="high:Q"
    )
    points = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        y="auroc:Q"
    )
    chance = (
        alt.Chart(pl.DataFrame({"y": [0.5]}))
        .mark_rule(strokeDash=[4, 3], color="grey")
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = alt.layer(bars, points, chance).properties(
        width=WIDTH, height=260,
        title=alt.TitleParams(
            "Host rows separate ended temporary moves from permanent changes, not sustained ones",
            subtitle=[
                "AUROC on the second generator for permanent changes against each group of",
                "temporary moves. Dashed line: chance (0.50). Bars: 99.17% intervals;",
                "the ended-move reading is post hoc.",
            ],
        ),
    )  # fmt: skip
    figs.save(chart=chart, path=out / "auroc_in_force_vs_ended.svg")


def aligned_figure(*, out: Path) -> None:
    """Draw the pre-registered host AUROC beside the sign-aligned post hoc readings."""
    rows = [
        {"days": k, "set": name, "auroc": values[i]}
        for name, values in ALIGNED.items()
        for i, k in enumerate(HORIZONS)
    ]
    names = list(ALIGNED)
    colours = [theme.DATA_BLUE, theme.DATA_GREEN, theme.BRAND_ORANGE]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        x=alt.X("days:O", title="Days after the onset", axis=alt.Axis(labelAngle=0)),
        xOffset="set:N",
        color=alt.Color(
            "set:N", title=None, scale=alt.Scale(domain=names, range=colours),
            legend=alt.Legend(orient="top", columns=1, labelLimit=500),
        ),
    )  # fmt: skip # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("auroc:Q", title="AUROC", scale=alt.Scale(domain=[0.5, 0.8]))
    )
    text = base.mark_text(dy=-12, fontSize=10).encode(  # ty: ignore[unresolved-attribute]
        y="auroc:Q", text=alt.Text("auroc:Q", format=".2f")
    )
    bar = (
        alt.Chart(pl.DataFrame({"y": [0.7]}))
        .mark_rule(strokeDash=[4, 3], color="grey")
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = alt.layer(points, text, bar).properties(
        width=WIDTH,
        height=240,
        title=alt.TitleParams(
            "Sign-aligned features raise the AUROC by 0.006 to 0.052, still below 0.70",
            subtitle=[
                "AUROC of host rows on the second generator. Dashed line: the threshold of 0.70.",
                "The two sign-aligned sets are post hoc, from the audit, and carry no interval.",
            ],
        ),
    )
    figs.save(chart=chart, path=out / "aligned_vs_original_features.svg")


def raw_cue_figure(*, results: Path, out: Path) -> None:
    """Draw each raw cue's AUROC by permanent kind at 14 days against all temporary moves."""
    frame = (
        pl.read_csv(results / "raw_cues_by_kind.csv")
        .filter(
            (pl.col("world") == "g2_gas")
            & (pl.col("k") == 14)
            & (pl.col("negatives") == "all_temporary")
        )
        .with_columns(
            kind_label=pl.col("kind").replace_strict(KIND_LABEL),
            cue_label=pl.col("cue").replace_strict(CUE_LABEL),
        )
    )
    print(frame.select("kind", "cue", "auroc", "low95", "high95").sort("kind", "cue"))
    base = alt.Chart(frame).encode(
        y=alt.Y(
            "kind_label:N", title=None, sort=list(KIND_LABEL.values()),
            axis=alt.Axis(labelLimit=260),
        ),
        yOffset=alt.YOffset("cue_label:N", sort=list(CUE_LABEL.values())),
        color=alt.Color(
            "cue_label:N", title=None,
            scale=alt.Scale(domain=list(CUE_LABEL.values()), range=list(CUE_COLOUR)),
            legend=alt.Legend(orient="top", columns=2, labelLimit=500),
        ),
    )  # fmt: skip # ty: ignore[unresolved-attribute]
    bars = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low95:Q", title="AUROC of the cue alone", scale=alt.Scale(domain=[0.1, 1.0])),
        x2="high95:Q",
    )
    points = base.mark_point(filled=True, size=40).encode(x="auroc:Q")  # ty: ignore[unresolved-attribute]
    chance = (
        alt.Chart(pl.DataFrame({"x": [0.5]}))
        .mark_rule(strokeDash=[4, 3], color="grey")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = alt.layer(bars, points, chance).properties(
        width=WIDTH - 200, height=300,
        title=alt.TitleParams(
            "No single raw cue separates every kind of permanent change from temporary moves",
            subtitle=[
                "AUROC at 14 days of one cue with no fitting, oriented so that a higher score",
                "means a permanent change (below 0.50: the cue points the wrong way).",
                "Positives: one permanent kind. Negatives: all temporary moves.",
                "Dashed line: chance. Bars: 95% intervals, secondary.",
            ],
        ),
    )  # fmt: skip
    figs.save(chart=chart, path=out / "raw_cues_by_kind.svg")


def main() -> None:
    """Draw the three figures."""
    results = common.DATA_DIR / "results"
    in_force_figure(results=results, out=figs.DOCS)
    aligned_figure(out=figs.DOCS)
    raw_cue_figure(results=results, out=figs.DOCS)


if __name__ == "__main__":
    main()

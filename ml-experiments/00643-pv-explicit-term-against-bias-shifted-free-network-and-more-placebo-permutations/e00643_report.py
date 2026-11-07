# ruff: noqa: E501
"""Experiment 643: print every number the write-up quotes and draw its figures.

Usage: python e00643_report.py
"""

import json
import os
from pathlib import Path

import altair as alt
import e00639_analyse as a639  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00643_analyse as an
import e00643_common as k
import numpy as np
import polars as pl

DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / k.NAME
W = 560
NAMES = {
    "free": "Free network (plain)", "fbias": "Free, shifted by training-month constant",
    "foff": "Free, trainable offset", "fhind": "Free, shifted in hindsight (bound)",
}  # fmt: skip
COLOURS = {"free": "#FF4901", "fbias": "#306BFF", "foff": "#10C5F7", "fhind": "#B701FF"}


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write the SVG, and a PNG for review when `PNG_DIR` is set."""
    chart.save(DOCS / f"{name}.svg")
    if os.environ.get("PNG_DIR"):
        chart.save(Path(os.environ["PNG_DIR"]) / f"{name}.png", scale_factor=1.0)


def contrasts_figure(results: list[dict]) -> None:
    """Draw each control minus `cal` on sunny middays with its interval, per world."""
    recs = [
        {"world": "Real development months" if r["world"] == "real" else "Step world (4 MW added from April 2023)",
         "arm": NAMES[r["arm"]], "key": r["arm"], "diff": r["diff"], "lo": r["lo"], "hi": r["hi"]}
        for r in results
        if "absolute" not in r and r["subset"] == "sunny" and r["arm"] in NAMES
    ]  # fmt: skip
    order = [NAMES[a] for a in NAMES]
    panels = []
    for world in dict.fromkeys(r["world"] for r in recs):
        df = pl.DataFrame([r for r in recs if r["world"] == world])
        base = alt.Chart(df).encode(
            y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("arm:N", scale=alt.Scale(domain=order, range=[COLOURS[a] for a in NAMES]), legend=None),
        )  # fmt: skip
        pts = base.mark_point(filled=True, size=70).encode(
            x=alt.X(
                "diff:Q",
                title="Free arm's CRPS minus the explicit network's (points; right favours the explicit network)",
            )
        )
        bars = base.mark_rule(strokeWidth=2).encode(x="lo:Q", x2="hi:Q")
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color="#999999", strokeDash=[4, 3])
            .encode(x="x:Q")
        )
        panels.append(alt.layer(zero, bars, pts).properties(width=W, height=120, title=world))
    chart = alt.vconcat(*panels).resolve_scale(x="independent")
    save(chart, "free_arms_minus_explicit.png".replace(".png", ""))


def placebo_figure(results: list[dict]) -> None:
    """Draw each permutation's CRPS minus `cal`'s in the step world, with the mean and the bar."""
    recs = [
        {"label": r["arm"].replace("perm_mean", "Mean of 5").replace("perm", "Shift "), "diff": r["diff"], "lo": r["lo"], "hi": r["hi"]}
        for r in results
        if "absolute" not in r and r["world"] == "step" and r["subset"] == "sunny" and r["arm"].startswith("perm")
    ]  # fmt: skip
    df = pl.DataFrame(recs)
    order = [f"Shift {s}" for s in k.SHIFTS] + ["Mean of 5"]
    base = alt.Chart(df).encode(y=alt.Y("label:N", sort=order, title=None))
    pts = base.mark_point(filled=True, size=70, color="#B701FF").encode(
        x=alt.X(
            "diff:Q", title="Placebo CRPS minus explicit CRPS (points; step world, sunny middays)"
        )
    )
    bars = base.mark_rule(strokeWidth=2, color="#B701FF").encode(x="lo:Q", x2="hi:Q")
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#999999", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    bar = alt.Chart(pl.DataFrame({"x": [k.MDE["step"]]})).mark_rule(color="#FF4901").encode(x="x:Q")
    save(alt.layer(zero, bar, bars, pts).properties(width=W, height=170), "placebo_permutations")


def print_checks() -> None:
    """Print the learned shifts and offsets, per-cutoff contrasts, and the median checks."""
    for world in k639.WORLDS:
        for cutoff in k639.CUTOFFS:
            sh = json_load(k.WORK / "analysis" / f"fbias_shift_{cutoff}_{world}.json")
            arr = np.array(list(sh["shift_points_by_seed"].values()))
            six = [i for i in range(16) if np.any(arr[:, i] != 0)]
            print(
                f"fbias shift {world} {cutoff}: held-out sunny rows {sh['rows']}, points of capacity x100 by Primary index {six}: {np.round(100 * arr.mean(axis=0)[six], 2).tolist()}"
            )
            offs = [
                json_load(
                    k.weights_path(cutoff=cutoff, world=world, arm="foff", seed=s).with_suffix(
                        ".json"
                    )
                )["delta_mw"]
                for s in k.STAGE1_SEEDS
            ]
            nz = np.array(offs).mean(axis=0)
            print(
                f"foff offset {world} {cutoff}: mean over seeds (MW) {np.round(nz[nz != 0], 3).tolist()}"
            )
    for world in k639.WORLDS:
        rows = a639.load(world=world)
        cal = an._arm_scores(world=world, arm="cal", seeds=k.STAGE1_SEEDS)
        for arm in ("fbias", "foff"):
            s = an._arm_scores(world=world, arm=arm, seeds=k.STAGE1_SEEDS)
            d = (s - cal).mean(axis=0)
            parts = [float(d[rows["sunny"] & (rows["cutoff"] == ci)].mean()) for ci in range(3)]
            prim = [
                float(d[rows["sunny"] & (rows["primary"] == p)].mean())
                for p in np.unique(rows["primary"])
            ]
            print(
                f"{world} {arm} minus cal by cutoff {np.round(parts, 3).tolist()}, by Primary {np.round(prim, 3).tolist()}"
            )
        for arm in ("free", "fbias", "foff", "cal"):
            med, gap, shift = [], [], []
            for cutoff in k639.CUTOFFS:
                z = np.load(k639.test_rows_path(cutoff=cutoff, world=world))
                six = z["inc"] > 0
                q = np.mean(
                    [
                        np.load(k.prediction_path(cutoff=cutoff, world=world, arm=arm, seed=sd))[
                            "q"
                        ][six]
                        for sd in k.STAGE1_SEEDS
                    ],
                    axis=0,
                )
                approx = np.trapezoid(q, np.asarray(k639.LEVELS), axis=1) / (
                    k639.LEVELS[-1] - k639.LEVELS[0]
                )
                y = z["y"][six]
                med.append(100 * np.abs(q[:, 5] - y))
                gap.append(100 * (q[:, 5] - approx))
                shift.append(100 * (q[:, 5] - y))
            sun = rows["sunny"]
            print(
                f"{world} {arm}: sunny median abs error {np.concatenate(med)[sun].mean():.2f}, median minus mean {np.concatenate(gap)[sun].mean():+.2f}, median forecast minus actual {np.concatenate(shift)[sun].mean():+.2f}"
            )


def json_load(path: Path) -> dict:
    """Read a JSON file."""
    return json.loads(path.read_text())


def main() -> None:
    """Print the checks and draw the figures."""
    results = json_load(k.WORK / "analysis" / "contrasts_stage1.json")
    print_checks()
    DOCS.mkdir(parents=True, exist_ok=True)
    contrasts_figure(results)
    placebo_figure(results)


if __name__ == "__main__":
    main()

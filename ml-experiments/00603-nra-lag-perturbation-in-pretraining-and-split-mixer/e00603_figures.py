# ruff: noqa: E501, ANN001, ANN201, ANN003, E402, PT018, B023, F821, ANN202, D103, SIM115, PTH123, PERF401, UP037, F841, RUF100
"""Experiment 603: coverage of the normal running arrangement by arm, phase, and metering group (pilot month)."""

import csv
import sys
from pathlib import Path

import altair as alt
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = ML_EXPERIMENTS_DATA_DIR / "00603-nra-lag-perturbation-in-pretraining-and-split-mixer"
ARMS = {
    "c1": "J0",
    "j1": "J1",
    "j2": "J2",
    "j3": "J3",
    "j3c": "J3c",
    "j4": "J4",
    "j4b": "J4b",
    "j5": "J5",
}
PHASES = {
    "days4-end": "day 4 to end",
    "after0-7": "0-7 days after end",
    "after7-14": "7-14 days after end",
}
COLOURS = ["#FF4901", "#306BFF", "#10C5F7"]
out = Path(sys.argv[1])
data = []
for a, label in ARMS.items():
    for r in csv.DictReader(open(ROOT / f"score_{a}_pilot/bank_summary.csv")):
        if (
            r["length_class"] == "15 and 30 days"
            and r["band"] == "24-120 h"
            and r["arm"] == "V"
            and r["phase"] in PHASES
            and r["group"] in {"MVA-only eight", "signed-MW six"}
        ):
            data.append(
                {
                    "arm": label,
                    "group": "the eight MVA Primaries"
                    if r["group"].startswith("MVA")
                    else "the six MW Primaries",
                    "phase": PHASES[r["phase"]],
                    "coverage": float(r["cover90"]),
                }
            )
chart = (
    alt.Chart(alt.Data(values=data))
    .mark_point(filled=True, size=90, clip=True)
    .encode(
        x=alt.X("arm:N", sort=list(ARMS.values()), title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("phase:N", sort=list(PHASES.values())),
        y=alt.Y(
            "coverage:Q",
            scale=alt.Scale(domain=[0.6, 1.0]),
            title="Share of rows inside the q05-q95 band",
        ),
        color=alt.Color(
            "phase:N",
            sort=list(PHASES.values()),
            scale=alt.Scale(domain=list(PHASES.values()), range=COLOURS),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    .properties(width=620, height=280)
    .facet(
        row=alt.Row("group:N", title=None, sort=["the eight MVA Primaries", "the six MW Primaries"])
    )
    .properties(
        title="Coverage after injected 15 and 30 day load-shifts, 24-120 h, December 2025 only"
    )
)
chart.save(out / "coverage_by_arm.svg")

# ruff: noqa: E402
# ruff: noqa: E501
"""Experiment 732: the counted contrasts drawn in three groups of one unit each (new, read only, CPU).

`scores/readings.json` holds every contrast. The first figure of the stage `figures` draws C1 to C7 on
one axis, which hides C1, C5, C6, and C7 behind C4 because C4 is in points of solar error and the others
are in Brier score. This script writes three figures to `figures/`: the sign contrasts (C1, C6, R4, R1,
R2, R5, R13, R19, R21), the step-event contrasts (C5, C7, R15), and the solar contrasts (C2, C3, C4,
R6, R10, R23). It changes no stored file.
"""

import json
import sys

import e00732_paths as paths

paths.pin_threads()

import e00732_figures as figures

SIGN = ("C1", "C6", "R4", "R1", "R2", "R5", "R13", "R19", "R21")
STEP = ("C5", "C7", "R15")
SOLAR = ("C2", "C3", "C4", "R6", "R10", "R23")


def main() -> None:
    """Draw the three figures."""
    root = paths.data_root()
    rec = json.loads((root / "scores" / "readings.json").read_text())["readings"]
    allr = {**rec["counted"], **rec["report"]}
    for name, keys, title in (
        (
            "contrasts_sign",
            SIGN,
            "Sign: Brier score of P(X<0), first arm minus second (negative favours the first)",
        ),
        ("contrasts_step", STEP, "Transfer step: Brier score of the step event"),
        ("contrasts_solar", SOLAR, "Solar output: mean absolute error in points of capacity"),
    ):
        chart = figures.contrast_chart(readings={k: allr[k] for k in keys}, title=title)
        figures.save_svg(chart=chart, path=root / "figures" / f"{name}.svg")
        print("wrote", name)


if __name__ == "__main__":
    sys.exit(main())

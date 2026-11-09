# ruff: noqa: E501
"""Experiment 673, figure (b): S(u) curves for D19 (stored by experiment 671) and D12 (re-probed here).

D19's cell is read from experiment 671's stored `pf_twin_s0.json`. D12's cell is read from
`work/d12_d19_probe_seed0.json`, which `e00673_d12_probe.py` wrote with the same probe code on
experiment 662's stored uninjected network (2026-06 cutoff, seed 0). Nothing is fitted network-wise;
the linear and fold fits are the probe's own regressions on the probed readings.

Usage: python e00673_s_of_u.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00673-pv-capacity-for-nged-per-primary-with-register-cross-check"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
E671: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00671-pv-mva-fold-probe-fair-denominator-and-d3-d4-disagreement/work/part2/pf_twin_s0.json"
)


def fold_curve(*, c: dict, name: str) -> np.ndarray:
    """Return the fitted fold-model reading at the stored cell's rows (as in experiment 671)."""
    th = np.array(c[f"fold_{name}"]["theta"])
    a = c[f"fold_{name}"]["a"]
    kk = len(th) - 3
    ub, dt = np.array(c["u_basis"]), np.array(c["dt"])
    mm = a - ub @ th[:kk] + th[kk] * dt
    qq = th[kk + 1] + th[kk + 2] * a
    return np.sqrt(mm**2 + qq**2)


def main() -> None:
    """Draw figure (b) and print each cell's capacities and slopes."""
    d19 = json.loads(E671.read_text())["D19"]
    re = json.loads((WORK / "d12_d19_probe_seed0.json").read_text())
    cells = {
        "D19 (experiment 671's stored cell)": d19["curve"],
        "D12 (re-probed for this page)": re["D12"]["curve"],
    }
    check = re["D19"]
    print(
        "D19 re-probe check: L0",
        round(check["L0"], 3),
        "FA_free",
        round(check["FA_free"], 3),
        "vs 671 seed 0:",
        round(d19["L0"], 3),
        round(d19["FA_free"], 3),
    )
    rows = []
    for label, c in cells.items():
        u, y = np.array(c["u"]), np.array(c["y"])
        rows += [
            {"panel": label, "u": a, "S": b, "series": "Probed reading"}
            for a, b in zip(u, y, strict=True)
        ]
        rows += [
            {"panel": label, "u": a, "S": b, "series": "Linear fit"}
            for a, b in zip(u, c["a_lin"] - c["c_lin"] * u, strict=True)
        ]
        for name, lab in (
            ("free", "Fold fit, reactive terms free"),
            ("Z", "Fold fit, reactive terms zero"),
        ):
            rows += [
                {"panel": label, "u": a, "S": b, "series": lab}
                for a, b in zip(u, fold_curve(c=c, name=name), strict=True)
            ]
        slope = np.polyfit(u, y, 1)[0]
        print(
            label,
            "cell hh",
            c["cell_hh"],
            "month",
            c["cell_month"],
            "linear c",
            round(c["c_lin"], 2),
            "overall slope dS/du",
            round(float(slope), 2),
            "u range",
            round(float(u.min()), 2),
            round(float(u.max()), 2),
        )
    df = pl.DataFrame(rows)
    dom = [
        "Probed reading",
        "Linear fit",
        "Fold fit, reactive terms free",
        "Fold fit, reactive terms zero",
    ]
    rng = [theme.DATA_SKY, theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_PURPLE]
    base = alt.Chart(df).encode(
        x=alt.X("u:Q", title="Solar-output curve u (donor weather)"),
        y=alt.Y("S:Q", title="Probed reading S (MVA)", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=dom, range=rng),
            title=None,
            legend=alt.Legend(orient="top", columns=2, labelLimit=320),
        ),
    )
    pts = base.transform_filter(alt.datum.series == "Probed reading").mark_point(
        filled=True, size=14, opacity=0.5, aria=False
    )
    lines = base.transform_filter(alt.datum.series != "Probed reading").mark_line(
        strokeWidth=2, aria=False
    )
    chart = (
        (pts + lines)
        .properties(width=280, height=190)
        .facet(facet=alt.Facet("panel:N", sort=list(cells), title=None), columns=2)
        .properties(
            title=alt.TitleParams(
                "S(u) of one probe cell at D19 and D12, with the linear and fold fits",
                subtitle=[
                    "Uninjected 2026-06 network of experiment 662, seed 0, cell nearest 12:00 UTC in July.",
                    "Reactive model: power factor 0.97 (the PF arm).",
                ],
                anchor="start",
                fontSize=13,
                subtitleFontSize=11,
            )
        )
    )
    chart.save(str(DOCS / "s_of_u_d19_d12.svg"))


if __name__ == "__main__":
    main()

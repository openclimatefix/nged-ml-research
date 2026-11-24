# ruff: noqa: E501
"""Print every number the write-up quotes and draw its figures as SVG.

Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.
Figure 3 goes through `engine_figure_guard`. D8 is a substation.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00196-*/e00196_report.py
"""

import json
import re
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00196_common import (
    CLEAN_END,
    DATA_DIR,
    FIT_END,
    FIT_START,
    POST_START,
    stamp,
)
from e00196_hmm import rolling_offset
from e00196_pipeline import load_series

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine_figure_guard as guard

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00196-ps2-quantised-engine-detector-on-d8-with-known-answer-gate"
)
SOLAR_MONTH: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
GREY: Final[str] = "#888888"


def title(text: str, sub: str) -> alt.TitleParams:
    """A wrapped title and subtitle."""
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(sub, 90),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def load(variant: str) -> dict:
    """Gate results of one variant."""
    return json.loads((DATA_DIR / f"gate_results_{variant}.json").read_text())["wide"]


def overall_recall(*, sc: dict) -> float:
    """Recall over all running half-hours from the 1-unit and 2-3-unit recalls."""
    n = sc["n_units"]
    n1, n23 = n["1"], n["2"] + n["3"]
    return (sc["recall_1_unit"]["value"] * n1 + sc["recall_2_or_3_units"]["value"] * n23) / (
        n1 + n23
    )


def positive_spells() -> None:
    """Runs of 4 or more half-hours more than 1.5 MW above a 30-day rolling median, per 100 days."""
    s = load_series(label="D8")
    off = rolling_offset(y=s.y, load=s.load, g_unused=0.0, q=0.14)
    dev = s.resid - off
    up = np.nan_to_num(dev) > 1.5
    edges = np.flatnonzero(np.diff(np.concatenate([[0], up.astype(int), [0]])))
    starts, ends = edges[::2], edges[1::2]
    keep = (ends - starts) >= 4
    starts = starts[keep]
    for name, lo, hi in (
        ("clean history", 0, stamp(CLEAN_END)),
        (
            "2023-09 to 2025-02",
            stamp(FIT_START),
            stamp(SOLAR_MONTH),
        ),
        (
            "2025-03 to 2025-12",
            stamp(SOLAR_MONTH),
            stamp(FIT_END),
        ),
        ("2026-04 to 2026-06", stamp(POST_START), len(s.y)),
    ):
        n = int(((starts >= lo) & (starts < hi)).sum())
        print(
            f"positive spells (>=4 half-hours above +1.5 MW) {name}: {n} in "
            f"{(hi - lo) / 48:.0f} days = {100 * n / ((hi - lo) / 48):.1f} per 100 days"
        )


def draw_figure_3(*, base: dict, local: dict) -> None:
    """Draw the flagged share and left-tail share pooled over four eras; no dates, no power values."""
    stop = 2023 * 12 + 8
    solar = 2025 * 12 + 2
    fall = 2026 * 12 + 1
    eras = ["Era 1", "Era 2", "Era 3", "Era 4"]

    def era_of(month: int) -> str:
        if month < stop:
            return eras[0]
        if month < solar:
            return eras[1]
        return eras[2] if month < fall else eras[3]

    rows = []
    for variant, r in (("Fixed noise", base), ("Local noise (post hoc)", local)):
        rows += [
            {
                "era": era_of(m["month"] - 1),
                "series": f"Flagged running: {variant}",
                "share": m["flag"],
            }
            for m in r["monthly"]
            if m["month"] >= 2022 * 12 + 1
        ]
    rows += [
        {"era": era_of(m["month"] - 1), "series": "Left tail", "share": m["left"]}
        for m in base["monthly"]
    ]
    pooled = pl.DataFrame(rows).group_by("era", "series").agg(share=pl.col("share").mean())
    names = ["Flagged running: Fixed noise", "Flagged running: Local noise (post hoc)", "Left tail"]
    chart = (
        alt.Chart(pooled)
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("series:N", title=None, sort=names, axis=None),
            y=alt.Y("share:Q", title="Mean share of half-hours", scale=alt.Scale(domain=[0, 0.7])),
            color=alt.Color(
                "series:N",
                sort=names,
                scale=alt.Scale(
                    domain=names, range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=400),
            ),
            column=alt.Column(
                "era:N",
                title=None,
                sort=eras,
                header=alt.Header(labelAngle=0, labelLimit=120, labelFontSize=9),
            ),
        )
        .properties(
            width=90,
            height=240,
            title=title(
                "The detector over-flags after solar was connected, when the noise is larger",
                "D8, mean over the months of each era. Era 1: before the engine's readings stop. Era 2: readings stopped. Era 3: solar connected. Era 4: left tail fallen.",
            ),
        )
    )
    guard.save_chart(chart=chart, path=OUT_DIR / "figure_3_flagged_share_by_month.svg")


def main() -> None:
    """Print and draw."""
    base = load("fixed")
    local = load("local")
    print(
        "unit size (gate fit):",
        base["g"],
        base["g_path"],
        "local-noise:",
        local["g"],
        local["g_path"],
    )
    for variant, r in (("fixed noise (pre-registered)", base), ("local noise (post hoc)", local)):
        print("==", variant)
        for arm in ("K1", "K2", "synthetic_x0.5", "synthetic_x2.0", "q_0.0", "q_0.5", "q_1.0"):
            sc = r[arm]
            print(
                arm,
                "units",
                sc["n_units"],
                {
                    k: [round(sc[k][x], 3) for x in ("value", "lo", "hi")]
                    for k in (
                        "recall_1_unit",
                        "recall_2_or_3_units",
                        "precision",
                        "energy_error",
                        "flagged_share",
                    )
                },
                "filtered recall",
                round(sc["recall_filtered_any_running"], 3),
                "exact-k",
                round(sc["exact_k_accuracy"], 3),
                "pass",
                all(sc["passes"].values()),
                sc["passes"],
            )
        print("false alarms clean even months", r["false_alarms_clean_even_months"])
        print("false alarms clean all", r["false_alarms_clean_all"])
        for k in r:
            if k.startswith(("flagged_share_", "left_tail_share_")):
                print(k, round(r[k], 4))
    for variant in ("fixed", "local"):
        prim = json.loads((DATA_DIR / f"primaries_results_{variant}.json").read_text())
        for label, d in prim.items():
            print(
                variant,
                label,
                round(d["wide"]["flagged_share"], 4),
                round(d["wide"]["false_active_days_per_month"], 2),
                d["wide"]["flagged_share_by_year"],
            )
    baselines = json.loads((DATA_DIR / "baseline_results.json").read_text())
    for arm, d in baselines.items():
        for det, c in d.items():
            print(
                "baseline",
                arm,
                det,
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()},
            )
    positive_spells()

    # Figure 1: the engine detector against the step detectors
    rows = []
    for arm in ("K1", "K2"):
        sc = base[arm]
        rows.append(
            {
                "detector": "Engine detector (this experiment)",
                "arm": arm,
                "recall": overall_recall(sc=sc),
            }
        )
        for det, c in baselines[arm].items():
            rows.append({"detector": det, "arm": arm, "recall": c["half_hour_recall_found_spells"]})
    frame = pl.DataFrame(rows)
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("detector:N", title=None, sort=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X(
                "recall:Q",
                title="Share of injected running half-hours found (larger is better)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            yOffset="arm:N",
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=["K1", "K2"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=520,
            height=240,
            title=title(
                (
                    "The engine detector finds the injected engine; step detectors find "
                    "a quarter or "
                    "less"
                ),
                (
                    "Share of the half-hours in which the injected engine ran that each detector "
                    "flags. Step detectors get credit for the whole spell when an alarm lies "
                    "within 2 "
                    "hours of its start or end. K1: exact replay at D8. K2: nine shifted copies."
                ),
            ),
        )
    )
    chart.save(OUT_DIR / "figure_1_against_step_detectors.svg")

    # Figure 2: profile likelihood of the unit size
    log = (DATA_DIR / "fit_wide.log").read_text()
    prof = [
        (int(i), float(g), float(v))
        for i, (g, v) in enumerate(
            [(m[0], m[1]) for m in re.findall(r"g=(\d\.\d+) MW, objective (-?\d+)", log)]
        )
    ]
    per_round = 11
    rounds = [prof[i : i + per_round] for i in range(0, len(prof), per_round)]
    pf = pl.DataFrame(
        [
            {"round": f"fit {r + 1}", "g": g, "ll": v - max(x[2] for x in rr)}
            for r, rr in enumerate(rounds)
            for _, g, v in rr
        ]
    )
    (
        alt.Chart(pf)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("g:Q", title="Unit size (MW)", scale=alt.Scale(domain=[1, 3.5])),
            y=alt.Y("ll:Q", title="Log-likelihood minus the best (nats)"),
            color=alt.Color(
                "round:N",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=520,
            height=220,
            title=title(
                "The data prefer a unit of 1.5 to 1.75 MW, and a 2 MW unit fits worse",
                (
                    "Log-likelihood of D8's residual from 2023-09 to 2025-12 at each unit size, "
                    "transitions refitted at each size. Fit 1 is the first detection; fits 2 and 3 "
                    "follow each re-centring."
                ),
            ),
        )
    ).save(OUT_DIR / "figure_2_unit_size_profile.svg")

    draw_figure_3(base=base, local=local)
    # Figure 4 (one known-answer week of the engine's recorded output) is not drawn: an output
    # series of a metered generator cannot be published, even without dates.
    print("figures written")


if __name__ == "__main__":
    main()

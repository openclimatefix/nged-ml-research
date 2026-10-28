"""Print every number quoted in experiment 376's write-up, run the control gate, and draw figures.

Reads `work/scores.json` of this experiment and of experiment 372 (the D15 references).

    PYTHONPATH=ml-experiments/00376-all-primaries-gam-stack-with-mva-treatment-arms \
        uv run --no-sync python ml-experiments/00376-*/e00376_report.py
"""

import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Final

os.environ.setdefault("E00376_SOURCE", "control")
os.environ.setdefault("E00376_ARM", "a0")

import altair as alt
import e00376s2_common as s2common
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "all-primaries-gam-stack-with-mva-treatment-arms"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00376-{SLUG}" / "work"
SCORES_372: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00372-*/work/scores.json"))
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00376-{SLUG}"
)
SIGNED: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
MVA: Final[list[str]] = ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
ORDER: Final[list[str]] = [*SIGNED, "D18", *MVA, "D15"]
STORED_155: Final[str] = "00155-*/work/dp_preds/d_scored_{site}.parquet"


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _fmt(triple: list[float]) -> str:
    point, low, high = triple
    return f"{point:+.2f} ({low:+.2f} to {high:+.2f})"


def gate() -> None:
    """Compare this experiment's stage-2 arm d with experiment 155's stored arm d."""
    print("== control gate: arm d scored vs experiment 155's stored arm d (max abs q50 difference)")
    for site in SIGNED:
        mine = pl.read_parquet(s2common.out_path(arm="d", period="scored", site=site))
        stored = [
            pl.read_parquet(p) for p in ML_EXPERIMENTS_DATA_DIR.glob(STORED_155.format(site=site))
        ]
        if not stored:
            print(site, "no stored arm d")
            continue
        joined = mine.join(
            stored[0], on=["power_fcst_init_time", "valid_time", "seed"], suffix="_s", how="inner"
        )
        print(site, joined.height, (joined["q50"] - joined["q50_s"]).abs().max())
        assert joined.height == mine.height
        assert (joined["q50"] - joined["q50_s"]).abs().max() < 1e-6


def print_scores(*, scores: dict) -> None:
    """Print absolute scores, contrasts, tails, and checks."""
    for group, sites, arms in (
        ("signed", SIGNED, ("stack",)),
        ("mva", MVA, ("chosen2", "n2", "a1")),
        ("short", ["D18"], ("stack",)),
    ):
        data = scores[group]
        print(f"== {group}: absolute, all half-hours")
        for site in sites:
            print(site, {k: round(v, 2) for k, v in data["absolute"][site]["all"].items()})
        for flag in ("sunny", "winter_night", "cold_night", "e7_step"):
            print(
                group,
                flag,
                "absolute, mean over sites",
                {
                    n: round(mean(data["absolute"][s][flag][n] for s in sites), 2)
                    for n in (*arms, "bg2", "climatology", "manual")
                },
            )
        print(f"== {group}: contrasts, pooled")
        for label, by_flag in data["contrast"].items():
            for flag, result in by_flag.items():
                print(label, flag, _fmt(result["pooled"]))
        print(f"== {group}: contrasts per Primary, all half-hours")
        for label, by_flag in data["contrast"].items():
            for site, triple in zip(by_flag["all"]["sites"], by_flag["all"]["site"], strict=True):
                print(label, site, _fmt(triple))
        print(f"== {group}: leave one out")
        for label, by_site in data.get("leave_one_out", {}).items():
            for left, triple in by_site.items():
                print(label, "without", left, _fmt(triple))
        print(f"== {group}: tails, per Primary, all half-hours")
        for site in sites:
            print(
                site,
                {
                    a: {q: round(v, 3) for q, v in shares.items()}
                    for a, shares in data["tails"][site]["all"].items()
                    if a in (*arms, "bg2")
                },
            )
        print(f"== {group}: tails, mean over sites")
        for flag in ("all", "sunny", "cold_night"):
            print(
                flag,
                {
                    a: {
                        q: round(mean(data["tails"][s][flag][a][q] for s in sites), 3)
                        for q in ("q01", "q05", "q95", "q99")
                    }
                    for a in (*arms, "bg2")
                },
            )
        print(f"== {group}: checks")
        for site in sites:
            print(site, data["checks"][site])
        print(group, "rows", data["rows"])
    print("== mva without D8", scores["mva_without_d8"])


def figures(*, scores: dict, scores_372: dict) -> None:
    """Draw the contrast figure, the absolute-score figure, and the tail figure."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for site, (point, low, high) in zip(
        scores["signed"]["contrast"]["stack_minus_bg2"]["all"]["sites"],
        scores["signed"]["contrast"]["stack_minus_bg2"]["all"]["site"],
        strict=True,
    ):
        rows.append({"site": site, "arm": "Stack (signed MW)", "p": point, "lo": low, "hi": high})
    for label, arm in (
        ("chosen_minus_bg2", "Chosen fit (MVA)"),
        ("naive_minus_bg2", "Naive fit (MVA)"),
    ):
        for site, (point, low, high) in zip(
            scores["mva"]["contrast"][label]["all"]["sites"],
            scores["mva"]["contrast"][label]["all"]["site"],
            strict=True,
        ):
            rows.append({"site": site, "arm": arm, "p": point, "lo": low, "hi": high})
    point, low, high = scores["short"]["contrast"]["stack_minus_bg2"]["all"]["site"][0]
    rows.append({"site": "D18", "arm": "Stack (signed MW)", "p": point, "lo": low, "hi": high})
    domain = ["Stack (signed MW)", "Chosen fit (MVA)", "Naive fit (MVA)"]
    base = _chart(pl.DataFrame(rows).to_pandas()).encode(
        x=alt.X("site:N", sort=ORDER[:-1], title="Primary", axis=alt.Axis(labelAngle=0)),
        xOffset="arm:N",
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=domain, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_SKY]),
            legend=alt.Legend(title=None, orient="top", direction="vertical"),
        ),
    )
    interval = base.mark_rule().encode(
        y=alt.Y("lo:Q", title="Stack minus BG2, quantile score (points; <0 is better)"), y2="hi:Q"
    )
    dots = base.mark_circle(size=70).encode(y="p:Q")
    zero = (
        _chart(pl.DataFrame({"y": [0.0]}).to_pandas()).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    )
    (interval + dots + zero).properties(width=560, height=280).save(
        DOCS_DIR / "figure_1_minus_bg2_by_primary.svg"
    )
    absolute = []
    for site in ORDER:
        if site in SIGNED:
            row = scores["signed"]["absolute"][site]["all"]
            stack = row["stack"]
        elif site == "D18":
            row = scores["short"]["absolute"][site]["all"]
            stack = row["stack"]
        elif site in MVA:
            row = scores["mva"]["absolute"][site]["all"]
            stack = row["chosen2"]
        else:
            row = scores_372["absolute"][site]["all"]
            stack = None
        series = {
            "Stack / chosen fit": stack,
            "BG2": row["bg2"],
            "Manual heuristic": row["manual"],
            "Climatology": row["climatology"],
        }
        absolute.extend(
            {"site": site, "series": k, "score": v} for k, v in series.items() if v is not None
        )
    names = ["Stack / chosen fit", "BG2", "Manual heuristic", "Climatology"]
    (
        _chart(pl.DataFrame(absolute).to_pandas())
        .mark_circle(size=70)
        .encode(
            x=alt.X(
                "site:N",
                sort=ORDER,
                title="Primary (D15: references only)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "score:Q",
                title="Quantile score, all half-hours (points)",
                scale=alt.Scale(type="symlog", constant=2),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=names,
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_AMBER, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical"),
            ),
        )
        .properties(width=560, height=260)
        .save(DOCS_DIR / "figure_2_score_by_primary.svg")
    )
    tails = []
    for group, sites, arm in (("signed", SIGNED, "stack"), ("mva", MVA, "chosen2")):
        for site in sites:
            for name, label in ((arm, "Stack / chosen fit"), ("bg2", "BG2")):
                share = scores[group]["tails"][site]["all"][name]
                tails.append(
                    {"site": site, "series": label, "q05": share["q05"], "q95": share["q95"]}
                )
    frame = pl.DataFrame(tails).to_pandas()
    panels = []
    for column, target, title in (
        ("q05", 0.05, "Share of readings below the q05 forecast (ideal 0.05)"),
        ("q95", 0.95, "Share of readings below the q95 forecast (ideal 0.95)"),
    ):
        dots = (
            _chart(frame)
            .mark_circle(size=60)
            .encode(
                x=alt.X(
                    "site:N", sort=[*SIGNED, *MVA], title="Primary", axis=alt.Axis(labelAngle=0)
                ),
                y=alt.Y(f"{column}:Q", title=title),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["Stack / chosen fit", "BG2"],
                        range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                    ),
                    legend=alt.Legend(title=None, orient="top", direction="vertical"),
                ),
            )
        )
        ideal = (
            _chart(pl.DataFrame({"y": [target]}).to_pandas())
            .mark_rule(strokeDash=[4, 4])
            .encode(y="y:Q")
        )
        panels.append((dots + ideal).properties(width=560, height=160))
    alt.vconcat(*panels).save(DOCS_DIR / "figure_3_tail_coverage.svg")


def main() -> None:
    """Run the gate, print the report, and draw the figures."""
    scores = json.loads((WORK / "scores.json").read_text())
    gate()
    print_scores(scores=scores)
    figures(scores=scores, scores_372=json.loads(SCORES_372.read_text()))


if __name__ == "__main__":
    main()

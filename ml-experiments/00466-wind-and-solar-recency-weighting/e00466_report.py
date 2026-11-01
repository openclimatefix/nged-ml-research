"""Experiment 466: draw the two figures from `work/report.txt` (written by `e00466_score.py`).

Figure 1 is the pooled score change of every arm against its equal-weight reference, in percent of
the reference score in MW, with 95% month-block intervals and the pre-registered margin. Figure 2
is the pooled change by scored month in MW for the arms with the longest half-lives.

    source ml-experiments/00429-*/env.sh   # plus the 00466 folder
    uv run --no-sync python ml-experiments/00466-*/e00466_report.py
"""

import re
from pathlib import Path
from typing import Final

import altair as alt
import e00466_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00466-wind-and-solar-recency-weighting"
)
LINE: Final = re.compile(
    r"^(?P<arm>\S+) pooled: ref (?P<ref>[\d.]+) MW, arm [\d.]+ MW, diff (?P<diff>[+-][\d.]+) MW "
    r"\((?P<pct>[+-][\d.]+)%\), 95% \[(?P<lo>[+-][\d.]+), (?P<hi>[+-][\d.]+)\]"
)


def parse() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the pooled contrasts and the pooled monthly contrasts from the report."""
    rows, months = [], []
    lines = (common.DATA_DIR / "work" / "report.txt").read_text().splitlines()
    for i, line in enumerate(lines):
        m = LINE.match(line)
        if not m:
            continue
        ref = float(m["ref"])
        rows.append(
            {
                "arm": m["arm"],
                "tech": "Wind" if m["arm"].startswith("member") else "Solar",
                "pct": float(m["pct"]),
                "lo": float(m["lo"]) / ref * 100,
                "hi": float(m["hi"]) / ref * 100,
            }
        )
        months.extend(
            {"arm": m["arm"], "month": token[0], "diff_mw": float(token[1])}
            for token in re.findall(r"(\d{4}-\d{2}) ([+-][\d.]+)", lines[i + 1])
        )
    return pl.DataFrame(rows), pl.DataFrame(months)


def main() -> None:
    """Write both figures."""
    contrasts, months = parse()
    base = alt.Chart(contrasts).encode(
        y=alt.Y("arm:N", title=None, sort=None),
        color=alt.Color(
            "tech:N", legend=None, scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE])
        ),
    )
    chart = (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Score change against equal weights (%)"), x2="hi:Q"
        )
        + base.mark_point(filled=True, size=60).encode(x="pct:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="black").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=520, height=260, title="Pooled score change, positive is worse (95% interval)"
    )
    chart.save(DOCS / "figure_1_score_change.svg")
    pick = months.filter(
        pl.col("arm").is_in(
            ["member_plain_hl9m", "g0_onepass_hl2", "g0_onepass_hl3", "a2_member_hl3"]
        )
    )
    alt.Chart(pick).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("month:N", title="Scored month"),
        y=alt.Y("diff_mw:Q", title="Score change against equal weights (MW)"),
        color=alt.Color("arm:N", title=None),
    ).properties(width=560, height=260).save(DOCS / "figure_2_monthly_change.svg")


if __name__ == "__main__":
    main()

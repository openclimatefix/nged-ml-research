r"""Apply experiment 697's decision rule to the stored statistics and print the verdict.

    uv run --no-sync python $E/00697-*/e00697_verdict.py

The rule was written before the run (sections 2 and 9 of the experiment's plan). It reads
`stats.json`, which `e00697_stats.py` wrote, and writes `printed/verdict.txt`. **Q1 gets one of four
words, tested in this order: not supported, supported, partly supported, inconclusive. Inconclusive
is the default.** Every other question gets a reading, and none changes the verdict word.

- Not supported: D8's own weekday-minus-weekend difference has an interval that includes zero in
  both eras.
- Supported: the interval of D (D8's difference minus a null control's) lies above zero against
  every null control in both eras, and every point estimate of D is 0.10 or more.
- Partly supported: the interval of D lies above zero against every null control in one era only,
  or in both eras with a point estimate under 0.10.
- Inconclusive: anything else, including D8's own difference excluding zero while the interval of
  D includes zero against a null control in both eras, so tracker error is not excluded.

Exploratory readings: 1 is "not explained by the controls", 2 is "tracker error not excluded", 3 is
"no pattern found". A case that fits none (D8's interval excludes zero in one era only) is called
"one era only". A pattern could be passed on to a possible gas-engine plant-equations experiment
only if Q1 is supported or the question has reading 1, and Gas A at 2 or more units does not point
the other way. A result chosen after the tables are seen is labelled post hoc.
"""

import json
from typing import Final, TypedDict

from e00697_common import (  # ty: ignore[unresolved-import]
    PRINTED_DIR,
    Q1_MIN_DIFFERENCE,
    STATS_PATH,
    VERDICT_ERAS,
)
from e00697_stats import controls_for  # ty: ignore[unresolved-import]

EXPLORATORY: Final[tuple[str, ...]] = ("q2", "q2q", "q3", "q3a", "q6", "q7")
P1B_LIMIT: Final[float] = 0.10
"""The labels of a patternless injected engine must show a difference below this."""


class Interval(TypedDict):
    """A point estimate with its percentile interval and the number of weeks resampled."""

    point: float
    lo: float
    hi: float
    n_weeks: int


def excludes_zero(*, estimate: Interval | None) -> bool:
    """Return whether an interval has one sign."""
    return estimate is not None and (estimate["lo"] > 0 or estimate["hi"] < 0)


def above_zero(*, estimate: Interval | None) -> bool:
    """Return whether an interval lies wholly above zero."""
    return estimate is not None and estimate["lo"] > 0


def differences_for(*, stats: dict, question: str, era: str) -> dict[str, Interval | None]:
    """Return D against every null control for a question and era; missing means not estimable."""
    row = stats["difference"].get(question, {}).get(era, {})
    return {name: row.get(name) for name in controls_for(question=question)}


def q1_verdict(*, stats: dict) -> tuple[str, list[str]]:
    """Return the verdict word and the lines that justify it."""
    lines: list[str] = []
    own = {era: stats["d8"][era]["ts"].get("q1") for era in VERDICT_ERAS}
    for era, est in own.items():
        lines.append(f"D8's own Q1 in the {era} era: {fmt(est)}")
    if all(not excludes_zero(estimate=e) for e in own.values()):
        return "NOT SUPPORTED", [*lines, "D8's interval includes zero in both eras."]
    above: dict[str, bool] = {}
    big: dict[str, bool] = {}
    for era in VERDICT_ERAS:
        diffs = differences_for(stats=stats, question="q1", era=era)
        for name, d in diffs.items():
            lines.append(f"D against {name}, {era} era: {fmt(d)}")
        complete = all(d is not None for d in diffs.values())
        above[era] = complete and all(above_zero(estimate=d) for d in diffs.values())
        big[era] = complete and all(d["point"] >= Q1_MIN_DIFFERENCE for d in diffs.values() if d)
    if all(above.values()) and all(big.values()):
        return "SUPPORTED", lines
    if sum(above.values()) == 1 or (all(above.values()) and not all(big.values())):
        return "PARTLY SUPPORTED", lines
    return "INCONCLUSIVE", lines


def fmt(estimate: Interval | None) -> str:
    """Format an interval from `stats.json` on one line."""
    if estimate is None:
        return "not estimable"
    return f"{estimate['point']:+.3f} [{estimate['lo']:+.3f}, {estimate['hi']:+.3f}]"


def reading(*, stats: dict, question: str) -> tuple[str, str]:
    """Return the reading (1, 2, 3, or 'one era only') of an exploratory question and a note."""
    eras = ("net",) if question == "q6" else VERDICT_ERAS
    own = {era: stats["d8"][era]["ts"].get(question) for era in eras}
    if not any(excludes_zero(estimate=e) for e in own.values()):
        return "3 (no pattern found)", ""
    if not all(excludes_zero(estimate=e) for e in own.values()):
        return "one era only (not read as a pattern)", ""
    signs = {e["point"] > 0 for e in own.values() if e}
    if len(signs) > 1:
        return "one era only (the two eras differ in sign)", ""
    explained = True
    for era in eras:
        diffs = differences_for(stats=stats, question=question, era=era)
        if any(d is None for d in diffs.values()):
            return "not estimable (a control gave no estimate, so no reading is made)", ""
        if any(not excludes_zero(estimate=d) for d in diffs.values()):
            explained = False
    if explained:
        return "1 (not explained by the controls)", ""
    return "2 (tracker error not excluded)", ""


def r_agrees(*, stats: dict, question: str) -> str:
    """Return whether Gas A at 2 or more units points the same way as D8's net era."""
    record = stats["record"].get("r2", {}).get(question)
    d8 = stats["d8"]["net"]["ts"].get(question)
    if record is None or d8 is None:
        return "not available"
    return "same direction" if (record["point"] > 0) == (d8["point"] > 0) else "OPPOSITE direction"


def p1b(*, stats: dict) -> list[str]:
    """Return the P1b lines: each D8-history control's Q1 in the labels, and the largest.

    P1b is a check that the controls built on D8's history show a Q1 under the limit. It does not
    change the verdict word.
    """
    lines: list[str] = []
    points: list[float] = []
    for name in ("c1_null", "c1s_net_null", "c1s_solar_null"):
        for era in VERDICT_ERAS:
            estimate = stats.get(name, {}).get(era, {}).get("ts", {}).get("q1")
            if estimate is None:
                lines.append(f"  {name}, {era} era: not estimable")
                continue
            points.append(estimate["point"])
            lines.append(f"  {name}, {era} era: Q1 in the labels {fmt(estimate)}")
    if not points:
        return ["P1b: not available"]
    point = max(points)
    side = "below" if point < P1B_LIMIT else "NOT below"
    heading = (
        "P1b (a check that does not change the verdict word), Q1 in the labels of the three "
        "controls built on D8's history:"
    )
    return [
        heading,
        *lines,
        f"P1b: the largest Q1 is {point:+.3f}, which is {side} {P1B_LIMIT}",
    ]


def main() -> None:
    """Print and write the verdict, the P1b check, the readings, and the pass-on rule."""
    stats = json.loads(STATS_PATH.read_text())["questions"]
    word, justification = q1_verdict(stats=stats)
    lines = [f"Q1 verdict: {word}", *justification, "", *p1b(stats=stats), ""]
    lines.append("Exploratory readings (post hoc status: none of these changes the verdict word):")
    passed_on = word == "SUPPORTED"
    for question in EXPLORATORY:
        read, _ = reading(stats=stats, question=question)
        agree = r_agrees(stats=stats, question=question)
        lines.append(f"  {question}: reading {read}; Gas A at 2 or more units: {agree}")
        if read.startswith("1") and agree != "OPPOSITE direction":
            lines.append(
                f"    {question} could be passed on to a possible plant-equations experiment."
            )
    lines.append("")
    q1_r = r_agrees(stats=stats, question="q1")
    lines.append(
        "Q1 could be passed on to a possible plant-equations experiment: "
        f"{passed_on and q1_r != 'OPPOSITE direction'}"
    )
    lines.append(f"Gas A at 2 or more units against D8's net era for Q1: {q1_r}")
    cross = stats.get("cross_era", {})
    lines.append(
        "Q5 (night share, 6 months after the solar start against a year earlier), "
        f"D8 labels: {cross.get('d8_ts')}"
    )
    PRINTED_DIR.mkdir(parents=True, exist_ok=True)
    (PRINTED_DIR / "verdict.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

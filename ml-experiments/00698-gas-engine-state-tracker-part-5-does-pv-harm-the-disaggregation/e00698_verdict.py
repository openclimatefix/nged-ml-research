r"""Experiment 698's verdict, by the rule in the plan, in the order the plan states.

    uv run --no-sync python $E/00698-*/e00698_verdict.py

`decide` is a pure function of the gate, the scores, and the farm's coverage, so the selftest
checks each branch on synthetic inputs. The verdict word answers the weather question only: the
level contrast is reported in its own sentence and never changes the word.

Tested in this order:

1. `inconclusive` (the audit blocks the test): D8's reading before the engine fails the curve gate
   with the farm shape at both capacities, or its solar-free twin's midday F1 differs from 0.767 by
   more than 0.02, or the paired half-width of the F1 contrast at D8 exceeds 0.05 (in any contrast
   that exists), or the farm series covers under 90% of D8-pre's span or of D3's span (the design said 95% of D8's
   span; the clause was changed before the first real run, see `COVERAGE_MIN`).
2. `supported`: at one capacity, the farm weather contrast meets a harm criterion at D8 and at D3.
   If D3 failed the gate (no capacity passes with the farm shape), it must meet a criterion at D8
   at both capacities.
3. `partly supported`: the farm weather contrast meets a criterion at D8 at one capacity only, or
   only the CAMS weather contrast meets one at D8.
4. `refuted`: at D8, at both capacities and for both shapes, every weather contrast has an interval
   wholly below its threshold.
5. `not supported`: none of the above.

**Step 4 can lower a verdict and cannot raise one.** If the verdict is `supported` and step 4's
recall contrast (the solar era's bright-minus-dull recall minus the net era's) has a half-width of
0.10 or less and an interval wholly above zero, the verdict becomes `partly supported`.

A harm criterion is met when the point estimate reaches the threshold and the interval lies wholly
above zero (`meets_criterion`). The reading "inside its threshold" means an interval whose upper end
is below the threshold (`inside_threshold`).
"""

from typing import Final

from e00698_common import (  # ty: ignore[unresolved-import]
    AUDIT_PATH,
    CAPACITIES_MW,
    COVERAGE_MIN,
    CRITERIA,
    GATE_PATH,
    HALF_WIDTH_MAX,
    PRINTED_DIR,
    SCORES_PATH,
    STEP4_HALF_WIDTH_MAX,
    STRONG_BMD,
    THRESHOLDS,
    TWIN_F1_EXPECTED,
    TWIN_F1_TOLERANCE,
    VERDICT_PATH,
    Interval,
    capacity_key,
    half_width,
    inside_threshold,
    meets_criterion,
    read_json,
    write_json,
)

VERDICT_WORDS: Final[tuple[str, ...]] = (
    "inconclusive",
    "supported",
    "partly supported",
    "refuted",
    "not supported",
)


def contrast_of(*, scores: dict, host: str, key: str, name: str) -> dict | None:
    """Return the smoothed-label criteria of one contrast, or None if the run is missing."""
    host_doc = scores.get("hosts", {}).get(host)
    if host_doc is None:
        return None
    labels = host_doc["contrasts"].get(key, {}).get(name)
    return None if labels is None else labels["smooth"]


def meets_any(*, criteria: dict | None) -> bool:
    """Return whether any of the three criteria is met in a contrast."""
    if criteria is None:
        return False
    return any(meets_criterion(interval=criteria[c], criterion=c) for c in CRITERIA)


def all_inside(*, criteria: dict | None) -> bool:
    """Return whether every criterion's interval lies wholly below its threshold."""
    if criteria is None:
        return False
    return all(inside_threshold(interval=criteria[c], criterion=c) for c in CRITERIA)


def block_reasons(*, gate: dict, scores: dict, coverage: float) -> list[str]:
    """Return the reasons the audit blocks the test (branch 1), empty if it does not."""
    reasons: list[str] = []
    keys = [capacity_key(capacity_mw=c) for c in CAPACITIES_MW]
    if all(not gate["gate"]["d8pre"][k]["farm"] for k in keys):
        reasons.append("D8-pre fails the curve gate with the farm shape at both capacities")
    twin = scores.get("twin_f1_d8pre")
    if twin is None or abs(twin - TWIN_F1_EXPECTED) > TWIN_F1_TOLERANCE:
        reasons.append(
            f"D8-pre's solar-free twin midday F1 is {twin}, not {TWIN_F1_EXPECTED} within "
            f"{TWIN_F1_TOLERANCE}"
        )
    widths = [
        half_width(interval=labels["smooth"]["f1"])
        for table in scores.get("hosts", {}).get("d8pre", {"contrasts": {}})["contrasts"].values()
        for labels in table.values()
    ]
    if not widths:
        reasons.append("D8-pre has no paired contrast")
    elif max(widths) > HALF_WIDTH_MAX:
        reasons.append(
            f"the paired half-width of the F1 contrast at D8-pre reaches {max(widths):.3f}, "
            f"above {HALF_WIDTH_MAX}"
        )
    if coverage < COVERAGE_MIN:
        reasons.append(
            f"the farm series covers {coverage:.3f} of the smaller host span, under {COVERAGE_MIN}"
        )
    return reasons


def refuted_at_d8(*, gate: dict, scores: dict, strict: bool) -> bool:
    """Return whether every weather contrast at D8-pre lies wholly inside its threshold.

    The plan says "at both capacities and for both shapes". A capacity or a shape that failed the
    curve gate has no run, so that reading can never hold once the gate removes a pair. With
    `strict` False the test covers every pair that passed the gate at D8-pre and needs at least one
    farm contrast and one CAMS contrast. With `strict` True all four pairs must exist.
    """
    keys = [capacity_key(capacity_mw=c) for c in CAPACITIES_MW]
    seen = {"weather_farm": 0, "weather_cams": 0}
    for k in keys:
        for name, shape in (("weather_farm", "farm"), ("weather_cams", "cams")):
            criteria = contrast_of(scores=scores, host="d8pre", key=k, name=name)
            gated = gate["gate"]["d8pre"][k][shape]
            if criteria is None:
                if strict or gated:
                    return False
                continue
            if not all_inside(criteria=criteria):
                return False
            seen[name] += 1
    return all(n > 0 for n in seen.values())


def decide(
    *,
    gate: dict,
    scores: dict,
    coverage: float,
    strict_refuted: bool = False,
) -> tuple[str, list[str]]:
    """Return the verdict word and the lines that explain it.

    Args:
        gate: The parsed `gate.json`.
        scores: The parsed `scores.json`.
        coverage: The farm's smallest finite share over the D8-pre and D3 spans.
        strict_refuted: Read branch 4 as needing all four pairs (both capacities, both shapes).

    Returns:
        The word, one of `VERDICT_WORDS`, and the reasons in the order the branches were tested.
    """
    keys = [capacity_key(capacity_mw=c) for c in CAPACITIES_MW]
    notes: list[str] = []
    blocks = block_reasons(gate=gate, scores=scores, coverage=coverage)
    if blocks:
        return "inconclusive", [f"branch 1: {reason}" for reason in blocks]
    farm_d8 = {
        k: contrast_of(scores=scores, host="d8pre", key=k, name="weather_farm") for k in keys
    }
    farm_d3 = {
        k: contrast_of(scores=scores, host="d3solar", key=k, name="weather_farm") for k in keys
    }
    cams_d8 = {
        k: contrast_of(scores=scores, host="d8pre", key=k, name="weather_cams") for k in keys
    }
    d3_failed = all(not gate["gate"].get("d3solar", {}).get(k, {}).get("farm", False) for k in keys)
    d8_meets = {k: meets_any(criteria=farm_d8[k]) for k in keys}
    d3_meets = {k: meets_any(criteria=farm_d3[k]) for k in keys}
    notes.append(f"farm weather contrast meets a criterion at D8-pre: {d8_meets}")
    notes.append(
        f"farm weather contrast meets a criterion at D3 (D3 failed the gate: {d3_failed}): "
        f"{d3_meets}"
    )
    if d3_failed:
        supported = all(d8_meets.values())
    else:
        supported = any(d8_meets[k] and d3_meets[k] for k in keys)
    word: str
    if supported:
        word = "supported"
        notes.append("branch 2: supported")
        step4 = scores.get("step4", {}).get("contrast_solar_minus_net")
        if (
            step4 is not None
            and half_width(interval=step4) <= STEP4_HALF_WIDTH_MAX
            and step4["lo"] > 0
        ):
            word = "partly supported"
            notes.append(
                "step 4 lowers the verdict: recall on bright half-hours is better in the solar "
                f"era (interval {step4['lo']:+.3f} to {step4['hi']:+.3f})"
            )
        return word, notes
    cams_meets = any(meets_any(criteria=cams_d8[k]) for k in keys)
    if any(d8_meets.values()) or cams_meets:
        notes.append(
            "branch 3: partly supported (farm weather contrast at one capacity only, or CAMS only)"
        )
        return "partly supported", notes
    if refuted_at_d8(gate=gate, scores=scores, strict=strict_refuted):
        notes.append(
            "branch 4: refuted (every weather contrast wholly inside its threshold at D8-pre)"
        )
        return "refuted", notes
    notes.append(
        "branch 5: not supported (no weather contrast meets a criterion, and branch 4 fails)"
    )
    return "not supported", notes


def level_sentence(*, scores: dict) -> str:
    """Return one sentence on the level contrast at D8-pre, which never changes the verdict word."""
    parts = []
    for cap in CAPACITIES_MW:
        key = capacity_key(capacity_mw=cap)
        criteria = contrast_of(scores=scores, host="d8pre", key=key, name="level")
        if criteria is None:
            parts.append(f"{cap} MW: not run")
            continue
        met = [c for c in CRITERIA if meets_criterion(interval=criteria[c], criterion=c)]
        parts.append(f"{cap} MW: " + (", ".join(met) if met else "no criterion met"))
    return (
        "Level contrast (smooth twin minus solar-free twin) at D8-pre, criteria met: "
        + "; ".join(parts)
    )


def strong_statement(*, scores: dict) -> str:
    """Return whether solar weather explains D8's contrast: criterion 3 at 0.125 at D8-pre."""
    hits = []
    for cap in CAPACITIES_MW:
        key = capacity_key(capacity_mw=cap)
        criteria = contrast_of(scores=scores, host="d8pre", key=key, name="weather_farm")
        if criteria is None:
            continue
        bmd: Interval = criteria["bmd"]
        hits.append(f"{cap} MW: {bmd['point'] >= STRONG_BMD and bmd['lo'] > 0}")
    return (
        f"Weather explains D8's swing (farm bright-minus-dull harm at least {STRONG_BMD} with the "
        "interval above zero): " + "; ".join(hits)
    )


def main() -> None:
    """Read the stored documents, print the verdict, and write `verdict.json`."""
    gate = read_json(path=GATE_PATH)
    scores = read_json(path=SCORES_PATH)
    audit = read_json(path=AUDIT_PATH)
    coverage = float(audit["numbers"]["farm_coverage_min"])
    word, notes = decide(gate=gate, scores=scores, coverage=coverage)
    lines = [f"Verdict (the plan's rule): {word.upper()}", *[f"  {n}" for n in notes]]
    strict_word, _ = decide(gate=gate, scores=scores, coverage=coverage, strict_refuted=True)
    if strict_word != word:
        lines.append(
            "Under the strict reading of branch 4 (all four capacity and shape pairs must exist): "
            f"{strict_word.upper()}"
        )
    lines.append(level_sentence(scores=scores))
    lines.append(strong_statement(scores=scores))
    lines.append(f"Thresholds: {THRESHOLDS}")
    print("\n".join(lines))
    PRINTED_DIR.mkdir(parents=True, exist_ok=True)
    (PRINTED_DIR / "verdict.txt").write_text("\n".join(lines) + "\n")
    write_json(
        path=VERDICT_PATH,
        payload={"verdict": word, "notes": notes},
    )


if __name__ == "__main__":
    main()

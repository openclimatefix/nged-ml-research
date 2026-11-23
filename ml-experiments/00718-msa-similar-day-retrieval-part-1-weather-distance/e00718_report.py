# ruff: noqa: E501
"""Experiment 718, WP8: print the tables of a part's `result.md` from the stored score files.

Usage:
    e00718_report.py PART        # contrast tables, verdict, guards, thresholds, closure, and departures, as markdown
    e00718_report.py self-test   # renders the tables from the synthetic contrasts of `e00718_contrasts.py self-test`

Every number comes from `score/screen.json`, `score/headline.json`, `score/clean_folds.json`, and
`score/report_only.json` of the part (written by `e00718_contrasts.py`). The script prints and never
writes into `docs/`: the author pastes the output and adds the prose.
"""

import json
import sys
from typing import Final

import e00718_decision as dec

DEPARTURES: Final = (
    "The pool's weather is read from experiment 713's float16 standardised array and converted to float64, where the design asked for float64 from the source rows. The table builder and its independent recomputation read the same array.",
    "Each placebo draw is seeded by (placebo seed 0, Primary, origin day, lead day, an arm tag), where the design names the first four. The arm tag keeps the draws of different placebo arms independent.",
    'The negative control "a pool that admits the block of age 0" is a block of age 1 under a power cap of 0, because the block of age 0 lies beyond the grid index that a cap of 0 exposes.',
    "The newest-row dropout is applied after experiment 713's `Source.assemble`, to the rows of age 2, so that experiment 713's assembly stays unedited.",
    "The 27-row arms have more rows than the 24 age buckets, so a placebo draw for them can put two rows in one bucket.",
    "The age-bucket tables are read with experiment 713's slices when the rows are in the standard layout of ages 2 to 15, so that an F9 fit of arm `Z` is bit-identical to experiment 713's loop.",
)
"""Departures from the design that both pages state."""


def fmt(c: dict) -> str:
    """Return a point estimate with its interval."""
    return f"{c['point']:+.3f} [{c['lower']:+.3f}, {c['upper']:+.3f}]"


def contrast_table(*, title: str, results: dict[str, dict], names: list[str]) -> list[str]:
    """Return a markdown table of the counted contrasts: the point estimate and interval by group, the guards, and the reading."""
    lines = [
        f"**{title}**",
        "",
        "| Contrast | Group | Point and interval (points of CRPS) | t guard | Seed guard | Floor guard | Reading |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in names:
        for g in dec.GROUPS:
            c = results[name][g]
            lines.append(
                f"| {name} | {g} | {fmt(c)} | {c['t_guard']} | {c['seed_guard']} | {c['floor_guard']} | {c['reading']} |"
            )
    return lines


def headline_block(*, h: dict, level_note: str = "98.75%") -> list[str]:
    """Return the markdown of one part's headline: the counted contrasts, the verdict, the guards, and the sensitivity reading."""
    a, b = [k for k in h if k[0] in {"A", "B"} and k[1:] in {"1", "2"}]
    lines = contrast_table(
        title=f"Counted contrasts at {level_note} ({h['n_seeds']} seeds per arm)",
        results=h,
        names=[a, b],
    )
    lines += [
        "",
        f"**Verdict: {h['verdict']}.** Readings of the first contrast: "
        + ", ".join(f"{g} {r}" for g, r in h["readings"].items())
        + ".",
    ]
    if h["unread_guards"]:
        lines.append(
            "Guards not read yet, which fail by rule: " + ", ".join(h["unread_guards"]) + "."
        )
    guards = h["guards"]
    lines += ["", "| Guard | " + " | ".join(dec.GROUPS) + " |", "|---|---|---|"]
    for key in guards["six"]:
        if key != "all_hold":
            lines.append(f"| {key} | " + " | ".join(str(guards[g][key]) for g in dec.GROUPS) + " |")
    lines += [
        "",
        "**Sensitivity reading at 99.375% (eight readings across both parts; changes no verdict):**",
    ]
    for name, c in h["sensitivity_99.375"].items():
        lines.append(
            f"- {name}: "
            + "; ".join(f"{g} [{c[g]['lower']:+.3f}, {c[g]['upper']:+.3f}]" for g in dec.GROUPS)
        )
    return lines


def screen_block(*, s: dict) -> list[str]:
    """Return the markdown of the screen: stop rule and audit trigger 1."""
    lines = [
        "**Screen (interim, folds 2025-11, 2026-01, 2026-03, 2026-06; seeds 0 and 1; 95%).**",
        f"Stop rule fires: {s['stop_rule']}. Audit trigger 1 from the placebo contrast: {s['audit_trigger_1_placebo']}.",
    ]
    if "audit_trigger_1_z_rule" in s:
        lines.append(f"Audit trigger 1 from the rule on `Z`: {s['audit_trigger_1_z_rule']}.")
    return lines


def thresholds_block(*, n_seeds: int) -> list[str]:
    """Return the reachable thresholds for the seed count."""
    return [
        "**Reachable thresholds (printed before any fit).**",
        *[f"- {x}" for x in dec.reachable_thresholds(n_seeds=n_seeds)],
    ]


def render(*, h: dict, s: dict, ro: dict, clean: dict) -> str:
    """Return the whole report as markdown."""
    out = [
        *screen_block(s=s),
        "",
        *headline_block(h=h),
        "",
        f"**Reading clean of the exploration's selection (report-only, 95%):** {json.dumps({k: v for k, v in clean.items() if k != 'label'}, default=float)[:400]}",
        "",
    ]
    out += [*thresholds_block(n_seeds=h["n_seeds"]), ""]
    out += ["**Report-only contrasts at 95%** (CRPS point estimates, all 16 Primaries pooled):"]
    for k, v in ro["contrasts"].items():
        out.append(
            f"- {k}: "
            + (
                v
                if isinstance(v, str)
                else f"{v['all']['crps'][1]:+.3f} [{v['all']['crps'][0]:+.3f}, {v['all']['crps'][2]:+.3f}]"
            )
        )
    out += ["", "**Departures from the design:**", *[f"- {d}" for d in DEPARTURES]]
    return "\n".join(out)


def self_test() -> int:
    """Render the report from the synthetic contrasts."""
    import e00718_contrasts as ct

    ct.self_test()
    text = render(
        h=ct.LAST["headline"],
        s=ct.LAST["screen"],
        ro=ct.LAST["report_only"],
        clean=ct.LAST["clean"],
    )
    for needle in (
        "Verdict:",
        "| A1 | six |",
        "| A2 | eight |",
        "Sensitivity reading at 99.375%",
        "Reachable thresholds",
        "Departures from the design",
        "R-P90",
    ):
        assert needle in text, needle
    print(text[:600])
    print("e00718_report: all checks passed")
    return 0


def main() -> int:
    """Print the report of one part."""
    if len(sys.argv) > 1 and sys.argv[1] == "self-test":
        return self_test()
    part = int(sys.argv[1])
    import e00718_contrasts as ct

    def read(name: str) -> dict:
        return json.loads((ct._score_dir(part=part) / f"{name}.json").read_text())

    print(
        render(
            h=read("headline"), s=read("screen"), ro=read("report_only"), clean=read("clean_folds")
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

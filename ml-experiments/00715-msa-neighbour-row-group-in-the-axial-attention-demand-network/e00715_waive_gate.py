# ruff: noqa: E501
"""Experiment 715: write the known-answer gate file as a waived gate.

Usage: e00715_waive_gate.py [--selftest]

The gate of experiment 715 ran on 2026-11-24 at head ac3e768e4 (code hash e9809f648a885ae1). It
failed KN6 and KN7 and passed KN11(a) and KN12. The maintainer decided, through the manager, to waive
KN6 and KN7 with disclosure and to run F4's real screen. This script turns
`work/known_answers_last_run.json` into `work/known_answers_passed.json` with `waived: true`, the
failed and the passed checks, the failure values, the decision, the date, the code record, and the
sha256 of `e00715_known_answer.py`. **It records a waived known-answer check and never a passed
one, and no threshold and no verdict rule changes.**

The script refuses to write unless the last-run file is the very file whose sha256 is pinned below,
exactly KN6 and KN7 failed with the recorded values, KN11a and KN12 passed, the code hash is the
hash at which the gate ran, the known-answer script is the pinned file, and the gate file is not a
real pass. The fit scripts need no change: `e00715_fit.py` checks the `code` record of the gate file
with `arms.check_code` and prints the gate's known-answer sha256 beside the current file's.
"""

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Final

import e00715_arms as arms
from studies.sources import ML_EXPERIMENTS_DATA_DIR

GATE_HEAD: Final = "ac3e768e4"
GATE_CODE_HASH: Final = "e9809f648a885ae1"
EXPECTED_FAILED: Final = frozenset({"KN6", "KN7"})
EXPECTED_PASSED: Final = frozenset({"KN11a", "KN12"})
EXPECTED_LAST_RUN_SHA256: Final = "4bf839fcfede051ef5830ac081c5c5c364aab8dbd5497d0118816e9b8215f2c6"
EXPECTED_KNOWN_ANSWER_SHA256: Final = (
    "f6870d58d59013c7c36ca9dafcafa68e0c40db14d3f52bf5d5a012467fba2e85"
)
KN6_F4_GAP: Final = 0.02876130850798119
KN6_THRESHOLD: Final = 0.60
WAIVER_DATE: Final = "2026-11-24"
DECISION: Final = "WAIVE KN6 and KN7 with disclosure and run F4's real screen"
DECISION_NOTE: Final = (
    "The maintainer decided this through the manager on 2026-11-24, after an audit of the failed gate "
    "(f4_gate_failure_audit.md) found no fault in F4's neighbour path and three other causes."
)
AUDIT_FINDINGS: Final = [
    "The harness trains the synthetic fits for at most 150 passes with patience 8, experiment 713's budget before its amendment 3 (400 passes, patience 30). The W3 fits stopped at 42 to 54 passes while their validation loss was still falling, and the W0 pairs F4 fit hit the 150-pass cap.",
    "KN6's gap rests on one permanent step and two temporary steps in the held-out window, so it cannot separate a working F4 from a broken one.",
    "Arm M follows only 0.019 of a permanent step on W3 at that budget, as in experiment 713's waived check KA3, so F4 built on the same trunk is unlikely to reach a gap of 0.60.",
]
MAINTAINER_NOTE: Final = (
    "In general, it is hard to discriminate between a temporary step and a permanent step; the only "
    "two hints are what a primary substation's electrical neighbours do, and the rough shape of the "
    "step (although the shape is probably less informative)."
)


def problems(*, results: dict | None, code_hash: str) -> list[str]:
    """Return the reasons the waiver must be refused, an empty list when it may be written."""
    if results is None:
        return ["work/known_answers_last_run.json is missing or unreadable"]
    out = []
    if code_hash != GATE_CODE_HASH:
        out.append(f"the code hash is {code_hash}, not {GATE_CODE_HASH} of the gate run")
    failed = {
        k for k in ("KN6", "KN7", "KN11a", "KN12")
        if not (isinstance(results.get(k), dict) and results[k].get("pass") is True)
    }  # fmt: skip
    if failed != EXPECTED_FAILED:
        out.append(
            f"the checks that did not pass are {sorted(failed)}, not {sorted(EXPECTED_FAILED)}"
        )
    for name in sorted(EXPECTED_FAILED):
        entry = results.get(name)
        if not (isinstance(entry, dict) and entry.get("pass") is False):
            out.append(f"{name} has no recorded `pass: false`")
    gap = (
        (results.get("KN6") or {}).get("gaps", {}).get("F4")
        if isinstance(results.get("KN6"), dict)
        else None
    )
    if not (isinstance(gap, float) and math.isclose(gap, KN6_F4_GAP, rel_tol=1e-9)):
        out.append(f"the KN6 F4 gap is {gap!r}, not {KN6_F4_GAP}")
    kn7 = results.get("KN7")
    if not (isinstance(kn7, dict) and "150-pass cap" in str(kn7.get("not evaluated", ""))):
        out.append("KN7 is not recorded as a fit that ended at the 150-pass cap")
    return out


def build_record(
    *, results: dict, code: dict, known_answer_sha256: str, last_run_sha256: str
) -> dict:
    """Return the waived gate record: the last run's results plus the waiver and the code record."""
    return {
        **results,
        "waived": True,
        "waiver": {
            "date": WAIVER_DATE,
            "decision_quoted": DECISION,
            "decision_note": DECISION_NOTE,
            "decided_by": "the maintainer",
            "failed_checks": sorted(EXPECTED_FAILED),
            "passed_checks": sorted(EXPECTED_PASSED),
            "gate_head": GATE_HEAD,
            "thresholds_changed": False,
            "verdict_rules_changed": False,
            "audit_findings": AUDIT_FINDINGS,
            "maintainer_note": MAINTAINER_NOTE,
            "statement": (
                "Waived known-answer check, not a passed one. KN6 failed (the F4 gap on world W3 is "
                f"{results['KN6']['gaps']['F4']:.4f} against at least {KN6_THRESHOLD}), and KN7 was not "
                "evaluated because the W0 pairs F4 fit ended at the 150-pass cap. KN11(a) and KN12 passed. "
                "The maintainer waived KN6 and KN7 and F4's real screen runs. Every statement about "
                "whether F4's neighbour rows follow or ignore a level step is unsupported by a passing "
                "known-answer test."
            ),
        },
        "code": code,
        "gate_run_code": results.get("code"),
        "known_answer_sha256": known_answer_sha256,
        "last_run_sha256": last_run_sha256,
    }


def waive(
    *,
    work: Path,
    code: dict,
    known_answer_sha256: str,
    expected_last_run_sha256: str = EXPECTED_LAST_RUN_SHA256,
    expected_known_answer_sha256: str = EXPECTED_KNOWN_ANSWER_SHA256,
) -> Path:
    """Write the waived gate file in `work`, or raise AssertionError with every reason for refusal."""
    last = work / "known_answers_last_run.json"
    results = json.loads(last.read_text()) if last.exists() else None
    reasons = problems(results=results, code_hash=code["code_hash"])
    last_sha = hashlib.sha256(last.read_bytes()).hexdigest() if last.exists() else ""
    if last_sha != expected_last_run_sha256:
        reasons.append(f"the last-run file has sha256 {last_sha}, not {expected_last_run_sha256}")
    if known_answer_sha256 != expected_known_answer_sha256:
        reasons.append(
            f"e00715_known_answer.py has sha256 {known_answer_sha256}, not {expected_known_answer_sha256}"
        )
    if results is not None and results.get("known_answer_sha256") != known_answer_sha256:
        reasons.append("the last-run file was written by a different e00715_known_answer.py")
    gate = work / "known_answers_passed.json"
    if gate.exists() and not json.loads(gate.read_text()).get("waived"):
        reasons.append(f"{gate} exists and is not a waived gate, so it is not overwritten")
    assert not reasons, "waiver refused: " + "; ".join(reasons)
    assert results is not None
    tmp = Path(str(gate) + ".tmp")
    tmp.write_text(
        json.dumps(
            build_record(
                results=results, code=code, known_answer_sha256=known_answer_sha256,
                last_run_sha256=last_sha,
            ),
            indent=1,
            default=str,
        )
    )  # fmt: skip
    tmp.rename(gate)
    return gate


def selftest() -> int:
    """Check, with tiny files and no data, that the waiver refuses what it must and writes what it may."""
    good = {
        "KN12": {"pass": True}, "KN11a": {"pass": True},
        "KN6": {"gaps": {"F4": KN6_F4_GAP}, "pass": False},
        "KN7": {"not evaluated": "the fits ['W0 pairs F4'] ended at the 150-pass cap", "pass": False},
        "known_answer_sha256": "s", "code": {"code_hash": GATE_CODE_HASH, "git_commit": "gate", "git_dirty": False},
    }  # fmt: skip
    code = {"code_hash": GATE_CODE_HASH, "git_commit": "x", "git_dirty": False}
    assert problems(results=good, code_hash=GATE_CODE_HASH) == []
    assert problems(results=None, code_hash=GATE_CODE_HASH)
    assert problems(results=good, code_hash="0" * 16)
    refused = [
        {**good, "KN12": {"pass": False}},
        {**good, "KN11a": {"pass": False}},
        {**good, "KN6": {**good["KN6"], "pass": True}},
        {**good, "KN7": {"pass": False}},
        {**good, "KN6": {"gaps": {"F4": 0.5}, "pass": False}},
        {**good, "KN6": {**good["KN6"], "pass": None}},
    ]
    for bad in refused:
        assert problems(results=bad, code_hash=GATE_CODE_HASH), bad

    def refuses(**kwargs: object) -> bool:
        try:
            waive(**kwargs)  # ty: ignore[invalid-argument-type]
        except AssertionError:
            return True
        return False

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        last = work / "known_answers_last_run.json"
        pins = {"expected_known_answer_sha256": "s"}
        assert refuses(work=work, code=code, known_answer_sha256="s", **pins)  # no last-run file
        last.write_text(json.dumps(good))
        last_sha = hashlib.sha256(last.read_bytes()).hexdigest()
        pins["expected_last_run_sha256"] = last_sha
        assert refuses(
            work=work, code={**code, "code_hash": "f" * 16}, known_answer_sha256="s", **pins
        )
        assert refuses(work=work, code=code, known_answer_sha256="other", **pins)
        assert refuses(work=work, code=code, known_answer_sha256="s")  # last-run sha not pinned
        assert not (work / "known_answers_passed.json").exists()
        last.write_text(json.dumps({**good, "KN12": {"pass": False}}))  # a different run
        assert refuses(work=work, code=code, known_answer_sha256="s", **pins)
        last.write_text(json.dumps(good))
        path = waive(work=work, code=code, known_answer_sha256="s", **pins)
        record = json.loads(path.read_text())
        assert record["waived"] is True
        assert record["code"] == code
        assert record["gate_run_code"] == good["code"]
        assert record["last_run_sha256"] == last_sha
        assert record["waiver"]["failed_checks"] == ["KN6", "KN7"]
        assert record["waiver"]["thresholds_changed"] is False
        assert record["waiver"]["verdict_rules_changed"] is False
        assert "Waived known-answer check, not a passed one" in record["waiver"]["statement"]
        assert record["waiver"]["maintainer_note"] == MAINTAINER_NOTE
        path.write_text(json.dumps({"code": code}))  # a real pass is never overwritten
        assert refuses(work=work, code=code, known_answer_sha256="s", **pins)
    print("waiver self-test passed (no data, no fit)")
    return 0


def main() -> int:
    """Write the waived gate file for this experiment's data folder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    work = arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR) / "work"
    digest = hashlib.sha256(
        (Path(__file__).parent / "e00715_known_answer.py").read_bytes()
    ).hexdigest()
    path = waive(work=work, code=arms.code_record(), known_answer_sha256=digest)
    print("waived gate written:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ruff: noqa: E501, PT018
"""Experiment 713: write the known-answer gate file as a waived gate (design amendment 4).

Usage: e00713_waive_gate.py [--selftest]

The amended known-answer gate (commit 94dfb9652, code hash e85b84e2ea04b3cb) failed KA3 and KA12 and
passed every other check. On 2026-11-23 the maintainer decided "Go ahead with option 1" (waive KA3 for
the real folds, failure disclosed). This script turns `work/known_answers_last_run.json` into
`work/known_answers_passed.json` with `waived: true`, the failed and the passed checks, the decision,
the date, the code record, and the sha256 of `e00713_known_answer.py`. No threshold changes.

The script refuses to write unless the last-run file exists, shows exactly KA3 and KA12 failing with
KA2, KA4, and KA10 passing and KA5 and KA6 recorded, the code hash equals the hash at which the amended
gate ran, and the gate file is not a real pass. The fit scripts need no change: `e00713_fit.py` still
checks the `code` record of the gate file with `arms.check_code`, so a waived gate written under one
code hash is refused under another unless `--accept-gate-change OLD_HASH` names it.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Final

import e00713_arms as arms
from studies.sources import ML_EXPERIMENTS_DATA_DIR

AMENDED_GATE_COMMIT: Final = "94dfb9652"
AMENDED_GATE_CODE_HASH: Final = "e85b84e2ea04b3cb"
EXPECTED_FAILED: Final = frozenset({"KA3", "KA12"})
EXPECTED_PASSED: Final = frozenset({"KA2", "KA4", "KA10"})
EXPECTED_RECORDED: Final = ("KA5", "KA6")
DECISION: Final = "Go ahead with option 1"
DECISION_OPTION: Final = (
    "option 1 of gate_options.md: waive KA3 for the real folds, and disclose the failure"
)
WAIVER_DATE: Final = "2026-11-23"


def problems(*, results: dict | None, code_hash: str) -> list[str]:
    """Return the reasons the waiver must be refused, an empty list when it may be written."""
    if results is None:
        return ["work/known_answers_last_run.json is missing or unreadable"]
    out = []
    if code_hash != AMENDED_GATE_CODE_HASH:
        out.append(
            f"the code hash is {code_hash}, not {AMENDED_GATE_CODE_HASH} of the amended gate "
            f"(commit {AMENDED_GATE_COMMIT})"
        )
    failed = {
        k for k, v in results.items() if k.startswith("KA") and isinstance(v, dict) and not v.get("pass", True)
    }  # fmt: skip
    if failed != EXPECTED_FAILED:
        out.append(f"the failed checks are {sorted(failed)}, not {sorted(EXPECTED_FAILED)}")
    for name in sorted(EXPECTED_PASSED):
        entry = results.get(name)
        if not (isinstance(entry, dict) and entry.get("pass") is True):
            out.append(f"{name} is not recorded as passed")
    out += [f"{name} is not recorded" for name in EXPECTED_RECORDED if name not in results]
    for name in sorted(EXPECTED_FAILED):
        if not isinstance(results.get(name), dict) or "pass" not in results[name]:
            out.append(f"{name} has no recorded value")
    return out


def build_record(*, results: dict, code: dict, known_answer_sha256: str) -> dict:
    """Return the waived gate record: the last run's results plus the waiver and the code record."""
    return {
        **results,
        "waived": True,
        "waiver": {
            "date": WAIVER_DATE,
            "decision_quoted": DECISION,
            "decision_option": DECISION_OPTION,
            "decided_by": "the maintainer",
            "failed_checks": sorted(EXPECTED_FAILED),
            "passed_checks": sorted(EXPECTED_PASSED | set(EXPECTED_RECORDED)),
            "amended_gate_commit": AMENDED_GATE_COMMIT,
            "thresholds_changed": False,
            "statement": (
                "The pre-registered known-answer gate failed KA3 (follow share of arm M "
                f"{results['KA3']['follow']['M']:.4f} against 0.75) and KA12. The maintainer waived it "
                "for the real folds. Every statement about following level shifts, and the era "
                "secondary set R1 to R3, is unsupported by a passing known-answer test."
            ),
        },
        "code": code,
        "known_answer_sha256": known_answer_sha256,
    }


def waive(*, work: Path, code: dict, known_answer_sha256: str) -> Path:
    """Write the waived gate file in `work`, or raise AssertionError with every reason for refusal."""
    last = work / "known_answers_last_run.json"
    results = json.loads(last.read_text()) if last.exists() else None
    reasons = problems(results=results, code_hash=code["code_hash"])
    gate = work / "known_answers_passed.json"
    if gate.exists() and not json.loads(gate.read_text()).get("waived"):
        reasons.append(f"{gate} exists and is not a waived gate, so it is not overwritten")
    assert not reasons, "waiver refused: " + "; ".join(reasons)
    assert results is not None
    tmp = Path(str(gate) + ".tmp")
    tmp.write_text(
        json.dumps(
            build_record(results=results, code=code, known_answer_sha256=known_answer_sha256),
            indent=1,
            default=str,
        )
    )
    tmp.rename(gate)
    return gate


def selftest() -> int:
    """Check, with tiny files and no data, that the waiver refuses what it must and writes what it may."""
    good = {
        "KA5": "ok", "KA6": "ok", "KA2": {"pass": True}, "KA4": {"pass": True},
        "KA10": {"pass": True}, "KA3": {"follow": {"M": 0.1418}, "pass": False},
        "KA12": {"pass": False},
    }  # fmt: skip
    code = {"code_hash": AMENDED_GATE_CODE_HASH, "git_commit": "x", "git_dirty": False}
    assert problems(results=good, code_hash=AMENDED_GATE_CODE_HASH) == []
    assert problems(results=None, code_hash=AMENDED_GATE_CODE_HASH)
    assert problems(results=good, code_hash="0" * 16)
    refused = [
        {**good, "KA2": {"pass": False}},  # a different failure set
        {**good, "KA12": {"pass": True}},  # KA12 did not fail
        {**good, "KA10": {"pass": False}},
        {k: v for k, v in good.items() if k != "KA6"},
    ]
    for bad in refused:
        assert problems(results=bad, code_hash=AMENDED_GATE_CODE_HASH), bad
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for bad_code in (
            {**code, "code_hash": "f" * 16},
        ):  # last-run file missing, then hash differs
            try:
                waive(work=work, code=bad_code, known_answer_sha256="s")
            except AssertionError:
                pass
            else:
                raise AssertionError("waiver written without a last-run file")
        (work / "known_answers_last_run.json").write_text(json.dumps(good))
        try:
            waive(work=work, code={**code, "code_hash": "f" * 16}, known_answer_sha256="s")
        except AssertionError:
            pass
        else:
            raise AssertionError("waiver written under a different code hash")
        assert not (work / "known_answers_passed.json").exists()
        path = waive(work=work, code=code, known_answer_sha256="s")
        record = json.loads(path.read_text())
        assert record["waived"] is True and record["code"] == code
        assert record["waiver"]["decision_quoted"] == DECISION
        assert record["waiver"]["failed_checks"] == ["KA12", "KA3"]
        # a real pass is never overwritten
        path.write_text(json.dumps({"code": code}))
        try:
            waive(work=work, code=code, known_answer_sha256="s")
        except AssertionError:
            pass
        else:
            raise AssertionError("a real gate pass was overwritten")
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
        (Path(__file__).parent / "e00713_known_answer.py").read_bytes()
    ).hexdigest()
    path = waive(work=work, code=arms.code_record(), known_answer_sha256=digest)
    print("waived gate written:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

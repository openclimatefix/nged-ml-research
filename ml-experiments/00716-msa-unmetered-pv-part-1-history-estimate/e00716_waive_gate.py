# ruff: noqa: E501
"""Experiment 716: write the known-answer gate file as a waived known-answer check (maintainer decision on KF9).

Usage: e00716_waive_gate.py [--selftest]

The known-answer gate of experiment 716 (code hash 7d6a4381ffe3d298) failed KF9 and passed every other
gating check. The maintainer decided, through the manager, to waive KF9 with a disclosure. This script
turns `work/known_answers_last_run.json` into `work/known_answers_passed.json` with `waived: true`,
the failed check and its values, the passed checks, the decision, the code record, and the sha256 of
`e00716_known_answer.py` and of the last-run file. No threshold and no verdict rule changes. The file
says "waived known-answer check" and never "passed".

The script refuses to write unless the last-run file has the sha256 below, shows exactly KF9 failing
(with the value and the need below) and KF10, KF11, and KF12 passing, the secondary readings and the
pre-check are recorded and the pre-check passed, the current code hash equals the hash at which the
gate ran, `e00716_known_answer.py` has the pinned sha256, and the gate file is not a real pass. The fit
script `e00716_fit.py` needs no change: it still checks the `code` record of the gate file with
`arms.check_code`, so a waiver written under one code hash is refused under another unless
`--accept-gate-change OLD_HASH` names it.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Final

import e00716_arms as arms
from studies.sources import ML_EXPERIMENTS_DATA_DIR

GATE_CODE_HASH: Final = "7d6a4381ffe3d298"
GATE_COMMITS: Final = ("126577d1d", "e4617f721")
"""The commits at which the gate ran: the `M` fits, the pre-check, the wiring records, and `WH_s0` at the first, the other eight gating jobs at the second. Both have code hash 7d6a4381ffe3d298."""
EXPECTED_FAILED: Final = frozenset({"KF9"})
EXPECTED_PASSED: Final = frozenset({"KF10", "KF11", "KF12"})
EXPECTED_RECORDED: Final = (
    "KF15",
    "secondary_days_1_60",
    "secondary_days_61_236",
    "secondary_pass60",
    "early_stopping_precheck",
    "wiring_records",
)
KF9_VALUE: Final = 0.0007037622944964586
KF9_NEED: Final = 0.007290675990783289
DECISION: Final = "WAIVE KF9 with disclosure; then phase 2 may start"
WAIVER_DATE: Final = "2026-11-24"
EXPECTED_LAST_RUN_SHA256: Final = "c1bf18ee0f2e18089f304f7ace0891b446a14114a185e45f36d9db4792622cba"
EXPECTED_KNOWN_ANSWER_SHA256: Final = (
    "a6f877ada2da550329bcde48b4697fd5997b2c66c07f0845a119b81d88445c5d"
)
STATEMENT: Final = (
    "Waived known-answer check, not a passed one. KF9 (arm H-S minus arm H on the solar block in world "
    "WH, the difference of the per-arm medians over seeds 0 to 2) is {value:.5f} against the need of {need:.5f}. The secondary "
    "readings of KF9 do not reach the need either: days 1 to 60 {d1:.5f}, days 61 to 236 {d2:.5f}, "
    "pass 60 {p60:.5f}. KF10, KF11, and KF12 pass. The maintainer waived KF9. The audit of the failure "
    "found no implementation fault and two method findings: the patience-8 rule scored the history "
    "arm at passes 4, 23, and 5 against global bests at passes 43, 55, and 32, and the pass mark "
    "forces arm H below the oracle ridge (arm H-S scores 0.0394 on the solar block and arm Z 0.0397, "
    "the naive ridge 0.0470, and the oracle ridge 0.0324, so arm H must score 0.0321 or better, "
    "below the oracle ridge's 0.0324). Every statement that the history path recovers the hidden sky "
    "from the PV estimate is unsupported by a passing known-answer test."
)


def problems(*, results: dict | None, code_hash: str) -> list[str]:
    """Return the reasons the waiver must be refused, an empty list when it may be written."""
    if results is None:
        return ["work/known_answers_last_run.json is missing or unreadable"]
    out = []
    if code_hash != GATE_CODE_HASH:
        out.append(f"the code hash is {code_hash}, not {GATE_CODE_HASH} of the gate run")
    failed = {
        k for k, v in results.items() if k.startswith("KF") and isinstance(v, dict) and v.get("pass") is False
    }  # fmt: skip
    if failed != EXPECTED_FAILED:
        out.append(f"the failed checks are {sorted(failed)}, not {sorted(EXPECTED_FAILED)}")
    if results.get("failed") != sorted(EXPECTED_FAILED):
        out.append(f"the failed list is {results.get('failed')!r}, not {sorted(EXPECTED_FAILED)}")
    kf9 = results.get("KF9")
    if not (
        isinstance(kf9, dict)
        and kf9.get("pass") is False
        and isinstance(kf9.get("value"), float)
        and isinstance(kf9.get("need"), float)
        and abs(kf9["value"] - KF9_VALUE) < 1e-12
        and abs(kf9["need"] - KF9_NEED) < 1e-12
    ):
        out.append("KF9 does not have the recorded failing value and the unchanged need")
    for name in sorted(EXPECTED_PASSED):
        entry = results.get(name)
        if not (isinstance(entry, dict) and entry.get("pass") is True):
            out.append(f"{name} is not recorded as passed")
    out += [f"{name} is not recorded" for name in EXPECTED_RECORDED if not results.get(name)]
    pre = results.get("early_stopping_precheck")
    if isinstance(pre, dict) and not all(
        isinstance(v, dict) and v.get("pass") is True for v in pre.values()
    ):
        out.append("the early-stopping pre-check is not recorded as passed in both worlds")
    for key in ("secondary_days_1_60", "secondary_days_61_236", "secondary_pass60"):
        entry = results.get(key)
        if not (isinstance(entry, dict) and isinstance(entry.get("KF9"), dict)):
            out.append(f"{key} has no KF9 reading")
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
            "decided_by": "the maintainer, relayed by the manager",
            "failed_checks": sorted(EXPECTED_FAILED),
            "passed_checks": sorted(EXPECTED_PASSED),
            "recorded_checks": list(EXPECTED_RECORDED),
            "gate_commits": list(GATE_COMMITS),
            "thresholds_changed": False,
            "verdict_rules_changed": False,
            "statement": STATEMENT.format(
                value=results["KF9"]["value"],
                need=results["KF9"]["need"],
                d1=results["secondary_days_1_60"]["KF9"]["value"],
                d2=results["secondary_days_61_236"]["KF9"]["value"],
                p60=results["secondary_pass60"]["KF9"]["value"],
            ),
        },
        "code": code,
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
    """Write the waived gate file in `work`, or raise AssertionError with every reason for refusal.

    The last-run file and the known-answer script must be the very files whose sha256 the waiver
    pins, because neither the code hash nor the last-run file records which thresholds were run.
    """
    last = work / "known_answers_last_run.json"
    results = json.loads(last.read_text()) if last.exists() else None
    reasons = problems(results=results, code_hash=code["code_hash"])
    last_sha = hashlib.sha256(last.read_bytes()).hexdigest() if last.exists() else ""
    if last_sha != expected_last_run_sha256:
        reasons.append(f"the last-run file has sha256 {last_sha}, not {expected_last_run_sha256}")
    if known_answer_sha256 != expected_known_answer_sha256:
        reasons.append(
            f"e00716_known_answer.py has sha256 {known_answer_sha256}, not {expected_known_answer_sha256}"
        )
    if code.get("git_dirty"):
        reasons.append("the 716 folder has uncommitted changes")
    gate = work / "known_answers_passed.json"
    if gate.exists() and not json.loads(gate.read_text()).get("waived"):
        reasons.append(f"{gate} exists and is not a waived gate, so it is not overwritten")
    assert not reasons, "waiver refused: " + "; ".join(reasons)
    assert results is not None
    tmp = Path(str(gate) + ".tmp")
    tmp.write_text(
        json.dumps(
            build_record(
                results=results,
                code=code,
                known_answer_sha256=known_answer_sha256,
                last_run_sha256=last_sha,
            ),
            indent=1,
            default=str,
        )
    )
    tmp.rename(gate)
    return gate


def selftest() -> int:
    """Check, with tiny files and no data, that the waiver refuses what it must and writes what it may."""
    sec = {"KF9": {"value": 0.001, "need": KF9_NEED, "pass": False}}
    good = {
        "KF9": {"value": KF9_VALUE, "need": KF9_NEED, "pass": False},
        "KF10": {"pass": True}, "KF11": {"pass": True}, "KF12": {"pass": True},
        "secondary_days_1_60": sec, "secondary_days_61_236": sec, "secondary_pass60": sec,
        "KF15": {"all_in_range": True},
        "early_stopping_precheck": {"WH": {"pass": True}, "WT": {"pass": True}},
        "wiring_records": {"KF1": {"pass": True}},
        "failed": ["KF9"],
    }  # fmt: skip
    code = {"code_hash": GATE_CODE_HASH, "git_commit": "x", "git_dirty": False}
    assert problems(results=good, code_hash=GATE_CODE_HASH) == []
    assert problems(results=None, code_hash=GATE_CODE_HASH)
    assert problems(results=good, code_hash="0" * 16)
    refused = [
        {**good, "KF10": {"pass": False}, "failed": ["KF10", "KF9"]},  # a different failure set
        {**good, "KF9": {**good["KF9"], "pass": True}, "failed": []},  # KF9 did not fail
        {**good, "KF9": {**good["KF9"], "need": 0.001}},  # the need was changed
        {**good, "KF9": {**good["KF9"], "value": 0.005}},  # a different run
        {**good, "KF11": {"pass": False}, "failed": ["KF11", "KF9"]},
        {**good, "early_stopping_precheck": {"WH": {"pass": False}, "WT": {"pass": True}}},
        {k: v for k, v in good.items() if k != "secondary_pass60"},
        {**good, "secondary_pass60": {}},
        {**good, "KF9": {**good["KF9"], "pass": None}},  # falsy but not False
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
        assert refuses(work=work, code={**code, "git_dirty": True}, known_answer_sha256="s", **pins)
        assert refuses(work=work, code=code, known_answer_sha256="other", **pins)  # script differs
        assert refuses(work=work, code=code, known_answer_sha256="s")  # last-run sha not pinned
        assert not (work / "known_answers_passed.json").exists()
        last.write_text(json.dumps({**good, "KF12": {"pass": True, "x": 1}}))  # a different run
        assert refuses(work=work, code=code, known_answer_sha256="s", **pins)
        last.write_text(json.dumps(good))
        path = waive(work=work, code=code, known_answer_sha256="s", **pins)
        record = json.loads(path.read_text())
        assert record["waived"] is True
        assert record["code"] == code
        assert record["last_run_sha256"] == last_sha
        assert record["waiver"]["decision_quoted"] == DECISION
        assert record["waiver"]["failed_checks"] == ["KF9"]
        assert record["waiver"]["gate_commits"] == ["126577d1d", "e4617f721"]
        assert "0.0397" in record["waiver"]["statement"]
        assert "0.0321" in record["waiver"]["statement"]
        assert record["waiver"]["thresholds_changed"] is False
        assert "Waived known-answer check" in record["waiver"]["statement"]
        assert "unsupported by a passing known-answer test" in record["waiver"]["statement"]
        # a waived gate is rewritten only by the same script, and a real pass is never overwritten
        assert not refuses(work=work, code=code, known_answer_sha256="s", **pins)
        path.write_text(json.dumps({"code": code}))
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
    work = arms.gate_file(data_dir=ML_EXPERIMENTS_DATA_DIR).parent
    digest = hashlib.sha256(
        (Path(__file__).parent / "e00716_known_answer.py").read_bytes()
    ).hexdigest()
    path = waive(work=work, code=arms.code_record(), known_answer_sha256=digest)
    print("waived known-answer check written:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

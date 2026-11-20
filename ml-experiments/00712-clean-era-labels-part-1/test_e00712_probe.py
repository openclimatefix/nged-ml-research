"""Tests of the year-guessing probe, with a stand-in for the command-line tool (no call)."""

# ruff: noqa: D103, ANN001 - a test's name documents it, as for `tests/` (pyproject.toml)
import json
import shutil
import sys
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for folder in (HERE, *HERE.parent.glob("00709-*")):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

probe = pytest.importorskip("e00712_probe")


def _reply(year: int, month: int = 6) -> str:
    return json.dumps({"year": year, "month": month, "confidence": 0.5})


class _Fake:
    """Scripted replies, no network, no `claude`."""

    def __init__(
        self, replies: list[str], *, limit: bool = False, key_source: str = "none"
    ) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.limit = limit
        self.key_source = key_source

    def __call__(self, *, prompt: object) -> probe.trial.RawRun:
        self.calls += 1
        text = self.replies.pop(0) if self.replies else "OK"
        events: list[dict] = [
            {"type": "system", "tools": [], "skills": [], "mcp_servers": [], "cwd": "/empty",
             "apiKeySource": self.key_source},
            {"type": "assistant", "message": {"model": probe.trial.MODEL_ID}},
            {"type": "result", "is_error": False, "result": text, "total_cost_usd": 0.03,
             "usage": {"input_tokens": 100, "output_tokens": 20}},
        ]  # fmt: skip
        status = "rejected" if self.limit else "allowed"
        events.insert(
            1,
            {"type": "rate_limit_event",
             "rate_limit_info": {"status": status, "rateLimitType": "seven_day",
                                 "unifiedWindows": {"seven_day": {"utilization": 0.25}}}},
        )  # fmt: skip
        stdout = "\n".join(json.dumps(e) for e in events) + "\n"
        return probe.trial.RawRun(
            stdout=stdout, stderr="", returncode=0, seconds=0.1, directory="/empty"
        )


@pytest.fixture
def scratch(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    if not probe.OUT_DIR.parent.exists():
        pytest.skip("the private data folder is not on this machine")
    folder = probe.OUT_DIR / "test_scratch_probe"
    shutil.rmtree(folder, ignore_errors=True)
    monkeypatch.setattr(probe, "PROBE_DIR", folder)
    monkeypatch.setattr(probe, "LOG_PATH", folder / "probe_log.jsonl")
    monkeypatch.setattr(probe, "STOP_PATH", folder / "stopped.json")
    monkeypatch.setattr(probe, "GO_FILE", folder / "go.txt")
    monkeypatch.setattr(probe, "CALLS_DIR", folder / "calls")
    monkeypatch.setattr(probe.trial, "cli_version", lambda: "stub")
    monkeypatch.setattr(
        probe, "probe_prompt", lambda *, stem: probe.Prompt(system="s", text=stem, images=())
    )
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


STEMS = [f"D10_x{i}_alarm" for i in range(10)]


def test_parse_reply_is_strict() -> None:
    assert probe.parse_reply(text=_reply(2023)).year == 2023
    assert probe.parse_reply(text="```json\n" + _reply(2023) + "\n```").month == 6
    for bad in (
        "2023",
        '{"year": 2023, "month": 6}',
        '{"year": "2023", "month": 6, "confidence": 0.5}',
        '{"year": 2023, "month": 13, "confidence": 0.5}',
        '{"year": 2023, "month": 6, "confidence": 1.5}',
        '{"year": true, "month": 6, "confidence": 0.5}',
        '{"year": 2023, "month": 6, "confidence": 0.5, "why": "x"}',
    ):
        with pytest.raises(probe.SchemaError):
            probe.parse_reply(text=bad)


def test_the_holiday_helpers() -> None:
    assert probe.easter_sunday(year=2023) == date(2023, 4, 9)
    assert probe.easter_sunday(year=2025) == date(2025, 4, 20)
    assert probe.distance_to_holiday(day=date(2021, 12, 23)) == 2


def test_score_counts_exact_and_within_one_and_the_rule() -> None:
    truth = dict.fromkeys(STEMS, 2023)
    entries: list[dict] = [
        {"tag": "probe", "status": "ok", "call_id": s, "answer": {"year": y, "month": 1}}
        for s, y in zip(STEMS, [2023] * 3 + [2024] * 3 + [2021] * 4, strict=True)
    ]
    result = probe.score(entries=entries, truth=truth, stems=STEMS)
    assert (result["exact_year_hits"], result["within_one_year_hits"]) == (3, 6)
    assert result["passed"] is True
    entries[3]["answer"]["year"] = 2023
    assert probe.score(entries=entries, truth=truth, stems=STEMS)["passed"] is False
    assert probe.score(entries=entries[:5], truth=truth, stems=STEMS)["passed"] is None


def test_the_go_token_must_equal_the_first_line(scratch: Path) -> None:
    assert not probe.go_token_valid(token="x" * 20)
    scratch.mkdir(parents=True)
    probe.GO_FILE.write_text("a" * 20 + "\nsecond\n")
    assert probe.go_token_valid(token="a" * 20)
    assert not probe.go_token_valid(token="b" * 20)
    assert not probe.go_token_valid(token=None)


def test_a_full_run_logs_everything_and_a_resume_sends_nothing(scratch: Path, monkeypatch) -> None:
    fake = _Fake(["OK", *[_reply(2023)] * 10])
    monkeypatch.setattr(probe.trial, "run_claude", fake)
    assert probe.run_probe(stems=STEMS, resume=False, expect_key_source="none") == 0
    assert fake.calls == 11
    entries = probe.read_log()
    assert all(e["rate_limit_events"] and e["wall_seconds"] is not None for e in entries)
    assert probe.seven_day_utilisation(entries=entries, last=True) == 0.25
    again = _Fake([])
    monkeypatch.setattr(probe.trial, "run_claude", again)
    assert probe.run_probe(stems=STEMS, resume=True, expect_key_source="none") == 0
    assert again.calls == 0


def test_an_invalid_reply_is_retried_once(scratch: Path, monkeypatch) -> None:
    fake = _Fake(["OK", "nonsense", _reply(2023), *[_reply(2023)] * 9])
    monkeypatch.setattr(probe.trial, "run_claude", fake)
    assert probe.run_probe(stems=STEMS, resume=False, expect_key_source="none") == 0
    assert fake.calls == 12


def test_a_rate_limit_stops_at_once(scratch: Path, monkeypatch) -> None:
    fake = _Fake([], limit=True)
    monkeypatch.setattr(probe.trial, "run_claude", fake)
    assert probe.run_probe(stems=STEMS, resume=False, expect_key_source="none") == 2
    assert fake.calls == 1
    assert probe.STOP_PATH.exists()


def test_a_different_key_source_aborts_before_any_probe_call(scratch: Path, monkeypatch) -> None:
    fake = _Fake([], key_source="none")
    monkeypatch.setattr(probe.trial, "run_claude", fake)
    assert probe.run_probe(stems=STEMS, resume=False, expect_key_source="ANTHROPIC_API_KEY") == 4
    assert fake.calls == 1


def test_the_call_cap_stops_the_run(scratch: Path, monkeypatch) -> None:
    monkeypatch.setattr(probe, "MAX_CALLS", 3)
    fake = _Fake(["OK", *[_reply(2023)] * 10])
    monkeypatch.setattr(probe.trial, "run_claude", fake)
    assert probe.run_probe(stems=STEMS, resume=False, expect_key_source="none") == 3
    assert fake.calls == 3


@pytest.fixture
def v2() -> Iterator[None]:
    probe.configure_attempt(name="v2")
    yield
    probe.configure_attempt(name="v1")


def test_the_v2_probe_has_its_own_files_and_the_same_limits(v2: None) -> None:
    assert probe.LOG_PATH.name == "probe_log_v2.jsonl"
    assert probe.STOP_PATH.name == "stopped_v2.json"
    assert probe.RESULT_PATH.name == "probe_result_v2.json"
    assert probe.GO_FILE.name == "probe_go_token_v2.txt"
    assert probe.CALLS_DIR.name == "calls_v2"
    assert probe.PROMPT_SOURCE_DIR.name == "prompts_v2"
    assert (probe.MAX_CALLS, probe.MAX_USD, probe.MAX_INPUT_TOKENS) == (14, 2.0, 200_000)
    assert probe.MAX_EXACT_HITS == 3


def test_the_prompt_stamp_must_match_the_current_prompt_version(
    v2: None, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(probe, "PROMPT_SOURCE_DIR", tmp_path)
    assert not probe.prompts_match_the_code()  # no stamp
    (tmp_path / "prompt_version_sha256.txt").write_text("0" * 64)
    assert not probe.prompts_match_the_code()  # a wrong stamp
    (tmp_path / "prompt_version_sha256.txt").write_text(probe.label.prompt_version_hash())
    assert probe.prompts_match_the_code()


def test_attempt_1_is_closed_for_spending(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["e00712_probe.py", "--run-probe", "--go-token", "x" * 20])
    monkeypatch.setattr(
        probe, "select_prompts", lambda **_: [f"llm_D10_x{i}_alarm" for i in range(10)]
    )
    monkeypatch.setattr(probe.asof, "load_stored_candidates", list)
    monkeypatch.setattr(probe, "probe_prompt", lambda *, stem: probe.Prompt("s", stem, ()))
    assert probe.main() == 1
    assert "attempt 1 is closed" in capsys.readouterr().err


def test_the_probe_refuses_a_stem_that_is_not_d10(v2: None, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["e00712_probe.py", "--attempt", "v2", "--run-probe"])
    monkeypatch.setattr(
        probe, "select_prompts", lambda **_: [f"llm_D8_x{i}_alarm" for i in range(10)]
    )
    monkeypatch.setattr(probe.asof, "load_stored_candidates", list)
    assert probe.main() == 1
    assert "only use D10" in capsys.readouterr().err


def test_the_score_counts_only_calls_made_with_the_stamped_prompts(v2: None) -> None:
    truth = dict.fromkeys(STEMS, 2023)
    hashes = {s: f"h-{s}" for s in STEMS}
    entries = [
        {"tag": "probe", "status": "ok", "call_id": s, "answer": {"year": 2020},
         "prompt_version_sha256": "calls-hash", "prompt_sha256": hashes[s]}
        for s in STEMS
    ]  # fmt: skip
    result = probe.score(
        entries=entries, truth=truth, stems=STEMS, version="calls-hash", prompt_hashes=hashes
    )
    assert result["prompt_version_sha256"] == "calls-hash"  # never the hash at scoring time
    assert result["complete"]
    old = [e | {"prompt_version_sha256": "older"} for e in entries]
    assert (
        probe.score(entries=old, truth=truth, stems=STEMS, version="calls-hash")["n_answered"] == 0
    )
    edited = [e | {"prompt_sha256": "edited"} for e in entries]
    assert (
        probe.score(
            entries=edited, truth=truth, stems=STEMS, version="calls-hash", prompt_hashes=hashes
        )["n_answered"]
        == 0
    )


def test_every_probe_call_records_the_prompt_version(scratch: Path, monkeypatch) -> None:
    fake = _Fake(["OK", *[_reply(2023)] * 10])
    monkeypatch.setattr(probe.trial, "run_claude", fake)
    assert probe.run_probe(stems=STEMS, resume=False, expect_key_source="none") == 0
    versions = {e["prompt_version_sha256"] for e in probe.read_log()}
    assert versions == {probe.label.prompt_version_hash()}


def test_a_score_refuses_to_replace_a_different_complete_result(scratch: Path, monkeypatch) -> None:
    monkeypatch.setattr(probe, "RESULT_PATH", scratch / "result.json")
    first = {"complete": True, "passed": True, "n": 1}
    assert probe.write_result(result=first)
    assert probe.write_result(result=first)
    assert not probe.write_result(result={**first, "passed": False})
    assert json.loads(probe.RESULT_PATH.read_text()) == first

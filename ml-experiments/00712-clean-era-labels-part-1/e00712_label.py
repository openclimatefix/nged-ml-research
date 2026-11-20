"""Experiment 712, stage 1: label the detector candidates of the real record, and build the arms.

**Attempt 2 (`--attempt v2`) is a second D8-only attempt with a new identity.** Its prompt text is
`e00712_prompt_v2`, its pin is `prompt_pin_v2.json`, its output folder is `stage1_v2/`, its probe
files carry `_v2`, and its go-token files are new and single-use (renamed to `<name>.used` after the
paid step succeeds). The default `--attempt v1` is unchanged and still reproducible. **Every change
of attempt 2 is justified by the schema failures of attempt 1 and by reply-format robustness
alone. None was decided from the label content or the agreement with the known answers of D8.**

**Attempt 2 spends only on `--primary D8 --arm llm`, within US$4, 140 calls and 1.1 million input
tokens.** A go token is claimed by an atomic rename to `<name>.used.<timestamp>` before the first
call, and a token found in any `*_go_token*.used*` file is refused, so a token is single-use across
the probe and the labelling and across processes. A stopped or failed run needs a new token for any
resume, and so does a billing-probe failure before any label call. Attempt 1 is closed and refuses
to spend. A check whose first attempt timed out gets one further attempt, logged as retry reason
`other`; if that reply fails the schema, the check ends with one real reply.

**Nothing in this script spends unless the maintainer's manager says so.** The default is a dry
run. A call needs `--run-stage1`, a `--go-token` equal to the first line of `STAGE1_GO_FILE` (a
file in the private data folder that the manager writes), an `--expect-key-source` that the billing
probe must report before any check call, a prompt pin that matches the code (`--pin-prompt`), and
explicit `--max-usd`, `--max-calls`, and `--max-input-tokens` within the limits of the plan. The
script never calls `BudgetGuard.with_round`.

**What runs without a call.** `--list-candidates` prints the union candidates of the 16 Primaries.
`--dry-run` builds every prompt a check could need, asserts the cut on a sample, and prints token
estimates. `--build-arms det ceiling ecr` writes `signals/<arm>.parquet` in experiment 709's format.
`--example` prints the as-of rows of one real candidate for arm `det`. `--schema-report` counts the
valid and invalid replies of a stored log and never prints a label, a reason, or a reply.

**What needs calls.** `--run-stage1 --arm llm` (or `llm-ecr`) labels the candidates, resumes with
`--resume`, and stops at once on a rate-limit message. `--build-arms llm` and `--build-arms
llm-ecr` then read the log and write the arm's signal.

**The plan.** Candidates are the union of experiment 709's two detectors on all 16 Primaries, with
no exclusion of D8 or D5 (the pin comes first). Checks follow `e00712_asof.next_check`. At most 2
calls run at a time, and the budget guard counts the calls in flight. A reply is parsed against a
strict schema, with one retry on an invalid reply. A class at confidence 0.8 or more sets the
state, and anything else gives `unknown`.

**Design points this script does not implement.** The 13 evidence columns and the ensemble weather
cells of the prompt (see `e00712_prompt`); the year probe of design section 2.2.1 (the pin is a
hash of the prompt files that the manager commits after that probe); the register file and its
coordinate match (arms `ecr` and `llm-ecr` read the three dated rows of the era table); and the
bank (stage 2).
"""

import argparse
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import e00709_features as features  # ty: ignore[unresolved-import]
import e00712_asof as asof
import e00712_prompt as prompts
import e00712_prompt_v2 as prompts_v2
import e00712_trial as trial
import numpy as np
import polars as pl
from studies.capacity_labeller import Prompt
from studies.capacity_llm import (
    PER_CALL_BUDGET_USD,
    BudgetExceeded,
    BudgetGuard,
    CallStore,
    ModelError,
    ModelReply,
    estimate_input_tokens,
    parse_stream,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

JsonObject = dict[str, Any]
"""A JSON object read from the command-line tool's output or from the call log."""

OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00712-clean-era-labels-part-1"
"""Private output folder: call log, prompts, signals. Never committed."""
RUN_DIR: Final[Path] = OUT_DIR / "stage1"
PROMPT_DIR: Final[Path] = RUN_DIR / "prompts"
LOG_PATH: Final[Path] = RUN_DIR / "call_log.jsonl"
STOP_PATH: Final[Path] = RUN_DIR / "stopped.json"
SIGNALS_DIR: Final[Path] = OUT_DIR / "signals"
STAGE1_GO_FILE: Final[Path] = OUT_DIR / "stage1_go_token.txt"
PIN_PATH: Final[Path] = OUT_DIR / "prompt_pin.json"
TRIAL_LOG_PATH: Final[Path] = OUT_DIR / "trial" / "trial_log.jsonl"
PROBE_PROMPT_DIR: Final[Path] = OUT_DIR / "probe" / "prompts"
"""Where `--build-probe-prompts` writes the year probe's prompts; never the trial's folder."""
PROBE_RESULT_PATH: Final[Path] = OUT_DIR / "probe" / "probe_result.json"
"""The year probe's score, written by `e00712_probe.py`."""
PROBE_PRIMARY: Final[str] = "D10"
MIN_TOKEN_CHARACTERS: Final[int] = 16
ATTEMPTS: Final[tuple[str, ...]] = ("v1", "v2")
V1_PROMPT_TEXTS: Final[tuple[str, str, str]] = (
    prompts.SYSTEM_PROMPT,
    prompts.SCHEMA_NO_REGISTER,
    prompts.SCHEMA_WITH_REGISTER,
)
"""The attempt 1 prompt texts, kept so that `configure_attempt("v1")` can restore them."""
ATTEMPT_STATE: dict[str, Any] = {"name": "v1"}
"""The attempt that the module's paths and prompt texts are set for."""
MIN_WILSON_N: Final[int] = 5
WILSON_Z: Final[float] = 1.96
WINDOWS: Final[tuple[str, ...]] = ("seven_day", "five_hour")

STAGE1_MAX_USD: Final[float] = 20.0
STAGE1_MAX_CALLS: Final[int] = 520
STAGE1_MAX_INPUT_TOKENS: Final[int] = 5_000_000
"""The most a stage 1 run may be given: the plan's table has at most 480 calls, 3.8 million input
tokens, and about US$17."""
V2_MAX_USD: Final[float] = 4.0
V2_MAX_CALLS: Final[int] = 140
"""The 65 D8 checks, one retry each, and the billing probes of any restarts."""
V2_MAX_INPUT_TOKENS: Final[int] = 1_100_000
"""Twice the estimated 492,000 input tokens of the 65 D8 prompts, plus margin."""
V2_PRIMARY: Final[str] = "D8"
MAX_WORKERS: Final[int] = 2
MAX_ATTEMPTS: Final[int] = 2
"""A reply that fails the schema is asked for once more; each attempt is a call."""
PRINT_EVERY: Final[int] = 25
CUT_SAMPLE: Final[int] = 40
CUT_SEED: Final[int] = 712_001
WORKED_EXAMPLE_SITE: Final[str] = "D8"
WORKED_EXAMPLE_ALARM: Final[date] = date(2025, 4, 26)
STATE_NAMES: Final[tuple[str, ...]] = ("none", "unknown", "era", "switch")

LABELS: Final[frozenset[str]] = frozenset({"era", "switch", "noise", "undecided"})
MIRRORS: Final[frozenset[str]] = frozenset({"yes", "no", "unmetered"})
SHAPES: Final[frozenset[str]] = frozenset({"solar", "flat", "demand-like", "irregular"})
SCHEMA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "label",
        "onset_offset_days",
        "confidence",
        "neighbour_mirror",
        "shape",
        "register_row",
        "reason",
    }
)
MAX_REASON_WORDS: Final[int] = 60
MAX_KEY_TEXT: Final[int] = 40
PARSER_STRICT: Final[str] = "strict-1"
"""The first parser: every key present, `reason` of at most 60 words."""
PARSER_LENIENT: Final[str] = "lenient-2"
"""`register_row` may be absent (read as null); a long `reason` is cut to 60 words and flagged."""
REPARSED_LOG_PATH: Final[Path] = RUN_DIR / "call_log_reparsed.jsonl"


class SchemaError(ValueError):
    """A reply does not satisfy the output schema; the message names a field and never a value."""


@dataclass(frozen=True)
class Answer:
    """A reply that satisfies the schema.

    Attributes:
        label: `era`, `switch`, `noise`, or `undecided`.
        onset_offset_days: The estimated first day of the shift, 0 or below.
        confidence: The probability that `label` is right, from 0 to 1.
        neighbour_mirror: `yes`, `no`, or `unmetered`.
        shape: `solar`, `flat`, `demand-like`, or `irregular`.
        register_row: The number of the register row relied on, or `None`.
        reason: The audit text, never parsed.
        reason_truncated: Whether `reason` was cut to its first 60 words.
        parser_version: `PARSER_STRICT` or `PARSER_LENIENT`.
    """

    label: str
    onset_offset_days: int
    confidence: float
    neighbour_mirror: str
    shape: str
    register_row: int | None
    reason: str
    reason_truncated: bool = False
    parser_version: str = PARSER_LENIENT

    def as_json(self) -> JsonObject:
        """Return the answer as a JSON-ready mapping."""
        return {
            "label": self.label,
            "onset_offset_days": self.onset_offset_days,
            "confidence": self.confidence,
            "neighbour_mirror": self.neighbour_mirror,
            "shape": self.shape,
            "register_row": self.register_row,
            "reason": self.reason,
            "reason_truncated": self.reason_truncated,
            "parser_version": self.parser_version,
        }


def _is_int(*, value: object) -> bool:
    """Return True for an integer that is not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def parse_reply(  # noqa: PLR0912 - one branch per schema field
    *, text: str, n_register_rows: int = 0, strict: bool = False
) -> Answer:
    """Validate a reply against the output schema of design section 2.3.

    Args:
        text: The reply text, as one JSON object (a surrounding code fence is removed).
        n_register_rows: How many register rows the prompt showed; `register_row` must be `None`
            when this is 0, and otherwise `None` or a number from 1 to this count.
        strict: Use the first parser (`PARSER_STRICT`), which rejects an absent `register_row`
            and a `reason` of more than 60 words.

    Returns:
        The answer.

    Raises:
        SchemaError: If the text is not one JSON object that satisfies the schema. The message
            names a field or a problem and never repeats a value of the reply.
    """
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as error:
        msg = f"not valid JSON ({error.msg})"
        raise SchemaError(msg) from error
    if not isinstance(obj, dict):
        msg = "not a JSON object"
        raise SchemaError(msg)
    required = SCHEMA_KEYS if strict else SCHEMA_KEYS - {"register_row"}
    if not required <= set(obj) or not set(obj) <= SCHEMA_KEYS:
        missing = sorted(required - set(obj))
        extra = sorted(str(k)[:MAX_KEY_TEXT] for k in set(obj) - SCHEMA_KEYS)
        msg = f"keys differ from the schema: missing {missing}, extra {extra}"
        raise SchemaError(msg)
    if obj["label"] not in LABELS:
        msg = "field 'label' is not one of the four classes"
        raise SchemaError(msg)
    if not _is_int(value=obj["onset_offset_days"]) or obj["onset_offset_days"] > 0:
        msg = "field 'onset_offset_days' is not an integer of 0 or below"
        raise SchemaError(msg)
    confidence = obj["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        msg = "field 'confidence' is not a number"
        raise SchemaError(msg)
    if not 0.0 <= confidence <= 1.0:
        msg = "field 'confidence' is outside 0 to 1"
        raise SchemaError(msg)
    if obj["neighbour_mirror"] not in MIRRORS:
        msg = "field 'neighbour_mirror' is not one of its three values"
        raise SchemaError(msg)
    if obj["shape"] not in SHAPES:
        msg = "field 'shape' is not one of its four values"
        raise SchemaError(msg)
    row = obj.get("register_row")
    if row is not None and (not _is_int(value=row) or not 1 <= row <= n_register_rows):
        msg = "field 'register_row' is not null or a shown row number"
        raise SchemaError(msg)
    reason = obj["reason"]
    if not isinstance(reason, str):
        msg = "field 'reason' is not a string"
        raise SchemaError(msg)
    words = reason.split()
    truncated = len(words) > MAX_REASON_WORDS
    if truncated and strict:
        msg = "field 'reason' is not a string of at most 60 words"
        raise SchemaError(msg)
    if truncated:
        reason = " ".join(words[:MAX_REASON_WORDS])
    return Answer(
        label=obj["label"],
        onset_offset_days=obj["onset_offset_days"],
        confidence=float(confidence),
        neighbour_mirror=obj["neighbour_mirror"],
        shape=obj["shape"],
        register_row=row,
        reason=reason,
        reason_truncated=truncated,
        parser_version=PARSER_STRICT if strict else PARSER_LENIENT,
    )


def reparse_log(*, entries: Sequence[JsonObject]) -> list[JsonObject]:
    """Re-parse the stored replies of an `llm` arm log with the current parser, without a call.

    The input entries are not changed. Each label entry is copied with its `answer` read again from
    `raw_output`, its stored answer kept as `original_answer`, and `parser_version` set. Once a
    check has a valid answer, a later attempt of the same check is marked unused, because a live
    run would not have made it. A reply that is still invalid keeps `answer` as `None` and gets a
    `parse_error` that names a field and never a value.

    Args:
        entries: The log entries (arm `llm`, so no register row is shown).

    Returns:
        The new entries, in the same order.
    """
    out: list[JsonObject] = []
    settled: set[str] = set()
    for entry in entries:
        new = {k: v for k, v in entry.items() if k != "answer"}
        if entry.get("tag") != "label" or entry.get("status") != "ok":
            new["answer"] = entry.get("answer")
            out.append(new)
            continue
        new["original_answer"] = entry.get("answer")
        new["parser_version"] = PARSER_LENIENT
        call_id = str(entry["call_id"])
        if call_id in settled:
            new |= {"answer": None, "parse_error": "unused: the check was already answered"}
        else:
            try:
                new["answer"] = parse_reply(text=str(entry.get("raw_output") or "")).as_json()
                settled.add(call_id)
            except SchemaError as error:
                new |= {"answer": None, "parse_error": str(error)}
        out.append(new)
    return out


def _confident_classes(*, entries: Sequence[JsonObject]) -> Counter[str]:
    """Count the answers that set an era or a switch at the 0.8 confidence rule."""
    found: Counter[str] = Counter()
    for entry in entries:
        answer = entry.get("answer")
        if answer and answer["label"] in ("era", "switch") and answer["confidence"] >= 0.8:
            found[answer["label"]] += 1
    return found


def print_reparse_comparison(
    *, original: Sequence[JsonObject], reparsed: Sequence[JsonObject]
) -> None:
    """Print how many replies are valid, and how many set an era or a switch, under each parser.

    No label, reason, or reply text is printed; a reply that is still invalid is listed by the
    error type (the schema message, which names a field only).
    """
    for name, entries in (("strict", original), ("lenient", reparsed)):
        labels = [e for e in entries if e.get("tag") == "label" and e.get("status") == "ok"]
        valid = [e for e in labels if e.get("answer")]
        used = [
            e for e in labels if e.get("parse_error") != "unused: the check was already answered"
        ]
        print(
            f"{name} parser: {len(valid)} of {len(used)} replies valid; "
            f"at the 0.8 rule {dict(_confident_classes(entries=valid))}"
        )
    still: Counter[str] = Counter(
        str(e["parse_error"])
        for e in reparsed
        if e.get("tag") == "label"
        and e.get("answer") is None
        and e.get("parse_error")
        and not str(e["parse_error"]).startswith("unused")
    )
    print(f"still invalid under the lenient parser: {sum(still.values())}")
    for message, count in still.most_common():
        print(f"  {count} x {message}")
    cut = sum(1 for e in reparsed if (e.get("answer") or {}).get("reason_truncated"))
    print(f"replies whose reason was cut to 60 words: {cut}")


def schema_report(*, log_path: Path, strict: bool = False) -> tuple[int, int, Counter[str]]:
    """Count the valid and invalid replies of a call log, with the schema errors by message.

    Only the structure of each reply is tested. No label, reason, or reply text is returned.

    Args:
        log_path: A trial or label call log.
        strict: Use the first parser.

    Returns:
        The number of valid replies, the number of invalid replies, and a count of each message.
    """
    valid, invalid, errors = 0, 0, Counter()
    for line in log_path.read_text().splitlines():
        entry = json.loads(line)
        if entry.get("status") != "ok" or entry.get("tag") in ("billing-probe", "year-probe"):
            continue
        try:
            parse_reply(text=str(entry.get("raw_output", "")), strict=strict)
        except SchemaError as error:
            invalid += 1
            errors[str(error)] += 1
        else:
            valid += 1
    return valid, invalid, errors


# ---------------------------------------------------------------- the pin and the go token


def prompt_version_hash() -> str:
    """Return the hash of everything that fixes a prompt.

    The hash covers the system prompt, the two schema texts, and the source of the prompt builder.
    For attempt 2 it also covers the source of `e00712_prompt_v2`.
    """
    source = Path(prompts.__file__).read_bytes()
    parts = (prompts.SYSTEM_PROMPT, prompts.SCHEMA_NO_REGISTER, prompts.SCHEMA_WITH_REGISTER)
    data = "\x00".join(parts).encode() + b"\x00" + source
    if ATTEMPT_STATE["name"] == "v2":
        data += b"\x00" + Path(prompts_v2.__file__).read_bytes()
    return hashlib.sha256(data).hexdigest()


def configure_attempt(*, name: str) -> None:
    """Point the paths and the prompt texts of this module at one attempt.

    Attempt `v1` uses the original names. Attempt `v2` uses `stage1_v2/`, `prompt_pin_v2.json`,
    `stage1_go_token_v2.txt`, `probe/prompts_v2/`, and `probe/probe_result_v2.json`, and the
    reply-format texts of `e00712_prompt_v2`.

    Args:
        name: `v1` or `v2`.

    Raises:
        ValueError: If the name is not an attempt.
    """
    if name not in ATTEMPTS:
        msg = f"unknown attempt {name!r}"
        raise ValueError(msg)
    suffix = "" if name == "v1" else "_v2"
    texts = V1_PROMPT_TEXTS
    if name == "v2":
        texts = (
            prompts_v2.SYSTEM_PROMPT,
            prompts_v2.SCHEMA_NO_REGISTER,
            prompts_v2.SCHEMA_WITH_REGISTER,
        )
    prompts.SYSTEM_PROMPT, prompts.SCHEMA_NO_REGISTER, prompts.SCHEMA_WITH_REGISTER = texts
    run_dir = OUT_DIR / f"stage1{suffix}"
    paths = {
        "RUN_DIR": run_dir,
        "PROMPT_DIR": run_dir / "prompts",
        "LOG_PATH": run_dir / "call_log.jsonl",
        "STOP_PATH": run_dir / "stopped.json",
        "REPARSED_LOG_PATH": run_dir / "call_log_reparsed.jsonl",
        "SIGNALS_DIR": OUT_DIR / f"signals{suffix}",
        "STAGE1_GO_FILE": OUT_DIR / f"stage1_go_token{suffix}.txt",
        "PIN_PATH": OUT_DIR / f"prompt_pin{suffix}.json",
        "PROBE_PROMPT_DIR": OUT_DIR / "probe" / f"prompts{suffix}",
        "PROBE_RESULT_PATH": OUT_DIR / "probe" / f"probe_result{suffix}.json",
    }
    globals().update(paths)
    ATTEMPT_STATE["name"] = name


def attempt_name() -> str:
    """Return the attempt that the module is configured for."""
    return str(ATTEMPT_STATE["name"])


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class BillingRouteError(RuntimeError):
    """A call reported an `apiKeySource` other than the one the run expects."""


def pin_matches() -> bool:
    """Return True if a pin file exists and holds the current prompt version hash."""
    if not PIN_PATH.exists():
        return False
    return json.loads(PIN_PATH.read_text()).get("prompt_version_sha256") == prompt_version_hash()


def probe_blocker() -> str | None:
    """Return why the year probe does not allow a pin, or `None` if it passed on this prompt code.

    The probe passes when `probe_result.json` reports `passed` true (at most 3 of 10 exact years)
    and holds the current prompt version hash.
    """
    if not PROBE_RESULT_PATH.exists():
        return "the year probe has no probe_result.json"
    result = json.loads(PROBE_RESULT_PATH.read_text())
    if result.get("prompt_version_sha256") != prompt_version_hash():
        return "probe_result.json was produced from a different prompt version"
    if result.get("passed") is not True:
        return "the year probe did not pass (more than 3 of 10 exact years, or incomplete)"
    return None


def token_valid(*, token: str | None, path: Path) -> bool:
    """Return True if the token equals the first line of the go file and was never used.

    A token is refused if the first line of any `*_go_token*.used*` file in the go file's folder
    holds it, so a token is single-use across the probe and the labelling, across uses, and across
    processes.

    Args:
        token: The token given on the command line.
        path: The go file.

    Returns:
        Whether the token may start a paid step.
    """
    if not token or len(token) < MIN_TOKEN_CHARACTERS or not path.exists():
        return False
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != token:
        return False
    for used in path.parent.glob("*_go_token*.used*"):
        spent = used.read_text().splitlines()
        if spent and spent[0].strip() == token:
            return False
    return True


def claim_token(*, token: str | None, path: Path) -> Path | None:
    """Claim a go token by an atomic rename to `<name>.used.<timestamp>`, before the first call.

    A second process that checks the token afterwards finds no go file. A run that stops or fails
    after the claim needs a new token for any resume, and so does a billing-probe failure before
    any label call.

    Args:
        token: The token given on the command line.
        path: The go file.

    Returns:
        The path of the claimed (renamed) file, or `None` if the token is not valid or another
        process claimed it first.
    """
    if not token_valid(token=token, path=path):
        return None
    used = path.with_name(f"{path.name}.used.{time.time_ns()}")
    try:
        path.rename(used)
    except FileNotFoundError:
        return None
    return used


def go_token_valid(*, token: str | None) -> bool:
    """Return True if the token equals the first line of the go file the manager wrote."""
    return token_valid(token=token, path=STAGE1_GO_FILE)


# ---------------------------------------------------------------- the log and resume


def read_log(*, path: Path | None = None) -> list[JsonObject]:
    """Return every entry of a call log (default: the stage 1 log), or an empty list if none."""
    path = path or LOG_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_call_id(*, call_id: str) -> tuple[str, str, str]:
    """Split `arm:key_kind` into the arm, the candidate key, and the check kind."""
    arm, rest = call_id.split(":", 1)
    key, kind = rest.rsplit("_", 1)
    return arm, key, kind


def resolved_from_log(
    *,
    entries: Sequence[JsonObject],
    candidates: Sequence[prompts.Candidate],
    arm: str,
    version: str | None = None,
) -> dict[str, list[asof.Resolved]]:
    """Rebuild the finished checks of every candidate of one arm from a call log.

    A check is finished when one attempt gave an answer that satisfies the schema, or when
    `MAX_ATTEMPTS` attempts did not (the check then has no valid reply and its label is `None`).

    Args:
        entries: The log entries.
        candidates: The arm's candidates.
        arm: The arm.
        version: Use only entries made with this prompt version hash, or every entry if `None`.

    Returns:
        The finished checks by `Candidate.key`, in day order. An entry of a candidate that is not
        in `candidates` (a run restricted to some Primaries) is skipped.
    """
    by_key = {c.key: c for c in candidates}
    grouped: dict[str, list[JsonObject]] = {}
    for entry in entries:
        if version is not None and entry.get("prompt_version_sha256") != version:
            continue
        if entry.get("tag") == "label" and str(entry["call_id"]).startswith(f"{arm}:"):
            grouped.setdefault(str(entry["call_id"]), []).append(entry)
    out: dict[str, list[asof.Resolved]] = {}
    for call_id, group in grouped.items():
        _, key, kind = split_call_id(call_id=call_id)
        if key not in by_key:
            continue
        candidate = by_key[key]
        good = [e for e in group if e.get("status") == "ok" and e.get("answer") is not None]
        if good:
            answer = good[-1]["answer"]
            item = asof.Resolved(
                kind=kind,
                day=candidate.alarm + int(good[-1]["check_offset_days"]),
                label=answer["label"],
                confidence=float(answer["confidence"]),
                onset_offset_days=int(answer["onset_offset_days"]),
            )
        elif len(group) >= MAX_ATTEMPTS:
            item = asof.Resolved(
                kind=kind,
                day=candidate.alarm + int(group[-1]["check_offset_days"]),
                label=None,
                confidence=0.0,
            )
        else:
            continue
        out.setdefault(key, []).append(item)
    return {key: sorted(items, key=lambda r: r.day) for key, items in out.items()}


# ---------------------------------------------------------------- the call loop


@dataclass(frozen=True)
class Limits:
    """The explicit limits of one run.

    Attributes:
        max_usd: The reported-cost ceiling.
        max_calls: The call ceiling, retries and the probe included.
        max_input_tokens: The input-token ceiling.
    """

    max_usd: float
    max_calls: int
    max_input_tokens: int


class Labeller:
    """Runs the checks with the guard, the call log, and the stop flag.

    A stop cannot interrupt a call that the other worker already has in flight, which can last up
    to `CALL_TIMEOUT_SECONDS`; it only prevents every later call.
    """

    def __init__(
        self, *, limits: Limits, arm: str, resume: bool, expect_key_source: str | None = None
    ) -> None:
        """Create the guard, the log, and the stop flag.

        Args:
            limits: The explicit limits.
            arm: `llm` or `llm-ecr`.
            resume: Whether to continue from the existing log, whose totals the guard then counts.
            expect_key_source: The `apiKeySource` every call must report, or `None` for no check.
        """
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        self.arm = arm
        self.store = CallStore(directory=RUN_DIR / "calls", log_path=LOG_PATH)
        self.guard = BudgetGuard.from_log(log_path=LOG_PATH) if resume else BudgetGuard()
        self.guard.ceiling_usd = limits.max_usd
        self.guard.max_calls = limits.max_calls
        self.guard.max_input_tokens = limits.max_input_tokens
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.inflight = 0
        self.version_hash = prompt_version_hash()
        earlier = read_log() if resume else []
        self.attempts: dict[str, int] = {}
        self.parse_failed: set[str] = set()
        for entry in earlier:
            if entry.get("prompt_version_sha256") != self.version_hash:
                continue
            call_id = str(entry["call_id"])
            self.attempts[call_id] = self.attempts.get(call_id, 0) + 1
            if entry.get("parse_error"):
                self.parse_failed.add(call_id)
        self.done = len(earlier)
        self.output_tokens = sum(int(e["output_tokens"]) for e in earlier)
        self.api_key_source = "unknown"
        self.expect_key_source = expect_key_source
        self.cli_version = ""

    def totals_line(self) -> str:
        """One line of running totals."""
        g = self.guard
        return (
            f"running totals: calls={g.calls} input_tokens={g.input_tokens} "
            f"output_tokens={self.output_tokens} reported_cost_usd={g.spent_usd:.3f}"
        )

    def trip(self, *, reason: str) -> None:
        """Set the stop flag and write what was done, for every kind of stop."""
        self.stop.set()
        STOP_PATH.write_text(json.dumps({"reason": reason, "totals": self.totals_line()}, indent=2))

    def _reserve(self, *, prompt: Prompt) -> None:
        """Count the call in flight against every limit before it starts.

        Args:
            prompt: The prompt about to be sent.

        Raises:
            BudgetExceeded: If the call, with those already in flight, could pass a limit.
        """
        with self.lock:
            g = self.guard
            pending = self.inflight + 1
            step = max(g.largest_call_usd, PER_CALL_BUDGET_USD)
            if g.calls + pending > g.max_calls:
                msg = f"call {g.calls + pending} would pass the {g.max_calls} cap"
            elif (
                g.input_tokens + pending * estimate_input_tokens(prompt=prompt) > g.max_input_tokens
            ):
                msg = f"the input tokens would pass {g.max_input_tokens}"
            elif g.spent_usd + pending * step >= g.ceiling_usd:
                msg = (
                    f"spent US${g.spent_usd:.2f}; {pending} calls in flight would reach the ceiling"
                )
            else:
                self.inflight += 1
                return
        raise BudgetExceeded(msg)

    def _log(self, *, entry: JsonObject) -> None:
        """Append one entry to the log under the lock, stamped with the time it finished."""
        entry.setdefault("finished_utc", _now())
        with self.lock:
            self.store.append_log(entry=entry)

    def _finish(
        self, *, entry: JsonObject, reply: ModelReply, events: Sequence[JsonObject]
    ) -> None:
        """Add the token counts and the cost to a log entry and to the running totals."""
        result = next((e for e in events if e.get("type") == "result"), {})
        usage: JsonObject = result.get("usage") or {}
        entry |= {
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
            "uncached_input_tokens": usage.get("input_tokens"),
            "cache_creation_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_tokens": usage.get("cache_read_input_tokens"),
            "cost_usd": reply.cost_usd,
            "reported_total_cost_usd": result.get("total_cost_usd"),
            "cost_is_placeholder": not events,
        }
        with self.lock:
            self.guard.record(reply=reply)
            self.done += 1
            self.output_tokens += reply.output_tokens
            if entry.get("apiKeySource") is not None:
                self.api_key_source = str(entry["apiKeySource"])
            if self.done % PRINT_EVERY == 0:
                print(self.totals_line(), flush=True)

    @staticmethod
    def _cap_cost(*, prompt: Prompt) -> ModelReply:
        """Return the placeholder reply of a call whose bill is unknown, costed at the cap."""
        return ModelReply(
            text="",
            input_tokens=estimate_input_tokens(prompt=prompt),
            output_tokens=0,
            cost_usd=PER_CALL_BUDGET_USD,
            model=trial.MODEL_ID,
        )

    def call(
        self,
        *,
        call_id: str,
        tag: str,
        offset: int | None,
        prompt: Prompt,
        n_register_rows: int = 0,
    ) -> Answer | None:
        """Make one check call, with one retry on an invalid reply, and log every attempt.

        Any error sets the stop flag before it propagates, so the other worker takes no new call.

        Args:
            call_id: A date-free identifier, `arm:Primary_episode_kind`.
            tag: `billing-probe` or `label`.
            offset: The check day as an offset from the alarm day, or `None` for the probe.
            prompt: The prompt.
            n_register_rows: The number of register rows the prompt shows.

        Returns:
            The answer, or `None` for the probe and for a check whose attempts all failed the
            schema.

        Raises:
            RateLimited: On a rate-limit message.
            BudgetExceeded: If a limit would be passed.
            ModelError: If the call fails or times out.
            BillingRouteError: If the call reports an unexpected `apiKeySource`.
        """
        for attempt in range(self.attempts.get(call_id, 0), MAX_ATTEMPTS):
            if self.stop.is_set():
                return None
            entry: JsonObject = {
                "call_id": call_id,
                "attempt": attempt,
                "tag": tag,
                "arm": self.arm,
                "check_offset_days": offset,
                "model": trial.MODEL_ID,
                "claude_code_version": self.cli_version,
                "prompt_version_sha256": self.version_hash,
                "prompt_sha256": prompts.sha256(text=prompt.text),
                "prompt": prompt.text,
                "estimated_input_tokens": estimate_input_tokens(prompt=prompt),
                "started_utc": _now(),
                "run_attempt": attempt_name(),
                "retry_reason": (
                    None
                    if attempt == 0
                    else ("parse_failure" if call_id in self.parse_failed else "other")
                ),
            }
            try:
                self._reserve(prompt=prompt)
            except BudgetExceeded as error:
                self.trip(reason=f"budget: {error}")
                raise
            try:
                finished, answer = self._attempt(
                    entry=entry, tag=tag, prompt=prompt, n_register_rows=n_register_rows
                )
            finally:
                with self.lock:
                    self.inflight -= 1
            if finished:
                return answer
            self.parse_failed.add(call_id)
        return None

    def _attempt(
        self, *, entry: JsonObject, tag: str, prompt: Prompt, n_register_rows: int
    ) -> tuple[bool, Answer | None]:
        """Run one attempt of a call that is already reserved, and log it.

        Args:
            entry: The log entry so far.
            tag: `billing-probe` or `label`.
            prompt: The prompt.
            n_register_rows: The number of register rows the prompt shows.

        Returns:
            `(False, None)` if the reply failed the schema and a retry may follow, else `(True,
            answer)`.

        Raises:
            RateLimited: On a rate-limit message.
            ModelError: If the call fails or times out.
            BillingRouteError: If the call reports an unexpected `apiKeySource`.
        """
        try:
            run = trial.run_claude(prompt=prompt)
        except ModelError as error:
            self._finish(entry=entry, reply=self._cap_cost(prompt=prompt), events=[])
            entry |= {"status": "failed", "error": str(error)[:500], "wall_seconds": None}
            self._log(entry=entry)
            self.trip(reason=f"timeout or launch failure: {error}")
            raise
        events = trial.stream_events(stdout=run.stdout)
        limit = trial.rate_limit_hit(events=events, run=run)
        init = next((e for e in events if e.get("type") == "system" and "tools" in e), {})
        result = next((e for e in events if e.get("type") == "result"), {})
        entry |= {
            "apiKeySource": init.get("apiKeySource"),
            "init_event": init,
            "result_event": {k: v for k, v in result.items() if k != "result"},
            "rate_limit_events": trial.rate_limit_payloads(events=events),
            "wall_seconds": round(run.seconds, 2),
        }
        if (
            self.expect_key_source is not None
            and init.get("apiKeySource") != self.expect_key_source
        ):
            self._finish(entry=entry, reply=self._cap_cost(prompt=prompt), events=events)
            msg = (
                f"the call reports apiKeySource {init.get('apiKeySource')!r}, "
                f"expected {self.expect_key_source!r}"
            )
            entry |= {"status": "failed", "error": msg}
            self._log(entry=entry)
            self.trip(reason=msg)
            raise BillingRouteError(msg)
        try:
            reply = parse_stream(
                output=run.stdout,
                requested_model=trial.MODEL_ID,
                expected_cwd=Path(run.directory),
            )
        except ModelError as error:
            billed = error.reply or self._cap_cost(prompt=prompt)
            self._finish(entry=entry, reply=billed, events=events)
            entry |= {"status": "failed", "error": str(error)[:500], "raw_output": run.stdout}
            self._log(entry=entry)
            self.trip(reason=limit or f"model error: {error}")
            if limit:
                raise trial.RateLimited(limit) from error
            raise
        self._finish(entry=entry, reply=reply, events=events)
        entry |= {"status": "ok", "raw_output": reply.text}
        if limit:
            self._log(entry=entry)
            self.trip(reason=limit)
            raise trial.RateLimited(limit)
        if tag == "billing-probe":
            entry["answer"] = None
            self._log(entry=entry)
            return True, None
        try:
            answer = parse_reply(text=reply.text, n_register_rows=n_register_rows)
        except SchemaError as error:
            entry |= {"answer": None, "parse_error": str(error)}
            self._log(entry=entry)
            return False, None
        entry["answer"] = answer.as_json()
        self._log(entry=entry)
        return True, answer


def label_candidate(
    *,
    labeller: Labeller,
    candidate: prompts.Candidate,
    resolved: list[asof.Resolved],
    inputs: prompts.Inputs,
    register: tuple[prompts.RegisterRow, ...] | None,
    offsets: Sequence[int],
) -> None:
    """Run the checks of one candidate in order until none remains or the run stops.

    Any error that is not already a stop (the cut test raising, a bug) sets the stop flag before it
    propagates, so the other worker takes no new call.

    Args:
        labeller: The labeller.
        candidate: The candidate.
        resolved: The checks already finished (from the log, on a resume); extended in place.
        inputs: The real record.
        register: The register rows of arm `llm-ecr`, or `None`.
        offsets: The scheduled offsets from the alarm day.
    """
    try:
        while not labeller.stop.is_set():
            check = asof.next_check(
                candidate=candidate, resolved=resolved, offsets=offsets, last_day=inputs.last_day
            )
            if check is None:
                return
            prompt = prompts.prompt_of(inputs=inputs, check=check, register=register)
            prompts.assert_cut_changes_nothing(
                inputs=inputs, check=check, prompt=prompt, register=register
            )
            shown = (
                len(
                    prompts.shown_register_rows(
                        rows=register,
                        candidate=candidate,
                        day0=check.day,
                        lag=prompts.DEFAULT_PUBLICATION_LAG_DAYS,
                    )
                )
                if register is not None
                else 0
            )
            answer = labeller.call(
                call_id=check.call_id(arm=labeller.arm),
                tag="label",
                offset=check.offset,
                prompt=prompt,
                n_register_rows=shown,
            )
            if labeller.stop.is_set() and answer is None:
                return
            resolved.append(
                asof.Resolved(
                    kind=check.kind,
                    day=check.day,
                    label=answer.label if answer else None,
                    confidence=answer.confidence if answer else 0.0,
                    onset_offset_days=answer.onset_offset_days if answer else None,
                )
            )
    except BudgetExceeded, ModelError, trial.RateLimited, BillingRouteError:
        raise
    except Exception as error:
        labeller.trip(reason=f"unexpected {type(error).__name__}: {error}")
        raise


def window_utilisation(*, entries: Sequence[JsonObject], window: str) -> list[float]:
    """Return every `unifiedWindows.<window>.utilization` that the logged rate-limit events hold."""
    out: list[float] = []
    for entry in entries:
        for payload in entry.get("rate_limit_events") or []:
            info = payload.get("rate_limit_info") if isinstance(payload, dict) else None
            windows = info.get("unifiedWindows") if isinstance(info, dict) else None
            item = windows.get(window) if isinstance(windows, dict) else None
            if isinstance(item, dict) and isinstance(item.get("utilization"), int | float):
                out.append(float(item["utilization"]))
    return out


def run_totals(*, entries: Sequence[JsonObject]) -> JsonObject:
    """Return the calls, tokens, cost, and first and last utilisation of some log entries."""
    totals: JsonObject = {
        "calls": len(entries),
        "input_tokens": sum(int(e.get("input_tokens") or 0) for e in entries),
        "output_tokens": sum(int(e.get("output_tokens") or 0) for e in entries),
        "usd": round(sum(float(e.get("cost_usd") or 0.0) for e in entries), 4),
    }
    for window in WINDOWS:
        seen = window_utilisation(entries=entries, window=window)
        totals[f"{window}_utilization_first"] = seen[0] if seen else None
        totals[f"{window}_utilization_last"] = seen[-1] if seen else None
    return totals


def run_stage1(
    *,
    candidates: list[prompts.Candidate],
    inputs: prompts.Inputs,
    register: tuple[prompts.RegisterRow, ...] | None,
    limits: Limits,
    arm: str,
    resume: bool,
    expect_key_source: str | None,
    probe_only: bool,
) -> int:
    """Run the probe and the labelling, and write `run_<n>.json` at the start and at the end.

    The record holds the start and end times, and the calls, tokens, US$, and first and last
    seven-day and five-hour utilisation of this invocation (the log entries it added).

    Args:
        candidates: The arm's candidates.
        inputs: The real record.
        register: The register rows of arm `llm-ecr`, or `None`.
        limits: The explicit limits.
        arm: `llm` or `llm-ecr`.
        resume: Whether to continue from the existing log.
        expect_key_source: The `apiKeySource` every call must report.
        probe_only: Whether to stop after the probe.

    Returns:
        The exit status; see `_run_stage1`.
    """
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    number = len(list(RUN_DIR.glob("run_*.json"))) + 1
    path = RUN_DIR / f"run_{number}.json"
    before = len(read_log())
    started, clock = _now(), time.monotonic()
    record: JsonObject = {"run": number, "arm": arm, "started_utc": started, "finished_utc": None}
    path.write_text(json.dumps(record, indent=2))
    status = 3
    try:
        status = _run_stage1(
            candidates=candidates,
            inputs=inputs,
            register=register,
            limits=limits,
            arm=arm,
            resume=resume,
            expect_key_source=expect_key_source,
            probe_only=probe_only,
        )
    finally:
        elapsed = time.monotonic() - clock
        record |= {
            "finished_utc": _now(),
            "elapsed_seconds": round(elapsed, 1),
            "exit_status": status,
            **run_totals(entries=read_log()[before:]),
        }
        path.write_text(json.dumps(record, indent=2))
        print(f"elapsed {elapsed:.0f} s; run record {path}")
    return status


def _run_stage1(  # noqa: PLR0911, PLR0912 - one return per stop reason
    *,
    candidates: list[prompts.Candidate],
    inputs: prompts.Inputs,
    register: tuple[prompts.RegisterRow, ...] | None,
    limits: Limits,
    arm: str,
    resume: bool,
    expect_key_source: str | None,
    probe_only: bool,
) -> int:
    """Send the billing probe, check its route, then label every candidate, 2 at a time at most.

    The probe is sent on every start with its own call id, and every later call must report the
    same `apiKeySource`.

    Args:
        candidates: The arm's candidates.
        inputs: The real record.
        register: The register rows of arm `llm-ecr`, or `None`.
        limits: The explicit limits.
        arm: `llm` or `llm-ecr`.
        resume: Whether to continue from the existing log.
        expect_key_source: The `apiKeySource` every call must report, or `None` for a probe only.
        probe_only: Whether to stop after the probe.

    Returns:
        The exit status: 0 if all calls ran, 2 on a rate limit, 3 on any other stop, 4 if a call's
        key source differs from the expected one.
    """
    if resume and STOP_PATH.exists():
        STOP_PATH.rename(STOP_PATH.with_name(f"stopped_before_resume_{int(time.time())}.json"))
    labeller = Labeller(limits=limits, arm=arm, resume=resume, expect_key_source=expect_key_source)
    labeller.cli_version = trial.cli_version()
    probe = Prompt(system=trial.PROBE_SYSTEM, text=trial.PROBE_TEXT, images=())
    probes = sum(1 for e in read_log() if e.get("tag") == "billing-probe")
    try:
        labeller.call(
            call_id=f"{arm}:billing_probe_{probes}", tag="billing-probe", offset=None, prompt=probe
        )
    except trial.RateLimited as error:
        print(f"RATE LIMIT on the billing probe: {error}", file=sys.stderr)
        return 2
    except BillingRouteError as error:
        print(f"ABORT: {error}; no check call made", file=sys.stderr)
        return 4
    except (BudgetExceeded, ModelError) as error:
        print(f"STOP on the billing probe: {error}", file=sys.stderr)
        return 3
    except Exception as error:  # noqa: BLE001 - any error must stop the run
        labeller.trip(reason=f"unexpected {type(error).__name__}: {error}")
        print(f"STOP on the billing probe, unexpected error: {error!r}", file=sys.stderr)
        return 3
    print(f"billing probe: apiKeySource = {labeller.api_key_source!r}", flush=True)
    if probe_only:
        print("probe only: no check call made")
        return 0
    if labeller.api_key_source != expect_key_source:
        print(
            f"ABORT: expected apiKeySource {expect_key_source!r}, the probe reports "
            f"{labeller.api_key_source!r}; no check call made",
            file=sys.stderr,
        )
        return 4
    earlier = resolved_from_log(
        entries=read_log(), candidates=candidates, arm=arm, version=labeller.version_hash
    )
    offsets = prompts.REAL_OFFSETS
    status = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(
                label_candidate,
                labeller=labeller,
                candidate=c,
                resolved=list(earlier.get(c.key, [])),
                inputs=inputs,
                register=register,
                offsets=offsets,
            )
            for c in candidates
        ]
        for future in futures:
            try:
                future.result()
            except trial.RateLimited as error:
                print(f"RATE LIMIT, stopping: {error}", file=sys.stderr)
                status = 2
            except BillingRouteError as error:
                print(f"ABORT: {error}", file=sys.stderr)
                status = status or 4
            except (BudgetExceeded, ModelError) as error:
                print(f"STOP: {error}", file=sys.stderr)
                status = status or 3
            except Exception as error:  # noqa: BLE001 - any error must stop the run
                print(f"STOP, unexpected error: {error!r}", file=sys.stderr)
                status = status or 3
    print(labeller.totals_line())
    if status:
        print(f"stopped early, exit status {status}; see {STOP_PATH}", file=sys.stderr)
    return status


# ---------------------------------------------------------------- arms and printed output


def arm_inputs(
    *,
    arm: str,
    candidates: list[prompts.Candidate],
    low: list[prompts.Candidate],
    era_table: pl.DataFrame,
) -> tuple[list[prompts.Candidate], tuple[prompts.RegisterRow, ...] | None]:
    """Return the candidates of an arm and, for the register arms, the register rows."""
    rows = asof.register_rows_from_era_table(era_table=era_table)
    windows = asof.watch_windows(rows=rows)
    if arm == "ecr":
        return asof.ecr_candidates(windows=windows, low_threshold=low), rows
    if arm == "llm-ecr":
        return asof.llm_ecr_candidates(union=candidates, windows=windows, low_threshold=low), rows
    return candidates, None


def ceiling_changes(
    *, candidates: Sequence[prompts.Candidate], era_table: pl.DataFrame, known_events: pl.DataFrame
) -> int:
    """Count the candidates whose `ceiling` class changes when `other_site` also counts."""
    kwargs = {"candidates": candidates, "era_table": era_table, "known_events": known_events}
    strict = asof.true_classes(**kwargs, use_other_site=False)
    wide = asof.true_classes(**kwargs, use_other_site=True)
    return sum(1 for key in wide if wide[key] != strict[key])


def build_arm_table(
    *,
    arm: str,
    candidates: list[prompts.Candidate],
    low: list[prompts.Candidate],
    bank: asof.DetectorBank,
    era_table: pl.DataFrame,
    known_events: pl.DataFrame,
) -> asof.AsOfTable:
    """Build the table of an arm that needs no call, or of an `llm` arm from the call log."""
    arm_candidates, _ = arm_inputs(arm=arm, candidates=candidates, low=low, era_table=era_table)
    n_sites = len(features.PRIMARIES)
    if arm in ("det", "ecr"):
        answers = asof.det_answers(candidates=arm_candidates)
    elif arm == "ceiling":
        truth = asof.true_classes(
            candidates=arm_candidates, era_table=era_table, known_events=known_events
        )
        answers = asof.ceiling_answers(candidates=arm_candidates, truth=truth)
    else:
        entries = read_log()
        if len(log_versions(entries=entries)) > 1:
            msg = "the call log holds more than one prompt version; move the old log aside"
            raise ValueError(msg)
        answers = resolved_from_log(entries=entries, candidates=arm_candidates, arm=arm)
    return asof.build_table(
        arm=arm, candidates=arm_candidates, answers=answers, bank=bank, n_sites=n_sites
    )


def print_example(*, table: asof.AsOfTable, candidates: list[prompts.Candidate]) -> None:
    """Print the as-of rows of one real candidate, for arm `det` only.

    The rows are those of the D8 candidate that alarms on 2025-04-26 (D8 is a substation, so its
    rows may be shown). No language-model column is printed.

    Args:
        table: The `det` table.
        candidates: The union candidates.
    """
    site = features.PRIMARIES.index(WORKED_EXAMPLE_SITE)
    target = features.day_index(day=WORKED_EXAMPLE_ALARM)
    mine = [c for c in candidates if c.site == site]
    candidate = min(mine, key=lambda c: abs(c.alarm - target))
    cols = table.columns
    print(
        f"\nWorked example, arm {table.arm}: primary={WORKED_EXAMPLE_SITE} "
        f"detector={candidate.detector} onset_est={features.day_of(index=candidate.onset)} "
        f"alarm={features.day_of(index=candidate.alarm)} "
        f"episode_end={features.day_of(index=candidate.end)} closed_by={candidate.ended_by}"
    )
    print(
        "entry day    state    era_id  onset_est    anchor   c1(morning) c2(midday) c3(evening)"
        "  withdrawn  first origin reading it"
    )
    shown: list[int] = []
    last: tuple[object, ...] | None = None
    for day in range(
        candidate.alarm - 3, min(candidate.alarm + 150, table.columns["state"].shape[1])
    ):
        key = (
            int(cols["state"][site, day]),
            int(cols["era_id"][site, day]),
            int(cols["onset_day"][site, day]),
            int(cols["withdrawn"][site, day]),
        )
        if key != last or day in (candidate.alarm - 1, candidate.alarm, candidate.alarm + 1):
            shown.append(day)
        last = key
    for day in shown:
        onset = int(cols["onset_day"][site, day])
        onset_text = "-" if onset < 0 else str(features.day_of(index=onset))
        origin = features.day_of(index=day + 1)
        print(
            f"{features.day_of(index=day)}   {STATE_NAMES[int(cols['state'][site, day])]:<7}  "
            f"{int(cols['era_id'][site, day]):>6}  {onset_text:<10}  "
            f"{cols['anchor'][site, day]:>7.3f}  {cols['c1'][site, day]:>10.2f} "
            f"{cols['c2'][site, day]:>10.2f} {cols['c3'][site, day]:>11.2f}  "
            f"{int(cols['withdrawn'][site, day]):>9}  {origin} 09:00 UTC"
        )


def reply_accounting(*, entries: Sequence[JsonObject]) -> JsonObject:
    """Count the checks by how they were answered: first reply, retry, or never.

    A retry is an attempt after the first one of the same check, and the retries are counted by
    their `retry_reason`. No reply text is read.

    Args:
        entries: The log entries.

    Returns:
        `first_reply_valid`, `retry_valid`, `never_valid` (two attempts, none valid), `unfinished`
        (no valid reply and fewer than two attempts), and `retries_by_reason`.
    """
    groups: dict[str, list[JsonObject]] = {}
    for entry in entries:
        if entry.get("tag") == "label":
            groups.setdefault(str(entry["call_id"]), []).append(entry)
    first = retry = never = unfinished = 0
    reasons: Counter[str] = Counter()
    for group in groups.values():
        ordered = sorted(group, key=lambda e: int(e.get("attempt", 0)))
        valid = [e for e in ordered if e.get("answer") is not None]
        if valid and int(valid[0].get("attempt", 0)) == 0:
            first += 1
        elif valid:
            retry += 1
        elif len(ordered) >= MAX_ATTEMPTS:
            never += 1
        else:
            unfinished += 1
        reasons.update(str(e.get("retry_reason") or "unrecorded") for e in ordered[1:])
    return {
        "first_reply_valid": first,
        "retry_valid": retry,
        "never_valid": never,
        "unfinished": unfinished,
        "retries_by_reason": dict(reasons),
    }


V2_MAX_REASON_WORDS: Final[int] = 40


def _reply_object(*, text: str) -> JsonObject | None:
    """Return the JSON object of a reply (any code fence removed), or `None` if there is none."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def reply_format_counts(*, entries: Sequence[JsonObject]) -> JsonObject:
    """Count the format of the label replies of an `llm` arm log, reading structure only.

    No label, confidence, or reason text is returned or printed. A reply is valid under the
    attempt 2 strict parser if it is bare JSON (no fence, no prose), has every key but
    `register_row` (which it may carry), satisfies the lenient parser, and has a `reason` of at
    most 40 words.

    Args:
        entries: The log entries.

    Returns:
        Counts of replies that are not bare JSON, carry a `register_row` key, have a `reason` over
        40 words, or fail the lenient parser, and of checks valid at the first reply, at the retry,
        or never under the attempt 2 strict parser.
    """
    counts = {"replies": 0, "not_bare_json": 0, "register_row_present": 0,
              "reason_over_40_words": 0, "fail_lenient_parser": 0}  # fmt: skip
    groups: dict[str, list[tuple[int, bool]]] = {}
    for entry in entries:
        if entry.get("tag") != "label" or entry.get("status") != "ok":
            continue
        text = str(entry.get("raw_output") or "")
        counts["replies"] += 1
        bare = text.strip().startswith("{") and text.strip().endswith("}")
        counts["not_bare_json"] += not bare
        obj = _reply_object(text=text)
        words_over = False
        if obj is not None:
            counts["register_row_present"] += "register_row" in obj
            reason = obj.get("reason")
            words_over = isinstance(reason, str) and len(reason.split()) > V2_MAX_REASON_WORDS
            counts["reason_over_40_words"] += words_over
        try:
            parse_reply(text=text)
            lenient = True
        except SchemaError:
            lenient = False
        counts["fail_lenient_parser"] += not lenient
        groups.setdefault(str(entry["call_id"]), []).append(
            (int(entry.get("attempt", 0)), bare and lenient and not words_over)
        )
    first = retry = never = 0
    for group in groups.values():
        valid = sorted(a for a, ok in group if ok)
        if valid and valid[0] == 0:
            first += 1
        elif valid:
            retry += 1
        else:
            never += 1
    return counts | {
        "v2_first_reply_valid": first,
        "v2_retry_valid": retry,
        "v2_never_valid": never,
    }


def print_summary(*, path: Path | None = None) -> None:
    """Print the totals of a call log, the plan-quota readings, and the count of invalid replies.

    No label, confidence, reason, or reply text is printed.
    """
    path = path or LOG_PATH
    entries = read_log(path=path)
    if not entries:
        print("no call log yet")
        return
    ok = [e for e in entries if e.get("status") == "ok"]
    valid, invalid, errors = schema_report(log_path=path)
    print(f"calls logged: {len(entries)}; ok: {len(ok)}; failed: {len(entries) - len(ok)}")
    print(f"replies that satisfy the schema: {valid}; that fail it: {invalid}")
    for message, count in errors.most_common():
        print(f"  {count} x {message}")
    accounting = reply_accounting(entries=entries)
    print(
        f"checks answered: valid at the first reply {accounting['first_reply_valid']}, "
        f"valid at the retry {accounting['retry_valid']}, never valid {accounting['never_valid']}, "
        f"unfinished {accounting['unfinished']}; "
        f"retries by reason {accounting['retries_by_reason']}"
    )
    if attempt_name() == "v2":
        print(f"reply format (structure only): {reply_format_counts(entries=entries)}")
    inputs = [int(e.get("input_tokens") or 0) for e in entries]
    outputs = [int(e.get("output_tokens") or 0) for e in entries]
    print(
        f"input tokens: {sum(inputs):,} (per call: mean {sum(inputs) / len(inputs):,.0f}, "
        f"max {max(inputs):,})"
    )
    print(
        f"output tokens: {sum(outputs):,} (per call: mean {sum(outputs) / len(outputs):,.0f}, "
        f"max {max(outputs):,})"
    )
    print(f"reported cost: US${sum(e.get('cost_usd') or 0.0 for e in entries):.3f}")
    waits = [float(e["wall_seconds"]) for e in entries if e.get("wall_seconds") is not None]
    if waits:
        print(f"wall time inside calls: {sum(waits):,.0f} s over {len(waits)} calls")
    starts = [e["started_utc"] for e in entries if e.get("started_utc")]
    ends = [e["finished_utc"] for e in entries if e.get("finished_utc")]
    if starts and ends:
        print(f"first call started {min(starts)}; last call finished {max(ends)}")
    kinds = Counter(split_call_id(call_id=str(e["call_id"]))[2] for e in ok if e["tag"] == "label")
    print(f"checks by kind: {dict(kinds)}")
    print(f"apiKeySource values seen: {sorted({str(e.get('apiKeySource')) for e in entries})}")
    totals = run_totals(entries=entries)
    for window in WINDOWS:
        print(
            f"{window} utilization (account-wide): "
            f"first {totals[f'{window}_utilization_first']!r}, "
            f"last {totals[f'{window}_utilization_last']!r}"
        )
    trial.print_plan_quota(entries=entries)
    print(
        "The utilization figures are account-wide, so every other session moves them. The trial "
        "saw a constant 0.32 over its 22 calls, so the figure may not move per call. Read /usage "
        "by hand before and after a run."
    )
    for record in sorted(RUN_DIR.glob("run_*.json")):
        print(f"  {record.name}: {json.loads(record.read_text())}")


def build_all_prompts(
    *,
    candidates: list[prompts.Candidate],
    inputs: prompts.Inputs,
    register: tuple[prompts.RegisterRow, ...] | None,
    arm: str,
) -> dict[str, Prompt]:
    """Build the prompt of every check that the schedule could ask for, and cut-test a sample.

    Args:
        candidates: The arm's candidates.
        inputs: The real record.
        register: The register rows of arm `llm-ecr`, or `None`.
        arm: `llm` or `llm-ecr`.

    Returns:
        The prompts by call id.
    """
    checks = [
        check
        for c in candidates
        for check in asof.all_possible_checks(
            candidate=c, offsets=prompts.REAL_OFFSETS, last_day=inputs.last_day
        )
        if check.day >= prompts.DAILY_DAYS - 1
    ]
    built: dict[str, Prompt] = {}
    for check in checks:
        built[check.call_id(arm=arm)] = prompts.prompt_of(
            inputs=inputs, check=check, register=register
        )
    sample = random.Random(CUT_SEED).sample(checks, k=min(CUT_SAMPLE, len(checks)))
    for check in sample:
        prompts.assert_cut_changes_nothing(
            inputs=inputs, check=check, prompt=built[check.call_id(arm=arm)], register=register
        )
    print(f"cut test passed on a seeded sample of {len(sample)} of {len(checks)} prompts")
    return built


def check_schedule_causal(*, candidates: Sequence[prompts.Candidate], last_day: int) -> int:
    """Check that no check on or before a day depends on a fact revealed after that day.

    For every candidate and every day from its alarm to 30 days after its closure, the checks
    that `all_possible_checks` lists on or before the day must be the same when the candidate
    is shown as it is known on that day.

    Args:
        candidates: The candidates.
        last_day: Last day index with data.

    Returns:
        The number of (candidate, day) pairs tested.

    Raises:
        ValueError: If a check depends on a later fact.
    """
    pairs = 0
    for c in candidates:
        stop = (c.closure_day or c.alarm + 60) + 30
        for day in range(c.alarm, min(stop, last_day) + 1):
            full = asof.all_possible_checks(
                candidate=c, offsets=prompts.REAL_OFFSETS, last_day=last_day
            )
            seen = asof.all_possible_checks(
                candidate=asof.visible_candidate(candidate=c, day=day),
                offsets=prompts.REAL_OFFSETS,
                last_day=last_day,
            )
            now = [(k.kind, k.day) for k in full if k.day <= day]
            then = [(k.kind, k.day) for k in seen if k.day <= day]
            if now != then:
                msg = f"a check of {c.key} on or before day {day} depends on a later fact"
                raise ValueError(msg)
            pairs += 1
    return pairs


def restrict_primaries(
    *, candidates: Sequence[prompts.Candidate], primaries: Sequence[str] | None
) -> list[prompts.Candidate]:
    """Keep the candidates of the named Primaries (all of them if none is named).

    Args:
        candidates: The candidates.
        primaries: Primary codes such as `D8`, or `None` for no restriction.

    Returns:
        The candidates of those Primaries.

    Raises:
        ValueError: If a name is not one of `features.PRIMARIES`.
    """
    if not primaries:
        return list(candidates)
    unknown = sorted(set(primaries) - set(features.PRIMARIES))
    if unknown:
        msg = f"not a Primary: {unknown}; the Primaries are {list(features.PRIMARIES)}"
        raise ValueError(msg)
    keep = {features.PRIMARIES.index(name) for name in primaries}
    return [c for c in candidates if c.site in keep]


def log_versions(*, entries: Sequence[JsonObject]) -> set[str | None]:
    """Return the prompt version hashes that appear in a call log."""
    return {e.get("prompt_version_sha256") for e in entries}


def move_log_aside() -> list[Path]:
    """Rename the stage 1 log, its stop file, and its call folder, so a new version starts clean.

    Returns:
        The new paths.
    """
    stamp = f"superseded_{int(time.time())}"
    moved = []
    for path in (LOG_PATH, STOP_PATH, RUN_DIR / "calls"):
        if path.exists():
            target = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            path.rename(target)
            moved.append(target)
    return moved


def wilson(*, k: int, n: int) -> tuple[float, float]:
    """Return the 95% Wilson score interval of k successes in n trials."""
    p = k / n
    z2 = WILSON_Z**2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n**2)) / (1 + z2 / n)
    return max(centre - half, 0.0), min(centre + half, 1.0)


def _known_text(*, known: asof.KnownAnswer, candidate: prompts.Candidate) -> str:
    """Describe a known answer with offsets only."""
    if known.kind == "none":
        return "no known answer"
    offset = candidate.onset - (known.reference_day or 0)
    if known.kind == "switch":
        return f"switch ({known.source}, via {known.via}, onset {offset:+d} d from the event)"
    return f"{known.kind} (onset {offset:+d} d from the era date)"


def _delay_text(*, a: int | None, b: int | None) -> str:
    """Return `b - a` in days, or `n/a`."""
    return "n/a" if a is None or b is None else f"{b - a:+d}"


def print_agreement(  # noqa: PLR0915 - one block per printed table
    *,
    arm: str,
    candidates: Sequence[prompts.Candidate],
    entries: Sequence[JsonObject],
    era_table: pl.DataFrame,
    known_events: pl.DataFrame,
    n_days: int,
) -> None:
    """Print how an arm's labels agree with the known answers, from the call log, with no call.

    Per-event rows give an ordinal and offsets, never a date (the one exception is the register
    date of D8, a substation row in a public register). A row of the confusion table is the known
    answer and a column is the state the check set.

    Args:
        arm: `llm` or `llm-ecr`.
        candidates: The arm's candidates (after any `--primary` restriction).
        entries: The call-log entries.
        era_table: `era_table.parquet`.
        known_events: `known_events.parquet`.
        n_days: Number of days of the grid.
    """
    resolved = resolved_from_log(entries=entries, candidates=candidates, arm=arm)
    known = asof.known_answers(
        candidates=candidates, era_table=era_table, known_events=known_events
    )
    names = sorted({features.PRIMARIES[c.site] for c in candidates})
    scope = f"{names[0]} only" if len(names) == 1 else ", ".join(names)
    ordered = sorted(candidates, key=lambda c: (c.site, c.alarm, c.episode_id))
    print(f"\nAgreement with the known answers, arm {arm}, scope: {scope}")
    print(f"{len(ordered)} candidates; {sum(1 for c in ordered if c.key in resolved)} have a check")
    print("ordinal  Primary detector onset-alarm  check            label      conf  state    known")
    cells: dict[tuple[str, str, str], int] = {}
    timelines: dict[str, np.ndarray] = {}
    for ordinal, c in enumerate(ordered, start=1):
        items = resolved.get(c.key, [])
        timelines[c.key] = asof.class_timeline(candidate=c, resolved=items, n_days=n_days)
        name = features.PRIMARIES[c.site]
        for item in items:
            state = asof.state_on_day(candidate=c, resolved=items, day=item.day)
            shown = f" alarm={features.day_of(index=c.alarm)}" if name == "D8" else ""
            print(
                f"{ordinal:>6}  {name:<7} {c.detector:<8} {c.onset - c.alarm:>+5d} d  "
                f"{item.kind:<8}{item.day - c.alarm:>+4d} d  {item.label or 'invalid':<10} "
                f"{item.confidence:>4.2f}  {STATE_NAMES[state]:<7}  "
                f"{_known_text(known=known[c.key], candidate=c)}{shown}"
            )
            column = (
                "invalid"
                if item.label is None
                else STATE_NAMES[asof.class_of(label=item.label, confidence=item.confidence)]
            )
            key = (item.kind, known[c.key].kind, column)
            cells[key] = cells.get(key, 0) + 1
    columns = ("none", "unknown", "era", "switch", "invalid")
    for kind in ("alarm", "plus7", "plus21", "closure"):
        print(f"\nConfusion table, check kind {kind} (scope: {scope}); rows known, columns state")
        print(f"{'known':<12}" + "".join(f"{c:>9}" for c in columns) + "      n")
        for truth in ("era", "switch", "unresolved", "none"):
            row = [cells.get((kind, truth, c), 0) for c in columns]
            n = sum(row)
            line = f"{truth if truth != 'none' else 'no answer':<12}" + "".join(
                f"{v:>9}" for v in row
            )
            line += f"  {n:>5}"
            hit = {"era": "era", "switch": "switch"}.get(truth)
            if hit is not None and n >= MIN_WILSON_N:
                low, high = wilson(k=cells.get((kind, truth, hit), 0), n=n)
                got = cells.get((kind, truth, hit), 0)
                line += f"   share set to {hit}: {got / n:.2f} (95% Wilson {low:.2f} to {high:.2f})"
            elif hit is not None and n:
                line += f"   n < {MIN_WILSON_N}: no interval"
            print(line)
    print("\nDelays for each known era, in days (offsets; D8 also gives its register date)")
    print(
        "ordinal  Primary  register->alarm  telemetry->alarm  register->first era  "
        "telemetry->first era  register->first origin  telemetry->first origin"
    )
    seen: set[tuple[int, int | None]] = set()
    ordinal = 0
    for c in ordered:
        k = known[c.key]
        if k.kind != "era" or (c.site, k.reference_day) in seen:
            continue
        seen.add((c.site, k.reference_day))
        ordinal += 1
        mine = [
            m
            for m in ordered
            if m.site == c.site
            and known[m.key].kind == "era"
            and known[m.key].reference_day == k.reference_day
        ]
        alarm = min(m.alarm for m in mine)
        eras = [
            int(np.flatnonzero(timelines[m.key] == asof.STATE_ERA)[0])
            for m in mine
            if (timelines[m.key] == asof.STATE_ERA).any()
        ]
        first_era = min(eras) if eras else None
        origin = first_era + 1 if first_era is not None else None
        name = features.PRIMARIES[c.site]
        date_text = (
            f"  register date {features.day_of(index=k.register_day)}"
            if name == "D8" and k.register_day is not None
            else ""
        )
        print(
            f"{ordinal:>6}  {name:<7}  {_delay_text(a=k.register_day, b=alarm):>15}  "
            f"{_delay_text(a=k.telemetry_day, b=alarm):>16}  "
            f"{_delay_text(a=k.register_day, b=first_era):>18}  "
            f"{_delay_text(a=k.telemetry_day, b=first_era):>20}  "
            f"{_delay_text(a=k.register_day, b=origin):>22}  "
            f"{_delay_text(a=k.telemetry_day, b=origin):>23}{date_text}"
        )


def reparse_command(
    *,
    candidates: Sequence[prompts.Candidate],
    rerun_bank: asof.DetectorBank,
    era_table: pl.DataFrame,
    known_events: pl.DataFrame,
) -> None:
    """Re-parse the stored log offline, build both signals, and compare the two parsers.

    The call log is only read. The re-parsed entries go to `REPARSED_LOG_PATH`; the table of the
    first parser is written as `llm_strict` and the table of the new parser as `llm`.

    Args:
        candidates: The `llm` arm's candidates.
        rerun_bank: The detector bank.
        era_table: `era_table.parquet`.
        known_events: `known_events.parquet`.
    """
    original = read_log()
    reparsed = reparse_log(entries=original)
    REPARSED_LOG_PATH.write_text("".join(json.dumps(e) + "\n" for e in reparsed))
    print(f"re-parsed {len(reparsed)} entries into {REPARSED_LOG_PATH}; the call log is unchanged")
    print_reparse_comparison(original=original, reparsed=reparsed)
    for name, entries in (("llm_strict", original), ("llm", reparsed)):
        answers = resolved_from_log(entries=entries, candidates=candidates, arm="llm")
        table = asof.build_table(
            arm=name,
            candidates=candidates,
            answers=answers,
            bank=rerun_bank,
            n_sites=len(features.PRIMARIES),
        )
        print(f"{name}: wrote {asof.write_table(table=table, folder=SIGNALS_DIR)}")
    for name, entries in (("first parser", original), ("lenient parser", reparsed)):
        print(f"\n===== {name} =====")
        print_agreement(
            arm="llm",
            candidates=candidates,
            entries=entries,
            era_table=era_table,
            known_events=known_events,
            n_days=features.N_DAYS,
        )


def build_probe_prompts(*, candidates: Sequence[prompts.Candidate], inputs: prompts.Inputs) -> int:
    """Build the probe Primary's prompts with the current builder into `PROBE_PROMPT_DIR`.

    The trial's own prompt folder and log are never read or written. Every prompt is cut-tested.

    Args:
        candidates: The union candidates.
        inputs: The real record.

    Returns:
        The number of prompts written.
    """
    mine = restrict_primaries(candidates=candidates, primaries=[PROBE_PRIMARY])
    built = build_all_prompts(candidates=mine, inputs=inputs, register=None, arm="llm")
    PROBE_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    for call_id, prompt in built.items():
        (PROBE_PROMPT_DIR / f"{call_id.replace(':', '_')}.txt").write_text(prompt.text)
    (PROBE_PROMPT_DIR / "system_prompt.txt").write_text(prompts.SYSTEM_PROMPT)
    (PROBE_PROMPT_DIR / "prompt_version_sha256.txt").write_text(prompt_version_hash())
    tokens = [estimate_input_tokens(prompt=p) for p in built.values()]
    print(
        f"{len(built)} {PROBE_PRIMARY} prompts written to {PROBE_PROMPT_DIR}; estimated input "
        f"tokens total {sum(tokens):,}, max {max(tokens):,}; no call made"
    )
    return len(built)


# ---------------------------------------------------------------- the command line


def _limits_from(*, args: argparse.Namespace) -> Limits | None:
    """Return the explicit limits, or `None` with a message if any is missing or too large."""
    if None in (args.max_usd, args.max_calls, args.max_input_tokens):
        print("--max-usd, --max-calls, and --max-input-tokens are all required", file=sys.stderr)
        return None
    maxima = (STAGE1_MAX_USD, STAGE1_MAX_CALLS, STAGE1_MAX_INPUT_TOKENS)
    if attempt_name() == "v2":
        maxima = (V2_MAX_USD, V2_MAX_CALLS, V2_MAX_INPUT_TOKENS)
    if args.max_usd > maxima[0] or args.max_calls > maxima[1] or args.max_input_tokens > maxima[2]:
        print(
            f"a limit is above the maximum of attempt {attempt_name()}: {maxima}", file=sys.stderr
        )
        return None
    return Limits(args.max_usd, args.max_calls, args.max_input_tokens)


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--list-candidates", action="store_true", help="print candidates (default)")
    parser.add_argument("--dry-run", action="store_true", help="build prompts, make no call")
    parser.add_argument("--arm", choices=("llm", "llm-ecr"), default="llm")
    parser.add_argument(
        "--attempt", choices=ATTEMPTS, default="v1", help="v2: the second D8-only attempt"
    )
    parser.add_argument("--build-arms", nargs="+", choices=asof.ARMS, help="write signals/<arm>")
    parser.add_argument("--example", action="store_true", help="print the det arm worked example")
    parser.add_argument("--pin-prompt", action="store_true", help="write the prompt pin (no call)")
    parser.add_argument("--schema-report", type=Path, nargs="?", const=TRIAL_LOG_PATH)
    parser.add_argument("--summary", action="store_true", help="summarise the stage 1 call log")
    parser.add_argument("--probe-only", action="store_true", help="send the billing probe only")
    parser.add_argument("--run-stage1", action="store_true", help="probe, then label (spends)")
    parser.add_argument("--go-token", help="the token the manager wrote to the go file")
    parser.add_argument("--expect-key-source", help="apiKeySource the probe must report")
    parser.add_argument("--max-usd", type=float)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--resume", action="store_true", help="continue from the existing log")
    parser.add_argument(
        "--new-version",
        action="store_true",
        help="move a log made with another prompt version aside and start a new one",
    )
    parser.add_argument(
        "--primary",
        action="append",
        choices=sorted(features.PRIMARIES),
        help="restrict the candidates, schedule, and calls to this Primary (repeatable)",
    )
    parser.add_argument(
        "--reparse", action="store_true", help="re-parse the stored replies with the lenient parser"
    )
    parser.add_argument(
        "--agreement", action="store_true", help="compare the log with the known answers"
    )
    parser.add_argument(
        "--build-probe-prompts", action="store_true", help="write the D10 probe prompts"
    )
    return parser


def _spend(  # noqa: PLR0911 - one return per refused gate
    *, args: argparse.Namespace, candidates: list[prompts.Candidate], low: list[prompts.Candidate]
) -> int:
    """Check every gate and, if all pass, run the probe and the labelling.

    Args:
        args: The parsed command line.
        candidates: The union candidates.
        low: The lower-threshold candidates (only arm `llm-ecr` reads them).

    Returns:
        The exit status.
    """
    if attempt_name() == "v1":
        print("refused: attempt 1 is closed; use --attempt v2", file=sys.stderr)
        return 1
    if args.primary != [V2_PRIMARY] or args.arm != "llm":
        print(f"refused: attempt 2 runs arm llm on --primary {V2_PRIMARY} only", file=sys.stderr)
        return 1
    if not go_token_valid(token=args.go_token):
        print("refused: --go-token is missing, does not match, or was used", file=sys.stderr)
        return 1
    if not pin_matches():
        print("refused: the prompt pin is missing or does not match the code", file=sys.stderr)
        return 1
    if not args.probe_only and not args.expect_key_source:
        print("refused: --expect-key-source is required for the labelling run", file=sys.stderr)
        return 1
    others = log_versions(entries=read_log()) - {prompt_version_hash()}
    if others and args.new_version:
        print(f"moved aside: {[str(p) for p in move_log_aside()]}")
    elif others and args.resume:
        print(
            "refused: the log holds entries of another prompt version; --resume would mix them. "
            "Pass --new-version to move the old log aside",
            file=sys.stderr,
        )
        return 1
    elif args.new_version:
        print("refused: --new-version needs a log of another prompt version", file=sys.stderr)
        return 1
    if (LOG_PATH.exists() or STOP_PATH.exists()) and not args.resume:
        print(f"refused: {LOG_PATH} or {STOP_PATH} exists; pass --resume", file=sys.stderr)
        return 1
    limits = _limits_from(args=args)
    if limits is None:
        return 1
    era_table = pl.read_parquet(asof.ERA_TABLE_PATH)
    arm_candidates, register = arm_inputs(
        arm=args.arm, candidates=candidates, low=low, era_table=era_table
    )
    inputs = prompts.load_inputs()
    claimed = claim_token(token=args.go_token, path=STAGE1_GO_FILE)
    if claimed is None:
        print("refused: the go token could not be claimed", file=sys.stderr)
        return 1
    print(f"go token claimed: {claimed.name}; a resume needs a new token")
    return run_stage1(
        candidates=arm_candidates,
        inputs=inputs,
        register=register,
        limits=limits,
        arm=args.arm,
        resume=args.resume,
        expect_key_source=args.expect_key_source,
        probe_only=args.probe_only,
    )


def main() -> int:  # noqa: PLR0911, PLR0912, PLR0915 - one branch per command
    """Run one command of the script; with no spending flag nothing is ever sent.

    Returns:
        The exit status.
    """
    args = _parser().parse_args()
    configure_attempt(name=args.attempt)
    if args.schema_report is not None:
        valid, invalid, errors = schema_report(log_path=args.schema_report)
        print(f"replies that satisfy the schema: {valid}; that fail it: {invalid}")
        for message, count in errors.most_common():
            print(f"  {count} x {message}")
        return 0
    if args.summary:
        print_summary()
        return 0
    if args.pin_prompt:
        blocker = probe_blocker()
        if blocker is not None:
            print(f"refused: {blocker}", file=sys.stderr)
            return 1
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        PIN_PATH.write_text(json.dumps({"prompt_version_sha256": prompt_version_hash()}, indent=2))
        print(f"prompt pin written: {prompt_version_hash()}")
        return 0
    everything = asof.load_stored_candidates()
    candidates = restrict_primaries(candidates=everything, primaries=args.primary)
    counts = Counter(features.PRIMARIES[c.site] for c in candidates)
    restricted = f" (restricted to {sorted(set(args.primary))})" if args.primary else ""
    print(f"{len(candidates)} union candidates{restricted}: {dict(sorted(counts.items()))}")
    if args.build_probe_prompts:
        build_probe_prompts(candidates=everything, inputs=prompts.load_inputs())
        return 0
    if args.reparse and attempt_name() == "v2":
        print("refused: attempt 2 is parsed live; there is nothing to re-parse", file=sys.stderr)
        return 1
    if args.reparse:
        bank, rerun = asof.build_detector_bank(stat3_sites=_register_sites())
        union, _ = _candidates_from_rerun(rerun=rerun)
        reparse_command(
            candidates=restrict_primaries(candidates=union, primaries=args.primary),
            rerun_bank=bank,
            era_table=pl.read_parquet(asof.ERA_TABLE_PATH),
            known_events=pl.read_parquet(asof.KNOWN_EVENTS_PATH),
        )
        return 0
    if args.agreement:
        low: list[prompts.Candidate] = []
        if args.arm == "llm-ecr":
            _, low = _low_candidates()
        era_table = pl.read_parquet(asof.ERA_TABLE_PATH)
        arm_candidates, _ = arm_inputs(
            arm=args.arm,
            candidates=candidates,
            low=restrict_primaries(candidates=low, primaries=args.primary),
            era_table=era_table,
        )
        print_agreement(
            arm=args.arm,
            candidates=arm_candidates,
            entries=read_log(),
            era_table=era_table,
            known_events=pl.read_parquet(asof.KNOWN_EVENTS_PATH),
            n_days=features.N_DAYS,
        )
        return 0
    spends = args.run_stage1 or args.probe_only
    if spends:
        if not args.run_stage1 and args.probe_only:
            print("refused: --probe-only needs --run-stage1 as well", file=sys.stderr)
            return 1
        low = []
        if args.arm == "llm-ecr":
            _, low = _low_candidates()
        low = restrict_primaries(candidates=low, primaries=args.primary)
        return _spend(args=args, candidates=candidates, low=low)
    if args.build_arms or args.example:
        arms = args.build_arms or ["det"]
        sites = _register_sites()
        bank, rerun = asof.build_detector_bank(stat3_sites=sites)
        asof.compare_with_stored(rerun=rerun)
        print("the re-run detectors reproduce the stored episode tables")
        union, low = _candidates_from_rerun(rerun=rerun)
        if [c.key for c in union] != [c.key for c in everything]:
            print("the re-run candidates differ from the stored ones", file=sys.stderr)
            return 1
        union = restrict_primaries(candidates=union, primaries=args.primary)
        low = restrict_primaries(candidates=low, primaries=args.primary)
        era_table = pl.read_parquet(asof.ERA_TABLE_PATH)
        known = pl.read_parquet(asof.KNOWN_EVENTS_PATH)
        for arm in arms:
            table = build_arm_table(
                arm=arm,
                candidates=union,
                low=low,
                bank=bank,
                era_table=era_table,
                known_events=known,
            )
            if arm == "ceiling":
                changed = ceiling_changes(candidates=union, era_table=era_table, known_events=known)
                print(f"ceiling: {changed} candidates change class when other_site counts")
            if args.build_arms:
                if attempt_name() == "v2" and arm in ("llm", "llm-ecr"):
                    table = replace(table, arm=f"{arm.replace('-', '_')}_v2")
                print(f"{arm}: wrote {asof.write_table(table=table, folder=SIGNALS_DIR)}")
            if arm == "det" and args.example:
                print_example(table=table, candidates=union)
        return 0
    inputs = prompts.load_inputs()
    pairs = check_schedule_causal(candidates=candidates, last_day=inputs.last_day)
    print(f"the check schedule is causal on {pairs} (candidate, day) pairs")
    era_table = pl.read_parquet(asof.ERA_TABLE_PATH)
    _, low = _low_candidates()
    low = restrict_primaries(candidates=low, primaries=args.primary)
    arm_candidates, register = arm_inputs(
        arm=args.arm, candidates=candidates, low=low, era_table=era_table
    )
    if not args.dry_run:
        print(f"arm {args.arm}: {len(arm_candidates)} candidates; --dry-run builds the prompts")
        return 0
    built = build_all_prompts(
        candidates=arm_candidates, inputs=inputs, register=register, arm=args.arm
    )
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    for call_id, prompt in built.items():
        (PROMPT_DIR / f"{call_id.replace(':', '_')}.txt").write_text(prompt.text)
    (PROMPT_DIR / "system_prompt.txt").write_text(prompts.SYSTEM_PROMPT)
    tokens = [estimate_input_tokens(prompt=p) for p in built.values()]
    print(
        f"{len(built)} prompts for {len(arm_candidates)} candidates; estimated input tokens: "
        f"total {sum(tokens):,}, max {max(tokens):,}; prompts written to {PROMPT_DIR}; no call made"
    )
    return 0


def _register_sites() -> list[int]:
    """Return the Primaries that have a register watch window."""
    rows = asof.register_rows_from_era_table(era_table=pl.read_parquet(asof.ERA_TABLE_PATH))
    return sorted({site for site, _, _ in asof.watch_windows(rows=rows)})


def _candidates_from_rerun(
    *, rerun: pl.DataFrame
) -> tuple[list[prompts.Candidate], list[prompts.Candidate]]:
    """Return the union and lower-threshold candidates from the re-run detector episodes."""
    parts = {
        name: rerun.filter(pl.col("detector") == name).select("site", "onset", "alarm", "end")
        for name in ("bocpd", "stat", "stat3")
    }
    return asof.candidates_from_tables(
        bocpd_rows=parts["bocpd"],
        stat_rows=parts["stat"],
        stat3_rows=parts["stat3"] if parts["stat3"].height else None,
    )


def _low_candidates() -> tuple[list[prompts.Candidate], list[prompts.Candidate]]:
    """Re-run the year-on-year detector at its lower threshold where a watch window needs it."""
    bank, rerun = asof.build_detector_bank(stat3_sites=_register_sites())
    del bank
    return _candidates_from_rerun(rerun=rerun)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    sys.exit(main())

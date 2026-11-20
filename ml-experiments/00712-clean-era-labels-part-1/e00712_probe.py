"""Experiment 712: the year-guessing probe of design section 2.3.

**The probe asks whether a check prompt reveals the calendar year.** The check prompts hold no date,
but a language model may still infer the year from weather, weekday, and holiday patterns. This
script sends 10 of the 20 check prompts of Primary D10 to the model, each with one appended
question that asks for the calendar year and month of the data. The plan's rule is that the
prompts pass if at most 3 of the 10 years are exactly right. **A probe reply is not a label and is
never used as one.**

**Nothing here spends unless the manager says so.** The default is a dry run, which builds the
prompts and prints their file names and token estimates. A call needs `--run-probe`, a `--go-token`
equal to the first line of `GO_FILE` (a file in the private data folder that the manager writes),
and an `--expect-key-source` that the billing probe must report before any probe call. The limits
are fixed: at most `MAX_CALLS` calls (the billing probe and retries included), `MAX_USD` of
reported cost, and `MAX_INPUT_TOKENS` input tokens. At most 2 calls run at a time, and any
rate-limit event stops the run at once. `--resume` continues from the log and never repeats a
finished call. `--score` reads the log and prints the result without a call.

**The true year of each prompt is read privately** from the stored candidates (the check day of
the candidate and the check kind named in the prompt file) and never appears in a prompt.
"""

import argparse
import json
import re
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import e00709_features as features  # ty: ignore[unresolved-import]
import e00712_asof as asof
import e00712_label as label
import e00712_prompt as prompts
import e00712_trial as trial
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
"""The experiment's folder in the private data store."""

PROBE_DIR: Final[Path] = OUT_DIR / "probe"
LOG_PATH: Final[Path] = PROBE_DIR / "probe_log.jsonl"
STOP_PATH: Final[Path] = PROBE_DIR / "stopped.json"
RESULT_PATH: Final[Path] = PROBE_DIR / "probe_result.json"
GO_FILE: Final[Path] = OUT_DIR / "probe_go_token.txt"
PROMPT_SOURCE_DIR: Final[Path] = label.PROBE_PROMPT_DIR
FILE_PREFIX: Final[str] = "llm_"
"""The arm prefix of the file names `e00712_label.py --build-probe-prompts` writes."""
PRIMARY: Final[str] = "D10"
MIN_TOKEN_CHARACTERS: Final[int] = 16

MAX_CALLS: Final[int] = 14
MAX_USD: Final[float] = 2.0
MAX_INPUT_TOKENS: Final[int] = 200_000
MAX_WORKERS: Final[int] = 2
MAX_ATTEMPTS: Final[int] = 2
N_PROMPTS: Final[int] = 10
PER_CANDIDATE: Final[int] = 2
MAX_EXACT_HITS: Final[int] = 3
"""The plan's rule: the prompts pass if at most this many of the 10 years are exactly right."""

CHECK_KINDS: Final[tuple[str, ...]] = ("alarm", "plus7", "plus21", "closure")
MIN_YEAR: Final[int] = 2018
MAX_YEAR: Final[int] = 2026

PROBE_SYSTEM_PROMPT: Final[str] = """\
You are an analyst of electricity-demand and solar-generation time series. You are given tables \
describing one electricity substation in Great Britain. Answer only with the JSON object that the \
user asks for, with no other text."""

PROBE_QUESTION: Final[str] = """\

---
NEW QUESTION, replacing the task above. Do not label any change. The data above end on one day \
somewhere between 2018 and 2026. From everything the tables show (weekday patterns, \
holiday dips, the season, the weather columns, and anything else), estimate the calendar YEAR and \
MONTH in which the last day of the data fell. Guess even if you are unsure. Reply with exactly \
one JSON object and nothing else:
{"year": <integer>, "month": <integer from 1 to 12>, "confidence": <number from 0 to 1>}"""

ANSWER_KEYS: Final[frozenset[str]] = frozenset({"year", "month", "confidence"})
FENCE: Final[re.Pattern[str]] = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class SchemaError(ValueError):
    """A reply does not satisfy the strict answer schema."""


@dataclass(frozen=True)
class Guess:
    """One parsed year guess.

    Attributes:
        year: The guessed calendar year.
        month: The guessed month, 1 to 12.
        confidence: The model's stated confidence, 0 to 1.
    """

    year: int
    month: int
    confidence: float

    def as_json(self) -> JsonObject:
        """Return the guess as a JSON object."""
        return {"year": self.year, "month": self.month, "confidence": self.confidence}


# ---------------------------------------------------------------- selection and prompts


def easter_sunday(*, year: int) -> date:
    """Return Easter Sunday of a year (the anonymous Gregorian computus)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    j, k = c // 4, c % 4
    m = (a + 11 * h) // 319
    r = (2 * e + 2 * j - k - h + m + 32) % 7
    month, day = divmod(h - m + r + 90, 25)
    return date(year, month, day + 1)


def _last_monday(*, year: int, month: int) -> date:
    """Return the last Monday of a month."""
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=last.weekday())


def holiday_anchors(*, year: int) -> list[date]:
    """Return the dates of a year around which bank holidays fall (Christmas to August)."""
    easter = easter_sunday(year=year)
    return [
        date(year, 12, 25),
        date(year, 1, 1),
        easter,
        easter - timedelta(days=2),
        _last_monday(year=year, month=5),
        _last_monday(year=year, month=8),
    ]


def distance_to_holiday(*, day: date) -> int:
    """Return the days from a date to the nearest holiday anchor of a nearby year."""
    return min(
        abs((day - anchor).days)
        for year in (day.year - 1, day.year, day.year + 1)
        for anchor in holiday_anchors(year=year)
    )


def check_days(*, candidate: prompts.Candidate) -> dict[str, int]:
    """Return the check day index of each check kind of a candidate."""
    offsets = dict(zip(CHECK_KINDS[:3], trial.SCHEDULED_OFFSETS, strict=True))
    days = {kind: candidate.alarm + offset for kind, offset in offsets.items()}
    days["closure"] = candidate.end + 1
    return days


def select_prompts(*, candidates: Sequence[prompts.Candidate]) -> list[str]:
    """Choose the probe prompts: for each D10 candidate, the checks nearest a holiday.

    The selection is deterministic. Each candidate contributes its `PER_CANDIDATE` checks whose
    check day lies closest to Christmas, New Year, Good Friday, Easter Sunday, or the May or August
    bank holiday (ties go to the earlier kind), so the 10 prompts spread across all 5 candidates
    and oversample holiday-bearing weeks.

    Args:
        candidates: The stored union candidates.

    Returns:
        The prompt file stems (`llm_D10_<episode>_<kind>`) of the files that exist, sorted.
    """
    chosen: list[str] = []
    for candidate in candidates:
        if features.PRIMARIES[candidate.site] != PRIMARY:
            continue
        scored = sorted(
            (
                distance_to_holiday(day=features.day_of(index=index)),
                CHECK_KINDS.index(kind),
                kind,
            )
            for kind, index in check_days(candidate=candidate).items()
            if (
                PROMPT_SOURCE_DIR / f"{FILE_PREFIX}{PRIMARY}_{candidate.episode_id}_{kind}.txt"
            ).exists()
        )
        chosen.extend(
            f"{FILE_PREFIX}{PRIMARY}_{candidate.episode_id}_{kind}"
            for *_, kind in scored[:PER_CANDIDATE]
        )
    return sorted(chosen)


def true_years(*, candidates: Sequence[prompts.Candidate]) -> dict[str, int]:
    """Return the true calendar year of every D10 check prompt, keyed by prompt stem (private)."""
    out: dict[str, int] = {}
    for candidate in candidates:
        if features.PRIMARIES[candidate.site] != PRIMARY:
            continue
        for kind, index in check_days(candidate=candidate).items():
            out[f"{FILE_PREFIX}{PRIMARY}_{candidate.episode_id}_{kind}"] = features.day_of(
                index=index
            ).year
    return out


def probe_prompt(*, stem: str) -> Prompt:
    """Build the probe prompt of one stored check prompt: its text, then the fixed question.

    Args:
        stem: The prompt file stem.

    Returns:
        The prompt, with no image.

    Raises:
        ValueError: If the stored text holds a Primary name or a year.
    """
    text = (PROMPT_SOURCE_DIR / f"{stem}.txt").read_text()
    if re.search(r"\bD\d+\b", text) or re.search(r"\b20[12]\d\b", text):
        msg = f"the stored prompt {stem} holds a Primary name or a year"
        raise ValueError(msg)
    return Prompt(system=PROBE_SYSTEM_PROMPT, text=text + PROBE_QUESTION, images=())


# ---------------------------------------------------------------- the reply


def parse_reply(*, text: str) -> Guess:
    """Parse a reply strictly: exactly the keys `year`, `month`, and `confidence`.

    Args:
        text: The reply, optionally inside a code fence.

    Returns:
        The guess.

    Raises:
        SchemaError: If the reply is not exactly that object. The message names the field, never
            the value.
    """
    body = text.strip()
    fenced = FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as error:
        msg = "the reply is not JSON"
        raise SchemaError(msg) from error
    if not isinstance(obj, dict) or set(obj) != ANSWER_KEYS:
        msg = "the reply is not an object with exactly year, month, and confidence"
        raise SchemaError(msg)
    year, month, confidence = obj["year"], obj["month"], obj["confidence"]
    if (
        isinstance(year, bool)
        or not isinstance(year, int)
        or not MIN_YEAR - 10 <= year <= MAX_YEAR + 10
    ):
        msg = "year is not a plausible integer"
        raise SchemaError(msg)
    if isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12:
        msg = "month is not an integer from 1 to 12"
        raise SchemaError(msg)
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        msg = "confidence is not a number"
        raise SchemaError(msg)
    if not 0.0 <= confidence <= 1.0:
        msg = "confidence is outside 0 to 1"
        raise SchemaError(msg)
    return Guess(year=year, month=month, confidence=float(confidence))


def seven_day_utilisation(*, entries: Sequence[JsonObject], last: bool) -> object:
    """Return the account-wide seven-day utilisation from the logged rate-limit events.

    Args:
        entries: The log entries, in order.
        last: Whether to return the latest reading (`True`) or the earliest (`False`).

    Returns:
        `unifiedWindows.seven_day.utilization` of the first or last event that carries it, or
        `None` if no event does.
    """
    readings: list[object] = []
    for entry in entries:
        for payload in entry.get("rate_limit_events") or []:
            info = payload.get("rate_limit_info")
            windows = info.get("unifiedWindows") if isinstance(info, dict) else None
            window = windows.get("seven_day") if isinstance(windows, dict) else None
            if isinstance(window, dict) and "utilization" in window:
                readings.append(window["utilization"])
    if not readings:
        return None
    return readings[-1] if last else readings[0]


# ---------------------------------------------------------------- the gates and the log


def go_token_valid(*, token: str | None) -> bool:
    """Return True if the token equals the first line of the go file the manager wrote."""
    if not token or len(token) < MIN_TOKEN_CHARACTERS or not GO_FILE.exists():
        return False
    lines = GO_FILE.read_text().splitlines()
    return bool(lines) and lines[0].strip() == token


def read_log() -> list[JsonObject]:
    """Return every entry of the probe log, or an empty list if there is none."""
    if not LOG_PATH.exists():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]


class Prober:
    """Runs the probe calls with the budget guard, the call log, and the stop flag.

    A stop cannot interrupt a call that the other worker already has in flight; it only prevents
    every later call.
    """

    def __init__(self, *, resume: bool) -> None:
        """Create the guard, the log, and the stop flag.

        Args:
            resume: Whether to continue from the existing log, whose totals the guard then counts.
        """
        PROBE_DIR.mkdir(parents=True, exist_ok=True)
        self.store = CallStore(directory=PROBE_DIR / "calls", log_path=LOG_PATH)
        self.guard = BudgetGuard.from_log(log_path=LOG_PATH) if resume else BudgetGuard()
        self.guard.ceiling_usd = MAX_USD
        self.guard.max_calls = MAX_CALLS
        self.guard.max_input_tokens = MAX_INPUT_TOKENS
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.inflight = 0
        earlier = read_log() if resume else []
        self.attempts: dict[str, int] = {}
        self.finished: set[str] = set()
        for entry in earlier:
            call_id = str(entry["call_id"])
            self.attempts[call_id] = self.attempts.get(call_id, 0) + 1
            if entry.get("status") == "ok" and (
                entry["tag"] == "billing-probe" or entry.get("answer") is not None
            ):
                self.finished.add(call_id)
        self.output_tokens = sum(int(e["output_tokens"]) for e in earlier)
        sources = [e["apiKeySource"] for e in earlier if e.get("apiKeySource") is not None]
        self.api_key_source = str(sources[-1]) if sources else "unknown"
        self.cli_version = ""

    def totals_line(self) -> str:
        """One line of running totals."""
        g = self.guard
        return (
            f"running totals: calls={g.calls} input_tokens={g.input_tokens} "
            f"output_tokens={self.output_tokens} reported_cost_usd={g.spent_usd:.3f}"
        )

    def _trip(self, *, reason: str) -> None:
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
        """Append one entry to the log under the lock."""
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
            self.inflight -= 1
            self.output_tokens += reply.output_tokens
            if entry.get("apiKeySource") is not None:
                self.api_key_source = str(entry["apiKeySource"])

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

    def call(self, *, call_id: str, tag: str, prompt: Prompt) -> Guess | None:
        """Make one call, with one retry on an invalid reply, and log every attempt.

        Any error sets the stop flag before it propagates, so the other worker takes no new call.

        Args:
            call_id: The prompt stem, or `billing_probe`.
            tag: `billing-probe` or `probe`.
            prompt: The prompt.

        Returns:
            The guess, or `None` for the billing probe, for a finished call, and for a call whose
            attempts all failed the schema.

        Raises:
            RateLimited: On a rate-limit message.
            BudgetExceeded: If a limit would be passed.
            ModelError: If the call fails or times out.
        """
        if call_id in self.finished:
            return None
        for attempt in range(self.attempts.get(call_id, 0), MAX_ATTEMPTS):
            if self.stop.is_set():
                return None
            entry: JsonObject = {
                "call_id": call_id,
                "attempt": attempt,
                "tag": tag,
                "model": trial.MODEL_ID,
                "claude_code_version": self.cli_version,
                "system_prompt": prompt.system,
                "prompt_sha256": prompts.sha256(text=prompt.text),
                "estimated_input_tokens": estimate_input_tokens(prompt=prompt),
            }
            try:
                self._reserve(prompt=prompt)
                run = trial.run_claude(prompt=prompt)
            except BudgetExceeded as error:
                self._trip(reason=f"budget: {error}")
                raise
            except ModelError as error:
                self._finish(entry=entry, reply=self._cap_cost(prompt=prompt), events=[])
                entry |= {"status": "failed", "error": str(error)[:500], "wall_seconds": None}
                self._log(entry=entry)
                self._trip(reason=f"timeout or launch failure: {error}")
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
                self._trip(reason=limit or f"model error: {error}")
                if limit:
                    raise trial.RateLimited(limit) from error
                raise
            self._finish(entry=entry, reply=reply, events=events)
            entry |= {"status": "ok", "raw_output": reply.text}
            if limit:
                self._log(entry=entry)
                self._trip(reason=limit)
                raise trial.RateLimited(limit)
            if tag == "billing-probe":
                entry["answer"] = None
                self._log(entry=entry)
                return None
            try:
                guess = parse_reply(text=reply.text)
            except SchemaError as error:
                entry |= {"answer": None, "parse_error": str(error)}
                self._log(entry=entry)
                continue
            entry["answer"] = guess.as_json()
            self._log(entry=entry)
            return guess
        return None


def run_probe(*, stems: Sequence[str], resume: bool, expect_key_source: str) -> int:
    """Send the billing probe, check its route, then the probe prompts, 2 at a time at most.

    Args:
        stems: The prompt stems to send.
        resume: Whether to continue from the existing log.
        expect_key_source: The `apiKeySource` the billing probe must report before any probe call.

    Returns:
        The exit status: 0 if all calls ran, 2 on a rate limit, 3 on any other stop, 4 if the
        billing probe's key source differs from the expected one.
    """
    if resume and STOP_PATH.exists():
        STOP_PATH.rename(STOP_PATH.with_name(f"stopped_before_resume_{int(time.time())}.json"))
    prober = Prober(resume=resume)
    prober.cli_version = trial.cli_version()
    billing = Prompt(system=trial.PROBE_SYSTEM, text=trial.PROBE_TEXT, images=())
    try:
        prober.call(call_id="billing_probe", tag="billing-probe", prompt=billing)
    except trial.RateLimited as error:
        print(f"RATE LIMIT on the billing probe: {error}", file=sys.stderr)
        return 2
    except (BudgetExceeded, ModelError) as error:
        print(f"STOP on the billing probe: {error}", file=sys.stderr)
        return 3
    print(f"billing probe: apiKeySource = {prober.api_key_source!r}", flush=True)
    before = seven_day_utilisation(entries=read_log(), last=False)
    print(f"account-wide seven-day utilisation before the probe calls: {before!r}", flush=True)
    if prober.api_key_source != expect_key_source:
        print(
            f"ABORT: expected apiKeySource {expect_key_source!r}, the billing probe reports "
            f"{prober.api_key_source!r}; no probe call made",
            file=sys.stderr,
        )
        return 4
    status = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(prober.call, call_id=s, tag="probe", prompt=probe_prompt(stem=s))
            for s in stems
        ]
        for future in futures:
            try:
                future.result()
            except trial.RateLimited as error:
                print(f"RATE LIMIT, stopping: {error}", file=sys.stderr)
                status = 2
            except (BudgetExceeded, ModelError) as error:
                print(f"STOP: {error}", file=sys.stderr)
                status = status or 3
    print(prober.totals_line())
    after = seven_day_utilisation(entries=read_log(), last=True)
    print(f"account-wide seven-day utilisation after the probe calls: {after!r}")
    if status:
        print(f"stopped early, exit status {status}; see {STOP_PATH}", file=sys.stderr)
    return status


# ---------------------------------------------------------------- scoring and output


def score(
    *, entries: Sequence[JsonObject], truth: dict[str, int], stems: Sequence[str]
) -> JsonObject:
    """Score the logged guesses against the true years.

    Args:
        entries: The log entries.
        truth: The true year of each prompt stem.
        stems: The selected stems.

    Returns:
        A dict with the number answered, the exact-year hits, the within-1-year hits, whether all
        prompts were answered, and `passed` (`None` if the probe is incomplete).
    """
    guessed: dict[str, int] = {}
    for entry in entries:
        answer = entry.get("answer")
        if entry.get("tag") == "probe" and entry.get("status") == "ok" and answer is not None:
            guessed[str(entry["call_id"])] = int(answer["year"])
    answered = [s for s in stems if s in guessed]
    exact = sum(guessed[s] == truth[s] for s in answered)
    within_one = sum(abs(guessed[s] - truth[s]) <= 1 for s in answered)
    complete = len(answered) == len(stems)
    return {
        "prompt_version_sha256": label.prompt_version_hash(),
        "n_selected": len(stems),
        "n_answered": len(answered),
        "exact_year_hits": exact,
        "within_one_year_hits": within_one,
        "complete": complete,
        "passed": (exact <= MAX_EXACT_HITS) if complete else None,
    }


def print_score(*, result: JsonObject) -> None:
    """Print the score and the verdict against the plan's rule."""
    n = result["n_selected"]
    print(
        f"answered {result['n_answered']} of {n}; exact year {result['exact_year_hits']} of {n}; "
        f"within 1 year {result['within_one_year_hits']} of {n}"
    )
    if result["passed"] is None:
        print(f"INCOMPLETE: the rule needs all {n} answers; no verdict")
    else:
        verdict = "PASS" if result["passed"] else "FAIL"
        print(f"{verdict}: the rule passes at most {MAX_EXACT_HITS} of {n} exactly correct years")


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="build prompts, no call (default)")
    parser.add_argument("--score", action="store_true", help="score the stored log, no call")
    parser.add_argument("--run-probe", action="store_true", help="make the calls (spends)")
    parser.add_argument("--go-token", help="the token the manager wrote to the go file")
    parser.add_argument("--expect-key-source", help="apiKeySource the billing probe must report")
    parser.add_argument("--resume", action="store_true", help="continue from the existing log")
    return parser


def main() -> int:  # noqa: PLR0911 - one exit status per refusal
    """Run one command of the script; with no spending flag nothing is ever sent.

    Returns:
        The exit status.
    """
    args = _parser().parse_args()
    candidates = asof.load_stored_candidates()
    stems = select_prompts(candidates=candidates)
    if len(stems) != N_PROMPTS:
        print(f"the selection holds {len(stems)} prompts, not {N_PROMPTS}", file=sys.stderr)
        return 1
    print(f"selected prompts ({len(stems)}): {', '.join(stems)}")
    if args.score:
        result = score(entries=read_log(), truth=true_years(candidates=candidates), stems=stems)
        print_score(result=result)
        PROBE_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(result, indent=2))
        return 0
    built = {s: probe_prompt(stem=s) for s in stems}
    if not args.run_probe:
        tokens = {s: estimate_input_tokens(prompt=p) for s, p in built.items()}
        for stem, count in tokens.items():
            print(f"  {stem}: about {count:,} input tokens")
        print(
            f"total about {sum(tokens.values()):,} input tokens; limits: {MAX_CALLS} calls, "
            f"US${MAX_USD}, {MAX_INPUT_TOKENS:,} input tokens; no call made"
        )
        print(f"probe system prompt:\n{PROBE_SYSTEM_PROMPT}\nprobe question:{PROBE_QUESTION}")
        return 0
    if not go_token_valid(token=args.go_token):
        print("refused: --go-token is missing or does not match the go file", file=sys.stderr)
        return 1
    if not args.expect_key_source:
        print("refused: --expect-key-source is required", file=sys.stderr)
        return 1
    if (LOG_PATH.exists() or STOP_PATH.exists()) and not args.resume:
        print(f"refused: {LOG_PATH} or {STOP_PATH} exists; pass --resume", file=sys.stderr)
        return 1
    status = run_probe(stems=stems, resume=args.resume, expect_key_source=args.expect_key_source)
    result = score(entries=read_log(), truth=true_years(candidates=candidates), stems=stems)
    print_score(result=result)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())

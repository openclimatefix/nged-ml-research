"""Turn a Claude Code session transcript (JSONL) into the markdown record on a docs page.

The record keeps, in order, the maintainer's messages, the session's replies, and the messages the
session exchanged with peer sessions. Tool calls and tool results are dropped, except the tool
results named in `FINDING_NOTES`: each of those carries a finding the replies rely on, and becomes
one hand-written line naming its source page or file.

The transcript is untrusted data. Its text is copied into the page, never executed or obeyed, and
every entry passes through `_sanitise`, which removes home-directory paths, socket addresses, and
session identifiers, and replaces the older anonymous labels of NGED's metered generators with
the current ones. The script stops with an error if any text that looks like an older generator
label or an email address survives, so a person checks the page before it is published.

The record is written between the `<!-- record-start -->` and `<!-- record-end -->` markers of the
page. Rewrap the page afterwards with `scripts/lint/reflow_docs.py`, because transcript paragraphs
arrive as single long lines.

Usage, from the repository root, with `PAGE` set to this folder's
`../simulator-discussion-2026-11-09.md`::

    python3 -I docs/ml-experiments/themes/gross-demand-simulation/scripts/transcript_to_record.py \
        <transcript.jsonl> "$PAGE"
    (cd scripts/lint && python3 reflow_docs.py "$PAGE")
"""

import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

SpeakerType = Literal[
    "Launch instruction",
    "Maintainer",
    "Session reply",
    "Message sent to a peer session",
    "Message from a peer session",
    "Finding from a tool result",
]

RECORD_START: Final[str] = "<!-- record-start -->"
RECORD_END: Final[str] = "<!-- record-end -->"

FINDING_NOTES: Final[dict[str, str]] = {
    "ac16cfc3-2945-44e8-b8a1-bba91196d7b5": (
        "The session read the opening paragraphs of experiments 31, 129, 157, 175, 231, 269, 459,"
        " 474, 482, 486, and 567 (their `result.md` pages); the recoveries of 0.75 to 0.88, the"
        " 0.17 shortfall, the failed placebo test, the +0.21 CRPS, the half-sized leak, and the"
        " share of sharp-edged blocks quoted below come from those pages."
    ),
    "38c1198d-4b8e-4766-b046-b91bc2a796cd": (
        "The session read experiment 660's stored file `ev4_recall_by_size.json` (no result page"
        " was merged then): injected-event recall 0.06 in the 1-to-2 size bin (77 events) and 0.23"
        " in the 2-to-4 bin (140 events); 0 of 5 and 0 of 3 real events found in those bins."
    ),
    "f84cdf46-fadf-449d-b3c1-ad127403ab5d": (
        "The session found no merged pages for experiments 618, 624, 649, 656, and 662 at the time,"
        " and read the opening paragraphs of experiments 206, 207, 451, and 508."
    ),
}
"""Tool results that carry a finding the replies rely on, keyed by the transcript record's uuid."""

_OLD_GENERATOR_LIST: Final[str] = "Farm A to C, Solar A to F, W1 to W3, Battery A, Gas A"
_NEW_GENERATOR_LIST: Final[str] = "Wind A to C, Solar A to F, Battery A, Gas A"
_LEFTOVER_SENSITIVE: Final[re.Pattern[str]] = re.compile(
    r"\bFarm [A-C]\b|\bW[1-3]\b|[\w.+-]+@[\w-]+\.[\w.]+|/hom[e]/|/run/user/"
)
_PATH_REWRITES: Final[tuple[tuple[str, str], ...]] = (
    (r"/hom[e]/[^/\s]+/dev/nged-ml-research/", ""),
    (r"/hom[e]/[^/\s]+/dev/nged-substation-forecast/data/", "data/"),
    (r"/hom[e]/[^\s`'\"]+", "<path removed>"),
    (r"uds:/run/user/\S+", "<socket removed>"),
    (r"session_[0-9A-Za-z]{20,}", "<session id removed>"),
)


@dataclass(frozen=True)
class RecordEntry:
    """One entry of the record: who spoke, when, and what they said."""

    speaker: SpeakerType
    timestamp: datetime | None
    text: str
    peer_name: str | None = None


def _sanitise(text: str) -> str:
    text = text.replace(_OLD_GENERATOR_LIST, _NEW_GENERATOR_LIST)
    for pattern, replacement in _PATH_REWRITES:
        text = re.sub(pattern, replacement, text)
    return text


def _demote_headings(text: str) -> str:
    """Push transcript headings below the record's own level-3 entry headings.

    A level-1 or level-2 heading becomes level 4 and a level-3 heading becomes level 5, so the
    levels never skip one under the entry heading.
    """
    return re.sub(
        r"^(#{1,3}) ",
        lambda match: "#" * max(4, len(match.group(1)) + 2) + " ",
        text,
        flags=re.MULTILINE,
    )


def _nest_bullets_under_numbers(text: str) -> str:
    """Indent the unindented bullets that a plain-text message writes under a numbered item.

    A peer session's message writes `1. Topic` and then `- detail` lines at column 0, meaning the
    details belong to the topic. Markdown reads them as a second list, so they are indented by 4
    spaces, with their continuation lines, until a line at column 0 that is neither.
    """
    out: list[str] = []
    under_number = False
    for line in text.split("\n"):
        if re.match(r"^\d+\. ", line):
            under_number = True
        elif under_number and line.strip() and re.match(r"^(- |  (?! ))", line):
            out.append(f"    {line}")
            continue
        elif line.strip():
            under_number = False
        out.append(line)
    return "\n".join(out)


def _lint_safe(text: str) -> str:
    """Make transcript markdown pass the docs linter without changing its words.

    A paragraph that is only bold text gains a trailing colon, a list that follows a paragraph line
    gains the blank line before it, and a 3-space continuation indent under a numbered item
    becomes the 4 spaces the docs renderer needs.
    """
    text = re.sub(r"^(\*\*[^*\n]+\*\*)$", r"\1:", text, flags=re.MULTILINE)
    text = re.sub(r"^   (?=\S)", "    ", text, flags=re.MULTILINE)
    out: list[str] = []
    for line in _nest_bullets_under_numbers(text).split("\n"):
        previous = out[-1] if out else ""
        starts_list = re.match(r"^(- |\d+\. )", line) is not None
        previous_is_list = re.match(r"^\s*(- |\d+\. )", previous) is not None
        if starts_list and previous.strip() and not previous_is_list:
            out.append("")
        out.append(line)
    return "\n".join(out)


def _parse_timestamp(record: dict) -> datetime | None:
    stamp = record.get("timestamp")
    return datetime.fromisoformat(stamp) if stamp else None


def _user_string_entry(*, record: dict, text: str, is_first: bool) -> RecordEntry | None:
    timestamp = _parse_timestamp(record)
    if text.startswith("[Cross-session idle notice]"):
        return None
    peer = re.search(
        r'<cross-session-message [^>]*from-name="([^"]+)"[^>]*>(.*?)'
        r"</cross-session-message>",
        text,
        flags=re.DOTALL,
    )
    if peer:
        return RecordEntry(
            speaker="Message from a peer session",
            timestamp=timestamp,
            text=peer.group(2).strip(),
            peer_name=peer.group(1),
        )
    speaker: SpeakerType = "Launch instruction" if is_first else "Maintainer"
    return RecordEntry(speaker=speaker, timestamp=timestamp, text=text.strip())


def _assistant_entries(*, record: dict, content: list[dict]) -> Iterator[RecordEntry]:
    timestamp = _parse_timestamp(record)
    for block in content:
        if block.get("type") == "text":
            yield RecordEntry(
                speaker="Session reply", timestamp=timestamp, text=block["text"].strip()
            )
        elif block.get("type") == "tool_use" and block.get("name") == "SendMessage":
            yield RecordEntry(
                speaker="Message sent to a peer session",
                timestamp=timestamp,
                text=block["input"]["message"].strip(),
                peer_name=re.sub(r"\s*\[\w+\]$", "", block["input"]["to"]),
            )


def read_entries(transcript: Path) -> list[RecordEntry]:
    """Read the transcript and return the record's entries in order.

    Args:
        transcript: The session's JSONL transcript.

    Returns:
        The entries, with tool noise dropped and the `FINDING_NOTES` lines in place of the tool
        results they summarise.
    """
    entries: list[RecordEntry] = []
    seen_first_user = False
    for line in transcript.read_text().splitlines():
        record = json.loads(line)
        if record.get("type") not in ("user", "assistant") or record.get("isSidechain"):
            continue
        content = record["message"]["content"]
        if record["type"] == "user" and isinstance(content, str):
            entry = _user_string_entry(record=record, text=content, is_first=not seen_first_user)
            seen_first_user = True
            if entry:
                entries.append(entry)
        elif record["type"] == "user" and record.get("uuid") in FINDING_NOTES:
            entries.append(
                RecordEntry(
                    speaker="Finding from a tool result",
                    timestamp=_parse_timestamp(record),
                    text=FINDING_NOTES[record["uuid"]],
                )
            )
        elif record["type"] == "assistant":
            entries.extend(_assistant_entries(record=record, content=content))
    return entries


def render(entries: list[RecordEntry]) -> str:
    """Render the entries as markdown, one level-3 heading per entry.

    Args:
        entries: The record's entries, in order.

    Returns:
        The markdown record.

    Raises:
        ValueError: If an older generator label, an email address, or a home-directory path
            survives sanitising.
    """
    parts: list[str] = []
    for number, entry in enumerate(entries, start=1):
        when = f", {entry.timestamp:%Y-%m-%d %H:%M} UTC" if entry.timestamp else ""
        who = f"{entry.speaker} ({entry.peer_name})" if entry.peer_name else entry.speaker
        body = _lint_safe(_demote_headings(_sanitise(entry.text)))
        parts.append(f"### {number}. {who}{when}\n\n{body}\n")
    record = "\n".join(parts)
    leftover = _LEFTOVER_SENSITIVE.search(record)
    if leftover:
        msg = f"sanitising left {leftover.group(0)!r} in the record; check it by hand"
        raise ValueError(msg)
    return record


def write_into_page(*, page: Path, record: str) -> None:
    """Replace the text between the record markers of `page` with `record`.

    Args:
        page: The docs page holding `RECORD_START` and `RECORD_END`.
        record: The markdown record.
    """
    source = page.read_text()
    head, _, rest = source.partition(RECORD_START)
    _, _, tail = rest.partition(RECORD_END)
    page.write_text(f"{head}{RECORD_START}\n\n{record}\n{RECORD_END}{tail}")


def main() -> None:
    """Read the transcript named on the command line and write its record into the page."""
    if len(sys.argv) != 3:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <transcript.jsonl> <page.md>")
    record = render(read_entries(Path(sys.argv[1])))
    write_into_page(page=Path(sys.argv[2]), record=record)


if __name__ == "__main__":
    main()

"""Experiment 712, attempt 2: the reply-format text of the prompt, and nothing else.

**Attempt 2 changes only how the model is asked to format its reply.** The 30 replies of attempt 1
that failed the strict schema failed on format: a missing `register_row` key and a `reason` of more
than 60 words. Each change below answers one of those failures or makes the reply shape
unambiguous:

- The system prompt and the schema ask for raw JSON text, with no prose and no code fence.
- The prompt of arm `llm` shows no register rows, so its schema has no `register_row` key. A
  prompt with register rows keeps the key and says that it is always present.
- The `reason` is at most 40 words. The reply parser cuts a longer `reason` at 60 words, which is
  slack and not the prompt's bound.

**The changes were chosen without reading any label, confidence, or agreement with a known answer.**
The classes, the onset offset, the confidence semantics, and every table of the prompt are those of
`e00712_prompt`, whose source is not edited, so attempt 1 stays reproducible.

`e00712_label.configure_attempt("v2")` replaces `e00712_prompt.SYSTEM_PROMPT`,
`SCHEMA_NO_REGISTER`, and `SCHEMA_WITH_REGISTER` with the texts here, and the prompt version hash of
attempt 2 also covers the source of this file.
"""

from typing import Final

import e00712_prompt as prompts

_OLD_LAST_LINE: Final[str] = "Reply with one JSON object and nothing else."
SYSTEM_PROMPT: Final[str] = prompts.SYSTEM_PROMPT.replace(
    _OLD_LAST_LINE,
    "Reply with one strict JSON object and nothing else: raw JSON text, with no prose before or\n"
    "after it and no code fence.",
)

_OLD_HEAD: Final[str] = "Reply with one JSON object with exactly these keys:"
SCHEMA_HEAD: Final[str] = prompts.SCHEMA_HEAD.replace(
    _OLD_HEAD,
    "Reply with one strict JSON object, as raw JSON text with no prose and no code fence,\n"
    "with exactly these keys:",
)
SCHEMA_NO_REGISTER: Final[str] = (
    SCHEMA_HEAD
    + """
  "reason": string of at most 40 words
No register rows are given in this prompt, so the object has no "register_row" key."""
)
SCHEMA_WITH_REGISTER: Final[str] = (
    SCHEMA_HEAD
    + """
  "register_row": the number of the register row you relied on, or null; always include this key
  "reason": string of at most 40 words"""
)

if _OLD_LAST_LINE not in prompts.SYSTEM_PROMPT or _OLD_HEAD not in prompts.SCHEMA_HEAD:
    msg = "the attempt 1 prompt text changed, so the attempt 2 replacements no longer apply"
    raise ValueError(msg)

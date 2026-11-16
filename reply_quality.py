"""Reply cleanup: cut the echo tail a small model leaves behind.

A 147M model under SFT does not fail by being wrong, it fails by *looping*:
``"Hello! I'm Jin, a small language model — a small language model — a small
language model..."``. The token-level helpers in :mod:`model` detect that loop
while decoding; this module cleans up the text afterwards, because a streamed
reply has already sent the first copies to the browser.

Character-level on purpose: it also catches loops the tokenizer hides
("—-—-—", "… …", repeated ASCII art), and it needs no tokenizer at all.
"""

from __future__ import annotations

from model import REPEAT_MAX_PERIOD, REPEAT_MIN_REPEATS, REPEAT_MIN_SPAN
from model import repeated_repeat_count

# Text spans are longer than token spans: 24 BPE tokens can be ~100 characters.
TEXT_MAX_PERIOD = 400


def repeated_tail(text: str, *, min_repeats: int = REPEAT_MIN_REPEATS,
                  min_span: int = REPEAT_MIN_SPAN,
                  max_period: int = TEXT_MAX_PERIOD) -> tuple[int, int] | None:
    """Return ``(period, repeats)`` of a looped tail in ``text``, else ``None``.

    ``period`` is the length in characters of the repeated chunk and
    ``repeats`` how many times it appears at the end.
    """
    n = len(text)
    for period in range(1, min(max_period, n) + 1):
        repeats = repeated_repeat_count(period, min_repeats, min_span)
        if period * repeats > n:
            break
        chunk = text[n - period:]
        if all(
            text[n - period * (k + 1): n - period * k] == chunk
            for k in range(1, repeats)
        ):
            return period, repeats
    return None


# The loop chunk usually carries the separator with it (" a small language
# model —"), so after keeping one copy a dangling connector is left behind.
_DANGLING = " \t\r\n\u2014\u2013-\u2026;:,|/\\*\u00b7\u2022\u00ab\u00bb"


def _strip_dangling_connector(text: str) -> str:
    """Drop trailing separators left over after cutting a repeated tail."""
    stripped = text.rstrip(_DANGLING)
    # Never strip a whole (possibly empty) reply into nothing.
    return stripped if stripped.strip() else text.rstrip()


def trim_repeated_tail(text: str, *, keep_copies: int = 1,
                       min_repeats: int = REPEAT_MIN_REPEATS,
                       min_span: int = REPEAT_MIN_SPAN,
                       max_period: int = TEXT_MAX_PERIOD) -> str:
    """Keep the first ``keep_copies`` of a looped tail, drop the rest.

    Returns ``text`` unchanged when the tail does not loop.
    """
    if not text:
        return text
    hit = repeated_tail(text, min_repeats=min_repeats, min_span=min_span,
                        max_period=max_period)
    if hit is None:
        return text
    period, repeats = hit
    drop = period * max(0, repeats - keep_copies)
    if not drop:
        return text
    return _strip_dangling_connector(text[: len(text) - drop])


def looks_like_echo(text: str, **kwargs) -> bool:
    """True when the reply's tail is a verbatim loop."""
    return repeated_tail(text, **kwargs) is not None


def stop_reason_label(reason: str | None) -> str:
    """Human wording for ``model.generate``'s ``stop_reason``."""
    return {
        "eos": "ended on the stop token",
        "repeat": "cut off: the model started repeating itself",
        "length": "hit the token budget",
    }.get(reason or "", "unknown")

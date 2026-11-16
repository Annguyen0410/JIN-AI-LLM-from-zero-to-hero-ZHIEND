"""
JIN AI chat template.

All messages are wrapped with the following special tokens so the model
learns an explicit conversational structure:

    <|system|>   ... system message ... <|end|>
    <|user|>     ... user message ...   <|end|>
    <|assistant|> ... reply ...         <|end|>
    <|endoftext|>   (document / conversation terminator)

The template is intentionally simple and deterministic so it can be
emitted by Python code and recognized at generation time.
"""

from __future__ import annotations

from typing import Iterable
import re


ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

ROLE_TAG = {
    ROLE_SYSTEM: "<|system|>",
    ROLE_USER: "<|user|>",
    ROLE_ASSISTANT: "<|assistant|>",
    ROLE_TOOL: "<|tool|>",
}
END_TAG = "<|end|>"
EOT_TAG = "<|endoftext|>"

# Phase 3 reasoning delimiters (CoT data).
THINK_TAG = "<|think|>"
THINK_END_TAG = "<|/think|>"


DEFAULT_SYSTEM_PROMPT = (
    "You are Jin, a small but capable AI assistant built from scratch in PyTorch. "
    "You speak naturally and clearly, give direct answers, show your reasoning when "
    "it helps, write code in fenced blocks, and admit when you do not know. "
    "You are honest, curious, and respectful."
)


def format_message(role: str, content: str) -> str:
    if role not in ROLE_TAG:
        raise ValueError(f"Unknown role: {role!r}")
    tag = ROLE_TAG[role]
    clean = (content or "").strip()
    # User/retrieved text cannot manufacture ChatML roles or terminate a turn.
    # Assistant/tool training targets retain their legitimate action tokens.
    if role == ROLE_USER:
        clean = re.sub(r"<\|[^\n<>]{1,40}\|>",
                       lambda m: m[0].replace("<", "＜").replace(">", "＞"), clean)
    return f"{tag}\n{clean}\n{END_TAG}\n"


def format_conversation(
    messages: Iterable[dict],
    add_generation_prompt: bool = False,
    default_system: str | None = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Render a list of ``{"role": ..., "content": ...}`` messages.

    If ``add_generation_prompt`` is True, the output ends with the
    ``<|assistant|>`` tag (no content and no closing tag), so the model
    can continue directly into a reply.

    If no system message is provided and ``default_system`` is set,
    a system message is prepended.
    """

    messages = list(messages)
    has_system = any(m.get("role") == ROLE_SYSTEM for m in messages)

    parts: list[str] = []
    if not has_system and default_system:
        parts.append(format_message(ROLE_SYSTEM, default_system))

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role not in ROLE_TAG:
            continue
        parts.append(format_message(role, content))

    if add_generation_prompt:
        parts.append(f"{ROLE_TAG[ROLE_ASSISTANT]}\n")

    return "".join(parts)


def format_training_document(
    messages: Iterable[dict],
    default_system: str | None = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Render a conversation as a self-contained training document.

    A trailing ``<|endoftext|>`` tells the model the conversation has
    ended, which also doubles as the generation stop signal.
    """

    body = format_conversation(
        messages,
        add_generation_prompt=False,
        default_system=default_system,
    )
    return body + EOT_TAG + "\n"


def format_tool_trace(
    user_query: str,
    call: str,
    result: str,
    final_answer: str,
    system_prompt: str | None = None,
) -> str:
    """Render a tool-use trace as a training document (Phase 3.4).

    Structure:
        <|user|> query <|end|>
        <|tool|> {"name": ..., "args": ...} <|end|>
        <|user|> <tool result> ... <|end|>
        <|assistant|> answer with citation <|end|>
        <|endoftext|>

    ``call`` should be a JSON string like ``{"name": "web_search",
    "args": {...}}``; ``result`` the tool output text.
    """

    messages = [
        {"role": ROLE_USER, "content": user_query},
        {"role": ROLE_TOOL, "content": call},
        {"role": ROLE_USER, "content": f"<tool result> {result}"},
        {"role": ROLE_ASSISTANT, "content": final_answer},
    ]
    return format_training_document(messages, default_system=system_prompt)


def format_cot_document(
    user_query: str,
    thinking: str,
    answer: str,
    system_prompt: str | None = None,
) -> str:
    """Render a chain-of-thought example (Phase 3.3).

    The reasoning lives between ``<|think|>`` and ``<|/think|>`` so the
    model learns to separate private reasoning from the final answer.
    """

    messages = [
        {"role": ROLE_USER, "content": user_query},
        {
            "role": ROLE_ASSISTANT,
            "content": f"{THINK_TAG}\n{thinking}\n{THINK_END_TAG}\n{answer}",
        },
    ]
    return format_training_document(messages, default_system=system_prompt)


STOP_TOKEN_STRINGS: tuple[str, ...] = (EOT_TAG, END_TAG)

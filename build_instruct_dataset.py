"""
Build the JIN AI instruction / chat training corpus.

Pipeline
--------
1. Load hand-crafted seed dialogues from ``data/seed/conversations.jsonl``.
2. (Optional) Synthesize additional simple Q&A dialogues from the raw
   Gutenberg corpus using pure, deterministic templates. This gives the
   model exposure to "given a book passage, here is a short question and
   grounded answer" format, which is great for teaching it to hold a
   conversation about text without using any external API.
3. Mix in a reasonable amount of the raw literary corpus so that the
   model does not forget English fluency.
4. Emit a single file ``data/chat_train.txt`` where every example is
   wrapped with the ChatML-style template in ``chat_template.py``.

Usage:
    python build_instruct_dataset.py
    python build_instruct_dataset.py --synth-samples 4000 --literary-fraction 0.5
    python build_instruct_dataset.py --no-synth --no-literary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
import random
import re
import sys
from pathlib import Path

# stdout/stderr default to the console code page (cp1252 on Windows) which
# cannot encode Vietnamese or emoji-laden model answers; every stats/sample
# print below would raise UnicodeEncodeError and abort the build mid-run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - exotic streams
        pass

import config
from chat_template import (
    DEFAULT_SYSTEM_PROMPT,
    EOT_TAG,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
    format_training_document,
)


SEED_FILE = os.path.join(config.ROOT_DIR, "data", "seed", "conversations.jsonl")
HF_SEED_DIR = os.path.join(config.ROOT_DIR, "data", "seed", "hf")
OUTPUT_FILE = os.path.join(config.ROOT_DIR, "data", "chat_train.txt")


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")


QUESTION_TEMPLATES = [
    "Could you explain this passage in simple words?",
    "Can you summarize what is happening here?",
    "What is the main idea of this passage?",
    "In your own words, what is going on?",
    "What do you take away from this paragraph?",
    "Retell this passage briefly.",
    "What is the feeling of this paragraph?",
]

CONTINUATION_TEMPLATES = [
    "Please continue this passage in a similar style.",
    "What might come next in this text?",
    "Write a short continuation of this paragraph.",
    "Keep the tone of this passage and continue it for a few sentences.",
]

DEFINITION_TEMPLATES = [
    "What does the word '{word}' mean?",
    "Can you explain the word '{word}' simply?",
    "Give me a short definition of '{word}'.",
]


# ----------------------------------------------------------------------
# Seed loader
# ----------------------------------------------------------------------

def _load_jsonl_conversations(
    path: str,
    *,
    max_messages: int | None = None,
    max_chars_per_message: int | None = None,
    max_total_chars: int | None = None,
    max_conversations: int | None = None,
) -> list[list[dict]]:
    """Load ``{"messages": [...]}`` JSONL file with optional length caps.

    Long OASST-style trees can have 10+ turns and very large bodies.
    We cap so a handful of huge conversations don't dominate the corpus.

    ``max_conversations`` keeps memory bounded with **reservoir sampling** over
    the whole file instead of parsing everything and truncating afterwards:
    ``openhermes`` alone has 1,000,551 rows / 1.6 GB, which does not fit beside
    the rest of the mix on a 16 GB machine.
    """

    conversations: list[list[dict]] = []
    if not os.path.exists(path):
        return conversations
    accepted = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = record.get("messages")
            if not isinstance(messages, list):
                continue
            kept: list[dict] = []
            total_chars = 0
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                content = m.get("content")
                if role not in {ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT} or not content:
                    continue
                content = str(content)
                if max_chars_per_message and len(content) > max_chars_per_message:
                    content = content[:max_chars_per_message].rstrip() + "..."
                kept.append({"role": role, "content": content})
                total_chars += len(content)
                if max_messages and len(kept) >= max_messages:
                    break
                if max_total_chars and total_chars >= max_total_chars:
                    break
            # Must end with an assistant turn to be useful as training signal.
            while kept and kept[-1]["role"] != ROLE_ASSISTANT:
                kept.pop()
            if len(kept) >= 2:
                if max_conversations is None or max_conversations <= 0:
                    conversations.append(kept)
                    continue
                accepted += 1
                if len(conversations) < max_conversations:
                    conversations.append(kept)
                else:
                    j = random.randrange(accepted)
                    if j < max_conversations:
                        conversations[j] = kept
    return conversations


def normalize_text(text: str) -> str:
    """Collapse whitespace + lowercase, for duplicate detection only."""
    return " ".join(str(text).split()).lower()


def conversation_fingerprint(conv: list[dict]) -> str:
    """Stable hash of a whole conversation (role + normalised text)."""
    body = "\n".join(f"{m.get('role')}:{normalize_text(m.get('content', ''))}" for m in conv)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def dedup_conversations(
    conversations: list[list[dict]],
    *,
    max_identical_answers: int = 4,
) -> tuple[list[list[dict]], dict[str, int]]:
    """Drop duplicate conversations and cap repeated answer sequences.

    The May corpus had 30,020 assistant turns but only **755 distinct answers**
    (``--seed-duplication 40`` over 365 hand-authored dialogues), so the model
    learned a canned set of strings and echoed them. Whole-conversation
    duplicates cover that; the answer cap additionally limits the *same* answer
    pair appearing across many different questions ("2 + 2 = 4." appeared
    120 times). ``max_identical_answers <= 0`` disables the cap.
    """
    kept: list[list[dict]] = []
    seen: set[str] = set()
    answer_counts: dict[str, int] = {}
    dropped_duplicates = 0
    dropped_repeated_answers = 0
    for conv in conversations:
        fingerprint = conversation_fingerprint(conv)
        if fingerprint in seen:
            dropped_duplicates += 1
            continue
        seen.add(fingerprint)
        answers = [normalize_text(m.get("content", ""))
                   for m in conv if m.get("role") == ROLE_ASSISTANT]
        key = " | ".join(a for a in answers if a)
        if max_identical_answers > 0 and key:
            if answer_counts.get(key, 0) >= max_identical_answers:
                dropped_repeated_answers += 1
                continue
            answer_counts[key] = answer_counts.get(key, 0) + 1
        kept.append(conv)
    stats = {
        "dropped_duplicates": dropped_duplicates,
        "dropped_repeated_answers": dropped_repeated_answers,
        "kept": len(kept),
    }
    return kept, stats


def answer_stats(conversations: list[list[dict]]) -> dict:
    """Assistant-turn count vs distinct answers — the echo-risk metric."""
    counter: Counter = Counter()
    turns = 0
    for conv in conversations:
        for m in conv:
            if m.get("role") != ROLE_ASSISTANT:
                continue
            turns += 1
            counter[normalize_text(m.get("content", ""))] += 1
    return {
        "turns": turns,
        "unique_answers": len(counter),
        "repeat_ratio": (round(1 - len(counter) / turns, 4) if turns else 0.0),
        "top_answers": counter.most_common(5),
    }


def load_seed(path: str) -> list[list[dict]]:
    """Load the primary seed file plus any other top-level *.jsonl files
    in the same directory (e.g. ``knowledge_pack.jsonl``).

    Files inside the ``hf/`` sub-directory are intentionally skipped here
    because they have a separate, capped loader.
    """

    convs = _load_jsonl_conversations(path)
    seed_dir = Path(path).parent
    if seed_dir.exists():
        for extra in sorted(seed_dir.glob("*.jsonl")):
            if extra.resolve() == Path(path).resolve():
                continue
            extra_convs = _load_jsonl_conversations(str(extra))
            if extra_convs:
                print(
                    f"[Seed] +{len(extra_convs):>4} from {extra.name}"
                )
                convs.extend(extra_convs)
    return convs


def load_hf_seed(
    directory: str,
    *,
    max_per_dataset: int | None = None,
    max_messages: int = 12,
    max_chars_per_message: int = 2_000,
    max_total_chars: int = 8_000,
) -> list[list[dict]]:
    """Load every ``*.jsonl`` produced by ``fetch_datasets.py``.

    Each file corresponds to one upstream dataset (Alpaca, Dolly, OASST,
    etc.). We cap per dataset so large sources don't monopolize the mix.
    """

    out: list[list[dict]] = []
    root = Path(directory)
    if not root.exists():
        return out
    for path in sorted(root.glob("*.jsonl")):
        convs = _load_jsonl_conversations(
            str(path),
            max_messages=max_messages,
            max_chars_per_message=max_chars_per_message,
            max_total_chars=max_total_chars,
            max_conversations=max_per_dataset,
        )
        print(f"[HF  ] {path.name}: {len(convs):,} conversations")
        out.extend(convs)
    return out


# ----------------------------------------------------------------------
# Synthetic Q&A from raw Gutenberg text
# ----------------------------------------------------------------------

# Max chars read per raw file when synthesizing dialogues / sampling
# literary passages. Full books are fine; multi-GB dumps (TinyStories,
# Wikipedia shards) only contribute their opening slice.
MAX_CHARS_PER_RAW_DOC = 400_000


def iter_raw_documents(raw_dir: str, max_chars_per_doc: int = MAX_CHARS_PER_RAW_DOC):
    """Yield (doc_id, text) pairs for every usable raw text file.

    Reads at most ``max_chars_per_doc`` characters per file so a 2 GB
    dump cannot exhaust RAM (the old code did ``path.read_text()``).
    """

    root = Path(raw_dir)
    if not root.exists():
        return
    for path in sorted(root.rglob("*.txt")):
        try:
            if path.stat().st_size < 5_000:
                continue
            if max_chars_per_doc and path.stat().st_size > max_chars_per_doc:
                from clean_corpus import read_text_prefix
                text = read_text_prefix(path, max_chars_per_doc)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(text) < 5_000:
            continue
        doc_id = path.stem
        yield doc_id, text


def clean_paragraph(block: str) -> str:
    block = block.strip()
    block = re.sub(r"\s+", " ", block)
    return block


def good_paragraph(block: str) -> bool:
    if len(block) < 180 or len(block) > 900:
        return False
    # avoid metadata / ToC style lines
    lowered = block.lower()
    bad_markers = (
        "project gutenberg",
        "produced by",
        "end of the project",
        "start of the project",
        "chapter contents",
        "table of contents",
        "illustration:",
    )
    if any(marker in lowered for marker in bad_markers):
        return False
    letters = sum(c.isalpha() for c in block)
    if letters / max(len(block), 1) < 0.6:
        return False
    return True


def summarize_paragraph(paragraph: str) -> str:
    """Deterministic tiny 'summary': take the first sentence or two.

    This is intentionally shallow: it is just a teacher signal that says
    "here is a reasonable short restatement". We rely on seed dialogues
    to teach general conversational ability; the synth data teaches the
    model how to react to *passages*.
    """

    sentences = SENTENCE_SPLIT_RE.split(paragraph.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return paragraph.strip()[:220]
    first = sentences[0]
    if len(first) > 220 or len(sentences) == 1:
        summary = first
    else:
        summary = first + " " + sentences[1]
    if len(summary) > 320:
        summary = summary[:317].rsplit(" ", 1)[0] + "..."
    return summary


def synthesize_summary_dialogue(doc_id: str, paragraph: str) -> list[dict] | None:
    paragraph = clean_paragraph(paragraph)
    if not good_paragraph(paragraph):
        return None
    question = random.choice(QUESTION_TEMPLATES)
    user_content = f'Here is a short passage from "{doc_id}":\n\n{paragraph}\n\n{question}'
    assistant_content = summarize_paragraph(paragraph)
    return [
        {"role": ROLE_USER, "content": user_content},
        {"role": ROLE_ASSISTANT, "content": assistant_content},
    ]


def synthesize_continuation_dialogue(doc_id: str, paragraph: str) -> list[dict] | None:
    paragraph = clean_paragraph(paragraph)
    if not good_paragraph(paragraph):
        return None
    sentences = SENTENCE_SPLIT_RE.split(paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 4:
        return None
    split = max(1, len(sentences) // 2)
    prefix = " ".join(sentences[:split])
    suffix = " ".join(sentences[split : split + 3])
    if len(prefix) < 120 or len(suffix) < 60:
        return None
    prompt = random.choice(CONTINUATION_TEMPLATES)
    user_content = f'Here is the start of a passage:\n\n{prefix}\n\n{prompt}'
    return [
        {"role": ROLE_USER, "content": user_content},
        {"role": ROLE_ASSISTANT, "content": suffix},
    ]


def synthesize_definition_dialogue(text: str, long_words: list[str]) -> list[dict] | None:
    if not long_words:
        return None
    word = random.choice(long_words)
    # Find the first sentence in text that contains the word and explains it.
    pattern = re.compile(r"[^.]*\b" + re.escape(word) + r"\b[^.]*\.", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return None
    sentence = match.group(0).strip()
    if len(sentence) < 40 or len(sentence) > 320:
        return None
    question = random.choice(DEFINITION_TEMPLATES).format(word=word.lower())
    user_content = question
    assistant_content = (
        f"The word '{word.lower()}' appears in this kind of context: "
        f'"{sentence}" From that, it roughly refers to the idea or action '
        f"being described in the sentence. If you'd like a formal "
        f"dictionary definition, a dictionary will be more precise than me."
    )
    return [
        {"role": ROLE_USER, "content": user_content},
        {"role": ROLE_ASSISTANT, "content": assistant_content},
    ]


def gather_long_words(text: str, min_len: int = 9, max_words: int = 200) -> list[str]:
    words = [w for w in WORD_RE.findall(text) if len(w) >= min_len]
    # keep only reasonably distinct words
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in seen:
            continue
        seen.add(lw)
        out.append(w)
        if len(out) >= max_words:
            break
    return out


def synthesize_from_corpus(
    raw_dir: str,
    samples_per_doc: int = 40,
    total_cap: int = 6000,
) -> list[list[dict]]:
    """Generate synthetic instruction dialogues grounded in the raw corpus."""

    results: list[list[dict]] = []
    for doc_id, text in iter_raw_documents(raw_dir):
        paragraphs = [
            p for p in PARAGRAPH_SPLIT_RE.split(text)
            if good_paragraph(clean_paragraph(p))
        ]
        if not paragraphs:
            continue
        random.shuffle(paragraphs)
        long_words = gather_long_words(text)

        produced = 0
        for paragraph in paragraphs:
            if produced >= samples_per_doc:
                break
            pick = random.random()
            if pick < 0.55:
                dialogue = synthesize_summary_dialogue(doc_id, paragraph)
            elif pick < 0.85:
                dialogue = synthesize_continuation_dialogue(doc_id, paragraph)
            else:
                dialogue = synthesize_definition_dialogue(paragraph, long_words)
            if dialogue is None:
                continue
            results.append(dialogue)
            produced += 1
            if len(results) >= total_cap:
                return results

    return results


# ----------------------------------------------------------------------
# Literary echo (keep some raw prose in the chat dataset)
# ----------------------------------------------------------------------

def sample_literary_chunks(raw_dir: str, chunk_chars: int, total_chars: int) -> list[str]:
    """Take random ~``chunk_chars`` slices from books until we reach
    ``total_chars`` characters. Each chunk is wrapped into a fake
    dialogue where the user asks for a reading and the model delivers
    the text verbatim. This keeps the model's prose fluency alive.
    """

    out: list[str] = []
    produced = 0
    docs: list[tuple[str, str]] = list(iter_raw_documents(raw_dir))
    if not docs:
        return out
    random.shuffle(docs)
    for doc_id, text in docs:
        if produced >= total_chars:
            break
        for _ in range(10):
            if produced >= total_chars:
                break
            if len(text) < chunk_chars + 2:
                continue
            start = random.randint(0, len(text) - chunk_chars - 1)
            slice_ = text[start : start + chunk_chars]
            # Snap to paragraph-ish boundaries.
            first_break = slice_.find("\n\n")
            if first_break > 0:
                slice_ = slice_[first_break + 2 :]
            last_break = slice_.rfind("\n\n")
            if last_break > 0:
                slice_ = slice_[:last_break]
            slice_ = slice_.strip()
            if len(slice_) < 400:
                continue
            out.append(f"__LIT__::{doc_id}::{slice_}")
            produced += len(slice_)
    return out


def literary_chunk_to_dialogue(token: str) -> list[dict] | None:
    if not token.startswith("__LIT__::"):
        return None
    _, doc_id, slice_ = token.split("::", 2)
    user_content = f'Read me a passage from "{doc_id}".'
    return [
        {"role": ROLE_USER, "content": user_content},
        {"role": ROLE_ASSISTANT, "content": slice_},
    ]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the JIN AI chat-format training corpus."
    )
    parser.add_argument("--seed-file", default=SEED_FILE)
    parser.add_argument("--hf-seed-dir", default=HF_SEED_DIR,
                        help="Directory of *.jsonl files produced by fetch_datasets.py.")
    parser.add_argument("--hf-max-per-dataset", type=int, default=20_000,
                        help="Cap conversations taken from any single HF dataset.")
    parser.add_argument("--no-hf-seed", action="store_true",
                        help="Ignore data/seed/hf/ even if it exists.")
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--raw-dir", default=config.RAW_DATA_DIR)
    parser.add_argument("--seed-duplication", type=int, default=None,
                        help="Each seed dialogue is emitted this many times. Default: "
                             "1 when HF conversations are available, else 40. The old "
                             "unconditional 40 was what collapsed the corpus to 755 "
                             "distinct answers.")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Keep duplicate conversations and repeated answers.")
    parser.add_argument("--max-identical-answers", type=int, default=4,
                        help="Cap how often one answer sequence may repeat.")
    parser.add_argument("--synth-samples", type=int, default=2_000,
                        help="Max number of synthetic Q&A dialogues to generate from the raw corpus.")
    parser.add_argument("--synth-per-doc", type=int, default=20)
    parser.add_argument("--no-synth", action="store_true")
    parser.add_argument("--no-literary", action="store_true",
                        help="Skip the literary echo entirely. Recommended for "
                             "chat tuning — Gutenberg passages can drown out chat data.")
    parser.add_argument("--literary-chunk-chars", type=int, default=1_600)
    parser.add_argument("--literary-target-mb", type=float, default=2.0,
                        help="Target size (MB) of literary echo content. "
                             "Lower than before (was 15.0) so chat data dominates.")
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument(
        "--no-contamination-check",
        action="store_true",
        help="Skip the benchmark-leakage guard (removes any document sharing "
             "an n-gram with data/eval/probes.jsonl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = config.output_path(args.output)
    random.seed(args.random_seed)

    print("=" * 72)
    print("JIN AI chat corpus builder")
    print("=" * 72)

    seed_conversations = load_seed(args.seed_file)
    print(f"[Seed] Loaded {len(seed_conversations)} hand-authored conversations.")

    hf_conversations: list[list[dict]] = []
    if not args.no_hf_seed:
        hf_conversations = load_hf_seed(
            args.hf_seed_dir,
            max_per_dataset=args.hf_max_per_dataset,
        )
        if hf_conversations:
            print(f"[HF  ] Loaded {len(hf_conversations):,} total HF conversations.")
        else:
            print(f"[HF  ] No HF datasets found in {args.hf_seed_dir} (run fetch_datasets.py).")
    else:
        print("[HF  ] Skipped (--no-hf-seed).")

    synth_conversations: list[list[dict]] = []
    if not args.no_synth:
        synth_conversations = synthesize_from_corpus(
            raw_dir=args.raw_dir,
            samples_per_doc=args.synth_per_doc,
            total_cap=args.synth_samples,
        )
        print(f"[Synth] Generated {len(synth_conversations)} Q&A dialogues from raw corpus.")
    else:
        print("[Synth] Skipped (--no-synth).")

    literary_tokens: list[str] = []
    if not args.no_literary:
        literary_tokens = sample_literary_chunks(
            raw_dir=args.raw_dir,
            chunk_chars=args.literary_chunk_chars,
            total_chars=int(args.literary_target_mb * 1024 * 1024),
        )
        print(f"[Lit]  Sampled {len(literary_tokens)} literary passages.")
    else:
        print("[Lit]  Skipped (--no-literary).")

    # Assemble master list.
    # 40 repeats existed to give 365 hand-authored dialogues enough weight when
    # they were the whole corpus. With HF data loaded it only teaches the model
    # to parrot a canned answer set, so default to no repetition.
    duplication = args.seed_duplication
    if duplication is None:
        duplication = 1 if hf_conversations else 40
    print(f"[Seed] seed-duplication = {duplication} "
          f"({'HF data present' if hf_conversations else 'hand-authored only'})")

    master: list[list[dict]] = []
    for conv in seed_conversations:
        for _ in range(max(1, duplication)):
            master.append(conv)
    master.extend(synth_conversations)
    master.extend(hf_conversations)

    before = answer_stats(master)
    print(f"[Mix ] {len(master):,} dialogues | {before['turns']:,} assistant turns | "
          f"{before['unique_answers']:,} distinct answers "
          f"(repeat ratio {before['repeat_ratio']:.3f})")
    if not args.no_dedup:
        master, dedup_stats = dedup_conversations(
            master, max_identical_answers=args.max_identical_answers,
        )
        print(f"[Dedup] dropped {dedup_stats['dropped_duplicates']:,} duplicate dialogues "
              f"+ {dedup_stats['dropped_repeated_answers']:,} repeated-answer dialogues "
              f"-> kept {dedup_stats['kept']:,}")
    after = answer_stats(master)
    print(f"[Mix ] after dedup: {after['turns']:,} turns | "
          f"{after['unique_answers']:,} distinct answers "
          f"(repeat ratio {after['repeat_ratio']:.3f})")
    for text, count in after["top_answers"]:
        print(f"        {count:>5}x  {text[:70]!r}")

    # Literary chunks are resolved to dialogue at emit time.
    random.shuffle(master)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    written_docs = 0
    written_chars = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for conv in master:
            doc = format_training_document(conv, default_system=DEFAULT_SYSTEM_PROMPT)
            out.write(doc)
            written_docs += 1
            written_chars += len(doc)

        # Interleave the literary passages so the model keeps its prose.
        for token in literary_tokens:
            conv = literary_chunk_to_dialogue(token)
            if conv is None:
                continue
            doc = format_training_document(conv, default_system=None)
            out.write(doc)
            written_docs += 1
            written_chars += len(doc)

    print()
    print(f"[Output] Wrote {written_docs:,} documents "
          f"({written_chars / (1024 * 1024):.2f} MB) -> {args.output}")

    if not args.no_contamination_check:
        from data_quality import check_contamination
        from chat_template import EOT_TAG
        # Stream the output file (constant memory) and screen one
        # conversation document at a time — each doc ends with EOT_TAG.
        removed_total = 0
        kept_path = args.output + ".kept"
        with open(args.output, "r", encoding="utf-8") as handle, \
                open(kept_path, "w", encoding="utf-8") as out:
            pending = ""
            for line in handle:
                pending += line
                if EOT_TAG in line:
                    doc = pending.strip()
                    pending = ""
                    if not doc:
                        continue
                    kept, removed = check_contamination([doc])
                    removed_total += len(removed)
                    for d in kept:
                        out.write(d + "\n")
            tail_doc = pending.strip()
            if tail_doc:
                kept, removed = check_contamination([tail_doc])
                removed_total += len(removed)
                for d in kept:
                    out.write(d + "\n")
        os.replace(kept_path, args.output)
        if removed_total:
            print(f"[Output] Contamination check: removed {removed_total:,} docs "
                  f"(leaked data/eval/probes.jsonl n-grams)")

    print("[Output] Sample (first 400 chars):")
    with open(args.output, "r", encoding="utf-8") as handle:
        print(handle.read(400))


if __name__ == "__main__":
    main()

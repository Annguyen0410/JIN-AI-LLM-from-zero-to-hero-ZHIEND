"""
JIN AI byte-level BPE tokenizer (from scratch, no external ML libs).

Design
------
- Byte-level: every byte of UTF-8 input is representable, so the tokenizer
  can encode ANY text losslessly (Vietnamese, emoji, code, etc.).
- Pre-tokenization with a GPT-2-style regex so merges do not cross word /
  whitespace / punctuation boundaries too aggressively.
- Incremental merge training: we keep a reverse index `pair -> {word}` and
  only touch the affected words each step. This lets us train 16k merges
  on a ~80 MB corpus in a few minutes of pure Python.
- Special tokens (e.g. ``<|user|>``, ``<|assistant|>``, ``<|end|>``) are
  reserved IDs that are NEVER broken by BPE and always encoded atomically.

This tokenizer is self-contained: it does not depend on tiktoken,
transformers, sentencepiece, or any pretrained vocabulary.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Iterable


DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|end|>",
    "<|pad|>",
)

# Phase 3 reasoning / tool-use tokens. APPENDED at the tail of the vocab
# (IDs come after all existing merges + specials), so adding them never
# shifts existing token IDs. They are ONLY used when building a fresh
# tokenizer (``extended=True``) — existing checkpoints keep their 6-token
# vocab and load unchanged; their embeddings simply have no rows for these
# three ids until the next tokenizer rebuild (per the Phase 3 plan).
EXTENDED_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|think|>",
    "<|/think|>",
    "<|tool|>",
)


def default_special_tokens(extended: bool = False) -> list[str]:
    """The base 6 special tokens, plus the 3 Phase-3 tokens if ``extended``."""
    toks = list(DEFAULT_SPECIAL_TOKENS)
    if extended:
        toks.extend(EXTENDED_SPECIAL_TOKENS)
    return toks


# GPT-2-ish pre-tokenization pattern, adapted for stdlib `re`.
# It groups: english contractions, letter runs, digit runs, punctuation runs,
# and whitespace runs. Leading-space capture keeps tokens like " the"
# distinct from "the" which is important for fluent generation.
#: Bumped when the pre-tokenizer changes the token stream. Token caches are
#: keyed on this, so a splitter fix automatically rebuilds them instead of
#: silently serving tokens produced by the old (buggy) rules.
BPE_IMPL_VERSION = 2

#: Pre-tokenizer. Every alternative must cover a slice of the input: `findall`
#: DISCARDS unmatched characters, which is how the underscore used to vanish
#: (`LAST_UPDATED` -> `LASTUPDATED`) and with it every snake_case identifier in
#: the code corpus. `_` is a word character (\w) yet excluded from the letter
#: class [^\W\d_], so it needs its own alternative.
_PRETOKEN_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|"""
    r""" ?[^\W\d_]+|"""
    r""" ?\d+|"""
    r""" ?_+|"""
    r""" ?[^\s\w]+|"""
    r"""\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _pretokenize(text: str) -> list[str]:
    return _PRETOKEN_PATTERN.findall(text)


class BPETokenizer:
    """Byte-level BPE tokenizer with support for reserved special tokens."""

    def __init__(self, special_tokens: Iterable[str] | None = None):
        self.special_tokens: list[str] = list(
            special_tokens if special_tokens is not None else default_special_tokens()
        )
        self.merges: list[tuple[int, int]] = []
        self.merge_ranks: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self.token_to_id: dict[bytes, int] = {}
        self.special_to_id: dict[str, int] = {}
        self.id_to_special: dict[int, str] = {}
        self.vocab_size: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        text: str,
        vocab_size: int = 16_384,
        min_pair_freq: int = 2,
        verbose: bool = True,
    ) -> None:
        """Train BPE merges on ``text`` until the vocabulary reaches ``vocab_size``.

        The final vocab layout is:
            [0 .. 255]               raw bytes (always present)
            [256 .. 256+M-1]         learned merges (M = vocab_size - 256 - S)
            [end - S .. end - 1]     S special tokens (fixed IDs at the tail)
        """

        if vocab_size <= 256 + len(self.special_tokens):
            raise ValueError(
                "vocab_size must be greater than 256 + number of special tokens"
            )

        if verbose:
            print(f"[BPE] Pre-tokenizing {len(text):,} characters...")

        t_start = time.time()
        word_freqs: Counter[str] = Counter(_pretokenize(text))
        if verbose:
            print(
                f"[BPE] Unique pre-tokens: {len(word_freqs):,} "
                f"(took {time.time() - t_start:.1f}s)"
            )

        # Each word becomes an ordered list of int token IDs (initially byte IDs).
        word_ids: dict[str, list[int]] = {}
        for word in word_freqs:
            word_ids[word] = list(word.encode("utf-8"))

        # Initial vocab: 256 raw bytes.
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        next_id = 256

        # Count pair frequencies and maintain an inverted index
        # pair -> set of words containing it. That lets every merge step
        # touch only the impacted words instead of the whole corpus.
        pair_freq: Counter[tuple[int, int]] = Counter()
        pair_words: dict[tuple[int, int], set[str]] = defaultdict(set)
        for word, freq in word_freqs.items():
            ids = word_ids[word]
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                pair_freq[pair] += freq
                pair_words[pair].add(word)

        target_merges = vocab_size - 256 - len(self.special_tokens)
        t_train = time.time()
        for step in range(target_merges):
            if not pair_freq:
                break

            best_pair, best_count = max(
                pair_freq.items(), key=lambda kv: (kv[1], kv[0])
            )
            if best_count < min_pair_freq:
                if verbose:
                    print(
                        f"[BPE] Stopping early at {step} merges "
                        f"(top pair freq={best_count} below min={min_pair_freq})."
                    )
                break

            new_id = next_id
            next_id += 1
            self.merges.append(best_pair)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            affected_words = list(pair_words.pop(best_pair, ()))
            pair_freq.pop(best_pair, None)

            for word in affected_words:
                ids = word_ids[word]
                freq = word_freqs[word]

                # Remove contributions of every pair this word currently holds.
                for i in range(len(ids) - 1):
                    old_pair = (ids[i], ids[i + 1])
                    pair_freq[old_pair] -= freq
                    if pair_freq[old_pair] <= 0:
                        pair_freq.pop(old_pair, None)
                    bucket = pair_words.get(old_pair)
                    if bucket is not None:
                        bucket.discard(word)
                        if not bucket:
                            pair_words.pop(old_pair, None)

                # Rebuild the word, merging every occurrence of ``best_pair``.
                new_ids: list[int] = []
                i = 0
                while i < len(ids):
                    if (
                        i < len(ids) - 1
                        and ids[i] == best_pair[0]
                        and ids[i + 1] == best_pair[1]
                    ):
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                word_ids[word] = new_ids

                # Re-add contributions for this word.
                for i in range(len(new_ids) - 1):
                    new_pair = (new_ids[i], new_ids[i + 1])
                    pair_freq[new_pair] += freq
                    pair_words[new_pair].add(word)

            if verbose and (step + 1) % 500 == 0:
                elapsed = time.time() - t_train
                done = step + 1
                rate = done / max(elapsed, 1e-6)
                remain = (target_merges - done) / max(rate, 1e-6)
                print(
                    f"[BPE] merge {done:>6}/{target_merges} | "
                    f"top freq {best_count:>8} | {rate:5.1f} merges/s | "
                    f"ETA {remain/60:5.1f} min"
                )

        # Append special tokens with reserved IDs at the very end.
        self.special_to_id = {}
        self.id_to_special = {}
        for tok in self.special_tokens:
            self.vocab[next_id] = tok.encode("utf-8")
            self.special_to_id[tok] = next_id
            self.id_to_special[next_id] = tok
            next_id += 1

        self.vocab_size = next_id
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.token_to_id = {b: i for i, b in self.vocab.items()}

        if verbose:
            print(
                f"[BPE] Training done: {len(self.merges):,} merges, "
                f"vocab_size={self.vocab_size:,} "
                f"(took {time.time() - t_train:.1f}s)"
            )

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------
    def _encode_word(self, word: str) -> list[int]:
        """Apply learned merges to a single pre-tokenized word."""

        if not word:
            return []
        ids = list(word.encode("utf-8"))
        if not self.merge_ranks:
            return ids

        while len(ids) >= 2:
            best_rank = None
            best_i = -1
            for i in range(len(ids) - 1):
                rank = self.merge_ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_rank is None:
                break
            merged_id = 256 + best_rank
            ids = ids[:best_i] + [merged_id] + ids[best_i + 2 :]
        return ids

    def encode(
        self,
        text: str,
        allow_special: bool = True,
        return_stats: bool = False,
    ) -> list[int] | tuple[list[int], int]:
        """Encode text to token ids.

        If ``allow_special`` is True, occurrences of any registered special
        token string are emitted as that single reserved ID.
        """

        out: list[int] = []
        if not text:
            return (out, 0) if return_stats else out

        if allow_special and self.special_to_id:
            pattern = "|".join(re.escape(tok) for tok in self.special_to_id)
            chunks = re.split(f"({pattern})", text)
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue
            if allow_special and chunk in self.special_to_id:
                out.append(self.special_to_id[chunk])
                continue
            for word in _pretokenize(chunk):
                out.extend(self._encode_word(word))

        if return_stats:
            return out, 0
        return out

    def decode(self, ids: Iterable[int]) -> str:
        buf = bytearray()
        for i in ids:
            token_bytes = self.vocab.get(int(i))
            if token_bytes is None:
                continue
            buf.extend(token_bytes)
        return buf.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_state_dict(self) -> dict:
        return {
            "kind": "bpe_v1",
            "special_tokens": list(self.special_tokens),
            "merges": [list(pair) for pair in self.merges],
            "vocab_size": self.vocab_size,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != "bpe_v1":
            raise ValueError("Not a BPE tokenizer state dict")
        self.special_tokens = list(state.get("special_tokens", []))
        self.merges = [tuple(pair) for pair in state.get("merges", [])]
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256
        for a, b in self.merges:
            self.vocab[next_id] = self.vocab[a] + self.vocab[b]
            next_id += 1
        self.special_to_id = {}
        self.id_to_special = {}
        for tok in self.special_tokens:
            self.vocab[next_id] = tok.encode("utf-8")
            self.special_to_id[tok] = next_id
            self.id_to_special[next_id] = tok
            next_id += 1
        self.vocab_size = int(state.get("vocab_size", next_id))
        self.token_to_id = {b: i for i, b in self.vocab.items()}

    def save(self, path: str) -> None:
        from storage import output_path
        path = output_path(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_state_dict(), f, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.load_state_dict(json.load(f))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def preview(self, text: str, n: int = 20) -> list[str]:
        ids = self.encode(text)
        out: list[str] = []
        for i in ids[:n]:
            b = self.vocab.get(int(i), b"?")
            try:
                out.append(b.decode("utf-8"))
            except UnicodeDecodeError:
                out.append(repr(b))
        return out


if __name__ == "__main__":
    import argparse
    import codecs
    import os

    parser = argparse.ArgumentParser(description="Train the JIN AI BPE tokenizer")
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--out", default="bpe.json")
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--min-pair-freq", type=int, default=2)
    parser.add_argument(
        "--sample-mb", type=float, default=60.0,
        help="Train BPE on a strided sample of this many MB (0 = read whole "
             "corpus). Sampling keeps memory bounded on multi-GB corpora and "
             "spans the whole file so every language/domain is represented.",
    )
    args = parser.parse_args()
    from storage import configure_runtime, output_path
    configure_runtime()
    args.out = output_path(args.out)

    sample_bytes = int(args.sample_mb * 1024 * 1024) if args.sample_mb > 0 else None
    file_size = os.path.getsize(args.data)
    if sample_bytes is None or file_size <= sample_bytes:
        with open(args.data, "r", encoding="utf-8") as f:
            corpus = f.read()
    else:
        # Strided sample: keep the head of every 1 MB block. This spans the
        # entire corpus (all languages/domains) with constant memory.
        chunk_size = 1 << 20
        chunks = max(1, (file_size + chunk_size - 1) // chunk_size)
        keep_per_chunk = max(4_096, min(chunk_size, sample_bytes // chunks))
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        parts: list[str] = []
        with open(args.data, "rb") as f:
            while True:
                block = f.read(chunk_size)
                if not block:
                    break
                text = decoder.decode(block)
                if len(text) > keep_per_chunk:
                    text = text[:keep_per_chunk]
                parts.append(text)
        corpus = "".join(parts)
        print(f"[BPE] Sampled {len(corpus)/1e6:.1f} MB of {file_size/1e6:.1f} MB corpus "
              f"({len(parts)} blocks x {keep_per_chunk} chars)")

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=args.vocab_size, min_pair_freq=args.min_pair_freq)
    tok.save(args.out)
    print(f"[BPE] Saved -> {args.out}")
    print("[BPE] Sample:", tok.preview("Hello, world! How are you today?", n=30))

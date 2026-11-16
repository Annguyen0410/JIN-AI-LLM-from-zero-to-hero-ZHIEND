"""GPT-2 byte-level BPE tokenizer (Phase 2 reference profile).

Two backends, same interface as :class:`bpe_tokenizer.BPETokenizer`:

1. ``transformers.GPT2TokenizerFast`` when available — Rust BPE, exact parity
   with OpenAI GPT-2, batch encoding fast enough for corpus work.
2. A dependency-free fallback that downloads the official
   ``encoder.json`` + ``vocab.bpe`` and implements the byte-level merge loop
   in pure Python. Slower, but guarantees the profile still works on a machine
   without ``transformers``.

Special-token policy
--------------------
GPT-2's vocabulary is fixed at 50,257 ids (50,256 BPE merges + ``<|endoftext|>``).
Adding BOS/PAD ids would change the vocabulary and break parity with the
reference model, so:

* ``<|endoftext|>`` (id 50256) is used as **both** EOS and BOS, exactly as the
  GPT-2 paper does (documents are separated by EOT).
* PAD reuses the EOS id. Loss on padding positions is masked out by
  ``loss_mask`` in the trainer, so padding never contributes gradients.

Every document is therefore encoded as ``<|endoftext|> + body + <|endoftext|>``
when ``add_special_tokens=True``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

EOT = "<|endoftext|>"
GPT2_VOCAB_SIZE = 50_257
GPT2_EOT_ID = 50_256

_ENCODER_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"
_VOCAB_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "cache", "gpt2_bpe")


def _download(url: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        tmp = path + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-unicode map."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class GPT2BPETokenizer:
    """Byte-level BPE identical to OpenAI GPT-2 (vocab 50,257)."""

    kind = "gpt2_bpe"
    vocab_size = GPT2_VOCAB_SIZE
    eot_id = GPT2_EOT_ID
    bos_id = GPT2_EOT_ID
    eos_id = GPT2_EOT_ID
    pad_id = GPT2_EOT_ID           # masked out of the loss by the trainer

    def __init__(self, backend: str = "auto", cache_dir: str = _CACHE_DIR):
        self.cache_dir = cache_dir
        self.special_to_id = {EOT: GPT2_EOT_ID}
        self.id_to_special = {GPT2_EOT_ID: EOT}
        self.special_tokens = [EOT]
        self._fast = None
        self.backend = None
        if backend in ("auto", "transformers"):
            try:
                from transformers import GPT2TokenizerFast
                self._fast = GPT2TokenizerFast.from_pretrained(
                    "gpt2", cache_dir=cache_dir)
                self.backend = "transformers"
            except Exception:
                if backend == "transformers":
                    raise
        if self._fast is None:
            self._load_pure()
            self.backend = "pure-python"

    # ── pure-python backend ────────────────────────────────────────────
    def _load_pure(self) -> None:
        import regex  # need \p{L}/\p{N} classes, same as GPT-2's splitter

        encoder = json.load(open(_download(_ENCODER_URL, os.path.join(
            self.cache_dir, "encoder.json")), encoding="utf-8"))
        bpe_file = _download(_VOCAB_URL, os.path.join(self.cache_dir, "vocab.bpe"))
        ranks: dict[tuple[str, str], int] = {}
        with open(bpe_file, encoding="utf-8") as fh:
            next(fh)  # version header
            for rank, line in enumerate(fh):
                if line.strip():
                    a, b = line.rstrip("\n").split(" ")
                    ranks[(a, b)] = rank
        self._encoder = encoder
        self._decoder = {i: tok for tok, i in encoder.items()}
        self._ranks = ranks
        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
        self._pat = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def _bpe_pure(self, token: str) -> list[str]:
        word = list(token)
        if len(word) < 2:
            return word
        while True:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            candidates = [(self._ranks[p], p) for p in pairs if p in self._ranks]
            if not candidates:
                return word
            _, (a, b) = min(candidates)
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
            if len(word) == 1:
                return word

    def _encode_pure(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._pat.findall(text):
            mapped = "".join(self._byte_encoder[b] for b in chunk.encode("utf-8"))
            ids.extend(self._encoder[p] for p in self._bpe_pure(mapped))
        return ids

    def _decode_pure(self, ids: list[int]) -> str:
        text = "".join(self._decoder.get(int(i), "") for i in ids
                       if int(i) != GPT2_EOT_ID)
        raw = bytearray(self._byte_decoder[ch] for ch in text
                        if ch in self._byte_decoder)
        return raw.decode("utf-8", errors="replace")

    # ── public interface ───────────────────────────────────────────────
    def encode(self, text: str, allow_special: bool = True,
               return_stats: bool = False) -> list[int] | tuple[list[int], int]:
        if not text:
            return ([], 0) if return_stats else []
        if self._fast is not None:
            ids = self._fast.encode(text, add_special_tokens=False)
        else:
            ids = self._encode_pure(text)
        return (ids, 0) if return_stats else ids

    def decode(self, ids) -> str:
        ids = [int(i) for i in ids]
        if self._fast is not None:
            return self._fast.decode(ids, skip_special_tokens=False)
        return self._decode_pure(ids)

    def encode_document(self, text: str) -> list[int]:
        """``<|endoftext|> body <|endoftext|>`` — the GPT-2 document format."""
        return [GPT2_EOT_ID] + self.encode(text) + [GPT2_EOT_ID]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if self._fast is not None:
            return self._fast(texts, add_special_tokens=False)["input_ids"]
        return [self.encode(t) for t in texts]

    def token_text(self, token_id: int) -> str:
        if self._fast is not None:
            return self._fast.decode([int(token_id)], skip_special_tokens=False)
        return self._decode_pure([int(token_id)])

    # ── persistence (checkpoint embedded state) ────────────────────────
    def to_state_dict(self) -> dict:
        return {"kind": self.kind, "vocab_size": self.vocab_size,
                "special_tokens": list(self.special_tokens)}

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != self.kind:
            raise ValueError("Not a GPT-2 BPE tokenizer state dict")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_state_dict(), fh)


def load_tokenizer(profile: str = "gpt2") -> object:
    """Return a tokenizer for a named profile (``gpt2`` | ``jin``)."""
    if profile == "gpt2":
        return GPT2BPETokenizer()
    if profile == "jin":
        from bpe_tokenizer import BPETokenizer
        import config
        tok = BPETokenizer()
        tok.load(config.BPE_PATH)
        return tok
    raise ValueError(f"Unknown tokenizer profile: {profile!r}")


if __name__ == "__main__":
    tok = load_tokenizer("gpt2")
    t0 = time.time()
    text = "Xin chào! JIN AI học ngôn ngữ từ số 0. The quick brown fox jumps over 13 lazy dogs."
    ids = tok.encode(text)
    print(f"backend={tok.backend} vocab={tok.vocab_size} tokens={len(ids)} "
          f"({len(text)/max(1,len(ids)):.2f} chars/token)")
    print("roundtrip:", tok.decode(ids) == text)
    print("ids:", ids[:20])
    print(f"encode speed: {len(text)/max(1e-9, time.time()-t0)/1000:.1f} kB/s")

# Tokenizer research (Phase 2)

Measured on real corpus slices with `scripts/tokenizer_eval.py`.
Raw numbers: `bench_results/gpt2_upgrade/tokenizer_eval.json`.

| Sample (400k chars unless noted) | JIN BPE 16,384 | GPT-2 BPE 50,257 |
|---|---:|---:|
| Vietnamese — tokens/char | 0.2764 (3.62 chars/token) | 0.8761 (1.14 chars/token) |
| Vietnamese — tokens/word | 1.258 | 3.987 |
| English (knowledge) — tokens/char | 0.2536 (3.94 c/t) | 0.2241 (4.46 c/t) |
| Stories — tokens/char | 0.2460 (4.06 c/t) | 0.2467 (4.05 c/t) |
| Code — tokens/char | 0.3911 (2.56 c/t) | 0.3516 (2.84 c/t) |
| Math — tokens/char | 0.3293 (3.04 c/t) | 0.2874 (3.48 c/t) |
| Mixed 425k chars — total tokens | 125,243 | 174,978 |
| Throughput | ~0.94 M chars/s | ~1.38 M chars/s |
| UTF-8 round-trip (VI / code) | **False / False** | True / True |

## What this means

**The JIN 16k vocabulary is strongly Vietnamese-optimised.** It needs
**3.2× fewer tokens** for Vietnamese than GPT-2's 50k vocabulary
(1.14 vs 3.62 chars/token), and slightly fewer on English stories. On English
knowledge, code and math GPT-2's vocabulary is 10–15 % more efficient.

That is a real advantage for a Vietnamese-first model, and it is why swapping
in GPT-2's tokenizer for the existing JIN checkpoint would be a downgrade, not
an upgrade. The directive's rule — never change the vocabulary of an existing
checkpoint without a proven migration — is therefore also the engineering
choice: **the modern JIN profile keeps its 16k vocabulary.**

## Bug found and fixed: the underscore was being dropped

`bpe_tokenizer._PRETOKEN_PATTERN` had no alternative that matched `_`:

```
's|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+
```

`_` is a word character (`\w`) but explicitly excluded from the letter class
`[^\W\d_]`, and ` ?[^\s\w]+` excludes word characters. `re.findall` therefore
**discarded every underscore** in the input.

Consequences measured on the real corpus:

* `|LAST_UPDATED2|` was tokenized as `|LASTUPDATED2|`.
* Every `snake_case_name`, `self.x_1`, `__init__` in the code corpus was
  silently mangled before training. That is training-data corruption, not a
  cosmetic issue, and it degrades exactly the domain the model is weakest at.

Fix (`BPE_IMPL_VERSION = 2`): an explicit ` ?_+` alternative, so the splitter
covers the whole input. Because a splitter change alters the token stream for a
*vocabulary that did not change*, the token-cache fingerprint now includes
`BPE_IMPL_VERSION` — stale caches rebuild automatically instead of being
reused. The checkpoint-embedded tokenizer state is untouched, so existing
checkpoints still load and resume.

Regression tests: `tests/test_tokenizer_upgrade.py` (20 cases, including
120,000 characters of real Vietnamese and a full UTF-8 round-trip).

## Decision for each profile

| Profile | Tokenizer | Reason |
|---|---|---|
| `jin_small_modern` (existing checkpoints) | JIN BPE 16,384 (unchanged) | 3.2× cheaper Vietnamese; changing it would invalidate every checkpoint |
| `gpt2_small_reference` (new) | GPT-2 BPE 50,257 | Required for parameter/layout parity with GPT-2 Small and for a fair comparison |

Special tokens for the GPT-2 profile: the vocabulary is fixed at 50,257 ids, so
adding BOS/PAD would break parity. `<|endoftext|>` (50,256) is used as both BOS
and EOS, exactly as GPT-2 does, and PAD reuses the same id with loss masked on
padding positions. Documents are encoded as `<|endoftext|> body <|endoftext|>`.

## Deferred (not done in this pass)

A purpose-built **32k multilingual byte-level BPE** would dominate both
profiles (English/code efficiency of 50k, Vietnamese efficiency of the current
16k, plus guaranteed byte round-trip). It is deliberately deferred: it requires
retraining the vocabulary on the full corpus, re-tokenizing every dataset and
retraining every model from scratch, and the directive rules out changing an
existing checkpoint's vocabulary without a migration. The measurement harness
to evaluate such a candidate already exists (`scripts/tokenizer_eval.py`).

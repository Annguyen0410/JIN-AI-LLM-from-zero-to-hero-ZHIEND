"""
JIN AI benchmark — one command for every number (Phase 0 of v3.0).

Measures, for a given checkpoint:

- **Quality** — held-out val loss + perplexity on a fixed, locked eval set
  (``data/eval/heldout.txt``, carved out of the training corpus and never
  trained on afterwards).
- **Knowledge & capability** — ``data/eval/probes.jsonl`` (>=150 hand-written
  probes across 8 domains). Each probe is scored automatically, no external
  model: exact match / numeric match / keyword coverage, plus a universal
  logprob check (is the correct answer more likely than a wrong distractor?).
- **Fluency** — reuses ``fluency_check.py`` verbatim (same prompts, same
  heuristics) so the number stays comparable with v2.5 history.
- **Speed** — prefill tok/s, decode tok/s (KV cache), TTFT, peak VRAM.
- **Training throughput** — tok/s forward+backward on the same architecture
  (per device + preset, ``--no-train-bench`` to skip).
- **Forgetting** — via ``--compare A.json B.json``: per-domain probe deltas
  of the newer run against the older one on the shared probe set.

Usage::

    python bench.py --checkpoint checkpoints/jingpt_chat.pt
    python bench.py --init-eval          # carve heldout.txt out of train.txt once
    python bench.py --compare bench_results/baseline_v25.json bench_results/new.json

Output: ``bench_results/<timestamp>_<checkpoint>.json`` + a markdown table
printed to the terminal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

import config
from chat_template import DEFAULT_SYSTEM_PROMPT, END_TAG, EOT_TAG, ROLE_SYSTEM, ROLE_USER, ROLE_TAG, format_conversation
from fluency_check import PROBES as FLUENCY_PROBES
from fluency_check import chat_once as fluency_chat_once
from fluency_check import fluency_score, verdict
from utils import (
    build_inference_bundle,
    configure_cuda_performance,
    extract_model_config,
    load_checkpoint_file,
    tokenizer_from_state,
)

BENCH_DIR = os.path.join(config.ROOT_DIR, "bench_results")
EVAL_DIR = os.path.join(config.ROOT_DIR, "data", "eval")
HELDOUT_PATH = os.path.join(EVAL_DIR, "heldout.txt")
PROBES_PATH = os.path.join(EVAL_DIR, "probes.jsonl")

# Held-out loss is computed on non-overlapping 512-token blocks; cap the
# number of blocks so a huge heldout set can't make the bench run forever.
HELDOUT_BLOCK = 512
HELDOUT_MAX_BLOCKS = 512

PROBE_MAX_NEW_TOKENS = 48
PROBE_TEMPERATURE = 0.0          # greedy -> deterministic, reproducible scores
PROBE_REPETITION_PENALTY = 1.0   # no penalty: keep greedy output raw

_DOMAIN_ORDER = [
    "factual", "math", "code", "reasoning", "vietnamese",
    "instruction_following", "safety", "long_context",
]


# ── Small helpers ───────────────────────────────────────────────────────

def _safe(s: str) -> str:
    """Return *s* encoded in a way that survives a Windows console."""
    enc = sys.stdout.encoding or "utf-8"
    return s.encode(enc, errors="replace").decode(enc, errors="replace")


def normalize(text: str) -> str:
    """Lowercase, drop non-alphanumerics, collapse whitespace."""
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9à-ỹăâđêôơư]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+(?:[.,]\d+)?", text.replace(",", ""))]


# ── Eval-set bootstrap ──────────────────────────────────────────────────

def init_eval(heldout_mb: float, force: bool) -> None:
    """Carve ``data/eval/heldout.txt`` out of ``data/train.txt`` (once).

    Snapshots the original corpus first, then removes the carved slice so
    the held-out set is never trained on.
    """
    if os.path.exists(HELDOUT_PATH) and not force:
        print(f"[init-eval] {HELDOUT_PATH} already exists. Use --force to re-carve.")
        return
    if not os.path.exists(config.DATA_PATH):
        raise SystemExit(f"[init-eval] {config.DATA_PATH} not found — nothing to carve from.")

    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(config.DATA_SNAPSHOT_DIR, exist_ok=True)

    # Stream the corpus (constant memory) instead of readlines(): corpora
    # can now be multi-GB. Cut at the last <doc ...> boundary that leaves
    # roughly ``heldout_mb`` at the tail.
    total_chars = 0
    doc_boundary_char_offsets: list[int] = []
    with open(config.DATA_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("<doc "):
                doc_boundary_char_offsets.append(total_chars)
            total_chars += len(line)
    target = int(heldout_mb * 1024 * 1024)

    cut_char = total_chars
    for offset in doc_boundary_char_offsets:
        if total_chars - offset <= target:
            cut_char = offset
            break
    if cut_char == total_chars and doc_boundary_char_offsets:
        cut_char = doc_boundary_char_offsets[-1]

    ts = time.strftime("%Y%m%d_%H%M%S")
    snapshot = os.path.join(config.DATA_SNAPSHOT_DIR, f"train_pre_phase0_{ts}.txt")
    shutil.copyfile(config.DATA_PATH, snapshot)
    print(f"[init-eval] Snapshot of original corpus -> {snapshot}")

    heldout_chars = 0
    with open(config.DATA_PATH, "r", encoding="utf-8") as fh, \
            open(HELDOUT_PATH, "w", encoding="utf-8") as heldout_fh, \
            open(config.DATA_PATH + ".kept", "w", encoding="utf-8") as kept_fh:
        seen = 0
        for line in fh:
            if seen >= cut_char:
                heldout_fh.write(line)
                heldout_chars += len(line)
            else:
                kept_fh.write(line)
            seen += len(line)
    os.replace(config.DATA_PATH + ".kept", config.DATA_PATH)
    kept_chars = cut_char

    print(f"[init-eval] Carved {heldout_chars:,} chars ({heldout_mb:.2f}MB target) into {HELDOUT_PATH}")
    print(f"[init-eval] Remaining train corpus: {kept_chars:,} chars (was {total_chars:,})")
    print("[init-eval] NOTE: token cache fingerprint changes — train.py will re-encode once.")


# ── Quality (held-out loss / perplexity) ────────────────────────────────

def run_quality(model, tokenizer, device: str) -> dict[str, Any]:
    if not os.path.exists(HELDOUT_PATH):
        return {"error": f"heldout set missing — run `python bench.py --init-eval` first"}
    with open(HELDOUT_PATH, "r", encoding="utf-8") as fh:
        text = fh.read()
    encoded = tokenizer.encode(text, return_stats=True)
    ids = encoded[0] if isinstance(encoded, tuple) else encoded
    if len(ids) <= HELDOUT_BLOCK:
        return {"error": f"heldout set too short ({len(ids)} tokens)"}

    data = torch.tensor(ids, dtype=torch.long)
    n_blocks = (len(data) - 1) // HELDOUT_BLOCK
    n_blocks = min(n_blocks, HELDOUT_MAX_BLOCKS)
    use_amp = device == "cuda"
    autocast_dtype = torch.float16

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for b in range(n_blocks):
            x = data[b * HELDOUT_BLOCK : (b + 1) * HELDOUT_BLOCK].unsqueeze(0).to(device)
            y = data[b * HELDOUT_BLOCK + 1 : (b + 1) * HELDOUT_BLOCK + 1].unsqueeze(0).to(device)
            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", dtype=autocast_dtype, enabled=use_amp):
                _, loss = model(x, y)
            total_loss += float(loss.item())
    avg_loss = total_loss / n_blocks
    return {
        "heldout_path": HELDOUT_PATH,
        "heldout_blocks": n_blocks,
        "heldout_loss": round(avg_loss, 4),
        "heldout_ppl": round(math.exp(avg_loss), 2),
    }


# ── Probes ──────────────────────────────────────────────────────────────

def _load_probes(path: str = PROBES_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"[bench] {path} not found.")
    probes = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                probes.append(json.loads(line))
    return probes


def _exact_pass(answer_norm: str, probe: dict) -> bool:
    for cand in probe.get("exact", []):
        if cand and normalize(cand) in answer_norm:
            return True
    return False


def _numeric_pass(answer: str, probe: dict) -> bool:
    ref = float(probe["reference"])
    tol = max(0.5, abs(ref) * 0.01)
    numbers = extract_numbers(answer)
    return any(abs(n - ref) <= tol for n in numbers)


def _keyword_pass(answer_norm: str, probe: dict) -> tuple[bool, float]:
    required = [normalize(k) for k in probe.get("keywords", [])]
    if not required:
        return False, 0.0
    hits = sum(1 for k in required if k in answer_norm)
    return hits / len(required) >= float(probe.get("threshold", 0.6)), hits / len(required)


def _forbidden_hit(answer_norm: str, probe: dict) -> bool:
    for bad in probe.get("forbidden", []):
        if bad and normalize(bad) in answer_norm:
            return True
    return False


def _primary_score(answer: str, probe: dict) -> tuple[bool, float]:
    """Return ``(passed, detail)`` for the probe's primary scoring kind."""
    answer_norm = normalize(answer)
    if _forbidden_hit(answer_norm, probe):
        return False, 0.0
    kind = probe.get("kind", "exact")
    if kind == "exact":
        return _exact_pass(answer_norm, probe), 1.0 if _exact_pass(answer_norm, probe) else 0.0
    if kind == "numeric":
        return _numeric_pass(answer, probe), 1.0 if _numeric_pass(answer, probe) else 0.0
    if kind == "keyword":
        ok, cov = _keyword_pass(answer_norm, probe)
        return ok, round(cov, 3)
    return False, 0.0


def _answer_logprob(model, tokenizer, device, prefix_ids: list[int], answer: str) -> float:
    """Mean log-prob per token of *answer* given *prefix* (self-scoring, no KV cache)."""
    answer_ids = tokenizer.encode(answer)
    if not answer_ids:
        return float("-inf")
    full = prefix_ids + answer_ids
    if len(full) > model.block_size:
        full = full[-model.block_size:]
    start = len(full) - len(answer_ids)
    if start <= 0:
        return float("-inf")
    x = torch.tensor([full], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
    logp = F.log_softmax(logits[0], dim=-1)
    tgt = full[1:]
    total = 0.0
    n = 0
    for i in range(start, len(full) - 1):
        total += float(logp[i, tgt[i]])
        n += 1
    return total / max(1, n)


def run_probes(model, tokenizer, device: str, max_probes: int | None = None) -> dict[str, Any]:
    probes = _load_probes()
    if max_probes:
        probes = probes[:max_probes]

    stop_ids: set[int] = set()
    if hasattr(tokenizer, "special_to_id"):
        for tag in (END_TAG, EOT_TAG):
            tid = tokenizer.special_to_id.get(tag)
            if tid is not None:
                stop_ids.add(tid)

    by_domain: dict[str, list[dict]] = {d: [] for d in _DOMAIN_ORDER}
    results: list[dict] = []
    lp_gaps: list[float] = []

    model.eval()
    for probe in probes:
        domain = probe.get("domain", "other")
        passage = probe.get("passage", "")
        question = probe.get("prompt", "")
        full_prompt = f"{passage}\n\nQuestion: {question}" if passage else question

        text = format_conversation(
            [{"role": ROLE_SYSTEM, "content": DEFAULT_SYSTEM_PROMPT},
             {"role": ROLE_USER, "content": full_prompt}],
            add_generation_prompt=True, default_system=None,
        )
        ids = tokenizer.encode(text, allow_special=True)
        prefix_len = len(ids)

        # Greedy answer (deterministic -> reproducible across runs).
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx, max_new_tokens=PROBE_MAX_NEW_TOKENS,
            temperature=PROBE_TEMPERATURE, top_k=None, top_p=None,
            repetition_penalty=PROBE_REPETITION_PENALTY, stop_ids=stop_ids if stop_ids else None,
        )
        new = out[0].tolist()[prefix_len:]
        if stop_ids and new and new[-1] in stop_ids:
            new = new[:-1]
        answer = tokenizer.decode(new).strip()
        if hasattr(tokenizer, "special_to_id"):
            for tag in ROLE_TAG.values():
                cut = answer.find(tag)
                if cut != -1:
                    answer = answer[:cut].strip()

        passed, detail = _primary_score(answer, probe)

        # Universal logprob check: correct answer vs wrong distractor.
        lp_ref_text = probe.get("logprob_ref") or probe.get("reference", "")
        lp_wrong_text = probe.get("wrong", "")
        lp_gap = None
        lp_pass = None
        if lp_ref_text and lp_wrong_text and len(ids) < model.block_size - 16:
            lp_correct = _answer_logprob(model, tokenizer, device, ids, lp_ref_text)
            lp_wrong = _answer_logprob(model, tokenizer, device, ids, lp_wrong_text)
            if lp_correct > float("-inf") and lp_wrong > float("-inf"):
                lp_gap = round(lp_correct - lp_wrong, 4)
                lp_pass = lp_correct > lp_wrong
                lp_gaps.append(lp_gap)

        row = {
            "id": probe.get("id", "?"), "domain": domain, "kind": probe.get("kind"),
            "passed": passed, "detail": detail, "prefix_tokens": prefix_len,
            "answer": answer[:120], "lp_gap": lp_gap, "lp_pass": lp_pass,
        }
        results.append(row)
        by_domain.setdefault(domain, []).append(row)

    domain_stats: dict[str, dict] = {}
    for d, rows in by_domain.items():
        if not rows:
            continue
        n_pass = sum(1 for r in rows if r["passed"])
        lp_rows = [r for r in rows if r["lp_pass"] is not None]
        domain_stats[d] = {
            "n": len(rows),
            "pass_rate": round(n_pass / len(rows), 3),
            "lp_pass_rate": round(sum(1 for r in lp_rows if r["lp_pass"]) / max(1, len(lp_rows)), 3),
        }

    n_total = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    return {
        "n_probes": n_total,
        "pass_rate": round(n_pass / max(1, n_total), 3),
        "by_domain": domain_stats,
        "mean_lp_gap": round(sum(lp_gaps) / max(1, len(lp_gaps)), 4) if lp_gaps else None,
        "details": results,
    }


# ── Fluency (verbatim reuse of fluency_check.py) ────────────────────────

def run_fluency(model, tokenizer) -> dict[str, Any]:
    scores = []
    details = []
    for prompt in FLUENCY_PROBES:
        reply = fluency_chat_once(model, tokenizer, prompt,
                                  system=DEFAULT_SYSTEM_PROMPT, max_new_tokens=80)
        s = fluency_score(reply)
        scores.append(s["score"])
        details.append({"prompt": prompt, "reply": reply[:120], "score": s["score"]})
    avg = sum(scores) / max(1, len(scores))
    label, _ = verdict(scores)
    return {"avg_score": round(avg, 3), "verdict": label, "n": len(scores), "details": details}


# ── Speed (prefill / decode / TTFT / peak VRAM) ─────────────────────────

def _median_seconds(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def run_speed(model, tokenizer, device: str) -> dict[str, Any]:
    # Build a fixed prompt from the heldout set (or fallback text).
    prompt_ids: list[int] = []
    if os.path.exists(HELDOUT_PATH):
        with open(HELDOUT_PATH, "r", encoding="utf-8") as fh:
            enc = tokenizer.encode(fh.read(), return_stats=True)
            prompt_ids = (enc[0] if isinstance(enc, tuple) else enc)[:2048]
    if len(prompt_ids) < 600:
        prompt_ids = tokenizer.encode(
            "The history of science is full of surprises. " * 40)
    if len(prompt_ids) < 600:
        prompt_ids = list(range(600))[:600]
    prompt_ids = prompt_ids[: model.block_size]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prefill_len = 512
    prefill_ids = prompt_ids[:prefill_len]
    x_prefill = torch.tensor([prefill_ids], dtype=torch.long, device=device)

    def _prefill_once():
        with torch.no_grad():
            model(x_prefill)
    prefill_s = _median_seconds(_prefill_once, 8)
    prefill_tok_s = prefill_len / max(prefill_s, 1e-9)

    decode_len = 64
    dec_prompt = prompt_ids[:64]
    x_dec = torch.tensor([dec_prompt], dtype=torch.long, device=device)

    def _decode_once():
        kv = model.empty_kv_cache()
        cur = x_dec
        with torch.no_grad():
            for _ in range(decode_len):
                logits, _ = model(cur, kv_caches=kv)
                nxt = torch.argmax(logits[:, -1:, :], dim=-1)
                cur = nxt
    decode_s = _median_seconds(_decode_once, 3)
    decode_tok_s = decode_len / max(decode_s, 1e-9)

    def _ttft_once():
        x_ttft = torch.tensor([prompt_ids[:64]], dtype=torch.long, device=device)
        kv = model.empty_kv_cache()
        with torch.no_grad():
            logits, _ = model(x_ttft, kv_caches=kv)
            torch.argmax(logits[:, -1:, :], dim=-1)
    ttft_s = _median_seconds(_ttft_once, 5)

    peak_vram_gb = None
    if torch.cuda.is_available():
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    return {
        "prefill_tok_per_s": round(prefill_tok_s, 1),
        "decode_tok_per_s": round(decode_tok_s, 1),
        "ttft_ms": round(ttft_s * 1000, 1),
        "peak_vram_gb": peak_vram_gb,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
    }


# ── Training throughput (same architecture, random data) ────────────────
#
# Measured in a SUBPROCESS on purpose: fp32 AdamW on a 151M model needs
# ~2.4GB of optimizer state + ~0.6GB of temp buffers, which can exceed the
# usable VRAM ceiling of small WDDM hosts (desktop reserves part of the
# card). An OOM there can wedge the CUDA allocator for the rest of the
# process — a subprocess contains any crash. Falls back to the tiny preset
# when the same-architecture run does not fit, and says so in the report.


def _train_bench_impl(
    vocab_size: int,
    model_config: dict,
    device: str,
    *,
    label: str,
    block: int,
) -> dict[str, Any]:
    from model import JinAI
    cfg = dict(model_config)
    cfg["vocab_size"] = vocab_size
    cfg["device"] = device
    torch.manual_seed(1234)
    m = JinAI.from_config(cfg)
    optim = torch.optim.AdamW(m.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    m.train()

    batch, warmup, timed = 1, 1, 6
    x = torch.randint(0, vocab_size, (batch, block), device=device)
    y = torch.randint(0, vocab_size, (batch, block), device=device)

    def _step():
        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _, loss = m(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()

    for _ in range(warmup):
        _step()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(timed):
        _step()
    dt = time.perf_counter() - t0
    tok_s = (timed * batch * block) / max(dt, 1e-9)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None
    return {
        "train_tok_per_s": round(tok_s, 1),
        "train_steps": timed, "batch": batch, "block": block,
        "train_peak_vram_gb": peak, "device": device, "label": label,
        "params_m": round(m.num_parameters() / 1e6, 1),
    }


def run_train_bench_subprocess(vocab_size: int, model_config: dict, device: str) -> dict[str, Any]:
    """Measure train throughput, one fresh subprocess per attempt.

    Attempt 1 mirrors the checkpoint's architecture. On hosts where fp32
    AdamW state does not fit usable VRAM (measured on 6GB WDDM cards), that
    attempt OOMs; attempt 2 then measures the tiny preset, clearly labelled.
    Each attempt gets a clean process so an OOM can never wedge the rest of
    the bench or poison the parent's CUDA context.
    """

    import json as _json
    import subprocess
    import sys as _sys
    import tempfile

    tiny = config.SCALE_PRESETS["tiny"]
    tiny_cfg = dict(model_config)
    tiny_cfg.update({
        "embed_dim": tiny["embed_dim"],
        "n_layers": tiny["n_layers"],
        "n_heads": tiny["n_heads"],
        "block_size": tiny["block_size"],
    })
    attempts = [
        ("same-arch", dict(model_config), None),
        ("tiny-preset (fallback)", tiny_cfg,
         "same-arch fp32 AdamW does not fit usable VRAM on this host (measured "
         "constraint); throughput measured on tiny preset."),
    ]
    last_error = "no result"
    for label, cfg, note in attempts:
        payload = {"vocab_size": vocab_size, "model_config": cfg, "device": device}
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            _json.dump(payload, fh)
            cfg_path = fh.name
        try:
            proc = subprocess.run(
                [_sys.executable, os.path.abspath(__file__), "--train-bench-only", cfg_path],
                capture_output=True, text=True, timeout=900,
            )
            for line in proc.stdout.splitlines():
                if line.startswith("__BENCH_JSON__ "):
                    res = _json.loads(line[len("__BENCH_JSON__ "):])
                    if "error" not in res:
                        res["label"] = label
                        if note:
                            res["note"] = note
                        return res
                    last_error = res["error"]
            tail = (proc.stderr or "").strip().splitlines()[-2:]
            last_error = f"subprocess produced no result ({tail})"
        finally:
            try:
                os.unlink(cfg_path)
            except OSError:
                pass
    return {"error": last_error}


# ── Orchestration ───────────────────────────────────────────────────────

def _model_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _vocab_from_checkpoint(ck: dict) -> int:
    """Vocab size without building a model or touching CUDA."""
    state = ck.get("tokenizer_state")
    if isinstance(state, dict):
        try:
            return tokenizer_from_state(state).vocab_size
        except Exception:
            pass
    mc = ck.get("model_config") or {}
    if mc.get("vocab_size"):
        return int(mc["vocab_size"])
    sd = ck.get("model_state_dict") or {}
    w = sd.get("tok_emb.weight")
    if w is not None:
        return int(w.shape[0])
    raise ValueError("cannot infer vocab size from checkpoint")


def run_all(checkpoint_path: str, device: str, *, train_bench: bool, max_probes: int | None,
            ctx: int | None = None) -> dict[str, Any]:
    # The train-throughput subprocesses must start BEFORE this process creates
    # a CUDA context: on low-commit Windows hosts a torch parent + child
    # simultaneously trips the paging-file limit (WinError 1455), and CUDA
    # init in the parent right after CUDA-using children can segfault the
    # flaky WDDM driver. When train_bench is True we spawn them first.
    meta = None
    result_train = {"skipped": True}
    if train_bench:
        print("[1/5] Training throughput (subprocess) ...")
        ck = load_checkpoint_file(checkpoint_path, allow_fallback=True, map_location="cpu")
        vocab_size = _vocab_from_checkpoint(ck)
        cfg = extract_model_config(ck, vocab_size=vocab_size)
        result_train = run_train_bench_subprocess(vocab_size, cfg, device)
        print(f"[bench] train result: {result_train.get('label', result_train.get('error', '?'))}")

    # Deliberately NOT configure_cuda_performance(): its cudnn.benchmark flag
    # + fp16 warmup matmul segfaults this host's WDDM driver when followed by
    # a checkpoint load + big model creation. TF32 flags alone keep matmul
    # precision consistent with training without the unstable parts.
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[bench] Loading checkpoint: {checkpoint_path}  (device={device})")
    t_load = time.time()
    ctx_override = config.context_override(ctx) if ctx else None
    model, tokenizer, meta = build_inference_bundle(
        checkpoint_path, device=device, override_config=ctx_override)
    model = model.to(device)
    model.eval()
    print(f"[bench] Loaded in {time.time() - t_load:.1f}s | iter {meta.get('iteration', '?')} | "
          f"{model.num_parameters() / 1e6:.2f}M params | ctx {model.block_size}"
          f"{f' -> serving {model.max_positions}' if ctx else ''}")

    result: dict[str, Any] = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_name": _model_basename(checkpoint_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": device,
        "params_m": round(model.num_parameters() / 1e6, 1),
        "arch": {"modern": model.modern, "layers": model.n_layers, "heads": model.n_heads,
                 "n_kv_heads": model.n_kv_heads, "qk_norm": model.qk_norm,
                 "embed_dim": model.embed_dim, "block_size": model.block_size,
                 "max_positions": model.max_positions, "moe": bool(model.moe),
                 "sliding_window": model.sliding_window,
                 "params_active_m": round(model.active_parameters() / 1e6, 1)},
        "iteration": meta.get("iteration"),
        "checkpoint_loss": meta.get("loss"),
    }
    result["train"] = result_train

    print("\n[2/5] Quality (held-out loss / ppl) ...")
    result["quality"] = run_quality(model, tokenizer, device)

    print("[3/5] Probes ...")
    result["probes"] = run_probes(model, tokenizer, device, max_probes=max_probes)

    print("[4/5] Fluency ...")
    result["fluency"] = run_fluency(model, tokenizer)

    print("[5/5] Speed ...")
    result["speed"] = run_speed(model, tokenizer, device)

    # Forgetting requires a baseline to compare against — filled by --compare.
    result["forgetting"] = {"score": None, "note": "run `bench.py --compare <baseline.json> <new.json>`"}

    os.makedirs(BENCH_DIR, exist_ok=True)
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{_model_basename(checkpoint_path)}.json"
    out_path = os.path.join(BENCH_DIR, name)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n[bench] Saved -> {out_path}")
    append_history(result, out_path)

    print_markdown(result)
    return result


def append_history(result: dict[str, Any], result_path: str | None = None) -> None:
    """Append a compact run record to ``bench_results/history.json``.

    Feeds the Growth Dashboard (``/api/bench/history``): val loss, probe
    pass rate, fluency, tok/s and VRAM over time, per checkpoint. The
    record only carries the keys the dashboard plots, not the full JSON.
    """
    q = result.get("quality") or {}
    p = result.get("probes") or {}
    f = result.get("fluency") or {}
    s = result.get("speed") or {}
    t = result.get("train") or {}
    record = {
        "timestamp": result.get("timestamp"),
        "checkpoint": result.get("checkpoint_name"),
        "file": os.path.basename(result_path) if result_path else None,
        "params_m": result.get("params_m"),
        "heldout_loss": q.get("heldout_loss"),
        "heldout_ppl": q.get("heldout_ppl"),
        "probe_pass_rate": p.get("pass_rate"),
        "probe_mean_lp_gap": p.get("mean_lp_gap"),
        "fluency_score": f.get("avg_score"),
        "prefill_tok_per_s": s.get("prefill_tok_per_s"),
        "decode_tok_per_s": s.get("decode_tok_per_s"),
        "ttft_ms": s.get("ttft_ms"),
        "peak_vram_gb": s.get("peak_vram_gb"),
        "train_tok_per_s": t.get("train_tok_per_s"),
    }
    os.makedirs(BENCH_DIR, exist_ok=True)
    hist_path = os.path.join(BENCH_DIR, "history.json")
    history: list[dict] = []
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8") as fh:
                history = json.load(fh)
            if not isinstance(history, list):
                history = []
        except (OSError, json.JSONDecodeError):
            history = []
    history.append(record)
    with open(hist_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)


def print_markdown(r: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"BENCHMARK: {r['checkpoint_name']}  ({r['timestamp']})")
    print("=" * 60)
    q = r.get("quality") or {}
    p = r.get("probes") or {}
    f = r.get("fluency") or {}
    s = r.get("speed") or {}
    t = r.get("train") or {}

    rows = [
        ("params (M)", r.get("params_m")),
        ("held-out loss", q.get("heldout_loss")),
        ("held-out ppl", q.get("heldout_ppl")),
        ("probe pass rate", p.get("pass_rate")),
        ("probe mean logprob gap", p.get("mean_lp_gap")),
        ("fluency avg score", f.get("avg_score")),
        ("fluency verdict", f.get("verdict")),
        ("prefill tok/s", s.get("prefill_tok_per_s")),
        ("decode tok/s", s.get("decode_tok_per_s")),
        ("TTFT (ms)", s.get("ttft_ms")),
        ("peak VRAM (GB)", s.get("peak_vram_gb")),
    ]
    if t.get("train_tok_per_s") is not None:
        rows.append(("train tok/s", t.get("train_tok_per_s")))
        if t.get("label"):
            rows.append(("train bench label", t.get("label")))
    if t.get("train_peak_vram_gb") is not None:
        rows.append(("train peak VRAM (GB)", t.get("train_peak_vram_gb")))

    print("| Metric | Value |")
    print("|---|---|")
    for label, value in rows:
        print(f"| {label} | {value} |")

    by_domain = p.get("by_domain") or {}
    if by_domain:
        print("\n| Domain | n | pass rate | lp pass rate |")
        print("|---|---|---|---|")
        for d in _DOMAIN_ORDER:
            st = by_domain.get(d)
            if st:
                print(f"| {d} | {st['n']} | {st['pass_rate']} | {st['lp_pass_rate']} |")


def compare(a_path: str, b_path: str) -> None:
    with open(a_path, "r", encoding="utf-8") as fh:
        a = json.load(fh)
    with open(b_path, "r", encoding="utf-8") as fh:
        b = json.load(fh)

    print("=" * 66)
    print(f"COMPARE: {a.get('checkpoint_name')}  ->  {b.get('checkpoint_name')}")
    print("=" * 66)

    def _num(d: dict, key: str):
        for part in ("quality", "probes", "fluency", "speed", "train"):
            if d.get(part) and key in d[part] and isinstance(d[part][key], (int, float)):
                return float(d[part][key])
        return None

    pairs = [
        ("held-out loss", "heldout_loss"), ("held-out ppl", "heldout_ppl"),
        ("probe pass rate", "pass_rate"), ("fluency avg score", "avg_score"),
        ("prefill tok/s", "prefill_tok_per_s"), ("decode tok/s", "decode_tok_per_s"),
        ("TTFT (ms)", "ttft_ms"), ("peak VRAM (GB)", "peak_vram_gb"),
        ("train tok/s", "train_tok_per_s"),
    ]
    print("| Metric | A | B | diff | diff% |")
    print("|---|---|---|---|---|")
    for label, key in pairs:
        va, vb = _num(a, key), _num(b, key)
        if va is None or vb is None:
            continue
        delta = vb - va
        pct = (delta / va * 100) if va else float("nan")
        print(f"| {label} | {va:g} | {vb:g} | {delta:+.3g} | {pct:+.1f}% |")

    # Per-domain probe diff + forgetting score.
    da = (a.get("probes") or {}).get("by_domain") or {}
    db = (b.get("probes") or {}).get("by_domain") or {}
    if da and db:
        print("\n| Domain | A pass | B pass | diff |")
        print("|---|---|---|---|")
        shared = [d for d in _DOMAIN_ORDER if d in da and d in db]
        deltas = []
        for d in shared:
            pa, pb = da[d]["pass_rate"], db[d]["pass_rate"]
            deltas.append(pb - pa)
            print(f"| {d} | {pa:.3f} | {pb:.3f} | {pb - pa:+.3f} |")
        if deltas:
            # Forgetting = mean degradation of the older run's strengths.
            forgetting = sum(max(0.0, -dd) for dd in deltas) / len(deltas)
            print(f"\nforgetting_score (mean probe pass-rate drop on old domains): {forgetting:.4f}")
            print("(>=0; 0 = the new checkpoint lost nothing on the old probe set)")


def train_bench_only(cfg_path: str) -> None:
    """Subprocess entry point: measure train throughput, print one JSON line."""

    with open(cfg_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    vocab_size = int(payload["vocab_size"])
    model_config = payload["model_config"]
    device = payload["device"]
    config.set_device(device)
    configure_cuda_performance(config.DEVICE)

    try:
        result = _train_bench_impl(vocab_size, model_config, config.DEVICE,
                                   label="attempt", block=64)
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"[train-bench] attempt failed: {type(exc).__name__}: {exc}")
    print("__BENCH_JSON__ " + json.dumps(result))


def train_bench_command(checkpoint_path: str, device: str, out_path: str | None) -> None:
    """Standalone train-throughput benchmark (no CUDA in this process).

    Spawns fresh subprocesses for each attempt so an OOM can never wedge
    anything. On hosts where the same-architecture fp32 AdamW state does not
    fit, falls back to the tiny preset and says so.
    """

    ck = load_checkpoint_file(checkpoint_path, allow_fallback=True, map_location="cpu")
    vocab_size = _vocab_from_checkpoint(ck)
    cfg = extract_model_config(ck, vocab_size=vocab_size)
    result = run_train_bench_subprocess(vocab_size, cfg, device)
    result["checkpoint_name"] = _model_basename(checkpoint_path)
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path = out_path or os.path.join(
        BENCH_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_train_{_model_basename(checkpoint_path)}.json")
    os.makedirs(BENCH_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"[train-bench] {result.get('label', 'error')}: "
          f"{result.get('train_tok_per_s', 'n/a')} tok/s -> {out_path}")


def merge_train(bench_path: str, train_path: str) -> None:
    """Copy a train-throughput JSON into a benchmark JSON (same checkpoint)."""
    with open(bench_path, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    with open(train_path, "r", encoding="utf-8") as fh:
        train = json.load(fh)
    bench["train"] = train
    out = bench_path
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(bench, fh, ensure_ascii=False, indent=2)
    print(f"[merge-train] merged into {out}")
    print_markdown(bench)


def main() -> None:
    p = argparse.ArgumentParser(description="JIN AI benchmark (Phase 0).")
    p.add_argument("--checkpoint", default=config.get_checkpoint_path())
    p.add_argument("--device", default=None, choices=config.DEVICE_CHOICES)
    p.add_argument("--ctx", type=int, default=None, metavar="N",
                   help="Serve N tokens via RoPE scaling (e.g. 2048) and "
                        "measure whether the long_context probe actually "
                        "improves without a long-context fine-tune.")
    p.add_argument("--no-train-bench", action="store_true",
                   help="skip the train-throughput stage of the full run")
    p.add_argument("--train-bench", action="store_true",
                   help="only measure train throughput (standalone, subprocess-isolated)")
    p.add_argument("--train-out", default=None, help="output file for --train-bench")
    p.add_argument("--merge-train", nargs=2, metavar=("bench.json", "train.json"),
                   help="copy a --train-bench result into a full-run result")
    p.add_argument("--max-probes", type=int, default=None)
    p.add_argument("--init-eval", action="store_true", help="carve data/eval/heldout.txt out of train.txt")
    p.add_argument("--heldout-mb", type=float, default=1.5)
    p.add_argument("--force", action="store_true", help="re-carve heldout even if it exists")
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    p.add_argument("--train-bench-only", metavar="CFG.json",
                   help=argparse.SUPPRESS)  # internal subprocess entry point
    args = p.parse_args()

    if args.train_bench_only:
        train_bench_only(args.train_bench_only)
        return
    if args.train_bench:
        train_bench_command(args.checkpoint, args.device or "auto", args.train_out)
        return
    if args.merge_train:
        merge_train(args.merge_train[0], args.merge_train[1])
        return
    if args.compare:
        compare(args.compare[0], args.compare[1])
        return
    if args.init_eval:
        init_eval(args.heldout_mb, args.force)
        return

    # bench.py always resolves the device explicitly ("auto" -> CUDA/MPS/CPU)
    # instead of relying on set_device(None), which no-ops for other scripts.
    config.set_device(args.device or "auto")
    run_all(args.checkpoint, config.DEVICE,
            train_bench=not args.no_train_bench, max_probes=args.max_probes, ctx=args.ctx)


if __name__ == "__main__":
    main()
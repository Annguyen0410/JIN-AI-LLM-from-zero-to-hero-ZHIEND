"""
JIN AI web server.

This is a Flask app that exposes:

    GET  /                  - the chat UI
    GET  /api/status        - model + server status
    POST /api/chat          - synchronous chat completion (JSON)
    POST /api/chat/stream   - server-sent-events streaming chat completion
    POST /api/reset         - clear server-side conversation memory

Hardening:
    - Default bind address is 127.0.0.1; override via ``JINGPT_HOST`` only
      when you intentionally want LAN exposure.
    - Per-IP token-bucket rate limit on /api/chat and /api/chat/stream.
    - Hard caps on incoming JSON body size and on user message length.
    - Per-session conversation history kept on the server, capped to a
      reasonable number of turns to bound prompt length and memory.
    - Optional shared bearer token (``JINGPT_TOKEN``) for cases where the
      server is exposed behind a tunnel.
    - JSON / SSE responses set ``X-Content-Type-Options: nosniff`` and a
      conservative ``Content-Security-Policy``.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
import uuid
from collections import deque
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory
import torch

sys.path.insert(0, os.path.dirname(__file__))

import config
from bpe_tokenizer import BPETokenizer
from chat_template import (
    DEFAULT_SYSTEM_PROMPT,
    END_TAG,
    EOT_TAG,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TAG,
    ROLE_USER,
    format_conversation,
)
from utils import (
    build_inference_bundle,
    configure_cuda_performance,
    count_parameters,
)
from reply_quality import stop_reason_label, trim_repeated_tail
from web_search import looks_like_search_query, web_context
from grounding import bounded_prompt_ids, grounded_answer, grounding_prompt, escape_external
import logging

logger = logging.getLogger(__name__)


class JsonEventFormatter(logging.Formatter):
    def format(self, record):
        event = {"timestamp": self.formatTime(record), "level":record.levelname,
                 "component":record.name, "event":record.getMessage()}
        for key in ("request_id", "session_id", "endpoint", "status", "elapsed_ms",
                    "checkpoint_step", "source_count", "grounding_status", "error_type"):
            if hasattr(record,key):
                event[key] = getattr(record,key)
        return json.dumps(event, ensure_ascii=True)


# Long-term memory (Phase 4) — optional; --no-memory keeps v2.5 behaviour.
MEMORY_STORE = None
MEMORY_RETRIEVER = None
MEMORY_ENABLED = config.MEMORY_ENABLED


# ── Flask app ──────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_REQUEST_BYTES

# ── Global model state ─────────────────────────────────────────────────
_model_lock = threading.Lock()                 # generation is not thread-safe
tokenizer = None
model = None
checkpoint_meta: dict[str, Any] = {}
stop_token_ids: set[int] = set()
# Inference-time knobs (CLI flags / env): weight quant + KV cache modes.
WEIGHT_QUANT: str | None = None
KV_STATIC: bool = False
KV_INT8: bool = False

# Liveness counters surfaced by /api/status (Model Lab panel).
SERVER_START = time.time()
REQUESTS_SERVED = 0
_STATS_LOCK = threading.Lock()

# Extended-context serving (--ctx N): RoPE rescaling at load time, kept so
# /api/model/reload serves A/B checkpoints under the same context ceiling.
CONTEXT_REQUESTED: int | None = None
CONTEXT_OVERRIDE: dict[str, Any] | None = None


def _count_request() -> None:
    global REQUESTS_SERVED
    with _STATS_LOCK:
        REQUESTS_SERVED += 1

# ── Per-session chat memory ────────────────────────────────────────────
SESSIONS: dict[str, dict[str, Any]] = {}
SESSION_LOCK = threading.Lock()
MAX_SESSIONS = 256
MAX_TURNS_PER_SESSION = 16

# Prefix-cache budget: reuse KV cache between turns of the same session
# (TTFT cut on turn 2+). Bounded LRU so a busy server cannot OOM.
PREFIX_CACHE_MAX_SESSIONS = 16


# ── Per-IP token-bucket rate limiter ───────────────────────────────────
class TokenBucket:
    __slots__ = ("rate", "burst", "tokens", "stamp")

    def __init__(self, rate: float, burst: int):
        self.rate = float(rate)
        self.burst = int(burst)
        self.tokens = float(burst)
        self.stamp = time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.stamp
        self.stamp = now
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


_RATE_BUCKETS: dict[str, TokenBucket] = {}
_RATE_LOCK = threading.Lock()


def _client_ip() -> str:
    # Trust only the direct peer; we are not behind a reverse proxy by default.
    return request.remote_addr or "0.0.0.0"


def _rate_limit_ok() -> bool:
    ip = _client_ip()
    with _RATE_LOCK:
        bucket = _RATE_BUCKETS.get(ip)
        if bucket is None:
            bucket = TokenBucket(config.RATE_LIMIT_RPS, config.RATE_LIMIT_BURST)
            _RATE_BUCKETS[ip] = bucket
        return bucket.take()


def _auth_ok() -> bool:
    if not config.SERVER_AUTH_TOKEN:
        return True
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return secrets.compare_digest(header[7:], config.SERVER_AUTH_TOKEN)
    return False


def _client_is_loopback() -> bool:
    return _client_ip().startswith(("127.", "::1"))


def _admin_ok() -> bool:
    """Admin endpoints (/api/learn/*, /api/checkpoints, /api/model/reload…).

    Personal-first policy (2026-10-02): with **no** ``JINGPT_TOKEN`` configured,
    a loopback client talking to a server that is *itself* bound to loopback is
    trusted — that is the everyday `python app.py` case, and the UI needs it to
    hot-swap checkpoints without a password. Expose the server on the LAN
    (``JINGPT_HOST=0.0.0.0``) and admin closes again until a token is set.

    With a token configured the rules are strict: valid ``Bearer`` token *and*
    loopback, unless ``JINGPT_ALLOW_REMOTE_ADMIN=1`` says otherwise.
    """
    if not config.SERVER_AUTH_TOKEN:
        return _client_is_loopback() and config.host_is_loopback()
    if not config.ALLOW_REMOTE_ADMIN and not _client_is_loopback():
        return False
    return _auth_ok()


def _admin_mode() -> str:
    """``"local"`` (open on loopback) / ``"token"`` / ``"off"`` — for the UI."""
    if config.SERVER_AUTH_TOKEN:
        return "token"
    return "local" if _admin_ok() else "off"


def _admin_denied() -> tuple[Any, int]:
    if not config.SERVER_AUTH_TOKEN:
        return jsonify({"error": (
            "Admin API is closed: the server is exposed beyond loopback "
            "(JINGPT_HOST) — set JINGPT_TOKEN, or bind 127.0.0.1"
        )}), 403
    return jsonify({"error": "Forbidden: admin requires loopback + valid token"}), 403


# ── Security headers ───────────────────────────────────────────────────
@app.before_request
def _request_metadata():
    from flask import g
    g.request_id = uuid.uuid4().hex
    g.started_at = time.perf_counter()
    if request.path.startswith("/api/") and request.method == "POST":
        if not isinstance(request.get_json(silent=True), dict):
            return jsonify({"error":"JSON object required"}), 400


@app.errorhandler(ValueError)
@app.errorhandler(TypeError)
def _invalid_request(exc):
    return jsonify({"error":"Invalid request parameters"}), 400


@app.after_request
def _security_headers(response: Response) -> Response:
    from flask import g
    response.headers["X-Request-ID"] = getattr(g, "request_id", "")
    logger.info("api_request", extra={"request_id":getattr(g,"request_id",None),
        "endpoint":request.path,"status":response.status_code,
        "elapsed_ms":round((time.perf_counter()-getattr(g,"started_at",time.perf_counter()))*1000,2),
        "checkpoint_step":checkpoint_meta.get("iteration")})
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        # Only allow same-origin assets + the Google Fonts we explicitly use.
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "script-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'",
    )
    return response


# ── Model loading ──────────────────────────────────────────────────────
def load_model(
    checkpoint_path: str | None = None,
    quant: str | None = None,
    static_kv: bool = False,
    int8_kv: bool = False,
    context: int | None = None,
) -> bool:
    global model, tokenizer, checkpoint_meta, stop_token_ids
    global WEIGHT_QUANT, KV_STATIC, KV_INT8
    global CONTEXT_REQUESTED, CONTEXT_OVERRIDE
    WEIGHT_QUANT, KV_STATIC, KV_INT8 = quant, bool(static_kv), bool(int8_kv)
    CONTEXT_REQUESTED = context
    CONTEXT_OVERRIDE = config.context_override(context) if context else None
    # Resolve at CALL time so the newest chat checkpoint wins (checked into
    # this process start, not at import time).
    checkpoint_path = checkpoint_path or config.get_checkpoint_path()
    try:
        model, tokenizer, checkpoint_meta = build_inference_bundle(
            checkpoint_path,
            quant=quant,
            override_config=CONTEXT_OVERRIDE,
        )
        stop_token_ids = _stop_token_ids(tokenizer)
        _init_memory(model, tokenizer)
        print("[OK] JIN AI model loaded successfully.")
        print(
            f"[KV ] static={KV_STATIC} int8={KV_INT8} "
            f"quant={WEIGHT_QUANT or 'none'}"
        )
        print(f"[Mem] enabled={MEMORY_ENABLED} store={config.MEMORY_PATH}")
        if CONTEXT_OVERRIDE:
            print(f"[Ctx] serving {model.max_positions:,} tokens "
                  f"(trained {model.block_size:,}, RoPE "
                  f"{CONTEXT_OVERRIDE['rope_scaling']['type']} "
                  f"x{CONTEXT_OVERRIDE['rope_scaling']['factor']:.1f})")
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        print("        Train the model first with 'python train.py'.")
        model = None
        tokenizer = None
        checkpoint_meta = {}
        stop_token_ids = set()
        return False


def _stop_token_ids(tok) -> set[int]:
    if not isinstance(tok, BPETokenizer):
        return set()
    ids: set[int] = set()
    for tag in (END_TAG, EOT_TAG):
        tid = tok.special_to_id.get(tag)
        if tid is not None:
            ids.add(tid)
    return ids


# ── Long-term memory (Phase 4) ────────────────────────────────────────
def set_memory_enabled(value: bool) -> None:
    global MEMORY_ENABLED
    MEMORY_ENABLED = bool(value)


def _init_memory(mdl, tok) -> None:
    """Build the MemoryStore + HybridRetriever over persisted memory.

    Embeds every stored entry lazily on first search (index build can be
    slow for a large store); indexing happens incrementally in
    ``_memory_index()``.
    """
    global MEMORY_STORE, MEMORY_RETRIEVER
    if not MEMORY_ENABLED:
        return
    try:
        from memory import MemoryStore
        from retriever import BM25Index, HybridRetriever, VectorIndex

        store = MemoryStore(config.MEMORY_PATH)
        vec = VectorIndex(embed_dim) if (embed_dim := _memory_embed_dim(mdl)) else VectorIndex(0)
        retriever = HybridRetriever(vec, BM25Index(), texts={})
        for rec in store.all():
            if rec.get("text"):
                retriever.texts[rec["id"]] = rec["text"]
        MEMORY_STORE, MEMORY_RETRIEVER = store, retriever
        print(f"[Mem] loaded {store.stats()['total']} entries")
    except Exception as exc:  # memory is best-effort; never break the server
        print(f"[Mem] disabled (init failed): {exc}")
        MEMORY_STORE = MEMORY_RETRIEVER = None


def _memory_embed_dim(mdl) -> int:
    """Embedding dimension == model hidden size (mean-pooled ln_f)."""
    try:
        return mdl.embed_dim
    except AttributeError:
        return 0


def _memory_index() -> None:
    """Incremental index sync: embed new store entries into the retriever.

    Cheap: only embeds entries whose id is not yet indexed.
    """
    if not MEMORY_ENABLED or MEMORY_STORE is None or MEMORY_RETRIEVER is None:
        return
    try:
        from retriever import embed_texts
        store, retriever = MEMORY_STORE, MEMORY_RETRIEVER
        indexed = set(retriever.vector.ids)
        todo = [r for r in store.all() if r["id"] not in indexed and r.get("text")]
        if not todo:
            return
        texts = [r["text"] for r in todo]
        ids = [r["id"] for r in todo]
        embs = embed_texts(texts, model, tokenizer, device=config.DEVICE)
        retriever.vector.add(ids, embs)
        retriever.bm25.add(ids, texts)
        retriever.register_texts(dict(zip(ids, texts)))
        print(f"[Mem] indexed {len(ids)} new entries")
    except Exception as exc:
        print(f"[Mem] index failed (best-effort): {exc}")


def _maybe_use_memory(message: str, data: dict) -> tuple[str, list[dict]]:
    """Ground the user message with long-term memory (RAG + citations).

    Mirrors ``_maybe_run_web_search``: ``memory`` request field can be
    ``True``/``False``/``"auto"``; returns ``(rewritten_message, sources)``
    where each source carries ``title/url/snippet`` and ``kind='memory'``
    so the UI can render them like web citations.
    """
    if not MEMORY_ENABLED or MEMORY_STORE is None or MEMORY_RETRIEVER is None:
        return message, []
    # Skip the (expensive) query embedding when the store has nothing to
    # retrieve from — keeps the per-request cost near zero on a fresh store.
    if not MEMORY_STORE.all():
        return message, []
    raw_flag = data.get("memory", "auto")
    if isinstance(raw_flag, str):
        flag = raw_flag.lower()
    else:
        flag = "on" if raw_flag else "off"
    if flag in {"off", "false", "0", "no"}:
        return message, []
    if flag in {"on", "true", "1", "yes"}:
        do_retrieve = True
    else:
        # Auto: use memory for any non-trivial question.
        do_retrieve = len(message.split()) >= 3
    if not do_retrieve:
        return message, []

    try:
        # Lexical primary: this checkpoint has no trained embedding/rerank head.
        # Do not let arbitrary vector neighbors become factual evidence.
        from retriever import BM25Index, RetrievalResult
        records = {r["id"]: r for r in MEMORY_STORE.all()
                   if not r.get("needs_verification") and not r.get("contradicts")}
        bm = BM25Index()
        bm.add(list(records), [r["text"] for r in records.values()])
        hits = [RetrievalResult(mid, records[mid]["text"], score)
                for mid, score in bm.search(message, k=3)]
    except Exception as exc:
        print(f"[Mem] search failed (best-effort): {exc}")
        return message, []
    hits = [h for h in hits if h.text]
    if not hits:
        return message, []

    block_lines = [f"[{i}] {escape_external(h.text)}" for i, h in enumerate(hits[:3], 1)]
    block = "\n".join(block_lines)
    grounded = (
        f"{block}\n\n"
        f"---\n"
        f"User question: {message}\n"
        f"Answer the user using the memory entries above. If they do not "
        f"contain the answer, say so honestly. Cite sources by [number]."
    )
    sources = []
    for i, h in enumerate(hits[:3]):
        rec = MEMORY_STORE.get(h.id) or {}
        sources.append({
            "title": rec.get("claim") or h.text[:60],
            "url": rec.get("source_url") or "memory://" + h.id,
            "snippet": h.text[:200],
            "text": h.text,
            "citation_id": i + 1,
            "needs_verification": rec.get("needs_verification", False),
            "contradicts": rec.get("contradicts", []),
            "kind": "memory",
            "memory_id": h.id,
            "confidence": rec.get("confidence"),
        })
    return grounded, sources


# ── Session helpers ────────────────────────────────────────────────────
def _get_or_create_session(session_id: str | None, system_prompt: str) -> tuple[str, dict]:
    with SESSION_LOCK:
        if not session_id or session_id not in SESSIONS:
            session_id = uuid.uuid4().hex
            # If we are over the cap, evict the oldest session.
            if len(SESSIONS) >= MAX_SESSIONS:
                oldest = min(SESSIONS, key=lambda k: SESSIONS[k]["touched"])
                SESSIONS.pop(oldest, None)
            SESSIONS[session_id] = {
                "messages": [{"role": ROLE_SYSTEM, "content": system_prompt}],
                "touched": time.time(),
            }
        sess = SESSIONS[session_id]
        sess["touched"] = time.time()
        return session_id, sess


def _trim_history(messages: list[dict], max_turns: int) -> list[dict]:
    system = [m for m in messages if m["role"] == ROLE_SYSTEM][-1:]
    other = [m for m in messages if m["role"] != ROLE_SYSTEM]
    other = other[-(max_turns * 2):]
    return system + other


# ── Prefix cache (KV reuse between turns) ──────────────────────────────
def _evict_oldest_prefix_cache() -> None:
    """Drop the least-recently-touched session's cached KV if over budget."""
    if PREFIX_CACHE_MAX_SESSIONS <= 0:
        return
    with SESSION_LOCK:
        cached = {k: v.get("prefix_cache") for k, v in SESSIONS.items()}
        cached = {k: c for k, c in cached.items() if c is not None}
        if len(cached) <= PREFIX_CACHE_MAX_SESSIONS:
            return
        oldest = min(cached, key=lambda k: SESSIONS[k]["touched"])
        SESSIONS[oldest]["prefix_cache"] = None


def disable_prefix_cache() -> None:
    """Turn off cross-turn KV reuse (--no-prefix-cache)."""
    global PREFIX_CACHE_MAX_SESSIONS
    PREFIX_CACHE_MAX_SESSIONS = 0


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _reset_prefix_cache(sess: dict) -> None:
    sess["prefix_cache"] = None


def _reuse_prefix(
    sess: dict, prompt_ids: list[int]
) -> tuple[list[dict] | None, int]:
    """Return (kv_caches, skip_prefill_tokens) for prefix reuse, or (None, 0).

    Safe reuse requires the new prompt to start with exactly the previously
    cached prompt (history truncation invalidates the cache naturally via
    the common-prefix check).
    """
    cached = sess.get("prefix_cache")
    if not cached:
        return None, 0
    old_ids = cached.get("prompt_ids") or []
    kv = cached.get("kv_caches")
    common = _common_prefix_len(old_ids, prompt_ids)
    if kv is None or common < len(old_ids) or common >= len(prompt_ids):
        return None, 0
    if common == 0:
        return None, 0
    # Safety: only reuse when the cache was never trimmed (no sliding-window
    # or max_positions eviction) — otherwise the cached prefix no longer
    # starts at token 0 and reuse would silently corrupt attention.
    max_pos = getattr(model, "max_positions", 1024)
    if len(old_ids) >= max_pos or len(prompt_ids) - common >= max_pos:
        return None, 0
    return kv, common


# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


def _vram_stats() -> dict[str, Any]:
    if config.DEVICE != "cuda" or not torch.cuda.is_available():
        return {"vram_used_mb": None, "vram_total_mb": None, "gpu_name": None}
    free, total = torch.cuda.mem_get_info()
    return {
        "vram_used_mb": round((total - free) / 1e6, 1),
        "vram_total_mb": round(total / 1e6, 1),
        "gpu_name": torch.cuda.get_device_name(0),
    }


def _default_checkpoint() -> dict[str, Any]:
    """Which checkpoint a fresh start would load, and why (for the UI hint)."""
    path, why = config.resolve_checkpoint()
    return {"path": path, "name": os.path.basename(path), "why": why}


# Checkpoint headers are re-read only when a file's mtime *and* size change,
# so the 5s status poll never touches the disk beyond a stat call.
_CKPT_META_CACHE: dict[str, tuple[float, int, dict[str, Any]]] = {}
_CKPT_META_LOCK = threading.Lock()


def _probe_checkpoint(path: str) -> dict[str, Any]:
    """Read ``iteration`` / ``val_loss`` from a checkpoint without loading weights.

    ``mmap=True`` keeps the 1.7GB of tensors off the heap: only the pickle
    header is parsed, which takes ~0.03s for zip-format files. Legacy
    (non-zip) checkpoints have to be read whole, hence the cache.
    """
    try:
        info = os.stat(path)
    except OSError:
        return {}
    key = os.path.abspath(path)
    with _CKPT_META_LOCK:
        cached = _CKPT_META_CACHE.get(key)
        if cached and cached[0] == info.st_mtime and cached[1] == info.st_size:
            return cached[2]
    meta: dict[str, Any] = {}
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        step = ckpt.get("iteration")
        val = ckpt.get("val_loss")
        best = ckpt.get("best_loss")
        meta = {
            "step": int(step) if isinstance(step, (int, float)) else None,
            "val_loss": round(float(val), 4) if isinstance(val, (int, float)) else None,
            "best_loss": round(float(best), 4) if isinstance(best, (int, float)) else None,
            "trained_with": str(ckpt.get("scale") or ""),
        }
    except Exception as exc:  # unreadable file should not break the picker
        meta = {"probe_error": str(exc)[:140]}
    with _CKPT_META_LOCK:
        _CKPT_META_CACHE[key] = (info.st_mtime, info.st_size, meta)
    return meta


def _checkpoint_files(probe: bool = False) -> list[dict[str, Any]]:
    """On-disk checkpoints for the Model Lab picker.

    Walks ``checkpoints/`` *recursively* (see ``config.iter_checkpoint_paths``):
    Kaggle runs keep their weights in a folder of their own, and a flat listing
    hid them from the picker — and therefore from the UI.

    ``probe=False`` (the status poll) returns only stat-level facts; ``probe``
    additionally reads each file's header for step / val loss. Either way the
    list carries ``role`` (chat vs instruct vs pretrain), ``folder`` and
    ``default`` so the UI can explain *why* a given file is the one being
    served.
    """
    ckpt_dir = config.CHECKPOINT_DIR
    loaded = os.path.abspath(checkpoint_meta.get("_loaded_from") or "")
    default_path = os.path.abspath(config.get_checkpoint_path())
    files: list[dict[str, Any]] = []
    for path in config.iter_checkpoint_paths():
        try:
            info = os.stat(path)
        except OSError:
            continue
        try:
            folder = os.path.relpath(os.path.dirname(path), ckpt_dir)
        except ValueError:                      # different drive on Windows
            folder = os.path.dirname(path)
        folder = "" if folder == os.curdir else folder.replace(os.sep, "/")
        entry: dict[str, Any] = {
            "name": os.path.basename(path),
            "path": path,
            "folder": folder,
            "size_mb": round(info.st_size / 1e6, 1),
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(info.st_mtime)),
            "mtime_epoch": info.st_mtime,
            "role": config.checkpoint_role(os.path.basename(path)),
            "current": os.path.abspath(path) == loaded,
            "default": os.path.abspath(path) == default_path,
        }
        if probe:
            entry.update(_probe_checkpoint(path))
        files.append(entry)
    # Current first, then the auto-picked one, then the chat-capable ones
    # (instruct tunes included), then newest first — so the picker's useful
    # entries are at the top.
    files.sort(key=lambda f: (
        not f["current"], not f["default"],
        not config.is_chat_capable(f["role"]), -f["mtime_epoch"],
    ))
    return files


@app.route("/api/status")
def status():
    payload = {
        "model_loaded": model is not None,
        "vocab_size": tokenizer.vocab_size if tokenizer is not None else 0,
        "model_params": (
            f"{count_parameters(model) / 1e6:.2f}M" if model is not None else "N/A"
        ),
        "context_window": (
            getattr(model, "block_size", 0) if model is not None else 0
        ),
        "max_positions": (
            getattr(model, "max_positions", 0) if model is not None else 0
        ),
        "checkpoint_file": checkpoint_meta.get("_loaded_from"),
        "checkpoint_step": checkpoint_meta.get("iteration"),
        "best_val_loss": checkpoint_meta.get("best_loss"),
        "device": config.DEVICE,
        "kv_cache": {"static": KV_STATIC, "int8": KV_INT8},
        "quant": WEIGHT_QUANT,
        "scale": config.MODEL_SCALE,
        "tokenizer_kind": "bpe" if isinstance(tokenizer, BPETokenizer) else "char",
        "auth_required": bool(config.SERVER_AUTH_TOKEN),
        # "local" = admin is open for loopback (personal default), "token" =
        # needs JINGPT_TOKEN, "off" = server exposed and no token configured.
        "admin_mode": _admin_mode(),
        "web_search_available": True,
        # Decoding defaults the UI mirrors into its controls.
        "defaults": {
            "temperature": config.TEMPERATURE,
            "top_k": config.TOP_K,
            "top_p": config.TOP_P,
            "repetition_penalty": config.REPETITION_PENALTY,
            "max_tokens": config.MAX_NEW_TOKENS,
            "no_repeat_ngram_size": config.NO_REPEAT_NGRAM_SIZE,
            "stop_on_repeat": config.STOP_ON_REPEAT,
        },
        "uptime_s": round(time.time() - SERVER_START, 1),
        "requests_served": REQUESTS_SERVED,
        "checkpoints": _checkpoint_files(),
        # Why the current file was chosen — jingpt_best/latest are pretrain
        # artifacts, so "why" answers "why not best?" right in the UI.
        "checkpoint_default": _default_checkpoint(),
    }
    payload.update(_vram_stats())
    return jsonify(payload)


def _as_bool(value: Any, default: bool = False) -> bool:
    """Coerce JSON/query values (bool, 0/1, "on"/"true") to bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "on", "yes")


def _gen_params_from_request(data: dict) -> dict:
    max_tokens = max(1, min(int(data.get("max_tokens", config.MAX_NEW_TOKENS)), 1024))
    temperature = max(0.0, min(float(data.get("temperature", config.TEMPERATURE)), 2.0))

    raw_top_k = data.get("top_k", config.TOP_K)
    top_k = int(raw_top_k) if raw_top_k not in (None, "") else None
    if top_k is not None and top_k <= 0:
        top_k = None

    raw_top_p = data.get("top_p", config.TOP_P)
    top_p = float(raw_top_p) if raw_top_p not in (None, "") else None
    if top_p is not None:
        top_p = max(0.0, min(top_p, 1.0))

    raw_rep = data.get("repetition_penalty", config.REPETITION_PENALTY)
    repetition_penalty = (
        max(1.0, float(raw_rep)) if raw_rep not in (None, "") else 1.0
    )

    # Anti-echo: ban n-grams already seen + stop early on a looping tail.
    raw_ngram = data.get("no_repeat_ngram_size", config.NO_REPEAT_NGRAM_SIZE)
    no_repeat_ngram_size = int(raw_ngram) if raw_ngram not in (None, "") else 0
    no_repeat_ngram_size = max(0, min(no_repeat_ngram_size, 8))
    stop_on_repeat = _as_bool(data.get("stop_on_repeat"), config.STOP_ON_REPEAT)

    return {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "stop_on_repeat": stop_on_repeat,
    }


def _maybe_run_web_search(message: str, data: dict) -> tuple[str, list[dict]]:
    """Decide whether to search the web and, if so, rewrite the user
    message with grounded context.

    The behaviour is controlled by two request fields:

    - ``web_search``: ``True`` forces a search, ``False`` disables it,
      ``"auto"`` (the default) uses a heuristic based on the message.
    - ``search_max_results`` (int) and ``search_fetch_top`` (int) cap
      how aggressive the lookup is.

    Returns ``(possibly_rewritten_message, sources)``.
    """

    raw_flag = data.get("web_search", "auto")
    if isinstance(raw_flag, str):
        flag = raw_flag.lower()
    else:
        flag = "on" if raw_flag else "off"

    if flag in {"off", "false", "0", "no"}:
        return message, []
    if flag in {"on", "true", "1", "yes"}:
        do_search = True
    else:
        do_search = looks_like_search_query(message)

    if not do_search:
        return message, []

    max_results = max(1, min(int(data.get("search_max_results", 4)), 8))
    fetch_top = max(0, min(int(data.get("search_fetch_top", 1)), 3))

    block, sources = web_context(
        message,
        max_results=max_results,
        fetch_top=fetch_top,
    )
    if not sources:
        return message, []

    grounded = (
        f"{block}\n\n"
        f"---\n"
        f"User question: {message}\n"
        f"Answer the user using the search results above. If they do not "
        f"contain the answer, say so honestly. Cite sources by [number]."
    )
    return grounded, sources


def _build_prompt_ids(session_messages: list[dict]) -> list[int]:
    if isinstance(tokenizer, BPETokenizer):
        prompt_text = format_conversation(
            session_messages,
            add_generation_prompt=True,
            default_system=None,
        )
        return bounded_prompt_ids(session_messages, tokenizer, model.max_positions,
                                  reserve=64, default_system=None)
    # Legacy char-tokenizer fallback: just join with newlines.
    text = "\n".join(f"{m['role']}: {m['content']}" for m in session_messages)
    text += "\nassistant: "
    return tokenizer.encode(text)


def _collect_evidence(message: str, data: dict) -> tuple[str, list[dict]]:
    # Search each channel with the ORIGINAL query, never with injected context.
    _, web_sources = _maybe_run_web_search(message, data)
    _, mem_sources = _maybe_use_memory(message, data)
    sources = web_sources + mem_sources
    prompt, selected = grounding_prompt(message, sources)
    return (prompt if selected else escape_external(message)), sources


def _arithmetic_answer(message: str):
    import re
    from tools import calculate, CalculatorError
    from grounding import EvidenceAnswer
    expr = re.sub(r"^(?:calculate|compute|evaluate|tính|what is|kết quả của)\s*:?\s*", "", message, flags=re.I).rstrip("?= ")
    if not re.fullmatch(r"[\d\s.+*/()%\-]+", expr) or not re.search(r"[+*/%\-]", expr):
        return None
    try:
        return EvidenceAnswer(calculate(expr), [], "tool_verified", [], "verified_calculator")
    except CalculatorError as exc:
        return EvidenceAnswer("Calculation failed safely.", [], "tool_failed", [str(exc)], "verified_calculator")


def _grounded_chat_response(message, sources, session_id, sess, *, streaming=False, verified_result=None):
    t0 = time.perf_counter()
    result = verified_result or grounded_answer(message, sources)
    sess["messages"].extend([{"role": ROLE_USER, "content": escape_external(message)},
                             {"role": ROLE_ASSISTANT, "content": result.answer}])
    sess["messages"] = _trim_history(sess["messages"], MAX_TURNS_PER_SESSION)
    _reset_prefix_cache(sess)
    payload = {"response": result.answer, "reply": result.answer,
               "session_id": session_id, "sources": result.sources,
               "grounding_status": result.status, "issues": result.issues,
               "answer_mode": result.mode, "tokens_generated": 0,
               "stop_reason": "evidence", "elapsed_s": round(time.perf_counter()-t0, 4),
               "web_search_used": any(s.get("kind") != "memory" for s in result.sources)}
    logger.info("grounded_answer", extra={"session_id": session_id,
                "grounding_status": result.status, "source_count": len(result.sources)})
    if not streaming:
        return jsonify(payload)
    def events():
        yield f"event: session\ndata: {json.dumps({'session_id':session_id})}\n\n"
        yield f"event: sources\ndata: {json.dumps({'sources':result.sources})}\n\n"
        yield f"data: {json.dumps({'delta':result.answer})}\n\n"
        yield f"event: done\ndata: {json.dumps(dict(payload, tokens=0))}\n\n"
    return Response(events(), mimetype="text/event-stream", headers={"Cache-Control":"no-cache"})


@app.route("/api/agent", methods=["POST"])
def agent_chat():
    """Bounded tool agent. No host Python or implicit persistent memory writes."""
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    if not _rate_limit_ok():
        return jsonify({"error": "Rate limit exceeded"}), 429
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object required"}), 400
    message = str(data.get("message", "")).strip()[:config.MAX_USER_MESSAGE_CHARS]
    if not message:
        return jsonify({"error": "Empty message"}), 400
    arithmetic = _arithmetic_answer(message)
    if arithmetic is not None:
        return jsonify(dict(arithmetic.to_dict(), answer_mode=arithmetic.mode,
                       trace=[{"type":"tool_result", "name":"calculator", "text":arithmetic.answer}]))
    _, sources = _collect_evidence(message, data)
    if sources or _as_bool(data.get("grounded")):
        result = grounded_answer(message, sources)
        return jsonify(dict(result.to_dict(), trace=[]))
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 503
    from agent import Agent
    with _model_lock:
        runtime = Agent(model=model, tokenizer=tokenizer, memory=MEMORY_STORE,
                        search_enabled=_as_bool(data.get("web_search"), False),
                        max_tool_calls=4, device=config.DEVICE, rigor="normal",
                        max_new_tokens=128)
        result = runtime.run(message)
    _count_request()
    return jsonify(dict(result, answer_mode="model_tool_agent"))


@app.route("/api/chat", methods=["POST"])
def chat():
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    if not _rate_limit_ok():
        return jsonify({"error": "Rate limit exceeded"}), 429
    _count_request()
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object required"}), 400
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    if len(message) > config.MAX_USER_MESSAGE_CHARS:
        message = message[: config.MAX_USER_MESSAGE_CHARS]

    system_prompt = str(data.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)[:1000]
    session_id, sess = _get_or_create_session(data.get("session_id"), system_prompt)

    arithmetic = _arithmetic_answer(message)
    if arithmetic is not None:
        return _grounded_chat_response(message, [], session_id, sess,
            streaming=request.path.endswith("/stream"), verified_result=arithmetic)
    grounded_message, sources = _collect_evidence(message, data)
    strict = _as_bool(data.get("grounded"), True)
    web_flag = str(data.get("web_search", "auto")).lower()
    evidence_requested = (_as_bool(data.get("web_search")) or
                          (web_flag == "auto" and looks_like_search_query(message)) or
                          _as_bool(data.get("memory")) or _as_bool(data.get("grounded")))
    if strict and (sources or evidence_requested):
        return _grounded_chat_response(message, sources, session_id, sess, streaming=request.path.endswith("/stream"))
    sess["messages"].append({"role": ROLE_USER, "content": grounded_message})
    sess["messages"] = _trim_history(sess["messages"], MAX_TURNS_PER_SESSION)

    prompt_ids = _build_prompt_ids(sess["messages"])
    if not prompt_ids:
        return jsonify({"error": "Prompt encoded to zero tokens."}), 400

    params = _gen_params_from_request(data)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=config.DEVICE)

    kv_caches, skip_prefill = _reuse_prefix(sess, prompt_ids)
    if kv_caches is None:
        kv_caches = model.empty_kv_cache(static=KV_STATIC, int8=KV_INT8)

    gen_stats: dict[str, Any] = {}
    t0 = time.perf_counter()
    with _model_lock, torch.inference_mode():
        output = model.generate(
            idx,
            stop_ids=stop_token_ids if stop_token_ids else None,
            kv_caches=kv_caches,
            skip_prefill_tokens=skip_prefill,
            static_kv=KV_STATIC,
            kv_int8=KV_INT8,
            keep_ids=stop_token_ids or None,
            stats=gen_stats,
            **params,
        )
    elapsed = time.perf_counter() - t0

    # Remember the KV so the next turn only prefills its new tail.
    sess["prefix_cache"] = {"prompt_ids": prompt_ids, "kv_caches": kv_caches}
    _evict_oldest_prefix_cache()

    new_ids = output[0].tolist()[len(prompt_ids):]
    if stop_token_ids and new_ids and new_ids[-1] in stop_token_ids:
        new_ids = new_ids[:-1]
    reply = tokenizer.decode(new_ids).strip() if new_ids else ""
    if isinstance(tokenizer, BPETokenizer):
        for tag in ROLE_TAG.values():
            cut = reply.find(tag)
            if cut != -1:
                reply = reply[:cut].strip()
    # Cut an echo tail the loop breaker could only catch one copy late.
    reply = trim_repeated_tail(reply)

    if not reply:
        reply = "(no reply)"
    sess["messages"].append({"role": ROLE_ASSISTANT, "content": reply})

    stop_reason = gen_stats.get("stop_reason")
    return jsonify(
        {
            "response": reply,
            "session_id": session_id,
            "checkpoint_step": checkpoint_meta.get("iteration"),
            "tokens_generated": len(new_ids),
            "stop_reason": stop_reason,
            "stop_label": stop_reason_label(stop_reason),
            "elapsed_s": round(elapsed, 3),
            "tok_per_s": round(len(new_ids) / elapsed, 1) if elapsed > 0 else None,
            "sources": [
                {"title": s.get("title"), "url": s.get("url"), "snippet": s.get("snippet")}
                for s in sources
            ],
            "web_search_used": bool(sources),
        }
    )


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    if not _rate_limit_ok():
        return jsonify({"error": "Rate limit exceeded"}), 429
    _count_request()
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object required"}), 400
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    if len(message) > config.MAX_USER_MESSAGE_CHARS:
        message = message[: config.MAX_USER_MESSAGE_CHARS]

    system_prompt = str(data.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)[:1000]
    session_id, sess = _get_or_create_session(data.get("session_id"), system_prompt)

    arithmetic = _arithmetic_answer(message)
    if arithmetic is not None:
        return _grounded_chat_response(message, [], session_id, sess,
            streaming=request.path.endswith("/stream"), verified_result=arithmetic)
    grounded_message, sources = _collect_evidence(message, data)
    strict = _as_bool(data.get("grounded"), True)
    web_flag = str(data.get("web_search", "auto")).lower()
    evidence_requested = (_as_bool(data.get("web_search")) or
                          (web_flag == "auto" and looks_like_search_query(message)) or
                          _as_bool(data.get("memory")) or _as_bool(data.get("grounded")))
    if strict and (sources or evidence_requested):
        return _grounded_chat_response(message, sources, session_id, sess, streaming=request.path.endswith("/stream"))
    sess["messages"].append({"role": ROLE_USER, "content": grounded_message})
    sess["messages"] = _trim_history(sess["messages"], MAX_TURNS_PER_SESSION)

    prompt_ids = _build_prompt_ids(sess["messages"])
    params = _gen_params_from_request(data)

    def event_stream():
        yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"
        if sources:
            light_sources = [
                {"title": s.get("title"), "url": s.get("url"), "snippet": s.get("snippet")}
                for s in sources
            ]
            yield f"event: sources\ndata: {json.dumps({'sources': light_sources})}\n\n"

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=config.DEVICE)
        kv_caches, skip_prefill = _reuse_prefix(sess, prompt_ids)
        if kv_caches is None:
            kv_caches = model.empty_kv_cache(static=KV_STATIC, int8=KV_INT8)
        collected: list[int] = []
        decode_buffer = bytearray()
        gen_stats: dict[str, Any] = {}
        t0 = time.perf_counter()
        first_token_ms: float | None = None

        with _model_lock, torch.inference_mode():
            try:
                for tok in model.stream_generate(
                    idx,
                    stop_ids=stop_token_ids if stop_token_ids else None,
                    kv_caches=kv_caches,
                    skip_prefill_tokens=skip_prefill,
                    static_kv=KV_STATIC,
                    kv_int8=KV_INT8,
                    keep_ids=stop_token_ids or None,
                    stats=gen_stats,
                    **params,
                ):
                    collected.append(tok)
                    if first_token_ms is None:
                        # Prefill+first sample: the number that actually feels
                        # like "how fast is it" while streaming.
                        first_token_ms = (time.perf_counter() - t0) * 1000.0
                    if isinstance(tokenizer, BPETokenizer):
                        token_bytes = tokenizer.vocab.get(int(tok), b"")
                        decode_buffer.extend(token_bytes)
                        try:
                            chunk = decode_buffer.decode("utf-8")
                            decode_buffer.clear()
                        except UnicodeDecodeError:
                            # Wait for the next token — we are mid multi-byte char.
                            continue
                    else:
                        chunk = tokenizer.decode([tok])
                    if not chunk:
                        continue
                    yield f"data: {json.dumps({'delta': chunk})}\n\n"
            except GeneratorExit:
                return
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
                return

        reply = tokenizer.decode(collected).strip() if collected else ""
        if isinstance(tokenizer, BPETokenizer):
            for tag in ROLE_TAG.values():
                cut = reply.find(tag)
                if cut != -1:
                    reply = reply[:cut].strip()
        reply = trim_repeated_tail(reply)
        if not reply:
            reply = "(no reply)"
        sess["messages"].append({"role": ROLE_ASSISTANT, "content": reply})
        sess["prefix_cache"] = {"prompt_ids": prompt_ids, "kv_caches": kv_caches}
        _evict_oldest_prefix_cache()
        elapsed = time.perf_counter() - t0
        stop_reason = gen_stats.get("stop_reason")
        yield "event: done\ndata: %s\n\n" % json.dumps({
            "tokens": len(collected),
            # Authoritative text: the streamed deltas can contain a trailing
            # role tag or an echo tail that only the final pass can strip.
            "reply": reply,
            "stop_reason": stop_reason,
            "stop_label": stop_reason_label(stop_reason),
            "elapsed_s": round(elapsed, 3),
            "tok_per_s": round(len(collected) / elapsed, 1) if elapsed > 0 else None,
            "first_token_ms": round(first_token_ms, 1) if first_token_ms else None,
        })

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/search", methods=["POST"])
def search_only():
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    if not _rate_limit_ok():
        return jsonify({"error": "Rate limit exceeded"}), 429

    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    if len(query) > config.MAX_USER_MESSAGE_CHARS:
        query = query[: config.MAX_USER_MESSAGE_CHARS]

    max_results = max(1, min(int(data.get("max_results", 5)), 10))
    fetch_top = max(0, min(int(data.get("fetch_top", 0)), 3))

    try:
        block, sources = web_context(
            query, max_results=max_results, fetch_top=fetch_top
        )
    except Exception as exc:
        return jsonify({"error": f"Search failed: {exc}"}), 502
    return jsonify({"query": query, "context": block, "sources": sources})


@app.route("/api/reset", methods=["POST"])
def reset_session():
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    with SESSION_LOCK:
        if session_id and session_id in SESSIONS:
            SESSIONS.pop(session_id)
    return jsonify({"ok": True})


@app.route("/api/kv-cache", methods=["POST"])
def clear_kv_cache():
    """Drop all cached KV prefixes (free VRAM / force full prefill)."""
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    with SESSION_LOCK:
        for sess in SESSIONS.values():
            sess["prefix_cache"] = None
    return jsonify({"ok": True})


# ── Long-term memory API (Phase 4) ────────────────────────────────────
@app.route("/api/memory/search", methods=["POST"])
def memory_search():
    """Retrieve top-k memory entries for a query (RAG-style)."""
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    if MEMORY_STORE is None or MEMORY_RETRIEVER is None:
        return jsonify({"error": "Memory disabled (--no-memory or init failed)"}), 503
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    k = max(1, min(int(data.get("k", 5)), 20))
    try:
        from retriever import BM25Index, RetrievalResult
        entries = {r["id"]:r for r in MEMORY_STORE.all()
                   if not r.get("needs_verification") and not r.get("contradicts")}
        index = BM25Index()
        index.add(list(entries), [r["text"] for r in entries.values()])
        hits = [RetrievalResult(mid, entries[mid]["text"],score) for mid,score in index.search(query,k=k)]
    except Exception as exc:
        return jsonify({"error": f"Memory search failed: {exc}"}), 500
    results = []
    for h in hits:
        rec = MEMORY_STORE.get(h.id) or {}
        results.append({
            "id": h.id,
            "text": h.text,
            "claim": rec.get("claim"),
            "source_url": rec.get("source_url"),
            "license": rec.get("license"),
            "confidence": rec.get("confidence"),
            "fetched_at": rec.get("fetched_at"),
            "score": round(h.score, 4),
        })
    return jsonify({"results": results, "count": len(results)})


@app.route("/api/memory/add", methods=["POST"])
def memory_add():
    """Add one knowledge entry (semantic) with optional provenance."""
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    if MEMORY_STORE is None:
        return jsonify({"error": "Memory disabled"}), 503
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    tier = str(data.get("tier", "semantic"))
    try:
        mid = MEMORY_STORE.add(
            tier=tier,
            text=text,
            claim=data.get("claim"),
            source_url=data.get("source_url"),
            license_=data.get("license"),
            confidence=float(data.get("confidence", 0.5)),
            verified_by=data.get("verified_by"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    # Lexical retrieval sees the persistent store immediately; no GPU work
    # or hidden-state forward hook is needed for memory writes.
    return jsonify({"ok": True, "id": mid})


@app.route("/api/memory/correct", methods=["POST"])
def memory_correct():
    if not _admin_ok():
        return _admin_denied()
    if MEMORY_STORE is None:
        return jsonify({"error":"Memory disabled"}), 503
    data = request.get_json()
    mid = MEMORY_STORE.correct(str(data.get("id", "")), text=str(data.get("text", "")),
        verified_by=str(data.get("verified_by", "")), source_url=data.get("source_url"),
        confidence=float(data.get("confidence", .5)))
    return jsonify({"ok":True,"id":mid})


@app.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    if MEMORY_STORE is None:
        return jsonify({"error": "Memory disabled"}), 503
    stats = MEMORY_STORE.stats()
    stats["index"] = MEMORY_RETRIEVER.vector.stats() if MEMORY_RETRIEVER else {}
    return jsonify(stats)


@app.route("/api/memory/forget", methods=["POST"])
def memory_forget():
    """Delete by id or by source URL (privacy: forget what Jin learned)."""
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    if MEMORY_STORE is None:
        return jsonify({"error": "Memory disabled"}), 503
    data = request.get_json(silent=True) or {}
    mid = data.get("id")
    url = data.get("source_url")
    if not mid and not url:
        return jsonify({"error": "Provide id or source_url"}), 400
    # Figure out which ids will be removed FIRST so both indexes can be
    # synced (vector AND bm25) — deleting by source_url must not leave the
    # entries searchable.
    affected = [
        r["id"] for r in MEMORY_STORE.all()
        if (mid and r["id"] == mid)
        or (url and r.get("source_url") == url)
    ]
    removed = MEMORY_STORE.delete(mid=mid, source_url=url)
    if removed and MEMORY_RETRIEVER is not None:
        gone = set(affected)
        MEMORY_RETRIEVER.vector.remove(gone)
        MEMORY_RETRIEVER.bm25.remove(gone)
    return jsonify({"ok": True, "removed": removed})


@app.route("/api/memory/consolidate", methods=["POST"])
def memory_consolidate():
    """Run one consolidation pass (merge dups, decay, flag contradictions)."""
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    if MEMORY_STORE is None:
        return jsonify({"error": "Memory disabled"}), 503
    report = MEMORY_STORE.consolidate()
    return jsonify(report)


# ── Admin API (Phase 7) — token-gated + loopback-only ─────────────────

_LEARN_LOCK = threading.Lock()
_LEARN_RUNNING = False


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@app.route("/api/bench/history", methods=["GET"])
def bench_history():
    """Bench run history for the Growth Dashboard (val loss, probes, tok/s)."""
    if not _admin_ok():
        return _admin_denied()
    import bench as bench_mod
    hist_path = os.path.join(bench_mod.BENCH_DIR, "history.json")
    history = _read_json(hist_path)
    if not isinstance(history, list):
        history = []
    return jsonify({"runs": history, "total": len(history)})


@app.route("/api/learn/status", methods=["GET"])
def learn_status():
    """Live growth stats: frontier, ledger summary, memory, registry."""
    if not _admin_ok():
        return _admin_denied()
    frontier = _read_json(config.CURIOSITY_FRONTIER_PATH)
    ledger_summary: dict = {}
    try:
        from knowledge_ledger import KnowledgeLedger
        ledger_summary = KnowledgeLedger().stats()
    except Exception:
        ledger_summary = {"error": "ledger unavailable"}
    registry = {}
    try:
        from curiosity_engine import load_registry
        registry = load_registry()
    except Exception:
        registry = {"error": "registry unavailable"}
    status = {
        "frontier": frontier if isinstance(frontier, list) else [],
        "ledger": ledger_summary,
        "registry": registry,
        "memory": MEMORY_STORE.stats() if MEMORY_STORE is not None else None,
        "budget": {
            "max_requests": config.CURIOSITY_MAX_REQUESTS,
            "max_gb_day": config.CURIOSITY_MAX_GB_DAY,
            "max_cycles": config.CURIOSITY_MAX_CYCLES,
            "replay_ratio": config.CURIOSITY_REPLAY_RATIO,
        },
        "running": _LEARN_RUNNING,
    }
    return jsonify(status)


@app.route("/api/learn/trigger", methods=["POST"])
def learn_trigger():
    """Run ONE curiosity cycle (harvest -> verify -> self-edit -> ledger).

    Safely dry-run: with the train provider not yet wired (Phase 5 debt),
    this exercises the full loop end-to-end (real web harvest + claim
    verification + SEAL self-edit + ledger + gate simulation) WITHOUT
    touching model weights. Set ``dry_run=false`` in the body once
    ``train.py --mode curiosity`` exists.
    """
    if not _admin_ok():
        return _admin_denied()
    global _LEARN_RUNNING
    data = request.get_json(silent=True) or {}
    topic = str(data.get("topic", "")).strip()
    if not topic:
        return jsonify({"error": "Provide a topic"}), 400
    if _LEARN_RUNNING:
        return jsonify({"error": "A learning cycle is already running"}), 409

    dry_run = bool(data.get("dry_run", True))  # default safe

    def _worker() -> None:
        global _LEARN_RUNNING
        try:
            from learn_daemon import run_daemon
            run_daemon([topic], cycles=1, dry_run=dry_run, device="cpu")
        except Exception as exc:  # never take the server down
            import traceback
            traceback.print_exc()
            print(f"[learn/trigger] cycle failed: {exc}")
        finally:
            _LEARN_RUNNING = False

    _LEARN_RUNNING = True
    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({
        "ok": True, "started": True, "topic": topic, "dry_run": dry_run,
        "note": "cycle runs in background; poll /api/learn/status",
    })


@app.route("/api/checkpoints", methods=["GET"])
def checkpoints_list():
    """Registry + on-disk checkpoints (promote/rollback targets)."""
    if not _admin_ok():
        return _admin_denied()
    from curiosity_engine import load_registry
    registry = load_registry()
    files = []
    for path in config.iter_checkpoint_paths():
        try:
            size_mb = round(os.path.getsize(path) / 1e6, 1)
        except OSError:
            continue
        files.append({"name": os.path.basename(path), "path": path,
                      "role": config.checkpoint_role(os.path.basename(path)),
                      "size_mb": size_mb})
    return jsonify({"registry": registry, "files": files})


@app.route("/api/checkpoints/promote", methods=["POST"])
def checkpoints_promote():
    """Promote a checkpoint into the registry (after it passed the gate)."""
    if not _admin_ok():
        return _admin_denied()
    data = request.get_json(silent=True) or {}
    path = str(data.get("path", "")).strip()
    if not path:
        return jsonify({"error": "Provide checkpoint path"}), 400
    if not os.path.exists(path):
        return jsonify({"error": f"Checkpoint not found: {path}"}), 404
    changelog = data.get("changelog") or {}
    if not isinstance(changelog, dict):
        changelog = {"note": str(changelog)}
    from curiosity_engine import promote_checkpoint
    registry = promote_checkpoint(path, changelog=changelog)
    return jsonify({"ok": True, "registry": registry})


@app.route("/api/checkpoints/rollback", methods=["POST"])
def checkpoints_rollback():
    """Roll back to a previous checkpoint; records the action in the ledger.

    The registry keeps the last N promoted checkpoints, so rollback is a
    promotion of an older entry annotated with ``action: rollback`` — the
    gate already guarantees the newer weights never silently win.
    """
    if not _admin_ok():
        return _admin_denied()
    data = request.get_json(silent=True) or {}
    path = str(data.get("path", "")).strip()
    if not path:
        return jsonify({"error": "Provide checkpoint path to roll back to"}), 400
    if not os.path.exists(path):
        return jsonify({"error": f"Checkpoint not found: {path}"}), 404
    from curiosity_engine import promote_checkpoint
    registry = promote_checkpoint(path, changelog={
        "action": "rollback",
        "reason": str(data.get("reason", "manual rollback")),
        "by": "api",
    })
    return jsonify({"ok": True, "rolled_back_to": path, "registry": registry})


@app.route("/api/model/checkpoints", methods=["GET"])
def model_checkpoints():
    """Model Lab picker with real headers (step / val loss).

    Separate from ``/api/status`` on purpose: reading every checkpoint header
    costs seconds the first time (legacy non-zip files must be read whole), so
    the 5s poll stays stat-only and only the picker pays for the probe. Results
    are cached until a file's mtime/size changes.
    """
    return jsonify({
        "checkpoints": _checkpoint_files(probe=True),
        "current": checkpoint_meta.get("_loaded_from"),
        "default": _default_checkpoint(),
    })


@app.route("/api/model/reload", methods=["POST"])
def model_reload():
    """Hot-swap the served checkpoint — A/B two weights without a restart.

    The day-to-day use is comparing ``jingpt_chat.pt`` (newest) against a
    golden copy (``jingpt_chat_46k_sft4k.pt``) from the same browser tab.
    Admin-gated (loopback, plus JINGPT_TOKEN when one is configured) because the
    caller supplies a path and the process replaces 1.7GB of live weights;
    generation waits on the model lock, so an in-flight reply finishes first.
    """
    if not _admin_ok():
        return _admin_denied()
    data = request.get_json(silent=True) or {}
    raw_path = str(data.get("path", "")).strip()
    path: str | None = os.path.abspath(raw_path) if raw_path else None
    if path is not None and (not path.endswith(".pt") or not os.path.isfile(path)):
        return jsonify({"error": f"Not a checkpoint file: {path}"}), 404

    t0 = time.perf_counter()
    with _model_lock:
        ok = load_model(path, quant=WEIGHT_QUANT, static_kv=KV_STATIC,
                        int8_kv=KV_INT8, context=CONTEXT_REQUESTED)
    if not ok:
        return jsonify({"error": "Failed to load checkpoint (check server log)"}), 500
    # Cached KV prefixes belong to the previous weights.
    with SESSION_LOCK:
        for sess in SESSIONS.values():
            sess["prefix_cache"] = None
    return jsonify({
        "ok": True,
        "checkpoint_file": checkpoint_meta.get("_loaded_from"),
        "checkpoint_step": checkpoint_meta.get("iteration"),
        "best_val_loss": checkpoint_meta.get("best_loss"),
        "model_params": (
            f"{count_parameters(model) / 1e6:.2f}M" if model is not None else None
        ),
        "load_s": round(time.perf_counter() - t0, 2),
        "checkpoints": _checkpoint_files(),
        "checkpoint_default": _default_checkpoint(),
    })


# ── Error handlers ─────────────────────────────────────────────────────
@app.errorhandler(413)
def _too_large(_exc):
    return jsonify({"error": "Request body too large"}), 413


@app.errorhandler(404)
def _not_found(_exc):
    return jsonify({"error": "Not found"}), 404


# ── Background session GC ──────────────────────────────────────────────
def _session_gc_loop():
    while True:
        time.sleep(300)
        cutoff = time.time() - 60 * 30
        with SESSION_LOCK:
            stale = [k for k, v in SESSIONS.items() if v["touched"] < cutoff]
            for k in stale:
                SESSIONS.pop(k, None)


def _start_background_workers() -> None:
    threading.Thread(target=_session_gc_loop, daemon=True).start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JIN AI web server")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a trained checkpoint "
                             "(default: jingpt_chat.pt, else jingpt_best.pt).")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="NAME",
        choices=config.DEVICE_CHOICES,
        help="Device override. Default: JINGPT_DEVICE or auto (cuda if present).",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=config.WEIGHT_QUANT or "none",
        choices=["none", "int8", "int4"],
        help="Weight-only quantization (int8 per-channel / int4 group-128).",
    )
    parser.add_argument("--static-kv", action="store_true",
                        help="Pre-allocated KV buffers (no per-step cat).")
    parser.add_argument("--int8-kv", action="store_true",
                        help="int8 per-head-scale KV cache (4x less memory).")
    parser.add_argument("--no-prefix-cache", action="store_true",
                        help="Disable cross-turn KV reuse.")
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable long-term memory (v2.5 behaviour).")
    parser.add_argument("--ctx", type=int, default=None, metavar="N",
                        help="Serve a longer context than trained via RoPE "
                             "scaling (e.g. --ctx 2048). Untrained extension: "
                             "measure with `python bench.py --ctx N`.")
    args = parser.parse_args()

    if args.no_prefix_cache:
        disable_prefix_cache()
    if args.no_memory:
        set_memory_enabled(False)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonEventFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    # Without this the module keeps ``config.DEVICE == "cpu"`` (import-time
    # placeholder) and the whole web UI serves on CPU even when a GPU exists.
    config.set_device(args.device)
    configure_cuda_performance(config.DEVICE)

    load_model(
        args.checkpoint,
        quant=None if args.quant == "none" else args.quant,
        static_kv=args.static_kv,
        int8_kv=args.int8_kv,
        context=args.ctx,
    )
    _start_background_workers()
    print("\n" + "=" * 60)
    host = config.SERVER_HOST
    print(f"JIN AI Web UI -> http://{host}:{config.SERVER_PORT}")
    if config.SERVER_AUTH_TOKEN:
        print("  (Bearer token required: set JINGPT_TOKEN to authorise)")
    print("=" * 60 + "\n")
    app.run(host=host, port=config.SERVER_PORT, debug=False, threaded=True)

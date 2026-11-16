"""
JIN AI configuration (v3).

The active architecture is selected by ``MODEL_SCALE``. Shipped presets:

    "gpt2_small_reference" ~124M parameters (GPT-2 Small-compatible, from scratch)
    "jin_small_modern"     ~143M parameters (JIN modern architecture)
    "tiny"     ~ 35M   parameters  (laptop / smoke test)
    "small"    ~143M   parameters  (the existing v2.5 checkpoint)
    "medium"   ~360M   parameters  (GQA + QK-norm, grad-checkpointing)
    "large"    ~1.0B   parameters  (needs a real GPU)
    "xl-moe"   ~0.77B  total / ~0.30B active (MoE 12L, expert offload for 6GB VRAM)

Override at runtime with the ``JINGPT_SCALE`` environment variable.

Depth-based auto-scaling (the "one dial"): ``python config.py --describe
--depth N`` derives width/heads/KV-heads/LR/tokens from a single depth
number, following the nanochat-style philosophy that one dial drives
everything else.

Compute device: ``JINGPT_DEVICE`` (``auto`` default -> CUDA -> MPS -> CPU),
``resolve_device()``, ``set_device()`` (CLI ``--device`` overrides).
"""

from __future__ import annotations

import argparse
import math
import os
import warnings

from storage import configure_runtime, output_path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Apply before torch initializes compiler/download caches. Process-local only.
STORAGE_ENV = configure_runtime(ROOT_DIR)

import torch as _torch

# ── Paths ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(ROOT_DIR, "data", "train.txt")
CHAT_DATA_PATH = os.path.join(ROOT_DIR, "data", "chat_train.txt")
SEED_CONVERSATIONS_PATH = os.path.join(ROOT_DIR, "data", "seed", "conversations.jsonl")
VOCAB_PATH = os.path.join(ROOT_DIR, "vocab.json")
BPE_PATH = os.path.join(ROOT_DIR, "bpe.json")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
CURATED_DATA_DIR = os.path.join(ROOT_DIR, "data", "curated")
DATA_SNAPSHOT_DIR = os.path.join(CURATED_DATA_DIR, "snapshots")
LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "jingpt_latest.pt")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "jingpt_best.pt")
CHAT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "jingpt_chat.pt")
LEGACY_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "jingpt.pt")


CHECKPOINT_ENV = "JINGPT_CHECKPOINT"

# Kaggle runs land in their own folders — ``checkpoints/kaggle_v4_60020/`` and
# even ``checkpoints/pretrain/<run>/jin/checkpoints/`` — so a flat ``listdir``
# (*and* the old resolver) hid the newest weights from both the picker and the
# startup pick. The walk below reads every ``.pt`` under ``CHECKPOINT_DIR``.
CHECKPOINT_WALK_DEPTH = 6

#: Roles that saw the chat template, i.e. can hold a conversation instead of
#: continuing raw text.
CHAT_ROLES = frozenset({"chat", "instruct"})


def checkpoint_role(name: str) -> str:
    """Classify a checkpoint filename so the UI can say what it actually is.

    ``jingpt_best.pt`` / ``jingpt_latest.pt`` sound like "the good ones" but
    they are *pretrain* artifacts: they never saw the chat template, so they
    answer with raw text continuation. Instruct tunes (``jingpt_chat*`` and the
    Kaggle ``*instruct*.pt``/``*pretrain*.pt`` family) are classified by name so
    the resolver below can put the chat-capable ones first.
    """
    stem = name[:-3] if name.endswith(".pt") else name
    low = stem.lower()
    if stem.startswith("jingpt_chat"):
        return "chat"
    if "instruct" in low:
        return "instruct"
    if stem == "jingpt_best":
        return "pretrain-best"
    if stem == "jingpt_latest":
        return "pretrain-latest"
    if "pretrain" in low:
        return "pretrain-run"
    return "legacy"


def is_chat_capable(role: str) -> bool:
    """True when ``role`` knows the chat template (not raw continuation)."""
    return role in CHAT_ROLES


def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:      # deleted mid-walk — keep sorting instead of raising
        return 0.0


def iter_checkpoint_paths(max_depth: int = CHECKPOINT_WALK_DEPTH) -> list[str]:
    """Every ``.pt`` under :data:`CHECKPOINT_DIR`, subfolders included.

    Newest first, so callers that only want "the latest" read ``paths[0]``.
    Hidden folders and ``__pycache__`` are skipped; the depth cap keeps a deep
    Kaggle export (``pretrain/<run>/jin/checkpoints/``) in reach without
    walking data folders too far.
    """
    root = CHECKPOINT_DIR
    if not os.path.isdir(root):
        return []
    found: list[str] = []
    base_depth = os.path.abspath(root).rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.abspath(dirpath).rstrip(os.sep).count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d != "__pycache__"
        )
        for name in sorted(filenames):
            if name.endswith(".pt"):
                found.append(os.path.join(dirpath, name))
    found.sort(key=_safe_mtime, reverse=True)
    return found


def chat_snapshot_paths() -> list[str]:
    """``jingpt_chat_*.pt`` snapshots (golden copies), newest first."""
    chat_main = os.path.abspath(CHAT_CHECKPOINT_PATH)
    return [
        path
        for path in iter_checkpoint_paths()
        if os.path.basename(path).startswith("jingpt_chat")
        and os.path.abspath(path) != chat_main
    ]


def servable_checkpoint_paths() -> list[str]:
    """Chat-capable checkpoints (instruct tunes), newest first.

    ``jingpt_chat.pt`` is one of these, not a privileged one: the newest file
    wins, which is how a freshly downloaded Kaggle v4 instruct tune takes over
    from the older local chat checkpoint on the next start.
    """
    return [
        path
        for path in iter_checkpoint_paths()
        if is_chat_capable(checkpoint_role(os.path.basename(path)))
    ]


def resolve_checkpoint() -> tuple[str, str]:
    """Return ``(path, why)`` for the checkpoint this process should serve.

    Order (first hit wins):
      1. ``JINGPT_CHECKPOINT`` env var — explicit override, no guessing.
      2. newest chat-capable checkpoint anywhere under ``checkpoints/``:
         ``jingpt_chat.pt``, a ``jingpt_chat_*.pt`` snapshot, or an instruct
         tune copied in from a Kaggle run (``kaggle_v4_instruct.pt``). Newest
         by mtime, so a fresh download is picked up on the next start.
      3. ``jingpt_best.pt`` — last resort, pretrain only (its answers are text
         continuation, not chat).
    """
    override = os.environ.get(CHECKPOINT_ENV, "").strip()
    if override:
        path = override if os.path.isabs(override) else os.path.join(ROOT_DIR, override)
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path, CHECKPOINT_ENV
    servable = servable_checkpoint_paths()
    if servable:
        path = servable[0]
        if os.path.abspath(path) == os.path.abspath(CHAT_CHECKPOINT_PATH):
            return path, "chat"
        if os.path.basename(path).startswith("jingpt_chat"):
            return path, "chat-snapshot"
        return path, "instruct"
    return BEST_CHECKPOINT_PATH, "pretrain-fallback"


def get_checkpoint_path() -> str:
    """Resolve the best available checkpoint at call time, not import time."""
    return resolve_checkpoint()[0]


# ``CHECKPOINT_PATH`` is resolved lazily (PEP 562) so every entry point reading
# ``config.CHECKPOINT_PATH`` gets the best available checkpoint at process
# start: the newest chat-capable file (``jingpt_chat.pt``, a snapshot, or an
# instruct tune from a Kaggle run), else ``jingpt_best.pt``. As a plain
# constant it made ``python app.py`` /
# ``python chat.py`` / ``generate.py`` silently serve the *pretrain*
# checkpoint even after instruct tuning.


def __getattr__(name: str) -> str:
    if name == "CHECKPOINT_PATH":
        return get_checkpoint_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ── Tokenizer ──────────────────────────────────────────────────────────
TOKENIZER_KIND = "bpe"
BPE_VOCAB_SIZE = 16_384

# ── Model scale presets ────────────────────────────────────────────────
# Every preset carries the v3 knobs; defaults reproduce the v2.5 behaviour,
# and old checkpoints load unchanged because they carry their own config.
SCALE_PRESETS: dict[str, dict] = {
    # ── GPT-2 Small reference (Phase 1) ────────────────────────────────
    # Exact GPT-2 Small shape: 12L / 12H / 768d / 3072 FFN / 1024 ctx /
    # 50257 vocab / learned absolute positions / pre-LN / GELU(tanh) /
    # tied token embedding + LM head.  Measured count with vocab 50257:
    # 124,439,808 parameters (see tests/test_gpt2_reference.py).
    # ``modern=False`` selects the classic block; ``gelu_tanh`` and
    # ``resid_scaled_init`` are the two GPT-2 init/activation details the
    # classic path was missing.
    "gpt2_small_reference": {
        "arch_id": "gpt2_small",
        "modern": False,
        "embed_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 12,          # full MHA, no GQA — GPT-2 has 12 KV heads
        "block_size": 1024,
        "target_params": 124_439_808,
        "ffn_multiplier": 4.0,     # 768 -> 3072 (not SwiGLU's 8/3)
        "gelu_tanh": True,         # GPT-2 "GELU New" = tanh approximation
        "resid_scaled_init": True, # c_proj init scaled by 1/sqrt(2*n_layer)
        "qk_norm": False,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "sliding_window": None,
        "local_layer_every": None,
        "moe": None,
        "expert_offload": False,
        "gradient_checkpointing": False,
        "static_kv_cache": False,
        "kv_cache_int8": False,
    },
    "tiny": {
        "embed_dim": 384,
        "n_layers": 6,
        "n_heads": 6,
        "n_kv_heads": 6,
        "block_size": 512,
        "target_params": 35_000_000,
        "qk_norm": False,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "sliding_window": None,
        "local_layer_every": None,
        "moe": None,
        "expert_offload": False,
        "gradient_checkpointing": False,
        "static_kv_cache": False,
        "kv_cache_int8": False,
    },
    "small": {
        "embed_dim": 992,
        "n_layers": 12,
        "n_heads": 16,
        "n_kv_heads": 8,          # GQA: 16 query heads share 8 KV heads
        "block_size": 1024,
        "target_params": 143_000_000,
        "qk_norm": True,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "sliding_window": None,
        "local_layer_every": None,
        "moe": None,
        "expert_offload": False,
        "gradient_checkpointing": False,
        "static_kv_cache": False,
        "kv_cache_int8": False,
    },
    "medium": {
        "embed_dim": 1280,
        "n_layers": 20,
        "n_heads": 20,
        "n_kv_heads": 4,          # GQA: 20 query heads share 4 KV heads (5x smaller cache)
        "block_size": 2048,
        "target_params": 360_000_000,
        "qk_norm": True,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "sliding_window": None,
        "local_layer_every": None,
        "moe": None,
        "expert_offload": False,
        "gradient_checkpointing": True,   # needed to fit 6GB VRAM
        "static_kv_cache": False,
        "kv_cache_int8": False,
    },
    "large": {
        "embed_dim": 2048,
        "n_layers": 24,
        "n_heads": 32,
        "n_kv_heads": 8,
        "block_size": 2048,
        "target_params": 1_000_000_000,
        "qk_norm": True,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "sliding_window": None,
        "local_layer_every": None,
        "moe": None,
        "expert_offload": False,
        "gradient_checkpointing": True,
        "static_kv_cache": False,
        "kv_cache_int8": False,
    },
    "xl-moe": {
        "embed_dim": 1280,
        "n_layers": 12,
        "n_heads": 20,
        "n_kv_heads": 4,
        "block_size": 4096,
        "target_params": 770_000_000,   # measured ~0.77B total / ~0.30B active (12L, 8 exp top-2)
        "qk_norm": True,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "sliding_window": 1024,
        "local_layer_every": 4,   # 3 local layers : 1 global
        "moe": {
            "n_experts": 8,
            "top_k": 2,
            "capacity_factor": 1.25,
            "aux_loss_coef": 0.01,
            "z_loss_coef": 0.001,
            "layers": "even",     # MoE-ify the even layers only
        },
        "expert_offload": True,   # experts on CPU; router prefetches to GPU
        "gradient_checkpointing": True,
        "static_kv_cache": False,  # set JINGPT_STATIC_KV=1 to enable
        "kv_cache_int8": False,    # set JINGPT_KV_INT8=1 to enable
    },
}

# ``jin_small_modern`` is the explicit name for the JIN modern profile that
# ``small`` has always described (RMSNorm + RoPE + GQA + QK-norm + SwiGLU).
# Both keys exist so downstream tooling can name the profile it means.
SCALE_PRESETS["jin_small_modern"] = {**SCALE_PRESETS["small"],
                                     "arch_id": "jin_modern"}
for _name, _preset in SCALE_PRESETS.items():
    _preset.setdefault("arch_id", _name)
    _preset.setdefault("modern", True)
    _preset.setdefault("gelu_tanh", False)
    _preset.setdefault("resid_scaled_init", False)

MODEL_SCALE = os.environ.get("JINGPT_SCALE", "small").lower()
if MODEL_SCALE not in SCALE_PRESETS:
    raise ValueError(
        f"Unknown JINGPT_SCALE={MODEL_SCALE!r}. "
        f"Choose one of: {sorted(SCALE_PRESETS)}"
    )
_PRESET = SCALE_PRESETS[MODEL_SCALE]

# Architecture identity + GPT-2 fidelity switches (backward compatible:
# absent keys in an old checkpoint resolve to the pre-upgrade behaviour).
ARCH_ID = _PRESET["arch_id"]
GELU_TANH = bool(_PRESET["gelu_tanh"])
RESID_SCALED_INIT = bool(_PRESET["resid_scaled_init"])

MODERN_ARCHITECTURE = bool(_PRESET["modern"])
TARGET_PARAMETERS = _PRESET["target_params"]
N_LAYERS = _PRESET["n_layers"]
N_HEADS = _PRESET["n_heads"]
N_KV_HEADS = _PRESET["n_kv_heads"]
EMBED_DIM = _PRESET["embed_dim"]
BLOCK_SIZE = _PRESET["block_size"]
DROPOUT = 0.1
QK_NORM = _PRESET["qk_norm"]
ROPE_THETA = _PRESET["rope_theta"]
ROPE_SCALING = _PRESET["rope_scaling"]
SLIDING_WINDOW = _PRESET["sliding_window"]
LOCAL_LAYER_EVERY = _PRESET["local_layer_every"]
MOE = _PRESET["moe"]
EXPERT_OFFLOAD = _PRESET["expert_offload"]
# Inference KV-cache modes (env-overridable):
#   JINGPT_STATIC_KV=1  -> pre-allocated buffers, no per-step torch.cat
#   JINGPT_KV_INT8=1     -> int8 per-head-scale KV cache (4x less memory)
STATIC_KV_CACHE = bool(int(os.environ.get("JINGPT_STATIC_KV", "0")))
KV_CACHE_INT8 = bool(int(os.environ.get("JINGPT_KV_INT8", "0")))
WEIGHT_QUANT = os.environ.get("JINGPT_QUANT", "").lower() or None  # "int8" | "int4"
# Disabled on CPU (activation recomputation kills throughput);
# set True on GPU to save VRAM (see per-preset value above).
GRADIENT_CHECKPOINTING = _PRESET["gradient_checkpointing"]

# ── Eval & data quality (Phase 3) ─────────────────────────────────────
EVAL_DIR = os.path.join(ROOT_DIR, "data", "eval")
HELDOUT_PATH = os.path.join(EVAL_DIR, "heldout.txt")
PROBES_PATH = os.path.join(EVAL_DIR, "probes.jsonl")
MIXTURE_PATH = os.path.join(ROOT_DIR, "data", "mixture.json")
LANG_SEED_DIR = os.path.join(ROOT_DIR, "data", "lang")

# ── Long-term memory (Phase 4) ────────────────────────────────────────
MEMORY_DIR = os.path.join(ROOT_DIR, "data", "memory")
MEMORY_PATH = os.path.join(MEMORY_DIR, "memory.jsonl")
MEMORY_INDEX_PATH = os.path.join(MEMORY_DIR, "embeddings.npy")
MEMORY_ENABLED = os.environ.get("JINGPT_NO_MEMORY", "").lower() not in ("1", "true")
# Embedding pool target: layer name whose hidden state is mean-pooled.
MEMORY_POOL_LAYER = os.environ.get("JINGPT_MEMORY_POOL", "ln_f")

# ── Curiosity Engine (Phase 5) ────────────────────────────────────────
CURIOSITY_DIR = os.path.join(ROOT_DIR, "data", "curiosity")
CURIOSITY_LEDGER_PATH = os.path.join(CURIOSITY_DIR, "ledger.jsonl")
CURIOSITY_FRONTIER_PATH = os.path.join(CURIOSITY_DIR, "frontier.json")
CURIOSITY_QUIZ_PATH = os.path.join(CURIOSITY_DIR, "quiz.jsonl")
CURIOSITY_HARVEST_DIR = os.path.join(CURIOSITY_DIR, "harvest")
CHECKPOINT_REGISTRY_PATH = os.path.join(ROOT_DIR, "checkpoints", "registry.json")

# Budgets & safety brakes (env-overridable):
#   JINGPT_CURIOSITY_MAX_REQUESTS  — max outgoing web requests per cycle
#   JINGPT_CURIOSITY_MAX_GB_DAY    — max harvested bytes per day
#   JINGPT_CURIOSITY_MAX_CYCLES    — cycles before forced stop
CURIOSITY_MAX_REQUESTS = int(os.environ.get("JINGPT_CURIOSITY_MAX_REQUESTS", "200"))
CURIOSITY_MAX_GB_DAY = float(os.environ.get("JINGPT_CURIOSITY_MAX_GB_DAY", "0.5"))
CURIOSITY_MAX_CYCLES = int(os.environ.get("JINGPT_CURIOSITY_MAX_CYCLES", "10"))
CURIOSITY_MAX_CONSECUTIVE_FAILS = 3   # stop after 3 failed eval gates
CURIOSITY_SAMPLES_PER_CLAIM = 8       # SEAL self-edit samples per claim
CURIOSITY_VERIFY_MIN_SOURCES = 2      # independent sources for confidence=high
CURIOSITY_REPLAY_RATIO = 0.05         # default replay for weak domain shift
CURIOSITY_REPLAY_RATIO_STRONG = 0.3   # replay for strong shift
CURIOSITY_LORA_RANK = 16              # LoRA rank for light updates

# Quality domain allowlist / denylist for harvesting (also used by the
# web_search polite layer). Empty allowlist = everything allowed except
# denylist.
CURIOSITY_DOMAIN_ALLOWLIST: tuple[str, ...] = tuple(
    d.strip().lower() for d in os.environ.get(
        "JINGPT_CURIOSITY_ALLOWLIST", "").split(",") if d.strip()
)  # e.g. "wikipedia.org,arxiv.org,edu,gov"
CURIOSITY_DOMAIN_DENYLIST: tuple[str, ...] = (
    "facebook.com", "instagram.com", "tiktok.com", "twitter.com",
    "x.com", "reddit.com", "pinterest.com", "onlyfans.com",
)

# Eval gate thresholds (Phase 5 step 7) — absolute minimums to promote a
# candidate checkpoint into registry.json.
GATE_QUIZ_PASS_RATE = 0.80            # new-topic quiz accuracy
GATE_FORGETTING_MAX = 0.01            # absolute forgetting on old probes
GATE_VAL_LOSS_MAX_DELTA = 0.01        # relative val-loss increase allowed
GATE_FLUENCY_MIN = 0.70

# ── MoE (Phase 1) ──────────────────────────────────────────────────────
# Defaults follow the active preset (``_PRESET["moe"]``); env overrides
# (``JINGPT_MOE_*``) let you tune without touching presets.
MOE_N_EXPERTS = int(os.environ.get(
    "JINGPT_MOE_N_EXPERTS", str((MOE or {}).get("n_experts", 0))))
MOE_TOP_K = int(os.environ.get(
    "JINGPT_MOE_TOP_K", str((MOE or {}).get("top_k", 0))))
MOE_CAPACITY_FACTOR = float(os.environ.get(
    "JINGPT_MOE_CAPACITY", str((MOE or {}).get("capacity_factor", 1.25))))
MOE_AUX_LOSS_COEF = float(os.environ.get(
    "JINGPT_MOE_AUX_LOSS", str((MOE or {}).get("aux_loss_coef", 0.01))))
MOE_Z_LOSS_COEF = float(os.environ.get(
    "JINGPT_MOE_Z_LOSS", str((MOE or {}).get("z_loss_coef", 0.001))))
MOE_EXPERT_OFFLOAD = bool(int(os.environ.get(
    "JINGPT_MOE_EXPERT_OFFLOAD", str(int(EXPERT_OFFLOAD)))))

# ── Quantization (Phase 2) ─────────────────────────────────────────────
# Weight-only: JINGPT_QUANT=int8 | int4 (per-channel / group-128).
QUANT_WEIGHT = os.environ.get("JINGPT_QUANT", "").lower() or None
QUANT_GROUP = int(os.environ.get("JINGPT_QUANT_GROUP", "128"))
# KV cache: JINGPT_KV_INT8=1 -> int8 per-head-scale KV (4x less memory).
QUANT_KV_INT8 = bool(int(os.environ.get("JINGPT_KV_INT8", "0")))

# ── Speed (Phase 2) ────────────────────────────────────────────────────
# JINGPT_STATIC_KV=1        -> pre-allocated KV buffers (no per-step cat)
# JINGPT_COMPILE=1          -> torch.compile (auto-falls back on failure)
# JINGPT_PROMPT_LOOKUP_K=N  -> prompt-lookup speculative decoding depth
# JINGPT_GC_MODE=layer2|attn-> selective gradient checkpointing profile
SPEED_STATIC_KV = bool(int(os.environ.get("JINGPT_STATIC_KV", "0")))
# NOTE: train.py's compile decision reads ``COMPILE_MODEL`` (Training block
# below); this env-only twin lets scripts query the requested mode.
SPEED_COMPILE = bool(int(os.environ.get("JINGPT_COMPILE", "0")))
SPEED_PROMPT_LOOKUP_K = int(os.environ.get("JINGPT_PROMPT_LOOKUP_K", "4"))
SPEED_GC_MODE = os.environ.get("JINGPT_GC_MODE", "").lower() or None

# ── Training ───────────────────────────────────────────────────────────
BATCH_SIZE = 1
GRAD_ACCUM = 2
TRAIN_BLOCK_SIZE = 256
RUN_ITERS = 100_000
LR = 2e-4
MIN_LR_RATIO = 0.1
WARMUP_ITERS = 200
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
MAX_GRAD_NORM = 1.0
EVAL_INTERVAL = 1_000
EVAL_ITERS = 20
CHECKPOINT_INTERVAL = 1_000
TRAIN_SPLIT = 0.9
SEED = 1337
COMPILE_MODEL = False
AMP_DTYPE = "float16"

# ── Generation ─────────────────────────────────────────────────────────
TEMPERATURE = 0.65
TOP_K = 40
TOP_P = 0.9
REPETITION_PENALTY = 1.12
MAX_NEW_TOKENS = 256
# Anti-echo decoding (see ``model.repeated_tail_period`` / ``reply_quality``).
# A token-level penalty cannot stop a verbatim echo — every token of
# "a small language model" is legal, only the n-gram is not — so the SFT
# 4,000-step checkpoint still ran to the token cap repeating itself.
NO_REPEAT_NGRAM_SIZE = 3     # ban n-grams already in the sequence (0 = off)
STOP_ON_REPEAT = True        # break decoding once the generated tail loops
SAMPLE_PROMPTS = (
    "The ",
    "Once upon a time, ",
    "I think that ",
)

# ── Web server ─────────────────────────────────────────────────────────
# Default to localhost-only — override via JINGPT_HOST=0.0.0.0 only when you
# really want to expose the model on your LAN.
SERVER_HOST = os.environ.get("JINGPT_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("JINGPT_PORT", "5000"))
MAX_REQUEST_BYTES = 32 * 1024              # 32 KiB upper bound on JSON bodies
MAX_USER_MESSAGE_CHARS = 4_000             # per-message clamp
RATE_LIMIT_RPS = 4                         # rolling per-IP requests/second
RATE_LIMIT_BURST = 12                      # short burst budget
REQUIRE_LOOPBACK_TOKEN = False             # set True for shared boxes
SERVER_AUTH_TOKEN = os.environ.get("JINGPT_TOKEN", "")
# Admin API (/api/learn/*, /api/checkpoints, /api/model/reload):
#   * with JINGPT_TOKEN set  → valid token required, loopback-only unless the
#     explicit LAN escape hatch below is on;
#   * without a token        → trusted *only* when the server itself is bound to
#     loopback (see ``host_is_loopback``) and the request comes from loopback.
#     This is a personal model on your own machine: the everyday case
#     (`python app.py`) should not need a password to switch checkpoints.
ALLOW_REMOTE_ADMIN = os.environ.get(
    "JINGPT_ALLOW_REMOTE_ADMIN", "").lower() in ("1", "true")


def host_is_loopback(host: str | None = None) -> bool:
    """True when the server only listens on localhost (127.x / ::1 / localhost).

    The admin API leans on this: bound to loopback, the only clients that can
    reach it are processes on this machine, so no token is demanded. Bind to
    ``0.0.0.0`` (JINGPT_HOST) and admin goes back behind JINGPT_TOKEN.
    """
    value = (host if host is not None else SERVER_HOST) or ""
    value = value.strip().strip("[]").lower()
    return value == "localhost" or value.startswith("127.") or value in (
        "::1", "0:0:0:0:0:0:0:1",
    )

# ── Device ─────────────────────────────────────────────────────────────
DEVICE_CHOICES = ("auto", "gpu", "cuda", "mps", "cpu")


def _mps_available() -> bool:
    return bool(
        getattr(_torch.backends, "mps", None)
        and _torch.backends.mps.is_available()
    )


def resolve_device(request: str | None = None) -> str:
    """Pick ``cuda``, ``mps``, or ``cpu``.

    If ``request`` is ``None``, read ``JINGPT_DEVICE`` (default ``auto``).
    """
    raw = (
        request
        if request is not None
        else os.environ.get("JINGPT_DEVICE", "auto")
    ).lower().strip()
    if raw in ("", "auto"):
        if _torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"
    if raw == "gpu":
        if _torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        warnings.warn(
            "JINGPT_DEVICE=gpu (or --device gpu) but neither CUDA nor MPS is "
            "available — using CPU. For NVIDIA GPU, install CUDA-enabled PyTorch "
            "and drivers.",
            UserWarning,
            stacklevel=2,
        )
        return "cpu"
    if raw == "cuda":
        if not _torch.cuda.is_available():
            raise RuntimeError(
                "Requested --device cuda but torch.cuda.is_available() is False.\n"
                "  • Training on CPU instead: omit --device or use --device auto / --device cpu.\n"
                "  • Prefer GPU when available without failing: use --device gpu.\n"
                "  • Enable CUDA: install an NVIDIA driver (nvidia-smi), then reinstall PyTorch "
                "with CUDA, e.g.\n"
                "      pip install torch torchvision torchaudio --index-url "
                "https://download.pytorch.org/whl/cu124\n"
                "    (pick the cu1xx wheel that matches your setup from https://pytorch.org/get-started/locally/)"
            )
        return "cuda"
    if raw == "mps":
        if not _mps_available():
            raise RuntimeError("Requested MPS but the Metal backend is not available.")
        return "mps"
    if raw == "cpu":
        return "cpu"
    raise ValueError(
        f"Invalid device {raw!r}. Choose one of: {', '.join(DEVICE_CHOICES)}."
    )


def _apply_device_side_effects() -> None:
    if DEVICE == "cuda" and _torch.cuda.is_available():
        _torch.cuda.set_device(0)


def set_device(request: str | None) -> str:
    """Resolve and set :data:`DEVICE` (e.g. ``--device`` from a CLI).

    ``None`` means *auto* (``JINGPT_DEVICE``, else cuda/mps/cpu) — matching what
    ``--device`` documents. It used to be a no-op, but since the module default
    is the string ``"cpu"`` a no-op left every script that omits ``--device``
    (``python app.py``, ``python chat.py``, ``python generate.py``,
    ``python train.py``) running on CPU even with a GPU idle.
    """
    global DEVICE
    DEVICE = resolve_device(request)
    _apply_device_side_effects()
    return DEVICE


DEVICE = "cpu"  # placeholder — overridden at first call to set_device()


# ── Model config assembly ──────────────────────────────────────────────

def get_model_config(vocab_size: int) -> dict:
    """Build the model config for the active preset."""
    return {
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "block_size": BLOCK_SIZE,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "n_kv_heads": N_KV_HEADS,
        "dropout": DROPOUT,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "modern": MODERN_ARCHITECTURE,
        "arch_id": ARCH_ID,
        "gelu_tanh": GELU_TANH,
        "resid_scaled_init": RESID_SCALED_INIT,
        "qk_norm": QK_NORM,
        "rope_theta": ROPE_THETA,
        "rope_scaling": ROPE_SCALING,
        "sliding_window": SLIDING_WINDOW,
        "local_layer_every": LOCAL_LAYER_EVERY,
        "moe": MOE,
        "expert_offload": EXPERT_OFFLOAD,
        "static_kv_cache": STATIC_KV_CACHE or _PRESET.get("static_kv_cache", False),
        "kv_cache_int8": KV_CACHE_INT8 or _PRESET.get("kv_cache_int8", False),
    }


def context_override(
    target_ctx: int,
    base_ctx: int | None = None,
    mode: str = "ntk",
) -> dict:
    """Model-config override that serves a longer context than trained.

    Inference-side twin of ``train.py --extend-context``: the checkpoint keeps
    its 1024-token training window, but RoPE is rescaled so positions beyond it
    still mean something (``factor = target / base``, NTK-aware by default).

    Honest expectation: without a fine-tune on long blocks this buys *usable*
    context, not *trained* context — measure with the ``long_context`` probe
    (``python bench.py --ctx 2048``) before trusting it.
    """
    base = int(base_ctx or BLOCK_SIZE)
    if target_ctx <= base:
        raise ValueError(
            f"--ctx {target_ctx} does not extend the trained window ({base})."
        )
    return {
        "rope_scaling": {
            "type": mode,
            "factor": target_ctx / float(base),
            "target_ctx": int(target_ctx),
        }
    }


def get_legacy_model_config(vocab_size: int) -> dict:
    return {
        "vocab_size": vocab_size,
        "embed_dim": 512,
        "block_size": 256,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 8,
        "dropout": 0.1,
        "gradient_checkpointing": False,
        "modern": False,
        "qk_norm": False,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "sliding_window": None,
        "local_layer_every": None,
        "moe": None,
        "expert_offload": False,
        "static_kv_cache": False,
        "kv_cache_int8": False,
    }


# ── Depth-based auto-scaling (the "one dial") ──────────────────────────

# Anchor points (depth -> embed_dim) from the shipped presets; a log-log
# interpolation between them drives every other knob.
_DEPTH_ANCHORS = [(6, 384), (12, 992), (20, 1280), (24, 2048)]


def _interp_embed(depth: int) -> int:
    xs = [a[0] for a in _DEPTH_ANCHORS]
    ys = [a[1] for a in _DEPTH_ANCHORS]
    if depth <= xs[0]:
        return ys[0]
    if depth >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= depth <= xs[i + 1]:
            logx = math.log(depth / xs[i]) / math.log(xs[i + 1] / xs[i])
            val = ys[i] * (ys[i + 1] / ys[i]) ** logx
            break
    return int(round(val / 64.0) * 64)


def derive_config_from_depth(depth: int, vocab_size: int) -> dict:
    """Derive a full model + training config from a single depth number.

    Heuristics (documented, not sacred): width grows log-linearly between
    preset anchors; head dim is 64; KV heads = heads/4; LR scales as
    1/sqrt(width) around 6e-4 at 384-wide; recommended tokens ~20x params
    (SLM budget, capped by what the corpus actually has).
    """
    embed = _interp_embed(depth)
    n_heads = max(4, round(embed / 64 / 2) * 2)          # even, head_dim 64
    n_kv_heads = max(2, n_heads // 4)
    if embed % n_heads != 0:
        n_heads = max(4, (embed // 64 // 2) * 2)
        while embed % n_heads != 0:
            n_heads -= 2
    if n_heads % n_kv_heads != 0:
        n_kv_heads = max(2, (n_heads // 4 // 2) * 2)
        while n_heads % n_kv_heads != 0:
            n_kv_heads += 1
    block_size = 512 if depth <= 6 else (1024 if depth <= 12 else (2048 if depth <= 20 else 4096))
    lr = 6e-4 * math.sqrt(384.0 / embed)
    lr = min(6e-4, max(1e-4, lr))
    warmup = max(100, int(200 * depth / 12))
    ffn_hidden = int(embed * 8 / 3)
    ffn_hidden = ((ffn_hidden + 63) // 64) * 64
    attn = embed * (embed + 2 * (embed // n_heads) * n_kv_heads) + embed * embed
    ffn = 3 * embed * ffn_hidden
    params = (embed * vocab_size) + depth * (attn + ffn)   # tied embeddings
    tokens_rec = int(params * 20)
    return {
        "depth": depth,
        "embed_dim": embed,
        "n_layers": depth,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "block_size": block_size,
        "lr": round(lr, 7),
        "warmup": warmup,
        "estimated_params": int(params),
        "recommended_tokens": tokens_rec,
    }


# ── CLI: config.py --describe [--depth N] ──────────────────────────────

def _estimate_preset_stats(preset: dict, vocab_size: int) -> dict:
    embed = preset["embed_dim"]
    depth = preset["n_layers"]
    heads = preset["n_heads"]
    kv = preset["n_kv_heads"]
    head_dim = embed // heads
    ffn_mult = float(preset.get("ffn_multiplier", 8.0 / 3.0))
    ffn_hidden = int(embed * ffn_mult)
    ffn_hidden = ((ffn_hidden + 63) // 64) * 64
    moe = preset.get("moe")
    n_experts = (moe or {}).get("n_experts", 0)
    top_k = (moe or {}).get("top_k", 0)
    moe_layers = depth // 2 if moe and moe.get("layers") == "even" else (depth if moe else 0)
    dense_layers = depth - moe_layers
    attn = embed * (embed + 2 * head_dim * kv) + embed * embed
    ffn = 3 * embed * ffn_hidden
    tied_embed = embed * vocab_size                    # head.weight ties tok_emb.weight
    total = tied_embed + dense_layers * (attn + ffn) + moe_layers * (attn + n_experts * ffn)
    active = tied_embed + dense_layers * (attn + ffn) + moe_layers * (attn + top_k * ffn)
    vram_fp32_adamw = total * 16 / 1e9      # weights + grads + 2 moments (fp32)
    return {
        "params_m": total / 1e6,
        "active_m": active / 1e6,
        "vram_est_gb": vram_fp32_adamw,
        "tokens_est": total * 20,
    }


def _describe(vocab_size: int, depth: int | None) -> None:
    print(f"{'Preset/depth':<12} {'params':>10} {'active':>10} {'dim':>5} {'L':>3} "
          f"{'H':>3} {'KV':>3} {'block':>6} {'lr':>10} {'VRAM fp32 AdamW':>15} {'tokens rec.':>14}")
    print("-" * 120)
    for name, preset in SCALE_PRESETS.items():
        st = _estimate_preset_stats(preset, vocab_size)
        print(f"{name:<12} {st['params_m']/1e3:>8.2f}B {st['active_m']/1e3:>8.2f}B "
              f"{preset['embed_dim']:>5} {preset['n_layers']:>3} {preset['n_heads']:>3} "
              f"{preset['n_kv_heads']:>3} {preset['block_size']:>6} "
              f"{'--':>10} {st['vram_est_gb']:>11.1f}GB {st['tokens_est']/1e9:>10.2f}B")
    if depth is not None:
        d = derive_config_from_depth(depth, vocab_size)
        ffn_hidden = int(d["embed_dim"] * 8 / 3)
        ffn_hidden = ((ffn_hidden + 63) // 64) * 64
        attn = d["embed_dim"] * (d["embed_dim"] + 2 * (d["embed_dim"] // d["n_heads"]) * d["n_kv_heads"]) + d["embed_dim"] ** 2
        ffn = 3 * d["embed_dim"] * ffn_hidden
        total = 2 * d["embed_dim"] * vocab_size + d["depth"] * (attn + ffn)
        vram = total * 16 / 1e9
        print("\nDepth-derived config (--depth %d):" % depth)
        print(f"  embed_dim={d['embed_dim']}  n_layers={d['depth']}  n_heads={d['n_heads']}  "
              f"n_kv_heads={d['n_kv_heads']}  block_size={d['block_size']}")
        print(f"  lr={d['lr']}  warmup={d['warmup']}  est_params={total/1e6:.1f}M  "
              f"recommended_tokens={d['recommended_tokens']/1e9:.2f}B  "
              f"VRAM(fp32 AdamW)~{vram:.1f}GB")


def _cli() -> None:
    p = argparse.ArgumentParser(description="JIN AI config inspector.")
    p.add_argument("--describe", action="store_true",
                   help="print the preset table (params/active/VRAM/tokens estimate)")
    p.add_argument("--depth", type=int, default=None,
                   help="with --describe: also print the depth-derived auto-scaled config")
    p.add_argument("--vocab-size", type=int, default=BPE_VOCAB_SIZE)
    args = p.parse_args()
    if args.describe:
        _describe(args.vocab_size, args.depth)
    else:
        print(f"Active preset: {MODEL_SCALE} ({EMBED_DIM}d, {N_LAYERS}L, {N_HEADS}H, {N_KV_HEADS}KV)")
        print("Run `python config.py --describe [--depth N]` for the full table.")


if __name__ == "__main__":
    _cli()
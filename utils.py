"""
JIN AI utilities.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Any, Protocol, Union

import torch
from torch.utils.data import Dataset

import config
from bpe_tokenizer import BPETokenizer
from model import JinAI
from tokenizer import CharTokenizer


class TokenizerLike(Protocol):
    vocab_size: int

    def encode(self, text: str, return_stats: bool = False): ...
    def decode(self, ids) -> str: ...
    def to_state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


AnyTokenizer = Union[CharTokenizer, BPETokenizer]


def make_tokenizer(kind: str | None = None) -> AnyTokenizer:
    """Create an empty tokenizer of the configured kind."""

    kind = (kind or config.TOKENIZER_KIND or "char").lower()
    if kind == "bpe":
        return BPETokenizer()
    if kind == "char":
        return CharTokenizer()
    raise ValueError(f"Unknown TOKENIZER_KIND: {kind!r}")


def tokenizer_from_state(state: dict | None) -> AnyTokenizer:
    """Recreate a tokenizer from a checkpoint's ``tokenizer_state``."""

    if isinstance(state, dict) and state.get("kind") == "bpe_v1":
        tok = BPETokenizer()
        tok.load_state_dict(state)
        return tok
    tok = CharTokenizer()
    if state:
        tok.load_state_dict(state)
    return tok


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int = config.BLOCK_SIZE):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def load_training_text(path: str = config.DATA_PATH) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _tokenizer_fingerprint(tokenizer: AnyTokenizer) -> tuple:
    if isinstance(tokenizer, BPETokenizer):
        from lineage import object_hash
        return ("bpe", object_hash(tokenizer.to_state_dict()))
    if isinstance(tokenizer, CharTokenizer):
        return ("char", tuple(sorted(tokenizer.char_to_id.items())))
    return ("unknown",)


def _token_cache_path(data_path: str, tokenizer: AnyTokenizer) -> str:
    data_abs_path = os.path.abspath(data_path)
    try:
        stat = os.stat(data_abs_path)
        stat_part = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        # Source file gone (e.g. moved to /kaggle/temp to slim the Files
        # panel after encoding). The stream cache stays resolvable because
        # it was named with the fingerprint captured at creation time.
        stat_part = (0, 0)
    fingerprint = repr(
        (
            data_abs_path,
            *stat_part,
            _tokenizer_fingerprint(tokenizer),
        )
    ).encode("utf-8")
    digest = hashlib.sha256(fingerprint).hexdigest()[:16]
    filename = f"tokens_{digest}.pt"
    return config.output_path(os.path.join(config.CACHE_DIR, filename))


def load_or_create_token_data(
    text: str,
    tokenizer: AnyTokenizer,
    data_path: str = config.DATA_PATH,
) -> tuple[torch.Tensor, int, bool]:
    os.makedirs(config.output_path(config.CACHE_DIR), exist_ok=True)
    cache_path = _token_cache_path(data_path, tokenizer)

    if os.path.exists(cache_path):
        cached = _torch_load(cache_path)
        token_ids = cached.get("token_ids")
        skipped = int(cached.get("skipped", 0))
        if isinstance(token_ids, torch.Tensor):
            print(f"[Cache] Hit -> {cache_path}")
            return token_ids.long(), skipped, True

    print(f"[Cache] Miss -> {cache_path}")
    encoded = tokenizer.encode(text, return_stats=True)
    if isinstance(encoded, tuple):
        token_ids, skipped = encoded
    else:
        token_ids, skipped = encoded, 0
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    torch.save(
        {
            "token_ids": token_tensor.cpu(),
            "skipped": skipped,
        },
        cache_path,
    )
    print(f"[Cache] Saved -> {cache_path}")
    return token_tensor, skipped, False


def split_train_val(
    data: torch.Tensor,
    train_split: float = config.TRAIN_SPLIT,
) -> tuple[torch.Tensor, torch.Tensor]:
    split_index = int(len(data) * train_split)
    split_index = max(config.BLOCK_SIZE + 1, split_index)
    split_index = min(split_index, len(data) - (config.BLOCK_SIZE + 1))
    if split_index <= config.BLOCK_SIZE:
        raise ValueError("Dataset is too small for the configured block size.")
    return data[:split_index], data[split_index:]


def get_batch(
    data: torch.Tensor,
    block_size: int = config.TRAIN_BLOCK_SIZE,
    batch_size: int = config.BATCH_SIZE,
    device: str | None = None,
):
    if device is None:
        # Resolve at CALL time — config.DEVICE is a placeholder ("cpu") at
        # import and only becomes correct after set_device() runs in main().
        device = config.DEVICE
    max_idx = len(data) - block_size - 1
    if max_idx <= 0:
        raise ValueError("Dataset is shorter than block_size + 1.")
    starts = torch.randint(0, max_idx + 1, (batch_size,))
    # .long() per slice: stream-cached corpora are stored as uint16 memmaps
    # (constant RAM); conversion happens per batch, never for the whole set.
    x = torch.stack([data[i : i + block_size].long() for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1].long() for i in starts]).to(device)
    return x, y


# ── Streaming token encode (multi-GB corpora, constant RAM) ─────────────

STREAM_ENCODE_THRESHOLD_BYTES = 200_000_000
STREAM_CHUNK_BYTES = 16 << 20  # 16 MB per encode chunk


def _token_stream_cache_path(data_path: str, tokenizer: AnyTokenizer) -> str:
    # Explicit override (JINGPT_TOKEN_CACHE=/abs/path/cache.stream.u16) — lets a
    # resumed session reuse the existing stream cache even when the source
    # corpus file was moved/deleted after encoding (fingerprint would change).
    override = os.environ.get("JINGPT_TOKEN_CACHE")
    if override:
        return config.output_path(override)
    base = _token_cache_path(data_path, tokenizer)
    return base[:-3] + "stream.u16" if base.endswith(".pt") else base + ".stream.u16"


def load_or_create_token_data_from_file(
    data_path: str,
    tokenizer: AnyTokenizer,
) -> tuple[torch.Tensor, int, bool]:
    """Token-encode a corpus file with constant memory.

    Corpora above ``STREAM_ENCODE_THRESHOLD_BYTES`` are streamed in chunks,
    encoded, and appended to a uint16 binary cache that is memory-mapped
    on load (np.memmap -> torch.from_numpy) so the OS pages tokens in on
    demand instead of holding the whole int64 tensor in RAM. ``get_batch``
    converts each slice to int64.
    """
    import codecs

    import numpy as np

    os.makedirs(config.output_path(config.CACHE_DIR), exist_ok=True)
    stream_bin = _token_stream_cache_path(data_path, tokenizer)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    fits_uint16 = isinstance(vocab_size, int) and vocab_size <= 65_535

    if fits_uint16 and os.path.exists(stream_bin):
        import json
        from lineage import object_hash
        metadata_path = stream_bin + ".manifest.json"
        if os.path.getsize(stream_bin) == 0 or os.path.getsize(stream_bin) % 2:
            raise ValueError("Token cache is empty or truncated (uint16 requires even bytes)")
        if os.path.isfile(metadata_path):
            with open(metadata_path, encoding="utf-8") as fh:
                metadata = json.load(fh)
            if metadata.get("tokenizer_hash") != object_hash(tokenizer.to_state_dict()):
                raise ValueError("Token cache tokenizer hash mismatch")
            if metadata.get("bytes") != os.path.getsize(stream_bin):
                raise ValueError("Token cache size differs from manifest")
        elif os.environ.get("JINGPT_TOKEN_CACHE") and os.environ.get("JINGPT_ALLOW_UNVERIFIED_CACHE") != "1":
            raise ValueError("Explicit legacy token cache lacks lineage. Re-encode, or explicitly set JINGPT_ALLOW_UNVERIFIED_CACHE=1 only after independently verifying its tokenizer.")
        arr = np.memmap(stream_bin, dtype=np.uint16, mode="r")
        import warnings as _warnings
        with _warnings.catch_warnings():
            # Read-only memmap-backed tensor is intended: we never write to it.
            _warnings.filterwarnings("ignore", message=".*not writable.*")
            token_tensor = torch.from_numpy(arr)
        print(f"[Cache] Hit -> {stream_bin} ({len(token_tensor):,} tokens)")
        return token_tensor, 0, True

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Corpus file not found: {data_path}\n"
            f"No token cache matched either ({stream_bin}). Restore the corpus "
            f"file, or set JINGPT_TOKEN_CACHE to an existing cache to reuse it."
        )

    if os.path.getsize(data_path) <= STREAM_ENCODE_THRESHOLD_BYTES or not fits_uint16:
        text = load_training_text(data_path)
        return load_or_create_token_data(text, tokenizer, data_path=data_path)

    print(f"[Cache] Miss -> {stream_bin}")
    print(f"[Cache] Streaming encode of {os.path.getsize(data_path)/1e6:.0f} MB "
          f"corpus (constant memory, one-time)...")
    tmp_bin = stream_bin + ".tmp"
    incremental = codecs.getincrementaldecoder("utf-8")(errors="replace")
    skipped_total = 0
    total_tokens = 0
    total_bytes = os.path.getsize(data_path)
    bytes_done = 0
    import time as _time
    _t0 = _time.time()
    with open(data_path, "rb") as src, open(tmp_bin, "wb") as dst:
        carry = ""
        while True:
            block = src.read(STREAM_CHUNK_BYTES)
            final = not block
            if final:
                text = carry + incremental.decode(b"", final=True)
            else:
                text = carry + incremental.decode(block)
            if final and not text:
                break
            if not final:
                cut = max(text.rfind(" "), text.rfind("\n"), text.rfind("\t"))
                if cut <= 0 and len(text) < (64 << 20):
                    carry = text
                    continue
                cut = cut if cut > 0 else len(text) // 2
                chunk, carry = text[:cut], text[cut:]
            else:
                chunk, carry = text, ""
            encoded = tokenizer.encode(chunk, return_stats=True)
            if isinstance(encoded, tuple):
                ids, skipped = encoded
            else:
                ids, skipped = encoded, 0
            if isinstance(skipped, int):
                skipped_total += skipped
            if ids:
                np.asarray(ids, dtype=np.uint16).tofile(dst)
                total_tokens += len(ids)
            bytes_done += len(block)
            if final or bytes_done % (64 << 20) < STREAM_CHUNK_BYTES:
                elapsed = max(_time.time() - _t0, 1e-6)
                mbps = bytes_done / 1e6 / elapsed
                eta_s = (total_bytes - bytes_done) / 1e6 / max(mbps, 1e-6)
                sys.stdout.write(
                    f"\r[Encode] {bytes_done/1e6:8.0f}/{total_bytes/1e6:.0f} MB "
                    f"({100*bytes_done/max(total_bytes,1):5.1f}%) | "
                    f"{total_tokens:,} tokens | {mbps:.1f} MB/s | "
                    f"ETA {eta_s/60:4.1f} min   "
                )
                sys.stdout.flush()
            if final:
                break
    sys.stdout.write("\n")
    os.replace(tmp_bin, stream_bin)
    from lineage import atomic_json, object_hash, sha256_file
    atomic_json(stream_bin + ".manifest.json", {
        "artifact_kind":"token_cache_uint16", "tokenizer_hash":object_hash(tokenizer.to_state_dict()),
        "dataset_sha256":sha256_file(data_path), "sha256":sha256_file(stream_bin),
        "bytes":os.path.getsize(stream_bin), "tokens":total_tokens})
    arr = np.memmap(stream_bin, dtype=np.uint16, mode="r")
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*not writable.*")
        token_tensor = torch.from_numpy(arr)
    print(f"[Cache] Saved -> {stream_bin} ({total_tokens:,} tokens)")
    return token_tensor, skipped_total, False


class MixtureSampler:
    """Curriculum data mixing (Phase 3).

    Loads ``data/mixture.json``: each domain maps to a plain-text source
    file; ``stages`` define per-progress weight schedules. Each ``batch()``
    call samples ``batch_size`` domains according to the active stage's
    weights, draws a random block from that domain's token stream, and
    records the realized domain so the trainer can log the true mix.

    Falls back gracefully: missing domain files are skipped (weights are
    renormalized); a missing mixture file raises unless ``strict=False``.
    """

    def __init__(
        self,
        tokenizer: AnyTokenizer,
        mixture_path: str = config.MIXTURE_PATH,
        block_size: int = config.TRAIN_BLOCK_SIZE,
        batch_size: int = config.BATCH_SIZE,
        device: str | None = None,
        strict: bool = True,
    ):
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.batch_size = int(batch_size)
        self.device = device if device is not None else config.DEVICE
        self.strict = strict

        import json
        with open(mixture_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        self.stages: list[dict] = spec.get("stages", [])
        if spec.get("status") == "template_not_materialized":
            raise ValueError("Mixture is a template: materialize and validate domain files before training")
        domains: dict[str, str] = spec.get("domains", {})
        if spec.get("requires_manifest_status"):
            from lineage import object_hash, sha256_file
            manifest_path = spec.get("provenance_report")
            if not manifest_path:
                raise ValueError("Prepared mixture requires a provenance manifest")
            if not os.path.isabs(manifest_path):
                manifest_path = os.path.join(config.ROOT_DIR, manifest_path)
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("status") != spec["requires_manifest_status"]:
                raise ValueError("Prepared mixture is not ready; preparation may have been interrupted")
            if manifest.get("mixture_hash") != object_hash(spec):
                raise ValueError("Prepared mixture differs from its manifest")
            for name, path in domains.items():
                source = path if os.path.isabs(path) else os.path.join(config.ROOT_DIR, path)
                if sha256_file(source) != manifest["pretrain"][name]["train_sha256"]:
                    raise ValueError(f"Prepared domain hash mismatch: {name}")
            validation = spec.get("validation_path")
            if not validation:
                raise ValueError("Prepared mixture requires a heldout validation file")
            source = validation if os.path.isabs(validation) else os.path.join(config.ROOT_DIR, validation)
            if sha256_file(source) != manifest.get("validation_sha256"):
                raise ValueError("Prepared validation hash mismatch")

        self.validation_path = spec.get("validation_path")
        if self.validation_path:
            source = self.validation_path if os.path.isabs(self.validation_path) else os.path.join(config.ROOT_DIR, self.validation_path)
            if not os.path.isfile(source) or not os.path.getsize(source):
                raise ValueError("Mixture validation file is missing or empty")
        if os.environ.get("JINGPT_TOKEN_CACHE"):
            raise ValueError("Mixtures require separate per-file token caches; unset JINGPT_TOKEN_CACHE")

        self.train_split = float(spec.get("train_split", config.TRAIN_SPLIT))
        if not 0 < self.train_split < 1:
            raise ValueError("Mixture train_split must be in (0,1)")
        # Tokenize each domain once, cached by file fingerprint.
        self.domains: dict[str, torch.Tensor] = {}
        for name, path in domains.items():
            if not os.path.isabs(path):
                path = os.path.join(config.ROOT_DIR, path)
            if not os.path.exists(path):
                if strict:
                    raise FileNotFoundError(
                        f"Mixture domain {name!r}: file not found: {path}"
                    )
                continue
            data, _, _ = load_or_create_token_data_from_file(path, tokenizer)
            if not self.validation_path:
                # Legacy specs had no heldout split: exclude each domain's tail
                # instead of sampling validation tokens during training.
                data = data[:int(len(data) * self.train_split)]
            if data.numel() > block_size + 1:
                self.domains[name] = data
        if not self.domains:
            raise ValueError("Mixture: no usable domain files.")

        self._realized: dict[str, int] = {name: 0 for name in self.domains}

    def active_weights(self, progress: float) -> dict[str, float]:
        stage = self.stages[0] if self.stages else {}
        for cand in self.stages:
            if cand.get("start", 0.0) <= progress <= cand.get("end", 1.0):
                stage = cand
                break
        weights = {k: v for k, v in stage.get("weights", {}).items()
                   if k in self.domains}
        total = sum(weights.values())
        if total <= 0:
            weights = {k: 1.0 for k in self.domains}
            total = float(len(weights))
        return {k: v / total for k, v in weights.items()}

    def batch(self, progress: float):
        """Sample one (x, y) batch drawn from the active mixture."""
        weights = self.active_weights(progress)
        names = list(weights.keys())
        probs = torch.tensor([weights[n] for n in names], dtype=torch.float64)
        chosen = torch.multinomial(probs, self.batch_size, replacement=True).tolist()
        xs, ys = [], []
        for idx in chosen:
            name = names[idx]
            data = self.domains[name]
            max_idx = len(data) - self.block_size - 1
            start = torch.randint(0, max_idx + 1, (1,)).item()
            xs.append(data[start : start + self.block_size].long())
            ys.append(data[start + 1 : start + self.block_size + 1].long())
            self._realized[name] += 1
        return (
            torch.stack(xs).to(self.device),
            torch.stack(ys).to(self.device),
        )

    def realized_mix(self) -> dict[str, int]:
        return dict(self._realized)


def get_batch_with_doc_mask(
    data: torch.Tensor,
    eot_id: int,
    block_size: int = config.TRAIN_BLOCK_SIZE,
    batch_size: int = config.BATCH_SIZE,
    device: str | None = None,
):
    """Batch of packed sequences with a real document mask.

    ``doc_mask`` (B, T) is 1 at every token that starts a new document —
    the first token of the block, and any token right after ``eot_id``.
    The model uses it to block attention across document boundaries, so
    sequence packing trains real documents instead of concatenated noise.
    """
    if device is None:
        device = config.DEVICE
    max_idx = len(data) - block_size - 1
    if max_idx <= 0:
        raise ValueError("Dataset is shorter than block_size + 1.")
    starts = torch.randint(0, max_idx + 1, (batch_size,))
    x = torch.stack([data[i : i + block_size].long() for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1].long() for i in starts]).to(device)
    prev = torch.stack(
        [torch.cat([data[i : i + 1].long(), data[i : i + block_size - 1].long()]) for i in starts]
    ).to(device)
    doc_mask = (prev == eot_id).long()
    doc_mask[:, 0] = 1
    return x, y, doc_mask


def _torch_load(
    path: str,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint dict, mmap-ing tensor storage when possible.

    Checkpoints hold optimizer state (often 2-3x the weights), so a plain
    load can spike RAM and, on Windows WDDM, wedge subsequent CUDA
    allocations. ``mmap=True`` keeps storage on disk; falls back silently.
    """

    target_location = config.DEVICE if map_location is None else map_location
    if target_location in ("cpu", None):
        try:
            return torch.load(path, map_location=target_location or "cpu",
                              weights_only=True, mmap=True)
        except Exception:
            pass  # e.g. legacy pickle-format checkpoints — fall through
    try:
        return torch.load(path, map_location=target_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Safe checkpoint loading requires PyTorch with weights_only support") from exc


def load_checkpoint_file(
    path: str,
    allow_fallback: bool = False,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    resolved_path = path
    if not os.path.exists(resolved_path) and allow_fallback:
        fallbacks = []
        if path in {
            config.CHECKPOINT_PATH,
            config.BEST_CHECKPOINT_PATH,
            config.LATEST_CHECKPOINT_PATH,
        }:
            fallbacks = [
                config.BEST_CHECKPOINT_PATH,
                config.LATEST_CHECKPOINT_PATH,
                config.LEGACY_CHECKPOINT_PATH,
            ]
        for candidate in fallbacks:
            if os.path.exists(candidate):
                resolved_path = candidate
                break
        else:
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    elif not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = _torch_load(resolved_path, map_location=map_location)
    checkpoint.setdefault("_loaded_from", resolved_path)
    return checkpoint


def extract_model_config(
    checkpoint: dict[str, Any] | None,
    vocab_size: int | None = None,
) -> dict[str, Any]:
    if checkpoint and checkpoint.get("model_config"):
        model_config = dict(checkpoint["model_config"])
    else:
        inferred_vocab_size = vocab_size or (
            checkpoint.get("vocab_size") if checkpoint else None
        )
        if inferred_vocab_size is None:
            raise ValueError("vocab_size is required to build the model.")
        loaded_from = checkpoint.get("_loaded_from") if checkpoint else None
        if loaded_from and os.path.normcase(loaded_from) == os.path.normcase(
            config.LEGACY_CHECKPOINT_PATH
        ):
            model_config = config.get_legacy_model_config(inferred_vocab_size)
        else:
            model_config = config.get_model_config(inferred_vocab_size)

    final_vocab_size = vocab_size or model_config.get("vocab_size")
    if final_vocab_size is None:
        raise ValueError("Unable to resolve vocab_size for model config.")

    model_config["vocab_size"] = int(final_vocab_size)
    model_config.setdefault("gradient_checkpointing", config.GRADIENT_CHECKPOINTING)
    return model_config


def create_model(
    vocab_size: int,
    checkpoint: dict[str, Any] | None = None,
    device: str | None = None,
    override_config: dict[str, Any] | None = None,
) -> JinAI:
    """Build a model from a checkpoint's config (or the active preset).

    ``override_config`` is merged on top of the resolved config — used by
    ``--depth`` (auto-scaled architecture) and ``--extend-context`` (RoPE
    scaling) in train.py.
    """
    if device is None:
        device = config.DEVICE
    model_config = extract_model_config(checkpoint, vocab_size=vocab_size)
    if override_config:
        model_config.update(override_config)
    model_config["device"] = device
    return JinAI.from_config(model_config)


def save_checkpoint(
    model: JinAI,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    best_loss: float,
    path: str,
    tokenizer: AnyTokenizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    train_loss: float | None = None,
    val_loss: float | None = None,
) -> None:
    path = config.output_path(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": 2,
        "iteration": iteration,
        "best_loss": best_loss,
        "loss": val_loss if val_loss is not None else best_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_config": model.export_config(),
        "vocab_size": model.vocab_size,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "rng_state": torch.random.get_rng_state(),
    }

    if tokenizer is not None:
        payload["tokenizer_state"] = tokenizer.to_state_dict()

    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    from lineage import object_hash, atomic_json, sha256_file
    payload["format_version"] = 3
    payload["artifact_kind"] = "training_checkpoint" if optimizer else "inference_weights"
    run = getattr(model, "_run_manifest", None)
    lineage = dict(run or {})
    lineage.update(model_config_hash=object_hash(payload["model_config"]),
                   tokenizer_hash=object_hash(payload.get("tokenizer_state")), step=iteration,
                   validation_metrics={"val_loss": val_loss, "train_loss": train_loss},
                   status="checkpoint_saved")
    payload["lineage"] = lineage
    # Atomic save: write to a temp file ONCE, then os.replace(). A crash
    # mid-write can no longer corrupt the last good checkpoint. If the swap
    # is blocked (antivirus scanning the fresh 1-2 GB temp file, or another
    # process holding the destination open — e.g. an app.py server that
    # mmap-loaded it), retry the swap with backoff, and as a last resort
    # publish under an alternate path instead of killing training.
    import uuid
    tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp_path, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            artifact = {"path": os.path.abspath(path), "sha256": sha256_file(path),
                        "bytes": os.path.getsize(path), "kind": payload["artifact_kind"]}
            lineage["output_artifact"] = artifact
            atomic_json(path + ".manifest.json", lineage)
            if run:
                run.update(lineage)
                atomic_json(run["manifest_path"], run)
            print(f"[Checkpoint] Saved -> {path}")
            return
        except (OSError, PermissionError, RuntimeError) as exc:
            last_err = exc
            wait_s = 15 * (attempt + 1)
            print(f"[Checkpoint] replace attempt {attempt + 1}/5 blocked "
                  f"({exc.__class__.__name__}); retrying in {wait_s}s ...")
            time.sleep(wait_s)

    # Last resort: the temp file holds a VALID checkpoint — keep the data
    # under an alternate name instead of losing it.
    alt_path = path.replace(".pt", f"_{iteration}.pt")
    try:
        os.replace(tmp_path, alt_path)
        print(f"[Checkpoint] Destination locked — data saved to ALTERNATE path:\n"
              f"[Checkpoint]   {alt_path}\n"
              f"[Checkpoint] Resume later with: --checkpoint \"{alt_path}\" "
              f"(or replace {os.path.basename(path)} with this file).")
        return
    except Exception as exc2:
        raise RuntimeError(
            f"Could not save checkpoint to {path}: {last_err}. "
            f"Valid checkpoint data remains at {tmp_path} — rename it manually."
        ) from exc2


def load_checkpoint(
    model: JinAI,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    path: str = config.CHECKPOINT_PATH,
    restore_rng: bool = False,
) -> dict[str, Any]:
    checkpoint = load_checkpoint_file(path, allow_fallback=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as exc:
            # E.g. switching AdamW <-> Muon checkpoints: weights still load;
            # optimizer state is skipped so training restarts fresh.
            print(f"[Checkpoint] WARNING: optimizer state not loaded ({type(exc).__name__}: {exc})")

    if scaler is not None and checkpoint.get("scaler_state_dict"):
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except Exception as exc:
            print(f"[Checkpoint] WARNING: scaler state not loaded ({type(exc).__name__}: {exc})")

    if restore_rng and checkpoint.get("rng_state") is not None:
        rng_state = checkpoint["rng_state"]
        if not isinstance(rng_state, torch.Tensor):
            rng_state = torch.tensor(rng_state, dtype=torch.uint8)
        rng_state = rng_state.detach().to(device="cpu", dtype=torch.uint8)
        torch.random.set_rng_state(rng_state)

    if (
        restore_rng
        and torch.cuda.is_available()
        and checkpoint.get("cuda_rng_state_all") is not None
    ):
        cuda_rng_state_all = []
        for state in checkpoint["cuda_rng_state_all"]:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.uint8)
            cuda_rng_state_all.append(state.detach().to(device="cpu", dtype=torch.uint8))
        torch.cuda.set_rng_state_all(cuda_rng_state_all)

    loaded_from = checkpoint.get("_loaded_from", path)
    print(
        f"[Checkpoint] Loaded <- {loaded_from} "
        f"(iter {checkpoint.get('iteration', '?')})"
    )
    return checkpoint


def build_inference_bundle(
    checkpoint_path: str = config.CHECKPOINT_PATH,
    device: str | None = None,
    quant: str | None = None,
    quant_group: int = 128,
    override_config: dict[str, Any] | None = None,
) -> tuple[JinAI, AnyTokenizer, dict[str, Any]]:
    """Load a checkpoint for inference.

    ``override_config`` (see :func:`config.context_override`) changes the
    *runtime* architecture — e.g. a larger RoPE ceiling at serve time — without
    touching the stored weights.
    """
    if device is None:
        device = config.DEVICE
    # Load on CPU: checkpoints can hold optimizer state (often 2-3x the
    # weights), which would blow up small-VRAM GPUs if resident on device.
    # load_state_dict() below copies only the weights over.
    checkpoint = load_checkpoint_file(checkpoint_path, allow_fallback=True, map_location="cpu")

    state = checkpoint.get("tokenizer_state")
    if state:
        tokenizer = tokenizer_from_state(state)
        print(
            f"[Tokenizer] Restored {'BPE' if isinstance(tokenizer, BPETokenizer) else 'char'} "
            f"vocabulary from checkpoint metadata (size={tokenizer.vocab_size:,})."
        )
    elif os.path.exists(config.BPE_PATH):
        tokenizer = BPETokenizer()
        tokenizer.load(config.BPE_PATH)
        print(f"[Tokenizer] Loaded BPE from {config.BPE_PATH}.")
    elif os.path.exists(config.VOCAB_PATH):
        tokenizer = CharTokenizer()
        tokenizer.load()
    else:
        raise FileNotFoundError(
            "No tokenizer metadata in checkpoint and no vocab/bpe file found."
        )

    model = create_model(
        vocab_size=tokenizer.vocab_size,
        checkpoint=checkpoint,
        device=device,
        override_config=override_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if quant:
        from quantize import apply_weight_quantization
        bits = 8 if quant == "int8" else 4
        replaced = apply_weight_quantization(model, bits=bits, group=quant_group)
        print(
            f"[Quant] Applied {quant} weight-only quantization "
            f"to {replaced} linear layers (embeddings/head kept full-precision)."
        )
    return model, tokenizer, checkpoint


def configure_cuda_performance(device: str | torch.device) -> None:
    """Speed up CUDA paths and stabilize cuBLAS on Windows + small VRAM GPUs."""

    d = device if isinstance(device, str) else device.type
    if d != "cuda" or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Tiny matmul so cuBLAS initializes before heavier training tensors land.
    t = torch.empty(384, 384, device="cuda", dtype=torch.float16)
    torch.mm(t, t)
    torch.cuda.synchronize()

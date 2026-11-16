"""
JIN AI v3 — decoder-only transformer.

Architecture v3 (Phase 1 of the v3.0 plan) adds, behind backward-compatible
config flags:

- **GQA** (Grouped-Query Attention): ``n_kv_heads < n_heads`` shrinks the KV
  cache ~4x and speeds up decode with near-zero quality loss.
- **QK-norm**: RMSNorm on ``q``/``k`` before attention — stabilizes training
  at higher LR and suppresses loss spikes.
- **RoPE scaling**: configurable ``rope_theta`` plus NTK-aware / YaRN-style
  interpolation to extend the context window (1024 -> 4096 -> 8192) without
  training from scratch (needs a short fine-tune on long data).
- **Sliding-window attention**: local layers attend only to the last
  ``window`` tokens; the KV cache trims per-window.
- **MoE feed-forward**: Switch-style router (top-k, capacity factor,
  load-balancing auxiliary loss + optional z-loss) with expert offload to CPU
  for small-VRAM GPUs.

Two architectural variants remain:

- ``modern=False``: classic GPT block (LayerNorm + GELU MLP + learned pos).
- ``modern=True``: RMSNorm + SwiGLU + RoPE (+ the v3 features above).

Backward compatibility: every new knob defaults to the old behaviour, and
the ``qkv_proj`` shape is identical when ``n_kv_heads == n_heads``, so v2.5
checkpoints load unchanged when built from their own ``model_config``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

import config


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no bias, no mean subtraction)."""

    def __init__(self, embed_dim: int, eps: float = 1e-6, *, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def build_rope_cache(
    max_positions: int,
    head_dim: int,
    device: torch.device | str | None = None,
    base: float = 10_000.0,
    scaling: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute ``cos``/``sin`` tables for rotary position embeddings.

    Returns two tensors of shape ``(max_positions, head_dim)``.

    ``scaling`` supports two context-extension schemes:

    - ``{"type": "ntk", "factor": f, ...}``: NTK-aware — rescale ``base`` by
      ``f ** (head_dim / (head_dim - 2))`` so longer positions alias correctly
      without touching the position grid.
    - ``{"type": "yarn", "factor": f, "target_ctx": N}``: YaRN-lite — build a
      table for ``N`` positions but divide positions by ``f`` (linear
      interpolation of the position grid).
    """
    half = head_dim // 2
    if scaling:
        kind = scaling.get("type")
        factor = float(scaling.get("factor", 1.0))
        target = int(scaling.get("target_ctx", 0)) or max_positions
        if kind == "ntk":
            base = base * (factor ** (head_dim / max(2.0, head_dim - 2.0)))
        elif kind == "yarn":
            max_positions = target
        # unknown kind -> plain cache (no scaling)

    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    if scaling and scaling.get("type") == "yarn":
        positions = torch.arange(max_positions, device=device).float() / factor
    else:
        positions = torch.arange(max_positions, device=device).float()
    angles = torch.outer(positions, freqs)  # (max_positions, half)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to ``x`` of shape ``(B, H, T, D)``."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rotated * sin).to(x.dtype)


# ----------------------------------------------------------------------
# Repetition control (decoding)
# ----------------------------------------------------------------------

# A tail is "looping" once the same token chunk repeats this many times. Short
# chunks need more repeats (a lone "!" or "ha ha" is legitimate style), long
# chunks need only 3 — mirrors what a human would call an echo.
REPEAT_MIN_REPEATS = 3
REPEAT_MIN_SPAN = 12
REPEAT_MAX_PERIOD = 24


def repeated_repeat_count(period: int, min_repeats: int = REPEAT_MIN_REPEATS,
                          min_span: int = REPEAT_MIN_SPAN) -> int:
    """How many copies of a ``period``-long chunk count as an echo."""
    return max(min_repeats, -(-min_span // max(1, period)))


def repeated_tail_period(
    ids: list[int],
    max_period: int = REPEAT_MAX_PERIOD,
    min_repeats: int = REPEAT_MIN_REPEATS,
    min_span: int = REPEAT_MIN_SPAN,
) -> int | None:
    """Return the period of a looped tail in ``ids``, or ``None``.

    ``[.., a, b, c, a, b, c, a, b, c]`` -> ``3`` (the last three tokens are
    the third copy of an already-seen chunk). The smallest period wins, so a
    collapse into one repeated token is reported as period 1.
    """
    n = len(ids)
    for period in range(1, min(max_period, n) + 1):
        repeats = repeated_repeat_count(period, min_repeats, min_span)
        if period * repeats > n:
            break                    # need grows with period — nothing longer fits
        chunk = ids[n - period:]
        if all(
            ids[n - period * (k + 1): n - period * k] == chunk
            for k in range(1, repeats)
        ):
            return period
    return None


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block (gate x up -> down)."""

    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        hidden_multiplier: float = 8.0 / 3.0,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        hidden = int(embed_dim * hidden_multiplier)
        hidden = ((hidden + 63) // 64) * 64  # round to a multiple of 64
        self.w_gate = nn.Linear(embed_dim, hidden, bias=False, **factory)
        self.w_up = nn.Linear(embed_dim, hidden, bias=False, **factory)
        self.w_down = nn.Linear(hidden, embed_dim, bias=False, **factory)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ----------------------------------------------------------------------
# Attention (GQA + QK-norm + RoPE + sliding window + KV cache)
# ----------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float,
        *,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10_000.0,
        rope_scaling: dict | None = None,
        window: int | None = None,
        max_positions: int = 0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else int(n_kv_heads)
        if self.n_kv_heads <= 0 or n_heads % self.n_kv_heads != 0:
            raise ValueError("n_kv_heads must be a positive divisor of n_heads")
        self.dropout = dropout
        self.use_rope = use_rope
        self.window = window                      # None -> global attention
        self.max_positions = max(max_positions, 1)

        factory_kwargs = {"device": device, "dtype": dtype}
        kv_dim = self.n_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(embed_dim, embed_dim + 2 * kv_dim, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.resid_drop = nn.Dropout(dropout)

        self.q_norm = RMSNorm(self.head_dim, **factory_kwargs) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, **factory_kwargs) if qk_norm else None

        # RoPE tables are sized for the extended context (rope_scaling target).
        rope_len = rope_scaling.get("target_ctx", 0) if rope_scaling else 0
        rope_len = max(rope_len, self.max_positions)
        if use_rope:
            cos, sin = build_rope_cache(
                rope_len, self.head_dim, device=device,
                base=rope_theta, scaling=rope_scaling,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

        self._masks: dict[int, torch.Tensor] = {}  # cached local masks by length

    def _local_mask(self, seq_len: int, device) -> torch.Tensor | None:
        if self.window is None:
            return None
        cached = self._masks.get(seq_len)
        if cached is not None and cached.device == device:
            return cached
        ar = torch.arange(seq_len, device=device)
        allowed = (ar[:, None] >= ar[None, :]) & (ar[:, None] - ar[None, :] < self.window)
        mask = torch.where(allowed, torch.zeros((), device=device),
                           torch.full((), float("-inf"), device=device))
        self._masks[seq_len] = mask
        return mask

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([channels, self.n_kv_heads * self.head_dim,
                             self.n_kv_heads * self.head_dim], dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm before RoPE (stabilizes training at high LR).
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Absolute position offset: the KV cache may have been trimmed to the
        # sliding window, so track a running position counter separately.
        offset = 0
        if kv_cache is not None:
            offset = int(kv_cache.get("pos", 0))

        if self.use_rope and rope_cos is not None and rope_sin is not None:
            # ``offset`` (absolute position) can exceed the precomputed rope
            # table once the cache is trimmed and generation continues past
            # ``max_positions`` total tokens. An empty slice would collapse
            # q/k to 0 tokens and crash SDPA, so clamp the lookup to the
            # table. The cache only ever keeps the last ``max_positions``
            # keys, so this is the standard sliding-window position cap
            # (same trick as Llama.cpp/Mistral): positions stay exact until
            # the table is exhausted, then alias gracefully instead of
            # crashing. q and k share the clamp, so the pair stays consistent.
            rope_off = min(offset, max(0, rope_cos.size(0) - seq_len))
            q = apply_rope(q, rope_cos[rope_off: rope_off + seq_len],
                           rope_sin[rope_off: rope_off + seq_len])
            k = apply_rope(k, rope_cos[rope_off: rope_off + seq_len],
                           rope_sin[rope_off: rope_off + seq_len])

        is_incremental = False
        window = kv_cache.get("window") if kv_cache is not None else None
        if kv_cache is not None and kv_cache.get("k_buf") is not None:
            # ── Static (pre-allocated, optional int8) KV cache ────────────
            # No per-step torch.cat: write into pre-allocated buffers.
            k_buf = kv_cache["k_buf"]
            v_buf = kv_cache["v_buf"]
            start = int(kv_cache.get("len", 0))
            is_incremental = seq_len == 1
            # Make room: drop keys that NO query in this forward could attend.
            #   - windowed layers: keep at most ``window`` total keys
            #   - global layers:  keep at most buffer capacity
            buf_cap = k_buf.size(-2)
            max_keep = min(window, buf_cap) if window else buf_cap
            if start + seq_len > max_keep:
                keep_old = max(0, max_keep - seq_len)
                if start > keep_old:
                    shift = start - keep_old
                    k_buf[:, :, :keep_old] = k_buf[:, :, shift:start]
                    v_buf[:, :, :keep_old] = v_buf[:, :, shift:start]
                    if kv_cache.get("int8"):
                        kv_cache["k_scale"][:, :, :keep_old] = kv_cache["k_scale"][:, :, shift:start]
                        kv_cache["v_scale"][:, :, :keep_old] = kv_cache["v_scale"][:, :, shift:start]
                start = keep_old
            if kv_cache.get("int8"):
                k_scale = k.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
                v_scale = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
                k_buf[:, :, start:start + seq_len] = (k / k_scale * 127).round().to(torch.int8)
                v_buf[:, :, start:start + seq_len] = (v / v_scale * 127).round().to(torch.int8)
                kv_cache["k_scale"][:, :, start:start + seq_len] = k_scale
                kv_cache["v_scale"][:, :, start:start + seq_len] = v_scale
            else:
                k_buf[:, :, start:start + seq_len] = k
                v_buf[:, :, start:start + seq_len] = v
            new_len = start + seq_len
            kv_cache["len"] = new_len
            kv_cache["pos"] = offset + seq_len
            if kv_cache.get("int8"):
                # Stored as round(k / scale * 127); dequant = buf / 127 * scale.
                k = k_buf[:, :, :new_len].float() / 127.0 * kv_cache["k_scale"][:, :, :new_len]
                v = v_buf[:, :, :new_len].float() / 127.0 * kv_cache["v_scale"][:, :, :new_len]
            else:
                k = k_buf[:, :, :new_len]
                v = v_buf[:, :, :new_len]
        elif kv_cache is not None:
            # ── Dynamic KV cache (original path, backward compatible) ─────
            cached_k = kv_cache.get("k")
            cached_v = kv_cache.get("v")
            if cached_k is not None and cached_v is not None:
                k = torch.cat([cached_k, k], dim=-2)
                v = torch.cat([cached_v, v], dim=-2)
            is_incremental = seq_len == 1
            # Trim only on single-token decode (keeps the cache bounded); a
            # multi-token prefill needs the full keys so early query positions
            # can still see their local window. Positions stay absolute via pos.
            if is_incremental and self.window is not None and k.size(-2) > self.window:
                k = k[..., -self.window:, :]
                v = v[..., -self.window:, :]
            kv_cache["k"] = k
            kv_cache["v"] = v
            kv_cache["pos"] = offset + seq_len

        # GQA: expand KV heads to match the query head count for SDPA.
        if self.n_kv_heads != self.n_heads:
            expand = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(expand, dim=1)
            v = v.repeat_interleave(expand, dim=1)

        # Masking rules (absolute positions, robust to cache trimming):
        # - single-token decode: cache is causal + window-bounded already,
        #   so no mask and no is_causal flag.
        # - multi-token forward into a cache: explicit absolute-position-aware
        #   mask. Cache keys may be a trimmed window (fewer keys than query
        #   tokens) or the full history, so a plain local mask sliced to the
        #   key count is WRONG — build it from absolute positions instead.
        # - no cache (training / full pass): doc-aware ``attn_mask`` wins;
        #   otherwise causal, with a sliding-window bound for local layers.
        key_len = k.size(-2)
        if is_incremental:
            attn_mask_use, use_causal = None, False
        elif kv_cache is not None:
            # Query tokens live at absolute [offset, offset+seq_len); keys at
            # [offset + seq_len - key_len, offset + seq_len).
            q_abs = torch.arange(offset, offset + seq_len, device=x.device)
            k_abs = torch.arange(offset + seq_len - key_len, offset + seq_len,
                                 device=x.device)
            causal = q_abs[:, None] >= k_abs[None, :]
            if window is not None:
                allowed = causal & (q_abs[:, None] - k_abs[None, :] < window)
            else:
                allowed = causal
            attn_mask_use = torch.where(
                allowed, torch.zeros((), device=x.device),
                torch.full((), float("-inf"), device=x.device),
            )
            use_causal = False
        elif attn_mask is not None:
            attn_mask_use, use_causal = attn_mask, False
            if self.window is not None:
                attn_mask_use = attn_mask_use + self._local_mask(seq_len, x.device)
        elif self.window is not None:
            attn_mask_use = self._local_mask(seq_len, x.device)
            use_causal = False
        else:
            attn_mask_use, use_causal = None, True

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask_use,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=use_causal,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            if not is_incremental:
                if attn_mask_use is not None:
                    attn = attn + attn_mask_use
                else:
                    mask = torch.triu(
                        torch.ones(seq_len, k.size(-2), device=x.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    attn = attn.masked_fill(mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_drop(self.out_proj(out))


# ----------------------------------------------------------------------
# Feed-forward (dense + MoE)
# ----------------------------------------------------------------------

class FeedForward(nn.Module):
    """Classic 4x GELU MLP (modern=False variant).

    ``gelu_tanh=True`` selects GPT-2's "GELU New" (tanh approximation).
    Exact GELU (the default) reproduces every pre-upgrade checkpoint.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        *,
        gelu_tanh: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, **factory_kwargs),
            nn.GELU(approximate="tanh") if gelu_tanh else nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, **factory_kwargs),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEFeedForward(nn.Module):
    """Switch-style Mixture-of-Experts feed-forward.

    ``n_experts`` SwiGLU experts, gated by a linear router. The top ``top_k``
    experts per token are combined with their router probabilities. Training
    losses: a Switch load-balancing auxiliary loss (to keep routing balanced)
    plus an optional z-loss (to keep router logits well-scaled). Tokens are
    dispatched under a ``capacity_factor``; over-capacity tokens are dropped
    (their gradient flows through the router only).

    With ``expert_offload=True`` the expert weights live on CPU and are moved
    to the compute device on demand, selected by the router (prefetch), so a
    1.2B-parameter model fits in 6GB VRAM.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        *,
        n_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.0,
        expert_offload: bool = False,
        compute_device: str | torch.device | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)
        self.aux_loss_coef = float(aux_loss_coef)
        self.z_loss_coef = float(z_loss_coef)
        self.expert_offload = bool(expert_offload)
        self.compute_device = compute_device

        self.gate = nn.Linear(embed_dim, self.n_experts, bias=False, **factory_kwargs)
        self.experts = nn.ModuleList(
            [SwiGLU(embed_dim, dropout, device=device, dtype=dtype) for _ in range(self.n_experts)]
        )
        if self.expert_offload:
            self.experts.to("cpu")

        # Aux losses accumulated during forward (read by JinAI.forward).
        self.aux_loss = torch.zeros((), dtype=torch.float32)
        self.z_loss = torch.zeros((), dtype=torch.float32)

    def _to_compute(self, needed: set[int]) -> None:
        device = self.compute_device
        if device is None:
            return  # CPU model: experts already resident
        for e in needed:
            if self.experts[e].w_gate.weight.device != device:
                self.experts[e].to(device)
        if self.training:
            # During training keep the footprint low: evict experts that were
            # moved but are not used this step. At eval time experts stay on
            # device once loaded (routing is sparse but re-loading each step
            # would dominate latency).
            for e in range(self.n_experts):
                if e not in needed and self.experts[e].w_gate.weight.device == device:
                    self.experts[e].to("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        n_tokens = batch * seq
        tokens = x.reshape(-1, dim)  # (n_tokens, dim)

        router_logits = self.gate(tokens)          # (n_tokens, n_experts)
        probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)  # (n_tokens, top_k)

        if self.expert_offload:
            needed = set(int(i) for i in topk_idx.unique())
            self._to_compute(needed)

        capacity = math.ceil((n_tokens / self.n_experts) * self.capacity_factor)
        out = torch.zeros_like(tokens)

        for e in range(self.n_experts):
            chosen = (topk_idx == e).any(dim=1)          # (n_tokens,)
            if self.capacity_factor > 0 and int(chosen.sum()) > capacity:
                # Enforce capacity: keep the first `capacity` tokens in order;
                # dropped tokens contribute nothing (router-only gradient).
                idxs = chosen.nonzero(as_tuple=False).flatten()
                chosen = torch.zeros_like(chosen)
                chosen[idxs[:capacity]] = True
            token_ids = chosen.nonzero(as_tuple=False).flatten()
            if token_ids.numel() == 0:
                continue
            # Router probability assigned to expert e for each selected token.
            slot = (topk_idx[token_ids] == e).int().argmax(dim=1)  # first matching slot
            w = topk_probs[token_ids].gather(1, slot.unsqueeze(1)).squeeze(1)
            out[token_ids] += self.experts[e](tokens[token_ids]) * w.unsqueeze(-1)

        # Load-balancing auxiliary loss (Switch-style).
        if self.training:
            dispatch_count = torch.zeros(self.n_experts, dtype=torch.float32, device=x.device)
            for e in range(self.n_experts):
                dispatch_count[e] = (topk_idx == e).any(dim=1).sum().float()
            f_i = dispatch_count.detach() / max(1, n_tokens)
            p_i = probs.mean(dim=0)
            self.aux_loss = self.n_experts * (f_i * p_i).sum()
            self.z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)

        return out.reshape(batch, seq, dim)

    def active_parameters(self) -> int:
        """Parameters actually used per forward, beyond the router/gate.

        The gate is counted by the model-wide loop; this adds only the
        fraction of expert weights routed per forward (top_k / n_experts).
        """
        expert_n = sum(p.numel() for p in self.experts.parameters())
        return int(expert_n * (self.top_k / self.n_experts))


# ----------------------------------------------------------------------
# Transformer block
# ----------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float,
        *,
        modern: bool = False,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        rope_theta: float = 10_000.0,
        rope_scaling: dict | None = None,
        window: int | None = None,
        max_positions: int = 0,
        moe: dict | None = None,
        expert_offload: bool = False,
        compute_device: str | torch.device | None = None,
        attn_checkpoint: bool = False,
        gelu_tanh: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.modern = modern
        self.attn_checkpoint = bool(attn_checkpoint)

        if modern:
            self.ln1 = RMSNorm(embed_dim, **factory_kwargs)
            self.ln2 = RMSNorm(embed_dim, **factory_kwargs)
        else:
            self.ln1 = nn.LayerNorm(embed_dim, **factory_kwargs)
            self.ln2 = nn.LayerNorm(embed_dim, **factory_kwargs)

        if modern and moe:
            self.ffn = MoEFeedForward(
                embed_dim, dropout,
                **{k: v for k, v in moe.items() if k in (
                    "n_experts", "top_k", "capacity_factor", "aux_loss_coef", "z_loss_coef")},
                expert_offload=expert_offload,
                compute_device=compute_device,
                device=device, dtype=dtype,
            )
        elif modern:
            self.ffn = SwiGLU(embed_dim, dropout, device=device, dtype=dtype)
        else:
            self.ffn = FeedForward(embed_dim, dropout, gelu_tanh=gelu_tanh,
                                   device=device, dtype=dtype)

        self.attn = CausalSelfAttention(
            embed_dim,
            n_heads,
            dropout,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            use_rope=modern,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            window=window,
            max_positions=max_positions,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_in = self.ln1(x)
        if self.attn_checkpoint:
            x = x + activation_checkpoint(
                self.attn, attn_in, rope_cos, rope_sin, kv_cache, attn_mask,
                use_reentrant=False,
            )
        else:
            x = x + self.attn(
                attn_in, rope_cos=rope_cos, rope_sin=rope_sin,
                kv_cache=kv_cache, attn_mask=attn_mask,
            )
        x = x + self.ffn(self.ln2(x))
        return x


# ----------------------------------------------------------------------
# JinAI — model
# ----------------------------------------------------------------------

def _is_local_layer(layer_idx: int, local_layer_every: int | None) -> bool:
    """3 local : 1 global when ``local_layer_every == 4`` (layer i global iff (i+1) % 4 == 0)."""
    if not local_layer_every or local_layer_every <= 0:
        return False
    return (layer_idx + 1) % local_layer_every != 0


class JinAI(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.EMBED_DIM,
        block_size: int = config.BLOCK_SIZE,
        n_layers: int = config.N_LAYERS,
        n_heads: int = config.N_HEADS,
        dropout: float = config.DROPOUT,
        gradient_checkpointing: bool = config.GRADIENT_CHECKPOINTING,
        modern: bool = False,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        rope_theta: float = 10_000.0,
        rope_scaling: dict | None = None,
        sliding_window: int | None = None,
        local_layer_every: int | None = None,
        moe: dict | None = None,
        expert_offload: bool = False,
        static_kv_cache: bool = False,
        kv_cache_int8: bool = False,
        gelu_tanh: bool = False,
        resid_scaled_init: bool = False,
        arch_id: str = "jin_modern",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.arch_id = arch_id
        self.gelu_tanh = bool(gelu_tanh)
        self.resid_scaled_init = bool(resid_scaled_init)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else int(n_kv_heads)
        self.dropout = dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.modern = modern
        self.qk_norm = bool(qk_norm)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.local_layer_every = local_layer_every
        self.moe = moe
        self.expert_offload = bool(expert_offload)
        self.static_kv_cache = bool(static_kv_cache)
        self.kv_cache_int8 = bool(kv_cache_int8)
        # gradient_checkpointing: False | True | "attn" (attention only) |
        # "layer2" (every 2nd block full-checkpointed).
        self.gc_mode = str(gradient_checkpointing) if gradient_checkpointing else False
        if gradient_checkpointing is True:
            self.gc_mode = True

        # Extended context ceiling (RoPE scaling target) vs training block size.
        self.max_positions = int((rope_scaling or {}).get("target_ctx", 0)) or block_size
        if self.max_positions < block_size:
            self.max_positions = block_size

        self.tok_emb = nn.Embedding(vocab_size, embed_dim, **factory_kwargs)
        if not modern:
            self.pos_emb = nn.Embedding(block_size, embed_dim, **factory_kwargs)
        else:
            self.pos_emb = None
        self.drop = nn.Dropout(dropout)

        windows = [
            sliding_window if _is_local_layer(i, local_layer_every) else None
            for i in range(n_layers)
        ]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    n_heads,
                    dropout,
                    modern=modern,
                    n_kv_heads=self.n_kv_heads,
                    qk_norm=self.qk_norm,
                    rope_theta=self.rope_theta,
                    rope_scaling=rope_scaling,
                    window=windows[i],
                    max_positions=self.max_positions,
                    moe=(moe if (moe and (moe.get("layers") in (None, "all")
                                          or (moe.get("layers") == "even" and i % 2 == 0))) else None),
                    expert_offload=self.expert_offload,
                    compute_device=device,
                    attn_checkpoint=(self.gc_mode == "attn"),
                    gelu_tanh=self.gelu_tanh,
                    device=device,
                    dtype=dtype,
                )
                for i in range(n_layers)
            ]
        )

        if modern:
            self.ln_f = RMSNorm(embed_dim, **factory_kwargs)
        else:
            self.ln_f = nn.LayerNorm(embed_dim, **factory_kwargs)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False, **factory_kwargs)
        self.head.weight = self.tok_emb.weight

        if modern:
            cos, sin = build_rope_cache(
                self.max_positions,
                embed_dim // n_heads,
                device=device,
                base=self.rope_theta,
                scaling=rope_scaling,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

        self.apply(self._init_weights)
        if self.resid_scaled_init:
            # GPT-2 scales the residual-writing projections (attention c_proj
            # and MLP c_proj) by 1/sqrt(2*n_layer) so the residual stream does
            # not blow up with depth. Only the projections that WRITE into the
            # residual stream are scaled — qkv/up/gate stay at std 0.02.
            scale = 1.0 / math.sqrt(2 * n_layers)
            with torch.no_grad():
                for block in self.blocks:
                    block.attn.out_proj.weight.mul_(scale)
                    if isinstance(block.ffn, FeedForward):
                        block.ffn.net[2].weight.mul_(scale)
                    elif isinstance(block.ffn, SwiGLU):
                        block.ffn.w_down.weight.mul_(scale)
        print(
            f"[JIN AI] {self.num_parameters() / 1e6:.2f}M total / "
            f"{self.active_parameters() / 1e6:.2f}M active parameters"
            f" ({'modern' if modern else 'classic'}, "
            f"{n_layers}L, {n_heads}H, {self.n_kv_heads}KV, dim {embed_dim})"
        )

    @classmethod
    def from_config(cls, model_config: dict[str, int | float | bool | dict | None]) -> "JinAI":
        return cls(**model_config)

    def export_config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "block_size": self.block_size,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "dropout": self.dropout,
            "gradient_checkpointing": self.gradient_checkpointing,
            "modern": self.modern,
            "arch_id": self.arch_id,
            "gelu_tanh": self.gelu_tanh,
            "resid_scaled_init": self.resid_scaled_init,
            "qk_norm": self.qk_norm,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "sliding_window": self.sliding_window,
            "local_layer_every": self.local_layer_every,
            "moe": self.moe,
            "expert_offload": self.expert_offload,
            "static_kv_cache": self.static_kv_cache,
            "kv_cache_int8": self.kv_cache_int8,
        }

    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def active_parameters(self) -> int:
        """Total parameters excluding MoE experts not routed per forward."""
        active = 0
        for name, param in self.named_parameters():
            if ".experts." in name and ".ffn." in name and self.moe:
                continue  # counted per MoE layer below
            active += param.numel()
        for block in self.blocks:
            if isinstance(block.ffn, MoEFeedForward):
                active += block.ffn.active_parameters()
        return active

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[dict] | None = None,
        loss_mask: torch.Tensor | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        ``doc_mask`` (B, T) marks document starts (1 at the first token of
        each document): attention never crosses document boundaries, so
        packed sequences train without leakage. Used by the packed chat
        dataset (real packing, not just separator tokens).
        """
        _, seq_len = idx.shape
        if seq_len > self.max_positions:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_positions {self.max_positions}"
            )

        # Document-aware attention mask (causal AND same-document).
        attn_mask = None
        if doc_mask is not None and seq_len > 1:
            starts = doc_mask.cumsum(dim=-1)                      # (B, T)
            same_doc = starts[:, :, None] == starts[:, None, :]   # (B, T, T)
            causal = torch.tril(torch.ones(
                seq_len, seq_len, device=idx.device, dtype=torch.bool)).unsqueeze(0)
            allowed = same_doc & causal
            attn_mask = torch.where(
                allowed, torch.zeros((), device=idx.device),
                torch.full((), float("-inf"), device=idx.device),
            )
            # SDPA wants (B, H, T, T) or a broadcastable (B, 1, T, T). A raw
            # (B, T, T) mask only broadcasts at batch size 1, so packed
            # training silently crashed for batch > 1. Add the head axis.
            attn_mask = attn_mask.unsqueeze(1)

        if self.modern:
            x = self.tok_emb(idx)
        else:
            offset = 0
            if kv_caches is not None:
                first_cache = kv_caches[0] if kv_caches else None
                if first_cache:
                    offset = int(first_cache.get("pos", 0))
            positions = torch.arange(
                offset, offset + seq_len, device=idx.device, dtype=torch.long
            )
            x = self.tok_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)

        rope_cos = self.rope_cos if self.modern else None
        rope_sin = self.rope_sin if self.modern else None

        for layer_idx, block in enumerate(self.blocks):
            cache = kv_caches[layer_idx] if kv_caches is not None else None
            if self.gc_mode is True and self.training:
                x = activation_checkpoint(
                    block, x, rope_cos, rope_sin, cache, attn_mask, use_reentrant=False
                )
            elif self.gc_mode == "layer2" and self.training and layer_idx % 2 == 0:
                x = activation_checkpoint(
                    block, x, rope_cos, rope_sin, cache, attn_mask, use_reentrant=False
                )
            else:
                x = block(x, rope_cos=rope_cos, rope_sin=rope_sin,
                          kv_cache=cache, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            if loss_mask is None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )
            else:
                per_token = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="none",
                )
                mask_flat = loss_mask.view(-1).to(per_token.dtype)
                denom = mask_flat.sum().clamp(min=1.0)
                loss = (per_token * mask_flat).sum() / denom

            # MoE auxiliary + z losses (training only).
            if self.training:
                for block in self.blocks:
                    ffn = block.ffn
                    if isinstance(ffn, MoEFeedForward):
                        loss = loss + ffn.aux_loss_coef * ffn.aux_loss.to(loss.dtype)
                        loss = loss + ffn.z_loss_coef * ffn.z_loss.to(loss.dtype)

        return logits, loss

    def empty_kv_cache(
        self,
        batch_size: int = 1,
        static: bool | None = None,
        int8: bool | None = None,
        max_len: int | None = None,
    ) -> list[dict]:
        """Fresh per-layer KV cache.

        ``static=True`` pre-allocates ``(B, n_kv_heads, max_len, head_dim)``
        buffers (no per-step ``cat``) and, with ``int8=True``, stores int8
        values + per-head scales (4x less memory, dequantized on read).
        Local layers keep only their sliding window.
        """
        static = self.static_kv_cache if static is None else bool(static)
        int8 = self.kv_cache_int8 if int8 is None else bool(int8)
        max_len = self.max_positions if max_len is None else int(max_len)
        caches = []
        device = self.rope_cos.device if self.rope_cos is not None else "cpu"
        for block in self.blocks:
            window = block.attn.window
            head_dim = block.attn.head_dim
            kv_heads = block.attn.n_kv_heads
            if static:
                dtype = torch.int8 if int8 else torch.float32
                caches.append({
                    "k_buf": torch.zeros(batch_size, kv_heads, max_len, head_dim,
                                          device=device, dtype=dtype),
                    "v_buf": torch.zeros(batch_size, kv_heads, max_len, head_dim,
                                          device=device, dtype=dtype),
                    "k_scale": torch.ones(batch_size, kv_heads, max_len, 1,
                                           device=device) if int8 else None,
                    "v_scale": torch.ones(batch_size, kv_heads, max_len, 1,
                                           device=device) if int8 else None,
                    "len": 0, "pos": 0, "window": window, "int8": int8,
                })
            else:
                caches.append({"k": None, "v": None, "window": window, "pos": 0})
        return caches

    def _trim_kv_cache(self, kv_caches: list[dict], max_keep: int) -> None:
        """Drop the oldest entries per layer.

        Global layers keep at most ``max_keep`` tokens; local layers keep at
        most their sliding window. The running ``pos`` counter is unaffected
        so RoPE positions stay absolute. Static buffers are shifted in place;
        dynamic caches are re-sliced.
        """
        for cache in kv_caches:
            window = cache.get("window")
            keep = window if window else max_keep
            if cache.get("k_buf") is not None:
                cur = int(cache.get("len", 0))
                if cur > keep:
                    shift = cur - keep
                    k_buf = cache["k_buf"]
                    v_buf = cache["v_buf"]
                    k_buf[:, :, :keep] = k_buf[:, :, shift:cur]
                    v_buf[:, :, :keep] = v_buf[:, :, shift:cur]
                    if cache.get("int8"):
                        cache["k_scale"][:, :, :keep] = cache["k_scale"][:, :, shift:cur]
                        cache["v_scale"][:, :, :keep] = cache["v_scale"][:, :, shift:cur]
                    cache["len"] = keep
                continue
            k = cache.get("k")
            v = cache.get("v")
            if k is not None and k.size(-2) > keep:
                cache["k"] = k[..., -keep:, :]
                cache["v"] = v[..., -keep:, :]

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, seen_ids: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """Sign-aware repetition penalty (HF-style)."""
        if penalty is None or penalty <= 1.0 or seen_ids.numel() == 0:
            return logits
        gathered = logits.gather(-1, seen_ids)
        gathered = torch.where(gathered < 0, gathered * penalty, gathered / penalty)
        logits.scatter_(-1, seen_ids, gathered)
        return logits

    @staticmethod
    def _apply_no_repeat_ngram(
        logits: torch.Tensor,
        idx: torch.Tensor,
        n: int,
        keep_ids: set[int] | None = None,
    ) -> torch.Tensor:
        """Ban tokens that would complete an ``n``-gram already in ``idx``.

        HuggingFace-style ``no_repeat_ngram_size``: take the last ``n-1``
        tokens as the prefix, find every earlier occurrence of that prefix and
        force the token that followed it to ``-inf``. A token-level repetition
        penalty cannot stop a verbatim echo — every token of
        "a small language model" is individually legal, only the *n-gram* is
        not — which is exactly the failure mode seen at SFT step 4,000.

        ``keep_ids`` (EOS / role tags) is never banned: blocking the stop
        token would make the model unable to end its turn.
        """
        if not n or n <= 0 or idx.size(1) < n:
            return logits
        span = idx.size(0)
        for row in range(span):
            seq = idx[row]
            prefix = seq[-(n - 1):]
            windows = seq.unfold(0, n - 1, 1)      # windows[i] = seq[i:i+n-1]
            matches = (windows == prefix).all(dim=-1)
            matches = matches[: seq.size(0) - n + 1]   # drop the partial tail
            if not bool(matches.any()):
                continue
            followers = seq[n - 1:][matches]           # window i -> seq[i+n-1]
            if keep_ids:
                keep = torch.tensor(sorted(keep_ids), device=seq.device,
                                    dtype=seq.dtype)
                followers = followers[~torch.isin(followers, keep)]
            if followers.numel() == 0:
                continue
            row_logits = logits[row]
            # Never blank the entire row — an all -inf row makes softmax NaN.
            if followers.numel() >= int(torch.isfinite(row_logits).sum()):
                continue
            row_logits[followers] = float("-inf")
        return logits

    @staticmethod
    def _steer(
        logits: torch.Tensor,
        idx: torch.Tensor,
        repetition_penalty: float,
        no_repeat_ngram_size: int = 0,
        keep_ids: set[int] | None = None,
    ) -> torch.Tensor:
        """Apply every decoding-time penalty to ``logits`` in place.

        Single choke point so the KV-cache path, the speculative-decoding path
        and the streaming path cannot drift apart in what they forbid.
        """
        logits = JinAI._apply_no_repeat_ngram(
            logits, idx, no_repeat_ngram_size, keep_ids)
        seen = torch.unique(idx).unsqueeze(0).expand(logits.size(0), -1)
        return JinAI._apply_repetition_penalty(logits, seen, repetition_penalty)

    @staticmethod
    def _filter_logits(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits
        logits = logits / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            sorted_logits[remove_mask] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        return logits

    @staticmethod
    def _rfind_sublist(haystack: list, needle: list) -> int:
        """Index of the LAST occurrence of ``needle`` inside ``haystack``, or -1.

        ``str/list.rfind`` semantics for sublists: Python's ``in`` on lists
        only matches whole elements, so a multi-token suffix never matches —
        this is the actual search prompt-lookup needs.
        """
        n_len, h_len = len(needle), len(haystack)
        if n_len == 0 or n_len > h_len:
            return -1
        for start in range(h_len - n_len, -1, -1):
            if haystack[start:start + n_len] == needle:
                return start
        return -1

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        idx: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        repetition_penalty: float,
        no_repeat_ngram_size: int = 0,
        keep_ids: set[int] | None = None,
    ) -> torch.Tensor:
        """Sample one token from ``logits`` (already the right position)."""
        logits = JinAI._steer(logits, idx, repetition_penalty,
                               no_repeat_ngram_size, keep_ids)
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        filtered = JinAI._filter_logits(logits, temperature, top_k, top_p)
        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
        top_k: int | None = config.TOP_K,
        top_p: float | None = config.TOP_P,
        repetition_penalty: float = config.REPETITION_PENALTY,
        stop_ids: set[int] | None = None,
        use_kv_cache: bool = True,
        kv_caches: list[dict] | None = None,
        skip_prefill_tokens: int = 0,
        static_kv: bool | None = None,
        kv_int8: bool | None = None,
        prompt_lookup_k: int = 4,
        prompt_lookup_max_ngram: int = 64,
        no_repeat_ngram_size: int = 0,
        stop_on_repeat: bool = False,
        keep_ids: set[int] | None = None,
        stats: dict | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation.

        ``kv_caches`` + ``skip_prefill_tokens`` support prefix reuse: pass a
        cache already populated with the shared conversation prefix and the
        number of prompt tokens already in it; only the tail gets prefilled.

        ``no_repeat_ngram_size`` bans n-grams already present (see
        :meth:`_apply_no_repeat_ngram`); ``stop_on_repeat`` breaks out early
        when the generated tail loops (see :func:`repeated_tail_period`).
        ``stats``, when given, is filled with ``stop_reason`` (``length`` |
        ``eos`` | ``repeat``), ``generated`` and ``prompt_tokens``.
        """
        was_training = self.training
        self.eval()

        prompt_len = idx.size(1)
        if kv_caches is None and use_kv_cache:
            kv_caches = self.empty_kv_cache(
                batch_size=idx.size(0), static=static_kv, int8=kv_int8)
        cur_input = idx if not use_kv_cache else idx[:, -self.max_positions:]
        if kv_caches is not None and skip_prefill_tokens > 0:
            # Prefix already in cache: only prefill the new tail. Guard:
            # there must be at least one token to prefill (an empty tail
            # would break the decode loop, so fall back to a fresh prefill).
            if skip_prefill_tokens >= idx.size(1):
                skip_prefill_tokens = 0
                kv_caches = self.empty_kv_cache(
                    batch_size=idx.size(0), static=static_kv, int8=kv_int8)
            cur_input = idx[:, skip_prefill_tokens:]

        remaining = max_new_tokens
        generated_ids: list[int] = []
        stop_reason = "length"
        while remaining > 0:
            if stop_on_repeat and generated_ids and \
                    repeated_tail_period(generated_ids) is not None:
                # The model has stopped saying anything new: every extra token
                # is a copy of the last chunk. Cheaper and far more pleasant
                # than spending the rest of the budget on an echo.
                stop_reason = "repeat"
                break
            # ── Prompt-lookup speculative decoding ────────────────────────
            # Draft tokens by copying the longest repeated n-gram from the
            # prompt (free, no extra model). Verified against the real model
            # in one batched forward; wrong drafts are discarded and the KV
            # cache is rewound so rejected tokens never leak in.
            draft: list[torch.Tensor] = []
            can_draft = False
            if use_kv_cache and prompt_lookup_k > 0 and idx.size(1) > 2 and kv_caches:
                # Safety: never draft when the cache could hit its sliding
                # window / max length mid-batch (window slide cannot be
                # rewound by a simple length restore).
                min_room = min(
                    c.get("window") or self.max_positions for c in kv_caches
                )
                start_len = max(
                    int(c.get("len", 0)) if c.get("k_buf") is not None
                    else (int(c["k"].size(-2)) if c.get("k") is not None else 0)
                    for c in kv_caches
                )
                # The batch write covers cur_input PLUS the drafts; the room
                # must fit both, otherwise the write hits a window slide
                # mid-batch, which is unrewindable (buffer content shifted).
                max_draft = min(prompt_lookup_k, remaining,
                                min_room - start_len - cur_input.size(1))
                # Need >= 2 to beat the normal single-token path, and the
                # batch write must never trigger a window slide (unrewindable).
                can_draft = max_draft >= 2
            if can_draft:
                prompt_ids = idx[0].tolist()
                while len(draft) < max_draft:
                    n = min(prompt_lookup_max_ngram, len(prompt_ids) - 1)
                    match_len = 0
                    while n >= 2:
                        suffix = prompt_ids[-n:]
                        # Sublist search (last occurrence), NOT element ``in``
                        # (which can never match a multi-token suffix).
                        found = self._rfind_sublist(prompt_ids[:-n], suffix)
                        if found >= 0:
                            match_len = n
                            break
                        n -= 1
                    if match_len < 2:
                        break
                    nxt = prompt_ids[found + match_len]
                    draft.append(torch.tensor([[nxt]], dtype=idx.dtype, device=idx.device))
                    prompt_ids = prompt_ids + [nxt]

            if draft:
                # Snapshot cache lengths so a partial acceptance can rewind
                # the rejected tail (batch forward writes ALL drafts).
                snaps = [
                    (
                        int(c.get("len", 0)) if c.get("k_buf") is not None
                        else (int(c["k"].size(-2)) if c.get("k") is not None else 0)
                    )
                    for c in kv_caches
                ]
                # Forward [cur_input, *drafts] as one batch: this writes the
                # pending real token AND all drafts. logits[:, i] then
                # predicts draft[i] (the token AFTER the i-th position), so
                # draft[i] is verified against logits[:, i] — no off-by-one.
                draft_ids = torch.cat([cur_input] + draft, dim=1)
                logits, _ = self(draft_ids, kv_caches=kv_caches)
                # Draft d_i sits at batch position cur_len - 1 + d_i, so the
                # logits that PREDICT it are at that same index (logits[:, j]
                # predicts the token at position j+1). When cur_input is the
                # whole prompt (fresh-cache prefill), this is NOT ``d_i``.
                cur_len = cur_input.size(1)
                accepted: list[torch.Tensor] = []
                for d_i in range(len(draft)):
                    logits_t = self._steer(
                        logits[:, cur_len - 1 + d_i, :], idx,
                        repetition_penalty, no_repeat_ngram_size, keep_ids,
                    )
                    target = draft[d_i]
                    if temperature <= 0:
                        guess = torch.argmax(logits_t, dim=-1)
                        ok = (guess == target).item()
                    else:
                        filtered = self._filter_logits(logits_t, temperature, top_k, top_p)
                        probs = F.softmax(filtered, dim=-1)
                        ok = (torch.multinomial(probs, 1).item() == target.item())
                    if not ok:
                        break
                    # draft[i] is already (1, 1); unsqueeze would make it 3D.
                    idx = torch.cat([idx, target], dim=1)
                    accepted.append(target)
                    generated_ids.append(int(target.item()))
                    if stop_ids and int(target.item()) in stop_ids:
                        break
                n_accept = len(accepted)
                # The batch wrote cur_input + ALL drafts. On a rewind we keep
                # every token up to and including the last accepted draft:
                #   base_len (pre-batch) + cur_input + accepted drafts.
                # ``cur_input`` can be multi-token (fresh prefill or a
                # multi-token prefix-reuse tail), so it must be counted — the
                # old ``1 + n_accept`` truncated the prompt out of the cache.
                keep_len = cur_input.size(1) + n_accept
                for c, base_len in zip(kv_caches, snaps):
                    if c.get("k_buf") is not None:
                        c["len"] = base_len + keep_len
                        c["pos"] = c.get("pos", 0) - (len(draft) - n_accept)
                    elif c.get("k") is not None:
                        c["k"] = c["k"][..., : base_len + keep_len, :]
                        c["v"] = c["v"][..., : base_len + keep_len, :]
                        c["pos"] = c.get("pos", 0) - (len(draft) - n_accept)
                if use_kv_cache:
                    self._trim_kv_cache(kv_caches, self.max_positions - 1)
                remaining -= n_accept
                if stop_ids and accepted and int(accepted[-1].item()) in stop_ids:
                    stop_reason = "eos"
                    break
                if n_accept == len(draft):
                    # All drafts accepted: the budget may be exhausted by the
                    # drafts alone (e.g. remaining == len(draft)). Fall
                    # through to the partial branch would sample ONE EXTRA
                    # token beyond max_new_tokens, so stop here if done.
                    if remaining <= 0:
                        break
                    # Otherwise sample the next real token from the
                    # already-computed last logits (predicts the token after
                    # the final draft). cur_input = a fresh, not-yet-cached
                    # token.
                    next_id = self._sample_token(
                        logits[:, -1, :], idx, temperature, top_k, top_p,
                        repetition_penalty, no_repeat_ngram_size, keep_ids,
                    )
                    idx = torch.cat([idx, next_id], dim=1)
                    generated_ids.append(int(next_id.item()))
                    remaining -= 1
                    if stop_ids and int(next_id.item()) in stop_ids:
                        stop_reason = "eos"
                        break
                    cur_input = next_id
                    continue
                # Partial / zero acceptance: the next real token is sampled
                # from the logits at the first rejected position (the draft
                # index is offset by cur_len - 1, same as the verification).
                # n_accept < len(draft) guarantees remaining >= 1 here, so
                # the +1 token stays inside the budget.
                next_id = self._sample_token(
                    logits[:, cur_len - 1 + n_accept, :], idx, temperature,
                    top_k, top_p, repetition_penalty, no_repeat_ngram_size,
                    keep_ids,
                )
                idx = torch.cat([idx, next_id], dim=1)
                generated_ids.append(int(next_id.item()))
                remaining -= 1
                if stop_ids and int(next_id.item()) in stop_ids:
                    stop_reason = "eos"
                    break
                cur_input = next_id
                continue

            if remaining <= 0:
                break
            if cur_input.size(1) > self.max_positions:
                cur_input = cur_input[:, -self.max_positions:]
            logits, _ = self(cur_input, kv_caches=kv_caches)
            logits = logits[:, -1, :]

            logits = self._steer(logits, idx, repetition_penalty,
                                 no_repeat_ngram_size, keep_ids)

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = self._filter_logits(logits, temperature, top_k, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            generated_ids.append(int(next_id.item()))
            remaining -= 1

            if stop_ids and int(next_id.item()) in stop_ids:
                stop_reason = "eos"
                break

            if use_kv_cache:
                self._trim_kv_cache(kv_caches, self.max_positions - 1)
                cur_input = next_id
            else:
                cur_input = idx[:, -self.max_positions:]

        if stats is not None:
            stats["stop_reason"] = stop_reason
            stats["generated"] = len(generated_ids)
            stats["prompt_tokens"] = prompt_len
        if was_training:
            self.train()
        return idx

    @torch.no_grad()
    def stream_generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
        top_k: int | None = config.TOP_K,
        top_p: float | None = config.TOP_P,
        repetition_penalty: float = config.REPETITION_PENALTY,
        stop_ids: set[int] | None = None,
        kv_caches: list[dict] | None = None,
        skip_prefill_tokens: int = 0,
        static_kv: bool | None = None,
        kv_int8: bool | None = None,
        no_repeat_ngram_size: int = 0,
        stop_on_repeat: bool = False,
        keep_ids: set[int] | None = None,
        stats: dict | None = None,
    ):
        """Token-by-token generator (KV cache) for SSE streaming.

        ``kv_caches`` + ``skip_prefill_tokens``: prefix reuse, same contract
        as :meth:`generate`. ``no_repeat_ngram_size`` / ``stop_on_repeat`` /
        ``stats`` behave exactly as in :meth:`generate` — a stream must be
        able to cut an echo short just like a batch call.
        """
        was_training = self.training
        self.eval()

        if kv_caches is None:
            kv_caches = self.empty_kv_cache(
                batch_size=idx.size(0), static=static_kv, int8=kv_int8)
        if skip_prefill_tokens > 0:
            # Same guard as generate(): an empty tail would crash the first
            # ``logits[:, -1]`` slice, so fall back to a fresh prefill.
            if skip_prefill_tokens >= idx.size(1):
                skip_prefill_tokens = 0
                kv_caches = self.empty_kv_cache(
                    batch_size=idx.size(0), static=static_kv, int8=kv_int8)
            cur_input = idx[:, skip_prefill_tokens:]
            logits, _ = self(cur_input, kv_caches=kv_caches)
            logits = logits[:, -1, :]
        else:
            cur_input = idx[:, -self.max_positions:]
            logits, _ = self(cur_input, kv_caches=kv_caches)
            logits = logits[:, -1, :]

        generated_ids: list[int] = []
        stop_reason = "length"
        for _ in range(max_new_tokens):
            if stop_on_repeat and generated_ids and \
                    repeated_tail_period(generated_ids) is not None:
                # Stop before spending the rest of the budget on an echo. The
                # caller trims the last (already streamed) partial copy, see
                # reply_quality.trim_repeated_tail.
                stop_reason = "repeat"
                break
            logits = self._steer(logits, idx, repetition_penalty,
                                 no_repeat_ngram_size, keep_ids)

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = self._filter_logits(logits, temperature, top_k, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            tok = int(next_id.item())
            if stop_ids and tok in stop_ids:
                stop_reason = "eos"
                break
            generated_ids.append(tok)
            yield tok

            idx = torch.cat([idx, next_id], dim=1)
            self._trim_kv_cache(kv_caches, self.max_positions - 1)
            logits, _ = self(next_id, kv_caches=kv_caches)
            logits = logits[:, -1, :]

        if stats is not None:
            stats["stop_reason"] = stop_reason
            stats["generated"] = len(generated_ids)
        if was_training:
            self.train()
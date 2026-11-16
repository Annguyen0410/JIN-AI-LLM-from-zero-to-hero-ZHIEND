"""
LoRA — self-written low-rank adapters (Phase 5, step 6).

Curiosity updates come in two modes:

- **full fine-tune**: the candidate checkpoint is trained directly
  (small LR, replay) and promoted only if it passes the eval gate.
- **LoRA (light)**: a rank-r adapter is trained on just the new-topic
  data, then merged into the base weights ONLY if the eval gate passes.
  Rollback is a one-second unmerge — nothing else touched the weights.

This module implements LoRA from scratch:

- ``wrap_lora(model, rank)`` — replaces the 2D ``nn.Linear`` layers of the
  attention (q/k/v/o) and FFN (gate/up/down) with ``LoRALinear``, which
  keeps the original weight frozen and adds ``A (in,r) @ B (r,out)``.
- ``trainable_parameters`` — the adapter matrices only.
- ``merge()`` / ``unmerge()`` — fold ``B @ A`` into ``weight`` (merge is
  reversible while the original weight is cached; unmerge restores it).
- ``enable_adapters(on/off)`` — bypass the adapter at inference time
  (for A/B comparison: merged vs unmerged) without touching weights.

The default fan-in/out scale is ``1/r`` (standard LoRA init: A ~ N(0, 1/r),
B = 0 so the adapter starts as identity).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank adapter (A @ B)."""

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.has_bias = linear.bias is not None
        # Keep the original weight/bias as the frozen base.
        self.base_weight = nn.Parameter(linear.weight.detach().clone(),
                                        requires_grad=False)
        self.base_bias = None
        if self.has_bias:
            self.base_bias = nn.Parameter(linear.bias.detach().clone(),
                                          requires_grad=False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        # Adapter: x -> (x @ A) @ B; init A ~ N(0, 1/r), B = 0 (identity start).
        self.A = nn.Parameter(
            torch.randn(self.in_features, self.rank) / math.sqrt(max(rank, 1))
        )
        self.B = nn.Parameter(torch.zeros(self.rank, self.out_features))
        self.enabled = True
        self._merged = False
        self._saved_weight: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.linear(x, self.base_weight, self.base_bias)
        if self.enabled and self.rank > 0:
            scale = self.alpha / max(self.rank, 1)
            out = out + (x @ self.A @ self.B) * scale
        return out

    def merge(self) -> None:
        """Fold the adapter into ``base_weight`` (reversible).

        forward:  x @ (A @ B)            shape (..., in) -> (..., out)
        weight:   (out, in)              so the folded delta is
                  delta = (B.t() @ A.t()) * (alpha / rank)   shape (out, in)
        """
        if self._merged or self.rank <= 0:
            return
        with torch.no_grad():
            delta = (self.B.t() @ self.A.t()) * (self.alpha / self.rank)
            self._saved_weight = self.base_weight.detach().clone()
            self.base_weight.add_(delta)
            self._merged = True
            self.enabled = False

    def unmerge(self) -> None:
        """Restore the original weight (one-second rollback)."""
        if not self._merged:
            return
        with torch.no_grad():
            if self._saved_weight is not None:
                self.base_weight.copy_(self._saved_weight)
                self._saved_weight = None
            self._merged = False
            self.enabled = True

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, merged={self._merged}")


_ATTN_LORA_NAMES = ("qkv_proj", "o_proj")
_FFN_LORA_NAMES = ("w_gate", "w_up", "w_down")


def wrap_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    *,
    target_names: tuple[str, ...] = _ATTN_LORA_NAMES + _FFN_LORA_NAMES,
) -> list[LoRALinear]:
    """Replace target 2D Linears with LoRALinear; returns the adapters."""
    adapters: list[LoRALinear] = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in target_names:
            continue
        # Only swap the actual leaf Linear, not containers.
        if not isinstance(module, nn.Linear):
            continue
        parent = model
        for part in name.split(".")[:-1]:
            parent = getattr(parent, part)
        child_name = name.rsplit(".", 1)[-1]
        lora_lin = LoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, child_name, lora_lin)
        adapters.append(lora_lin)
    return adapters


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters that belong to LoRA adapters (A/B) only."""
    out = []
    for name, param in model.named_parameters():
        if ".A" in name or ".B" in name:
            out.append(param)
    return out


def freeze_base(model: nn.Module) -> None:
    """Freeze everything except LoRA A/B parameters."""
    for name, param in model.named_parameters():
        if ".A" in name or ".B" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def merge_all(adapters: list[LoRALinear]) -> None:
    for a in adapters:
        a.merge()


def unmerge_all(adapters: list[LoRALinear]) -> None:
    for a in adapters:
        a.unmerge()


def add_lora_args(parser) -> None:
    parser.add_argument("--lora", action="store_true",
                        help="Train a LoRA adapter instead of full fine-tune.")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (default 16).")
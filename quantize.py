"""
Weight-only quantization (from scratch, no external deps).

Two schemes, applied AFTER loading a checkpoint:

- ``int8``: symmetric per-output-channel — one scale per output row,
  values in [-127, 127]. ~4x less weight memory.
- ``int4``: group-wise 128 — one scale per group of 128 input columns,
  4-bit precision (values stored as int8 bytes in [-8, 7] for simplicity;
  the *precision* is 4-bit). ~8x less weight memory.

Quantized weights are dequantized lazily on first use per device into a
cached fp16 buffer, so memory is saved at rest and compute stays fp16
(matmul precision unchanged from the checkpoint's perspective). Measure the
perplexity delta with ``bench.py --quant int8`` before trusting the numbers.

Why this exists: on a 6GB card, weight-only quantization is what lets a
larger checkpoint (or bigger batch) fit at inference time.

Usage:
    from quantize import apply_weight_quantization
    apply_weight_quantization(model, bits=8)          # int8 per-channel
    apply_weight_quantization(model, bits=4, group=128)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QuantizedLinear(nn.Module):
    """Linear whose weight is stored quantized; dequantized lazily per device."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 8,
        group: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = int(bits)
        self.group = int(group)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Resolve the effective group size ONCE: it must divide in_features
        # evenly (e.g. 992 % 128 != 0 for the small preset), so shrink until
        # it does. Stored so quantize and dequantize always agree.
        g = self.group
        while g > 1 and in_features % g != 0:
            g -= 1
        self.group_used = max(1, g)
        self.n_groups = max(1, in_features // self.group_used)
        # Quantized payload (filled by from_linear): int8 storage always.
        self.qweight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.scales = nn.Parameter(torch.empty(self.n_groups, out_features), requires_grad=False)
        self._dequantized: torch.Tensor | None = None
        self._dequant_device = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 8, group: int = 128) -> "QuantizedLinear":
        ql = cls(linear.in_features, linear.out_features, bias=linear.bias is not None,
                 bits=bits, group=group)
        w = linear.weight.detach()
        if bits == 8:
            scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)   # (out, 1)
            q = torch.round(w / scale * 127.0).to(torch.int8)
            ql.qweight.data.copy_(q)
            ql.scales.data.copy_(scale.squeeze(1).unsqueeze(0))         # (1, out)
        else:  # int4 group-wise
            out, inn = w.shape
            g = ql.group_used
            n_groups = ql.n_groups
            wg = w.view(out, n_groups, g)
            scale = wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)  # (out, n_groups, 1)
            q = torch.round(wg / scale * 7.0).to(torch.int8)            # [-7, 7]
            ql.qweight.data.copy_(q.view(out, inn))
            ql.scales.data.copy_(scale.view(n_groups, out))
        if ql.bias is not None and linear.bias is not None:
            ql.bias.data.copy_(linear.bias.detach())
        return ql

    def dequantize(self) -> torch.Tensor:
        if self._dequantized is not None and self._dequant_device == self.qweight.device:
            return self._dequantized
        w = self.qweight.float()
        if self.bits == 8:
            scale = self.scales.squeeze(0)                     # (out,)
            deq = w / 127.0 * scale.unsqueeze(1)
        else:
            out, inn = w.shape
            g = self.group_used
            n_groups = self.n_groups
            scale = self.scales.view(n_groups, out)            # (n_groups, out)
            deq = (w.view(out, n_groups, g) / 7.0 * scale.t().unsqueeze(-1)).view(out, inn)
        self._dequantized = deq.to(dtype=torch.float16).to(self.qweight.device)
        self._dequant_device = self.qweight.device
        return self._dequantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize()
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        return nn.functional.linear(x, w, self.bias)


def apply_weight_quantization(model: nn.Module, bits: int = 8, group: int = 128,
                              skip_embedding: bool = True) -> int:
    """Replace nn.Linear weights with quantized versions, in place.

    Returns the number of layers quantized. Embeddings / output head can be
    kept at full precision (they dominate accuracy at small scales).
    """
    bits = int(bits)
    if bits not in (8, 4):
        raise ValueError("bits must be 8 (per-channel) or 4 (group-wise 128)")
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if skip_embedding and (name.endswith("head") or "tok_emb" in name):
            continue
        ql = QuantizedLinear.from_linear(module, bits=bits, group=group)
        if name:
            parent_name, _, attr = name.rpartition(".")
            parent = model if not parent_name else model.get_submodule(parent_name)
            setattr(parent, attr, ql)
        else:
            # Root-level Linear passed directly as ``model``.
            model.__class__ = QuantizedLinear  # swap in place (same ctor args)
            model.__dict__.update(ql.__dict__)
        replaced += 1
    return replaced
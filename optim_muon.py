"""
Muon optimizer (from scratch).

Muon ("An optimizer for hidden layers", Keller Jordan; "Muon is Scalable
for LLM Training", Moonshot AI, arXiv 2502.16982) replaces Adam's per-
parameter diagonal scaling with a matrix-orthogonalizing update for 2D
hidden-layer weights:

    m = beta1 * m + (1 - beta1) * g            (momentum)
    w -= lr * scale * orth(m)                  (Newton-Schulz 5 steps)

Why it helps here:
- One momentum buffer per matrix (Adam keeps two) -> ~half the optimizer
  memory, precious on a 6GB card.
- Empirically converges ~1.1-1.4x faster than a well-tuned AdamW (measured
  by Stanford's "Fantastic Pretraining Optimizers", not the 2x some blogs
  promise) — this is what our own A/B bench is for.

The two conditions Moonshot found necessary for scaling:
- decoupled weight decay applied to the weight itself, and
- update scaled by ``sqrt(max(1, fan_out / fan_in))`` for non-square mats.

Params that are NOT 2D hidden-layer matrices (embeddings, head, norms,
biases, 1D tensors) use AdamW inside the same optimizer, so checkpoint
state stays in one place.

Reference behaviours worth keeping in mind:
- orth() is the polar-ish factor; Newton-Schulz 5 approximates sign(G)
  for square G and the near-unitary factor otherwise.
- Recommended LR for Muon is higher than AdamW (typically 2-4x).

Usage:
    from optim_muon import build_muon_optimizer
    optim = build_muon_optimizer(model, lr=6e-4, weight_decay=0.1)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# Newton-Schulz coefficients for 5 steps (from the Muon reference impl).
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def newton_schulz_5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of ``g`` via Newton-Schulz.

    Works on the smaller side of ``g @ g.T``; the result is the polar /
    sign factor (dimension-preserving). ``g`` is normalized first.
    """
    a, b, c = _NS_A, _NS_B, _NS_C
    x = g / (g.norm() + 1e-6)
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.t()
    for _ in range(steps):
        a_ = x @ x.t()
        b_ = b * a_ + c * (a_ @ a_)
        x = a * x + b_ @ x
    if transpose:
        x = x.t()
    return x


class Muon(torch.optim.Optimizer):
    """Muon + AdamW hybrid.

    ``param_groups`` each carry ``mode``:

    - ``"muon"``: 2D hidden-layer matrix -> momentum + orthogonalization.
    - ``"adamw"``: everything else -> standard AdamW.

    The constructor takes plain parameters; use :func:`build_muon_optimizer`
    to get the correct split from a model.
    """

    def __init__(
        self,
        params,
        lr: float = 6e-4,
        betas: tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        nesterov: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            nesterov = group.get("nesterov", True)
            mode = group.get("mode", "adamw")

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                # muon keeps one momentum buffer; adamw keeps two.
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                if mode == "adamw" and "momentum2" not in state:
                    state["momentum2"] = torch.zeros_like(p)

                if wd != 0.0:
                    # Decoupled weight decay, applied to the weight itself
                    # (required for Muon to scale, per Moonshot).
                    p.mul_(1.0 - lr * wd)

                if mode == "muon":
                    m = state["momentum"]
                    # Velocity convention (exactly the Keller-Jordan
                    # reference): buf = beta*buf + g, lookahead g + beta*buf.
                    # Re-mutating ``m`` twice would double-count the current
                    # gradient and corrupt the stored momentum, so the
                    # lookahead is computed WITHOUT touching ``m`` again.
                    m.mul_(beta1).add_(grad)
                    if nesterov:
                        update = newton_schulz_5(grad.add(m, alpha=beta1))
                    else:
                        update = newton_schulz_5(m)
                    # Non-square update scaling (fan_out / fan_in).
                    fan_in, fan_out = p.size(1), p.size(0)
                    scale = math.sqrt(max(1.0, fan_out / fan_in))
                    p.add_(update, alpha=-lr * scale)
                else:
                    m = state["momentum"]
                    v = state["momentum2"]
                    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = v.sqrt().add_(eps)
                    p.addcdiv_(m, denom, value=-lr)

        return loss


def is_muon_param(name: str, param: nn.Parameter) -> bool:
    """Is this parameter a 2D hidden-layer matrix (Muon-eligible)?

    Excludes embeddings (tok_emb / pos_emb), the output head (tied to
    tok_emb), norms and all 1D tensors.
    """
    if param.dim() != 2:
        return False
    if name.startswith("tok_emb.") or name.startswith("pos_emb."):
        return False
    if name == "head.weight" or name.endswith(".head.weight"):
        return False
    if ".ln" in name or "norm" in name:
        return False
    return True


def build_muon_optimizer(
    model: nn.Module,
    lr: float = 6e-4,
    betas: tuple[float, float] = (0.95, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    nesterov: bool = True,
    muon_lr_scale: float = 1.0,
) -> Muon:
    """Split model params into Muon (2D hidden matrices) and AdamW groups."""
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_muon_param(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    groups = []
    if muon_params:
        groups.append({"params": muon_params, "mode": "muon",
                       "lr": lr * muon_lr_scale})
    if adamw_params:
        groups.append({"params": adamw_params, "mode": "adamw", "lr": lr})
    if not groups:
        raise ValueError("No trainable parameters found.")
    return Muon(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                nesterov=nesterov)
"""
Phase 2 tests: Muon optimizer (optim_muon.py) and weight-only quantization
(quantize.py). Pure CPU, stdlib unittest.

Run:  python -m unittest tests.test_optim_muon -v
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

import config
from optim_muon import build_muon_optimizer
from quantize import apply_weight_quantization


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)
        self.bias = nn.Parameter(torch.zeros(8))

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))) + self.bias


class TestMuon(unittest.TestCase):
    def test_muon_steps_and_decreases_loss(self):
        torch.manual_seed(0)
        net = ToyNet()
        opt = build_muon_optimizer(net, lr=1e-2, weight_decay=0.1)
        x = torch.randn(64, 16)
        y = torch.randn(64, 8)

        def loss_fn():
            return (net(x) - y).pow(2).mean()

        l0 = loss_fn().item()
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            loss_fn().backward()
            opt.step()
        l1 = loss_fn().item()
        self.assertLess(l1, l0 * 0.6,
                        f"Muon should reduce loss: {l0:.4f} -> {l1:.4f}")

    def test_muon_groups_partition_parameters(self):
        net = ToyNet()
        opt = build_muon_optimizer(net, lr=1e-3)
        modes = {group.get("mode") for group in opt.param_groups}
        self.assertEqual(modes, {"muon", "adamw"})
        # Every parameter belongs to exactly one group.
        all_params = []
        for group in opt.param_groups:
            all_params.extend(p for p in group["params"] if p.requires_grad)
        self.assertEqual(len(all_params), sum(1 for p in net.parameters() if p.requires_grad))
        # 2D hidden matrices go to Muon; 1D/head/embed stay AdamW.
        muon_names = set()
        for group in opt.param_groups:
            if group.get("mode") == "muon":
                for p in group["params"]:
                    self.assertEqual(p.dim(), 2)

    def test_state_dict_roundtrip(self):
        net = ToyNet()
        opt = build_muon_optimizer(net, lr=1e-3)
        x = torch.randn(8, 16)
        opt.zero_grad()
        net(x).sum().backward()
        opt.step()
        sd = opt.state_dict()
        opt2 = build_muon_optimizer(ToyNet(), lr=1e-3)
        opt2.load_state_dict(sd)
        # After loading, stepping produces identical updates.
        torch.manual_seed(1)
        g1 = torch.randn_like(net.fc1.weight)
        net.fc1.weight.grad = g1.clone()
        opt.step()
        net2 = ToyNet()
        net2.load_state_dict(net.state_dict())
        net2.fc1.weight.grad = g1.clone()
        opt2.step()
        torch.testing.assert_close(net.fc1.weight, net2.fc1.weight)


class TestQuantize(unittest.TestCase):
    def test_int8_roundtrip_preserves_logits(self):
        torch.manual_seed(0)
        net = ToyNet().eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            ref = net(x)
        replaced = apply_weight_quantization(net, bits=8)
        self.assertGreater(replaced, 0)
        with torch.no_grad():
            q = net(x)
        # int8 per-channel: max error ~1/127 of row scale; logits stay close.
        err = (ref - q).abs().mean().item()
        self.assertLess(err, 0.05, f"int8 logit drift too large: {err}")

    def test_int4_roundtrip_preserves_logits(self):
        torch.manual_seed(0)
        net = ToyNet().eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            ref = net(x)
        apply_weight_quantization(net, bits=4, group=8)
        with torch.no_grad():
            q = net(x)
        err = (ref - q).abs().mean().item()
        # int4 group-wise is lossier than int8 but should stay reasonable.
        self.assertLess(err, 0.2, f"int4 logit drift too large: {err}")

    def test_rejects_bad_bits(self):
        net = ToyNet()
        with self.assertRaises(ValueError):
            apply_weight_quantization(net, bits=3)


if __name__ == "__main__":
    unittest.main()
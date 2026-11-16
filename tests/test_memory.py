"""
Phase 4 tests — long-term memory + retrieval (pure numpy/stdlib + torch).

Run:  python -m unittest tests.test_memory -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import torch

from memory import MemoryStore, new_id
from retriever import (
    BM25Index,
    HybridRetriever,
    VectorIndex,
    embed_texts,
)

VOCAB = 256


def make_tiny_model():
    from model import JinAI
    return JinAI(vocab_size=VOCAB, embed_dim=32, block_size=64,
                  n_layers=2, n_heads=2, n_kv_heads=2, dropout=0.0,
                  modern=True, qk_norm=True)


class DummyTokenizer:
    """Minimal tokenizer standing in for BPETokenizer in retriever tests."""

    def __init__(self):
        self.special_to_id = {"<|pad|>": 0}

    def encode(self, text, allow_special=True):
        ids = []
        for ch in text:
            ids.append((ord(ch) * 7 + 1) % 200 + 10)
        return ids[:64]


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "memory.jsonl")

    def test_add_get_delete_roundtrip(self):
        store = MemoryStore(self.path)
        mid = store.add(tier="semantic", text="Paris is the capital of France",
                        claim="Paris is capital", source_url="https://wiki.example",
                        confidence=0.9, embedding=np.random.randn(16).astype(np.float32))
        self.assertTrue(mid.startswith("semantic:"))
        rec = store.get(mid)
        self.assertEqual(rec["text"], "Paris is the capital of France")
        self.assertEqual(rec["source_url"], "https://wiki.example")
        # Reload from disk -> persisted.
        store2 = MemoryStore(self.path)
        self.assertEqual(store2.get(mid)["text"], "Paris is the capital of France")
        # Delete by id.
        self.assertEqual(store.delete(mid=mid), 1)
        self.assertIsNone(store.get(mid))

    def test_delete_by_source_url(self):
        store = MemoryStore(self.path)
        store.add(tier="semantic", text="a", source_url="https://x.example")
        store.add(tier="semantic", text="b", source_url="https://x.example")
        store.add(tier="semantic", text="c", source_url="https://y.example")
        self.assertEqual(store.delete(source_url="https://x.example"), 2)
        self.assertEqual(store.stats()["total"], 1)

    def test_ttl_compaction_drops_expired_episodic(self):
        store = MemoryStore(self.path, episodic_ttl_days=1)
        # Backdate an episodic entry.
        from datetime import datetime, timedelta, timezone
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        mid = store.add(tier="episodic", text="old session",
                        fetched_at=old, session_id="s1")
        store.add(tier="semantic", text="stays forever")
        dropped = store.compact()
        self.assertGreaterEqual(dropped, 1)
        self.assertIsNone(store.get(mid))
        self.assertEqual(store.stats()["semantic"], 1)

    def test_consolidation_merges_near_duplicates(self):
        store = MemoryStore(self.path, consolidate_threshold=0.95)
        a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.99, 0.05, 0.0, 0.0], dtype=np.float32)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        # Near-duplicate TEXTS (different ids) with the SAME claim -> merged.
        m1 = store.add(tier="semantic", text="The sky is blue on a clear day",
                       claim="sky is blue", embedding=a)
        m2 = store.add(tier="semantic",
                       text="The sky appears blue during a clear day",
                       claim="sky is blue", embedding=b)
        self.assertNotEqual(m1, m2)
        report = store.consolidate()
        self.assertGreaterEqual(report["merged"], 1)
        self.assertEqual(store.stats()["total"], 1)
        self.assertIn(m1, [r["id"] for r in store.all()])

    def test_contradiction_detection_flags_disputed(self):
        store = MemoryStore(self.path)
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.98, 0.1, 0.0], dtype=np.float32)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        m1 = store.add(tier="semantic", text="The capital of X is A",
                       claim="capital of X is A", embedding=a)
        m2 = store.add(tier="semantic", text="The capital of X is B",
                       claim="capital of X is B", embedding=b)
        report = store.consolidate()
        self.assertGreaterEqual(report["disputed"], 1)
        nv = store.needs_verification()
        self.assertTrue(any(r["id"] in (m1, m2) for r in nv))

    def test_new_id_content_addressed(self):
        self.assertEqual(new_id("semantic", "hello world"),
                         new_id("semantic", "hello world"))
        self.assertNotEqual(new_id("semantic", "hello world"),
                            new_id("semantic", "hello there"))


class TestVectorIndex(unittest.TestCase):
    def test_brute_force_search_finds_nearest(self):
        idx = VectorIndex(4)
        ids = ["a", "b", "c"]
        embs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                        dtype=np.float32)
        idx.add(ids, embs)
        hits = idx.search(np.array([0.9, 0.1, 0, 0], dtype=np.float32), k=2)
        self.assertEqual(hits[0][0], "a")

    def test_ivf_path(self):
        idx = VectorIndex(4, ivf_threshold=50, nlist=4)
        rng = np.random.default_rng(0)
        embs = rng.normal(size=(100, 4)).astype(np.float32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        idx.add([f"d{i}" for i in range(100)], embs)
        self.assertEqual(idx.stats()["mode"], "ivf")
        hits = idx.search(embs[7], k=5)
        self.assertEqual(hits[0][0], "d7")


class TestBM25(unittest.TestCase):
    def test_keyword_ranking(self):
        bm = BM25Index()
        bm.add(["d1", "d2", "d3"],
               ["the cat sat on the mat",
                "dogs bark loudly at night",
                "a cat and a dog are friends"])
        hits = bm.search("cat mat", k=2)
        self.assertEqual(hits[0][0], "d1")


class TestHybridRetriever(unittest.TestCase):
    def test_rrf_fusion_ranks_relevant_above_irrelevant(self):
        model = make_tiny_model()
        texts = {
            "m1": "Paris is the capital of France",
            "m2": "The Eiffel Tower is in Paris",
            "m3": "How to bake bread at home",
        }
        embs = embed_texts(list(texts.values()), model, DummyTokenizer(),
                           device="cpu")
        vec = VectorIndex(embs.shape[1])
        vec.add(list(texts.keys()), embs)
        bm = BM25Index()
        bm.add(list(texts.keys()), list(texts.values()))
        ret = HybridRetriever(vec, bm, texts)
        q_emb = embed_texts(["Where is the Eiffel Tower?"],
                            model, DummyTokenizer(), device="cpu")[0]
        res = ret.search("Eiffel Tower Paris", q_emb, k=3)
        self.assertEqual(res[0].text, texts["m2"])
        # RRF order: m2 should be top-1 (vector + bm25 both agree).


if __name__ == "__main__":
    unittest.main()
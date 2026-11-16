"""
Retrieval for JIN AI long-term memory — pure numpy, no faiss (Phase 4).

Pipeline
--------
1. ``embed_text`` — embed text with Jin itself: run the model, capture the
   mean-pooled hidden state of the second-to-last layer (via a forward
   hook on the final norm), L2-normalize. fp16 storage.
2. ``VectorIndex`` — brute-force cosine for <= ~100k vectors; beyond that
   an IVF-flat index (numpy kmeans, nlist ~ sqrt(N), nprobe configurable)
   with the same API, so callers do not care which is active.
3. ``BM25Index`` — keyword retrieval from scratch (log-frequency
   saturation, IDF), over the same corpus.
4. ``hybrid_search`` — merge BM25 + vector scores with Reciprocal Rank
   Fusion, then rerank the top-20 with Jin's own logprob
   (score = logprob of "the passage answers the question"), keep top-5.

Everything is deterministic and CPU-safe; the heavy math is numpy.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import torch


def embed_texts(
    texts: list[str],
    model,
    tokenizer,
    *,
    device: str = "cpu",
    batch_size: int = 8,
    pool_layer: str = "ln_f",
) -> np.ndarray:
    """Mean-pooled hidden-state embeddings from Jin (no external model).

    Hooks ``pool_layer`` (default the final norm) and averages token
    hidden states, then L2-normalizes. Returns ``(N, embed_dim)`` fp32
    (the caller may cast to fp16 for storage).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    captured: list[np.ndarray] = []

    def hook(module, inp, out):
        h = out if isinstance(out, torch.Tensor) else out[0]
        captured.append(h.detach().float())

    handle = None
    target = None
    for name, mod in model.named_modules():
        if name == pool_layer or name.endswith("." + pool_layer):
            target = mod
            break
    if target is None:
        raise RuntimeError(f"Model has no module named {pool_layer!r}")
    handle = target.register_forward_hook(hook)

    model.eval()
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            ids_list = []
            for t in batch:
                toks = tokenizer.encode(t, allow_special=True)
                ids_list.append(toks[: model.max_positions])
            seq_lens = [len(t) for t in ids_list]
            max_len = max(seq_lens)
            pad_id = 0
            try:
                pad_id = tokenizer.special_to_id.get("<|pad|>", 0)
            except AttributeError:
                pad_id = 0
            batch_t = torch.zeros((len(batch), max_len), dtype=torch.long,
                                  device=device)
            for r, (toks, sl) in enumerate(zip(ids_list, seq_lens)):
                batch_t[r, :sl] = torch.tensor(toks, dtype=torch.long, device=device)
            captured.clear()
            model(batch_t)
            if not captured:
                raise RuntimeError("Embedding hook never fired")
            h = captured[-1]  # (B, T, D)
            mask = (batch_t != pad_id).unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            embs.append(pooled.cpu().numpy())
    handle.remove()
    return np.concatenate(embs, axis=0)


# ── Vector index (brute-force / IVF-flat) ──────────────────────────────

def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vs every row of ``matrix``."""
    return matrix @ query


class VectorIndex:
    """numpy vector store. Brute-force below a size threshold; IVF-flat
    (numpy kmeans) above it. API: add/remove/search/stats."""

    def __init__(self, dim: int, *, ivf_threshold: int = 100_000,
                 nlist: int = 64, nprobe: int = 8, seed: int = 1337):
        self.dim = int(dim)
        self.ivf_threshold = int(ivf_threshold)
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.seed = int(seed)
        self.ids: list[str] = []
        self.matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._ivf: dict | None = None  # populated lazily past the threshold

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], embs: np.ndarray) -> None:
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs[None, :]
        assert embs.shape[1] == self.dim
        assert len(ids) == embs.shape[0]
        self.matrix = np.concatenate([self.matrix, embs], axis=0)
        self.ids.extend(ids)
        if len(self) > self.ivf_threshold:
            self._build_ivf()

    def remove(self, id_set: set[str]) -> None:
        keep = [i for i, mid in enumerate(self.ids) if mid not in id_set]
        self.ids = [self.ids[i] for i in keep]
        self.matrix = self.matrix[keep]
        self._ivf = None

    def _build_ivf(self) -> None:
        rng = np.random.default_rng(self.seed)
        n = len(self)
        k = min(self.nlist, n)
        # Simple kmeans (forgy init, 10 iters) — good enough for an index.
        centroids = self.matrix[rng.choice(n, k, replace=False)]
        for _ in range(10):
            dists = self.matrix @ centroids.T
            labels = dists.argmax(axis=1)
            new_c = np.stack([
                self.matrix[labels == c].mean(axis=0) if (labels == c).any()
                else centroids[c]
                for c in range(k)
            ])
            centroids = new_c
        lists = {c: np.where(labels == c)[0] for c in range(k)}
        self._ivf = {"centroids": centroids, "lists": lists}

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if self._ivf is None or len(self) <= self.ivf_threshold:
            scores = _cosine_scores(query, self.matrix)
            order = np.argsort(-scores)[:k]
            return [(self.ids[i], float(scores[i])) for i in order]
        # IVF: score only the nprobe closest clusters.
        centroids = self._ivf["centroids"]
        cl_scores = centroids @ query
        probes = np.argsort(-cl_scores)[: self.nprobe]
        cand_idx = np.concatenate([self._ivf["lists"][c] for c in probes])
        scores = _cosine_scores(query, self.matrix[cand_idx])
        order = np.argsort(-scores)[:k]
        return [(self.ids[cand_idx[i]], float(scores[i])) for i in order]

    def stats(self) -> dict:
        return {
            "size": len(self),
            "dim": self.dim,
            "mode": "ivf" if self._ivf is not None else "brute-force",
            "nlist": self.nlist if self._ivf is not None else 0,
        }


# ── BM25 (keyword) index ───────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(unicodedata.normalize("NFC", text).casefold())


class BM25Index:
    """BM25 from scratch (k1=1.5, b=0.75), over pre-tokenized docs."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.doc_lens: list[int] = []
        self.docs: list[list[str]] = []
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avgdl = 1.0

    def add(self, ids: list[str], texts: list[str]) -> None:
        for mid, text in zip(ids, texts):
            toks = _tokenize(text)
            self.ids.append(mid)
            self.docs.append(toks)
            self.doc_lens.append(len(toks))
            for tok in set(toks):
                self._df[tok] = self._df.get(tok, 0) + 1
        n = len(self.ids)
        self._avgdl = (sum(self.doc_lens) / n) if n else 1.0
        self._idf = {
            tok: math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for tok, df in self._df.items()
        }

    def remove(self, id_set: set[str]) -> None:
        keep = [i for i, mid in enumerate(self.ids) if mid not in id_set]
        self.ids = [self.ids[i] for i in keep]
        self.docs = [self.docs[i] for i in keep]
        self.doc_lens = [self.doc_lens[i] for i in keep]
        n = len(self.ids)
        self._avgdl = (sum(self.doc_lens) / n) if n else 1.0
        self._df = {}
        for tokens in self.docs:
            for token in set(tokens):
                self._df[token] = self._df.get(token, 0) + 1
        self._idf = {token: math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                     for token, df in self._df.items()}

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        qtoks = list(dict.fromkeys(_tokenize(query)))
        if not qtoks or not self.ids:
            return []
        scores: list[tuple[int, float]] = []
        for i, toks in enumerate(self.docs):
            s = 0.0
            tf = {t: 0 for t in qtoks}
            for t in toks:
                if t in tf:
                    tf[t] += 1
            dl = self.doc_lens[i]
            norm = 1.0 - self.b + self.b * dl / self._avgdl
            for t in qtoks:
                if t not in self._idf:
                    continue
                ft = tf.get(t, 0)
                s += self._idf[t] * ft * (self.k1 + 1) / (ft + self.k1 * norm)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return [(self.ids[i], float(s)) for i, s in scores[:k]]

    def stats(self) -> dict:
        return {"size": len(self.ids), "vocab": len(self._df)}


# ── Hybrid retrieval + rerank ──────────────────────────────────────────

def _rrf(docs: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over several ranked lists."""
    fused: dict[str, float] = {}
    for ranked in docs:
        for rank, (mid, _score) in enumerate(ranked):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (k + rank + 1)
    order = sorted(fused.items(), key=lambda x: -x[1])
    return order


@dataclass
class RetrievalResult:
    id: str
    text: str
    score: float
    sources: dict = field(default_factory=dict)


class HybridRetriever:
    """BM25 + vector -> RRF -> optional logprob rerank (top-20 -> top-5)."""

    def __init__(self, vector: VectorIndex, bm25: BM25Index,
                 texts: dict[str, str] | None = None):
        self.vector = vector
        self.bm25 = bm25
        self.texts: dict[str, str] = texts if texts is not None else {}

    def register_texts(self, texts: dict[str, str]) -> None:
        self.texts.update(texts)

    def _rerank(self, mids: list[str], query: str, model, tokenizer,
                device: str) -> list[tuple[str, float]]:
        """Score each candidate with Jin's own logprob: P(query | passage)."""
        if model is None or not mids:
            return [(m, 0.0) for m in mids]
        scores: list[float] = []
        with torch.no_grad():
            for mid in mids:
                passage = self.texts.get(mid, "")
                text = f"Question: {query}\nPassage: {passage}\nThe passage answers the question."
                toks = tokenizer.encode(text, allow_special=True)
                toks = toks[: model.max_positions]
                if len(toks) < 3:
                    scores.append(0.0)
                    continue
                idx = torch.tensor([toks[:-1]], dtype=torch.long, device=device)
                tgt = torch.tensor([toks[1:]], dtype=torch.long, device=device)
                logits, _ = model(idx)
                logp = torch.nn.functional.log_softmax(
                    logits.float(), dim=-1)
                gathered = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                scores.append(float(gathered.mean().item()))
        order = sorted(range(len(mids)), key=lambda i: -scores[i])
        return [(mids[i], scores[i]) for i in order]

    def search(
        self,
        query: str,
        query_emb: np.ndarray | None,
        k: int = 5,
        *,
        model=None,
        tokenizer=None,
        device: str = "cpu",
    ) -> list[RetrievalResult]:
        vec_hits = self.vector.search(query_emb, k=20) if query_emb is not None else []
        bm25_hits = self.bm25.search(query, k=20)
        fused = _rrf([vec_hits, bm25_hits])
        # Ties should prefer lexical evidence, not the accidental insertion
        # order of an untrained embedding model's vector neighbors.
        lexical = dict(bm25_hits)
        fused.sort(key=lambda item: (-item[1], -lexical.get(item[0], 0.0)))
        top20 = [mid for mid, _ in fused][:20]
        if model is not None:
            ranked = self._rerank(top20, query, model, tokenizer, device)
        else:
            ranked = fused[:20]
        out: list[RetrievalResult] = []
        for mid, score in ranked[:k]:
            out.append(RetrievalResult(
                id=mid, text=self.texts.get(mid, ""), score=score,
                sources={"rrf_rank": next(
                    (r for r, (_m, _s) in enumerate(fused) if _m == mid), -1)},
            ))
        return out
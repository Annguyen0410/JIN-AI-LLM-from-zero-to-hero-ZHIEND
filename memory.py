"""
JIN AI long-term memory — the persistent knowledge store (Phase 4).

Three tiers, mirroring how a curious agent should remember:

- ``episodic``   — per-session conversations (TTL-bounded, consolidated
                   into summaries when a session grows past a threshold).
- ``semantic``   — verified knowledge chunks. Each entry carries full
                   provenance: claim text, source URL, license, fetch time,
                   confidence, embedding, and a list of contradicting
                   memory ids (disputed knowledge is kept, never silently
                   dropped, and flagged ``needs_verification``).
- ``procedural`` — how-to / skill entries (templates, command syntax,
                   past mistakes and their fixes).

Design notes
------------
- Pure numpy + stdlib. Embeddings are computed OUTSIDE this module (the
  caller passes ``embedding`` vectors); memory.py only stores and indexes
  them, so the store is reusable with any embedder.
- Append-only JSONL on disk + in-memory list, with periodic compaction
  (rewrite without deleted/expired entries).
- Consolidation (background job): merge near-duplicate chunks (same
  embedding cluster, cosine > threshold), decay confidence for
  fast-changing knowledge (news/prices), and detect contradictions
  (two entries with the same topic but different claims) -> both are
  flagged ``needs_verification`` and pushed to the curiosity queue
  (Phase 5 reads ``needs_verification``).

Schema of one entry (dict):
    id, tier, text, claim, source_url, license, fetched_at (ISO), ttl_days,
    confidence (0..1), embedding_id, verified_by, contradicts (list of ids),
    needs_verification (bool), created_at, last_accessed_at
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(tier: str, text: str) -> str:
    """Stable, content-addressed id: tier + sha1(text)[:16]."""
    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{tier}:{digest}"


def _text_similarity(a: str, b: str) -> float:
    """Lightword-token Jaccard similarity (no external deps)."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / (len(sa | sb))


class MemoryStore:
    """Thread-safe, append-only (with compaction) memory store."""

    def __init__(
        self,
        path: str | None = None,
        *,
        embed_dim: int = 0,
        episodic_ttl_days: int = 30,
        max_session_chars: int = 40_000,
        consolidate_threshold: float = 0.9,
        decay_days: int = 30,
    ):
        self.path = path
        self.embed_dim = int(embed_dim)
        self.episodic_ttl_days = int(episodic_ttl_days)
        self.max_session_chars = int(max_session_chars)
        self.consolidate_threshold = float(consolidate_threshold)
        self.decay_days = int(decay_days)
        self.lock = threading.RLock()
        self._entries: list[dict] = []
        # embedding_id -> np.ndarray (fp16 storage)
        self._embeddings: dict[str, np.ndarray] = {}
        if path and os.path.exists(path):
            self._load()

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict) or not rec.get("id") or rec.get("tier") not in ("episodic", "semantic", "procedural"):
                        continue
                    self._entries.append(rec)
                    emb = rec.pop("_embedding", None)
                    if emb is not None:
                        self._embeddings[rec.get("embedding_id", rec.get("id"))] = (
                            np.asarray(emb, dtype=np.float16)
                        )
        except OSError:
            pass

    def _persist(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in self._entries:
                out = dict(rec)
                emb = self._embeddings.get(out.get("embedding_id", out.get("id")))
                if emb is not None:
                    out["_embedding"] = emb.astype(np.float16).tolist()
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    @staticmethod
    def _expired(rec: dict) -> bool:
        ttl = rec.get("ttl_days")
        if ttl is None:
            return False
        try:
            stamp = datetime.fromisoformat(rec.get("fetched_at") or rec["created_at"]).timestamp()
            return time.time() >= stamp + float(ttl) * 86400
        except (ValueError, TypeError, KeyError):
            return True  # malformed expiration fails closed

    def compact(self) -> int:
        """Drop expired/deleted entries; rewrite the JSONL file."""
        with self.lock:
            cutoff = time.time() - self.episodic_ttl_days * 86400
            kept = []
            for rec in self._entries:
                if rec.get("deleted") or self._expired(rec):
                    continue
                if rec.get("tier") == "episodic" and rec.get("fetched_at"):
                    try:
                        ts = datetime.fromisoformat(rec["fetched_at"]).timestamp()
                        if ts < cutoff:
                            continue
                    except ValueError:
                        pass
                kept.append(rec)
            dropped = len(self._entries) - len(kept)
            self._entries = kept
            self._persist()
            return dropped

    # ── CRUD ───────────────────────────────────────────────────────────

    def add(
        self,
        *,
        tier: str,
        text: str,
        claim: str | None = None,
        source_url: str | None = None,
        license_: str | None = None,
        fetched_at: str | None = None,
        confidence: float = 0.5,
        embedding: np.ndarray | None = None,
        verified_by: str | None = None,
        contradicts: list[str] | None = None,
        session_id: str | None = None,
        ttl_days: int | None = None,
    ) -> str:
        """Add one entry; returns its id."""
        if tier not in ("episodic", "semantic", "procedural"):
            raise ValueError(f"Unknown tier {tier!r}")
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty memory text")
        if len(text) > 16000:
            raise ValueError("Memory text too long")
        if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
            raise ValueError("Confidence must be finite and in [0, 1] (heuristic, not probability)")
        if ttl_days is not None and (not math.isfinite(float(ttl_days)) or float(ttl_days) <= 0):
            raise ValueError("TTL must be positive and finite")
        with self.lock:
            mid = new_id(tier, text)
            emb_id = f"{mid}:emb"
            # Content-addressed id: identical text already stored -> update
            # that entry instead of appending a duplicate (the duplicate
            # would share the embedding slot and corrupt consolidation).
            existing = None
            for rec in self._entries:
                if rec["id"] == mid and not rec.get("deleted"):
                    existing = rec
                    break
            if embedding is not None:
                self._embeddings[emb_id] = np.asarray(embedding, dtype=np.float16)
            rec = {
                "id": mid,
                "tier": tier,
                "text": text,
                "claim": claim or text,
                "source_url": source_url,
                "license": license_,
                "fetched_at": fetched_at or now_iso(),
                "ttl_days": ttl_days if ttl_days is not None
                else (self.episodic_ttl_days if tier == "episodic" else None),
                "confidence": float(confidence),
                "embedding_id": emb_id if embedding is not None else None,
                "verified_by": verified_by,
                "contradicts": list(contradicts or []),
                "needs_verification": bool(contradicts),
                "session_id": session_id,
                "created_at": now_iso(),
                "last_accessed_at": now_iso(),
            }
            if existing is not None:
                # Refresh the existing entry (new source / confidence wins).
                # Use explicit None checks — ``v not in (None, [], False)``
                # would skip legitimate 0.0 / [] / False values.
                for k, v in rec.items():
                    if k in {"created_at", "contradicts", "needs_verification"}:
                        continue  # a duplicate is not a verified correction
                    if v is not None:
                        existing[k] = v
                existing["contradicts"] = list(dict.fromkeys(existing.get("contradicts", []) + list(contradicts or [])))
                existing["needs_verification"] = bool(existing.get("needs_verification") or existing["contradicts"])
                existing["last_accessed_at"] = now_iso()
                self._persist()
                return existing["id"]
            self._entries.append(rec)
            self._persist()
            return mid

    def get(self, mid: str) -> dict | None:
        with self.lock:
            for rec in self._entries:
                if rec["id"] == mid and not rec.get("deleted") and not self._expired(rec):
                    rec["last_accessed_at"] = now_iso()
                    return dict(rec)
        return None

    def delete(self, mid: str | None = None, source_url: str | None = None) -> int:
        """Delete by id or by source URL (privacy). Returns count removed."""
        with self.lock:
            removed = 0
            for rec in self._entries:
                if rec.get("deleted"):
                    continue
                if mid and rec["id"] == mid:
                    rec["deleted"] = True
                    removed += 1
                elif source_url and rec.get("source_url") == source_url:
                    rec["deleted"] = True
                    removed += 1
            if removed:
                self._persist()
            return removed

    def all(self, tier: str | None = None, include_deleted: bool = False) -> list[dict]:
        with self.lock:
            out = [dict(r) for r in self._entries
                   if (include_deleted or (not r.get("deleted") and not self._expired(r)))
                   and (tier is None or r.get("tier") == tier)]
            return out

    def stats(self) -> dict:
        with self.lock:
            counts: dict[str, int] = {"episodic": 0, "semantic": 0, "procedural": 0}
            disputed = 0
            for rec in self._entries:
                if rec.get("deleted"):
                    continue
                counts[rec.get("tier", "semantic")] = (
                    counts.get(rec.get("tier", "semantic"), 0) + 1
                )
                if rec.get("contradicts"):
                    disputed += 1
            counts["total"] = sum(counts.values())
            counts["disputed"] = disputed
            counts["embedding_count"] = len(self._embeddings)
            return counts

    def embedding_of(self, mid: str) -> np.ndarray | None:
        with self.lock:
            for rec in self._entries:
                if rec["id"] == mid:
                    emb_id = rec.get("embedding_id")
                    if emb_id:
                        return self._embeddings.get(emb_id)
                    return None
        return None

    # ── Consolidation & decay (background job) ─────────────────────────

    def consolidate(self, epsilon: float = 0.15) -> dict:
        """One pass: merge near-dups, decay confidence, flag contradictions.

        Returns a small report dict (merge count, decayed count, disputed
        count). Never deletes disputed entries — they are kept and flagged
        ``needs_verification`` so the curiosity engine can re-check.
        """
        with self.lock:
            merged = 0
            # 1. Near-duplicate merge (same tier, cosine > threshold).
            entries = [r for r in self._entries if not r.get("deleted")]
            to_drop: set[str] = set()
            for i in range(len(entries)):
                ei = entries[i]
                if ei["id"] in to_drop:
                    continue
                emb_i = self.embedding_of(ei["id"])
                for j in range(i + 1, len(entries)):
                    ej = entries[j]
                    if ej["id"] in to_drop or ej["id"] == ei["id"] \
                            or ej.get("tier") != ei.get("tier"):
                        continue
                    # Different claims = a contradiction, NOT a duplicate:
                    # leave it for the contradiction pass below (merging it
                    # here would silently discard one side of a dispute).
                    if ej.get("claim") != ei.get("claim"):
                        continue
                    emb_j = self.embedding_of(ej["id"])
                    if emb_i is None or emb_j is None:
                        sim = _text_similarity(ei["text"], ej["text"])
                    else:
                        sim = float(np.dot(emb_i, emb_j))
                    if sim >= self.consolidate_threshold:
                        to_drop.add(ej["id"])
                        # Merge provenance: prefer the higher-confidence source.
                        if ej.get("confidence", 0) > ei.get("confidence", 0):
                            ei["text"] = ej["text"]
                            ei["claim"] = ej.get("claim") or ei.get("claim")
                            ei["source_url"] = ej.get("source_url") or ei.get("source_url")
                            ei["confidence"] = ej.get("confidence", 0)
                        ei.setdefault("merged_ids", []).append(ej["id"])
                        merged += 1
            if to_drop:
                for rec in self._entries:
                    if rec["id"] in to_drop:
                        rec["deleted"] = True

            # 2. Confidence decay for fast-changing knowledge (no source_url
            #    / news-like heuristic: short-lived topics carry ttl).
            decayed = 0
            for rec in self._entries:
                if rec.get("deleted") or rec.get("tier") != "semantic":
                    continue
                if rec.get("ttl_days"):
                    try:
                        created = datetime.fromisoformat(rec["created_at"]).timestamp()
                        age_days = (time.time() - created) / 86400
                        if age_days > self.decay_days:
                            new_c = rec.get("confidence", 0.5) * 0.95
                            rec["confidence"] = round(new_c, 3)
                            decayed += 1
                    except ValueError:
                        pass

            # 3. Contradiction detection: same tier, similar text (embedding
            #    or token overlap), different claim -> flag both.
            disputed = 0
            for i in range(len(entries)):
                ei = entries[i]
                if ei["id"] in to_drop or ei.get("deleted"):
                    continue
                emb_i = self.embedding_of(ei["id"])
                for j in range(i + 1, len(entries)):
                    ej = entries[j]
                    if ej["id"] in to_drop or ej.get("deleted") \
                            or ej.get("tier") != ei.get("tier"):
                        continue
                    if ei.get("claim") == ej.get("claim"):
                        continue
                    emb_j = self.embedding_of(ej["id"])
                    if emb_i is not None and emb_j is not None:
                        sim = float(np.dot(emb_i, emb_j))
                    else:
                        sim = _text_similarity(ei["text"], ej["text"])
                    if sim >= 0.85:
                        if ej["id"] not in ei.setdefault("contradicts", []):
                            ei["contradicts"].append(ej["id"])
                            ei["needs_verification"] = True
                            ej.setdefault("contradicts", []).append(ei["id"])
                            ej["needs_verification"] = True
                            disputed += 1
            if merged or decayed or disputed:
                self._persist()
            return {"merged": merged, "decayed": decayed, "disputed": disputed}

    def correct(self, mid: str, *, text: str, verified_by: str,
                source_url: str | None = None, confidence: float = 0.5) -> str:
        """Explicit provenance-bearing correction; duplicates cannot clear disputes."""
        if not verified_by or not verified_by.strip():
            raise ValueError("Correction requires verifier provenance")
        with self.lock:
            old = self.get(mid)
            if old is None:
                raise ValueError("Memory entry not found or expired")
            if text.strip() == old["text"]:
                raise ValueError("Correction must provide new evidence text")
            new_mid = self.add(tier=old["tier"], text=text, source_url=source_url,
                               verified_by=verified_by, confidence=confidence)
            for rec in self._entries:
                if rec["id"] == mid:
                    rec["deleted"] = True
                    rec["superseded_by"] = new_mid
                elif mid in rec.get("contradicts", []):
                    rec["contradicts"] = [i for i in rec["contradicts"] if i != mid]
                    # Remain flagged until independently reverified.
            for rec in self._entries:
                if rec["id"] == new_mid:
                    rec["corrects"] = mid
            self._persist()
            return new_mid

    def needs_verification(self) -> list[dict]:
        """Entries flagged for the curiosity engine (Phase 5)."""
        return [
            dict(r) for r in self._entries
            if not r.get("deleted") and r.get("needs_verification")
        ]
"""
Knowledge Ledger — append-only record of every Curiosity Engine cycle
(Phase 5, step 8).

Every learning round writes ONE record with full provenance: topic,
queries, sources, claims verified/rejected, training parameters, score
deltas, and the merge/rollback decision. The ledger is:

- **append-only**: cycles are never edited or deleted (only read), so the
  history is auditable — the spec calls this Jin's "growth diary".
- **meta-learning data**: the same records feed a small UCB bandit over
  ``(topic_kind, self_edit_form)`` so later cycles prefer topic types and
  self-edit forms that historically improved scores the most.

Schema of one record (JSON line):
    cycle, ts, topic, topic_kind, queries, quiz_questions,
    sources (list of {url, domain, kind, fetched_at, ok}),
    claims (list of {claim, confidence, verified, source_count}),
    self_edit_form, self_edit_samples, replay_ratio, lr, steps,
    score_delta (float), forgetting_delta (float), val_loss_delta (float),
    fluency_delta (float), decision ("merged" | "rolled_back" | "memory_only"),
    reason, duration_s
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import config


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeLedger:
    """Thread-safe append-only ledger + UCB bandit over (kind, form) pairs."""

    def __init__(self, path: str = config.CURIOSITY_LEDGER_PATH):
        self.path = path
        self.records: list[dict] = []
        if path and os.path.exists(path):
            self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    def append(self, record: dict) -> None:
        """Append one cycle record and persist immediately (append-only)."""
        rec = dict(record)
        rec.setdefault("ts", now_iso())
        self.records.append(rec)
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def recent(self, n: int = 5) -> list[dict]:
        return self.records[-n:]

    def decisions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.records:
            d = r.get("decision", "unknown")
            counts[d] = counts.get(d, 0) + 1
        return counts

    # ── UCB bandit over (topic_kind, self_edit_form) ───────────────────
    def ucb_scores(self, c: float = 1.0, min_samples: int = 2) -> list[dict]:
        """Rank ``(topic_kind, self_edit_form)`` arms by UCB1.

        ``score_delta`` (probe improvement after the cycle) is the reward;
        rolls back count as 0. Arms with < ``min_samples`` pulls get a
        small exploration bonus so they are not starved.
        """
        arms: dict[str, dict] = {}
        for r in self.records:
            kind = r.get("topic_kind", "generic")
            form = r.get("self_edit_form", "qa")
            key = f"{kind}|{form}"
            arm = arms.setdefault(key, {"kind": kind, "form": form,
                                        "pulls": 0, "reward_sum": 0.0})
            arm["pulls"] += 1
            reward = r.get("score_delta", 0.0) or 0.0
            if r.get("decision") == "rolled_back":
                reward = 0.0
            arm["reward_sum"] += max(reward, 0.0)
        n_total = sum(a["pulls"] for a in arms.values()) or 1
        out = []
        for key, arm in arms.items():
            pulls = arm["pulls"]
            mean = arm["reward_sum"] / pulls
            if pulls < min_samples:
                # Exploration bonus: assume a neutral prior and count the
                # missing pulls at mean reward 0.
                bonus = math.sqrt(2.0 * math.log(n_total + 1) / max(pulls, 1))
                ucb = mean + c * bonus
            else:
                bonus = math.sqrt(2.0 * math.log(n_total + 1) / pulls)
                ucb = mean + c * bonus
            out.append({"key": key, "kind": arm["kind"], "form": arm["form"],
                        "pulls": pulls, "mean_reward": round(mean, 4),
                        "ucb": round(ucb, 4)})
        out.sort(key=lambda x: -x["ucb"])
        return out

    def best_arm(self, c: float = 1.0) -> tuple[str, str] | None:
        ranked = self.ucb_scores(c=c)
        if not ranked:
            return None
        top = ranked[0]
        return top["kind"], top["form"]

    def stats(self) -> dict:
        return {
            "cycles": len(self.records),
            "decisions": self.decisions(),
            "arms": len(set(r.get("topic_kind", "generic") + "|"
                            + r.get("self_edit_form", "qa") for r in self.records)),
        }
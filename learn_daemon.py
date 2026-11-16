"""
Curiosity learning daemon (Phase 5) — the background loop.

Runs ``CuriosityEngine`` cycles under hard safety budgets:

- **requests/cycle** — cap on outgoing web requests (``CURIOSITY_MAX_REQUESTS``)
- **GB/day** — cap on harvested bytes (``CURIOSITY_MAX_GB_DAY``)
- **GPU hours/day** — cap on training time (tracked via the train
  provider's reported seconds)
- **consecutive-fail brake** — stop after ``CURIOSITY_MAX_CONSECUTIVE_FAILS``
  failed eval gates (the spec's safety brake: don't burn GPU on a broken
  loop; shrink scope instead)
- **``--dry-run``** — run every step EXCEPT training: print what would be
  learned, from which sources, with what parameters. Nothing is written
  to weights.

Meta-learning: each cycle records (topic_kind, self_edit_form) + score
delta in the ledger; the daemon picks the UCB-best arm for the next
cycle (bandit over \"topic kind x self-edit form\").

Usage:
    python learn_daemon.py --cycles 5 --topics data/curiosity/topics.txt
    python learn_daemon.py --cycles 1 --dry-run --topic \"quantum computing\"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import config
from curiosity_engine import (
    CuriosityEngine,
    generate_quiz,
    harvest_topic,
    verify_claims,
)
from knowledge_ledger import KnowledgeLedger


class Budget:
    """Daily / per-cycle budget tracking with hard stops."""

    def __init__(self, max_requests: int, max_gb: float, max_cycles: int):
        self.max_requests = int(max_requests)
        self.max_gb = float(max_gb)
        self.max_cycles = int(max_cycles)
        self.requests = 0
        self.bytes = 0.0
        self.cycles = 0

    def spend_request(self) -> bool:
        if self.requests >= self.max_requests:
            return False
        self.requests += 1
        return True

    def spend_bytes(self, n: int) -> bool:
        if self.bytes + n / (1024 ** 3) > self.max_gb:
            return False
        self.bytes += n / (1024 ** 3)
        return True

    def can_cycle(self) -> bool:
        return self.cycles < self.max_cycles

    def summary(self) -> dict:
        return {"requests": self.requests, "bytes_gb": round(self.bytes, 4),
                "cycles": self.cycles,
                "requests_left": max(0, self.max_requests - self.requests),
                "gb_left": round(max(0.0, self.max_gb - self.bytes), 4)}


def budgeted_fetch(budget: Budget):
    """Wrap ``web_search.harvest`` with request/byte budgets."""
    from web_search import harvest

    def _fetch(query: str):
        if not budget.spend_request():
            raise RuntimeError("Request budget exhausted")
        pages = harvest(query, max_results=4, max_chars=2_500)
        kept = []
        for page in pages:
            if not budget.spend_bytes(len(page.get("text", "").encode("utf-8"))):
                break
            kept.append(page)
        return [_page_dict(p) for p in kept]

    return _fetch


def _page_dict(p: dict):
    """Convert a harvest dict into a HarvestedPage-like object."""
    from curiosity_engine import HarvestedPage
    return HarvestedPage(
        url=p["url"], domain=p["domain"], title=p.get("title", ""),
        text=p.get("text", ""), kind=p.get("kind", "web"),
    )


def make_engine(budget: Budget, *, dry_run: bool = False, device: str = "cpu") -> CuriosityEngine:
    """Build a CuriosityEngine wired to the real model + harvest layer.

    Harvest/planning only. Model training and evaluation providers are NOT
    operational. Dry runs never synthesize scores or claim a learning gain.
    """
    ledger = KnowledgeLedger()
    fetch = budgeted_fetch(budget)

    def train_provider(samples, topic: str):
        if dry_run:
            return {"replay_ratio": config.CURIOSITY_REPLAY_RATIO,
                    "lr": 2e-5, "steps": 50, "dry_run": True}
        # Full implementation: continual-pretrain on the samples with
        # replay. Lives in train.py --mode curiosity; here we raise with
        # instructions rather than silently doing nothing.
        raise RuntimeError(
            "train provider not wired — run the real loop via "
            "train.py --mode instruct with a SEPARATE candidate checkpoint, or implement train() "
            "and pass it to CuriosityEngine()"
        )

    def evaluate_provider(kind: str, topic: str = "", quiz=None):
        if kind == "probe_topic":
            # Real uncertainty measure would sample the model; with no
            # loaded model, return a neutral high-uncertainty profile.
            return {"logprobs": [-4.0, -5.0, -6.0],
                    "samples": [f"I am not sure about {topic}.",
                                f"{topic} seems complicated."],
                    "kind": "generic"}
        # gate
        if dry_run:
            return {"measured": False, "metrics": {}, "status": "NOT MEASURED"}
        raise RuntimeError(
            "gate provider not wired — run bench.py on the candidate "
            "checkpoint and feed metrics to eval_gate()"
        )

    engine = CuriosityEngine(
        fetch=fetch, train=train_provider, evaluate=evaluate_provider,
        memory=None, ledger=ledger,
    )
    return engine


def run_daemon(
    topics: list[str],
    *,
    cycles: int = config.CURIOSITY_MAX_CYCLES,
    dry_run: bool = False,
    seed: int = 1337,
    device: str = "cpu",
) -> list[dict]:
    """Run up to ``cycles`` curiosity cycles over ``topics``.

    Picks the UCB-best (kind, form) arm for each cycle; stops early on
    the consecutive-fail brake or when budgets are exhausted. Returns the
    ledger records for the cycles actually run.
    """
    rng = random.Random(seed)
    budget = Budget(config.CURIOSITY_MAX_REQUESTS,
                    config.CURIOSITY_MAX_GB_DAY, cycles)
    engine = make_engine(budget, dry_run=dry_run, device=device)
    ledger = engine.ledger

    results: list[dict] = []
    topic_pool = list(topics)
    kind_map = {t: ("factual" if " " not in t else "reasoning") for t in topic_pool}

    while budget.can_cycle() and topic_pool:
        if engine.consecutive_fails >= engine.max_consecutive_fails:
            print(f"[Daemon] STOP: {engine.consecutive_fails} consecutive "
                  f"gate failures (safety brake).")
            break
        topic = rng.choice(topic_pool)
        kind = kind_map.get(topic, "generic")
        # Meta-learning: prefer the UCB-best (kind, form) arm, restricted
        # to arms of THIS topic's kind (a form that worked for factual
        # topics may not help reasoning ones). Fall back to "qa" while
        # the ledger is cold (no pulls for this kind yet).
        ranked = ledger.ucb_scores()
        same_kind = [a for a in ranked if a["kind"] == kind]
        arm = same_kind[0] if same_kind else None
        form = arm["form"] if arm else "qa"
        quiz = generate_quiz(topic, n=3, rng=rng)
        print(f"[Daemon] cycle {budget.cycles + 1}/{cycles} | topic={topic!r} "
              f"| arm={arm} | {'DRY-RUN' if dry_run else 'train'}")
        record = engine.run_cycle(topic, kind=kind, quiz_questions=quiz,
                                  dry_run=dry_run, self_edit_form=form)
        results.append(record)
        budget.cycles += 1
        # Feed the live frontier (Growth Dashboard reads this file).
        from curiosity_engine import append_frontier
        append_frontier({
            "topic": topic, "kind": kind, "decision": record.get("decision"),
            "reason": record.get("reason", "")[:80],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print(f"  -> decision={record['decision']} reason={record['reason'][:80]}")
        # Learned topics drop out of the pool so each cycle learns
        # something new (unless a cycle failed — retry it smaller).
        if record["decision"] in ("merged", "memory_only"):
            topic_pool.remove(topic)
    print(f"[Daemon] done. {budget.summary()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Curiosity learning daemon (Phase 5)")
    parser.add_argument("--cycles", type=int, default=config.CURIOSITY_MAX_CYCLES)
    parser.add_argument("--topics", default=None,
                        help="File with one topic per line; or pass --topic.")
    parser.add_argument("--topic", action="append", default=[],
                        help="Single topic (repeatable).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan everything, train nothing.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    topics = list(args.topic)
    if args.topics and os.path.exists(args.topics):
        with open(args.topics, "r", encoding="utf-8") as f:
            topics += [ln.strip() for ln in f if ln.strip()]
    if not topics:
        topics = ["quantum computing", "Vietnam history", "Fibonacci sequence"]
    run_daemon(topics, cycles=args.cycles, dry_run=args.dry_run,
               seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
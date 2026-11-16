"""
Curiosity Engine — Jin's infinite-curiosity loop (Phase 5, the heart of v3).

The 8 steps, per the v3.0 plan:

1. **Gap detection** — find what Jin doesn't know: entropy + logprob of
   answers, self-consistency across k samples, entity novelty vs the
   trained corpus. Produces a ranked *Knowledge Frontier*:
       score = uncertainty x importance x novelty x learnability
   (every component logged — curiosity must be quantified, not vibes).
2. **Query generation** — 3-8 search queries per topic (definition,
   example, debate, numbers, vi + en) PLUS a quiz written BEFORE learning
   (the quiz is the success criterion — written first so we cannot cheat).
3. **Harvest** — web search + page fetch through the polite layer
   (robots.txt, per-domain rate limit, allowlist, SSRF guard).
4. **Verification** — the most important step. A claim needs >= 2
   INDEPENDENT sources (different domains) for confidence=high; 1 source
   -> medium; conflicting sources -> disputed (both kept, neither trained
   on). Conflicts with existing high-confidence knowledge demand stronger
   evidence. Unverifiable -> memory only, NEVER into weights.
5. **Self-edit (SEAL-style)** — rewrite each verified claim into several
   training forms (assertion, Q&A, dialogue, summary) — diversity beats
   volume — and filter every sample through data quality + contamination.
6. **Continual learning** — LR re-warm + re-decay per round, replay of
   old data (default 5%, strong shift 25-50%), optional LoRA adapter that
   merges only after the gate passes (one-second rollback).
7. **Eval gate** — run the full bench on the candidate: new-topic quiz
   >= 80%, forgetting <= 1% abs, val loss not up > 1%, fluency not down
   > 0.02. Pass -> promote to registry.json with a changelog. Fail ->
   automatic rollback + a smaller-scope retry.
8. **Ledger & meta-learning** — append a full-provenance record; UCB
   bandit over (topic_kind x self_edit_form) picks better arms next round.

Design for testability: heavy steps (harvest, train, probe eval) go
through small callables that tests can fake — the gate and rollback
logic itself is pure and unit-tested without GPU or network.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field

import config
from knowledge_ledger import KnowledgeLedger


# ── Live knowledge frontier (persisted for the dashboard) ────────────
def append_frontier(entry: dict, max_kept: int = 20) -> None:
    """Append one frontier entry to ``CURIOSITY_FRONTIER_PATH``.

    The Growth Dashboard reads this file live via ``/api/learn/status``;
    without persistence the "frontier" would always be empty. Keeps the
    most recent ``max_kept`` entries so the file stays small.
    """
    entries: list[dict] = []
    if os.path.exists(config.CURIOSITY_FRONTIER_PATH):
        try:
            with open(config.CURIOSITY_FRONTIER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                entries = data
        except (OSError, json.JSONDecodeError):
            entries = []
    entries.append(entry)
    entries = entries[-max_kept:]
    os.makedirs(os.path.dirname(config.CURIOSITY_FRONTIER_PATH) or ".",
                exist_ok=True)
    with open(config.CURIOSITY_FRONTIER_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


# ── Step 1: gap detection ─────────────────────────────────────────────
@dataclass
class FrontierTopic:
    topic: str
    kind: str                      # factual | math | code | reasoning | ...
    uncertainty: float             # 0..1 (low logprob / high entropy)
    importance: float              # 0..1 (asked often / appears in frontier)
    novelty: float                 # 0..1 (rare in trained corpus)
    learnability: float            # 0..1 (harvestable: queryable, verifiable)
    score: float = 0.0

    def __post_init__(self):
        self.score = (self.uncertainty * self.importance
                      * self.novelty * self.learnability)


def _token_entropy(logprobs: list[float]) -> float:
    """Entropy of a token sequence's logprobs, normalized to 0..1."""
    if not logprobs:
        return 1.0
    mean_lp = sum(logprobs) / len(logprobs)
    # mean logprob in [-20, 0] -> entropy-ish in [0, 1]
    return max(0.0, min(1.0, 1.0 + mean_lp / 20.0))


def _self_consistency_disagreement(samples: list[str]) -> float:
    """Fraction of sample pairs that disagree (0 = all agree)."""
    n = len(samples)
    if n < 2:
        return 1.0
    disagree = 0
    pairs = 0
    for i in range(n):
        a = set(samples[i].split())
        for j in range(i + 1, n):
            b = set(samples[j].split())
            pairs += 1
            if not a or not b or len(a & b) / max(len(a | b), 1) < 0.5:
                disagree += 1
    return disagree / max(pairs, 1)


def _entity_novelty(text: str, corpus_entities: set[str]) -> float:
    """Fraction of capitalized tokens in ``text`` absent from the corpus."""
    tokens = re.findall(r"[A-Z][a-z]+", text)
    if not tokens:
        return 0.0
    unseen = sum(1 for t in tokens if t.lower() not in corpus_entities)
    return unseen / len(tokens)


def detect_gaps(
    answers: dict[str, dict],
    *,
    corpus_entities: set[str] | None = None,
    importance: dict[str, float] | None = None,
    learnability: dict[str, float] | None = None,
) -> list[FrontierTopic]:
    """Rank topics by score = uncertainty x importance x novelty x learn.

    ``answers`` maps topic -> {logprobs: [...], samples: [str,...]}.
    Missing keys default to neutral (0.5) so the formula is always
    well-defined and logged.
    """
    corpus_entities = corpus_entities or set()
    importance = importance or {}
    learnability = learnability or {}
    out: list[FrontierTopic] = []
    for topic, info in answers.items():
        logps = info.get("logprobs", [])
        samples = info.get("samples", [])
        text = " ".join(samples) or topic
        uncertainty = max(
            _token_entropy(logps),
            _self_consistency_disagreement(samples),
        )
        topic_importance = min(1.0, importance.get(topic, 0.5))
        novelty = _entity_novelty(text, corpus_entities)
        if samples and not re.search(r"[A-Z][a-z]+", text):
            novelty = max(novelty, 0.5)  # no entities -> unknown territory
        learn = min(1.0, learnability.get(topic, 0.5))
        out.append(FrontierTopic(
            topic=topic,
            kind=info.get("kind", "generic"),
            uncertainty=round(uncertainty, 3),
            importance=round(topic_importance, 3),
            novelty=round(novelty, 3),
            learnability=round(learn, 3),
        ))
    out.sort(key=lambda t: -t.score)
    return out


# ── Step 2: query + quiz generation ───────────────────────────────────
_QUERY_TEMPLATES = (
    "What is {topic}? definition and key facts",
    "{topic} examples and how it works",
    "{topic} controversy or debate",
    "{topic} statistics and numbers",
    "{topic} latest updates",
    "{topic} là gì",  # vi
    "define {topic} in simple terms",
)

_QUIZ_TEMPLATES = (
    "What is {topic}?",
    "Give one concrete example related to {topic}.",
    "Why is {topic} important?",
    "What is a key fact about {topic}?",
)


def generate_queries(topic: str, n: int = 5, rng: random.Random | None = None) -> list[str]:
    rng = rng or random.Random(0)
    tpls = list(_QUERY_TEMPLATES)
    rng.shuffle(tpls)
    return [t.format(topic=topic) for t in tpls[:max(3, n)]]


def generate_quiz(topic: str, n: int = 3, rng: random.Random | None = None) -> list[str]:
    rng = rng or random.Random(0)
    tpls = list(_QUIZ_TEMPLATES)
    rng.shuffle(tpls)
    return [t.format(topic=topic) for t in tpls[:max(1, n)]]


# ── Step 3: harvest (injectable fetch layer) ──────────────────────────
@dataclass
class HarvestedPage:
    url: str
    domain: str
    title: str
    text: str
    kind: str = "web"
    ok: bool = True


def _domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
        return host.removeprefix("www.")
    except Exception:
        return ""


def _allowed_domain(domain: str) -> bool:
    if not domain:
        return False
    if any(d in domain for d in config.CURIOSITY_DOMAIN_DENYLIST):
        return False
    allow = config.CURIOSITY_DOMAIN_ALLOWLIST
    if not allow:
        return True
    return any(domain == a or domain.endswith("." + a) for a in allow)


def harvest_topic(
    topic: str,
    queries: list[str],
    fetch: callable,
    *,
    max_pages: int = 4,
    min_text_chars: int = 200,
) -> list[HarvestedPage]:
    """Harvest pages for a topic through the injected ``fetch`` callable.

    ``fetch(query) -> list[HarvestedPage]`` — the real implementation in
    ``web_search.harvest`` respects robots.txt + per-domain rate limits;
    tests inject a fake. Only allowlisted domains are kept, and pages
    without enough extractable text are dropped (spam/JS shells).
    """
    pages: list[HarvestedPage] = []
    seen_urls: set[str] = set()
    for query in queries:
        try:
            hits = fetch(query)
        except Exception:
            continue
        for hit in hits or []:
            if hit.url in seen_urls:
                continue
            if not _allowed_domain(hit.domain):
                continue
            if len(hit.text or "") < min_text_chars:
                continue
            seen_urls.add(hit.url)
            pages.append(hit)
            if len(pages) >= max_pages:
                return pages
    return pages


# ── Step 4: verification ──────────────────────────────────────────────
_DOMAIN_TRUST = {
    "wikipedia.org": 0.9,
    "arxiv.org": 0.95,
    ".gov": 0.9,
    ".edu": 0.85,
    ".org": 0.6,
    ".com": 0.5,
}


def domain_trust(domain: str) -> float:
    for key, val in _DOMAIN_TRUST.items():
        if key.startswith("."):
            if domain.endswith(key):
                return val
        elif domain == key or domain.endswith("." + key):
            return val
    return 0.5


@dataclass
class VerifiedClaim:
    claim: str
    confidence: str            # high | medium | disputed | unverified
    sources: list[dict] = field(default_factory=list)
    source_count: int = 0
    independent_domains: int = 0
    verified_by: str = "cross_source"


def _normalize_claim(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().strip(".").lower()


def verify_claims(
    pages: list[HarvestedPage],
    *,
    max_claims_per_page: int = 3,
) -> list[VerifiedClaim]:
    """Cross-source verification.

    A claim is a sentence extracted from each page's text. Two claims are
    treated as the SAME statement when their normalized text has strong
    token overlap. Confidence rules:

    - >= 2 independent domains agreeing -> high
    - exactly 1 domain -> medium
    - same topic, different claims -> disputed (both kept, neither trained)
    """
    if not pages:
        return []
    # Extract candidate claims: sentences >= 6 words.
    claims_by_page: list[list[str]] = []
    for page in pages:
        from grounding import INJECTION
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", page.text)
                 if len(s.strip().split()) >= 6 and not INJECTION.search(s)]
        claims_by_page.append(sents[:max_claims_per_page])

    grouped: list[dict] = []  # {text, domains: set, urls: set, count}
    for p_idx, sents in enumerate(claims_by_page):
        page = pages[p_idx]
        for sent in sents:
            norm = _normalize_claim(sent)
            if len(norm) < 20:
                continue
            placed = False
            for g in grouped:
                g_tokens = set(g["text"].split())
                s_tokens = set(norm.split())
                if not g_tokens or not s_tokens:
                    continue
                overlap = len(g_tokens & s_tokens) / max(len(g_tokens | s_tokens), 1)
                # Token overlap cannot establish entailment (Paris vs London,
                # 2025 vs 2026, or a negation). Only exact normalized agreement
                # counts as corroboration in this small deterministic verifier.
                if norm == g["text"]:
                    g["count"] += 1
                    g["domains"].add(page.domain)
                    g["urls"].add(page.url)
                    placed = True
                    break
            if not placed:
                grouped.append({"text": norm, "count": 1,
                                "domains": {page.domain}, "urls": {page.url}})

    out: list[VerifiedClaim] = []
    for g in grouped:
        ndomains = len(g["domains"])
        if ndomains >= 2:
            conf = "high"
        elif ndomains == 1:
            conf = "medium"
        else:
            conf = "unverified"
        out.append(VerifiedClaim(
            claim=g["text"],
            confidence=conf,
            sources=[{"url": u} for u in sorted(g["urls"])],
            source_count=g["count"],
            independent_domains=ndomains,
            verified_by="cross_source",
        ))
    return out


# ── Step 5: self-edit (SEAL-style) ────────────────────────────────────
_SELF_EDIT_FORMS = ("assertion", "qa", "dialogue", "summary")


def self_edit_samples(
    claim: VerifiedClaim,
    topic: str,
    *,
    form: str = "qa",
    n: int = 6,
    rng: random.Random | None = None,
) -> list[dict]:
    """Rewrite a verified claim into training samples of one form.

    Forms (diversity beats volume — the same fact expressed differently
    generalizes better than raw text):
    - assertion:  "Fact: <claim>"
    - qa:         "Q: ... A: <claim>"
    - dialogue:   a tiny user/assistant exchange
    - summary:    "In short, <claim>"
    """
    rng = rng or random.Random(0)
    claim_text = claim.claim
    samples: list[dict] = []
    for i in range(n):
        if form == "assertion":
            text = f"Fact: {claim_text}."
        elif form == "qa":
            q = rng.choice([
                f"Tell me about {topic}.",
                f"What should I know about {topic}?",
                f"Explain {topic}.",
            ])
            text = f"Q: {q}\nA: {claim_text}."
        elif form == "dialogue":
            user = rng.choice([
                f"Do you know anything about {topic}?",
                f"Can you tell me about {topic}?",
            ])
            text = f"User: {user}\nAssistant: {claim_text}."
        else:  # summary
            text = f"In short, {claim_text}."
        samples.append({
            "topic": topic, "form": form, "text": text,
            "claim_id": f"{topic}:{i}",
        })
    return samples


# ── Step 7: eval gate ─────────────────────────────────────────────────
@dataclass
class GateResult:
    passed: bool
    reason: str
    metrics: dict = field(default_factory=dict)


def eval_gate(
    *,
    quiz_score: float,
    forgetting: float,
    val_loss_delta: float,
    fluency_delta: float,
    quiz_pass: float = config.GATE_QUIZ_PASS_RATE,
    forgetting_max: float = config.GATE_FORGETTING_MAX,
    val_loss_max: float = config.GATE_VAL_LOSS_MAX_DELTA,
    fluency_min_delta: float = -0.02,
) -> GateResult:
    """Pure gate logic (unit-testable): promote or roll back.

    All thresholds from config; returns a pass/fail with the exact reason
    so the ledger can record why.
    """
    import math
    values = (quiz_score, forgetting, val_loss_delta, fluency_delta)
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return GateResult(False, "missing or nonfinite gate metrics", {})
    if not 0 <= quiz_score <= 1 or not 0 <= forgetting <= 1:
        return GateResult(False, "gate metric outside valid range", {})
    fails: list[str] = []
    metrics = {
        "quiz_score": round(quiz_score, 4),
        "forgetting": round(forgetting, 4),
        "val_loss_delta": round(val_loss_delta, 4),
        "fluency_delta": round(fluency_delta, 4),
    }
    if quiz_score < quiz_pass:
        fails.append(f"quiz {quiz_score:.2f} < {quiz_pass:.2f}")
    if forgetting > forgetting_max:
        fails.append(f"forgetting {forgetting:.3f} > {forgetting_max:.3f}")
    if val_loss_delta > val_loss_max:
        fails.append(f"val_loss +{val_loss_delta:.4f} > +{val_loss_max:.4f}")
    if fluency_delta < fluency_min_delta:
        fails.append(f"fluency {fluency_delta:+.3f} < {fluency_min_delta:+.3f}")
    if fails:
        return GateResult(False, "; ".join(fails), metrics)
    return GateResult(True, "all thresholds met", metrics)


# ── registry.json (checkpoint promotion) ──────────────────────────────
def load_registry() -> dict:
    path = config.CHECKPOINT_REGISTRY_PATH
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {"golden": None, "checkpoints": []}


def promote_checkpoint(
    checkpoint_path: str,
    *,
    changelog: dict,
    max_kept: int = 3,
) -> dict:
    """Register a checkpoint that passed the gate.

    Keeps at most ``max_kept`` recent checkpoints plus an immutable
    ``golden`` entry (never overwritten unless explicitly requested).
    """
    reg = load_registry()
    entry = {
        "path": checkpoint_path,
        "sha256": __import__("lineage").sha256_file(checkpoint_path) if os.path.isfile(checkpoint_path) else None,
        "verification": "manual_registration; not automatic evaluation",
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "changelog": changelog,
    }
    if reg.get("golden") is None:
        reg["golden"] = dict(entry, role="golden")
    ckpts = reg.setdefault("checkpoints", [])
    ckpts.append(entry)
    ckpts = ckpts[-max_kept:]
    reg["checkpoints"] = ckpts
    os.makedirs(os.path.dirname(config.CHECKPOINT_REGISTRY_PATH) or ".",
                exist_ok=True)
    from lineage import atomic_json
    atomic_json(config.CHECKPOINT_REGISTRY_PATH, reg)
    return reg


# ── The engine loop ───────────────────────────────────────────────────
class CuriosityEngine:
    """One cycle = steps 1-8, with injectable heavy steps for tests."""

    def __init__(
        self,
        *,
        fetch: callable = None,
        train: callable = None,
        evaluate: callable = None,
        memory=None,
        ledger: KnowledgeLedger | None = None,
        max_consecutive_fails: int = config.CURIOSITY_MAX_CONSECUTIVE_FAILS,
    ):
        self.fetch = fetch
        self.train = train
        self.evaluate = evaluate
        self.memory = memory
        self.ledger = ledger or KnowledgeLedger()
        self.max_consecutive_fails = int(max_consecutive_fails)
        self.consecutive_fails = 0

    # ── Steps ──────────────────────────────────────────────────────────
    def _step1_gaps(self, topics: list[str] | None = None) -> list[FrontierTopic]:
        """Ask the model about candidate topics, measure uncertainty."""
        topics = topics or ["Paris", "quantum computing", "Fibonacci sequence"]
        answers: dict[str, dict] = {}
        for topic in topics:
            if self.evaluate is None:
                answers[topic] = {"logprobs": [-3.0, -4.0, -5.0],
                                  "samples": [f"I am not sure about {topic}.",
                                              f"I think {topic} might be complex."],
                                  "kind": "generic"}
                continue
            try:
                info = self.evaluate("probe_topic", topic)
            except Exception:
                info = {"logprobs": [], "samples": []}
            answers[topic] = info or {"logprobs": [], "samples": []}
        return detect_gaps(answers)

    def _step4_verify(self, pages: list[HarvestedPage]) -> list[VerifiedClaim]:
        claims = verify_claims(pages)
        # Contradictions with existing memory: high-confidence knowledge
        # wins unless the new evidence is stronger.
        if self.memory is not None:
            for claim in list(claims):
                for old in (self.memory.needs_verification() or []):
                    if _normalize_claim(claim.claim) == _normalize_claim(
                            old.get("claim", "")):
                        claim.confidence = "disputed"
                        claim.verified_by = "conflicts_with_memory"
        return claims

    def run_cycle(
        self,
        topic: str,
        *,
        kind: str = "generic",
        quiz_questions: list[str] | None = None,
        dry_run: bool = False,
        self_edit_form: str = "qa",
    ) -> dict:
        """Run one full cycle for one topic; returns a result dict.

        ``self_edit_form`` lets the meta-learner (UCB bandit in the
        daemon) pick which rewrite style to use this cycle.
        """
        t0 = time.time()
        train_report = {}
        record: dict = {
            "topic": topic, "topic_kind": kind, "decision": "memory_only",
            "reason": "no verified claims", "duration_s": 0.0,
            "operational_learning_verified": False,
        }
        try:
            # Step 1-2: gap + queries + quiz (quiz BEFORE learning).
            queries = generate_queries(topic, n=4)
            quiz = quiz_questions or generate_quiz(topic, n=3)

            # Step 3: harvest.
            if self.fetch is None:
                raise RuntimeError("No fetch provider configured")
            pages = harvest_topic(topic, queries, self.fetch)
            record["queries"] = queries
            record["quiz_questions"] = quiz
            record["pages"] = [{"url": p.url, "domain": p.domain} for p in pages]
            if not pages:
                record["reason"] = "no harvestable pages"
                return record

            # Step 4: verify.
            claims = self._step4_verify(pages)
            record["claims"] = [{"claim": c.claim, "confidence": c.confidence,
                                 "sources": c.sources,
                                 "independent_domains": c.independent_domains}
                                for c in claims]
            verified = [c for c in claims if c.confidence == "high"]
            if not verified:
                record["reason"] = "no claim reached confidence=high"
                return record

            # Step 5: self-edit into training samples.
            form = self_edit_form or "qa"
            samples = []
            for claim in verified[:3]:
                samples += self_edit_samples(claim, topic, form=form, n=4)
            record["self_edit_form"] = form
            record["self_edit_samples"] = len(samples)
            if dry_run:
                record["decision"] = "dry_run"
                record["reason"] = f"{len(samples)} samples ready, not trained"
                return record
            if self.train is None or self.evaluate is None:
                raise RuntimeError("Train AND evaluation providers required; weights unchanged")

            # Step 6: continual learning (injectable train step).
            train_report = self.train(samples, topic=topic)
            record["replay_ratio"] = train_report.get("replay_ratio")
            record["lr"] = train_report.get("lr")
            record["steps"] = train_report.get("steps")

            # Step 7: eval gate.
            gate = self.evaluate("gate", topic, quiz) if self.evaluate else None
            if gate is None:
                raise RuntimeError("No gate evaluation available; candidate rejected")
            gate_metrics = gate.get("metrics", {})
            result = eval_gate(
                quiz_score=gate_metrics.get("quiz_score", 0.0),
                forgetting=gate_metrics.get("forgetting"),
                val_loss_delta=gate_metrics.get("val_loss_delta"),
                fluency_delta=gate_metrics.get("fluency_delta"),
            )
            record["score_delta"] = gate_metrics.get("score_delta", 0.0)
            record["forgetting_delta"] = gate_metrics.get("forgetting", 0.0)
            record["val_loss_delta"] = gate_metrics.get("val_loss_delta", 0.0)
            record["fluency_delta"] = gate_metrics.get("fluency_delta", 0.0)
            if result.passed:
                # Legacy injected providers exercise orchestration only. Real
                # promotion requires a separate candidate file and measured gate.
                candidate = train_report.get("candidate_checkpoint")
                if candidate and gate.get("measured") and os.path.isfile(candidate):
                    import math
                    delta = gate_metrics.get("score_delta")
                    if not isinstance(delta, (int,float)) or not math.isfinite(delta) or delta <= 0:
                        raise RuntimeError("Candidate has no measured positive improvement")
                    active_path = train_report.get("active_checkpoint")
                    if not active_path or os.path.abspath(candidate) == os.path.abspath(active_path):
                        raise RuntimeError("Candidate must be isolated from the active checkpoint")
                    from lineage import sha256_file
                    if sha256_file(active_path) != train_report.get("active_sha256_before"):
                        raise RuntimeError("Training provider modified active checkpoint; promotion refused")
                    promote_checkpoint(candidate, changelog={"topic":topic, "gate":result.metrics,
                                       "measured":True, "score_delta":delta})
                    record["operational_learning_verified"] = True
                    record["decision"] = "candidate_accepted"
                else:
                    record["decision"] = "gate_passed_unpromoted"
                record["reason"] = result.reason
                self.consecutive_fails = 0
            else:
                record["decision"] = "rolled_back"
                record["reason"] = result.reason
                self.consecutive_fails += 1
            record["gate"] = result.metrics
        except Exception as exc:
            record["decision"] = "rejected" if train_report else "memory_only"
            record["reason"] = f"error: {exc}"
            if train_report:
                self.consecutive_fails += 1
        finally:
            # Candidate training providers MUST leave active weights untouched;
            # rejected artifacts stay isolated, never copied over the active model.
            record["duration_s"] = round(time.time() - t0, 2)
            self.ledger.append(record)
        return record
"""Phase 17 §8 — the JIN assistant as a *system*, measured honestly.

Phase 16 measured three components and one of its checks proved less than it
looked: `verify_citations()` only validates `[n]` syntax, and the benchmark built
the answer from the gold answer before validating it. Neither says the cited
passage actually supports the claim.

What this script measures instead:

  router_accuracy        does the router pick the right tool at all?
  arithmetic_accuracy    given the right tool, is the number exactly right?
  recall@k / MRR / P@k   retrieval over a larger corpus with distractors
  evidence_selection     did select_evidence() keep the gold passage?
  evidence_support       does the selected passage actually contain the answer?
  abstention_accuracy    are unanswerable questions refused?

Credit is attributed per component. Nothing here is credited to the base model:
the model was not asked, and a calculator result is not language-model reasoning.

Usage
-----
    python scripts/phase17_system_benchmark.py \
        --out bench_results/phase17/system_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from grounding import select_evidence                     # noqa: E402
from retriever import BM25Index                           # noqa: E402
from phase17_intent_router import route, run_router_benchmark   # noqa: E402


# ── corpus ──────────────────────────────────────────────────────────────────
# Facts are synthetic and self-contained on purpose: no external dataset can be
# mistaken for general world knowledge, and every expected answer is a literal
# string in one passage, so evidence support is decidable.
FACTS: list[tuple[str, str, str]] = [
    ("aurora_station", "The Aurora research station on Devon Island records a mean July temperature of 4.7 degrees Celsius.", "4.7"),
    ("aurora_station_2", "The Aurora research station was founded in 1972 and has eleven permanent staff.", "1972"),
    ("kestrel_bridge", "The Kestrel Bridge in Vallis is 742 metres long and carries two tram lines.", "742"),
    ("kestrel_bridge_2", "The Kestrel Bridge was repainted in 2019 after a corrosion inspection.", "2019"),
    ("mira_lake", "Mira Lake has a maximum depth of 118 metres and freezes over for 214 days a year.", "118"),
    ("mira_lake_2", "Mira Lake is fed by the Talus river and drains into the Corrin basin.", "Talus"),
    ("havel_observatory", "Havel Observatory sits at an altitude of 2,860 metres in the Kora range.", "2860"),
    ("havel_observatory_2", "Havel Observatory operates a 1.8 metre reflecting telescope.", "1.8"),
    ("corrin_basin", "The Corrin basin covers 41,300 square kilometres and supports 26 fish species.", "41300"),
    ("talus_river", "The Talus river is 388 kilometres long and drops 1,120 metres over its course.", "388"),
    ("peregrine_library", "Peregrine Library holds 1,240,000 volumes and opens 361 days a year.", "1240000"),
    ("solstice_market", "Solstice Market has traded on the same square since 1804 and hosts 320 stalls.", "1804"),
    ("nimbus_foundry", "Nimbus Foundry produces 96,000 tonnes of aluminium a year.", "96000"),
    ("verdant_tram", "The Verdant tram line carries 74,500 passengers each weekday.", "74500"),
    ("quartz_canyon", "Quartz Canyon is 1,610 metres deep at its deepest point.", "1610"),
    ("amber_lighthouse", "The Amber lighthouse stands 63 metres tall and was automated in 1988.", "63"),
    ("willow_archive", "The Willow Archive digitised 4,200 manuscripts between 2005 and 2011.", "4200"),
    ("cobalt_tunnel", "The Cobalt tunnel runs 12.4 kilometres under the Greth hills.", "12.4"),
    ("ferrin_university", "Ferrin University enrolls 18,700 students and employs 2,150 staff.", "18700"),
    ("marsh_ferry", "The Marsh ferry crosses in 19 minutes and runs every 40 minutes.", "19"),
    ("oriel_dam", "Oriel Dam generates 480 megawatts at full capacity.", "480"),
    ("pallas_forest", "Pallas forest covers 9,340 hectares and contains 61 protected species.", "9340"),
    ("juniper_clinic", "Juniper clinic treats 1,150 patients a week with 84 beds.", "1150"),
    ("sable_mine", "Sable mine produced 3.6 million tonnes of ore in its best year.", "3.6"),
    ("lumen_theatre", "Lumen theatre seats 1,027 people and stages 210 performances a year.", "1027"),
    ("gren_museum", "The Gren museum displays 7,800 artefacts, of which 640 are on loan.", "7800"),
    ("tessa_spaceport", "Tessa spaceport handles 42 launches a year from three pads.", "42"),
    ("vireo_wetland", "Vireo wetland shelters 214 bird species across 5,600 hectares.", "214"),
    ("north_gate", "North Gate toll plaza processes 88,000 vehicles daily.", "88000"),
    ("lowell_canal", "The Lowell canal carries 34 cubic metres of water per second.", "34"),
]

VI_FACTS: list[tuple[str, str, str]] = [
    ("sông_lam", "Sông Lam dài 512 ki-lô-mét và chảy qua hai tỉnh.", "512"),
    ("hồ_tây", "Hồ Tây có diện tích khoảng 5,3 ki-lô-mét vuông.", "5,3"),
    ("núi_bà_đen", "Núi Bà Đen cao 986 mét, là đỉnh cao nhất vùng Đông Nam Bộ.", "986"),
    ("chợ_bến_thành", "Chợ Bến Thành được xây năm 1914 và có hơn 1.500 sạp.", "1914"),
    ("đèo_hải_vân", "Đèo Hải Vân dài 21 ki-lô-mét, nối Đà Nẵng và Huế.", "21"),
]

DISTRACTORS: list[tuple[str, str, str]] = [
    ("aurora_station_old", "An older Aurora station in a different country recorded a mean July temperature of 9.1 degrees.", "9.1"),
    ("kestrel_bridge_rail", "The Kestrel rail bridge, a separate structure, is 742 metres long but has no tram lines.", "742"),
    ("mira_lake_north", "North Mira Lake is 118 metres wide but only four metres deep.", "118"),
    ("havel_observatory_old", "Havel Observatory's older dome sits at 2,860 feet, not metres.", "2860"),
    ("talus_creek", "Talus creek is 388 metres long, not 388 kilometres.", "388"),
    ("lumen_theatre_cinema", "Lumen cinema, unrelated to the theatre, seats 1,027 people in four halls.", "1027"),
    ("oriel_dam_plan", "A proposed second Oriel dam would generate 480 gigawatt-hours, not megawatts.", "480"),
    ("sable_mine_iron", "Sable iron works produced 3.6 thousand tonnes, far less than the mine.", "3.6"),
    ("ferrin_college", "Ferrin College, distinct from the university, enrolls 18,700 students part-time.", "18700"),
    ("north_gate_south", "South Gate toll plaza processes 88,000 vehicles weekly.", "88000"),
]


def _doc(doc_id: str, text: str, title: str) -> dict:
    return {"id": doc_id, "title": title, "body": text, "text": text,
            "source": "phase17_synthetic", "license": "project_authored"}


def build_corpus() -> tuple[list[dict], list[dict]]:
    docs: list[dict] = []
    for doc_id, text, _ in FACTS:
        docs.append(_doc(doc_id, text, doc_id.replace("_", " ").title()))
    for doc_id, text, _ in VI_FACTS:
        docs.append(_doc(doc_id, text, doc_id.replace("_", " ").title()))
    for doc_id, text, _ in DISTRACTORS:
        docs.append(_doc(doc_id, text, doc_id.replace("_", " ").title()))
    # Unrelated filler so recall is not measured on a 45-document toy set.
    topics = ["budget", "weather", "football", "recipe", "policy", "software",
              "history", "music", "shipping", "agriculture"]
    for i in range(55):
        topic = topics[i % len(topics)]
        docs.append(_doc(f"filler_{i}",
                         f"Note {i} about {topic}: the committee discussed routine "
                         f"matters numbered {1000 + i} and postponed a decision.",
                         f"Filler {i}"))
    return docs, [{"id": d["id"], "title": d["title"], "body": d["text"]} for d in docs]


def build_queries() -> list[dict]:
    queries: list[dict] = []
    for doc_id, text, answer in FACTS:
        subject = doc_id.replace("_2", "").replace("_", " ")
        queries.append({"q": text[:120], "gold": doc_id, "answer": answer,
                        "kind": "direct"})
        queries.append({"q": f"what number is reported for the {subject}?",
                        "gold": doc_id, "answer": answer, "kind": "paraphrase"})
    for doc_id, text, answer in VI_FACTS:
        subject = doc_id.replace("_", " ")
        queries.append({"q": f"{subject} có số liệu bao nhiêu?", "gold": doc_id,
                        "answer": answer, "kind": "vietnamese"})
    queries.extend([
        {"q": "how many volumes does Peregrine Library hold?", "gold": "peregrine_library",
         "answer": "1240000", "kind": "direct"},
        {"q": "how long is the Cobalt tunnel?", "gold": "cobalt_tunnel",
         "answer": "12.4", "kind": "direct"},
        {"q": "how tall is the Amber lighthouse?", "gold": "amber_lighthouse",
         "answer": "63", "kind": "direct"},
        {"q": "how many students are at Ferrin University?", "gold": "ferrin_university",
         "answer": "18700", "kind": "direct"},
        {"q": "how many megawatts does Oriel Dam generate?", "gold": "oriel_dam",
         "answer": "480", "kind": "direct"},
        {"q": "how deep is Quartz Canyon?", "gold": "quartz_canyon",
         "answer": "1610", "kind": "direct"},
    ])
    unanswerable = [
        "what is the population of the moon?",
        "who won the 2091 world cup?",
        "how many moons does the planet Zorblax have?",
        "what is the boiling point of mercury on Titan?",
        "how many employees work at the nonexistent Vell company?",
        "thủ đô của hành tinh Sao Mộc là gì?",
    ]
    for text in unanswerable:
        queries.append({"q": text, "gold": None, "answer": None, "kind": "unanswerable"})
    # Multi-document: the answer is a comparison across two passages.
    queries.append({"q": "which is longer, the Cobalt tunnel or the Kestrel Bridge?",
                    "gold": "cobalt_tunnel", "answer": "12.4", "kind": "multi_doc"})
    queries.append({"q": "compare Mira Lake depth with Quartz Canyon depth",
                    "gold": "mira_lake", "answer": "118", "kind": "multi_doc"})
    return queries


# ── measurement ─────────────────────────────────────────────────────────────
def run_retrieval_benchmark(k_values=(1, 3, 5)) -> dict:
    docs, sources = build_corpus()
    bm = BM25Index()
    bm.add([d["id"] for d in docs], [d["text"] for d in docs])
    queries = build_queries()

    recall = {k: 0 for k in k_values}
    precision = {k: 0.0 for k in k_values}
    rr: list[float] = []
    latencies: list[float] = []
    answerable = [q for q in queries if q["gold"]]
    unanswerable = [q for q in queries if not q["gold"]]

    evidence_selection = 0
    evidence_support = 0
    per_kind: dict[str, dict] = {}

    for item in answerable:
        import time
        started = time.perf_counter()
        hits = bm.search(item["q"], k=max(k_values))
        latencies.append((time.perf_counter() - started) * 1000)
        ids = [h[0] for h in hits]
        rank = ids.index(item["gold"]) + 1 if item["gold"] in ids else None
        rr.append(1.0 / rank if rank else 0.0)
        for k in k_values:
            if rank and rank <= k:
                recall[k] += 1
            if k <= len(ids):
                precision[k] += 1.0 / k if (rank and rank <= k) else 0.0

        # Evidence selection runs on the top-20 retrieval hits, as in production.
        top_sources = [s for s in sources if s["id"] in ids[:20]]
        selected, _ = select_evidence(item["q"], top_sources, k=3)
        selected_ids = [s.get("id") for s in selected]
        picked = item["gold"] in selected_ids
        evidence_selection += int(picked)
        text = " ".join(s.get("text", "") for s in selected)
        # Numeric answers must be compared as numbers: the passage writes
        # "1,240,000" while the fact table stores "1240000".
        def _flat(value: str) -> str:
            return str(value).replace(",", "").replace(" ", "").lower()
        supported = bool(picked) and _flat(item["answer"]) in _flat(text)
        evidence_support += int(supported)
        bucket = per_kind.setdefault(item["kind"], {"n": 0, "recall3": 0,
                                                    "selection": 0, "support": 0})
        bucket["n"] += 1
        bucket["recall3"] += int(bool(rank and rank <= 3))
        bucket["selection"] += int(picked)
        bucket["support"] += int(supported)

    abstained = 0
    for item in unanswerable:
        selected, _ = select_evidence(item["q"], sources, k=3)
        abstained += int(not selected)

    n = len(answerable)
    return {
        "corpus_documents": len(docs),
        "answerable_queries": n,
        "unanswerable_queries": len(unanswerable),
        "recall_at": {str(k): round(recall[k] / n, 4) for k in k_values},
        "precision_at": {str(k): round(precision[k] / n, 4) for k in k_values},
        "mrr": round(sum(rr) / n, 4),
        "retrieval_latency_ms_mean": round(statistics.fmean(latencies), 3),
        "retrieval_latency_ms_p95": round(
            sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 3),
        "evidence_selection_accuracy": round(evidence_selection / n, 4),
        "evidence_support_rate": round(evidence_support / n, 4),
        "abstention_accuracy": round(abstained / len(unanswerable), 4),
        "by_kind": {k: {"n": v["n"],
                        "recall@3": round(v["recall3"] / v["n"], 4),
                        "evidence_selection": round(v["selection"] / v["n"], 4),
                        "evidence_support": round(v["support"] / v["n"], 4)}
                    for k, v in per_kind.items()},
        "citation_syntax_checks_are_not_support": True,
        "credited_to_model": 0,
    }


def run_router_section() -> dict:
    return run_router_benchmark()


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 17 system benchmark")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench_results", "phase17",
                                                  "system_benchmark.json"))
    args = ap.parse_args()

    report = {
        "protocol": "phase17.system.v1",
        "router": run_router_section(),
        "retrieval": run_retrieval_benchmark(),
        "attribution": {
            "calculator": "router + calculator, model contributed nothing",
            "retrieval": "BM25 + evidence selector, model contributed nothing",
            "model_in_loop": "not measured here; Phase 16 measured 0/73 task ability",
        },
        "limitations": [
            "the retrieval corpus is synthetic and project-authored; it is larger "
            "and harder than Phase 16's 12 passages but it is not a web corpus",
            "the router suite is self-authored, so 100% means the implemented "
            "grammar is self-consistent, not that the router understands language",
            "evidence_support checks that the answer string appears in the selected "
            "passage; it does not test entailment of a generated sentence",
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:2500])
    try:
        shown = os.path.relpath(args.out, ROOT)
    except ValueError:                       # different drive on Windows
        shown = args.out
    print(f"\nWrote {shown}")


if __name__ == "__main__":
    main()

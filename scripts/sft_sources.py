"""Phase 16 §5 — reviewed SFT source table.

Every entry records what the upstream dataset card actually says, with the URL
that was read. A source that is not in this table is ineligible: no inference from
a self-declared license string, and no "it is on disk so it must be fine".

`aggregated` marks a dataset whose records are themselves mixtures of other
datasets (smoltalk). Its own license cannot grant rights over the constituents, so
it stays ineligible until each constituent is reviewed separately.
"""

SFT_SOURCES = {
    "dolly-15k": {
        "name": "Databricks Dolly 15k",
        "card": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "license": "cc-by-sa-3.0",
        "commercial": True,
        "redistribution": True,
        "attribution": True,
        "share_alike": True,
        "eligible": True,
        "aggregated": False,
        "review": ("Human-written instruction/response pairs. CC-BY-SA-3.0: "
                   "attribution and share-alike obligations follow the data."),
    },
    "gsm8k-train": {
        "name": "GSM8K (train split)",
        "card": "https://huggingface.co/datasets/openai/gsm8k",
        "license": "mit",
        "commercial": True,
        "redistribution": True,
        "attribution": True,
        "share_alike": False,
        "eligible": True,
        "aggregated": False,
        "review": "MIT per dataset card. Grade-school math word problems with worked solutions.",
    },
    "metamathqa": {
        "name": "MetaMathQA",
        "card": "https://huggingface.co/datasets/meta-math/MetaMathQA",
        "license": "mit",
        "commercial": True,
        "redistribution": True,
        "attribution": True,
        "share_alike": False,
        "eligible": True,
        "aggregated": False,
        "review": ("MIT per dataset card. Augmented from GSM8K and MATH **train** "
                   "splits only; the card states no test-set data is included."),
    },
    "oasst2": {
        "name": "OpenAssistant Conversations (OASST2)",
        "card": "https://huggingface.co/datasets/OpenAssistant/oasst2",
        "license": "apache-2.0",
        "commercial": True,
        "redistribution": True,
        "attribution": True,
        "share_alike": False,
        "eligible": True,
        "aggregated": False,
        "review": ("Apache-2.0 per dataset card. Crowd-written conversations: user "
                   "content, so the PII gate and quality filter are mandatory, not optional."),
    },
    "opencodeinstruct": {
        "name": "NVIDIA OpenCodeInstruct",
        "card": "https://huggingface.co/datasets/nvidia/OpenCodeInstruct",
        "license": "cc-by-4.0",
        "commercial": True,
        "redistribution": True,
        "attribution": True,
        "share_alike": False,
        "eligible": True,
        "aggregated": False,
        "review": "CC-BY-4.0 per dataset card. Synthetic code instructions with unit-test fields.",
    },
    "smol-smoltalk": {
        "name": "HuggingFaceTB smol-smoltalk",
        "card": "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk",
        "license": "apache-2.0",
        "commercial": False,
        "redistribution": False,
        "attribution": True,
        "share_alike": False,
        "eligible": False,
        "aggregated": True,
        "review": ("Aggregated from many upstream datasets; the mixture licence does "
                   "not grant rights over constituents that were not reviewed here."),
    },
    "chat_train_txt": {
        "name": "data/chat_train.txt",
        "card": None,
        "license": "unknown",
        "commercial": False,
        "redistribution": False,
        "attribution": False,
        "share_alike": False,
        "eligible": False,
        "aggregated": False,
        "review": ("No per-record provenance: source, licence and author are unknown. "
                   "Excluded from SFT until that is resolved."),
    },
}


def decision(source_id: str) -> dict:
    entry = SFT_SOURCES.get(source_id)
    if entry is None:
        return {"source_id": source_id, "eligible": False, "license": "unknown",
                "reason": "no reviewed SFT entry"}
    return {"source_id": source_id, "eligible": entry["eligible"],
            "license": entry["license"],
            "reason": entry["review"] if not entry["eligible"] else
            f"reviewed: {entry['license']}"}

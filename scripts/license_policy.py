"""Explicit license and PII policy for the corpus pipeline (Phase 15C).

Two rules this module exists to enforce:

1. **A license string is not permission.** Permission is granted only by a
   reviewed entry in ``REVIEWED_SOURCES`` / ``LICENSE_POLICIES``. Anything not
   matched there is ``unknown`` and ineligible for the training mixture, no
   matter how plausible its name looks.
2. **Flagging is not removing.** PII detection produces structured findings;
   the policy decides whether the document is redacted or excluded, and every
   decision is logged with counts only (never the sensitive text itself).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

POLICY_VERSION = 2   # v2: tightened phone/card detectors (fewer false positives)

# ── reviewed license policies ────────────────────────────────────────────
# Every field is a decision made for THIS project, not an inference from the
# license name. `eligible` means "may enter the default training mixture".
LICENSE_POLICIES: dict[str, dict] = {
    "cc-by-sa-3.0": {
        "commercial": True, "redistribution": True, "attribution": True,
        "share_alike": True, "nc_restriction": False, "eligible": True,
        "obligations": ["attribute upstream", "share-alike on redistributed text"],
        "review": "Wikimedia terms reviewed: text is CC-BY-SA-3.0 (dual GFDL)",
    },
    "cc-by-sa-3.0+gfdl": {
        "commercial": True, "redistribution": True, "attribution": True,
        "share_alike": True, "nc_restriction": False, "eligible": True,
        "obligations": ["attribute upstream", "share-alike", "GFDL notice where applicable"],
        "review": "Wikipedia dump: dual-licensed CC-BY-SA-3.0 and GFDL",
    },
    "odc-by-1.0": {
        "commercial": True, "redistribution": True, "attribution": True,
        "share_alike": False, "nc_restriction": False, "eligible": True,
        "obligations": ["attribute upstream dataset"],
        "review": "Open Data Commons Attribution License 1.0",
    },
    "project-authored": {
        "commercial": True, "redistribution": True, "attribution": False,
        "share_alike": False, "nc_restriction": False, "eligible": True,
        "obligations": [],
        "review": "authored inside this repository",
    },
    "project-vetted-mixture": {
        "commercial": True, "redistribution": True, "attribution": True,
        "share_alike": False, "nc_restriction": False, "eligible": True,
        "obligations": ["per-source attribution preserved in pretrain_sources.jsonl"],
        "review": "prepared_v4 policy: NC and unknown sources already excluded upstream",
    },
    "unknown": {
        "commercial": False, "redistribution": False, "attribution": False,
        "share_alike": False, "nc_restriction": False, "eligible": False,
        "obligations": ["record license before use"],
        "review": "not reviewed",
    },
}

# Source-level mapping: the only place a source becomes eligible. A source that
# is absent here is excluded, including any local folder that merely exists on
# disk (e.g. data/raw books, whose per-file provenance was never recorded).
REVIEWED_SOURCES: dict[str, dict] = {
    "wikipedia-en-20231101": {
        "canonical_license": "cc-by-sa-3.0+gfdl",
        "upstream": "wikimedia/wikipedia",
        "dataset_card": "https://huggingface.co/datasets/wikimedia/wikipedia",
    },
    "cosmopedia-v2-sample": {
        "canonical_license": "odc-by-1.0",
        "upstream": "HuggingFaceTB/smollm-corpus",
        "dataset_card": "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus",
        "notes": "synthetic (model-generated) educational text",
    },
    "prepared_v4": {
        "canonical_license": "project-vetted-mixture",
        "upstream": "data/prepared_v4/manifest.json",
        "dataset_card": "data/prepared_v4/pretrain_sources.jsonl",
    },
    "project_seed": {
        "canonical_license": "project-authored",
        "upstream": "this repository",
        "dataset_card": "data/seed/",
    },
}


@dataclass
class LicenseDecision:
    source_id: str
    canonical: str
    original: str | None
    commercial: bool
    redistribution: bool
    attribution: bool
    share_alike: bool
    nc_restriction: bool
    eligible: bool
    obligations: list[str] = field(default_factory=list)
    review: str = ""
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "canonical_license": self.canonical,
            "original_license_string": self.original,
            "commercial_use_permitted": self.commercial,
            "redistribution_permitted": self.redistribution,
            "attribution_required": self.attribution,
            "share_alike": self.share_alike,
            "non_commercial_restriction": self.nc_restriction,
            "eligible_for_training_mixture": self.eligible,
            "obligations": self.obligations,
            "review_note": self.review,
            "decision_reason": self.reason,
            "policy_version": POLICY_VERSION,
        }


def classify_source(source_id: str, original_license: str | None) -> LicenseDecision:
    """Map a source to a reviewed policy. No inference from the string."""
    # Documents carry the id with underscores (wikipedia_en_20231101) while the
    # reviewed table uses hyphens; both spellings must resolve to the same entry,
    # and an id that resolves to nothing stays unknown.
    reviewed = REVIEWED_SOURCES.get(source_id)
    if reviewed is None and source_id:
        for candidate in (source_id.replace("_", "-"), source_id.replace("-", "_")):
            if candidate in REVIEWED_SOURCES:
                reviewed = REVIEWED_SOURCES[candidate]
                source_id = candidate
                break
    if reviewed is None:
        policy = LICENSE_POLICIES["unknown"]
        return LicenseDecision(
            source_id=source_id, canonical="unknown", original=original_license,
            commercial=False, redistribution=False, attribution=False,
            share_alike=False, nc_restriction=False, eligible=False,
            obligations=policy["obligations"], review=policy["review"],
            reason=f"source {source_id!r} has no reviewed license entry",
        )
    canonical = reviewed["canonical_license"]
    policy = LICENSE_POLICIES[canonical]
    return LicenseDecision(
        source_id=source_id, canonical=canonical, original=original_license,
        commercial=policy["commercial"], redistribution=policy["redistribution"],
        attribution=policy["attribution"], share_alike=policy["share_alike"],
        nc_restriction=policy["nc_restriction"], eligible=policy["eligible"],
        obligations=policy["obligations"], review=policy["review"],
        reason=f"reviewed mapping {source_id} -> {canonical}",
    )


def decision_from_document(doc: dict) -> "LicenseDecision":
    """Prefer the reviewed decision already recorded on the document.

    Ingestion is the only place a source is mapped to a policy; verification
    re-uses that recorded decision when present and falls back to the source
    mapping otherwise (so a hand-written document without the fields is still
    classified, not trusted).
    """
    source_id = str(doc.get("source_id", "?"))
    # The reviewed source mapping is authoritative. A record's own
    # `canonical_license` field is data, not a permission: trusting it would let
    # any record (or any compromised upstream file) declare itself
    # "project-authored" or "cc-by-sa-3.0+gfdl" and become eligible.
    # A mounted re-upload keeps its folder name in source_id while the reviewed
    # entry it was ingested under is recorded as policy_source. Resolving the
    # policy through that id still uses the reviewed table: a record cannot
    # invent a licence, it can only point at one that was already reviewed.
    policy_key = str(doc.get("policy_source") or source_id)
    reviewed = classify_source(policy_key, doc.get("license"))
    recorded = doc.get("canonical_license")

    if recorded and recorded != reviewed.canonical:
        return LicenseDecision(
            source_id=source_id, canonical=reviewed.canonical,
            original=doc.get("license"),
            commercial=False, redistribution=False, attribution=False,
            share_alike=False, nc_restriction=False, eligible=False,
            obligations=[], review=reviewed.review,
            reason=(f"license_conflict: record declares {recorded!r} but source "
                    f"{source_id!r} is reviewed as {reviewed.canonical!r}"),
        )

    return LicenseDecision(
        source_id=source_id, canonical=reviewed.canonical,
        original=doc.get("license"),
        commercial=reviewed.commercial, redistribution=reviewed.redistribution,
        attribution=reviewed.attribution, share_alike=reviewed.share_alike,
        nc_restriction=reviewed.nc_restriction, eligible=reviewed.eligible,
        obligations=reviewed.obligations, review=reviewed.review,
        reason=reviewed.reason if not recorded else f"reviewed mapping confirms {recorded!r}",
    )


# ── PII detection ────────────────────────────────────────────────────────
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
# A phone number must look like a phone number: a country code or parentheses,
# and 9-15 digits. Without that, "1902-1955" (a year range) was being counted.
PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]\d{3,4}){1,3}"
    r"|\(\d{2,4}\)[\s.-]?\d{3,4}(?:[\s.-]\d{3,4})?)(?!\w)")
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
                     r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
CARD_RE = re.compile(r"(?<![\d])(?:\d[ -]?){14,18}\d(?![\d])")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def detect_pii(text: str) -> dict[str, int]:
    """Return {kind: count}. No text is returned, so findings are safe to log."""
    findings: dict[str, int] = {}
    emails = EMAIL_RE.findall(text)
    if emails:
        findings["email"] = len(emails)
    phones = 0
    for candidate in PHONE_RE.findall(text):
        digits = re.sub(r"\D", "", candidate)
        if 9 <= len(digits) <= 15:
            phones += 1
    if phones:
        findings["phone"] = phones
    ips = IPV4_RE.findall(text)
    if ips:
        findings["ipv4"] = len(ips)
    ssns = SSN_RE.findall(text)
    if ssns:
        findings["ssn"] = len(ssns)
    cards = 0
    for candidate in CARD_RE.findall(text):
        digits = re.sub(r"\D", "", candidate)
        # Payment cards are 15-16 digits and start with 3, 4, 5 or 6. Anything
        # else that happens to pass Luhn is a false positive.
        if len(digits) in (15, 16) and digits[0] in "3456" and _luhn_ok(digits):
            cards += 1
    if cards:
        findings["payment_card"] = cards
    return findings


REDACTION_PATTERNS = [
    ("email", EMAIL_RE, "[EMAIL]"),
    ("phone", PHONE_RE, "[PHONE]"),
    ("ipv4", IPV4_RE, "[IP]"),
    ("ssn", SSN_RE, "[SSN]"),
]


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Deterministic redaction. Payment-card-like numbers are not redacted in
    place (a wrong span is worse than excluding the document), so callers that
    choose `redact` still exclude documents with `payment_card` findings."""
    findings = detect_pii(text)
    out = text
    for _kind, pattern, replacement in REDACTION_PATTERNS:
        out = pattern.sub(replacement, out)
    return out, findings


PII_POLICIES = {
    "exclude": "documents with any finding are removed from all splits",
    "redact": "emails/phones/IPs/SSNs replaced with placeholders; payment-card "
              "findings still exclude the document",
    "keep": "findings recorded only (NOT recommended; must be disclosed)",
}

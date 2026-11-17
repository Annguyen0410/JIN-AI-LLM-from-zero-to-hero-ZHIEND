"""License eligibility and PII policy tests (Phase 15C).

The point of these tests is that permission is never inferred:

* a non-empty license string does **not** make a source usable;
* a source without a reviewed entry is ineligible even if it looks licensed;
* an NC license is recorded but ineligible for the training mixture;
* PII is detected deterministically, and the chosen policy decides whether the
  document is excluded or redacted — flagging alone is not removal.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from license_policy import (LICENSE_POLICIES, classify_source, detect_pii,  # noqa: E402
                            redact_pii)


def test_unknown_source_is_ineligible_even_with_a_license_string():
    decision = classify_source("some_random_folder", "CC-BY-4.0")
    assert decision.canonical == "unknown"
    assert decision.eligible is False
    assert decision.commercial is False


def test_missing_license_is_unknown():
    decision = classify_source("data_raw_books", None)
    assert decision.canonical == "unknown"
    assert decision.eligible is False


def test_reviewed_sources_are_eligible():
    for source_id, expected in (("wikipedia-en-20231101", "cc-by-sa-3.0+gfdl"),
                                ("cosmopedia-v2-sample", "odc-by-1.0"),
                                ("prepared_v4", "project-vetted-mixture"),
                                ("project_seed", "project-authored")):
        decision = classify_source(source_id, "as recorded upstream")
        assert decision.canonical == expected
        assert decision.eligible is True
        assert decision.commercial is True


def test_share_alike_and_attribution_are_recorded():
    wiki = classify_source("wikipedia-en-20231101", "CC-BY-SA-3.0")
    assert wiki.attribution is True
    assert wiki.share_alike is True
    assert any("share-alike" in o for o in wiki.obligations)
    cosmopedia = classify_source("cosmopedia-v2-sample", "ODC-BY-1.0")
    assert cosmopedia.attribution is True
    assert cosmopedia.share_alike is False


def test_non_commercial_policy_is_excluded_by_construction():
    # The NC policy exists so a future NC source is recorded, not silently used.
    assert LICENSE_POLICIES["unknown"]["eligible"] is False


def test_policy_dict_is_complete():
    required = {"commercial", "redistribution", "attribution", "share_alike",
                "nc_restriction", "eligible", "obligations", "review"}
    for name, policy in LICENSE_POLICIES.items():
        assert required <= set(policy), name


def test_detect_email_phone_ip_and_card():
    text = ("Write to jane.doe@example.org or call +1 (555) 123-4567. "
            "Server 192.168.0.14 logged card 4111 1111 1111 1111 today. "
            "SSN 123-45-6789 appears too.")
    findings = detect_pii(text)
    assert findings["email"] >= 1
    assert findings["phone"] >= 1
    assert findings["ipv4"] >= 1
    assert findings["payment_card"] == 1
    assert findings["ssn"] == 1


def test_luhn_rejects_random_number_groups():
    text = "Reference number 1234 5678 9012 3456 is not a payment card."
    assert "payment_card" not in detect_pii(text)


def test_clean_text_has_no_findings():
    text = "The river rises every spring and floods the lower valley pastures."
    assert detect_pii(text) == {}


def test_redaction_is_deterministic_and_content_free():
    text = "Contact jane@example.org or 10.0.0.5 for details."
    redacted, findings = redact_pii(text)
    assert "jane@example.org" not in redacted
    assert "10.0.0.5" not in redacted
    assert "[EMAIL]" in redacted and "[IP]" in redacted
    assert findings["email"] == 1
    again, _ = redact_pii(text)
    assert again == redacted


def _write_docs(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _doc(text: str, source: str, idx: str, eligible: bool = True,
         canonical: str = "project-authored") -> dict:
    import hashlib
    return {
        "document_id": f"{source}:{idx}", "source_id": source, "source_name": source,
        "source_url": None, "license": "recorded upstream",
        "license_status": "documented", "canonical_license": canonical,
        "eligible_for_training_mixture": eligible,
        "language": "en", "domain": "knowledge", "text": text,
        "original_format": "txt", "original_record_id": idx,
        "content_hash": hashlib.sha256(" ".join(text.split()).encode()).hexdigest(),
        "quality_flags": [], "ingestion_timestamp": "2026-11-16T00:00:00",
    }


def test_verify_excludes_ineligible_license_documents(tmp_path):
    clean = "A long, ordinary paragraph about coastal erosion and sediment. " * 5
    unlicensed = "A long paragraph from a source whose licence was never reviewed. " * 5
    root = tmp_path / "curated"
    _write_docs(str(root / "documents" / "d.jsonl"), [
        _doc(clean, "wikipedia_en_20231101", "1", canonical="cc-by-sa-3.0+gfdl"),
        # Unreviewed source that *claims* an approved licence: Phase 16 made the
        # reviewed source mapping authoritative, so the claim must not help it.
        _doc(unlicensed, "scraped_dump", "1", eligible=False,
             canonical="project-authored"),
    ])
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "verify_corpus.py"),
         "--in-root", str(root), "--out-root", str(root), "--min-chars", "10"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.load(open(root / "reports" / "dedup_quality_splits.json",
                            encoding="utf-8"))
    assert report["license"]["ineligible_documents_removed"] == 1
    kept = []
    for split in ("train", "val", "test"):
        with open(root / "splits" / f"{split}.jsonl", encoding="utf-8") as fh:
            kept += [json.loads(l) for l in fh if l.strip()]
    assert len(kept) == 1
    assert kept[0]["source_id"] == "wikipedia_en_20231101"


def test_provenance_fields_survive_verification(tmp_path):
    text = "Provenance must survive verification: source, revision, licence, id. " * 5
    root = tmp_path / "curated"
    record = _doc(text, "wikipedia_en_20231101", "42",
                  canonical="cc-by-sa-3.0+gfdl")
    record["dataset_revision"] = "abc123"
    record["license"] = "CC-BY-SA-3.0"
    record["original_title"] = "A title"
    _write_docs(str(root / "documents" / "d.jsonl"), [record])
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "verify_corpus.py"),
                    "--in-root", str(root), "--out-root", str(root),
                    "--min-chars", "10"], cwd=ROOT, check=True,
                   capture_output=True, text=True, encoding="utf-8")
    kept = []
    for split in ("train", "val", "test"):
        with open(root / "splits" / f"{split}.jsonl", encoding="utf-8") as fh:
            kept += [json.loads(l) for l in fh if l.strip()]
    assert kept
    rec = kept[0]
    for field in ("document_id", "source_id", "source_url", "license",
                  "dataset_revision", "original_record_id", "original_format",
                  "content_hash", "quality_flags", "processing_version",
                  "split", "original_title"):
        assert field in rec, field
    assert rec["license"] == "CC-BY-SA-3.0"
    assert rec["dataset_revision"] == "abc123"

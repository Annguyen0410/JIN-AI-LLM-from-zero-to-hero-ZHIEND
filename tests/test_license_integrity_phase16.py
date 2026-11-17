"""Phase 16 §5 — a record cannot grant itself a licence.

Phase 15's `decision_from_document()` trusted the record's own
`canonical_license`. These tests pin the replacement: the reviewed source mapping
decides, and a conflicting self-declaration is rejected rather than honoured.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from license_policy import decision_from_document   # noqa: E402


def test_unknown_source_claiming_project_authored_is_ineligible():
    dec = decision_from_document({"source_id": "scraped_web_dump",
                                  "canonical_license": "project-authored"})
    assert dec.eligible is False
    assert dec.canonical == "unknown"


def test_unknown_source_claiming_cc_by_sa_is_ineligible():
    dec = decision_from_document({"source_id": "mystery_corpus",
                                  "canonical_license": "cc-by-sa-3.0+gfdl"})
    assert dec.eligible is False


def test_reviewed_source_with_conflicting_licence_is_rejected():
    dec = decision_from_document({"source_id": "wikipedia_en_20231101",
                                  "canonical_license": "project-authored"})
    assert dec.eligible is False
    assert "license_conflict" in dec.reason


def test_reviewed_source_with_its_real_licence_is_eligible():
    dec = decision_from_document({"source_id": "wikipedia_en_20231101",
                                  "canonical_license": "cc-by-sa-3.0+gfdl"})
    assert dec.eligible is True
    assert dec.canonical == "cc-by-sa-3.0+gfdl"


def test_underscore_and_hyphen_ids_resolve_to_the_same_reviewed_entry():
    a = decision_from_document({"source_id": "cosmopedia_v2_sample"})
    b = decision_from_document({"source_id": "cosmopedia-v2-sample"})
    assert a.canonical == b.canonical == "odc-by-1.0"
    assert a.eligible and b.eligible

def test_project_sources_stay_eligible():
    for sid in ("prepared_v4", "project_seed"):
        dec = decision_from_document({"source_id": sid})
        assert dec.eligible is True, sid


def test_sft_sources_reject_aggregated_and_unprovenanced_data():
    from sft_sources import decision as sft_decision
    assert sft_decision("smol-smoltalk")["eligible"] is False
    assert sft_decision("chat_train_txt")["eligible"] is False
    for sid in ("dolly-15k", "gsm8k-train", "metamathqa", "oasst2",
                "opencodeinstruct"):
        assert sft_decision(sid)["eligible"] is True, sid
    assert sft_decision("unknown-dataset")["eligible"] is False

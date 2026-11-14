"""Acceptance tests for governace.py (v2)"""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import AllowInfNan, ValidationError


from lexitalk.governance import(
    SERVABLE,
    GovernanceError,
    Snapshot,
    check_snapshot,
    load_allowlist,
    load_snapshots,
    sha256_text,
)


ALLOWLIST = Path(__file__).resolve().parents[1] / "corpus" / "sources.json"

def make_snapshot(**over) -> Snapshot:
    """ A valid snapshot: pass keyword arguments to override fields. """

    text = over.pop("text", "SUMMARY: this rule changes things.")
    data = {
        "document_number" : "2025-01234",
        "source_id": "fr-immigration-rules",
        "title": "A rule",
        "doc_type": "Rule",
        "agencies": ["U.S. Citizenship and Immigration Services"],
        "publication_date": date(2025, 3, 1),
        "effective_on": date(2025, 4, 1),
        "html_url": "https://www.federalregister.gov/documents/2025/03/01/2025-01234/a-rule",
        "raw_text_url": "https://www.federalregister.gov/documents/full_text/text/2025/03/01/2025-01234.txt",
        "fetched_at": datetime.now(UTC),
        "sha256": sha256_text(text),
        "text": text,
    }

    data.update(over)
    return Snapshot(**data)



def test_loads_repo_allowlist():
    a = load_allowlist(ALLOWLIST)
    assert a.get("fr-immigration-rules") is not None
    assert a.get("nope") is None
    assert a.sources[0].query is not None and a.sources[0].query.since == date(2023, 1,1)
    pdfs = a.get("official-pdfs")
    assert pdfs is not None and len(pdfs.documents) == 4

    def test_servable_is_only_public_and_immutable():
        assert set(SERVABLE) == {"public"}
        with pytest.raises(AttributeError):
            SERVABLE.add("internal")

def test_unknown_keys_and_bad_values_rejected(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(ALLOWLIST.read_text().replace('"agencies"', '"agency"'))
    with pytest.raises(ValidationError):
        load_allowlist(p)
    p.write_text(ALLOWLIST.read_text().replace('"max_documents": 250', '"max_documents": 50000'))
    with pytest.raises(ValidationError):
        load_allowlist(p)


def test_pdf_source_needs_documents(tmp_path):
    import json

    data = json.loads(ALLOWLIST.read_text())
    for s in data["sources"]:
        if s["kind"] == "pdf":
            s["documents"] = []
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="needs at least one document"):
        load_allowlist(p)


def test_duplicate_source_ids_rejected(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(ALLOWLIST.read_text().replace("fr-presidential-immigration", "fr-immigration-rules"))
    with pytest.raises(GovernanceError):
        load_allowlist(p)


def test_snapshot_field_rules():
    with pytest.raises(ValidationError):
        make_snapshot(document_number="../../etc/passwd")
    with pytest.raises(ValidationError):
        make_snapshot(sha256="not-a-hash")


def test_check_snapshot_rules():
    a = load_allowlist(ALLOWLIST)
    check_snapshot(make_snapshot(), a)  # valid: no exception
    check_snapshot(make_snapshot(html_url="https://www.uscis.gov/policy-manual"), a)  # other official origin
    with pytest.raises(GovernanceError, match="not allowlisted"):
        check_snapshot(make_snapshot(source_id="random-blog"), a)
    with pytest.raises(GovernanceError, match="unexpected origin"):
        check_snapshot(make_snapshot(html_url="https://evil.example/x"), a)
    tampered = make_snapshot().model_copy(update={"text": "edited"})
    with pytest.raises(GovernanceError, match="checksum"):
        check_snapshot(tampered, a)


def test_non_public_source_not_servable(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(ALLOWLIST.read_text().replace('"classification": "public"', '"classification": "internal"', 1))
    with pytest.raises(GovernanceError, match="not servable"):
        check_snapshot(make_snapshot(), load_allowlist(p))


def test_load_snapshots_skips_catalog(tmp_path):
    (tmp_path / "2025-00002.json").write_text(make_snapshot(document_number="2025-00002").model_dump_json())
    (tmp_path / "2025-00001.json").write_text(make_snapshot(document_number="2025-00001").model_dump_json())
    (tmp_path / "catalog.json").write_text("{}")
    assert [s.document_number for s in load_snapshots(tmp_path)] == ["2025-00001", "2025-00002"]




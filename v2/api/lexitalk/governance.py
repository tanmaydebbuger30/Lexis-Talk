"""Corpus governance for v2: a source allowlist plus per-document provenance.

Two kinds of sources are allowed:
  - federal_register: documents fetched from the Federal Register API by query
  - pdf: official PDFs (8 U.S.C., 8 CFR, USCIS Policy Manual, Visa Bulletin)
Only documents from an allowlisted, public source, with an official origin URL
and an unchanged checksum, may be indexed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# The four data classifications. Only "public" may ever reach the public index.
Classification = Literal["public", "internal", "confidential", "restricted"]
SERVABLE: frozenset[str] = frozenset({"public"})

# Federal Register document types: final rule, proposed rule, notice, presidential document.
DocType = Literal["RULE", "PRORULE", "NOTICE", "PRESDOCU"]

# Only documents whose official URL starts with one of these may be indexed.
OFFICIAL_ORIGINS: tuple[str, ...] = (
    "https://www.federalregister.gov/",
    "https://uscode.house.gov/",
    "https://www.govinfo.gov/",
    "https://www.uscis.gov/",
    "https://travel.state.gov/",
)


class Strict(BaseModel):
    # extra="forbid": unknown keys are an error, so a typo in sources.json fails loudly.
    model_config = ConfigDict(extra="forbid")

class FederalRegisterQuery(Strict):
    agencies: list[str] = Field(default_factory=list)  # agency slugs; empty = any agency
    types: list[DocType]
    term: str | None = None  # optional full-text search
    since: date  # "2023-01-01" in JSON becomes a real date
    max_documents: int = Field(default=200, ge=1, le=2000)  # cost cap


class PdfDocument(Strict):
    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")  # e.g. "usc-title-08"
    file: str  # filename inside v2/data/pdf/
    title: str  # shown to users in citations
    doc_type: str  # "Statute", "Regulation", "Policy Manual", "Visa Bulletin"
    source_url: str  # the official page it came from
    as_of: date  # the date the text is current as of
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # fingerprint of the PDF file


class SourceSpec(Strict):
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    kind: Literal["federal_register", "pdf"]
    title: str
    classification: Classification
    license: str
    owner: str
    query: FederalRegisterQuery | None = None  # only for kind="federal_register"
    documents: list[PdfDocument] = Field(default_factory=list)  # only for kind="pdf"

    @model_validator(mode="after")
    def _kind_matches(self) -> SourceSpec:
        # Each kind must carry what it needs: an API query, or a list of files.
        if self.kind == "federal_register" and self.query is None:
            raise ValueError("a federal_register source needs a query")
        if self.kind == "pdf" and not self.documents:
            raise ValueError("a pdf source needs at least one document")
        return self


class SourceAllowlist(Strict):
    version: int
    description: str = ""
    sources: list[SourceSpec]

    def get(self, source_id: str) -> SourceSpec | None:
        """Look up a source by id; None if it isn't allowlisted."""
        return next((s for s in self.sources if s.source_id == source_id), None)


class Snapshot(Strict):
    """One fetched document = the raw (bronze) layer.

    Stored as data/federal_register/<document_number>.json.
    """

    document_number: str = Field(pattern=r"^[A-Za-z0-9-]{3,40}$")  # also safe as a filename
    source_id: str  # which allowlist entry produced it (lineage)
    title: str
    doc_type: str
    abstract: str | None = None
    action: str | None = None
    agencies: list[str]
    publication_date: date
    effective_on: date | None = None  # when the change takes effect
    citation: str | None = None
    html_url: str  # official page we link users to
    raw_text_url: str
    fetched_at: datetime
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # fingerprint of `text` at download time
    text: str


class GovernanceError(Exception):
    pass


def load_allowlist(path: Path) -> SourceAllowlist:
    """Read and validate sources.json."""
    a = SourceAllowlist.model_validate(json.loads(path.read_text(encoding="utf-8")))
    ids = [s.source_id for s in a.sources]
    if len(ids) != len(set(ids)):
        raise GovernanceError("duplicate source_id in allowlist")
    return a


def sha256_text(text: str) -> str:
    """SHA-256 fingerprint of a text (hex)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_snapshot(snap: Snapshot, allowlist: SourceAllowlist) -> None:
    """Raise GovernanceError unless this snapshot may be indexed."""
    spec = allowlist.get(snap.source_id)
    if spec is None:
        raise GovernanceError(f"{snap.document_number}: source {snap.source_id!r} is not allowlisted")
    if spec.classification not in SERVABLE:
        raise GovernanceError(f"{snap.document_number}: source classification {spec.classification!r} is not servable")
    if not snap.html_url.startswith(OFFICIAL_ORIGINS):
        raise GovernanceError(f"{snap.document_number}: unexpected origin {snap.html_url}")
    if sha256_text(snap.text) != snap.sha256:
        raise GovernanceError(f"{snap.document_number}: checksum mismatch (snapshot modified after fetch)")


def load_snapshots(snapshot_dir: Path) -> list[Snapshot]:
    """Load every saved document in a folder, skipping the catalog summary."""
    return [
        Snapshot.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(snapshot_dir.glob("*.json"))
        if p.name != "catalog.json"
    ]
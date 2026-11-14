"""Corpus governance: manifest loading, integrity checks, classification gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Classification = Literal["public", "internal", "confidential", "restricted"]
SERVABLE: frozenset[str] = frozenset({"public"})


class ManifestDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    file: str
    title: str
    classification: Classification
    source: str
    license: str
    owner: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int
    description: str = ""
    documents: list[ManifestDocument]


class GovernanceError(Exception):
    pass


def load_manifest(path: Path) -> Manifest:
    m = Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    ids = [d.document_id for d in m.documents]
    if len(ids) != len(set(ids)):
        raise GovernanceError("duplicate document_id in manifest")
    return m


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def resolve_servable(manifest: Manifest, corpus_dir: Path) -> list[tuple[ManifestDocument, Path]]:
    """Return only public documents whose bytes match the manifest checksum.

    Raises on checksum mismatch or path traversal: a changed file must be
    re-reviewed and re-approved in the manifest, not silently ingested.
    """
    root = corpus_dir.resolve()
    out: list[tuple[ManifestDocument, Path]] = []
    for doc in manifest.documents:
        if doc.classification not in SERVABLE:
            continue
        p = (root / doc.file).resolve()
        if root not in p.parents:
            raise GovernanceError(f"{doc.document_id}: path escapes corpus dir")
        if not p.is_file():
            raise GovernanceError(f"{doc.document_id}: file not found: {doc.file}")
        actual = sha256_file(p)
        if actual != doc.sha256:
            raise GovernanceError(f"{doc.document_id}: checksum mismatch (got {actual})")
        out.append((doc, p))
    return out


def unlisted_files(manifest: Manifest, corpus_dir: Path) -> list[str]:
    listed = {d.file for d in manifest.documents}
    return sorted(p.name for p in corpus_dir.glob("*.pdf") if p.name not in listed)

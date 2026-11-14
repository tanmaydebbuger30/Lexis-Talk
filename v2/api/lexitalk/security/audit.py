"""Structured audit logging that never records raw user content.

Each record has: request id, route, a salted hash of the client IP (so abuse
can be correlated without storing the IP), sizes, languages, redaction counts,
source document ids, latency and outcome. Message text, transcripts and answers
are never logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from typing import Any

_logger = logging.getLogger("lexitalk.audit")

# Keys that must never appear in an audit record.
FORBIDDEN_KEYS = frozenset({"message", "text", "answer", "transcript", "history", "content", "ip"})


def configure_logging(level: str = "INFO") -> None:
    if _logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(level)
    _logger.propagate = False


def hash_client(ip: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()[:16]


def audit(event: str, **fields: Any) -> dict[str, Any]:
    leaked = FORBIDDEN_KEYS.intersection(fields)
    if leaked:
        raise ValueError(f"audit record must not contain raw content: {sorted(leaked)}")
    record = {"event": event, **fields}
    _logger.info(json.dumps(record, default=str, ensure_ascii=False))
    return record

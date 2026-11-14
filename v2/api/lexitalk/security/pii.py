"""Deterministic PII detection and masking.

Applied at four points (see DATA_GOVERNANCE.md):
  1. ingestion  - chunks are masked before they are embedded/stored
  2. inbound    - user messages and history are masked before reaching the LLM
  3. outbound   - answers and snippets are masked before reaching the client
  4. logging    - raw text is never logged; only counts/hashes

Regex + checksum validation (Luhn for cards, Verhoeff for Aadhaar) keeps false
positives low and makes the behaviour fully testable offline. It is not a
complete PII solution: names and free-form addresses are NOT detected. A
model-based detector (e.g. Microsoft Presidio) is the documented upgrade path.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn_valid(number: str) -> bool:
    d = _digits(number)
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


_V_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_V_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    d = _digits(number)
    if len(d) != 12 or d[0] in "01":
        return False
    c = 0
    for i, ch in enumerate(reversed(d)):
        c = _V_D[c][_V_P[i % 8][int(ch)]]
    return c == 0


@dataclass(frozen=True)
class Detector:
    type: str
    pattern: re.Pattern[str]
    validate: Callable[[str], bool] | None = None


# Order matters: more specific detectors run first so their spans win.
DETECTORS: tuple[Detector, ...] = (
    Detector("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    Detector("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), luhn_valid),
    Detector("AADHAAR", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), verhoeff_valid),
    Detector("US_SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    Detector("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    Detector("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")),
    Detector(
        "PHONE",
        re.compile(
            r"(?<![\w+])(?:"
            r"\+\d{1,3}[ .-]?\d{2,5}(?:[ .-]?\d{2,5}){1,4}"  # international, leading +
            r"|(?:1[ .-])?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}"  # North American
            r"|[6-9]\d{4}[ -]\d{5}"  # Indian mobile, grouped
            r")(?![\w])"
        ),
        lambda s: 10 <= len(_digits(s)) <= 15,
    ),
    Detector("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
)


@dataclass(frozen=True)
class MaskResult:
    text: str
    counts: dict[str, int]

    @property
    def redacted(self) -> bool:
        return bool(self.counts)


Span = tuple[int, int, str]


def find_spans(text: str) -> list[Span]:
    taken: list[Span] = []
    for det in DETECTORS:
        for m in det.pattern.finditer(text):
            s, e = m.span()
            if any(s < te and ts < e for ts, te, _ in taken):
                continue
            if det.validate and not det.validate(m.group()):
                continue
            taken.append((s, e, det.type))
    taken.sort()
    return taken


def _apply(text: str, spans: list[Span], start: int, end: int) -> str:
    out: list[str] = []
    pos = start
    for s, e, t in spans:
        if s < start or e > end:
            continue
        out.append(text[pos:s])
        out.append(f"[{t}]")
        pos = e
    out.append(text[pos:end])
    return "".join(out)


def mask(text: str) -> MaskResult:
    """Replace detected PII with typed placeholders like ``[EMAIL]``."""
    spans = find_spans(text)
    if not spans:
        return MaskResult(text, {})
    return MaskResult(_apply(text, spans, 0, len(text)), dict(Counter(t for _, _, t in spans)))


class StreamMasker:
    """Mask PII in a token stream without leaking partial entities.

    The last ``hold`` characters are held back because they may be the start of
    an entity that is not complete yet; the emit cut is also moved so it never
    falls inside a detected span. ``flush()`` emits the remainder.
    """

    def __init__(self, hold: int = 96) -> None:
        self.hold = hold
        self._buf = ""
        self._pos = 0  # raw chars already emitted

    def feed(self, delta: str) -> str:
        self._buf += delta
        return self._emit(len(self._buf) - self.hold)

    def flush(self) -> str:
        return self._emit(len(self._buf))

    def _emit(self, cut: int) -> str:
        if cut <= self._pos:
            return ""
        spans = find_spans(self._buf)
        for s, e, _ in spans:
            if s < cut < e:
                cut = s
                break
        if cut <= self._pos:
            return ""
        out = _apply(self._buf, spans, self._pos, cut)
        self._pos = cut
        return out

    @property
    def counts(self) -> dict[str, int]:
        return mask(self._buf).counts


def merge_counts(*counts: dict[str, int]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for c in counts:
        total.update(c)
    return dict(total)

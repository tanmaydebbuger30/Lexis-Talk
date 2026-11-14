import random

import pytest

from lexitalk.security.pii import StreamMasker, luhn_valid, mask, verhoeff_valid


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("mail me at jane.doe+law@example.co.uk please", "EMAIL"),
        ("card 4111 1111 1111 1111 exp", "CREDIT_CARD"),
        ("card 4111-1111-1111-1111", "CREDIT_CARD"),
        ("my aadhaar is 2345 6789 0124", "AADHAAR"),
        ("ssn 123-45-6789", "US_SSN"),
        ("PAN ABCDE1234F on file", "PAN"),
        ("call (410) 555-0199 today", "PHONE"),
        ("call +91 98765 43210", "PHONE"),
        ("call 410-555-0199", "PHONE"),
        ("from 192.168.10.4", "IP_ADDRESS"),
        ("IBAN GB82 WEST 1234 5698 7654 32", "IBAN"),
    ],
)
def test_detects_and_masks(text: str, kind: str) -> None:
    r = mask(text)
    assert r.counts.get(kind) == 1, r
    assert f"[{kind}]" in r.text


@pytest.mark.parametrize(
    "text",
    [
        "See 2 U.S.C. § 4301 and section 192 of title 3.",
        "Enacted July 30, 1947, ch. 388, 61 Stat. 633.",
        "Pub. L. 119-36 amended the section in 2025.",
        "card 4111 1111 1111 1112",  # fails Luhn
        "id 2345 6789 0125",  # fails Verhoeff
    ],
)
def test_no_false_positives_on_legal_text(text: str) -> None:
    assert mask(text).counts == {}, mask(text)


def test_checksums() -> None:
    assert luhn_valid("4111111111111111") and not luhn_valid("4111111111111112")
    assert verhoeff_valid("234567890124") and not verhoeff_valid("234567890125")


def test_multiple_entities_counted() -> None:
    r = mask("a@b.io and c@d.io, phone 410-555-0199")
    assert r.counts == {"EMAIL": 2, "PHONE": 1}
    assert "a@b.io" not in r.text and "410-555-0199" not in r.text


SECRETS = ["jane.doe@example.com", "410-555-0199", "4111 1111 1111 1111", "123-45-6789"]


@pytest.mark.parametrize("seed", range(40))
def test_stream_masker_never_leaks_split_entities(seed: int) -> None:
    rnd = random.Random(seed)
    text = " ".join(
        [rnd.choice(["The statute says", "contact", "or", "regarding §102,"]) for _ in range(10)]
        + [rnd.choice(SECRETS)]
        + ["end of answer."] * 3
    )
    sm = StreamMasker(hold=48)
    out, i = [], 0
    while i < len(text):
        n = rnd.randint(1, 7)
        out.append(sm.feed(text[i : i + n]))
        i += n
    out.append(sm.flush())
    streamed = "".join(out)
    assert streamed == mask(text).text
    for s in SECRETS:
        assert s not in streamed

import pytest

from lexitalk.config import Settings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://a.app", ["https://a.app"]),
        ("https://a.app, https://b.app", ["https://a.app", "https://b.app"]),
        ('["https://a.app","https://b.app"]', ["https://a.app", "https://b.app"]),
    ],
)
def test_allowed_origins_from_env(monkeypatch, raw, expected) -> None:
    monkeypatch.setenv("LEXITALK_ALLOWED_ORIGINS", raw)
    assert Settings(_env_file=None).allowed_origins == expected


def test_secret_is_not_exposed_in_repr(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-very-secret")
    s = Settings(_env_file=None)
    assert "sk-test-very-secret" not in repr(s) and "sk-test-very-secret" not in str(s.model_dump())
    assert s.openai_api_key is not None and s.openai_api_key.get_secret_value() == "sk-test-very-secret"

from types import SimpleNamespace

import pytest

from src.llm.gemini import GeminiClient, GeminiError, get_api_key
from src.llm.summarize import summarize


class FakeModels:
    def __init__(self, models=None, response="Grounded response [Source 1].", error=None):
        self._models = models or []
        self.response = response
        self.error = error
        self.generated = []

    def list(self):
        if self.error:
            raise self.error
        return self._models

    def generate_content(self, *, model, contents):
        self.generated.append((model, contents))
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.response)


def test_gemini_requires_key_without_constructing_sdk(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiClient(api_key="")
    assert client.available() is False
    with pytest.raises(GeminiError, match="not configured"):
        client.generate("private test text")


def test_api_key_environment_has_priority(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-only")
    assert get_api_key() == "test-key-only"


def test_models_are_discovered_and_filtered_by_generation_support():
    models = [
        SimpleNamespace(name="models/account-flash", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/image-only", supported_actions=["embedContent"]),
    ]
    fake = FakeModels(models)
    client = GeminiClient(api_key="unit-test-key", sdk_client=SimpleNamespace(models=fake))
    assert client.models() == ["account-flash"]


def test_generation_uses_discovered_model_and_returns_text():
    fake = FakeModels([SimpleNamespace(name="models/available-model", supported_actions=["generateContent"])])
    client = GeminiClient(api_key="unit-test-key", sdk_client=SimpleNamespace(models=fake))
    assert client.generate("Only use this evidence.") == "Grounded response [Source 1]."
    assert fake.generated == [("available-model", "Only use this evidence.")]


def test_quota_errors_are_user_friendly_and_do_not_echo_raw_error():
    class QuotaError(Exception):
        code = 429
        message = "RESOURCE_EXHAUSTED hidden diagnostic"

    fake = FakeModels(error=QuotaError("RESOURCE_EXHAUSTED private diagnostic"))
    client = GeminiClient(api_key="unit-test-key", sdk_client=SimpleNamespace(models=fake))
    with pytest.raises(GeminiError, match="quota") as error:
        client.models()
    assert "private diagnostic" not in str(error.value)


@pytest.mark.parametrize(
    ("status", "detail", "message"),
    [
        (403, "permission denied confidential response", "authenticate"),
        (None, "connection timed out with private endpoint", "network connection"),
    ],
)
def test_auth_and_network_errors_are_safe(status, detail, message):
    class ProviderFailure(Exception):
        code = status
        message = detail

    fake = FakeModels(error=ProviderFailure(detail))
    client = GeminiClient(model="account-model", api_key="unit-test-key", sdk_client=SimpleNamespace(models=fake))
    with pytest.raises(GeminiError, match=message) as error:
        client.generate("non-sensitive test")
    assert "confidential" not in str(error.value)
    assert "private endpoint" not in str(error.value)


def test_gemini_summary_errors_are_not_silently_reported_as_success():
    class BrokenGemini:
        provider = "Gemini"

        def available(self):
            return True

        def generate(self, prompt):
            raise GeminiError("Gemini quota reached.")

    with pytest.raises(GeminiError, match="quota"):
        summarize("The report contains enough readable source text for a useful extractive summary.", BrokenGemini())

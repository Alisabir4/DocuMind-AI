import pytest

from src.llm.gemini import GeminiError
from src.llm.providers import OllamaUnavailable, generate_text, normalize_provider


class FakeGemini:
    def __init__(self, response="Gemini answer [Source 1].", error=None):
        self.response = response
        self.error = error
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.response


class FakeOllama:
    def __init__(self, available=True, response="Local answer [Source 1]."):
        self.is_available = available
        self.response = response
        self.prompts = []

    def available(self):
        return self.is_available

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


@pytest.mark.parametrize("provider", ["gemini", "Gemini", "Cloud · Google Gemini"])
def test_gemini_selection_routes_to_gemini_only(provider):
    gemini = FakeGemini()
    ollama_created = []
    answer = generate_text(
        provider,
        "evidence-based prompt",
        cloud_consent=True,
        gemini_factory=lambda: gemini,
        ollama_factory=lambda: ollama_created.append(True),
    )
    assert answer == "Gemini answer [Source 1]."
    assert gemini.prompts == ["evidence-based prompt"]
    assert ollama_created == []


def test_gemini_consent_is_required_before_client_creation():
    client_created = []
    with pytest.raises(GeminiError, match="consent is not enabled"):
        generate_text(
            "Cloud · Google Gemini",
            "private retrieved passage",
            cloud_consent=False,
            gemini_factory=lambda: client_created.append(True),
        )
    assert client_created == []


def test_gemini_failure_is_not_silently_routed_to_ollama():
    gemini = FakeGemini(error=GeminiError("Gemini could not authenticate this request."))
    ollama_created = []
    with pytest.raises(GeminiError, match="authenticate"):
        generate_text(
            "gemini",
            "prompt",
            cloud_consent=True,
            gemini_factory=lambda: gemini,
            ollama_factory=lambda: ollama_created.append(True),
        )
    assert ollama_created == []


def test_local_selection_uses_ollama():
    ollama = FakeOllama()
    gemini_created = []
    answer = generate_text(
        "Local · Ollama",
        "prompt",
        cloud_consent=False,
        gemini_factory=lambda: gemini_created.append(True),
        ollama_factory=lambda: ollama,
    )
    assert answer == "Local answer [Source 1]."
    assert ollama.prompts == ["prompt"]
    assert gemini_created == []


def test_unavailable_local_model_has_local_provider_error():
    with pytest.raises(OllamaUnavailable, match="Ollama model is not running"):
        generate_text("ollama", "prompt", cloud_consent=False, ollama_factory=lambda: FakeOllama(available=False))


def test_unknown_provider_is_not_treated_as_ollama():
    assert normalize_provider("unexpected") is None
    with pytest.raises(ValueError, match="not recognized"):
        generate_text("unexpected", "prompt", cloud_consent=True)

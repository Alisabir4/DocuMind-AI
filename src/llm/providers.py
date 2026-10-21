"""Provider selection and guarded text generation for document-grounded requests."""
from __future__ import annotations

from typing import Callable

from .gemini import GeminiClient, GeminiError
from .ollama import OllamaClient


class OllamaUnavailable(RuntimeError):
    """The selected local model is not available."""


def normalize_provider(value: object) -> str | None:
    """Normalize canonical values and older/display-label session values."""
    normalized = str(value or "").strip().casefold()
    if normalized in {"gemini", "google gemini", "cloud · google gemini", "cloud - google gemini"}:
        return "gemini"
    if normalized in {"ollama", "local ollama", "local · ollama", "local - ollama"}:
        return "ollama"
    return None


def generate_text(
    provider: object,
    prompt: str,
    *,
    cloud_consent: bool,
    gemini_factory: Callable[[], GeminiClient] = GeminiClient,
    ollama_factory: Callable[[], OllamaClient] = OllamaClient,
) -> str:
    """Generate using the selected provider; never downgrade cloud to local."""
    selected = normalize_provider(provider)
    if selected == "gemini":
        if not cloud_consent:
            raise GeminiError(
                "Gemini cloud mode is selected, but cloud-data consent is not enabled. "
                "No document text was sent. Enable consent in Settings or switch to local mode."
            )
        return gemini_factory().generate(prompt)
    if selected == "ollama":
        client = ollama_factory()
        if not client.available():
            raise OllamaUnavailable("The configured Ollama model is not running.")
        return client.generate(prompt)
    raise ValueError("The selected AI provider is not recognized. Choose Local · Ollama or Cloud · Google Gemini in Settings.")

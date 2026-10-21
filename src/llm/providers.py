"""Provider selection and guarded text generation for document-grounded requests."""
from __future__ import annotations

import logging
from typing import Callable, MutableMapping, Any

from .gemini import GeminiClient, GeminiError, get_api_key
from .ollama import OllamaClient

logger = logging.getLogger("documind.providers")


class OllamaUnavailable(RuntimeError):
    """The selected local model is not available."""


class OllamaGenerationError(RuntimeError):
    """A local Ollama generation request failed after the service was found."""


def _safe_diagnostic(exc: Exception) -> str:
    """Return only known-safe provider messages or metadata, never raw payloads."""
    if isinstance(exc, GeminiError):
        message = str(exc).casefold()
        if any(term in message for term in ("quota", "rate limit", "resource_exhausted")):
            return "Gemini rate limit or quota error."
        if any(term in message for term in ("authenticate", "api key", "unauthenticated", "permission denied", "forbidden")):
            return "Gemini authentication or permission error."
        if any(term in message for term in ("network", "connection", "timeout", "could not be reached")):
            return "Gemini network request error."
        if "not configured" in message:
            return "Gemini API key is not configured."
        return "Gemini request failed; provider error details omitted."
    if isinstance(exc, (OllamaUnavailable, OllamaGenerationError)):
        return str(exc)[:240]
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(status, (int, str)) and str(status).isdigit():
        return f"Provider request failed with status {status}."
    return f"{type(exc).__name__}; provider error details omitted."


def normalize_provider(value: object) -> str | None:
    """Normalize canonical values and older/display-label session values."""
    normalized = str(value or "").strip().casefold()
    if normalized in {"gemini", "google gemini", "cloud · google gemini", "cloud - google gemini"}:
        return "gemini"
    if normalized in {"ollama", "local ollama", "local · ollama", "local - ollama"}:
        return "ollama"
    return None


def remember_provider_selection(state: MutableMapping[str, Any], value: object) -> str | None:
    """Persist provider choice outside legacy and Settings-only widget keys."""
    selected = normalize_provider(value)
    if selected:
        state["dm_selected_provider"] = selected
    return selected


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
    key_configured = bool(get_api_key())
    logger.warning(
        "provider_diagnostic received_provider=%s cloud_consent=%s gemini_api_key_configured=%s",
        selected or "unrecognized",
        bool(cloud_consent),
        key_configured,
    )
    if selected == "gemini":
        if not cloud_consent:
            error = GeminiError(
                "Gemini cloud mode is selected, but cloud-data consent is not enabled. "
                "No document text was sent. Enable consent in Settings or switch to local mode."
            )
            logger.warning("provider_diagnostic adapter=gemini exception_type=%s error=%s", type(error).__name__, _safe_diagnostic(error))
            raise error
        logger.warning("provider_diagnostic adapter_called=gemini")
        try:
            return gemini_factory().generate(prompt)
        except Exception as exc:
            logger.warning(
                "provider_diagnostic adapter=gemini exception_type=%s error=%s",
                type(exc).__name__,
                _safe_diagnostic(exc),
            )
            if isinstance(exc, GeminiError):
                raise
            raise GeminiError("Gemini could not complete the request. Try again, or check your API key and connection.") from None
    if selected == "ollama":
        logger.warning("provider_diagnostic adapter_called=ollama")
        client = ollama_factory()
        if not client.available():
            error = OllamaUnavailable("The configured Ollama model is not running.")
            logger.warning("provider_diagnostic adapter=ollama exception_type=%s error=%s", type(error).__name__, _safe_diagnostic(error))
            raise error
        try:
            return client.generate(prompt)
        except Exception as exc:
            logger.warning(
                "provider_diagnostic adapter=ollama exception_type=%s error=%s",
                type(exc).__name__,
                _safe_diagnostic(exc),
            )
            if isinstance(exc, OllamaGenerationError):
                raise
            raise OllamaGenerationError("Ollama could not complete the request. Check that the local service is running.") from None
    error = ValueError("The selected AI provider is not recognized. Choose Local · Ollama or Cloud · Google Gemini in Settings.")
    logger.warning("provider_diagnostic adapter=none exception_type=%s error=provider identifier not recognized", type(error).__name__)
    raise error

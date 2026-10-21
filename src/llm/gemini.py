"""Optional Google Gemini text generation through the supported Google Gen AI SDK."""
from __future__ import annotations

import os
from typing import Any

GEMINI_FAILURE_MESSAGE = "Gemini could not generate a response. Check the GEMINI_API_KEY Streamlit secret and try again."


class GeminiError(RuntimeError):
    """A safe, user-facing Gemini setup or request error."""


def get_api_key() -> str | None:
    """Read credentials without displaying, logging, or persisting them."""
    try:
        import streamlit as st

        key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        # Streamlit secrets can be unavailable outside an app context or when unset.
        key = ""
    return key or os.getenv("GEMINI_API_KEY", "").strip() or None


class GeminiClient:
    provider = "Gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None, sdk_client: Any = None):
        self.model = model
        self._api_key = api_key if api_key is not None else get_api_key()
        self._client = sdk_client

    def available(self) -> bool:
        return bool(self._api_key or self._client is not None)

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise GeminiError("Gemini is not configured. Add GEMINI_API_KEY to the environment or Streamlit secrets.")
        try:
            from google import genai
        except ImportError:
            raise GeminiError("The Google Gen AI SDK is not installed. Install the project requirements to enable Gemini.") from None
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    @staticmethod
    def _friendly_error(exc: Exception) -> GeminiError:
        # Classify status/message without returning raw SDK diagnostics to the UI.
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        detail = f"{status or ''} {getattr(exc, 'message', '')} {exc}".casefold()
        if str(status) == "429" or any(term in detail for term in ("resource_exhausted", "quota", "rate limit", "too many requests")):
            return GeminiError("Gemini rate limit or API quota reached. Check your Google AI API quota, then try again later.")
        if str(status) in {"401", "403"} or any(term in detail for term in ("api key", "permission denied", "unauthenticated", "forbidden")):
            return GeminiError("Gemini could not authenticate this request. Check the API key and its access in Google AI Studio.")
        if any(term in detail for term in ("timeout", "connection", "network", "dns", "unavailable")):
            return GeminiError("Gemini could not be reached. Check your network connection and try again.")
        return GeminiError(GEMINI_FAILURE_MESSAGE)

    def models(self) -> list[str]:
        """List models this key can use for text generation."""
        client = self._get_client()
        try:
            available = []
            for model in client.models.list():
                actions = getattr(model, "supported_actions", None) or []
                if any("generatecontent" == str(action).replace("_", "").casefold() for action in actions):
                    name = str(getattr(model, "name", "")).removeprefix("models/")
                    if name:
                        available.append(name)
            return sorted(set(available))
        except GeminiError:
            raise
        except Exception as exc:
            raise self._friendly_error(exc) from None

    def generate(self, prompt: str) -> str:
        if not self.available():
            raise GeminiError("Gemini is not configured. Add GEMINI_API_KEY to Streamlit secrets.")
        client = self._get_client()
        model = self.model
        if not model:
            choices = self.models()
            if not choices:
                raise GeminiError("No text generation models are available for this Gemini API key.")
            model = choices[0]
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            result = (getattr(response, "text", None) or "").strip()
            if not result:
                raise GeminiError("Gemini returned no text.")
            return result
        except GeminiError:
            raise
        except Exception as exc:
            raise self._friendly_error(exc) from None


def answer_with_gemini(question: str, sources: list[dict[str, Any]], *, cloud_consent: bool,
                       client_factory=GeminiClient) -> str:
    """Answer a question from retrieved evidence with Gemini only."""
    if not cloud_consent:
        raise GeminiError("Enable Gemini cloud consent in Settings before asking. No document text was sent.")
    evidence = "\n\n".join(
        f"[Source {index + 1}: {source['name']}, page {source['page']}]\n{source['text']}"
        for index, source in enumerate(sources)
    )
    prompt = (
        "Answer only from the retrieved evidence. If evidence is insufficient, say so plainly. "
        "Cite evidence only with the exact labels [Source 1], [Source 2], etc. shown below. "
        "Do not invent document names, page numbers, or source labels.\n\n"
        f"QUESTION: {question}\n\nSOURCES:\n{evidence}"
    )
    try:
        return client_factory().generate(prompt)
    except Exception:
        raise GeminiError(GEMINI_FAILURE_MESSAGE) from None

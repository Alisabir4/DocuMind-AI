"""Optional Google Gemini text generation through the supported Google Gen AI SDK."""
from __future__ import annotations

import os
from typing import Any


class GeminiError(RuntimeError):
    """A safe, user-facing Gemini setup or request error."""


def get_api_key() -> str | None:
    """Read credentials without displaying, logging, or persisting them."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        return key or None
    except Exception:
        # Streamlit secrets can be unavailable outside an app context or when unset.
        return None


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
        return GeminiError("Gemini could not complete the request. Try again, or switch back to local AI.")

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
            raise GeminiError("Gemini is not configured. Add GEMINI_API_KEY in Settings before selecting Gemini.")
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
                raise GeminiError("Gemini returned no text. Review the request and try again.")
            return result
        except GeminiError:
            raise
        except Exception as exc:
            raise self._friendly_error(exc) from None

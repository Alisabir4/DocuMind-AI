"""Optional local Ollama HTTP client."""
from __future__ import annotations
import os
import requests

class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    def models(self) -> list[str]:
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=2)
            res.raise_for_status()
            return [m.get("name", "") for m in res.json().get("models", [])]
        except Exception:
            return []

    def available(self) -> bool:
        return self.model in self.models()

    def generate(self, prompt: str) -> str:
        response = requests.post(f"{self.base_url}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False}, timeout=180)
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        if not result:
            raise RuntimeError("Ollama returned an empty response.")
        return result

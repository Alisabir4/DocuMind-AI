"""Grounded, chunk-aware summaries with deterministic extractive fallback."""
from __future__ import annotations
import re
from collections import Counter
from src.document_processing.pdf import Chunk

def extractive_summary(text: str, limit: int = 5) -> dict:
    page_blocks = re.findall(r"\[Page\s+(\d+)\]\s*(.*?)(?=\[Page\s+\d+\]|$)", text, re.S | re.I)
    if page_blocks:
        sentences_with_pages = [(int(page), s.strip()) for page, body in page_blocks for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 35]
    else:
        sentences_with_pages = [(None, s.strip()) for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 35]
    sentences = [s for _, s in sentences_with_pages]
    if not sentences:
        return {"overview": "No readable text was available for summarization.", "key_points": [], "key_point_pages": [], "dates": [], "numbers": [], "entities": [], "action_items": [], "method": "extractive"}
    words = Counter(w.lower() for w in re.findall(r"[A-Za-z]{4,}", text))
    ranked = sorted(enumerate(sentences), key=lambda pair: (sum(words[w.lower()] for w in re.findall(r"[A-Za-z]{4,}", pair[1])) / max(len(pair[1].split()), 1), -pair[0]), reverse=True)
    picked = sorted((i, s) for i, s in ranked[:limit])
    dates = sorted(set(re.findall(r"\b(?:\d{1,2}[/-]){1,2}\d{2,4}\b|\b(?:19|20)\d{2}\b", text)))
    actions = [s for s in sentences if re.search(r"\b(must|shall|required to|action item|please|deadline|due by)\b", s, re.I)][:5]
    numbers = sorted(set(re.findall(r"(?<!\w)(?:\$|€|£)?\d[\d,]*(?:\.\d+)?%?(?!\w)", text)))
    entities = sorted(set(re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b", text)))
    entities = [e for e in entities if e not in {"Page", "The", "This", "That", "These", "Those", "However", "Whereas"}]
    pages = [page for page, _ in sentences_with_pages]
    return {"overview": " ".join(s for _, s in picked[:2]), "key_points": [s for _, s in picked],
            "key_point_pages": [pages[i] for i, _ in picked], "dates": dates, "numbers": numbers,
            "entities": entities, "action_items": actions, "method": "extractive"}

def summarize(text: str, llm=None, max_chars: int = 12000) -> dict:
    fallback = extractive_summary(text)
    if llm is None or not getattr(llm, "available", lambda: False)():
        return fallback
    prompt = ("Summarize only facts supported by the source. Return concise Markdown sections: Overview, Key points, Important dates, Action items. "
              "Do not infer missing facts; say None found when unsupported.\n\nSOURCE:\n" + text[:max_chars])
    try:
        generated = llm.generate(prompt)
        provider = getattr(llm, "provider", "local model")
        return {**fallback, "generated": generated, "method": f"{provider} with extractive evidence"}
    except Exception:
        # Cloud errors need to reach the UI so quota, auth, and connectivity issues
        # are not silently presented as a successful local model summary.
        if getattr(llm, "provider", "") == "Gemini":
            raise
        return fallback

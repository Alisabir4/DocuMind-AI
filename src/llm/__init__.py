"""Gemini generation and deterministic document summaries."""

from .gemini import GeminiClient, GeminiError
from .summarize import summarize, extractive_summary

__all__ = ["GeminiClient", "GeminiError", "summarize", "extractive_summary"]

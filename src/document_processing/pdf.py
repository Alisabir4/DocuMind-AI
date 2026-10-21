"""Safe PDF text extraction and page-aware chunking."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable
import pymupdf as fitz

MAX_PDF_BYTES = 50 * 1024 * 1024

class DocumentError(ValueError):
    """A user-correctable document processing error."""

@dataclass(frozen=True)
class PageText:
    page: int
    text: str

@dataclass(frozen=True)
class Chunk:
    page: int
    text: str

def extract_pdf(data: bytes) -> list[PageText]:
    if not data:
        raise DocumentError("The uploaded file is empty.")
    if len(data) > MAX_PDF_BYTES:
        raise DocumentError("PDF exceeds the 50 MB upload limit.")
    if not data.startswith(b"%PDF"):
        raise DocumentError("This file is not a valid PDF.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        if doc.is_encrypted:
            if not doc.authenticate(""):
                raise DocumentError("This PDF is password protected. Remove its password and try again.")
        pages = [PageText(i + 1, normalize_text(page.get_text("text"))) for i, page in enumerate(doc)]
        doc.close()
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("The PDF could not be opened. It may be corrupted or encrypted.") from exc
    if not pages or not any(p.text.strip() for p in pages):
        raise DocumentError("No selectable text was found. This may be a scanned PDF; OCR is required before import.")
    return pages

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00ad", "")
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()

def chunk_pages(pages: Iterable[PageText], chunk_size: int = 900, overlap: int = 120) -> list[Chunk]:
    if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be at least 100 and overlap must be smaller than chunk size.")
    chunks: list[Chunk] = []
    for page in pages:
        words = page.text.split()
        if not words:
            continue
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            body = " ".join(words[start:start + chunk_size]).strip()
            if body:
                chunks.append(Chunk(page.page, body))
            if start + chunk_size >= len(words):
                break
    return chunks

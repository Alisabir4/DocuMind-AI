"""Document intake orchestration."""
from __future__ import annotations
import re
from pathlib import PurePath
from src.document_processing.pdf import extract_pdf, chunk_pages

def safe_filename(name: str) -> str:
    base = PurePath(name.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return cleaned[:120] or "document.pdf"

def ingest(store, name: str, data: bytes, chunk_size: int = 900, overlap: int = 120) -> int:
    if not name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    safe = safe_filename(name)
    digest = store.digest(data)
    with store.connect() as db:
        exists = db.execute("SELECT id FROM documents WHERE digest=?", (digest,)).fetchone()
    if exists:
        raise ValueError("This document has already been uploaded.")
    try:
        pages = extract_pdf(data)
        chunks = chunk_pages(pages, chunk_size, overlap)
        return store.add_document(safe, safe, digest, pages, chunks)
    except Exception as exc:
        # Persist only failed, non-duplicate attempts with readable diagnostic text.
        try:
            store.add_failure(safe, safe, digest, str(exc))
        except Exception:
            pass
        raise

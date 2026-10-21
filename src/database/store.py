"""SQLite persistence for documents, chunks, and local activity."""
from __future__ import annotations
import hashlib, json, sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _init(self):
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL, safe_name TEXT NOT NULL,
                digest TEXT NOT NULL UNIQUE, status TEXT NOT NULL, page_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0, extracted_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL, error TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, page INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB,
                UNIQUE(document_id, ordinal));
                CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY, message TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);''')

    @staticmethod
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def add_document(self, name: str, safe_name: str, digest: str, pages: list, chunks: list) -> int:
        now = datetime.now(timezone.utc).isoformat()
        joined = "\n\n".join(f"[Page {p.page}]\n{p.text}" for p in pages)
        with self.connect() as db:
            cur = db.execute("INSERT INTO documents(name,safe_name,digest,status,page_count,chunk_count,extracted_text,created_at) VALUES(?,?,?,?,?,?,?,?)",
                             (name, safe_name, digest, "processed", len(pages), len(chunks), joined, now))
            doc_id = cur.lastrowid
            db.executemany("INSERT INTO chunks(document_id,ordinal,page,text) VALUES(?,?,?,?)",
                           [(doc_id, i, c.page, c.text) for i, c in enumerate(chunks)])
            db.execute("INSERT INTO activity(message,created_at) VALUES(?,?)", (f"Processed {name}", now))
            return int(doc_id)

    def add_failure(self, name: str, safe_name: str, digest: str, error: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            cur = db.execute("INSERT INTO documents(name,safe_name,digest,status,created_at,error) VALUES(?,?,?,?,?,?)",
                             (name, safe_name, digest, "failed", now, error[:500]))
            db.execute("INSERT INTO activity(message,created_at) VALUES(?,?)", (f"Failed to process {name}", now))
            return int(cur.lastrowid)

    def documents(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT id,name,status,page_count,chunk_count,created_at,error FROM documents ORDER BY id DESC")]

    def document(self, doc_id: int):
        with self.connect() as db:
            r = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return dict(r) if r else None

    def chunks(self, doc_ids: list[int] | None = None):
        with self.connect() as db:
            if doc_ids is None:
                rows = db.execute("SELECT c.*,d.name FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.status='processed' ORDER BY c.id")
            elif not doc_ids:
                return []
            else:
                marks = ",".join("?" for _ in doc_ids)
                rows = db.execute(f"SELECT c.*,d.name FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.status='processed' AND d.id IN ({marks}) ORDER BY c.id", doc_ids)
            return [dict(r) for r in rows]

    def set_embedding(self, chunk_id: int, embedding: bytes):
        with self.connect() as db:
            db.execute("UPDATE chunks SET embedding=? WHERE id=?", (embedding, chunk_id))

    def delete_document(self, doc_id: int):
        with self.connect() as db:
            db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            db.execute("INSERT INTO activity(message,created_at) VALUES(?,?)", (f"Removed document #{doc_id}", datetime.now(timezone.utc).isoformat()))

    def stats(self) -> dict[str, int]:
        with self.connect() as db:
            return {"documents": db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                    "processed": db.execute("SELECT COUNT(*) FROM documents WHERE status='processed'").fetchone()[0],
                    "chunks": db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]}

    def activity(self, limit=8):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,))]

    def page_text(self, doc_id: int) -> str:
        doc = self.document(doc_id)
        return doc["extracted_text"] if doc else ""

    def clear(self):
        with self.connect() as db:
            db.executescript("DELETE FROM chunks; DELETE FROM documents; DELETE FROM activity;")

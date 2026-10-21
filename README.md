# DocuMind AI

DocuMind AI is a local-first document intelligence workspace. Upload text-based PDFs, inspect extracted pages, create evidence-based summaries, and retrieve passages with page citations. SQLite stores metadata, extracted text, chunks, and activity locally.

## Features

- Six-section Streamlit dashboard: Overview, My Documents, AI Analysis, Ask Your Documents, Evaluation Lab, and Settings.
- Midnight Emerald dark design with responsive navigation and evidence panels.
- PDF validation and page-aware PyMuPDF extraction, SHA-256 duplicate detection, and OCR guidance for scanned PDFs.
- Adjustable word chunking with overlap and CPU Sentence Transformers embeddings. FAISS performs local similarity search.
- Gemini-only generative summaries and grounded Q&A after explicit cloud-data consent. Retrieval, indexing, and citations remain local.
- Deterministic extractive summary and retrieval-only results when Gemini is not configured.
- Evaluation Lab describes the planned workflow; it does not display fabricated benchmark scores.

## Windows setup

Python 3.11 is recommended:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set `GEMINI_API_KEY` in Streamlit Community Cloud under **App settings → Secrets**:

```toml
GEMINI_API_KEY = "your-key"
```

For local development, the same secret can be placed in `.streamlit/secrets.toml` (excluded from Git), or provided as a `GEMINI_API_KEY` environment variable. The key is never displayed or logged. In Settings, enable the cloud-data consent before requesting generated answers or summaries. Gemini receives the question and retrieved passages for Q&A, or selected extracted document text for summaries. Without consent, document text is not sent to Gemini.

Run the app and tests:

```powershell
streamlit run app.py
pytest -q
```

## Local processing and privacy

PDF extraction, SQLite-backed document management, embeddings, indexing, semantic search, retrieval, and citation locations run locally. The original PDFs are processed in memory and are not retained; extracted text and chunks are stored in the local SQLite database. The embedding model is downloaded on first use and cached on-device. By default, data is stored under `data/`; set `DOCUMIND_DATA_DIR` to use another local folder. Protect and back up this folder as appropriate. This prototype has no authentication, encryption-at-rest, or multi-user access control.

Gemini is the only generative AI provider. If the API key is missing or a Gemini request fails, the app displays a single clear Gemini error; it does not fall back to another provider. Local semantic search and evidence citations remain available.

## Limitations

Phase 1 supports selectable-text PDFs only and does not include OCR, authentication, multi-user isolation, original-file archives, cross-session chat history, or advanced entity extraction. Similarity scores are not calibrated probabilities. The in-memory search index is rebuilt from SQLite passages; SQLite remains the source of truth.

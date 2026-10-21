# DocuMind AI

DocuMind AI is a local-first Phase 1 document intelligence workspace. Upload text-based PDFs, inspect extracted pages, create evidence-based summaries, and retrieve passages with page citations. SQLite stores metadata, extracted text, chunks, and activity locally.

## Features

- Six-section Streamlit dashboard: Overview, My Documents, AI Analysis, Ask Your Documents, Evaluation Lab, and Settings.
- Midnight Emerald dark design system with a persistent local-workspace sidebar, global document search, direct import action, responsive layouts, and page-aware evidence panels.
- PDF validation and page-aware PyMuPDF extraction, 50 MB limit, SHA-256 duplicate detection, safe display names, and OCR guidance for scanned PDFs.
- Adjustable word chunking with overlap and a CPU Sentence Transformers embedding adapter (`all-MiniLM-L6-v2`). FAISS CPU performs similarity search; numpy is a small runtime fallback if FAISS is missing.
- Evidence-based extraction summary fallback; optional local Ollama or explicitly enabled Google Gemini generation.
- Semantic Q&A retrieves passages locally before response generation and displays document/page citations and excerpts. When generation is unavailable, it displays retrieval results without inventing an answer.
- Evaluation Lab currently explains the evaluation workflow that is planned; it does not display fabricated benchmark scores.
- SQLite-backed stats, document lifecycle, and activity history. Deleting a document cascades to its chunks.

## Architecture

`app.py` contains Streamlit presentation and navigation. `src/document_processing` handles extraction/chunking; `src/database` owns SQLite; `src/embeddings` provides a lazy CPU encoder; `src/retrieval` provides FAISS search; `src/llm` holds optional Ollama, Gemini, and summary logic; `src/services` coordinates ingestion. Unit tests use generated PDFs and fake encoders/providers.

## Windows PowerShell setup

Python 3.11 is recommended. From this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or invoke `.\.venv\Scripts\python.exe` directly.

Run the app:

```powershell
streamlit run app.py
```

Run automated tests:

```powershell
pytest -q
```

## Embeddings and first run

The application starts without loading or downloading an embedding model. The first semantic search or index rebuild lazily loads `sentence-transformers/all-MiniLM-L6-v2` on CPU. Hugging Face downloads the model files on that first use (roughly 90 MB, plus dependency/runtime overhead) and caches them in the standard user model cache. This requires internet access once. The selected model is intentionally lightweight for CPU use. No GPU is used.

## Optional Ollama

Install Ollama for Windows from [ollama.com](https://ollama.com), then in PowerShell:

```powershell
ollama pull llama3.2:3b
$env:OLLAMA_MODEL = "llama3.2:3b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
streamlit run app.py
```

`OLLAMA_MODEL` and `OLLAMA_BASE_URL` are optional; defaults are `llama3.2:3b` and `http://localhost:11434`. Ollama runs on the same computer. A model must already be pulled; DocuMind does not download it silently.

## Optional Google Gemini

Install the project requirements to add the supported Google Gen AI Python SDK (`google-genai`). Gemini is off by default. Configure a key using either environment variables:

```powershell
$env:GEMINI_API_KEY = "your-key"
streamlit run app.py
```

Or add `GEMINI_API_KEY = "your-key"` to `.streamlit/secrets.toml`. That file is excluded by `.gitignore`; never commit it. The app does not display or log the key. In **Settings**, choose **Cloud · Google Gemini**, read the disclosure, and explicitly enable the cloud-data consent checkbox. Discover models to list text-generation models available to that API key; the app does not assume or hardcode an account model name. If none is selected, it discovers an available text model when generation is requested.

After enabling consent, Gemini receives the selected document's extracted text (up to 12,000 characters) and prompt for summaries, or the question and locally retrieved passages for Q&A, when you request generation. The original PDF, SQLite database, embeddings/index, and chat history are not uploaded. Parsing, indexing, retrieval, and citation locations remain local. Without cloud consent, Gemini is never called; local Ollama or extractive/retrieval-only behavior remains available. Gemini quota, authentication, and network failures are shown as user-friendly messages.

## What works without a generative model

PDF extraction, page labeling, SQLite-backed dashboard and document management, semantic indexing/search (after the embedding model is downloaded), retrieval-only Q&A with supporting passages, and deterministic extractive summaries work without a generative model. Natural-language answers and model-assisted summaries require an available configured Ollama model or explicitly enabled Gemini provider. If generation is unavailable, Q&A gives retrieved excerpts and clearly says generation is not active.

The extraction summary is deterministic and selects source sentences. Model-generated text is labeled separately and is not guaranteed to be factually correct; verify against the displayed source text. Q&A citations are built only from retrieved database chunks and page values.

## Storage and privacy

By default, the SQLite database is created under `data/` beside the app. Set `DOCUMIND_DATA_DIR` to choose another local directory. Uploaded PDFs are processed in memory and are not retained as original files. Extracted text and chunks are stored locally in SQLite. Ollama inference is local. Gemini receives extracted document text for a requested summary or retrieved passage text with the question for Q&A only after the user selects it and enables the explicit cloud-data consent in Settings. The embedding model is fetched from Hugging Face on first use; document text is encoded on-device. Do not use this prototype as a secure multi-user service: there is no authentication, encryption-at-rest layer, or access control. Back up or protect the local data folder according to your needs.

## Troubleshooting

- **Scanned PDF / no text:** Phase 1 does not include OCR. Run OCR externally and import the searchable PDF.
- **Password-protected or damaged PDF:** remove the password or repair the file before import.
- **Model download fails:** check internet access and retry; the app and all non-semantic features still start without the model.
- **Ollama unavailable:** start Ollama and confirm `ollama list` includes the configured model. Generation is optional.
- **Gemini not configured:** set `GEMINI_API_KEY` in the process environment or `.streamlit/secrets.toml`, then restart Streamlit.
- **Gemini model discovery or generation fails:** confirm the key can access Gemini and that the account has available quota. Model discovery uses the account's current supported text-generation models.
- **FAISS installation issue:** use 64-bit Python 3.11 and the pinned `faiss-cpu` wheel. Search has a numpy fallback if FAISS is unavailable.
- **PowerShell activation:** use the execution-policy command above or run the venv Python executable directly.

## Known limitations and future phases

Phase 1 supports selectable-text PDFs only, has no OCR, authentication, multi-user isolation, original-file archive, cross-session chat history, or advanced entity extraction. Extractive summary action/date recognition uses transparent text heuristics. Similarity scores are cosine similarity and should not be interpreted as calibrated probabilities. The local index is rebuilt in memory from SQLite passages; SQLite remains the source of truth. Future work can add OCR, stronger structured extraction with validated schemas, durable FAISS snapshots, richer audit controls, and additional document formats.

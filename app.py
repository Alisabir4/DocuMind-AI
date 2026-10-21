"""DocuMind AI — local-first document intelligence workspace."""
from __future__ import annotations

import os
import html
import re
from pathlib import Path
from typing import Any

import streamlit as st

from src.database.store import Store
from src.embeddings.encoder import Encoder, MODEL_NAME
from src.llm.gemini import GeminiClient, GeminiError, get_api_key
from src.llm.ollama import OllamaClient
from src.llm.summarize import summarize
from src.retrieval.index import Retriever
from src.services.documents import ingest

ROOT = Path(__file__).parent
DATA = Path(os.getenv("DOCUMIND_DATA_DIR", str(ROOT / "data")))
STORE = Store(DATA / "documind.sqlite3")

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Overview": ("dashboard", "Workspace overview", "A clear view of your local document library."),
    "My Documents": ("folder_open", "My documents", "Find, inspect, and manage your source files."),
    "AI Analysis": ("auto_awesome", "AI analysis", "Evidence-led summaries and extracted information."),
    "Ask Your Documents": ("forum", "Ask your documents", "Research across your library with page-level evidence."),
    "Evaluation Lab": ("science", "Evaluation lab", "A transparent space for measuring answer quality."),
    "Settings": ("settings", "Settings", "Local processing, retrieval, and workspace preferences."),
}
LINE_ICONS = {
    "document": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h8"/></svg>',
    "check": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M22 11.1V12a10 10 0 1 1-5.9-9.1"/><path d="m9 11 3 3L22 4"/></svg>',
    "chunks": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16M4 12h10M4 19h16"/><path d="M18 10v4m-2-2h4"/></svg>',
    "storage": '<svg viewBox="0 0 24 24" aria-hidden="true"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>',
}

st.markdown(
    """
<style>
:root {
  color-scheme: dark;
  --dm-ink: #FFFFFF;
  --dm-muted: #C5D8CE;
  --dm-label: #A8C0B3;
  --dm-line: #355747;
  --dm-paper: #091713;
  --dm-sidebar: #07110E;
  --dm-card: #10251E;
  --dm-elevated: #163127;
  --dm-emerald: #35C98A;
  --dm-green: #35C98A;
  --dm-mint: #70E6AC;
  --dm-shadow: 0 8px 24px rgba(0, 0, 0, .24);
}
html, body { color-scheme: dark; background: var(--dm-paper); }
body, .stApp, [data-testid="stAppViewContainer"] { font-family: Inter, "Segoe UI", Arial, sans-serif; color: var(--dm-ink); }
.stApp { background: radial-gradient(ellipse at 72% -14%, rgba(53, 201, 138, .055), transparent 42%), var(--dm-paper); }
[data-testid="stSidebar"] {
  background: var(--dm-sidebar);
  border-right: 1px solid var(--dm-line);
  width: 258px !important;
  flex: 0 0 258px !important;
  min-width: 258px;
  max-width: 258px;
}
[data-testid="stSidebar"] > div { padding: 1.1rem .85rem 1rem; }
[data-testid="stSidebar"] button { text-align: left; justify-content: flex-start; border-radius: 9px; }
[data-testid="stSidebar"] button[kind="secondary"] { border-color: transparent; background: transparent; color: var(--dm-muted); }
[data-testid="stSidebar"] button[kind="secondary"]:hover { background: rgba(22,49,39,.92); color: var(--dm-ink); }
[data-testid="stSidebar"] button[kind="primary"] { background: rgba(53,201,138,.13); color: var(--dm-mint); border: 1px solid rgba(112,230,172,.28); box-shadow: inset 2px 0 var(--dm-emerald); }
[data-testid="stSidebar"] button[kind="primary"]:hover { background: rgba(53,201,138,.2); border-color: var(--dm-mint); color: #FFFFFF; }
[data-testid="stSidebar"] button svg { color: currentColor; fill: currentColor; }
.block-container { padding: 1.2rem 2.5rem 3rem; max-width: 1500px; }
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 { color: #FFFFFF !important; letter-spacing: -.35px; }
h1 { font-size: 1.62rem !important; line-height: 1.25 !important; font-weight: 680 !important; margin: 0 !important; }
h2 { font-size: 1.22rem !important; font-weight: 650 !important; }
h3 { font-size: 1.02rem !important; font-weight: 650 !important; }
p, li, blockquote, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] { color: var(--dm-muted); }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, label,
[data-testid="stCheckbox"] label, [data-testid="stRadio"] label { color: var(--dm-muted) !important; }
.topbar { border-bottom: 1px solid var(--dm-line); padding: .1rem 0 .95rem; margin-bottom: 1.4rem; }
.page-kicker { color: var(--dm-label); font-size: .68rem; letter-spacing: .11em; text-transform: uppercase; font-weight: 700; margin-bottom: .3rem; }
.page-subtitle { color: var(--dm-muted); font-size: .88rem; margin-top: .3rem; }
.brand { display:flex; gap:11px; align-items:center; padding: .25rem .35rem 1.05rem; border-bottom:1px solid var(--dm-line); margin-bottom:1.1rem; }
.brand-mark { width:36px; height:36px; border-radius:11px; display:grid; place-items:center; color:#06150F; background:linear-gradient(140deg,#70E6AC,#35C98A 72%,#268B67); font-weight:750; font-size:1rem; box-shadow:0 0 18px rgba(53,201,138,.13); }
.brand-name { font-size:.96rem; font-weight:720; letter-spacing:-.025em; color:var(--dm-ink); }
.brand-caption { color:var(--dm-label); font-size:.61rem; letter-spacing:.105em; margin-top:2px; }
.side-label { font-size:.64rem; font-weight:700; letter-spacing:.1em; color:var(--dm-label); margin:.55rem .45rem .45rem; text-transform:uppercase; }
.workspace-status { margin:.8rem .15rem 0; padding:.85rem .8rem; background:var(--dm-card); border:1px solid var(--dm-line); border-radius:11px; }
.status-dot { display:inline-block; width:7px; height:7px; background:var(--dm-green); border-radius:50%; margin-right:7px; box-shadow:0 0 0 3px rgba(82,224,160,.12); }
.status-label { font-size:.75rem; color:var(--dm-ink); font-weight:650; }
.status-meta { font-size:.68rem; color:var(--dm-muted); margin-top:.45rem; line-height:1.5; }
.card-title { font-size:.94rem; font-weight:680; color:var(--dm-ink); margin-bottom:.12rem; }
.card-copy { font-size:.78rem; color:var(--dm-muted); line-height:1.55; }
.metric-card { background:var(--dm-card); border:1px solid var(--dm-line); border-radius:12px; padding:1rem 1.05rem .8rem; min-height:112px; box-shadow:var(--dm-shadow); transition:border-color .15s ease, transform .15s ease; }
.metric-card:hover { border-color:#35644F; transform:translateY(-1px); }
.metric-label { font-size:.7rem; font-weight:650; color:var(--dm-muted); }
.metric-icon { display:inline-flex; width:18px; height:18px; vertical-align:-4px; color:var(--dm-green); margin-right:5px; }
.metric-icon svg { width:18px; height:18px; fill:none; stroke:currentColor; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
.metric-value { font-size:1.65rem; line-height:1.2; font-weight:680; letter-spacing:-.045em; color:var(--dm-ink); margin-top:.48rem; }
.metric-note { font-size:.68rem; color:var(--dm-muted); margin-top:.2rem; }
.section-head { display:flex; align-items:center; justify-content:space-between; margin:1.55rem 0 .7rem; }
.section-head-title { font-size:.98rem; font-weight:680; color:var(--dm-ink); }
.section-head-note { font-size:.72rem; color:var(--dm-muted); }
.drop-hint { text-align:center; padding:.5rem 0 .12rem; }
.drop-icon { color:var(--dm-mint); font-size:1.25rem; font-weight:700; }
.drop-title { font-size:.93rem; color:var(--dm-ink); font-weight:670; margin-top:.35rem; }
.drop-copy { font-size:.75rem; color:var(--dm-muted); margin-top:.2rem; }
.doc-name { font-weight:650; color:var(--dm-ink); font-size:.82rem; }
.doc-meta { color:var(--dm-muted); font-size:.68rem; margin-top:.18rem; }
.doc-row-grid { display:grid; grid-template-columns:minmax(180px,2.1fr) repeat(4,minmax(64px,.72fr)); align-items:center; gap:.75rem; padding:.1rem 0 .75rem; }
.doc-primary { display:flex; align-items:center; gap:.72rem; min-width:0; }
.doc-file-icon { width:34px; height:38px; flex:0 0 34px; display:grid; place-items:center; border-radius:8px; color:var(--dm-mint); background:linear-gradient(145deg,rgba(53,201,138,.13),rgba(82,224,160,.035)); border:1px solid rgba(53,201,138,.17); }
.doc-file-icon svg { width:18px; height:18px; fill:none; stroke:currentColor; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
.doc-info { min-width:0; }
.doc-info .doc-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.doc-cell-label { display:block; color:var(--dm-label); font-size:.59rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.2rem; }
.doc-cell-value { color:var(--dm-ink); font-size:.73rem; }
.badge { display:inline-block; border-radius:100px; font-size:.65rem; font-weight:650; padding:.25rem .55rem; white-space:nowrap; }
.badge-ready { background:#153A2B; color:#8AF0C2; border:1px solid rgba(82,224,160,.2); }
.badge-failed { background:#3A2225; color:#FFB8B8; border:1px solid rgba(255,130,130,.22); }
.badge-pdf { background:#173329; color:#8AF0C2; border:1px solid rgba(53,201,138,.2); }
.empty-state { padding:2rem 1.2rem; border:1px dashed #315342; background:rgba(16,37,30,.72); border-radius:12px; text-align:center; }
.empty-title { font-size:.9rem; font-weight:650; color:var(--dm-ink); }
.empty-copy { font-size:.76rem; color:var(--dm-muted); margin-top:.35rem; }
.panel { background:var(--dm-card); border:1px solid var(--dm-line); border-radius:12px; padding:1rem 1.05rem; box-shadow:var(--dm-shadow); }
.source-card { padding:.8rem .85rem; background:var(--dm-card); border:1px solid var(--dm-line); border-left:2px solid rgba(53,201,138,.62); border-radius:9px; margin-bottom:.6rem; }
.source-head { font-size:.72rem; font-weight:680; color:var(--dm-ink); margin-bottom:.35rem; }
.source-body { font-size:.74rem; line-height:1.55; color:var(--dm-muted); }
.answer-label { font-size:.62rem; font-weight:750; letter-spacing:.09em; color:var(--dm-mint); margin-bottom:.35rem; }
div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
  border-radius:8px; font-weight:620; color:var(--dm-ink);
  transition:background .15s ease, border-color .15s ease, transform .15s ease;
}
div[data-testid="stButton"] button[kind="primary"], div[data-testid="stFormSubmitButton"] button[kind="primary"] {
  background:#35C98A; border-color:#35C98A; color:#06150F !important; box-shadow:0 3px 14px rgba(53,201,138,.12);
}
div[data-testid="stButton"] button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
  background:#70E6AC; border-color:#70E6AC; color:#06150F !important;
}
div[data-testid="stButton"] button[kind="secondary"], div[data-testid="stFormSubmitButton"] button[kind="secondary"] {
  border-color:var(--dm-line); background:var(--dm-card); color:var(--dm-ink);
}
div[data-testid="stButton"] button[kind="secondary"]:hover, div[data-testid="stFormSubmitButton"] button[kind="secondary"]:hover {
  border-color:var(--dm-mint); background:var(--dm-elevated); color:#FFFFFF;
}
div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover { transform:translateY(-1px); }
button:disabled, button[aria-disabled="true"] { opacity:.58; cursor:not-allowed !important; transform:none !important; box-shadow:none !important; }
button:disabled[kind="primary"] { background:#246D50 !important; border-color:#39735A !important; color:#D7E6DD !important; }
a, [data-testid="stMarkdownContainer"] a { color:#70E6AC; text-decoration-color:rgba(112,230,172,.55); }
a:hover, [data-testid="stMarkdownContainer"] a:hover { color:#FFFFFF; }

/* Inputs and menus */
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input, div[data-testid="stDateInput"] input,
div[data-testid="stTimeInput"] input, div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  border-radius:8px; border-color:var(--dm-line); background:var(--dm-elevated); color:#FFFFFF;
}
div[data-testid="stTextInput"] input::placeholder, div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stChatInput"] textarea::placeholder { color:var(--dm-label); opacity:1; }
input:disabled, textarea:disabled { color:var(--dm-muted) !important; -webkit-text-fill-color:var(--dm-muted); opacity:1; cursor:not-allowed; }
div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stNumberInput"] input:focus, div[data-testid="stDateInput"] input:focus {
  border-color:var(--dm-mint); box-shadow:0 0 0 1px rgba(112,230,172,.28);
}
div[data-baseweb="select"] input { color:#FFFFFF !important; caret-color:var(--dm-mint); }
div[data-baseweb="select"] [aria-selected="true"], [role="option"][aria-selected="true"] { background:rgba(53,201,138,.2) !important; color:#FFFFFF !important; }
div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] [role="option"], [role="listbox"] [role="option"] {
  background:var(--dm-elevated); color:#FFFFFF;
}
div[data-baseweb="popover"] [role="option"]:hover, [role="listbox"] [role="option"]:hover {
  background:#244437; color:#FFFFFF;
}
input[type="checkbox"], input[type="radio"] { accent-color:var(--dm-emerald); }
div[data-testid="stSlider"] [role="slider"] { background:var(--dm-emerald); border-color:var(--dm-emerald); }

/* Upload, tables, chat, tabs, disclosures, and status feedback */
div[data-testid="stFileUploader"] { border:1px dashed #547665; border-radius:10px; background:linear-gradient(115deg,rgba(53,201,138,.07),rgba(16,37,30,.94) 70%); padding:.35rem .55rem; transition:border-color .15s ease, background .15s ease; color:var(--dm-muted); }
div[data-testid="stFileUploader"]:hover, div[data-testid="stFileUploader"]:focus-within { border-color:var(--dm-mint); background:linear-gradient(115deg,rgba(53,201,138,.12),rgba(22,49,39,.96) 70%); }
div[data-testid="stFileUploader"] button { color:var(--dm-ink); border-color:var(--dm-line); background:var(--dm-elevated); }
div[data-testid="stFileUploader"] section { color:var(--dm-muted); }
div[data-testid="stVerticalBlockBorderWrapper"]:has(.drop-hint) { border-color:#547665; background:linear-gradient(115deg,rgba(53,201,138,.085),rgba(22,49,39,.96) 82%); box-shadow:0 8px 26px rgba(0,0,0,.2),0 0 24px rgba(53,201,138,.045); }
div[data-testid="stDataFrame"], div[data-testid="stTable"] { border:1px solid var(--dm-line); border-radius:10px; overflow:hidden; background:var(--dm-card); color:var(--dm-ink); }
div[data-testid="stDataFrame"] canvas { color-scheme:dark; }
table { color:var(--dm-ink); border-color:var(--dm-line); }
thead tr th { background:var(--dm-elevated); color:#FFFFFF; border-color:var(--dm-line); }
tbody tr td { background:var(--dm-card); color:var(--dm-muted); border-color:var(--dm-line); }
div[data-testid="stChatMessage"] { background:var(--dm-card); border:1px solid var(--dm-line); border-radius:10px; padding:.55rem .75rem; color:var(--dm-ink); }
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { color:var(--dm-ink); }
div[data-testid="stChatInput"] textarea { background:var(--dm-elevated); border-color:var(--dm-line); color:#FFFFFF; }
div[data-testid="stTabs"] [role="tablist"] { border-bottom:1px solid var(--dm-line); gap:.55rem; }
div[data-testid="stTabs"] button[role="tab"] { color:var(--dm-muted); border-radius:8px 8px 0 0; }
div[data-testid="stTabs"] button[role="tab"]:hover { color:#FFFFFF; background:rgba(22,49,39,.8); }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#70E6AC; border-bottom:2px solid #35C98A; }
div[data-testid="stExpander"] { border:1px solid var(--dm-line); border-radius:10px; background:var(--dm-card); }
div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary p { color:var(--dm-ink) !important; }
div[data-testid="stExpander"] summary:hover { color:#70E6AC !important; }
div[data-testid="stAlert"], [data-testid="stToast"] { border-radius:10px; color:#FFFFFF; background:var(--dm-elevated); border:1px solid var(--dm-line); }
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stStatus"] [data-testid="stMarkdownContainer"] p { color:inherit; }
div[data-testid="stStatus"] { color:var(--dm-ink); background:var(--dm-card); border-color:var(--dm-line); }
div[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#35C98A,#70E6AC); }
div[role="dialog"], [data-testid="stDialog"] > div { background:var(--dm-card); color:var(--dm-ink); border-color:var(--dm-line); }
button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible,
[role="combobox"]:focus-visible, [role="option"]:focus-visible {
  outline:2px solid #70E6AC !important; outline-offset:2px;
}
hr { border-color:var(--dm-line) !important; }
@media(max-width:850px) {
  .block-container { padding:1rem 1rem 2rem; }
  [data-testid="stSidebar"] { width:220px !important; flex-basis:220px !important; min-width:220px; max-width:220px; }
  .doc-row-grid { grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:.65rem; }
  .doc-primary { grid-column:1 / -1; }
}
</style>
""",
    unsafe_allow_html=True,
)


def set_page(page: str) -> None:
    st.session_state["dm_page"] = page


def section_heading(title: str, note: str | None = None) -> None:
    st.markdown(
        f'<div class="section-head"><div class="section-head-title">{title}</div>'
        f'<div class="section-head-note">{note or ""}</div></div>',
        unsafe_allow_html=True,
    )


def metric(label: str, value: str | int, note: str, icon: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label"><span class="metric-icon">{LINE_ICONS[icon]}</span>{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def matching_documents(docs: list[dict[str, Any]], query: str, status: str = "All statuses") -> list[dict[str, Any]]:
    normalized = query.strip().casefold()
    return [
        doc for doc in docs
        if (not normalized or normalized in doc["name"].casefold())
        and (status == "All statuses" or doc["status"].casefold() == status.casefold())
    ]


def import_area(key: str = "overview_upload") -> None:
    st.markdown('<div class="drop-hint"><div class="drop-icon">↥</div><div class="drop-title">Bring your documents into focus</div><div class="drop-copy">Drop a PDF here or browse from your device · up to 50 MB per file</div></div>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=key,
        help="Text-based PDFs are supported. Scanned PDFs require OCR before import.",
        label_visibility="collapsed",
    )
    if files and st.button("Process selected PDFs", type="primary", icon=":material/upload_file:", key=f"process_{key}"):
        progress = st.progress(0, text="Preparing your documents…")
        for index, uploaded in enumerate(files):
            try:
                ingest(
                    STORE,
                    uploaded.name,
                    uploaded.getvalue(),
                    int(st.session_state.get("chunk_size", 900)),
                    int(st.session_state.get("overlap", 120)),
                )
                st.success(f"{uploaded.name} is ready for analysis.")
            except Exception as exc:
                st.error(f"{uploaded.name}: {exc}")
            progress.progress((index + 1) / len(files), text=f"Processed {index + 1} of {len(files)}")
        st.session_state["show_import"] = False
        st.rerun()


def show_document_rows(docs: list[dict[str, Any]], prefix: str, allow_delete: bool = False) -> None:
    if not docs:
        st.markdown(
            '<div class="empty-state"><div class="empty-title">No matching documents</div>'
            '<div class="empty-copy">Try another search or clear the status filter.</div></div>',
            unsafe_allow_html=True,
        )
        return

    for doc in docs:
        with st.container(border=True):
            safe_name = html.escape(doc["name"])
            if doc["status"] == "processed":
                badge = '<span class="badge badge-ready">Ready</span>'
            else:
                badge = '<span class="badge badge-failed">Needs attention</span>'
            st.markdown(
                f'<div class="doc-row-grid"><div class="doc-primary"><span class="doc-file-icon">{LINE_ICONS["document"]}</span>'
                f'<div class="doc-info"><div class="doc-name">{safe_name}</div><div class="doc-meta">{doc["chunk_count"]} indexed passages</div></div></div>'
                f'<div><span class="doc-cell-label">Type</span><span class="badge badge-pdf">PDF</span></div>'
                f'<div><span class="doc-cell-label">Pages</span><span class="doc-cell-value">{doc["page_count"]}</span></div>'
                f'<div><span class="doc-cell-label">Status</span>{badge}</div>'
                f'<div><span class="doc-cell-label">Added</span><span class="doc-cell-value">{doc["created_at"][:10]}</span></div></div>',
                unsafe_allow_html=True,
            )
            open_col, actions_col = st.columns([1, .22], vertical_alignment="center")
            if open_col.button("Open document", icon=":material/visibility:", key=f"open_{prefix}_{doc['id']}", use_container_width=True):
                st.session_state["view_doc"] = doc["id"]
                set_page("My Documents")
                st.rerun()
            with actions_col:
                with st.popover("", icon=":material/more_horiz:", help="Document actions", use_container_width=True):
                    if st.button("Analyze document", icon=":material/auto_awesome:", key=f"analyze_{prefix}_{doc['id']}", disabled=doc["status"] != "processed", use_container_width=True):
                        st.session_state["analysis_doc"] = doc["id"]
                        set_page("AI Analysis")
                        st.rerun()
                    if allow_delete and st.button("Remove document", icon=":material/delete:", key=f"remove_{doc['id']}", use_container_width=True):
                        st.session_state["pending_delete_doc"] = doc["id"]
                        st.rerun()
        if st.session_state.get("pending_delete_doc") == doc["id"]:
            st.warning(f'Remove “{doc["name"]}” and its indexed chunks from this local workspace?')
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("Confirm removal", type="primary", key=f"confirm_remove_{doc['id']}"):
                STORE.delete_document(doc["id"])
                st.session_state.pop("pending_delete_doc", None)
                if st.session_state.get("view_doc") == doc["id"]:
                    st.session_state.pop("view_doc", None)
                st.rerun()
            if cancel_col.button("Cancel", key=f"cancel_remove_{doc['id']}"):
                st.session_state.pop("pending_delete_doc", None)
                st.rerun()
        if doc["status"] == "failed" and doc.get("error"):
            st.caption(f'Processing detail: {doc["error"]}')


def sidebar(stats: dict[str, int]) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">D</div><div><div class="brand-name">DocuMind AI</div>'
            '<div class="brand-caption">PRIVATE DOCUMENT WORKSPACE</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)
        active = st.session_state.get("dm_page", "Overview")
        for label, (icon, _, _) in PAGES.items():
            if st.button(
                label,
                key=f"nav_{label}",
                icon=f":material/{icon}:",
                type="primary" if active == label else "secondary",
                use_container_width=True,
            ):
                set_page(label)
                st.rerun()
        st.markdown('<div class="side-label" style="margin-top:1.2rem">Your workspace</div>', unsafe_allow_html=True)
        db_bytes = STORE.path.stat().st_size if STORE.path.exists() else 0
        st.markdown(
            f'<div class="workspace-status"><div class="status-label"><span class="status-dot"></span>Local and private</div>'
            f'<div class="status-meta">{stats["processed"]} ready documents<br>{format_bytes(db_bytes)} used on this device</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
        st.caption("Documents and search run locally. Model downloads happen only when needed.")


def topbar(page: str) -> str:
    icon, title, subtitle = PAGES[page]
    st.markdown(f'<div class="page-kicker">DocuMind / {page}</div><h1>{title}</h1><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    search_col, import_col, settings_col = st.columns([1.0, .23, .12], vertical_alignment="center")
    with search_col:
        search = st.text_input("Search documents", placeholder="Search your library…", label_visibility="collapsed", key="global_search")
    with import_col:
        if st.button("Import", type="primary", icon=":material/add:", use_container_width=True, key="top_import"):
            st.session_state["show_import"] = True
            set_page("My Documents")
            st.rerun()
    with settings_col:
        if st.button("", icon=":material/settings:", help="Open settings", key="top_settings", use_container_width=True):
            set_page("Settings")
            st.rerun()
    st.markdown("<div style='height:.2rem;border-bottom:1px solid var(--dm-line);margin-bottom:1rem'></div>", unsafe_allow_html=True)
    return search


def overview(docs: list[dict[str, Any]], query: str) -> None:
    stats = STORE.stats()
    db_size = STORE.path.stat().st_size if STORE.path.exists() else 0

    with st.container(border=True):
        import_area("overview_upload")

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1: metric("Total documents", stats["documents"], "In your local library", "document")
    with k2: metric("Processed", stats["processed"], "Ready to explore", "check")
    with k3: metric("Indexed chunks", stats["chunks"], "Searchable passages", "chunks")
    with k4: metric("Local storage", format_bytes(db_size), "SQLite metadata and text", "storage")

    left, right = st.columns([1.65, .9], gap="large")
    with left:
        section_heading("Search results" if query else "Recent documents", "Matching files in your library" if query else "Latest files in your library")
        recent = matching_documents(docs, query)[:5]
        show_document_rows(recent, "overview")
        if docs and st.button("View all documents", icon=":material/arrow_forward:", key="view_all_docs"):
            set_page("My Documents")
            st.rerun()
    with right:
        section_heading("Recent activity", "From your local workspace")
        activity = STORE.activity(6)
        if activity:
            for item in activity:
                with st.container(border=True):
                    st.markdown(f"**{item['message']}**")
                    st.caption(item["created_at"][:16].replace("T", " "))
        else:
            empty("Nothing here yet", "Upload a document to start your activity timeline.")


def empty(title: str, copy: str) -> None:
    st.markdown(f'<div class="empty-state"><div class="empty-title">{title}</div><div class="empty-copy">{copy}</div></div>', unsafe_allow_html=True)


def documents_page(docs: list[dict[str, Any]], query: str) -> None:
    search_col, filter_col = st.columns([1.8, .8], vertical_alignment="bottom")
    with search_col:
        library_query = st.text_input("Filter library", value=query, placeholder="Search by document name…", key="library_search")
    with filter_col:
        status_filter = st.selectbox("Processing status", ["All statuses", "Processed", "Failed"], label_visibility="collapsed", key="library_status")
    visible = matching_documents(docs, library_query, status_filter)
    st.caption(f"Showing {len(visible)} of {len(docs)} documents")

    if st.session_state.get("show_import", False):
        with st.container(border=True):
            import_area("library_upload")
    elif st.button("Import documents", icon=":material/upload_file:", key="library_import_action"):
        st.session_state["show_import"] = True
        st.rerun()

    show_document_rows(visible, "library", allow_delete=True)
    view_id = st.session_state.get("view_doc")
    doc = STORE.document(view_id) if view_id else None
    if doc:
        section_heading(f"Extracted text · {doc['name']}", f"{doc['page_count']} source pages")
        with st.container(border=True):
            st.text_area("Page-labeled source text", STORE.page_text(doc["id"]), height=310, disabled=True, label_visibility="collapsed")
            if st.button("Close preview", icon=":material/close:", key="close_preview"):
                st.session_state.pop("view_doc", None)
                st.rerun()


def analysis_page(docs: list[dict[str, Any]]) -> None:
    available = [doc for doc in docs if doc["status"] == "processed"]
    if not available:
        empty("No analyzed documents yet", "Import a text-based PDF to generate a source-grounded analysis.")
        return
    ids = [doc["id"] for doc in available]
    preferred = st.session_state.get("analysis_doc")
    default_index = ids.index(preferred) if preferred in ids else 0
    chosen = st.selectbox("Document", available, index=default_index, format_func=lambda item: item["name"], label_visibility="collapsed")
    action_col, info_col = st.columns([.28, .72], vertical_alignment="center")
    with action_col:
        generate = st.button("Generate analysis", type="primary", icon=":material/auto_awesome:", use_container_width=True, key="generate_analysis")
    with info_col:
        provider = st.session_state.get("ai_provider", "ollama")
        caption = "Uses extractive evidence by default. Ollama can add a local generated overview when available."
        if provider == "gemini":
            caption = "Gemini cloud mode is selected. Source text is sent only after you enable consent in Settings."
        st.caption(caption)
    if generate:
        if provider == "gemini" and not st.session_state.get("gemini_cloud_consent", False):
            st.warning("Enable Gemini cloud consent in Settings before sending document text for analysis. No document text was sent.")
        elif provider == "gemini" and not get_api_key():
            st.error("Gemini API key not found. Configure GEMINI_API_KEY in the environment or Streamlit secrets. No document text was sent.")
        else:
            if provider == "gemini":
                st.info("This analysis sends the selected document’s extracted text (up to 12,000 characters) and the summary prompt to Google Gemini. The PDF file itself stays local.")
            llm = GeminiClient(model=st.session_state.get("gemini_model")) if provider == "gemini" else OllamaClient()
            try:
                with st.spinner("Reading and organizing source evidence…"):
                    result = summarize(STORE.page_text(chosen["id"]), llm)
                st.session_state["analysis_result"] = (chosen["id"], result)
            except GeminiError as exc:
                st.error(str(exc))
    cached = st.session_state.get("analysis_result")
    if not cached or cached[0] != chosen["id"]:
        empty("Ready when you are", "Generate an analysis to see the summary, key points, dates, and source pages.")
        return

    result = cached[1]
    method = result.get("method", "extractive")
    st.caption(f"Analysis method: {method}. Values come from extracted source text; review the page references before relying on them.")
    tabs = st.tabs(["Summary", "Key Points", "Extracted Information", "Sources"])
    with tabs[0]:
        with st.container(border=True):
            if result.get("generated"):
                st.markdown("#### Local model overview")
                st.write(result["generated"])
                st.caption("Model-generated interpretation. Verify each claim against the source passages.")
        with st.container(border=True):
            st.markdown("#### Overview")
            st.write(result["overview"])
            st.caption("Extractive summary assembled from source sentences.")
    with tabs[1]:
        points = result.get("key_points", [])
        if points:
            for index, point in enumerate(points):
                page = result.get("key_point_pages", [])[index] if index < len(result.get("key_point_pages", [])) else None
                with st.container(border=True):
                    st.write(point)
                    if page is not None:
                        st.caption(f"Source · {chosen['name']} · page {page}")
        else:
            empty("No supported key points found", "The extracted text did not contain enough complete sentences for a useful summary.")
        actions = result.get("action_items", [])
        if actions:
            section_heading("Potential action items", "Heuristic matches from source language")
            for action in actions:
                st.markdown(f"- {action}")
    with tabs[2]:
        x, y, z = st.columns(3)
        with x:
            with st.container(border=True):
                st.markdown("#### Dates")
                st.write(", ".join(result.get("dates", [])) or "None found")
        with y:
            with st.container(border=True):
                st.markdown("#### Numbers")
                st.write(", ".join(result.get("numbers", [])) or "None reliably identified")
        with z:
            with st.container(border=True):
                st.markdown("#### Named entities")
                st.write(", ".join(result.get("entities", [])) or "None reliably identified")
        st.caption("Dates, numbers, and names are extracted using local text patterns. They are not independently verified or classified.")
    with tabs[3]:
        with st.container(border=True):
            for line in STORE.page_text(chosen["id"]).split("\n\n"):
                if line.strip():
                    st.markdown(f"**{line.splitlines()[0]}**")
                    st.write("\n".join(line.splitlines()[1:]))


def answer_question(question: str, selected: list[dict[str, Any]]) -> None:
    st.session_state.setdefault("conversation", [])
    st.session_state["conversation"].append({"role": "user", "content": question})
    try:
        chunks = STORE.chunks([doc["id"] for doc in selected])
        matches = Retriever(Encoder()).search(question, chunks, int(st.session_state.get("top_k", 5)))
        sources = [
            {"name": match["name"], "page": match["page"], "text": match["text"], "score": match["score"]}
            for match in matches
        ]
        evidence = "\n\n".join(
            f"[Source {index + 1}: {source['name']}, page {source['page']}]\n{source['text']}"
            for index, source in enumerate(sources)
        )
        provider = st.session_state.get("ai_provider", "ollama")
        if provider == "gemini" and not st.session_state.get("gemini_cloud_consent", False):
            answer = "Gemini cloud mode is selected, but cloud-data consent is not enabled. No document text was sent. Enable consent in Settings or switch to local mode."
        elif matches and provider == "gemini":
            llm = GeminiClient(model=st.session_state.get("gemini_model"))
            try:
                answer = llm.generate(
                    "Answer only from the retrieved evidence. If evidence is insufficient, say so plainly. "
                    "Cite evidence only with the exact labels [Source 1], [Source 2], etc. shown below. "
                    "Do not invent document names, page numbers, or source labels.\n\n"
                    f"QUESTION: {question}\n\nSOURCES:\n{evidence}"
                )
                valid_labels = {f"Source {index + 1}" for index in range(len(sources))}
                for label in re.findall(r"Source\s+\d+", answer):
                    if label not in valid_labels:
                        answer = answer.replace(label, "")
            except GeminiError as exc:
                answer = str(exc)
        elif matches and OllamaClient().available():
            answer = OllamaClient().generate(
                "Answer only from the sources below. If evidence is insufficient, say so. "
                "Cite only the source labels shown. Never create a page number.\n\n"
                f"QUESTION: {question}\n\nSOURCES:\n{evidence}"
            )
            valid_labels = {f"Source {index + 1}" for index in range(len(sources))}
            for label in re.findall(r"Source\s+\d+", answer):
                if label not in valid_labels:
                    answer = answer.replace(label, "")
        elif matches:
            answer = "I found relevant passages, but a generative answer is unavailable because the configured Ollama model is not running. Review the cited evidence on the right."
        else:
            answer = "The selected documents do not contain enough indexed evidence to answer this question."
        st.session_state["conversation"].append({"role": "assistant", "content": answer, "sources": sources})
    except Exception as exc:
        st.session_state["conversation"].append({
            "role": "assistant",
            "content": f"Local semantic search could not start: {exc}. Install the project dependencies and allow the embedding model to download on first use.",
            "sources": [],
        })


def ask_page(docs: list[dict[str, Any]]) -> None:
    available = [doc for doc in docs if doc["status"] == "processed"]
    if not available:
        empty("Your research space is empty", "Import and process a text-based PDF to ask questions with page citations.")
        return
    if st.session_state.get("ai_provider") == "gemini" and st.session_state.get("gemini_cloud_consent", False):
        st.info("Cloud AI is enabled. Your question and retrieved passages will be sent to Google Gemini; source retrieval and citations remain local.")
    selected = st.multiselect("Search across", available, default=available, format_func=lambda doc: doc["name"], key="qa_documents")
    chat_col, sources_col = st.columns([1.15, .85], gap="large")
    with chat_col:
        section_heading("Conversation", "Answers stay grounded in retrieved passages")
        with st.container(height=465, border=True):
            turns = st.session_state.get("conversation", [])
            if not turns:
                st.markdown("**Start with a question about your library.**")
                st.caption("Choose a starting point or write your own research question below.")
            for turn in turns:
                with st.chat_message(turn["role"]):
                    if turn["role"] == "assistant":
                        st.markdown('<div class="answer-label">DOCUMIND AI · GROUNDED RESPONSE</div>', unsafe_allow_html=True)
                    st.markdown(turn["content"])

        if not st.session_state.get("conversation"):
            suggestion_a, suggestion_b = st.columns(2)
            examples = ["What dates or deadlines are mentioned?", "What actions are required?"]
            if suggestion_a.button(examples[0], key="suggest_deadlines", use_container_width=True):
                st.session_state["qa_question"] = examples[0]
                st.rerun()
            if suggestion_b.button(examples[1], key="suggest_actions", use_container_width=True):
                st.session_state["qa_question"] = examples[1]
                st.rerun()
        with st.form("document_question_form", clear_on_submit=True, border=False):
            question = st.text_area(
                "Your question",
                key="qa_question",
                placeholder="Ask a question about the selected documents…",
                height=78,
                label_visibility="collapsed",
                disabled=not selected,
            )
            submitted = st.form_submit_button(
                "Ask your documents",
                type="primary",
                icon=":material/arrow_upward:",
                disabled=not selected,
                use_container_width=True,
            )
        if submitted and question.strip():
            with st.spinner("Searching your local document index…"):
                answer_question(question.strip(), selected)
            st.rerun()
    with sources_col:
        section_heading("Sources", "Retrieved passages with actual page references")
        assistant_turns = [turn for turn in st.session_state.get("conversation", []) if turn["role"] == "assistant" and turn.get("sources")]
        if assistant_turns:
            options = list(range(len(assistant_turns)))
            selected_turn = st.selectbox("Evidence for answer", options, index=len(options) - 1, format_func=lambda index: f"Question {index + 1}", label_visibility="collapsed")
            sources = assistant_turns[selected_turn]["sources"]
            for source in sources:
                with st.container(border=True):
                    st.markdown(f"**{source['name']}** · page **{source['page']}**")
                    st.caption(f"Cosine relevance · {source['score']:.2f}")
                    st.write(source["text"])
        else:
            empty("Evidence will appear here", "Each answer shows the retrieved text and its document page.")
        if st.session_state.get("conversation") and st.button("Clear this session’s conversation", icon=":material/delete_sweep:", key="clear_chat"):
            st.session_state["conversation"] = []
            st.rerun()


def evaluation_page() -> None:
    with st.container(border=True):
        st.markdown("### Evaluation is not implemented yet")
        st.write("This workspace does not currently store a benchmark set, human ratings, or answer-quality scores. No evaluation results are shown here.")
        st.caption("A future evaluation workflow can compare retrieved passages and generated answers against questions you provide, entirely on this device.")
    st.markdown(" ")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("#### Planned evaluation inputs")
            st.write("Question, expected evidence, selected documents, and optional reviewer notes.")
    with b:
        with st.container(border=True):
            st.markdown("#### Planned reporting")
            st.write("Citation coverage, retrieval relevance, and groundedness review. These metrics are not available in this release.")


def settings_page() -> None:
    encoder = Encoder()
    _, embedding_status = encoder.status()
    ollama = OllamaClient()
    models = ollama.models()
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown("#### Semantic search")
            st.write(f"**Embedding model**  \n`{MODEL_NAME}`")
            st.write(f"**Status**  \n{embedding_status}")
            st.caption("Runs on CPU. The model is fetched from Hugging Face the first time you index or search.")
    with b:
        with st.container(border=True):
            st.markdown("#### Local language model")
            st.write(f"**Ollama**  \n{'Available' if models else 'Not detected'}")
            st.write(f"**Configured model**  \n`{ollama.model}`")
            st.caption("Optional local generation. Ollama processes prompts on this device.")

    section_heading("AI provider and privacy", "Cloud use is off until you choose Gemini and enable consent")
    provider = st.radio(
        "Generation provider",
        ["ollama", "gemini"],
        format_func=lambda item: "Local · Ollama" if item == "ollama" else "Cloud · Google Gemini",
        horizontal=True,
        key="ai_provider",
    )
    if provider == "gemini":
        st.warning("Gemini sends the question and retrieved passages for Q&A, or the selected document’s extracted text for summaries, to Google for generation. PDF files, the local database, embeddings, and chat history are not uploaded. Review your organization’s data policy before enabling cloud mode.")
        st.checkbox(
            "I understand document text for summaries, or retrieved passages and my question for Q&A, will be sent to Google Gemini when I request generation.",
            key="gemini_cloud_consent",
        )
        api_key_present = bool(get_api_key())
        if not api_key_present:
            st.info("Gemini API key not found. Set GEMINI_API_KEY in the environment or add it to .streamlit/secrets.toml. The key is never shown in this app.")
        else:
            st.caption("The key is configured. Model discovery contacts Google’s model-list endpoint but does not send document contents.")
            if st.button("Discover available Gemini models", icon=":material/travel_explore:", key="discover_gemini_models"):
                try:
                    with st.spinner("Checking models available to this API key…"):
                        st.session_state["gemini_available_models"] = GeminiClient().models()
                    if not st.session_state["gemini_available_models"]:
                        st.info("This API key has no models available for text generation.")
                except GeminiError as exc:
                    st.error(str(exc))
            models = st.session_state.get("gemini_available_models", [])
            if models:
                previous_model = st.session_state.get("gemini_model")
                model_index = models.index(previous_model) if previous_model in models else 0
                st.selectbox("Account-available text model", models, index=model_index, key="gemini_model")
    else:
        st.session_state["gemini_cloud_consent"] = False
        st.caption("Local Ollama remains the selected generation provider. Retrieved document text is not sent to Gemini.")

    section_heading("Retrieval preferences", "Applied to new processing and searches")
    left, right = st.columns(2)
    with left:
        st.session_state["chunk_size"] = st.slider("Chunk size (words)", 200, 1600, int(st.session_state.get("chunk_size", 900)), 100)
        st.session_state["overlap"] = st.slider("Chunk overlap (words)", 0, min(400, st.session_state["chunk_size"] - 100), int(st.session_state.get("overlap", 120)), 20)
    with right:
        st.session_state["top_k"] = st.slider("Passages per answer", 1, 10, int(st.session_state.get("top_k", 5)))
        st.markdown("**Local storage**")
        st.code(str(DATA), language=None)

    st.caption("PDF extraction, SQLite, embeddings, and retrieval remain local. Gemini receives extracted text for requested summaries or retrieved passages with the question for Q&A, only after explicit selection and consent.")
    section_heading("Index maintenance", "SQLite remains the source of truth")
    if st.button("Rebuild saved embeddings", icon=":material/refresh:", key="rebuild_index"):
        rows = STORE.chunks()
        if rows:
            try:
                progress = st.progress(0, text="Loading the CPU embedding model…")
                vectors = encoder.encode([row["text"] for row in rows])
                for index, (row, vector) in enumerate(zip(rows, vectors)):
                    STORE.set_embedding(row["id"], vector.tobytes())
                    progress.progress((index + 1) / len(rows), text=f"Saved vectors for {index + 1} of {len(rows)} passages")
                st.success("Saved embeddings to the existing SQLite database. A FAISS CPU index is rebuilt from these vectors during search.")
            except Exception as exc:
                st.error(f"Embedding rebuild failed: {exc}")
        else:
            st.info("There are no processed passages to rebuild.")

    section_heading("Delete workspace data", "This action removes local document data")
    st.warning("This permanently removes document metadata, extracted text, chunks, embeddings, and activity from the DocuMind database.")
    confirmed = st.checkbox("I understand and want to delete all DocuMind data", key="confirm_delete_all")
    if st.button("Delete all data", type="secondary", icon=":material/delete_forever:", disabled=not confirmed, key="delete_all"):
        STORE.clear()
        st.session_state["conversation"] = []
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("view_doc", None)
        st.success("Local DocuMind data has been deleted.")
        st.rerun()


def run() -> None:
    st.session_state.setdefault("dm_page", "Overview")
    st.session_state.setdefault("conversation", [])
    docs = STORE.documents()
    stats = STORE.stats()
    sidebar(stats)
    page = st.session_state.get("dm_page", "Overview")
    global_query = topbar(page)

    if global_query.strip() and page not in {"Overview", "My Documents"}:
        matches = matching_documents(docs, global_query)
        section_heading("Library search", f"{len(matches)} matching document(s)")
        show_document_rows(matches[:6], "global")

    if page == "Overview":
        overview(docs, global_query)
    elif page == "My Documents":
        documents_page(docs, global_query)
    elif page == "AI Analysis":
        analysis_page(docs)
    elif page == "Ask Your Documents":
        ask_page(docs)
    elif page == "Evaluation Lab":
        evaluation_page()
    else:
        settings_page()


if __name__ == "__main__":
    run()

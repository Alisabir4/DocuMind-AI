from pathlib import Path
import sqlite3
import pymupdf as fitz
import numpy as np
import pytest
from src.database.store import Store
from src.document_processing.pdf import DocumentError, PageText, chunk_pages, extract_pdf
from src.services.documents import ingest
from src.llm.summarize import summarize
from src.retrieval.index import Retriever

def pdf_bytes(text="A useful agreement begins on 2025-04-10. The team must submit the report by Friday."):
    doc=fitz.open(); page=doc.new_page(); page.insert_text((72,72),text); out=doc.tobytes(); doc.close(); return out

def test_pdf_extraction_page_numbering():
    doc=fitz.open(); doc.new_page().insert_text((72,72),"First page text"); doc.new_page().insert_text((72,72),"Second page text"); data=doc.tobytes(); doc.close()
    pages=extract_pdf(data); assert [p.page for p in pages]==[1,2]; assert "Second" in pages[1].text

def test_chunk_overlap():
    pages=[PageText(3," ".join(f"w{i}" for i in range(20)))]
    chunks=chunk_pages(pages,100,10) if False else chunk_pages(pages,100,20) # valid minimum chunk, short page
    assert chunks[0].page==3
    many=chunk_pages([PageText(1," ".join(str(i) for i in range(240)))],100,20)
    assert len(many)==3 and many[0].text.split()[-20:]==many[1].text.split()[:20]

def test_empty_and_invalid_pdf():
    with pytest.raises(DocumentError): extract_pdf(b"")
    with pytest.raises(DocumentError): extract_pdf(b"not a pdf")
    d=fitz.open(); d.new_page(); data=d.tobytes(); d.close()
    with pytest.raises(DocumentError, match="scanned"): extract_pdf(data)

def test_db_duplicate_and_delete_cleanup(tmp_path):
    store=Store(tmp_path/"state.db"); data=pdf_bytes(); docid=ingest(store,"test.pdf",data)
    assert store.stats()=={"documents":1,"processed":1,"chunks":1}
    with pytest.raises(ValueError,match="already"): ingest(store,"copy.pdf",data)
    store.delete_document(docid); assert store.stats()["chunks"]==0 and store.documents()==[]

class FakeEncoder:
    def encode(self, texts):
        return np.array([[1.,0.] if "contract" in t.lower() else [0.,1.] for t in texts],dtype="float32")

def test_index_retrieval_and_citation_page():
    chunks=[{"id":1,"name":"terms.pdf","page":7,"text":"contract renewal terms"},{"id":2,"name":"other.pdf","page":2,"text":"weather report"}]
    got=Retriever(FakeEncoder()).search("contract",chunks,1)
    assert got[0]["name"]=="terms.pdf" and got[0]["page"]==7

def test_extractive_summary_without_generator():
    result=summarize("The agreement starts in 2025. The parties must respond by 2025-09-01. This is a sufficiently long sentence about the main provisions of the agreement.")
    assert result["method"]=="extractive" and "2025" in result["dates"]

def test_page_labels_persist(tmp_path):
    store=Store(tmp_path/"db.sqlite"); doc=fitz.open(); doc.new_page().insert_text((72,72),"Evidence on this page."); doc.new_page().insert_text((72,72),"More evidence on page two."); data=doc.tobytes(); doc.close()
    did=ingest(store,"pages.pdf",data); assert "[Page 2]" in store.page_text(did)

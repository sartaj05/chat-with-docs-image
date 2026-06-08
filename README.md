# Chat With Docs Image - Phase 7

Local Streamlit document assistant with PDF, scanned PDF OCR, image OCR, DOCX, Gemini Q&A, FAISS + BM25 hybrid search, multi-collection search, and local ZIP backup/import/export.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example`, then run:

```bash
streamlit run app.py
```

Generated local folders are ignored by Git: `collections/`, `backups/`, `faiss_index/`.

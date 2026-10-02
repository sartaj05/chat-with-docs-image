# Chat With Docs Image

Local Streamlit document assistant with document comparison, OCR, multi-collection hybrid retrieval, source navigation, file management, chat export, optional ML reranking, Ollama offline mode, streaming answers, and local multi-user workspaces.

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

Gemini is the default provider. For offline mode, install Ollama, pull a chat model such as `llama3.2` and an embedding model such as `nomic-embed-text`, then choose `Ollama Offline` inside the app. ML reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2` and is enabled from the AI settings panel.

Implemented feature areas:

- Compare two documents with an AI change summary and unified diff.
- Jump from each source result to its highlighted preview.
- Filter by file type, filename, page, category, tag, and minimum relevance.
- Manage individual indexed files: preview, rename, delete, reindex, tag, and categorize.
- Export collection chat history as TXT, Markdown, or DOCX.
- Optional cross-encoder reranking.
- Gemini Cloud or Ollama Offline provider selection.
- Streaming answers with `st.write_stream`.
- Local account registration, login, logout, isolated workspaces, and persisted chat history.

Generated local folders are ignored by Git: `collections/`, `backups/`, `faiss_index/`, `workspaces/`, and `users.json`.

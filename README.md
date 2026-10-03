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

Upload safety limits are configurable through `.env`: uploads default to 50 MB, collection ZIPs to 100 files and 250 MB uncompressed, PDFs to 200 pages, and document processing to 180 seconds. Streamlit's upload cap is set to 50 MB in `.streamlit/config.toml`.

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
- Optional layout-aware OCR that preserves detected block, paragraph, and line order.
- Local account registration, login, logout, isolated workspaces, and persisted chat history.

Authentication uses SQLite with PBKDF2 password hashes. For production OIDC, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, replace its client values, and register both your local and deployed callback URLs with the identity provider. The app exposes an OIDC sign-in path when Streamlit authentication is configured.

Collection imports accept JSON documents and metadata only. Any supplied FAISS index is ignored and rebuilt locally; the app loads only native FAISS index binaries and reconstructs the document store from `documents.json`.

CI runs compilation and pytest through `.github/workflows/ci.yml`. A Docker-based Render deployment template is provided in `render.yaml` with a persistent `/app/data` disk. The app stores JSON, SQLite, FAISS, backups, and workspaces under `APP_DATA_DIR`; Docker deployments can mount `/app/data` as a volume.

External integration tests are opt-in: set `RUN_EXTERNAL_TESTS=1` with `GOOGLE_API_KEY` for Gemini, or set `RUN_OLLAMA_TESTS=1` with a running local Ollama service. Plain `pytest` skips these network-dependent tests.

PDF extraction uses the maintained `pypdf` package.

Generated local folders are ignored by Git: `collections/`, `backups/`, `faiss_index/`, `workspaces/`, and `users.json`.

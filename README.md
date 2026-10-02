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
- Optional layout-aware OCR that preserves detected block, paragraph, and line order.
- Local account registration, login, logout, isolated workspaces, and persisted chat history.

Authentication uses SQLite with PBKDF2 password hashes. For production OIDC, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, replace its client values, and register both your local and deployed callback URLs with the identity provider. The app exposes an OIDC sign-in path when Streamlit authentication is configured.

CI runs compilation and pytest through `.github/workflows/ci.yml`. A Docker-based Render deployment template is provided in `render.yaml`. Local JSON/FAISS/workspace data needs persistent storage in production.

Generated local folders are ignored by Git: `collections/`, `backups/`, `faiss_index/`, `workspaces/`, and `users.json`.

import importlib.util
import io
import json
import os
import zipfile

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from PIL import Image

import app


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


def test_password_hash_round_trip():
    stored = app.hash_password("correct horse battery staple")
    assert app.verify_password("correct horse battery staple", stored)
    assert not app.verify_password("wrong password", stored)


def test_document_diff_reports_added_text():
    diff = app.build_document_diff("payment: 10", "payment: 20")
    assert "-payment: 10" in diff
    assert "+payment: 20" in diff


def test_citation_documents_keep_source_metadata():
    source = Document(page_content="Eligibility is required.", metadata={"file_name": "rules.txt", "file_type": "TXT", "chunk": 1})
    cited = app.prepare_citation_documents([source])[0]
    assert cited.page_content.startswith("[1] rules.txt | TXT | Chunk 1")
    assert cited.metadata["file_name"] == "rules.txt"


def test_optional_ai_dependencies_are_installed():
    assert importlib.util.find_spec("langchain_ollama") is not None
    assert importlib.util.find_spec("sentence_transformers") is not None


def test_visual_comparison_detects_changed_pixels():
    original = Image.new("RGB", (10, 10), "white")
    updated = Image.new("RGB", (10, 10), "white")
    updated.putpixel((5, 5), (0, 0, 0))
    result = app.build_visual_comparison([original], [updated])
    assert result[0][1] is not None
    assert result[0][2] > 0


def test_authenticated_workspace_paths_are_isolated_with_absolute_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(app, "WORKSPACES_DIR", str(data_dir / "workspaces"))
    monkeypatch.setattr(app, "COLLECTIONS_DIR", str(data_dir / "collections"))
    monkeypatch.setattr(app, "BACKUPS_DIR", str(data_dir / "backups"))
    app.st.session_state["authenticated_user"] = "alice"

    try:
        assert app.get_collections_dir() == os.path.join(str(data_dir), "workspaces", "alice", "collections")
        assert app.get_backups_dir() == os.path.join(str(data_dir), "workspaces", "alice", "backups")
    finally:
        app.st.session_state["authenticated_user"] = None


def test_collection_import_rebuilds_index_without_importing_pickle(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(app, "COLLECTIONS_DIR", str(data_dir / "collections"))
    monkeypatch.setattr(app, "BACKUPS_DIR", str(data_dir / "backups"))
    monkeypatch.setattr(app, "WORKSPACES_DIR", str(data_dir / "workspaces"))
    monkeypatch.setattr(app, "get_embeddings", lambda: FakeEmbeddings())
    app.st.session_state["authenticated_user"] = None

    archive = io.BytesIO()
    archive.name = "collection.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            app.DOCUMENTS_JSON,
            json.dumps([{"page_content": "alpha document", "metadata": {"file_name": "alpha.txt"}}])
        )
        zip_file.writestr(app.FILE_METADATA_JSON, json.dumps({}))
        zip_file.writestr(app.CHAT_HISTORY_JSON, json.dumps([]))
        zip_file.writestr("index.pkl", b"this must never be imported")
    archive.seek(0)

    success, message = app.import_collection_zip(archive, "imported")

    assert success, message
    imported_path = data_dir / "collections" / "imported"
    assert (imported_path / "documents.json").exists()
    assert (imported_path / "index.faiss").exists()
    assert not (imported_path / "index.pkl").exists()

    store = app.load_faiss_vector_store("imported")
    assert store is not None
    assert store.similarity_search("alpha", k=1)[0].page_content == "alpha document"


def test_collection_import_rejects_unsafe_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "COLLECTIONS_DIR", str(tmp_path / "collections"))
    app.st.session_state["authenticated_user"] = None

    archive = io.BytesIO()
    archive.name = "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(app.DOCUMENTS_JSON, "[]")
        zip_file.writestr("../outside.txt", "blocked")
    archive.seek(0)

    success, message = app.import_collection_zip(archive, "unsafe")

    assert not success
    assert "unsafe path" in message.lower()


def test_upload_size_validation(monkeypatch):
    monkeypatch.setattr(app, "MAX_UPLOAD_SIZE_BYTES", 1024 * 1024)

    class UploadedFile:
        name = "large.txt"
        size = 1024 * 1024 + 1

    errors = app.validate_uploaded_files([UploadedFile()])

    assert errors == ["large.txt: maximum file size is 1 MB."]

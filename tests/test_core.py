import importlib.util

from langchain_core.documents import Document
from PIL import Image

import app


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

from langchain_core.documents import Document

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

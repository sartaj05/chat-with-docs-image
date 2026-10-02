import os

import pytest

import app


@pytest.mark.external
def test_gemini_external_round_trip():
    if os.getenv("RUN_EXTERNAL_TESTS") != "1" or not app.api_key:
        pytest.skip("Set RUN_EXTERNAL_TESTS=1 and GOOGLE_API_KEY to run Gemini integration tests.")
    response = app.GeminiChatModel(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=app.api_key
    ).invoke("Reply with exactly: OK")
    assert "OK" in str(response.content).upper()


@pytest.mark.external
def test_ollama_external_round_trip():
    if os.getenv("RUN_OLLAMA_TESTS") != "1" or app.ChatOllama is None:
        pytest.skip("Set RUN_OLLAMA_TESTS=1 and run Ollama locally to run Ollama integration tests.")
    response = app.ChatOllama(model="llama3.2", temperature=0).invoke("Reply with exactly: OK")
    assert "OK" in str(response.content).upper()

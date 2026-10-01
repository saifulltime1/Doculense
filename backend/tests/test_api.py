import io

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE_TEXT = (
    "Machine learning is transforming software development. "
    "Developers now use models to write code, find bugs, and review pull requests. "
    "However, machine learning systems require careful evaluation. "
    "Evaluation ensures that models behave safely in production. "
    "Teams that invest in evaluation ship more reliable software. "
    "Reliable software builds user trust over time."
)


@pytest.fixture()
def client(monkeypatch):
    # Ensure the LLM path is disabled by default for deterministic tests.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return TestClient(app)


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["llm_enabled"] is False


def test_summarize_text(client):
    resp = client.post(
        "/api/summarize", data={"text": SAMPLE_TEXT, "length": "short"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "extractive"
    assert body["length"] == "short"
    assert body["summary"]
    assert len(body["summary"]) < len(SAMPLE_TEXT)
    assert body["sentences_used"] == 3
    assert isinstance(body["keywords"], list) and body["keywords"]
    assert "machine" in body["keywords"] or "learning" in body["keywords"]
    assert body["char_count"] == len(SAMPLE_TEXT)
    assert body["word_count"] == len(SAMPLE_TEXT.split())


def test_summarize_file(client):
    files = {"file": ("notes.md", io.BytesIO(SAMPLE_TEXT.encode()), "text/markdown")}
    resp = client.post("/api/summarize", data={"length": "medium"}, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "extractive"
    assert body["summary"]
    assert body["sentences_used"] == 6


def test_summarize_rejects_unsupported_file(client):
    files = {
        "file": ("doc.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")
    }
    resp = client.post("/api/summarize", files=files)
    assert resp.status_code == 400


def test_summarize_requires_input(client):
    resp = client.post("/api/summarize", data={"text": "   ", "length": "short"})
    assert resp.status_code == 400


def test_summarize_rejects_bad_length(client):
    resp = client.post("/api/summarize", data={"text": SAMPLE_TEXT, "length": "huge"})
    assert resp.status_code == 400


def test_fallback_to_extractive_when_llm_fails(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import app.main as main

    def boom(_text, _length="medium"):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(main, "abstractive_summarize", boom)
    client = TestClient(app)
    resp = client.post("/api/summarize", data={"text": SAMPLE_TEXT})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "extractive"
    assert body["summary"]


def test_abstractive_mode_when_llm_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import app.main as main

    monkeypatch.setattr(
        main, "abstractive_summarize", lambda _t, _l="medium": "AI summary."
    )
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.json()["llm_enabled"] is True
    resp = client.post("/api/summarize", data={"text": SAMPLE_TEXT})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "abstractive"
    assert body["summary"] == "AI summary."
    assert body["sentences_used"] is None

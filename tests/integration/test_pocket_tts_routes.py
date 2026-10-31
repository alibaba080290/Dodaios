"""Pocket TTS through the real provider routes: card, voice picker, Test button."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from dodaios.core.bus import EventBus
from dodaios.core.config import JarvisConfig
from dodaios.ui.web.server import WebServer


def _client(monkeypatch, url: str) -> Iterator[TestClient]:
    monkeypatch.setattr("dodaios.plugins.tts.pocket_local._card_base_url", lambda: url)
    cfg = JarvisConfig()
    cfg.ui.dev_mode = True
    cfg.tts.provider = "pocket-local"
    server = WebServer(cfg, bus=EventBus())
    server.app.state.config = cfg
    try:
        with TestClient(server.app) as client:
            yield client
    finally:
        # GET /api/providers memoises every card's credential lookup; later
        # tests that fake a key must not see this run's answers.
        from dodaios.dictation.polish_client import reset_credential_cache

        reset_credential_cache()


@pytest.fixture
def client(monkeypatch) -> Iterator[TestClient]:
    # Nothing listens on port 9 (discard) on loopback: a stopped server.
    yield from _client(monkeypatch, "http://127.0.0.1:9")


def _cards(client: TestClient) -> list[dict]:
    response = client.get("/api/providers")
    assert response.status_code == 200
    body = response.json()
    return body if isinstance(body, list) else body["providers"]


def test_pocket_card_is_listed_next_to_piper(client: TestClient) -> None:
    cards = {card["id"]: card for card in _cards(client)}
    assert "piper-local" in cards
    pocket = cards["pocket-local"]
    assert pocket["tier"] == "tts"
    assert pocket["label"] == "Pocket TTS French (on this machine)"
    assert pocket["supports_base_url"] is True
    assert pocket["default_base_url"] == "http://127.0.0.1:18765"


def test_pocket_voice_picker_lists_estelle_first(client: TestClient) -> None:
    response = client.get("/api/providers/pocket-local/models")
    assert response.status_code == 200
    ids = [model["id"] for model in response.json()["models"]]
    assert ids[0] == "estelle"
    assert {"fantine", "eponine", "charles", "paul", "george"} <= set(ids)


def test_test_button_reports_a_stopped_server(client: TestClient) -> None:
    response = client.post("/api/providers/pocket-local/test", json={})
    assert response.status_code == 200
    assert response.json()["status"] != "ok"


@pytest.mark.skipif(not os.environ.get("POCKET_TTS_LIVE_URL"), reason="set POCKET_TTS_LIVE_URL")
def test_test_button_passes_against_a_live_server(monkeypatch) -> None:
    for client in _client(monkeypatch, os.environ["POCKET_TTS_LIVE_URL"]):
        response = client.post("/api/providers/pocket-local/test", json={})
        assert response.status_code == 200
        assert response.json()["status"] == "ok", response.json()

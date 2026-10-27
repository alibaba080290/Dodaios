"""Regressions from the activated BC-250's actual browser voice path."""

import os
from types import SimpleNamespace

import pytest

from dodaios.brain.manager import BrainManager
from dodaios.browser_voice.route import _resolve_language
from dodaios.browser_voice.session import BrowserVoiceSession
from dodaios.core.bus import EventBus
from dodaios.core.config import JarvisConfig
from dodaios.plugins.brain.claude_cli import ClaudeCliBrain


@pytest.mark.parametrize("level", ["fast", "deep", "code"])
def test_claude_subscription_default_stays_in_the_real_fallback_chain(level):
    config = JarvisConfig()
    config.brain.primary = "claude-cli"
    config.brain.deep_brain = None
    config.brain.providers.pop("claude-cli", None)
    manager = BrainManager(config=config, bus=EventBus(), tools={})
    manager._registry._classes = {"claude-cli": ClaudeCliBrain}
    manager._registry._loaded = True
    assert manager._fast_model("claude-cli") is None
    assert manager._deep_model("claude-cli") is None
    assert manager._build_fallback_chain(level) == [("claude-cli", None)]


def test_browser_route_resolves_french_pin_to_the_french_voice():
    config = JarvisConfig()
    config.brain.reply_language = "fr"
    assert _resolve_language(config) == "fr-FR"


@pytest.mark.asyncio
@pytest.mark.parametrize("pin, expected", [("fr", "fr-FR"), ("auto", "en-US")])
async def test_browser_ui_language_cannot_override_an_explicit_reply_pin(pin, expected):
    config = JarvisConfig()
    config.brain.reply_language = pin
    sent = []

    async def send(value):
        sent.append(value)

    session = BrowserVoiceSession(
        session_id="language-pin",
        send_binary=send,
        send_json=send,
        stt=None,
        brain=None,
        tts=None,
        language_code=_resolve_language(config),
        config=config,
    )
    await session.handle_control(
        {"type": "audio_start", "sample_rate": 48000, "language": "en-US"}
    )
    assert session.language_code == expected
    assert sent == [{"type": "audio_ready"}]


@pytest.mark.asyncio
async def test_headless_launcher_disables_only_the_nonexistent_native_warmup(monkeypatch):
    from dodaios.core import loop_executor, socket_budget
    from dodaios.ui.web import launcher

    monkeypatch.setenv("JARVIS_VOICE", "1")
    monkeypatch.setattr(loop_executor, "install_prewarmed_default_executor", lambda _: None)
    monkeypatch.setattr(socket_budget, "start_default_watchdog", lambda: None)

    def reached_before_server_construction(_):
        assert os.environ.get("JARVIS_VOICE") == "0"
        return None

    monkeypatch.setattr(launcher, "_prepare_background_service", reached_before_server_construction)
    assert await launcher._run_headless(SimpleNamespace(background_service=True)) == 0

"""Activation preserves unrelated settings and refuses unknown installed code."""

import importlib.util
from pathlib import Path

import pytest
import tomlkit


def module():
    path = Path(__file__).resolve().parents[3] / "scripts/enable_french_voice.py"
    assert path.is_file(), "French voice activation is not implemented"
    spec = importlib.util.spec_from_file_location("enable_french_voice", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_profile_preserves_existing_devices_and_unrelated_settings(tmp_path):
    m = module()
    config = tmp_path / "jarvis.toml"
    config.write_text('# keep this comment\n[audio]\ninput_device = 2\n[brain]\nprimary = "old"\n')
    m.configure(config, None)
    text = config.read_text()
    data = tomlkit.parse(text)
    assert data["audio"]["input_device"] == 2
    assert "# keep this comment" in text
    assert data["brain"]["primary"] == "claude-cli"
    assert data["brain"]["reply_language"] == "fr"
    assert data["stt"]["provider"] == "nemotron-local"
    assert data["tts"]["provider"] == "piper-local"
    assert data["tts"]["speed"] == 0.9
    assert data["performance"]["streaming_tts"] is False
    assert data["voice"]["mode"] == "pipeline"


def test_unknown_code_is_refused_before_any_file_changes(tmp_path):
    m = module()
    target = tmp_path / "installed"
    for relative in m.ORIGINAL_HASHES:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unrecognized installed code")
    with pytest.raises(RuntimeError, match="unrecognized"):
        m.validate_overlay(m.ROOT, target)
    assert all(p.read_text() == "unrecognized installed code" for p in target.rglob("*.py"))


@pytest.mark.parametrize("modified", [False, True])
def test_reported_bc250_catalog_is_accepted_without_accepting_other_edits(tmp_path, modified):
    """Reproduce the installed catalog from the user's complete unified diff."""
    m = module()
    target = tmp_path / "installed"
    for relative in m.ORIGINAL_HASHES:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((m.ROOT / relative).read_bytes())
    catalog = target / "dodaios/speech/local_models.py"
    text = catalog.read_text()
    start = text.index("    # French is opt-in until")
    end = text.index('    "vits-piper-de_DE-ramona-low":', start)
    text = (text[:start] + text[end:]).replace(
        'SHERPA_ONNX_PACKAGE = "sherpa-onnx>=1.13.3,<2"',
        'SHERPA_ONNX_PACKAGE = "sherpa-onnx>=1.13.3"',
    )
    if modified:
        text += "\n# Unreviewed local change\n"
    catalog.write_text(text)
    if modified:
        with pytest.raises(RuntimeError, match="unrecognized.*local_models"):
            m.validate_overlay(m.ROOT, target)
    else:
        m.validate_overlay(m.ROOT, target)
    assert catalog.read_text() == text


def test_restore_recovers_original_bytes_and_permissions(tmp_path):
    m = module()
    target = tmp_path / "config.toml"
    target.write_bytes(b"original\n")
    target.chmod(0o600)
    snapshots = m.snapshot([target], tmp_path / "backup")
    m.atomic_write(target, b"changed\n")
    m.restore(snapshots)
    assert target.read_bytes() == b"original\n"
    assert target.stat().st_mode & 0o777 == 0o600


def test_configuration_updates_existing_drift_baseline(tmp_path):
    import json

    m = module()
    config, baseline = tmp_path / "config.toml", tmp_path / "baseline.json"
    config.write_text('[brain]\nprimary = "old"\n')
    baseline.write_text('{"brain":{"primary":"old"},"unrelated":true}')
    m.configure(config, baseline)
    data = json.loads(baseline.read_text())
    assert data["brain"]["primary"] == "claude-cli"
    assert data["tts"]["speed"] == 0.9
    assert data["unrelated"] is True


def test_conflicting_profile_is_rejected_without_changing_active_config(tmp_path, monkeypatch):
    from dodaios.core import config as configuration

    m = module()
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "conflict.yaml").write_text("voice:\n  mode: realtime\n")
    monkeypatch.setattr(configuration, "PROFILES_DIR", profiles)
    monkeypatch.setenv("JARVIS_PROFILE", "conflict")
    config = tmp_path / "jarvis.toml"
    original = '[brain]\nprimary = "old"\n'
    config.write_text(original)
    with pytest.raises(RuntimeError, match="voice.mode"):
        m.verify_candidate(config)
    assert config.read_text() == original


def test_rollback_restores_files_even_if_stopping_the_service_times_out(tmp_path, monkeypatch):
    import subprocess

    m = module()
    config = tmp_path / "config.toml"
    config.write_bytes(b"original")
    saved = m.snapshot([config], tmp_path / "backup")
    m.atomic_write(config, b"changed")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("systemctl", 60)

    monkeypatch.setattr(m.subprocess, "run", timeout)
    message = m.rollback(saved, active=False, env={}, working=tmp_path)
    assert config.read_bytes() == b"original"
    assert "restored" in message


@pytest.mark.parametrize("token", [None, "test-private-control-key"])
def test_readiness_check_waits_for_the_actual_http_voice_status(token):
    import http.server
    import threading

    m = module()
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ready":false}' if len(seen) == 1 else b'{"ready":true}')

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        m.wait_for_voice_ready(f"http://127.0.0.1:{server.server_port}", timeout=2, token=token)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    assert seen == ["/api/voice/status", "/api/voice/status"]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["Bonjour.", ""])
async def test_activation_checks_the_routed_subscription_answer(monkeypatch, answer):
    from dodaios.brain.provider_registry import BrainProviderRegistry
    from dodaios.core.config import JarvisConfig
    from dodaios.core.protocols import BrainDelta
    from dodaios.plugins.brain.claude_cli import ClaudeCliBrain

    m = module()
    calls = []

    class Subscription(ClaudeCliBrain):
        async def complete(self, request):
            calls.append(request)
            yield BrainDelta(content=answer)

    def load(self):
        self._classes = {"claude-cli": Subscription}
        self._loaded = True

    monkeypatch.setattr(BrainProviderRegistry, "_load", load)
    monkeypatch.setattr("dodaios.core.config.get_secret_any", lambda _: None)
    config = JarvisConfig()
    config.brain.primary = "claude-cli"
    if answer:
        await m.check_brain(config)
    else:
        with pytest.raises(RuntimeError, match="no answer"):
            await m.check_brain(config)
    assert len(calls) == 1
    assert calls[0].tools == ()

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

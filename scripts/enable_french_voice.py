#!/usr/bin/env python3
"""Activate the tested French voice path on a known Dodaios Linux installation.

Run with the production venv and -I. Default: read-only checks. --apply backs up
the five patched modules and configuration, stops the user service, validates
the native engines, configures French and starts the service. No pip changes.
Unknown installed source or conflicting service overrides are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = "dodaios.service"
ORIGINAL_HASHES = {
    "dodaios/core/turn_language.py": {
        "22a4486b62c7667184f1221e381a44ed06dcafa438522c3cf6cd468b36cf98f2"
    },
    "dodaios/brain/manager.py": {
        "e5bf6ac8f210ac7a7e6517687e738c4aa00c7f72c0759a535bb160fca63caf47"
    },
    "dodaios/speech/pipeline.py": {
        "4a3402b8e1c9bf489f7864dbcf3756555463cfa219081d68fdd6a42671761d48"
    },
    "dodaios/speech/local_models.py": {
        "7742c21ba39fbb52c4cb1d1ca2c539458cbcefc86a59cc9f8e7ea7749e662c29",
        "79befe1739924ca07415ed3617db21d7227206ccbe53bd3a3f015a9a62bf379f",
        # BC-250 installation: unbounded Sherpa requirement, no French bundle.
        "990be5320cb6109603ee23a75c3c15a1c7e75e2451f33453872c793305286faa",
    },
    "dodaios/plugins/tts/piper_local.py": {
        "5c6fbfe7690495b60b116e4ddad8f934f6ae82290f2bf275e82501a2c3734ddd",
        "c97182528821a7768fd539d091878a337e49e35a3eb4a16a8033a4112e53fe38",
    },
}
PROFILE = {
    "voice": {"mode": "pipeline", "profile": ""},
    "brain": {"primary": "claude-cli", "reply_language": "fr"},
    "stt": {
        "provider": "nemotron-local",
        "provider_user_selected": True,
        "language": "fr",
        "fallback": "",
    },
    "tts": {"provider": "piper-local", "language_code": "fr-FR", "speed": 0.9, "fallback": ""},
    # One complete synthesis at a time, using the path exercised by the smoke test.
    "performance": {"streaming_tts": False},
}


def atomic_write(path: Path, data: bytes, mode: int | None = None) -> None:
    permissions = mode if mode is not None else stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary = tempfile.mkstemp(prefix=".french-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, permissions)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def snapshot(paths: list[Path], directory: Path) -> list[tuple[Path, Path, int]]:
    directory.mkdir(parents=True, mode=0o700)
    result = []
    for index, path in enumerate(paths):
        saved = directory / f"{index:02d}-{path.name}"
        mode = stat.S_IMODE(path.stat().st_mode)
        saved.write_bytes(path.read_bytes())
        saved.chmod(0o600)
        result.append((path, saved, mode))
    return result


def restore(snapshots: list[tuple[Path, Path, int]]) -> None:
    for path, saved, mode in snapshots:
        atomic_write(path, saved.read_bytes(), mode)
        clear_bytecode(path)


def rollback(snapshots, *, active, env, working) -> str:
    try:
        run(["systemctl", "--user", "stop", UNIT], env=env, cwd=working)
    except (OSError, subprocess.SubprocessError):
        pass
    restore(snapshots)
    if active:
        try:
            run(["systemctl", "--user", "start", UNIT], env=env, cwd=working)
        except (OSError, subprocess.SubprocessError):
            return "Original files restored; restarting the original service failed."
    return "Original code/config restored."


def clear_bytecode(path: Path) -> None:
    if path.suffix == ".py":
        for cached in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
            cached.unlink()


def validate_overlay(source: Path, target: Path) -> None:
    for relative, originals in ORIGINAL_HASHES.items():
        incoming, installed = source / relative, target / relative
        compile(incoming.read_bytes(), str(incoming), "exec")
        current = hashlib.sha256(installed.read_bytes()).hexdigest()
        desired = hashlib.sha256(incoming.read_bytes()).hexdigest()
        if current not in originals | {desired}:
            raise RuntimeError(f"Refusing unrecognized installed source: {relative}")


def configure(config: Path, baseline: Path | None) -> None:
    import tomlkit

    raw = config.read_text(encoding="utf-8")
    bom = raw.startswith("\ufeff")
    document = tomlkit.parse(raw.lstrip("\ufeff"))
    for section, values in PROFILE.items():
        if section not in document:
            document[section] = tomlkit.table()
        for key, value in values.items():
            document[section][key] = value
    output = ("\ufeff" if bom else "") + tomlkit.dumps(document)
    atomic_write(config, output.encode("utf-8"))
    if baseline is not None and baseline.exists():
        data = json.loads(baseline.read_text())
        for section, values in PROFILE.items():
            data.setdefault(section, {}).update(values)
        atomic_write(baseline, (json.dumps(data, indent=2) + "\n").encode())


def verify_candidate(config: Path) -> None:
    from dodaios.core.config import load_config

    # The real YAML profile and environment must not undo the TOML changes.
    with tempfile.TemporaryDirectory(prefix=".french-check-", dir=config.parent) as folder:
        candidate = Path(folder) / config.name
        shutil.copyfile(config, candidate)
        configure(candidate, None)
        effective = load_config(config_file=candidate)
        for section, values in PROFILE.items():
            for key, wanted in values.items():
                if getattr(getattr(effective, section), key) != wanted:
                    raise RuntimeError(f"Configuration override prevents {section}.{key}")


def run(
    args: list[str], *, env: dict[str, str], cwd: Path, timeout: int = 60
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, env=env, cwd=cwd, check=True, capture_output=True, text=True, timeout=timeout
    )


def service_context() -> tuple[dict[str, str], Path]:
    raw = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            UNIT,
            "-p",
            "LoadState",
            "-p",
            "WorkingDirectory",
            "-p",
            "Environment",
            "-p",
            "EnvironmentFiles",
        ],
        text=True,
        timeout=15,
    )
    properties = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    if properties.get("LoadState") != "loaded":
        raise RuntimeError("The Dodaios user service is not installed.")
    if properties.get("EnvironmentFiles"):
        raise RuntimeError("Service EnvironmentFiles need review before automatic activation.")
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("JARVIS", "DODAIOS")):
            env.pop(name)
    manager_env = subprocess.check_output(
        ["systemctl", "--user", "show-environment"], text=True, timeout=15
    )
    for line in manager_env.splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            env[name] = value
    for assignment in shlex.split(properties.get("Environment", "")):
        name, _, value = assignment.partition("=")
        env[name] = value
    for section, values in PROFILE.items():
        for key, value in values.items():
            name = f"JARVIS__{section.upper()}__{key.upper()}"
            alternate = name.replace("JARVIS", "DODAIOS", 1)
            wanted = str(value).lower() if isinstance(value, bool) else str(value)
            for candidate in (name, alternate):
                if candidate in env and env[candidate] != wanted:
                    raise RuntimeError(f"Conflicting service override: {candidate}")
    working = properties.get("WorkingDirectory")
    if not working or not Path(working).is_dir():
        raise RuntimeError("Cannot determine the Dodaios service working directory.")
    return env, Path(working)


def probe() -> dict[str, str]:
    import dodaios
    from dodaios.core.config import ensure_project_root_cwd, resolve_config_path
    from dodaios.core.config_writer import _config_soll_path
    from dodaios.plugins.brain.claude_cli import ClaudeCliBrain
    from dodaios.speech.local_models import models_root

    # Match the real launcher's chdir before resolving any relative paths.
    ensure_project_root_cwd()
    if not shutil.which("espeak-ng"):
        raise RuntimeError("eSpeak NG is missing")
    if not ClaudeCliBrain.subscription_connected():
        raise RuntimeError("Claude subscription login is unavailable")
    config = resolve_config_path().resolve()
    if not config.is_file():
        raise RuntimeError("The active Dodaios configuration is missing.")
    verify_candidate(config)
    return {
        "root": str(Path(dodaios.__file__).resolve().parents[1]),
        "config": str(config),
        "baseline": str(_config_soll_path().resolve()),
        "models": str(models_root().resolve()),
    }


async def native_check(data_dir: str) -> None:
    import numpy as np

    from dodaios.plugins.stt.nemotron_local import NemotronLocalSTT
    from dodaios.plugins.tts.piper_local import PiperLocalTTS
    from dodaios.speech.local_models import (
        NEMOTRON_MODEL_ID,
        PIPER_DEFAULT_VOICES,
    )
    from dodaios.speech.sherpa_models import download_sherpa_model

    for model in (NEMOTRON_MODEL_ID, *PIPER_DEFAULT_VOICES, "vits-piper-fr_FR-siwis-medium"):
        download_sherpa_model(model, data_dir=data_dir)
    recognizer = NemotronLocalSTT(language="fr")
    voice = PiperLocalTTS(speed=0.9)
    try:
        recognizer._ensure_model()
        chunks = [
            chunk.pcm
            async for chunk in voice.synthesize(
                "Bonjour. Je suis prêt à parler français.", language_code="fr-FR"
            )
        ]
        samples = np.frombuffer(b"".join(chunks), dtype=np.int16)
        if samples.size < 10_000 or not np.any(samples):
            raise RuntimeError("The native French voice did not generate usable audio.")
        if voice.last_voice != "vits-piper-fr_FR-siwis-medium":
            raise RuntimeError("The French voice was not selected.")
    finally:
        await voice.aclose()
        await recognizer.aclose()


def finish(info: dict[str, str]) -> None:
    import asyncio

    from dodaios.brain.manager import normalize_reply_language
    from dodaios.core.config import ensure_project_root_cwd, load_config
    from dodaios.plugins.stt import build_stt_from_config
    from dodaios.plugins.tts import build_tts_from_config
    from dodaios.speech.pipeline import SpeechPipeline

    ensure_project_root_cwd()
    if normalize_reply_language("fr") != "fr" or SpeechPipeline._bcp47("fr") != "fr-FR":
        raise RuntimeError("The installed French route is incomplete.")
    data_dir = str(Path(info["models"]).parent)
    os.environ["JARVIS__MEMORY__DATA_DIR"] = data_dir
    asyncio.run(native_check(data_dir))
    configure(Path(info["config"]), Path(info["baseline"]))
    effective = load_config(config_file=Path(info["config"]))
    if build_stt_from_config(effective.stt).name != "nemotron-local":
        raise RuntimeError("The installed app did not select Nemotron.")
    if build_tts_from_config(effective.tts).name != "piper-local":
        raise RuntimeError("The installed app did not select Piper.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="activate French voice with backup")
    parser.add_argument("--finish", help=argparse.SUPPRESS)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe()))
        return 0
    if args.finish:
        finish(json.loads(args.finish))
        return 0
    if sys.platform != "linux" or os.getuid() == 0:
        raise RuntimeError("Run as the BC-250 desktop user, using the Dodaios venv.")
    env, working = service_context()
    result = run(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--probe"], env=env, cwd=working
    )
    info = json.loads(result.stdout.strip().splitlines()[-1])
    target, config = Path(info["root"]), Path(info["config"])
    if not config.is_file():
        raise RuntimeError("The active Dodaios configuration is missing.")
    if target.resolve() == ROOT.resolve():
        raise RuntimeError("Use the production venv, not an editable test checkout.")
    validate_overlay(ROOT, target)
    if not args.apply:
        print("Checks passed. Use --apply to enable Nemotron, Claude and Siwis in French.")
        return 0
    baseline = Path(info["baseline"])
    paths = [target / name for name in ORIGINAL_HASHES] + [config]
    if baseline.exists():
        paths.append(baseline)
    backup = config.parent / "french-voice-backups" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S-%f")
    saved = snapshot(paths, backup)
    active = (
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", UNIT], check=False, timeout=15
        ).returncode
        == 0
    )
    try:
        run(["systemctl", "--user", "stop", UNIT], env=env, cwd=working)
        print("Installing the checked French voice integration...", flush=True)
        for relative in ORIGINAL_HASHES:
            path = target / relative
            atomic_write(path, (ROOT / relative).read_bytes())
            clear_bytecode(path)
        run(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--finish", json.dumps(info)],
            env=env,
            cwd=working,
            timeout=1200,
        )
        run(["systemctl", "--user", "start", UNIT], env=env, cwd=working)
        run(["systemctl", "--user", "is-active", "--quiet", UNIT], env=env, cwd=working)
    except BaseException:
        try:
            status = rollback(saved, active=active, env=env, working=working)
        except Exception:
            raise RuntimeError(f"Automatic restore failed. Backup: {backup}") from None
        raise RuntimeError(f"Activation failed. {status} Backup: {backup}") from None
    print(f"Native French engines validated; Dodaios service active. Backup: {backup}")
    print("Open Dodaios and start a voice conversation.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # No child stdout/stderr, environment values or credentials are echoed.
        print(f"French voice activation: {error}", file=sys.stderr)
        raise SystemExit(1) from None

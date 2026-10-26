#!/usr/bin/env python3
"""Isolated coexistence smoke test: loaded Nemotron + speaking Siwis.

No microphone, service, config edits, downloads or permanent model installation.
Reads installed Nemotron bundle (or DODAIOS_NEMOTRON_DIR override). Uses Siwis
from /tmp through isolated model-store symlink. Measures /proc/self/status RSS.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from dodaios.plugins.stt.nemotron_local import NemotronLocalSTT
from dodaios.plugins.tts.piper_local import PiperLocalTTS
from dodaios.speech.local_models import NEMOTRON_MODEL_ID, bundle_present, sherpa_model_dir

SIWIS_ID = "vits-piper-fr_FR-siwis-medium"
SIWIS = Path("/tmp/dodaios-siwis") / SIWIS_ID
NEMOTRON_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def mem_mb() -> tuple[float, float]:
    status = Path("/proc/self/status").read_text()
    def value(key: str) -> float:
        for line in status.splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024
        return 0.0
    return value("VmRSS"), value("VmSwap")


def find_nemotron() -> Path:
    override = os.environ.get("DODAIOS_NEMOTRON_DIR", "").strip()
    home = Path.home()
    possible = (
        [Path(override).expanduser()] if override else [
            home / ".dodaios/data/models" / NEMOTRON_MODEL_ID,
            home / ".dodaios/models" / NEMOTRON_MODEL_ID,
            home / ".local/share/dodaios/data/models" / NEMOTRON_MODEL_ID,
            home / ".jarvis/data/models" / NEMOTRON_MODEL_ID,
            Path.cwd() / "data/models" / NEMOTRON_MODEL_ID,
        ]
    )
    for folder in possible:
        if folder.is_dir() and all((folder / name).is_file() for name in NEMOTRON_FILES):
            return folder
    print("Nemotron directory not located in known places; nothing loaded.")
    print("Search for encoder.int8.onnx in your existing Dodaios data/models folder.")
    print("Then rerun using: env DODAIOS_NEMOTRON_DIR=/actual/path ...")
    raise SystemExit(2)


async def speak() -> None:
    tts = PiperLocalTTS(gender="feminine")
    pcm_bytes = 0
    count = 0
    try:
        if tts._resolve_voice(None, "fr-FR") != SIWIS_ID:
            raise RuntimeError("Siwis not selected by normal French discovery")
        async for chunk in tts.synthesize(
            "Bonjour. Nemotron et Siwis fonctionnent ensemble, sans modifier Dodaios.",
            language_code="fr-FR",
        ):
            count += 1
            pcm_bytes += len(chunk.pcm)
    finally:
        await tts.aclose()
    if not pcm_bytes:
        raise RuntimeError("No French audio produced")
    print(f"TTS: {count} chunks, {pcm_bytes} PCM bytes")


def main() -> None:
    nemotron = find_nemotron()
    if not SIWIS.is_dir():
        raise RuntimeError(f"Siwis missing: {SIWIS}")
    print(f"Nemotron model: {nemotron}")
    print(f"Memory before load (RSS MiB, swap MiB): {mem_mb()}")
    with tempfile.TemporaryDirectory(prefix="dodaios-voice-both-") as root:
        os.environ["JARVIS__MEMORY__DATA_DIR"] = root
        link = sherpa_model_dir(SIWIS_ID)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(SIWIS, target_is_directory=True)
        if not bundle_present(SIWIS_ID):
            raise RuntimeError("Siwis model was not accepted by normal discovery")
        stt = NemotronLocalSTT(model_dir=nemotron, num_threads=2)
        stt._ensure_model()  # load, no microphone/capture
        print(f"Nemotron loaded. Memory (RSS MiB, swap MiB): {mem_mb()}")
        asyncio.run(speak())
        print(f"Both active. Memory (RSS MiB, swap MiB): {mem_mb()}")
        # Ensure native model lifetime includes entire TTS operation.
        assert stt._recognizer is not None
        print("SUCCESS: both native models loaded in one isolated process")
    print("Temporary links cleaned; production service unchanged.")


if __name__ == "__main__":
    main()

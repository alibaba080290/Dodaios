#!/usr/bin/env python3
"""Check actual Dodaios model discovery + French TTS in an isolated data root.

Uses a temporary symlink to the user's previously downloaded Siwis model.
Never writes to ~/.dodaios, the live model store, or service configuration.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import wave
from pathlib import Path

from dodaios.plugins.tts.piper_local import PiperLocalTTS
from dodaios.speech.local_models import bundle_present, sherpa_model_dir

MODEL_ID = "vits-piper-fr_FR-siwis-medium"
SOURCE = Path("/tmp/dodaios-siwis") / MODEL_ID
OUTPUT = Path("/tmp/dodaios-managed-siwis.wav")
TEXT = (
    "Bonjour, je suis Dodaios. "
    "La reconnaissance et la synthèse vocale française fonctionnent."
)


async def speak() -> None:
    adapter = PiperLocalTTS(gender="feminine")
    output = []
    rate = None
    try:
        chosen = adapter._resolve_voice(None, "fr-FR")
        if chosen != MODEL_ID:
            raise RuntimeError(f"Incorrect French voice selected: {chosen!r}")
        async for chunk in adapter.synthesize(TEXT, language_code="fr-FR"):
            if rate is not None and chunk.sample_rate != rate:
                raise RuntimeError("Unexpected audio sample rate change")
            rate = chunk.sample_rate
            output.append(chunk.pcm)
        if adapter.last_voice != MODEL_ID:
            raise RuntimeError("Incorrect reported last voice")
    finally:
        await adapter.aclose()
    if not output or not rate:
        raise RuntimeError("No audio generated")
    with wave.open(str(OUTPUT), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(output))
    total_seconds = sum(map(len, output)) / (2 * rate)
    print(f"SUCCESS: voice={MODEL_ID}; chunks={len(output)}; duration={total_seconds:.2f}s")
    print(f"Play: pw-play {OUTPUT}")


def main() -> None:
    if not SOURCE.is_dir():
        raise RuntimeError(f"Siwis source missing: {SOURCE}")
    with tempfile.TemporaryDirectory(prefix="dodaios-siwis-managed-") as root:
        os.environ["JARVIS__MEMORY__DATA_DIR"] = root
        link = sherpa_model_dir(MODEL_ID)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(SOURCE, target_is_directory=True)
        if not bundle_present(MODEL_ID):
            raise RuntimeError("Dodaios does not consider Siwis complete")
        print("Model resolved from isolated Dodaios model directory:", link)
        asyncio.run(speak())
    print("Temporary model link and isolated data root cleaned up.")


if __name__ == "__main__":
    main()

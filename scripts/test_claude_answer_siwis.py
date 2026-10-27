#!/usr/bin/env python3
"""Speak an existing Claude response through Dodaios Piper without cloud calls.

Reads /tmp/dodaios-claude-reponse.txt, resolves Siwis through an isolated
Dodaios model root, and writes a WAV. Never invokes Claude or changes service.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import wave
from pathlib import Path

from dodaios.plugins.tts.piper_local import PiperLocalTTS
from dodaios.speech.local_models import bundle_present, sherpa_model_dir

MODEL = "vits-piper-fr_FR-siwis-medium"
SOURCE = Path("/tmp/dodaios-siwis") / MODEL
TEXT = Path("/tmp/dodaios-claude-reponse.txt")
OUT = Path("/tmp/dodaios-claude-parle.wav")


async def run() -> None:
    if not TEXT.is_file():
        raise RuntimeError(f"No Claude answer at {TEXT}")
    answer = TEXT.read_text(encoding="utf-8").strip()
    if not answer:
        raise RuntimeError("Claude answer is empty")
    if len(answer) > 2000:
        raise RuntimeError("Answer too long for a short voice smoke test")
    if not SOURCE.is_dir():
        raise RuntimeError(f"Siwis source model not found at {SOURCE}")
    print("Texte Claude:", answer, flush=True)
    with tempfile.TemporaryDirectory(prefix="dodaios-claude-piper-") as root:
        os.environ["JARVIS__MEMORY__DATA_DIR"] = root
        link = sherpa_model_dir(MODEL)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(SOURCE, target_is_directory=True)
        if not bundle_present(MODEL):
            raise RuntimeError("Siwis failed model discovery")
        tts = PiperLocalTTS(gender="feminine", speed=0.90)
        parts = []
        rate = None
        try:
            async for chunk in tts.synthesize(answer, language_code="fr-FR"):
                if rate is not None and rate != chunk.sample_rate:
                    raise RuntimeError("Output sample rate changed")
                rate = chunk.sample_rate
                parts.append(chunk.pcm)
        finally:
            await tts.aclose()
        if not rate or not parts:
            raise RuntimeError("Piper yielded no speech")
        with wave.open(str(OUT), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(b"".join(parts))
        print(f"SUCCESS: {len(parts)} chunks, {sum(map(len, parts)) / (2*rate):.2f}s")
        print(f"Lecture: pw-play {OUT}")


if __name__ == "__main__":
    asyncio.run(run())

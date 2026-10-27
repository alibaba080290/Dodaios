#!/usr/bin/env python3
"""Offline WAV -> real Nemotron STT -> fixed French reply -> real Siwis TTS.

This is an isolated microphone-to-audio smoke test, NOT an agent/Claude test.
Run against a recorded 16kHz mono 16-bit WAV. No microphone control here;
no service/config changes, network requests, API tokens or downloads.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import wave
from pathlib import Path

from dodaios.plugins.stt.nemotron_local import NemotronLocalSTT
from dodaios.plugins.tts.piper_local import PiperLocalTTS
from dodaios.speech.local_models import NEMOTRON_MODEL_ID, sherpa_model_dir, bundle_present

MODEL = "vits-piper-fr_FR-siwis-medium"
SIWIS = Path("/tmp/dodaios-siwis") / MODEL
INPUT = Path("/tmp/dodaios-fr-micro.wav")
OUTPUT = Path("/tmp/dodaios-fr-reponse.wav")
NEMOTRON_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def find_nemotron() -> Path:
    override = os.environ.get("DODAIOS_NEMOTRON_DIR", "").strip()
    locations = ([Path(override).expanduser()] if override else [
        Path.home() / ".dodaios/data/models" / NEMOTRON_MODEL_ID,
    ])
    for folder in locations:
        if folder.is_dir() and all((folder / name).is_file() for name in NEMOTRON_FILES):
            return folder
    raise RuntimeError("Installed Nemotron model not found; do not download another copy.")


def read_wav() -> tuple[bytes, int]:
    with wave.open(str(INPUT), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16_000:
            raise RuntimeError("Input must be 16kHz, 16-bit mono PCM WAV")
        pcm = wav.readframes(wav.getnframes())
    if not pcm:
        raise RuntimeError("Input WAV has no samples")
    return pcm, 16_000


async def main() -> None:
    pcm, rate = read_wav()
    nemotron_dir = find_nemotron()
    if not SIWIS.is_dir():
        raise RuntimeError(f"Siwis missing: {SIWIS}")
    with tempfile.TemporaryDirectory(prefix="dodaios-fr-end-to-end-") as root:
        os.environ["JARVIS__MEMORY__DATA_DIR"] = root
        link = sherpa_model_dir(MODEL)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(SIWIS, target_is_directory=True)
        if not bundle_present(MODEL):
            raise RuntimeError("Siwis not discovered")
        stt = NemotronLocalSTT(model_dir=nemotron_dir, num_threads=2)
        tts = PiperLocalTTS(gender="feminine", speed=0.90)
        try:
            transcript = await stt.transcribe_pcm(pcm, sample_rate=rate, language="fr")
            print("Nemotron understood:", repr(transcript.text), flush=True)
            if not transcript.text.strip():
                raise RuntimeError("Nemotron returned an empty transcript")
            # Deliberately do NOT invoke Claude. This proves audio I/O and TTS
            # compatibility without touching cloud credentials.
            answer = "Je t'ai entendu. Tu peux maintenant me parler en français."
            frames = []
            output_rate = None
            async for chunk in tts.synthesize(answer, language_code="fr-FR"):
                frames.append(chunk.pcm)
                if output_rate is not None and chunk.sample_rate != output_rate:
                    raise RuntimeError("TTS sample rate changed")
                output_rate = chunk.sample_rate
            if not frames or not output_rate:
                raise RuntimeError("No TTS audio")
            with wave.open(str(OUTPUT), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(output_rate)
                wav.writeframes(b"".join(frames))
            print(f"SUCCESS: {len(frames)} TTS chunks at {output_rate}Hz")
            print(f"Listen: pw-play {OUTPUT}")
        finally:
            await tts.aclose()
            await stt.aclose()
    print("Temporary model link cleaned. Service configuration unchanged.")


if __name__ == "__main__":
    asyncio.run(main())

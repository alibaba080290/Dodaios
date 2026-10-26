#!/usr/bin/env python3
"""Exercise the real Dodaios PiperLocalTTS adapter with /tmp Siwis model.

Run using the existing 3.13 Dodaios venv, with PYTHONPATH set to branch.
No runtime configuration, installed model directory, service, or DB changes.
"""
import asyncio
import wave
from pathlib import Path

import sherpa_onnx

from dodaios.plugins.tts.piper_local import PiperLocalTTS

VOICE = "vits-piper-fr_FR-siwis-medium"
ROOT = Path("/tmp/dodaios-siwis/vits-piper-fr_FR-siwis-medium")
OUT = Path("/tmp/dodaios-piper-adapter-fr.wav")
TEXT = "Bonjour, je suis Dodaios. La reconnaissance et la synthèse vocale française fonctionnent."


class IsolatedSiwis(PiperLocalTTS):
    def _resolve_voice(self, voice, language_code):
        if language_code != "fr-FR":
            raise RuntimeError("This smoke test only permits fr-FR.")
        return VOICE

    def _build_engine(self, model_id):
        if model_id != VOICE:
            raise RuntimeError("Unexpected model")
        model = ROOT / "fr_FR-siwis-medium.onnx"
        tokens = ROOT / "tokens.txt"
        data_dir = ROOT / "espeak-ng-data"
        for path in (model, tokens, data_dir):
            if not path.exists():
                raise RuntimeError(f"Missing model file: {path}")
        return sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(model), tokens=str(tokens),
                        data_dir=str(data_dir),
                    ),
                    num_threads=2,
                    provider="cpu",
                )
            )
        )


async def run():
    adapter = IsolatedSiwis(speed=0.90)
    chunks = []
    rate = None
    try:
        async for chunk in adapter.synthesize(TEXT, language_code="fr-FR"):
            if rate is not None and chunk.sample_rate != rate:
                raise RuntimeError("Inconsistent sample rate between chunks")
            rate = chunk.sample_rate
            chunks.append(chunk.pcm)
    finally:
        await adapter.aclose()
    if not chunks or not rate:
        raise RuntimeError("No audio chunks")
    pcm = b"".join(chunks)
    with wave.open(str(OUT), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    print(f"SUCCESS: {len(chunks)} chunk(s), {len(pcm) / (rate * 2):.2f}s, {rate}Hz")
    print(f"Output: {OUT}")
    print(f"Play: pw-play {OUT}")


if __name__ == "__main__":
    asyncio.run(run())

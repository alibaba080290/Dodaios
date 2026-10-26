#!/usr/bin/env python3
"""Isolated Piper/sherpa-onnx 2.0.1 real-audio smoke test.

Uses one already-installed English voice; no network, service changes,
downloads, or microphone access. Output: /tmp/dodaios-piper-probe.wav.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import sherpa_onnx


def main() -> int:
    directory = (
        Path.home()
        / ".dodaios/data/models/vits-piper-en_US-ryan-medium"
    )
    models = sorted(directory.glob("*.onnx"))
    tokens = directory / "tokens.txt"
    if len(models) != 1 or not tokens.is_file():
        print("Ryan voice model or tokens missing; no audio generated.")
        return 1

    # Test IPA for English "hello". These are phonetic Unicode codepoints,
    # NOT model token IDs or a complete text-to-phoneme conversion solution.
    ipa = "həloʊ"
    available = {
        line.rsplit(" ", 1)[0]
        for line in tokens.read_text(encoding="utf-8").splitlines()
        if " " in line
    }
    unsupported = [char for char in ipa if char not in available]
    if unsupported:
        print("IPA phonemes not in voice tokens:", unsupported)
        return 2

    model = sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(models[0]),
            tokens=str(tokens),
            data_dir=str(directory / "espeak-ng-data"),
        ),
        num_threads=2,
        provider="cpu",
    )
    engine = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(model=model)
    )
    config = sherpa_onnx.GenerationConfig()
    config.phoneme_codepoints = [[ord(c) for c in ipa]]
    print("Generating Ryan test audio using IPA:", ipa, flush=True)
    audio = engine.generate("hello", config)
    samples = np.asarray(audio.samples, dtype=np.float32)
    rate = int(audio.sample_rate)
    if not samples.size or rate <= 0:
        print("FAILED: no audio samples produced")
        return 3

    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    target = Path("/tmp/dodaios-piper-probe.wav")
    with wave.open(str(target), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    print(f"SUCCESS: {target} | {len(samples) / rate:.2f}s | {rate} Hz")
    print("Listen manually using: aplay /tmp/dodaios-piper-probe.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

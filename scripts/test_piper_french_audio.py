#!/usr/bin/env python3
"""Isolated French Siwis Piper smoke test for sherpa-onnx 2.x.

Requires the official archive unpacked at
  /tmp/dodaios-siwis/vits-piper-fr_FR-siwis-medium
No service edits, model downloads, or external phonemizer used.
"""
from pathlib import Path
import sys
import wave

import numpy as np
import sherpa_onnx


def main() -> int:
    model_dir = Path("/tmp/dodaios-siwis/vits-piper-fr_FR-siwis-medium")
    onnx = model_dir / "fr_FR-siwis-medium.onnx"
    tokens = model_dir / "tokens.txt"
    if not onnx.is_file() or not tokens.is_file():
        print(f"Missing Siwis model or tokens in {model_dir}")
        return 1

    # IPA for "bonjour", solely for verifying native inference.
    # Not an automatic text-to-phoneme solution.
    ipa = "bɔ̃ʒuʁ"
    charset = {
        line.rsplit(" ", 1)[0]
        for line in tokens.read_text(encoding="utf-8").splitlines()
        if " " in line
    }
    missing = [c for c in ipa if c not in charset]
    if missing:
        print("IPA characters absent from tokens:", missing)
        return 2

    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(onnx), tokens=str(tokens),
                data_dir=str(model_dir / "espeak-ng-data"),
            ),
            num_threads=2,
            provider="cpu",
        )
    )
    engine = sherpa_onnx.OfflineTts(cfg)
    options = sherpa_onnx.GenerationConfig()
    options.phoneme_codepoints = [[ord(c) for c in ipa]]
    audio = engine.generate("bonjour", options)
    samples = np.asarray(audio.samples, dtype=np.float32)
    rate = int(audio.sample_rate)
    if samples.size == 0 or rate <= 0:
        print("FAILED: no audio samples")
        return 3

    dest = Path("/tmp/dodaios-siwis-bonjour.wav")
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())

    print(f"SUCCESS: {dest} ({samples.size / rate:.2f}s, {rate} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""French eSpeak-NG -> sherpa-onnx 2.x -> Siwis smoke test.

Standalone: no Dodaios runtime/config edits. Generates WAV under /tmp.
Requires espeak-ng and Siwis archive already unpacked at /tmp/dodaios-siwis.
"""
from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

MODEL_DIR = Path("/tmp/dodaios-siwis/vits-piper-fr_FR-siwis-medium")
OUTPUT = Path("/tmp/dodaios-siwis-phrase.wav")
DEFAULT_TEXT = "Bonjour, comment vas-tu aujourd'hui ?"


def phonemize(text: str) -> str:
    result = subprocess.run(
        ["espeak-ng", "-q", "-v", "fr", "--ipa", text],
        capture_output=True, text=True, timeout=10, check=True,
    )
    # eSpeak renders prosodic marks and whitespace. Sherpa's Piper model
    # needs phonemes, not text formatting, for its token mapping.
    return "".join(
        c for c in result.stdout
        if not c.isspace() and c not in "ˈˌ"
    )


def main() -> int:
    text = " ".join(sys.argv[1:]) or DEFAULT_TEXT
    onnx = MODEL_DIR / "fr_FR-siwis-medium.onnx"
    tokens = MODEL_DIR / "tokens.txt"
    if not onnx.is_file() or not tokens.is_file():
        print("Siwis model not found. Run the earlier isolated installation.")
        return 1
    ipa = phonemize(text)
    print("Text:", text)
    print("IPA:", ipa)
    token_lines = tokens.read_text(encoding="utf-8").splitlines()
    available = {
        line.rsplit(" ", 1)[0]
        for line in token_lines if " " in line
    }
    unknown = sorted(set(ipa) - available)
    if unknown:
        print("ABORT: phoneme characters missing from Siwis tokens:", unknown)
        return 2
    if not ipa:
        print("ABORT: eSpeak returned no phonemes")
        return 3

    engine = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(onnx), tokens=str(tokens),
                    data_dir=str(MODEL_DIR / "espeak-ng-data"),
                ),
                num_threads=2,
                provider="cpu",
            )
        )
    )
    cfg = sherpa_onnx.GenerationConfig()
    cfg.phoneme_codepoints = [[ord(ch) for ch in ipa]]
    cfg.speed = 0.70  # slower speech than default 1.0; listening test
    print("Generating at speed 0.70...", flush=True)
    audio = engine.generate(text, cfg)
    samples = np.asarray(audio.samples, dtype=np.float32)
    rate = int(audio.sample_rate)
    if not samples.size or rate <= 0:
        print("FAIL: no audio generated")
        return 4

    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(OUTPUT), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    print(f"SUCCESS: {OUTPUT} ({len(samples) / rate:.2f}s at {rate}Hz)")
    print(f"Listen: pw-play {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

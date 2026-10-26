#!/usr/bin/env python3
"""Read-only probe of Piper phoneme tokens on the installed sherpa-onnx runtime.

Uses no model inference, downloads or config changes. Run under Dodaios venv.
"""
from pathlib import Path
import sherpa_onnx


def main() -> None:
    root = Path.home() / ".dodaios/data/models"
    cfg = sherpa_onnx.GenerationConfig()
    print("GenerationConfig fields:", [x for x in dir(cfg) if not x.startswith("_")])
    print("phoneme_codepoints type:", type(cfg.phoneme_codepoints).__name__)
    try:
        cfg.phoneme_codepoints = [ord(ch) for ch in "bɔ̃ʒuʁ"]
        print("codepoint assignment: accepted", cfg.phoneme_codepoints)
        print("round trip:", "".join(map(chr, cfg.phoneme_codepoints)))
    except (TypeError, ValueError, AttributeError) as exc:
        print("codepoint assignment rejected:", exc)
    for directory in sorted(root.glob("vits-piper-*")):
        tokens = directory / "tokens.txt"
        if not tokens.exists():
            continue
        lines = tokens.read_text(encoding="utf-8").splitlines()
        print("\nvoice:", directory.name, "token count:", len(lines))
        print("first 24 token lines:", lines[:24])
        print("unicode lines for French phonemes:", [
            x for x in lines if any(c in x for c in ("ɔ", "̃", "ʒ", "ʁ"))
        ][:16])
        onnx = list(directory.glob("*.onnx"))
        print("onnx models:", [p.name for p in onnx])


if __name__ == "__main__":
    main()

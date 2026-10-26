#!/usr/bin/env python3
"""Read-only diagnostics for Piper with sherpa-onnx 2.x.

Run with the same Python interpreter as Dodaios:
  ~/.dodaios/.venv/bin/python scripts/diagnose_piper_runtime.py

No model loads, network requests, downloads, or configuration writes.
"""
from __future__ import annotations

import inspect
from importlib.metadata import PackageNotFoundError, version


def main() -> int:
    for name in ("sherpa-onnx", "sherpa-onnx-core"):
        try:
            print(f"{name}: {version(name)}")
        except PackageNotFoundError:
            print(f"{name}: not installed")

    try:
        import sherpa_onnx
    except ImportError as exc:
        print(f"Cannot import sherpa_onnx: {exc}")
        return 1

    for name in (
        "OfflineTts",
        "OfflineTtsGenerationConfig",
        "OfflineTtsVitsModelConfig",
    ):
        obj = getattr(sherpa_onnx, name, None)
        print(f"\n{name}: {'present' if obj is not None else 'absent'}")
        if obj is None:
            continue
        try:
            print("constructor/signature:", inspect.signature(obj))
        except (TypeError, ValueError):
            print("constructor/signature: not exposed by this binding")
        if name == "OfflineTts":
            method = getattr(obj, "generate", None)
            if method is not None:
                print("generate docstring:", inspect.getdoc(method) or "<none>")
        elif name == "OfflineTtsGenerationConfig":
            print("generation config documentation:", inspect.getdoc(obj) or "<none>")
            print(
                "phoneme_codepoints field:",
                hasattr(obj, "phoneme_codepoints")
                or "phoneme_codepoints" in (inspect.getdoc(obj) or ""),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

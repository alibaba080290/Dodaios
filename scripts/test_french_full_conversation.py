#!/usr/bin/env python3
"""Isolated end-to-end voice test: WAV -> Nemotron -> Claude CLI -> Siwis.

No production config/service/model changes. Uses the existing microphone WAV,
local Nemotron and Siwis model files, and user's Claude subscription CLI.
Claude is invoked from an empty temporary working directory with tools and
MCP disabled; a real subscription request is consumed.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
import wave
from pathlib import Path

from dodaios.plugins.stt.nemotron_local import NemotronLocalSTT
from dodaios.plugins.tts.piper_local import PiperLocalTTS
from dodaios.speech.local_models import NEMOTRON_MODEL_ID, bundle_present, sherpa_model_dir

MODEL = "vits-piper-fr_FR-siwis-medium"
SIWIS = Path("/tmp/dodaios-siwis") / MODEL
INPUT = Path("/tmp/dodaios-fr-micro.wav")
OUTPUT = Path("/tmp/dodaios-conversation-fr.wav")
NEMOTRON_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def find_nemotron() -> Path:
    env_path = os.environ.get("DODAIOS_NEMOTRON_DIR", "").strip()
    folder = (Path(env_path).expanduser() if env_path else
              Path.home() / ".dodaios/data/models" / NEMOTRON_MODEL_ID)
    if not all((folder / name).is_file() for name in NEMOTRON_FILES):
        raise RuntimeError(f"Existing Nemotron model not found: {folder}")
    return folder


def read_microphone_wav() -> bytes:
    with wave.open(str(INPUT), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise RuntimeError("Expected a 16 kHz mono signed 16-bit WAV")
        pcm = wav.readframes(wav.getnframes())
    if not pcm:
        raise RuntimeError("Microphone recording contains no audio")
    return pcm


async def ask_claude(question: str, cwd: str) -> str:
    binary = shutil.which("claude")
    if not binary:
        raise RuntimeError("Claude CLI is not on PATH")
    # Keep agent tools, MCP and local project context disabled. Using an
    # isolated working directory prevents accidental repo indexing.
    cmd = [
        binary, "-p", "--tools", "", "--strict-mcp-config",
        "--no-session-persistence",
        "--system-prompt",
        "Tu es Dodaios, un assistant vocal. Réponds exclusivement en français, "
        "en une ou deux phrases courtes, sans Markdown. N'utilise aucun outil.",
        question,
    ]
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=90)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError("Claude CLI timed out after 90 seconds") from None
    if proc.returncode != 0:
        # Avoid echoing potentially sensitive auth or environment data.
        raise RuntimeError(f"Claude CLI exit code {proc.returncode}; inspect CLI login")
    answer = stdout.decode("utf-8", errors="replace").strip()
    if not answer:
        raise RuntimeError("Claude CLI returned no text")
    if len(answer) > 1200:
        raise RuntimeError("Claude response too long for voice smoke test")
    print(f"Claude ({time.monotonic()-start:.1f}s): {answer}", flush=True)
    return answer


async def run() -> None:
    pcm = read_microphone_wav()
    if not SIWIS.is_dir():
        raise RuntimeError(f"Siwis model missing: {SIWIS}")
    model_path = find_nemotron()
    with tempfile.TemporaryDirectory(prefix="dodaios-voice-full-") as root:
        os.environ["JARVIS__MEMORY__DATA_DIR"] = root
        link = sherpa_model_dir(MODEL)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(SIWIS, target_is_directory=True)
        if not bundle_present(MODEL):
            raise RuntimeError("Siwis model discovery failed")
        stt = NemotronLocalSTT(model_dir=model_path, num_threads=2)
        tts = PiperLocalTTS(gender="feminine", speed=0.90)
        started = time.monotonic()
        try:
            transcript = await stt.transcribe_pcm(pcm, sample_rate=16000, language="fr")
            question = transcript.text.strip()
            print(f"Nemotron ({time.monotonic()-started:.1f}s): {question!r}", flush=True)
            if not question:
                raise RuntimeError("Empty microphone transcription")
            answer = await ask_claude(question, root)
            frames = []
            rate = None
            async for chunk in tts.synthesize(answer, language_code="fr-FR"):
                if rate is not None and rate != chunk.sample_rate:
                    raise RuntimeError("Piper changed audio sample rate")
                rate = chunk.sample_rate
                frames.append(chunk.pcm)
            if not rate or not frames:
                raise RuntimeError("Piper produced no audio")
            with wave.open(str(OUTPUT), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(b"".join(frames))
            print(f"SUCCESS: {len(frames)} audio chunks, "
                  f"{sum(map(len,frames))/(2*rate):.2f}s speech, "
                  f"{time.monotonic()-started:.1f}s end-to-end", flush=True)
            print(f"Listen: pw-play {OUTPUT}", flush=True)
        finally:
            await tts.aclose()
            await stt.aclose()
    print("Temporary data removed; active Dodaios service unchanged.")


if __name__ == "__main__":
    asyncio.run(run())

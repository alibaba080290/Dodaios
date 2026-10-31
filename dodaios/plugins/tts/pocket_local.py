"""On-device text-to-speech through a local Kyutai Pocket TTS server.

Pocket TTS runs as its own process (``pocket-tts serve``), bound to the
loopback interface, and this provider talks to it over HTTP. Keeping the model
out of the Dodaios process keeps torch out of the voice pipeline's memory and
lets the server be restarted on its own.

The server API, checked against pocket-tts 4.0.1 (``pocket_tts/main.py``):

* ``GET /health`` returns ``{"status": "healthy"}`` once the model is loaded.
* ``POST /tts`` takes a multipart form: ``text`` (required) and an optional
  ``voice_url`` naming a built-in voice. It streams ``audio/wav``: a 44-byte
  header (mono, 16-bit PCM, 24 kHz for the released models) whose size fields
  are placeholders because the stream is not seekable, then raw PCM.

Decisions worth keeping:

**No remote fetch from here.** ``voice_url`` also accepts http(s) and hf://
URLs, which make the server download audio. This provider only ever sends the
name of a built-in voice, so nothing it does can reach the network.

**One model speaks one language.** A server loaded with ``--language french``
speaks French only. A turn in another language raises before any audio, so a
``[tts].fallback`` such as ``piper-local`` speaks it instead of a French accent
reading English.

The server address comes from the provider card (stored as
``[brain.providers.pocket-local].base_url``), else ``[tts.pocket].url``, else
the default below. Only loopback addresses are used. Other settings live in the
optional ``[tts.pocket]`` table::

    [tts.pocket]
    url = "http://127.0.0.1:18765"
    language = "fr"
    connect_timeout_s = 2.0
    read_timeout_s = 20.0
"""

from __future__ import annotations

import logging
import struct
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import httpx

from dodaios.core.protocols import AudioChunk

log = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:18765"
DEFAULT_LANGUAGE = "fr"
DEFAULT_CONNECT_TIMEOUT_S = 2.0
#: Longest wait for the next bytes of a stream. The first audio of a clause
#: arrives after the server has buffered a short opening chunk, well under this.
DEFAULT_READ_TIMEOUT_S = 20.0
WARMUP_TEXT = "Bonjour."

#: Built-in voices of pocket-tts 4.0.1 (``_ORIGINS_OF_PREDEFINED_VOICES``).
#: Every language model, ``french`` included, ships one embedding per name.
POCKET_VOICES: frozenset[str] = frozenset(
    {
        "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles",
        "cosette", "daan", "eponine", "estelle", "eve", "fantine", "george",
        "giovanni", "jane", "javert", "jean", "juergen", "lola", "marius",
        "mary", "michael", "paul", "peter_yearsley", "rafael", "stuart_bell",
        "vera",
    }
)

#: The voice the server uses for a language when a request names none.
DEFAULT_VOICE_FOR_LANGUAGE = {
    "fr": "estelle", "de": "juergen", "es": "lola", "it": "giovanni",
    "pt": "rafael", "nl": "daan", "en": "alba",
}

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class PocketTTSError(RuntimeError):
    """The Pocket TTS server could not produce audio for this request."""


class PocketTTSUnavailable(PocketTTSError):
    """The Pocket TTS server is not reachable."""


def _card_base_url() -> str | None:
    """The server URL saved on the provider card, if any. Never raises."""
    try:
        from dodaios.core import config as _cfg  # noqa: PLC0415 - avoid import cycle

        provider = _cfg.load_config().brain.providers.get(PocketLocalTTS.name)
    except Exception:  # noqa: BLE001 - a config problem must not stop the voice
        return None
    return (getattr(provider, "base_url", None) or "").strip() or None


def _primary_language(code: str | None) -> str:
    return (code or "").strip().lower().replace("_", "-").split("-", 1)[0]


class _WavStream:
    """Incremental parser for the streamed WAV: header first, then PCM.

    The size fields are ignored on purpose (the server writes placeholders).
    PCM is handed out in whole frames so a 16-bit sample is never split
    across two chunks.
    """

    def __init__(self) -> None:
        self._buffer = b""
        self._in_data = False
        self.sample_rate = 0
        self.channels = 0
        self._frame_bytes = 0

    def feed(self, data: bytes) -> bytes:
        self._buffer += data
        if not self._in_data:
            self._parse_header()
            if not self._in_data:
                return b""
        usable = len(self._buffer) - len(self._buffer) % self._frame_bytes
        pcm, self._buffer = self._buffer[:usable], self._buffer[usable:]
        return pcm

    def _parse_header(self) -> None:
        buf = self._buffer
        if len(buf) < 12:
            return
        if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
            raise PocketTTSError("Pocket TTS did not return a WAV stream.")
        pos = 12
        while len(buf) >= pos + 8:
            chunk_id = buf[pos : pos + 4]
            size = struct.unpack("<I", buf[pos + 4 : pos + 8])[0]
            body = pos + 8
            if chunk_id == b"data":
                if not self.sample_rate:
                    raise PocketTTSError("Pocket TTS sent audio before its format.")
                self._buffer = buf[body:]
                self._in_data = True
                return
            if len(buf) < body + size:
                return
            if chunk_id == b"fmt ":
                fmt, channels, rate, _, _, bits = struct.unpack("<HHIIHH", buf[body : body + 16])
                if fmt != 1 or bits != 16:
                    raise PocketTTSError(
                        f"Pocket TTS sent unsupported audio (format {fmt}, {bits}-bit)."
                    )
                if channels != 1 or rate <= 0:
                    raise PocketTTSError(
                        f"Pocket TTS sent {channels} channel(s) at {rate} Hz; expected mono."
                    )
                self.sample_rate = rate
                self.channels = channels
                self._frame_bytes = 2 * channels
            pos = body + size + (size & 1)


class PocketLocalTTS:
    """Natural-sounding local voice served by ``pocket-tts serve`` on loopback."""

    name = "pocket-local"
    supports_streaming = True
    #: Which voice spoke last — read by the per-turn transcript label.
    last_voice: str | None = None
    last_voice_provider: str | None = None
    #: Nothing spoken here leaves the machine (the server is loopback-only).
    runs_on_device = True

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_URL,
        voice: str | None = None,
        language: str = DEFAULT_LANGUAGE,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        sync_transport: httpx.BaseTransport | None = None,
    ) -> None:
        url = (base_url or DEFAULT_URL).strip().rstrip("/")
        #: Set when the configured address is not on this machine. Construction
        #: still succeeds so the TTS factory keeps the fallback voice working.
        self._refusal: str | None = None
        if (urlsplit(url).hostname or "") not in _LOOPBACK_HOSTS:
            self._refusal = (
                f"Pocket TTS must run on this machine; {url!r} is not a loopback address."
            )
            log.error("%s", self._refusal)
        self._base_url = url
        self._language = _primary_language(language) or DEFAULT_LANGUAGE
        requested = (voice or "").strip().lower()
        self._voice = requested if requested in POCKET_VOICES else None
        if requested and self._voice is None:
            log.warning("Unknown Pocket TTS voice %r; using the server default.", voice)
        self._timeout = httpx.Timeout(
            connect=float(connect_timeout_s),
            read=float(read_timeout_s),
            write=float(connect_timeout_s),
            pool=float(connect_timeout_s),
        )
        self._transport = transport
        self._sync_transport = sync_transport
        self._status: dict[str, Any] = {
            "state": "unknown",
            "url": self._base_url,
            "language": self._language,
            "voice": self.default_voice,
            "detail": "",
            "checked_at": None,
            "warmup_ms": None,
        }
        if self._refusal:
            self._set_status("error", self._refusal)

    @classmethod
    def from_config(cls, tts_cfg: Any) -> PocketLocalTTS:
        """Build from ``[tts]``: the voice from voice_de/voice_en, the rest from ``[tts.pocket]``."""
        extra = getattr(tts_cfg, "model_extra", None) or {}
        section = extra.get("pocket") or {}
        if not isinstance(section, dict):
            section = {}
        voice = next(
            (
                str(v).strip().lower()
                for v in (getattr(tts_cfg, "voice_de", None), getattr(tts_cfg, "voice_en", None))
                if v and str(v).strip().lower() in POCKET_VOICES
            ),
            None,
        )
        return cls(
            base_url=_card_base_url() or str(section.get("url") or DEFAULT_URL),
            voice=voice,
            language=str(section.get("language") or DEFAULT_LANGUAGE),
            connect_timeout_s=float(section.get("connect_timeout_s") or DEFAULT_CONNECT_TIMEOUT_S),
            read_timeout_s=float(section.get("read_timeout_s") or DEFAULT_READ_TIMEOUT_S),
        )

    # -- voices -------------------------------------------------------------
    @property
    def default_voice(self) -> str:
        return self._voice or DEFAULT_VOICE_FOR_LANGUAGE.get(self._language, "alba")

    def list_voices(self, language: str | None = None) -> list[str]:
        """Built-in voices; none for a language the loaded model does not speak."""
        if language and _primary_language(language) != self._language:
            return []
        return sorted(POCKET_VOICES)

    def _resolve_voice(self, voice: str | None) -> str | None:
        """A built-in voice name, or None to let the server use its default."""
        requested = (voice or "").strip().lower()
        if requested in POCKET_VOICES:
            return requested
        return self._voice

    def _check_language(self, language_code: str | None) -> None:
        code = _primary_language(language_code)
        if code and code != "auto" and code != self._language:
            raise PocketTTSError(
                f"The Pocket TTS server speaks {self._language!r}, not {code!r}."
            )

    # -- status -------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Last known server state: unknown, ready, unreachable or error."""
        return dict(self._status)

    def _set_status(self, state: str, detail: str = "", **extra: Any) -> None:
        self._status.update(state=state, detail=detail, checked_at=time.time(), **extra)

    def _sync_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._base_url, timeout=self._timeout, transport=self._sync_transport
        )

    def check_health(self) -> bool:
        """Ask ``GET /health``; updates :meth:`status`. Never raises."""
        if self._refusal:
            return False
        try:
            with self._sync_client() as client:
                response = client.get("/health")
            healthy = response.status_code == 200 and response.json().get("status") == "healthy"
        except (httpx.HTTPError, ValueError) as exc:
            self._set_status("unreachable", f"{type(exc).__name__}: {exc}")
            return False
        if healthy:
            self._set_status("ready")
        else:
            self._set_status("error", f"/health answered HTTP {response.status_code}")
        return healthy

    def _ensure_client(self) -> None:
        """Warm-up hook: health check, then one short synthesis. Never raises.

        Synchronous on purpose: the pipeline runs it with ``asyncio.to_thread``
        and ``FallbackTTS`` calls it directly, so a coroutine here would never run.
        """
        if not self.check_health():
            log.warning(
                "Pocket TTS server not reachable at %s (%s); the fallback voice will "
                "speak until it is up.",
                self._base_url, self._status["detail"],
            )
            return
        start = time.perf_counter()
        form = {"text": WARMUP_TEXT}
        if self._voice:
            form["voice_url"] = self._voice
        try:
            with self._sync_client() as client:
                with client.stream("POST", "/tts", data=form) as response:
                    if response.status_code != 200:
                        raise PocketTTSError(f"HTTP {response.status_code}")
                    for _ in response.iter_bytes():
                        pass
        except (httpx.HTTPError, PocketTTSError) as exc:
            self._set_status("error", f"warm-up failed: {exc}")
            log.warning("Pocket TTS warm-up failed: %s", exc)
            return
        warm_ms = round((time.perf_counter() - start) * 1000.0, 1)
        self._set_status("ready", warmup_ms=warm_ms)
        log.info("Pocket TTS ready at %s (warm-up %.0f ms)", self._base_url, warm_ms)

    # -- synthesis ----------------------------------------------------------
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language_code: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """Stream 16-bit mono PCM at the server's sample rate as it is generated."""
        spoken = (text or "").strip()
        if not spoken:
            return
        if self._refusal:
            raise PocketTTSError(self._refusal)
        self._check_language(language_code)
        chosen = self._resolve_voice(voice)
        form = {"text": spoken}
        if chosen:
            form["voice_url"] = chosen
        wav = _WavStream()
        emitted = False
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=self._timeout, transport=self._transport
            ) as client:
                async with client.stream("POST", "/tts", data=form) as response:
                    if response.status_code != 200:
                        body = (await response.aread())[:200].decode("utf-8", "replace")
                        raise PocketTTSError(
                            f"Pocket TTS answered HTTP {response.status_code}: {body}"
                        )
                    self.last_voice = chosen or self.default_voice
                    self.last_voice_provider = self.name
                    async for raw in response.aiter_bytes():
                        pcm = wav.feed(raw)
                        if pcm:
                            emitted = True
                            yield AudioChunk(
                                pcm=pcm,
                                sample_rate=wav.sample_rate,
                                timestamp_ns=0,
                                channels=wav.channels,
                            )
        except httpx.ConnectError as exc:
            self._set_status("unreachable", str(exc))
            raise PocketTTSUnavailable(
                f"Pocket TTS server not reachable at {self._base_url}. Start it with "
                "'systemctl --user start dodaios-pocket-tts'."
            ) from exc
        except httpx.TimeoutException as exc:
            self._set_status("error", f"timeout: {type(exc).__name__}")
            raise PocketTTSError(f"Pocket TTS timed out ({type(exc).__name__}).") from exc
        except httpx.HTTPError as exc:
            self._set_status("error", str(exc))
            raise PocketTTSError(f"Pocket TTS request failed: {exc}") from exc
        if not emitted:
            raise PocketTTSError("Pocket TTS returned no audio.")
        self._set_status("ready")

    async def aclose(self) -> None:
        """Nothing to release: every request opens and closes its own connection."""
        return None

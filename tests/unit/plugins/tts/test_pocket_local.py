"""Pocket TTS provider: HTTP contract, audio format, failure modes, wiring."""

from __future__ import annotations

import asyncio
import os
import struct
import tomllib

import httpx
import numpy as np
import pytest

from dodaios.core.protocols import AudioChunk
from dodaios.plugins.tts.pocket_local import (
    POCKET_VOICES,
    PocketLocalTTS,
    PocketTTSError,
    PocketTTSUnavailable,
    _WavStream,
)

RATE = 24000


def _wav_header(rate: int = RATE, channels: int = 1, bits: int = 16) -> bytes:
    """The header Python's wave module writes for an unseekable stream."""
    block = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 0x77359424) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * block, block, bits)
        + b"data" + struct.pack("<I", 0x77359400)
    )


def _pcm(n: int) -> bytes:
    return (np.arange(n, dtype=np.int16) % 1000).tobytes()


class _Body(httpx.AsyncByteStream):
    """Response body delivered in caller-chosen pieces; records being closed."""

    def __init__(self, pieces: list[bytes], delay: float = 0.0) -> None:
        self.pieces = pieces
        self.delay = delay
        self.closed = False

    async def __aiter__(self):
        for piece in self.pieces:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield piece

    async def aclose(self) -> None:
        self.closed = True


def _provider(handler, **kwargs) -> PocketLocalTTS:
    return PocketLocalTTS(transport=httpx.MockTransport(handler), **kwargs)


async def _collect(tts: PocketLocalTTS, text: str = "Bonjour.", **kwargs) -> list[AudioChunk]:
    return [chunk async for chunk in tts.synthesize(text, **kwargs)]


def _form(request: httpx.Request) -> str:
    return request.read().decode("utf-8", "replace")


# -- audio --------------------------------------------------------------------
def test_wav_stream_reads_rate_from_header_and_keeps_frames_whole():
    raw = _wav_header(22050) + _pcm(101)
    parser = _WavStream()
    out = b""
    for i in range(0, len(raw), 7):  # odd-sized pieces split header and samples
        out += parser.feed(raw[i : i + 7])
        assert len(out) % 2 == 0
    assert parser.sample_rate == 22050
    assert parser.channels == 1
    assert out == _pcm(101)


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (b"NOPE" * 4, "WAV"),
        (_wav_header(bits=8), "unsupported"),
        (_wav_header(channels=2), "mono"),
    ],
)
def test_wav_stream_rejects_unexpected_audio(header, message):
    with pytest.raises(PocketTTSError, match=message):
        _WavStream().feed(header + b"\0" * 16)


def test_synthesize_streams_int16_mono_at_server_rate():
    body = _wav_header() + _pcm(48000)
    pieces = [body[:30], body[30:4001], body[4001:]]

    def handler(request):
        assert request.method == "POST" and request.url.path == "/tts"
        return httpx.Response(200, headers={"content-type": "audio/wav"}, stream=_Body(pieces))

    tts = _provider(handler)
    chunks = asyncio.run(_collect(tts, language_code="fr-FR"))
    assert chunks and all(c.sample_rate == RATE and c.channels == 1 for c in chunks)
    assert all(len(c.pcm) % 2 == 0 for c in chunks)
    assert b"".join(c.pcm for c in chunks) == _pcm(48000)
    assert tts.last_voice == "estelle" and tts.last_voice_provider == "pocket-local"
    assert tts.status()["state"] == "ready"


# -- voices and language ------------------------------------------------------
def test_voice_names_are_sent_but_urls_never_are():
    forms: list[str] = []

    def handler(request):
        forms.append(_form(request))
        return httpx.Response(200, stream=_Body([_wav_header() + _pcm(10)]))

    tts = _provider(handler, voice="charles")
    asyncio.run(_collect(tts))
    asyncio.run(_collect(tts, voice="fantine"))
    asyncio.run(_collect(tts, voice="https://example.com/x.wav"))
    assert "voice_url=charles" in forms[0]
    assert "voice_url=fantine" in forms[1]
    assert "example.com" not in forms[2] and "charles" in forms[2]


def test_unknown_configured_voice_lets_the_server_choose():
    forms: list[str] = []

    def handler(request):
        forms.append(_form(request))
        return httpx.Response(200, stream=_Body([_wav_header() + _pcm(10)]))

    tts = _provider(handler, voice="hf://someone/voice.wav")
    asyncio.run(_collect(tts))
    assert "voice_url" not in forms[0]
    assert tts.default_voice == "estelle"


def test_other_language_raises_before_any_request():
    def handler(request):  # pragma: no cover - must not be reached
        raise AssertionError("no request expected")

    with pytest.raises(PocketTTSError, match="'fr', not 'en'"):
        asyncio.run(_collect(_provider(handler), language_code="en-US"))
    assert _provider(handler).list_voices("en") == []
    assert set(_provider(handler).list_voices("fr-FR")) == POCKET_VOICES


def test_empty_text_makes_no_request():
    def handler(request):  # pragma: no cover
        raise AssertionError("no request expected")

    assert asyncio.run(_collect(_provider(handler), text="   ")) == []


def test_non_loopback_url_is_refused_without_breaking_the_factory():
    tts = PocketLocalTTS(base_url="http://192.168.1.10:8000")
    assert tts.status()["state"] == "error"
    assert tts.check_health() is False
    with pytest.raises(PocketTTSError, match="loopback"):
        asyncio.run(_collect(tts))


# -- failures -----------------------------------------------------------------
def test_http_error_is_reported_with_status():
    def handler(request):
        return httpx.Response(500, json={"detail": "boom"})

    with pytest.raises(PocketTTSError, match="HTTP 500"):
        asyncio.run(_collect(_provider(handler)))


def test_server_down_raises_unavailable_and_sets_status():
    def handler(request):
        raise httpx.ConnectError("Connection refused", request=request)

    tts = _provider(handler)
    with pytest.raises(PocketTTSUnavailable, match="not reachable"):
        asyncio.run(_collect(tts))
    assert tts.status()["state"] == "unreachable"


def test_read_timeout_maps_to_provider_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    tts = _provider(handler)
    with pytest.raises(PocketTTSError, match="timed out"):
        asyncio.run(_collect(tts))
    assert tts.status()["state"] == "error"


def test_empty_audio_is_an_error():
    def handler(request):
        return httpx.Response(200, stream=_Body([_wav_header()]))

    with pytest.raises(PocketTTSError, match="no audio"):
        asyncio.run(_collect(_provider(handler)))


def test_cancelling_the_consumer_closes_the_http_stream():
    body = _Body([_wav_header() + _pcm(100)] + [_pcm(100)] * 50, delay=0.01)
    tts = _provider(lambda r: httpx.Response(200, stream=body))

    async def run():
        stream = tts.synthesize("Bonjour.")
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(run())
    assert body.closed


# -- warm-up and health -------------------------------------------------------
def test_warmup_is_synchronous_and_never_raises_when_down():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    tts = PocketLocalTTS(sync_transport=httpx.MockTransport(handler))
    assert tts._ensure_client() is None
    assert tts.check_health() is False
    assert tts.status()["state"] == "unreachable"


def test_warmup_checks_health_then_synthesizes():
    paths: list[str] = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        return httpx.Response(200, content=_wav_header() + _pcm(10))

    tts = PocketLocalTTS(sync_transport=httpx.MockTransport(handler))
    tts._ensure_client()
    assert paths == ["/health", "/tts"]
    status = tts.status()
    assert status["state"] == "ready" and status["warmup_ms"] is not None


# -- fallback -----------------------------------------------------------------
class _FakePiper:
    name = "piper-local"
    supports_streaming = True

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def synthesize(self, text, voice=None, language_code=None):
        self.calls.append(language_code)
        yield AudioChunk(pcm=b"\1\0" * 10, sample_rate=22050, timestamp_ns=0)


@pytest.mark.parametrize("language", ["fr-FR", "en-US"])
def test_fallback_speaks_when_pocket_cannot(language):
    from dodaios.plugins.tts.fallback_tts import FallbackTTS

    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    piper = _FakePiper()
    wrapped = FallbackTTS(_provider(handler), piper)

    async def run():
        return [c async for c in wrapped.synthesize("Bonjour.", language_code=language)]

    chunks = asyncio.run(run())
    assert [c.sample_rate for c in chunks] == [22050]
    assert piper.calls == [language]
    assert wrapped.last_voice_provider == "piper-local"


# -- factory and config -------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_card_url(monkeypatch):
    """Keep the user's real config out of these tests."""
    monkeypatch.setattr("dodaios.plugins.tts.pocket_local._card_base_url", lambda: None)


def test_card_url_wins_over_the_tts_table(monkeypatch):
    monkeypatch.setattr(
        "dodaios.plugins.tts.pocket_local._card_base_url", lambda: "http://localhost:18800"
    )
    cfg = _tts_cfg(provider="pocket-local", pocket={"url": "http://127.0.0.1:18999"})
    assert PocketLocalTTS.from_config(cfg).status()["url"] == "http://localhost:18800"


def _tts_cfg(**fields):
    from dodaios.core.config import TTSConfig

    return TTSConfig(**fields)


@pytest.mark.parametrize("alias", ["pocket-local", "pocket-tts", "pocket"])
def test_factory_builds_pocket_from_config(alias):
    from dodaios.plugins.tts import _build_provider, _canonical_tts_name

    cfg = _tts_cfg(
        provider=alias, voice_de="paul", voice_en="paul",
        pocket={"url": "http://127.0.0.1:18999", "read_timeout_s": 7},
    )
    tts = _build_provider(cfg, alias)
    assert isinstance(tts, PocketLocalTTS)
    assert tts.status()["url"] == "http://127.0.0.1:18999"
    assert tts.default_voice == "paul"
    assert tts._timeout.read == 7
    assert _canonical_tts_name(alias) == "pocket-local"


def test_factory_wraps_pocket_with_piper_fallback_without_loading_models():
    from dodaios.plugins.tts import build_tts_from_config
    from dodaios.plugins.tts.fallback_tts import FallbackTTS
    from dodaios.plugins.tts.piper_local import PiperLocalTTS

    tts = build_tts_from_config(
        _tts_cfg(provider="pocket-local", fallback="piper-local", language_code="fr-FR")
    )
    assert isinstance(tts, FallbackTTS)
    assert isinstance(tts.primary, PocketLocalTTS)
    assert isinstance(tts.fallback, PiperLocalTTS)
    assert tts.primary.default_voice == "estelle"


def test_piper_alone_is_still_built_unchanged():
    from dodaios.plugins.tts import build_tts_from_config
    from dodaios.plugins.tts.piper_local import PiperLocalTTS

    assert isinstance(build_tts_from_config(_tts_cfg(provider="piper-local")), PiperLocalTTS)


def test_switching_to_pocket_writes_french_defaults(tmp_path, monkeypatch):
    from dodaios.core import config_writer

    monkeypatch.setattr(config_writer, "_sync_tts_provider_drift_soll", lambda applied: None)
    monkeypatch.setattr(config_writer, "_update_config_soll_section", lambda top, values: None)
    path = tmp_path / "config.toml"
    path.write_text(
        '[tts]\nprovider = "piper-local"\nvoice_de = "vits-piper-en_US-ryan-medium"\n'
        'voice_en = "vits-piper-en_US-ryan-medium"\nlanguage_code = "fr-FR"\n'
        'fallback = "piper-local"\n',
        encoding="utf-8",
    )
    config_writer.set_tts_provider("pocket-local", path=path)
    tts = tomllib.loads(path.read_text(encoding="utf-8"))["tts"]
    assert tts["provider"] == "pocket-local"
    assert tts["voice_de"] == tts["voice_en"] == "estelle"
    assert tts["language_code"] == "fr-FR"
    assert tts["fallback"] == "piper-local"

    config_writer.set_tts_voice("charles", path=path)
    config_writer.set_tts_provider("pocket-local", path=path)
    tts = tomllib.loads(path.read_text(encoding="utf-8"))["tts"]
    assert tts["voice_de"] == "charles"  # a valid Pocket voice survives a re-select


def test_ui_card_and_voice_picker_are_consistent():
    from dodaios.brain.model_catalog import TTS_CATALOG
    from dodaios.core.config_writer import _TTS_DEFAULTS, _VOICES_FOR_PROVIDER
    from dodaios.costs.speech_rates import LOCAL_SPEECH_PROVIDERS
    from dodaios.ui.web.provider_spec import PROVIDERS

    spec = next(p for p in PROVIDERS if p.id == "pocket-local")
    assert spec.tier == "tts" and spec.auth_mode == "none" and not spec.secret_keys
    assert any(p.id == "piper-local" for p in PROVIDERS)
    selects, voices = TTS_CATALOG["pocket-local"]
    assert selects == "voice"
    ids = [v.id for v in voices]
    assert ids[0] == "estelle"
    assert set(ids) <= POCKET_VOICES == _VOICES_FOR_PROVIDER["pocket-local"]
    assert _TTS_DEFAULTS["pocket-local"]["voice_de"] in ids
    assert "pocket-local" in LOCAL_SPEECH_PROVIDERS


def test_provider_test_button_reports_unreachable_server(monkeypatch):
    from dodaios.brain import provider_test
    from dodaios.ui.web.provider_spec import PROVIDERS

    spec = next(p for p in PROVIDERS if p.id == "pocket-local")

    def make(cfg, provider):
        return PocketLocalTTS(
            transport=httpx.MockTransport(
                lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r))
            )
        )

    class Cfg:
        tts = _tts_cfg(provider="pocket-local")

    result = asyncio.run(provider_test.run_provider_test(spec, Cfg(), make_tts=make, timeout_s=5))
    assert result.status != provider_test.OK


# -- live server (opt-in) -----------------------------------------------------
@pytest.mark.skipif(not os.environ.get("POCKET_TTS_LIVE_URL"), reason="set POCKET_TTS_LIVE_URL")
def test_live_server_streams_french_audio():
    tts = PocketLocalTTS(base_url=os.environ["POCKET_TTS_LIVE_URL"])
    assert tts.check_health()
    chunks = asyncio.run(_collect(tts, text="Bonjour, je suis Dodaios.", language_code="fr-FR"))
    pcm = b"".join(c.pcm for c in chunks)
    assert {c.sample_rate for c in chunks} == {RATE}
    seconds = len(pcm) / 2 / RATE
    assert 0.5 < seconds < 10
    assert np.abs(np.frombuffer(pcm, dtype=np.int16)).max() > 1000  # not silence

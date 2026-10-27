"""French speech must reach the French voice, independently of UI catalogs."""

from types import SimpleNamespace

import pytest

from dodaios.core.turn_language import resolve_output_language
from dodaios.plugins.tts.piper_local import PiperLocalTTS


@pytest.mark.parametrize(
    "text",
    [
        "Bonjour, comment vas-tu ce matin ?",
        "Arrête la musique.",
        "Mets la musique.",
        "Quelle est la météo à Belfort aujourd'hui ?",
    ],
)
def test_french_reply_pin_survives_ambiguous_command(text):
    assert resolve_output_language("fr", "fr", text) == "fr"


def test_detected_french_is_not_replaced_with_a_translated_phrase_locale():
    assert resolve_output_language("auto", "fr", "Bonjour, comment vas-tu ce matin ?") == "fr"


def test_resolved_french_selects_siwis():
    tts = PiperLocalTTS()
    tts._installed_voices = lambda: {
        "english": SimpleNamespace(language="en", gender="masculine"),
        "siwis": SimpleNamespace(language="fr", gender="feminine"),
    }
    language = resolve_output_language("fr", "fr", "Mets la musique.")
    assert tts._resolve_voice(None, language) == "siwis"

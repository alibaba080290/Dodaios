"""Public mission news must not expose private mission state."""
import pytest

from dodaios.brain.turn_planner import TurnPath, TurnReason, plan_turn


@pytest.mark.parametrize("text", [
    "Quelles sont les dernières nouvelles de la mission Artemis ?",
    "Donne-moi les actualités de la mission spatiale Artemis",
    "What's the latest news on the Artemis II mission?",
    "Was sind die neuesten Nachrichten zur Artemis-II-Mission?",
])
def test_public_mission_news(text):
    plan = plan_turn(text, tool_names=("search_web",))
    assert TurnReason.MISSION not in plan.reasons
    assert plan.requires_public_fact_grounding


@pytest.mark.parametrize("text", [
    "Quel est l'état de ma mission Artemis dans Dodaios ?",
    "Relance la mission Artemis de mon équipe",
    "Quelles sont les nouvelles de la mission Artemis que je t'ai confiée ?",
    "Quelles sont les nouvelles de la mission Artemis que tu as lancée ?",
    "What's the latest news on my Artemis II mission?",
    "Was macht meine Artemis-II-Mission?",
    "Quelles sont les nouvelles de la mission Artemis dans mon projet ?",
    "Quelles sont les nouvelles de la mission Artemis dans Dodaios ?",
    "Quel est le statut de la mission Artemis ?",
    "How is the mission going?",
    "Quelles sont les dernières nouvelles de la mission Phoenix ?",
])
def test_internal_or_ambiguous_mission_stays_private(text):
    plan = plan_turn(text, tool_names=("search_web",))
    assert plan.path is TurnPath.ORCHESTRATOR
    assert TurnReason.MISSION in plan.reasons
    assert not plan.requires_public_fact_grounding


@pytest.mark.parametrize("subject, public", [
    ("Quelles sont les dernières nouvelles de la mission Artemis ?", True),
    ("Quel est l'état de ma mission Artemis dans Dodaios ?", False),
])
def test_follow_up_preserves_mission_domain(subject, public):
    plan = plan_turn("Quelles nouvelles à ce sujet aujourd'hui ?", context=(subject,), tool_names=("search_web",))
    assert (TurnReason.MISSION in plan.reasons) is not public
    assert plan.requires_public_fact_grounding is public


@pytest.mark.parametrize("text", [
    "Relance la mission spatiale Artemis",
    "Quelles sont les dernières nouvelles de la mission Artemis dans mes fichiers ?",
    "Search my emails for the latest news on the Artemis II mission",
])
def test_public_name_does_not_remove_private_or_action_exclusions(text):
    plan = plan_turn(text, tool_names=("search_web",))
    assert not plan.requires_public_fact_grounding


def test_relaunch_is_an_internal_action():
    plan = plan_turn("Relance la mission Artemis de mon équipe")
    assert {TurnReason.MISSION, TurnReason.ACTION} <= plan.reasons

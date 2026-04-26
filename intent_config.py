"""Shared configuration for the zero-shot intent classification project.

This module keeps the intent schema and model registry in one place so the
Streamlit app and the evaluation script always stay consistent.
"""

from __future__ import annotations

# The assignment explicitly requires seven hardcoded smart-home intent classes.
INTENTS = [
    "PlayMusic",
    "TurnOnTV",
    "GetWeather",
    "SetTimer",
    "DimLights",
    "SetAlarm",
    "SendMessage",
]

# Zero-shot models used in the Streamlit ablation study.
MODEL_REGISTRY = {
    "Primary: facebook/bart-large-mnli": {
        "model_id": "facebook/bart-large-mnli",
        "summary": "Best overall accuracy baseline, but larger and slower.",
    },
    "Ablation: valhalla/distilbart-mnli-12-3": {
        "model_id": "valhalla/distilbart-mnli-12-3",
        "summary": "Smaller and faster distilled variant for comparison.",
    },
}

# Natural-language intent descriptions work better than raw CamelCase labels
# when using zero-shot NLI classification.
INTENT_TO_CANDIDATE_LABEL = {
    "PlayMusic": "play music or audio",
    "TurnOnTV": "turn on the television",
    "GetWeather": "get the weather forecast",
    "SetTimer": "set a countdown timer",
    "DimLights": "dim the lights in the room",
    "SetAlarm": "set an alarm",
    "SendMessage": "send a text message",
}

# Reverse lookup so model output can be mapped back to the assignment labels.
CANDIDATE_LABEL_TO_INTENT = {
    candidate_label: intent
    for intent, candidate_label in INTENT_TO_CANDIDATE_LABEL.items()
}

CANDIDATE_LABELS = list(INTENT_TO_CANDIDATE_LABEL.values())

# The hypothesis template guides the NLI model toward instruction-style intents.
HYPOTHESIS_TEMPLATE = "This voice command is asking the assistant to {}."

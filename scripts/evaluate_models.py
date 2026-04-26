"""Evaluate the zero-shot models on a small held-out intent set.

This script is included so the assignment repository has a reproducible way to
compare the primary model with the ablation model outside the Streamlit UI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "held_out_intents.json"

# Allow the script to import project modules when run from the repo root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intent_config import (  # noqa: E402
    CANDIDATE_LABELS,
    CANDIDATE_LABEL_TO_INTENT,
    HYPOTHESIS_TEMPLATE,
    MODEL_REGISTRY,
)


def load_examples() -> list[dict[str, str]]:
    """Load held-out evaluation examples from disk."""

    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_model(model_id: str, examples: list[dict[str, str]]) -> float:
    """Compute simple accuracy for one model on the held-out set."""

    classifier = pipeline("zero-shot-classification", model=model_id)
    correct_predictions = 0

    for example in examples:
        result = classifier(
            sequences=example["text"],
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
        )

        predicted_intent = CANDIDATE_LABEL_TO_INTENT[result["labels"][0]]
        if predicted_intent == example["label"]:
            correct_predictions += 1

    return correct_predictions / len(examples)


def main() -> None:
    """Run the evaluation for all configured models."""

    examples = load_examples()
    print(f"Loaded {len(examples)} held-out evaluation examples.\n")

    for model_label, model_config in MODEL_REGISTRY.items():
        accuracy = evaluate_model(model_config["model_id"], examples)
        print(
            f"{model_label}\n"
            f"  Model ID: {model_config['model_id']}\n"
            f"  Accuracy: {accuracy:.2%}\n"
        )


if __name__ == "__main__":
    main()

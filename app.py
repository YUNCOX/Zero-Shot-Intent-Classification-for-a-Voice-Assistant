"""Streamlit UI for zero-shot smart-home intent classification.

The app uses Hugging Face's zero-shot-classification pipeline so it can map
unseen user commands to one of seven predefined intents without fine-tuning.
"""

from __future__ import annotations

from typing import List, Dict

import streamlit as st
from transformers import pipeline

from intent_config import (
    CANDIDATE_LABELS,
    CANDIDATE_LABEL_TO_INTENT,
    HYPOTHESIS_TEMPLATE,
    INTENTS,
    MODEL_REGISTRY,
)


st.set_page_config(
    page_title="Zero-Shot Intent Classifier",
    page_icon="AI",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_zero_shot_pipeline(model_id: str):
    """Load and cache a Hugging Face zero-shot pipeline for the selected model.

    Streamlit keeps this resource in memory across reruns, which prevents the
    application from downloading and rebuilding the model on every button click.
    """

    return pipeline(
        task="zero-shot-classification",
        model=model_id,
    )


def classify_utterance(user_text: str, model_id: str) -> List[Dict[str, float]]:
    """Run zero-shot inference and return ranked intent predictions."""

    classifier = load_zero_shot_pipeline(model_id)

    result = classifier(
        sequences=user_text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )

    ranked_predictions: List[Dict[str, float]] = []
    for candidate_label, score in zip(result["labels"], result["scores"]):
        ranked_predictions.append(
            {
                "intent": CANDIDATE_LABEL_TO_INTENT[candidate_label],
                "candidate_label": candidate_label,
                "score": float(score),
            }
        )

    return ranked_predictions


def render_confidence_bars(predictions: List[Dict[str, float]]) -> None:
    """Show all intent scores as readable confidence bars."""

    st.subheader("Confidence Across All 7 Intents")

    for prediction in predictions:
        label_col, score_col = st.columns([4, 1])
        label_col.markdown(f"**{prediction['intent']}**")
        score_col.markdown(f"**{prediction['score'] * 100:.2f}%**")
        st.progress(int(round(prediction["score"] * 100)))


def main() -> None:
    """Render the full Streamlit application."""

    st.title("Zero-Shot Intent Classification for a Voice Assistant")
    st.write(
        "Zero-shot classification lets a model assign unseen text commands to "
        "predefined intents without any task-specific training examples."
    )

    st.sidebar.header("Ablation Study")
    selected_model_label = st.sidebar.selectbox(
        "Choose the model used for inference",
        options=list(MODEL_REGISTRY.keys()),
        help=(
            "Switch between the full BART MNLI model and the smaller distilled "
            "version to compare speed and confidence behaviour."
        ),
    )

    selected_model = MODEL_REGISTRY[selected_model_label]
    st.sidebar.info(f"**Model ID:** `{selected_model['model_id']}`")
    st.sidebar.caption(selected_model["summary"])

    st.sidebar.subheader("Available Intents")
    for intent in INTENTS:
        st.sidebar.write(f"- {intent}")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Ablation idea: run the same utterance with both models and compare the "
        "top prediction confidence and response speed."
    )

    user_text = st.text_area(
        "Enter a natural voice command",
        placeholder=(
            "Examples: 'Set a timer for 20 minutes', "
            "'Could you send a message to Sarah?', "
            "'It's way too bright in this room.'"
        ),
        height=140,
    )

    if st.button("Classify", type="primary", use_container_width=True):
        cleaned_text = user_text.strip()

        if not cleaned_text:
            st.warning("Please enter a voice command before clicking Classify.")
            return

        try:
            with st.spinner("Loading model and classifying the command..."):
                predictions = classify_utterance(
                    user_text=cleaned_text,
                    model_id=selected_model["model_id"],
                )
        except Exception as error:  # pragma: no cover - UI path
            st.error("The model could not complete the classification request.")
            st.exception(error)
            return

        top_prediction = predictions[0]

        st.markdown("---")
        st.subheader("Top Predicted Intent")
        st.success(
            f"{top_prediction['intent']} "
            f"({top_prediction['score'] * 100:.2f}% confidence)"
        )

        render_confidence_bars(predictions)

        with st.expander("Why this is an ablation study"):
            st.write(
                "The sidebar switches between a large baseline model "
                "(`facebook/bart-large-mnli`) and a smaller distilled model "
                "(`valhalla/distilbart-mnli-12-3`). By keeping the same 7 intents "
                "and the same user command while changing only the model, you can "
                "measure how model size affects prediction confidence and speed."
            )


if __name__ == "__main__":
    main()

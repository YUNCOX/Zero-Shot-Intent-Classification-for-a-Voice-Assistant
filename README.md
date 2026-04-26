# Zero-Shot Intent Classification for a Voice Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://zero-shot-intent-classification-for-a-voice-assistant.streamlit.app)

This project implements **Task 13: Zero-Shot Intent Classification for a Voice Assistant** as a Streamlit web app using Hugging Face Transformers. The system maps unseen natural-language commands to one of seven smart-home intents **without task-specific training** by using the `zero-shot-classification` pipeline.

## Live Demo

- Streamlit App: [zero-shot-intent-classification-for-a-voice-assistant.streamlit.app](https://zero-shot-intent-classification-for-a-voice-assistant.streamlit.app)

## Academic Information

- Student: `YUNCOX`
- University: `Al-Farabi University`
- Course: `Artificial Intelligence and Applications`
- Supervisor / Instructor: `Almuntadher Alwhelat`

## Features

- Streamlit front-end for interactive testing
- Hugging Face `pipeline("zero-shot-classification")` backend
- Primary model: `facebook/bart-large-mnli`
- Ablation model: `valhalla/distilbart-mnli-12-3`
- Seven hardcoded smart-home intents
- Cached model loading with `@st.cache_resource`
- Confidence display for all classes
- Small held-out evaluation script for quick model comparison

## Intents

The application classifies each utterance into one of these seven intents:

1. `PlayMusic`
2. `TurnOnTV`
3. `GetWeather`
4. `SetTimer`
5. `DimLights`
6. `SetAlarm`
7. `SendMessage`

## Project Structure

```text
.
|-- .streamlit/
|   `-- config.toml
|-- app.py
|-- intent_config.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- data/
|   `-- held_out_intents.json
`-- scripts/
    `-- evaluate_models.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

## Run the Held-Out Evaluation

```bash
python scripts/evaluate_models.py
```

This evaluation script uses a small held-out set of unseen utterances stored in `data/held_out_intents.json` and reports the accuracy of both models.

## How the Ablation Study Works

The ablation study is built directly into the Streamlit sidebar. The user keeps the same 7 intents and the same input command, but switches the inference model between:

- `facebook/bart-large-mnli` as the **primary baseline**
- `valhalla/distilbart-mnli-12-3` as the **smaller, faster ablation model**

Because the intent list, hypothesis template, and input text stay constant, the only changing variable is the model itself. This makes it easy to compare how model size affects:

- predicted intent
- confidence distribution
- practical responsiveness

## Notes for the Assignment Report

- The classifier is **zero-shot**, so it does not require task-specific supervised training on the 7 intent classes.
- The candidate labels are written as natural-language intent descriptions internally to improve NLI matching quality, then mapped back to the required class names for display.
- The repository includes a small held-out evaluation set to support quick testing and report screenshots.

## References

- Hugging Face Transformers zero-shot pipeline
- `facebook/bart-large-mnli`
- `valhalla/distilbart-mnli-12-3`
- SNIPS voice-command style intent classification task

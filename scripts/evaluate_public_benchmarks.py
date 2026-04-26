"""Evaluate the zero-shot models on public SNIPS or ATIS test splits.

This script fetches real benchmark data from the official Hugging Face dataset
server and evaluates the same zero-shot classification pipeline used in the
Streamlit app. It is intentionally separate from the small curated held-out set
so the repository can support both:

1. a lightweight assignment demo evaluation, and
2. a reproducible public-benchmark evaluation.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests
from transformers import logging as hf_logging
from transformers import pipeline
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow the script to import project modules when run from the repo root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intent_config import MODEL_REGISTRY  # noqa: E402


HF_DATASET_SERVER = "https://datasets-server.huggingface.co"
PAGE_SIZE = 100

# Keep command-line benchmark output focused on the evaluation metrics.
hf_logging.set_verbosity_error()


@dataclass(frozen=True)
class BenchmarkConfig:
    """Describe how to load and verbalize one public benchmark."""

    dataset_id: str
    split: str
    text_field: str
    label_field: str
    hypothesis_template: str
    label_to_candidate: Callable[[str], str]


def verbalize_snips_label(label: str) -> str:
    """Convert raw SNIPS labels into natural candidate labels."""

    label_map = {
        "AddToPlaylist": "add music to a playlist",
        "BookRestaurant": "book a restaurant",
        "GetWeather": "get the weather forecast",
        "PlayMusic": "play music",
        "RateBook": "rate a book",
        "SearchCreativeWork": "search for a creative work",
        "SearchScreeningEvent": "search for a movie or screening event",
    }
    return label_map[label]


def verbalize_atis_label(label: str) -> str:
    """Convert raw ATIS labels into readable candidate labels."""

    label_map = {
        "atis_abbreviation": "ask for an airport or airline abbreviation",
        "atis_aircraft": "ask about aircraft type",
        "atis_airfare": "ask about airfare or ticket cost",
        "atis_airline": "ask about airline information",
        "atis_flight": "find a flight",
        "atis_flight_time": "ask about flight time",
        "atis_ground_service": "ask about ground transportation",
        "atis_quantity": "ask about counts or quantities",
    }
    return label_map.get(label, label.replace("atis_", "").replace("_", " "))


BENCHMARKS = {
    "snips": BenchmarkConfig(
        dataset_id="benayas/snips",
        split="test",
        text_field="text",
        label_field="category",
        hypothesis_template="This user request is asking the assistant to {}.",
        label_to_candidate=verbalize_snips_label,
    ),
    "atis": BenchmarkConfig(
        dataset_id="DeepPavlov/atis_intent_classification",
        split="test",
        text_field="text",
        label_field="label_text",
        hypothesis_template="This airline-travel request is asking to {}.",
        label_to_candidate=verbalize_atis_label,
    ),
}

MODEL_ALIASES = {
    "primary": MODEL_REGISTRY["Primary: facebook/bart-large-mnli"]["model_id"],
    "ablation": MODEL_REGISTRY["Ablation: valhalla/distilbart-mnli-12-3"]["model_id"],
}


def fetch_rows(config: BenchmarkConfig) -> list[dict[str, str]]:
    """Download one complete split from the official HF dataset server."""

    examples: list[dict[str, str]] = []
    offset = 0

    while True:
        response = requests.get(
            f"{HF_DATASET_SERVER}/rows",
            params={
                "dataset": config.dataset_id,
                "config": "default",
                "split": config.split,
                "offset": offset,
                "length": PAGE_SIZE,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows", [])

        if not rows:
            break

        for row in rows:
            record = row["row"]
            examples.append(
                {
                    "text": str(record[config.text_field]).strip(),
                    "label": str(record[config.label_field]).strip(),
                }
            )

        if len(rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return examples


def stratified_sample(
    examples: list[dict[str, str]],
    max_examples: int | None,
    seed: int,
) -> list[dict[str, str]]:
    """Downsample while preserving class balance as much as possible."""

    if max_examples is None or max_examples >= len(examples):
        return examples

    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example in examples:
        buckets[example["label"]].append(example)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    ordered_labels = sorted(buckets)
    sampled: list[dict[str, str]] = []

    while len(sampled) < max_examples and ordered_labels:
        next_round: list[str] = []
        for label in ordered_labels:
            if buckets[label] and len(sampled) < max_examples:
                sampled.append(buckets[label].pop())
            if buckets[label]:
                next_round.append(label)
        ordered_labels = next_round

    return sampled


def build_candidate_space(
    examples: list[dict[str, str]],
    label_to_candidate: Callable[[str], str],
) -> tuple[list[str], dict[str, str]]:
    """Create natural-language candidate labels and reverse lookup mapping."""

    unique_labels = sorted({example["label"] for example in examples})
    candidate_labels = [label_to_candidate(label) for label in unique_labels]
    candidate_to_raw_label = dict(zip(candidate_labels, unique_labels))
    return candidate_labels, candidate_to_raw_label


def compute_macro_f1(true_labels: list[str], predicted_labels: list[str]) -> float:
    """Compute macro F1 without adding a heavy metrics dependency."""

    all_labels = sorted(set(true_labels) | set(predicted_labels))
    f1_scores: list[float] = []

    for label in all_labels:
        true_positive = sum(
            1
            for true_label, predicted_label in zip(true_labels, predicted_labels)
            if true_label == label and predicted_label == label
        )
        false_positive = sum(
            1
            for true_label, predicted_label in zip(true_labels, predicted_labels)
            if true_label != label and predicted_label == label
        )
        false_negative = sum(
            1
            for true_label, predicted_label in zip(true_labels, predicted_labels)
            if true_label == label and predicted_label != label
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 0.0
        )

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores)


def evaluate_model(
    model_id: str,
    config: BenchmarkConfig,
    examples: list[dict[str, str]],
) -> dict[str, float | int]:
    """Evaluate one zero-shot model on one public benchmark split."""

    candidate_labels, candidate_to_raw_label = build_candidate_space(
        examples=examples,
        label_to_candidate=config.label_to_candidate,
    )

    load_start = time.perf_counter()
    classifier = pipeline("zero-shot-classification", model=model_id)
    load_time_seconds = time.perf_counter() - load_start

    predictions: list[str] = []
    truths: list[str] = []

    inference_start = time.perf_counter()
    for example in tqdm(
        examples,
        desc=model_id,
        leave=False,
        disable=not sys.stderr.isatty(),
    ):
        result = classifier(
            sequences=example["text"],
            candidate_labels=candidate_labels,
            hypothesis_template=config.hypothesis_template,
            multi_label=False,
        )
        predicted_raw_label = candidate_to_raw_label[result["labels"][0]]
        predictions.append(predicted_raw_label)
        truths.append(example["label"])
    inference_time_seconds = time.perf_counter() - inference_start

    correct = sum(
        1 for predicted_label, true_label in zip(predictions, truths)
        if predicted_label == true_label
    )

    return {
        "examples": len(examples),
        "num_labels": len(candidate_labels),
        "correct": correct,
        "accuracy": correct / len(examples),
        "macro_f1": compute_macro_f1(truths, predictions),
        "load_time_seconds": load_time_seconds,
        "inference_time_seconds": inference_time_seconds,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the zero-shot intent models on real public SNIPS or ATIS "
            "test splits fetched from Hugging Face."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(BENCHMARKS),
        default="snips",
        help="Public benchmark dataset to evaluate.",
    )
    parser.add_argument(
        "--model",
        choices=["primary", "ablation", "all"],
        default="all",
        help="Model selection shortcut.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=200,
        help=(
            "Maximum number of test examples to evaluate. Use 0 to evaluate the "
            "full public test split. The default keeps runtime practical on CPU."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when downsampling the benchmark split.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the public benchmark evaluation."""

    args = parse_args()
    config = BENCHMARKS[args.dataset]

    examples = fetch_rows(config)
    max_examples = None if args.max_examples == 0 else args.max_examples
    sampled_examples = stratified_sample(examples, max_examples=max_examples, seed=args.seed)

    if args.model == "all":
        model_ids = [MODEL_ALIASES["primary"], MODEL_ALIASES["ablation"]]
    else:
        model_ids = [MODEL_ALIASES[args.model]]

    print(f"Dataset: {args.dataset} ({config.dataset_id})")
    print(f"Split: {config.split}")
    print(f"Downloaded examples: {len(examples)}")
    print(f"Evaluated examples: {len(sampled_examples)}")
    print(f"Seed: {args.seed}\n")

    for model_id in model_ids:
        metrics = evaluate_model(
            model_id=model_id,
            config=config,
            examples=sampled_examples,
        )
        print(f"Model: {model_id}")
        print(f"  Labels evaluated: {metrics['num_labels']}")
        print(f"  Correct predictions: {metrics['correct']}/{metrics['examples']}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Load time: {metrics['load_time_seconds']:.2f}s")
        print(f"  Inference time: {metrics['inference_time_seconds']:.2f}s\n")


if __name__ == "__main__":
    main()

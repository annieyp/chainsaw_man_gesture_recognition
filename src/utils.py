from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "config" / "gesture_labels.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gesture_recognizer.task"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw"


def load_labels(path: str | Path = DEFAULT_LABELS_PATH) -> list[str]:
    labels_path = Path(path)
    with labels_path.open("r", encoding="utf-8") as label_file:
        labels = json.load(label_file)

    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"{labels_path} must contain a JSON list of label strings.")

    normalized = [label.strip().lower() for label in labels if label.strip()]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{labels_path} contains duplicate labels after normalization.")

    return normalized


def ensure_required_labels(labels: Iterable[str]) -> None:
    label_set = {label.strip().lower() for label in labels}
    missing = {"bomb", "chainsaw", "kon", "none"} - label_set
    if missing:
        raise ValueError(
            "Missing required gesture labels: " + ", ".join(sorted(missing))
        )


def label_display_name(label: str) -> str:
    names = {
        "bomb": "Bomb Girl",
        "chainsaw": "Chainsaw Man",
        "kon": "Kon",
        "none": "No gesture",
    }
    return names.get(label.lower(), label.replace("_", " ").title())


def make_output_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

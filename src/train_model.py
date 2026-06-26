from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import DEFAULT_DATASET_PATH, ensure_required_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MediaPipe Gesture Recognizer with Model Maker.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Folder with one subdirectory per label.")
    parser.add_argument("--export-dir", type=Path, default=Path("models"), help="Directory for exported gesture_recognizer.task.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset}")

    labels = sorted(path.name.lower() for path in args.dataset.iterdir() if path.is_dir())
    ensure_required_labels(labels)

    try:
        from mediapipe_model_maker import gesture_recognizer
    except ImportError as exc:
        raise SystemExit(
            "Install cloud training dependencies first: pip install -r requirements-cloud.txt"
        ) from exc

    args.export_dir.mkdir(parents=True, exist_ok=True)

    data = gesture_recognizer.Dataset.from_folder(
        dirname=str(args.dataset),
        hparams=gesture_recognizer.HandDataPreprocessingParams(
            min_detection_confidence=args.min_detection_confidence
        ),
    )
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    hparams = gesture_recognizer.HParams(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        export_dir=str(args.export_dir),
    )
    model_options = gesture_recognizer.ModelOptions(dropout_rate=args.dropout_rate)
    options = gesture_recognizer.GestureRecognizerOptions(
        model_options=model_options,
        hparams=hparams,
    )

    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options,
    )
    loss, accuracy = model.evaluate(test_data, batch_size=1)
    print(f"Test loss: {loss:.4f}, test accuracy: {accuracy:.4f}")

    model.export_model()
    print(f"Exported model to {args.export_dir / 'gesture_recognizer.task'}")


if __name__ == "__main__":
    main()

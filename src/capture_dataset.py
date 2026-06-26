from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2

from src.utils import DEFAULT_DATASET_PATH, DEFAULT_LABELS_PATH, ensure_required_labels, load_labels, make_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture webcam samples for one gesture label.")
    parser.add_argument("label", help="Gesture label to capture, for example bomb, chainsaw, kon, or none.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH, help="Dataset root directory.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH, help="JSON label list.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--interval", type=int, default=1, help="Save every Nth frame while recording.")
    parser.add_argument("--no-mirror", action="store_true", help="Disable selfie-style horizontal mirroring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    ensure_required_labels(labels)

    label = args.label.strip().lower()
    if label not in labels:
        raise ValueError(f"Unknown label '{args.label}'. Expected one of: {', '.join(labels)}")

    output_dir = make_output_dir(args.output / label)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}.")

    recording = False
    frame_index = 0
    saved = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            frame_index += 1
            if recording and frame_index % max(1, args.interval) == 0:
                stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                image_path = output_dir / f"{label}_{stamp}.jpg"
                cv2.imwrite(str(image_path), frame)
                saved += 1

            status = "REC" if recording else "PAUSED"
            cv2.putText(frame, f"{label} | {status} | saved {saved}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "space: record/pause   q/esc: quit", (18, frame.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 235, 255), 2)
            cv2.imshow("Dataset capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = not recording
            elif key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} images to {output_dir}")


if __name__ == "__main__":
    main()

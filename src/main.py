from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.inference import GestureRecognizerRunner, PredictionSmoother, draw_hand_landmarks
from src.overlays import OverlayState, apply_gesture_overlay
from src.person_tracking import FaceTracker, choose_target
from src.utils import DEFAULT_LABELS_PATH, DEFAULT_MODEL_PATH, ensure_required_labels, load_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chainsaw Man gesture recognition overlays.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to a MediaPipe .task gesture model.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH, help="JSON label list used by the trained model.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--min-score", type=float, default=0.55, help="Minimum gesture score before smoothing.")
    parser.add_argument("--effect-seconds", type=float, default=2.0, help="How long an effect remains after a stable gesture.")
    parser.add_argument("--hide-hands", action="store_true", help="Do not draw detected hand landmarks.")
    parser.add_argument("--no-mirror", action="store_true", help="Disable selfie-style horizontal mirroring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    ensure_required_labels(labels)

    recognizer = GestureRecognizerRunner(model_path=args.model)
    face_tracker = FaceTracker()
    smoother = PredictionSmoother()
    overlay_state = OverlayState()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30
    effect_frames = max(1, int(args.effect_seconds * fps))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            prediction, hand_landmarks = recognizer.recognize(frame)
            if prediction and prediction.score < args.min_score:
                prediction = None

            stable_label = smoother.update(prediction)
            overlay_state.trigger(stable_label, effect_frames)
            active_label = overlay_state.tick()

            faces = face_tracker.detect(frame)
            target = choose_target(faces, hand_landmarks, frame.shape)

            if not args.hide_hands:
                draw_hand_landmarks(frame, hand_landmarks)

            if active_label:
                frame = apply_gesture_overlay(frame, active_label, target, overlay_state)

            if prediction:
                cv2.putText(
                    frame,
                    f"{prediction.label} {prediction.score:.2f}",
                    (18, frame.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (220, 255, 220),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Chainsaw Man Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        recognizer.close()
        face_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

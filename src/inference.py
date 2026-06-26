from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.utils import DEFAULT_MODEL_PATH


IGNORED_LABELS = {"", "none", "unknown", "no_gesture", "no gesture"}


@dataclass(frozen=True)
class GesturePrediction:
    label: str | None
    score: float
    hand_landmarks: list


class PredictionSmoother:
    def __init__(self, window_size: int = 8, min_votes: int = 4) -> None:
        self.history: deque[str] = deque(maxlen=window_size)
        self.min_votes = min_votes

    def update(self, prediction: GesturePrediction | None) -> str | None:
        label = prediction.label if prediction else None
        self.history.append(label or "")
        votes = Counter(item for item in self.history if item)
        if not votes:
            return None

        top_label, top_count = votes.most_common(1)[0]
        majority = len(self.history) // 2 + 1
        if top_count >= self.min_votes and top_count >= majority:
            return top_label
        return None


class GestureRecognizerRunner:
    """Small wrapper around MediaPipe Tasks GestureRecognizer."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.65,
        min_hand_presence_confidence: float = 0.65,
        min_tracking_confidence: float = 0.65,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Custom gesture model not found at {self.model_path}. "
                "Train/export it first, or pass --model to a .task file."
            )

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision
        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._timestamp_ms = 0

    def recognize(self, bgr_frame: np.ndarray) -> tuple[GesturePrediction | None, list]:
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        self._timestamp_ms += 33
        result = self._recognizer.recognize_for_video(mp_image, self._timestamp_ms)

        hand_landmarks = result.hand_landmarks or []
        candidates: list[GesturePrediction] = []
        for index, gestures in enumerate(result.gestures or []):
            if not gestures:
                continue
            top = gestures[0]
            label = normalize_label(top.category_name)
            if label in IGNORED_LABELS:
                continue
            landmarks = hand_landmarks[index] if index < len(hand_landmarks) else []
            candidates.append(GesturePrediction(label=label, score=top.score, hand_landmarks=landmarks))

        if not candidates:
            return None, hand_landmarks
        return max(candidates, key=lambda item: item.score), hand_landmarks

    def close(self) -> None:
        self._recognizer.close()


def normalize_label(label: str | None) -> str:
    if not label:
        return ""
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def draw_hand_landmarks(frame: np.ndarray, hand_landmarks: Sequence[Sequence]) -> None:
    if not hand_landmarks:
        return

    import mediapipe as mp

    h, w = frame.shape[:2]
    connections = mp.solutions.hands.HAND_CONNECTIONS
    for hand in hand_landmarks:
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (80, 240, 180), -1)
        for start_idx, end_idx in connections:
            start = hand[start_idx]
            end = hand[end_idx]
            cv2.line(
                frame,
                (int(start.x * w), int(start.y * h)),
                (int(end.x * w), int(end.y * h)),
                (255, 255, 255),
                1,
            )

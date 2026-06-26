from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceTarget:
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


class FaceTracker:
    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        import mediapipe as mp

        self._mp = mp
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame: np.ndarray) -> list[FaceTarget]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb_frame)
        if not results.detections:
            return []

        height, width = frame.shape[:2]
        faces: list[FaceTarget] = []
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            x = max(0, int(box.xmin * width))
            y = max(0, int(box.ymin * height))
            w = min(width - x, int(box.width * width))
            h = min(height - y, int(box.height * height))
            if w <= 0 or h <= 0:
                continue
            faces.append(FaceTarget(x=x, y=y, w=w, h=h, score=detection.score[0]))
        return faces

    def close(self) -> None:
        self._detector.close()


def choose_target(faces: list[FaceTarget], hand_landmarks: list, frame_shape: tuple[int, ...]) -> FaceTarget | None:
    if not faces:
        return None
    if not hand_landmarks:
        return max(faces, key=lambda face: face.w * face.h)

    height, width = frame_shape[:2]
    points: list[tuple[int, int]] = []
    for hand in hand_landmarks:
        if not hand:
            continue
        wrist = hand[0]
        points.append((int(wrist.x * width), int(wrist.y * height)))

    if not points:
        return max(faces, key=lambda face: face.w * face.h)

    def score(face: FaceTarget) -> float:
        cx, cy = face.center
        return min((cx - px) ** 2 + (cy - py) ** 2 for px, py in points)

    return min(faces, key=score)

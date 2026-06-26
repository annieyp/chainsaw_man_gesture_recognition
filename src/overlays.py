from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

from src.person_tracking import FaceTarget
from src.utils import label_display_name


FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class OverlayState:
    active_label: str | None = None
    frames_left: int = 0
    phase: float = 0.0

    def trigger(self, label: str | None, duration_frames: int) -> None:
        if not label:
            return
        if label != self.active_label:
            self.phase = 0.0
        self.active_label = label
        self.frames_left = duration_frames

    def tick(self) -> str | None:
        if not self.active_label or self.frames_left <= 0:
            self.active_label = None
            self.frames_left = 0
            return None
        self.frames_left -= 1
        self.phase += 0.12
        return self.active_label


def apply_gesture_overlay(frame: np.ndarray, label: str, target: FaceTarget | None, state: OverlayState) -> np.ndarray:
    label = label.lower()
    if label == "bomb":
        frame = _bomb_devil(frame, target, state.phase)
    elif label == "chainsaw":
        frame = _chainsaw_hybrid(frame, target, state.phase)
    elif label == "kon":
        frame = _kon_fox(frame, target, state.phase)

    _draw_label(frame, label)
    return frame


def _draw_label(frame: np.ndarray, label: str) -> None:
    text = label_display_name(label)
    cv2.rectangle(frame, (18, 18), (310, 64), (0, 0, 0), -1)
    cv2.rectangle(frame, (18, 18), (310, 64), _label_color(label), 2)
    cv2.putText(frame, text, (32, 50), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def _label_color(label: str) -> tuple[int, int, int]:
    return {
        "bomb": (0, 160, 255),
        "chainsaw": (0, 90, 255),
        "kon": (40, 220, 255),
    }.get(label, (255, 255, 255))


def _head_anchor(frame: np.ndarray, target: FaceTarget | None) -> tuple[int, int, int]:
    height, width = frame.shape[:2]
    if target:
        cx, cy = target.center
        radius = max(target.w, target.h) // 2
        return cx, cy, max(34, radius)
    return width // 2, height // 2, min(width, height) // 7


def _bomb_devil(frame: np.ndarray, target: FaceTarget | None, phase: float) -> np.ndarray:
    cx, cy, r = _head_anchor(frame, target)
    helmet_center = (cx, cy - r // 7)
    shell = frame.copy()

    cv2.circle(shell, helmet_center, int(r * 1.08), (14, 14, 20), -1)
    cv2.circle(shell, helmet_center, int(r * 0.92), (38, 38, 48), -1)
    cv2.circle(shell, (cx - r // 3, cy - r // 3), max(4, r // 12), (235, 235, 240), -1)
    frame[:] = cv2.addWeighted(frame, 0.45, shell, 0.55, 0)

    fuse_start = (cx + int(r * 0.45), cy - int(r * 0.95))
    fuse_end = (cx + int(r * 0.88), cy - int(r * 1.35))
    cv2.line(frame, fuse_start, fuse_end, (70, 70, 85), max(3, r // 16), cv2.LINE_AA)

    spark = (
        fuse_end[0] + int(math.sin(phase * 5) * r * 0.1),
        fuse_end[1] + int(math.cos(phase * 4) * r * 0.1),
    )
    for angle in np.linspace(0, math.tau, 9)[:-1]:
        length = int(r * random.uniform(0.18, 0.34))
        end = (spark[0] + int(math.cos(angle) * length), spark[1] + int(math.sin(angle) * length))
        cv2.line(frame, spark, end, (0, 210, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, spark, max(5, r // 10), (0, 95, 255), -1)

    return _screen_tint(frame, (0, 70, 120), 0.18)


def _chainsaw_hybrid(frame: np.ndarray, target: FaceTarget | None, phase: float) -> np.ndarray:
    cx, cy, r = _head_anchor(frame, target)
    mask = frame.copy()

    points = np.array(
        [
            (cx - int(r * 0.82), cy - int(r * 0.35)),
            (cx + int(r * 0.82), cy - int(r * 0.35)),
            (cx + int(r * 0.58), cy + int(r * 0.72)),
            (cx - int(r * 0.58), cy + int(r * 0.72)),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, points, (16, 82, 225))
    cv2.polylines(mask, [points], True, (20, 20, 25), max(3, r // 18), cv2.LINE_AA)
    frame[:] = cv2.addWeighted(frame, 0.5, mask, 0.5, 0)

    blade_top = cy - int(r * 1.75)
    blade_bottom = cy - int(r * 0.2)
    blade_width = max(12, r // 4)
    blade = np.array(
        [
            (cx - blade_width, blade_bottom),
            (cx, blade_top),
            (cx + blade_width, blade_bottom),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, blade, (205, 205, 195))
    cv2.polylines(frame, [blade], True, (60, 60, 60), 2, cv2.LINE_AA)

    tooth_step = max(8, r // 8)
    jitter = int(math.sin(phase * 8) * 3)
    for y in range(blade_top + tooth_step, blade_bottom, tooth_step):
        side = -1 if (y // tooth_step) % 2 == 0 else 1
        x1 = cx + side * blade_width
        x2 = cx + side * (blade_width + max(7, r // 8) + jitter)
        cv2.line(frame, (x1, y), (x2, y + tooth_step // 2), (240, 240, 230), 2, cv2.LINE_AA)

    cv2.rectangle(frame, (cx - r // 2, cy + r // 2), (cx + r // 2, cy + int(r * 0.72)), (25, 25, 25), -1)
    return _screen_tint(frame, (0, 40, 90), 0.12)


def _kon_fox(frame: np.ndarray, target: FaceTarget | None, phase: float) -> np.ndarray:
    cx, cy, r = _head_anchor(frame, target)
    mask = frame.copy()
    ear_l = np.array(
        [(cx - int(r * 0.72), cy - int(r * 0.38)), (cx - int(r * 1.08), cy - int(r * 1.2)), (cx - int(r * 0.18), cy - int(r * 0.7))],
        dtype=np.int32,
    )
    ear_r = np.array(
        [(cx + int(r * 0.72), cy - int(r * 0.38)), (cx + int(r * 1.08), cy - int(r * 1.2)), (cx + int(r * 0.18), cy - int(r * 0.7))],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, ear_l, (35, 150, 235))
    cv2.fillConvexPoly(mask, ear_r, (35, 150, 235))
    cv2.ellipse(mask, (cx, cy), (int(r * 0.86), int(r * 0.62)), 0, 0, 360, (35, 150, 235), -1)
    cv2.ellipse(mask, (cx, cy + r // 10), (int(r * 0.52), int(r * 0.36)), 0, 0, 360, (235, 235, 225), -1)
    frame[:] = cv2.addWeighted(frame, 0.46, mask, 0.54, 0)

    eye_y = cy - r // 10 + int(math.sin(phase * 3) * 2)
    cv2.line(frame, (cx - r // 2, eye_y), (cx - r // 5, eye_y + r // 10), (30, 30, 30), 3, cv2.LINE_AA)
    cv2.line(frame, (cx + r // 5, eye_y + r // 10), (cx + r // 2, eye_y), (30, 30, 30), 3, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy + r // 4), max(5, r // 12), (25, 25, 28), -1)

    for offset in (-1, 1):
        start = (cx + offset * r // 7, cy + r // 3)
        for whisker in (-1, 0, 1):
            end = (cx + offset * int(r * 0.82), cy + r // 4 + whisker * r // 7)
            cv2.line(frame, start, end, (245, 245, 235), 2, cv2.LINE_AA)

    return _screen_tint(frame, (20, 100, 80), 0.1)


def _screen_tint(frame: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    overlay = np.zeros_like(frame)
    overlay[:] = color
    return cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)

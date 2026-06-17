import cv2
import numpy as np
import json
from pathlib import Path
 
def load_gesture_labels(path="config/gesture_labels.json"):
    """Load gesture labels from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)
 
def save_gesture_labels(labels, path="config/gesture_labels.json"):
    """Save gesture labels to JSON file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(labels, f, indent=2)
 
def create_directories():
    """Create necessary project directories"""
    dirs = [
        "models",
        "data/gestures",
        "config",
        "output"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Project directories created")
 
def draw_confidence_bar(frame, confidence, x=10, y=70, width=200, height=20):
    """Draw confidence bar on frame"""
    bar_width = int(width * confidence)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), -1)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + height), (0, 255, 0), -1)
    cv2.putText(frame, f"{confidence:.1%}", (x + width + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame
 
def resize_frame(frame, width=640):
    """Resize frame maintaining aspect ratio"""
    h, w = frame.shape[:2]
    if w > width:
        scale = width / w
        new_h = int(h * scale)
        return cv2.resize(frame, (width, new_h))
    return frame
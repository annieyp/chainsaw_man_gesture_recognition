import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import json

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GestureRecognizer:
    def __init__(self, model_path="models/gesture_recognizer.task", labels_path="config/gesture_labels.json"):
        self.model_path = model_path
        self.labels_path = labels_path
        
        if not os.path.exists(self.model_path):
            print(f"Error: Model not found at {self.model_path}")
            print("Download gesture_recognizer.task from Colab training output")
            self.recognizer = None
            return
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        
        with open(self.labels_path, 'r') as f:
            self.labels = json.load(f)
        
        print(f"✓ Model loaded: {self.model_path}")
        print(f"✓ Gestures: {', '.join(self.labels)}")
    
    def recognize(self, frame):
        if self.recognizer is None:
            return None, None
        
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.recognizer.recognize(image)
        
        if results.gestures and len(results.gestures) > 0:
            gesture = results.gestures[0][0]
            gesture_name = gesture.category_name
            confidence = gesture.score
            return gesture_name, confidence
        
        return None, None

def main():
    recognizer = GestureRecognizer()
    
    if recognizer.recognizer is None:
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("\nStarting gesture recognition...")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        gesture_name, confidence = recognizer.recognize(frame)
        
        if gesture_name:
            text = f"{gesture_name} ({confidence:.2f})"
            color = (0, 255, 0)
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "No gesture detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        cv2.imshow('Gesture Recognizer', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done")

if __name__ == "__main__":
    main()


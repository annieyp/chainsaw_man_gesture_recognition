# chainsaw_man_gesture_recognition

Real-time Chainsaw Man gesture recognition using OpenCV and a custom MediaPipe Gesture Recognizer model.

Starter classes:

- `bomb`
- `chainsaw`
- `kon`
- `none`

The webcam app follows the same shape as `montasirmoyen/domain-expansion`: detect a stable hand sign, smooth the prediction over a short history, then trigger an on-screen visual effect. Instead of hard-coded gesture geometry, this project expects a custom MediaPipe `.task` model trained from your anime gesture dataset.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Layout

MediaPipe Model Maker expects one folder per label. The `none` class is required and should contain hands/poses that are not one of your target gestures.

```text
data/raw/
  bomb/
  chainsaw/
  kon/
  none/
```

You can collect quick webcam samples locally:

```bash
python -m src.capture_dataset bomb
python -m src.capture_dataset chainsaw
python -m src.capture_dataset kon
python -m src.capture_dataset none
```

Press `space` to start/pause recording and `q` to quit.

## Train In Google Cloud Or Colab

Install the training-only dependencies in your cloud environment:

```bash
pip install -r requirements-cloud.txt
```

Then run:

```bash
python -m src.train_model \
  --dataset data/raw \
  --export-dir models \
  --epochs 20 \
  --batch-size 8
```

The script exports:

```text
models/gesture_recognizer.task
```

That file is ignored by git because it can be large.

## Run The Webcam App

After exporting or copying in your trained model:

```bash
python -m src.main --model models/gesture_recognizer.task
```

Press `q` or `esc` to close the camera window.

## Notes

- `bomb` draws a Bomb Devil-style helmet with a lit fuse on the detected face.
- `chainsaw` draws an orange hybrid mask and animated forehead blade.
- `kon` draws a fox mask and ears.
- If a gesture uses both hands, include complete examples of that sign in the target class. MediaPipe Gesture Recognizer returns per-hand gesture categories, so very complex two-hand contracts may eventually need a small custom landmark classifier on top of MediaPipe hand landmarks.

# main.py
#
# Real-time face detection and VAD emotion analysis.
# Stack: YuNet (detection) + VAD multi-task model (emotion)
#
# Controls:
#   Q — quit

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from yunet import YuNet
from vad_model import VADModel


# --- Configuration ------------------------------------------------------------

YUNET_MODEL      = "models/face_detection_yunet_2023mar.onnx"
VAD_MODEL_PATH   = "models/vad_multitask.onnx"

EMOTION_INTERVAL = 5       # run VAD inference every N frames
MARGIN_RATIO     = 0.20    # face crop margin percentage

# Display colors (BGR)
COLOR_BOX  = (0, 200, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_BG   = (30, 30, 30)


# --- Utilities ----------------------------------------------------------------

def crop_face_with_margin(frame: np.ndarray, box: list, margin: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, fw, fh = [int(v) for v in box]
    mx = int(fw * margin)
    my = int(fh * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + fw + mx)
    y2 = min(h, y + fh + my)
    return frame[y1:y2, x1:x2]


def draw_vad_bar(frame, x, y, label, value, color):
    """Draws a small horizontal bar for a VAD dimension in [-1, 1]."""
    bar_w  = 80
    bar_h  = 8
    filled = int((value + 1) / 2 * bar_w)
    filled = max(0, min(bar_w, filled))

    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), color, -1)
    cv2.putText(frame, f"{label} {value:+.2f}",
                (x + bar_w + 6, y + bar_h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT, 1, cv2.LINE_AA)


def draw_face_overlay(frame, box, result):
    """Draws bounding box, emotion label, confidence, and VAD bars."""
    x, y, w, h = [int(v) for v in box]

    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BOX, 2)

    if result is None:
        cv2.putText(frame, "analyzing...", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_BOX, 1, cv2.LINE_AA)
        return

    # Emotion label + confidence
    label = f"{result['emotion'].upper()}  {result['confidence']:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(frame, (x, max(y - th - 10, 0)), (x + tw + 8, y), COLOR_BOX, -1)
    cv2.putText(frame, label, (x + 4, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_BG, 1, cv2.LINE_AA)

    # VAD bars below box
    bx = x
    by = y + h + 6
    draw_vad_bar(frame, bx, by,      "V", result["valence"],   (100, 220, 100))
    draw_vad_bar(frame, bx, by + 14, "A", result["arousal"],   (100, 160, 255))
    draw_vad_bar(frame, bx, by + 28, "D", result["dominance"], (200, 130, 255))


# --- Main Pipeline ------------------------------------------------------------

def main():
    detector  = YuNet(
        modelPath=YUNET_MODEL,
        inputSize=[320, 320],
        confThreshold=0.6,
        nmsThreshold=0.3,
        topK=5,
    )
    vad_model = VADModel(VAD_MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count   = 0
    emotion_cache = {}

    print("Running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]
        detector.setInputSize([w, h])
        detections = detector.infer(frame)

        face_count = 0

        if detections is not None and len(detections) > 0:
            face_count = len(detections)

            for i, det in enumerate(detections):
                box       = det[:4]
                cache_key = f"face_{i}"

                if frame_count % EMOTION_INTERVAL == 0 or cache_key not in emotion_cache:
                    crop   = crop_face_with_margin(frame, box, MARGIN_RATIO)
                    result = vad_model.predict(crop)
                    if result is not None:
                        emotion_cache[cache_key] = result

                draw_face_overlay(frame, box, emotion_cache.get(cache_key))

        # Clear stale cache entries
        for key in list(emotion_cache.keys()):
            if int(key.split("_")[1]) >= face_count:
                del emotion_cache[key]

        # HUD
        cv2.putText(frame,
                    f"Faces: {face_count}  |  Frame: {frame_count}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "Q: quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("face-emotion-vad", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exited.")


if __name__ == "__main__":
    main()
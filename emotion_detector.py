import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from download_model import download_weights, MODEL_PATH

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# colours per emotion (BGR)
EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 0),
    "fear":     (128, 0,   128),
    "happy":    (0,   215, 255),
    "sad":      (200, 100, 0),
    "surprise": (0,   165, 255),
    "neutral":  (180, 180, 180),
}


def preprocess_face(face_gray):
    face = cv2.resize(face_gray, (64, 64))
    face = face.astype("float32") / 255.0
    return face.reshape(1, 64, 64, 1)


def draw_overlay(frame, x, y, w, h, emotion, confidence, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"{emotion}  {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # background pill behind text
    pad = 4
    cv2.rectangle(frame, (x, y - th - pad * 2), (x + tw + pad * 2, y), color, -1)
    cv2.putText(frame, label, (x + pad, y - pad), font, font_scale, (255, 255, 255), thickness)


def draw_bar_chart(frame, probabilities, chart_x=10, chart_y=10):
    bar_width = 18
    bar_max_height = 80
    gap = 6
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probabilities)):
        color = EMOTION_COLORS[emotion]
        bar_h = int(prob * bar_max_height)
        bx = chart_x + i * (bar_width + gap)
        by_bottom = chart_y + bar_max_height
        cv2.rectangle(frame, (bx, by_bottom - bar_h), (bx + bar_width, by_bottom), color, -1)
        # emotion initial label
        cv2.putText(frame, emotion[0].upper(), (bx + 4, by_bottom + 14), font, 0.4, color, 1)


def run(camera_index=0):
    if not os.path.exists(MODEL_PATH):
        download_weights()

    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        sys.exit("ERROR: Could not load Haar Cascade. Check your OpenCV installation.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open camera index {camera_index}")

    print("Running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            input_tensor = preprocess_face(face_roi)

            predictions = model.predict(input_tensor, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            emotion = EMOTIONS[emotion_idx]
            confidence = predictions[emotion_idx]
            color = EMOTION_COLORS[emotion]

            draw_overlay(frame, x, y, w, h, emotion, confidence, color)
            draw_bar_chart(frame, predictions)

        # FPS counter
        cv2.putText(frame, "Press Q to quit", (frame.shape[1] - 150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Emotion Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(camera_idx)

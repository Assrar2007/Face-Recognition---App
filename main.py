import os
import sys
import traceback
import cv2
import numpy as np
import tensorflow as tf

# --- 1) Locate model file (tries a few common spots) ---
candidates = [
    "Data/emotion_model.h5",
    "emotion_model.h5",
    "./emotion_model.h5",
    "./Data/emotion_model.h5"
]

model_path = None
for p in candidates:
    if os.path.isfile(p):
        model_path = p
        break

if model_path is None:
    print("ERROR: could not find emotion_model.h5. Tried paths:")
    for p in candidates:
        print("  -", os.path.abspath(p))
    print("\nPlace your saved model (emotion_model.h5) in the project root or Data/ folder.")
    sys.exit(1)

print("Loading model from:", model_path)

# --- 2) Load model safely ---
try:
    model = tf.keras.models.load_model(model_path, compile=False)
except Exception as e:
    print("Failed to load model. Traceback:")
    traceback.print_exc()
    sys.exit(1)

print("Model loaded OK.")
try:
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
except Exception:
    pass

# --- 3) Haar cascade check ---
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.isfile(cascade_path):
    print("ERROR: Haarcascade file not found at:", cascade_path)
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(cascade_path)
print("Using Haar cascade:", cascade_path)

# --- 4) Labels (make sure order matches training) ---
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
print("Using emotion labels (in this order):", emotion_labels)

# --- 5) Start webcam and run predictions (with debug prints) ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam (cv2.VideoCapture(0) failed).")
    sys.exit(1)

print("Webcam opened. Press 'q' to quit.")

# <<< added: smoothing variables >>>
smooth_preds = None
alpha = 0.7  # smoothing factor (higher = smoother, but slower to react)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            try:
                roi_resized = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)
            except Exception as ex:
                print("Resize failed:", ex)
                continue

            roi = roi_resized.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=0)
            if roi.ndim == 3:
                roi = np.expand_dims(roi, axis=-1)

            try:
                preds = model.predict(roi, verbose=0)[0]
            except Exception as ex:
                print("Prediction failed. Traceback:")
                traceback.print_exc()
                print("ROI shape:", roi.shape)
                print("Model input shape:", getattr(model, "input_shape", None))
                raise

            # <<< smoothing step >>>
            if smooth_preds is None:
                smooth_preds = preds
            else:
                smooth_preds = alpha * smooth_preds + (1 - alpha) * preds

            max_i = int(np.argmax(smooth_preds))
            label = emotion_labels[max_i]
            confidence = float(smooth_preds[max_i])

            # draw box + label
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # draw probability bars (smoothed)
            start_y = y + h + 10
            for i,(emo,prob) in enumerate(zip(emotion_labels, smooth_preds)):
                bar_w = int(prob * 150)
                top_left = (x, start_y + i*22)
                bottom_right = (x + bar_w, start_y + 18 + i*22)
                cv2.rectangle(frame, top_left, bottom_right, (0,255,0), -1)
                cv2.putText(frame, f"{emo} {prob*100:.1f}%", (x + 155, start_y + 15 + i*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        cv2.imshow("Emotion Detector (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception:
    print("Unexpected error during webcam loop. Traceback:")
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Clean exit.")

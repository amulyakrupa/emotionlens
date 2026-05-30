"""
EmotionLens - Module 1
Real-time stress detection from webcam using facial landmarks.
Run: python app.py
Works with mediapipe 0.10.35
"""

import cv2
import numpy as np
from collections import deque
import urllib.request
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# ── Download model if missing ─────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading model (~2MB), please wait...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        MODEL_PATH
    )
    print("Download complete!")

# ── Landmark indices ──────────────────────────────────────────────────────────
L_EYE_TOP, L_EYE_BOT = 159, 145
L_EYE_L,   L_EYE_R   = 33,  133
R_EYE_TOP, R_EYE_BOT = 386, 374
R_EYE_L,   R_EYE_R   = 362, 263
L_BROW_IN, R_BROW_IN  = 107, 336
M_TOP,  M_BOT         = 13,  14
LOWER_LIP             = 17
CHIN                  = 152

# ── Helpers ───────────────────────────────────────────────────────────────────
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def ear(top, bot, left, right):
    return dist(top, bot) / (dist(left, right) + 1e-6)

def pt(lm, idx, w, h):
    return (lm[idx].x * w, lm[idx].y * h)

def compute_stress(lm, w, h):
    iod         = dist(pt(lm, L_EYE_L, w, h), pt(lm, R_EYE_R, w, h)) + 1e-6
    brow_furrow = 1.0 - np.clip(dist(pt(lm, L_BROW_IN, w, h), pt(lm, R_BROW_IN, w, h)) / iod, 0, 1)
    l_ear       = ear(pt(lm, L_EYE_TOP, w, h), pt(lm, L_EYE_BOT, w, h),
                      pt(lm, L_EYE_L,   w, h), pt(lm, L_EYE_R,   w, h))
    r_ear       = ear(pt(lm, R_EYE_TOP, w, h), pt(lm, R_EYE_BOT, w, h),
                      pt(lm, R_EYE_L,   w, h), pt(lm, R_EYE_R,   w, h))
    eye_tense   = 1.0 - np.clip(min(l_ear, r_ear) * 5, 0, 1)
    asym        = abs(l_ear - r_ear) / (max(l_ear, r_ear) + 1e-6)
    mouth_open  = dist(pt(lm, M_TOP, w, h), pt(lm, M_BOT, w, h)) / iod
    lip_press   = 1.0 - np.clip(mouth_open * 8, 0, 1)
    chin_raise  = 1.0 - np.clip(dist(pt(lm, LOWER_LIP, w, h), pt(lm, CHIN, w, h)) / iod, 0, 1)
    return float(np.clip(
        0.35 * brow_furrow + 0.25 * eye_tense +
        0.20 * asym + 0.10 * lip_press + 0.10 * chin_raise,
        0, 1
    ))

# ── UI ────────────────────────────────────────────────────────────────────────
score_buffer = deque(maxlen=15)

def stress_colour(s):
    if s < 0.42:   return (80, 200, 80)
    elif s < 0.46: return (0, 200, 255)
    else:          return (50, 50, 220)

def stress_label(s):
    if s < 0.42:   return "CALM"
    elif s < 0.46: return "MILD STRESS"
    else:          return "STRESSED"

def draw_overlay(frame, smoothed):
    h, w    = frame.shape[:2]
    colour  = stress_colour(smoothed)
    overlay = frame.copy()

    # Top banner
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Project name + status
    cv2.putText(frame, "EmotionLens", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, stress_label(smoothed), (14, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)

    # Score bar bottom right
    bx, by, bw, bh = w - 180, h - 40, 160, 18
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * smoothed), by + bh), colour, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (120, 120, 120), 1)
    cv2.putText(frame, f"{smoothed:.0%}", (bx + bw + 6, by + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Raw score
    cv2.putText(frame, f"score: {smoothed:.3f}", (14, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q = quit  |  S = screenshot", (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    options  = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
    )
    detector = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Try changing VideoCapture(0) to VideoCapture(1)")
        return

    print("EmotionLens running...")
    print("Press Q to quit, S to save a screenshot")
    print("Watch the 'score:' number at bottom left to calibrate thresholds for your face")

    shot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        if result.face_landmarks:
            lm    = result.face_landmarks[0]
            score = compute_stress(lm, w, h)
            score_buffer.append(score)
            for l in lm:
                cv2.circle(frame, (int(l.x * w), int(l.y * h)), 1, (0, 180, 150), -1)
        else:
            score_buffer.append(0.0)

        smoothed = float(np.mean(score_buffer)) if score_buffer else 0.0
        draw_overlay(frame, smoothed)
        cv2.imshow("EmotionLens - Stress Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            shot_count += 1
            filename = f"screenshot_{shot_count}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Done.")

if __name__ == "__main__":
    main()
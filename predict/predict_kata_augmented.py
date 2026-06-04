"""
predict_kata_augmented.py
=========================
Predict real-time untuk model BISINDO Kata yang dihasilkan oleh
training dengan augmentasi (bisindo_holistic_gru_new.h5).

Struktur mengikuti Predict_mix_new.py tapi:
- Hanya mode KATA (tidak ada huruf/angka)
- Load model dari kiro_train/bisindo_holistic_gru_new.h5
- Load label encoder dari kiro_train/label_encoder_new.pkl

Kontrol keyboard:
    R = reset buffer
    Q = keluar
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque, Counter
from pathlib import Path


# ============================================================
# 1. PATH SETUP
# ============================================================

SCRIPT_DIR   = Path(__file__).resolve().parent
KIRO_TRAIN   = SCRIPT_DIR.parent / "kiro_train"

MODEL_PATH   = KIRO_TRAIN / "bisindo_holistic_gru_new.h5"
ENCODER_PATH = KIRO_TRAIN / "label_encoder_new.pkl"


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("[INFO] Memuat model kata augmented...")
kata_model = load_model(str(MODEL_PATH), compile=False)

with open(str(ENCODER_PATH), "rb") as f:
    kata_le = pickle.load(f)

print("[INFO] Model berhasil dimuat!")
print(f"      Input shape : {kata_model.input_shape}")
print(f"      Jumlah kelas: {len(kata_le.classes_)}")
print("      R = reset buffer | Q = keluar")


# ============================================================
# 3. KONFIGURASI
# ============================================================

MAX_FRAMES       = 50      # harus sama dengan SEQUENCE_LENGTH saat training
NUM_FEATURES     = 258     # Pose 132 + tangan kiri 63 + tangan kanan 63
KATA_THRESHOLD   = 0.75    # minimum confidence untuk diterima
PREDICT_EVERY    = 10      # predict tiap N frame (biar tidak terlalu sering)
SMOOTHING_WINDOW = 5       # ambil label paling sering dari N prediksi terakhir


# ============================================================
# 4. INISIALISASI MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# 5. EKSTRAKSI FITUR (harus identik dengan training)
# ============================================================

def extract_holistic_landmarks(results) -> np.ndarray:
    """
    Pose       : 33 landmark x 4 fitur = 132
    Left hand  : 21 landmark x 3 fitur = 63
    Right hand : 21 landmark x 3 fitur = 63
    Total      = 258 fitur
    """
    if results.pose_landmarks:
        pose = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(33 * 4)

    if results.left_hand_landmarks:
        left_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.left_hand_landmarks.landmark
        ]).flatten()
    else:
        left_hand = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        right_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.right_hand_landmarks.landmark
        ]).flatten()
    else:
        right_hand = np.zeros(21 * 3)

    return np.concatenate([pose, left_hand, right_hand]).astype(np.float32)


# ============================================================
# 6. SMOOTHING
# ============================================================

def get_stable_prediction(prediction_history):
    """Ambil label paling sering + rata-rata confidence-nya."""
    if len(prediction_history) == 0:
        return "", 0.0

    labels           = [item[0] for item in prediction_history]
    most_common_label = Counter(labels).most_common(1)[0][0]
    confidences      = [item[1] for item in prediction_history
                        if item[0] == most_common_label]
    avg_confidence   = sum(confidences) / len(confidences)

    return most_common_label, avg_confidence


# ============================================================
# 7. UI HELPERS
# ============================================================

BOX_COLOR = (0, 255, 255)   # kuning — beda dari mix yang hijau/biru


def draw_landmarks(frame, results):
    """Gambar skeleton tangan + pose di frame."""
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )


def draw_prediction(frame, label: str, confidence: float):
    """Tampilkan teks prediksi besar di tengah bawah frame."""
    h, w = frame.shape[:2]
    if not label:
        return

    # kotak background
    cv2.rectangle(frame, (0, h // 2 - 70), (w, h // 2 + 35), (0, 0, 0), -1)

    # teks label
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 4)[0]
    label_x    = max(10, (w - label_size[0]) // 2)
    cv2.putText(frame, label, (label_x, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, BOX_COLOR, 4)

    # teks confidence
    conf_text = f"{confidence:.1f}%"
    conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    conf_x    = max(10, (w - conf_size[0]) // 2)
    cv2.putText(frame, conf_text, (conf_x, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)


def draw_header(frame):
    cv2.rectangle(frame, (0, 0), (420, 45), (0, 0, 0), -1)
    cv2.putText(frame, "Mode: KATA AUGMENTED",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, BOX_COLOR, 2)


def draw_footer(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "R = Reset Buffer | Q = Keluar",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (220, 220, 220), 1)


def draw_buffer_bar(frame, sequence_len: int):
    h, w = frame.shape[:2]
    progress = int((sequence_len / MAX_FRAMES) * (w - 40))
    cv2.rectangle(frame, (20, h - 65), (w - 20, h - 45), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 65), (20 + progress, h - 45), BOX_COLOR, -1)
    cv2.putText(frame, f"Buffer: {sequence_len}/{MAX_FRAMES}",
                (20, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)


def draw_top_predictions(frame, preds_array, class_names, top_n=3):
    """Tampilkan top-N prediksi di sudut kanan atas."""
    h, w = frame.shape[:2]
    top_indices = np.argsort(preds_array)[::-1][:top_n]

    cv2.rectangle(frame, (w - 230, 55), (w, 55 + top_n * 30 + 10), (0, 0, 0), -1)
    cv2.putText(frame, "Top prediksi:", (w - 225, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    for rank, idx in enumerate(top_indices):
        label   = class_names[idx]
        conf    = preds_array[idx]
        y_pos   = 75 + (rank + 1) * 25
        color   = BOX_COLOR if rank == 0 else (160, 160, 160)
        cv2.putText(frame, f"{rank+1}. {label} ({conf*100:.0f}%)",
                    (w - 225, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)


# ============================================================
# 8. MAIN LOOP
# ============================================================

def predict_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam tidak bisa dibuka.")
        return

    sequence           = deque(maxlen=MAX_FRAMES)
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    frame_count        = 0
    current_label      = "MULAI GERAKAN"
    current_conf       = 0.0
    last_preds         = None

    print("\n[INFO] Webcam aktif. R=Reset | Q=Keluar")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame webcam gagal dibaca.")
            break

        frame   = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = holistic.process(img_rgb)
        img_rgb.flags.writeable = True

        # — Ekstraksi landmark & akumulasi sequence
        features = extract_holistic_landmarks(results)
        sequence.append(features)
        frame_count += 1

        # — Prediksi periodik
        if len(sequence) == MAX_FRAMES and frame_count % PREDICT_EVERY == 0:
            seq_input = np.array(sequence, dtype=np.float32)
            seq_input = seq_input.reshape(1, MAX_FRAMES, NUM_FEATURES)

            preds     = kata_model.predict(seq_input, verbose=0)[0]
            last_preds = preds

            pred_idx   = int(np.argmax(preds))
            pred_conf  = float(preds[pred_idx])
            pred_label = kata_le.inverse_transform([pred_idx])[0]

            if pred_conf >= KATA_THRESHOLD:
                prediction_history.append((pred_label, pred_conf))
                stable_label, stable_conf = get_stable_prediction(prediction_history)
                current_label = stable_label.upper()
                current_conf  = stable_conf * 100
            else:
                current_label = "TIDAK YAKIN"
                current_conf  = pred_conf * 100

        # — Render
        draw_landmarks(frame, results)
        draw_prediction(frame, current_label, current_conf)
        draw_buffer_bar(frame, len(sequence))
        draw_header(frame)
        draw_footer(frame)

        if last_preds is not None:
            draw_top_predictions(frame, last_preds, kata_le.classes_, top_n=3)

        cv2.imshow("BISINDO Kata Augmented", frame)

        # — Keyboard
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key in (ord("r"), ord("R")):
            sequence.clear()
            prediction_history.clear()
            frame_count   = 0
            current_label = "MULAI ULANG"
            current_conf  = 0.0
            last_preds    = None
            print("[INFO] Buffer direset.")

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
    print("[INFO] Selesai.")


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    predict_webcam()

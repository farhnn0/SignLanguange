import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque

# ─────────────────────────────────────────
# 1. LOAD MODEL & LABEL
# ─────────────────────────────────────────
print("[INFO] Memuat model kata...")
with tf.device('/CPU:0'):
    kata_model = load_model("kata_model.h5")

with open("kata_labels.pkl", "rb") as f:
    kata_le = pickle.load(f)

print(f"[INFO] Model siap. Kelas: {list(kata_le.classes_)}")

# ─────────────────────────────────────────
# 2. KONFIGURASI
# ─────────────────────────────────────────
MAX_FRAMES   = 50    # Harus sama dengan saat training
NUM_FEATURES = 225   # Harus sama dengan saat training
THRESHOLD    = 0.6   # Minimum confidence untuk tampilkan hasil
PREDICT_EVERY = 25   # Prediksi setiap 25 frame baru (sliding window)

# ─────────────────────────────────────────
# 3. INISIALISASI MEDIAPIPE
# ─────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────────────────────────────
# 4. FUNGSI EKSTRAKSI FITUR (225 fitur per frame)
# ─────────────────────────────────────────
def extract_features(results):
    """
    Ekstrak 225 fitur dari MediaPipe Holistic per frame:
    - Pose: 33 landmark × 3 = 99 fitur
    - Tangan kiri: 21 landmark × 3 = 63 fitur
    - Tangan kanan: 21 landmark × 3 = 63 fitur
    Total = 225 fitur
    """
    pose  = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(99)
    left  = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(63)
    right = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, left, right])

# ─────────────────────────────────────────
# 5. WEBCAM REAL-TIME (SLIDING WINDOW)
# ─────────────────────────────────────────
def predict_webcam():
    cap = cv2.VideoCapture(0)

    # Buffer sliding window — otomatis buang frame lama saat penuh
    sequence = deque(maxlen=MAX_FRAMES)

    current_label = ""
    current_conf  = 0.0
    frame_count   = 0  # Hitung frame untuk trigger prediksi

    print("\n[INFO] Webcam aktif — prediksi otomatis setiap 25 frame baru.")
    print("       Tekan Q untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        # Gambar landmark tangan
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                   mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                   mp_holistic.HAND_CONNECTIONS)

        # Ekstrak fitur dan tambah ke buffer
        features = extract_features(results)
        sequence.append(features)
        frame_count += 1

        # Prediksi otomatis setiap PREDICT_EVERY frame, setelah buffer penuh
        if len(sequence) == MAX_FRAMES and frame_count % PREDICT_EVERY == 0:
            seq_array = np.array(list(sequence), dtype=np.float32)
            seq_input = seq_array.reshape(1, MAX_FRAMES, NUM_FEATURES)

            with tf.device('/CPU:0'):
                preds = kata_model.predict(seq_input, verbose=0)

            pred_idx  = np.argmax(preds[0])
            pred_conf = preds[0][pred_idx]

            if pred_conf >= THRESHOLD:
                current_label = kata_le.inverse_transform([pred_idx])[0].upper()
                current_conf  = pred_conf * 100
                print(f"[PREDIKSI] {current_label} ({current_conf:.1f}%)")
            else:
                # Confidence rendah — tidak tampilkan
                current_label = ""
                current_conf  = 0.0

        # Tampilkan hasil prediksi di tengah layar
        if current_label:
            cv2.rectangle(frame, (0, h // 2 - 60), (w, h // 2 + 20), (0, 0, 0), -1)

            label_size = cv2.getTextSize(current_label, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 4)[0]
            label_x = (w - label_size[0]) // 2
            cv2.putText(frame, current_label, (label_x, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 4)

            conf_text = f"{current_conf:.1f}%"
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            conf_x = (w - conf_size[0]) // 2
            cv2.putText(frame, conf_text, (conf_x, h // 2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        # Progress buffer di bagian bawah
        progress = int((len(sequence) / MAX_FRAMES) * (w - 20))
        cv2.rectangle(frame, (10, h - 50), (w - 10, h - 30), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h - 50), (10 + progress, h - 30), (0, 200, 255), -1)
        cv2.putText(frame, f"Buffer: {len(sequence)}/{MAX_FRAMES}",
                    (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Status di pojok kiri atas
        cv2.rectangle(frame, (0, 0), (200, 40), (0, 0, 0), -1)
        cv2.putText(frame, "Mode: KATA", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Instruksi di pojok bawah
        cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "Q=Keluar",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("BISINDO Kata Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ─────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    predict_webcam()
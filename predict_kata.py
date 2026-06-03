import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque, Counter


# =========================
# 1. LOAD MODEL KATA BARU
# =========================

print("[INFO] Memuat model kata BISINDO baru...")

kata_model = load_model("bisindo_holistic_gru.h5", compile=False)

with open("label_encoder.pkl", "rb") as f:
    kata_le = pickle.load(f)

print("[INFO] Model berhasil dimuat.")
print("[INFO] Input shape model:", kata_model.input_shape)
print("[INFO] Jumlah kelas:", len(kata_le.classes_))
print("[INFO] Label:", list(kata_le.classes_))


# =========================
# 2. KONFIGURASI
# =========================

MAX_FRAMES = 50          # Harus sama dengan SEQUENCE_LENGTH saat training
NUM_FEATURES = 258       # Pose 132 + left hand 63 + right hand 63
THRESHOLD = 0.75         # Minimal confidence agar prediksi dianggap yakin
PREDICT_EVERY = 10       # Prediksi setiap 10 frame agar tidak terlalu berat
SMOOTHING_WINDOW = 5     # Untuk menstabilkan hasil prediksi


# =========================
# 3. SETUP MEDIAPIPE HOLISTIC
# =========================

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# 4. EKSTRAKSI LANDMARK
# =========================

def extract_holistic_landmarks(results):
    """
    Harus sama persis dengan fitur saat training.

    Pose       : 33 landmark x 4 fitur = 132
    Left hand  : 21 landmark x 3 fitur = 63
    Right hand : 21 landmark x 3 fitur = 63

    Total = 258 fitur
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

    return np.concatenate([pose, left_hand, right_hand])


def get_stable_prediction(prediction_history):
    """
    Ambil prediksi yang paling sering muncul dari beberapa prediksi terakhir.
    Ini bikin output webcam tidak terlalu kedap-kedip.
    """
    if len(prediction_history) == 0:
        return "", 0.0

    labels = [item[0] for item in prediction_history]
    most_common_label = Counter(labels).most_common(1)[0][0]

    confidences = [
        item[1] for item in prediction_history
        if item[0] == most_common_label
    ]

    avg_confidence = sum(confidences) / len(confidences)

    return most_common_label, avg_confidence


# =========================
# 5. WEBCAM PREDICTION
# =========================

def predict_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam tidak bisa dibuka.")
        return

    sequence = deque(maxlen=MAX_FRAMES)
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)

    current_label = "MULAI GERAKAN"
    current_conf = 0.0
    frame_count = 0

    print("[INFO] Webcam aktif.")
    print("[INFO] Tekan Q untuk keluar.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Gagal membaca frame webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False

        results = holistic.process(img_rgb)

        img_rgb.flags.writeable = True

        # Gambar landmark tangan
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

        # Optional: gambar pose tubuh
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        features = extract_holistic_landmarks(results)
        sequence.append(features)
        frame_count += 1

        # Prediksi hanya jika buffer sudah penuh 50 frame
        if len(sequence) == MAX_FRAMES and frame_count % PREDICT_EVERY == 0:
            seq_input = np.array(sequence, dtype=np.float32)
            seq_input = seq_input.reshape(1, MAX_FRAMES, NUM_FEATURES)

            preds = kata_model.predict(seq_input, verbose=0)[0]

            pred_idx = int(np.argmax(preds))
            pred_conf = float(preds[pred_idx])
            pred_label = kata_le.inverse_transform([pred_idx])[0]

            if pred_conf >= THRESHOLD:
                prediction_history.append((pred_label, pred_conf))
                stable_label, stable_conf = get_stable_prediction(prediction_history)

                current_label = stable_label.upper()
                current_conf = stable_conf * 100
            else:
                current_label = "TIDAK YAKIN"
                current_conf = pred_conf * 100

        # =========================
        # UI DISPLAY
        # =========================

        # Background teks utama
        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f"Prediksi: {current_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"Confidence: {current_conf:.1f}%",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2
        )

        # Buffer progress
        progress = int((len(sequence) / MAX_FRAMES) * (w - 40))
        cv2.rectangle(frame, (20, h - 55), (w - 20, h - 35), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 55), (20 + progress, h - 35), (0, 255, 255), -1)

        cv2.putText(
            frame,
            f"Buffer: {len(sequence)}/{MAX_FRAMES}",
            (20, h - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "R = Reset | Q = Keluar",
            (20, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            2
        )

        cv2.imshow("Prediksi Kata BISINDO - GRU Holistic", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        elif key == ord("r") or key == ord("R"):
            sequence.clear()
            prediction_history.clear()
            frame_count = 0
            current_label = "MULAI ULANG"
            current_conf = 0.0
            print("[INFO] Buffer direset ke 0.")

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


# =========================
# 6. MAIN
# =========================

if __name__ == "__main__":
    predict_webcam()
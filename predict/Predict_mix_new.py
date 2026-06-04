import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque, Counter


# ============================================================
# 1. LOAD SEMUA MODEL
# ============================================================

print("[INFO] Memuat model huruf...")
with open("../hasil_train_label/huruf_model.pkl", "rb") as f:
    huruf_model = pickle.load(f)

with open("../hasil_train_label/huruf_labels.pkl", "rb") as f:
    huruf_le = pickle.load(f)


print("[INFO] Memuat model angka...")
with open("../hasil_train_label/angka_model.pkl", "rb") as f:
    angka_model = pickle.load(f)

with open("../hasil_train_label/angka_labels.pkl", "rb") as f:
    angka_le = pickle.load(f)


print("[INFO] Memuat model kata baru...")
kata_model = load_model("../hasil_train_label/bisindo_holistic_gru.h5", compile=False)

with open("../hasil_train_label/label_encoder.pkl", "rb") as f:
    kata_le = pickle.load(f)


print("[INFO] Semua model berhasil dimuat!")
print("      H = mode Huruf")
print("      A = mode Angka")
print("      K = mode Kata")
print("      R = reset buffer kata")
print("      Q = keluar")
print("[INFO] Input shape model kata:", kata_model.input_shape)
print("[INFO] Jumlah label kata:", len(kata_le.classes_))


# ============================================================
# 2. KONFIGURASI
# ============================================================

# Mode kata baru
MAX_FRAMES = 50          # Harus sama dengan SEQUENCE_LENGTH saat training
NUM_FEATURES = 258       # Pose 132 + tangan kiri 63 + tangan kanan 63
KATA_THRESHOLD = 0.75
PREDICT_EVERY = 10
SMOOTHING_WINDOW = 5

# Mode huruf/angka
HURUF_THRESHOLD = 0.0    # Boleh dinaikkan kalau mau filter confidence
ANGKA_THRESHOLD = 0.0


# ============================================================
# 3. INISIALISASI MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

# Huruf & Angka pakai Hands, lebih ringan
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Kata pakai Holistic, sesuai training baru
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
# 4. FUNGSI EKSTRAKSI FITUR HURUF & ANGKA
# ============================================================

def extract_126(hand_landmarks_list):
    """
    Untuk model huruf lama:
    2 tangan x 21 landmark x 3 fitur = 126.
    Kalau cuma 1 tangan, tangan kedua diisi 0.
    """
    features = []

    for lm in hand_landmarks_list[0].landmark:
        features.extend([lm.x, lm.y, lm.z])

    if len(hand_landmarks_list) > 1:
        for lm in hand_landmarks_list[1].landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features += [0.0] * 63

    return np.array(features, dtype=np.float32).reshape(1, -1)


def extract_63(hand_landmarks_list):
    """
    Untuk model angka lama:
    1 tangan x 21 landmark x 3 fitur = 63.
    """
    features = []

    for lm in hand_landmarks_list[0].landmark:
        features.extend([lm.x, lm.y, lm.z])

    return np.array(features, dtype=np.float32).reshape(1, -1)


# ============================================================
# 5. FUNGSI EKSTRAKSI FITUR KATA BARU
# ============================================================

def extract_holistic_landmarks(results):
    """
    Harus sama persis dengan training model kata baru.

    Pose       : 33 landmark x 4 fitur = 132
    Left hand  : 21 landmark x 3 fitur = 63
    Right hand : 21 landmark x 3 fitur = 63

    Total = 258 fitur.
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


def get_stable_prediction(prediction_history):
    """
    Biar hasil kata tidak terlalu kedap-kedip.
    Ambil label yang paling sering muncul dari beberapa prediksi terakhir.
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


# ============================================================
# 6. FUNGSI UI
# ============================================================

def draw_hand_box(frame, hand_lm, box_color, h, w):
    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

    x_coords = [lm.x * w for lm in hand_lm.landmark]
    y_coords = [lm.y * h for lm in hand_lm.landmark]

    x1 = max(0, int(min(x_coords)) - 20)
    y1 = max(0, int(min(y_coords)) - 20)
    x2 = min(w, int(max(x_coords)) + 20)
    y2 = min(h, int(max(y_coords)) + 20)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    return x1, y1, x2, y2


def draw_big_center_text(frame, label, confidence, color):
    h, w = frame.shape[:2]

    if not label:
        return

    cv2.rectangle(frame, (0, h // 2 - 70), (w, h // 2 + 35), (0, 0, 0), -1)

    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 4)[0]
    label_x = max(10, (w - label_size[0]) // 2)

    cv2.putText(
        frame,
        label,
        (label_x, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.4,
        color,
        4
    )

    conf_text = f"{confidence:.1f}%"
    conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    conf_x = max(10, (w - conf_size[0]) // 2)

    cv2.putText(
        frame,
        conf_text,
        (conf_x, h // 2 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (220, 220, 220),
        2
    )


def draw_mode_header(frame, current_mode, color):
    cv2.rectangle(frame, (0, 0), (330, 45), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Mode: {current_mode}",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        color,
        2
    )


def draw_footer(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "H=Huruf | A=Angka | K=Kata | R=Reset Kata | Q=Keluar",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )


def draw_buffer_bar(frame, sequence_len, color):
    h, w = frame.shape[:2]

    progress = int((sequence_len / MAX_FRAMES) * (w - 40))

    cv2.rectangle(frame, (20, h - 65), (w - 20, h - 45), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 65), (20 + progress, h - 45), color, -1)

    cv2.putText(
        frame,
        f"Buffer: {sequence_len}/{MAX_FRAMES}",
        (20, h - 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


# ============================================================
# 7. WEBCAM REAL-TIME
# ============================================================

def predict_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam tidak bisa dibuka.")
        return

    current_mode = "HURUF"
    current_label = ""
    current_conf = 0.0

    sequence = deque(maxlen=MAX_FRAMES)
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    frame_count = 0

    print("\n[INFO] Webcam aktif.")
    print("[INFO] H=Huruf | A=Angka | K=Kata | R=Reset Kata | Q=Keluar")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Frame webcam gagal dibaca.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if current_mode == "HURUF":
            box_color = (0, 255, 0)
        elif current_mode == "ANGKA":
            box_color = (255, 100, 0)
        else:
            box_color = (0, 255, 255)

        # ====================================================
        # MODE HURUF & ANGKA
        # ====================================================
        if current_mode in ["HURUF", "ANGKA"]:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)

                if current_mode == "HURUF":
                    features = extract_126(results.multi_hand_landmarks)
                    pred_idx = huruf_model.predict(features)[0]
                    pred_prob = huruf_model.predict_proba(features)[0]

                    current_label = huruf_le.inverse_transform([pred_idx])[0]
                    current_conf = float(pred_prob[pred_idx]) * 100

                else:
                    features = extract_63(results.multi_hand_landmarks)
                    pred_idx = angka_model.predict(features)[0]
                    pred_prob = angka_model.predict_proba(features)[0]

                    current_label = angka_le.inverse_transform([pred_idx])[0]
                    current_conf = float(pred_prob[pred_idx]) * 100

                for i, hand_lm in enumerate(results.multi_hand_landmarks):
                    x1, y1, x2, y2 = draw_hand_box(frame, hand_lm, box_color, h, w)

                    if i == 0:
                        strip_y1 = y2
                        strip_y2 = y2 + 70

                        if strip_y2 > h:
                            strip_y1 = y1 - 70
                            strip_y2 = y1

                        cv2.rectangle(frame, (x1, strip_y1), (x2, strip_y2), (0, 0, 0), -1)

                        label_text = str(current_label)
                        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
                        label_x = x1 + max(0, (x2 - x1 - label_size[0]) // 2)

                        cv2.putText(
                            frame,
                            label_text,
                            (label_x, strip_y1 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,
                            box_color,
                            3
                        )

                        conf_text = f"{current_conf:.1f}%"
                        conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        conf_x = x1 + max(0, (x2 - x1 - conf_size[0]) // 2)

                        cv2.putText(
                            frame,
                            conf_text,
                            (conf_x, strip_y1 + 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (220, 220, 220),
                            2
                        )

                cv2.putText(
                    frame,
                    f"{num_hands} tangan",
                    (w - 170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

            else:
                current_label = ""
                current_conf = 0.0

        # ====================================================
        # MODE KATA BARU
        # ====================================================
        else:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False

            results = holistic.process(img_rgb)

            img_rgb.flags.writeable = True

            # Gambar tangan
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

            # Gambar pose tubuh
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )

            features = extract_holistic_landmarks(results)
            sequence.append(features)
            frame_count += 1

            if len(sequence) == MAX_FRAMES and frame_count % PREDICT_EVERY == 0:
                seq_input = np.array(sequence, dtype=np.float32)
                seq_input = seq_input.reshape(1, MAX_FRAMES, NUM_FEATURES)

                preds = kata_model.predict(seq_input, verbose=0)[0]

                pred_idx = int(np.argmax(preds))
                pred_conf = float(preds[pred_idx])
                pred_label = kata_le.inverse_transform([pred_idx])[0]

                if pred_conf >= KATA_THRESHOLD:
                    prediction_history.append((pred_label, pred_conf))
                    stable_label, stable_conf = get_stable_prediction(prediction_history)

                    current_label = stable_label.upper()
                    current_conf = stable_conf * 100
                else:
                    current_label = "TIDAK YAKIN"
                    current_conf = pred_conf * 100

            draw_big_center_text(frame, current_label, current_conf, box_color)
            draw_buffer_bar(frame, len(sequence), box_color)

        # ====================================================
        # UI UMUM
        # ====================================================
        draw_mode_header(frame, current_mode, box_color)
        draw_footer(frame)

        cv2.imshow("BISINDO Recognition - Huruf Angka Kata", frame)

        # ====================================================
        # KEYBOARD CONTROL
        # ====================================================
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        elif key == ord("h") or key == ord("H"):
            current_mode = "HURUF"
            current_label = ""
            current_conf = 0.0
            sequence.clear()
            prediction_history.clear()
            frame_count = 0
            print("[INFO] Mode: HURUF")

        elif key == ord("a") or key == ord("A"):
            current_mode = "ANGKA"
            current_label = ""
            current_conf = 0.0
            sequence.clear()
            prediction_history.clear()
            frame_count = 0
            print("[INFO] Mode: ANGKA")

        elif key == ord("k") or key == ord("K"):
            current_mode = "KATA"
            current_label = "MULAI GERAKAN"
            current_conf = 0.0
            sequence.clear()
            prediction_history.clear()
            frame_count = 0
            print("[INFO] Mode: KATA")

        elif key == ord("r") or key == ord("R"):
            if current_mode == "KATA":
                sequence.clear()
                prediction_history.clear()
                frame_count = 0
                current_label = "MULAI ULANG"
                current_conf = 0.0
                print("[INFO] Buffer kata direset ke 0.")
            else:
                print("[INFO] Reset hanya dipakai untuk mode KATA.")

    cap.release()
    hands.close()
    holistic.close()
    cv2.destroyAllWindows()


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    predict_webcam()
import cv2                                  # Library buat buka kamera & gambar UI
import numpy as np                          # Buat ngitung matriks & array koordinat
import mediapipe as mp                      # AI Google buat ngedeteksi titik sendi tubuh
import pickle                               # Buat buka file model Random Forest (.pkl)
import tensorflow as tf                     # Mesin utama buat Deep Learning (LSTM)
from tensorflow.keras.models import load_model # Buat ngeload otak LSTM (.h5)
from collections import deque               # Antrean memori biar frame muter terus

# ─────────────────────────────────────────
# 1. LOAD SEMUA MODEL & LABEL
# ─────────────────────────────────────────
print("[INFO] Memuat model huruf...")
with open("../hasil_train_label/huruf_model.pkl", "rb") as f:
    huruf_model = pickle.load(f)            # Buka otak Random Forest (Huruf)
with open("../hasil_train_label/huruf_labels.pkl", "rb") as f:
    huruf_le = pickle.load(f)               # Buka kamus label Huruf

print("[INFO] Memuat model angka...")
with open("../hasil_train_label/angka_model.pkl", "rb") as f:
    angka_model = pickle.load(f)            # Buka otak Random Forest (Angka)
with open("../hasil_train_label/angka_labels.pkl", "rb") as f:
    angka_le = pickle.load(f)               # Buka kamus label Angka

print("[INFO] Memuat model kata...")
with tf.device('/CPU:0'):                   # Paksa jalan di CPU biar laptop ga ngos-ngosan
    kata_model = load_model("../hasil_train_label/kata_model.h5")# Buka otak Deep Learning LSTM (Kata)
with open("../hasil_train_label/kata_labels.pkl", "rb") as f:
    kata_le = pickle.load(f)                # Buka kamus label Kata

print("[INFO] Semua model berhasil dimuat!")
print("      H = mode Huruf")
print("      A = mode Angka")
print("      K = mode Kata")
print("      Q = keluar")

# ─────────────────────────────────────────
# 2. KONFIGURASI
# ─────────────────────────────────────────
MAX_FRAMES    = 50           # Wajib 50! Sesuaiin sama jumlah frame pas training kemaren
NUM_FEATURES  = 225          # Total fitur titik (badan + tangan kiri + kanan)
THRESHOLD     = 0.6          # AI cuma berani nampilin teks kalau dia yakin minimal 60%
PREDICT_EVERY = 25           # Jeda tebakan. AI nebak tiap 25 frame biar ga ngelag

# ─────────────────────────────────────────
# 3. INISIALISASI MEDIAPIPE
# ─────────────────────────────────────────
# Setup buat mode Huruf & Angka (Cuma nge-track jari, lebih ringan)
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Setup buat mode Kata (Nge-track full postur badan + tangan, agak berat)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────────────────────────────
# 4. FUNGSI EKSTRAKSI FITUR (Visual ke Angka)
# ─────────────────────────────────────────
def extract_126(hand_landmarks_list):
    # Mode Huruf: Butuh 2 tangan (126 angka)
    features = []
    for lm in hand_landmarks_list[0].landmark:
        features.extend([lm.x, lm.y, lm.z])     # Ambil titik tangan dominan
    if len(hand_landmarks_list) > 1:
        for lm in hand_landmarks_list[1].landmark:
            features.extend([lm.x, lm.y, lm.z]) # Ambil titik tangan kedua (kalo ada)
    else:
        features += [0.0] * 63                  # Kalo cuma 1 tangan, sisanya diganjal nol
    return np.array(features).reshape(1, -1)    # Bikin jadi 1 baris memanjang

def extract_63(hand_landmarks_list):
    # Mode Angka: Cuma butuh 1 tangan (63 angka)
    features = []
    for lm in hand_landmarks_list[0].landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features).reshape(1, -1)

def extract_holistic(results):
    # Mode Kata: Gabungin Badan(99) + Kiri(63) + Kanan(63) = 225 angka
    pose  = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(99)  # Kalo badan ga masuk frame, isi nol
    left  = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(63)
    right = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, left, right])           # Satukan semuanya

# ─────────────────────────────────────────
# 5. FUNGSI GAMBAR KOTAK (Bounding Box)
# ─────────────────────────────────────────
def draw_hand(frame, hand_lm, box_color, h, w):
    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS) # Gambar tulang jari
    x_coords = [lm.x * w for lm in hand_lm.landmark]                  # Cari koordinat X tiap jari
    y_coords = [lm.y * h for lm in hand_lm.landmark]                  # Cari koordinat Y tiap jari
    x1 = max(0, int(min(x_coords)) - 20)                              # Pojok kiri atas kotak
    y1 = max(0, int(min(y_coords)) - 20)
    x2 = min(w, int(max(x_coords)) + 20)                              # Pojok kanan bawah kotak
    y2 = min(h, int(max(y_coords)) + 20)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)            # Render kotaknya ke layar
    return x1, y1, x2, y2

# ─────────────────────────────────────────
# 6. WEBCAM REAL-TIME (Main Program)
# ─────────────────────────────────────────
def predict_webcam():
    cap = cv2.VideoCapture(0)                   # Nyalain webcam
    current_mode  = "HURUF"                     # Mode awal pas baru running
    current_label = ""
    current_conf  = 0.0

    # Memori jangka pendek khusus buat mode Kata (Sliding window)
    sequence    = deque(maxlen=MAX_FRAMES)
    frame_count = 0

    print("\n[INFO] Webcam aktif.")

    while True:                                 # Looping terus selama kamera nyala
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)              # Layarnya dimirror biar geraknya ga kebalik
        h, w = frame.shape[:2]

        # Nentuin warna UI berdasarkan mode yang aktif
        if current_mode == "HURUF":
            box_color = (0, 255, 0)      # Hijau
        elif current_mode == "ANGKA":
            box_color = (255, 100, 0)    # Biru
        else:
            box_color = (0, 255, 255)    # Kuning

        # ── LOGIKA MODE HURUF & ANGKA (Prediksi tiap frame) ──
        if current_mode in ["HURUF", "ANGKA"]:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Mediapipe butuh format RGB
            results = hands.process(img_rgb)                 # Cari tangan di jepretan ini

            if results.multi_hand_landmarks:                 # Kalo tangan ketemu...
                num_hands = len(results.multi_hand_landmarks)

                if current_mode == "HURUF":
                    features   = extract_126(results.multi_hand_landmarks) # Tarik 126 titik
                    pred_idx   = huruf_model.predict(features)[0]          # Langsung tembak tebakan!
                    pred_prob  = huruf_model.predict_proba(features)[0]
                    current_label = huruf_le.inverse_transform([pred_idx])[0] # Terjemahin balik ke teks
                    current_conf  = pred_prob[pred_idx] * 100
                else:
                    features   = extract_63(results.multi_hand_landmarks)  # Tarik 63 titik
                    pred_idx   = angka_model.predict(features)[0]
                    pred_prob  = angka_model.predict_proba(features)[0]
                    current_label = angka_le.inverse_transform([pred_idx])[0]
                    current_conf  = pred_prob[pred_idx] * 100

                # Bikin kotak sama tulisan hasil tebakan di atas tangan
                for i, hand_lm in enumerate(results.multi_hand_landmarks):
                    x1, y1, x2, y2 = draw_hand(frame, hand_lm, box_color, h, w)

                    if i == 0:  # Teksnya ditampilin di tangan yang pertama aja
                        strip_y1 = y2
                        strip_y2 = y2 + 70
                        if strip_y2 > h:                    # Kalo mentok layar bawah, pindah ke atas
                            strip_y1 = y1 - 70
                            strip_y2 = y1

                        cv2.rectangle(frame, (x1, strip_y1), (x2, strip_y2), (0, 0, 0), -1) # Background teks

                        label_size = cv2.getTextSize(current_label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
                        label_x = x1 + (x2 - x1 - label_size[0]) // 2
                        cv2.putText(frame, current_label, (label_x, strip_y1 + 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, box_color, 3)            # Print teksnya

                        conf_size = cv2.getTextSize(f"{current_conf:.1f}%", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        conf_x = x1 + (x2 - x1 - conf_size[0]) // 2
                        cv2.putText(frame, f"{current_conf:.1f}%", (conf_x, strip_y1 + 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)      # Print persennya

                # Teks info deteksi berapa tangan di pojok atas
                cv2.putText(frame, f"{num_hands} tangan",
                            (w - 160, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # ── LOGIKA MODE KATA (Prediksi dinamis kumpulin frame) ──
        else:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)             # Cari titik sekujur tubuh

            # Render visual tulang tangan (buat estetika)
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                       mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                       mp_holistic.HAND_CONNECTIONS)

            # Tarik 225 angka trus masukin ke antrean memori
            features = extract_holistic(results)
            sequence.append(features)
            frame_count += 1

            # Mengeksekusi tebakan LSTM HANYA JIKA memori full (50) & waktunya pas (kelipatan 25)
            if len(sequence) == MAX_FRAMES and frame_count % PREDICT_EVERY == 0:
                seq_input = np.array(list(sequence), dtype=np.float32).reshape(1, MAX_FRAMES, NUM_FEATURES) # Bikin format buat LSTM
                with tf.device('/CPU:0'):
                    preds = kata_model.predict(seq_input, verbose=0)        # Tembak ke otak LSTM
                pred_idx  = np.argmax(preds[0])                             # Cari jawaban skor tertinggi
                pred_conf = preds[0][pred_idx]

                if pred_conf >= THRESHOLD:                                  # Lolos filter Threshold ga?
                    current_label = kata_le.inverse_transform([pred_idx])[0].upper() # Ambil teksnya
                    current_conf  = pred_conf * 100

            # Render teks tebakan mode Kata di tengah layar
            if current_label:
                cv2.rectangle(frame, (0, h // 2 - 60), (w, h // 2 + 20), (0, 0, 0), -1)
                label_size = cv2.getTextSize(current_label, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 4)[0]
                label_x = (w - label_size[0]) // 2
                cv2.putText(frame, current_label, (label_x, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, box_color, 4)
                conf_text = f"{current_conf:.1f}%"
                conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                conf_x = (w - conf_size[0]) // 2
                cv2.putText(frame, conf_text, (conf_x, h // 2 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

            # Render UI Loading bar buffer memori di bawah
            progress = int((len(sequence) / MAX_FRAMES) * (w - 20))
            cv2.rectangle(frame, (10, h - 50), (w - 10, h - 30), (50, 50, 50), -1)      # Background bar
            cv2.rectangle(frame, (10, h - 50), (10 + progress, h - 30), box_color, -1)  # Isi bar-nya
            cv2.putText(frame, f"Buffer: {len(sequence)}/{MAX_FRAMES}",
                        (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # ── UI MENU UMUM ──
        # Boks penunjuk Mode aktif di kiri atas
        cv2.rectangle(frame, (0, 0), (230, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Mode: {current_mode}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        # Boks petunjuk tombol keyboard di bawah
        cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "H=Huruf | A=Angka | K=Kata | Q=Keluar",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("BISINDO Recognition", frame)        # Munculin jendela aplikasinya

        # Deteksi pencetan keyboard user
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break                                       # Tutup aplikasi
        elif key == ord('h') or key == ord('H'):
            current_mode  = "HURUF"                     # Pindah ke mode Huruf
            current_label = ""
            sequence.clear()
            print("[INFO] Mode: HURUF")
        elif key == ord('a') or key == ord('A'):
            current_mode  = "ANGKA"                     # Pindah ke mode Angka
            current_label = ""
            sequence.clear()
            print("[INFO] Mode: ANGKA")
        elif key == ord('k') or key == ord('K'):
            current_mode  = "KATA"                      # Pindah ke mode Kata
            current_label = ""
            sequence.clear()
            frame_count = 0
            print("[INFO] Mode: KATA")

    # Beres-beres pas selesai
    cap.release()               # Matiin webcam
    cv2.destroyAllWindows()     # Tutup semua tab

# ─────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    predict_webcam()            # Perintah jalanin aplikasinya
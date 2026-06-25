import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 1. KONFIGURASI PATH & FILE
TRAIN_PATH = r"bisindo\images\train"  # Path ke folder data latih
VAL_PATH = r"bisindo\images\val"  # Path ke folder data validasi
MODEL_SAVE = "huruf_model.pkl"  # Nama file output untuk model
LABEL_SAVE = "huruf_labels.pkl"  # Nama file output untuk encoder label

# 2. INISIALISASI MEDIAPIPE
# Set max_hands=2 karena bahasa isyarat (BISINDO) sering pakai 2 tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # Mode gambar statis (bukan video stream/kamera)
    max_num_hands=2,  # Maksimal deteksi 2 tangan
    min_detection_confidence=0.3  # Toleransi 30% agar tangan kurang jelas tetap terdeteksi
)


# 3. EKSTRAKSI LANDMARK (FITUR)
def extract_landmarks(image_path):
    img = cv2.imread(image_path)  # Baca file gambar
    if img is None:
        return None  # Skip jika gambar rusak/kosong

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ubah format BGR (OpenCV) ke RGB (MediaPipe)
    results = hands.process(img_rgb)  # Proses deteksi rangka tangan

    # Skip jika tidak ada tangan terdeteksi
    if not results.multi_hand_landmarks:
        return None

    features = []

    # Ekstrak 63 fitur (21 titik koordinat X,Y,Z) dari tangan pertama
    for lm in results.multi_hand_landmarks[0].landmark:
        features.extend([lm.x, lm.y, lm.z])  # Tambahkan koordinat X, Y, Z ke dalam list

    # Jika ada tangan kedua, tambahkan 63 fiturnya (Total: 126 fitur)
    if len(results.multi_hand_landmarks) > 1:
        for lm in results.multi_hand_landmarks[1].landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        # PENTING: Padding angka 0 jika hanya 1 tangan, agar dimensi data tetap 126
        features += [0.0] * 63  # Isi sisa 63 fitur dengan angka 0

    return features  # Return list berisi 126 angka koordinat


# 4. LOAD DATASET
def load_dataset(base_path, split_name):
    X, y, skipped = [], [], 0  # X: list fitur, y: list label
    classes = sorted(os.listdir(base_path))  # Ambil nama folder sebagai label (A, B, C...)
    print(f"\n[INFO] Loading {split_name} — {len(classes)} kelas")

    for label in tqdm(classes, desc=split_name):  # TQDM untuk memunculkan progress bar
        folder = os.path.join(base_path, label)
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):  # Filter ekstensi gambar
                continue

            features = extract_landmarks(os.path.join(folder, fname))
            if features:
                X.append(features)  # Simpan array 126 fitur koordinat
                y.append(label.upper())  # Simpan nama kelas/huruf (kapital)
            else:
                skipped += 1  # Hitung gambar tanpa tangan

    print(f"[INFO] {split_name}: {len(X)} berhasil, {skipped} dilewati")
    return np.array(X), np.array(y)  # Konversi list ke bentuk NumPy Array


# 5. MAIN PIPELINE (TRAINING & EVALUASI)
X_train, y_train = load_dataset(TRAIN_PATH,
 "train")
X_val, y_val = load_dataset(VAL_PATH, "val")

print(f"\n[INFO] Shape X_train: {X_train.shape}")  # Target shape: (Jumlah Gambar, 126)

# Ubah label huruf (A, B, C) menjadi angka (0, 1, 2)
le = LabelEncoder()
le.fit(np.concatenate([y_train, y_val]))  # Pelajari semua label unik yang ada
y_train_enc = le.transform(y_train)  # Encode label untuk data training
y_val_enc = le.transform(y_val)  # Encode label untuk data validasi

# Klasifikasi menggunakan Random Forest (sangat cocok untuk data koordinat/tabular)
print("\n[INFO] Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,  # Menggunakan 200 "Pohon Keputusan"
    random_state=42,  # Agar hasil training konsisten setiap kali dijalankan
    n_jobs=-1,  # Gunakan semua core CPU agar proses lebih cepat
    verbose=1
)
model.fit(X_train, y_train_enc)  # Mulai proses training

# Hitung akurasi pada data validasi
acc = np.mean(model.predict(X_val) == y_val_enc) * 100  # Bandingkan prediksi vs jawaban asli
print(f"[INFO] Akurasi Validasi: {acc:.2f}%")

# Simpan model dan encoder untuk dipakai saat deteksi real-time (kamera)
with open(MODEL_SAVE, 'wb') as f:
    pickle.dump(model, f)  # Simpan model machine learning (Random Forest)
with open(LABEL_SAVE, 'wb') as f:
    pickle.dump(le, f)  # Simpan penerjemah label (Label Encoder)

print(f"\n[DONE] Model disimpan ke {MODEL_SAVE}")
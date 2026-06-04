import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 1. KONFIGURASI PATH & FILE
DATASET_PATH = r"sign-language-for-numbers"  # Folder utama berisi subfolder kelas (0, 1, 2...)
MODEL_SAVE = "angka_model.pkl"  # Nama file output untuk model yang sudah pintar
LABEL_SAVE = "angka_labels.pkl"  # Nama file output untuk penerjemah label
VAL_SPLIT = 0.2  # Porsi data validasi 20% (80% untuk training)

# 2. INISIALISASI MEDIAPIPE (DETEKSI RANGKA)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # Mode gambar statis (untuk dataset foto, bukan video)
    max_num_hands=2,  # Parameter bawaan, tapi untuk angka biasanya butuh 1 tangan
    min_detection_confidence=0.3  # Toleransi 30% agar tangan yang blur tetap terdeteksi
)

# 3. EKSTRAKSI LANDMARK (MENGUBAH GAMBAR JADI ANGKA)
def extract_landmarks(image_path):
    img = cv2.imread(image_path)  # Baca gambar menggunakan OpenCV
    if img is None:
        return None  # Skip jika gambar rusak

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konversi BGR (OpenCV) ke RGB (MediaPipe)
    results = hands.process(img_rgb)  # Cari rangka tangan di gambar

    # Jika tidak ada tangan yang terdeteksi, lewati gambar ini
    if not results.multi_hand_landmarks:
        return None

    features = []

    # Fokus mengambil landmark dari tangan pertama saja (index ke-0)
    # Terdapat 21 titik, masing-masing punya X, Y, Z. Total = 63 angka/fitur.
    for lm in results.multi_hand_landmarks[0].landmark:
        features.extend([lm.x, lm.y, lm.z])

    return features  # Return 1 baris list berisi 63 angka koordinat

# 4. LOAD DATASET (BACA SEMUA FOLDER)
def load_dataset(base_path):
    X, y, skipped = [], [], 0  # X: list fitur (63 angka), y: list label (kelas)
    classes = sorted(os.listdir(base_path))  # Jadikan nama folder sebagai kelas
    print(f"\n[INFO] Loading dataset — {len(classes)} kelas: {classes}")

    for label in tqdm(classes, desc="Processing"):  # TQDM memunculkan progress bar
        folder = os.path.join(base_path, label)
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):  # Filter format gambar
                continue

            features = extract_landmarks(os.path.join(folder, fname))
            if features:
                X.append(features)  # Simpan 63 fitur koordinat
                y.append(label)  # Simpan nama folder sebagai label
            else:
                skipped += 1  # Hitung foto yang gagal dideteksi tangannya

    print(f"[INFO] Total: {len(X)} berhasil, {skipped} dilewati")
    return np.array(X), np.array(y)  # Jadikan array NumPy agar bisa masuk ke model ML

# 5. MAIN PIPELINE (TRAINING & EVALUASI)
X, y = load_dataset(DATASET_PATH)

# Memecah 1 dataset utuh menjadi 2 bagian: 80% untuk belajar (Train), 20% untuk ujian (Val)
# "stratify=y" memastikan pembagian porsi setiap kelas merata (adil)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SPLIT, random_state=42, stratify=y
)
print(f"\n[INFO] Train: {len(X_train)} | Val: {len(X_val)}")

# Mengubah label teks menjadi angka berurutan (0, 1, 2...)
le = LabelEncoder()
le.fit(y)  # Pelajari semua jenis label yang ada
y_train_enc = le.transform(y_train)  # Terjemahkan label data training
y_val_enc = le.transform(y_val)  # Terjemahkan label data validasi
print(f"[INFO] Kelas: {list(le.classes_)}")

# Tahap Training Model
print("\n[INFO] Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,  # Menggunakan 200 percabangan "Pohon Keputusan"
    random_state=42,  # Agar skor akurasi tidak berubah-ubah saat di-run ulang
    n_jobs=-1,  # Pakai semua inti CPU agar proses training lebih ngebut
    verbose=1
)
model.fit(X_train, y_train_enc)  # Model mulai belajar dari 80% data

# Ujian Model: Hitung akurasi pada 20% data validasi yang belum pernah ia lihat
acc = np.mean(model.predict(X_val) == y_val_enc) * 100
print(f"[INFO] Akurasi Validasi: {acc:.2f}%")

# Simpan otak model (Random Forest) dan penerjemahnya (Encoder)
with open(MODEL_SAVE, 'wb') as f:
    pickle.dump(model, f)
with open(LABEL_SAVE, 'wb') as f:
    pickle.dump(le, f)

print(f"\n[DONE] Model disimpan ke {MODEL_SAVE}")
print(f"[DONE] Label disimpan ke {LABEL_SAVE}")
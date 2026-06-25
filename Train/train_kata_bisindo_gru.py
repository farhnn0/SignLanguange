import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Nonaktifkan CUDA agar tidak konflik di Windows (pakai DirectML/CPU)
import cv2
import json
import pickle
import numpy as np
import mediapipe as mp
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
gpus = tf.config.list_physical_devices("GPU")
print("Jumlah GPU:", len(gpus))
print("GPU:", gpus)

# =========================
# 1. KONFIGURASI
# =========================

DATASET_PATH = r"bisindo-kata-baru"  # Folder utama dataset video kata BISINDO
OUTPUT_PATH = r"processed_bisindo"   # Folder penyimpanan hasil ekstraksi landmark (.npy)

SEQUENCE_LENGTH = 50                 # Standar durasi: semua video disamakan ke 50 frame
MIN_DETECTION_CONFIDENCE = 0.5       # Batas minimum keyakinan MediaPipe saat mendeteksi tangan/pose pertama kali
MIN_TRACKING_CONFIDENCE = 0.5        # Batas minimum keyakinan saat melacak gerakan antar frame

USE_LABEL_FROM = "folder"
# "folder"  = label diambil dari nama folder, contoh: bisindo-kata-baru/Kata Kerja/Berdiri/Berdiri-a-1.mp4 -> "Berdiri"
# "filename" = label diambil dari nama file sebelum tanda "-", contoh: Berdiri-a-1.mp4 -> "Berdiri"

MODEL_PATH = "bisindo_holistic_gru.h5"      # Nama file output model deep learning (format Keras)
LABEL_ENCODER_PATH = "label_encoder.pkl"    # File penyimpan urutan/daftar nama kata
CLASS_NAMES_PATH = "class_names.json"       # File JSON berisi daftar nama kelas (untuk prediksi real-time)


# =========================
# 2. SETUP MEDIAPIPE
# =========================

mp_holistic = mp.solutions.holistic  # Inisialisasi modul Holistic dari MediaPipe (deteksi pose + kedua tangan sekaligus)


def extract_landmarks(results):
    """
    Mengubah hasil deteksi MediaPipe menjadi array angka koordinat (fitur).
    Jika salah satu bagian tubuh tidak terdeteksi, isi dengan angka 0 agar ukuran tetap seragam.

    Total fitur per frame:
    - Pose      : 33 titik x 4 nilai (x, y, z, visibility) = 132
    - Tangan Kiri : 21 titik x 3 nilai (x, y, z) = 63
    - Tangan Kanan: 21 titik x 3 nilai (x, y, z) = 63
    Total = 258 fitur per frame
    """

    # Pose: 33 landmark x 4 fitur = 132,Kepala (0-10), Tubuh (11-24), Kaki (25-32)
    if results.pose_landmarks:
        pose = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(33 * 4)  # Jika pose tidak terdeteksi, isi nol sebanyak 132

    # Left hand: 21 landmark x 3 fitur = 63
    if results.left_hand_landmarks:
        left_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.left_hand_landmarks.landmark
        ]).flatten()
    else:
        left_hand = np.zeros(21 * 3)  # Jika tangan kiri tidak terdeteksi, isi nol sebanyak 63

    # Right hand: 21 landmark x 3 fitur = 63
    if results.right_hand_landmarks:
        right_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.right_hand_landmarks.landmark
        ]).flatten()
    else:
        right_hand = np.zeros(21 * 3)  # Jika tangan kanan tidak terdeteksi, isi nol sebanyak 63

    return np.concatenate([pose, left_hand, right_hand])  # Gabung semuanya jadi 1 array panjang 258


def normalize_sequence(sequence, target_length=50):
    """
    Menyamakan jumlah frame semua video menjadi tepat 50 frame.
    Syarat mutlak model GRU: semua input harus berukuran seragam.

    - Jika video terlalu panjang  : ambil 50 frame secara MERATA dari seluruh durasi (bukan potong depan)
    - Jika video terlalu pendek   : padding dengan menduplikat FRAME TERAKHIR (bukan angka nol)
    - Jika sudah tepat 50 frame   : langsung pakai
    """

    sequence = np.array(sequence)

    if len(sequence) == 0:
        return None  # Jika video kosong/tidak bisa dibaca, lewati

    if len(sequence) == target_length:
        return sequence  # Sudah pas, tidak perlu diubah

    if len(sequence) > target_length:
        # Ambil 50 frame yang tersebar merata dari seluruh video (linspace)
        # Contoh: video 100 frame -> ambil frame ke 0, 2, 4, ..., 99
        # Ini menjaga informasi gerakan dari awal sampai akhir tetap terwakili
        indices = np.linspace(0, len(sequence) - 1, target_length).astype(int)
        return sequence[indices]

    # Jika kurang dari target_length: duplikat frame terakhir sebagai padding
    # Ini lebih baik dari angka nol karena tangan tidak "menghilang" mendadak di akhir
    pad_count = target_length - len(sequence)
    last_frame = sequence[-1]  # Ambil posisi tangan/pose di frame terakhir
    padding = np.repeat(last_frame[np.newaxis, :], pad_count, axis=0)  # Duplikat sebanyak kekurangan frame

    return np.concatenate([sequence, padding], axis=0)  # Gabung: data asli + frame duplikat


def get_label(video_path):
    """Mengambil nama kata (label) dari path video berdasarkan konfigurasi USE_LABEL_FROM."""
    if USE_LABEL_FROM == "folder":
        return os.path.basename(os.path.dirname(video_path))  # Ambil nama folder induk sebagai label

    filename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(filename)[0]

    # Contoh: Berdiri-a-1.mp4 -> ambil teks sebelum tanda "-" pertama -> "Berdiri"
    return name_without_ext.split("-")[0]


def collect_video_paths(dataset_path):
    """
    Mengumpulkan semua path file video dari 5 kategori yang digunakan untuk training.
    Folder lain (dataset_sd, Videos, WL-BISINDO) sengaja tidak diikutkan.
    """
    video_extensions = (".mp4", ".mov", ".avi", ".mkv")

    # Hanya 5 kategori ini yang dipakai — sesuai dengan kosa kata yang ingin dikenali
    allowed_categories = [
        "5W+1H",
        "Kata Ganti Orang",
        "Kata Kerja",
        "Kata Lainnya",
        "Kata Sifat"
    ]

    video_paths = []

    for category in allowed_categories:
        category_path = os.path.join(dataset_path, category)

        if not os.path.exists(category_path):
            print(f"Folder tidak ditemukan, dilewati: {category_path}")
            continue

        # os.walk menelusuri semua subfolder secara rekursif
        for root, dirs, files in os.walk(category_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_paths.append(os.path.join(root, file))

    return video_paths


# =========================
# 3. EXTRACT VIDEO KE LANDMARK
# =========================

def process_video(video_path, holistic):
    """
    Membaca satu file video frame per frame, lalu mengekstrak koordinat landmark
    dari setiap frame menggunakan MediaPipe Holistic.
    Hasil akhir: sequence (daftar array fitur) yang sudah dinormalisasi ke 50 frame.
    """
    cap = cv2.VideoCapture(video_path)  # Buka file video
    sequence = []

    if not cap.isOpened():
        print(f"Gagal membuka video: {video_path}")
        return None

    while True:
        ret, frame = cap.read()  # Baca frame satu per satu
        if not ret:
            break  # Hentikan jika sudah habis

        # MediaPipe membutuhkan format RGB, sedangkan OpenCV menghasilkan BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Optimasi: tandai sebagai read-only agar lebih cepat

        results = holistic.process(image)       # Deteksi pose + tangan di frame ini
        landmarks = extract_landmarks(results)  # Konversi hasil deteksi ke array 258 angka

        sequence.append(landmarks)  # Kumpulkan 1 baris fitur untuk frame ini

    cap.release()

    sequence = normalize_sequence(sequence, SEQUENCE_LENGTH)  # Samakan jadi tepat 50 frame

    return sequence


def extract_dataset():
    """
    Mengekstrak semua video di dataset menjadi array NumPy (X dan y).
    Jika X.npy dan y.npy sudah ada, proses ekstraksi dilewati untuk menghemat waktu.

    X: shape (jumlah_video, 50, 258) -> koordinat landmark semua video
    y: shape (jumlah_video,) -> nama kata untuk setiap video
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    X_path = os.path.join(OUTPUT_PATH, "X.npy")
    y_path = os.path.join(OUTPUT_PATH, "y.npy")

    if os.path.exists(X_path) and os.path.exists(y_path):
        # Cache sudah ada: langsung load tanpa perlu proses ulang (hemat waktu)
        print("File X.npy dan y.npy sudah ada. Lewati proses ekstraksi.")
        X = np.load(X_path)
        y = np.load(y_path, allow_pickle=True)
        return X, y

    video_paths = collect_video_paths(DATASET_PATH)
    print(f"Total video ditemukan: {len(video_paths)}")

    X = []
    y = []

    # Inisialisasi MediaPipe Holistic sekali untuk semua video (lebih efisien)
    with mp_holistic.Holistic(
        static_image_mode=False,       # Mode video (bukan gambar statis) agar tracking antar frame aktif
        model_complexity=1,            # 0=cepat, 1=seimbang, 2=akurat. Pakai 1 untuk keseimbangan
        smooth_landmarks=True,         # Haluskan pergerakan landmark agar tidak goyang-goyang
        enable_segmentation=False,     # Matikan fitur segmentasi (tidak diperlukan, hemat memori)
        refine_face_landmarks=False,   # Matikan refinement wajah (tidak relevan untuk isyarat tangan)
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    ) as holistic:

        for video_path in tqdm(video_paths, desc="Extracting landmarks"):
            label = get_label(video_path)           # Ambil nama kata dari path video
            sequence = process_video(video_path, holistic)  # Ekstrak 50 frame fitur

            if sequence is None:
                continue  # Lewati video yang gagal dibaca

            X.append(sequence)
            y.append(label)

    X = np.array(X)  # Konversi list ke NumPy array: shape (N, 50, 258)
    y = np.array(y)  # Konversi list ke NumPy array: shape (N,)

    print("Shape X:", X.shape)           # Contoh: (2500, 50, 258)
    print("Shape y:", y.shape)           # Contoh: (2500,)
    print("Jumlah label:", len(set(y)))  # Jumlah kata unik yang berhasil diekstrak
    print("Label:", sorted(set(y)))

    np.save(X_path, X)  # Simpan ke processed_bisindo/X.npy
    np.save(y_path, y)  # Simpan ke processed_bisindo/y.npy

    return X, y


# =========================
# 4. TRAIN MODEL GRU
# =========================

def build_model(sequence_length, num_features, num_classes):
    """
    Membangun arsitektur model GRU (Gated Recurrent Unit).
    GRU dipilih karena lebih ringan dari LSTM namun tetap mampu
    menangkap pola temporal/urutan dalam data gerakan.
    """
    model = Sequential([
        Input(shape=(sequence_length, num_features)),  # Input: (50 frame, 258 fitur)

        # GRU Layer 1: Baca pola gerakan dari frame ke frame
        # return_sequences=True karena hasilnya akan diteruskan ke GRU berikutnya
        # recurrent_dropout=0.1: dropout khusus di dalam sel GRU untuk mencegah overfitting
        GRU(
            128,                     # 128 unit memori GRU
            return_sequences=True,   # Teruskan output tiap frame ke layer berikutnya
            reset_after=False,       # Versi GRU yang kompatibel dengan DirectML Windows
            recurrent_dropout=0.1    # Dropout 10% di dalam recurrent connection
        ),
        BatchNormalization(),  # Normalisasi output antar batch agar training lebih stabil
        Dropout(0.3),          # Matikan 30% neuron acak untuk cegah overfitting

        # GRU Layer 2: Mempertajam kesimpulan pola gerakan
        # return_sequences=False karena ini GRU terakhir, cukup output 1 vektor ringkasan
        GRU(
            64,                      # 64 unit memori GRU (lebih kecil dari layer 1)
            reset_after=False,
            recurrent_dropout=0.1
        ),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),  # Jaringan saraf biasa untuk memproses kesimpulan GRU
        Dropout(0.3),

        # Output: jumlah neuron = jumlah kelas kata. Softmax menghasilkan probabilitas tiap kata
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adam: algoritma optimasi adaptif
        loss="sparse_categorical_crossentropy",  # Loss function untuk label integer (bukan one-hot)
        metrics=["accuracy"]
    )

    return model


def train():
    """Pipeline training lengkap: load data → encode label → split → build model → train → evaluasi → simpan."""

    X, y = extract_dataset()  # Load atau ekstrak data landmark dari video

    # Ubah label teks ("Makan", "Halo", dst) menjadi angka (0, 1, 2, ...)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_names = list(label_encoder.classes_)
    print("Class names:", class_names)

    # Simpan label encoder dan daftar kelas untuk dipakai saat prediksi real-time
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    num_samples = X.shape[0]      # Total video dalam dataset
    sequence_length = X.shape[1]  # Panjang sequence (50 frame)
    num_features = X.shape[2]     # Jumlah fitur per frame (258)
    num_classes = len(class_names)  # Jumlah kata yang dikenali (50 kelas)

    print("Jumlah sample:", num_samples)
    print("Sequence length:", sequence_length)
    print("Jumlah fitur:", num_features)
    print("Jumlah kelas:", num_classes)

    # Pembagian data: 80% untuk belajar (train), 20% untuk ujian (test)
    # stratify=y_encoded memastikan proporsi tiap kelas merata di train dan test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,     # Kunci acakan agar hasil pembagian konsisten setiap run
        stratify=y_encoded
    )

    model = build_model(sequence_length, num_features, num_classes)
    model.summary()

    callbacks = [
        # Hentikan training otomatis jika val_loss tidak membaik dalam 15 putaran berturut-turut
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True  # Kembalikan bobot terbaik saat training berhenti
        ),
        # Simpan model HANYA saat ada rekor akurasi validasi baru
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Turunkan learning rate sebesar 50% jika val_loss stagnan selama 5 putaran
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,  # Batas bawah learning rate agar tidak terlalu kecil
            verbose=1
        )
    ]

    # Mulai proses training
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),  # Data ujian dipakai untuk memantau overfitting
        epochs=50,        # Maksimal 50 putaran (bisa berhenti lebih awal karena EarlyStopping)
        batch_size=32,    # Model memproses 32 video sekaligus per langkah update
        callbacks=callbacks
    )

    # Evaluasi akhir pada data test
    print("Evaluasi model:")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    # Classification report: akurasi per kelas kata (precision, recall, f1-score)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Ambil indeks kelas dengan probabilitas tertinggi

    print(classification_report(
        y_test,
        y_pred_classes,
        target_names=class_names  # Tampilkan nama kata, bukan angka
    ))

    # Simpan model final
    model.save(MODEL_PATH)
    print(f"Model disimpan ke: {MODEL_PATH}")
    print(f"Label encoder disimpan ke: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    train()

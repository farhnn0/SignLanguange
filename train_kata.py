import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

# 1. KONFIGURASI PARAMETER
DATASET_PATH = r"bisindo_kata/npy"  # Folder dataset berupa matriks angka (.npy), bukan gambar
MODEL_SAVE = "kata_model.h5"  # Model Deep Learning (Keras) disimpan dengan format .h5
LABEL_SAVE = "kata_labels.pkl"  # Penyimpan urutan label/nama kata
MAX_FRAMES = 50  # Standar durasi untuk 1 kata (50 frame/jepretan)
NUM_FEATURES = 225  # Total fitur per frame (Pose + Tangan Kiri + Tangan Kanan)
VAL_SPLIT = 0.2  # Porsi ujian 20%, porsi belajar 80%
EPOCHS = 50  # Maksimal proses belajar diulang 50 kali
BATCH_SIZE = 16  # Model mencerna 16 video sekaligus dalam satu waktu


# 2. FUNGSI PADDING & TRIMMING (MENYAMAKAN DURASI)
def normalize_sequence(sequence, max_frames):
    """
    Syarat mutlak model LSTM: Semua input ukurannya harus seragam.
    Fungsi ini menyamaratakan semua durasi gerakan kata menjadi 50 frame.
    """
    if len(sequence) > max_frames:
        # Jika gerakan terlalu lama/panjang, potong ujungnya sehingga jadi max_frames yaitu 50 (Trim)
        return sequence[:max_frames]
    elif len(sequence) < max_frames:
        # Jika gerakan terlalu cepat, isi kekosongan sisa frame dengan angka 0. Contoh jika gerakan terlalu cepat(misal 30 frame) maka 20 frame, maka program akan mengisi nilai 0 (Padding)
        pad = np.zeros((max_frames - len(sequence), sequence.shape[1]))
        return np.vstack([sequence, pad])
    return sequence


# 3. PROSES LOAD DATASET (.npy)
def load_dataset(base_path):
    X, y, skipped = [], [], 0  # X: list data gerakan, y: list label kata
    classes = sorted(os.listdir(base_path))  # Nama folder = nama kelas kata
    print(f"\n[INFO] Loading dataset — {len(classes)} kelas: {classes}")

    # Iterasi pertama: Membaca setiap nama folder (contoh: folder 'terimakasih', 'halo')
    for label in tqdm(classes, desc="Loading"):
        folder = os.path.join(base_path, label)

        # Keamanan: Pastikan yang sedang dibaca benar-benar sebuah folder, bukan file nyasar
        if not os.path.isdir(folder):
            continue

        # Iterasi kedua: Masuk ke dalam folder kelas, baca file di dalamnya satu per satu
        for fname in os.listdir(folder):

            # Filter ekstensi: Hanya proses file berekstensi .npy (Matriks NumPy)
            # Jika ada file foto (.jpg) atau teks (.txt), abaikan dan lanjut ke file berikutnya
            if not fname.endswith('.npy'):
                continue

            fpath = os.path.join(folder, fname)
            try:
                data = np.load(fpath, allow_pickle=True)  # Buka file koordinat

                # Validasi Data: Pastikan format datanya 2 Dimensi dan memiliki 225 kolom fitur
                # Jika datanya cacat/korup, hitung sebagai 'skipped' dan lewati
                if data.ndim != 2 or data.shape[1] != NUM_FEATURES:
                    skipped += 1
                    continue

                data = normalize_sequence(data, MAX_FRAMES)  # Samakan jadi 50 frame
                X.append(data)  # Simpan matriks angka ke dalam variabel X
                y.append(label)  # Simpan nama foldernya ke dalam variabel y sebagai jawaban/label
            except:
                skipped += 1  # Tangkap error jika file .npy gagal dibuka agar program tidak crash

    print(f"[INFO] Total: {len(X)} sampel berhasil, {skipped} dilewati")
    return np.array(X, dtype=np.float32), np.array(y)


# 4. PEMBAGIAN DATA (TRAIN & VAL)
X, y = load_dataset(DATASET_PATH)               # Load semua data gerakan (X) dan nama kata (y)
print(f"\n[INFO] Shape X: {X.shape}")           # Format akhir: (Jumlah Video, 50 frame, 225 fitur)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VAL_SPLIT,                        # Pisahkan 20% data untuk ujian, 80% untuk AI belajar
    random_state=42,                            # Kunci acakan agar hasil pembagian selalu sama tiap di-run
    stratify=y                                  # Memastikan pembagian jumlah sampel tiap kata merata/adil
)
print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)}")

# 5. ENCODE LABEL (UBAH TEKS KE ANGKA)
le = LabelEncoder()                             # Panggil fungsi penerjemah
le.fit(y)                                       # Komputer mendata & mengurutkan semua kata (misal: halo=0, maaf=1)

y_train_enc = le.transform(y_train)             # Terjemahkan kunci jawaban data belajar jadi angka
y_val_enc   = le.transform(y_val)               # Terjemahkan kunci jawaban data ujian jadi angka

NUM_CLASSES = len(le.classes_)                  # Hitung total kosa kata (Penting untuk penentu output LSTM nanti)
print(f"[INFO] Kelas: {list(le.classes_)}")

# Khusus Deep Learning (Categorical Crossentropy), label harus diubah ke format One-Hot
# Contoh: Kelas 1 dari 3 kelas berubah dari angka 1 menjadi array [0, 1, 0]
y_train_cat = tf.keras.utils.to_categorical(y_train_enc, NUM_CLASSES)
y_val_cat = tf.keras.utils.to_categorical(y_val_enc, NUM_CLASSES)

# 6. MEMBANGUN ARSITEKTUR MODEL LSTM
print("\n[INFO] Membangun model LSTM di CPU...")

# Kenapa pakai CPU? Karena pada laptop Windows biasa, DirectML sering crash
# jika disuruh memproses arsitektur recurrent layer (CudnnRNN) bawaan LSTM.
with tf.device('/CPU:0'):
    model = Sequential([
        # Layer 1: Minta AI MENGABAIKAN frame buatan (angka 0) dari hasil Padding tadi
        Masking(mask_value=0.0, input_shape=(MAX_FRAMES, NUM_FEATURES)),

        # Layer 2: Otak utama (LSTM). Mengekstrak memori pergerakan dari detik ke detik.
        # "implementation=1" memaksa LSTM memakai perhitungan standar (ramah CPU).
        LSTM(128, return_sequences=True, implementation=1),# LSTM 1: Ekstrak memori pola gerak. (implementation=1 agar ramah CPU)
        Dropout(0.3),  # Matikan 30% saraf acak untuk mencegah model sekadar menghafal (Overfitting)

        # Layer 3: LSTM kedua untuk mempertajam bacaan pola gerak
        LSTM(64, return_sequences=False, implementation=1),
        # LSTM 2: Pertajam hasil pola. (return_sequences=False karena ini LSTM terakhir)
        Dropout(0.3),  # Cegah overfitting

        # Layer 4: Saraf penghubung biasa (Fully Connected)
        Dense(64, activation='relu'),  # Jaringan saraf biasa (Fully Connected) untuk memproses kesimpulan LSTM
        Dropout(0.3),

        # Layer 5: Output/Hasil akhir. Pakai Softmax untuk memunculkan probabilitas/persentase tebakan.
        Dense(NUM_CLASSES, activation='softmax') # Output: Jumlah saraf sesuai jumlah kata. Softmax menghitung peluang/persentase tebakan.
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Algoritma pencari titik akurasi terbaik
        loss='categorical_crossentropy',  # Fungsi hitung error
        metrics=['accuracy']
    )

model.summary()

# 7. PENGAWAS TRAINING (CALLBACKS)
callbacks = [
    # 1. Simpan file model HANYA saat ada pemecahan rekor akurasi (save_best_only)
    ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy', save_best_only=True, verbose=1),
    # 2. Jika dalam 10 putaran akurasi tidak naik sama sekali, hentikan training otomatis!
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    # 3. Jika model mulai buntu belajarnya, turunkan kecepatan belajarnya sebesar 50%
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

# 8. PROSES BELAJAR (TRAINING)
print("\n[INFO] Training LSTM di CPU...")
with tf.device('/CPU:0'):
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

# 9. EVALUASI HASIL UJIAN
with tf.device('/CPU:0'):
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
print(f"\n[INFO] Akurasi Validasi: {val_acc * 100:.2f}%")

# 10. SIMPAN LABEL
with open(LABEL_SAVE, 'wb') as f:
    pickle.dump(le, f)  # Simpan daftar urutan kata (agar nanti bisa diterjemahkan ulang di webcam)

print(f"\n[DONE] Model disimpan ke {MODEL_SAVE}")
print(f"[DONE] Label disimpan ke {LABEL_SAVE}")
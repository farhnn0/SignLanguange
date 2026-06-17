import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

# ─────────────────────────────────────────
# 1. KONFIGURASI
# ─────────────────────────────────────────
DATASET_PATH  = r"bisindo_kata/npy"
MODEL_SAVE    = "kata_model.h5"
LABEL_SAVE    = "kata_labels.pkl"
MAX_FRAMES    = 50
NUM_FEATURES  = 225
VAL_SPLIT     = 0.2
EPOCHS        = 100
BATCH_SIZE    = 16
AUG_FACTOR    = 5    # Setiap sample diaugmentasi jadi 5x lipat

# ─────────────────────────────────────────
# 2. FUNGSI PADDING / TRIMMING
# ─────────────────────────────────────────
def normalize_sequence(sequence, max_frames):
    # Jika sequence lebih panjang dari max_frames, potong hingga max_frames
    if len(sequence) > max_frames:
        return sequence[:max_frames]
    # Jika sequence lebih pendek, tambahkan padding baris nol di akhir
    elif len(sequence) < max_frames:
        pad = np.zeros((max_frames - len(sequence), sequence.shape[1]))  # Buat matrix nol berukuran (selisih, jumlah fitur)
        return np.vstack([sequence, pad])  # Gabung sequence asli + padding di bawahnya
    # Jika panjangnya pas, langsung return tanpa perubahan
    return sequence

# ─────────────────────────────────────────
# 3. FUNGSI AUGMENTASI DATA
# ─────────────────────────────────────────
def augment_sequence(sequence):
    """
    Buat variasi dari 1 sequence untuk memperbanyak data training.
    Menghasilkan AUG_FACTOR variasi baru per sample.
    """
    augmented = []  # List kosong untuk simpan hasil augmentasi

    for _ in range(AUG_FACTOR):  # Loop AUG_FACTOR kali (misal: 5x)
        aug = sequence.copy()  # Copy sequence asli (jangan modifikasi original)

        # Augmentasi 1: Tambah noise Gaussian kecil ke koordinat
        noise = np.random.normal(0, 0.005, aug.shape)  # Buat derau acak dengan std dev 0.005
        aug = aug + noise  # Tambah derau ke semua koordinat (simulasi kamera goyang)

        # Augmentasi 2: Time warping — percepat atau perlambat sedikit
        warp_factor = np.random.uniform(0.85, 1.15)  # Random multiplier 0.85x - 1.15x
        original_len = len(aug)  # Simpan jumlah frame asli (misal: 50)
        new_len = int(original_len * warp_factor)  # Hitung frame baru (misal: 45 atau 55)
        if new_len > 1:  # Pastikan minimal ada 1 frame
            indices = np.linspace(0, original_len - 1, new_len)  # Buat index baru yg renggang/rapat
            aug = np.array([aug[int(i)] for i in indices])  # Interpolasi: ambil frame di index baru

        # Augmentasi 3: Scale koordinat sedikit (simulasi jarak tangan ke kamera)
        scale = np.random.uniform(0.92, 1.08)  # Random scale 0.92x - 1.08x
        aug = aug * scale  # Kalikan semua koordinat (tangan lebih besar/kecil)

        # Normalize kembali ke MAX_FRAMES
        aug = normalize_sequence(aug, MAX_FRAMES)  # Pastikan ukuran tetap (padding/trim jika perlu)
        augmented.append(aug)  # Simpan 1 variasi ke list

    return augmented  # Return list berisi AUG_FACTOR variasi

# ─────────────────────────────────────────
# 4. LOAD DATASET
# ─────────────────────────────────────────
def load_dataset(base_path):
    X, y, skipped = [], [], 0
    classes = sorted(os.listdir(base_path))  # Ambil nama semua folder
    print(f"\n[INFO] Loading dataset — {len(classes)} kelas: {classes}")

    for label in tqdm(classes, desc="Loading"): # Loop setiap folder (TQDM = progress bar)
        folder = os.path.join(base_path, label)
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.endswith('.npy'):  # Filter file .npy saja
                continue

            try:
                data = np.load(os.path.join(folder, fname), allow_pickle=True)

                if data.ndim != 2 or data.shape[1] != NUM_FEATURES:  # Validasi format
                    skipped += 1
                    continue

                data = normalize_sequence(data, MAX_FRAMES)  # Samakan durasi jadi 50 frame
                X.append(data)
                y.append(label)
            except:
                skipped += 1

    print(f"[INFO] Total: {len(X)} sampel berhasil, {skipped} dilewati")
    return np.array(X, dtype=np.float32), np.array(y)

# ─────────────────────────────────────────
# 5. LOAD, AUGMENTASI & SPLIT DATA
# ─────────────────────────────────────────
X, y = load_dataset(DATASET_PATH)
print(f"\n[INFO] Shape sebelum augmentasi: {X.shape}")

# Split dulu sebelum augmentasi — agar data val tidak ikut diaugmentasi
X_train_raw, X_val, y_train_raw, y_val = train_test_split(
    X, y, test_size=VAL_SPLIT, random_state=42, stratify=y
)

# Augmentasi hanya pada data training
print(f"\n[INFO] Augmentasi data training ({AUG_FACTOR}x lipat)...")
X_train_aug = list(X_train_raw)
y_train_aug = list(y_train_raw)

for i, seq in enumerate(tqdm(X_train_raw, desc="Augmenting")):
    augmented = augment_sequence(seq)
    X_train_aug.extend(augmented)
    y_train_aug.extend([y_train_raw[i]] * AUG_FACTOR)

X_train = np.array(X_train_aug, dtype=np.float32)
y_train  = np.array(y_train_aug)

print(f"[INFO] Train setelah augmentasi: {len(X_train)} sampel")
print(f"[INFO] Val (tanpa augmentasi):   {len(X_val)} sampel")

# ─────────────────────────────────────────
# 6. ENCODE LABEL
# ─────────────────────────────────────────
le = LabelEncoder()
le.fit(y)
y_train_enc = le.transform(y_train)
y_val_enc   = le.transform(y_val)

NUM_CLASSES = len(le.classes_)
print(f"\n[INFO] Kelas: {list(le.classes_)}")
print(f"[INFO] Jumlah kelas: {NUM_CLASSES}")

y_train_cat = tf.keras.utils.to_categorical(y_train_enc, NUM_CLASSES)
y_val_cat   = tf.keras.utils.to_categorical(y_val_enc, NUM_CLASSES)

# ─────────────────────────────────────────
# 7. BUILD MODEL GRU
# ─────────────────────────────────────────
print("\n[INFO] Membangun model GRU di CPU...")

with tf.device('/CPU:0'):
    model = Sequential([
        # Masking: abaikan frame padding (nilai 0)
        Masking(mask_value=0.0, input_shape=(MAX_FRAMES, NUM_FEATURES)),

        # GRU layer 1
        GRU(128, return_sequences=True, implementation=1),
        BatchNormalization(),
        Dropout(0.3),

        # GRU layer 2
        GRU(64, return_sequences=False, implementation=1),
        BatchNormalization(),
        Dropout(0.3),

        # Fully connected
        Dense(64, activation='relu'),
        Dropout(0.3),

        # Output layer
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()

# ─────────────────────────────────────────
# 8. CALLBACKS
# ─────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)
]

# ─────────────────────────────────────────
# 9. TRAINING
# ─────────────────────────────────────────
print("\n[INFO] Training GRU di CPU...")
with tf.device('/CPU:0'):
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True
    )

# ─────────────────────────────────────────
# 10. EVALUASI
# ─────────────────────────────────────────
with tf.device('/CPU:0'):
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
print(f"\n[INFO] Akurasi Validasi: {val_acc * 100:.2f}%")

# ─────────────────────────────────────────
# 11. SIMPAN MODEL & LABEL
# ─────────────────────────────────────────
with open(LABEL_SAVE, 'wb') as f:
    pickle.dump(le, f)

print(f"\n[DONE] Model disimpan ke {MODEL_SAVE}")
print(f"[DONE] Label disimpan ke {LABEL_SAVE}")
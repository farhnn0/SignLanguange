import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

DATASET_PATH = r"bisindo-kata-baru"  # ganti sesuai lokasi dataset kamu
OUTPUT_PATH = r"processed_bisindo"

SEQUENCE_LENGTH = 50
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

USE_LABEL_FROM = "folder"
# "folder"  = label dari nama folder, contoh dataset/Kata Kerja/Berdiri/Berdiri-a-1.mp4 -> Berdiri
# "filename" = label dari nama file sebelum tanda "-", contoh Berdiri-a-1.mp4 -> Berdiri

MODEL_PATH = "bisindo_holistic_gru.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"
CLASS_NAMES_PATH = "class_names.json"


# =========================
# 2. SETUP MEDIAPIPE
# =========================

mp_holistic = mp.solutions.holistic


def extract_landmarks(results):
    """
    Ambil landmark dari pose, tangan kiri, dan tangan kanan.
    Jika tidak terdeteksi, isi dengan 0 agar ukuran fitur tetap sama.
    """

    # Pose: 33 landmark x 4 fitur = 132
    if results.pose_landmarks:
        pose = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # Left hand: 21 landmark x 3 fitur = 63
    if results.left_hand_landmarks:
        left_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.left_hand_landmarks.landmark
        ]).flatten()
    else:
        left_hand = np.zeros(21 * 3)

    # Right hand: 21 landmark x 3 fitur = 63
    if results.right_hand_landmarks:
        right_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.right_hand_landmarks.landmark
        ]).flatten()
    else:
        right_hand = np.zeros(21 * 3)

    return np.concatenate([pose, left_hand, right_hand])


def normalize_sequence(sequence, target_length=50):
    """
    Membuat jumlah frame selalu sama.
    Kalau video terlalu panjang, ambil frame merata.
    Kalau video terlalu pendek, padding dengan frame terakhir.
    """

    sequence = np.array(sequence)

    if len(sequence) == 0:
        return None

    if len(sequence) == target_length:
        return sequence

    if len(sequence) > target_length:
        indices = np.linspace(0, len(sequence) - 1, target_length).astype(int)
        return sequence[indices]

    # Jika kurang dari target_length
    pad_count = target_length - len(sequence)
    last_frame = sequence[-1]
    padding = np.repeat(last_frame[np.newaxis, :], pad_count, axis=0)

    return np.concatenate([sequence, padding], axis=0)


def get_label(video_path):
    if USE_LABEL_FROM == "folder":
        return os.path.basename(os.path.dirname(video_path))

    filename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(filename)[0]

    # Contoh: Berdiri-a-1.mp4 -> Berdiri
    return name_without_ext.split("-")[0]


def collect_video_paths(dataset_path):
    video_extensions = (".mp4", ".mov", ".avi", ".mkv")

    # Hanya folder kategori utama yang mau dipakai untuk training
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

        for root, dirs, files in os.walk(category_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_paths.append(os.path.join(root, file))

    return video_paths


# =========================
# 3. EXTRACT VIDEO KE LANDMARK
# =========================

def process_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    if not cap.isOpened():
        print(f"Gagal membuka video: {video_path}")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR ke RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)
        landmarks = extract_landmarks(results)

        sequence.append(landmarks)

    cap.release()

    sequence = normalize_sequence(sequence, SEQUENCE_LENGTH)

    return sequence


def extract_dataset():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    X_path = os.path.join(OUTPUT_PATH, "X.npy")
    y_path = os.path.join(OUTPUT_PATH, "y.npy")

    if os.path.exists(X_path) and os.path.exists(y_path):
        print("File X.npy dan y.npy sudah ada. Lewati proses ekstraksi.")
        X = np.load(X_path)
        y = np.load(y_path, allow_pickle=True)
        return X, y

    video_paths = collect_video_paths(DATASET_PATH)

    print(f"Total video ditemukan: {len(video_paths)}")

    X = []
    y = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    ) as holistic:

        for video_path in tqdm(video_paths, desc="Extracting landmarks"):
            label = get_label(video_path)
            sequence = process_video(video_path, holistic)

            if sequence is None:
                continue

            X.append(sequence)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Jumlah label:", len(set(y)))
    print("Label:", sorted(set(y)))

    np.save(X_path, X)
    np.save(y_path, y)

    return X, y


# =========================
# 4. TRAIN MODEL GRU
# =========================

def build_model(sequence_length, num_features, num_classes):
    model = Sequential([
        Input(shape=(sequence_length, num_features)),

        GRU(
            128,
            return_sequences=True,
            reset_after=False,
            recurrent_dropout=0.1
        ),
        BatchNormalization(),
        Dropout(0.3),

        GRU(
            64,
            reset_after=False,
            recurrent_dropout=0.1
        ),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train():
    X, y = extract_dataset()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_names = list(label_encoder.classes_)

    print("Class names:", class_names)

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    num_samples = X.shape[0]
    sequence_length = X.shape[1]
    num_features = X.shape[2]
    num_classes = len(class_names)

    print("Jumlah sample:", num_samples)
    print("Sequence length:", sequence_length)
    print("Jumlah fitur:", num_features)
    print("Jumlah kelas:", num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    model = build_model(sequence_length, num_features, num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )

    print("Evaluasi model:")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(
        y_test,
        y_pred_classes,
        target_names=class_names
    ))

    model.save(MODEL_PATH)
    print(f"Model disimpan ke: {MODEL_PATH}")
    print(f"Label encoder disimpan ke: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    train()
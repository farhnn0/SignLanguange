from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Inisialisasi Web Server & CORS Configuration

# Inisialisasi backend server menggunakan FastAPI
app = FastAPI(title="Sign Language Recognition API")

# Konfigurasi perizinan akses lintas situs (CORS) untuk Next.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Path Lokasi File Model & Konfigurasi Dimensi Fitur

# Folder utama penyimpanan file model AI
MODEL_DIR = "models"

# Path model dan label untuk pengenalan Huruf (Random Forest)
HURUF_MODEL_PATH = os.path.join(MODEL_DIR, "huruf_model.pkl")
HURUF_LABELS_PATH = os.path.join(MODEL_DIR, "huruf_labels.pkl")

# Path model dan label untuk pengenalan Angka (Random Forest)
ANGKA_MODEL_PATH = os.path.join(MODEL_DIR, "angka_model.pkl")
ANGKA_LABELS_PATH = os.path.join(MODEL_DIR, "angka_labels.pkl")

# Path model dan label untuk pengenalan Kata (GRU Deep Learning)
KATA_MODEL_PATH = os.path.join(MODEL_DIR, "bisindo_holistic_gru.h5")
KATA_LABELS_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Spesifikasi input model kata (50 frame x 258 koordinat = 12.900 fitur)
KATA_MAX_FRAMES = 50
KATA_NUM_FEATURES = 258
KATA_TOTAL_FEATURES = KATA_MAX_FRAMES * KATA_NUM_FEATURES

# 3. Variabel Global Untuk Model Di RAM

# Variabel ini menampung objek model dan label encoder setelah dimuat ke memori server
huruf_model = None
huruf_labels = None
angka_model = None
angka_labels = None
kata_model = None
kata_labels = None

# 4. Struktur Data Request (Pydantic Validator)

# Skema data request untuk memvalidasi tipe data yang dikirim frontend
class PredictRequest(BaseModel):
    mode: str              # Mode pengenalan: "huruf", "angka", atau "kata"
    features: List[float]  # List array berisi angka koordinat landmark

# 5. Fungsi Utility (Loader & Decoder)

# Fungsi untuk memuat model biner menggunakan joblib (optimal untuk model Sklearn)
def load_pickle_joblib(path: str):
    if not os.path.exists(path):
        print(f"File tidak ditemukan: {path}")
        return None

    data = joblib.load(path)
    print(f"Loaded: {path} | type: {type(data)}")
    return data

# Fungsi untuk memuat encoder menggunakan pickle standar
def load_pickle_normal(path: str):
    if not os.path.exists(path):
        print(f"File tidak ditemukan: {path}")
        return None

    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded: {path} | type: {type(data)}")
    return data

# Fungsi untuk melihat preview daftar kelas/label yang terdaftar dalam model
def get_label_preview(labels):
    try:
        # Jika menggunakan scikit-learn LabelEncoder
        if hasattr(labels, "classes_"):
            return labels.classes_.tolist()

        # Jika berwujud dictionary mapping
        if isinstance(labels, dict):
            return labels

        # Jika berupa list array mentah
        if isinstance(labels, (list, tuple, np.ndarray)):
            return list(labels)

        return str(labels)
    except Exception as e:
        return f"Cannot preview labels: {e}"

# Fungsi penerjemah index prediksi angka integer kembali ke teks aslinya
def decode_prediction(raw_prediction, labels):
    if isinstance(raw_prediction, np.generic):
        raw_prediction = raw_prediction.item()

    if labels is None:
        return str(raw_prediction)

    # Terjemahkan jika menggunakan LabelEncoder scikit-learn
    if hasattr(labels, "inverse_transform"):
        try:
            decoded = labels.inverse_transform([raw_prediction])[0]
            return str(decoded)
        except Exception:
            pass

        try:
            decoded = labels.inverse_transform([int(raw_prediction)])[0]
            return str(decoded)
        except Exception:
            pass

    # Terjemahkan jika label berupa list biasa
    if isinstance(labels, (list, tuple, np.ndarray)):
        try:
            return str(labels[int(raw_prediction)])
        except Exception:
            return str(raw_prediction)

    # Terjemahkan jika label disimpan dalam bentuk dictionary
    if isinstance(labels, dict):
        try:
            key = str(raw_prediction)
            if key in labels:
                return str(labels[key])

            if raw_prediction in labels:
                return str(labels[raw_prediction])
        except Exception:
            return str(raw_prediction)

    return str(raw_prediction)

# 6. Lifecycle Events (Load Model Saat Server Startup)

# Handler untuk memuat semua model secara otomatis saat server FastAPI dinyalakan
@app.on_event("startup")
def load_models():
    global huruf_model, huruf_labels
    global angka_model, angka_labels
    global kata_model, kata_labels

    # Memuat model Huruf (Random Forest) dan labelnya
    huruf_model = load_pickle_joblib(HURUF_MODEL_PATH)
    huruf_labels = load_pickle_joblib(HURUF_LABELS_PATH)

    # Memuat model Angka (Random Forest) dan labelnya
    angka_model = load_pickle_joblib(ANGKA_MODEL_PATH)
    angka_labels = load_pickle_joblib(ANGKA_LABELS_PATH)

    # Memuat model Kata (Keras GRU)
    if os.path.exists(KATA_MODEL_PATH):
        try:
            # compile=False mencegah kebutuhan memuat konfigurasi optimizer saat training
            kata_model = load_model(KATA_MODEL_PATH, compile=False)
            print(f"Loaded: {KATA_MODEL_PATH} | input_shape: {kata_model.input_shape}")
        except Exception as e:
            print(f"Gagal load model kata: {e}")
            kata_model = None
    else:
        print(f"File tidak ditemukan: {KATA_MODEL_PATH}")

    # Memuat label kata
    kata_labels = load_pickle_normal(KATA_LABELS_PATH)

    # Verifikasi status pemuatan seluruh model di console
    print("Huruf model loaded:", huruf_model is not None)
    print("Huruf labels loaded:", huruf_labels is not None)
    print("Angka model loaded:", angka_model is not None)
    print("Angka labels loaded:", angka_labels is not None)
    print("Kata model loaded:", kata_model is not None)
    print("Kata labels loaded:", kata_labels is not None)

    if huruf_labels is not None:
        print("Huruf labels content:", get_label_preview(huruf_labels))

    if angka_labels is not None:
        print("Angka labels content:", get_label_preview(angka_labels))

    if kata_labels is not None:
        print("Kata labels content:", get_label_preview(kata_labels))

# 7. Inference Logic (Proses Prediksi Model)

# Proses prediksi model Random Forest (Huruf & Angka)
def predict_with_model(model, labels, features):
    input_data = np.array(features, dtype=np.float32).reshape(1, -1)

    raw_prediction = model.predict(input_data)[0]
    prediction = decode_prediction(raw_prediction, labels)

    # Hitung confidence score berdasarkan probabilitas keputusan model
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(np.max(probabilities))
    else:
        confidence = 1.0

    return prediction, confidence, raw_prediction

# Proses prediksi model GRU (Kata)
def predict_kata(features):
    input_data = np.array(features, dtype=np.float32)

    # Reshape koordinat datar 1D menjadi 3D sekuensial (batch_size=1, timesteps=50, features=258)
    input_data = input_data.reshape(1, KATA_MAX_FRAMES, KATA_NUM_FEATURES)

    # Jalankan inferensi di CPU agar aman dari masalah CuDNN di Windows
    with tf.device("/CPU:0"):
        probabilities = kata_model.predict(input_data, verbose=0)[0]

    # Ambil index kelas dengan kecocokan tertinggi
    pred_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_idx])

    prediction = decode_prediction(pred_idx, kata_labels)

    return prediction, confidence, pred_idx

# 8. API Endpoints

# Endpoint root untuk mengecek apakah API berjalan aktif
@app.get("/")
def root():
    return {
        "message": "Sign Language Recognition API is running"
    }

# Endpoint health check untuk menampilkan status model & daftar kelas terdaftar
@app.get("/health")
def health():
    return {
        "status": "ok",
        "api": "connected",
        "models": {
            "huruf_model_loaded": huruf_model is not None,
            "huruf_labels_loaded": huruf_labels is not None,
            "angka_model_loaded": angka_model is not None,
            "angka_labels_loaded": angka_labels is not None,
            "kata_model_loaded": kata_model is not None,
            "kata_labels_loaded": kata_labels is not None,
        },
        "labels_preview": {
            "huruf": get_label_preview(huruf_labels),
            "angka": get_label_preview(angka_labels),
            "kata": get_label_preview(kata_labels),
        },
        "feature_rules": {
            "huruf": 126,
            "angka": 63,
            "kata": {
                "max_frames": KATA_MAX_FRAMES,
                "num_features_per_frame": KATA_NUM_FEATURES,
                "total_features": KATA_TOTAL_FEATURES,
                "format": "flat array 50 x 258 = 12900"
            }
        }
    }

# Endpoint POST utama yang dipanggil frontend untuk prediksi koordinat real-time
@app.post("/predict")
def predict(data: PredictRequest):
    mode = data.mode.lower()

    # --- Mode Pengenalan Huruf ---
    if mode == "huruf":
        expected_features = 126

        if len(data.features) != expected_features:
            return {
                "error": "Invalid feature length for huruf",
                "expected": expected_features,
                "received": len(data.features)
            }

        if huruf_model is None:
            return {
                "error": "Model huruf belum dimuat."
            }

        prediction, confidence, raw_prediction = predict_with_model(
            huruf_model,
            huruf_labels,
            data.features
        )

    # --- Mode Pengenalan Angka ---
    elif mode == "angka":
        expected_features = 63

        if len(data.features) != expected_features:
            return {
                "error": "Invalid feature length for angka",
                "expected": expected_features,
                "received": len(data.features)
            }

        if angka_model is None:
            return {
                "error": "Model angka belum dimuat."
            }

        prediction, confidence, raw_prediction = predict_with_model(
            angka_model,
            angka_labels,
            data.features
        )

    # --- Mode Pengenalan Kata ---
    elif mode == "kata":
        expected_features = KATA_TOTAL_FEATURES

        if len(data.features) != expected_features:
            return {
                "error": "Invalid feature length for kata",
                "expected": expected_features,
                "received": len(data.features),
                "detail": "Mode kata membutuhkan 50 frame x 258 fitur = 12900 angka."
            }

        if kata_model is None:
            return {
                "error": "Model kata belum dimuat."
            }

        if kata_labels is None:
            return {
                "error": "Label encoder kata belum dimuat."
            }

        prediction, confidence, raw_prediction = predict_kata(data.features)

    else:
        return {
            "error": "Mode tidak valid. Gunakan: huruf, angka, atau kata."
        }

    # Kembalikan response json berisi prediksi dan confidence score ke frontend
    return {
        "prediction": prediction,
        "raw_prediction": str(raw_prediction),
        "confidence": confidence,
        "mode": mode,
        "feature_count": len(data.features)
    }
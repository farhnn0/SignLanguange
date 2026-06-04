/**
 * Konstanta global aplikasi Sign Language Recognition.
 *
 * Nilai-nilai ini HARUS konsisten dengan backend (backend/main.py) dan
 * dengan parameter saat training model (Train/train_kata_bisindo_gru.py).
 * Jangan mengubah angka feature/frame tanpa men-train ulang model.
 */

/** Endpoint FastAPI untuk prediksi. */
export const API_URL = "http://127.0.0.1:8000/predict";

/** Panjang sequence untuk model kata (GRU). Harus = SEQUENCE_LENGTH saat training. */
export const KATA_MAX_FRAMES = 50;

/** Jumlah fitur per frame: pose(33x4=132) + tangan kiri(21x3=63) + tangan kanan(21x3=63). */
export const KATA_NUM_FEATURES = 258;

/** Total fitur flat yang dikirim ke backend untuk satu prediksi kata. */
export const KATA_TOTAL_FEATURES = KATA_MAX_FRAMES * KATA_NUM_FEATURES;

/** Jumlah prediksi terakhir yang dipakai untuk majority voting (smoothing). */
export const KATA_SMOOTHING_WINDOW = 5;

/** Confidence minimum agar prediksi kata diterima ke dalam buffer smoothing. */
export const KATA_CONFIDENCE_THRESHOLD = 0.75;

/** Jeda minimum antar panggilan API (ms) agar tidak membanjiri backend. */
export const API_THROTTLE_MS = 500;

/** Jumlah fitur untuk model huruf (2 tangan x 21 x 3). */
export const HURUF_FEATURES = 126;

/** Jumlah fitur untuk model angka (1 tangan x 21 x 3). */
export const ANGKA_FEATURES = 63;

/** URL model MediaPipe (hand & pose). Pose pakai versi "full" agar akurat. */
export const HAND_LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";

export const POSE_LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";

export const MEDIAPIPE_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

/** Daftar mode deteksi yang tersedia di UI. */
export const modes = [
  { id: "huruf", label: "Huruf", description: "A-Z" },
  { id: "angka", label: "Angka", description: "0-9" },
  { id: "kata", label: "Kata", description: "GRU" },
] as const;

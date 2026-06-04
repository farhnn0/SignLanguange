/**
 * Tipe data bersama untuk aplikasi Sign Language Recognition.
 */

/** Satu landmark tangan (21 per tangan dari MediaPipe HandLandmarker). */
export type HandLandmark = {
  x: number;
  y: number;
  z: number;
};

/** Satu landmark pose (33 dari MediaPipe PoseLandmarker). */
export type PoseLandmark = {
  x: number;
  y: number;
  z: number;
  visibility?: number;
};

/** Info kiri/kanan dari MediaPipe untuk tiap tangan yang terdeteksi. */
export type Handedness = {
  score: number;
  index: number;
  categoryName: string;
  displayName: string;
};

/** Status koneksi ke backend FastAPI. */
export type ApiStatus = "Connected" | "Disconnected" | "Error";

/** Hasil prediksi tunggal untuk smoothing kata. */
export type KataPrediction = {
  label: string;
  confidence: number;
};

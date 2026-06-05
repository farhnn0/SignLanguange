/**
 * Loader model MediaPipe (HandLandmarker & PoseLandmarker).
 *
 * Keduanya di-cache lewat ref di pemanggil agar tidak dibuat ulang.
 */

import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

import {
  HAND_LANDMARKER_MODEL,
  POSE_LANDMARKER_MODEL,
  MEDIAPIPE_WASM,
} from "./constants";

/** Buat HandLandmarker (mode VIDEO, maksimal 2 tangan).
 *
 * Threshold sengaja diturunkan agar tangan tidak gampang "hilang":
 * - minHandDetectionConfidence rendah  -> tangan lebih mudah terdeteksi awal
 * - minHandPresenceConfidence rendah   -> tangan tetap dianggap ada walau ragu
 * - minTrackingConfidence rendah       -> tracking antar-frame lebih "lengket"
 */
export async function createHandLandmarker(): Promise<HandLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);

  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_LANDMARKER_MODEL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
}

/** Buat PoseLandmarker (mode VIDEO, 1 pose). Dipakai hanya untuk mode kata.
 *
 * Threshold diturunkan dengan alasan yang sama agar pose tubuh lebih stabil.
 */
export async function createPoseLandmarker(): Promise<PoseLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);

  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: POSE_LANDMARKER_MODEL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.3,
    minPosePresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
}

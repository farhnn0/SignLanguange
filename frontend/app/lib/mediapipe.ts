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

/** Buat HandLandmarker (mode VIDEO, maksimal 2 tangan). */
export async function createHandLandmarker(): Promise<HandLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);

  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_LANDMARKER_MODEL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

/** Buat PoseLandmarker (mode VIDEO, 1 pose). Dipakai hanya untuk mode kata. */
export async function createPoseLandmarker(): Promise<PoseLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);

  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: POSE_LANDMARKER_MODEL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

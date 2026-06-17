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

// 1. INISIALISASI HAND LANDMARKER
// Inisialisasi model MediaPipe Hand Landmarker untuk deteksi tangan (maksimal 2 tangan)
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

// 2. INISIALISASI POSE LANDMARKER
// Inisialisasi model MediaPipe Pose Landmarker untuk deteksi pose/badan (maksimal 1 pose)
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

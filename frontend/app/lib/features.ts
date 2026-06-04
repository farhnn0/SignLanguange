/**
 * Ekstraksi fitur landmark menjadi array angka yang dikirim ke backend.
 *
 * PENTING: urutan & jumlah fitur HARUS identik dengan saat training
 * (Train/train_kata_bisindo_gru.py). Mengubah urutan = model salah prediksi.
 */

import type { HandLandmark, PoseLandmark, Handedness } from "./types";

/**
 * Petakan daftar tangan MediaPipe ke slot { left, right }.
 *
 * Video di-mirror secara visual (scale-x-[-1]), sehingga:
 * - "Left" dari MediaPipe  -> tangan kanan user (tampak di kiri) -> slot left
 * - "Right" dari MediaPipe -> tangan kiri user (tampak di kanan) -> slot right
 *
 * Pemetaan ini sengaja dibuat agar cocok dengan urutan
 * left_hand lalu right_hand pada data training Python.
 */
function mapHands(
  landmarksList: HandLandmark[][],
  handednessList: Handedness[][]
): { left: HandLandmark[] | null; right: HandLandmark[] | null } {
  const hands: { left: HandLandmark[] | null; right: HandLandmark[] | null } = {
    left: null,
    right: null,
  };

  handednessList.forEach((handedness, index) => {
    const label = handedness[0]?.categoryName;
    if (label === "Left") hands.left = landmarksList[index];
    if (label === "Right") hands.right = landmarksList[index];
  });

  return hands;
}

/**
 * Fitur untuk model HURUF (2 tangan x 21 x 3 = 126).
 *
 * Catatan: untuk huruf memakai pemetaan lama (Left->right, Right->left)
 * mengikuti kode awal agar konsisten dengan model huruf yang sudah ada.
 */
export function extractHurufFeatures(
  landmarksList: HandLandmark[][],
  handednessList: Handedness[][]
): number[] {
  const features: number[] = [];

  const hands: { right: HandLandmark[] | null; left: HandLandmark[] | null } = {
    right: null,
    left: null,
  };

  handednessList.forEach((handedness, index) => {
    const label = handedness[0]?.categoryName;
    if (label === "Left") hands.right = landmarksList[index];
    if (label === "Right") hands.left = landmarksList[index];
  });

  const firstHand = hands.right ?? hands.left;
  if (firstHand) {
    firstHand.forEach((lm) => features.push(lm.x, lm.y, lm.z));
  } else {
    features.push(...Array(63).fill(0));
  }

  const secondHand = hands.right && hands.left ? hands.left : null;
  if (secondHand) {
    secondHand.forEach((lm) => features.push(lm.x, lm.y, lm.z));
  } else {
    features.push(...Array(63).fill(0));
  }

  return features;
}

/**
 * Fitur untuk model KATA (holistic): pose(132) + left hand(63) + right hand(63) = 258.
 * Urutan persis sama dengan extract_landmarks() di training Python.
 */
export function extractKataFeatures(
  poseLandmarks: PoseLandmark[] | undefined,
  landmarksList: HandLandmark[][],
  handednessList: Handedness[][]
): number[] {
  const features: number[] = [];

  // Pose: 33 x 4 = 132
  if (poseLandmarks && poseLandmarks.length > 0) {
    for (const lm of poseLandmarks) {
      features.push(lm.x, lm.y, lm.z, lm.visibility ?? 0);
    }
  } else {
    features.push(...Array(33 * 4).fill(0));
  }

  const hands = mapHands(landmarksList, handednessList);

  // Left hand: 21 x 3 = 63
  if (hands.left) {
    for (const lm of hands.left) features.push(lm.x, lm.y, lm.z);
  } else {
    features.push(...Array(21 * 3).fill(0));
  }

  // Right hand: 21 x 3 = 63
  if (hands.right) {
    for (const lm of hands.right) features.push(lm.x, lm.y, lm.z);
  } else {
    features.push(...Array(21 * 3).fill(0));
  }

  return features;
}

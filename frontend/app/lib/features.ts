import type { HandLandmark, PoseLandmark, Handedness } from "./types";

// 1. PEMETAAN TANGAN MEDIAPIPE
// Petakan tangan MediaPipe ke slot left/right agar cocok dengan urutan training Python
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

// 2. EKSTRAKSI FITUR HURUF
// Ekstrak 126 fitur koordinat (2 tangan x 21 landmark x 3 dimensi) untuk model huruf
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

// 3. EKSTRAKSI FITUR KATA
// Ekstrak 258 fitur (pose: 132 + left hand: 63 + right hand: 63) untuk model kata
export function extractKataFeatures(
  poseLandmarks: PoseLandmark[] | undefined,
  landmarksList: HandLandmark[][],
  handednessList: Handedness[][]
): number[] {
  const features: number[] = [];

  if (poseLandmarks && poseLandmarks.length > 0) {
    for (const lm of poseLandmarks) {
      features.push(lm.x, lm.y, lm.z, lm.visibility ?? 0);
    }
  } else {
    features.push(...Array(33 * 4).fill(0));
  }

  const hands = mapHands(landmarksList, handednessList);

  if (hands.left) {
    for (const lm of hands.left) features.push(lm.x, lm.y, lm.z);
  } else {
    features.push(...Array(21 * 3).fill(0));
  }

  if (hands.right) {
    for (const lm of hands.right) features.push(lm.x, lm.y, lm.z);
  } else {
    features.push(...Array(21 * 3).fill(0));
  }

  return features;
}

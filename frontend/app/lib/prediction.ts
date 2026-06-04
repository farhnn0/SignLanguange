/**
 * Smoothing prediksi kata dengan majority voting.
 *
 * Tujuannya agar label hasil tidak "kedap-kedip" ganti-ganti tiap frame.
 * Setara dengan get_stable_prediction() di predict/Predict_mix_new.py.
 */

import type { KataPrediction } from "./types";

/**
 * Ambil label yang paling sering muncul dari riwayat prediksi,
 * lalu rata-ratakan confidence untuk label tersebut.
 */
export function getStableKataPrediction(
  history: KataPrediction[]
): { label: string; confidence: number } {
  if (history.length === 0) return { label: "-", confidence: 0 };

  const counts: Record<string, number> = {};
  for (const item of history) {
    counts[item.label] = (counts[item.label] ?? 0) + 1;
  }

  let bestLabel = history[0].label;
  let bestCount = 0;
  for (const [label, count] of Object.entries(counts)) {
    if (count > bestCount) {
      bestCount = count;
      bestLabel = label;
    }
  }

  const matching = history.filter((h) => h.label === bestLabel);
  const avgConfidence =
    matching.reduce((sum, h) => sum + h.confidence, 0) / matching.length;

  return { label: bestLabel, confidence: avgConfidence };
}

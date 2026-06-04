/**
 * Bagian bawah: ringkasan model + preview beberapa nilai fitur terakhir.
 */

import { Brain } from "lucide-react";
import { Card } from "./ui";
import {
  KATA_TOTAL_FEATURES,
  HURUF_FEATURES,
  ANGKA_FEATURES,
} from "../lib/constants";

type ModelInfoProps = {
  activeMode: string;
  featureCount: number;
  latestFeatures: number[];
};

export function ModelInfo({
  activeMode,
  featureCount,
  latestFeatures,
}: ModelInfoProps) {
  const featureTotal =
    activeMode === "kata"
      ? KATA_TOTAL_FEATURES
      : activeMode === "angka"
      ? ANGKA_FEATURES
      : HURUF_FEATURES;

  return (
    <section className="mx-auto max-w-7xl px-5 pb-8 lg:px-8">
      <Card>
        <div className="mb-4 flex items-center gap-2">
          <Brain size={18} />
          <h2 className="text-lg font-semibold tracking-tight">Model Info</h2>
        </div>

        <div className="grid gap-4 md:grid-cols-5">
          <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
            <p className="text-sm text-neutral-500">Static Model</p>
            <p className="mt-1 font-semibold">Huruf & Angka</p>
          </div>

          <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
            <p className="text-sm text-neutral-500">Word Model</p>
            <p className="mt-1 font-semibold">Holistic + GRU</p>
          </div>

          <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
            <p className="text-sm text-neutral-500">API Endpoint</p>
            <p className="mt-1 font-semibold">/predict</p>
          </div>

          <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
            <p className="text-sm text-neutral-500">Input Format</p>
            <p className="mt-1 font-semibold">
              {activeMode === "kata" ? "50 x 258 Features" : "126 / 63 Features"}
            </p>
          </div>

          <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
            <p className="text-sm text-neutral-500">Current Features</p>
            <p className="mt-1 font-semibold">
              {featureCount}/{featureTotal}
            </p>
          </div>
        </div>

        <div className="mt-4 rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
          <p className="text-sm text-neutral-500">Feature Preview</p>
          <p className="mt-2 break-all font-mono text-xs text-neutral-600">
            {latestFeatures.length > 0
              ? latestFeatures
                  .slice(0, 12)
                  .map((value) => value.toFixed(4))
                  .join(", ") + " ..."
              : "No features yet"}
          </p>
        </div>
      </Card>
    </section>
  );
}

/**
 * Sidebar kanan: pemilih mode, hasil prediksi, confidence bar,
 * status sistem, dan panduan penggunaan.
 */

import { AlertCircle, Info } from "lucide-react";
import { Card, Badge, ConfidenceBar } from "./ui";
import { cn } from "../lib/utils";
import {
  modes,
  KATA_TOTAL_FEATURES,
  HURUF_FEATURES,
  ANGKA_FEATURES,
} from "../lib/constants";
import type { ApiStatus } from "../lib/types";

type SidebarProps = {
  activeMode: string;
  setActiveMode: (mode: string) => void;
  prediction: { value: string; label: string; confidence: number };
  apiStatus: ApiStatus;
  modelLoading: boolean;
  handDetected: boolean;
  handCount: number;
  landmarkCount: number;
  featureCount: number;
};

export function Sidebar({
  activeMode,
  setActiveMode,
  prediction,
  apiStatus,
  modelLoading,
  handDetected,
  handCount,
  landmarkCount,
  featureCount,
}: SidebarProps) {
  const featureTotal =
    activeMode === "kata"
      ? KATA_TOTAL_FEATURES
      : activeMode === "angka"
      ? ANGKA_FEATURES
      : HURUF_FEATURES;

  return (
    <aside className="space-y-6">
      {/* 1. Panel Pemilihan Mode Deteksi */}
      <Card>
        <div className="mb-4">
          <h2 className="text-lg font-semibold tracking-tight">
            Detection Mode
          </h2>
          <p className="mt-1 text-sm text-neutral-500">
            Pilih jenis gesture yang ingin dikenali.
          </p>
        </div>

        <div className="grid grid-cols-3 gap-2 rounded-2xl bg-neutral-100 p-1">
          {modes.map((mode) => {
            const active = activeMode === mode.id;
            return (
              <button
                key={mode.id}
                onClick={() => setActiveMode(mode.id)}
                className={cn(
                  "rounded-xl px-3 py-3 text-center text-sm font-medium transition",
                  active
                    ? "bg-neutral-950 text-white shadow-sm"
                    : "text-neutral-700 hover:bg-white"
                )}
              >
                <span className="block">{mode.label}</span>
                <span
                  className={cn(
                    "mt-0.5 block text-[11px]",
                    active ? "text-white/60" : "text-neutral-500"
                  )}
                >
                  {mode.description}
                </span>
              </button>
            );
          })}
        </div>
      </Card>

      {/* 2. Panel Hasil Prediksi Model */}
      <Card>
        <div className="mb-5 flex items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">
              Prediction Result
            </h2>
            <p className="mt-1 text-sm text-neutral-500">
              Hasil prediksi gesture terbaru.
            </p>
          </div>

          {activeMode === "kata" && <Badge tone="warning">GRU</Badge>}
        </div>

        <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-6 text-center">
          <p className="text-6xl font-bold tracking-tight text-neutral-950">
            {prediction.value}
          </p>
          <p className="mt-3 text-sm font-medium text-neutral-500">
            {prediction.label}
          </p>
        </div>
      </Card>

      {/* 3. Panel Indikator Keakuratan (Confidence) */}
      <Card>
        <ConfidenceBar value={prediction.confidence} />
      </Card>

      {/* 4. Panel Info Status Sistem Terperinci */}
      <Card>
        <div className="mb-4 flex items-center gap-2">
          <Info size={18} />
          <h2 className="text-lg font-semibold tracking-tight">
            System Status
          </h2>
        </div>

        <div className="space-y-3 text-sm">
          <div className="flex items-center justify-between gap-4">
            <span className="text-neutral-500">API Status</span>
            <Badge
              tone={
                apiStatus === "Connected"
                  ? "success"
                  : apiStatus === "Error"
                  ? "danger"
                  : "neutral"
              }
            >
              {apiStatus}
            </Badge>
          </div>

          <div className="flex items-center justify-between gap-4">
            <span className="text-neutral-500">Model Status</span>
            <Badge tone={modelLoading ? "warning" : "success"}>
              {modelLoading ? "Loading" : "Loaded"}
            </Badge>
          </div>

          <div className="flex items-center justify-between gap-4">
            <span className="text-neutral-500">Active Mode</span>
            <span className="font-medium capitalize">{activeMode}</span>
          </div>

          <div className="flex items-center justify-between gap-4">
            <span className="text-neutral-500">Hand Detection</span>
            <Badge tone={handDetected ? "success" : "neutral"}>
              {handDetected ? "Detected" : "Not Detected"}
            </Badge>
          </div>

          <div className="flex items-center justify-between gap-4">
            <span className="text-neutral-500">Detected Hands</span>
            <span className="font-medium">{handCount}/2 hands</span>
          </div>

          <div className="flex items-center justify-between gap-4">
            <span className="text-neutral-500">Input Landmark</span>
            <span className="font-medium">{landmarkCount} points</span>
          </div>

          <div className="flex items-center justify-between gap-4">
            <span className="text-neutral-500">Feature Count</span>
            <span className="font-medium">
              {featureCount}/{featureTotal}
            </span>
          </div>
        </div>
      </Card>

      {/* 5. Panel Panduan Cara Penggunaan */}
      <Card>
        <div className="mb-4 flex items-center gap-2">
          <AlertCircle size={18} />
          <h2 className="text-lg font-semibold tracking-tight">How to Use</h2>
        </div>

        <ol className="space-y-2 text-sm leading-6 text-neutral-600">
          <li>1. Click Start Camera.</li>
          <li>2. Select Huruf, Angka, or Kata mode.</li>
          <li>3. Place your hands clearly inside the camera frame.</li>
          <li>
            4. For Kata mode, perform the movement until the 50-frame buffer is
            full.
          </li>
          <li>5. The prediction result will appear automatically.</li>
        </ol>
      </Card>
    </aside>
  );
}

/**
 * Header aplikasi: judul + badge status (API, model, buffer kata).
 */

import { Hand } from "lucide-react";
import { Badge } from "./ui";
import { KATA_TOTAL_FEATURES } from "../lib/constants";
import type { ApiStatus } from "../lib/types";

type AppHeaderProps = {
  apiStatus: ApiStatus;
  modelLoading: boolean;
  activeMode: string;
  featureCount: number;
};

export function AppHeader({
  apiStatus,
  modelLoading,
  activeMode,
  featureCount,
}: AppHeaderProps) {
  return (
    <header className="border-b border-neutral-200 bg-white">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
        
        {/* Judul & Deskripsi Aplikasi */}
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-neutral-200 bg-neutral-950 text-white">
            <Hand size={22} />
          </div>

          <div>
            <h1 className="text-2xl font-bold tracking-tight lg:text-3xl">
              Sign Language Recognition
            </h1>
            <p className="mt-1 text-sm text-neutral-500">
              Deteksi huruf, angka, dan kata bahasa isyarat melalui kamera
              secara real-time.
            </p>
          </div>
        </div>

        {/* Status Indikator Sistem (FastAPI, MediaPipe, Buffer) */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Status Koneksi Backend */}
          <Badge
            tone={
              apiStatus === "Connected"
                ? "success"
                : apiStatus === "Error"
                ? "danger"
                : "neutral"
            }
          >
            FastAPI {apiStatus}
          </Badge>

          {/* Status Load Model MediaPipe */}
          <Badge tone={modelLoading ? "warning" : "success"}>
            {modelLoading ? "Model Loading" : "Model Loaded"}
          </Badge>

          {/* Status Buffer untuk Mode Kata */}
          <Badge tone={activeMode === "kata" ? "warning" : "neutral"}>
            {activeMode === "kata"
              ? `Word Buffer ${Math.min(
                  featureCount,
                  KATA_TOTAL_FEATURES
                )}/${KATA_TOTAL_FEATURES}`
              : "Static Model Ready"}
          </Badge>
        </div>

      </div>
    </header>
  );
}

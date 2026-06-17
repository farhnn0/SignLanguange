/**
 * Panel kamera: preview video + canvas overlay landmark, indikator deteksi,
 * progress buffer kata, dan tombol kontrol (Start / Stop / Reset Kata).
 */

import { RefObject } from "react";
import { Camera, CircleStop, Play, RotateCcw } from "lucide-react";
import { Card, Badge } from "./ui";
import { cn } from "../lib/utils";
import { KATA_MAX_FRAMES, KATA_NUM_FEATURES, KATA_TOTAL_FEATURES } from "../lib/constants";

type CameraPanelProps = {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  cameraActive: boolean;
  cameraError: string;
  modelLoading: boolean;
  activeMode: string;
  handDetected: boolean;
  handCount: number;
  landmarkCount: number;
  featureCount: number;
  onStart: () => void;
  onStop: () => void;
  onResetKata: () => void;
};

export function CameraPanel({
  videoRef,
  canvasRef,
  cameraActive,
  cameraError,
  modelLoading,
  activeMode,
  handDetected,
  handCount,
  landmarkCount,
  featureCount,
  onStart,
  onStop,
  onResetKata,
}: CameraPanelProps) {
  return (
    <Card className="p-0">
      {/* Header Panel Utama */}
      <div className="flex items-center justify-between gap-4 border-b border-neutral-200 p-5">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">
            Camera Preview
          </h2>
          <p className="mt-1 text-sm text-neutral-500">
            Pastikan tangan dan tubuh bagian atas terlihat jelas di dalam frame
            kamera.
          </p>
        </div>

        <Badge tone={cameraActive ? "success" : "neutral"}>
          {cameraActive ? "Camera Active" : "Camera Off"}
        </Badge>
      </div>

      <div className="p-5">
        {/* Tampilan Pesan Error Kamera */}
        {cameraError && (
          <div className="mb-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {cameraError}
          </div>
        )}

        {/* Video & Canvas Overlay Deteksi */}
        <div className="relative aspect-video overflow-hidden rounded-2xl bg-neutral-950">
          {cameraActive ? (
            <>
              {/* Element Video Webcam */}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="h-full w-full scale-x-[-1] object-cover"
              />

              {/* Element Canvas untuk Menggambar Kerangka */}
              <canvas
                ref={canvasRef}
                className="absolute inset-0 h-full w-full scale-x-[-1]"
              />

              {/* Indikator Status Deteksi Tangan */}
              <div className="absolute left-4 top-4 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-medium text-white backdrop-blur">
                {handDetected ? "Hand detected" : "No hand detected"}
              </div>

              {/* Info Jumlah Tangan & Titik Rangka */}
              <div className="absolute right-4 top-4 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-medium text-white backdrop-blur">
                {handCount} {handCount > 1 ? "hands" : "hand"} • {landmarkCount}{" "}
                landmarks
              </div>

              {/* Progress Buffer untuk Mode Kata Dinamis */}
              {activeMode === "kata" && (
                <div className="absolute bottom-4 left-4 right-4">
                  <div className="mb-2 flex items-center justify-between text-xs font-medium text-white">
                    <span>Word buffer</span>
                    <span>
                      {Math.floor(featureCount / KATA_NUM_FEATURES)}/
                      {KATA_MAX_FRAMES} frames
                    </span>
                  </div>

                  <div className="h-2 overflow-hidden rounded-full bg-white/20">
                    <div
                      className="h-full rounded-full bg-white"
                      style={{
                        width: `${Math.min(
                          100,
                          (featureCount / KATA_TOTAL_FEATURES) * 100
                        )}%`,
                      }}
                    />
                  </div>
                </div>
              )}
            </>
          ) : (
            /* Tampilan saat Kamera Mati */
            <div className="flex h-full flex-col items-center justify-center px-6 text-center text-white">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-white/10">
                <Camera size={30} />
              </div>

              <h3 className="text-lg font-semibold">Camera is off</h3>
              <p className="mt-2 max-w-sm text-sm text-white/60">
                Click Start Camera to begin detection and show your gesture.
              </p>
            </div>
          )}
        </div>

        {/* Tombol Aksi Kontrol Kamera & Buffer */}
        <div className="mt-5 flex flex-col gap-3 sm:flex-row">
          <button
            onClick={onStart}
            disabled={cameraActive || modelLoading}
            className={cn(
              "inline-flex items-center justify-center gap-2 rounded-xl border px-4 py-2.5 text-sm font-medium transition",
              cameraActive || modelLoading
                ? "cursor-not-allowed border-neutral-200 bg-neutral-100 text-neutral-400"
                : "border-neutral-950 bg-neutral-950 text-white hover:bg-neutral-800"
            )}
          >
            <Play size={16} />
            {modelLoading ? "Loading Model..." : "Start Camera"}
          </button>

          <button
            onClick={onStop}
            disabled={!cameraActive}
            className={cn(
              "inline-flex items-center justify-center gap-2 rounded-xl border px-4 py-2.5 text-sm font-medium transition",
              !cameraActive
                ? "cursor-not-allowed border-neutral-200 bg-neutral-100 text-neutral-400"
                : "border-neutral-200 bg-white text-neutral-950 hover:bg-neutral-50"
            )}
          >
            <CircleStop size={16} />
            Stop Camera
          </button>

          <button
            onClick={onResetKata}
            disabled={!cameraActive || activeMode !== "kata"}
            className={cn(
              "inline-flex items-center justify-center gap-2 rounded-xl border px-4 py-2.5 text-sm font-medium transition",
              !cameraActive || activeMode !== "kata"
                ? "cursor-not-allowed border-neutral-200 bg-neutral-100 text-neutral-400"
                : "border-neutral-200 bg-white text-neutral-950 hover:bg-neutral-50"
            )}
          >
            <RotateCcw size={16} />
            Reset Kata
          </button>
        </div>
      </div>
    </Card>
  );
}

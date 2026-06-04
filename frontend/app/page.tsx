/**
 * Halaman utama Sign Language Recognition.
 *
 * File ini sengaja dibuat tipis: seluruh logika ada di hook
 * `useSignRecognition`, dan tampilan dipecah menjadi komponen di
 * folder components/. Lihat README.md untuk arsitektur lengkap.
 */

"use client";

import { useSignRecognition } from "./hooks/useSignRecognition";
import { AppHeader } from "./components/AppHeader";
import { CameraPanel } from "./components/CameraPanel";
import { MetricsRow } from "./components/MetricsRow";
import { Sidebar } from "./components/Sidebar";
import { ModelInfo } from "./components/ModelInfo";
import { GestureGuide } from "./components/GestureGuide";

export default function Home() {
  const {
    videoRef,
    canvasRef,
    activeMode,
    setActiveMode,
    cameraActive,
    cameraError,
    modelLoading,
    handDetected,
    handCount,
    landmarkCount,
    featureCount,
    latestFeatures,
    apiStatus,
    responseTime,
    prediction,
    startCamera,
    stopCamera,
    resetKataBuffer,
  } = useSignRecognition();

  return (
    <main className="min-h-screen bg-neutral-50 text-neutral-950">
      <AppHeader
        apiStatus={apiStatus}
        modelLoading={modelLoading}
        activeMode={activeMode}
        featureCount={featureCount}
      />

      <section className="mx-auto grid max-w-7xl gap-6 px-5 py-6 lg:grid-cols-[1.75fr_1fr] lg:px-8">
        <div className="space-y-6">
          <CameraPanel
            videoRef={videoRef}
            canvasRef={canvasRef}
            cameraActive={cameraActive}
            cameraError={cameraError}
            modelLoading={modelLoading}
            activeMode={activeMode}
            handDetected={handDetected}
            handCount={handCount}
            landmarkCount={landmarkCount}
            featureCount={featureCount}
            onStart={startCamera}
            onStop={stopCamera}
            onResetKata={resetKataBuffer}
          />

          <MetricsRow
            responseTime={responseTime}
            activeMode={activeMode}
            apiStatus={apiStatus}
          />

          <GestureGuide />
        </div>

        <Sidebar
          activeMode={activeMode}
          setActiveMode={setActiveMode}
          prediction={prediction}
          apiStatus={apiStatus}
          modelLoading={modelLoading}
          handDetected={handDetected}
          handCount={handCount}
          landmarkCount={landmarkCount}
          featureCount={featureCount}
        />
      </section>

      <ModelInfo
        activeMode={activeMode}
        featureCount={featureCount}
        latestFeatures={latestFeatures}
      />
    </main>
  );
}

/**
 * useSignRecognition
 * ==================
 * Hook utama yang membungkus seluruh state & logika pengenalan isyarat:
 * - Memuat model MediaPipe (hand + pose)
 * - Mengakses webcam
 * - Loop deteksi per-frame (requestAnimationFrame)
 * - Ekstraksi fitur sesuai mode (huruf/angka/kata)
 * - Mengirim fitur ke backend & smoothing hasil kata
 *
 * Logika di sini sengaja dipertahankan 1:1 dengan versi sebelumnya
 * agar fungsionalitas dan akurasi tidak berubah. Komponen UI cukup
 * memakai nilai & fungsi yang dikembalikan hook ini.
 */

"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import {
  HandLandmarker,
  PoseLandmarker,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

import {
  API_URL,
  API_THROTTLE_MS,
  KATA_MAX_FRAMES,
  KATA_NUM_FEATURES,
  KATA_TOTAL_FEATURES,
  KATA_SMOOTHING_WINDOW,
  KATA_CONFIDENCE_THRESHOLD,
  HURUF_FEATURES,
  ANGKA_FEATURES,
} from "../lib/constants";
import type { ApiStatus, KataPrediction } from "../lib/types";
import { createHandLandmarker, createPoseLandmarker } from "../lib/mediapipe";
import { extractHurufFeatures, extractKataFeatures } from "../lib/features";
import { getStableKataPrediction } from "../lib/prediction";

export function useSignRecognition() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);

  const animationFrameRef = useRef<number | null>(null);
  const lastApiCallRef = useRef<number>(0);

  const kataBufferRef = useRef<number[][]>([]);
  const activeModeRef = useRef("huruf");

  // Riwayat prediksi kata untuk smoothing (majority voting).
  const kataPredictionHistoryRef = useRef<KataPrediction[]>([]);

  const [activeMode, setActiveMode] = useState("huruf");
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [modelLoading, setModelLoading] = useState(false);

  const [handDetected, setHandDetected] = useState(false);
  const [handCount, setHandCount] = useState(0);
  const [landmarkCount, setLandmarkCount] = useState(0);
  const [featureCount, setFeatureCount] = useState(0);
  const [latestFeatures, setLatestFeatures] = useState<number[]>([]);

  const [predictionValue, setPredictionValue] = useState("-");
  const [predictionLabel, setPredictionLabel] = useState("Waiting for camera");
  const [confidence, setConfidence] = useState(0);
  const [apiStatus, setApiStatus] = useState<ApiStatus>("Disconnected");
  const [responseTime, setResponseTime] = useState("-");

  // Reset buffer & status saat mode berganti.
  useEffect(() => {
    activeModeRef.current = activeMode;

    kataBufferRef.current = [];
    kataPredictionHistoryRef.current = [];
    lastApiCallRef.current = 0;

    setPredictionValue("-");
    setPredictionLabel(
      activeMode === "kata" ? "Mulai gerakan kata" : "Waiting for gesture"
    );
    setConfidence(0);
    setFeatureCount(0);
    setLatestFeatures([]);
  }, [activeMode]);

  // ---- Loader model (cache via ref) ----

  const loadHandLandmarker = async () => {
    if (handLandmarkerRef.current) return handLandmarkerRef.current;
    setModelLoading(true);
    const handLandmarker = await createHandLandmarker();
    handLandmarkerRef.current = handLandmarker;
    setModelLoading(false);
    return handLandmarker;
  };

  const loadPoseLandmarker = async () => {
    if (poseLandmarkerRef.current) return poseLandmarkerRef.current;
    setModelLoading(true);
    const poseLandmarker = await createPoseLandmarker();
    poseLandmarkerRef.current = poseLandmarker;
    setModelLoading(false);
    return poseLandmarker;
  };

  // ---- Kirim fitur ke backend + smoothing ----

  const sendFeaturesToAPI = async (
    features: number[],
    detectedHandCount: number
  ) => {
    const mode = activeModeRef.current;

    if (mode === "angka" && detectedHandCount < 1) {
      setPredictionValue("-");
      setPredictionLabel("Tampilkan tangan untuk mode angka");
      setConfidence(0);
      return;
    }

    if (mode === "huruf" && detectedHandCount < 1) {
      setPredictionValue("-");
      setPredictionLabel("Tampilkan tangan untuk mode huruf");
      setConfidence(0);
      return;
    }

    if (mode === "kata" && detectedHandCount < 1) {
      setPredictionValue("-");
      setPredictionLabel("Tampilkan tangan untuk mode kata");
      setConfidence(0);
      return;
    }

    const now = performance.now();
    if (now - lastApiCallRef.current < API_THROTTLE_MS) {
      return;
    }

    let payloadFeatures = features;

    if (mode === "angka") {
      payloadFeatures = features.slice(0, ANGKA_FEATURES);
    }

    if (mode === "huruf" && payloadFeatures.length !== HURUF_FEATURES) return;
    if (mode === "angka" && payloadFeatures.length !== ANGKA_FEATURES) return;
    if (mode === "kata" && payloadFeatures.length !== KATA_TOTAL_FEATURES)
      return;

    lastApiCallRef.current = now;

    try {
      const startTime = performance.now();

      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, features: payloadFeatures }),
      });

      const endTime = performance.now();
      const result = await response.json();

      if (!response.ok || result.error) {
        console.error("API error:", result);
        setApiStatus("Error");
        return;
      }

      setApiStatus("Connected");
      setResponseTime(`${Math.round(endTime - startTime)} ms`);

      const rawConfidence = result.confidence ?? 0;
      const rawPrediction = result.prediction ?? "-";

      if (mode === "kata") {
        // Hanya terima prediksi yang cukup yakin, lalu smoothing.
        if (rawConfidence >= KATA_CONFIDENCE_THRESHOLD) {
          kataPredictionHistoryRef.current.push({
            label: rawPrediction,
            confidence: rawConfidence,
          });
          if (kataPredictionHistoryRef.current.length > KATA_SMOOTHING_WINDOW) {
            kataPredictionHistoryRef.current.shift();
          }

          const stable = getStableKataPrediction(
            kataPredictionHistoryRef.current
          );
          setPredictionValue(stable.label);
          setPredictionLabel("kata terdeteksi");
          setConfidence(Math.round(stable.confidence * 100));
        } else {
          setPredictionValue(rawPrediction);
          setPredictionLabel("kurang yakin");
          setConfidence(Math.round(rawConfidence * 100));
        }
      } else {
        setPredictionValue(rawPrediction);
        setPredictionLabel(`${mode} terdeteksi`);
        setConfidence(Math.round(rawConfidence * 100));
      }
    } catch (error) {
      console.error("Fetch API error:", error);
      setApiStatus("Disconnected");
      setResponseTime("-");
    }
  };

  const resetKataBuffer = () => {
    kataBufferRef.current = [];
    kataPredictionHistoryRef.current = [];
    lastApiCallRef.current = 0;

    setFeatureCount(0);
    setLatestFeatures([]);
    setPredictionValue("-");
    setPredictionLabel("Buffer kata direset");
    setConfidence(0);
    setResponseTime("-");

    console.log("[INFO] Buffer kata direset.");
  };

  // ---- Loop deteksi per-frame ----

  const detectHands = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const handLandmarker = handLandmarkerRef.current;
    const poseLandmarker = poseLandmarkerRef.current;
    const mode = activeModeRef.current;

    if (!video || !canvas || !handLandmarker) return;
    if (mode === "kata" && !poseLandmarker) return;

    if (video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(detectHands);
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const now = performance.now();
    const handResults = handLandmarker.detectForVideo(video, now);
    const poseResults =
      mode === "kata" && poseLandmarker
        ? poseLandmarker.detectForVideo(video, now)
        : null;

    const drawingUtils = new DrawingUtils(ctx);

    if (handResults.landmarks && handResults.landmarks.length > 0) {
      const totalHands = handResults.landmarks.length;
      const totalHandLandmarks = handResults.landmarks.reduce(
        (total, handLandmarks) => total + handLandmarks.length,
        0
      );

      let features: number[] = [];

      if (mode === "kata") {
        features = extractKataFeatures(
          poseResults?.landmarks?.[0],
          handResults.landmarks,
          handResults.handedness
        );

        if (features.length === KATA_NUM_FEATURES) {
          kataBufferRef.current.push(features);
          if (kataBufferRef.current.length > KATA_MAX_FRAMES) {
            kataBufferRef.current.shift();
          }

          setLatestFeatures(features);
          setFeatureCount(kataBufferRef.current.length * KATA_NUM_FEATURES);

          if (kataBufferRef.current.length === KATA_MAX_FRAMES) {
            const flatFeatures = kataBufferRef.current.flat();
            sendFeaturesToAPI(flatFeatures, totalHands);
          } else {
            setPredictionValue("-");
            setPredictionLabel(
              `Mengumpulkan frame kata ${kataBufferRef.current.length}/${KATA_MAX_FRAMES}`
            );
            setConfidence(0);
          }
        }
      } else {
        features = extractHurufFeatures(
          handResults.landmarks,
          handResults.handedness
        );

        setLatestFeatures(features);
        setFeatureCount(features.length);

        sendFeaturesToAPI(features, totalHands);
      }

      setHandDetected(true);
      setHandCount(totalHands);

      const poseLandmarkCount =
        mode === "kata" && poseResults?.landmarks?.[0]
          ? poseResults.landmarks[0].length
          : 0;

      setLandmarkCount(totalHandLandmarks + poseLandmarkCount);

      for (const handLandmarks of handResults.landmarks) {
        drawingUtils.drawConnectors(
          handLandmarks,
          HandLandmarker.HAND_CONNECTIONS,
          { color: "#ffffff", lineWidth: 3 }
        );
        drawingUtils.drawLandmarks(handLandmarks, {
          color: "#ffffff",
          lineWidth: 2,
          radius: 4,
        });
      }

      if (mode === "kata" && poseResults?.landmarks?.[0]) {
        drawingUtils.drawLandmarks(poseResults.landmarks[0], {
          color: "#facc15",
          lineWidth: 1,
          radius: 2,
        });
      }
    } else {
      const emptyFeatures =
        mode === "kata"
          ? Array(KATA_NUM_FEATURES).fill(0)
          : Array(HURUF_FEATURES).fill(0);

      if (mode === "kata") {
        // Jangan reset buffer saat tangan hilang sesaat; isi frame kosong.
        kataBufferRef.current.push(Array(KATA_NUM_FEATURES).fill(0));
        if (kataBufferRef.current.length > KATA_MAX_FRAMES) {
          kataBufferRef.current.shift();
        }
      }

      setHandDetected(false);
      setHandCount(0);
      setLandmarkCount(0);
      setLatestFeatures(emptyFeatures);
      setFeatureCount(0);
    }

    animationFrameRef.current = requestAnimationFrame(detectHands);
  };

  // ---- Kontrol kamera ----

  const startCamera = async () => {
    try {
      setCameraError("");
      setApiStatus("Disconnected");

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraError("Browser tidak mendukung akses kamera.");
        return;
      }

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }

      await loadHandLandmarker();
      await loadPoseLandmarker();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;
      setCameraActive(true);

      setTimeout(async () => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          try {
            await videoRef.current.play();
            detectHands();
          } catch (playError) {
            console.error("Video play error:", playError);
          }
        }
      }, 100);
    } catch (error) {
      console.error("Camera error:", error);
      setCameraActive(false);
      setHandDetected(false);
      setHandCount(0);
      setLandmarkCount(0);
      setLatestFeatures([]);
      setFeatureCount(0);
      setModelLoading(false);
      setApiStatus("Disconnected");
      setCameraError(
        "Gagal mengakses kamera atau memuat MediaPipe. Pastikan permission kamera sudah diizinkan."
      );
    }
  };

  const stopCamera = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    kataBufferRef.current = [];
    kataPredictionHistoryRef.current = [];

    setCameraActive(false);
    setHandDetected(false);
    setHandCount(0);
    setLandmarkCount(0);
    setLatestFeatures([]);
    setFeatureCount(0);

    setPredictionValue("-");
    setPredictionLabel("Waiting for camera");
    setConfidence(0);
    setResponseTime("-");
    setApiStatus("Disconnected");
  };

  // Cleanup saat unmount.
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const prediction = useMemo(() => {
    if (!cameraActive) {
      return { value: "-", label: "Waiting for camera", confidence: 0 };
    }
    return {
      value: predictionValue,
      label: predictionLabel,
      confidence: confidence,
    };
  }, [cameraActive, predictionValue, predictionLabel, confidence]);

  return {
    // refs untuk elemen DOM
    videoRef,
    canvasRef,

    // state
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
    confidence,
    apiStatus,
    responseTime,
    prediction,

    // actions
    startCamera,
    stopCamera,
    resetKataBuffer,
  };
}

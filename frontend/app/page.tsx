"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

import {
  Activity,
  AlertCircle,
  Brain,
  Camera,
  CircleStop,
  Hand,
  Info,
  Play,
  RotateCcw,
  Server,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000/predict";

const KATA_MAX_FRAMES = 50;
const KATA_NUM_FEATURES = 258;
const KATA_TOTAL_FEATURES = KATA_MAX_FRAMES * KATA_NUM_FEATURES;

const modes = [
  { id: "huruf", label: "Huruf", description: "A-Z" },
  { id: "angka", label: "Angka", description: "0-9" },
  { id: "kata", label: "Kata", description: "GRU" },
];

function cn(...classes: string[]) {
  return classes.filter(Boolean).join(" ");
}

function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-neutral-200 bg-white p-5 shadow-sm",
        className
      )}
    >
      {children}
    </div>
  );
}

function Badge({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "success" | "warning" | "danger";
}) {
  const toneClass = {
    neutral: "border-neutral-200 bg-neutral-50 text-neutral-700",
    success: "border-green-200 bg-green-50 text-green-700",
    warning: "border-yellow-200 bg-yellow-50 text-yellow-800",
    danger: "border-red-200 bg-red-50 text-red-700",
  }[tone];

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium",
        toneClass
      )}
    >
      {children}
    </span>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const safeValue = Math.max(0, Math.min(100, value));

  const barColor =
    safeValue >= 85
      ? "bg-green-600"
      : safeValue >= 60
      ? "bg-yellow-600"
      : "bg-red-600";

  const label =
    safeValue >= 85
      ? "High confidence"
      : safeValue >= 60
      ? "Medium confidence"
      : "Low confidence";

  return (
    <div className="space-y-3">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-neutral-500">Confidence</p>
          <p className="mt-1 text-4xl font-bold tracking-tight text-neutral-950">
            {safeValue}%
          </p>
        </div>

        <Badge
          tone={
            safeValue >= 85 ? "success" : safeValue >= 60 ? "warning" : "danger"
          }
        >
          {label}
        </Badge>
      </div>

      <div className="h-2.5 overflow-hidden rounded-full bg-neutral-100">
        <div
          className={cn("h-full rounded-full", barColor)}
          style={{ width: `${safeValue}%` }}
        />
      </div>
    </div>
  );
}

type HandLandmark = {
  x: number;
  y: number;
  z: number;
};

type PoseLandmark = {
  x: number;
  y: number;
  z: number;
  visibility?: number;
};

type ApiStatus = "Connected" | "Disconnected" | "Error";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);

  const animationFrameRef = useRef<number | null>(null);
  const lastApiCallRef = useRef<number>(0);

  const kataBufferRef = useRef<number[][]>([]);
  const activeModeRef = useRef("huruf");

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

  useEffect(() => {
    activeModeRef.current = activeMode;

    kataBufferRef.current = [];
    lastApiCallRef.current = 0;

    setPredictionValue("-");
    setPredictionLabel(
      activeMode === "kata" ? "Mulai gerakan kata" : "Waiting for gesture"
    );
    setConfidence(0);
    setFeatureCount(0);
    setLatestFeatures([]);
  }, [activeMode]);

  const loadHandLandmarker = async () => {
    if (handLandmarkerRef.current) {
      return handLandmarkerRef.current;
    }

    setModelLoading(true);

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    const handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });

    handLandmarkerRef.current = handLandmarker;
    setModelLoading(false);

    return handLandmarker;
  };

  const loadPoseLandmarker = async () => {
    if (poseLandmarkerRef.current) {
      return poseLandmarkerRef.current;
    }

    setModelLoading(true);

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });

    poseLandmarkerRef.current = poseLandmarker;
    setModelLoading(false);

    return poseLandmarker;
  };

  const extractFeaturesFromLandmarks = (
    landmarksList: HandLandmark[][],
    handednessList: {
      score: number;
      index: number;
      categoryName: string;
      displayName: string;
    }[][]
  ) => {
    const features: number[] = [];

    const hands: { right: HandLandmark[] | null; left: HandLandmark[] | null } =
      {
        right: null,
        left: null,
      };

    handednessList.forEach((handedness, index) => {
      const label = handedness[0]?.categoryName;

      // Karena video/canvas di-flip secara visual.
      // Mapping ini mengikuti kode kamu sebelumnya.
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
  };

  const extractKataFeatures = (
    poseLandmarks: PoseLandmark[] | undefined,
    landmarksList: HandLandmark[][],
    handednessList: {
      score: number;
      index: number;
      categoryName: string;
      displayName: string;
    }[][]
  ) => {
    const features: number[] = [];

    // Pose: 33 x 4 = 132
    if (poseLandmarks && poseLandmarks.length > 0) {
      for (const lm of poseLandmarks) {
        features.push(lm.x, lm.y, lm.z, lm.visibility ?? 0);
      }
    } else {
      features.push(...Array(33 * 4).fill(0));
    }

    const hands: { right: HandLandmark[] | null; left: HandLandmark[] | null } =
      {
        right: null,
        left: null,
      };

    handednessList.forEach((handedness, index) => {
      const label = handedness[0]?.categoryName;

      // Tetap mengikuti mapping kode kamu sebelumnya karena video di-mirror.
      if (label === "Left") hands.right = landmarksList[index];
      if (label === "Right") hands.left = landmarksList[index];
    });

    // Left hand: 21 x 3 = 63
    if (hands.left) {
      for (const lm of hands.left) {
        features.push(lm.x, lm.y, lm.z);
      }
    } else {
      features.push(...Array(21 * 3).fill(0));
    }

    // Right hand: 21 x 3 = 63
    if (hands.right) {
      for (const lm of hands.right) {
        features.push(lm.x, lm.y, lm.z);
      }
    } else {
      features.push(...Array(21 * 3).fill(0));
    }

    return features;
  };

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

    if (now - lastApiCallRef.current < 500) {
      return;
    }

    let payloadFeatures = features;

    if (mode === "angka") {
      payloadFeatures = features.slice(0, 63);
    }

    if (mode === "huruf" && payloadFeatures.length !== 126) {
      return;
    }

    if (mode === "angka" && payloadFeatures.length !== 63) {
      return;
    }

    if (mode === "kata" && payloadFeatures.length !== KATA_TOTAL_FEATURES) {
      return;
    }

    lastApiCallRef.current = now;

    try {
      const startTime = performance.now();

      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          mode,
          features: payloadFeatures,
        }),
      });

      const endTime = performance.now();
      const result = await response.json();

      if (!response.ok || result.error) {
        console.error("API error:", result);
        setApiStatus("Error");
        return;
      }

      setPredictionValue(result.prediction ?? "-");
      setPredictionLabel(`${mode} terdeteksi`);
      setConfidence(Math.round((result.confidence ?? 0) * 100));
      setApiStatus("Connected");
      setResponseTime(`${Math.round(endTime - startTime)} ms`);
    } catch (error) {
      console.error("Fetch API error:", error);
      setApiStatus("Disconnected");
      setResponseTime("-");
    }
  };

  const resetKataBuffer = () => {
    kataBufferRef.current = [];
    lastApiCallRef.current = 0;

    setFeatureCount(0);
    setLatestFeatures([]);
    setPredictionValue("-");
    setPredictionLabel("Buffer kata direset");
    setConfidence(0);
    setResponseTime("-");

    console.log("[INFO] Buffer kata direset.");
  };

  const detectHands = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const handLandmarker = handLandmarkerRef.current;
    const poseLandmarker = poseLandmarkerRef.current;
    const mode = activeModeRef.current;

    if (!video || !canvas || !handLandmarker) {
      return;
    }

    if (mode === "kata" && !poseLandmarker) {
      return;
    }

    if (video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(detectHands);
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");

    if (!ctx) {
      return;
    }

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
        features = extractFeaturesFromLandmarks(
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
          {
            color: "#ffffff",
            lineWidth: 3,
          }
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
        mode === "kata" ? Array(KATA_NUM_FEATURES).fill(0) : Array(126).fill(0);

      if (mode === "kata") {
        kataBufferRef.current = [];
      }

      setHandDetected(false);
      setHandCount(0);
      setLandmarkCount(0);
      setLatestFeatures(emptyFeatures);
      setFeatureCount(0);

      if (mode === "kata") {
        setPredictionValue("-");
        setPredictionLabel("Tampilkan tangan untuk mode kata");
        setConfidence(0);
      }
    }

    animationFrameRef.current = requestAnimationFrame(detectHands);
  };

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
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });

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
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

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

  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, []);

  const prediction = useMemo(() => {
    if (!cameraActive) {
      return {
        value: "-",
        label: "Waiting for camera",
        confidence: 0,
      };
    }

    return {
      value: predictionValue,
      label: predictionLabel,
      confidence: confidence,
    };
  }, [cameraActive, predictionValue, predictionLabel, confidence]);

  return (
    <main className="min-h-screen bg-neutral-50 text-neutral-950">
      <header className="border-b border-neutral-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
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

          <div className="flex flex-wrap items-center gap-2">
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

            <Badge tone={modelLoading ? "warning" : "success"}>
              {modelLoading ? "Model Loading" : "Model Loaded"}
            </Badge>

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

      <section className="mx-auto grid max-w-7xl gap-6 px-5 py-6 lg:grid-cols-[1.75fr_1fr] lg:px-8">
        <div className="space-y-6">
          <Card className="p-0">
            <div className="flex items-center justify-between gap-4 border-b border-neutral-200 p-5">
              <div>
                <h2 className="text-lg font-semibold tracking-tight">
                  Camera Preview
                </h2>
                <p className="mt-1 text-sm text-neutral-500">
                  Pastikan tangan dan tubuh bagian atas terlihat jelas di dalam
                  frame kamera.
                </p>
              </div>

              <Badge tone={cameraActive ? "success" : "neutral"}>
                {cameraActive ? "Camera Active" : "Camera Off"}
              </Badge>
            </div>

            <div className="p-5">
              {cameraError && (
                <div className="mb-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {cameraError}
                </div>
              )}

              <div className="relative aspect-video overflow-hidden rounded-2xl bg-neutral-950">
                {cameraActive ? (
                  <>
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="h-full w-full scale-x-[-1] object-cover"
                    />

                    <canvas
                      ref={canvasRef}
                      className="absolute inset-0 h-full w-full scale-x-[-1]"
                    />

                    <div className="absolute left-4 top-4 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-medium text-white backdrop-blur">
                      {handDetected ? "Hand detected" : "No hand detected"}
                    </div>

                    <div className="absolute right-4 top-4 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-medium text-white backdrop-blur">
                      {handCount} {handCount > 1 ? "hands" : "hand"} •{" "}
                      {landmarkCount} landmarks
                    </div>

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
                  <div className="flex h-full flex-col items-center justify-center px-6 text-center text-white">
                    <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-white/10">
                      <Camera size={30} />
                    </div>

                    <h3 className="text-lg font-semibold">Camera is off</h3>
                    <p className="mt-2 max-w-sm text-sm text-white/60">
                      Click Start Camera to begin detection and show your
                      gesture.
                    </p>
                  </div>
                )}
              </div>

              <div className="mt-5 flex flex-col gap-3 sm:flex-row">
                <button
                  onClick={startCamera}
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
                  onClick={stopCamera}
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
                  onClick={resetKataBuffer}
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

          <div className="grid gap-6 md:grid-cols-3">
            <Card>
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-neutral-100">
                  <Activity size={18} />
                </div>
                <div>
                  <p className="text-sm text-neutral-500">Response Time</p>
                  <p className="text-xl font-semibold">{responseTime}</p>
                </div>
              </div>
            </Card>

            <Card>
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-neutral-100">
                  <Brain size={18} />
                </div>
                <div>
                  <p className="text-sm text-neutral-500">Active Model</p>
                  <p className="text-xl font-semibold">
                    {activeMode === "kata" ? "GRU Kata" : "Static RF"}
                  </p>
                </div>
              </div>
            </Card>

            <Card>
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-neutral-100">
                  <Server size={18} />
                </div>
                <div>
                  <p className="text-sm text-neutral-500">API Status</p>
                  <p className="text-xl font-semibold">{apiStatus}</p>
                </div>
              </div>
            </Card>
          </div>
        </div>

        <aside className="space-y-6">
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

          <Card>
            <ConfidenceBar value={prediction.confidence} />
          </Card>

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
                  {activeMode === "kata"
                    ? `${featureCount}/${KATA_TOTAL_FEATURES}`
                    : `${featureCount}/${activeMode === "angka" ? 63 : 126}`}
                </span>
              </div>
            </div>
          </Card>

          <Card>
            <div className="mb-4 flex items-center gap-2">
              <AlertCircle size={18} />
              <h2 className="text-lg font-semibold tracking-tight">
                How to Use
              </h2>
            </div>

            <ol className="space-y-2 text-sm leading-6 text-neutral-600">
              <li>1. Click Start Camera.</li>
              <li>2. Select Huruf, Angka, or Kata mode.</li>
              <li>3. Place your hands clearly inside the camera frame.</li>
              <li>
                4. For Kata mode, perform the movement until the 50-frame buffer
                is full.
              </li>
              <li>5. The prediction result will appear automatically.</li>
            </ol>
          </Card>
        </aside>
      </section>

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
                {activeMode === "kata"
                  ? `${featureCount}/${KATA_TOTAL_FEATURES}`
                  : `${featureCount}/${activeMode === "angka" ? 63 : 126}`}
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
    </main>
  );
}
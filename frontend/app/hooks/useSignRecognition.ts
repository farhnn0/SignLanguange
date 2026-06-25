"use client";

/**
 * useSignRecognition — Hook Utama Pengenalan Isyarat
 * ===================================================
 * Ini adalah "otak" dari seluruh aplikasi web.
 * Semua logika ada di sini: akses kamera, deteksi landmark, kirim data ke backend, tampilkan hasil.
 *
 * Alur kerja singkat:
 * Webcam → MediaPipe (deteksi tangan/pose) → Ekstrak koordinat angka → Kirim ke FastAPI → Tampilkan prediksi
 */

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

  // ============================================================
  // 1. INISIALISASI STATE & REFS
  // ============================================================
  // "ref" = referensi ke sesuatu, isinya tidak memicu render ulang UI
  // "state" = data yang jika berubah akan memperbarui tampilan UI

  const videoRef = useRef<HTMLVideoElement | null>(null);   // Referensi ke elemen <video> di HTML (tampilan kamera)
  const canvasRef = useRef<HTMLCanvasElement | null>(null); // Referensi ke <canvas> untuk menggambar rangka tangan di atas video
  const streamRef = useRef<MediaStream | null>(null);       // Menyimpan stream kamera yang sedang aktif agar bisa dihentikan nanti

  const handLandmarkerRef = useRef<HandLandmarker | null>(null); // Instance model deteksi tangan MediaPipe (di-cache agar tidak reload ulang)
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null); // Instance model deteksi pose/badan MediaPipe (hanya dipakai di mode kata)

  const animationFrameRef = useRef<number | null>(null); // ID untuk requestAnimationFrame — dipakai agar loop deteksi bisa dihentikan
  const lastApiCallRef = useRef<number>(0);              // Waktu terakhir kita kirim data ke backend (untuk throttling: batasi max 1 request per 500ms)

  // Buffer untuk mode KATA:
  // Menampung koordinat dari 50 frame terakhir. Baru dikirim ke backend kalau sudah penuh 50 frame.
  const kataBufferRef = useRef<number[][]>([]);
  const activeModeRef = useRef("huruf"); // Ref mode aktif — dipakai di dalam loop deteksi (tidak bisa pakai state langsung di dalam loop)

  // Riwayat 5 prediksi kata terakhir untuk smoothing (majority voting)
  // Tujuan: agar hasil tidak "kedap-kedip" ganti-ganti tiap frame
  const kataPredictionHistoryRef = useRef<KataPrediction[]>([]);

  // State UI — perubahan ini akan memperbarui tampilan web secara otomatis
  const [activeMode, setActiveMode] = useState("huruf");               // Mode aktif: "huruf", "angka", atau "kata"
  const [cameraActive, setCameraActive] = useState(false);             // Apakah kamera sedang menyala?
  const [cameraError, setCameraError] = useState("");                  // Pesan error jika kamera gagal diakses
  const [modelLoading, setModelLoading] = useState(false);             // Sedang loading model MediaPipe?

  const [handDetected, setHandDetected] = useState(false);             // Apakah tangan terdeteksi di kamera?
  const [handCount, setHandCount] = useState(0);                       // Jumlah tangan yang terdeteksi (0/1/2)
  const [landmarkCount, setLandmarkCount] = useState(0);               // Total titik koordinat yang terdeteksi
  const [featureCount, setFeatureCount] = useState(0);                 // Jumlah angka koordinat yang siap dikirim ke API
  const [latestFeatures, setLatestFeatures] = useState<number[]>([]); // Data koordinat terbaru (untuk debug/ditampilkan)

  const [predictionValue, setPredictionValue] = useState("-");          // Hasil prediksi: misal "A", "3", atau "Makan"
  const [predictionLabel, setPredictionLabel] = useState("Waiting for camera"); // Teks status di bawah hasil prediksi
  const [confidence, setConfidence] = useState(0);                      // Persentase keyakinan model (0-100%)
  const [apiStatus, setApiStatus] = useState<ApiStatus>("Disconnected"); // Status koneksi ke backend FastAPI
  const [responseTime, setResponseTime] = useState("-");                 // Kecepatan respons backend (misal: "45 ms")


  // ============================================================
  // 2. RESET OTOMATIS SAAT MODE BERGANTI
  // ============================================================
  // Setiap kali user pindah mode (huruf → kata → angka), buffer dan riwayat dikosongkan
  // agar prediksi mode sebelumnya tidak "mencemari" mode yang baru

  useEffect(() => {
    activeModeRef.current = activeMode; // Sinkronkan ref dengan state

    kataBufferRef.current = [];                   // Kosongkan buffer 50 frame kata
    kataPredictionHistoryRef.current = [];        // Kosongkan riwayat prediksi untuk smoothing
    lastApiCallRef.current = 0;                   // Reset timer throttling

    setPredictionValue("-");
    setPredictionLabel(
      activeMode === "kata" ? "Mulai gerakan kata" : "Waiting for gesture"
    );
    setConfidence(0);
    setFeatureCount(0);
    setLatestFeatures([]);
  }, [activeMode]);


  // ============================================================
  // 3. LOAD MODEL MEDIAPIPE (Hanya sekali, di-cache via ref)
  // ============================================================
  // Model MediaPipe diunduh dari internet (CDN) saat pertama kali dipakai.
  // Setelah itu disimpan di ref agar tidak diunduh ulang setiap frame.

  const loadHandLandmarker = async () => {
    if (handLandmarkerRef.current) return handLandmarkerRef.current; // Sudah ada, pakai yang lama
    setModelLoading(true);
    const handLandmarker = await createHandLandmarker(); // Download + inisialisasi model detektor tangan
    handLandmarkerRef.current = handLandmarker;
    setModelLoading(false);
    return handLandmarker;
  };

  const loadPoseLandmarker = async () => {
    if (poseLandmarkerRef.current) return poseLandmarkerRef.current; // Sudah ada, pakai yang lama
    setModelLoading(true);
    const poseLandmarker = await createPoseLandmarker(); // Download + inisialisasi model detektor pose tubuh
    poseLandmarkerRef.current = poseLandmarker;
    setModelLoading(false);
    return poseLandmarker;
  };


  // ============================================================
  // 4. KIRIM DATA KE BACKEND & TAMPILKAN HASIL
  // ============================================================
  // Fungsi ini mengirim array angka koordinat ke FastAPI via HTTP POST.
  // Backend akan memasukkannya ke model (RF atau GRU) dan mengembalikan prediksi.

  const sendFeaturesToAPI = async (
    features: number[],  // Array angka koordinat (126 untuk huruf, 63 untuk angka, 12900 untuk kata)
    detectedHandCount: number
  ) => {
    const mode = activeModeRef.current;

    // Jangan kirim jika tidak ada tangan di kamera (tidak ada data yang berarti)
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

    // Throttling: batasi request maksimal 1 kali setiap 500ms agar tidak membanjiri backend
    const now = performance.now();
    if (now - lastApiCallRef.current < API_THROTTLE_MS) {
      return; // Belum waktunya kirim lagi, lewati
    }

    let payloadFeatures = features;

    if (mode === "angka") {
      payloadFeatures = features.slice(0, ANGKA_FEATURES); // Angka hanya butuh 63 fitur (1 tangan)
    }

    // Validasi panjang data sebelum dikirim
    if (mode === "huruf" && payloadFeatures.length !== HURUF_FEATURES) return; // Harus 126
    if (mode === "angka" && payloadFeatures.length !== ANGKA_FEATURES) return; // Harus 63
    if (mode === "kata" && payloadFeatures.length !== KATA_TOTAL_FEATURES) return; // Harus 12900 (50×258)

    lastApiCallRef.current = now; // Catat waktu pengiriman untuk throttling berikutnya

    try {
      const startTime = performance.now();

      // HTTP POST ke backend FastAPI (localhost:8000/predict)
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, features: payloadFeatures }), // Kirim mode + data koordinat
      });

      const endTime = performance.now();
      const result = await response.json(); // Terima jawaban dari backend

      if (!response.ok || result.error) {
        console.error("API error:", result);
        setApiStatus("Error");
        return;
      }

      setApiStatus("Connected");
      setResponseTime(`${Math.round(endTime - startTime)} ms`); // Tampilkan kecepatan respons

      const rawConfidence = result.confidence ?? 0;  // Nilai keyakinan dari model (0.0 - 1.0)
      const rawPrediction = result.prediction ?? "-"; // Nama kata/huruf/angka hasil prediksi

      if (mode === "kata") {
        // Mode kata: ada smoothing (majority voting) agar hasil tidak loncat-loncat tiap frame
        if (rawConfidence >= KATA_CONFIDENCE_THRESHOLD) {
          // Confidence cukup tinggi (≥ 0.75): masukkan ke riwayat untuk voting
          kataPredictionHistoryRef.current.push({
            label: rawPrediction,
            confidence: rawConfidence,
          });

          // Pertahankan hanya 5 prediksi terakhir
          if (kataPredictionHistoryRef.current.length > KATA_SMOOTHING_WINDOW) {
            kataPredictionHistoryRef.current.shift(); // Hapus yang paling lama
          }

          // Ambil prediksi yang paling sering muncul dari 5 prediksi terakhir (majority voting)
          const stable = getStableKataPrediction(kataPredictionHistoryRef.current);
          setPredictionValue(stable.label);
          setPredictionLabel("kata terdeteksi");
          setConfidence(Math.round(stable.confidence * 100));
        } else {
          // Confidence rendah: tampilkan tapi tandai "kurang yakin", tidak masuk voting
          setPredictionValue(rawPrediction);
          setPredictionLabel("kurang yakin");
          setConfidence(Math.round(rawConfidence * 100));
        }
      } else {
        // Mode huruf/angka: langsung tampilkan hasil mentah tanpa smoothing
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


  // ============================================================
  // 5. RESET BUFFER KATA (dipanggil saat user klik tombol "Reset Kata")
  // ============================================================

  const resetKataBuffer = () => {
    kataBufferRef.current = [];            // Kosongkan 50 frame yang sudah terkumpul
    kataPredictionHistoryRef.current = []; // Kosongkan riwayat smoothing
    lastApiCallRef.current = 0;            // Reset timer throttling

    setFeatureCount(0);
    setLatestFeatures([]);
    setPredictionValue("-");
    setPredictionLabel("Buffer kata direset");
    setConfidence(0);
    setResponseTime("-");

    console.log("[INFO] Buffer kata direset.");
  };


  // ============================================================
  // 6. LOOP DETEKSI UTAMA (berjalan setiap frame, ~30-60 kali per detik)
  // ============================================================
  // Ini adalah fungsi terpenting: dijalankan secara terus-menerus menggunakan requestAnimationFrame
  // setiap frame → deteksi → ekstrak koordinat → (isi buffer / kirim ke API)

  const detectHands = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const handLandmarker = handLandmarkerRef.current;
    const poseLandmarker = poseLandmarkerRef.current;
    const mode = activeModeRef.current;

    if (!video || !canvas || !handLandmarker) return;
    if (mode === "kata" && !poseLandmarker) return;

    // Tunggu video siap dibaca (readyState ≥ 2 = sudah ada data)
    if (video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(detectHands);
      return;
    }

    // Samakan ukuran canvas dengan ukuran video agar overlay landmark pas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height); // Bersihkan gambar frame sebelumnya

    const now = performance.now();

    // Jalankan deteksi MediaPipe pada frame video saat ini
    const handResults = handLandmarker.detectForVideo(video, now); // Deteksi tangan
    const poseResults =
      mode === "kata" && poseLandmarker
        ? poseLandmarker.detectForVideo(video, now) // Deteksi pose tubuh (hanya di mode kata)
        : null;

    const drawingUtils = new DrawingUtils(ctx);

    if (handResults.landmarks && handResults.landmarks.length > 0) {
      // Ada tangan terdeteksi di frame ini
      const totalHands = handResults.landmarks.length; // Jumlah tangan (1 atau 2)
      const totalHandLandmarks = handResults.landmarks.reduce(
        (total, handLandmarks) => total + handLandmarks.length, 0
      );

      let features: number[] = [];

      if (mode === "kata") {
        // Mode KATA: ekstrak 258 fitur (pose 132 + tangan kiri 63 + tangan kanan 63)
        features = extractKataFeatures(
          poseResults?.landmarks?.[0], // Koordinat 33 titik pose tubuh
          handResults.landmarks,        // Koordinat titik kedua tangan
          handResults.handedness        // Info kiri/kanan untuk tiap tangan
        );

        if (features.length === KATA_NUM_FEATURES) {
          // Masukkan 258 angka ini ke dalam buffer (antrian 50 frame)
          kataBufferRef.current.push(features);

          // Buffer dibatasi 50 frame: kalau lebih, hapus yang paling lama (sliding window)
          if (kataBufferRef.current.length > KATA_MAX_FRAMES) {
            kataBufferRef.current.shift();
          }

          setLatestFeatures(features);
          setFeatureCount(kataBufferRef.current.length * KATA_NUM_FEATURES);

          if (kataBufferRef.current.length === KATA_MAX_FRAMES) {
            // Buffer penuh 50 frame → ratakan jadi 1 array panjang (50×258 = 12.900 angka) → kirim ke backend
            const flatFeatures = kataBufferRef.current.flat();
            sendFeaturesToAPI(flatFeatures, totalHands);
          } else {
            // Buffer belum penuh, tampilkan progres
            setPredictionValue("-");
            setPredictionLabel(
              `Mengumpulkan frame kata ${kataBufferRef.current.length}/${KATA_MAX_FRAMES}`
            );
            setConfidence(0);
          }
        }
      } else {
        // Mode HURUF/ANGKA: ekstrak 126 fitur (2 tangan × 21 × 3) lalu langsung kirim
        features = extractHurufFeatures(
          handResults.landmarks,
          handResults.handedness
        );

        setLatestFeatures(features);
        setFeatureCount(features.length);

        sendFeaturesToAPI(features, totalHands); // Kirim langsung tanpa menunggu buffer
      }

      setHandDetected(true);
      setHandCount(totalHands);

      const poseLandmarkCount =
        mode === "kata" && poseResults?.landmarks?.[0]
          ? poseResults.landmarks[0].length : 0;

      setLandmarkCount(totalHandLandmarks + poseLandmarkCount);

      // Gambar kerangka tangan di atas video (warna putih)
      for (const handLandmarks of handResults.landmarks) {
        drawingUtils.drawConnectors(
          handLandmarks,
          HandLandmarker.HAND_CONNECTIONS,
          { color: "#ffffff", lineWidth: 3 } // Garis penghubung antar titik
        );
        drawingUtils.drawLandmarks(handLandmarks, {
          color: "#ffffff",
          lineWidth: 2,
          radius: 4, // Titik-titik koordinat landmark
        });
      }

      // Gambar titik-titik pose tubuh di mode kata (warna kuning)
      if (mode === "kata" && poseResults?.landmarks?.[0]) {
        drawingUtils.drawLandmarks(poseResults.landmarks[0], {
          color: "#facc15",
          lineWidth: 1,
          radius: 2,
        });
      }
    } else {
      // Tidak ada tangan terdeteksi di frame ini
      const emptyFeatures =
        mode === "kata"
          ? Array(KATA_NUM_FEATURES).fill(0)
          : Array(HURUF_FEATURES).fill(0);

      if (mode === "kata") {
        // Mode kata: jangan reset buffer total. Isi dengan frame nol agar buffer terus bergerak.
        // Ini mencegah prediksi terhenti total saat tangan hilang sesaat.
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

    // Jadwalkan deteksi frame berikutnya (berjalan terus sampai kamera dimatikan)
    animationFrameRef.current = requestAnimationFrame(detectHands);
  };


  // ============================================================
  // 7. START / STOP KAMERA
  // ============================================================

  const startCamera = async () => {
    try {
      setCameraError("");
      setApiStatus("Disconnected");

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraError("Browser tidak mendukung akses kamera.");
        return;
      }

      // Hentikan loop dan stream yang mungkin masih berjalan dari sesi sebelumnya
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

      // Load model MediaPipe (kalau belum dimuat)
      await loadHandLandmarker();
      await loadPoseLandmarker();

      // Minta izin akses kamera dari browser
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },  // Resolusi ideal 720p
          height: { ideal: 720 },
          facingMode: "user",       // Kamera depan (selfie)
        },
        audio: false, // Tidak butuh mikrofon
      });

      streamRef.current = stream;
      setCameraActive(true);

      setTimeout(async () => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream; // Sambungkan stream ke elemen video HTML
          try {
            await videoRef.current.play();
            detectHands(); // Mulai loop deteksi per frame
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
    // Hentikan loop animasi
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Hentikan semua track kamera agar kamera fisik mati (lampu indikator padam)
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


  // ============================================================
  // 8. CLEANUP & NILAI KEMBALIAN HOOK
  // ============================================================

  // Cleanup otomatis saat komponen dihapus dari halaman (unmount)
  // Mencegah memory leak jika user menutup tab
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

  // useMemo: hanya hitung ulang nilai prediction jika ada perubahan yang relevan
  // Menghindari render ulang UI yang tidak perlu
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

  // Kembalikan semua nilai dan fungsi yang dibutuhkan komponen UI
  return {
    videoRef,         // Untuk elemen <video>
    canvasRef,        // Untuk elemen <canvas> overlay landmark
    activeMode,       // Mode aktif saat ini
    setActiveMode,    // Fungsi ganti mode
    cameraActive,     // Status kamera
    cameraError,      // Pesan error kamera
    modelLoading,     // Status loading model
    handDetected,     // Ada tangan di kamera?
    handCount,        // Jumlah tangan
    landmarkCount,    // Jumlah titik landmark
    featureCount,     // Jumlah fitur yang siap dikirim
    latestFeatures,   // Data koordinat terbaru
    confidence,       // Kepercayaan prediksi (%)
    apiStatus,        // Status koneksi backend
    responseTime,     // Waktu respons backend
    prediction,       // Hasil prediksi final { value, label, confidence }
    startCamera,      // Fungsi nyalakan kamera
    stopCamera,       // Fungsi matikan kamera
    resetKataBuffer,  // Fungsi reset buffer kata
  };
}

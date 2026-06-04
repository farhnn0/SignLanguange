# Sign Language Recognition — Frontend

Antarmuka web untuk pengenalan bahasa isyarat BISINDO secara real-time
(huruf, angka, dan kata) menggunakan webcam + MediaPipe + backend FastAPI.

Dibangun dengan Next.js (App Router) + TypeScript + Tailwind CSS.

---

## Arsitektur Kode

Sebelumnya seluruh logika ada di satu file `app/page.tsx` (~1000 baris).
Sekarang dipecah menjadi modul-modul kecil agar mudah dibaca dan dikembangkan,
**tanpa mengubah perilaku/akurasi sama sekali**.

```
app/
├── page.tsx                  # Komposisi halaman (tipis, hanya merangkai komponen)
│
├── hooks/
│   └── useSignRecognition.ts # SEMUA state & logika: kamera, loop deteksi,
│                             # ekstraksi fitur, kirim ke API, smoothing
│
├── lib/                      # Modul "pure" tanpa React (mudah diuji)
│   ├── constants.ts          # API_URL, jumlah frame/fitur, threshold, URL model
│   ├── types.ts              # Tipe: HandLandmark, PoseLandmark, ApiStatus, dll
│   ├── utils.ts              # Helper cn() untuk className
│   ├── mediapipe.ts          # Loader HandLandmarker & PoseLandmarker
│   ├── features.ts           # Ekstraksi fitur huruf (126) & kata (258)
│   └── prediction.ts         # Majority voting (smoothing) prediksi kata
│
└── components/               # Komponen tampilan (presentational)
    ├── ui.tsx                # Card, Badge, ConfidenceBar
    ├── AppHeader.tsx         # Header + badge status
    ├── CameraPanel.tsx       # Preview kamera + overlay + tombol kontrol
    ├── MetricsRow.tsx        # Kartu Response Time / Active Model / API Status
    ├── Sidebar.tsx           # Pemilih mode, hasil prediksi, status sistem
    └── ModelInfo.tsx         # Ringkasan model + preview fitur
```

### Prinsip pembagian

- **`lib/`** — fungsi murni, tidak menyentuh React/DOM. Paling aman dites & dipakai ulang.
- **`hooks/useSignRecognition.ts`** — satu sumber kebenaran untuk seluruh state dan
  efek samping (akses kamera, `requestAnimationFrame`, fetch API).
- **`components/`** — hanya menerima props dan menampilkan UI. Tidak memuat logika bisnis.
- **`page.tsx`** — hanya memanggil hook lalu meneruskan nilainya ke komponen.

Menambah fitur baru cukup menyentuh modul terkait, bukan satu file raksasa.

---

## Cara Menjalankan

```bash
# 1. Jalankan backend (terminal terpisah, dari folder backend/)
#    pakai venv yang sudah ada TensorFlow
C:\Users\Farhan\asl_env\Scripts\python.exe -m uvicorn main:app --reload

# 2. Jalankan frontend (dari folder frontend/)
npm install
npm run dev
# buka http://localhost:3000
```

Pastikan backend berjalan di `http://127.0.0.1:8000` (lihat `lib/constants.ts`).

---

## Alur Data (mode Kata)

```
Webcam ─► MediaPipe (Hand + Pose) ─► extractKataFeatures() ─► buffer 50 frame
        ─► fetch /predict (50 x 258 = 12900 fitur) ─► GRU di backend
        ─► smoothing majority voting ─► tampil di UI
```

Catatan penting yang menjaga akurasi:

1. **Urutan fitur kata** (pose 132 → tangan kiri 63 → tangan kanan 63) **harus
   identik** dengan `extract_landmarks()` saat training (`Train/train_kata_bisindo_gru.py`).
2. **Pose model** memakai versi `full` (bukan `lite`) agar landmark badan akurat.
3. **Smoothing**: hanya prediksi dengan confidence ≥ 0.75 yang diterima, lalu
   diambil label terbanyak dari 5 prediksi terakhir.

---

## Tips Pemakaian

- **Mundurkan kamera / turunkan layar** sehingga **wajah + tubuh atas + kedua tangan**
  terlihat penuh. Model kata dilatih dengan pose tubuh, jadi butuh ruang yang cukup.
- Untuk mode Kata, lakukan gerakan sampai progress buffer mencapai 50/50 frame.
- Gunakan tombol **Reset Kata** bila ingin mengulang dari awal.

---

## Daftar Gesture Kata (perwakilan)

Berikut contoh gesture kata beserta video perwakilan dari dataset.
Path video relatif terhadap root project (folder `bisindo-kata-baru/`).

| Kata | Kategori | Video Contoh | Catatan Gerakan |
|---|---|---|---|
| **Menulis** | Kata Kerja | `bisindo-kata-baru/Kata Kerja/Menulis/Menulis-a-1.mp4` | Gerakan menulis di telapak tangan; periodik sehingga paling stabil dideteksi. |
| **Terima Kasih** | Kata Lainnya | `bisindo-kata-baru/Kata Lainnya/Terima Kasih/Terima kasih-a-1.mp4` | Dua fase: tangan menyentuh dagu lalu disapukan ke depan. Pastikan gerakan utuh terekam. |
| **Makan** | Kata Kerja | `bisindo-kata-baru/Kata Kerja/Makan/Makan-a-1.mp4` | Tangan mengarah ke mulut, gerakan berulang. |
| **Belajar** | Kata Kerja | `bisindo-kata-baru/Kata Kerja/Belajar/Belajar-a-1.mp4` | Gerakan kedua tangan seperti membuka buku/membaca. |
| **Halo** | Kata Lainnya | `bisindo-kata-baru/Kata Lainnya/Halo/Halo-a-1.mp4` | Lambaian tangan singkat; gesture pendek satu gerakan. |
| **Sekian** | Kata Lainnya | `bisindo-kata-baru/Kata Lainnya/Sekian/Sekian-a-1.mp4` | Biasa dipakai sebagai penutup; gerakan tangan menutup. |
| **Olahraga** | Kata Lainnya | `bisindo-kata-baru/Kata Lainnya/Olahraga/Olahraga-a-1.mp4` | Gerakan kedua tangan menirukan aktivitas fisik. |
| **Keluarga** | Kata Lainnya | `bisindo-kata-baru/Kata Lainnya/Keluarga/Keluarga-a-1.mp4` | Gerakan dua tangan membentuk lingkaran/kelompok. |

> Total model mengenali 50 kata. Tabel di atas hanya perwakilan untuk pengujian cepat.
> Daftar lengkap kelas ada di `kiro_train/class_names_new.json`.

### Cara menampilkan video di markdown (opsional)

GitHub/VS Code preview tidak memutar `.mp4` inline lewat `![]()`. Untuk
menautkannya, gunakan link biasa agar bisa diklik:

```markdown
[Lihat contoh gesture "Menulis"](../bisindo-kata-baru/Kata%20Kerja/Menulis/Menulis-a-1.mp4)
```

Jika ingin thumbnail/preview tampil di halaman web, salin video ke
`frontend/public/samples/` lalu pakai tag `<video>`:

```html
<video src="/samples/Menulis-a-1.mp4" controls width="320"></video>
```

---

## Model & Backend Terkait

- Model kata: `kiro_train/bisindo_holistic_gru_new.h5` (GRU, val accuracy ~94%).
  Sudah disalin ke `backend/Models/bisindo_holistic_gru.h5` (model lama dibackup
  sebagai `*_old.h5`).
- Backend endpoint: `POST /predict` dengan body `{ mode, features }`.
  Lihat `backend/main.py` untuk format tiap mode (huruf 126, angka 63, kata 12900).

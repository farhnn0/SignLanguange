# HandSignature — Real-Time Sign Language Recognition (BISINDO)

Aplikasi berbasis web untuk pengenalan Bahasa Isyarat Indonesia (BISINDO) secara real-time yang mencakup **Huruf**, **Angka**, dan **Kata**. Sistem ini dibangun dengan membagi tugas antara **Frontend (Next.js)** untuk pemrosesan kamera & ekstraksi landmark tangan, serta **Backend (FastAPI)** untuk inferensi model Machine Learning & Deep Learning.

Repositori Resmi: [https://github.com/farhnn0/SignLanguange.git](https://github.com/farhnn0/SignLanguange.git)

---

## 📌 Alur Kerja Sistem

Sistem ini bekerja dengan meminimalkan beban komputasi server menggunakan pendekatan **Edge-Extraction + Server-Inference**:
1. **Frontend (Browser):** Mengakses webcam pengguna dan mengekstrak koordinat tangan/tubuh secara real-time menggunakan **MediaPipe** secara lokal di browser.
2. **Kirim Koordinat:** Koordinat yang diekstrak dikirim ke backend melalui HTTP POST request dalam bentuk list angka (fitur).
3. **Backend (FastAPI):** Menerima koordinat, memasukkannya ke model AI yang sesuai, lalu mengembalikan hasil prediksi beserta nilai kepercayaan (*confidence score*).
4. **Smoothing (Frontend):** Melakukan *majority voting* dari beberapa frame terakhir untuk menyaring hasil agar tidak berkedip (*flicker*).

---

## 📊 Informasi Dataset

Model dalam proyek ini dilatih menggunakan dataset publik berikut:

| Mode | Dataset Kaggle | Algoritma | Input Model (Fitur) |
| :--- | :--- | :--- | :--- |
| **Huruf** | [Indonesian Sign Language BISINDO (AgungMRF)](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo) | Random Forest | 126 Fitur (2 tangan × 21 titik × 3 koordinat XYZ) |
| **Angka** | [Sign Language for Numbers (Muhammad Khalid)](https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-numbers) | Random Forest | 63 Fitur (1 tangan × 21 titik × 3 koordinat XYZ) |
| **Kata** | [BISINDO Kata (xazhurea - Test500)](https://www.kaggle.com/datasets/xazhurea/test500) | GRU Deep Learning | 12.900 Fitur (50 frame × 258 koordinat per frame) |

---

## 📂 Struktur Direktori Proyek

```text
HandSignature/
├── Train/                     # Script python untuk melatih model AI
│   ├── train_huruf.py         # Training model Random Forest untuk huruf
│   ├── train_angka.py         # Training model Random Forest untuk angka
│   └── train_kata_bisindo_gru.py # Training model GRU untuk kata
│
├── backend/                   # Web Server API (Python)
│   ├── Models/                # Tempat menyimpan model hasil training (.pkl & .h5)
│   │   ├── huruf_model.pkl
│   │   ├── angka_model.pkl
│   │   └── bisindo_holistic_gru.h5
│   └── main.py                # Server FastAPI & logika prediksi
│
├── frontend/                  # Web Interface (Next.js + TypeScript)
│   ├── app/                   # Kode aplikasi Next.js (App Router)
│   ├── components/            # Komponen tampilan UI (Camera, Sidebar, dll)
│   ├── lib/                   # Logika ekstraksi MediaPipe & konfigurasi
│   ├── public/                # Asset gambar dan file model MediaPipe
│   └── package.json           # Dependensi modul Node.js
│
└── README.md                  # Dokumentasi proyek
```

---

## 🚀 Panduan Instalasi & Setup (Lokal)

Ikuti langkah-langkah di bawah ini untuk menjalankan proyek di komputer Anda.

### 1. Prasyarat Sistem
Pastikan komputer Anda sudah terinstall:
*   [Python 3.10+](https://www.python.org/downloads/)
*   [Node.js v18+](https://nodejs.org/)
*   Git

---

### 2. Setup Backend (FastAPI)

1. Buka terminal atau Command Prompt baru, lalu masuk ke folder `backend`:
   ```bash
   cd backend
   ```

2. Buat Virtual Environment agar package tidak bentrok dengan sistem global:
   ```bash
   python -m venv venv
   ```

3. Aktifkan Virtual Environment:
   *   **Windows (Command Prompt):**
       ```cmd
       venv\Scripts\activate
       ```
   *   **Windows (PowerShell):**
       ```powershell
       .\venv\Scripts\Activate.ps1
       ```
   *   **Linux/macOS:**
       ```bash
       source venv/bin/activate
       ```

4. Install semua pustaka yang dibutuhkan:
   ```bash
   pip install fastapi uvicorn pydantic numpy tensorflow scikit-learn joblib opencv-python mediapipe tqdm
   ```

5. Jalankan server backend:
   ```bash
   uvicorn main:app --reload
   ```
   *Backend akan berjalan di: `http://127.0.0.1:8000`*

---

### 3. Setup Frontend (Next.js)

1. Buka terminal baru (jangan matikan terminal backend), lalu masuk ke folder `frontend`:
   ```bash
   cd frontend
   ```

2. Install semua modul Node.js yang diperlukan:
   ```bash
   npm install
   ```

3. Jalankan server frontend dalam mode development:
   ```bash
   npm run dev
   ```
   *Frontend akan berjalan di: `http://localhost:3000`*

4. Buka browser Anda dan akses alamat `http://localhost:3000`.

---

## 🧠 Cara Melatih Model Baru (Opsional)

Jika Anda ingin melatih ulang model menggunakan dataset yang diunduh dari Kaggle, ikuti langkah berikut:

### Training Model Huruf
1. Download dataset **Bisindo Huruf** dan ekstrak ke folder `bisindo`. Pastikan strukturnya memiliki subfolder `bisindo/images/train` dan `bisindo/images/val`.
2. Jalankan script training:
   ```bash
   python Train/train_huruf.py
   ```
3. Pindahkan file `huruf_model.pkl` dan `huruf_labels.pkl` yang dihasilkan ke folder `backend/Models/`.

### Training Model Angka
1. Download dataset **Sign Language Numbers** dan ekstrak ke folder `sign-language-for-numbers`.
2. Jalankan script training:
   ```bash
   python Train/train_angka.py
   ```
3. Pindahkan file `angka_model.pkl` dan `angka_labels.pkl` yang dihasilkan ke folder `backend/Models/`.

### Training Model Kata (GRU)
1. Download dataset **BISINDO Kata** dan letakkan di folder `bisindo-kata-baru`.
2. Jalankan script training (proses ini membutuhkan waktu cukup lama karena mengekstrak frame video):
   ```bash
   python Train/train_kata_bisindo_gru.py
   ```
3. Pindahkan file `bisindo_holistic_gru.h5` dan `label_encoder.pkl` yang dihasilkan ke folder `backend/Models/`.

---

## 💡 Tips Penggunaan Aplikasi
*   **Pencahayaan Cukup:** Pastikan ruangan Anda memiliki pencahayaan yang baik agar MediaPipe dapat mendeteksi titik koordinat tangan secara stabil.
*   **Posisi Kamera:** Untuk mode **Kata**, posisikan diri Anda agak mundur sehingga **wajah, tubuh bagian atas, dan kedua tangan** terekam di layar. Model GRU dilatih menggunakan data pose tubuh + tangan.
*   **Kecepatan Gerakan:** Lakukan gerakan isyarat kata sampai indikator frame di layar mencapai **50/50 frame** untuk mengirimkannya ke model GRU di backend.

---

## 📝 Kontributor & Lisensi
*   **Pengembang:** Farhan (Repositori: [farhnn0](https://github.com/farhnn0))
*   Proyek ini dibuat untuk keperluan akademis praktikum Computer Vision.

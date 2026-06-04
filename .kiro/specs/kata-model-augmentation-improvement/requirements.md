# Requirements Document

## Introduction

Fitur ini meningkatkan akurasi model GRU pengenalan kata BISINDO (Bahasa Isyarat Indonesia)
dengan menambahkan pipeline video augmentation. Saat ini model `bisindo_holistic_gru.h5`
mencapai validation accuracy 85%+ namun saat real-time prediction (di `predict_kata.py` dan
backend web) akurasinya turun sekitar 50% karena training data kurang merepresentasikan
variasi user, lighting, kecepatan gerakan, dan kondisi webcam.

Solusi yang dipilih: tambahkan augmentation berbasis MediaPipe landmark dan/atau frame
video yang dieksekusi secara on-the-fly saat training (tidak disimpan ke disk), berjalan
di GPU NVIDIA dengan TensorFlow, sambil mempertahankan arsitektur GRU yang sudah ada
serta nama file output (`bisindo_holistic_gru.h5`, `label_encoder.pkl`, `class_names.json`)
agar `predict_kata.py` dan backend tidak rusak.

Target: real-time prediction accuracy naik dari ~50% menjadi minimal 75%, dengan training
time maksimal 2 jam di GPU NVIDIA.

## Glossary

- **Augmentation_Module**: Modul Python standalone yang menerapkan transformasi
  augmentation pada video atau pada urutan landmark MediaPipe Holistic.
- **Landmark_Augmenter**: Komponen di dalam `Augmentation_Module` yang menerapkan
  augmentation pada urutan landmark berdimensi `(SEQUENCE_LENGTH, 258)`.
- **Video_Augmenter**: Komponen di dalam `Augmentation_Module` yang menerapkan
  augmentation pada frame video mentah (BGR/RGB) sebelum ekstraksi MediaPipe.
- **Augmentation_Config**: File konfigurasi (YAML atau JSON) yang berisi parameter
  augmentation seperti probability, range, dan flag enable/disable per teknik.
- **Training_Pipeline**: Skrip training baru yang menggantikan/memperluas
  `train_kata_bisindo_gru.py`, menggunakan GPU dan augmentation on-the-fly.
- **Data_Generator**: `tf.keras.utils.Sequence` atau `tf.data.Dataset` yang
  menyajikan batch landmark sequence ke model, menerapkan augmentation per batch.
- **Original_Landmark_Cache**: Array NumPy `X.npy` dan `y.npy` di folder
  `processed_bisindo/` berisi landmark hasil ekstraksi MediaPipe Holistic dari
  video original (tanpa augmentation), digunakan sebagai sumber data.
- **GRU_Model**: Arsitektur model neural network yang sama persis dengan
  `train_kata_bisindo_gru.py` existing: GRU 128 → BatchNorm → Dropout 0.3 →
  GRU 64 → BatchNorm → Dropout 0.3 → Dense 64 → Dropout 0.3 → Dense softmax.
- **Holistic_Feature_Vector**: Vektor 258 dimensi per frame =
  pose (33×4=132) + left hand (21×3=63) + right hand (21×3=63).
- **Sequence_Length**: Jumlah frame per sample, fixed 50 frame (sama dengan
  predict_kata.py existing).
- **Validation_Set**: Subset 20% dari dataset original, hanya berisi sample
  tanpa augmentation, dipakai untuk evaluasi akhir dan early stopping.
- **Training_Set**: Subset 80% dari dataset original yang akan menjadi
  basis augmentation on-the-fly saat training.
- **Predict_Kata**: Skrip `predict/predict_kata.py` existing yang melakukan
  real-time prediction dari webcam.
- **Backend_App**: Aplikasi `backend/main.py` yang memuat
  `backend/Models/bisindo_holistic_gru.h5` dan `backend/Models/label_encoder.pkl`.
- **Comparison_Report**: Dokumen ringkasan yang membandingkan metrik model
  baseline (sebelum augmentation) vs model baru (sesudah augmentation).
- **Per_Class_Report**: Tabel berisi precision, recall, F1-score, dan support
  untuk masing-masing dari 50 kelas kata.
- **Confusion_Matrix_Image**: File gambar PNG berukuran cukup besar berisi
  confusion matrix 50×50 yang dapat dibaca per kelas.
- **Training_History_Plot**: File gambar PNG berisi kurva train/validation
  accuracy dan loss per epoch.

## Requirements

### Requirement 1: Augmentation Module Standalone

**User Story:** Sebagai developer, saya ingin modul augmentation yang dapat dipakai
secara standalone, sehingga saya dapat mengetes setiap teknik augmentation di luar
training loop dan menggunakannya kembali untuk eksperimen lain.

#### Acceptance Criteria

1. THE Augmentation_Module SHALL diletakkan di file Python yang terpisah dari
   skrip training, dengan path `Train/augmentation.py`.
2. THE Augmentation_Module SHALL menyediakan kelas atau fungsi `Landmark_Augmenter`
   yang menerima input array NumPy `(SEQUENCE_LENGTH, 258)` dan mengembalikan
   array NumPy dengan shape yang sama.
3. THE Augmentation_Module SHALL menyediakan kelas atau fungsi `Video_Augmenter`
   yang menerima list frame video (NumPy array `(H, W, 3)` uint8) dan
   mengembalikan list frame dengan jumlah frame yang sama atau berbeda
   tergantung augmentation yang diterapkan.
4. THE Augmentation_Module SHALL dapat di-import dengan
   `from augmentation import Landmark_Augmenter, Video_Augmenter` dari
   `Train/train_kata_bisindo_gru_augmented.py`.
5. WHEN Augmentation_Module dijalankan langsung sebagai script
   (`python Train/augmentation.py`), THE Augmentation_Module SHALL menjalankan
   smoke test yang menampilkan shape input dan output untuk setiap teknik
   augmentation yang aktif.
6. THE Augmentation_Module SHALL TIDAK memiliki dependensi pada file training
   atau pada path dataset spesifik, sehingga dapat dipakai ulang.

### Requirement 2: Horizontal Flip Augmentation

**User Story:** Sebagai user model, saya ingin model dapat mengenali isyarat
yang dilakukan oleh tangan kiri maupun tangan kanan, sehingga model tidak bias
terhadap satu sisi.

#### Acceptance Criteria

1. WHEN Landmark_Augmenter menerapkan horizontal flip pada satu sample,
   THE Landmark_Augmenter SHALL mengubah koordinat x dari semua landmark menjadi
   `1.0 - x` (karena MediaPipe normalize koordinat ke [0, 1]).
2. WHEN Landmark_Augmenter menerapkan horizontal flip,
   THE Landmark_Augmenter SHALL menukar block left hand (index 132–194) dengan
   block right hand (index 195–257) di dalam Holistic_Feature_Vector.
3. WHEN Video_Augmenter menerapkan horizontal flip,
   THE Video_Augmenter SHALL melakukan `cv2.flip(frame, 1)` pada setiap frame
   dalam sample.
4. WHILE training berjalan, THE Data_Generator SHALL menerapkan horizontal flip
   dengan probabilitas yang dapat dikonfigurasi via Augmentation_Config,
   dengan nilai default 0.5.
5. THE Augmentation_Module SHALL menerapkan horizontal flip secara konsisten
   pada semua frame dari satu sample (semua frame di-flip atau tidak sama sekali,
   tidak boleh sebagian).

### Requirement 3: Speed Variation Augmentation

**User Story:** Sebagai user model, saya ingin model dapat mengenali isyarat
dengan kecepatan gerakan yang bervariasi, sehingga model robust terhadap user
yang melakukan gerakan lebih cepat atau lebih lambat dari training data.

#### Acceptance Criteria

1. THE Augmentation_Config SHALL menyediakan parameter `speed_factors` berupa
   list float dengan nilai default `[0.8, 0.9, 1.0, 1.1, 1.2]`.
2. WHEN Landmark_Augmenter menerapkan speed variation dengan factor `s`,
   THE Landmark_Augmenter SHALL melakukan resampling temporal pada urutan
   landmark sehingga durasi efektif menjadi `1/s` kali dari original, lalu
   menormalisasi kembali menjadi tepat `SEQUENCE_LENGTH` frame.
3. WHEN sample hasil resampling memiliki frame kurang dari SEQUENCE_LENGTH,
   THE Landmark_Augmenter SHALL melakukan padding dengan frame terakhir
   (sesuai logika `normalize_sequence` existing).
4. WHEN sample hasil resampling memiliki frame lebih banyak dari SEQUENCE_LENGTH,
   THE Landmark_Augmenter SHALL melakukan sub-sampling dengan
   `np.linspace(0, len-1, SEQUENCE_LENGTH).astype(int)` (sesuai logika existing).
5. WHILE training berjalan, THE Data_Generator SHALL memilih satu speed factor
   secara acak dari `speed_factors` per sample per epoch dengan probabilitas
   yang dapat dikonfigurasi (default 0.7 dari sample akan mendapat speed
   variation, sisanya tetap original).
6. THE Augmentation_Module SHALL memastikan output speed variation tetap
   memiliki shape `(SEQUENCE_LENGTH, 258)`.

### Requirement 4: Brightness and Contrast Augmentation

**User Story:** Sebagai user model, saya ingin model dapat mengenali isyarat
pada berbagai kondisi pencahayaan webcam, sehingga model tidak gagal saat
ruangan terlalu terang atau terlalu gelap.

#### Acceptance Criteria

1. THE Augmentation_Config SHALL menyediakan parameter `brightness_range` berupa
   tuple float dengan nilai default `(-0.2, 0.2)`.
2. THE Augmentation_Config SHALL menyediakan parameter `contrast_range` berupa
   tuple float dengan nilai default `(0.8, 1.2)`.
3. WHEN Video_Augmenter menerapkan brightness adjustment dengan factor `b`,
   THE Video_Augmenter SHALL menggeser nilai pixel sebesar `b * 255` dengan
   clipping ke `[0, 255]` dan dtype uint8.
4. WHEN Video_Augmenter menerapkan contrast adjustment dengan factor `c`,
   THE Video_Augmenter SHALL mengalikan nilai pixel terhadap mean dengan
   formula `clip((pixel - 127.5) * c + 127.5, 0, 255)` dengan dtype uint8.
5. WHERE pipeline berjalan dalam mode landmark-level (tanpa video re-extraction),
   THE Training_Pipeline SHALL menonaktifkan brightness/contrast augmentation
   dan mencatat pesan informatif di log bahwa brightness/contrast tidak
   mempengaruhi koordinat landmark MediaPipe.
6. WHERE pipeline berjalan dalam mode video pre-pass dengan re-extraction
   MediaPipe, THE Video_Augmenter SHALL mengaplikasikan brightness dan
   contrast pada semua frame dari satu sample dengan factor yang sama
   (konsisten antar frame).

### Requirement 5: Random Crop and Resize Augmentation

**User Story:** Sebagai user model, saya ingin model dapat mengenali isyarat
saat user berada lebih dekat atau lebih jauh dari kamera, sehingga zoom level
tidak mempengaruhi prediksi.

#### Acceptance Criteria

1. THE Augmentation_Config SHALL menyediakan parameter `scale_range` berupa
   tuple float dengan nilai default `(0.85, 1.15)`.
2. WHEN Landmark_Augmenter menerapkan random crop and resize dengan scale `s`,
   THE Landmark_Augmenter SHALL mengalikan koordinat x dan y dari semua landmark
   dengan `s` dan menambahkan offset translasi acak yang dipilih sehingga
   landmark tetap berada di rentang `[0, 1]` setelah scaling.
3. WHEN Landmark_Augmenter menerapkan random crop and resize, THE Landmark_Augmenter
   SHALL TIDAK mengubah koordinat z dan visibility dari pose landmark.
4. WHEN Video_Augmenter menerapkan random crop and resize,
   THE Video_Augmenter SHALL meng-crop frame ke region acak berukuran
   `s * frame_size` lalu meresize kembali ke ukuran frame asli dengan
   `cv2.resize` interpolation `cv2.INTER_LINEAR`.
5. WHILE training berjalan, THE Data_Generator SHALL menerapkan random crop
   and resize dengan probabilitas yang dapat dikonfigurasi via
   Augmentation_Config dengan nilai default 0.5.
6. IF hasil augmentation menyebabkan koordinat landmark berada di luar
   rentang `[0, 1]`, THEN THE Landmark_Augmenter SHALL melakukan clipping
   ke rentang `[0, 1]`.

### Requirement 6: Gaussian Noise Augmentation

**User Story:** Sebagai user model, saya ingin model robust terhadap noise
deteksi MediaPipe yang muncul karena kualitas webcam yang rendah, sehingga
prediksi tetap stabil.

#### Acceptance Criteria

1. THE Augmentation_Config SHALL menyediakan parameter `landmark_noise_std`
   berupa float dengan nilai default `0.005` (relatif terhadap koordinat
   ternormalisasi `[0, 1]`).
2. WHEN Landmark_Augmenter menerapkan Gaussian noise pada satu sample,
   THE Landmark_Augmenter SHALL menambahkan noise dari distribusi
   `N(0, landmark_noise_std)` ke koordinat x, y, dan z dari semua landmark.
3. THE Landmark_Augmenter SHALL TIDAK menambahkan noise ke field
   visibility dari pose landmark.
4. THE Augmentation_Config SHALL menyediakan parameter `pixel_noise_std`
   berupa float dengan nilai default `5.0` (skala pixel `[0, 255]`).
5. WHEN Video_Augmenter menerapkan Gaussian noise, THE Video_Augmenter SHALL
   menambahkan noise dari distribusi `N(0, pixel_noise_std)` ke setiap pixel
   dengan clipping ke `[0, 255]` dan dtype uint8.
6. WHILE training berjalan, THE Data_Generator SHALL menerapkan Gaussian
   noise dengan probabilitas yang dapat dikonfigurasi via Augmentation_Config
   dengan nilai default 0.5.
7. THE Data_Generator SHALL TIDAK menerapkan Gaussian noise selama fase
   validation maupun inference.
8. IF Data_Generator gagal menerapkan Gaussian noise pada satu sample karena
   exception runtime, THEN THE Data_Generator SHALL meneruskan sample original
   tanpa noise dan mencatat warning ke log, tanpa menghentikan training.

### Requirement 7: Rotation Augmentation

**User Story:** Sebagai user model, saya ingin model dapat mengenali isyarat
saat user sedikit miring di depan kamera, sehingga model robust terhadap
rotasi kepala atau tubuh yang kecil.

#### Acceptance Criteria

1. THE Augmentation_Config SHALL menyediakan parameter `rotation_range_degrees`
   berupa tuple float dengan nilai default `(-10.0, 10.0)`.
2. WHEN Landmark_Augmenter menerapkan rotation dengan sudut `theta` derajat,
   THE Landmark_Augmenter SHALL melakukan rotasi 2D pada koordinat x dan y
   dari semua landmark di sekitar titik pusat `(0.5, 0.5)`.
3. WHEN Landmark_Augmenter menerapkan rotation, THE Landmark_Augmenter SHALL
   TIDAK mengubah koordinat z dan visibility.
4. WHEN Video_Augmenter menerapkan rotation dengan sudut `theta` derajat,
   THE Video_Augmenter SHALL menggunakan
   `cv2.warpAffine` dengan matriks rotasi dari `cv2.getRotationMatrix2D` di
   sekitar pusat frame, dengan mode `cv2.BORDER_REPLICATE`.
5. WHILE training berjalan, THE Data_Generator SHALL menerapkan rotation
   dengan probabilitas yang dapat dikonfigurasi via Augmentation_Config
   dengan nilai default 0.5, dan sudut yang sama untuk semua frame dalam
   satu sample.
6. IF hasil rotation menyebabkan koordinat landmark berada di luar rentang
   `[0, 1]`, THEN THE Landmark_Augmenter SHALL melakukan clipping ke
   rentang `[0, 1]`.

### Requirement 8: Augmentation Configuration File

**User Story:** Sebagai developer, saya ingin parameter augmentation
dipisahkan ke file konfigurasi, sehingga saya dapat melakukan eksperimen
tanpa harus mengubah kode training.

#### Acceptance Criteria

1. THE Augmentation_Config SHALL disimpan di file
   `Train/augmentation_config.yaml` dengan format YAML.
2. THE Augmentation_Config SHALL berisi minimal field berikut:
   `enable_horizontal_flip`, `flip_probability`,
   `enable_speed_variation`, `speed_factors`, `speed_probability`,
   `enable_brightness_contrast`, `brightness_range`, `contrast_range`,
   `enable_random_crop_resize`, `scale_range`, `crop_probability`,
   `enable_gaussian_noise`, `landmark_noise_std`, `pixel_noise_std`,
   `noise_probability`, `enable_rotation`, `rotation_range_degrees`,
   `rotation_probability`, `augmentation_multiplier`, `augmentation_mode`.
3. THE Augmentation_Config SHALL menyediakan field `augmentation_mode`
   dengan nilai yang valid `landmark` atau `video_prepass`, default `landmark`.
4. THE Augmentation_Config SHALL menyediakan field `augmentation_multiplier`
   berupa integer ≥ 1 dengan nilai default 3, yang menentukan berapa kali
   ukuran efektif Training_Set diperbesar oleh augmentation per epoch.
5. WHEN Training_Pipeline membaca Augmentation_Config, THE Training_Pipeline
   SHALL memvalidasi tipe dan rentang setiap field, dan jika tidak valid
   THE Training_Pipeline SHALL menampilkan error yang menjelaskan field
   yang tidak valid lalu menghentikan eksekusi.
6. IF file `augmentation_config.yaml` tidak ditemukan, THEN THE Training_Pipeline
   SHALL menggunakan nilai default yang di-hardcode di kode dan menampilkan
   warning bahwa file konfigurasi tidak ditemukan.

### Requirement 9: GPU-Accelerated Training

**User Story:** Sebagai developer, saya ingin training berjalan di GPU NVIDIA
saya, sehingga waktu training cepat dan saya bisa melakukan eksperimen
dalam waktu kurang dari 2 jam.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL TIDAK menyetel
   `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` (yang ada di skrip training
   existing) sehingga GPU NVIDIA dapat dideteksi oleh TensorFlow.
2. WHEN Training_Pipeline mulai berjalan, THE Training_Pipeline SHALL memanggil
   `tf.config.list_physical_devices("GPU")` dan menampilkan jumlah serta nama
   GPU yang terdeteksi di log.
3. IF tidak ada GPU NVIDIA yang terdeteksi, THEN THE Training_Pipeline SHALL
   menampilkan warning yang jelas dan menanyakan konfirmasi (atau menggunakan
   CLI flag `--allow-cpu`) sebelum melanjutkan training di CPU.
4. WHEN GPU terdeteksi, THE Training_Pipeline SHALL mengaktifkan
   `tf.config.experimental.set_memory_growth(gpu, True)` untuk menghindari
   alokasi seluruh memori GPU sekaligus.
5. THE Training_Pipeline SHALL menyelesaikan 50 epoch training (sebelum
   early stopping) dalam waktu maksimal 120 menit pada GPU NVIDIA dengan
   CUDA support pada dataset 50 kelas × 50 video dengan augmentation
   multiplier 3.
6. THE Training_Pipeline SHALL menggunakan GRU layer dengan parameter yang
   kompatibel dengan implementasi cuDNN GPU TensorFlow (yaitu `reset_after`
   yang sesuai, tanpa `recurrent_dropout` jika diperlukan untuk speed-up),
   selama hasil model tetap dapat diload kembali oleh `predict_kata.py`
   existing tanpa modifikasi.

### Requirement 10: On-the-fly Augmentation During Training

**User Story:** Sebagai developer, saya ingin augmentation dilakukan
secara on-the-fly saat training (tidak menyimpan video atau landmark
augmented ke disk), sehingga saya tidak menghabiskan storage dan dapat
bereksperimen dengan parameter augmentation tanpa pre-processing ulang.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL TIDAK menyimpan video augmented ke disk.
2. THE Training_Pipeline SHALL TIDAK menyimpan array landmark augmented
   ke disk.
3. WHERE `augmentation_mode` bernilai `landmark`, THE Data_Generator SHALL
   memuat Original_Landmark_Cache dari `processed_bisindo/X.npy` dan
   `processed_bisindo/y.npy` lalu menerapkan augmentation pada landmark
   sequence di memori per batch saat training.
4. WHERE `augmentation_mode` bernilai `video_prepass`, THE Training_Pipeline
   SHALL membaca video original dari folder `bisindo-kata-baru/`, menerapkan
   Video_Augmenter di memori, lalu mengekstrak landmark dengan MediaPipe
   Holistic, lalu memasukkan hasil ke training tanpa menyimpan ke disk.
5. THE Data_Generator SHALL memastikan setiap epoch menghasilkan kombinasi
   augmentation yang berbeda untuk sample yang sama (random per epoch).
6. THE Data_Generator SHALL menyajikan jumlah sample efektif per epoch
   sebesar `len(Training_Set) * augmentation_multiplier`.
7. THE Data_Generator SHALL TIDAK menerapkan augmentation pada Validation_Set.

### Requirement 11: Validation Set Without Augmentation

**User Story:** Sebagai developer, saya ingin validation set tetap berisi
data original (tanpa augmentation), sehingga metrik validation merepresentasikan
performa fair model pada data nyata.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL melakukan train/validation split dengan rasio
   80:20, `random_state=42`, dan `stratify` berdasarkan label (sama dengan
   skrip existing).
2. THE Validation_Set SHALL berisi sample landmark dari video original
   (tanpa augmentation).
3. WHEN Data_Generator menyajikan batch untuk validation,
   THE Data_Generator SHALL menonaktifkan semua augmentation (probability
   semua teknik = 0).
4. THE Training_Pipeline SHALL menggunakan Validation_Set yang sama untuk
   `EarlyStopping` dan `ModelCheckpoint` callback.
5. THE Training_Pipeline SHALL melaporkan validation accuracy dan validation
   loss per epoch di log dan menyimpannya di Training_History_Plot.

### Requirement 12: Preserve GRU Architecture

**User Story:** Sebagai developer, saya ingin tetap memakai arsitektur GRU
yang sudah ada, sehingga eksperimen ini terisolasi pada augmentation
dan dapat dibandingkan secara fair dengan baseline.

#### Acceptance Criteria

1. THE GRU_Model SHALL memiliki struktur layer yang sama persis dengan skrip
   existing: `Input(50, 258)` → `GRU(128, return_sequences=True)` →
   `BatchNorm` → `Dropout(0.3)` → `GRU(64)` → `BatchNorm` → `Dropout(0.3)` →
   `Dense(64, relu)` → `Dropout(0.3)` → `Dense(num_classes, softmax)`.
2. THE GRU_Model SHALL menggunakan optimizer `Adam` dengan
   `learning_rate=0.001` (sama dengan baseline).
3. THE GRU_Model SHALL menggunakan loss `sparse_categorical_crossentropy`
   (sama dengan baseline).
4. THE GRU_Model SHALL menggunakan metric `accuracy` (sama dengan baseline).
5. THE Training_Pipeline SHALL menggunakan callback `EarlyStopping(patience=15,
   restore_best_weights=True)`, `ModelCheckpoint(save_best_only=True)`, dan
   `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)`
   (sama dengan baseline).
6. THE Training_Pipeline SHALL menggunakan `batch_size=32` sebagai default,
   yang dapat dikonfigurasi via Augmentation_Config field `batch_size`.
7. WHERE parameter cuDNN-compatible diperlukan untuk speedup di GPU,
   THE Training_Pipeline MAY menyesuaikan `reset_after` dan menghapus
   `recurrent_dropout` PADA syarat hasil model tetap dapat diload oleh
   `predict_kata.py` existing tanpa perubahan kode.

### Requirement 13: Output File Naming Compatibility

**User Story:** Sebagai pengelola backend, saya ingin model baru memiliki
nama file yang sama dengan model lama, sehingga saya tidak perlu mengubah
kode `predict_kata.py` atau `backend/main.py`.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL menyimpan model akhir ke file dengan nama
   `bisindo_holistic_gru.h5` di root direktori workspace
   (sama dengan baseline).
2. THE Training_Pipeline SHALL menyimpan label encoder ke file dengan nama
   `label_encoder.pkl` di root direktori workspace.
3. THE Training_Pipeline SHALL menyimpan class names ke file dengan nama
   `class_names.json` di root direktori workspace.
4. THE Training_Pipeline SHALL menghasilkan file `bisindo_holistic_gru.h5`
   yang dapat diload oleh
   `tensorflow.keras.models.load_model("bisindo_holistic_gru.h5",
   compile=False)` tanpa custom_objects.
5. THE Training_Pipeline SHALL menghasilkan model dengan
   `model.input_shape == (None, 50, 258)`.
6. IF jumlah kelas yang terdeteksi pada dataset bukan tepat 50,
   THEN THE Training_Pipeline SHALL menampilkan error yang menjelaskan
   jumlah kelas yang terdeteksi vs jumlah yang diharapkan dan menghentikan
   eksekusi tanpa menyimpan file model.
7. THE LabelEncoder SHALL berisi 50 class names yang persis sama dengan
   yang dihasilkan oleh skrip training existing pada dataset `bisindo-kata-baru/`.
8. WHEN file model baru di-deploy ke `backend/Models/bisindo_holistic_gru.h5`
   dan `backend/Models/label_encoder.pkl`, THE Backend_App SHALL dapat
   memuat model dan melakukan inference tanpa perubahan kode pada
   `backend/main.py`.

### Requirement 14: Detailed Training Logging

**User Story:** Sebagai developer, saya ingin log training yang detail,
sehingga saya dapat memantau progress, melakukan debugging, dan
mendokumentasikan eksperimen.

#### Acceptance Criteria

1. WHEN Training_Pipeline mulai berjalan, THE Training_Pipeline SHALL mencatat
   ke log: timestamp mulai, versi TensorFlow, daftar GPU yang terdeteksi,
   dan isi Augmentation_Config yang digunakan.
2. WHEN Training_Pipeline memuat dataset, THE Training_Pipeline SHALL mencatat
   ke log: total sample original, jumlah sample train, jumlah sample
   validation, jumlah kelas, dan distribusi sample per kelas.
3. THE Training_Pipeline SHALL mencatat per-epoch ke log: nomor epoch,
   training accuracy, training loss, validation accuracy, validation loss,
   learning rate, dan durasi epoch dalam detik.
4. THE Training_Pipeline SHALL menyimpan log lengkap ke file
   `logs/training_<timestamp>.log` dengan format
   `<timestamp> | <level> | <message>`.
5. WHEN training selesai, THE Training_Pipeline SHALL mencatat ke log:
   total durasi training, best validation accuracy, epoch best validation
   accuracy, dan path output file.
6. THE Training_Pipeline SHALL menyimpan history training (per epoch
   accuracy/loss/lr) ke file `logs/training_history_<timestamp>.json`.

### Requirement 15: Confusion Matrix and Per-Class Report

**User Story:** Sebagai developer, saya ingin laporan confusion matrix dan
per-class accuracy, sehingga saya dapat mengidentifikasi kelas mana yang
masih sulit dikenali model dan menjadi target peningkatan berikutnya.

#### Acceptance Criteria

1. WHEN training selesai, THE Training_Pipeline SHALL menghasilkan
   `Confusion_Matrix_Image` di path `reports/confusion_matrix_<timestamp>.png`
   dengan ukuran minimal 1500×1500 piksel agar 50×50 cell terbaca jelas.
2. THE Confusion_Matrix_Image SHALL menampilkan label kelas pada sumbu x
   dan y, dengan rotasi label x sebesar 45–90 derajat agar tidak overlap.
3. THE Confusion_Matrix_Image SHALL menampilkan nilai count atau persentase
   di setiap cell (annotated heatmap) selama jumlah kelas memungkinkan
   tampilan yang terbaca.
4. WHEN training selesai, THE Training_Pipeline SHALL menghasilkan
   `Per_Class_Report` di path `reports/per_class_report_<timestamp>.csv`
   yang berisi kolom: `class_name`, `precision`, `recall`, `f1_score`,
   `support`.
5. THE Per_Class_Report SHALL berisi tepat satu baris per kelas, dengan
   jumlah baris sama dengan jumlah kelas pada LabelEncoder yang dipakai
   pada run training (50 baris untuk dataset BISINDO standar).
6. THE Training_Pipeline SHALL juga mencetak `classification_report` dari
   sklearn ke stdout dan file log.

### Requirement 16: Training History Visualization

**User Story:** Sebagai developer, saya ingin visualisasi kurva training,
sehingga saya dapat melihat tren overfitting/underfitting dan dampak
augmentation terhadap learning curve.

#### Acceptance Criteria

1. WHEN training selesai dan plot dapat dirender pada resolusi minimal
   1200×800 piksel, THE Training_Pipeline SHALL menghasilkan
   `Training_History_Plot` di path `reports/training_history_<timestamp>.png`.
2. THE Training_History_Plot SHALL berisi dua subplot: subplot pertama
   menampilkan training accuracy dan validation accuracy per epoch, subplot
   kedua menampilkan training loss dan validation loss per epoch.
3. THE Training_History_Plot SHALL memiliki sumbu x = nomor epoch dan
   sumbu y diberi label yang sesuai.
4. THE Training_History_Plot SHALL memiliki legend yang membedakan kurva
   train dan validation pada masing-masing subplot.
5. THE Training_History_Plot SHALL memiliki resolusi minimal 1200×800 piksel.
6. IF resolusi minimal 1200×800 tidak dapat dipenuhi karena keterbatasan
   environment grafik, THEN THE Training_Pipeline SHALL TIDAK menulis file
   plot dan SHALL mencatat error yang menjelaskan penyebabnya ke log,
   sambil tetap menyimpan history training ke file JSON sesuai
   Requirement 14.

### Requirement 17: Comparison Report Baseline vs Augmented

**User Story:** Sebagai developer, saya ingin laporan perbandingan model
baseline dan model dengan augmentation, sehingga saya dapat menilai
efektivitas augmentation secara objektif.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL menyediakan flag CLI `--compare-baseline
   <path_to_baseline_model.h5>` yang ketika diaktifkan akan memuat model
   baseline dan model baru, lalu mengevaluasi keduanya pada Validation_Set
   yang sama.
2. WHEN flag `--compare-baseline` aktif, THE Training_Pipeline SHALL
   menghasilkan `Comparison_Report` di path
   `reports/comparison_<timestamp>.md` berisi tabel dengan kolom:
   `metric`, `baseline_value`, `augmented_value`, `delta`.
3. THE Comparison_Report SHALL minimal mencakup metrik berikut: validation
   accuracy, validation loss, macro precision, macro recall, macro F1-score,
   weighted F1-score.
4. THE Comparison_Report SHALL menyertakan top-5 kelas dengan peningkatan
   F1-score terbesar dan top-5 kelas dengan penurunan F1-score terbesar.
5. WHEN flag `--compare-baseline` tidak aktif, THE Training_Pipeline SHALL
   tetap menghasilkan laporan model baru saja tanpa perbandingan dan
   tanpa error.

### Requirement 18: Class Balance Preservation

**User Story:** Sebagai developer, saya ingin augmentation tidak menyebabkan
imbalance kelas, sehingga model tetap fair antar kelas.

#### Acceptance Criteria

1. THE Data_Generator SHALL menerapkan augmentation_multiplier yang sama
   untuk semua kelas, sehingga distribusi kelas pada Training_Set efektif
   tetap balanced.
2. WHEN Data_Generator menyajikan satu epoch, THE Data_Generator SHALL
   memastikan setiap kelas memiliki tepat
   `samples_per_class_in_train * augmentation_multiplier` sample per epoch.
3. THE Training_Pipeline SHALL mencatat ke log distribusi sample per kelas
   pada Training_Set efektif (setelah augmentation_multiplier).

### Requirement 19: Reproducibility

**User Story:** Sebagai developer, saya ingin eksperimen dapat direproduksi,
sehingga hasil yang dilaporkan dapat diverifikasi.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL menyetel random seed untuk `numpy`, `random`,
   dan `tensorflow` di awal eksekusi, dengan nilai default 42 yang dapat
   dikonfigurasi via Augmentation_Config field `random_seed`.
2. THE Training_Pipeline SHALL menyimpan salinan Augmentation_Config yang
   dipakai ke `reports/run_config_<timestamp>.yaml` agar terikat ke output run.
3. THE Training_Pipeline SHALL mencatat hash atau ukuran file dataset
   (`processed_bisindo/X.npy` dan `processed_bisindo/y.npy`) di log untuk
   memastikan dataset yang dipakai dapat diidentifikasi.

### Requirement 20: Windows Compatibility

**User Story:** Sebagai user yang bekerja di Windows, saya ingin pipeline
ini dapat berjalan di Windows 10/11 dengan TensorFlow GPU, sehingga saya
tidak perlu pindah environment.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL menggunakan path-handling yang kompatibel
   dengan Windows, yaitu menggunakan `os.path.join` atau `pathlib.Path`
   dan TIDAK menggunakan separator path Unix (`/`) yang di-hardcode.
2. THE Training_Pipeline SHALL dapat dijalankan dari command prompt
   Windows dengan perintah
   `python Train\train_kata_bisindo_gru_augmented.py`.
3. THE Training_Pipeline SHALL kompatibel dengan TensorFlow versi yang
   mendukung GPU di Windows (TensorFlow 2.10.x adalah versi terakhir
   dengan GPU support native di Windows; jika versi lebih baru, gunakan
   WSL2 atau dokumentasikan persyaratan).
4. WHEN Training_Pipeline membaca file YAML, THE Training_Pipeline SHALL
   menggunakan encoding `utf-8` secara eksplisit untuk menghindari masalah
   encoding default Windows (`cp1252`).
5. WHEN Training_Pipeline menulis file log atau report,
   THE Training_Pipeline SHALL menggunakan encoding `utf-8` secara eksplisit.

### Requirement 21: Real-Time Prediction Accuracy Improvement

**User Story:** Sebagai user akhir aplikasi, saya ingin prediksi real-time
dari webcam akurat, sehingga saya dapat mempercayai output sistem.

#### Acceptance Criteria

1. THE GRU_Model hasil augmentation SHALL mencapai validation accuracy
   minimal 80% pada Validation_Set 20% dari dataset
   `bisindo-kata-baru/`.
2. WHEN model baru dievaluasi pada test set real-time (rekaman terpisah dari
   training set), THE GRU_Model SHALL mencapai accuracy minimal 75%
   diukur pada minimal 5 sample per kelas yang direkam dengan
   `predict_kata.py` atau prosedur evaluasi setara.
3. THE Training_Pipeline SHALL menyertakan instruksi prosedur evaluasi
   real-time di README atau di log akhir, yang mendeskripsikan cara
   merekam test set real-time dan menghitung accuracy.
4. IF target accuracy 75% real-time TIDAK tercapai pada eksperimen pertama,
   THEN THE Training_Pipeline SHALL memungkinkan eksperimen ulang dengan
   parameter Augmentation_Config yang berbeda tanpa modifikasi kode
   Training_Pipeline.

### Requirement 22: Error Handling and Robustness

**User Story:** Sebagai developer, saya ingin pipeline gagal dengan pesan
error yang jelas, sehingga saya dapat mendiagnosis masalah dengan cepat.

#### Acceptance Criteria

1. IF `processed_bisindo/X.npy` atau `processed_bisindo/y.npy` tidak ada
   dan `augmentation_mode` bernilai `landmark`, THEN THE Training_Pipeline
   SHALL menampilkan error yang menjelaskan bahwa user perlu menjalankan
   ekstraksi landmark terlebih dahulu, lalu menghentikan eksekusi.
2. IF folder `bisindo-kata-baru/` tidak ada dan `augmentation_mode`
   bernilai `video_prepass`, THEN THE Training_Pipeline SHALL menampilkan
   error yang menjelaskan bahwa folder dataset tidak ditemukan, lalu
   menghentikan eksekusi.
3. IF shape array `X` yang dimuat tidak `(N, 50, 258)`, THEN THE
   Training_Pipeline SHALL menampilkan error yang menjelaskan shape yang
   diharapkan vs shape yang ditemukan, lalu menghentikan eksekusi.
4. IF ekstraksi MediaPipe gagal pada sebuah video saat
   `video_prepass`, THEN THE Video_Augmenter SHALL melewati video
   tersebut dan mencatat warning ke log, tanpa menghentikan training.
5. IF GPU OOM (Out of Memory) terjadi saat training, THEN THE
   Training_Pipeline SHALL menyarankan menurunkan `batch_size` di
   Augmentation_Config dan menghentikan eksekusi dengan pesan yang jelas.

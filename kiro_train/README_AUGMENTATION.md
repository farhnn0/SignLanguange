# BISINDO Kata Augmented Training

Pipeline untuk train model GRU dengan augmentasi on-the-fly. Target: akurasi real-time 75%+.

## File-file penting

| File | Fungsi |
|---|---|
| `augmentation_config.yaml` | Konfigurasi semua augmentasi |
| `augmentation.py` | `LandmarkAugmenter` + `VideoAugmenter` |
| `data_generator.py` | `AugmentedSequenceGenerator` (Keras Sequence) |
| `train_kata_bisindo_gru_augmented.py` | **Main training script** |

## Quick Start (Local)

```bash
# Pakai venv yang sudah ada
C:\Users\Farhan\asl_env\Scripts\python.exe kiro_train\train_kata_bisindo_gru_augmented.py
```

Prasyarat: `processed_bisindo/X.npy` dan `processed_bisindo/y.npy` sudah ada (dari `Train/train_kata_bisindo_gru.py` yang sudah kamu jalankan).

**Output:** Semua file (model, logs, reports) disimpan di folder `kiro_train/`

## Quick Start (Kaggle / Colab)

```python
# Cell 1: setup
!pip install -q tensorflow opencv-python mediapipe scikit-learn pyyaml seaborn

# Cell 2: clone / upload kode dan dataset, lalu jalankan
!python Train/train_kata_bisindo_gru_augmented.py \
    --processed-dir /kaggle/input/bisindo-processed \
    --compare-baseline /kaggle/input/baseline-model/bisindo_holistic_gru.h5
```

## Arguments

| Flag | Default | Keterangan |
|---|---|---|
| `--config` | `Train/augmentation_config.yaml` | Path config file |
| `--processed-dir` | `processed_bisindo/` | Folder berisi X.npy, y.npy |
| `--compare-baseline` | (none) | Path .h5 baseline untuk comparison report |

## Output

Setelah training selesai, semua file disimpan di **`kiro_train/`**:

- **`bisindo_holistic_gru_new.h5`** — model baru (hasil augmentasi)
- **`label_encoder_new.pkl`** — label encoder
- **`class_names_new.json`** — daftar nama kelas
- **`logs/training_<timestamp>.log`** — log training lengkap
- **`logs/training_history_<timestamp>.json`** — history + metadata
- **`reports/confusion_matrix_<timestamp>.png`**
- **`reports/per_class_report_<timestamp>.csv`**
- **`reports/training_history_<timestamp>.png`**
- **`reports/run_config_<timestamp>.yaml`**
- **`reports/comparison_<timestamp>.md`** (kalau pakai `--compare-baseline`)

## Tuning Augmentasi

Edit `kiro_train/augmentation_config.yaml`:

```yaml
augmentation_multiplier: 3   # 3x dataset per epoch (besar = lambat tapi lebih banyak variasi)
batch_size: 32               # naikin kalau GPU besar
epochs: 50                   # early stopping akan stop lebih cepat kalau converge

# Mau matikan teknik tertentu?
enable_horizontal_flip: false
```

## Estimasi waktu

| Hardware | Estimasi |
|---|---|
| RTX 3060+ local | ~30-90 menit |
| Kaggle T4 | ~1-2 jam |
| Colab T4 free | ~1-2 jam |
| CPU only | 4+ jam (tidak disarankan) |

## Backward Compatibility

Output `kiro_train/bisindo_holistic_gru_new.h5` punya signature yang sama dengan baseline di `backend/Models/`:
- Input shape: `(None, 50, 258)`
- Output shape: `(None, num_classes)`

**Untuk pakai model baru:** 
1. Backup model lama: `backend/Models/bisindo_holistic_gru.h5` → `bisindo_holistic_gru_old.h5`
2. Copy model baru: `kiro_train/bisindo_holistic_gru_new.h5` → `backend/Models/bisindo_holistic_gru.h5`

Jadi `predict_kata.py` dan `backend/main.py` tetap jalan tanpa perubahan.

## Troubleshooting

**`FileNotFoundError: Landmark cache not found`**
→ Jalankan `Train/train_kata_bisindo_gru.py` dulu untuk extract landmark dari video.

**`OOM (Out of Memory)`**
→ Kurangi `batch_size` di config (32 → 16 → 8).

**Training terlalu lama**
→ Kurangi `augmentation_multiplier` dari 3 ke 2 atau 1.

**Akurasi turun dari baseline**
→ Cek `reports/comparison_<timestamp>.md` untuk lihat kelas mana yang degraded. Mungkin ada augmentasi yang terlalu agresif (misalnya rotasi >10°).

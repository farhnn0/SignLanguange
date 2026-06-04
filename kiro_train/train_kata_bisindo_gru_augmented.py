"""
BISINDO Kata Model Training with Augmentation
==============================================

Training script untuk model GRU dengan augmentasi on-the-fly.
Target: meningkatkan akurasi prediksi real-time dari ~50% ke 75%+.

Usage:
    python Train/train_kata_bisindo_gru_augmented.py
    python Train/train_kata_bisindo_gru_augmented.py --compare-baseline bisindo_holistic_gru.h5

Output (kiro_train/ folder):
    - bisindo_holistic_gru_new.h5   (model)
    - label_encoder_new.pkl          (label encoder)
    - class_names_new.json           (class names)
    - logs/training_*.log        (training logs)
    - reports/*.png, *.csv, *.md (evaluation reports)
"""

import os
import sys
import json
import pickle
import random
import logging
import argparse
import datetime
from pathlib import Path

import numpy as np
import yaml

# Path setup - works for both local and Kaggle/Colab
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Local modules
from augmentation import LandmarkAugmenter
from data_generator import AugmentedSequenceGenerator


# =============================
# CONFIG & CONSTANTS
# =============================

DEFAULT_CONFIG_PATH = SCRIPT_DIR / "augmentation_config.yaml"
DEFAULT_PROCESSED_DIR = WORKSPACE_ROOT / "processed_bisindo"

# Output paths - all in kiro_train folder
MODEL_PATH = SCRIPT_DIR / "bisindo_holistic_gru_new.h5"
LABEL_ENCODER_PATH = SCRIPT_DIR / "label_encoder_new.pkl"
CLASS_NAMES_PATH = SCRIPT_DIR / "class_names_new.json"

LOGS_DIR = SCRIPT_DIR / "logs"
REPORTS_DIR = SCRIPT_DIR / "reports"

DEFAULT_CONFIG = {
    "random_seed": 42,
    "batch_size": 32,
    "epochs": 50,
    "augmentation_multiplier": 3,
    "augmentation_mode": "landmark",
    "enable_horizontal_flip": True,
    "flip_probability": 0.5,
    "enable_speed_variation": True,
    "speed_factors": [0.8, 0.9, 1.0, 1.1, 1.2],
    "speed_probability": 0.7,
    "enable_brightness_contrast": True,
    "brightness_range": [-0.2, 0.2],
    "contrast_range": [0.8, 1.2],
    "brightness_contrast_probability": 0.5,
    "enable_random_crop_resize": True,
    "scale_range": [0.85, 1.15],
    "crop_probability": 0.5,
    "enable_gaussian_noise": True,
    "landmark_noise_std": 0.005,
    "pixel_noise_std": 5.0,
    "noise_probability": 0.5,
    "enable_rotation": True,
    "rotation_range_degrees": [-10.0, 10.0],
    "rotation_probability": 0.5,
}


# =============================
# LOGGING SETUP
# =============================

def setup_logger(timestamp: str) -> logging.Logger:
    """Setup logger with file + console output."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"training_{timestamp}.log"

    logger = logging.getLogger("kata_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


# =============================
# CONFIG LOADING
# =============================

def load_config(config_path: Path, logger: logging.Logger) -> dict:
    """Load YAML config or fall back to defaults."""
    if not config_path.exists():
        logger.warning(f"Config not found at {config_path}, using defaults")
        return DEFAULT_CONFIG.copy()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Fill in any missing keys with defaults
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)

    logger.info(f"Loaded config from {config_path}")
    return cfg


# =============================
# REPRODUCIBILITY & GPU
# =============================

def set_random_seeds(seed: int = 42) -> None:
    """Set seeds for numpy, random, tensorflow."""
    import tensorflow as tf
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu(logger: logging.Logger) -> list:
    """Detect GPU and enable memory growth."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"Detected {len(gpus)} GPU(s):")
        for g in gpus:
            logger.info(f"  - {g.name}")
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError as e:
                logger.warning(f"Memory growth failed: {e}")
    else:
        logger.warning("No GPU detected. Training will run on CPU (slow).")
    return gpus


# =============================
# DATA LOADING
# =============================

def load_landmark_cache(processed_dir: Path, logger: logging.Logger):
    """Load X.npy and y.npy."""
    x_path = processed_dir / "X.npy"
    y_path = processed_dir / "y.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Landmark cache not found at {processed_dir}.\n"
            f"Expected: X.npy and y.npy.\n"
            f"Run baseline extraction first (Train/train_kata_bisindo_gru.py)."
        )

    X = np.load(x_path)
    y = np.load(y_path, allow_pickle=True)

    logger.info(f"Loaded X: {X.shape} {X.dtype} ({x_path.stat().st_size/1e6:.1f} MB)")
    logger.info(f"Loaded y: {y.shape} {y.dtype}")

    return X, y


# =============================
# MODEL BUILDING
# =============================

def build_gru_model(sequence_length: int, num_features: int, num_classes: int,
                    learning_rate: float = 0.001):
    """Build GRU model matching baseline architecture."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        GRU(128, return_sequences=True, reset_after=False),
        BatchNormalization(),
        Dropout(0.3),
        GRU(64, reset_after=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_callbacks(model_path: Path, logger: logging.Logger):
    """EarlyStopping + ModelCheckpoint + ReduceLROnPlateau."""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    return [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]


# =============================
# EVALUATION & REPORTING
# =============================

def save_confusion_matrix(y_true, y_pred, class_names, output_path: Path, logger: logging.Logger):
    """Save confusion matrix PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        try:
            import seaborn as sns
            has_sns = True
        except ImportError:
            has_sns = False

        cm = confusion_matrix(y_true, y_pred)
        n = len(class_names)
        size = max(15, n * 0.4)
        fig, ax = plt.subplots(figsize=(size, size))

        if has_sns:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
        else:
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=90)
            ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6)

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title("Confusion Matrix - BISINDO Kata Augmented")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix: {e}")


def save_per_class_report(y_true, y_pred, class_names, output_path: Path, logger: logging.Logger):
    """Save per-class metrics CSV."""
    try:
        from sklearn.metrics import classification_report
        import csv

        report_dict = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_name", "precision", "recall", "f1_score", "support"])
            for cls in class_names:
                if cls in report_dict:
                    r = report_dict[cls]
                    writer.writerow([cls, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                                     f"{r['f1-score']:.4f}", int(r["support"])])

        logger.info(f"Per-class report saved: {output_path}")

        # Also print to log
        text_report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        logger.info("Classification report:\n" + text_report)
    except Exception as e:
        logger.error(f"Failed to save per-class report: {e}")


def save_history_plot(history_dict, output_path: Path, logger: logging.Logger):
    """Plot training curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history_dict["accuracy"], label="train")
        axes[0].plot(history_dict["val_accuracy"], label="val")
        axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(history_dict["loss"], label="train")
        axes[1].plot(history_dict["val_loss"], label="val")
        axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"History plot saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save history plot: {e}")


def save_history_json(history_dict, metadata: dict, output_path: Path, logger: logging.Logger):
    """Save training history JSON."""
    try:
        out = {
            "metadata": metadata,
            "history": {k: [float(v) for v in vals] for k, vals in history_dict.items()},
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        logger.info(f"History JSON saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save history JSON: {e}")


def compare_with_baseline(baseline_path: Path, new_model, X_val, y_val, class_names,
                          output_path: Path, logger: logging.Logger):
    """Compare baseline vs new model."""
    try:
        import tensorflow as tf
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support, f1_score, classification_report
        )

        if not baseline_path.exists():
            logger.warning(f"Baseline not found at {baseline_path}, skipping comparison")
            return

        logger.info(f"Loading baseline: {baseline_path}")
        baseline = tf.keras.models.load_model(str(baseline_path))

        # Evaluate
        b_loss, b_acc = baseline.evaluate(X_val, y_val, verbose=0)
        n_loss, n_acc = new_model.evaluate(X_val, y_val, verbose=0)

        b_pred = np.argmax(baseline.predict(X_val, verbose=0), axis=1)
        n_pred = np.argmax(new_model.predict(X_val, verbose=0), axis=1)

        b_p, b_r, b_f1m, _ = precision_recall_fscore_support(y_val, b_pred, average="macro", zero_division=0)
        n_p, n_r, n_f1m, _ = precision_recall_fscore_support(y_val, n_pred, average="macro", zero_division=0)
        b_f1w = f1_score(y_val, b_pred, average="weighted", zero_division=0)
        n_f1w = f1_score(y_val, n_pred, average="weighted", zero_division=0)

        # Per-class F1 deltas
        b_rep = classification_report(y_val, b_pred, target_names=class_names, output_dict=True, zero_division=0)
        n_rep = classification_report(y_val, n_pred, target_names=class_names, output_dict=True, zero_division=0)

        deltas = []
        for cls in class_names:
            if cls in b_rep and cls in n_rep:
                deltas.append((cls, n_rep[cls]["f1-score"] - b_rep[cls]["f1-score"]))
        deltas.sort(key=lambda x: x[1], reverse=True)
        top_improve = deltas[:5]
        top_degrade = deltas[-5:][::-1]

        # Markdown
        lines = [
            "# Baseline vs Augmented Model Comparison\n",
            f"Generated: {datetime.datetime.now().isoformat()}\n",
            f"Baseline: `{baseline_path}`\n",
            "## Metrics\n",
            "| Metric | Baseline | Augmented | Delta |",
            "|---|---|---|---|",
            f"| Val Accuracy | {b_acc:.4f} | {n_acc:.4f} | {n_acc-b_acc:+.4f} |",
            f"| Val Loss | {b_loss:.4f} | {n_loss:.4f} | {n_loss-b_loss:+.4f} |",
            f"| Macro Precision | {b_p:.4f} | {n_p:.4f} | {n_p-b_p:+.4f} |",
            f"| Macro Recall | {b_r:.4f} | {n_r:.4f} | {n_r-b_r:+.4f} |",
            f"| Macro F1 | {b_f1m:.4f} | {n_f1m:.4f} | {n_f1m-b_f1m:+.4f} |",
            f"| Weighted F1 | {b_f1w:.4f} | {n_f1w:.4f} | {n_f1w-b_f1w:+.4f} |",
            "\n## Top 5 Improvements (F1)\n",
            "| Class | Delta |", "|---|---|",
        ] + [f"| {c} | {d:+.4f} |" for c, d in top_improve] + [
            "\n## Top 5 Degradations (F1)\n",
            "| Class | Delta |", "|---|---|",
        ] + [f"| {c} | {d:+.4f} |" for c, d in top_degrade]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Comparison report saved: {output_path}")
    except Exception as e:
        logger.error(f"Comparison failed: {e}")


# =============================
# MAIN TRAINING
# =============================

def main(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(timestamp)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BISINDO Kata Augmented Training")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {timestamp}")

    # Step 1: Config
    config = load_config(Path(args.config), logger)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Step 2: Reproducibility
    set_random_seeds(config["random_seed"])
    logger.info(f"Random seeds set to {config['random_seed']}")

    # Step 3: TensorFlow & GPU
    import tensorflow as tf
    logger.info(f"TensorFlow: {tf.__version__}")
    setup_gpu(logger)

    # Step 4: Data
    processed_dir = Path(args.processed_dir)
    X, y = load_landmark_cache(processed_dir, logger)

    # Step 5: Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)
    logger.info(f"Classes ({len(class_names)}): {class_names}")

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved: {LABEL_ENCODER_PATH}, {CLASS_NAMES_PATH}")

    # Step 6: Train/val split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Step 7: Augmenter + generators
    augmenter = LandmarkAugmenter(config)
    train_gen = AugmentedSequenceGenerator(
        X_train, y_train, augmenter,
        batch_size=config["batch_size"],
        augmentation_multiplier=config["augmentation_multiplier"],
        shuffle=True, is_validation=False,
    )
    val_gen = AugmentedSequenceGenerator(
        X_val, y_val, augmenter,
        batch_size=config["batch_size"],
        shuffle=False, is_validation=True,
    )
    logger.info(f"Train batches: {len(train_gen)}, Val batches: {len(val_gen)}")

    # Step 8: Model
    seq_len, num_feat = X.shape[1], X.shape[2]
    num_classes = len(class_names)
    model = build_gru_model(seq_len, num_feat, num_classes)
    model.summary(print_fn=logger.info)

    # Step 9: Train
    callbacks = build_callbacks(MODEL_PATH, logger)
    logger.info(f"Starting training: {config['epochs']} epochs, batch={config['batch_size']}")
    start_time = datetime.datetime.now()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    duration = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {duration:.1f}s ({duration/60:.1f} min)")

    # Step 10: Final save (in case ModelCheckpoint didn't catch best)
    model.save(str(MODEL_PATH))
    logger.info(f"Model saved: {MODEL_PATH}")

    # Step 11: Evaluation
    logger.info("Evaluating on validation set...")
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    logger.info(f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.4f}")

    # Predictions for reports
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_val[:len(y_pred)]  # match length

    # Step 12: Reports
    save_confusion_matrix(y_true, y_pred, class_names,
                          REPORTS_DIR / f"confusion_matrix_{timestamp}.png", logger)
    save_per_class_report(y_true, y_pred, class_names,
                          REPORTS_DIR / f"per_class_report_{timestamp}.csv", logger)
    save_history_plot(history.history,
                      REPORTS_DIR / f"training_history_{timestamp}.png", logger)

    # Best epoch
    best_epoch = int(np.argmax(history.history.get("val_accuracy", [0]))) + 1
    best_val_acc = float(np.max(history.history.get("val_accuracy", [0])))

    metadata = {
        "timestamp": timestamp,
        "training_duration_seconds": duration,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": float(val_acc),
        "final_val_loss": float(val_loss),
        "num_classes": num_classes,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "config": config,
    }
    save_history_json(history.history, metadata,
                      LOGS_DIR / f"training_history_{timestamp}.json", logger)

    # Save run config snapshot
    with open(REPORTS_DIR / f"run_config_{timestamp}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, allow_unicode=True, sort_keys=False)

    # Step 13: Optional baseline comparison
    if args.compare_baseline:
        # Use raw arrays for fair comparison (no augmentation)
        compare_with_baseline(
            Path(args.compare_baseline), model, X_val, y_val, class_names,
            REPORTS_DIR / f"comparison_{timestamp}.md", logger,
        )

    logger.info("=" * 60)
    logger.info(f"DONE! Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Reports: {REPORTS_DIR}")
    logger.info("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="BISINDO Kata Augmented Training")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                   help="Path to augmentation_config.yaml")
    p.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR),
                   help="Path to processed_bisindo/ (X.npy, y.npy)")
    p.add_argument("--compare-baseline", default=None,
                   help="Path to baseline .h5 model for comparison")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

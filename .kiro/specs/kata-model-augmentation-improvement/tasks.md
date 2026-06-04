# Implementation Plan: Kata Model Augmentation Improvement

## Overview

This implementation plan converts the BISINDO kata model augmentation feature design into actionable coding tasks. The goal is to improve real-time prediction accuracy from ~50% to 75%+ by implementing a comprehensive video data augmentation pipeline for the GRU-based gesture recognition model, while maintaining backward compatibility with existing inference scripts.

## Tasks

- [x] 1. Create augmentation configuration file and validation utilities
  - [x] 1.1 Create `Train/augmentation_config.yaml` with all augmentation parameters
    - Define default configuration with all 7 augmentation techniques
    - Include parameters: flip, speed, brightness/contrast, crop/resize, noise, rotation
    - Add global settings: random_seed, batch_size, epochs, augmentation_multiplier, augmentation_mode
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  
  - [x] 1.2 Implement configuration validation function in Python
    - Create `validate_config()` function to check types and ranges for all config fields
    - Validate rotation_range_degrees is tuple, speed_factors is list, probabilities in [0,1]
    - Handle missing config file by using hardcoded defaults with warning
    - _Requirements: 8.5, 8.6_

- [x] 2. Implement standalone augmentation module
  - [x] 2.1 Create `Train/augmentation.py` with LandmarkAugmenter class
    - Implement `__init__()` to load configuration dictionary
    - Implement `horizontal_flip()` for landmark sequences (flip x coords, swap hands)
    - Implement `speed_variation()` with temporal resampling and padding/subsampling
    - Implement `random_crop_resize()` with scaling and translation of coordinates
    - Implement `gaussian_noise()` for landmark coordinates (skip visibility field)
    - Implement `rotation()` with 2D rotation matrix around center point (0.5, 0.5)
    - Implement `augment()` orchestrator that applies enabled augmentations based on probabilities
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 3.3, 5.1, 5.2, 5.3, 6.1, 6.2, 7.1, 7.2_
  
  - [ ]* 2.2 Add smoke test to augmentation.py for standalone execution
    - When run as `python Train/augmentation.py`, test each augmentation technique
    - Display input/output shapes for each technique with sample data
    - _Requirements: 1.5_
  
  - [x] 2.3 Create VideoAugmenter class in `Train/augmentation.py`
    - Implement `horizontal_flip()` using cv2.flip for video frames
    - Implement `brightness_contrast()` with pixel value adjustments
    - Implement `random_crop_resize()` with cv2 crop and resize operations
    - Implement `gaussian_noise()` for pixel-level noise with clipping
    - Implement `rotation()` using cv2.warpAffine
    - Implement `augment()` orchestrator for video frames
    - _Requirements: 1.3, 2.3, 4.1, 4.2, 4.3, 4.4, 5.4, 6.4, 6.5, 7.4_

- [x] 3. Implement data generator with on-the-fly augmentation
  - [x] 3.1 Create `Train/data_generator.py` with AugmentedSequenceGenerator class
    - Implement `__init__()` to accept X, y, augmenter, batch_size, augmentation_multiplier, is_validation
    - Implement `__len__()` to return number of batches considering augmentation_multiplier
    - Implement `__getitem__()` to generate augmented batches on-the-fly
    - Implement `on_epoch_end()` to shuffle indexes after each epoch
    - Ensure validation mode disables all augmentations
    - _Requirements: 10.3, 10.5, 10.6, 10.7, 11.3, 18.1, 18.2_
  
  - [ ]* 3.2 Write unit tests for data generator batch shapes and augmentation toggle
    - Test batch shape is correct (batch_size, 50, 258)
    - Test validation generator disables augmentation
    - Test effective dataset size matches augmentation_multiplier
    - _Requirements: 10.3, 10.6, 11.3_

- [x] 4. Implement GPU setup and training utilities
  - [x] 4.1 Create GPU detection and configuration functions
    - Implement `setup_gpu()` to detect NVIDIA GPUs using tf.config.list_physical_devices
    - Enable memory growth with tf.config.experimental.set_memory_growth
    - Display warning and prompt confirmation if no GPU detected
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [x] 4.2 Create random seed setting function
    - Implement `set_random_seeds()` to set seeds for numpy, random, tensorflow
    - Use configurable seed with default 42
    - _Requirements: 19.1_
  
  - [x] 4.3 Create data loading utilities
    - Implement `load_landmark_cache()` to load X.npy and y.npy from processed_bisindo/
    - Add hash/size logging for dataset identification
    - Handle FileNotFoundError with informative message
    - _Requirements: 19.3_

- [x] 5. Implement GRU model architecture
  - [x] 5.1 Create `build_gru_model()` function with exact baseline architecture
    - Input layer: (50, 258)
    - GRU(128, return_sequences=True) → BatchNorm → Dropout(0.3)
    - GRU(64) → BatchNorm → Dropout(0.3)
    - Dense(64, relu) → Dropout(0.3)
    - Dense(num_classes, softmax)
    - Compile with Adam(lr=0.001), sparse_categorical_crossentropy, accuracy metric
    - _Requirements: 12.1, 12.2, 12.3, 12.4_
  
  - [x] 5.2 Create training callbacks
    - Implement EarlyStopping(patience=15, restore_best_weights=True)
    - Implement ModelCheckpoint(save_best_only=True)
    - Implement ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    - _Requirements: 12.5_

- [x] 6. Implement main training pipeline script
  - [x] 6.1 Create `Train/train_kata_bisindo_gru_augmented.py` with main() function
    - Load and validate augmentation config
    - Set random seeds for reproducibility
    - Detect and configure GPU
    - Load cached landmarks from processed_bisindo/
    - Encode labels with LabelEncoder and save to label_encoder.pkl
    - Perform train/validation split (80:20, stratified, random_state=42)
    - Initialize LandmarkAugmenter with config
    - Create train and validation data generators
    - Build GRU model
    - Train model with callbacks
    - _Requirements: 9.1, 10.3, 11.1, 11.2, 12.6, 13.5, 19.1, 19.3_
  
  - [x] 6.2 Implement model and artifact saving
    - Save best model to `bisindo_holistic_gru.h5` in workspace root
    - Save label_encoder to `label_encoder.pkl` in workspace root
    - Save class_names to `class_names.json` in workspace root
    - Validate model can be loaded with tf.keras.models.load_model without custom_objects
    - Validate output has 50 classes matching dataset
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.6, 13.7_

- [x] 7. Implement detailed logging system
  - [x] 7.1 Create logging utilities with file and console output
    - Set up logging to `logs/training_<timestamp>.log` with format `<timestamp> | <level> | <message>`
    - Log startup info: timestamp, TensorFlow version, detected GPUs, config contents
    - Log dataset info: total samples, train/val split, num classes, per-class distribution
    - Log per-epoch info: epoch number, train/val accuracy/loss, learning rate, duration
    - Log completion info: total duration, best val accuracy, best epoch, output paths
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_
  
  - [x] 7.2 Save training history to JSON
    - Implement function to save history dict to `logs/training_history_<timestamp>.json`
    - Include metadata: timestamp, config_hash, dataset_info, training_duration, best_epoch
    - _Requirements: 14.6, 19.2_

- [x] 8. Checkpoint - Ensure core training pipeline runs successfully
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement evaluation and reporting utilities
  - [x] 9.1 Create confusion matrix generation
    - Implement function to generate confusion matrix on validation set
    - Save as PNG to `reports/confusion_matrix_<timestamp>.png` (min 1500×1500 pixels)
    - Display class labels on both axes with rotated x-axis labels (45-90 degrees)
    - Annotate heatmap cells with count or percentage values
    - _Requirements: 15.1, 15.2, 15.3_
  
  - [x] 9.2 Create per-class report generation
    - Implement function to generate classification report using sklearn
    - Save to `reports/per_class_report_<timestamp>.csv` with columns: class_name, precision, recall, f1_score, support
    - Print classification_report to stdout and log file
    - _Requirements: 15.4, 15.5, 15.6_
  
  - [x] 9.3 Create training history visualization
    - Implement function to plot training curves with matplotlib
    - Create two subplots: accuracy (train/val) and loss (train/val)
    - Save to `reports/training_history_<timestamp>.png` (min 1200×800 pixels)
    - Include legend, axis labels, and proper formatting
    - Handle gracefully if graphics environment unavailable (log error, skip plot)
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_

- [x] 10. Implement baseline comparison feature
  - [x] 10.1 Add CLI flag `--compare-baseline` with path parameter
    - Parse command-line arguments using argparse
    - Load baseline model from provided path
    - Evaluate both baseline and new model on same validation set
    - _Requirements: 17.1_
  
  - [x] 10.2 Create comparison report generation
    - Calculate metrics: val accuracy, val loss, macro precision/recall/F1, weighted F1
    - Generate markdown report at `reports/comparison_<timestamp>.md`
    - Include table with columns: metric, baseline_value, augmented_value, delta
    - List top-5 classes with largest F1-score improvements and degradations
    - _Requirements: 17.2, 17.3, 17.4, 17.5_

- [x] 11. Implement error handling and recovery
  - [x] 11.1 Add error handling for common failure scenarios
    - Handle missing config file (use defaults with warning)
    - Handle no GPU detected (prompt user confirmation)
    - Handle GPU OOM (reduce batch size and retry)
    - Handle missing landmark cache (display clear error message)
    - Handle class count mismatch (halt with descriptive error)
    - Handle augmentation runtime errors (log warning, use original sample)
    - Handle model divergence (halt with suggestion to reduce learning rate)
    - Handle file I/O errors (try alternative filenames, save critical files first)
    - _Requirements: Error Handling section_
  
  - [ ]* 11.2 Write integration tests for error handling
    - Test config validation with invalid types
    - Test GPU fallback behavior
    - Test handling of missing files
    - _Requirements: Error Handling section_

- [x] 12. Final integration and documentation
  - [x] 12.1 Create README or documentation for augmentation pipeline
    - Document how to run training script
    - Explain all config parameters in augmentation_config.yaml
    - Provide examples of different augmentation modes
    - Document output files and their purposes
    - _Requirements: 1.6, 8.1_
  
  - [x] 12.2 Verify backward compatibility with existing inference scripts
    - Test that predict_kata.py can load new model without modifications
    - Test that backend/main.py can load new model without modifications
    - Verify input shape (None, 50, 258) and output shape (None, 50)
    - _Requirements: 13.4, 13.5, 13.8_
  
  - [x] 12.3 Create experiment tracking and results summary
    - Save run configuration to `reports/run_config_<timestamp>.yaml`
    - Document baseline vs augmented model comparison
    - Include training duration, final accuracy, and key insights
    - _Requirements: 17.1, 19.2_

- [x] 13. Final checkpoint - Ensure all tests pass and model meets performance targets
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements from the requirements document for traceability
- The implementation uses Python with TensorFlow, OpenCV, MediaPipe, NumPy, and scikit-learn
- All augmentations are applied on-the-fly during training (no disk writes)
- The GRU architecture is preserved exactly from the baseline for fair comparison
- Output files maintain the same names as baseline for backward compatibility
- Checkpoints ensure incremental validation of the training pipeline
- Target performance: real-time prediction accuracy ≥75%, training time ≤2 hours on NVIDIA GPU

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "4.2", "4.3"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.1", "5.1"] },
    { "id": 3, "tasks": ["3.2", "4.1", "5.2"] },
    { "id": 4, "tasks": ["6.1", "7.1"] },
    { "id": 5, "tasks": ["6.2", "7.2"] },
    { "id": 6, "tasks": ["9.1", "9.2", "9.3"] },
    { "id": 7, "tasks": ["10.1", "11.1"] },
    { "id": 8, "tasks": ["10.2", "11.2"] },
    { "id": 9, "tasks": ["12.1", "12.2", "12.3"] }
  ]
}
```

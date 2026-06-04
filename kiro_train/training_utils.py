"""
Training utilities for augmented GRU model.

This module provides essential utility functions for the training pipeline:
- Random seed setting for reproducibility
- GPU detection and configuration
- Data loading utilities
"""

import os
import numpy as np
import random
import tensorflow as tf
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy, random, and tensorflow.
    
    This function ensures deterministic behavior across different random number
    generators used in the training pipeline. When set to the same value, the
    same random sequence will be generated across multiple runs.
    
    Args:
        seed: Random seed value (default: 42). Must be a non-negative integer.
              Common convention uses 42 for reproducible experiments.
    
    Returns:
        None
    
    Raises:
        ValueError: If seed is not an integer or is negative.
    
    Requirements:
        - Requirement 19.1: Sets random seed for numpy, random, and tensorflow
        - Requirement 19.1: Configurable seed with default 42
    
    Examples:
        >>> set_random_seeds(42)
        >>> # All subsequent random operations will be deterministic
        
        >>> set_random_seeds(seed=123)
        >>> # Using custom seed for different experiment
    
    Notes:
        - Call this function EARLY in the training script, before any random
          operations or model initialization
        - Setting the same seed will NOT guarantee identical results across
          different TensorFlow versions or GPU drivers
        - For full reproducibility on GPU, also set TF_DETERMINISTIC_OPS=1
    """
    if not isinstance(seed, int):
        raise ValueError(f"Seed must be an integer, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")
    
    # Set seed for numpy
    np.random.seed(seed)
    logger.info(f"Set numpy random seed to {seed}")
    
    # Set seed for Python's random module
    random.seed(seed)
    logger.info(f"Set Python random seed to {seed}")
    
    # Set seed for TensorFlow
    tf.random.set_seed(seed)
    logger.info(f"Set TensorFlow random seed to {seed}")


def setup_gpu(allow_cpu: bool = False) -> list:
    """
    Detect and configure NVIDIA GPU for TensorFlow training.
    
    This function:
    1. Lists available physical GPU devices
    2. Enables memory growth to avoid allocating entire GPU memory upfront
    3. Prompts for confirmation if no GPU detected (unless allow_cpu=True)
    4. Returns information about detected GPUs
    
    Requirements:
        - Requirement 9.2: Calls tf.config.list_physical_devices("GPU")
        - Requirement 9.3: Prompts confirmation if no GPU detected
        - Requirement 9.4: Activates tf.config.experimental.set_memory_growth
    
    Args:
        allow_cpu: If True, skip user confirmation when no GPU detected.
                   Useful for CLI flag --allow-cpu (default: False)
    
    Returns:
        List of GPU devices detected. Empty list if no GPUs found.
    
    Raises:
        SystemExit: If no GPU detected and user declines to continue on CPU
    
    Examples:
        >>> gpus = setup_gpu()
        >>> if gpus:
        ...     print(f"Found {len(gpus)} GPU(s)")
        ... else:
        ...     print("Training on CPU")
        
        >>> # Skip confirmation with CLI flag
        >>> gpus = setup_gpu(allow_cpu=True)
    
    Notes:
        - Call this function after setting random seeds but before building model
        - Memory growth should be set before any GPU memory allocation
        - Must be called after tf.random.set_seed() for deterministic GPU ops
        - Training on CPU may take >4 hours instead of <2 hours on GPU
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
        
        if len(gpus) > 0:
            logger.info(f"Detected {len(gpus)} GPU device(s):")
            for gpu in gpus:
                logger.info(f"  - {gpu}")
                # Enable memory growth to avoid OOM
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"    Memory growth enabled for {gpu.name}")
                except RuntimeError as e:
                    logger.warning(f"Could not set memory growth for {gpu.name}: {e}")
        else:
            # No GPU detected - handle according to Requirement 9.3
            logger.warning("=" * 70)
            logger.warning("WARNING: No NVIDIA GPU detected!")
            logger.warning("Training will run on CPU and may take >4 hours instead of <2 hours.")
            logger.warning("=" * 70)
            
            cpus = tf.config.list_physical_devices("CPU")
            if cpus:
                logger.info(f"CPU device(s) available: {cpus}")
            
            # Prompt for confirmation unless allow_cpu flag is set
            if not allow_cpu:
                logger.info("")
                logger.info("Continue training on CPU? [y/N]: ")
                try:
                    response = input().strip().lower()
                    if response not in ('y', 'yes'):
                        logger.info("Training aborted by user.")
                        raise SystemExit("No GPU detected and user declined to continue on CPU.")
                    else:
                        logger.info("User confirmed: continuing training on CPU.")
                except (KeyboardInterrupt, EOFError):
                    logger.info("\nTraining aborted by user.")
                    raise SystemExit("No GPU detected and user declined to continue on CPU.")
            else:
                logger.info("--allow-cpu flag set: continuing training on CPU without confirmation.")
        
        return gpus
    
    except SystemExit:
        # Re-raise SystemExit from user declining to continue
        raise
    except Exception as e:
        logger.error(f"Error detecting GPU devices: {e}")
        return []


def load_landmark_cache(cache_dir: str = "processed_bisindo") -> tuple:
    """
    Load cached landmark sequences from disk.
    
    This function loads pre-extracted MediaPipe Holistic landmarks from the
    processed_bisindo directory. The cache should contain X.npy (landmark
    sequences) and y.npy (labels).
    
    Args:
        cache_dir: Path to the cached landmark directory (default: "processed_bisindo")
    
    Returns:
        Tuple of (X, y) where:
        - X: numpy array of shape (N, 50, 258) - landmark sequences
        - y: numpy array of shape (N,) - integer class labels
    
    Raises:
        FileNotFoundError: If X.npy or y.npy not found in cache_dir
        ValueError: If loaded arrays have unexpected shapes
    
    Requirements:
        - Requirement 19.3: Logs hash or size of dataset for identification
    
    Examples:
        >>> X, y = load_landmark_cache("processed_bisindo")
        >>> print(f"Loaded {X.shape[0]} samples with {X.shape[1]} frames each")
    
    Notes:
        - Assumes landmarks were extracted using MediaPipe Holistic
        - Expects landmark dimension of 258 (33 pose × 4 + 21 left hand × 3 + 21 right hand × 3)
        - Should be called after verifying cache_dir exists
    """
    cache_path = Path(cache_dir)
    X_path = cache_path / "X.npy"
    y_path = cache_path / "y.npy"
    
    if not X_path.exists():
        raise FileNotFoundError(f"Landmark cache not found: {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Label cache not found: {y_path}")
    
    logger.info(f"Loading landmark cache from {cache_dir}/")
    
    # Load arrays
    X = np.load(X_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    
    # Validate shapes
    if X.ndim != 3 or X.shape[1] != 50 or X.shape[2] != 258:
        raise ValueError(
            f"Unexpected X shape: {X.shape}. Expected (N, 50, 258)"
        )
    if y.ndim != 1:
        raise ValueError(
            f"Unexpected y shape: {y.shape}. Expected (N,)"
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have mismatched lengths: {X.shape[0]} vs {y.shape[0]}"
        )
    
    # Log dataset info
    X_size_mb = X.nbytes / (1024 * 1024)
    y_size_kb = y.nbytes / 1024
    logger.info(f"Loaded X: shape={X.shape}, size={X_size_mb:.2f} MB, dtype={X.dtype}")
    logger.info(f"Loaded y: shape={y.shape}, size={y_size_kb:.2f} KB, dtype={y.dtype}")
    
    # Log file sizes for dataset identification (Requirement 19.3)
    logger.info(f"Dataset identification: X.npy size = {X_path.stat().st_size} bytes")
    logger.info(f"Dataset identification: y.npy size = {y_path.stat().st_size} bytes")
    
    return X, y


def create_callbacks(
    model_path: str = "bisindo_holistic_gru.h5",
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6
) -> list:
    """
    Create training callbacks for model.fit().
    
    This function creates three essential callbacks for GRU model training:
    1. EarlyStopping: Stops training when validation loss stops improving
    2. ModelCheckpoint: Saves the best model based on validation accuracy
    3. ReduceLROnPlateau: Reduces learning rate when validation loss plateaus
    
    These callbacks help improve training efficiency and prevent overfitting.
    
    Args:
        model_path: Path where the best model will be saved (default: "bisindo_holistic_gru.h5")
        early_stopping_patience: Number of epochs with no improvement before stopping
                                (default: 15)
        reduce_lr_patience: Number of epochs with no improvement before reducing LR
                           (default: 5)
        reduce_lr_factor: Factor by which learning rate will be reduced (default: 0.5)
        min_lr: Minimum learning rate (default: 1e-6)
    
    Returns:
        List of Keras callbacks ready to be passed to model.fit()
    
    Requirements:
        - Requirement 12.5: Implements EarlyStopping(patience=15, restore_best_weights=True)
        - Requirement 12.5: Implements ModelCheckpoint(save_best_only=True)
        - Requirement 12.5: Implements ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    
    Examples:
        >>> callbacks = create_callbacks("my_model.h5")
        >>> model.fit(X_train, y_train, validation_data=(X_val, y_val), 
        ...           epochs=50, callbacks=callbacks)
        
        >>> # Custom parameters
        >>> callbacks = create_callbacks(
        ...     model_path="best_model.h5",
        ...     early_stopping_patience=20,
        ...     reduce_lr_patience=7
        ... )
    
    Notes:
        - EarlyStopping monitors 'val_loss' and restores best weights on stop
        - ModelCheckpoint monitors 'val_accuracy' and only saves improvements
        - ReduceLROnPlateau monitors 'val_loss' and reduces LR when plateauing
        - These callbacks work together to optimize training and model quality
    """
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    logger.info(f"Creating training callbacks with model_path={model_path}")
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    logger.info(f"  - EarlyStopping: patience={early_stopping_patience}, restore_best_weights=True")
    
    # Model checkpoint callback
    model_checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    logger.info(f"  - ModelCheckpoint: save_best_only=True, monitor=val_accuracy")
    
    # Reduce learning rate on plateau callback
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1
    )
    logger.info(f"  - ReduceLROnPlateau: factor={reduce_lr_factor}, patience={reduce_lr_patience}, min_lr={min_lr}")
    
    callbacks = [early_stopping, model_checkpoint, reduce_lr]
    
    return callbacks


# For backward compatibility - allow direct import of commonly used functions
__all__ = [
    'set_random_seeds',
    'setup_gpu',
    'load_landmark_cache',
    'create_callbacks',
]

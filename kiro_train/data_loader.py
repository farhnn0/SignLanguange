"""
Data Loading Utilities for BISINDO Kata Model Training

This module provides utilities for loading preprocessed landmark data
from cached numpy files. It includes dataset identification via hash/size
logging and comprehensive error handling.

Requirements: 19.3 (Data Loading Utilities)
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Tuple
import numpy as np


def calculate_file_hash(filepath: str, algorithm: str = "md5") -> str:
    """
    Calculate hash of a file for dataset identification.
    
    Args:
        filepath: Path to the file
        algorithm: Hash algorithm to use (default: md5)
    
    Returns:
        Hex string of the file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def log_dataset_info(X: np.ndarray, y: np.ndarray, 
                    X_path: str, y_path: str,
                    logger: logging.Logger = None) -> None:
    """
    Log dataset information for identification and reproducibility.
    
    Logs:
    - File paths
    - File sizes (in bytes and MB)
    - File hashes (MD5)
    - Data shapes and dtypes
    - Dataset statistics
    
    Args:
        X: Landmark sequences array (N, 50, 258)
        y: Labels array (N,)
        X_path: Path to X.npy file
        y_path: Path to y.npy file
        logger: Optional logger instance (prints to console if None)
    """
    if logger is None:
        import sys
        handler = logging.StreamHandler(sys.stdout)
        logger = logging.getLogger(__name__)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    logger.info("=" * 60)
    logger.info("Dataset Information for Identification")
    logger.info("=" * 60)
    
    # File paths
    logger.info(f"X.npy path: {X_path}")
    logger.info(f"y.npy path: {y_path}")
    
    # File sizes
    X_size_bytes = os.path.getsize(X_path)
    y_size_bytes = os.path.getsize(y_path)
    X_size_mb = X_size_bytes / (1024 * 1024)
    y_size_mb = y_size_bytes / (1024 * 1024)
    
    logger.info(f"X.npy size: {X_size_bytes:,} bytes ({X_size_mb:.2f} MB)")
    logger.info(f"y.npy size: {y_size_bytes:,} bytes ({y_size_mb:.2f} MB)")
    
    # File hashes (MD5 for identification)
    logger.info("Calculating MD5 hashes for dataset identification...")
    X_hash = calculate_file_hash(X_path, "md5")
    y_hash = calculate_file_hash(y_path, "md5")
    
    logger.info(f"X.npy MD5: {X_hash}")
    logger.info(f"y.npy MD5: {y_hash}")
    
    # Data shapes and types
    logger.info(f"X shape: {X.shape}, dtype: {X.dtype}")
    logger.info(f"y shape: {y.shape}, dtype: {y.dtype}")
    
    # Dataset statistics
    num_samples = X.shape[0]
    num_classes = len(np.unique(y))
    
    logger.info(f"Total samples: {num_samples}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Per-class distribution
    unique_labels, class_counts = np.unique(y, return_counts=True)
    logger.info("Per-class sample distribution:")
    for label, count in zip(unique_labels, class_counts):
        logger.info(f"  Class '{label}': {count} samples")
    
    logger.info("=" * 60)


def load_landmark_cache(processed_dir: str = "processed_bisindo",
                       logger: logging.Logger = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed landmark data from cached numpy files.
    
    Loads X.npy and y.npy from the specified directory. Includes comprehensive
    error handling for missing files and hash/size logging for dataset
    identification and reproducibility.
    
    Args:
        processed_dir: Path to directory containing X.npy and y.npy
                      (default: "processed_bisindo")
        logger: Optional logger instance (creates default logger if None)
    
    Returns:
        Tuple of (X, y) where:
        - X: Landmark sequences array with shape (N, 50, 258)
        - y: Labels array with shape (N,)
    
    Raises:
        FileNotFoundError: If X.npy or y.npy not found with informative message
        ValueError: If loaded data has unexpected shapes
        IOError: If files cannot be read (permission denied, corrupted, etc.)
    
    Example:
        >>> X, y = load_landmark_cache()
        >>> print(X.shape, y.shape)
        (2500, 50, 258) (2500,)
    """
    
    # Setup logger
    if logger is None:
        import sys
        handler = logging.StreamHandler(sys.stdout)
        logger = logging.getLogger(__name__)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Construct file paths
    processed_path = Path(processed_dir)
    X_path = processed_path / "X.npy"
    y_path = processed_path / "y.npy"
    
    logger.info(f"Loading landmark cache from: {processed_path.absolute()}")
    
    # Check if directory exists
    if not processed_path.exists():
        error_msg = (
            f"Processed data directory not found: {processed_path.absolute()}\n"
            f"Expected directory to contain X.npy and y.npy files.\n"
            f"Please run MediaPipe extraction first or provide correct path."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Check for X.npy
    if not X_path.exists():
        error_msg = (
            f"X.npy not found at: {X_path.absolute()}\n"
            f"This file contains landmark sequences and is required for training.\n"
            f"Please run the MediaPipe extraction preprocessing step first."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Check for y.npy
    if not y_path.exists():
        error_msg = (
            f"y.npy not found at: {y_path.absolute()}\n"
            f"This file contains class labels and is required for training.\n"
            f"Please run the MediaPipe extraction preprocessing step first."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info("Loading X.npy...")
        X = np.load(str(X_path))
        logger.info(f"Loaded X with shape: {X.shape}, dtype: {X.dtype}")
        
        logger.info("Loading y.npy...")
        y = np.load(str(y_path), allow_pickle=True)
        logger.info(f"Loaded y with shape: {y.shape}, dtype: {y.dtype}")
        
    except Exception as e:
        error_msg = (
            f"Error reading landmark cache files:\n"
            f"  X.npy: {X_path.absolute()}\n"
            f"  y.npy: {y_path.absolute()}\n"
            f"Details: {str(e)}\n"
            f"Please ensure files are valid numpy arrays and not corrupted."
        )
        logger.error(error_msg)
        raise IOError(error_msg) from e
    
    # Validate shapes
    try:
        if len(X.shape) != 3:
            raise ValueError(
                f"X should be 3D array (N, seq_len, features), "
                f"got shape {X.shape}"
            )
        
        if len(y.shape) != 1:
            raise ValueError(
                f"y should be 1D array (N,), got shape {y.shape}"
            )
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatched number of samples: X has {X.shape[0]} "
                f"but y has {y.shape[0]}"
            )
        
        # Expected format: (N, 50, 258) for MediaPipe Holistic
        sequence_length = X.shape[1]
        num_features = X.shape[2]
        
        if sequence_length != 50:
            logger.warning(
                f"Expected sequence length 50, got {sequence_length}. "
                f"This may cause model input shape mismatch."
            )
        
        if num_features != 258:
            logger.warning(
                f"Expected 258 features (pose 132 + left hand 63 + right hand 63), "
                f"got {num_features}. This may cause model input shape mismatch."
            )
    
    except ValueError as e:
        error_msg = (
            f"Loaded data has unexpected shape:\n"
            f"  X shape: {X.shape} (expected: (N, 50, 258))\n"
            f"  y shape: {y.shape} (expected: (N,))\n"
            f"Details: {str(e)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    
    # Log dataset information for reproducibility
    log_dataset_info(X, y, str(X_path), str(y_path), logger)
    
    logger.info(f"Successfully loaded landmark cache:")
    logger.info(f"  Samples: {X.shape[0]}")
    logger.info(f"  Sequence length: {X.shape[1]}")
    logger.info(f"  Features per frame: {X.shape[2]}")
    logger.info(f"  Number of classes: {len(np.unique(y))}")
    
    return X, y


if __name__ == "__main__":
    """Smoke test for data loader."""
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    try:
        print("Testing load_landmark_cache()...")
        X, y = load_landmark_cache()
        print("\n✓ Data loaded successfully!")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Sample labels: {np.unique(y)[:5]}")
        
    except FileNotFoundError as e:
        print(f"\n✗ File not found error:\n{e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n✗ Validation error:\n{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error:\n{e}", file=sys.stderr)
        sys.exit(1)

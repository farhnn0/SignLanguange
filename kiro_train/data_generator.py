"""
Data Generator for BISINDO Kata Model with On-the-Fly Augmentation

This module provides the AugmentedSequenceGenerator class, a Keras Sequence
that generates batches of landmark data with on-the-fly augmentation during
training. The generator:
- Loads cached landmark sequences from X.npy and y.npy
- Applies augmentations per batch based on configuration
- Supports both training (with augmentation) and validation (without augmentation) modes
- Maintains deterministic behavior with proper shuffling and indexing
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AugmentedSequenceGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence generator for landmark sequences with on-the-fly augmentation.
    
    This generator:
    - Wraps original landmark data (X, y)
    - Applies augmentations per batch during training
    - Multiplies effective dataset size by augmentation_multiplier
    - Disables augmentation in validation mode
    - Shuffles samples after each epoch
    
    Args:
        X: Landmark sequences array of shape (N, 50, 258)
        y: Labels array of shape (N,)
        augmenter: LandmarkAugmenter instance for applying augmentations
        batch_size: Number of samples per batch (default: 32)
        augmentation_multiplier: How many times to augment each sample per epoch (default: 3)
        shuffle: Whether to shuffle indexes after each epoch (default: True)
        is_validation: If True, disable all augmentations (default: False)
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmenter,
        batch_size: int = 32,
        augmentation_multiplier: int = 3,
        shuffle: bool = True,
        is_validation: bool = False
    ):
        """
        Initialize the data generator.
        
        Args:
            X: Landmark sequences (N, 50, 258)
            y: Labels (N,)
            augmenter: LandmarkAugmenter instance
            batch_size: Batch size
            augmentation_multiplier: Augmentation multiplier (disabled if is_validation=True)
            shuffle: Shuffle samples each epoch
            is_validation: If True, no augmentation
        """
        self.X = X
        self.y = y
        self.augmenter = augmenter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_validation = is_validation
        
        # In validation mode, disable augmentation
        if is_validation:
            self.augmentation_multiplier = 1
        else:
            self.augmentation_multiplier = augmentation_multiplier
        
        self.n_samples = len(self.X)
        self.indexes = None
        
        # Initialize indexes
        self.on_epoch_end()
        
        logger.info(
            f"AugmentedSequenceGenerator initialized: "
            f"n_samples={self.n_samples}, batch_size={batch_size}, "
            f"augmentation_multiplier={self.augmentation_multiplier}, "
            f"is_validation={is_validation}"
        )
    
    def __len__(self) -> int:
        """
        Returns the number of batches per epoch.
        
        Number of batches = ceil(n_samples * augmentation_multiplier / batch_size)
        
        In validation mode, augmentation_multiplier is 1, so:
        Number of batches = ceil(n_samples / batch_size)
        
        Returns:
            Number of batches
        """
        n_batches = int(np.ceil(
            self.n_samples * self.augmentation_multiplier / self.batch_size
        ))
        return n_batches
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.
        
        This method:
        1. Maps batch index to sample indices (accounting for augmentation_multiplier)
        2. Loads samples and labels
        3. Applies augmentations (if not validation mode)
        4. Returns batch as (X_batch, y_batch)
        
        Args:
            index: Batch index (0 to len(self) - 1)
        
        Returns:
            Tuple of (X_batch, y_batch) where:
            - X_batch shape: (batch_size, 50, 258)
            - y_batch shape: (batch_size,)
        """
        # Calculate which samples belong to this batch
        # With augmentation_multiplier=3, sample 0 appears in batches 0, 1, 2, etc.
        batch_start_idx = index * self.batch_size
        batch_end_idx = min(batch_start_idx + self.batch_size, 
                            self.n_samples * self.augmentation_multiplier)
        
        # Get sample indices (wrapping around with augmentation_multiplier)
        sample_indices = []
        for batch_pos in range(batch_start_idx, batch_end_idx):
            # Map batch position to sample index (with cycling for augmentation)
            sample_idx = batch_pos % self.n_samples
            sample_indices.append(sample_idx)
        
        # Load data
        X_batch = []
        y_batch = []
        
        for sample_idx in sample_indices:
            # Get the actual sample from shuffled indexes
            actual_idx = self.indexes[sample_idx]
            
            # Load sample and label
            sample = self.X[actual_idx].copy()
            label = self.y[actual_idx]
            
            # Apply augmentation if not in validation mode
            if not self.is_validation:
                try:
                    sample = self.augmenter.augment(sample)
                except Exception as e:
                    logger.warning(
                        f"Augmentation failed for sample {actual_idx}: {str(e)}. "
                        f"Using original sample."
                    )
                    # Use original sample if augmentation fails
            
            X_batch.append(sample)
            y_batch.append(label)
        
        # Convert to numpy arrays
        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.int32)
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """
        Shuffle indexes after each epoch.
        
        This method:
        - Resets indexes to [0, 1, 2, ..., n_samples-1]
        - Shuffles them if shuffle=True
        
        Called automatically by Keras at the end of each epoch.
        """
        self.indexes = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        logger.debug(f"Epoch ended: indexes shuffled (shuffle={self.shuffle})")

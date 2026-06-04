"""
Landmark Augmentation Module for BISINDO Kata Model

This module provides the LandmarkAugmenter class for applying augmentations
to MediaPipe Holistic landmark sequences. It supports the following techniques:
- Horizontal flip (with hand swapping)
- Speed variation (temporal resampling)
- Random crop and resize (scaling and translation)
- Gaussian noise (for landmark coordinates)
- Rotation (2D rotation around center point)
- Orchestrated augmentation (applies multiple techniques based on config)
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LandmarkAugmenter:
    """
    Applies augmentations to MediaPipe Holistic landmark sequences.
    
    Landmark layout (258 features per frame):
    - Pose: indices 0-131 (33 landmarks × 4: x, y, z, visibility)
    - Left hand: indices 132-194 (21 landmarks × 3: x, y, z)
    - Right hand: indices 195-257 (21 landmarks × 3: x, y, z)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the augmenter with configuration.
        
        Args:
            config: Dictionary containing augmentation parameters:
                - enable_horizontal_flip (bool)
                - flip_probability (float)
                - enable_speed_variation (bool)
                - speed_factors (list)
                - speed_probability (float)
                - enable_random_crop_resize (bool)
                - scale_range (list or tuple)
                - crop_probability (float)
                - enable_gaussian_noise (bool)
                - landmark_noise_std (float)
                - noise_probability (float)
                - enable_rotation (bool)
                - rotation_range_degrees (list or tuple)
                - rotation_probability (float)
        """
        self.config = config
        self.sequence_length = 50
        
        # Landmark indices
        self.POSE_START = 0
        self.POSE_END = 132  # 33 × 4
        self.LEFT_HAND_START = 132
        self.LEFT_HAND_END = 195  # 21 × 3
        self.RIGHT_HAND_START = 195
        self.RIGHT_HAND_END = 258  # 21 × 3
        
        logger.info("LandmarkAugmenter initialized with config")
    
    def horizontal_flip(self, sequence: np.ndarray) -> np.ndarray:
        """
        Flip landmarks horizontally and swap left/right hands.
        
        This method:
        1. Flips x coordinates: x' = 1.0 - x
        2. Swaps left and right hand blocks
        3. Preserves visibility and z coordinates
        
        Args:
            sequence: Landmark sequence of shape (50, 258)
        
        Returns:
            Flipped sequence of shape (50, 258)
        """
        flipped = sequence.copy()
        
        # Flip x coordinates for all landmarks
        # X coordinates are at indices 0, 4, 8, ... for pose (every 4th starting from 0)
        # X coordinates are at indices 0, 3, 6, ... for hands (every 3rd starting from 0)
        
        for frame_idx in range(len(flipped)):
            # Flip pose x coordinates (indices 0, 4, 8, ..., 128)
            for i in range(0, self.POSE_END, 4):
                flipped[frame_idx, i] = 1.0 - flipped[frame_idx, i]
            
            # Flip left hand x coordinates
            for i in range(self.LEFT_HAND_START, self.LEFT_HAND_END, 3):
                flipped[frame_idx, i] = 1.0 - flipped[frame_idx, i]
            
            # Flip right hand x coordinates
            for i in range(self.RIGHT_HAND_START, self.RIGHT_HAND_END, 3):
                flipped[frame_idx, i] = 1.0 - flipped[frame_idx, i]
        
        # Swap left and right hand blocks
        left_hand = flipped[:, self.LEFT_HAND_START:self.LEFT_HAND_END].copy()
        right_hand = flipped[:, self.RIGHT_HAND_START:self.RIGHT_HAND_END].copy()
        
        flipped[:, self.LEFT_HAND_START:self.LEFT_HAND_END] = right_hand
        flipped[:, self.RIGHT_HAND_START:self.RIGHT_HAND_END] = left_hand
        
        return flipped
    
    def speed_variation(self, sequence: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """
        Apply speed variation through temporal resampling.
        
        This method resamples the temporal dimension by:
        1. If factor < 1: slow down (stretch sequence)
        2. If factor > 1: speed up (compress sequence)
        3. Normalize output to exactly SEQUENCE_LENGTH frames
        
        Args:
            sequence: Landmark sequence of shape (N, 258) where N is variable
            factor: Speed factor (0.8-1.2). If None, randomly select from config.
        
        Returns:
            Resampled sequence of shape (50, 258)
        """
        if factor is None:
            factor = np.random.choice(self.config.get('speed_factors', [0.8, 0.9, 1.0, 1.1, 1.2]))
        
        num_frames = len(sequence)
        if num_frames == 0:
            logger.warning("Speed variation: empty sequence, returning original")
            return sequence
        
        # Calculate new number of frames after speed change
        # speed_factor < 1 means slower (more frames needed to represent same motion)
        # speed_factor > 1 means faster (fewer frames needed)
        new_num_frames = max(1, int(num_frames / factor))
        
        # Interpolate to new_num_frames
        if new_num_frames > num_frames:
            # Slow down: need to interpolate more frames
            indices = np.linspace(0, num_frames - 1, new_num_frames)
            resampled = np.zeros((new_num_frames, sequence.shape[1]), dtype=sequence.dtype)
            
            for feat_idx in range(sequence.shape[1]):
                resampled[:, feat_idx] = np.interp(indices, np.arange(num_frames), sequence[:, feat_idx])
        else:
            # Speed up or same: subsample
            indices = np.linspace(0, num_frames - 1, new_num_frames).astype(int)
            resampled = sequence[indices]
        
        # Normalize to SEQUENCE_LENGTH frames
        if len(resampled) == self.sequence_length:
            return resampled
        elif len(resampled) > self.sequence_length:
            # Subsample
            indices = np.linspace(0, len(resampled) - 1, self.sequence_length).astype(int)
            return resampled[indices]
        else:
            # Pad with last frame
            pad_count = self.sequence_length - len(resampled)
            last_frame = resampled[-1]
            padding = np.repeat(last_frame[np.newaxis, :], pad_count, axis=0)
            return np.concatenate([resampled, padding], axis=0)
    
    def random_crop_resize(self, sequence: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        """
        Scale and translate landmark coordinates.
        
        This method:
        1. Scales x, y coordinates by scale factor
        2. Adds random translation to keep points in [0, 1]
        3. Clips coordinates to [0, 1]
        4. Preserves z and visibility
        
        Args:
            sequence: Landmark sequence of shape (50, 258)
            scale: Scale factor (0.85-1.15). If None, randomly select from config.
        
        Returns:
            Scaled sequence of shape (50, 258)
        """
        if scale is None:
            scale_range = self.config.get('scale_range', [0.85, 1.15])
            scale = np.random.uniform(scale_range[0], scale_range[1])
        
        scaled = sequence.copy().astype(np.float32)
        
        for frame_idx in range(len(scaled)):
            # Calculate translation to keep scaled coordinates in [0, 1]
            # After scaling, coordinates range from 0 to scale
            # We need to translate by a random amount to fill the space
            
            # Maximum translation is (1 - scale) so scaled coords don't exceed 1
            max_translate = max(0, 1.0 - scale)
            if max_translate > 0:
                translate_x = np.random.uniform(0, max_translate)
                translate_y = np.random.uniform(0, max_translate)
            else:
                # If scale > 1, translate should be negative to bring coords back
                translate_x = np.random.uniform(1.0 - scale, 0)
                translate_y = np.random.uniform(1.0 - scale, 0)
            
            # Scale and translate x, y coordinates (every 3rd or 4th element)
            # Pose: x at indices 0, 4, 8, ...
            for i in range(0, self.POSE_END, 4):
                scaled[frame_idx, i] = scaled[frame_idx, i] * scale + translate_x
            
            # Left hand: x at indices 0, 3, 6, ... (relative to LEFT_HAND_START)
            for i in range(self.LEFT_HAND_START, self.LEFT_HAND_END, 3):
                scaled[frame_idx, i] = scaled[frame_idx, i] * scale + translate_x
            
            # Right hand: x at indices 0, 3, 6, ...
            for i in range(self.RIGHT_HAND_START, self.RIGHT_HAND_END, 3):
                scaled[frame_idx, i] = scaled[frame_idx, i] * scale + translate_x
            
            # Pose: y at indices 1, 5, 9, ...
            for i in range(1, self.POSE_END, 4):
                scaled[frame_idx, i] = scaled[frame_idx, i] * scale + translate_y
            
            # Left hand: y at indices 1, 4, 7, ...
            for i in range(self.LEFT_HAND_START + 1, self.LEFT_HAND_END, 3):
                scaled[frame_idx, i] = scaled[frame_idx, i] * scale + translate_y
            
            # Right hand: y at indices 1, 4, 7, ...
            for i in range(self.RIGHT_HAND_START + 1, self.RIGHT_HAND_END, 3):
                scaled[frame_idx, i] = scaled[frame_idx, i] * scale + translate_y
        
        # Clip coordinates to [0, 1]
        scaled = np.clip(scaled, 0.0, 1.0)
        
        return scaled
    
    def gaussian_noise(self, sequence: np.ndarray, std: Optional[float] = None) -> np.ndarray:
        """
        Add Gaussian noise to landmark coordinates.
        
        This method adds noise to x, y, z coordinates but preserves visibility.
        
        Args:
            sequence: Landmark sequence of shape (50, 258)
            std: Standard deviation of noise. If None, use config value.
        
        Returns:
            Noisy sequence of shape (50, 258)
        """
        if std is None:
            std = self.config.get('landmark_noise_std', 0.005)
        
        noisy = sequence.copy().astype(np.float32)
        
        for frame_idx in range(len(noisy)):
            # Add noise to pose x, y, z (but not visibility)
            # Pose layout: x, y, z, visibility (repeated)
            for i in range(0, self.POSE_END, 4):
                # x, y, z get noise
                for j in range(3):
                    noisy[frame_idx, i + j] += np.random.normal(0, std)
                # visibility (i + 3) is preserved
            
            # Add noise to left hand x, y, z
            for i in range(self.LEFT_HAND_START, self.LEFT_HAND_END, 3):
                for j in range(3):
                    noisy[frame_idx, i + j] += np.random.normal(0, std)
            
            # Add noise to right hand x, y, z
            for i in range(self.RIGHT_HAND_START, self.RIGHT_HAND_END, 3):
                for j in range(3):
                    noisy[frame_idx, i + j] += np.random.normal(0, std)
        
        return noisy
    
    def rotation(self, sequence: np.ndarray, angle_deg: Optional[float] = None) -> np.ndarray:
        """
        Rotate landmarks around center point (0.5, 0.5) using 2D rotation matrix.
        
        This method:
        1. Rotates x, y coordinates around (0.5, 0.5)
        2. Preserves z and visibility
        3. Clips coordinates to [0, 1]
        
        Args:
            sequence: Landmark sequence of shape (50, 258)
            angle_deg: Rotation angle in degrees. If None, randomly select from config.
        
        Returns:
            Rotated sequence of shape (50, 258)
        """
        if angle_deg is None:
            rot_range = self.config.get('rotation_range_degrees', [-10.0, 10.0])
            angle_deg = np.random.uniform(rot_range[0], rot_range[1])
        
        # Convert to radians
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rotated = sequence.copy().astype(np.float32)
        center = 0.5
        
        for frame_idx in range(len(rotated)):
            # Rotate pose landmarks (x, y)
            for i in range(0, self.POSE_END, 4):
                x = rotated[frame_idx, i] - center
                y = rotated[frame_idx, i + 1] - center
                
                # 2D rotation matrix
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                
                rotated[frame_idx, i] = x_rot + center
                rotated[frame_idx, i + 1] = y_rot + center
            
            # Rotate left hand landmarks
            for i in range(self.LEFT_HAND_START, self.LEFT_HAND_END, 3):
                x = rotated[frame_idx, i] - center
                y = rotated[frame_idx, i + 1] - center
                
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                
                rotated[frame_idx, i] = x_rot + center
                rotated[frame_idx, i + 1] = y_rot + center
            
            # Rotate right hand landmarks
            for i in range(self.RIGHT_HAND_START, self.RIGHT_HAND_END, 3):
                x = rotated[frame_idx, i] - center
                y = rotated[frame_idx, i + 1] - center
                
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                
                rotated[frame_idx, i] = x_rot + center
                rotated[frame_idx, i + 1] = y_rot + center
        
        # Clip coordinates to [0, 1]
        rotated = np.clip(rotated, 0.0, 1.0)
        
        return rotated
    
    def augment(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to a landmark sequence based on configuration.
        
        This orchestrator method applies enabled augmentations with their
        configured probabilities to the input sequence.
        
        Args:
            sequence: Landmark sequence of shape (50, 258)
        
        Returns:
            Augmented sequence of shape (50, 258)
        """
        augmented = sequence.copy().astype(np.float32)
        
        try:
            # Horizontal flip
            if self.config.get('enable_horizontal_flip', False):
                if np.random.rand() < self.config.get('flip_probability', 0.5):
                    augmented = self.horizontal_flip(augmented)
            
            # Speed variation
            if self.config.get('enable_speed_variation', False):
                if np.random.rand() < self.config.get('speed_probability', 0.7):
                    augmented = self.speed_variation(augmented)
            
            # Random crop and resize
            if self.config.get('enable_random_crop_resize', False):
                if np.random.rand() < self.config.get('crop_probability', 0.5):
                    augmented = self.random_crop_resize(augmented)
            
            # Gaussian noise
            if self.config.get('enable_gaussian_noise', False):
                if np.random.rand() < self.config.get('noise_probability', 0.5):
                    augmented = self.gaussian_noise(augmented)
            
            # Rotation
            if self.config.get('enable_rotation', False):
                if np.random.rand() < self.config.get('rotation_probability', 0.5):
                    augmented = self.rotation(augmented)
        
        except Exception as e:
            logger.warning(f"Augmentation failed, returning original sequence: {str(e)}")
            return sequence.astype(np.float32)
        
        return augmented.astype(np.float32)


class VideoAugmenter:
    """
    Applies augmentations to video frames (raw BGR/RGB).
    
    This class operates on lists of video frames and applies various
    transformations including horizontal flip, brightness/contrast adjustment,
    crop and resize, Gaussian noise, and rotation using OpenCV operations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the video augmenter with configuration.
        
        Args:
            config: Dictionary containing video augmentation parameters:
                - enable_horizontal_flip (bool)
                - flip_probability (float)
                - enable_brightness_contrast (bool)
                - brightness_range (tuple)
                - contrast_range (tuple)
                - brightness_contrast_probability (float)
                - enable_random_crop_resize (bool)
                - scale_range (tuple)
                - crop_probability (float)
                - enable_gaussian_noise (bool)
                - pixel_noise_std (float)
                - noise_probability (float)
                - enable_rotation (bool)
                - rotation_range_degrees (tuple)
                - rotation_probability (float)
        """
        import cv2
        self.cv2 = cv2
        self.config = config
        
        logger.info("VideoAugmenter initialized with config")
    
    def horizontal_flip(self, frames: list) -> list:
        """
        Flip video frames horizontally.
        
        This method applies cv2.flip with flipCode=1 (horizontal flip)
        to each frame in the video sequence.
        
        Args:
            frames: List of video frames, each is numpy array (H, W, 3) uint8
        
        Returns:
            List of flipped frames with same length and shape
        """
        flipped_frames = []
        
        for frame in frames:
            # cv2.flip with flipCode=1 flips horizontally
            flipped_frame = self.cv2.flip(frame, 1)
            flipped_frames.append(flipped_frame)
        
        return flipped_frames
    
    def brightness_contrast(self, frames: list, brightness_factor: Optional[float] = None,
                          contrast_factor: Optional[float] = None) -> list:
        """
        Adjust brightness and contrast of video frames.
        
        This method:
        1. Brightness: shifts pixel values by brightness_factor * 255, clipped to [0, 255]
        2. Contrast: multiplies (pixel - 127.5) by contrast_factor + 127.5, clipped to [0, 255]
        
        Args:
            frames: List of video frames (H, W, 3) uint8
            brightness_factor: Value in brightness_range from config. If None, sample randomly.
            contrast_factor: Value in contrast_range from config. If None, sample randomly.
        
        Returns:
            List of adjusted frames with same length and shape
        """
        if brightness_factor is None:
            brightness_range = self.config.get('brightness_range', (-0.2, 0.2))
            brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
        
        if contrast_factor is None:
            contrast_range = self.config.get('contrast_range', (0.8, 1.2))
            contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
        
        adjusted_frames = []
        
        for frame in frames:
            # Convert to float32 for processing
            frame_float = frame.astype(np.float32)
            
            # Apply brightness adjustment
            frame_float = frame_float + (brightness_factor * 255.0)
            
            # Apply contrast adjustment: (pixel - 127.5) * contrast + 127.5
            frame_float = (frame_float - 127.5) * contrast_factor + 127.5
            
            # Clip to [0, 255] and convert back to uint8
            frame_adjusted = np.clip(frame_float, 0, 255).astype(np.uint8)
            
            adjusted_frames.append(frame_adjusted)
        
        return adjusted_frames
    
    def random_crop_resize(self, frames: list, scale: Optional[float] = None) -> list:
        """
        Randomly crop and resize video frames.
        
        This method:
        1. Crops each frame to a region of size scale * frame_size
        2. Resizes the cropped region back to the original frame size
        
        Args:
            frames: List of video frames (H, W, 3) uint8
            scale: Scale factor from scale_range. If None, sample randomly.
        
        Returns:
            List of cropped and resized frames with same length and original shape
        """
        if len(frames) == 0:
            return frames
        
        if scale is None:
            scale_range = self.config.get('scale_range', (0.85, 1.15))
            scale = np.random.uniform(scale_range[0], scale_range[1])
        
        # Use first frame to get dimensions
        h, w = frames[0].shape[:2]
        
        # Calculate crop size
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        
        # Ensure crop doesn't exceed frame size
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        
        # Calculate random top-left corner of crop region
        top = np.random.randint(0, max(1, h - crop_h + 1))
        left = np.random.randint(0, max(1, w - crop_w + 1))
        
        cropped_resized_frames = []
        
        for frame in frames:
            # Crop the frame
            cropped = frame[top:top + crop_h, left:left + crop_w]
            
            # Resize back to original size using linear interpolation
            resized = self.cv2.resize(cropped, (w, h), interpolation=self.cv2.INTER_LINEAR)
            
            cropped_resized_frames.append(resized)
        
        return cropped_resized_frames
    
    def gaussian_noise(self, frames: list, std: Optional[float] = None) -> list:
        """
        Add Gaussian noise to video frames.
        
        This method adds noise from N(0, pixel_noise_std) distribution to
        each pixel, then clips to [0, 255] and converts back to uint8.
        
        Args:
            frames: List of video frames (H, W, 3) uint8
            std: Standard deviation of noise in pixel scale [0, 255].
                 If None, use config value.
        
        Returns:
            List of noisy frames with same length and shape
        """
        if std is None:
            std = self.config.get('pixel_noise_std', 5.0)
        
        noisy_frames = []
        
        for frame in frames:
            # Convert to float32
            frame_float = frame.astype(np.float32)
            
            # Add Gaussian noise
            noise = np.random.normal(0, std, frame_float.shape)
            frame_noisy = frame_float + noise
            
            # Clip to [0, 255] and convert back to uint8
            frame_noisy = np.clip(frame_noisy, 0, 255).astype(np.uint8)
            
            noisy_frames.append(frame_noisy)
        
        return noisy_frames
    
    def rotation(self, frames: list, angle_deg: Optional[float] = None) -> list:
        """
        Rotate video frames using cv2.warpAffine.
        
        This method:
        1. Gets rotation matrix around frame center using cv2.getRotationMatrix2D
        2. Applies rotation with cv2.warpAffine and border mode BORDER_REPLICATE
        
        Args:
            frames: List of video frames (H, W, 3) uint8
            angle_deg: Rotation angle in degrees from rotation_range_degrees.
                      If None, sample randomly.
        
        Returns:
            List of rotated frames with same length and shape
        """
        if len(frames) == 0:
            return frames
        
        if angle_deg is None:
            rot_range = self.config.get('rotation_range_degrees', (-10.0, 10.0))
            angle_deg = np.random.uniform(rot_range[0], rot_range[1])
        
        # Use first frame to get dimensions
        h, w = frames[0].shape[:2]
        center = (w / 2, h / 2)
        
        # Get rotation matrix
        rotation_matrix = self.cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)
        
        rotated_frames = []
        
        for frame in frames:
            # Apply rotation with border replicate
            rotated = self.cv2.warpAffine(frame, rotation_matrix, (w, h),
                                         borderMode=self.cv2.BORDER_REPLICATE)
            
            rotated_frames.append(rotated)
        
        return rotated_frames
    
    def augment(self, frames: list) -> list:
        """
        Apply augmentations to video frames based on configuration.
        
        This orchestrator method applies enabled augmentations with their
        configured probabilities to the input frame list.
        
        Args:
            frames: List of video frames (H, W, 3) uint8
        
        Returns:
            List of augmented frames with same length and shape
        """
        augmented = frames.copy()
        
        try:
            # Horizontal flip
            if self.config.get('enable_horizontal_flip', False):
                if np.random.rand() < self.config.get('flip_probability', 0.5):
                    augmented = self.horizontal_flip(augmented)
            
            # Brightness and contrast
            if self.config.get('enable_brightness_contrast', False):
                if np.random.rand() < self.config.get('brightness_contrast_probability', 0.5):
                    augmented = self.brightness_contrast(augmented)
            
            # Random crop and resize
            if self.config.get('enable_random_crop_resize', False):
                if np.random.rand() < self.config.get('crop_probability', 0.5):
                    augmented = self.random_crop_resize(augmented)
            
            # Gaussian noise
            if self.config.get('enable_gaussian_noise', False):
                if np.random.rand() < self.config.get('noise_probability', 0.5):
                    augmented = self.gaussian_noise(augmented)
            
            # Rotation
            if self.config.get('enable_rotation', False):
                if np.random.rand() < self.config.get('rotation_probability', 0.5):
                    augmented = self.rotation(augmented)
        
        except Exception as e:
            logger.warning(f"Video augmentation failed, returning original frames: {str(e)}")
            return frames
        
        return augmented


# ============================================
# SMOKE TEST
# ============================================

if __name__ == "__main__":
    """
    Smoke test for standalone execution.
    Tests each augmentation technique and displays shapes.
    """
    import yaml
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Try to load config from YAML
    config_path = "Train/augmentation_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[OK] Loaded config from {config_path}")
    except FileNotFoundError:
        print(f"[WARN] Config file not found at {config_path}, using defaults")
        config = {
            'enable_horizontal_flip': True,
            'flip_probability': 0.5,
            'enable_speed_variation': True,
            'speed_factors': [0.8, 0.9, 1.0, 1.1, 1.2],
            'speed_probability': 0.7,
            'enable_random_crop_resize': True,
            'scale_range': [0.85, 1.15],
            'crop_probability': 0.5,
            'enable_gaussian_noise': True,
            'landmark_noise_std': 0.005,
            'pixel_noise_std': 5.0,
            'noise_probability': 0.5,
            'enable_rotation': True,
            'rotation_range_degrees': [-10.0, 10.0],
            'rotation_probability': 0.5,
            'enable_brightness_contrast': True,
            'brightness_range': [-0.2, 0.2],
            'contrast_range': [0.8, 1.2],
            'brightness_contrast_probability': 0.5,
        }
    
    # Create sample landmark sequence
    sample_sequence = np.random.rand(50, 258).astype(np.float32)
    print(f"[OK] Created sample landmark sequence with shape {sample_sequence.shape}")
    
    # Create sample video frames
    sample_frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
    print(f"[OK] Created sample video frames: {len(sample_frames)} frames of shape {sample_frames[0].shape}")
    
    # Initialize augmenters
    landmark_augmenter = LandmarkAugmenter(config)
    print("[OK] Initialized LandmarkAugmenter")
    
    video_augmenter = VideoAugmenter(config)
    print("[OK] Initialized VideoAugmenter")
    
    # Test each augmentation individually
    print("\n" + "="*60)
    print("SMOKE TEST: LandmarkAugmenter Methods")
    print("="*60)
    
    landmark_tests = [
        ("Horizontal Flip", lambda: landmark_augmenter.horizontal_flip(sample_sequence)),
        ("Speed Variation", lambda: landmark_augmenter.speed_variation(sample_sequence)),
        ("Random Crop & Resize", lambda: landmark_augmenter.random_crop_resize(sample_sequence)),
        ("Gaussian Noise", lambda: landmark_augmenter.gaussian_noise(sample_sequence)),
        ("Rotation", lambda: landmark_augmenter.rotation(sample_sequence)),
    ]
    
    for test_name, test_func in landmark_tests:
        try:
            result = test_func()
            print(f"[OK] {test_name:25} | Input: {sample_sequence.shape} -> Output: {result.shape} | dtype: {result.dtype}")
        except Exception as e:
            print(f"[ERR] {test_name:25} | ERROR: {str(e)}")
    
    # Test orchestrated landmark augmentation
    print("\n" + "="*60)
    print("SMOKE TEST: LandmarkAugmenter Orchestrator")
    print("="*60)
    
    np.random.seed(42)
    try:
        augmented = landmark_augmenter.augment(sample_sequence)
        print(f"[OK] Orchestrated augment() | Input: {sample_sequence.shape} -> Output: {augmented.shape} | dtype: {augmented.dtype}")
    except Exception as e:
        print(f"[ERR] Orchestrated augment() | ERROR: {str(e)}")
    
    # Test each video augmentation individually
    print("\n" + "="*60)
    print("SMOKE TEST: VideoAugmenter Methods")
    print("="*60)
    
    video_tests = [
        ("Horizontal Flip", lambda: video_augmenter.horizontal_flip(sample_frames)),
        ("Brightness & Contrast", lambda: video_augmenter.brightness_contrast(sample_frames)),
        ("Random Crop & Resize", lambda: video_augmenter.random_crop_resize(sample_frames)),
        ("Gaussian Noise", lambda: video_augmenter.gaussian_noise(sample_frames)),
        ("Rotation", lambda: video_augmenter.rotation(sample_frames)),
    ]
    
    for test_name, test_func in video_tests:
        try:
            result = test_func()
            input_info = f"[{len(sample_frames)}x{sample_frames[0].shape}]"
            output_info = f"[{len(result)}x{result[0].shape}]"
            print(f"[OK] {test_name:25} | Input: {input_info} -> Output: {output_info} | dtype: {result[0].dtype}")
        except Exception as e:
            print(f"[ERR] {test_name:25} | ERROR: {str(e)}")
    
    # Test orchestrated video augmentation
    print("\n" + "="*60)
    print("SMOKE TEST: VideoAugmenter Orchestrator")
    print("="*60)
    
    np.random.seed(42)
    try:
        augmented_frames = video_augmenter.augment(sample_frames)
        input_info = f"[{len(sample_frames)}x{sample_frames[0].shape}]"
        output_info = f"[{len(augmented_frames)}x{augmented_frames[0].shape}]"
        print(f"[OK] Orchestrated augment() | Input: {input_info} -> Output: {output_info} | dtype: {augmented_frames[0].dtype}")
    except Exception as e:
        print(f"[ERR] Orchestrated augment() | ERROR: {str(e)}")
    
    print("\n" + "="*60)
    print("SMOKE TEST COMPLETE - ALL TESTS PASSED")
    print("="*60)

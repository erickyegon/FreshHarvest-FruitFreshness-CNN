"""
Data Preprocessing Component for FreshHarvest Classification System
=================================================================

This module provides comprehensive data preprocessing functionality
including image normalization, resizing, and format conversion.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import os

from ..utils.common import read_yaml

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing for FreshHarvest image classification.

    Handles image loading, resizing, normalization, and format conversion
    for optimal model training and inference.
    """

    def __init__(self, config_path: str):
        """
        Initialize data preprocessor.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.data_config = self.config['data']

        # Image specifications
        self.image_size = tuple(self.data_config['image_size'])
        self.channels = self.data_config['channels']
        self.normalization_method = self.data_config.get('normalization', 'standard')

        logger.info(f"Data preprocessor initialized: {self.image_size}, {self.channels} channels")

    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load image from file path.

        Args:
            image_path: Path to image file

        Returns:
            Loaded image array or None if failed
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None

            # Load image using OpenCV
            image = cv2.imread(str(image_path))

            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def resize_image(self, image: np.ndarray,
                    target_size: Optional[Tuple[int, int]] = None,
                    interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize image to target dimensions.

        Args:
            image: Input image array
            target_size: Target size (width, height)
            interpolation: Interpolation method

        Returns:
            Resized image
        """
        try:
            if target_size is None:
                target_size = self.image_size

            # Resize image
            resized = cv2.resize(image, target_size, interpolation=interpolation)

            return resized

        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image

    def normalize_image(self, image: np.ndarray,
                       method: Optional[str] = None) -> np.ndarray:
        """
        Normalize image pixel values.

        Args:
            image: Input image array
            method: Normalization method ('standard', 'minmax', 'imagenet')

        Returns:
            Normalized image
        """
        try:
            if method is None:
                method = self.normalization_method

            image = image.astype(np.float32)

            if method == 'standard':
                # Normalize to [0, 1]
                normalized = image / 255.0

            elif method == 'minmax':
                # Min-max normalization
                min_val = np.min(image)
                max_val = np.max(image)
                normalized = (image - min_val) / (max_val - min_val)

            elif method == 'imagenet':
                # ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                normalized = (image / 255.0 - mean) / std

            elif method == 'zscore':
                # Z-score normalization
                mean = np.mean(image)
                std = np.std(image)
                normalized = (image - mean) / std

            else:
                logger.warning(f"Unknown normalization method: {method}. Using standard.")
                normalized = image / 255.0

            return normalized

        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            return image

    def preprocess_single_image(self, image: Union[np.ndarray, str, Path],
                               target_size: Optional[Tuple[int, int]] = None,
                               normalize: bool = True) -> Optional[np.ndarray]:
        """
        Preprocess a single image for model input.

        Args:
            image: Input image (array or path)
            target_size: Target size for resizing
            normalize: Whether to normalize pixel values

        Returns:
            Preprocessed image array
        """
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                image = self.load_image(image)
                if image is None:
                    return None

            # Ensure image is numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)

            # Resize image
            if target_size or self.image_size:
                image = self.resize_image(image, target_size)

            # Normalize if requested
            if normalize:
                image = self.normalize_image(image)

            # Ensure correct number of channels
            if len(image.shape) == 2 and self.channels == 3:
                # Convert grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4 and self.channels == 3:
                # Convert RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    def preprocess_batch(self, images: List[Union[np.ndarray, str, Path]],
                        target_size: Optional[Tuple[int, int]] = None,
                        normalize: bool = True) -> Tuple[np.ndarray, List[int]]:
        """
        Preprocess a batch of images.

        Args:
            images: List of images (arrays or paths)
            target_size: Target size for resizing
            normalize: Whether to normalize pixel values

        Returns:
            Tuple of (preprocessed_images, valid_indices)
        """
        try:
            preprocessed_images = []
            valid_indices = []

            for i, image in enumerate(images):
                processed = self.preprocess_single_image(image, target_size, normalize)

                if processed is not None:
                    preprocessed_images.append(processed)
                    valid_indices.append(i)
                else:
                    logger.warning(f"Failed to preprocess image at index {i}")

            if preprocessed_images:
                # Convert to numpy array
                batch_array = np.array(preprocessed_images)
                logger.info(f"Preprocessed batch: {batch_array.shape}")
                return batch_array, valid_indices
            else:
                logger.error("No images were successfully preprocessed")
                return np.array([]), []

        except Exception as e:
            logger.error(f"Error preprocessing batch: {e}")
            return np.array([]), []

    def create_tensorflow_preprocessing(self) -> tf.keras.Sequential:
        """
        Create TensorFlow preprocessing pipeline.

        Returns:
            TensorFlow Sequential model for preprocessing
        """
        try:
            preprocessing_layers = []

            # Resizing layer
            preprocessing_layers.append(
                tf.keras.layers.Resizing(
                    height=self.image_size[1],
                    width=self.image_size[0]
                )
            )

            # Normalization layer
            if self.normalization_method == 'standard':
                preprocessing_layers.append(
                    tf.keras.layers.Rescaling(1./255)
                )
            elif self.normalization_method == 'imagenet':
                preprocessing_layers.append(
                    tf.keras.layers.Rescaling(1./255)
                )
                preprocessing_layers.append(
                    tf.keras.layers.Normalization(
                        mean=[0.485, 0.456, 0.406],
                        variance=[0.229**2, 0.224**2, 0.225**2]
                    )
                )

            return tf.keras.Sequential(preprocessing_layers)

        except Exception as e:
            logger.error(f"Error creating TensorFlow preprocessing: {e}")
            return tf.keras.Sequential([])

    def validate_image_format(self, image: np.ndarray) -> bool:
        """
        Validate image format and dimensions.

        Args:
            image: Input image array

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if image is numpy array
            if not isinstance(image, np.ndarray):
                return False

            # Check dimensions
            if len(image.shape) not in [2, 3]:
                return False

            # Check channels
            if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                return False

            # Check data type
            if image.dtype not in [np.uint8, np.float32, np.float64]:
                return False

            # Check value range
            if image.dtype == np.uint8:
                if np.min(image) < 0 or np.max(image) > 255:
                    return False
            elif image.dtype in [np.float32, np.float64]:
                if np.min(image) < 0 or np.max(image) > 1:
                    # Allow for normalized images
                    if not (np.min(image) >= -3 and np.max(image) <= 3):  # Allow ImageNet normalization
                        return False

            return True

        except Exception as e:
            logger.error(f"Error validating image format: {e}")
            return False

    def get_preprocessing_stats(self, images: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate preprocessing statistics for a dataset.

        Args:
            images: List of preprocessed images

        Returns:
            Dictionary containing statistics
        """
        try:
            if not images:
                return {}

            # Convert to numpy array
            image_array = np.array(images)

            stats = {
                'mean_pixel_value': float(np.mean(image_array)),
                'std_pixel_value': float(np.std(image_array)),
                'min_pixel_value': float(np.min(image_array)),
                'max_pixel_value': float(np.max(image_array)),
                'total_images': len(images),
                'image_shape': image_array.shape[1:],
                'total_pixels': int(np.prod(image_array.shape))
            }

            # Channel-wise statistics
            if len(image_array.shape) == 4 and image_array.shape[3] == 3:
                for i, channel in enumerate(['red', 'green', 'blue']):
                    channel_data = image_array[:, :, :, i]
                    stats[f'{channel}_mean'] = float(np.mean(channel_data))
                    stats[f'{channel}_std'] = float(np.std(channel_data))

            logger.info("Preprocessing statistics calculated")
            return stats

        except Exception as e:
            logger.error(f"Error calculating preprocessing stats: {e}")
            return {}

def preprocess_image_for_inference(image_path: str,
                                 target_size: Tuple[int, int] = (224, 224),
                                 normalize: bool = True) -> Optional[np.ndarray]:
    """
    Convenience function for preprocessing a single image for inference.

    Args:
        image_path: Path to image file
        target_size: Target size for resizing
        normalize: Whether to normalize pixel values

    Returns:
        Preprocessed image array ready for model input
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, target_size)

        # Normalize
        if normalize:
            image = image.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    except Exception as e:
        logger.error(f"Error preprocessing image for inference: {e}")
        return None
"""
Data augmentation component for the FreshHarvest project.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Callable
import logging
import random
from PIL import Image, ImageEnhance
import tensorflow as tf

from ..utils.common import read_yaml


class DataAugmentation:
    """
    Data augmentation component for image transformations.
    """

    def __init__(self, config_path: str):
        """
        Initialize data augmentation component.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.data_config = self.config['data']

        # Image specifications
        self.image_size = tuple(self.data_config['image_size'])
        self.channels = self.data_config['channels']

        logging.info("Data augmentation component initialized")

    def random_flip(self, image: np.ndarray, probability: float = 0.5) -> np.ndarray:
        """
        Randomly flip image horizontally.

        Args:
            image: Input image array
            probability: Probability of applying flip

        Returns:
            Augmented image
        """
        if random.random() < probability:
            return cv2.flip(image, 1)  # Horizontal flip
        return image

    def random_rotation(self, image: np.ndarray,
                       max_angle: float = 30.0,
                       probability: float = 0.5) -> np.ndarray:
        """
        Randomly rotate image.

        Args:
            image: Input image array
            max_angle: Maximum rotation angle in degrees
            probability: Probability of applying rotation

        Returns:
            Augmented image
        """
        if random.random() < probability:
            angle = random.uniform(-max_angle, max_angle)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)

            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Apply rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   borderMode=cv2.BORDER_REFLECT)
            return rotated
        return image

    def random_brightness(self, image: np.ndarray,
                         brightness_range: Tuple[float, float] = (0.8, 1.2),
                         probability: float = 0.5) -> np.ndarray:
        """
        Randomly adjust brightness.

        Args:
            image: Input image array
            brightness_range: Range of brightness factors
            probability: Probability of applying brightness adjustment

        Returns:
            Augmented image
        """
        if random.random() < probability:
            factor = random.uniform(*brightness_range)

            # Convert to PIL for enhancement
            pil_image = Image.fromarray(image.astype(np.uint8))
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = enhancer.enhance(factor)

            return np.array(enhanced)
        return image

    def random_contrast(self, image: np.ndarray,
                       contrast_range: Tuple[float, float] = (0.8, 1.2),
                       probability: float = 0.5) -> np.ndarray:
        """
        Randomly adjust contrast.

        Args:
            image: Input image array
            contrast_range: Range of contrast factors
            probability: Probability of applying contrast adjustment

        Returns:
            Augmented image
        """
        if random.random() < probability:
            factor = random.uniform(*contrast_range)

            # Convert to PIL for enhancement
            pil_image = Image.fromarray(image.astype(np.uint8))
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(factor)

            return np.array(enhanced)
        return image

    def random_saturation(self, image: np.ndarray,
                         saturation_range: Tuple[float, float] = (0.8, 1.2),
                         probability: float = 0.5) -> np.ndarray:
        """
        Randomly adjust saturation.

        Args:
            image: Input image array
            saturation_range: Range of saturation factors
            probability: Probability of applying saturation adjustment

        Returns:
            Augmented image
        """
        if random.random() < probability:
            factor = random.uniform(*saturation_range)

            # Convert to PIL for enhancement
            pil_image = Image.fromarray(image.astype(np.uint8))
            enhancer = ImageEnhance.Color(pil_image)
            enhanced = enhancer.enhance(factor)

            return np.array(enhanced)
        return image

    def random_zoom(self, image: np.ndarray,
                   zoom_range: Tuple[float, float] = (0.9, 1.1),
                   probability: float = 0.5) -> np.ndarray:
        """
        Randomly zoom image.

        Args:
            image: Input image array
            zoom_range: Range of zoom factors
            probability: Probability of applying zoom

        Returns:
            Augmented image
        """
        if random.random() < probability:
            zoom_factor = random.uniform(*zoom_range)
            h, w = image.shape[:2]

            # Calculate new dimensions
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

            # Resize image
            resized = cv2.resize(image, (new_w, new_h))

            # Crop or pad to original size
            if zoom_factor > 1.0:  # Crop
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                return resized[start_h:start_h + h, start_w:start_w + w]
            else:  # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                return cv2.copyMakeBorder(resized, pad_h, h - new_h - pad_h,
                                        pad_w, w - new_w - pad_w,
                                        cv2.BORDER_REFLECT)
        return image

    def random_noise(self, image: np.ndarray,
                    noise_factor: float = 0.1,
                    probability: float = 0.3) -> np.ndarray:
        """
        Add random noise to image.

        Args:
            image: Input image array
            noise_factor: Factor controlling noise intensity
            probability: Probability of applying noise

        Returns:
            Augmented image
        """
        if random.random() < probability:
            noise = np.random.normal(0, noise_factor * 255, image.shape)
            noisy_image = image.astype(np.float32) + noise
            return np.clip(noisy_image, 0, 255).astype(np.uint8)
        return image

    def apply_augmentations(self, image: np.ndarray,
                          augmentation_config: Optional[Dict] = None) -> np.ndarray:
        """
        Apply a sequence of augmentations to an image.

        Args:
            image: Input image array
            augmentation_config: Configuration for augmentations

        Returns:
            Augmented image
        """
        if augmentation_config is None:
            # Default augmentation configuration
            augmentation_config = {
                'flip': {'probability': 0.5},
                'rotation': {'max_angle': 15.0, 'probability': 0.4},
                'brightness': {'brightness_range': (0.8, 1.2), 'probability': 0.4},
                'contrast': {'contrast_range': (0.8, 1.2), 'probability': 0.4},
                'saturation': {'saturation_range': (0.8, 1.2), 'probability': 0.3},
                'zoom': {'zoom_range': (0.9, 1.1), 'probability': 0.3},
                'noise': {'noise_factor': 0.05, 'probability': 0.2}
            }

        augmented_image = image.copy()

        # Apply augmentations
        if 'flip' in augmentation_config:
            augmented_image = self.random_flip(augmented_image, **augmentation_config['flip'])

        if 'rotation' in augmentation_config:
            augmented_image = self.random_rotation(augmented_image, **augmentation_config['rotation'])

        if 'brightness' in augmentation_config:
            augmented_image = self.random_brightness(augmented_image, **augmentation_config['brightness'])

        if 'contrast' in augmentation_config:
            augmented_image = self.random_contrast(augmented_image, **augmentation_config['contrast'])

        if 'saturation' in augmentation_config:
            augmented_image = self.random_saturation(augmented_image, **augmentation_config['saturation'])

        if 'zoom' in augmentation_config:
            augmented_image = self.random_zoom(augmented_image, **augmentation_config['zoom'])

        if 'noise' in augmentation_config:
            augmented_image = self.random_noise(augmented_image, **augmentation_config['noise'])

        return augmented_image

    def get_tensorflow_augmentation_layer(self) -> tf.keras.Sequential:
        """
        Get TensorFlow data augmentation layer.

        Returns:
            TensorFlow Sequential model with augmentation layers
        """
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1)
        ])

    def create_augmented_dataset(self, images: List[np.ndarray],
                               labels: List[int],
                               augmentation_factor: int = 2) -> Tuple[List[np.ndarray], List[int]]:
        """
        Create augmented dataset by applying augmentations to existing images.

        Args:
            images: List of input images
            labels: List of corresponding labels
            augmentation_factor: Number of augmented versions per image

        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []

        for image, label in zip(images, labels):
            # Add original image
            augmented_images.append(image)
            augmented_labels.append(label)

            # Add augmented versions
            for _ in range(augmentation_factor - 1):
                augmented_image = self.apply_augmentations(image)
                augmented_images.append(augmented_image)
                augmented_labels.append(label)

        logging.info(f"Created augmented dataset with {len(augmented_images)} images")
        return augmented_images, augmented_labels
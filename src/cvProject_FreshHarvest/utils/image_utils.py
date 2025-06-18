"""
Image utility functions for the FreshHarvest project.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, List, Union, Optional
import logging
from pathlib import Path


def load_image(image_path: Union[str, Path],
               target_size: Optional[Tuple[int, int]] = None,
               color_mode: str = 'RGB') -> np.ndarray:
    """
    Load and preprocess an image.

    Args:
        image_path: Path to the image file
        target_size: Target size (width, height) for resizing
        color_mode: Color mode ('RGB', 'BGR', 'GRAYSCALE')

    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load image using PIL
        image = Image.open(image_path)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if target size is specified
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(image)

        # Convert color mode if needed
        if color_mode == 'BGR':
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif color_mode == 'GRAYSCALE':
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        return img_array

    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        raise


def save_image(image: np.ndarray,
               save_path: Union[str, Path],
               quality: int = 95) -> None:
    """
    Save image to file.

    Args:
        image: Image array to save
        save_path: Path where to save the image
        quality: JPEG quality (1-100)
    """
    try:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = Image.fromarray(image.astype(np.uint8), mode='L')

        # Save image
        pil_image.save(save_path, quality=quality, optimize=True)

    except Exception as e:
        logging.error(f"Error saving image to {save_path}: {e}")
        raise


def normalize_image(image: np.ndarray,
                   method: str = 'standard') -> np.ndarray:
    """
    Normalize image pixel values.

    Args:
        image: Input image array
        method: Normalization method ('standard', 'minmax', 'imagenet')

    Returns:
        Normalized image array
    """
    image = image.astype(np.float32)

    if method == 'standard':
        # Normalize to [0, 1]
        return image / 255.0
    elif method == 'minmax':
        # Min-max normalization
        return (image - image.min()) / (image.max() - image.min())
    elif method == 'imagenet':
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image / 255.0
        return (image - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize_image(image: np.ndarray,
                     method: str = 'standard') -> np.ndarray:
    """
    Denormalize image pixel values.

    Args:
        image: Normalized image array
        method: Normalization method used ('standard', 'imagenet')

    Returns:
        Denormalized image array
    """
    if method == 'standard':
        return (image * 255.0).astype(np.uint8)
    elif method == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image * std) + mean
        return (image * 255.0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_image(image: np.ndarray,
                target_size: Tuple[int, int],
                interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
    """
    Resize image to target size.

    Args:
        image: Input image array
        target_size: Target size (width, height)
        interpolation: Interpolation method

    Returns:
        Resized image array
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def crop_center(image: np.ndarray,
               crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop image from center.

    Args:
        image: Input image array
        crop_size: Crop size (width, height)

    Returns:
        Cropped image array
    """
    h, w = image.shape[:2]
    crop_w, crop_h = crop_size

    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2

    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]


def get_image_stats(image: np.ndarray) -> dict:
    """
    Get basic statistics of an image.

    Args:
        image: Input image array

    Returns:
        Dictionary containing image statistics
    """
    stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(image.min()),
        'max': float(image.max()),
        'mean': float(image.mean()),
        'std': float(image.std())
    }

    if len(image.shape) == 3:
        stats['channels'] = image.shape[2]
        for i in range(image.shape[2]):
            channel_name = ['R', 'G', 'B'][i] if image.shape[2] == 3 else f'C{i}'
            stats[f'{channel_name}_mean'] = float(image[:, :, i].mean())
            stats[f'{channel_name}_std'] = float(image[:, :, i].std())

    return stats


def enhance_image(image: np.ndarray,
                 brightness: float = 1.0,
                 contrast: float = 1.0,
                 saturation: float = 1.0,
                 sharpness: float = 1.0) -> np.ndarray:
    """
    Enhance image with brightness, contrast, saturation, and sharpness.

    Args:
        image: Input image array
        brightness: Brightness factor (1.0 = no change)
        contrast: Contrast factor (1.0 = no change)
        saturation: Saturation factor (1.0 = no change)
        sharpness: Sharpness factor (1.0 = no change)

    Returns:
        Enhanced image array
    """
    # Convert to PIL Image
    pil_image = Image.fromarray(image.astype(np.uint8))

    # Apply enhancements
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast)

    if saturation != 1.0:
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(saturation)

    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(sharpness)

    return np.array(pil_image)
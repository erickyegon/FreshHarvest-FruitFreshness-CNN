"""
Image Quality Checker for FreshHarvest
=====================================

This module provides comprehensive image quality assessment
for the FreshHarvest fruit freshness classification system.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import cv2
from PIL import Image, ImageStat
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
from skimage import measure, filters
from scipy import ndimage

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from cvProject_FreshHarvest.utils.common import read_yaml

logger = logging.getLogger(__name__)

class ImageQualityChecker:
    """
    Comprehensive image quality assessment for FreshHarvest.

    Evaluates various quality metrics including sharpness, brightness,
    contrast, noise levels, and overall image quality for the 96.50% accuracy model.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize image quality checker.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.quality_config = self.config.get('data_quality', {}).get('image_quality', {})

        # Quality thresholds for 96.50% accuracy model
        self.thresholds = {
            'min_sharpness': self.quality_config.get('min_sharpness', 50),
            'max_noise_level': self.quality_config.get('max_noise_level', 0.1),
            'min_contrast': self.quality_config.get('min_contrast', 0.3),
            'max_saturation': self.quality_config.get('max_saturation', 0.9),
            'min_brightness': self.quality_config.get('min_brightness', 20),
            'max_brightness': self.quality_config.get('max_brightness', 235),
            'min_resolution': self.quality_config.get('min_resolution', [100, 100]),
            'max_resolution': self.quality_config.get('max_resolution', [4000, 4000]),
            'overall_quality_threshold': self.quality_config.get('overall_quality_threshold', 0.7)
        }

        logger.info("Image quality checker initialized for 96.50% accuracy model")

    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.

        Args:
            image: Input image

        Returns:
            Sharpness score (higher = sharper)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()

            return float(sharpness)

        except Exception as e:
            logger.error(f"Error calculating sharpness: {e}")
            return 0.0

    def calculate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate noise level in the image.

        Args:
            image: Input image

        Returns:
            Noise level (0-1, lower = less noise)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Apply Gaussian blur and calculate difference
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)

            # Normalize noise level
            noise_level = np.mean(noise) / 255.0

            return float(noise_level)

        except Exception as e:
            logger.error(f"Error calculating noise level: {e}")
            return 1.0

    def calculate_contrast(self, image: np.ndarray) -> float:
        """
        Calculate image contrast using standard deviation.

        Args:
            image: Input image

        Returns:
            Contrast score (0-1, higher = more contrast)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Calculate contrast as normalized standard deviation
            contrast = np.std(gray) / 255.0

            return float(contrast)

        except Exception as e:
            logger.error(f"Error calculating contrast: {e}")
            return 0.0

    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        Calculate average brightness of the image.

        Args:
            image: Input image

        Returns:
            Average brightness (0-255)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            brightness = np.mean(gray)

            return float(brightness)

        except Exception as e:
            logger.error(f"Error calculating brightness: {e}")
            return 0.0

    def calculate_saturation(self, image: np.ndarray) -> float:
        """
        Calculate color saturation of the image.

        Args:
            image: Input image (RGB)

        Returns:
            Average saturation (0-1)
        """
        try:
            if len(image.shape) != 3:
                return 0.0

            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # Extract saturation channel and normalize
            saturation = hsv[:, :, 1] / 255.0
            avg_saturation = np.mean(saturation)

            return float(avg_saturation)

        except Exception as e:
            logger.error(f"Error calculating saturation: {e}")
            return 0.0

    def detect_blur(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect if image is blurry using multiple methods.

        Args:
            image: Input image

        Returns:
            Blur detection results
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Method 1: Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Method 2: Sobel gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            sobel_mean = np.mean(sobel_magnitude)

            # Method 3: FFT-based blur detection
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            high_freq_energy = np.mean(magnitude_spectrum[gray.shape[0]//4:3*gray.shape[0]//4,
                                                        gray.shape[1]//4:3*gray.shape[1]//4])

            # Determine if blurry
            is_blurry = (laplacian_var < self.thresholds['min_sharpness'] or
                        sobel_mean < 10 or
                        high_freq_energy < 5)

            return {
                'is_blurry': is_blurry,
                'laplacian_variance': float(laplacian_var),
                'sobel_magnitude': float(sobel_mean),
                'high_freq_energy': float(high_freq_energy),
                'blur_confidence': 1.0 - min(laplacian_var / 100.0, 1.0)
            }

        except Exception as e:
            logger.error(f"Error detecting blur: {e}")
            return {
                'is_blurry': True,
                'error': str(e)
            }

    def check_resolution(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Check if image resolution meets requirements.

        Args:
            image: Input image

        Returns:
            Resolution check results
        """
        try:
            height, width = image.shape[:2]

            min_h, min_w = self.thresholds['min_resolution']
            max_h, max_w = self.thresholds['max_resolution']

            resolution_ok = (width >= min_w and height >= min_h and
                           width <= max_w and height <= max_h)

            return {
                'resolution_ok': resolution_ok,
                'width': width,
                'height': height,
                'min_required': self.thresholds['min_resolution'],
                'max_allowed': self.thresholds['max_resolution'],
                'aspect_ratio': width / height,
                'total_pixels': width * height
            }

        except Exception as e:
            logger.error(f"Error checking resolution: {e}")
            return {
                'resolution_ok': False,
                'error': str(e)
            }

    def detect_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect compression artifacts and other image defects.

        Args:
            image: Input image

        Returns:
            Artifact detection results
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Detect blocking artifacts (JPEG compression)
            # Apply DCT to detect 8x8 block patterns
            blocks = []
            for i in range(0, gray.shape[0] - 8, 8):
                for j in range(0, gray.shape[1] - 8, 8):
                    block = gray[i:i+8, j:j+8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    blocks.append(np.std(dct_block))

            blocking_score = np.std(blocks) if blocks else 0

            # Detect ringing artifacts
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])

            # Simple artifact detection
            has_artifacts = blocking_score > 50 or edge_density > 0.1

            return {
                'has_artifacts': has_artifacts,
                'blocking_score': float(blocking_score),
                'edge_density': float(edge_density),
                'artifact_confidence': min(blocking_score / 100.0, 1.0)
            }

        except Exception as e:
            logger.error(f"Error detecting artifacts: {e}")
            return {
                'has_artifacts': False,
                'error': str(e)
            }

    def assess_overall_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive quality assessment of the image.

        Args:
            image: Input image

        Returns:
            Complete quality assessment results
        """
        try:
            # Calculate individual metrics
            sharpness = self.calculate_sharpness(image)
            noise_level = self.calculate_noise_level(image)
            contrast = self.calculate_contrast(image)
            brightness = self.calculate_brightness(image)
            saturation = self.calculate_saturation(image)

            # Additional checks
            blur_results = self.detect_blur(image)
            resolution_results = self.check_resolution(image)
            artifact_results = self.detect_artifacts(image)

            # Calculate quality scores (0-1, higher = better)
            sharpness_score = min(sharpness / 100.0, 1.0)
            noise_score = 1.0 - min(noise_level / self.thresholds['max_noise_level'], 1.0)
            contrast_score = min(contrast / 0.5, 1.0)  # Normalize to 0.5 as good contrast

            # Brightness score (penalize too dark or too bright)
            brightness_score = 1.0 - abs(brightness - 127.5) / 127.5

            # Saturation score (moderate saturation is good)
            saturation_score = 1.0 - abs(saturation - 0.5) / 0.5

            # Resolution score
            resolution_score = 1.0 if resolution_results['resolution_ok'] else 0.5

            # Artifact score
            artifact_score = 0.0 if artifact_results['has_artifacts'] else 1.0

            # Blur score
            blur_score = 0.0 if blur_results['is_blurry'] else 1.0

            # Calculate weighted overall quality score
            weights = {
                'sharpness': 0.25,
                'noise': 0.15,
                'contrast': 0.15,
                'brightness': 0.10,
                'saturation': 0.10,
                'resolution': 0.10,
                'artifacts': 0.10,
                'blur': 0.05
            }

            overall_score = (
                weights['sharpness'] * sharpness_score +
                weights['noise'] * noise_score +
                weights['contrast'] * contrast_score +
                weights['brightness'] * brightness_score +
                weights['saturation'] * saturation_score +
                weights['resolution'] * resolution_score +
                weights['artifacts'] * artifact_score +
                weights['blur'] * blur_score
            )

            # Determine if image passes quality check
            quality_passed = overall_score >= self.thresholds['overall_quality_threshold']

            # Quality assessment
            if overall_score >= 0.8:
                quality_grade = "Excellent"
            elif overall_score >= 0.7:
                quality_grade = "Good"
            elif overall_score >= 0.5:
                quality_grade = "Fair"
            else:
                quality_grade = "Poor"

            return {
                'overall_quality_score': float(overall_score),
                'quality_grade': quality_grade,
                'quality_passed': quality_passed,
                'individual_scores': {
                    'sharpness': float(sharpness_score),
                    'noise': float(noise_score),
                    'contrast': float(contrast_score),
                    'brightness': float(brightness_score),
                    'saturation': float(saturation_score),
                    'resolution': float(resolution_score),
                    'artifacts': float(artifact_score),
                    'blur': float(blur_score)
                },
                'raw_metrics': {
                    'sharpness': float(sharpness),
                    'noise_level': float(noise_level),
                    'contrast': float(contrast),
                    'brightness': float(brightness),
                    'saturation': float(saturation)
                },
                'detailed_results': {
                    'blur_detection': blur_results,
                    'resolution_check': resolution_results,
                    'artifact_detection': artifact_results
                },
                'recommendations': self._generate_recommendations(overall_score, {
                    'sharpness': sharpness_score,
                    'noise': noise_score,
                    'contrast': contrast_score,
                    'brightness': brightness_score,
                    'blur': blur_score
                })
            }

        except Exception as e:
            logger.error(f"Error in overall quality assessment: {e}")
            return {
                'overall_quality_score': 0.0,
                'quality_grade': "Error",
                'quality_passed': False,
                'error': str(e)
            }

    def _generate_recommendations(self, overall_score: float, scores: Dict[str, float]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        if overall_score < 0.7:
            recommendations.append("Image quality below recommended threshold for 96.50% accuracy model")

        if scores['sharpness'] < 0.5:
            recommendations.append("Image appears blurry - ensure proper focus when capturing")

        if scores['noise'] < 0.7:
            recommendations.append("High noise levels detected - use better lighting or lower ISO")

        if scores['contrast'] < 0.5:
            recommendations.append("Low contrast - adjust lighting or camera settings")

        if scores['brightness'] < 0.5:
            recommendations.append("Brightness issues - ensure adequate and even lighting")

        if scores['blur'] < 0.5:
            recommendations.append("Motion blur detected - use faster shutter speed or stabilization")

        if not recommendations:
            recommendations.append("Image quality is good for the 96.50% accuracy model")

        return recommendations

    def batch_quality_check(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Perform quality check on a batch of images.

        Args:
            image_paths: List of image file paths

        Returns:
            Batch quality assessment results
        """
        try:
            results = []
            summary_stats = {
                'total_images': len(image_paths),
                'passed_quality': 0,
                'failed_quality': 0,
                'average_quality_score': 0.0,
                'quality_distribution': {'Excellent': 0, 'Good': 0, 'Fair': 0, 'Poor': 0}
            }

            for i, image_path in enumerate(image_paths):
                logger.info(f"Checking quality for image {i+1}/{len(image_paths)}: {image_path}")

                try:
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        result = {
                            'image_path': image_path,
                            'error': 'Failed to load image'
                        }
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        result = self.assess_overall_quality(image)
                        result['image_path'] = image_path

                        # Update summary stats
                        if result['quality_passed']:
                            summary_stats['passed_quality'] += 1
                        else:
                            summary_stats['failed_quality'] += 1

                        summary_stats['average_quality_score'] += result['overall_quality_score']
                        summary_stats['quality_distribution'][result['quality_grade']] += 1

                    results.append(result)

                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    results.append({
                        'image_path': image_path,
                        'error': str(e)
                    })

            # Calculate final summary stats
            if summary_stats['total_images'] > 0:
                summary_stats['average_quality_score'] /= summary_stats['total_images']
                summary_stats['pass_rate'] = summary_stats['passed_quality'] / summary_stats['total_images']

            return {
                'summary': summary_stats,
                'individual_results': results,
                'model_accuracy': '96.50%',
                'quality_threshold': self.thresholds['overall_quality_threshold']
            }

        except Exception as e:
            logger.error(f"Error in batch quality check: {e}")
            return {'error': str(e)}

def check_image_quality(image_path: str, config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Convenience function to check quality of a single image.

    Args:
        image_path: Path to image file
        config_path: Path to configuration file

    Returns:
        Quality assessment results
    """
    try:
        # Initialize quality checker
        checker = ImageQualityChecker(config_path)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Failed to load image: {image_path}'}

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Assess quality
        results = checker.assess_overall_quality(image)
        results['image_path'] = image_path

        return results

    except Exception as e:
        logger.error(f"Error checking image quality: {e}")
        return {'error': str(e)}
"""
Feature Extraction Component for FreshHarvest Classification System
=================================================================

This module provides comprehensive feature extraction functionality
including traditional computer vision features and deep learning
feature extraction for fruit freshness classification.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import feature, measure, filters
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..utils.common import read_yaml

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Comprehensive feature extraction for FreshHarvest image classification.

    Extracts both traditional computer vision features and deep learning
    features for optimal fruit freshness classification.
    """

    def __init__(self, config_path: str):
        """
        Initialize feature extractor.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.data_config = self.config['data']

        # Image specifications
        self.image_size = tuple(self.data_config['image_size'])
        self.channels = self.data_config['channels']

        # Feature extraction settings
        self.feature_config = self.config.get('feature_extraction', {})
        self.extract_traditional = self.feature_config.get('traditional_features', True)
        self.extract_deep = self.feature_config.get('deep_features', True)

        # Initialize feature extractors
        self.deep_feature_extractor = None
        self.scaler = StandardScaler()

        logger.info("Feature extractor initialized")

    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract color-based features from image.

        Args:
            image: Input image array (RGB)

        Returns:
            Dictionary containing color features
        """
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            features = {}

            # RGB statistics
            for i, channel in enumerate(['red', 'green', 'blue']):
                channel_data = image[:, :, i].flatten()
                features[f'rgb_{channel}_mean'] = float(np.mean(channel_data))
                features[f'rgb_{channel}_std'] = float(np.std(channel_data))
                features[f'rgb_{channel}_skew'] = float(self._calculate_skewness(channel_data))
                features[f'rgb_{channel}_kurtosis'] = float(self._calculate_kurtosis(channel_data))

            # HSV statistics
            for i, channel in enumerate(['hue', 'saturation', 'value']):
                channel_data = hsv[:, :, i].flatten()
                features[f'hsv_{channel}_mean'] = float(np.mean(channel_data))
                features[f'hsv_{channel}_std'] = float(np.std(channel_data))

            # LAB statistics
            for i, channel in enumerate(['l', 'a', 'b']):
                channel_data = lab[:, :, i].flatten()
                features[f'lab_{channel}_mean'] = float(np.mean(channel_data))
                features[f'lab_{channel}_std'] = float(np.std(channel_data))

            # Color ratios and relationships
            features['rg_ratio'] = float(np.mean(image[:, :, 0]) / (np.mean(image[:, :, 1]) + 1e-8))
            features['rb_ratio'] = float(np.mean(image[:, :, 0]) / (np.mean(image[:, :, 2]) + 1e-8))
            features['gb_ratio'] = float(np.mean(image[:, :, 1]) / (np.mean(image[:, :, 2]) + 1e-8))

            # Color dominance
            rgb_means = [np.mean(image[:, :, i]) for i in range(3)]
            dominant_color = np.argmax(rgb_means)
            features['dominant_color'] = float(dominant_color)
            features['color_variance'] = float(np.var(rgb_means))

            return features

        except Exception as e:
            logger.error(f"Error extracting color features: {e}")
            return {}

    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture-based features from image.

        Args:
            image: Input image array

        Returns:
            Dictionary containing texture features
        """
        try:
            # Convert to grayscale for texture analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            features = {}

            # Local Binary Pattern (LBP)
            radius = 3
            n_points = 8 * radius
            lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-8)

            for i, val in enumerate(lbp_hist):
                features[f'lbp_bin_{i}'] = float(val)

            # Gray Level Co-occurrence Matrix (GLCM) features
            try:
                from skimage.feature import greycomatrix, greycoprops

                # Calculate GLCM
                distances = [1, 2, 3]
                angles = [0, 45, 90, 135]

                glcm = greycomatrix(gray, distances=distances, angles=angles,
                                  levels=256, symmetric=True, normed=True)

                # Extract GLCM properties
                properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
                for prop in properties:
                    prop_values = greycoprops(glcm, prop)
                    features[f'glcm_{prop}_mean'] = float(np.mean(prop_values))
                    features[f'glcm_{prop}_std'] = float(np.std(prop_values))

            except ImportError:
                logger.warning("scikit-image not available for GLCM features")

            # Edge features
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / edges.size)
            features['edge_mean'] = float(np.mean(edges))
            features['edge_std'] = float(np.std(edges))

            # Gradient features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            features['gradient_mean'] = float(np.mean(gradient_magnitude))
            features['gradient_std'] = float(np.std(gradient_magnitude))
            features['gradient_max'] = float(np.max(gradient_magnitude))

            # Laplacian variance (measure of blurriness)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features['laplacian_variance'] = float(np.var(laplacian))

            return features

        except Exception as e:
            logger.error(f"Error extracting texture features: {e}")
            return {}

    def extract_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract shape-based features from image.

        Args:
            image: Input image array

        Returns:
            Dictionary containing shape features
        """
        try:
            # Convert to grayscale and create binary mask
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Threshold to create binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            features = {}

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get largest contour (assuming it's the main object)
                largest_contour = max(contours, key=cv2.contourArea)

                # Basic shape measurements
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)

                features['contour_area'] = float(area)
                features['contour_perimeter'] = float(perimeter)
                features['circularity'] = float(4 * np.pi * area / (perimeter**2 + 1e-8))

                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                features['aspect_ratio'] = float(w / (h + 1e-8))
                features['extent'] = float(area / (w * h + 1e-8))

                # Convex hull
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                features['solidity'] = float(area / (hull_area + 1e-8))

                # Moments
                moments = cv2.moments(largest_contour)
                if moments['m00'] != 0:
                    features['centroid_x'] = float(moments['m10'] / moments['m00'])
                    features['centroid_y'] = float(moments['m01'] / moments['m00'])
                else:
                    features['centroid_x'] = 0.0
                    features['centroid_y'] = 0.0

                # Hu moments (shape descriptors)
                hu_moments = cv2.HuMoments(moments)
                for i, hu in enumerate(hu_moments.flatten()):
                    features[f'hu_moment_{i}'] = float(-np.sign(hu) * np.log10(np.abs(hu) + 1e-8))

            else:
                # No contours found, set default values
                default_features = [
                    'contour_area', 'contour_perimeter', 'circularity',
                    'aspect_ratio', 'extent', 'solidity', 'centroid_x', 'centroid_y'
                ]
                for feat in default_features:
                    features[feat] = 0.0

                for i in range(7):
                    features[f'hu_moment_{i}'] = 0.0

            # Image moments
            image_moments = cv2.moments(gray)
            features['image_area'] = float(image_moments['m00'])

            return features

        except Exception as e:
            logger.error(f"Error extracting shape features: {e}")
            return {}

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

    def extract_deep_features(self, image: np.ndarray,
                             model_name: str = 'resnet50') -> np.ndarray:
        """
        Extract deep learning features using pre-trained models.

        Args:
            image: Input image array
            model_name: Name of pre-trained model to use

        Returns:
            Deep feature vector
        """
        try:
            if self.deep_feature_extractor is None:
                self._initialize_deep_extractor(model_name)

            # Preprocess image for the model
            if len(image.shape) == 3:
                # Ensure image is in correct format
                processed_image = cv2.resize(image, (224, 224))
                processed_image = np.expand_dims(processed_image, axis=0)
                processed_image = processed_image.astype(np.float32) / 255.0
            else:
                raise ValueError("Image must be 3-channel RGB")

            # Extract features
            features = self.deep_feature_extractor.predict(processed_image, verbose=0)
            return features.flatten()

        except Exception as e:
            logger.error(f"Error extracting deep features: {e}")
            return np.array([])

    def _initialize_deep_extractor(self, model_name: str):
        """Initialize deep feature extractor model."""
        try:
            if model_name.lower() == 'resnet50':
                base_model = tf.keras.applications.ResNet50(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )
            elif model_name.lower() == 'vgg16':
                base_model = tf.keras.applications.VGG16(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )
            elif model_name.lower() == 'mobilenet':
                base_model = tf.keras.applications.MobileNetV2(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )
            else:
                logger.warning(f"Unknown model {model_name}, using ResNet50")
                base_model = tf.keras.applications.ResNet50(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )

            self.deep_feature_extractor = base_model
            logger.info(f"Deep feature extractor initialized: {model_name}")

        except Exception as e:
            logger.error(f"Error initializing deep extractor: {e}")

    def extract_all_features(self, image: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract all available features from an image.

        Args:
            image: Input image array

        Returns:
            Dictionary containing all extracted features
        """
        try:
            all_features = {}

            # Extract traditional features
            if self.extract_traditional:
                color_features = self.extract_color_features(image)
                texture_features = self.extract_texture_features(image)
                shape_features = self.extract_shape_features(image)

                all_features.update(color_features)
                all_features.update(texture_features)
                all_features.update(shape_features)

            # Extract deep features
            if self.extract_deep:
                deep_features = self.extract_deep_features(image)
                all_features['deep_features'] = deep_features

            return all_features

        except Exception as e:
            logger.error(f"Error extracting all features: {e}")
            return {}

    def extract_features_batch(self, images: List[np.ndarray]) -> List[Dict[str, Union[float, np.ndarray]]]:
        """
        Extract features from a batch of images.

        Args:
            images: List of input images

        Returns:
            List of feature dictionaries
        """
        try:
            batch_features = []

            for i, image in enumerate(images):
                features = self.extract_all_features(image)
                batch_features.append(features)

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(images)} images")

            logger.info(f"Feature extraction completed for {len(images)} images")
            return batch_features

        except Exception as e:
            logger.error(f"Error in batch feature extraction: {e}")
            return []

    def create_feature_matrix(self, feature_dicts: List[Dict[str, Union[float, np.ndarray]]]) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix from list of feature dictionaries.

        Args:
            feature_dicts: List of feature dictionaries

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        try:
            if not feature_dicts:
                return np.array([]), []

            # Separate traditional and deep features
            traditional_features = []
            deep_features = []
            feature_names = []

            # Get traditional feature names from first dictionary
            first_dict = feature_dicts[0]
            traditional_keys = [k for k in first_dict.keys() if k != 'deep_features']
            feature_names.extend(traditional_keys)

            # Extract traditional features
            for feat_dict in feature_dicts:
                trad_feat = [feat_dict.get(key, 0.0) for key in traditional_keys]
                traditional_features.append(trad_feat)

                # Extract deep features if available
                if 'deep_features' in feat_dict:
                    deep_feat = feat_dict['deep_features']
                    if len(deep_feat) > 0:
                        deep_features.append(deep_feat)

            # Combine features
            feature_matrix = np.array(traditional_features)

            if deep_features:
                deep_matrix = np.array(deep_features)
                feature_matrix = np.hstack([feature_matrix, deep_matrix])

                # Add deep feature names
                deep_feature_names = [f'deep_feat_{i}' for i in range(deep_matrix.shape[1])]
                feature_names.extend(deep_feature_names)

            logger.info(f"Feature matrix created: {feature_matrix.shape}")
            return feature_matrix, feature_names

        except Exception as e:
            logger.error(f"Error creating feature matrix: {e}")
            return np.array([]), []

    def reduce_features(self, feature_matrix: np.ndarray,
                       n_components: int = 50,
                       method: str = 'pca') -> Tuple[np.ndarray, Any]:
        """
        Reduce feature dimensionality.

        Args:
            feature_matrix: Input feature matrix
            n_components: Number of components to keep
            method: Dimensionality reduction method ('pca')

        Returns:
            Tuple of (reduced_features, reducer_object)
        """
        try:
            if method.lower() == 'pca':
                reducer = PCA(n_components=n_components)
                reduced_features = reducer.fit_transform(feature_matrix)

                logger.info(f"PCA reduction: {feature_matrix.shape} -> {reduced_features.shape}")
                logger.info(f"Explained variance ratio: {np.sum(reducer.explained_variance_ratio_):.3f}")

                return reduced_features, reducer
            else:
                logger.error(f"Unknown reduction method: {method}")
                return feature_matrix, None

        except Exception as e:
            logger.error(f"Error in feature reduction: {e}")
            return feature_matrix, None

    def normalize_features(self, feature_matrix: np.ndarray,
                          fit_scaler: bool = True) -> np.ndarray:
        """
        Normalize feature matrix.

        Args:
            feature_matrix: Input feature matrix
            fit_scaler: Whether to fit the scaler (True for training, False for inference)

        Returns:
            Normalized feature matrix
        """
        try:
            if fit_scaler:
                normalized_features = self.scaler.fit_transform(feature_matrix)
                logger.info("Feature scaler fitted and applied")
            else:
                normalized_features = self.scaler.transform(feature_matrix)
                logger.info("Feature scaler applied")

            return normalized_features

        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return feature_matrix

def extract_features_from_images(image_paths: List[str],
                                config_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to extract features from image paths.

    Args:
        image_paths: List of image file paths
        config_path: Path to configuration file

    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    extractor = FeatureExtractor(config_path)

    # Load and process images
    images = []
    for img_path in image_paths:
        try:
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")

    # Extract features
    feature_dicts = extractor.extract_features_batch(images)
    feature_matrix, feature_names = extractor.create_feature_matrix(feature_dicts)

    return feature_matrix, feature_names
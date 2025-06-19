"""
Model Deployment Component for FreshHarvest Classification System
===============================================================

This module provides comprehensive model deployment functionality
including model serving, API endpoints, and production deployment.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import pickle
from pathlib import Path
import time
import os
from datetime import datetime

from ..utils.common import read_yaml, create_directories

logger = logging.getLogger(__name__)

class ModelDeployment:
    """
    Comprehensive model deployment for FreshHarvest classification.

    Handles model loading, serving, API creation, and production deployment
    with monitoring and health checks.
    """

    def __init__(self, config_path: str):
        """
        Initialize model deployment.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.model_config = self.config['model']
        self.deployment_config = self.config.get('deployment', {})

        # Model artifacts
        self.model = None
        self.preprocessor = None
        self.class_names = None
        self.model_metadata = {}

        # Deployment settings
        self.model_path = self.deployment_config.get('model_path', 'artifacts/model_trainer/model.h5')
        self.serving_port = self.deployment_config.get('port', 8080)
        self.max_batch_size = self.deployment_config.get('max_batch_size', 32)

        logger.info("Model deployment component initialized")

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load trained model for deployment.

        Args:
            model_path: Path to model file

        Returns:
            True if successful, False otherwise
        """
        try:
            if model_path is None:
                model_path = self.model_path

            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            # Load TensorFlow model
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")

            # Load model metadata
            metadata_path = Path(model_path).parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("Model metadata loaded")

            # Set class names
            self.class_names = [
                "Fresh Apple", "Fresh Banana", "Fresh Orange",
                "Rotten Apple", "Rotten Banana", "Rotten Orange"
            ]

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def preprocess_input(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess input image for model inference.

        Args:
            image: Input image array

        Returns:
            Preprocessed image ready for model
        """
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3:
                # Add batch dimension if needed
                if image.shape[0] != 1:
                    image = np.expand_dims(image, axis=0)
            elif len(image.shape) == 4:
                # Already has batch dimension
                pass
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")

            # Resize to model input size
            target_size = (224, 224)
            if image.shape[1:3] != target_size:
                # Resize using TensorFlow
                image = tf.image.resize(image, target_size)
                image = tf.cast(image, tf.float32)

            # Normalize to [0, 1]
            if image.dtype != tf.float32:
                image = tf.cast(image, tf.float32)

            if tf.reduce_max(image) > 1.0:
                image = image / 255.0

            return image.numpy()

        except Exception as e:
            logger.error(f"Error preprocessing input: {e}")
            return None

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction on input image.

        Args:
            image: Input image array

        Returns:
            Dictionary containing prediction results
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")

            start_time = time.time()

            # Preprocess input
            processed_image = self.preprocess_input(image)
            if processed_image is None:
                raise ValueError("Failed to preprocess input image")

            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)

            # Process predictions
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            # Get all class probabilities
            class_probabilities = {}
            for i, class_name in enumerate(self.class_names):
                class_probabilities[class_name] = float(predictions[0][i])

            # Calculate inference time
            inference_time = time.time() - start_time

            # Determine freshness
            is_fresh = predicted_class < 3  # First 3 classes are fresh
            freshness_confidence = confidence if is_fresh else 1 - confidence

            results = {
                'predicted_class': self.class_names[predicted_class],
                'predicted_class_id': int(predicted_class),
                'confidence': confidence,
                'is_fresh': is_fresh,
                'freshness_confidence': freshness_confidence,
                'class_probabilities': class_probabilities,
                'inference_time_ms': inference_time * 1000,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Prediction completed: {results['predicted_class']} ({confidence:.3f})")
            return results

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Make predictions on batch of images.

        Args:
            images: List of input images

        Returns:
            List of prediction results
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")

            if len(images) > self.max_batch_size:
                logger.warning(f"Batch size {len(images)} exceeds maximum {self.max_batch_size}")
                # Process in chunks
                results = []
                for i in range(0, len(images), self.max_batch_size):
                    chunk = images[i:i + self.max_batch_size]
                    chunk_results = self.predict_batch(chunk)
                    results.extend(chunk_results)
                return results

            start_time = time.time()

            # Preprocess all images
            processed_images = []
            valid_indices = []

            for i, image in enumerate(images):
                processed = self.preprocess_input(image)
                if processed is not None:
                    processed_images.append(processed[0])  # Remove batch dimension
                    valid_indices.append(i)

            if not processed_images:
                logger.error("No valid images in batch")
                return []

            # Stack images for batch prediction
            batch_images = np.array(processed_images)

            # Make batch prediction
            predictions = self.model.predict(batch_images, verbose=0)

            # Process results
            results = []
            for i, pred in enumerate(predictions):
                predicted_class = np.argmax(pred)
                confidence = float(pred[predicted_class])

                class_probabilities = {}
                for j, class_name in enumerate(self.class_names):
                    class_probabilities[class_name] = float(pred[j])

                is_fresh = predicted_class < 3
                freshness_confidence = confidence if is_fresh else 1 - confidence

                result = {
                    'predicted_class': self.class_names[predicted_class],
                    'predicted_class_id': int(predicted_class),
                    'confidence': confidence,
                    'is_fresh': is_fresh,
                    'freshness_confidence': freshness_confidence,
                    'class_probabilities': class_probabilities,
                    'batch_index': valid_indices[i],
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)

            batch_time = time.time() - start_time
            logger.info(f"Batch prediction completed: {len(results)} images in {batch_time:.3f}s")

            return results

        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on deployed model.

        Returns:
            Health status dictionary
        """
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'model_loaded': self.model is not None,
                'model_metadata': self.model_metadata,
                'uptime': time.time(),
                'memory_usage': self._get_memory_usage(),
                'version': '1.0.0'
            }

            # Test prediction with dummy data
            if self.model is not None:
                try:
                    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
                    start_time = time.time()
                    _ = self.model.predict(dummy_input, verbose=0)
                    test_time = time.time() - start_time

                    health_status['test_prediction_time_ms'] = test_time * 1000
                    health_status['model_responsive'] = True

                except Exception as e:
                    health_status['status'] = 'unhealthy'
                    health_status['model_responsive'] = False
                    health_status['error'] = str(e)

            return health_status

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}

    def save_deployment_config(self, save_path: str) -> bool:
        """
        Save deployment configuration.

        Args:
            save_path: Path to save configuration

        Returns:
            True if successful
        """
        try:
            deployment_info = {
                'model_path': self.model_path,
                'class_names': self.class_names,
                'model_metadata': self.model_metadata,
                'deployment_timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'config': self.deployment_config
            }

            with open(save_path, 'w') as f:
                json.dump(deployment_info, f, indent=2)

            logger.info(f"Deployment config saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving deployment config: {e}")
            return False

def deploy_model(model_path: str, config_path: str) -> ModelDeployment:
    """
    Convenience function to deploy a model.

    Args:
        model_path: Path to trained model
        config_path: Path to configuration file

    Returns:
        Deployed model instance
    """
    deployment = ModelDeployment(config_path)

    if deployment.load_model(model_path):
        logger.info("Model deployed successfully")
        return deployment
    else:
        logger.error("Failed to deploy model")
        return None
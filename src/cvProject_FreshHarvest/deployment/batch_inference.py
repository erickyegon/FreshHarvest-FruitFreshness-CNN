"""
Batch Inference for FreshHarvest Model
=====================================

This module provides batch processing capabilities for the FreshHarvest
fruit freshness classification system.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from cvProject_FreshHarvest.components.model_deployment import ModelDeployment
from cvProject_FreshHarvest.utils.common import read_yaml, create_directories

logger = logging.getLogger(__name__)

class BatchInference:
    """
    Batch inference processor for FreshHarvest model.

    Handles large-scale batch processing of fruit images
    with parallel processing and progress tracking.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize batch inference processor.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.batch_config = self.config.get('batch_inference', {})

        # Batch processing settings
        self.batch_size = self.batch_config.get('batch_size', 32)
        self.max_workers = self.batch_config.get('max_workers', 4)
        self.output_format = self.batch_config.get('output_format', 'json')

        # Initialize model deployment
        self.deployment = ModelDeployment(config_path)
        self.model_loaded = False

        # Results storage
        self.results = []
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }

        logger.info("Batch inference processor initialized")

    def load_model(self) -> bool:
        """Load the model for batch inference."""
        try:
            self.model_loaded = self.deployment.load_model()
            if self.model_loaded:
                logger.info("✅ Model loaded for batch inference")
            else:
                logger.error("❌ Failed to load model")
            return self.model_loaded
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def load_images_from_directory(self, directory_path: str,
                                 supported_formats: List[str] = None) -> List[Dict[str, Any]]:
        """
        Load images from directory.

        Args:
            directory_path: Path to directory containing images
            supported_formats: List of supported image formats

        Returns:
            List of image information dictionaries
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        image_files = []
        directory = Path(directory_path)

        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []

        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                image_files.append({
                    'path': str(file_path),
                    'filename': file_path.name,
                    'relative_path': str(file_path.relative_to(directory)),
                    'size': file_path.stat().st_size
                })

        logger.info(f"Found {len(image_files)} images in {directory_path}")
        return image_files

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image array or None if failed
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def process_single_image(self, image_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single image.

        Args:
            image_info: Image information dictionary

        Returns:
            Processing result dictionary
        """
        try:
            start_time = time.time()

            # Load image
            image = self.load_image(image_info['path'])
            if image is None:
                return {
                    'filename': image_info['filename'],
                    'path': image_info['path'],
                    'status': 'failed',
                    'error': 'Failed to load image',
                    'processing_time': time.time() - start_time
                }

            # Make prediction
            prediction_result = self.deployment.predict(image)

            if 'error' in prediction_result:
                return {
                    'filename': image_info['filename'],
                    'path': image_info['path'],
                    'status': 'failed',
                    'error': prediction_result['error'],
                    'processing_time': time.time() - start_time
                }

            # Format result
            result = {
                'filename': image_info['filename'],
                'path': image_info['path'],
                'relative_path': image_info['relative_path'],
                'status': 'success',
                'prediction': {
                    'class': prediction_result['predicted_class'],
                    'class_id': prediction_result['predicted_class_id'],
                    'confidence': prediction_result['confidence'],
                    'is_fresh': prediction_result['is_fresh'],
                    'freshness_confidence': prediction_result['freshness_confidence']
                },
                'class_probabilities': prediction_result['class_probabilities'],
                'inference_time_ms': prediction_result['inference_time_ms'],
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error processing image {image_info['filename']}: {e}")
            return {
                'filename': image_info['filename'],
                'path': image_info['path'],
                'status': 'failed',
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    def process_batch(self, image_files: List[Dict[str, Any]],
                     parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of images.

        Args:
            image_files: List of image file information
            parallel: Whether to use parallel processing

        Returns:
            List of processing results
        """
        if not self.model_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return []

        self.processing_stats['start_time'] = datetime.now()
        self.processing_stats['total_processed'] = 0
        self.processing_stats['successful'] = 0
        self.processing_stats['failed'] = 0

        results = []

        if parallel and len(image_files) > 1:
            # Parallel processing
            logger.info(f"Processing {len(image_files)} images in parallel (workers: {self.max_workers})")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(self.process_single_image, image_info): image_info
                    for image_info in image_files
                }

                # Collect results
                for future in as_completed(future_to_image):
                    result = future.result()
                    results.append(result)

                    # Update stats
                    self.processing_stats['total_processed'] += 1
                    if result['status'] == 'success':
                        self.processing_stats['successful'] += 1
                    else:
                        self.processing_stats['failed'] += 1

                    # Progress logging
                    if self.processing_stats['total_processed'] % 100 == 0:
                        logger.info(f"Processed {self.processing_stats['total_processed']}/{len(image_files)} images")

        else:
            # Sequential processing
            logger.info(f"Processing {len(image_files)} images sequentially")

            for i, image_info in enumerate(image_files):
                result = self.process_single_image(image_info)
                results.append(result)

                # Update stats
                self.processing_stats['total_processed'] += 1
                if result['status'] == 'success':
                    self.processing_stats['successful'] += 1
                else:
                    self.processing_stats['failed'] += 1

                # Progress logging
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_files)} images")

        self.processing_stats['end_time'] = datetime.now()
        self.results = results

        logger.info(f"Batch processing completed: {self.processing_stats['successful']} successful, {self.processing_stats['failed']} failed")

        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Save processing results to file.

        Args:
            results: List of processing results
            output_path: Path to save results

        Returns:
            True if successful, False otherwise
        """
        try:
            output_file = Path(output_path)
            create_directories([str(output_file.parent)])

            if self.output_format.lower() == 'json':
                # Save as JSON
                output_data = {
                    'metadata': {
                        'total_images': len(results),
                        'successful': self.processing_stats['successful'],
                        'failed': self.processing_stats['failed'],
                        'processing_time': str(self.processing_stats['end_time'] - self.processing_stats['start_time']),
                        'model_accuracy': '96.50%',
                        'timestamp': datetime.now().isoformat()
                    },
                    'results': results
                }

                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)

            elif self.output_format.lower() == 'csv':
                # Save as CSV
                df_data = []
                for result in results:
                    if result['status'] == 'success':
                        row = {
                            'filename': result['filename'],
                            'path': result['path'],
                            'predicted_class': result['prediction']['class'],
                            'confidence': result['prediction']['confidence'],
                            'is_fresh': result['prediction']['is_fresh'],
                            'freshness_confidence': result['prediction']['freshness_confidence'],
                            'inference_time_ms': result['inference_time_ms'],
                            'processing_time': result['processing_time'],
                            'timestamp': result['timestamp']
                        }
                    else:
                        row = {
                            'filename': result['filename'],
                            'path': result['path'],
                            'predicted_class': 'ERROR',
                            'confidence': 0.0,
                            'is_fresh': False,
                            'freshness_confidence': 0.0,
                            'inference_time_ms': 0.0,
                            'processing_time': result['processing_time'],
                            'timestamp': result.get('timestamp', datetime.now().isoformat())
                        }
                    df_data.append(row)

                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)

            logger.info(f"Results saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of batch processing."""

        if not self.results:
            return {'error': 'No results available'}

        # Calculate statistics
        successful_results = [r for r in self.results if r['status'] == 'success']

        if successful_results:
            # Class distribution
            class_counts = {}
            freshness_counts = {'fresh': 0, 'rotten': 0}
            confidences = []

            for result in successful_results:
                pred_class = result['prediction']['class']
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

                if result['prediction']['is_fresh']:
                    freshness_counts['fresh'] += 1
                else:
                    freshness_counts['rotten'] += 1

                confidences.append(result['prediction']['confidence'])

            # Calculate processing time
            processing_duration = self.processing_stats['end_time'] - self.processing_stats['start_time']

            summary = {
                'processing_summary': {
                    'total_images': len(self.results),
                    'successful': self.processing_stats['successful'],
                    'failed': self.processing_stats['failed'],
                    'success_rate': self.processing_stats['successful'] / len(self.results),
                    'processing_duration': str(processing_duration),
                    'average_time_per_image': processing_duration.total_seconds() / len(self.results)
                },
                'prediction_summary': {
                    'class_distribution': class_counts,
                    'freshness_distribution': freshness_counts,
                    'average_confidence': np.mean(confidences),
                    'min_confidence': np.min(confidences),
                    'max_confidence': np.max(confidences)
                },
                'model_info': {
                    'accuracy': '96.50%',
                    'version': '1.0.0'
                }
            }
        else:
            summary = {
                'processing_summary': {
                    'total_images': len(self.results),
                    'successful': 0,
                    'failed': len(self.results),
                    'success_rate': 0.0
                },
                'error': 'No successful predictions'
            }

        return summary

def run_batch_inference(input_directory: str, output_path: str,
                       config_path: str = "config/config.yaml",
                       parallel: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run batch inference.

    Args:
        input_directory: Directory containing images to process
        output_path: Path to save results
        config_path: Path to configuration file
        parallel: Whether to use parallel processing

    Returns:
        Summary of batch processing
    """
    # Initialize batch processor
    processor = BatchInference(config_path)

    # Load model
    if not processor.load_model():
        return {'error': 'Failed to load model'}

    # Load images
    image_files = processor.load_images_from_directory(input_directory)
    if not image_files:
        return {'error': 'No images found in directory'}

    # Process images
    results = processor.process_batch(image_files, parallel=parallel)

    # Save results
    processor.save_results(results, output_path)

    # Generate summary
    summary = processor.generate_summary_report()

    return summary
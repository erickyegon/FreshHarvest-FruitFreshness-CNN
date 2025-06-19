"""
Data Validation Component for FreshHarvest Classification System
==============================================================

This module provides comprehensive data validation functionality
including dataset integrity checks, image quality validation, and
data distribution analysis.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.common import read_yaml, create_directories

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Comprehensive data validation for FreshHarvest image datasets.

    Validates dataset structure, image quality, class distribution,
    and data integrity for optimal model training.
    """

    def __init__(self, config_path: str):
        """
        Initialize data validator.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.data_config = self.config['data']
        self.validation_results = {}

        # Expected image specifications
        self.expected_image_size = tuple(self.data_config['image_size'])
        self.expected_channels = self.data_config['channels']
        self.supported_formats = self.data_config.get('supported_formats', ['.jpg', '.jpeg', '.png'])

        logger.info("Data validator initialized")

    def validate_dataset_structure(self, dataset_path: str) -> Dict[str, Any]:
        """
        Validate the overall dataset structure and organization.

        Args:
            dataset_path: Path to the dataset directory

        Returns:
            Dictionary containing structure validation results
        """
        try:
            dataset_path = Path(dataset_path)

            if not dataset_path.exists():
                return {
                    'valid': False,
                    'error': f"Dataset path does not exist: {dataset_path}"
                }

            # Check for expected subdirectories
            expected_dirs = ['train', 'test', 'validation']
            found_dirs = []
            missing_dirs = []

            for dir_name in expected_dirs:
                dir_path = dataset_path / dir_name
                if dir_path.exists() and dir_path.is_dir():
                    found_dirs.append(dir_name)
                else:
                    missing_dirs.append(dir_name)

            # Check class directories within each split
            class_structure = {}
            for split_dir in found_dirs:
                split_path = dataset_path / split_dir
                class_dirs = [d.name for d in split_path.iterdir() if d.is_dir()]
                class_structure[split_dir] = class_dirs

            # Validate class consistency across splits
            all_classes = set()
            for classes in class_structure.values():
                all_classes.update(classes)

            class_consistency = {}
            for split, classes in class_structure.items():
                missing_classes = all_classes - set(classes)
                class_consistency[split] = {
                    'present_classes': classes,
                    'missing_classes': list(missing_classes),
                    'class_count': len(classes)
                }

            results = {
                'valid': len(missing_dirs) == 0,
                'dataset_path': str(dataset_path),
                'found_directories': found_dirs,
                'missing_directories': missing_dirs,
                'class_structure': class_structure,
                'class_consistency': class_consistency,
                'total_classes': len(all_classes),
                'all_classes': sorted(list(all_classes))
            }

            self.validation_results['dataset_structure'] = results
            logger.info(f"Dataset structure validation completed: {len(found_dirs)}/{len(expected_dirs)} directories found")

            return results

        except Exception as e:
            logger.error(f"Error validating dataset structure: {e}")
            return {'valid': False, 'error': str(e)}

    def validate_image_quality(self, dataset_path: str,
                              sample_size: int = 100) -> Dict[str, Any]:
        """
        Validate image quality across the dataset.

        Args:
            dataset_path: Path to the dataset directory
            sample_size: Number of images to sample for quality check

        Returns:
            Dictionary containing image quality validation results
        """
        try:
            dataset_path = Path(dataset_path)

            # Collect all image files
            image_files = []
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.supported_formats):
                        image_files.append(Path(root) / file)

            if not image_files:
                return {
                    'valid': False,
                    'error': "No image files found in dataset"
                }

            # Sample images for quality check
            sample_files = np.random.choice(image_files,
                                          min(sample_size, len(image_files)),
                                          replace=False)

            quality_issues = {
                'corrupted_images': [],
                'wrong_format': [],
                'wrong_size': [],
                'wrong_channels': [],
                'low_quality': [],
                'empty_files': []
            }

            valid_images = 0
            image_stats = {
                'sizes': [],
                'channels': [],
                'file_sizes': [],
                'formats': []
            }

            for img_path in sample_files:
                try:
                    # Check file size
                    file_size = img_path.stat().st_size
                    if file_size == 0:
                        quality_issues['empty_files'].append(str(img_path))
                        continue

                    # Try to load image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        quality_issues['corrupted_images'].append(str(img_path))
                        continue

                    # Check image properties
                    height, width, channels = image.shape

                    # Collect statistics
                    image_stats['sizes'].append((width, height))
                    image_stats['channels'].append(channels)
                    image_stats['file_sizes'].append(file_size)
                    image_stats['formats'].append(img_path.suffix.lower())

                    # Check for quality issues
                    if (width, height) != self.expected_image_size:
                        quality_issues['wrong_size'].append(str(img_path))

                    if channels != self.expected_channels:
                        quality_issues['wrong_channels'].append(str(img_path))

                    # Check for low quality (very small file size relative to dimensions)
                    expected_min_size = width * height * 0.1  # Rough heuristic
                    if file_size < expected_min_size:
                        quality_issues['low_quality'].append(str(img_path))

                    valid_images += 1

                except Exception as e:
                    quality_issues['corrupted_images'].append(str(img_path))
                    logger.warning(f"Error processing image {img_path}: {e}")

            # Calculate quality metrics
            total_sampled = len(sample_files)
            quality_score = valid_images / total_sampled if total_sampled > 0 else 0

            # Analyze image statistics
            size_distribution = Counter(image_stats['sizes'])
            channel_distribution = Counter(image_stats['channels'])
            format_distribution = Counter(image_stats['formats'])

            results = {
                'valid': quality_score > 0.95,  # 95% of images should be valid
                'total_images_found': len(image_files),
                'sampled_images': total_sampled,
                'valid_images': valid_images,
                'quality_score': quality_score,
                'quality_issues': quality_issues,
                'image_statistics': {
                    'size_distribution': dict(size_distribution),
                    'channel_distribution': dict(channel_distribution),
                    'format_distribution': dict(format_distribution),
                    'avg_file_size': np.mean(image_stats['file_sizes']) if image_stats['file_sizes'] else 0,
                    'file_size_range': [np.min(image_stats['file_sizes']), np.max(image_stats['file_sizes'])] if image_stats['file_sizes'] else [0, 0]
                }
            }

            self.validation_results['image_quality'] = results
            logger.info(f"Image quality validation completed: {quality_score:.2%} quality score")

            return results

        except Exception as e:
            logger.error(f"Error validating image quality: {e}")
            return {'valid': False, 'error': str(e)}

    def validate_class_distribution(self, dataset_path: str) -> Dict[str, Any]:
        """
        Validate class distribution and balance across dataset splits.

        Args:
            dataset_path: Path to the dataset directory

        Returns:
            Dictionary containing class distribution validation results
        """
        try:
            dataset_path = Path(dataset_path)

            # Count images per class per split
            distribution = {}
            total_counts = Counter()

            for split in ['train', 'test', 'validation']:
                split_path = dataset_path / split
                if not split_path.exists():
                    continue

                split_counts = Counter()
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        # Count image files in class directory
                        image_count = sum(1 for f in class_dir.iterdir()
                                        if f.is_file() and any(f.name.lower().endswith(ext)
                                                             for ext in self.supported_formats))
                        split_counts[class_dir.name] = image_count
                        total_counts[class_dir.name] += image_count

                distribution[split] = dict(split_counts)

            # Calculate distribution statistics
            total_images = sum(total_counts.values())
            num_classes = len(total_counts)

            # Class balance analysis
            if num_classes > 0:
                avg_per_class = total_images / num_classes
                class_balance = {}
                imbalance_ratio = 0

                for class_name, count in total_counts.items():
                    balance_ratio = count / avg_per_class
                    class_balance[class_name] = {
                        'count': count,
                        'percentage': count / total_images * 100,
                        'balance_ratio': balance_ratio
                    }
                    imbalance_ratio = max(imbalance_ratio, abs(1 - balance_ratio))

                # Split distribution analysis
                split_analysis = {}
                for split, split_dist in distribution.items():
                    split_total = sum(split_dist.values())
                    split_analysis[split] = {
                        'total_images': split_total,
                        'percentage_of_dataset': split_total / total_images * 100 if total_images > 0 else 0,
                        'classes_present': len(split_dist),
                        'avg_per_class': split_total / len(split_dist) if split_dist else 0
                    }
            else:
                class_balance = {}
                imbalance_ratio = 0
                split_analysis = {}

            # Determine if distribution is acceptable
            is_balanced = imbalance_ratio < 0.5  # Allow up to 50% imbalance
            has_all_splits = all(split in distribution for split in ['train', 'test'])
            min_images_per_class = min(total_counts.values()) if total_counts else 0
            sufficient_data = min_images_per_class >= 10  # At least 10 images per class

            results = {
                'valid': is_balanced and has_all_splits and sufficient_data,
                'total_images': total_images,
                'total_classes': num_classes,
                'distribution_by_split': distribution,
                'class_balance': class_balance,
                'split_analysis': split_analysis,
                'imbalance_ratio': imbalance_ratio,
                'is_balanced': is_balanced,
                'has_all_splits': has_all_splits,
                'sufficient_data': sufficient_data,
                'min_images_per_class': min_images_per_class
            }

            self.validation_results['class_distribution'] = results
            logger.info(f"Class distribution validation completed: {num_classes} classes, {total_images} total images")

            return results

        except Exception as e:
            logger.error(f"Error validating class distribution: {e}")
            return {'valid': False, 'error': str(e)}

    def run_comprehensive_validation(self, dataset_path: str) -> Dict[str, Any]:
        """
        Run comprehensive validation on the entire dataset.

        Args:
            dataset_path: Path to the dataset directory

        Returns:
            Dictionary containing all validation results
        """
        try:
            logger.info("Starting comprehensive dataset validation")

            # Run all validation checks
            structure_results = self.validate_dataset_structure(dataset_path)
            quality_results = self.validate_image_quality(dataset_path)
            distribution_results = self.validate_class_distribution(dataset_path)

            # Overall validation status
            overall_valid = (
                structure_results.get('valid', False) and
                quality_results.get('valid', False) and
                distribution_results.get('valid', False)
            )

            # Compile comprehensive results
            comprehensive_results = {
                'overall_valid': overall_valid,
                'validation_timestamp': pd.Timestamp.now().isoformat(),
                'dataset_path': dataset_path,
                'structure_validation': structure_results,
                'quality_validation': quality_results,
                'distribution_validation': distribution_results,
                'summary': {
                    'total_issues': sum([
                        len(quality_results.get('quality_issues', {}).get('corrupted_images', [])),
                        len(quality_results.get('quality_issues', {}).get('wrong_size', [])),
                        len(quality_results.get('quality_issues', {}).get('wrong_channels', [])),
                        len(structure_results.get('missing_directories', [])),
                    ]),
                    'critical_issues': not structure_results.get('valid', False),
                    'quality_score': quality_results.get('quality_score', 0),
                    'balance_score': 1 - distribution_results.get('imbalance_ratio', 1)
                }
            }

            self.validation_results['comprehensive'] = comprehensive_results

            if overall_valid:
                logger.info("✅ Dataset validation PASSED - Ready for training")
            else:
                logger.warning("❌ Dataset validation FAILED - Issues need to be addressed")

            return comprehensive_results

        except Exception as e:
            logger.error(f"Error in comprehensive validation: {e}")
            return {'overall_valid': False, 'error': str(e)}

    def generate_validation_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive validation report.

        Args:
            save_path: Path to save the report

        Returns:
            Formatted report string
        """
        try:
            if 'comprehensive' not in self.validation_results:
                return "No validation results available. Run validation first."

            results = self.validation_results['comprehensive']

            report = []
            report.append("="*80)
            report.append("FRESHHARVEST DATASET VALIDATION REPORT")
            report.append("="*80)
            report.append(f"Generated: {results.get('validation_timestamp', 'Unknown')}")
            report.append(f"Dataset Path: {results.get('dataset_path', 'Unknown')}")
            report.append("")

            # Overall status
            status = "✅ PASSED" if results['overall_valid'] else "❌ FAILED"
            report.append(f"OVERALL VALIDATION STATUS: {status}")
            report.append("")

            # Structure validation
            struct = results.get('structure_validation', {})
            report.append("DATASET STRUCTURE:")
            report.append("-" * 30)
            report.append(f"Valid Structure: {'✅' if struct.get('valid', False) else '❌'}")
            report.append(f"Found Directories: {struct.get('found_directories', [])}")
            report.append(f"Missing Directories: {struct.get('missing_directories', [])}")
            report.append(f"Total Classes: {struct.get('total_classes', 0)}")
            report.append("")

            # Quality validation
            quality = results.get('quality_validation', {})
            report.append("IMAGE QUALITY:")
            report.append("-" * 20)
            report.append(f"Quality Score: {quality.get('quality_score', 0):.2%}")
            report.append(f"Total Images: {quality.get('total_images_found', 0)}")
            report.append(f"Sampled Images: {quality.get('sampled_images', 0)}")
            report.append(f"Valid Images: {quality.get('valid_images', 0)}")

            issues = quality.get('quality_issues', {})
            report.append(f"Corrupted Images: {len(issues.get('corrupted_images', []))}")
            report.append(f"Wrong Size: {len(issues.get('wrong_size', []))}")
            report.append(f"Wrong Channels: {len(issues.get('wrong_channels', []))}")
            report.append("")

            # Distribution validation
            dist = results.get('distribution_validation', {})
            report.append("CLASS DISTRIBUTION:")
            report.append("-" * 25)
            report.append(f"Balanced Distribution: {'✅' if dist.get('is_balanced', False) else '❌'}")
            report.append(f"Total Images: {dist.get('total_images', 0)}")
            report.append(f"Total Classes: {dist.get('total_classes', 0)}")
            report.append(f"Imbalance Ratio: {dist.get('imbalance_ratio', 0):.2f}")
            report.append(f"Min Images per Class: {dist.get('min_images_per_class', 0)}")
            report.append("")

            # Summary
            summary = results.get('summary', {})
            report.append("VALIDATION SUMMARY:")
            report.append("-" * 25)
            report.append(f"Total Issues Found: {summary.get('total_issues', 0)}")
            report.append(f"Critical Issues: {'Yes' if summary.get('critical_issues', False) else 'No'}")
            report.append(f"Quality Score: {summary.get('quality_score', 0):.2%}")
            report.append(f"Balance Score: {summary.get('balance_score', 0):.2%}")
            report.append("")

            # Recommendations
            report.append("RECOMMENDATIONS:")
            report.append("-" * 20)
            if results['overall_valid']:
                report.append("✅ Dataset is ready for training!")
                report.append("✅ All validation checks passed successfully.")
            else:
                if not struct.get('valid', False):
                    report.append("❌ Fix dataset structure issues first")
                if quality.get('quality_score', 0) < 0.95:
                    report.append("❌ Address image quality issues")
                if not dist.get('is_balanced', False):
                    report.append("❌ Consider data augmentation for class balance")
                if not dist.get('sufficient_data', False):
                    report.append("❌ Collect more data for underrepresented classes")

            report.append("")
            report.append("="*80)

            report_text = "\n".join(report)

            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Validation report saved to {save_path}")

            return report_text

        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            return ""

def validate_dataset(dataset_path: str, config_path: str) -> Dict[str, Any]:
    """
    Convenience function for comprehensive dataset validation.

    Args:
        dataset_path: Path to the dataset directory
        config_path: Path to configuration file

    Returns:
        Dictionary containing validation results
    """
    validator = DataValidator(config_path)
    return validator.run_comprehensive_validation(dataset_path)
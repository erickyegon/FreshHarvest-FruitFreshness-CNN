"""
Model Validator for FreshHarvest Classification System
====================================================

This module provides comprehensive model validation functionality
including cross-validation, holdout validation, and performance assessment.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Comprehensive model validation for FreshHarvest classification models.

    Provides various validation strategies including holdout validation,
    cross-validation, and performance assessment.
    """

    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the model validator.

        Args:
            model_path: Path to the trained model
            config: Configuration dictionary
        """
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.validation_results = {}

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model for validation.

        Args:
            model_path: Path to the model file

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            path = model_path or self.model_path
            if not path or not Path(path).exists():
                logger.error(f"Model file not found: {path}")
                return False

            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded successfully from {path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def validate_holdout(self, X_test: np.ndarray, y_test: np.ndarray,
                        class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform holdout validation on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            class_names: List of class names

        Returns:
            Dictionary containing validation results
        """
        try:
            if self.model is None:
                logger.error("No model loaded for validation")
                return {}

            # Make predictions
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Classification report
            report = classification_report(
                y_test, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )

            results = {
                'validation_type': 'holdout',
                'test_size': len(X_test),
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': report,
                'model_path': self.model_path
            }

            self.validation_results['holdout'] = results
            logger.info(f"Holdout validation completed: Accuracy = {accuracy:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error in holdout validation: {e}")
            return {}

    def validate_cross_validation(self, X: np.ndarray, y: np.ndarray,
                                 cv_folds: int = 5,
                                 class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.

        Args:
            X: Features
            y: Labels
            cv_folds: Number of cross-validation folds
            class_names: List of class names

        Returns:
            Dictionary containing cross-validation results
        """
        try:
            if self.model is None:
                logger.error("No model loaded for validation")
                return {}

            # Initialize stratified k-fold
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            fold_results = []
            fold_accuracies = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                logger.info(f"Processing fold {fold + 1}/{cv_folds}")

                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Clone and train model for this fold
                model_fold = tf.keras.models.clone_model(self.model)
                model_fold.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                # Train on fold
                history = model_fold.fit(
                    X_train_fold, y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    epochs=10,  # Reduced for CV
                    batch_size=32,
                    verbose=0
                )

                # Evaluate on validation set
                y_pred_proba = model_fold.predict(X_val_fold, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
                accuracy = accuracy_score(y_val_fold, y_pred)

                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(X_train_fold),
                    'val_size': len(X_val_fold),
                    'accuracy': accuracy,
                    'history': history.history
                }

                fold_results.append(fold_result)
                fold_accuracies.append(accuracy)

            # Calculate overall statistics
            mean_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)

            results = {
                'validation_type': 'cross_validation',
                'cv_folds': cv_folds,
                'fold_results': fold_results,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'fold_accuracies': fold_accuracies,
                'model_path': self.model_path
            }

            self.validation_results['cross_validation'] = results
            logger.info(f"Cross-validation completed: Mean accuracy = {mean_accuracy:.4f} ± {std_accuracy:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {}

    def validate_model_robustness(self, X_test: np.ndarray, y_test: np.ndarray,
                                 noise_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict[str, Any]:
        """
        Test model robustness against input noise.

        Args:
            X_test: Test features
            y_test: Test labels
            noise_levels: List of noise levels to test

        Returns:
            Dictionary containing robustness test results
        """
        try:
            if self.model is None:
                logger.error("No model loaded for validation")
                return {}

            robustness_results = []

            # Test with original data
            y_pred_clean = self.model.predict(X_test, verbose=0)
            accuracy_clean = accuracy_score(y_test, np.argmax(y_pred_clean, axis=1))

            robustness_results.append({
                'noise_level': 0.0,
                'accuracy': accuracy_clean,
                'accuracy_drop': 0.0
            })

            # Test with different noise levels
            for noise_level in noise_levels:
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, X_test.shape)
                X_noisy = X_test + noise
                X_noisy = np.clip(X_noisy, 0, 1)  # Ensure valid range

                # Make predictions
                y_pred_noisy = self.model.predict(X_noisy, verbose=0)
                accuracy_noisy = accuracy_score(y_test, np.argmax(y_pred_noisy, axis=1))
                accuracy_drop = accuracy_clean - accuracy_noisy

                robustness_results.append({
                    'noise_level': noise_level,
                    'accuracy': accuracy_noisy,
                    'accuracy_drop': accuracy_drop
                })

                logger.info(f"Noise level {noise_level}: Accuracy = {accuracy_noisy:.4f}, Drop = {accuracy_drop:.4f}")

            results = {
                'validation_type': 'robustness',
                'clean_accuracy': accuracy_clean,
                'robustness_results': robustness_results,
                'model_path': self.model_path
            }

            self.validation_results['robustness'] = results
            logger.info("Robustness validation completed")

            return results

        except Exception as e:
            logger.error(f"Error in robustness validation: {e}")
            return {}

    def generate_validation_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive validation report.

        Args:
            save_path: Path to save the report

        Returns:
            Formatted report string
        """
        try:
            report = []
            report.append("="*80)
            report.append("FRESHHARVEST MODEL VALIDATION REPORT")
            report.append("="*80)
            report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Model: {self.model_path}")
            report.append("")

            # Holdout validation results
            if 'holdout' in self.validation_results:
                holdout = self.validation_results['holdout']
                report.append("HOLDOUT VALIDATION:")
                report.append("-" * 40)
                report.append(f"Test Size: {holdout['test_size']}")
                report.append(f"Accuracy: {holdout['accuracy']:.4f}")
                report.append("")

            # Cross-validation results
            if 'cross_validation' in self.validation_results:
                cv = self.validation_results['cross_validation']
                report.append("CROSS-VALIDATION:")
                report.append("-" * 40)
                report.append(f"Folds: {cv['cv_folds']}")
                report.append(f"Mean Accuracy: {cv['mean_accuracy']:.4f} ± {cv['std_accuracy']:.4f}")
                report.append("Fold Accuracies:")
                for i, acc in enumerate(cv['fold_accuracies']):
                    report.append(f"  Fold {i+1}: {acc:.4f}")
                report.append("")

            # Robustness results
            if 'robustness' in self.validation_results:
                robust = self.validation_results['robustness']
                report.append("ROBUSTNESS TESTING:")
                report.append("-" * 40)
                report.append(f"Clean Accuracy: {robust['clean_accuracy']:.4f}")
                report.append("Noise Level Results:")
                for result in robust['robustness_results']:
                    report.append(f"  Noise {result['noise_level']:.2f}: "
                                f"Accuracy {result['accuracy']:.4f}, "
                                f"Drop {result['accuracy_drop']:.4f}")
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

    def save_validation_results(self, save_path: str) -> bool:
        """
        Save validation results to JSON file.

        Args:
            save_path: Path to save the results

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}

            for key, value in self.validation_results.items():
                serializable_results[key] = self._make_serializable(value)

            with open(save_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"Validation results saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving validation results: {e}")
            return False

    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

def validate_model(model_path: str, X_test: np.ndarray, y_test: np.ndarray,
                  class_names: Optional[List[str]] = None,
                  validation_type: str = 'holdout') -> Dict[str, Any]:
    """
    Convenience function for model validation.

    Args:
        model_path: Path to the trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        validation_type: Type of validation ('holdout', 'cv', 'robustness')

    Returns:
        Dictionary containing validation results
    """
    validator = ModelValidator(model_path)

    if not validator.load_model():
        return {}

    if validation_type == 'holdout':
        return validator.validate_holdout(X_test, y_test, class_names)
    elif validation_type == 'cv':
        return validator.validate_cross_validation(X_test, y_test, class_names=class_names)
    elif validation_type == 'robustness':
        return validator.validate_model_robustness(X_test, y_test)
    else:
        logger.error(f"Unknown validation type: {validation_type}")
        return {}
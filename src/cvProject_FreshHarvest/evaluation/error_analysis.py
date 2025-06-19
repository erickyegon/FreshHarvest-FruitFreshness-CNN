"""
Error Analysis Module for FreshHarvest Classification System
==========================================================

This module provides comprehensive error analysis tools for understanding
model failures and improving classification performance.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ErrorAnalyzer:
    """
    Comprehensive error analysis for FreshHarvest classification models.

    Analyzes prediction errors, identifies patterns, and provides insights
    for model improvement.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the error analyzer.

        Args:
            class_names: List of class names for labeling
        """
        self.class_names = class_names or []
        self.error_analysis = {}

    def analyze_classification_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze classification errors and patterns.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            Dictionary containing error analysis results
        """
        try:
            # Basic error statistics
            total_samples = len(y_true)
            correct_predictions = np.sum(y_true == y_pred)
            incorrect_predictions = total_samples - correct_predictions
            error_rate = incorrect_predictions / total_samples

            # Find misclassified samples
            misclassified_mask = y_true != y_pred
            misclassified_indices = np.where(misclassified_mask)[0]

            # Analyze error patterns by class
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            class_errors = {}

            for cls in unique_classes:
                class_mask = y_true == cls
                class_total = np.sum(class_mask)
                class_correct = np.sum((y_true == cls) & (y_pred == cls))
                class_error_rate = (class_total - class_correct) / class_total if class_total > 0 else 0

                # Find what this class is misclassified as
                class_misclassified = y_pred[class_mask & misclassified_mask]
                misclassified_as = {}
                for target_cls in unique_classes:
                    count = np.sum(class_misclassified == target_cls)
                    if count > 0:
                        misclassified_as[target_cls] = count

                class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class_{cls}"
                class_errors[class_name] = {
                    'total_samples': int(class_total),
                    'correct_predictions': int(class_correct),
                    'error_rate': float(class_error_rate),
                    'misclassified_as': misclassified_as
                }

            # Confidence analysis for errors
            confidence_analysis = {}
            if y_pred_proba is not None:
                confidence_scores = np.max(y_pred_proba, axis=1)

                # Confidence of correct vs incorrect predictions
                correct_confidence = confidence_scores[~misclassified_mask]
                incorrect_confidence = confidence_scores[misclassified_mask]

                confidence_analysis = {
                    'mean_confidence_correct': float(np.mean(correct_confidence)) if len(correct_confidence) > 0 else 0,
                    'mean_confidence_incorrect': float(np.mean(incorrect_confidence)) if len(incorrect_confidence) > 0 else 0,
                    'std_confidence_correct': float(np.std(correct_confidence)) if len(correct_confidence) > 0 else 0,
                    'std_confidence_incorrect': float(np.std(incorrect_confidence)) if len(incorrect_confidence) > 0 else 0
                }

            results = {
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
                'incorrect_predictions': incorrect_predictions,
                'error_rate': error_rate,
                'accuracy': 1 - error_rate,
                'misclassified_indices': misclassified_indices.tolist(),
                'class_errors': class_errors,
                'confidence_analysis': confidence_analysis
            }

            self.error_analysis['classification_errors'] = results
            logger.info(f"Error analysis completed: Error rate = {error_rate:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error in classification error analysis: {e}")
            return {}

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """
        Plot detailed confusion matrix with error analysis.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Create figure
            plt.figure(figsize=(12, 10))

            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names if self.class_names else None,
                       yticklabels=self.class_names if self.class_names else None)

            plt.title('Confusion Matrix - Error Analysis')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            # Add error statistics
            total_errors = np.sum(cm) - np.trace(cm)
            total_samples = np.sum(cm)
            error_rate = total_errors / total_samples

            plt.figtext(0.02, 0.02, f'Total Errors: {total_errors}/{total_samples} ({error_rate:.2%})',
                       fontsize=10, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix plot saved to {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")

    def analyze_confidence_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Analyze relationship between prediction confidence and errors.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary containing confidence error analysis
        """
        try:
            confidence_scores = np.max(y_pred_proba, axis=1)
            correct_mask = y_true == y_pred

            # Bin confidence scores
            confidence_bins = np.linspace(0, 1, 11)
            bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2

            bin_accuracies = []
            bin_counts = []
            bin_error_rates = []

            for i in range(len(confidence_bins) - 1):
                mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
                bin_count = np.sum(mask)

                if bin_count > 0:
                    bin_accuracy = np.mean(correct_mask[mask])
                    bin_error_rate = 1 - bin_accuracy
                else:
                    bin_accuracy = 0
                    bin_error_rate = 0

                bin_accuracies.append(bin_accuracy)
                bin_counts.append(bin_count)
                bin_error_rates.append(bin_error_rate)

            # High confidence errors (confidence > 0.8 but wrong)
            high_conf_mask = confidence_scores > 0.8
            high_conf_errors = np.sum(high_conf_mask & ~correct_mask)
            high_conf_total = np.sum(high_conf_mask)

            # Low confidence correct (confidence < 0.6 but correct)
            low_conf_mask = confidence_scores < 0.6
            low_conf_correct = np.sum(low_conf_mask & correct_mask)
            low_conf_total = np.sum(low_conf_mask)

            results = {
                'confidence_bins': bin_centers.tolist(),
                'bin_accuracies': bin_accuracies,
                'bin_error_rates': bin_error_rates,
                'bin_counts': bin_counts,
                'high_confidence_errors': int(high_conf_errors),
                'high_confidence_total': int(high_conf_total),
                'low_confidence_correct': int(low_conf_correct),
                'low_confidence_total': int(low_conf_total),
                'mean_confidence_correct': float(np.mean(confidence_scores[correct_mask])),
                'mean_confidence_incorrect': float(np.mean(confidence_scores[~correct_mask]))
            }

            self.error_analysis['confidence_errors'] = results
            logger.info("Confidence error analysis completed")

            return results

        except Exception as e:
            logger.error(f"Error in confidence error analysis: {e}")
            return {}

    def identify_problematic_classes(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Identify classes that are most problematic for the model.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary containing problematic class analysis
        """
        try:
            # Calculate per-class metrics
            unique_classes = np.unique(y_true)
            class_metrics = []

            for cls in unique_classes:
                class_mask = y_true == cls
                class_total = np.sum(class_mask)
                class_correct = np.sum((y_true == cls) & (y_pred == cls))
                class_accuracy = class_correct / class_total if class_total > 0 else 0

                # Calculate precision and recall for this class
                pred_mask = y_pred == cls
                precision = class_correct / np.sum(pred_mask) if np.sum(pred_mask) > 0 else 0
                recall = class_accuracy  # Same as accuracy for individual class

                class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class_{cls}"

                class_metrics.append({
                    'class': class_name,
                    'class_id': int(cls),
                    'total_samples': int(class_total),
                    'correct_predictions': int(class_correct),
                    'accuracy': float(class_accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'error_count': int(class_total - class_correct)
                })

            # Sort by accuracy (ascending) to identify most problematic
            class_metrics.sort(key=lambda x: x['accuracy'])

            # Identify top problematic classes
            n_problematic = min(5, len(class_metrics))
            most_problematic = class_metrics[:n_problematic]
            best_performing = class_metrics[-n_problematic:]

            results = {
                'all_class_metrics': class_metrics,
                'most_problematic_classes': most_problematic,
                'best_performing_classes': best_performing,
                'overall_accuracy': float(np.mean(y_true == y_pred))
            }

            self.error_analysis['problematic_classes'] = results
            logger.info("Problematic class analysis completed")

            return results

        except Exception as e:
            logger.error(f"Error in problematic class analysis: {e}")
            return {}

    def generate_error_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive error analysis report.

        Args:
            save_path: Path to save the report

        Returns:
            Formatted report string
        """
        try:
            report = []
            report.append("="*80)
            report.append("FRESHHARVEST ERROR ANALYSIS REPORT")
            report.append("="*80)
            report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")

            # Classification errors
            if 'classification_errors' in self.error_analysis:
                errors = self.error_analysis['classification_errors']
                report.append("CLASSIFICATION ERROR ANALYSIS:")
                report.append("-" * 40)
                report.append(f"Total Samples: {errors['total_samples']}")
                report.append(f"Correct Predictions: {errors['correct_predictions']}")
                report.append(f"Incorrect Predictions: {errors['incorrect_predictions']}")
                report.append(f"Error Rate: {errors['error_rate']:.4f}")
                report.append(f"Accuracy: {errors['accuracy']:.4f}")
                report.append("")

                # Per-class errors
                report.append("Per-Class Error Analysis:")
                for class_name, class_data in errors['class_errors'].items():
                    report.append(f"  {class_name}:")
                    report.append(f"    Total: {class_data['total_samples']}")
                    report.append(f"    Correct: {class_data['correct_predictions']}")
                    report.append(f"    Error Rate: {class_data['error_rate']:.4f}")
                report.append("")

            # Confidence errors
            if 'confidence_errors' in self.error_analysis:
                conf_errors = self.error_analysis['confidence_errors']
                report.append("CONFIDENCE ERROR ANALYSIS:")
                report.append("-" * 40)
                report.append(f"High Confidence Errors: {conf_errors['high_confidence_errors']}/{conf_errors['high_confidence_total']}")
                report.append(f"Low Confidence Correct: {conf_errors['low_confidence_correct']}/{conf_errors['low_confidence_total']}")
                report.append(f"Mean Confidence (Correct): {conf_errors['mean_confidence_correct']:.4f}")
                report.append(f"Mean Confidence (Incorrect): {conf_errors['mean_confidence_incorrect']:.4f}")
                report.append("")

            # Problematic classes
            if 'problematic_classes' in self.error_analysis:
                prob_classes = self.error_analysis['problematic_classes']
                report.append("MOST PROBLEMATIC CLASSES:")
                report.append("-" * 40)
                for i, cls_data in enumerate(prob_classes['most_problematic_classes']):
                    report.append(f"{i+1}. {cls_data['class']}: "
                                f"Accuracy={cls_data['accuracy']:.4f}, "
                                f"Errors={cls_data['error_count']}")
                report.append("")

            report.append("="*80)

            report_text = "\n".join(report)

            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Error analysis report saved to {save_path}")

            return report_text

        except Exception as e:
            logger.error(f"Error generating error analysis report: {e}")
            return ""

def analyze_model_errors(y_true: np.ndarray, y_pred: np.ndarray,
                        y_pred_proba: Optional[np.ndarray] = None,
                        class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function for comprehensive error analysis.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        class_names: List of class names

    Returns:
        Dictionary containing all error analysis results
    """
    analyzer = ErrorAnalyzer(class_names)

    results = {}

    # Analyze classification errors
    results['classification_errors'] = analyzer.analyze_classification_errors(y_true, y_pred, y_pred_proba)

    # Analyze confidence errors if probabilities provided
    if y_pred_proba is not None:
        results['confidence_errors'] = analyzer.analyze_confidence_errors(y_true, y_pred, y_pred_proba)

    # Identify problematic classes
    results['problematic_classes'] = analyzer.identify_problematic_classes(y_true, y_pred)

    return results
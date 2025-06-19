"""
Comprehensive Metrics Calculator for FreshHarvest Model Evaluation
================================================================

This module provides comprehensive metrics calculation for evaluating
the performance of fruit freshness classification models.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Comprehensive metrics calculator for multi-class classification.

    Calculates various performance metrics including accuracy, precision,
    recall, F1-score, AUC, and provides detailed analysis.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the metrics calculator.

        Args:
            class_names: List of class names for labeling
        """
        self.class_names = class_names or []
        self.metrics_history = []

    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary containing basic metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }

            logger.info(f"Basic metrics calculated: Accuracy={metrics['accuracy']:.4f}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            return {}

    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Calculate per-class metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            DataFrame with per-class metrics
        """
        try:
            # Get unique classes
            classes = np.unique(np.concatenate([y_true, y_pred]))

            # Calculate per-class metrics
            precision = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
            recall = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)

            # Create DataFrame
            class_labels = [self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
                           for i in classes]

            df = pd.DataFrame({
                'Class': class_labels,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Support': [np.sum(y_true == cls) for cls in classes]
            })

            logger.info(f"Per-class metrics calculated for {len(classes)} classes")
            return df

        except Exception as e:
            logger.error(f"Error calculating per-class metrics: {e}")
            return pd.DataFrame()

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            logger.info(f"Confusion matrix calculated: {cm.shape}")
            return cm

        except Exception as e:
            logger.error(f"Error calculating confusion matrix: {e}")
            return np.array([])

    def calculate_auc_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate AUC-related metrics.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary containing AUC metrics
        """
        try:
            n_classes = len(np.unique(y_true))

            if n_classes == 2:
                # Binary classification
                auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
                return {'auc_roc': auc_score}
            else:
                # Multi-class classification
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

                # Calculate macro and weighted AUC
                auc_macro = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
                auc_weighted = roc_auc_score(y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')

                return {
                    'auc_roc_macro': auc_macro,
                    'auc_roc_weighted': auc_weighted
                }

        except Exception as e:
            logger.error(f"Error calculating AUC metrics: {e}")
            return {}

    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate comprehensive metrics for model evaluation.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            Dictionary containing all metrics
        """
        try:
            results = {}

            # Basic metrics
            results['basic_metrics'] = self.calculate_basic_metrics(y_true, y_pred)

            # Per-class metrics
            results['per_class_metrics'] = self.calculate_per_class_metrics(y_true, y_pred)

            # Confusion matrix
            results['confusion_matrix'] = self.calculate_confusion_matrix(y_true, y_pred)

            # Classification report
            results['classification_report'] = classification_report(
                y_true, y_pred,
                target_names=self.class_names if self.class_names else None,
                output_dict=True,
                zero_division=0
            )

            # AUC metrics (if probabilities provided)
            if y_pred_proba is not None:
                results['auc_metrics'] = self.calculate_auc_metrics(y_true, y_pred_proba)

            # Store in history
            self.metrics_history.append({
                'timestamp': pd.Timestamp.now(),
                'accuracy': results['basic_metrics'].get('accuracy', 0),
                'f1_weighted': results['basic_metrics'].get('f1_weighted', 0)
            })

            logger.info("Comprehensive metrics calculation completed")
            return results

        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            return {}

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix heatmap.

        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 8))

            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names if self.class_names else None,
                       yticklabels=self.class_names if self.class_names else None)

            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix plot saved to {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")

    def plot_metrics_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot metrics history over time.

        Args:
            save_path: Path to save the plot
        """
        try:
            if not self.metrics_history:
                logger.warning("No metrics history available")
                return

            df = pd.DataFrame(self.metrics_history)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.plot(df['timestamp'], df['accuracy'], marker='o')
            plt.title('Accuracy Over Time')
            plt.xlabel('Time')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)

            plt.subplot(1, 2, 2)
            plt.plot(df['timestamp'], df['f1_weighted'], marker='o', color='orange')
            plt.title('F1-Score Over Time')
            plt.xlabel('Time')
            plt.ylabel('F1-Score (Weighted)')
            plt.xticks(rotation=45)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Metrics history plot saved to {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Error plotting metrics history: {e}")

    def generate_metrics_report(self, metrics: Dict, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive metrics report.

        Args:
            metrics: Metrics dictionary from calculate_comprehensive_metrics
            save_path: Path to save the report

        Returns:
            Formatted report string
        """
        try:
            report = []
            report.append("="*80)
            report.append("FRESHHARVEST MODEL EVALUATION REPORT")
            report.append("="*80)
            report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")

            # Basic metrics
            if 'basic_metrics' in metrics:
                report.append("BASIC METRICS:")
                report.append("-" * 40)
                basic = metrics['basic_metrics']
                report.append(f"Accuracy:           {basic.get('accuracy', 0):.4f}")
                report.append(f"Precision (Macro):  {basic.get('precision_macro', 0):.4f}")
                report.append(f"Precision (Weighted): {basic.get('precision_weighted', 0):.4f}")
                report.append(f"Recall (Macro):     {basic.get('recall_macro', 0):.4f}")
                report.append(f"Recall (Weighted):  {basic.get('recall_weighted', 0):.4f}")
                report.append(f"F1-Score (Macro):   {basic.get('f1_macro', 0):.4f}")
                report.append(f"F1-Score (Weighted): {basic.get('f1_weighted', 0):.4f}")
                report.append("")

            # AUC metrics
            if 'auc_metrics' in metrics:
                report.append("AUC METRICS:")
                report.append("-" * 40)
                auc = metrics['auc_metrics']
                for key, value in auc.items():
                    report.append(f"{key.upper()}: {value:.4f}")
                report.append("")

            # Per-class metrics
            if 'per_class_metrics' in metrics and not metrics['per_class_metrics'].empty:
                report.append("PER-CLASS METRICS:")
                report.append("-" * 40)
                df = metrics['per_class_metrics']
                report.append(df.to_string(index=False, float_format='%.4f'))
                report.append("")

            # Confusion matrix summary
            if 'confusion_matrix' in metrics and metrics['confusion_matrix'].size > 0:
                cm = metrics['confusion_matrix']
                report.append("CONFUSION MATRIX SUMMARY:")
                report.append("-" * 40)
                report.append(f"Matrix Shape: {cm.shape}")
                report.append(f"Total Predictions: {cm.sum()}")
                report.append(f"Correct Predictions: {np.trace(cm)}")
                report.append("")

            report.append("="*80)

            report_text = "\n".join(report)

            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Metrics report saved to {save_path}")

            return report_text

        except Exception as e:
            logger.error(f"Error generating metrics report: {e}")
            return ""

def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: Optional[np.ndarray] = None,
                          class_names: Optional[List[str]] = None) -> Dict:
    """
    Convenience function to calculate comprehensive model metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        class_names: List of class names

    Returns:
        Dictionary containing all metrics
    """
    calculator = MetricsCalculator(class_names)
    return calculator.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
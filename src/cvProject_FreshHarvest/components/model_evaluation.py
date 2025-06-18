"""
Model evaluation component for the FreshHarvest project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..utils.common import read_yaml, write_json, save_pickle


class ModelEvaluator:
    """
    Comprehensive model evaluation component.
    """

    def __init__(self, config_path: str):
        """
        Initialize model evaluator.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.data_config = self.config['data']
        self.class_names = self.data_config['classes']
        self.num_classes = self.data_config['num_classes']

        logging.info("Model evaluator initialized")

    def evaluate_model(self, model, test_generator) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Args:
            model: Trained model
            test_generator: Test data generator

        Returns:
            Dictionary containing evaluation results
        """
        if not TF_AVAILABLE:
            logging.error("TensorFlow not available for evaluation")
            return {}

        logging.info("Starting comprehensive model evaluation")

        # Get predictions
        predictions = model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Per-class metrics
        class_report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # ROC AUC (for multiclass)
        try:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            roc_auc = roc_auc_score(y_true_bin, predictions, average='weighted', multi_class='ovr')
        except Exception as e:
            logging.warning(f"Could not compute ROC AUC: {e}")
            roc_auc = None

        # Compile results
        results = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc) if roc_auc is not None else None
            },
            'per_class_metrics': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'true_labels': y_true.tolist(),
            'predicted_labels': y_pred.tolist()
        }

        logging.info(f"Evaluation completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return results

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 10))

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create heatmap
        sns.heatmap(cm_normalized,
                   annot=True,
                   fmt='.2f',
                   cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)

        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_classification_report(self, class_report: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot classification report as heatmap.

        Args:
            class_report: Classification report dictionary
            save_path: Path to save the plot
        """
        # Extract metrics for each class
        metrics_data = []
        for class_name in self.class_names:
            if class_name in class_report:
                metrics_data.append([
                    class_report[class_name]['precision'],
                    class_report[class_name]['recall'],
                    class_report[class_name]['f1-score']
                ])

        metrics_df = pd.DataFrame(
            metrics_data,
            index=self.class_names,
            columns=['Precision', 'Recall', 'F1-Score']
        )

        plt.figure(figsize=(8, 12))
        sns.heatmap(metrics_df,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   vmin=0,
                   vmax=1)

        plt.title('Per-Class Classification Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Classification report plot saved to {save_path}")

        plt.show()

    def plot_roc_curves(self, y_true: np.ndarray, predictions: np.ndarray,
                       save_path: Optional[str] = None) -> None:
        """
        Plot ROC curves for multiclass classification.

        Args:
            y_true: True labels
            predictions: Model predictions (probabilities)
            save_path: Path to save the plot
        """
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        plt.figure(figsize=(15, 10))

        # Plot ROC curve for each class
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], predictions[:, i])

            plt.plot(fpr, tpr,
                    label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})',
                    linewidth=2)

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-Class Classification')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"ROC curves saved to {save_path}")

        plt.show()

    def analyze_misclassifications(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 predictions: np.ndarray) -> Dict[str, Any]:
        """
        Analyze misclassified samples.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            predictions: Model predictions (probabilities)

        Returns:
            Dictionary containing misclassification analysis
        """
        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]

        # Analyze misclassifications
        misclassification_analysis = {
            'total_misclassified': int(len(misclassified_indices)),
            'misclassification_rate': float(len(misclassified_indices) / len(y_true)),
            'misclassified_by_class': {},
            'confusion_pairs': {}
        }

        # Count misclassifications by true class
        for i, class_name in enumerate(self.class_names):
            true_class_mask = y_true == i
            misclassified_in_class = np.sum(misclassified_mask & true_class_mask)
            total_in_class = np.sum(true_class_mask)

            misclassification_analysis['misclassified_by_class'][class_name] = {
                'count': int(misclassified_in_class),
                'total': int(total_in_class),
                'rate': float(misclassified_in_class / total_in_class) if total_in_class > 0 else 0.0
            }

        # Analyze confusion pairs
        for idx in misclassified_indices:
            true_class = self.class_names[y_true[idx]]
            pred_class = self.class_names[y_pred[idx]]
            confidence = float(predictions[idx][y_pred[idx]])

            pair_key = f"{true_class} -> {pred_class}"
            if pair_key not in misclassification_analysis['confusion_pairs']:
                misclassification_analysis['confusion_pairs'][pair_key] = {
                    'count': 0,
                    'avg_confidence': 0.0,
                    'confidences': []
                }

            misclassification_analysis['confusion_pairs'][pair_key]['count'] += 1
            misclassification_analysis['confusion_pairs'][pair_key]['confidences'].append(confidence)

        # Calculate average confidences
        for pair_key in misclassification_analysis['confusion_pairs']:
            confidences = misclassification_analysis['confusion_pairs'][pair_key]['confidences']
            avg_conf = np.mean(confidences)
            misclassification_analysis['confusion_pairs'][pair_key]['avg_confidence'] = float(avg_conf)

        return misclassification_analysis

    def generate_evaluation_report(self, results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            results: Evaluation results dictionary
            save_path: Path to save the report

        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("FRESHHARVEST MODEL EVALUATION REPORT")
        report_lines.append("="*60)
        report_lines.append("")

        # Overall metrics
        overall = results['overall_metrics']
        report_lines.append("OVERALL PERFORMANCE METRICS:")
        report_lines.append("-" * 30)
        report_lines.append(f"Accuracy:  {overall['accuracy']:.4f}")
        report_lines.append(f"Precision: {overall['precision']:.4f}")
        report_lines.append(f"Recall:    {overall['recall']:.4f}")
        report_lines.append(f"F1-Score:  {overall['f1_score']:.4f}")
        if overall['roc_auc'] is not None:
            report_lines.append(f"ROC AUC:   {overall['roc_auc']:.4f}")
        report_lines.append("")

        # Performance assessment
        accuracy = overall['accuracy']
        if accuracy >= 0.95:
            performance = "EXCELLENT"
        elif accuracy >= 0.90:
            performance = "VERY GOOD"
        elif accuracy >= 0.85:
            performance = "GOOD"
        elif accuracy >= 0.80:
            performance = "FAIR"
        else:
            performance = "NEEDS IMPROVEMENT"

        report_lines.append(f"PERFORMANCE ASSESSMENT: {performance}")
        report_lines.append("")

        # Per-class performance
        report_lines.append("PER-CLASS PERFORMANCE:")
        report_lines.append("-" * 30)
        per_class = results['per_class_metrics']

        for class_name in self.class_names:
            if class_name in per_class:
                metrics = per_class[class_name]
                report_lines.append(f"{class_name}:")
                report_lines.append(f"  Precision: {metrics['precision']:.3f}")
                report_lines.append(f"  Recall:    {metrics['recall']:.3f}")
                report_lines.append(f"  F1-Score:  {metrics['f1-score']:.3f}")
                report_lines.append(f"  Support:   {metrics['support']}")
                report_lines.append("")

        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 30)

        if accuracy < 0.90:
            report_lines.append("• Consider increasing training epochs")
            report_lines.append("• Try data augmentation techniques")
            report_lines.append("• Experiment with different model architectures")

        if overall['f1_score'] < overall['accuracy']:
            report_lines.append("• Address class imbalance issues")
            report_lines.append("• Consider weighted loss functions")

        report_lines.append("• Monitor for overfitting with validation curves")
        report_lines.append("• Consider ensemble methods for improved performance")
        report_lines.append("")

        report_lines.append("="*60)

        report_text = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logging.info(f"Evaluation report saved to {save_path}")

        return report_text

    def save_evaluation_results(self, results: Dict[str, Any],
                              output_dir: str) -> None:
        """
        Save all evaluation results and plots.

        Args:
            results: Evaluation results dictionary
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results as JSON
        results_path = output_path / "evaluation_results.json"
        write_json(results, results_path)

        # Generate and save report
        report_path = output_path / "evaluation_report.txt"
        report = self.generate_evaluation_report(results, report_path)

        # Plot and save visualizations
        cm = np.array(results['confusion_matrix'])
        self.plot_confusion_matrix(cm, output_path / "confusion_matrix.png")

        self.plot_classification_report(
            results['per_class_metrics'],
            output_path / "classification_report.png"
        )

        # ROC curves if predictions available
        if 'predictions' in results and 'true_labels' in results:
            y_true = np.array(results['true_labels'])
            predictions = np.array(results['predictions'])
            self.plot_roc_curves(y_true, predictions, output_path / "roc_curves.png")

        # Misclassification analysis
        if 'predicted_labels' in results:
            y_pred = np.array(results['predicted_labels'])
            y_true = np.array(results['true_labels'])
            predictions = np.array(results['predictions'])

            misclass_analysis = self.analyze_misclassifications(y_true, y_pred, predictions)
            misclass_path = output_path / "misclassification_analysis.json"
            write_json(misclass_analysis, misclass_path)

        logging.info(f"All evaluation results saved to {output_dir}")

        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        print(f"F1-Score: {results['overall_metrics']['f1_score']:.4f}")
        print(f"Results saved to: {output_dir}")
        print("="*50)
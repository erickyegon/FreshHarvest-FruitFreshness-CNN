"""
Performance Analysis Module for FreshHarvest Classification System
================================================================

This module provides comprehensive performance analysis tools including
learning curves, feature importance analysis, and model comparison.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import learning_curve, validation_curve
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for FreshHarvest models.

    Provides tools for analyzing model performance, learning curves,
    feature importance, and comparative analysis.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the performance analyzer.

        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.model = None
        self.analysis_results = {}

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model for analysis.

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

    def analyze_training_history(self, history: Dict[str, List[float]],
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze training history and plot learning curves.

        Args:
            history: Training history dictionary
            save_path: Path to save the plots

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(history)

            # Calculate statistics
            final_train_acc = df['accuracy'].iloc[-1] if 'accuracy' in df else 0
            final_val_acc = df['val_accuracy'].iloc[-1] if 'val_accuracy' in df else 0
            best_val_acc = df['val_accuracy'].max() if 'val_accuracy' in df else 0
            best_epoch = df['val_accuracy'].idxmax() if 'val_accuracy' in df else 0

            final_train_loss = df['loss'].iloc[-1] if 'loss' in df else 0
            final_val_loss = df['val_loss'].iloc[-1] if 'val_loss' in df else 0
            min_val_loss = df['val_loss'].min() if 'val_loss' in df else 0

            # Detect overfitting
            overfitting_score = 0
            if 'val_accuracy' in df and 'accuracy' in df:
                acc_gap = final_train_acc - final_val_acc
                overfitting_score = max(0, acc_gap)

            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Accuracy plot
            if 'accuracy' in df and 'val_accuracy' in df:
                axes[0, 0].plot(df.index, df['accuracy'], label='Training Accuracy', marker='o')
                axes[0, 0].plot(df.index, df['val_accuracy'], label='Validation Accuracy', marker='s')
                axes[0, 0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
                axes[0, 0].set_title('Model Accuracy')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Loss plot
            if 'loss' in df and 'val_loss' in df:
                axes[0, 1].plot(df.index, df['loss'], label='Training Loss', marker='o')
                axes[0, 1].plot(df.index, df['val_loss'], label='Validation Loss', marker='s')
                axes[0, 1].set_title('Model Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

            # Learning rate plot (if available)
            if 'lr' in df:
                axes[1, 0].plot(df.index, df['lr'], label='Learning Rate', marker='o', color='green')
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_yscale('log')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Performance summary
            axes[1, 1].axis('off')
            summary_text = f"""
            Training Summary:

            Best Validation Accuracy: {best_val_acc:.4f}
            Best Epoch: {best_epoch}
            Final Training Accuracy: {final_train_acc:.4f}
            Final Validation Accuracy: {final_val_acc:.4f}

            Final Training Loss: {final_train_loss:.4f}
            Final Validation Loss: {final_val_loss:.4f}
            Minimum Validation Loss: {min_val_loss:.4f}

            Overfitting Score: {overfitting_score:.4f}
            """
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {save_path}")

            plt.show()

            # Compile results
            results = {
                'analysis_type': 'training_history',
                'final_train_accuracy': final_train_acc,
                'final_val_accuracy': final_val_acc,
                'best_val_accuracy': best_val_acc,
                'best_epoch': int(best_epoch),
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'min_val_loss': min_val_loss,
                'overfitting_score': overfitting_score,
                'total_epochs': len(df)
            }

            self.analysis_results['training_history'] = results
            logger.info("Training history analysis completed")

            return results

        except Exception as e:
            logger.error(f"Error analyzing training history: {e}")
            return {}

    def analyze_model_complexity(self) -> Dict[str, Any]:
        """
        Analyze model complexity and architecture.

        Returns:
            Dictionary containing complexity analysis
        """
        try:
            if self.model is None:
                logger.error("No model loaded for analysis")
                return {}

            # Model architecture analysis
            total_params = self.model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
            non_trainable_params = total_params - trainable_params

            # Layer analysis
            layer_info = []
            for i, layer in enumerate(self.model.layers):
                layer_params = layer.count_params()
                layer_info.append({
                    'layer_index': i,
                    'layer_name': layer.name,
                    'layer_type': type(layer).__name__,
                    'output_shape': str(layer.output_shape),
                    'parameters': layer_params
                })

            # Model size estimation (MB)
            model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32

            results = {
                'analysis_type': 'model_complexity',
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'non_trainable_parameters': non_trainable_params,
                'model_size_mb': model_size_mb,
                'total_layers': len(self.model.layers),
                'layer_info': layer_info,
                'input_shape': str(self.model.input_shape),
                'output_shape': str(self.model.output_shape)
            }

            self.analysis_results['model_complexity'] = results
            logger.info(f"Model complexity analysis completed: {total_params:,} parameters")

            return results

        except Exception as e:
            logger.error(f"Error analyzing model complexity: {e}")
            return {}

    def analyze_prediction_confidence(self, X_test: np.ndarray, y_test: np.ndarray,
                                    class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze prediction confidence distribution.

        Args:
            X_test: Test features
            y_test: Test labels
            class_names: List of class names

        Returns:
            Dictionary containing confidence analysis
        """
        try:
            if self.model is None:
                logger.error("No model loaded for analysis")
                return {}

            # Make predictions
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)

            # Calculate confidence scores
            confidence_scores = np.max(y_pred_proba, axis=1)

            # Analyze confidence distribution
            confidence_stats = {
                'mean_confidence': np.mean(confidence_scores),
                'std_confidence': np.std(confidence_scores),
                'min_confidence': np.min(confidence_scores),
                'max_confidence': np.max(confidence_scores),
                'median_confidence': np.median(confidence_scores)
            }

            # Confidence by correctness
            correct_predictions = (y_pred == y_test)
            correct_confidence = confidence_scores[correct_predictions]
            incorrect_confidence = confidence_scores[~correct_predictions]

            # Create confidence analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Overall confidence distribution
            axes[0, 0].hist(confidence_scores, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(confidence_stats['mean_confidence'], color='red',
                              linestyle='--', label=f"Mean: {confidence_stats['mean_confidence']:.3f}")
            axes[0, 0].set_title('Overall Confidence Distribution')
            axes[0, 0].set_xlabel('Confidence Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Confidence by correctness
            axes[0, 1].hist(correct_confidence, bins=30, alpha=0.7, label='Correct', color='green')
            axes[0, 1].hist(incorrect_confidence, bins=30, alpha=0.7, label='Incorrect', color='red')
            axes[0, 1].set_title('Confidence by Prediction Correctness')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Confidence vs Accuracy
            confidence_bins = np.linspace(0, 1, 11)
            bin_accuracies = []
            bin_counts = []

            for i in range(len(confidence_bins) - 1):
                mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
                if np.sum(mask) > 0:
                    bin_accuracy = np.mean(correct_predictions[mask])
                    bin_accuracies.append(bin_accuracy)
                    bin_counts.append(np.sum(mask))
                else:
                    bin_accuracies.append(0)
                    bin_counts.append(0)

            bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
            axes[1, 0].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7)
            axes[1, 0].set_title('Accuracy vs Confidence Bins')
            axes[1, 0].set_xlabel('Confidence Bin')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].grid(True, alpha=0.3)

            # Summary statistics
            axes[1, 1].axis('off')
            summary_text = f"""
            Confidence Analysis Summary:

            Mean Confidence: {confidence_stats['mean_confidence']:.4f}
            Std Confidence: {confidence_stats['std_confidence']:.4f}
            Min Confidence: {confidence_stats['min_confidence']:.4f}
            Max Confidence: {confidence_stats['max_confidence']:.4f}

            Correct Predictions: {np.sum(correct_predictions)}/{len(y_test)}
            Mean Confidence (Correct): {np.mean(correct_confidence):.4f}
            Mean Confidence (Incorrect): {np.mean(incorrect_confidence):.4f}

            High Confidence (>0.9): {np.sum(confidence_scores > 0.9)}/{len(confidence_scores)}
            Low Confidence (<0.5): {np.sum(confidence_scores < 0.5)}/{len(confidence_scores)}
            """
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')

            plt.tight_layout()
            plt.show()

            results = {
                'analysis_type': 'prediction_confidence',
                'confidence_stats': confidence_stats,
                'correct_confidence_mean': np.mean(correct_confidence) if len(correct_confidence) > 0 else 0,
                'incorrect_confidence_mean': np.mean(incorrect_confidence) if len(incorrect_confidence) > 0 else 0,
                'high_confidence_count': int(np.sum(confidence_scores > 0.9)),
                'low_confidence_count': int(np.sum(confidence_scores < 0.5)),
                'total_predictions': len(y_test),
                'correct_predictions': int(np.sum(correct_predictions))
            }

            self.analysis_results['prediction_confidence'] = results
            logger.info("Prediction confidence analysis completed")

            return results

        except Exception as e:
            logger.error(f"Error analyzing prediction confidence: {e}")
            return {}

    def generate_performance_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance analysis report.

        Args:
            save_path: Path to save the report

        Returns:
            Formatted report string
        """
        try:
            report = []
            report.append("="*80)
            report.append("FRESHHARVEST PERFORMANCE ANALYSIS REPORT")
            report.append("="*80)
            report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Model: {self.model_path}")
            report.append("")

            # Training history analysis
            if 'training_history' in self.analysis_results:
                history = self.analysis_results['training_history']
                report.append("TRAINING HISTORY ANALYSIS:")
                report.append("-" * 40)
                report.append(f"Best Validation Accuracy: {history['best_val_accuracy']:.4f}")
                report.append(f"Best Epoch: {history['best_epoch']}")
                report.append(f"Final Training Accuracy: {history['final_train_accuracy']:.4f}")
                report.append(f"Final Validation Accuracy: {history['final_val_accuracy']:.4f}")
                report.append(f"Overfitting Score: {history['overfitting_score']:.4f}")
                report.append(f"Total Epochs: {history['total_epochs']}")
                report.append("")

            # Model complexity analysis
            if 'model_complexity' in self.analysis_results:
                complexity = self.analysis_results['model_complexity']
                report.append("MODEL COMPLEXITY ANALYSIS:")
                report.append("-" * 40)
                report.append(f"Total Parameters: {complexity['total_parameters']:,}")
                report.append(f"Trainable Parameters: {complexity['trainable_parameters']:,}")
                report.append(f"Model Size: {complexity['model_size_mb']:.2f} MB")
                report.append(f"Total Layers: {complexity['total_layers']}")
                report.append(f"Input Shape: {complexity['input_shape']}")
                report.append(f"Output Shape: {complexity['output_shape']}")
                report.append("")

            # Confidence analysis
            if 'prediction_confidence' in self.analysis_results:
                confidence = self.analysis_results['prediction_confidence']
                report.append("PREDICTION CONFIDENCE ANALYSIS:")
                report.append("-" * 40)
                stats = confidence['confidence_stats']
                report.append(f"Mean Confidence: {stats['mean_confidence']:.4f}")
                report.append(f"Confidence Std: {stats['std_confidence']:.4f}")
                report.append(f"Correct Predictions Confidence: {confidence['correct_confidence_mean']:.4f}")
                report.append(f"Incorrect Predictions Confidence: {confidence['incorrect_confidence_mean']:.4f}")
                report.append(f"High Confidence Predictions (>0.9): {confidence['high_confidence_count']}")
                report.append(f"Low Confidence Predictions (<0.5): {confidence['low_confidence_count']}")
                report.append("")

            report.append("="*80)

            report_text = "\n".join(report)

            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Performance report saved to {save_path}")

            return report_text

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return ""

def analyze_model_performance(model_path: str, history: Optional[Dict] = None,
                            X_test: Optional[np.ndarray] = None,
                            y_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Convenience function for comprehensive model performance analysis.

    Args:
        model_path: Path to the trained model
        history: Training history dictionary
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary containing all analysis results
    """
    analyzer = PerformanceAnalyzer(model_path)

    if not analyzer.load_model():
        return {}

    results = {}

    # Analyze training history if provided
    if history:
        results['training_history'] = analyzer.analyze_training_history(history)

    # Analyze model complexity
    results['model_complexity'] = analyzer.analyze_model_complexity()

    # Analyze prediction confidence if test data provided
    if X_test is not None and y_test is not None:
        results['prediction_confidence'] = analyzer.analyze_prediction_confidence(X_test, y_test)

    return results
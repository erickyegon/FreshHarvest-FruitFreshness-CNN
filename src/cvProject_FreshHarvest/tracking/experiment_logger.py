"""
Experiment Logger for FreshHarvest
=================================

This module provides comprehensive experiment tracking and logging
for the FreshHarvest fruit freshness classification system.

Author: FreshHarvest Team
Version: 1.0.0
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import pickle
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from cvProject_FreshHarvest.utils.common import read_yaml

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """
    Comprehensive experiment tracking for FreshHarvest model development.

    Tracks experiments, metrics, parameters, and artifacts for the 96.50% accuracy model.
    """

    def __init__(self, config_path: str = "config/config.yaml", experiment_name: str = None):
        """
        Initialize experiment logger.

        Args:
            config_path: Path to configuration file
            experiment_name: Name of the experiment
        """
        self.config = read_yaml(config_path)
        self.tracking_config = self.config.get('tracking', {})

        # Experiment settings
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_id = f"{self.experiment_name}_{int(time.time())}"

        # Tracking directories
        self.base_dir = Path(self.tracking_config.get('base_dir', 'experiments'))
        self.experiment_dir = self.base_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize experiment data
        self.experiment_data = {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'running',
            'target_accuracy': 0.965,  # 96.50% target
            'parameters': {},
            'metrics': {},
            'artifacts': {},
            'logs': [],
            'tags': ['fruit_classification', '96.50_accuracy'],
            'notes': ''
        }

        # Save initial experiment data
        self._save_experiment_data()

        logger.info(f"Experiment logger initialized: {self.experiment_id}")

    def log_parameters(self, parameters: Dict[str, Any]):
        """
        Log experiment parameters.

        Args:
            parameters: Dictionary of parameters to log
        """
        try:
            self.experiment_data['parameters'].update(parameters)
            self._save_experiment_data()

            logger.info(f"Logged parameters: {list(parameters.keys())}")

        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_metric(self, metric_name: str, value: float, step: int = None, epoch: int = None):
        """
        Log a single metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Training step (optional)
            epoch: Training epoch (optional)
        """
        try:
            if metric_name not in self.experiment_data['metrics']:
                self.experiment_data['metrics'][metric_name] = []

            metric_entry = {
                'value': float(value),
                'timestamp': datetime.now().isoformat(),
                'step': step,
                'epoch': epoch
            }

            self.experiment_data['metrics'][metric_name].append(metric_entry)

            # Special handling for accuracy metrics
            if metric_name == 'validation_accuracy' and value >= 0.965:
                self.log_event(f"🎯 Target accuracy achieved: {value:.4f}")

            self._save_experiment_data()

            logger.info(f"Logged metric {metric_name}: {value}")

        except Exception as e:
            logger.error(f"Failed to log metric {metric_name}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None, epoch: int = None):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step (optional)
            epoch: Training epoch (optional)
        """
        try:
            for metric_name, value in metrics.items():
                self.log_metric(metric_name, value, step, epoch)

            logger.info(f"Logged {len(metrics)} metrics")

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_artifact(self, artifact_name: str, artifact_path: str, artifact_type: str = 'file'):
        """
        Log an experiment artifact.

        Args:
            artifact_name: Name of the artifact
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (file, model, plot, etc.)
        """
        try:
            artifact_info = {
                'path': str(artifact_path),
                'type': artifact_type,
                'timestamp': datetime.now().isoformat(),
                'size_bytes': Path(artifact_path).stat().st_size if Path(artifact_path).exists() else 0
            }

            self.experiment_data['artifacts'][artifact_name] = artifact_info
            self._save_experiment_data()

            logger.info(f"Logged artifact {artifact_name}: {artifact_path}")

        except Exception as e:
            logger.error(f"Failed to log artifact {artifact_name}: {e}")

    def log_model(self, model_path: str, model_metrics: Dict[str, float] = None):
        """
        Log a trained model.

        Args:
            model_path: Path to the saved model
            model_metrics: Model performance metrics
        """
        try:
            model_info = {
                'path': str(model_path),
                'type': 'model',
                'timestamp': datetime.now().isoformat(),
                'size_bytes': Path(model_path).stat().st_size if Path(model_path).exists() else 0,
                'metrics': model_metrics or {}
            }

            self.experiment_data['artifacts']['model'] = model_info

            # Log model metrics
            if model_metrics:
                for metric_name, value in model_metrics.items():
                    self.log_metric(f"model_{metric_name}", value)

            self._save_experiment_data()

            logger.info(f"Logged model: {model_path}")

        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_event(self, message: str, level: str = 'info'):
        """
        Log an experiment event.

        Args:
            message: Event message
            level: Log level (info, warning, error)
        """
        try:
            event = {
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat()
            }

            self.experiment_data['logs'].append(event)
            self._save_experiment_data()

            # Also log to application logger
            if level == 'error':
                logger.error(f"Experiment event: {message}")
            elif level == 'warning':
                logger.warning(f"Experiment event: {message}")
            else:
                logger.info(f"Experiment event: {message}")

        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None):
        """
        Log confusion matrix data.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        """
        try:
            from sklearn.metrics import confusion_matrix, classification_report

            # Default class names for 96.50% accuracy model
            if class_names is None:
                class_names = [
                    "Fresh Apple", "Fresh Banana", "Fresh Orange",
                    "Rotten Apple", "Rotten Banana", "Rotten Orange"
                ]

            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Generate classification report
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

            # Save confusion matrix data
            cm_data = {
                'confusion_matrix': cm.tolist(),
                'class_names': class_names,
                'classification_report': report,
                'timestamp': datetime.now().isoformat()
            }

            # Save to file
            cm_file = self.experiment_dir / 'confusion_matrix.json'
            with open(cm_file, 'w') as f:
                json.dump(cm_data, f, indent=2)

            self.log_artifact('confusion_matrix', str(cm_file), 'analysis')

            # Log key metrics
            self.log_metric('accuracy', report['accuracy'])
            self.log_metric('macro_avg_f1', report['macro avg']['f1-score'])
            self.log_metric('weighted_avg_f1', report['weighted avg']['f1-score'])

            logger.info("Logged confusion matrix and classification report")

        except Exception as e:
            logger.error(f"Failed to log confusion matrix: {e}")

    def log_training_history(self, history: Dict[str, List[float]]):
        """
        Log training history from model training.

        Args:
            history: Training history dictionary
        """
        try:
            # Save training history
            history_file = self.experiment_dir / 'training_history.json'
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)

            self.log_artifact('training_history', str(history_file), 'history')

            # Log final metrics
            for metric_name, values in history.items():
                if values:
                    final_value = values[-1]
                    self.log_metric(f"final_{metric_name}", final_value)

                    # Log best value
                    if 'loss' in metric_name.lower():
                        best_value = min(values)
                        best_epoch = values.index(best_value)
                    else:
                        best_value = max(values)
                        best_epoch = values.index(best_value)

                    self.log_metric(f"best_{metric_name}", best_value)
                    self.log_metric(f"best_{metric_name}_epoch", best_epoch)

            logger.info("Logged training history")

        except Exception as e:
            logger.error(f"Failed to log training history: {e}")

    def add_tag(self, tag: str):
        """
        Add a tag to the experiment.

        Args:
            tag: Tag to add
        """
        try:
            if tag not in self.experiment_data['tags']:
                self.experiment_data['tags'].append(tag)
                self._save_experiment_data()

                logger.info(f"Added tag: {tag}")

        except Exception as e:
            logger.error(f"Failed to add tag: {e}")

    def add_note(self, note: str):
        """
        Add a note to the experiment.

        Args:
            note: Note to add
        """
        try:
            if self.experiment_data['notes']:
                self.experiment_data['notes'] += f"\n{note}"
            else:
                self.experiment_data['notes'] = note

            self._save_experiment_data()

            logger.info("Added experiment note")

        except Exception as e:
            logger.error(f"Failed to add note: {e}")

    def finish_experiment(self, status: str = 'completed'):
        """
        Mark experiment as finished.

        Args:
            status: Final experiment status
        """
        try:
            self.experiment_data['end_time'] = datetime.now().isoformat()
            self.experiment_data['status'] = status

            # Calculate experiment duration
            start_time = datetime.fromisoformat(self.experiment_data['start_time'])
            end_time = datetime.fromisoformat(self.experiment_data['end_time'])
            duration = (end_time - start_time).total_seconds()

            self.experiment_data['duration_seconds'] = duration

            self._save_experiment_data()

            # Generate experiment summary
            self._generate_experiment_summary()

            logger.info(f"Experiment finished with status: {status}")

        except Exception as e:
            logger.error(f"Failed to finish experiment: {e}")

    def _save_experiment_data(self):
        """Save experiment data to file."""
        try:
            experiment_file = self.experiment_dir / 'experiment.json'
            with open(experiment_file, 'w') as f:
                json.dump(self.experiment_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save experiment data: {e}")

    def _generate_experiment_summary(self):
        """Generate experiment summary report."""
        try:
            summary = []
            summary.append("🍎 FreshHarvest Experiment Summary")
            summary.append("=" * 50)
            summary.append(f"Experiment ID: {self.experiment_id}")
            summary.append(f"Experiment Name: {self.experiment_name}")
            summary.append(f"Target Accuracy: 96.50%")
            summary.append(f"Status: {self.experiment_data['status']}")
            summary.append("")

            # Duration
            if 'duration_seconds' in self.experiment_data:
                duration = self.experiment_data['duration_seconds']
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                summary.append(f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
                summary.append("")

            # Key metrics
            summary.append("📊 Key Metrics:")
            metrics = self.experiment_data['metrics']

            # Find best accuracy
            if 'validation_accuracy' in metrics:
                accuracies = [entry['value'] for entry in metrics['validation_accuracy']]
                best_accuracy = max(accuracies)
                summary.append(f"  Best Validation Accuracy: {best_accuracy:.4f}")

                if best_accuracy >= 0.965:
                    summary.append("  ✅ Target accuracy (96.50%) achieved!")
                else:
                    summary.append(f"  ⚠️ Target accuracy not reached (need {0.965 - best_accuracy:.4f} more)")

            # Other key metrics
            for metric_name in ['accuracy', 'loss', 'val_loss']:
                if metric_name in metrics:
                    values = [entry['value'] for entry in metrics[metric_name]]
                    if 'loss' in metric_name:
                        best_value = min(values)
                    else:
                        best_value = max(values)
                    summary.append(f"  Best {metric_name}: {best_value:.4f}")

            summary.append("")

            # Parameters
            if self.experiment_data['parameters']:
                summary.append("⚙️ Key Parameters:")
                for param, value in self.experiment_data['parameters'].items():
                    summary.append(f"  {param}: {value}")
                summary.append("")

            # Artifacts
            if self.experiment_data['artifacts']:
                summary.append("📁 Artifacts:")
                for artifact_name, artifact_info in self.experiment_data['artifacts'].items():
                    summary.append(f"  {artifact_name}: {artifact_info['path']}")
                summary.append("")

            # Tags
            if self.experiment_data['tags']:
                summary.append(f"🏷️ Tags: {', '.join(self.experiment_data['tags'])}")
                summary.append("")

            # Notes
            if self.experiment_data['notes']:
                summary.append("📝 Notes:")
                summary.append(self.experiment_data['notes'])
                summary.append("")

            # Save summary
            summary_text = "\n".join(summary)
            summary_file = self.experiment_dir / 'summary.txt'
            with open(summary_file, 'w') as f:
                f.write(summary_text)

            self.log_artifact('experiment_summary', str(summary_file), 'report')

            logger.info("Generated experiment summary")

        except Exception as e:
            logger.error(f"Failed to generate experiment summary: {e}")

    def get_experiment_data(self) -> Dict[str, Any]:
        """
        Get current experiment data.

        Returns:
            Experiment data dictionary
        """
        return self.experiment_data.copy()

    def compare_with_experiment(self, other_experiment_id: str) -> Dict[str, Any]:
        """
        Compare current experiment with another experiment.

        Args:
            other_experiment_id: ID of experiment to compare with

        Returns:
            Comparison results
        """
        try:
            # Load other experiment data
            other_exp_file = self.base_dir / other_experiment_id / 'experiment.json'

            if not other_exp_file.exists():
                return {'error': f'Experiment {other_experiment_id} not found'}

            with open(other_exp_file, 'r') as f:
                other_data = json.load(f)

            # Compare key metrics
            comparison = {
                'current_experiment': self.experiment_id,
                'compared_experiment': other_experiment_id,
                'metric_comparison': {},
                'parameter_comparison': {},
                'summary': {}
            }

            # Compare metrics
            current_metrics = self.experiment_data['metrics']
            other_metrics = other_data['metrics']

            for metric_name in set(current_metrics.keys()) | set(other_metrics.keys()):
                current_values = [entry['value'] for entry in current_metrics.get(metric_name, [])]
                other_values = [entry['value'] for entry in other_metrics.get(metric_name, [])]

                if current_values and other_values:
                    if 'loss' in metric_name:
                        current_best = min(current_values)
                        other_best = min(other_values)
                    else:
                        current_best = max(current_values)
                        other_best = max(other_values)

                    comparison['metric_comparison'][metric_name] = {
                        'current': current_best,
                        'other': other_best,
                        'difference': current_best - other_best,
                        'improvement': current_best > other_best if 'loss' not in metric_name else current_best < other_best
                    }

            # Compare parameters
            current_params = self.experiment_data['parameters']
            other_params = other_data['parameters']

            for param_name in set(current_params.keys()) | set(other_params.keys()):
                comparison['parameter_comparison'][param_name] = {
                    'current': current_params.get(param_name),
                    'other': other_params.get(param_name),
                    'changed': current_params.get(param_name) != other_params.get(param_name)
                }

            # Generate summary
            improvements = sum(1 for comp in comparison['metric_comparison'].values() if comp['improvement'])
            total_metrics = len(comparison['metric_comparison'])

            comparison['summary'] = {
                'improved_metrics': improvements,
                'total_metrics': total_metrics,
                'improvement_rate': improvements / total_metrics if total_metrics > 0 else 0,
                'overall_better': improvements > total_metrics / 2
            }

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare experiments: {e}")
            return {'error': str(e)}

def create_experiment_logger(experiment_name: str = None, config_path: str = "config/config.yaml") -> ExperimentLogger:
    """
    Convenience function to create an experiment logger.

    Args:
        experiment_name: Name of the experiment
        config_path: Path to configuration file

    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(config_path, experiment_name)
"""
Data Drift Monitor for FreshHarvest
==================================

This module provides comprehensive data drift monitoring for the FreshHarvest
fruit freshness classification system targeting 96.50% accuracy.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
import json
import time
from datetime import datetime, timedelta
import pickle
from scipy import stats
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

logger = logging.getLogger(__name__)

class DataDriftMonitor:
    """
    Comprehensive data drift monitoring for FreshHarvest production system.

    Monitors statistical drift, feature drift, and prediction drift to maintain
    the 96.50% accuracy target in production.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data drift monitor.

        Args:
            config_path: Path to configuration file
        """
        try:
            from cvProject_FreshHarvest.utils.common import read_yaml
            self.config = read_yaml(config_path)
        except:
            self.config = {}

        self.drift_config = self.config.get('monitoring', {}).get('data_drift', {})

        # Drift detection thresholds
        self.statistical_threshold = self.drift_config.get('statistical_threshold', 0.05)
        self.feature_drift_threshold = self.drift_config.get('feature_drift_threshold', 0.1)
        self.prediction_drift_threshold = self.drift_config.get('prediction_drift_threshold', 0.05)
        self.accuracy_drop_threshold = self.drift_config.get('accuracy_drop_threshold', 0.02)

        # Reference data storage
        self.reference_data = None
        self.reference_predictions = None
        self.reference_features = None
        self.baseline_accuracy = 0.965  # 96.50% target

        # Drift history
        self.drift_history = []
        self.alert_history = []

        # Class names for 96.50% accuracy model
        self.class_names = [
            "Fresh Apple", "Fresh Banana", "Fresh Orange",
            "Rotten Apple", "Rotten Banana", "Rotten Orange"
        ]

        logger.info("Data drift monitor initialized for 96.50% accuracy model")

    def set_reference_data(self, reference_data: Dict[str, Any]):
        """
        Set reference data for drift detection.

        Args:
            reference_data: Dictionary containing reference statistics
        """
        try:
            self.reference_data = reference_data

            # Extract reference statistics
            self.reference_features = reference_data.get('features', {})
            self.reference_predictions = reference_data.get('predictions', {})
            self.baseline_accuracy = reference_data.get('accuracy', 0.965)

            logger.info(f"Reference data set with baseline accuracy: {self.baseline_accuracy:.4f}")

        except Exception as e:
            logger.error(f"Failed to set reference data: {e}")

    def extract_image_features(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract statistical features from images for drift detection.

        Args:
            images: Array of images

        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}

            # Basic statistical features
            features['mean_intensity'] = np.mean(images, axis=(1, 2, 3))
            features['std_intensity'] = np.std(images, axis=(1, 2, 3))
            features['min_intensity'] = np.min(images, axis=(1, 2, 3))
            features['max_intensity'] = np.max(images, axis=(1, 2, 3))

            # Color channel statistics
            if len(images.shape) == 4 and images.shape[-1] == 3:  # RGB images
                features['mean_red'] = np.mean(images[:, :, :, 0], axis=(1, 2))
                features['mean_green'] = np.mean(images[:, :, :, 1], axis=(1, 2))
                features['mean_blue'] = np.mean(images[:, :, :, 2], axis=(1, 2))

                features['std_red'] = np.std(images[:, :, :, 0], axis=(1, 2))
                features['std_green'] = np.std(images[:, :, :, 1], axis=(1, 2))
                features['std_blue'] = np.std(images[:, :, :, 2], axis=(1, 2))

            # Texture features (simplified)
            features['contrast'] = np.std(images, axis=(1, 2, 3))
            features['brightness'] = np.mean(images, axis=(1, 2, 3))

            # Edge density (simplified)
            edge_density = []
            for img in images:
                if len(img.shape) == 3:
                    gray = np.mean(img, axis=2)
                else:
                    gray = img

                # Simple edge detection using gradient
                grad_x = np.gradient(gray, axis=0)
                grad_y = np.gradient(gray, axis=1)
                edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                edge_density.append(np.mean(edge_magnitude))

            features['edge_density'] = np.array(edge_density)

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}

    def detect_statistical_drift(self, current_features: Dict[str, np.ndarray],
                                reference_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect statistical drift using Kolmogorov-Smirnov test.

        Args:
            current_features: Current batch features
            reference_features: Reference features

        Returns:
            Drift detection results
        """
        try:
            drift_results = {
                'drift_detected': False,
                'drift_features': [],
                'p_values': {},
                'drift_scores': {},
                'overall_drift_score': 0.0
            }

            drift_scores = []

            for feature_name in current_features.keys():
                if feature_name in reference_features:
                    # Perform Kolmogorov-Smirnov test
                    current_values = current_features[feature_name]
                    reference_values = reference_features[feature_name]

                    ks_statistic, p_value = stats.ks_2samp(reference_values, current_values)

                    drift_results['p_values'][feature_name] = p_value
                    drift_results['drift_scores'][feature_name] = ks_statistic

                    # Check if drift detected
                    if p_value < self.statistical_threshold:
                        drift_results['drift_features'].append(feature_name)
                        drift_results['drift_detected'] = True

                    drift_scores.append(ks_statistic)

            # Calculate overall drift score
            if drift_scores:
                drift_results['overall_drift_score'] = np.mean(drift_scores)

            return drift_results

        except Exception as e:
            logger.error(f"Statistical drift detection failed: {e}")
            return {'error': str(e)}

    def detect_prediction_drift(self, current_predictions: np.ndarray,
                              reference_predictions: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in prediction distributions.

        Args:
            current_predictions: Current prediction probabilities
            reference_predictions: Reference prediction probabilities

        Returns:
            Prediction drift results
        """
        try:
            drift_results = {
                'drift_detected': False,
                'class_drift': {},
                'distribution_shift': {},
                'overall_drift_score': 0.0
            }

            # Calculate class distribution shifts
            current_dist = np.mean(current_predictions, axis=0)
            reference_dist = np.mean(reference_predictions, axis=0)

            class_drift_scores = []

            for i, class_name in enumerate(self.class_names):
                # Calculate distribution difference
                dist_diff = abs(current_dist[i] - reference_dist[i])
                drift_results['distribution_shift'][class_name] = {
                    'current': float(current_dist[i]),
                    'reference': float(reference_dist[i]),
                    'difference': float(dist_diff)
                }

                # Check for significant drift
                if dist_diff > self.prediction_drift_threshold:
                    drift_results['class_drift'][class_name] = True
                    drift_results['drift_detected'] = True
                    class_drift_scores.append(dist_diff)
                else:
                    drift_results['class_drift'][class_name] = False

            # Calculate overall prediction drift score
            if class_drift_scores:
                drift_results['overall_drift_score'] = np.mean(class_drift_scores)
            else:
                drift_results['overall_drift_score'] = np.mean([
                    abs(current_dist[i] - reference_dist[i])
                    for i in range(len(current_dist))
                ])

            return drift_results

        except Exception as e:
            logger.error(f"Prediction drift detection failed: {e}")
            return {'error': str(e)}

    def detect_accuracy_drift(self, current_accuracy: float) -> Dict[str, Any]:
        """
        Detect accuracy drift from baseline.

        Args:
            current_accuracy: Current model accuracy

        Returns:
            Accuracy drift results
        """
        try:
            accuracy_drop = self.baseline_accuracy - current_accuracy

            drift_results = {
                'accuracy_drift_detected': accuracy_drop > self.accuracy_drop_threshold,
                'current_accuracy': current_accuracy,
                'baseline_accuracy': self.baseline_accuracy,
                'accuracy_drop': accuracy_drop,
                'relative_drop_percent': (accuracy_drop / self.baseline_accuracy) * 100,
                'below_target': current_accuracy < 0.965  # 96.50% target
            }

            return drift_results

        except Exception as e:
            logger.error(f"Accuracy drift detection failed: {e}")
            return {'error': str(e)}

    def monitor_batch(self, images: np.ndarray, predictions: np.ndarray,
                     true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Monitor a batch of data for drift.

        Args:
            images: Batch of images
            predictions: Model predictions
            true_labels: True labels (optional)

        Returns:
            Comprehensive drift monitoring results
        """
        try:
            monitoring_results = {
                'timestamp': datetime.now().isoformat(),
                'batch_size': len(images),
                'drift_detected': False,
                'alerts': [],
                'statistical_drift': {},
                'prediction_drift': {},
                'accuracy_drift': {},
                'recommendations': []
            }

            # Extract features from current batch
            current_features = self.extract_image_features(images)

            # Statistical drift detection
            if self.reference_features:
                statistical_drift = self.detect_statistical_drift(current_features, self.reference_features)
                monitoring_results['statistical_drift'] = statistical_drift

                if statistical_drift.get('drift_detected', False):
                    monitoring_results['drift_detected'] = True
                    monitoring_results['alerts'].append({
                        'type': 'statistical_drift',
                        'severity': 'warning',
                        'message': f"Statistical drift detected in features: {statistical_drift['drift_features']}",
                        'drift_score': statistical_drift['overall_drift_score']
                    })

            # Prediction drift detection
            if self.reference_predictions is not None:
                prediction_drift = self.detect_prediction_drift(predictions, self.reference_predictions)
                monitoring_results['prediction_drift'] = prediction_drift

                if prediction_drift.get('drift_detected', False):
                    monitoring_results['drift_detected'] = True
                    drifted_classes = [cls for cls, drifted in prediction_drift['class_drift'].items() if drifted]
                    monitoring_results['alerts'].append({
                        'type': 'prediction_drift',
                        'severity': 'warning',
                        'message': f"Prediction drift detected in classes: {drifted_classes}",
                        'drift_score': prediction_drift['overall_drift_score']
                    })

            # Accuracy drift detection (if true labels available)
            if true_labels is not None:
                predicted_labels = np.argmax(predictions, axis=1)
                current_accuracy = accuracy_score(true_labels, predicted_labels)

                accuracy_drift = self.detect_accuracy_drift(current_accuracy)
                monitoring_results['accuracy_drift'] = accuracy_drift

                if accuracy_drift.get('accuracy_drift_detected', False):
                    monitoring_results['drift_detected'] = True
                    monitoring_results['alerts'].append({
                        'type': 'accuracy_drift',
                        'severity': 'critical' if accuracy_drift['below_target'] else 'warning',
                        'message': f"Accuracy dropped by {accuracy_drift['accuracy_drop']:.4f} from baseline",
                        'current_accuracy': current_accuracy,
                        'target_accuracy': 0.965
                    })

            # Generate recommendations
            monitoring_results['recommendations'] = self._generate_recommendations(monitoring_results)

            # Store in drift history
            self.drift_history.append(monitoring_results)

            # Store alerts
            if monitoring_results['alerts']:
                self.alert_history.extend(monitoring_results['alerts'])

            logger.info(f"Batch monitoring completed. Drift detected: {monitoring_results['drift_detected']}")

            return monitoring_results

        except Exception as e:
            logger.error(f"Batch monitoring failed: {e}")
            return {'error': str(e)}

    def _generate_recommendations(self, monitoring_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on drift detection results."""
        recommendations = []

        try:
            # Statistical drift recommendations
            if monitoring_results.get('statistical_drift', {}).get('drift_detected', False):
                recommendations.append("🔍 Statistical drift detected - investigate data collection process")
                recommendations.append("📊 Consider retraining model with recent data")
                recommendations.append("🔧 Review data preprocessing pipeline")

            # Prediction drift recommendations
            if monitoring_results.get('prediction_drift', {}).get('drift_detected', False):
                recommendations.append("🎯 Prediction distribution has shifted - model may need retraining")
                recommendations.append("📈 Monitor prediction confidence scores")
                recommendations.append("🔄 Consider gradual model updates")

            # Accuracy drift recommendations
            accuracy_drift = monitoring_results.get('accuracy_drift', {})
            if accuracy_drift.get('accuracy_drift_detected', False):
                if accuracy_drift.get('below_target', False):
                    recommendations.append("🚨 CRITICAL: Accuracy below 96.50% target - immediate action required")
                    recommendations.append("⚡ Consider emergency model rollback")
                    recommendations.append("🔧 Investigate data quality issues")
                else:
                    recommendations.append("⚠️ Accuracy decline detected - monitor closely")
                    recommendations.append("📊 Schedule model retraining")

            # General recommendations
            if not recommendations:
                recommendations.append("✅ No significant drift detected - system operating normally")
                recommendations.append("📊 Continue regular monitoring")

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["❌ Error generating recommendations"]

    def generate_drift_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive drift monitoring report.

        Args:
            time_window_hours: Time window for report generation

        Returns:
            Drift monitoring report
        """
        try:
            # Filter drift history by time window
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            recent_history = [
                entry for entry in self.drift_history
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time
            ]

            # Calculate summary statistics
            total_batches = len(recent_history)
            drift_detected_count = sum(1 for entry in recent_history if entry['drift_detected'])
            drift_rate = drift_detected_count / total_batches if total_batches > 0 else 0

            # Alert summary
            recent_alerts = [
                alert for entry in recent_history
                for alert in entry.get('alerts', [])
            ]

            alert_summary = {}
            for alert in recent_alerts:
                alert_type = alert['type']
                if alert_type not in alert_summary:
                    alert_summary[alert_type] = {'count': 0, 'severity_counts': {}}
                alert_summary[alert_type]['count'] += 1
                severity = alert['severity']
                alert_summary[alert_type]['severity_counts'][severity] = \
                    alert_summary[alert_type]['severity_counts'].get(severity, 0) + 1

            # Generate report
            report = {
                'report_generated_at': datetime.now().isoformat(),
                'time_window_hours': time_window_hours,
                'model_target_accuracy': '96.50%',
                'summary': {
                    'total_batches_monitored': total_batches,
                    'drift_detected_count': drift_detected_count,
                    'drift_detection_rate': drift_rate,
                    'total_alerts': len(recent_alerts)
                },
                'alert_summary': alert_summary,
                'drift_trends': self._calculate_drift_trends(recent_history),
                'recommendations': self._generate_report_recommendations(recent_history, alert_summary),
                'system_health': self._assess_system_health(drift_rate, recent_alerts)
            }

            return report

        except Exception as e:
            logger.error(f"Drift report generation failed: {e}")
            return {'error': str(e)}

    def _calculate_drift_trends(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate drift trends from history."""
        try:
            trends = {
                'statistical_drift_trend': [],
                'prediction_drift_trend': [],
                'accuracy_trend': []
            }

            for entry in history:
                # Statistical drift trend
                stat_drift = entry.get('statistical_drift', {})
                trends['statistical_drift_trend'].append({
                    'timestamp': entry['timestamp'],
                    'drift_score': stat_drift.get('overall_drift_score', 0)
                })

                # Prediction drift trend
                pred_drift = entry.get('prediction_drift', {})
                trends['prediction_drift_trend'].append({
                    'timestamp': entry['timestamp'],
                    'drift_score': pred_drift.get('overall_drift_score', 0)
                })

                # Accuracy trend
                acc_drift = entry.get('accuracy_drift', {})
                trends['accuracy_trend'].append({
                    'timestamp': entry['timestamp'],
                    'accuracy': acc_drift.get('current_accuracy', 0)
                })

            return trends

        except Exception as e:
            logger.error(f"Trend calculation failed: {e}")
            return {}

    def _generate_report_recommendations(self, history: List[Dict[str, Any]],
                                       alert_summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations for the drift report."""
        recommendations = []

        try:
            # Check for critical issues
            critical_alerts = sum(
                summary['severity_counts'].get('critical', 0)
                for summary in alert_summary.values()
            )

            if critical_alerts > 0:
                recommendations.append("🚨 CRITICAL ALERTS DETECTED - Immediate action required")
                recommendations.append("⚡ Consider emergency model rollback or intervention")

            # Check drift frequency
            total_batches = len(history)
            drift_count = sum(1 for entry in history if entry['drift_detected'])

            if total_batches > 0:
                drift_rate = drift_count / total_batches

                if drift_rate > 0.5:
                    recommendations.append("🔴 High drift rate detected - system requires attention")
                    recommendations.append("🔧 Investigate data pipeline and model performance")
                elif drift_rate > 0.2:
                    recommendations.append("🟡 Moderate drift rate - monitor closely")
                    recommendations.append("📊 Consider scheduled model retraining")
                else:
                    recommendations.append("🟢 Low drift rate - system operating normally")

            # Specific alert type recommendations
            if 'accuracy_drift' in alert_summary:
                recommendations.append("📉 Accuracy drift detected - prioritize model improvement")
                recommendations.append("🎯 Focus on maintaining 96.50% accuracy target")

            if 'statistical_drift' in alert_summary:
                recommendations.append("📊 Data distribution changes detected")
                recommendations.append("🔍 Review data collection and preprocessing")

            if 'prediction_drift' in alert_summary:
                recommendations.append("🎯 Model prediction patterns have changed")
                recommendations.append("🔄 Consider model recalibration")

            return recommendations

        except Exception as e:
            logger.error(f"Report recommendations generation failed: {e}")
            return ["❌ Error generating recommendations"]

    def _assess_system_health(self, drift_rate: float, recent_alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall system health."""
        try:
            # Count critical alerts
            critical_count = sum(1 for alert in recent_alerts if alert.get('severity') == 'critical')
            warning_count = sum(1 for alert in recent_alerts if alert.get('severity') == 'warning')

            # Determine health status
            if critical_count > 0:
                health_status = "CRITICAL"
                health_score = 0.2
            elif drift_rate > 0.5:
                health_status = "POOR"
                health_score = 0.4
            elif drift_rate > 0.2 or warning_count > 5:
                health_status = "FAIR"
                health_score = 0.6
            elif drift_rate > 0.1:
                health_status = "GOOD"
                health_score = 0.8
            else:
                health_status = "EXCELLENT"
                health_score = 1.0

            return {
                'status': health_status,
                'score': health_score,
                'drift_rate': drift_rate,
                'critical_alerts': critical_count,
                'warning_alerts': warning_count,
                'target_accuracy_maintained': health_score >= 0.8
            }

        except Exception as e:
            logger.error(f"Health assessment failed: {e}")
            return {
                'status': 'UNKNOWN',
                'score': 0.0,
                'error': str(e)
            }

    def save_monitoring_state(self, filepath: str):
        """Save monitoring state to file."""
        try:
            state = {
                'reference_data': self.reference_data,
                'reference_features': self.reference_features,
                'reference_predictions': self.reference_predictions,
                'baseline_accuracy': self.baseline_accuracy,
                'drift_history': self.drift_history[-100:],  # Keep last 100 entries
                'alert_history': self.alert_history[-100:]   # Keep last 100 alerts
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"Monitoring state saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save monitoring state: {e}")

    def load_monitoring_state(self, filepath: str):
        """Load monitoring state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.reference_data = state.get('reference_data')
            self.reference_features = state.get('reference_features')
            self.reference_predictions = state.get('reference_predictions')
            self.baseline_accuracy = state.get('baseline_accuracy', 0.965)
            self.drift_history = state.get('drift_history', [])
            self.alert_history = state.get('alert_history', [])

            logger.info(f"Monitoring state loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load monitoring state: {e}")

def create_data_drift_monitor(config_path: str = "config/config.yaml") -> DataDriftMonitor:
    """
    Convenience function to create a data drift monitor.

    Args:
        config_path: Path to configuration file

    Returns:
        DataDriftMonitor instance
    """
    return DataDriftMonitor(config_path)
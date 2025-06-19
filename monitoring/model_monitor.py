"""
Model Monitor for FreshHarvest
=============================

This module provides comprehensive model performance monitoring for the FreshHarvest
fruit freshness classification system targeting 96.50% accuracy.

Author: FreshHarvest Team
Version: 1.0.0
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta
import numpy as np
from collections import deque, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

logger = logging.getLogger(__name__)

class ModelMonitor:
    """
    Comprehensive model performance monitoring for FreshHarvest production system.

    Monitors accuracy, latency, throughput, and other performance metrics to ensure
    the 96.50% accuracy target is maintained in production.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize model monitor.

        Args:
            config_path: Path to configuration file
        """
        try:
            from cvProject_FreshHarvest.utils.common import read_yaml
            self.config = read_yaml(config_path)
        except:
            self.config = {}

        self.monitor_config = self.config.get('monitoring', {}).get('model_performance', {})

        # Performance thresholds
        self.target_accuracy = 0.965  # 96.50% target
        self.accuracy_warning_threshold = 0.96  # 96.00% warning
        self.accuracy_critical_threshold = 0.95  # 95.00% critical
        self.latency_threshold_ms = self.monitor_config.get('latency_threshold_ms', 100)
        self.throughput_threshold = self.monitor_config.get('throughput_threshold', 100)

        # Monitoring windows
        self.short_window_size = self.monitor_config.get('short_window_size', 100)
        self.long_window_size = self.monitor_config.get('long_window_size', 1000)

        # Performance metrics storage
        self.accuracy_history = deque(maxlen=self.long_window_size)
        self.latency_history = deque(maxlen=self.long_window_size)
        self.confidence_history = deque(maxlen=self.long_window_size)
        self.prediction_history = deque(maxlen=self.long_window_size)
        self.error_history = deque(maxlen=self.long_window_size)

        # Real-time metrics
        self.current_metrics = {
            'accuracy': 0.0,
            'avg_latency_ms': 0.0,
            'throughput_per_second': 0.0,
            'avg_confidence': 0.0,
            'error_rate': 0.0,
            'total_predictions': 0,
            'uptime_hours': 0.0
        }

        # Alert tracking
        self.alerts = []
        self.alert_history = deque(maxlen=1000)

        # Class names for 96.50% accuracy model
        self.class_names = [
            "Fresh Apple", "Fresh Banana", "Fresh Orange",
            "Rotten Apple", "Rotten Banana", "Rotten Orange"
        ]

        # Performance tracking
        self.start_time = time.time()
        self.last_prediction_time = None
        self.prediction_count = 0

        logger.info("Model monitor initialized for 96.50% accuracy target")

    def log_prediction(self, prediction_data: Dict[str, Any]):
        """
        Log a single prediction for monitoring.

        Args:
            prediction_data: Dictionary containing prediction information
        """
        try:
            current_time = time.time()

            # Extract prediction metrics
            predicted_class = prediction_data.get('predicted_class')
            confidence = prediction_data.get('confidence', 0.0)
            latency_ms = prediction_data.get('latency_ms', 0.0)
            true_label = prediction_data.get('true_label')
            error_occurred = prediction_data.get('error', False)

            # Update prediction count
            self.prediction_count += 1
            self.last_prediction_time = current_time

            # Log metrics
            self.confidence_history.append(confidence)
            self.latency_history.append(latency_ms)
            self.error_history.append(1 if error_occurred else 0)

            # Log prediction for class distribution tracking
            if predicted_class is not None:
                self.prediction_history.append({
                    'timestamp': current_time,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'true_label': true_label,
                    'correct': predicted_class == true_label if true_label is not None else None
                })

            # Calculate accuracy if true label available
            if true_label is not None and predicted_class is not None:
                is_correct = predicted_class == true_label
                self.accuracy_history.append(1 if is_correct else 0)

            # Update real-time metrics
            self._update_current_metrics()

            # Check for alerts
            self._check_alerts()

        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

    def log_batch_predictions(self, batch_data: List[Dict[str, Any]]):
        """
        Log a batch of predictions for monitoring.

        Args:
            batch_data: List of prediction dictionaries
        """
        try:
            for prediction_data in batch_data:
                self.log_prediction(prediction_data)

            logger.info(f"Logged batch of {len(batch_data)} predictions")

        except Exception as e:
            logger.error(f"Failed to log batch predictions: {e}")

    def _update_current_metrics(self):
        """Update current performance metrics."""
        try:
            current_time = time.time()

            # Calculate accuracy
            if self.accuracy_history:
                self.current_metrics['accuracy'] = np.mean(list(self.accuracy_history))

            # Calculate average latency
            if self.latency_history:
                self.current_metrics['avg_latency_ms'] = np.mean(list(self.latency_history))

            # Calculate average confidence
            if self.confidence_history:
                self.current_metrics['avg_confidence'] = np.mean(list(self.confidence_history))

            # Calculate error rate
            if self.error_history:
                self.current_metrics['error_rate'] = np.mean(list(self.error_history))

            # Calculate throughput (predictions per second)
            uptime_seconds = current_time - self.start_time
            if uptime_seconds > 0:
                self.current_metrics['throughput_per_second'] = self.prediction_count / uptime_seconds
                self.current_metrics['uptime_hours'] = uptime_seconds / 3600

            # Update total predictions
            self.current_metrics['total_predictions'] = self.prediction_count

        except Exception as e:
            logger.error(f"Failed to update current metrics: {e}")

    def _check_alerts(self):
        """Check for performance alerts."""
        try:
            current_time = datetime.now()
            new_alerts = []

            # Accuracy alerts
            current_accuracy = self.current_metrics['accuracy']

            if current_accuracy < self.accuracy_critical_threshold:
                new_alerts.append({
                    'type': 'accuracy_critical',
                    'severity': 'critical',
                    'message': f"Critical accuracy drop: {current_accuracy:.4f} < {self.accuracy_critical_threshold:.4f}",
                    'timestamp': current_time.isoformat(),
                    'value': current_accuracy,
                    'threshold': self.accuracy_critical_threshold
                })
            elif current_accuracy < self.accuracy_warning_threshold:
                new_alerts.append({
                    'type': 'accuracy_warning',
                    'severity': 'warning',
                    'message': f"Accuracy below warning threshold: {current_accuracy:.4f} < {self.accuracy_warning_threshold:.4f}",
                    'timestamp': current_time.isoformat(),
                    'value': current_accuracy,
                    'threshold': self.accuracy_warning_threshold
                })

            # Latency alerts
            current_latency = self.current_metrics['avg_latency_ms']
            if current_latency > self.latency_threshold_ms:
                new_alerts.append({
                    'type': 'high_latency',
                    'severity': 'warning',
                    'message': f"High latency detected: {current_latency:.2f}ms > {self.latency_threshold_ms}ms",
                    'timestamp': current_time.isoformat(),
                    'value': current_latency,
                    'threshold': self.latency_threshold_ms
                })

            # Error rate alerts
            current_error_rate = self.current_metrics['error_rate']
            if current_error_rate > 0.05:  # 5% error rate threshold
                new_alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'warning',
                    'message': f"High error rate: {current_error_rate:.2%}",
                    'timestamp': current_time.isoformat(),
                    'value': current_error_rate,
                    'threshold': 0.05
                })

            # Low confidence alerts
            current_confidence = self.current_metrics['avg_confidence']
            if current_confidence < 0.7:  # 70% confidence threshold
                new_alerts.append({
                    'type': 'low_confidence',
                    'severity': 'warning',
                    'message': f"Low average confidence: {current_confidence:.3f}",
                    'timestamp': current_time.isoformat(),
                    'value': current_confidence,
                    'threshold': 0.7
                })

            # Add new alerts
            self.alerts.extend(new_alerts)
            self.alert_history.extend(new_alerts)

            # Log alerts
            for alert in new_alerts:
                logger.warning(f"Model alert: {alert['message']}")

        except Exception as e:
            logger.error(f"Alert checking failed: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.

        Returns:
            Dictionary of current metrics
        """
        try:
            # Update metrics before returning
            self._update_current_metrics()

            # Add additional computed metrics
            metrics = self.current_metrics.copy()

            # Add target comparison
            metrics['target_accuracy'] = self.target_accuracy
            metrics['accuracy_vs_target'] = metrics['accuracy'] - self.target_accuracy
            metrics['target_achieved'] = metrics['accuracy'] >= self.target_accuracy

            # Add performance status
            if metrics['accuracy'] >= self.target_accuracy:
                metrics['performance_status'] = 'excellent'
            elif metrics['accuracy'] >= self.accuracy_warning_threshold:
                metrics['performance_status'] = 'good'
            elif metrics['accuracy'] >= self.accuracy_critical_threshold:
                metrics['performance_status'] = 'warning'
            else:
                metrics['performance_status'] = 'critical'

            # Add class distribution
            metrics['class_distribution'] = self._calculate_class_distribution()

            # Add recent alerts count
            metrics['active_alerts'] = len(self.alerts)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {}

    def _calculate_class_distribution(self) -> Dict[str, float]:
        """Calculate current class prediction distribution."""
        try:
            if not self.prediction_history:
                return {}

            # Get recent predictions (last 100)
            recent_predictions = list(self.prediction_history)[-100:]

            # Count predictions per class
            class_counts = defaultdict(int)
            total_predictions = len(recent_predictions)

            for pred in recent_predictions:
                predicted_class = pred.get('predicted_class')
                if predicted_class is not None:
                    class_counts[predicted_class] += 1

            # Calculate distribution
            distribution = {}
            for class_name in self.class_names:
                count = class_counts.get(class_name, 0)
                distribution[class_name] = count / total_predictions if total_predictions > 0 else 0.0

            return distribution

        except Exception as e:
            logger.error(f"Class distribution calculation failed: {e}")
            return {}

    def get_performance_summary(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """
        Get performance summary for a specific time window.

        Args:
            time_window_hours: Time window in hours

        Returns:
            Performance summary
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (time_window_hours * 3600)

            # Filter predictions by time window
            recent_predictions = [
                pred for pred in self.prediction_history
                if pred['timestamp'] > cutoff_time
            ]

            if not recent_predictions:
                return {
                    'time_window_hours': time_window_hours,
                    'total_predictions': 0,
                    'message': 'No predictions in time window'
                }

            # Calculate metrics for time window
            total_predictions = len(recent_predictions)
            correct_predictions = sum(1 for pred in recent_predictions if pred.get('correct', False))
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

            # Calculate confidence statistics
            confidences = [pred['confidence'] for pred in recent_predictions if 'confidence' in pred]
            avg_confidence = np.mean(confidences) if confidences else 0
            min_confidence = np.min(confidences) if confidences else 0
            max_confidence = np.max(confidences) if confidences else 0

            # Calculate class distribution
            class_counts = defaultdict(int)
            for pred in recent_predictions:
                predicted_class = pred.get('predicted_class')
                if predicted_class:
                    class_counts[predicted_class] += 1

            class_distribution = {
                class_name: class_counts[class_name] / total_predictions
                for class_name in self.class_names
            }

            # Calculate throughput
            time_span = max(pred['timestamp'] for pred in recent_predictions) - min(pred['timestamp'] for pred in recent_predictions)
            throughput = total_predictions / (time_span / 3600) if time_span > 0 else 0  # predictions per hour

            summary = {
                'time_window_hours': time_window_hours,
                'total_predictions': total_predictions,
                'accuracy': accuracy,
                'target_accuracy': self.target_accuracy,
                'accuracy_vs_target': accuracy - self.target_accuracy,
                'target_achieved': accuracy >= self.target_accuracy,
                'confidence_stats': {
                    'average': avg_confidence,
                    'minimum': min_confidence,
                    'maximum': max_confidence
                },
                'class_distribution': class_distribution,
                'throughput_per_hour': throughput,
                'performance_grade': self._calculate_performance_grade(accuracy)
            }

            return summary

        except Exception as e:
            logger.error(f"Performance summary calculation failed: {e}")
            return {'error': str(e)}

    def _calculate_performance_grade(self, accuracy: float) -> str:
        """Calculate performance grade based on accuracy."""
        if accuracy >= 0.965:  # 96.50%
            return 'A+'
        elif accuracy >= 0.96:  # 96.00%
            return 'A'
        elif accuracy >= 0.95:  # 95.00%
            return 'B'
        elif accuracy >= 0.90:  # 90.00%
            return 'C'
        else:
            return 'F'

    def get_alerts(self, severity: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent alerts.

        Args:
            severity: Filter by severity (critical, warning, info)
            limit: Maximum number of alerts to return

        Returns:
            List of alerts
        """
        try:
            alerts = list(self.alert_history)

            # Filter by severity if specified
            if severity:
                alerts = [alert for alert in alerts if alert.get('severity') == severity]

            # Sort by timestamp (most recent first)
            alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            # Limit results
            return alerts[:limit]

        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []

    def clear_alerts(self):
        """Clear current alerts."""
        try:
            self.alerts.clear()
            logger.info("Alerts cleared")

        except Exception as e:
            logger.error(f"Failed to clear alerts: {e}")

    def generate_monitoring_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.

        Returns:
            Monitoring report
        """
        try:
            current_metrics = self.get_current_metrics()

            # Performance summaries for different time windows
            performance_1h = self.get_performance_summary(1)
            performance_24h = self.get_performance_summary(24)

            # Alert summary
            critical_alerts = self.get_alerts('critical', 10)
            warning_alerts = self.get_alerts('warning', 20)

            # System health assessment
            health_score = self._calculate_health_score(current_metrics)

            report = {
                'report_generated_at': datetime.now().isoformat(),
                'model_target_accuracy': '96.50%',
                'current_metrics': current_metrics,
                'performance_summaries': {
                    'last_1_hour': performance_1h,
                    'last_24_hours': performance_24h
                },
                'alerts': {
                    'critical': critical_alerts,
                    'warning': warning_alerts,
                    'total_active': len(self.alerts)
                },
                'system_health': {
                    'score': health_score,
                    'status': self._get_health_status(health_score),
                    'uptime_hours': current_metrics.get('uptime_hours', 0),
                    'total_predictions': current_metrics.get('total_predictions', 0)
                },
                'recommendations': self._generate_monitoring_recommendations(current_metrics, critical_alerts, warning_alerts)
            }

            return report

        except Exception as e:
            logger.error(f"Monitoring report generation failed: {e}")
            return {'error': str(e)}

    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-1)."""
        try:
            score = 0.0

            # Accuracy score (40% weight)
            accuracy = metrics.get('accuracy', 0)
            if accuracy >= self.target_accuracy:
                accuracy_score = 1.0
            elif accuracy >= self.accuracy_warning_threshold:
                accuracy_score = 0.8
            elif accuracy >= self.accuracy_critical_threshold:
                accuracy_score = 0.5
            else:
                accuracy_score = 0.2

            score += accuracy_score * 0.4

            # Latency score (20% weight)
            latency = metrics.get('avg_latency_ms', 0)
            if latency <= self.latency_threshold_ms:
                latency_score = 1.0
            elif latency <= self.latency_threshold_ms * 2:
                latency_score = 0.7
            else:
                latency_score = 0.3

            score += latency_score * 0.2

            # Error rate score (20% weight)
            error_rate = metrics.get('error_rate', 0)
            if error_rate <= 0.01:  # 1%
                error_score = 1.0
            elif error_rate <= 0.05:  # 5%
                error_score = 0.7
            else:
                error_score = 0.3

            score += error_score * 0.2

            # Confidence score (20% weight)
            confidence = metrics.get('avg_confidence', 0)
            if confidence >= 0.8:
                confidence_score = 1.0
            elif confidence >= 0.7:
                confidence_score = 0.8
            else:
                confidence_score = 0.5

            score += confidence_score * 0.2

            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1

        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return 0.0

    def _get_health_status(self, health_score: float) -> str:
        """Get health status based on score."""
        if health_score >= 0.9:
            return 'EXCELLENT'
        elif health_score >= 0.8:
            return 'GOOD'
        elif health_score >= 0.6:
            return 'FAIR'
        elif health_score >= 0.4:
            return 'POOR'
        else:
            return 'CRITICAL'

    def _generate_monitoring_recommendations(self, metrics: Dict[str, Any],
                                           critical_alerts: List[Dict[str, Any]],
                                           warning_alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate monitoring recommendations."""
        recommendations = []

        try:
            # Critical alerts recommendations
            if critical_alerts:
                recommendations.append("🚨 CRITICAL ALERTS ACTIVE - Immediate action required")
                recommendations.append("⚡ Consider emergency intervention or model rollback")

            # Accuracy recommendations
            accuracy = metrics.get('accuracy', 0)
            if accuracy < self.accuracy_critical_threshold:
                recommendations.append("🔴 CRITICAL: Accuracy below 95% - immediate investigation required")
                recommendations.append("🔧 Check data quality and model integrity")
            elif accuracy < self.accuracy_warning_threshold:
                recommendations.append("🟡 Accuracy below 96% - monitor closely and investigate")
                recommendations.append("📊 Consider model retraining or recalibration")
            elif accuracy < self.target_accuracy:
                recommendations.append("📈 Accuracy below 96.50% target - optimization needed")
                recommendations.append("🎯 Focus on achieving target accuracy")
            else:
                recommendations.append("✅ Accuracy target achieved - maintain current performance")

            # Latency recommendations
            latency = metrics.get('avg_latency_ms', 0)
            if latency > self.latency_threshold_ms:
                recommendations.append(f"⏱️ High latency detected ({latency:.1f}ms) - optimize inference")
                recommendations.append("🚀 Consider model optimization or hardware scaling")

            # Error rate recommendations
            error_rate = metrics.get('error_rate', 0)
            if error_rate > 0.05:
                recommendations.append(f"❌ High error rate ({error_rate:.1%}) - investigate failures")
                recommendations.append("🔍 Review error logs and system stability")

            # Confidence recommendations
            confidence = metrics.get('avg_confidence', 0)
            if confidence < 0.7:
                recommendations.append(f"⚠️ Low confidence ({confidence:.2f}) - review model certainty")
                recommendations.append("🎯 Consider confidence threshold adjustments")

            # General recommendations
            if not recommendations:
                recommendations.append("🟢 System operating within normal parameters")
                recommendations.append("📊 Continue regular monitoring and maintenance")

            return recommendations

        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["❌ Error generating recommendations"]

    def reset_monitoring(self):
        """Reset all monitoring data."""
        try:
            self.accuracy_history.clear()
            self.latency_history.clear()
            self.confidence_history.clear()
            self.prediction_history.clear()
            self.error_history.clear()
            self.alerts.clear()

            self.prediction_count = 0
            self.start_time = time.time()
            self.last_prediction_time = None

            # Reset current metrics
            self.current_metrics = {
                'accuracy': 0.0,
                'avg_latency_ms': 0.0,
                'throughput_per_second': 0.0,
                'avg_confidence': 0.0,
                'error_rate': 0.0,
                'total_predictions': 0,
                'uptime_hours': 0.0
            }

            logger.info("Monitoring data reset")

        except Exception as e:
            logger.error(f"Failed to reset monitoring: {e}")

def create_model_monitor(config_path: str = "config/config.yaml") -> ModelMonitor:
    """
    Convenience function to create a model monitor.

    Args:
        config_path: Path to configuration file

    Returns:
        ModelMonitor instance
    """
    return ModelMonitor(config_path)
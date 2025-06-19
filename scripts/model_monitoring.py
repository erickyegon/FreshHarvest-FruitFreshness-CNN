#!/usr/bin/env python3
"""
FreshHarvest Model Monitoring Script
===================================

This script provides comprehensive monitoring for the FreshHarvest
model in production, including performance tracking, drift detection,
and alerting capabilities.

Author: FreshHarvest Team
Version: 1.0.0
Last Updated: 2025-06-18
"""

import sys
import json
import time
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from collections import deque, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    from cvProject_FreshHarvest.utils.common import read_yaml, create_directories
    from cvProject_FreshHarvest.components.model_deployment import ModelDeployment
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the project is properly set up")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """
    Comprehensive model monitoring for FreshHarvest production system.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize model monitor.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = read_yaml(config_path)
        self.monitoring_config = self.config.get('monitoring', {})

        # Monitoring settings
        self.check_interval = self.monitoring_config.get('check_interval_seconds', 60)
        self.alert_threshold = self.monitoring_config.get('alert_threshold', 0.05)
        self.max_history_size = self.monitoring_config.get('max_history_size', 1000)

        # Data storage
        self.performance_history = deque(maxlen=self.max_history_size)
        self.prediction_history = deque(maxlen=self.max_history_size)
        self.error_history = deque(maxlen=self.max_history_size)

        # Monitoring state
        self.is_monitoring = False
        self.baseline_metrics = self._load_baseline_metrics()
        self.alerts = []

        # Create monitoring directories
        self.monitoring_dir = Path("monitoring")
        self.logs_dir = Path("logs")
        create_directories([str(self.monitoring_dir), str(self.logs_dir)])

        logger.info("Model monitor initialized")

    def _load_baseline_metrics(self) -> Dict[str, float]:
        """Load baseline performance metrics."""

        try:
            baseline_path = Path("artifacts/model_trainer/training_metadata.json")
            if baseline_path.exists():
                with open(baseline_path, 'r') as f:
                    metadata = json.load(f)

                return {
                    'accuracy': metadata.get('best_val_accuracy', 0.965),
                    'training_date': metadata.get('training_date', ''),
                    'epochs': metadata.get('total_epochs', 23)
                }
            else:
                logger.warning("Baseline metrics not found, using defaults")
                return {
                    'accuracy': 0.965,
                    'training_date': datetime.now().isoformat(),
                    'epochs': 23
                }

        except Exception as e:
            logger.error(f"Error loading baseline metrics: {e}")
            return {'accuracy': 0.965}

    def check_model_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive model health check.

        Returns:
            Health status dictionary
        """
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'checks': {}
            }

            # Check 1: Model file existence
            model_path = Path("artifacts/model_trainer/model.h5")
            health_status['checks']['model_file'] = {
                'status': 'pass' if model_path.exists() else 'fail',
                'message': f"Model file {'found' if model_path.exists() else 'missing'}"
            }

            # Check 2: Model loading
            try:
                deployment = ModelDeployment(self.config_path)
                model_loaded = deployment.load_model()
                health_status['checks']['model_loading'] = {
                    'status': 'pass' if model_loaded else 'fail',
                    'message': f"Model {'loaded successfully' if model_loaded else 'failed to load'}"
                }

                # Check 3: Prediction test
                if model_loaded:
                    test_image = np.random.rand(224, 224, 3).astype(np.float32)
                    start_time = time.time()
                    result = deployment.predict(test_image)
                    inference_time = time.time() - start_time

                    prediction_success = 'error' not in result
                    health_status['checks']['prediction_test'] = {
                        'status': 'pass' if prediction_success else 'fail',
                        'message': f"Prediction {'successful' if prediction_success else 'failed'}",
                        'inference_time_ms': inference_time * 1000
                    }

                    # Check 4: Inference time
                    time_threshold = 1000  # 1 second
                    time_ok = inference_time * 1000 < time_threshold
                    health_status['checks']['inference_time'] = {
                        'status': 'pass' if time_ok else 'warn',
                        'message': f"Inference time: {inference_time*1000:.2f}ms",
                        'threshold': time_threshold
                    }

            except Exception as e:
                health_status['checks']['model_loading'] = {
                    'status': 'fail',
                    'message': f"Model loading error: {str(e)}"
                }

            # Check 5: System resources
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)
                disk_percent = psutil.disk_usage('.').percent

                health_status['checks']['system_resources'] = {
                    'status': 'pass' if memory_percent < 90 and cpu_percent < 90 and disk_percent < 90 else 'warn',
                    'memory_percent': memory_percent,
                    'cpu_percent': cpu_percent,
                    'disk_percent': disk_percent
                }

            except ImportError:
                health_status['checks']['system_resources'] = {
                    'status': 'skip',
                    'message': 'psutil not available'
                }

            # Determine overall status
            failed_checks = [check for check in health_status['checks'].values()
                           if check['status'] == 'fail']
            warning_checks = [check for check in health_status['checks'].values()
                            if check['status'] == 'warn']

            if failed_checks:
                health_status['overall_status'] = 'unhealthy'
            elif warning_checks:
                health_status['overall_status'] = 'degraded'
            else:
                health_status['overall_status'] = 'healthy'

            return health_status

        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }

    def monitor_performance(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Monitor model performance metrics.

        Args:
            predictions: List of prediction results

        Returns:
            Performance analysis
        """
        try:
            if not predictions:
                return {'status': 'no_data'}

            # Calculate current metrics
            confidences = [pred.get('confidence', 0) for pred in predictions]
            inference_times = [pred.get('inference_time_ms', 0) for pred in predictions]

            current_metrics = {
                'timestamp': datetime.now().isoformat(),
                'num_predictions': len(predictions),
                'avg_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'std_confidence': np.std(confidences),
                'avg_inference_time': np.mean(inference_times),
                'max_inference_time': np.max(inference_times)
            }

            # Store in history
            self.performance_history.append(current_metrics)

            # Detect anomalies
            anomalies = self._detect_performance_anomalies(current_metrics)

            # Generate alerts if needed
            if anomalies:
                self._generate_alerts(anomalies)

            return {
                'current_metrics': current_metrics,
                'anomalies': anomalies,
                'status': 'monitored'
            }

        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")
            return {'status': 'error', 'error': str(e)}

    def _detect_performance_anomalies(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""

        anomalies = []

        try:
            # Check confidence drift
            if len(self.performance_history) > 10:
                recent_confidences = [m['avg_confidence'] for m in list(self.performance_history)[-10:]]
                baseline_confidence = self.baseline_metrics.get('accuracy', 0.965)

                current_avg = np.mean(recent_confidences)
                confidence_drop = baseline_confidence - current_avg

                if confidence_drop > self.alert_threshold:
                    anomalies.append({
                        'type': 'confidence_drift',
                        'severity': 'high' if confidence_drop > 0.1 else 'medium',
                        'message': f"Confidence dropped by {confidence_drop:.3f}",
                        'current_value': current_avg,
                        'baseline_value': baseline_confidence
                    })

            # Check inference time
            current_time = current_metrics.get('avg_inference_time', 0)
            if current_time > 1000:  # 1 second threshold
                anomalies.append({
                    'type': 'slow_inference',
                    'severity': 'medium' if current_time < 2000 else 'high',
                    'message': f"Slow inference time: {current_time:.2f}ms",
                    'current_value': current_time,
                    'threshold': 1000
                })

            # Check low confidence predictions
            low_confidence_ratio = len([c for c in [current_metrics.get('min_confidence', 1)] if c < 0.5])
            if low_confidence_ratio > 0:
                anomalies.append({
                    'type': 'low_confidence',
                    'severity': 'low',
                    'message': f"Low confidence predictions detected",
                    'current_value': current_metrics.get('min_confidence', 1)
                })

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")

        return anomalies

    def _generate_alerts(self, anomalies: List[Dict[str, Any]]) -> None:
        """Generate alerts for detected anomalies."""

        for anomaly in anomalies:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': anomaly['type'],
                'severity': anomaly['severity'],
                'message': anomaly['message'],
                'details': anomaly
            }

            self.alerts.append(alert)

            # Log alert
            severity_level = {
                'low': logging.INFO,
                'medium': logging.WARNING,
                'high': logging.ERROR
            }.get(anomaly['severity'], logging.WARNING)

            logger.log(severity_level, f"ALERT: {anomaly['message']}")

            # Save alert to file
            self._save_alert(alert)

    def _save_alert(self, alert: Dict[str, Any]) -> None:
        """Save alert to file."""

        try:
            alerts_file = self.monitoring_dir / "alerts.jsonl"
            with open(alerts_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')

        except Exception as e:
            logger.error(f"Error saving alert: {e}")

    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report."""

        try:
            # Collect current status
            health_status = self.check_model_health()

            # Calculate summary statistics
            if self.performance_history:
                recent_metrics = list(self.performance_history)[-24:]  # Last 24 checks
                avg_confidence = np.mean([m['avg_confidence'] for m in recent_metrics])
                avg_inference_time = np.mean([m['avg_inference_time'] for m in recent_metrics])
            else:
                avg_confidence = 0
                avg_inference_time = 0

            # Count recent alerts
            recent_alerts = [a for a in self.alerts
                           if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=24)]

            # Generate report
            report_content = f"""# FreshHarvest Model Monitoring Report

## 📊 System Status
**Overall Health:** {health_status['overall_status'].upper()}
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Monitoring Duration:** {len(self.performance_history)} checks

## 🎯 Performance Summary
- **Average Confidence:** {avg_confidence:.3f}
- **Average Inference Time:** {avg_inference_time:.2f}ms
- **Baseline Accuracy:** {self.baseline_metrics.get('accuracy', 0.965):.3f}
- **Recent Alerts:** {len(recent_alerts)}

## 🔍 Health Checks
"""

            for check_name, check_result in health_status.get('checks', {}).items():
                status_emoji = {'pass': '✅', 'fail': '❌', 'warn': '⚠️', 'skip': '⏭️'}.get(check_result['status'], '❓')
                report_content += f"- **{check_name.replace('_', ' ').title()}:** {status_emoji} {check_result['message']}\n"

            report_content += f"""
## 🚨 Recent Alerts ({len(recent_alerts)})
"""

            if recent_alerts:
                for alert in recent_alerts[-5:]:  # Show last 5 alerts
                    severity_emoji = {'low': '🔵', 'medium': '🟡', 'high': '🔴'}.get(alert['severity'], '⚪')
                    report_content += f"- {severity_emoji} **{alert['type']}:** {alert['message']} ({alert['timestamp']})\n"
            else:
                report_content += "- ✅ No recent alerts\n"

            report_content += f"""
## 📈 Recommendations
"""

            if health_status['overall_status'] == 'healthy':
                report_content += "- ✅ System operating normally\n- 📊 Continue regular monitoring\n"
            elif health_status['overall_status'] == 'degraded':
                report_content += "- ⚠️ Monitor system closely\n- 🔧 Consider performance optimization\n"
            else:
                report_content += "- 🚨 Immediate attention required\n- 🔧 Check system logs and configuration\n"

            report_content += f"""
## 🔧 System Information
- **Model Version:** 1.0.0
- **Baseline Accuracy:** 96.50%
- **Monitoring Interval:** {self.check_interval}s
- **Alert Threshold:** {self.alert_threshold}

---
*Generated by FreshHarvest Model Monitor*
"""

            # Save report
            report_path = self.monitoring_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_path, 'w') as f:
                f.write(report_content)

            logger.info(f"Monitoring report generated: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Error generating monitoring report: {e}")
            return ""

    def start_continuous_monitoring(self) -> None:
        """Start continuous monitoring in background."""

        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return

        self.is_monitoring = True
        logger.info(f"Starting continuous monitoring (interval: {self.check_interval}s)")

        def monitoring_loop():
            while self.is_monitoring:
                try:
                    # Perform health check
                    health_status = self.check_model_health()

                    # Log status
                    if health_status['overall_status'] != 'healthy':
                        logger.warning(f"Health check: {health_status['overall_status']}")

                    # Save status
                    status_file = self.monitoring_dir / "current_status.json"
                    with open(status_file, 'w') as f:
                        json.dump(health_status, f, indent=2)

                    # Wait for next check
                    time.sleep(self.check_interval)

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(self.check_interval)

        # Start monitoring thread
        import threading
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()

        logger.info("Continuous monitoring started")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""

        self.is_monitoring = False
        logger.info("Monitoring stopped")

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""

        try:
            # Current health status
            health_status = self.check_model_health()

            # Performance metrics
            if self.performance_history:
                recent_metrics = list(self.performance_history)[-24:]
                performance_data = {
                    'timestamps': [m['timestamp'] for m in recent_metrics],
                    'confidences': [m['avg_confidence'] for m in recent_metrics],
                    'inference_times': [m['avg_inference_time'] for m in recent_metrics]
                }
            else:
                performance_data = {
                    'timestamps': [],
                    'confidences': [],
                    'inference_times': []
                }

            # Recent alerts
            recent_alerts = [a for a in self.alerts
                           if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=24)]

            return {
                'health_status': health_status,
                'performance_data': performance_data,
                'recent_alerts': recent_alerts,
                'baseline_metrics': self.baseline_metrics,
                'monitoring_config': {
                    'check_interval': self.check_interval,
                    'alert_threshold': self.alert_threshold,
                    'is_monitoring': self.is_monitoring
                }
            }

        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}


def main():
    """Main function for model monitoring."""

    import argparse

    parser = argparse.ArgumentParser(description="FreshHarvest Model Monitoring")
    parser.add_argument('--action', choices=['check', 'monitor', 'report', 'dashboard'],
                       default='check', help='Monitoring action to perform')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--interval', type=int, default=60,
                       help='Monitoring interval in seconds')
    parser.add_argument('--duration', type=int, default=0,
                       help='Monitoring duration in minutes (0 = continuous)')

    args = parser.parse_args()

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    print("🍎 FreshHarvest Model Monitoring")
    print("=" * 40)
    print()

    try:
        # Initialize monitor
        monitor = ModelMonitor(args.config)
        monitor.check_interval = args.interval

        if args.action == 'check':
            # Single health check
            print("Performing health check...")
            health_status = monitor.check_model_health()

            print(f"Overall Status: {health_status['overall_status'].upper()}")
            print("\nHealth Checks:")
            for check_name, check_result in health_status.get('checks', {}).items():
                status_symbol = {'pass': '✅', 'fail': '❌', 'warn': '⚠️', 'skip': '⏭️'}.get(check_result['status'], '❓')
                print(f"  {status_symbol} {check_name.replace('_', ' ').title()}: {check_result['message']}")

        elif args.action == 'monitor':
            # Continuous monitoring
            print(f"Starting continuous monitoring (interval: {args.interval}s)")
            if args.duration > 0:
                print(f"Duration: {args.duration} minutes")
            else:
                print("Duration: Continuous (Ctrl+C to stop)")

            monitor.start_continuous_monitoring()

            try:
                if args.duration > 0:
                    time.sleep(args.duration * 60)
                    monitor.stop_monitoring()
                    print("Monitoring completed")
                else:
                    # Run until interrupted
                    while True:
                        time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                monitor.stop_monitoring()

        elif args.action == 'report':
            # Generate monitoring report
            print("Generating monitoring report...")
            report_path = monitor.generate_monitoring_report()
            if report_path:
                print(f"✅ Report generated: {report_path}")
            else:
                print("❌ Failed to generate report")

        elif args.action == 'dashboard':
            # Get dashboard data
            print("Collecting dashboard data...")
            dashboard_data = monitor.get_monitoring_dashboard_data()

            print(f"Health Status: {dashboard_data.get('health_status', {}).get('overall_status', 'unknown').upper()}")
            print(f"Recent Alerts: {len(dashboard_data.get('recent_alerts', []))}")
            print(f"Performance History: {len(dashboard_data.get('performance_data', {}).get('timestamps', []))} records")

            # Save dashboard data
            dashboard_file = Path("monitoring/dashboard_data.json")
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            print(f"Dashboard data saved: {dashboard_file}")

        print("\n🎉 Monitoring operation completed successfully!")

    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
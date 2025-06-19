"""
Performance Dashboard for FreshHarvest Model Monitoring
======================================================

This module provides a real-time performance dashboard for monitoring
the FreshHarvest fruit freshness classification system.

Author: FreshHarvest Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from cvProject_FreshHarvest.utils.common import read_yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """
    Real-time performance dashboard for FreshHarvest model monitoring.

    Provides comprehensive visualization of model performance metrics,
    system health, and operational statistics.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the performance dashboard."""
        self.config = read_yaml(config_path)
        self.monitoring_config = self.config.get('monitoring', {})

        # Dashboard configuration
        self.refresh_interval = self.monitoring_config.get('dashboard_refresh_seconds', 30)
        self.metrics_history_days = self.monitoring_config.get('metrics_history_days', 7)

        # Data paths
        self.metrics_file = Path("monitoring/metrics.jsonl")
        self.alerts_file = Path("monitoring/alerts.jsonl")
        self.performance_file = Path("monitoring/performance.jsonl")

        logger.info("Performance dashboard initialized")

    def load_metrics_data(self) -> pd.DataFrame:
        """Load metrics data from monitoring files."""
        try:
            if not self.metrics_file.exists():
                return pd.DataFrame()

            # Read JSONL file
            metrics_data = []
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    if line.strip():
                        metrics_data.append(json.loads(line))

            if not metrics_data:
                return pd.DataFrame()

            df = pd.DataFrame(metrics_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Filter to recent data
            cutoff_date = datetime.now() - timedelta(days=self.metrics_history_days)
            df = df[df['timestamp'] >= cutoff_date]

            return df.sort_values('timestamp')

        except Exception as e:
            logger.error(f"Error loading metrics data: {e}")
            return pd.DataFrame()

    def load_alerts_data(self) -> pd.DataFrame:
        """Load alerts data from monitoring files."""
        try:
            if not self.alerts_file.exists():
                return pd.DataFrame()

            alerts_data = []
            with open(self.alerts_file, 'r') as f:
                for line in f:
                    if line.strip():
                        alerts_data.append(json.loads(line))

            if not alerts_data:
                return pd.DataFrame()

            df = pd.DataFrame(alerts_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Filter to recent data
            cutoff_date = datetime.now() - timedelta(days=self.metrics_history_days)
            df = df[df['timestamp'] >= cutoff_date]

            return df.sort_values('timestamp', ascending=False)

        except Exception as e:
            logger.error(f"Error loading alerts data: {e}")
            return pd.DataFrame()

    def create_accuracy_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create accuracy trend chart."""
        if df.empty or 'accuracy' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No accuracy data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        fig = go.Figure()

        # Add accuracy line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=6)
        ))

        # Add target accuracy line (96.50%)
        fig.add_hline(y=0.965, line_dash="dash", line_color="red",
                     annotation_text="Target: 96.50%")

        fig.update_layout(
            title="Model Accuracy Trend",
            xaxis_title="Time",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0.9, 1.0]),
            height=400
        )

        return fig

    def create_inference_time_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create inference time chart."""
        if df.empty or 'inference_time_ms' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No inference time data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        fig = go.Figure()

        # Add inference time line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['inference_time_ms'],
            mode='lines+markers',
            name='Inference Time',
            line=dict(color='#4169E1', width=3),
            marker=dict(size=6)
        ))

        # Add target inference time line (50ms)
        fig.add_hline(y=50, line_dash="dash", line_color="orange",
                     annotation_text="Target: 50ms")

        fig.update_layout(
            title="Inference Time Trend",
            xaxis_title="Time",
            yaxis_title="Inference Time (ms)",
            height=400
        )

        return fig

    def create_prediction_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create prediction distribution chart."""
        if df.empty or 'predicted_class' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No prediction data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Count predictions by class
        class_counts = df['predicted_class'].value_counts()

        fig = go.Figure(data=[
            go.Bar(
                x=class_counts.index,
                y=class_counts.values,
                marker_color=['#2E8B57' if 'Fresh' in cls else '#DC143C' for cls in class_counts.index]
            )
        ])

        fig.update_layout(
            title="Prediction Distribution",
            xaxis_title="Fruit Class",
            yaxis_title="Number of Predictions",
            height=400
        )

        return fig

    def create_confidence_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create confidence distribution chart."""
        if df.empty or 'confidence' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No confidence data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        fig = go.Figure(data=[
            go.Histogram(
                x=df['confidence'],
                nbinsx=20,
                marker_color='#4169E1',
                opacity=0.7
            )
        ])

        fig.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            height=400
        )

        return fig

    def create_system_metrics_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create system metrics chart."""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No system metrics available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Create subplots for multiple metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage', 'Network I/O'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # CPU Usage
        if 'cpu_percent' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['cpu_percent'], name='CPU %'),
                row=1, col=1
            )

        # Memory Usage
        if 'memory_percent' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['memory_percent'], name='Memory %'),
                row=1, col=2
            )

        # Disk Usage
        if 'disk_percent' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['disk_percent'], name='Disk %'),
                row=2, col=1
            )

        # Network I/O (if available)
        if 'network_io_mb' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['network_io_mb'], name='Network MB'),
                row=2, col=2
            )

        fig.update_layout(height=600, showlegend=False)

        return fig

    def get_current_stats(self, df: pd.DataFrame) -> dict:
        """Get current performance statistics."""
        if df.empty:
            return {
                'total_predictions': 0,
                'avg_accuracy': 0.0,
                'avg_confidence': 0.0,
                'avg_inference_time': 0.0,
                'fresh_percentage': 0.0
            }

        # Get recent data (last hour)
        recent_df = df[df['timestamp'] >= datetime.now() - timedelta(hours=1)]

        if recent_df.empty:
            recent_df = df.tail(100)  # Last 100 predictions

        stats = {
            'total_predictions': len(df),
            'recent_predictions': len(recent_df),
            'avg_accuracy': recent_df['accuracy'].mean() if 'accuracy' in recent_df.columns else 0.0,
            'avg_confidence': recent_df['confidence'].mean() if 'confidence' in recent_df.columns else 0.0,
            'avg_inference_time': recent_df['inference_time_ms'].mean() if 'inference_time_ms' in recent_df.columns else 0.0,
        }

        # Calculate fresh percentage
        if 'predicted_class' in recent_df.columns:
            fresh_predictions = recent_df[recent_df['predicted_class'].str.contains('Fresh', na=False)]
            stats['fresh_percentage'] = (len(fresh_predictions) / len(recent_df)) * 100 if len(recent_df) > 0 else 0.0
        else:
            stats['fresh_percentage'] = 0.0

        return stats

def run_dashboard():
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="FreshHarvest Performance Dashboard",
        page_icon="🍎",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize dashboard
    dashboard = PerformanceDashboard()

    # Dashboard header
    st.title("🍎 FreshHarvest Performance Dashboard")
    st.markdown("**Real-time monitoring of AI-powered fruit freshness classification (96.50% accuracy)**")

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 30)

    if st.sidebar.button("Refresh Now"):
        st.experimental_rerun()

    # Load data
    with st.spinner("Loading monitoring data..."):
        metrics_df = dashboard.load_metrics_data()
        alerts_df = dashboard.load_alerts_data()

    # Current statistics
    stats = dashboard.get_current_stats(metrics_df)

    # Display key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Predictions", f"{stats['total_predictions']:,}")

    with col2:
        st.metric("Avg Accuracy", f"{stats['avg_accuracy']:.3f}",
                 delta=f"{stats['avg_accuracy'] - 0.965:.3f}")

    with col3:
        st.metric("Avg Confidence", f"{stats['avg_confidence']:.3f}")

    with col4:
        st.metric("Avg Inference Time", f"{stats['avg_inference_time']:.1f}ms")

    with col5:
        st.metric("Fresh Percentage", f"{stats['fresh_percentage']:.1f}%")

    # Charts section
    st.header("Performance Metrics")

    # Row 1: Accuracy and Inference Time
    col1, col2 = st.columns(2)

    with col1:
        accuracy_chart = dashboard.create_accuracy_chart(metrics_df)
        st.plotly_chart(accuracy_chart, use_container_width=True)

    with col2:
        inference_chart = dashboard.create_inference_time_chart(metrics_df)
        st.plotly_chart(inference_chart, use_container_width=True)

    # Row 2: Prediction Distribution and Confidence
    col1, col2 = st.columns(2)

    with col1:
        distribution_chart = dashboard.create_prediction_distribution_chart(metrics_df)
        st.plotly_chart(distribution_chart, use_container_width=True)

    with col2:
        confidence_chart = dashboard.create_confidence_distribution_chart(metrics_df)
        st.plotly_chart(confidence_chart, use_container_width=True)

    # System metrics
    st.header("System Metrics")
    system_chart = dashboard.create_system_metrics_chart(metrics_df)
    st.plotly_chart(system_chart, use_container_width=True)

    # Recent alerts
    st.header("Recent Alerts")
    if not alerts_df.empty:
        # Display recent alerts
        recent_alerts = alerts_df.head(10)
        for _, alert in recent_alerts.iterrows():
            alert_type = alert.get('type', 'info')
            alert_color = {
                'critical': '🔴',
                'warning': '🟡',
                'info': '🔵'
            }.get(alert_type, '🔵')

            st.write(f"{alert_color} **{alert.get('message', 'No message')}** - {alert['timestamp']}")
    else:
        st.info("No recent alerts")

    # Data table
    if st.checkbox("Show Raw Data"):
        st.header("Raw Metrics Data")
        if not metrics_df.empty:
            st.dataframe(metrics_df.tail(100))
        else:
            st.info("No metrics data available")

    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.experimental_rerun()

if __name__ == "__main__":
    run_dashboard()
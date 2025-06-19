"""
FreshHarvest Evaluation Module
=============================

This module provides comprehensive evaluation capabilities for the FreshHarvest
fruit freshness classification system.

Author: FreshHarvest Team
Version: 1.0.0
"""

from .metrics_calculator import MetricsCalculator, calculate_all_metrics
from .model_validator import ModelValidator, validate_model_performance
from .performance_analysis import PerformanceAnalyzer, analyze_model_performance
from .cross_validation import CrossValidator, perform_cross_validation
from .error_analysis import ErrorAnalyzer, analyze_prediction_errors
from .statistical_tests import StatisticalTests, run_statistical_analysis

__all__ = [
    'MetricsCalculator',
    'calculate_all_metrics',
    'ModelValidator',
    'validate_model_performance',
    'PerformanceAnalyzer',
    'analyze_model_performance',
    'CrossValidator',
    'perform_cross_validation',
    'ErrorAnalyzer',
    'analyze_prediction_errors',
    'StatisticalTests',
    'run_statistical_analysis'
]
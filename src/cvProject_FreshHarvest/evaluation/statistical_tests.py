"""
Statistical Tests for FreshHarvest Model Evaluation
==================================================

This module provides statistical testing capabilities for the FreshHarvest
fruit freshness classification system.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import permutation_test_score
import warnings

logger = logging.getLogger(__name__)

class StatisticalTests:
    """
    Statistical testing suite for FreshHarvest model evaluation.

    Provides comprehensive statistical analysis including significance tests,
    confidence intervals, and model comparison statistics.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical tests.

        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

        logger.info(f"Statistical tests initialized with {confidence_level*100}% confidence level")

    def mcnemar_test(self, y_true: np.ndarray, y_pred1: np.ndarray,
                    y_pred2: np.ndarray) -> Dict[str, Any]:
        """
        Perform McNemar's test to compare two models.

        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2

        Returns:
            Dictionary containing test results
        """
        try:
            # Create contingency table
            correct1 = (y_pred1 == y_true)
            correct2 = (y_pred2 == y_true)

            # McNemar's table
            both_correct = np.sum(correct1 & correct2)
            model1_only = np.sum(correct1 & ~correct2)
            model2_only = np.sum(~correct1 & correct2)
            both_wrong = np.sum(~correct1 & ~correct2)

            # McNemar's statistic
            if model1_only + model2_only == 0:
                p_value = 1.0
                statistic = 0.0
            else:
                statistic = (abs(model1_only - model2_only) - 1)**2 / (model1_only + model2_only)
                p_value = 1 - stats.chi2.cdf(statistic, df=1)

            # Effect size (odds ratio)
            if model2_only == 0:
                odds_ratio = float('inf') if model1_only > 0 else 1.0
            else:
                odds_ratio = model1_only / model2_only

            results = {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'contingency_table': {
                    'both_correct': both_correct,
                    'model1_only_correct': model1_only,
                    'model2_only_correct': model2_only,
                    'both_wrong': both_wrong
                },
                'odds_ratio': odds_ratio,
                'interpretation': self._interpret_mcnemar(p_value, odds_ratio)
            }

            return results

        except Exception as e:
            logger.error(f"Error in McNemar's test: {e}")
            return {'error': str(e)}

    def _interpret_mcnemar(self, p_value: float, odds_ratio: float) -> str:
        """Interpret McNemar's test results."""

        if p_value >= self.alpha:
            return "No significant difference between models"
        else:
            if odds_ratio > 1:
                return "Model 1 significantly outperforms Model 2"
            elif odds_ratio < 1:
                return "Model 2 significantly outperforms Model 1"
            else:
                return "Significant difference detected"

    def bootstrap_confidence_interval(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    metric: str = 'accuracy', n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence interval for a metric.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric: Metric to calculate ('accuracy', 'precision', 'recall', 'f1')
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary containing confidence interval results
        """
        try:
            # Select metric function
            metric_functions = {
                'accuracy': accuracy_score,
                'precision': lambda y_t, y_p: precision_score(y_t, y_p, average='weighted', zero_division=0),
                'recall': lambda y_t, y_p: recall_score(y_t, y_p, average='weighted', zero_division=0),
                'f1': lambda y_t, y_p: f1_score(y_t, y_p, average='weighted', zero_division=0)
            }

            if metric not in metric_functions:
                raise ValueError(f"Unsupported metric: {metric}")

            metric_func = metric_functions[metric]

            # Original metric value
            original_score = metric_func(y_true, y_pred)

            # Bootstrap sampling
            n_samples = len(y_true)
            bootstrap_scores = []

            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]

                # Calculate metric for bootstrap sample
                score = metric_func(y_true_boot, y_pred_boot)
                bootstrap_scores.append(score)

            bootstrap_scores = np.array(bootstrap_scores)

            # Calculate confidence interval
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_scores, lower_percentile)
            ci_upper = np.percentile(bootstrap_scores, upper_percentile)

            results = {
                'metric': metric,
                'original_score': original_score,
                'bootstrap_mean': np.mean(bootstrap_scores),
                'bootstrap_std': np.std(bootstrap_scores),
                'confidence_interval': {
                    'lower': ci_lower,
                    'upper': ci_upper,
                    'level': self.confidence_level
                },
                'n_bootstrap': n_bootstrap,
                'bootstrap_scores': bootstrap_scores.tolist()
            }

            return results

        except Exception as e:
            logger.error(f"Error in bootstrap confidence interval: {e}")
            return {'error': str(e)}

    def permutation_test(self, y_true: np.ndarray, y_pred: np.ndarray,
                        n_permutations: int = 1000) -> Dict[str, Any]:
        """
        Perform permutation test for model significance.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            n_permutations: Number of permutations

        Returns:
            Dictionary containing permutation test results
        """
        try:
            # Original accuracy
            original_accuracy = accuracy_score(y_true, y_pred)

            # Permutation test
            permutation_scores = []

            for _ in range(n_permutations):
                # Randomly permute labels
                y_true_perm = np.random.permutation(y_true)
                perm_accuracy = accuracy_score(y_true_perm, y_pred)
                permutation_scores.append(perm_accuracy)

            permutation_scores = np.array(permutation_scores)

            # Calculate p-value
            p_value = np.mean(permutation_scores >= original_accuracy)

            results = {
                'original_accuracy': original_accuracy,
                'permutation_mean': np.mean(permutation_scores),
                'permutation_std': np.std(permutation_scores),
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'n_permutations': n_permutations,
                'interpretation': self._interpret_permutation(p_value, original_accuracy)
            }

            return results

        except Exception as e:
            logger.error(f"Error in permutation test: {e}")
            return {'error': str(e)}

    def _interpret_permutation(self, p_value: float, accuracy: float) -> str:
        """Interpret permutation test results."""

        if p_value < self.alpha:
            return f"Model performance ({accuracy:.3f}) is significantly better than random"
        else:
            return f"Model performance ({accuracy:.3f}) is not significantly better than random"

    def chi_square_test(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Perform chi-square test for independence.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary containing chi-square test results
        """
        try:
            # Create contingency table
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            contingency_table = np.zeros((len(unique_labels), len(unique_labels)))

            for i, true_label in enumerate(unique_labels):
                for j, pred_label in enumerate(unique_labels):
                    contingency_table[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

            # Chi-square test
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            # Cramér's V (effect size)
            n = np.sum(contingency_table)
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

            results = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < self.alpha,
                'cramers_v': cramers_v,
                'contingency_table': contingency_table.tolist(),
                'expected_frequencies': expected.tolist(),
                'interpretation': self._interpret_chi_square(p_value, cramers_v)
            }

            return results

        except Exception as e:
            logger.error(f"Error in chi-square test: {e}")
            return {'error': str(e)}

    def _interpret_chi_square(self, p_value: float, cramers_v: float) -> str:
        """Interpret chi-square test results."""

        if p_value >= self.alpha:
            return "No significant association between true and predicted labels"
        else:
            if cramers_v < 0.1:
                effect = "weak"
            elif cramers_v < 0.3:
                effect = "moderate"
            else:
                effect = "strong"

            return f"Significant association detected with {effect} effect size (Cramér's V = {cramers_v:.3f})"

    def normality_test(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Test for normality using multiple tests.

        Args:
            data: Data to test for normality

        Returns:
            Dictionary containing normality test results
        """
        try:
            results = {}

            # Shapiro-Wilk test (for small samples)
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                results['shapiro_wilk'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'normal': shapiro_p > self.alpha
                }

            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            results['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'normal': ks_p > self.alpha
            }

            # Anderson-Darling test
            ad_stat, ad_critical, ad_significance = stats.anderson(data, dist='norm')
            results['anderson_darling'] = {
                'statistic': ad_stat,
                'critical_values': ad_critical.tolist(),
                'significance_levels': ad_significance.tolist(),
                'normal': ad_stat < ad_critical[2]  # 5% significance level
            }

            # Overall assessment
            normal_tests = [test['normal'] for test in results.values()]
            results['overall_normal'] = np.mean(normal_tests) > 0.5

            return results

        except Exception as e:
            logger.error(f"Error in normality test: {e}")
            return {'error': str(e)}

    def generate_statistical_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_pred_baseline: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate comprehensive statistical report.

        Args:
            y_true: True labels
            y_pred: Model predictions
            y_pred_baseline: Baseline model predictions (optional)

        Returns:
            Dictionary containing comprehensive statistical analysis
        """
        try:
            report = {
                'summary': {
                    'n_samples': len(y_true),
                    'n_classes': len(np.unique(y_true)),
                    'confidence_level': self.confidence_level
                }
            }

            # Bootstrap confidence intervals for main metrics
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            report['confidence_intervals'] = {}

            for metric in metrics:
                ci_result = self.bootstrap_confidence_interval(y_true, y_pred, metric)
                if 'error' not in ci_result:
                    report['confidence_intervals'][metric] = ci_result

            # Permutation test
            perm_result = self.permutation_test(y_true, y_pred)
            if 'error' not in perm_result:
                report['permutation_test'] = perm_result

            # Chi-square test
            chi2_result = self.chi_square_test(y_true, y_pred)
            if 'error' not in chi2_result:
                report['chi_square_test'] = chi2_result

            # Model comparison (if baseline provided)
            if y_pred_baseline is not None:
                mcnemar_result = self.mcnemar_test(y_true, y_pred, y_pred_baseline)
                if 'error' not in mcnemar_result:
                    report['model_comparison'] = mcnemar_result

            # Statistical summary
            report['statistical_summary'] = self._create_statistical_summary(report)

            return report

        except Exception as e:
            logger.error(f"Error generating statistical report: {e}")
            return {'error': str(e)}

    def _create_statistical_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Create statistical summary from report."""

        summary = {
            'model_significance': 'Unknown',
            'confidence_intervals_available': False,
            'model_comparison_available': False
        }

        # Check permutation test
        if 'permutation_test' in report:
            perm_test = report['permutation_test']
            if perm_test.get('significant', False):
                summary['model_significance'] = 'Significantly better than random'
            else:
                summary['model_significance'] = 'Not significantly better than random'

        # Check confidence intervals
        if 'confidence_intervals' in report:
            summary['confidence_intervals_available'] = True

            # Get accuracy CI
            if 'accuracy' in report['confidence_intervals']:
                acc_ci = report['confidence_intervals']['accuracy']['confidence_interval']
                summary['accuracy_ci'] = f"[{acc_ci['lower']:.3f}, {acc_ci['upper']:.3f}]"

        # Check model comparison
        if 'model_comparison' in report:
            summary['model_comparison_available'] = True
            comparison = report['model_comparison']
            summary['model_comparison_result'] = comparison.get('interpretation', 'Unknown')

        return summary

def run_statistical_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                            y_pred_baseline: Optional[np.ndarray] = None,
                            confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive statistical analysis.

    Args:
        y_true: True labels
        y_pred: Model predictions
        y_pred_baseline: Baseline model predictions (optional)
        confidence_level: Confidence level for tests

    Returns:
        Dictionary containing statistical analysis results
    """
    tester = StatisticalTests(confidence_level)
    return tester.generate_statistical_report(y_true, y_pred, y_pred_baseline)
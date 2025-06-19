"""
Cross-Validation Module for FreshHarvest Classification System
============================================================

This module provides comprehensive cross-validation functionality
for robust model evaluation and performance assessment.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_score,
    cross_validate, train_test_split
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import time

logger = logging.getLogger(__name__)

class CrossValidator:
    """
    Comprehensive cross-validation for FreshHarvest models.

    Provides various cross-validation strategies including stratified k-fold,
    time series split, and custom validation schemes.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize the cross-validator.

        Args:
            n_splits: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = {}

    def stratified_kfold_cv(self, X: np.ndarray, y: np.ndarray,
                           model_builder: Callable,
                           compile_params: Dict = None,
                           fit_params: Dict = None) -> Dict[str, Any]:
        """
        Perform stratified k-fold cross-validation.

        Args:
            X: Features
            y: Labels
            model_builder: Function that returns a compiled model
            compile_params: Model compilation parameters
            fit_params: Model fitting parameters

        Returns:
            Dictionary containing cross-validation results
        """
        try:
            # Default parameters
            compile_params = compile_params or {
                'optimizer': 'adam',
                'loss': 'sparse_categorical_crossentropy',
                'metrics': ['accuracy']
            }
            fit_params = fit_params or {
                'epochs': 20,
                'batch_size': 32,
                'verbose': 0
            }

            # Initialize stratified k-fold
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            fold_results = []
            fold_scores = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'train_time': [],
                'eval_time': []
            }

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                logger.info(f"Processing fold {fold + 1}/{self.n_splits}")

                # Split data
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Build and compile model
                model = model_builder()
                model.compile(**compile_params)

                # Train model
                start_time = time.time()
                history = model.fit(
                    X_train_fold, y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    **fit_params
                )
                train_time = time.time() - start_time

                # Evaluate model
                start_time = time.time()
                y_pred_proba = model.predict(X_val_fold, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
                eval_time = time.time() - start_time

                # Calculate metrics
                accuracy = accuracy_score(y_val_fold, y_pred)
                precision = precision_score(y_val_fold, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val_fold, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val_fold, y_pred, average='weighted', zero_division=0)

                # Store fold results
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(X_train_fold),
                    'val_size': len(X_val_fold),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'train_time': train_time,
                    'eval_time': eval_time,
                    'history': history.history
                }

                fold_results.append(fold_result)
                fold_scores['accuracy'].append(accuracy)
                fold_scores['precision'].append(precision)
                fold_scores['recall'].append(recall)
                fold_scores['f1'].append(f1)
                fold_scores['train_time'].append(train_time)
                fold_scores['eval_time'].append(eval_time)

                logger.info(f"Fold {fold + 1} completed: Accuracy = {accuracy:.4f}")

            # Calculate overall statistics
            cv_stats = {}
            for metric, scores in fold_scores.items():
                cv_stats[f'{metric}_mean'] = np.mean(scores)
                cv_stats[f'{metric}_std'] = np.std(scores)
                cv_stats[f'{metric}_min'] = np.min(scores)
                cv_stats[f'{metric}_max'] = np.max(scores)

            results = {
                'cv_type': 'stratified_kfold',
                'n_splits': self.n_splits,
                'fold_results': fold_results,
                'cv_statistics': cv_stats,
                'fold_scores': fold_scores
            }

            self.cv_results['stratified_kfold'] = results
            logger.info(f"Stratified K-Fold CV completed: Mean accuracy = {cv_stats['accuracy_mean']:.4f} ± {cv_stats['accuracy_std']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error in stratified k-fold CV: {e}")
            return {}

    def holdout_validation(self, X: np.ndarray, y: np.ndarray,
                          model_builder: Callable,
                          test_size: float = 0.2,
                          compile_params: Dict = None,
                          fit_params: Dict = None) -> Dict[str, Any]:
        """
        Perform holdout validation.

        Args:
            X: Features
            y: Labels
            model_builder: Function that returns a compiled model
            test_size: Proportion of data for testing
            compile_params: Model compilation parameters
            fit_params: Model fitting parameters

        Returns:
            Dictionary containing validation results
        """
        try:
            # Default parameters
            compile_params = compile_params or {
                'optimizer': 'adam',
                'loss': 'sparse_categorical_crossentropy',
                'metrics': ['accuracy']
            }
            fit_params = fit_params or {
                'epochs': 30,
                'batch_size': 32,
                'verbose': 0
            }

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )

            # Build and compile model
            model = model_builder()
            model.compile(**compile_params)

            # Train model
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                **fit_params
            )
            train_time = time.time() - start_time

            # Evaluate model
            start_time = time.time()
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            eval_time = time.time() - start_time

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            results = {
                'validation_type': 'holdout',
                'test_size': test_size,
                'train_size': len(X_train),
                'test_size_actual': len(X_test),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'train_time': train_time,
                'eval_time': eval_time,
                'history': history.history,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            self.cv_results['holdout'] = results
            logger.info(f"Holdout validation completed: Accuracy = {accuracy:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error in holdout validation: {e}")
            return {}

    def repeated_cv(self, X: np.ndarray, y: np.ndarray,
                   model_builder: Callable,
                   n_repeats: int = 3,
                   compile_params: Dict = None,
                   fit_params: Dict = None) -> Dict[str, Any]:
        """
        Perform repeated cross-validation.

        Args:
            X: Features
            y: Labels
            model_builder: Function that returns a compiled model
            n_repeats: Number of repetitions
            compile_params: Model compilation parameters
            fit_params: Model fitting parameters

        Returns:
            Dictionary containing repeated CV results
        """
        try:
            all_results = []
            all_scores = []

            for repeat in range(n_repeats):
                logger.info(f"Repeat {repeat + 1}/{n_repeats}")

                # Use different random state for each repeat
                temp_validator = CrossValidator(
                    n_splits=self.n_splits,
                    random_state=self.random_state + repeat
                )

                # Perform stratified k-fold CV
                repeat_results = temp_validator.stratified_kfold_cv(
                    X, y, model_builder, compile_params, fit_params
                )

                if repeat_results:
                    all_results.append(repeat_results)
                    all_scores.extend(repeat_results['fold_scores']['accuracy'])

            # Calculate overall statistics
            overall_stats = {
                'mean_accuracy': np.mean(all_scores),
                'std_accuracy': np.std(all_scores),
                'min_accuracy': np.min(all_scores),
                'max_accuracy': np.max(all_scores),
                'total_folds': len(all_scores)
            }

            results = {
                'cv_type': 'repeated_stratified_kfold',
                'n_repeats': n_repeats,
                'n_splits': self.n_splits,
                'repeat_results': all_results,
                'overall_statistics': overall_stats,
                'all_scores': all_scores
            }

            self.cv_results['repeated_cv'] = results
            logger.info(f"Repeated CV completed: Mean accuracy = {overall_stats['mean_accuracy']:.4f} ± {overall_stats['std_accuracy']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error in repeated CV: {e}")
            return {}

    def generate_cv_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive cross-validation report.

        Args:
            save_path: Path to save the report

        Returns:
            Formatted report string
        """
        try:
            report = []
            report.append("="*80)
            report.append("FRESHHARVEST CROSS-VALIDATION REPORT")
            report.append("="*80)
            report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")

            # Stratified K-Fold results
            if 'stratified_kfold' in self.cv_results:
                skf_results = self.cv_results['stratified_kfold']
                stats = skf_results['cv_statistics']

                report.append("STRATIFIED K-FOLD CROSS-VALIDATION:")
                report.append("-" * 50)
                report.append(f"Number of Folds: {skf_results['n_splits']}")
                report.append(f"Mean Accuracy: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
                report.append(f"Mean Precision: {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}")
                report.append(f"Mean Recall: {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}")
                report.append(f"Mean F1-Score: {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}")
                report.append(f"Mean Training Time: {stats['train_time_mean']:.2f}s ± {stats['train_time_std']:.2f}s")
                report.append("")

                # Individual fold results
                report.append("Individual Fold Results:")
                for fold_result in skf_results['fold_results']:
                    report.append(f"  Fold {fold_result['fold']}: "
                                f"Acc={fold_result['accuracy']:.4f}, "
                                f"Prec={fold_result['precision']:.4f}, "
                                f"Rec={fold_result['recall']:.4f}, "
                                f"F1={fold_result['f1_score']:.4f}")
                report.append("")

            # Holdout validation results
            if 'holdout' in self.cv_results:
                holdout = self.cv_results['holdout']
                report.append("HOLDOUT VALIDATION:")
                report.append("-" * 30)
                report.append(f"Test Size: {holdout['test_size']:.1%}")
                report.append(f"Accuracy: {holdout['accuracy']:.4f}")
                report.append(f"Precision: {holdout['precision']:.4f}")
                report.append(f"Recall: {holdout['recall']:.4f}")
                report.append(f"F1-Score: {holdout['f1_score']:.4f}")
                report.append(f"Training Time: {holdout['train_time']:.2f}s")
                report.append("")

            # Repeated CV results
            if 'repeated_cv' in self.cv_results:
                repeated = self.cv_results['repeated_cv']
                stats = repeated['overall_statistics']

                report.append("REPEATED CROSS-VALIDATION:")
                report.append("-" * 40)
                report.append(f"Number of Repeats: {repeated['n_repeats']}")
                report.append(f"Folds per Repeat: {repeated['n_splits']}")
                report.append(f"Total Folds: {stats['total_folds']}")
                report.append(f"Overall Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
                report.append(f"Min Accuracy: {stats['min_accuracy']:.4f}")
                report.append(f"Max Accuracy: {stats['max_accuracy']:.4f}")
                report.append("")

            report.append("="*80)

            report_text = "\n".join(report)

            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Cross-validation report saved to {save_path}")

            return report_text

        except Exception as e:
            logger.error(f"Error generating CV report: {e}")
            return ""

def perform_cross_validation(X: np.ndarray, y: np.ndarray,
                           model_builder: Callable,
                           cv_type: str = 'stratified_kfold',
                           n_splits: int = 5,
                           **kwargs) -> Dict[str, Any]:
    """
    Convenience function for performing cross-validation.

    Args:
        X: Features
        y: Labels
        model_builder: Function that returns a compiled model
        cv_type: Type of cross-validation ('stratified_kfold', 'holdout', 'repeated')
        n_splits: Number of folds
        **kwargs: Additional parameters

    Returns:
        Dictionary containing cross-validation results
    """
    validator = CrossValidator(n_splits=n_splits)

    if cv_type == 'stratified_kfold':
        return validator.stratified_kfold_cv(X, y, model_builder, **kwargs)
    elif cv_type == 'holdout':
        return validator.holdout_validation(X, y, model_builder, **kwargs)
    elif cv_type == 'repeated':
        return validator.repeated_cv(X, y, model_builder, **kwargs)
    else:
        logger.error(f"Unknown CV type: {cv_type}")
        return {}
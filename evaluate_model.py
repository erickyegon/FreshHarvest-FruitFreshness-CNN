"""
Model evaluation script for FreshHarvest fruit freshness classification.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.cvProject_FreshHarvest.utils.common import read_yaml, setup_logging, get_timestamp
from src.cvProject_FreshHarvest.components.model_evaluation import ModelEvaluator

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Cannot proceed with evaluation.")
    TF_AVAILABLE = False


def load_model(model_path: str):
    """
    Load trained model.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        Loaded model
    """
    if not TF_AVAILABLE:
        logging.error("TensorFlow not available")
        return None
    
    try:
        model = keras.models.load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


def create_test_generator(config):
    """
    Create test data generator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Test data generator
    """
    if not TF_AVAILABLE:
        return None
    
    data_config = config['data']
    
    # Only rescaling for test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        data_config['processed_data_path'] + '/test',
        target_size=tuple(data_config['image_size']),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    logging.info(f"Created test generator with {test_generator.samples} samples")
    return test_generator


def evaluate_model_performance(model_path: str, config_path: str, output_dir: str):
    """
    Evaluate model performance comprehensively.
    
    Args:
        model_path: Path to the trained model
        config_path: Path to configuration file
        output_dir: Output directory for results
    """
    # Setup logging
    setup_logging(level="INFO")
    logging.info("Starting model evaluation")
    
    if not TF_AVAILABLE:
        logging.error("TensorFlow not available. Cannot proceed.")
        return
    
    # Load configuration
    config = read_yaml(config_path)
    
    # Load model
    model = load_model(model_path)
    if model is None:
        logging.error("Failed to load model")
        return
    
    # Create test generator
    test_generator = create_test_generator(config)
    if test_generator is None:
        logging.error("Failed to create test generator")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config_path)
    
    # Perform evaluation
    results = evaluator.evaluate_model(model, test_generator)
    
    if not results:
        logging.error("Evaluation failed")
        return
    
    # Save results
    timestamp = get_timestamp()
    output_path = f"{output_dir}/evaluation_{timestamp}"
    evaluator.save_evaluation_results(results, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL EVALUATION COMPLETED")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Test Samples: {test_generator.samples}")
    print(f"Accuracy: {results['overall_metrics']['accuracy']:.4f}")
    print(f"Precision: {results['overall_metrics']['precision']:.4f}")
    print(f"Recall: {results['overall_metrics']['recall']:.4f}")
    print(f"F1-Score: {results['overall_metrics']['f1_score']:.4f}")
    
    if results['overall_metrics']['roc_auc'] is not None:
        print(f"ROC AUC: {results['overall_metrics']['roc_auc']:.4f}")
    
    print(f"\nDetailed results saved to: {output_path}")
    print("="*60)


def compare_models(model_paths: list, config_path: str, output_dir: str):
    """
    Compare multiple models.
    
    Args:
        model_paths: List of model paths to compare
        config_path: Path to configuration file
        output_dir: Output directory for results
    """
    setup_logging(level="INFO")
    logging.info("Starting model comparison")
    
    if not TF_AVAILABLE:
        logging.error("TensorFlow not available. Cannot proceed.")
        return
    
    config = read_yaml(config_path)
    test_generator = create_test_generator(config)
    evaluator = ModelEvaluator(config_path)
    
    comparison_results = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).stem
        logging.info(f"Evaluating model: {model_name}")
        
        model = load_model(model_path)
        if model is None:
            continue
        
        results = evaluator.evaluate_model(model, test_generator)
        if results:
            comparison_results[model_name] = results['overall_metrics']
    
    # Save comparison results
    timestamp = get_timestamp()
    comparison_path = f"{output_dir}/model_comparison_{timestamp}.json"
    
    from src.cvProject_FreshHarvest.utils.common import write_json
    write_json(comparison_results, comparison_path)
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    for model_name, metrics in comparison_results.items():
        print(f"{model_name:<30} {metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f}")
    
    print("="*80)
    print(f"Comparison results saved to: {comparison_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate FreshHarvest classification model')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', default='outputs/evaluation', help='Output directory')
    parser.add_argument('--compare', nargs='+', help='Compare multiple models')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Compare multiple models
            compare_models(args.compare, args.config, args.output_dir)
        else:
            # Evaluate single model
            evaluate_model_performance(args.model_path, args.config, args.output_dir)
    
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()

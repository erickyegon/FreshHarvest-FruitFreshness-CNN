#!/usr/bin/env python3
"""
FreshHarvest Main Application
============================

Main entry point for the FreshHarvest fruit freshness classification system.
This script provides a comprehensive CLI interface for training, evaluation, and inference.

Author: FreshHarvest Team
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.cvProject_FreshHarvest.utils.common import read_yaml, setup_logging
from src.cvProject_FreshHarvest.data.data_loader import DataLoader
from src.cvProject_FreshHarvest.models.cnn_models import FreshHarvestCNN
from src.cvProject_FreshHarvest.training.trainer import ModelTrainer
from src.cvProject_FreshHarvest.evaluation.evaluator import ModelEvaluator
from src.cvProject_FreshHarvest.inference.predictor import FreshHarvestPredictor

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'logs',
        'models/trained',
        'models/checkpoints',
        'models/exports',
        'outputs/predictions',
        'outputs/reports',
        'outputs/visualizations',
        'data/processed/train',
        'data/processed/val',
        'data/processed/test'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info("✅ Directory structure verified")

def train_model(config_path: str, model_type: str = 'lightweight'):
    """Train a FreshHarvest model."""
    logger.info(f"🚀 Starting model training with {model_type} architecture")

    try:
        # Load configuration
        config = read_yaml(config_path)
        logger.info(f"📋 Configuration loaded from {config_path}")

        # Initialize data loader
        data_loader = DataLoader(config_path)
        train_gen, val_gen, test_gen = data_loader.create_generators()
        logger.info("📊 Data generators created successfully")

        # Initialize model
        cnn_builder = FreshHarvestCNN(config_path)

        if model_type == 'basic':
            model = cnn_builder.create_basic_cnn()
        elif model_type == 'improved':
            model = cnn_builder.create_improved_cnn()
        elif model_type == 'lightweight':
            model = cnn_builder.create_lightweight_cnn()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model = cnn_builder.compile_model(model)
        logger.info(f"🏗️ {model_type.capitalize()} CNN model created and compiled")

        # Initialize trainer
        trainer = ModelTrainer(config_path)

        # Train model
        history = trainer.train(
            model=model,
            train_generator=train_gen,
            validation_generator=val_gen,
            save_path=f"models/trained/best_{model_type}_model.h5"
        )

        logger.info("✅ Model training completed successfully")
        return model, history

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

def evaluate_model(config_path: str, model_path: str):
    """Evaluate a trained model."""
    logger.info(f"📈 Starting model evaluation")

    try:
        # Load configuration
        config = read_yaml(config_path)

        # Initialize data loader
        data_loader = DataLoader(config_path)
        _, _, test_gen = data_loader.create_generators()

        # Initialize evaluator
        evaluator = ModelEvaluator(config_path)

        # Evaluate model
        results = evaluator.evaluate_model(
            model_path=model_path,
            test_generator=test_gen,
            save_results=True
        )

        logger.info("✅ Model evaluation completed successfully")
        return results

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

def predict_single(config_path: str, model_path: str, image_path: str):
    """Make a single prediction."""
    logger.info(f"🔮 Making prediction for {image_path}")

    try:
        # Initialize predictor
        predictor = FreshHarvestPredictor(config_path, model_path)

        # Make prediction
        result = predictor.predict_single(image_path)

        # Display results
        print("\n" + "="*60)
        print("🍎 FRESHNESS PREDICTION RESULTS")
        print("="*60)
        print(f"📁 Image: {image_path}")
        print(f"🍎 Fruit Type: {result['fruit_type']}")
        print(f"📊 Condition: {result['condition']}")
        print(f"🎯 Confidence: {result['confidence']:.2%}")
        print(f"⏱️ Processing Time: {result['processing_time']:.3f}s")
        print("="*60)

        logger.info("✅ Prediction completed successfully")
        return result

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise

def run_data_pipeline(config_path: str):
    """Run the complete data processing pipeline."""
    logger.info("🔄 Running data processing pipeline")

    try:
        # Initialize data loader
        data_loader = DataLoader(config_path)

        # Process data
        data_loader.prepare_data()

        # Create generators
        train_gen, val_gen, test_gen = data_loader.create_generators()

        # Generate statistics
        stats = data_loader.generate_statistics()

        logger.info("✅ Data pipeline completed successfully")
        return stats

    except Exception as e:
        logger.error(f"❌ Data pipeline failed: {e}")
        raise

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="FreshHarvest Fruit Freshness Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --model-type lightweight
  python main.py evaluate --model-path models/trained/best_model.h5
  python main.py predict --model-path models/trained/best_model.h5 --image data/sample.jpg
  python main.py data-pipeline
        """
    )

    parser.add_argument(
        'command',
        choices=['train', 'evaluate', 'predict', 'data-pipeline'],
        help='Command to execute'
    )

    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )

    parser.add_argument(
        '--model-type',
        choices=['basic', 'improved', 'lightweight'],
        default='lightweight',
        help='Type of model to train (default: lightweight)'
    )

    parser.add_argument(
        '--model-path',
        help='Path to trained model file'
    )

    parser.add_argument(
        '--image',
        help='Path to image file for prediction'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Setup directories
    setup_directories()

    # Print header
    print("\n" + "="*80)
    print("🍎 FRESHHARVEST FRUIT FRESHNESS CLASSIFICATION SYSTEM")
    print("="*80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Command: {args.command}")
    print(f"📋 Config: {args.config}")
    print("="*80)

    try:
        if args.command == 'train':
            model, history = train_model(args.config, args.model_type)
            print(f"\n✅ Training completed! Model saved to models/trained/best_{args.model_type}_model.h5")

        elif args.command == 'evaluate':
            if not args.model_path:
                parser.error("--model-path is required for evaluation")

            results = evaluate_model(args.config, args.model_path)
            print(f"\n✅ Evaluation completed! Results saved to outputs/reports/")

        elif args.command == 'predict':
            if not args.model_path or not args.image:
                parser.error("--model-path and --image are required for prediction")

            result = predict_single(args.config, args.model_path, args.image)

        elif args.command == 'data-pipeline':
            stats = run_data_pipeline(args.config)
            print(f"\n✅ Data pipeline completed! Statistics saved to outputs/reports/")

        print(f"\n🎉 Operation completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
Training script for FreshHarvest fruit freshness classification model.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.cvProject_FreshHarvest.utils.common import read_yaml, write_json, setup_logging, get_timestamp
from src.cvProject_FreshHarvest.components.data_ingestion import DataIngestion
from src.cvProject_FreshHarvest.models.cnn_models import FreshHarvestCNN

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Using dummy implementation.")
    TF_AVAILABLE = False

def setup_gpu():
    """Setup GPU configuration for TensorFlow."""
    if not TF_AVAILABLE:
        return False
    
    try:
        # Check for GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Enable memory growth for GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
            return True
        else:
            logging.info("No GPU found. Using CPU.")
            return False
    except Exception as e:
        logging.warning(f"GPU setup failed: {e}. Using CPU.")
        return False

def create_data_generators(config):
    """Create data generators for training, validation, and testing."""
    if not TF_AVAILABLE:
        logging.warning("TensorFlow not available. Skipping data generator creation.")
        return None, None, None
    
    data_config = config['data']
    training_config = config['training']
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_config['processed_data_path'] + '/train',
        target_size=tuple(data_config['image_size']),
        batch_size=training_config['batch_size'],
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        data_config['processed_data_path'] + '/val',
        target_size=tuple(data_config['image_size']),
        batch_size=training_config['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        data_config['processed_data_path'] + '/test',
        target_size=tuple(data_config['image_size']),
        batch_size=training_config['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    
    logging.info(f"Created data generators - Train: {train_generator.samples}, "
                f"Val: {val_generator.samples}, Test: {test_generator.samples}")
    
    return train_generator, val_generator, test_generator

def create_callbacks(config):
    """Create training callbacks."""
    if not TF_AVAILABLE:
        return []
    
    training_config = config['training']
    paths_config = config['paths']
    
    # Create directories
    os.makedirs(paths_config['checkpoints'], exist_ok=True)
    os.makedirs(paths_config['logs'], exist_ok=True)
    
    timestamp = get_timestamp()
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor=training_config['early_stopping']['monitor'],
            patience=training_config['early_stopping']['patience'],
            restore_best_weights=training_config['early_stopping']['restore_best_weights'],
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=f"{paths_config['checkpoints']}/best_model_{timestamp}.h5",
            monitor=training_config['checkpoint']['monitor'],
            save_best_only=training_config['checkpoint']['save_best_only'],
            save_weights_only=training_config['checkpoint']['save_weights_only'],
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks

def train_model(config_path: str, model_type: str = 'basic'):
    """
    Train the FreshHarvest classification model.
    
    Args:
        config_path: Path to configuration file
        model_type: Type of model to train ('basic', 'improved', 'lightweight')
    """
    # Setup logging
    setup_logging(level="INFO")
    logging.info("Starting FreshHarvest model training")
    
    # Load configuration
    config = read_yaml(config_path)
    training_config = config['training']
    
    # Setup GPU
    gpu_available = setup_gpu()
    
    if not TF_AVAILABLE:
        logging.error("TensorFlow is not available. Cannot proceed with training.")
        return None
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators(config)
    
    if train_gen is None:
        logging.error("Failed to create data generators.")
        return None
    
    # Create model
    cnn_model = FreshHarvestCNN(config_path)
    
    if model_type == 'basic':
        model = cnn_model.create_basic_cnn()
    elif model_type == 'improved':
        model = cnn_model.create_improved_cnn()
    elif model_type == 'lightweight':
        model = cnn_model.create_lightweight_cnn()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compile model
    model = cnn_model.compile_model(
        model, 
        learning_rate=training_config['learning_rate'],
        optimizer=training_config['optimizer']
    )
    
    # Print model summary
    logging.info("Model Summary:")
    logging.info(cnn_model.get_model_summary(model))
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Train model
    logging.info(f"Starting training for {training_config['epochs']} epochs")
    
    history = model.fit(
        train_gen,
        epochs=training_config['epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    timestamp = get_timestamp()
    model_path = f"{config['paths']['trained_models']}/final_model_{model_type}_{timestamp}.h5"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Evaluate on test set
    logging.info("Evaluating model on test set")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_gen, verbose=1)
    
    # Calculate F1 score
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    
    # Save training results
    results = {
        'model_type': model_type,
        'timestamp': timestamp,
        'model_path': model_path,
        'training_config': training_config,
        'final_metrics': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1)
        },
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        },
        'gpu_used': gpu_available
    }
    
    results_path = f"{config['paths']['outputs']}/training_results_{model_type}_{timestamp}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    write_json(results, results_path)
    
    logging.info(f"Training completed successfully!")
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")
    logging.info(f"Results saved to {results_path}")
    
    return model, history, results

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train FreshHarvest classification model')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--model_type', default='basic', choices=['basic', 'improved', 'lightweight'],
                       help='Type of model to train')
    
    args = parser.parse_args()
    
    try:
        # Train model
        model, history, results = train_model(args.config, args.model_type)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        if results:
            print(f"Model Type: {results['model_type']}")
            print(f"Test Accuracy: {results['final_metrics']['test_accuracy']:.4f}")
            print(f"Test F1 Score: {results['final_metrics']['test_f1']:.4f}")
            print(f"Model saved to: {results['model_path']}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()

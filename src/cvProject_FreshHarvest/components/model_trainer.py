"""
Model Trainer Component for FreshHarvest Classification System
============================================================

This module provides comprehensive model training functionality
including architecture definition, training pipeline, and optimization.

Author: FreshHarvest Team
Version: 1.0.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import os
from pathlib import Path
import time
from datetime import datetime

from ..utils.common import read_yaml, create_directories

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Comprehensive model training for FreshHarvest classification.

    Handles model architecture creation, training pipeline setup,
    and training execution with monitoring and callbacks.
    """

    def __init__(self, config_path: str):
        """
        Initialize model trainer.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.model_config = self.config['model']
        self.training_config = self.config.get('training', {})

        # Model specifications
        self.input_shape = tuple(self.model_config['input_shape'])
        self.num_classes = self.model_config['num_classes']

        # Training parameters
        self.epochs = self.training_config.get('epochs', 50)
        self.batch_size = self.training_config.get('batch_size', 32)
        self.learning_rate = self.training_config.get('learning_rate', 0.001)

        # Model and training artifacts
        self.model = None
        self.history = None
        self.training_metadata = {}

        logger.info("Model trainer initialized")

    def create_model(self) -> keras.Model:
        """
        Create CNN model architecture for fruit freshness classification.

        Returns:
            Compiled Keras model
        """
        try:
            # Input layer
            inputs = keras.layers.Input(shape=self.input_shape)

            # First convolutional block
            x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.1)(x)

            x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            x = keras.layers.Dropout(0.2)(x)

            # Second convolutional block
            x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)

            x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            x = keras.layers.Dropout(0.3)(x)

            # Third convolutional block
            x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)

            x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            x = keras.layers.Dropout(0.4)(x)

            # Global average pooling and dense layers
            x = keras.layers.GlobalAveragePooling2D()(x)

            x = keras.layers.Dense(512, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.5)(x)

            x = keras.layers.Dense(256, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.5)(x)

            # Output layer
            outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)

            # Create model
            model = keras.Model(inputs=inputs, outputs=outputs, name='FreshHarvest_CNN')

            logger.info(f"Model created: {model.count_params()} parameters")
            return model

        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return None

    def compile_model(self, model: keras.Model) -> keras.Model:
        """
        Compile model with optimizer, loss, and metrics.

        Args:
            model: Keras model to compile

        Returns:
            Compiled model
        """
        try:
            # Optimizer
            optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )

            # Compile model
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            logger.info("Model compiled successfully")
            return model

        except Exception as e:
            logger.error(f"Error compiling model: {e}")
            return model

    def create_callbacks(self, model_save_path: str) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks.

        Args:
            model_save_path: Path to save best model

        Returns:
            List of callbacks
        """
        try:
            callbacks = []

            # Early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True,
                min_delta=0.001,
                verbose=1
            )
            callbacks.append(early_stopping)

            # Model checkpoint
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(model_checkpoint)

            # Reduce learning rate
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(reduce_lr)

            # TensorBoard logging
            tensorboard_dir = Path(model_save_path).parent / "tensorboard_logs"
            create_directories([str(tensorboard_dir)])

            tensorboard = keras.callbacks.TensorBoard(
                log_dir=str(tensorboard_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
            callbacks.append(tensorboard)

            logger.info(f"Created {len(callbacks)} training callbacks")
            return callbacks

        except Exception as e:
            logger.error(f"Error creating callbacks: {e}")
            return []

    def train_model(self, train_data, validation_data,
                   model_save_path: str) -> Dict[str, Any]:
        """
        Train the model with given data.

        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            model_save_path: Path to save trained model

        Returns:
            Training results dictionary
        """
        try:
            # Create and compile model
            self.model = self.create_model()
            if self.model is None:
                raise ValueError("Failed to create model")

            self.model = self.compile_model(self.model)

            # Create callbacks
            callbacks = self.create_callbacks(model_save_path)

            # Print model summary
            logger.info("Model Architecture:")
            self.model.summary(print_fn=logger.info)

            # Start training
            start_time = time.time()
            logger.info(f"Starting training for {self.epochs} epochs...")

            self.history = self.model.fit(
                train_data,
                epochs=self.epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )

            training_time = time.time() - start_time

            # Get best metrics
            best_epoch = np.argmax(self.history.history['val_accuracy'])
            best_val_accuracy = max(self.history.history['val_accuracy'])
            best_train_accuracy = self.history.history['accuracy'][best_epoch]
            final_val_loss = self.history.history['val_loss'][best_epoch]

            # Training metadata
            self.training_metadata = {
                'training_time_seconds': training_time,
                'total_epochs': len(self.history.history['accuracy']),
                'best_epoch': best_epoch + 1,
                'best_val_accuracy': float(best_val_accuracy),
                'best_train_accuracy': float(best_train_accuracy),
                'final_val_loss': float(final_val_loss),
                'training_date': datetime.now().isoformat(),
                'model_parameters': self.model.count_params(),
                'early_stopping': len(self.history.history['accuracy']) < self.epochs
            }

            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch + 1}")

            # Save training metadata
            metadata_path = Path(model_save_path).parent / "training_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.training_metadata, f, indent=2)

            return {
                'success': True,
                'model_path': model_save_path,
                'metadata': self.training_metadata,
                'history': self.history.history
            }

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def evaluate_model(self, test_data) -> Dict[str, float]:
        """
        Evaluate trained model on test data.

        Args:
            test_data: Test dataset

        Returns:
            Evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train_model() first.")

            logger.info("Evaluating model on test data...")

            # Evaluate model
            test_loss, test_accuracy = self.model.evaluate(test_data, verbose=1)

            # Get predictions for detailed metrics
            predictions = self.model.predict(test_data, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)

            # Get true labels
            true_labels = []
            for batch in test_data:
                true_labels.extend(batch[1].numpy())
            true_labels = np.array(true_labels)

            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score

            precision = precision_score(true_labels, predicted_classes, average='weighted')
            recall = recall_score(true_labels, predicted_classes, average='weighted')
            f1 = f1_score(true_labels, predicted_classes, average='weighted')

            evaluation_results = {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }

            logger.info(f"Test Results - Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            return evaluation_results

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}

    def save_model_artifacts(self, save_dir: str) -> bool:
        """
        Save all model artifacts including weights, config, and metadata.

        Args:
            save_dir: Directory to save artifacts

        Returns:
            True if successful
        """
        try:
            save_path = Path(save_dir)
            create_directories([str(save_path)])

            if self.model is None:
                raise ValueError("No model to save")

            # Save model
            model_path = save_path / "model.h5"
            self.model.save(str(model_path))
            logger.info(f"Model saved to {model_path}")

            # Save weights separately
            weights_path = save_path / "weights.h5"
            self.model.save_weights(str(weights_path))
            logger.info(f"Weights saved to {weights_path}")

            # Save model config
            config_path = save_path / "model_config.json"
            model_config = {
                'architecture': self.model.to_json(),
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'class_names': [
                    "Fresh Apple", "Fresh Banana", "Fresh Orange",
                    "Rotten Apple", "Rotten Banana", "Rotten Orange"
                ]
            }

            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            logger.info(f"Model config saved to {config_path}")

            # Save training history
            if self.history is not None:
                history_path = save_path / "training_history.json"
                with open(history_path, 'w') as f:
                    json.dump(self.history.history, f, indent=2)
                logger.info(f"Training history saved to {history_path}")

            return True

        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            return False

def train_fresh_harvest_model(train_data, validation_data, test_data,
                             config_path: str, save_dir: str) -> Dict[str, Any]:
    """
    Convenience function to train FreshHarvest model.

    Args:
        train_data: Training dataset
        validation_data: Validation dataset
        test_data: Test dataset
        config_path: Path to configuration file
        save_dir: Directory to save model artifacts

    Returns:
        Training and evaluation results
    """
    trainer = ModelTrainer(config_path)

    # Train model
    model_save_path = Path(save_dir) / "model.h5"
    training_results = trainer.train_model(train_data, validation_data, str(model_save_path))

    if not training_results['success']:
        return training_results

    # Evaluate model
    evaluation_results = trainer.evaluate_model(test_data)

    # Save all artifacts
    trainer.save_model_artifacts(save_dir)

    # Combine results
    results = {
        'training': training_results,
        'evaluation': evaluation_results,
        'model_path': str(model_save_path),
        'artifacts_dir': save_dir
    }

    return results
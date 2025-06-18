"""
CNN model architectures for the FreshHarvest project.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from typing import Tuple, Optional

from ..utils.common import read_yaml


class FreshHarvestCNN:
    """
    Custom CNN model for fruit freshness classification.
    """

    def __init__(self, config_path: str):
        """
        Initialize CNN model.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.model_config = self.config['model']
        self.data_config = self.config['data']

        # Model specifications
        self.input_shape = tuple(self.model_config['input_shape'])
        self.num_classes = self.model_config['num_classes']

        logging.info("CNN model initialized")

    def create_basic_cnn(self, dropout_rate: float = 0.5) -> keras.Model:
        """
        Create a basic CNN model for fruit freshness classification.

        Args:
            dropout_rate: Dropout rate for regularization

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),

            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Global average pooling
            layers.GlobalAveragePooling2D(),

            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        logging.info(f"Created basic CNN model with {model.count_params()} parameters")
        return model

    def create_improved_cnn(self, dropout_rate: float = 0.5) -> keras.Model:
        """
        Create an improved CNN model with residual connections.

        Args:
            dropout_rate: Dropout rate for regularization

        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape)

        # Initial convolution
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

        # Residual blocks
        x = self._residual_block(x, 64, stride=1)
        x = self._residual_block(x, 64, stride=1)

        x = self._residual_block(x, 128, stride=2)
        x = self._residual_block(x, 128, stride=1)

        x = self._residual_block(x, 256, stride=2)
        x = self._residual_block(x, 256, stride=1)

        x = self._residual_block(x, 512, stride=2)
        x = self._residual_block(x, 512, stride=1)

        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)

        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        logging.info(f"Created improved CNN model with {model.count_params()} parameters")
        return model

    def _residual_block(self, x, filters: int, stride: int = 1):
        """
        Create a residual block.

        Args:
            x: Input tensor
            filters: Number of filters
            stride: Stride for convolution

        Returns:
            Output tensor
        """
        shortcut = x

        # First convolution
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Second convolution
        x = layers.Conv2D(filters, (3, 3), strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Adjust shortcut if needed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

        return x

    def create_lightweight_cnn(self, dropout_rate: float = 0.3) -> keras.Model:
        """
        Create a lightweight CNN model for faster inference.

        Args:
            dropout_rate: Dropout rate for regularization

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),

            # Depthwise separable convolutions
            layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),

            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        logging.info(f"Created lightweight CNN model with {model.count_params()} parameters")
        return model

    def compile_model(self, model: keras.Model,
                     learning_rate: float = 0.001,
                     optimizer: str = 'adam') -> keras.Model:
        """
        Compile the model with optimizer, loss, and metrics.

        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer name

        Returns:
            Compiled model
        """
        # Choose optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Compile model
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        logging.info(f"Model compiled with {optimizer} optimizer (lr={learning_rate})")
        return model

    def get_model_summary(self, model: keras.Model) -> str:
        """
        Get model summary as string.

        Args:
            model: Keras model

        Returns:
            Model summary string
        """
        import io
        import sys

        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        sys.stdout = old_stdout

        return buffer.getvalue()
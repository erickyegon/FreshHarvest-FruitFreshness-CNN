"""
Configuration Entity Classes for FreshHarvest
============================================

This module defines configuration entity classes for the FreshHarvest
fruit freshness classification system.

Author: FreshHarvest Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion component."""
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration for data validation component."""
    root_dir: Path
    status_file: str
    required_files: List[str]

@dataclass(frozen=True)
class DataPreprocessingConfig:
    """Configuration for data preprocessing component."""
    root_dir: Path
    data_path: Path
    image_size: Tuple[int, int]
    batch_size: int
    validation_split: float
    test_split: float

@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration for model training component."""
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Configuration for model evaluation component."""
    path_of_model: Path
    training_data: Path
    all_params: Dict[str, Any]
    mlflow_uri: str
    params_image_size: List[int]
    params_batch_size: int

@dataclass(frozen=True)
class ModelDeploymentConfig:
    """Configuration for model deployment component."""
    model_path: Path
    api_host: str
    api_port: int
    max_batch_size: int
    timeout_seconds: int
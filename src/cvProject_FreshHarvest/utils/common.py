"""
Common utility functions for the FreshHarvest project.
"""

import os
import sys
import yaml
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd


def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read YAML file and return its contents.

    Args:
        file_path: Path to the YAML file

    Returns:
        Dictionary containing YAML contents
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error reading YAML file {file_path}: {e}")
        raise


def write_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Write data to YAML file.

    Args:
        data: Dictionary to write to YAML
        file_path: Path where to save the YAML file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
    except Exception as e:
        logging.error(f"Error writing YAML file {file_path}: {e}")
        raise


def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read JSON file and return its contents.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing JSON contents
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error reading JSON file {file_path}: {e}")
        raise


def write_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Write data to JSON file.

    Args:
        data: Dictionary to write to JSON
        file_path: Path where to save the JSON file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4, default=_json_serializer)
    except Exception as e:
        logging.error(f"Error writing JSON file {file_path}: {e}")
        raise


def _json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        file_path: Path where to save the pickle file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        logging.error(f"Error saving pickle file {file_path}: {e}")
        raise


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Load object from pickle file.

    Args:
        file_path: Path to the pickle file

    Returns:
        Loaded object
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logging.error(f"Error loading pickle file {file_path}: {e}")
        raise


def create_directories(paths: List[Union[str, Path]]) -> None:
    """
    Create directories if they don't exist.

    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created directory: {path}")


def get_size(file_path: Union[str, Path]) -> str:
    """
    Get file size in human readable format.

    Args:
        file_path: Path to the file

    Returns:
        File size as string
    """
    size_bytes = os.path.getsize(file_path)

    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s} {size_names[i]}"


def setup_logging(log_file: Optional[Union[str, Path]] = None,
                 level: str = "INFO") -> None:
    """
    Setup logging configuration.

    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def get_timestamp() -> str:
    """
    Get current timestamp as string.

    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(file_path: Union[str, Path]) -> None:
    """
    Ensure directory exists for the given file path.

    Args:
        file_path: File path
    """
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
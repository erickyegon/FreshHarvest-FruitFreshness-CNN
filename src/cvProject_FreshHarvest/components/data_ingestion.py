"""
Data ingestion component for the FreshHarvest project.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

from ..utils.common import read_yaml, write_json, create_directories


class DataIngestion:
    """
    Data ingestion component for loading and organizing the dataset.
    """

    def __init__(self, config_path: str):
        """
        Initialize data ingestion component.

        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.data_config = self.config['data']
        self.paths_config = self.config['paths']

        # Setup paths
        self.raw_data_path = Path(self.data_config['raw_data_path'])
        self.processed_data_path = Path(self.data_config['processed_data_path'])
        self.interim_data_path = Path(self.data_config['interim_data_path'])

        # Create directories
        create_directories([
            self.processed_data_path,
            self.interim_data_path,
            self.processed_data_path / 'train',
            self.processed_data_path / 'val',
            self.processed_data_path / 'test'
        ])

        logging.info("Data ingestion component initialized")

    def scan_dataset(self) -> Dict[str, int]:
        """
        Scan the raw dataset and count images per class.

        Returns:
            Dictionary with class names and image counts
        """
        class_counts = {}

        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path does not exist: {self.raw_data_path}")

        # Get all class directories
        class_dirs = [d for d in self.raw_data_path.iterdir() if d.is_dir()]

        for class_dir in class_dirs:
            # Count image files
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            class_counts[class_dir.name] = len(image_files)

        logging.info(f"Found {len(class_counts)} classes with {sum(class_counts.values())} total images")
        return class_counts

    def create_file_list(self) -> pd.DataFrame:
        """
        Create a DataFrame with all image files and their labels.

        Returns:
            DataFrame with columns: filepath, class_name, label
        """
        file_data = []

        # Get all class directories
        class_dirs = [d for d in self.raw_data_path.iterdir() if d.is_dir()]
        class_names = sorted([d.name for d in class_dirs])

        # Create label mapping
        label_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}

        for class_dir in class_dirs:
            class_name = class_dir.name
            label = label_mapping[class_name]

            # Get all image files
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

            for image_file in image_files:
                file_data.append({
                    'filepath': str(image_file),
                    'class_name': class_name,
                    'label': label,
                    'fruit_type': class_name[2:],  # Remove F_ or S_ prefix
                    'condition': 'Fresh' if class_name.startswith('F_') else 'Spoiled'
                })

        df = pd.DataFrame(file_data)

        # Save label mapping
        label_mapping_path = self.interim_data_path / 'label_mapping.json'
        write_json(label_mapping, label_mapping_path)

        # Save class names
        class_names_path = self.interim_data_path / 'class_names.json'
        write_json(class_names, class_names_path)

        logging.info(f"Created file list with {len(df)} images")
        return df

    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            df: DataFrame with image file information

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_split = self.data_config['train_split']
        val_split = self.data_config['val_split']
        test_split = self.data_config['test_split']

        # Ensure splits sum to 1
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")

        # Stratified split by class
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_split + test_split),
            stratify=df['class_name'],
            random_state=42
        )

        # Split temp into val and test
        val_size = val_split / (val_split + test_split)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['class_name'],
            random_state=42
        )

        logging.info(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def organize_data(self, train_df: pd.DataFrame,
                     val_df: pd.DataFrame,
                     test_df: pd.DataFrame) -> None:
        """
        Organize data into train/val/test directories.

        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame
            test_df: Test data DataFrame
        """
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        for split_name, split_df in splits.items():
            split_path = self.processed_data_path / split_name

            # Create class directories
            for class_name in split_df['class_name'].unique():
                class_dir = split_path / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for _, row in split_df.iterrows():
                src_path = Path(row['filepath'])
                dst_path = split_path / row['class_name'] / src_path.name

                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)

            logging.info(f"Organized {len(split_df)} images for {split_name} split")

    def generate_dataset_report(self, train_df: pd.DataFrame,
                              val_df: pd.DataFrame,
                              test_df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive dataset report.

        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame
            test_df: Test data DataFrame

        Returns:
            Dictionary containing dataset statistics
        """
        report = {
            'total_images': len(train_df) + len(val_df) + len(test_df),
            'num_classes': len(train_df['class_name'].unique()),
            'class_names': sorted(train_df['class_name'].unique()),
            'splits': {
                'train': {
                    'count': len(train_df),
                    'percentage': len(train_df) / (len(train_df) + len(val_df) + len(test_df)) * 100
                },
                'val': {
                    'count': len(val_df),
                    'percentage': len(val_df) / (len(train_df) + len(val_df) + len(test_df)) * 100
                },
                'test': {
                    'count': len(test_df),
                    'percentage': len(test_df) / (len(train_df) + len(val_df) + len(test_df)) * 100
                }
            },
            'class_distribution': {
                'train': dict(train_df['class_name'].value_counts()),
                'val': dict(val_df['class_name'].value_counts()),
                'test': dict(test_df['class_name'].value_counts())
            },
            'fruit_distribution': {
                'train': dict(train_df['fruit_type'].value_counts()),
                'val': dict(val_df['fruit_type'].value_counts()),
                'test': dict(test_df['fruit_type'].value_counts())
            },
            'condition_distribution': {
                'train': dict(train_df['condition'].value_counts()),
                'val': dict(val_df['condition'].value_counts()),
                'test': dict(test_df['condition'].value_counts())
            }
        }

        # Save report
        report_path = self.interim_data_path / 'dataset_report.json'
        write_json(report, report_path)

        logging.info("Dataset report generated")
        return report

    def run_data_ingestion(self) -> Dict:
        """
        Run the complete data ingestion pipeline.

        Returns:
            Dataset report dictionary
        """
        logging.info("Starting data ingestion pipeline")

        # Scan dataset
        class_counts = self.scan_dataset()

        # Create file list
        df = self.create_file_list()

        # Split dataset
        train_df, val_df, test_df = self.split_dataset(df)

        # Organize data
        self.organize_data(train_df, val_df, test_df)

        # Generate report
        report = self.generate_dataset_report(train_df, val_df, test_df)

        logging.info("Data ingestion pipeline completed successfully")
        return report
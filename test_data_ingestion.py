"""
Test script for data ingestion component.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.cvProject_FreshHarvest.components.data_ingestion import DataIngestion
from src.cvProject_FreshHarvest.utils.common import setup_logging

def main():
    """Main function to test data ingestion."""
    
    # Setup logging
    setup_logging(level="INFO")
    
    try:
        # Initialize data ingestion
        config_path = "config/config.yaml"
        data_ingestion = DataIngestion(config_path)
        
        # Run data ingestion pipeline
        report = data_ingestion.run_data_ingestion()
        
        # Print summary
        print("\n" + "="*50)
        print("DATA INGESTION SUMMARY")
        print("="*50)
        print(f"Total images: {report['total_images']}")
        print(f"Number of classes: {report['num_classes']}")
        print(f"Train images: {report['splits']['train']['count']}")
        print(f"Validation images: {report['splits']['val']['count']}")
        print(f"Test images: {report['splits']['test']['count']}")
        print("\nClass distribution (train):")
        for class_name, count in report['class_distribution']['train'].items():
            print(f"  {class_name}: {count}")
        
        print("\nData ingestion completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in data ingestion: {e}")
        raise

if __name__ == "__main__":
    main()

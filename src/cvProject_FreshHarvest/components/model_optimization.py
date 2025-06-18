"""
Model optimization and deployment component for the FreshHarvest project.
"""

import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..utils.common import read_yaml, write_json, get_timestamp


class ModelOptimizer:
    """
    Model optimization and deployment preparation component.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize model optimizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.data_config = self.config['data']
        
        logging.info("Model optimizer initialized")
    
    def quantize_model(self, model_path: str, output_path: str) -> str:
        """
        Quantize model for faster inference and smaller size.
        
        Args:
            model_path: Path to the original model
            output_path: Path to save quantized model
            
        Returns:
            Path to quantized model
        """
        if not TF_AVAILABLE:
            logging.error("TensorFlow not available for quantization")
            return ""
        
        try:
            # Load model
            model = keras.models.load_model(model_path)
            
            # Convert to TensorFlow Lite with quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert model
            quantized_model = converter.convert()
            
            # Save quantized model
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(quantized_model)
            
            logging.info(f"Model quantized and saved to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error quantizing model: {e}")
            return ""
    
    def optimize_for_inference(self, model_path: str, output_dir: str) -> Dict[str, str]:
        """
        Optimize model for inference deployment.
        
        Args:
            model_path: Path to the original model
            output_dir: Directory to save optimized models
            
        Returns:
            Dictionary with paths to optimized models
        """
        if not TF_AVAILABLE:
            logging.error("TensorFlow not available for optimization")
            return {}
        
        optimized_models = {}
        timestamp = get_timestamp()
        
        try:
            # Load original model
            model = keras.models.load_model(model_path)
            
            # 1. SavedModel format (for TensorFlow Serving)
            saved_model_path = f"{output_dir}/saved_model_{timestamp}"
            model.save(saved_model_path, save_format='tf')
            optimized_models['saved_model'] = saved_model_path
            
            # 2. TensorFlow Lite (standard)
            tflite_path = f"{output_dir}/model_{timestamp}.tflite"
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            optimized_models['tflite'] = tflite_path
            
            # 3. TensorFlow Lite (quantized)
            quantized_path = f"{output_dir}/model_quantized_{timestamp}.tflite"
            quantized_model_path = self.quantize_model(model_path, quantized_path)
            if quantized_model_path:
                optimized_models['tflite_quantized'] = quantized_model_path
            
            logging.info(f"Model optimization completed. Models saved to {output_dir}")
            return optimized_models
            
        except Exception as e:
            logging.error(f"Error optimizing model: {e}")
            return {}
    
    def benchmark_model(self, model_path: str, num_samples: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference performance.
        
        Args:
            model_path: Path to the model
            num_samples: Number of samples for benchmarking
            
        Returns:
            Dictionary with performance metrics
        """
        if not TF_AVAILABLE:
            logging.error("TensorFlow not available for benchmarking")
            return {}
        
        try:
            # Load model
            model = keras.models.load_model(model_path)
            
            # Generate random test data
            input_shape = tuple(self.config['model']['input_shape'])
            test_data = np.random.random((num_samples,) + input_shape).astype(np.float32)
            
            # Warm up
            _ = model.predict(test_data[:10], verbose=0)
            
            # Benchmark
            start_time = time.time()
            predictions = model.predict(test_data, verbose=0)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_sample = total_time / num_samples
            throughput = num_samples / total_time
            
            # Model size
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            benchmark_results = {
                'total_inference_time': total_time,
                'avg_time_per_sample': avg_time_per_sample,
                'throughput_samples_per_sec': throughput,
                'model_size_mb': model_size_mb,
                'num_samples': num_samples
            }
            
            logging.info(f"Benchmark completed - Throughput: {throughput:.2f} samples/sec")
            return benchmark_results
            
        except Exception as e:
            logging.error(f"Error benchmarking model: {e}")
            return {}
    
    def create_deployment_package(self, model_path: str, output_dir: str) -> str:
        """
        Create deployment package with all necessary files.
        
        Args:
            model_path: Path to the trained model
            output_dir: Output directory for deployment package
            
        Returns:
            Path to deployment package
        """
        timestamp = get_timestamp()
        package_dir = f"{output_dir}/deployment_package_{timestamp}"
        os.makedirs(package_dir, exist_ok=True)
        
        try:
            # 1. Copy model files
            model_dir = f"{package_dir}/model"
            os.makedirs(model_dir, exist_ok=True)
            
            # Optimize models
            optimized_models = self.optimize_for_inference(model_path, model_dir)
            
            # 2. Copy configuration
            config_dir = f"{package_dir}/config"
            os.makedirs(config_dir, exist_ok=True)
            
            import shutil
            shutil.copy2("config/config.yaml", f"{config_dir}/config.yaml")
            
            # Copy class mappings
            if os.path.exists("data/interim/label_mapping.json"):
                shutil.copy2("data/interim/label_mapping.json", f"{config_dir}/label_mapping.json")
            if os.path.exists("data/interim/class_names.json"):
                shutil.copy2("data/interim/class_names.json", f"{config_dir}/class_names.json")
            
            # 3. Create deployment files
            self._create_deployment_files(package_dir, optimized_models)
            
            # 4. Benchmark models
            benchmark_results = {}
            for model_type, model_path in optimized_models.items():
                if model_type == 'saved_model':
                    benchmark_results[model_type] = self.benchmark_model(model_path)
            
            # Save benchmark results
            if benchmark_results:
                write_json(benchmark_results, f"{package_dir}/benchmark_results.json")
            
            logging.info(f"Deployment package created at {package_dir}")
            return package_dir
            
        except Exception as e:
            logging.error(f"Error creating deployment package: {e}")
            return ""
    
    def _create_deployment_files(self, package_dir: str, optimized_models: Dict[str, str]) -> None:
        """Create all deployment files."""
        # Create inference script
        inference_script = '''"""
Production inference script for FreshHarvest model.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import json
import yaml

class FreshHarvestInference:
    def __init__(self, model_path: str, config_path: str):
        self.model = tf.keras.models.load_model(model_path)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load class names
        with open("config/class_names.json", 'r') as f:
            self.class_names = json.load(f)
    
    def predict(self, image_path: str) -> dict:
        """Make prediction on image."""
        image = Image.open(image_path).convert('RGB')
        image = image.resize(tuple(self.config['data']['image_size']))
        image_array = np.array(image) / 255.0
        processed_image = np.expand_dims(image_array, axis=0)
        
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = self.class_names[predicted_class_idx]
        
        condition = "Fresh" if predicted_class.startswith('F_') else "Spoiled"
        fruit_type = predicted_class[2:]
        
        return {
            'fruit_type': fruit_type,
            'condition': condition,
            'confidence': confidence
        }
'''
        
        with open(f"{package_dir}/inference.py", 'w') as f:
            f.write(inference_script)
        
        # Create Dockerfile
        dockerfile = '''FROM tensorflow/tensorflow:2.13.0
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
'''
        
        with open(f"{package_dir}/Dockerfile", 'w') as f:
            f.write(dockerfile)
        
        # Create requirements
        requirements = '''tensorflow>=2.13.0
streamlit>=1.25.0
Pillow>=10.0.0
numpy>=1.24.0
PyYAML>=6.0
'''
        
        with open(f"{package_dir}/requirements.txt", 'w') as f:
            f.write(requirements)

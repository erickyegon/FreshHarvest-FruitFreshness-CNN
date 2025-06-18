"""
Model deployment script for FreshHarvest fruit freshness classification.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.cvProject_FreshHarvest.utils.common import read_yaml, setup_logging, get_timestamp
from src.cvProject_FreshHarvest.components.model_optimization import ModelOptimizer


def optimize_and_deploy(model_path: str, config_path: str, output_dir: str):
    """
    Optimize model and create deployment package.
    
    Args:
        model_path: Path to the trained model
        config_path: Path to configuration file
        output_dir: Output directory for deployment package
    """
    # Setup logging
    setup_logging(level="INFO")
    logging.info("Starting model optimization and deployment preparation")
    
    # Initialize optimizer
    optimizer = ModelOptimizer(config_path)
    
    # Create deployment package
    package_path = optimizer.create_deployment_package(model_path, output_dir)
    
    if package_path:
        print("\n" + "="*60)
        print("DEPLOYMENT PACKAGE CREATED SUCCESSFULLY")
        print("="*60)
        print(f"Package location: {package_path}")
        print("\nContents:")
        print("- Optimized model files (SavedModel, TensorFlow Lite)")
        print("- Configuration files")
        print("- Inference script")
        print("- Docker files")
        print("- Deployment documentation")
        print("\nNext steps:")
        print("1. Test the inference script")
        print("2. Build Docker container")
        print("3. Deploy to production environment")
        print("="*60)
    else:
        print("Failed to create deployment package")


def benchmark_model(model_path: str, config_path: str):
    """
    Benchmark model performance.
    
    Args:
        model_path: Path to the trained model
        config_path: Path to configuration file
    """
    setup_logging(level="INFO")
    logging.info("Starting model benchmarking")
    
    optimizer = ModelOptimizer(config_path)
    results = optimizer.benchmark_model(model_path, num_samples=100)
    
    if results:
        print("\n" + "="*50)
        print("MODEL PERFORMANCE BENCHMARK")
        print("="*50)
        print(f"Model size: {results['model_size_mb']:.2f} MB")
        print(f"Average inference time: {results['avg_time_per_sample']*1000:.2f} ms")
        print(f"Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"Total time for {results['num_samples']} samples: {results['total_inference_time']:.2f} sec")
        print("="*50)
    else:
        print("Benchmarking failed")


def create_production_app(package_dir: str):
    """
    Create production-ready Streamlit app.
    
    Args:
        package_dir: Deployment package directory
    """
    app_code = '''"""
Production Streamlit application for FreshHarvest.
"""

import streamlit as st
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from inference import FreshHarvestInference

# Configure page
st.set_page_config(
    page_title="FreshHarvest - Production",
    page_icon="🍎",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the production model."""
    model_path = "model/saved_model"
    config_path = "config/config.yaml"
    return FreshHarvestInference(model_path, config_path)

def main():
    st.title("🍎 FreshHarvest - Production Classifier")
    st.write("AI-powered fruit freshness classification for FreshHarvest Logistics")
    
    # Load model
    try:
        inference_model = load_model()
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload fruit image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of a fruit for freshness analysis"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Get prediction
                result = inference_model.predict(temp_path)
                
                # Clean up
                import os
                os.remove(temp_path)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Fruit Type", result['fruit_type'])
                
                with col2:
                    condition_color = "🟢" if result['condition'] == "Fresh" else "🔴"
                    st.metric("Condition", f"{condition_color} {result['condition']}")
                
                # Confidence
                st.metric("Confidence", f"{result['confidence']:.1%}")
                
                # Recommendation
                if result['condition'] == "Fresh":
                    st.success("✅ This fruit appears fresh and safe for consumption")
                else:
                    st.warning("⚠️ This fruit appears spoiled and should be discarded")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open(f"{package_dir}/app.py", 'w') as f:
        f.write(app_code)
    
    print(f"Production app created at {package_dir}/app.py")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy FreshHarvest classification model')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', default='deployment', help='Output directory')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark model performance')
    parser.add_argument('--optimize_only', action='store_true', help='Only optimize, don\'t create full package')
    
    args = parser.parse_args()
    
    try:
        if args.benchmark:
            # Benchmark model
            benchmark_model(args.model_path, args.config)
        elif args.optimize_only:
            # Only optimize model
            optimizer = ModelOptimizer(args.config)
            optimized_models = optimizer.optimize_for_inference(args.model_path, args.output_dir)
            print(f"Optimized models saved to: {args.output_dir}")
            for model_type, path in optimized_models.items():
                print(f"  {model_type}: {path}")
        else:
            # Full deployment package
            optimize_and_deploy(args.model_path, args.config, args.output_dir)
            
            # Create production app
            package_dirs = [d for d in Path(args.output_dir).iterdir() 
                          if d.is_dir() and d.name.startswith('deployment_package_')]
            if package_dirs:
                latest_package = max(package_dirs, key=lambda x: x.stat().st_mtime)
                create_production_app(str(latest_package))
    
    except Exception as e:
        logging.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()

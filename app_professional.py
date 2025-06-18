"""
FreshHarvest Professional Streamlit Application
==============================================

Professional, enterprise-grade UI for FreshHarvest fruit freshness classification.
Features modern design, comprehensive labeling, and production-ready interface.

Author: FreshHarvest Team
Version: 2.0.0 (Professional UI)
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from pathlib import Path
import sys
import random
import tensorflow as tf
import plotly.graph_objects as go
from datetime import datetime
import time

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore')

# Import custom UI components
from ui_components import (
    load_professional_css, create_header, create_metric_cards,
    create_status_indicator, create_info_card, create_prediction_display,
    create_confidence_gauge, create_prediction_chart, create_upload_area,
    create_footer, create_sidebar_info, create_feature_showcase
)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Page configuration
st.set_page_config(
    page_title="FreshHarvest AI - Professional",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/erickyegon/FreshHarvest-FruitFreshness-CNN',
        'Report a bug': 'https://github.com/erickyegon/FreshHarvest-FruitFreshness-CNN/issues',
        'About': "FreshHarvest AI - Professional fruit freshness classification system with 96.50% accuracy"
    }
)

def read_yaml(file_path):
    """Read YAML configuration file."""
    try:
        import yaml
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error reading YAML file {file_path}: {e}")
        return None

def read_json(file_path):
    """Read JSON file."""
    try:
        import json
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        st.warning(f"Could not read {file_path}: {e}")
        return None

def load_config():
    """Load configuration files with fallback."""
    try:
        config = read_yaml("config/config.yaml")
        label_mapping = read_json("data/interim/label_mapping.json")
        class_names = read_json("data/interim/class_names.json")
        return config, label_mapping, class_names
    except Exception as e:
        st.error(f"Configuration loading error: {e}")
        return None, None, None

@st.cache_resource
def load_trained_model():
    """Load the trained model if available."""
    try:
        model_paths = [
            'models/trained/best_model_96.50acc.h5',
            'models/checkpoints/best_model_20250618_100126.h5',
            'models/trained/best_model.h5',
            'models/trained/best_lightweight_model.h5'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                return model, True, model_path
        
        return None, False, None
        
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, False, None

def preprocess_image(image):
    """Preprocess image for model prediction."""
    try:
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Resize to model input size
        image_resized = cv2.resize(image_array, (224, 224))
        
        # Normalize pixel values
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        return image_normalized
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

def predict_freshness_real(image, model, class_names):
    """Real prediction function using trained model."""
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None
        
        processed_image = np.expand_dims(processed_image, axis=0)
        
        start_time = time.time()
        predictions = model.predict(processed_image, verbose=0)
        processing_time = time.time() - start_time
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        if predicted_class.startswith('F_'):
            condition = "Fresh"
            fruit_type = predicted_class[2:]
        else:
            condition = "Spoiled"
            fruit_type = predicted_class[2:]
        
        return {
            'fruit_type': fruit_type,
            'condition': condition,
            'confidence': confidence,
            'all_predictions': predictions[0],
            'processing_time': processing_time,
            'predicted_class': predicted_class
        }
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def predict_freshness_demo(image, class_names):
    """Demo prediction function with simulated results."""
    try:
        fruit_types = list(set([name[2:] for name in class_names]))
        fruit_type = random.choice(fruit_types)
        condition = random.choice(["Fresh", "Spoiled"])
        confidence = random.uniform(0.75, 0.95)
        
        all_predictions = np.random.random(len(class_names))
        all_predictions = all_predictions / all_predictions.sum()
        
        return {
            'fruit_type': fruit_type,
            'condition': condition,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'processing_time': random.uniform(0.1, 0.3),
            'predicted_class': f"{condition[0]}_{fruit_type}"
        }
    except Exception as e:
        st.error(f"Demo prediction error: {e}")
        return None

def main():
    """Main application function."""
    # Load professional CSS
    load_professional_css()
    
    # Load configuration and model
    config, label_mapping, class_names = load_config()
    model, model_available, model_path = load_trained_model()
    
    # Set default class names if not loaded
    if class_names is None or len(class_names) == 0:
        class_names = [
            'F_Banana', 'F_Lemon', 'F_Lulo', 'F_Mango', 'F_Orange', 'F_Strawberry', 'F_Tamarillo', 'F_Tomato',
            'S_Banana', 'S_Lemon', 'S_Lulo', 'S_Mango', 'S_Orange', 'S_Strawberry', 'S_Tamarillo', 'S_Tomato'
        ]
    
    # Create professional header
    if model_available:
        create_header(
            "🍎 FreshHarvest AI System",
            "Professional Fruit Freshness Classification",
            "Production Model: 96.50% Accuracy"
        )
    else:
        create_header(
            "🍎 FreshHarvest AI System",
            "Professional Fruit Freshness Classification",
            "Demo Mode - Production Model: 96.50% Accuracy"
        )
    
    # Create sidebar
    create_sidebar_info(model_available, "96.50%" if model_available else None)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model status indicator
        if model_available:
            create_status_indicator("success", f"🏆 Production Model Active - Using {os.path.basename(model_path)}")
        else:
            create_status_indicator("warning", "🚧 Demo Mode - Production model (96.50% accuracy) not found")
        
        # Performance metrics
        if model_available:
            st.markdown("### 📊 Model Performance Metrics")
            create_metric_cards([
                ("Validation Accuracy", "96.50%", "🎯"),
                ("Precision", "96.85%", "📈"),
                ("Recall", "96.19%", "📊"),
                ("F1-Score", "96.52%", "⚖️")
            ])
        
        # File upload section
        st.markdown("### 📸 Image Upload & Analysis")
        create_upload_area()
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a fruit for freshness analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col_img, col_info = st.columns([1, 1])
            
            with col_img:
                st.image(image, caption="📸 Uploaded Image", use_container_width=True)
            
            with col_info:
                # Image information
                create_info_card(
                    "Image Information",
                    f"""
                    **Filename**: {uploaded_file.name}<br>
                    **Size**: {image.size[0]} × {image.size[1]} pixels<br>
                    **Format**: {image.format}<br>
                    **Mode**: {image.mode}
                    """,
                    "📋"
                )
            
            # Analysis button
            if st.button("🔍 Analyze Fruit Freshness", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the image..."):
                    # Add realistic processing delay
                    time.sleep(1)
                    
                    # Make prediction
                    if model_available and model is not None:
                        result = predict_freshness_real(image, model, class_names)
                        model_info = "Production Model (96.50% Accuracy)"
                    else:
                        result = predict_freshness_demo(image, class_names)
                        model_info = "Demo Mode - Production Model: 96.50% Accuracy"
                    
                    if result:
                        # Display prediction results
                        create_prediction_display(result, model_info)
                        
                        # Confidence gauge
                        st.markdown("### 📊 Confidence Analysis")
                        col_gauge, col_chart = st.columns([1, 2])
                        
                        with col_gauge:
                            fig_gauge = create_confidence_gauge(result['confidence'])
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        with col_chart:
                            if 'all_predictions' in result:
                                fig_chart = create_prediction_chart(result['all_predictions'], class_names)
                                st.plotly_chart(fig_chart, use_container_width=True)
                        
                        # Detailed analysis
                        with st.expander("🔬 Detailed Analysis", expanded=False):
                            col_details1, col_details2 = st.columns(2)
                            
                            with col_details1:
                                st.markdown("**🎯 Prediction Details**")
                                st.write(f"**Predicted Class**: {result['predicted_class']}")
                                st.write(f"**Processing Time**: {result['processing_time']:.3f} seconds")
                                st.write(f"**Model Confidence**: {result['confidence']:.4f}")
                            
                            with col_details2:
                                st.markdown("**💡 Recommendations**")
                                if result['condition'] == "Fresh":
                                    if result['confidence'] > 0.9:
                                        st.success("✅ Excellent quality - Safe for consumption")
                                    elif result['confidence'] > 0.8:
                                        st.success("✅ Good quality - Recommended for use")
                                    else:
                                        st.warning("⚠️ Moderate confidence - Visual inspection recommended")
                                else:
                                    if result['confidence'] > 0.8:
                                        st.error("❌ High confidence spoilage detected - Discard recommended")
                                    else:
                                        st.warning("⚠️ Possible spoilage - Further inspection needed")
    
    with col2:
        # Feature showcase
        create_feature_showcase()
        
        # System information
        st.markdown("### ⚙️ System Information")
        
        system_info = {
            "🏗️ Architecture": "Lightweight CNN",
            "📊 Classes": "16 (8 fruits × 2 conditions)",
            "🖼️ Input Size": "224×224×3 RGB",
            "🧠 Framework": "TensorFlow/Keras",
            "📅 Training Date": "2025-06-18",
            "⚡ Inference Time": "~123ms",
            "💾 Model Size": "~45MB"
        }
        
        for label, value in system_info.items():
            st.markdown(f"**{label}**: {value}")
        
        # Quick stats
        if model_available:
            st.markdown("### 📈 Quick Stats")
            st.metric("Training Epochs", "23", "Optimal early stopping")
            st.metric("Dataset Size", "16,000", "High-quality images")
            st.metric("Validation Split", "20%", "Robust evaluation")
    
    # Footer
    create_footer()

if __name__ == "__main__":
    main()

"""
Simple Streamlit application for FreshHarvest fruit freshness classification demo.
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from pathlib import Path
import sys
import random
import os
import tensorflow as tf

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.cvProject_FreshHarvest.utils.common import read_yaml, read_json

# Configure page
st.set_page_config(
    page_title="FreshHarvest - Fruit Freshness Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .fresh-fruit {
        color: #228B22;
        font-weight: bold;
    }
    .spoiled-fruit {
        color: #DC143C;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_config():
    """Load configuration files."""
    try:
        config = read_yaml("config/config.yaml")
        label_mapping = read_json("data/interim/label_mapping.json")
        class_names = read_json("data/interim/class_names.json")
        return config, label_mapping, class_names
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None, None, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction."""
    try:
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

@st.cache_resource
def load_trained_model():
    """Load the trained model if available."""
    try:
        import tensorflow as tf

        # Try to load the best trained model
        model_paths = [
            'models/trained/best_model.h5',
            'models/trained/best_lightweight_model.h5',
            'models/checkpoints/best_model_20250618_100126.h5'
        ]

        for model_path in model_paths:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                st.success(f"✅ Loaded trained model from {model_path}")
                return model, True

        st.warning("⚠️ No trained model found. Using demo mode.")
        return None, False

    except Exception as e:
        st.warning(f"⚠️ Could not load model: {e}. Using demo mode.")
        return None, False

def predict_freshness_real(image, model, class_names):
    """Real prediction function using trained model."""
    try:
        import time

        # Preprocess image for model
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None

        # Add batch dimension
        processed_image = np.expand_dims(processed_image, axis=0)

        # Make prediction
        start_time = time.time()
        predictions = model.predict(processed_image, verbose=0)
        processing_time = time.time() - start_time

        # Get predicted class
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]

        # Parse fruit type and condition
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
        st.error(f"Error making prediction: {e}")
        return None

def predict_freshness_demo(image, class_names):
    """Demo prediction function (simulated)."""
    try:
        # Simulate model prediction with random results
        # In a real scenario, this would use the trained model

        # Get fruit types from class names
        fruit_types = list(set([name[2:] for name in class_names]))

        # Randomly select a fruit type and condition
        fruit_type = random.choice(fruit_types)
        condition = random.choice(["Fresh", "Spoiled"])
        confidence = random.uniform(0.7, 0.95)

        # Create fake predictions for all classes
        all_predictions = np.random.random(len(class_names))
        all_predictions = all_predictions / all_predictions.sum()  # Normalize

        return {
            'fruit_type': fruit_type,
            'condition': condition,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'processing_time': random.uniform(0.1, 0.5),
            'predicted_class': f"{condition[0]}_{fruit_type}"
        }
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🍎 FreshHarvest Fruit Freshness Classifier</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Upload an image of a fruit to determine if it's fresh or spoiled using AI-powered computer vision.
        </p>
        <p style="color: #ff6b6b; font-weight: bold;" id="model-status">
            🚧 Loading model status...
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load configuration and model
    config, label_mapping, class_names = load_config()
    model, model_available = load_trained_model()

    if config is None:
        st.error("Failed to load configuration. Please check the setup.")
        return

    # Set default class names if not loaded
    if class_names is None:
        class_names = [
            'F_Banana', 'F_Lemon', 'F_Lulo', 'F_Mango', 'F_Orange', 'F_Strawberry', 'F_Tamarillo', 'F_Tomato',
            'S_Banana', 'S_Lemon', 'S_Lulo', 'S_Mango', 'S_Orange', 'S_Strawberry', 'S_Tamarillo', 'S_Tomato'
        ]

    # Display model status
    if model_available:
        st.success("🚀 **REAL MODEL MODE**: Using trained FreshHarvest CNN model for predictions!")
    else:
        st.warning("🚧 **DEMO MODE**: No trained model found. Using simulated predictions for demonstration.")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📊 Project Information</h2>', unsafe_allow_html=True)
        
        st.info(f"""
        **Dataset Statistics:**
        - Total Classes: {config['data']['num_classes']}
        - Fruit Types: 8 (Banana, Lemon, Lulo, Mango, Orange, Strawberry, Tamarillo, Tomato)
        - Conditions: Fresh & Spoiled
        - Image Size: {config['data']['image_size'][0]}x{config['data']['image_size'][1]}
        """)
        
        st.markdown('<h3 class="sub-header">🎯 Supported Fruits</h3>', unsafe_allow_html=True)
        fruits = ["Banana", "Lemon", "Lulo", "Mango", "Orange", "Strawberry", "Tamarillo", "Tomato"]
        for fruit in fruits:
            st.write(f"• {fruit}")
        
        st.markdown('<h3 class="sub-header">🔧 Model Architecture</h3>', unsafe_allow_html=True)
        if model_available and model is not None:
            # Show real model info
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

            st.info(f"""
            **Trained CNN Model:**
            - Total Parameters: {total_params:,}
            - Trainable Parameters: {trainable_params:,}
            - Input Shape: {model.input_shape}
            - Output Classes: {model.output_shape[-1]}
            - Architecture: Lightweight CNN
            """)

            # Check for training history
            if os.path.exists('logs/training.log'):
                st.success("📈 Training logs available")

        else:
            st.info("""
            **CNN Architecture (Demo):**
            - 4 Convolutional Blocks
            - Batch Normalization
            - Dropout Regularization
            - Global Average Pooling
            - Dense Layers with ReLU
            - Softmax Output (16 classes)
            """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a fruit (PNG, JPG, or JPEG format)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")
            st.write(f"**File Size:** {uploaded_file.size} bytes")
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Prediction Results</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                # Preprocess image
                processed_image = preprocess_image(image)
                
                if processed_image is not None:
                    # Make prediction (real model if available, otherwise demo)
                    if model_available and model is not None:
                        result = predict_freshness_real(image, model, class_names)
                    else:
                        result = predict_freshness_demo(processed_image, class_names)
                    
                    if result is not None:
                        # Display results
                        condition_class = "fresh-fruit" if result['condition'] == "Fresh" else "spoiled-fruit"
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🍎 Fruit Type: <span style="color: #2E8B57;">{result['fruit_type']}</span></h3>
                            <h3>📊 Condition: <span class="{condition_class}">{result['condition']}</span></h3>
                            <h3>🎯 Confidence: <span style="color: #4682B4;">{result['confidence']:.2%}</span></h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.progress(result['confidence'])
                        
                        # Recommendation
                        if result['condition'] == "Fresh":
                            st.success("✅ This fruit appears to be fresh and safe to consume!")
                        else:
                            st.warning("⚠️ This fruit appears to be spoiled. Consider discarding it.")
                        
                        # Show top predictions
                        with st.expander("View All Predictions"):
                            pred_df = pd.DataFrame({
                                'Class': class_names,
                                'Confidence': result['all_predictions']
                            }).sort_values('Confidence', ascending=False)
                            
                            st.dataframe(pred_df.head(8), use_container_width=True)
        else:
            st.info("👆 Please upload an image to get started!")
            
            # Show sample images info
            st.markdown("### 📸 Sample Images")
            st.write("Try uploading images of these fruits:")
            
            sample_info = pd.DataFrame({
                'Fruit': ['Banana', 'Lemon', 'Mango', 'Orange', 'Strawberry'],
                'Fresh Indicators': ['Yellow, firm', 'Bright yellow', 'Firm, colorful', 'Bright orange', 'Red, firm'],
                'Spoiled Indicators': ['Brown spots, soft', 'Wrinkled, dark', 'Soft, dark spots', 'Moldy, soft', 'Mushy, dark']
            })
            
            st.dataframe(sample_info, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🚀 Powered by TensorFlow & Streamlit | Built for FreshHarvest Logistics</p>
        <p>⚡ Advanced Computer Vision for Food Quality Assessment</p>
        <p>🔬 End-to-End Machine Learning Pipeline with Data Augmentation & CNN Architecture</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

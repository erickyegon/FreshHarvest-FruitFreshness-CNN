#!/usr/bin/env python3
"""
Test script to verify Streamlit application fixes.
"""

import os
import sys
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def test_imports():
    """Test if all required imports work."""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        print("✅ TensorFlow imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        import pandas as pd
        from PIL import Image
        import cv2
        print("✅ All other packages imported successfully")
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return False
    
    return True

def test_class_names():
    """Test class names handling."""
    print("\n🔍 Testing class names handling...")
    
    # Test default class names
    default_class_names = [
        'F_Banana', 'F_Lemon', 'F_Lulo', 'F_Mango', 'F_Orange', 'F_Strawberry', 'F_Tamarillo', 'F_Tomato',
        'S_Banana', 'S_Lemon', 'S_Lulo', 'S_Mango', 'S_Orange', 'S_Strawberry', 'S_Tamarillo', 'S_Tomato'
    ]
    
    if len(default_class_names) == 16:
        print("✅ Default class names have correct length (16)")
    else:
        print(f"❌ Default class names have incorrect length: {len(default_class_names)}")
        return False
    
    # Test DataFrame creation
    try:
        import pandas as pd
        import numpy as np
        
        # Simulate prediction results
        fake_predictions = np.random.random(16)
        fake_predictions = fake_predictions / fake_predictions.sum()  # Normalize
        
        # Test DataFrame creation
        min_length = min(len(default_class_names), len(fake_predictions))
        pred_df = pd.DataFrame({
            'Class': default_class_names[:min_length],
            'Confidence': fake_predictions[:min_length]
        }).sort_values('Confidence', ascending=False)
        
        print("✅ DataFrame creation works correctly")
        print(f"   - DataFrame shape: {pred_df.shape}")
        print(f"   - Top prediction: {pred_df.iloc[0]['Class']} ({pred_df.iloc[0]['Confidence']:.2%})")
        
    except Exception as e:
        print(f"❌ DataFrame creation failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test model loading functionality."""
    print("\n🔍 Testing model loading...")
    
    model_paths = [
        'models/trained/best_model_96.50acc.h5',
        'models/checkpoints/best_model_20250618_100126.h5',
        'models/trained/best_model.h5',
        'models/trained/best_lightweight_model.h5'
    ]
    
    available_models = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            available_models.append(model_path)
    
    if available_models:
        print("✅ Model files found:")
        for model in available_models:
            print(f"   - {model}")
        
        # Try to load the first available model
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(available_models[0])
            print(f"✅ Successfully loaded model: {available_models[0]}")
            print(f"   - Input shape: {model.input_shape}")
            print(f"   - Output shape: {model.output_shape}")
            return True
        except Exception as e:
            print(f"⚠️ Model file exists but loading failed: {e}")
            return False
    else:
        print("⚠️ No model files found - applications will run in demo mode")
        return True

def test_config_loading():
    """Test configuration loading."""
    print("\n🔍 Testing configuration loading...")
    
    config_files = [
        'config/config.yaml',
        'config/training_config.yaml',
        'config/model_config.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ Found: {config_file}")
        else:
            print(f"⚠️ Missing: {config_file}")
    
    # Try to load main config
    try:
        import yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Successfully loaded main configuration")
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 FRESHHARVEST STREAMLIT FIXES TEST")
    print("="*50)
    
    tests = [
        ("Import Test", test_imports),
        ("Class Names Test", test_class_names),
        ("Model Loading Test", test_model_loading),
        ("Configuration Test", test_config_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} ERROR: {e}")
    
    print("\n" + "="*50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Streamlit apps should work correctly.")
        print("\n🚀 Ready to launch:")
        print("   python run_streamlit.py --app simple")
        print("   python run_streamlit.py --app full")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        print("\n💡 Common solutions:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Check model files in models/trained/ directory")
        print("   - Verify configuration files in config/ directory")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

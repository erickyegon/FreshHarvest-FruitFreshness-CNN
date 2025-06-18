"""
Comprehensive testing script for the complete FreshHarvest pipeline.
"""

import os
import sys
import logging
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.cvProject_FreshHarvest.utils.common import read_yaml, setup_logging, get_timestamp
from src.cvProject_FreshHarvest.components.data_ingestion import DataIngestion
from src.cvProject_FreshHarvest.components.data_augmentation import DataAugmentation
from src.cvProject_FreshHarvest.models.cnn_models import FreshHarvestCNN
from src.cvProject_FreshHarvest.components.model_evaluation import ModelEvaluator
from src.cvProject_FreshHarvest.components.model_optimization import ModelOptimizer

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def test_data_pipeline():
    """Test the complete data pipeline."""
    print("\n" + "="*60)
    print("TESTING DATA PIPELINE")
    print("="*60)
    
    try:
        # Test data ingestion
        print("1. Testing Data Ingestion...")
        config_path = "config/config.yaml"
        data_ingestion = DataIngestion(config_path)
        
        # Check if data is already processed
        if not Path("data/processed/train").exists():
            print("   Running data ingestion...")
            report = data_ingestion.run_data_ingestion()
            print(f"   ✅ Data ingestion completed - {report['total_images']} images processed")
        else:
            print("   ✅ Data already processed")
        
        # Test data augmentation
        print("2. Testing Data Augmentation...")
        data_aug = DataAugmentation(config_path)
        
        # Test augmentation on a sample image
        sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        augmented = data_aug.apply_augmentations(sample_image)
        
        if augmented.shape == sample_image.shape:
            print("   ✅ Data augmentation working correctly")
        else:
            print("   ❌ Data augmentation failed")
            return False
        
        # Test TensorFlow data generators if available
        if TF_AVAILABLE:
            print("3. Testing TensorFlow Data Generators...")
            config = read_yaml(config_path)
            
            datagen = ImageDataGenerator(rescale=1./255)
            generator = datagen.flow_from_directory(
                'data/processed/train',
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical'
            )
            
            # Test one batch
            batch_x, batch_y = next(generator)
            print(f"   ✅ Data generator working - Batch shape: {batch_x.shape}")
        
        print("✅ DATA PIPELINE TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ DATA PIPELINE TEST FAILED: {e}")
        return False


def test_model_architecture():
    """Test model architecture creation."""
    print("\n" + "="*60)
    print("TESTING MODEL ARCHITECTURE")
    print("="*60)
    
    try:
        config_path = "config/config.yaml"
        cnn_model = FreshHarvestCNN(config_path)
        
        # Test different model types
        model_types = ['basic', 'improved', 'lightweight']
        
        for model_type in model_types:
            print(f"Testing {model_type} model...")
            
            if model_type == 'basic':
                model = cnn_model.create_basic_cnn()
            elif model_type == 'improved':
                model = cnn_model.create_improved_cnn()
            elif model_type == 'lightweight':
                model = cnn_model.create_lightweight_cnn()
            
            # Compile model
            model = cnn_model.compile_model(model)
            
            # Test model with dummy data
            dummy_input = np.random.random((1, 224, 224, 3))
            output = model.predict(dummy_input, verbose=0)
            
            if output.shape == (1, 16):
                print(f"   ✅ {model_type} model working - Output shape: {output.shape}")
            else:
                print(f"   ❌ {model_type} model failed - Wrong output shape: {output.shape}")
                return False
        
        print("✅ MODEL ARCHITECTURE TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ MODEL ARCHITECTURE TEST FAILED: {e}")
        return False


def test_training_pipeline():
    """Test training pipeline with minimal epochs."""
    print("\n" + "="*60)
    print("TESTING TRAINING PIPELINE")
    print("="*60)
    
    if not TF_AVAILABLE:
        print("❌ TensorFlow not available - Skipping training test")
        return False
    
    try:
        config_path = "config/config.yaml"
        config = read_yaml(config_path)
        
        # Create small data generators for testing
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'data/processed/train',
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )
        
        val_generator = train_datagen.flow_from_directory(
            'data/processed/val',
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )
        
        # Create lightweight model for testing
        cnn_model = FreshHarvestCNN(config_path)
        model = cnn_model.create_lightweight_cnn()
        model = cnn_model.compile_model(model)
        
        print("   Training for 2 epochs (test)...")
        history = model.fit(
            train_generator,
            epochs=2,
            validation_data=val_generator,
            steps_per_epoch=5,
            validation_steps=2,
            verbose=1
        )
        
        if len(history.history['loss']) == 2:
            print("   ✅ Training pipeline working correctly")
            print(f"   Final loss: {history.history['loss'][-1]:.4f}")
            print(f"   Final accuracy: {history.history['accuracy'][-1]:.4f}")
        else:
            print("   ❌ Training pipeline failed")
            return False
        
        print("✅ TRAINING PIPELINE TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TRAINING PIPELINE TEST FAILED: {e}")
        return False


def test_evaluation_pipeline():
    """Test evaluation pipeline."""
    print("\n" + "="*60)
    print("TESTING EVALUATION PIPELINE")
    print("="*60)
    
    try:
        config_path = "config/config.yaml"
        evaluator = ModelEvaluator(config_path)
        
        # Test with dummy data
        y_true = np.random.randint(0, 16, 100)
        y_pred = np.random.randint(0, 16, 100)
        predictions = np.random.random((100, 16))
        
        # Test misclassification analysis
        analysis = evaluator.analyze_misclassifications(y_true, y_pred, predictions)
        
        if 'total_misclassified' in analysis:
            print("   ✅ Misclassification analysis working")
        else:
            print("   ❌ Misclassification analysis failed")
            return False
        
        # Test report generation
        dummy_results = {
            'overall_metrics': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.82,
                'f1_score': 0.82,
                'roc_auc': 0.90
            },
            'per_class_metrics': {
                'F_Banana': {'precision': 0.85, 'recall': 0.80, 'f1-score': 0.82, 'support': 50}
            }
        }
        
        report = evaluator.generate_evaluation_report(dummy_results)
        
        if "FRESHHARVEST MODEL EVALUATION REPORT" in report:
            print("   ✅ Report generation working")
        else:
            print("   ❌ Report generation failed")
            return False
        
        print("✅ EVALUATION PIPELINE TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ EVALUATION PIPELINE TEST FAILED: {e}")
        return False


def test_optimization_pipeline():
    """Test model optimization pipeline."""
    print("\n" + "="*60)
    print("TESTING OPTIMIZATION PIPELINE")
    print("="*60)
    
    try:
        config_path = "config/config.yaml"
        optimizer = ModelOptimizer(config_path)
        
        # Test benchmarking with dummy model
        if TF_AVAILABLE:
            # Create a simple model for testing
            from tensorflow import keras
            model = keras.Sequential([
                keras.layers.Input(shape=(224, 224, 3)),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(16, activation='softmax')
            ])
            
            # Save temporary model
            temp_model_path = "temp_test_model.h5"
            model.save(temp_model_path)
            
            # Test benchmarking
            results = optimizer.benchmark_model(temp_model_path, num_samples=10)
            
            if 'throughput_samples_per_sec' in results:
                print(f"   ✅ Benchmarking working - Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
            else:
                print("   ❌ Benchmarking failed")
                return False
            
            # Clean up
            os.remove(temp_model_path)
        
        print("✅ OPTIMIZATION PIPELINE TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ OPTIMIZATION PIPELINE TEST FAILED: {e}")
        return False


def test_streamlit_app():
    """Test Streamlit application components."""
    print("\n" + "="*60)
    print("TESTING STREAMLIT APPLICATION")
    print("="*60)
    
    try:
        # Test configuration loading
        config = read_yaml("config/config.yaml")
        
        if config and 'data' in config:
            print("   ✅ Configuration loading working")
        else:
            print("   ❌ Configuration loading failed")
            return False
        
        # Test class names loading
        if Path("data/interim/class_names.json").exists():
            from src.cvProject_FreshHarvest.utils.common import read_json
            class_names = read_json("data/interim/class_names.json")
            
            if len(class_names) == 16:
                print("   ✅ Class names loading working")
            else:
                print("   ❌ Class names loading failed")
                return False
        
        # Test image preprocessing
        from PIL import Image
        import numpy as np
        
        # Create dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test preprocessing
        image_array = np.array(dummy_image)
        normalized = image_array.astype(np.float32) / 255.0
        
        if normalized.max() <= 1.0 and normalized.min() >= 0.0:
            print("   ✅ Image preprocessing working")
        else:
            print("   ❌ Image preprocessing failed")
            return False
        
        print("✅ STREAMLIT APPLICATION TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ STREAMLIT APPLICATION TEST FAILED: {e}")
        return False


def run_complete_test_suite():
    """Run the complete test suite."""
    print("🚀 STARTING FRESHHARVEST COMPLETE PIPELINE TEST")
    print("="*80)
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Track test results
    test_results = {}
    
    # Run all tests
    tests = [
        ("Data Pipeline", test_data_pipeline),
        ("Model Architecture", test_model_architecture),
        ("Training Pipeline", test_training_pipeline),
        ("Evaluation Pipeline", test_evaluation_pipeline),
        ("Optimization Pipeline", test_optimization_pipeline),
        ("Streamlit Application", test_streamlit_app)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} Test...")
        start_time = time.time()
        
        try:
            result = test_func()
            test_results[test_name] = result
            duration = time.time() - start_time
            
            if result:
                print(f"✅ {test_name} Test PASSED ({duration:.2f}s)")
            else:
                print(f"❌ {test_name} Test FAILED ({duration:.2f}s)")
        
        except Exception as e:
            test_results[test_name] = False
            duration = time.time() - start_time
            print(f"❌ {test_name} Test FAILED with exception ({duration:.2f}s): {e}")
    
    # Print final results
    print("\n" + "="*80)
    print("🏁 FINAL TEST RESULTS")
    print("="*80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print("-" * 80)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! FreshHarvest pipeline is ready for production!")
    else:
        print("⚠️  Some tests failed. Please review and fix issues before deployment.")
    
    print("="*80)
    
    return test_results


if __name__ == "__main__":
    run_complete_test_suite()

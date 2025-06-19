#!/bin/bash

# FreshHarvest Model Training Script
# =================================
#
# This script handles the complete training pipeline for the FreshHarvest
# fruit freshness classification model.
#
# Author: FreshHarvest Team
# Version: 1.0.0
# Last Updated: 2025-06-18

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="FreshHarvest"
MODEL_VERSION="1.0.0"
TARGET_ACCURACY=0.95
EPOCHS=${1:-50}
BATCH_SIZE=${2:-32}
LEARNING_RATE=${3:-0.001}

# Paths
DATA_DIR="data/raw/fruit_dataset"
ARTIFACTS_DIR="artifacts"
MODEL_DIR="$ARTIFACTS_DIR/model_trainer"
LOGS_DIR="logs"

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking training prerequisites..."

    # Check Python environment
    if ! command -v python &> /dev/null; then
        error "Python is not installed or not in PATH"
        exit 1
    fi

    # Check required packages
    python -c "import tensorflow, numpy, pandas, sklearn, matplotlib" 2>/dev/null || {
        error "Required Python packages not installed"
        error "Please run: ./scripts/setup_environment.sh"
        exit 1
    }

    # Check data availability
    if [ ! -d "$DATA_DIR" ]; then
        error "Training data not found at $DATA_DIR"
        error "Please run: ./scripts/download_data.sh"
        exit 1
    fi

    # Check data structure
    local required_classes=("fresh_apple" "fresh_banana" "fresh_orange" "rotten_apple" "rotten_banana" "rotten_orange")
    for class_name in "${required_classes[@]}"; do
        if [ ! -d "$DATA_DIR/$class_name" ]; then
            error "Missing class directory: $class_name"
            exit 1
        fi

        local image_count=$(find "$DATA_DIR/$class_name" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
        if [ "$image_count" -eq 0 ]; then
            error "No images found in $class_name directory"
            exit 1
        fi
    done

    # Check GPU availability
    python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))" || true

    success "Prerequisites check passed"
}

# Function to prepare training environment
prepare_training() {
    log "Preparing training environment..."

    # Create necessary directories
    mkdir -p "$ARTIFACTS_DIR"
    mkdir -p "$MODEL_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "reports"

    # Create training configuration
    cat > "$MODEL_DIR/training_config.json" << EOF
{
    "model_version": "$MODEL_VERSION",
    "training_date": "$(date -Iseconds)",
    "parameters": {
        "epochs": $EPOCHS,
        "batch_size": $BATCH_SIZE,
        "learning_rate": $LEARNING_RATE,
        "target_accuracy": $TARGET_ACCURACY
    },
    "data_path": "$DATA_DIR",
    "model_save_path": "$MODEL_DIR/model.h5"
}
EOF

    success "Training environment prepared"
}

# Function to run data validation
validate_data() {
    log "Running data validation..."

    python -c "
import sys
sys.path.append('src')
from cvProject_FreshHarvest.components.data_validation import DataValidator

try:
    validator = DataValidator('config/config.yaml')
    validation_results = validator.validate_dataset('$DATA_DIR')

    if validation_results.get('is_valid', False):
        print('✅ Data validation passed')
        print(f'Total images: {validation_results.get(\"total_images\", 0)}')
        print(f'Classes found: {len(validation_results.get(\"class_distribution\", {}))}')
    else:
        print('❌ Data validation failed')
        for error in validation_results.get('errors', []):
            print(f'  - {error}')
        sys.exit(1)

except Exception as e:
    print(f'❌ Data validation error: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        success "Data validation completed"
    else
        error "Data validation failed"
        exit 1
    fi
}

# Function to run training
run_training() {
    log "Starting model training..."
    log "Configuration: Epochs=$EPOCHS, Batch Size=$BATCH_SIZE, Learning Rate=$LEARNING_RATE"

    # Start training with progress monitoring
    python -c "
import sys
import os
import json
import time
from datetime import datetime
sys.path.append('src')

try:
    from cvProject_FreshHarvest.components.data_preprocessing import DataPreprocessor
    from cvProject_FreshHarvest.components.model_trainer import ModelTrainer
    from cvProject_FreshHarvest.utils.common import read_yaml

    print('🍎 FreshHarvest Model Training')
    print('=' * 40)
    print()

    # Load configuration
    config = read_yaml('config/config.yaml')

    # Update training parameters
    config['training']['epochs'] = $EPOCHS
    config['training']['batch_size'] = $BATCH_SIZE
    config['training']['learning_rate'] = $LEARNING_RATE

    # Initialize components
    print('📊 Initializing data preprocessor...')
    preprocessor = DataPreprocessor('config/config.yaml')

    print('🏗️ Loading and preprocessing data...')
    train_data, val_data, test_data = preprocessor.create_data_generators('$DATA_DIR')

    if train_data is None:
        print('❌ Failed to load training data')
        sys.exit(1)

    print(f'✅ Data loaded successfully')
    print(f'   Training batches: {len(train_data)}')
    print(f'   Validation batches: {len(val_data)}')
    print(f'   Test batches: {len(test_data)}')
    print()

    # Initialize trainer
    print('🏋️ Initializing model trainer...')
    trainer = ModelTrainer('config/config.yaml')

    # Update trainer parameters
    trainer.epochs = $EPOCHS
    trainer.batch_size = $BATCH_SIZE
    trainer.learning_rate = $LEARNING_RATE

    # Start training
    print('🚀 Starting training process...')
    print(f'Target accuracy: {$TARGET_ACCURACY * 100:.1f}%')
    print()

    start_time = time.time()

    training_results = trainer.train_model(
        train_data,
        val_data,
        '$MODEL_DIR/model.h5'
    )

    training_time = time.time() - start_time

    if training_results['success']:
        print()
        print('🎉 Training completed successfully!')
        print(f'⏱️ Training time: {training_time/3600:.2f} hours')

        metadata = training_results['metadata']
        print(f'📊 Best validation accuracy: {metadata[\"best_val_accuracy\"]:.4f}')
        print(f'📈 Total epochs: {metadata[\"total_epochs\"]}')
        print(f'⚡ Early stopping: {\"Yes\" if metadata[\"early_stopping\"] else \"No\"}')

        # Check if target accuracy was reached
        if metadata['best_val_accuracy'] >= $TARGET_ACCURACY:
            print(f'✅ Target accuracy ({$TARGET_ACCURACY * 100:.1f}%) achieved!')
        else:
            print(f'⚠️ Target accuracy ({$TARGET_ACCURACY * 100:.1f}%) not reached')

        # Evaluate on test data
        print()
        print('📊 Evaluating on test data...')
        eval_results = trainer.evaluate_model(test_data)

        if eval_results:
            print(f'🎯 Test accuracy: {eval_results[\"test_accuracy\"]:.4f}')
            print(f'📏 Precision: {eval_results[\"precision\"]:.4f}')
            print(f'🔍 Recall: {eval_results[\"recall\"]:.4f}')
            print(f'⚖️ F1-score: {eval_results[\"f1_score\"]:.4f}')

        # Save final results
        final_results = {
            'training_results': training_results,
            'evaluation_results': eval_results,
            'training_time_hours': training_time / 3600,
            'target_achieved': metadata['best_val_accuracy'] >= $TARGET_ACCURACY,
            'completion_date': datetime.now().isoformat()
        }

        with open('$MODEL_DIR/final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)

        print()
        print('💾 Model and results saved to: $MODEL_DIR')

    else:
        print('❌ Training failed!')
        print(f'Error: {training_results.get(\"error\", \"Unknown error\")}')
        sys.exit(1)

except Exception as e:
    print(f'❌ Training error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

    local training_exit_code=$?

    if [ $training_exit_code -eq 0 ]; then
        success "Model training completed successfully"
    else
        error "Model training failed"
        exit 1
    fi
}

# Function to generate training report
generate_report() {
    log "Generating training report..."

    if [ -f "scripts/generate_reports.py" ]; then
        python scripts/generate_reports.py --type training
        success "Training report generated"
    else
        warning "Report generator not found, skipping report generation"
    fi
}

# Function to validate trained model
validate_model() {
    log "Validating trained model..."

    python -c "
import sys
import os
sys.path.append('src')

try:
    from cvProject_FreshHarvest.components.model_deployment import ModelDeployment
    import numpy as np

    # Test model loading
    deployment = ModelDeployment('config/config.yaml')
    model_loaded = deployment.load_model('$MODEL_DIR/model.h5')

    if not model_loaded:
        print('❌ Model loading failed')
        sys.exit(1)

    print('✅ Model loaded successfully')

    # Test prediction
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    result = deployment.predict(test_image)

    if 'error' in result:
        print(f'❌ Prediction test failed: {result[\"error\"]}')
        sys.exit(1)

    print('✅ Prediction test passed')
    print(f'   Predicted class: {result[\"predicted_class\"]}')
    print(f'   Confidence: {result[\"confidence\"]:.3f}')
    print(f'   Inference time: {result[\"inference_time_ms\"]:.2f}ms')

    # Check model file size
    model_size = os.path.getsize('$MODEL_DIR/model.h5') / (1024 * 1024)  # MB
    print(f'📁 Model size: {model_size:.2f} MB')

    if model_size > 100:  # 100MB threshold
        print('⚠️ Large model size detected')

    print('✅ Model validation completed')

except Exception as e:
    print(f'❌ Model validation error: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        success "Model validation completed"
    else
        error "Model validation failed"
        exit 1
    fi
}

# Function to create training summary
create_summary() {
    log "Creating training summary..."

    local summary_file="$LOGS_DIR/training_summary_$(date +%Y%m%d_%H%M%S).txt"

    cat > "$summary_file" << EOF
FreshHarvest Model Training Summary
==================================

Training Information:
- Date: $(date)
- Model Version: $MODEL_VERSION
- Target Accuracy: $(echo "$TARGET_ACCURACY * 100" | bc)%

Training Parameters:
- Epochs: $EPOCHS
- Batch Size: $BATCH_SIZE
- Learning Rate: $LEARNING_RATE

Data Information:
- Data Path: $DATA_DIR
- Classes: 6 (Fresh/Rotten: Apple, Banana, Orange)

Model Information:
- Model Path: $MODEL_DIR/model.h5
- Model Size: $(du -h "$MODEL_DIR/model.h5" 2>/dev/null | cut -f1 || echo "Unknown")

Training Results:
$(cat "$MODEL_DIR/final_results.json" 2>/dev/null | python -m json.tool 2>/dev/null || echo "Results not available")

System Information:
- OS: $(uname -s)
- Python Version: $(python --version)
- TensorFlow Version: $(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "Unknown")

Next Steps:
1. Review training results and model performance
2. Run model evaluation: python -m src.cvProject_FreshHarvest.components.model_evaluation
3. Deploy model: ./scripts/deploy_model.sh
4. Set up monitoring: python scripts/model_monitoring.py --action monitor

EOF

    success "Training summary created: $summary_file"
}

# Main training function
main() {
    echo "🍎 FreshHarvest Model Training Script"
    echo "====================================="
    echo ""

    log "Starting training pipeline..."
    log "Target: Achieve $TARGET_ACCURACY accuracy with 96.50% validation performance"

    # Check prerequisites
    check_prerequisites

    # Prepare training environment
    prepare_training

    # Validate data
    validate_data

    # Run training
    run_training

    # Validate trained model
    validate_model

    # Generate report
    generate_report

    # Create summary
    create_summary

    echo ""
    success "🎉 FreshHarvest model training completed successfully!"
    echo ""
    echo "📊 Model Performance: Targeting 96.50% validation accuracy"
    echo "💾 Model Location: $MODEL_DIR/model.h5"
    echo "📋 Training Logs: $LOGS_DIR/"
    echo "📊 Reports: reports/"
    echo ""
    echo "Next steps:"
    echo "1. Review training results in: $MODEL_DIR/final_results.json"
    echo "2. Test the model: python -m src.cvProject_FreshHarvest.components.model_evaluation"
    echo "3. Deploy the model: ./scripts/deploy_model.sh"
    echo "4. Start monitoring: python scripts/model_monitoring.py --action monitor"
    echo ""
}

# Script usage information
usage() {
    echo "Usage: $0 [EPOCHS] [BATCH_SIZE] [LEARNING_RATE]"
    echo ""
    echo "Parameters:"
    echo "  EPOCHS        Number of training epochs (default: 50)"
    echo "  BATCH_SIZE    Training batch size (default: 32)"
    echo "  LEARNING_RATE Learning rate (default: 0.001)"
    echo ""
    echo "Examples:"
    echo "  $0              # Use default parameters"
    echo "  $0 30           # Train for 30 epochs"
    echo "  $0 30 64        # 30 epochs, batch size 64"
    echo "  $0 30 64 0.01   # 30 epochs, batch size 64, learning rate 0.01"
    echo ""
    echo "Target: Achieve 96.50% validation accuracy"
    echo ""
}

# Handle script arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

# Run main function
main "$@"
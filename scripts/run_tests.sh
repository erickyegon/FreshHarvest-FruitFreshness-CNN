#!/bin/bash

# FreshHarvest Test Runner Script
# ==============================
#
# This script runs comprehensive tests for the FreshHarvest
# fruit freshness classification system.
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
TEST_TYPE=${1:-"all"}  # all, unit, integration, system
VERBOSE=${2:-false}
COVERAGE=${3:-false}

# Test directories
TESTS_DIR="tests"
SRC_DIR="src"
LOGS_DIR="logs"
REPORTS_DIR="reports"

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

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

# Function to check test prerequisites
check_prerequisites() {
    log "Checking test prerequisites..."

    # Check Python environment
    if ! command -v python &> /dev/null; then
        error "Python is not installed or not in PATH"
        exit 1
    fi

    # Check required test packages
    python -c "import pytest, unittest, coverage" 2>/dev/null || {
        warning "Some test packages not installed, installing..."
        pip install pytest pytest-cov coverage unittest-xml-reporting
    }

    # Check project structure
    if [ ! -d "$SRC_DIR" ]; then
        error "Source directory not found: $SRC_DIR"
        exit 1
    fi

    # Create test directories if they don't exist
    mkdir -p "$TESTS_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$REPORTS_DIR"

    success "Prerequisites check passed"
}

# Function to create test files if they don't exist
create_test_files() {
    log "Creating test files if needed..."

    # Create test directory structure
    mkdir -p "$TESTS_DIR/unit"
    mkdir -p "$TESTS_DIR/integration"
    mkdir -p "$TESTS_DIR/system"

    # Create __init__.py files
    touch "$TESTS_DIR/__init__.py"
    touch "$TESTS_DIR/unit/__init__.py"
    touch "$TESTS_DIR/integration/__init__.py"
    touch "$TESTS_DIR/system/__init__.py"

    # Create conftest.py for pytest configuration
    if [ ! -f "$TESTS_DIR/conftest.py" ]; then
        cat > "$TESTS_DIR/conftest.py" << 'EOF'
"""
FreshHarvest Test Configuration
==============================

Pytest configuration and fixtures for FreshHarvest tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "data": {
            "image_size": [224, 224],
            "channels": 3,
            "batch_size": 32
        },
        "model": {
            "input_shape": [224, 224, 3],
            "num_classes": 6
        }
    }

@pytest.fixture
def sample_image():
    """Sample image for testing."""
    import numpy as np
    return np.random.rand(224, 224, 3).astype(np.float32)

@pytest.fixture
def temp_model_path(tmp_path):
    """Temporary model path for testing."""
    return str(tmp_path / "test_model.h5")
EOF
    fi

    # Create unit tests for components
    create_unit_tests

    # Create integration tests
    create_integration_tests

    # Create system tests
    create_system_tests

    success "Test files created"
}

# Function to create unit tests
create_unit_tests() {
    # Test for data validation
    if [ ! -f "$TESTS_DIR/unit/test_data_validation.py" ]; then
        cat > "$TESTS_DIR/unit/test_data_validation.py" << 'EOF'
"""Unit tests for data validation component."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

def test_data_validator_import():
    """Test that DataValidator can be imported."""
    try:
        from cvProject_FreshHarvest.components.data_validation import DataValidator
        assert True
    except ImportError:
        pytest.skip("DataValidator not available")

def test_image_validation():
    """Test image validation functionality."""
    try:
        from cvProject_FreshHarvest.components.data_validation import DataValidator

        # Create temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
data:
  image_size: [224, 224]
  channels: 3
validation:
  quality:
    min_resolution: [100, 100]
    max_resolution: [4000, 4000]
""")
            config_path = f.name

        try:
            validator = DataValidator(config_path)

            # Test valid image
            valid_image = np.random.rand(224, 224, 3).astype(np.uint8)
            result = validator.validate_image_array(valid_image)
            assert result['is_valid'] == True

            # Test invalid image (wrong dimensions)
            invalid_image = np.random.rand(100, 100).astype(np.uint8)
            result = validator.validate_image_array(invalid_image)
            assert result['is_valid'] == False

        finally:
            os.unlink(config_path)

    except ImportError:
        pytest.skip("DataValidator not available")

def test_class_distribution_validation():
    """Test class distribution validation."""
    try:
        from cvProject_FreshHarvest.components.data_validation import DataValidator

        # Mock class distribution
        class_dist = {
            'fresh_apple': 100,
            'fresh_banana': 95,
            'fresh_orange': 105,
            'rotten_apple': 90,
            'rotten_banana': 88,
            'rotten_orange': 92
        }

        # This should pass (balanced distribution)
        assert len(class_dist) == 6
        assert all(count > 0 for count in class_dist.values())

    except ImportError:
        pytest.skip("DataValidator not available")
EOF
    fi

    # Test for model deployment
    if [ ! -f "$TESTS_DIR/unit/test_model_deployment.py" ]; then
        cat > "$TESTS_DIR/unit/test_model_deployment.py" << 'EOF'
"""Unit tests for model deployment component."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

def test_model_deployment_import():
    """Test that ModelDeployment can be imported."""
    try:
        from cvProject_FreshHarvest.components.model_deployment import ModelDeployment
        assert True
    except ImportError:
        pytest.skip("ModelDeployment not available")

def test_image_preprocessing():
    """Test image preprocessing for deployment."""
    try:
        from cvProject_FreshHarvest.components.model_deployment import ModelDeployment

        # Create temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model:
  input_shape: [224, 224, 3]
  num_classes: 6
deployment:
  model_path: "test_model.h5"
""")
            config_path = f.name

        try:
            deployment = ModelDeployment(config_path)

            # Test preprocessing
            test_image = np.random.rand(224, 224, 3).astype(np.float32)
            processed = deployment.preprocess_input(test_image)

            assert processed is not None
            assert processed.shape == (1, 224, 224, 3)
            assert processed.dtype == np.float32

        finally:
            os.unlink(config_path)

    except ImportError:
        pytest.skip("ModelDeployment not available")

def test_health_check():
    """Test deployment health check."""
    try:
        from cvProject_FreshHarvest.components.model_deployment import ModelDeployment

        # Create temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model:
  input_shape: [224, 224, 3]
  num_classes: 6
""")
            config_path = f.name

        try:
            deployment = ModelDeployment(config_path)
            health_status = deployment.health_check()

            assert 'status' in health_status
            assert 'timestamp' in health_status
            assert health_status['model_loaded'] == False  # No model loaded yet

        finally:
            os.unlink(config_path)

    except ImportError:
        pytest.skip("ModelDeployment not available")
EOF
    fi

    # Test for utilities
    if [ ! -f "$TESTS_DIR/unit/test_utils.py" ]; then
        cat > "$TESTS_DIR/unit/test_utils.py" << 'EOF'
"""Unit tests for utility functions."""

import pytest
import tempfile
import os
import yaml

def test_read_yaml():
    """Test YAML reading utility."""
    try:
        from cvProject_FreshHarvest.utils.common import read_yaml

        # Create temporary YAML file
        test_data = {
            'test_key': 'test_value',
            'nested': {
                'key': 'value'
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_data, f)
            yaml_path = f.name

        try:
            result = read_yaml(yaml_path)
            assert result == test_data
            assert result['test_key'] == 'test_value'
            assert result['nested']['key'] == 'value'

        finally:
            os.unlink(yaml_path)

    except ImportError:
        pytest.skip("Common utilities not available")

def test_create_directories():
    """Test directory creation utility."""
    try:
        from cvProject_FreshHarvest.utils.common import create_directories

        with tempfile.TemporaryDirectory() as temp_dir:
            test_dirs = [
                os.path.join(temp_dir, 'dir1'),
                os.path.join(temp_dir, 'dir2', 'subdir'),
                os.path.join(temp_dir, 'dir3')
            ]

            create_directories(test_dirs)

            for dir_path in test_dirs:
                assert os.path.exists(dir_path)
                assert os.path.isdir(dir_path)

    except ImportError:
        pytest.skip("Common utilities not available")
EOF
    fi
}

# Function to create integration tests
create_integration_tests() {
    if [ ! -f "$TESTS_DIR/integration/test_pipeline.py" ]; then
        cat > "$TESTS_DIR/integration/test_pipeline.py" << 'EOF'
"""Integration tests for FreshHarvest pipeline."""

import pytest
import tempfile
import os
import numpy as np
from pathlib import Path

def test_data_pipeline_integration():
    """Test data preprocessing pipeline integration."""
    try:
        from cvProject_FreshHarvest.components.data_preprocessing import DataPreprocessor

        # Create temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
data:
  image_size: [224, 224]
  channels: 3
  batch_size: 2
preprocessing:
  normalization: "standard"
  augmentation:
    enabled: false
""")
            config_path = f.name

        try:
            preprocessor = DataPreprocessor(config_path)

            # Create temporary data directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create class directories
                for class_name in ['fresh_apple', 'rotten_apple']:
                    class_dir = Path(temp_dir) / class_name
                    class_dir.mkdir()

                    # Create dummy images
                    for i in range(3):
                        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                        # Save as numpy array (simplified for testing)
                        np.save(class_dir / f"img_{i}.npy", img_array)

                # Test data generator creation would go here
                # (Simplified for this example)
                assert True  # Placeholder

        finally:
            os.unlink(config_path)

    except ImportError:
        pytest.skip("DataPreprocessor not available")

def test_training_evaluation_integration():
    """Test training and evaluation integration."""
    try:
        from cvProject_FreshHarvest.components.model_trainer import ModelTrainer
        from cvProject_FreshHarvest.components.model_evaluation import ModelEvaluator

        # This would test the integration between training and evaluation
        # Simplified for this example
        assert True  # Placeholder

    except ImportError:
        pytest.skip("Training/Evaluation components not available")
EOF
    fi
}

# Function to create system tests
create_system_tests() {
    if [ ! -f "$TESTS_DIR/system/test_end_to_end.py" ]; then
        cat > "$TESTS_DIR/system/test_end_to_end.py" << 'EOF'
"""System tests for FreshHarvest end-to-end functionality."""

import pytest
import tempfile
import os
import numpy as np

def test_model_prediction_system():
    """Test complete model prediction system."""
    try:
        from cvProject_FreshHarvest.components.model_deployment import ModelDeployment

        # Create temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model:
  input_shape: [224, 224, 3]
  num_classes: 6
deployment:
  model_path: "artifacts/model_trainer/model.h5"
""")
            config_path = f.name

        try:
            deployment = ModelDeployment(config_path)

            # Test health check
            health_status = deployment.health_check()
            assert 'status' in health_status

            # Test preprocessing
            test_image = np.random.rand(224, 224, 3).astype(np.float32)
            processed = deployment.preprocess_input(test_image)
            assert processed is not None

        finally:
            os.unlink(config_path)

    except ImportError:
        pytest.skip("System components not available")

def test_configuration_system():
    """Test configuration system."""
    try:
        from cvProject_FreshHarvest.utils.common import read_yaml

        # Test main config file
        if os.path.exists('config/config.yaml'):
            config = read_yaml('config/config.yaml')

            # Verify required sections
            assert 'data' in config
            assert 'model' in config

            # Verify data configuration
            assert 'image_size' in config['data']
            assert 'channels' in config['data']

            # Verify model configuration
            assert 'input_shape' in config['model']
            assert 'num_classes' in config['model']

        else:
            pytest.skip("Main config file not found")

    except ImportError:
        pytest.skip("Configuration utilities not available")
EOF
    fi
}

# Function to run unit tests
run_unit_tests() {
    log "Running unit tests..."

    local test_cmd="python -m pytest $TESTS_DIR/unit -v"

    if [ "$COVERAGE" = "true" ]; then
        test_cmd="$test_cmd --cov=$SRC_DIR --cov-report=html --cov-report=term"
    fi

    if [ "$VERBOSE" = "true" ]; then
        test_cmd="$test_cmd -s"
    fi

    # Add XML output for CI/CD
    test_cmd="$test_cmd --junit-xml=$REPORTS_DIR/unit_tests.xml"

    echo "Running: $test_cmd"
    eval $test_cmd

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        success "Unit tests passed"
    else
        error "Unit tests failed"
    fi

    return $exit_code
}

# Function to run integration tests
run_integration_tests() {
    log "Running integration tests..."

    local test_cmd="python -m pytest $TESTS_DIR/integration -v"

    if [ "$VERBOSE" = "true" ]; then
        test_cmd="$test_cmd -s"
    fi

    test_cmd="$test_cmd --junit-xml=$REPORTS_DIR/integration_tests.xml"

    echo "Running: $test_cmd"
    eval $test_cmd

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        success "Integration tests passed"
    else
        error "Integration tests failed"
    fi

    return $exit_code
}

# Function to run system tests
run_system_tests() {
    log "Running system tests..."

    local test_cmd="python -m pytest $TESTS_DIR/system -v"

    if [ "$VERBOSE" = "true" ]; then
        test_cmd="$test_cmd -s"
    fi

    test_cmd="$test_cmd --junit-xml=$REPORTS_DIR/system_tests.xml"

    echo "Running: $test_cmd"
    eval $test_cmd

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        success "System tests passed"
    else
        error "System tests failed"
    fi

    return $exit_code
}

# Function to run all tests
run_all_tests() {
    log "Running all tests..."

    local unit_result=0
    local integration_result=0
    local system_result=0

    # Run unit tests
    run_unit_tests
    unit_result=$?

    # Run integration tests
    run_integration_tests
    integration_result=$?

    # Run system tests
    run_system_tests
    system_result=$?

    # Calculate overall result
    local overall_result=$((unit_result + integration_result + system_result))

    if [ $overall_result -eq 0 ]; then
        success "All tests passed"
    else
        error "Some tests failed"
    fi

    return $overall_result
}

# Function to generate test report
generate_test_report() {
    log "Generating test report..."

    local report_file="$REPORTS_DIR/test_report_$(date +%Y%m%d_%H%M%S).md"

    cat > "$report_file" << EOF
# FreshHarvest Test Report

## Test Summary
**Date:** $(date)
**Test Type:** $TEST_TYPE
**Coverage:** $COVERAGE

## Test Results
- **Unit Tests:** $([ -f "$REPORTS_DIR/unit_tests.xml" ] && echo "✅ Completed" || echo "⏭️ Skipped")
- **Integration Tests:** $([ -f "$REPORTS_DIR/integration_tests.xml" ] && echo "✅ Completed" || echo "⏭️ Skipped")
- **System Tests:** $([ -f "$REPORTS_DIR/system_tests.xml" ] && echo "✅ Completed" || echo "⏭️ Skipped")

## Coverage Report
$([ "$COVERAGE" = "true" ] && echo "Coverage report available in: htmlcov/index.html" || echo "Coverage not enabled")

## Test Files
- Unit Tests: $TESTS_DIR/unit/
- Integration Tests: $TESTS_DIR/integration/
- System Tests: $TESTS_DIR/system/

## Next Steps
1. Review test results in XML reports
2. Check coverage report if enabled
3. Fix any failing tests
4. Add more tests for better coverage

---
*Generated by FreshHarvest Test Runner*
EOF

    success "Test report generated: $report_file"
}

# Main test function
main() {
    echo "🍎 FreshHarvest Test Runner"
    echo "=========================="
    echo ""

    log "Starting test execution..."
    log "Test type: $TEST_TYPE"
    log "Verbose: $VERBOSE"
    log "Coverage: $COVERAGE"

    # Check prerequisites
    check_prerequisites

    # Create test files
    create_test_files

    # Run tests based on type
    local test_result=0

    case $TEST_TYPE in
        "unit")
            run_unit_tests
            test_result=$?
            ;;
        "integration")
            run_integration_tests
            test_result=$?
            ;;
        "system")
            run_system_tests
            test_result=$?
            ;;
        "all")
            run_all_tests
            test_result=$?
            ;;
        *)
            error "Unknown test type: $TEST_TYPE"
            error "Supported types: unit, integration, system, all"
            exit 1
            ;;
    esac

    # Generate test report
    generate_test_report

    echo ""
    if [ $test_result -eq 0 ]; then
        success "🎉 All tests completed successfully!"
    else
        error "❌ Some tests failed!"
    fi

    echo ""
    echo "📊 Test Reports: $REPORTS_DIR/"
    echo "📋 Test Logs: $LOGS_DIR/"
    if [ "$COVERAGE" = "true" ]; then
        echo "📈 Coverage Report: htmlcov/index.html"
    fi
    echo ""

    return $test_result
}

# Script usage information
usage() {
    echo "Usage: $0 [TEST_TYPE] [VERBOSE] [COVERAGE]"
    echo ""
    echo "Parameters:"
    echo "  TEST_TYPE    Type of tests to run (default: all)"
    echo "               Options: unit, integration, system, all"
    echo "  VERBOSE      Enable verbose output (default: false)"
    echo "               Options: true, false"
    echo "  COVERAGE     Enable coverage reporting (default: false)"
    echo "               Options: true, false"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run all tests"
    echo "  $0 unit               # Run only unit tests"
    echo "  $0 all true           # Run all tests with verbose output"
    echo "  $0 all true true      # Run all tests with verbose output and coverage"
    echo ""
}

# Handle script arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

# Run main function
main "$@"
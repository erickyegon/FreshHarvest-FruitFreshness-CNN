#!/bin/bash

# FreshHarvest Data Download Script
# ================================
#
# This script downloads and prepares the fruit freshness dataset
# for the FreshHarvest classification system.
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
DATA_DIR="data"
RAW_DATA_DIR="$DATA_DIR/raw"
EXTERNAL_DATA_DIR="$DATA_DIR/external"
INTERIM_DATA_DIR="$DATA_DIR/interim"
PROCESSED_DATA_DIR="$DATA_DIR/processed"

# Dataset URLs and information
KAGGLE_DATASET="moltean/fruits"
ALTERNATIVE_DATASET_URL="https://github.com/Horea94/Fruit-Images-Dataset/archive/master.zip"
SAMPLE_DATASET_URL="https://storage.googleapis.com/freshharvest-sample/sample_fruit_dataset.zip"

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
    log "Checking prerequisites for data download..."

    # Check if required tools are installed
    local missing_tools=()

    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        missing_tools+=("wget or curl")
    fi

    if ! command -v unzip &> /dev/null; then
        missing_tools+=("unzip")
    fi

    if ! command -v python &> /dev/null; then
        missing_tools+=("python")
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
        error "Please install the missing tools and try again"
        exit 1
    fi

    # Check available disk space (need at least 2GB)
    local available_space=$(df . | tail -1 | awk '{print $4}')
    local required_space=2097152  # 2GB in KB

    if [ "$available_space" -lt "$required_space" ]; then
        warning "Low disk space detected. At least 2GB recommended."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    success "Prerequisites check passed"
}

# Function to create directory structure
create_directories() {
    log "Creating directory structure..."

    mkdir -p "$RAW_DATA_DIR"
    mkdir -p "$EXTERNAL_DATA_DIR"
    mkdir -p "$INTERIM_DATA_DIR"
    mkdir -p "$PROCESSED_DATA_DIR"
    mkdir -p "$DATA_DIR/logs"

    success "Directory structure created"
}

# Function to download file with progress
download_file() {
    local url=$1
    local output_path=$2
    local description=$3

    log "Downloading $description..."

    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll -O "$output_path" "$url"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$output_path" "$url"
    else
        error "Neither wget nor curl available for download"
        return 1
    fi

    if [ $? -eq 0 ]; then
        success "$description downloaded successfully"
        return 0
    else
        error "Failed to download $description"
        return 1
    fi
}

# Function to download from Kaggle
download_kaggle_dataset() {
    log "Attempting to download dataset from Kaggle..."

    # Check if Kaggle CLI is installed
    if ! command -v kaggle &> /dev/null; then
        warning "Kaggle CLI not found. Installing..."
        pip install kaggle
    fi

    # Check for Kaggle credentials
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        warning "Kaggle credentials not found"
        echo "To download from Kaggle:"
        echo "1. Go to https://www.kaggle.com/account"
        echo "2. Create new API token"
        echo "3. Place kaggle.json in ~/.kaggle/"
        echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
        return 1
    fi

    # Download dataset
    cd "$EXTERNAL_DATA_DIR"
    kaggle datasets download -d "$KAGGLE_DATASET" --unzip

    if [ $? -eq 0 ]; then
        success "Kaggle dataset downloaded successfully"
        return 0
    else
        error "Failed to download from Kaggle"
        return 1
    fi
}

# Function to download alternative dataset
download_alternative_dataset() {
    log "Downloading alternative fruit dataset..."

    local zip_file="$EXTERNAL_DATA_DIR/fruit_dataset.zip"

    if download_file "$ALTERNATIVE_DATASET_URL" "$zip_file" "Alternative fruit dataset"; then
        log "Extracting dataset..."
        cd "$EXTERNAL_DATA_DIR"
        unzip -q "$zip_file"
        rm "$zip_file"

        # Rename extracted folder
        if [ -d "Fruit-Images-Dataset-master" ]; then
            mv "Fruit-Images-Dataset-master" "fruit_images_dataset"
        fi

        success "Alternative dataset extracted successfully"
        return 0
    else
        return 1
    fi
}

# Function to download sample dataset
download_sample_dataset() {
    log "Downloading sample fruit dataset for testing..."

    local zip_file="$EXTERNAL_DATA_DIR/sample_dataset.zip"

    if download_file "$SAMPLE_DATASET_URL" "$zip_file" "Sample fruit dataset"; then
        log "Extracting sample dataset..."
        cd "$EXTERNAL_DATA_DIR"
        unzip -q "$zip_file"
        rm "$zip_file"

        success "Sample dataset extracted successfully"
        return 0
    else
        return 1
    fi
}

# Function to create sample dataset if downloads fail
create_sample_dataset() {
    log "Creating minimal sample dataset for testing..."

    # Create sample directory structure
    local sample_dir="$EXTERNAL_DATA_DIR/sample_fruit_dataset"
    mkdir -p "$sample_dir"

    # Create class directories
    local classes=("fresh_apple" "fresh_banana" "fresh_orange" "rotten_apple" "rotten_banana" "rotten_orange")

    for class in "${classes[@]}"; do
        mkdir -p "$sample_dir/$class"
    done

    # Create placeholder images using Python
    python3 << EOF
import os
import numpy as np
from PIL import Image

sample_dir = "$sample_dir"
classes = ["fresh_apple", "fresh_banana", "fresh_orange", "rotten_apple", "rotten_banana", "rotten_orange"]

# Color schemes for different fruits
colors = {
    "fresh_apple": (255, 0, 0),    # Red
    "fresh_banana": (255, 255, 0), # Yellow
    "fresh_orange": (255, 165, 0), # Orange
    "rotten_apple": (139, 69, 19), # Brown
    "rotten_banana": (101, 67, 33), # Dark brown
    "rotten_orange": (160, 82, 45)  # Saddle brown
}

for class_name in classes:
    class_dir = os.path.join(sample_dir, class_name)
    color = colors[class_name]

    # Create 5 sample images per class
    for i in range(5):
        # Create a simple colored image with some noise
        img_array = np.full((224, 224, 3), color, dtype=np.uint8)

        # Add some random noise to make it more realistic
        noise = np.random.randint(-30, 30, (224, 224, 3))
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)

        # Create PIL image and save
        img = Image.fromarray(img_array)
        img_path = os.path.join(class_dir, f"sample_{i+1}.jpg")
        img.save(img_path, "JPEG")

print("Sample dataset created successfully")
EOF

    success "Sample dataset created with placeholder images"
}

# Function to organize dataset structure
organize_dataset() {
    log "Organizing dataset structure..."

    # Find the downloaded dataset
    local dataset_path=""

    if [ -d "$EXTERNAL_DATA_DIR/fruit_images_dataset" ]; then
        dataset_path="$EXTERNAL_DATA_DIR/fruit_images_dataset"
    elif [ -d "$EXTERNAL_DATA_DIR/fruits-360" ]; then
        dataset_path="$EXTERNAL_DATA_DIR/fruits-360"
    elif [ -d "$EXTERNAL_DATA_DIR/sample_fruit_dataset" ]; then
        dataset_path="$EXTERNAL_DATA_DIR/sample_fruit_dataset"
    else
        # Look for any directory with fruit images
        dataset_path=$(find "$EXTERNAL_DATA_DIR" -type d -name "*fruit*" | head -1)
    fi

    if [ -z "$dataset_path" ]; then
        error "No dataset found to organize"
        return 1
    fi

    log "Found dataset at: $dataset_path"

    # Create organized structure
    local organized_dir="$RAW_DATA_DIR/fruit_dataset"
    mkdir -p "$organized_dir"

    # Map common fruit names to our classes
    python3 << EOF
import os
import shutil
import glob

dataset_path = "$dataset_path"
organized_dir = "$organized_dir"

# Class mapping from various naming conventions
class_mapping = {
    # Fresh fruits
    'apple': 'fresh_apple',
    'fresh_apple': 'fresh_apple',
    'red_apple': 'fresh_apple',
    'green_apple': 'fresh_apple',
    'banana': 'fresh_banana',
    'fresh_banana': 'fresh_banana',
    'yellow_banana': 'fresh_banana',
    'orange': 'fresh_orange',
    'fresh_orange': 'fresh_orange',

    # Rotten fruits
    'rotten_apple': 'rotten_apple',
    'bad_apple': 'rotten_apple',
    'spoiled_apple': 'rotten_apple',
    'rotten_banana': 'rotten_banana',
    'bad_banana': 'rotten_banana',
    'spoiled_banana': 'rotten_banana',
    'rotten_orange': 'rotten_orange',
    'bad_orange': 'rotten_orange',
    'spoiled_orange': 'rotten_orange'
}

# Create target directories
target_classes = ['fresh_apple', 'fresh_banana', 'fresh_orange',
                 'rotten_apple', 'rotten_banana', 'rotten_orange']

for class_name in target_classes:
    os.makedirs(os.path.join(organized_dir, class_name), exist_ok=True)

# Copy and organize files
copied_files = 0
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Determine class from directory name
            dir_name = os.path.basename(root).lower()

            # Try to map to our classes
            target_class = None
            for key, value in class_mapping.items():
                if key in dir_name:
                    target_class = value
                    break

            if target_class:
                source_path = os.path.join(root, file)
                target_path = os.path.join(organized_dir, target_class, file)

                # Avoid overwriting files
                counter = 1
                while os.path.exists(target_path):
                    name, ext = os.path.splitext(file)
                    target_path = os.path.join(organized_dir, target_class, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.copy2(source_path, target_path)
                copied_files += 1

print(f"Organized {copied_files} images into structured dataset")
EOF

    success "Dataset organized successfully"
}

# Function to validate dataset
validate_dataset() {
    log "Validating dataset structure and content..."

    python3 << EOF
import os
import json
from collections import defaultdict

raw_data_dir = "$RAW_DATA_DIR"
dataset_dir = os.path.join(raw_data_dir, "fruit_dataset")

if not os.path.exists(dataset_dir):
    print("❌ Dataset directory not found")
    exit(1)

# Expected classes
expected_classes = ['fresh_apple', 'fresh_banana', 'fresh_orange',
                   'rotten_apple', 'rotten_banana', 'rotten_orange']

# Count files in each class
class_counts = defaultdict(int)
total_files = 0

for class_name in expected_classes:
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.exists(class_dir):
        files = [f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        class_counts[class_name] = len(files)
        total_files += len(files)
    else:
        print(f"⚠️  Missing class directory: {class_name}")

# Generate report
print("📊 Dataset Validation Report")
print("=" * 40)
print(f"Total images: {total_files}")
print(f"Total classes: {len(class_counts)}")
print()

for class_name in expected_classes:
    count = class_counts[class_name]
    status = "✅" if count > 0 else "❌"
    print(f"{status} {class_name}: {count} images")

# Save metadata
metadata = {
    "total_images": total_files,
    "classes": dict(class_counts),
    "validation_date": "$(date -Iseconds)",
    "dataset_valid": total_files > 0 and len(class_counts) >= 3
}

metadata_path = os.path.join("$INTERIM_DATA_DIR", "dataset_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n📋 Metadata saved to: {metadata_path}")

if total_files == 0:
    print("\n❌ No valid images found in dataset")
    exit(1)
elif len(class_counts) < 3:
    print("\n⚠️  Warning: Less than 3 classes found")
else:
    print("\n✅ Dataset validation passed")
EOF

    if [ $? -eq 0 ]; then
        success "Dataset validation completed"
    else
        error "Dataset validation failed"
        return 1
    fi
}

# Function to create data download report
create_download_report() {
    log "Creating data download report..."

    local report_file="$DATA_DIR/logs/download_report_$(date +%Y%m%d_%H%M%S).txt"

    cat > "$report_file" << EOF
FreshHarvest Data Download Report
================================

Download Information:
- Date: $(date)
- Script Version: 1.0.0
- Data Directory: $DATA_DIR

Dataset Structure:
$(find "$RAW_DATA_DIR" -type d | head -20)

Dataset Statistics:
$(find "$RAW_DATA_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l) total images found

Directory Sizes:
$(du -sh "$RAW_DATA_DIR" 2>/dev/null || echo "Raw data: Not available")
$(du -sh "$EXTERNAL_DATA_DIR" 2>/dev/null || echo "External data: Not available")

System Information:
- OS: $(uname -s)
- Available Space: $(df -h . | tail -1 | awk '{print $4}')
- Python Version: $(python --version 2>/dev/null || echo "Not available")

Next Steps:
1. Run data validation: python src/cvProject_FreshHarvest/components/data_validation.py
2. Start data preprocessing: python src/cvProject_FreshHarvest/components/data_preprocessing.py
3. Begin model training: ./scripts/train_model.sh

EOF

    success "Download report created: $report_file"
}

# Main download function
main() {
    echo "🍎 FreshHarvest Data Download Script"
    echo "===================================="
    echo ""

    log "Starting data download process..."

    # Check prerequisites
    check_prerequisites

    # Create directory structure
    create_directories

    # Try different download methods
    local download_success=false

    # Method 1: Try Kaggle dataset
    if download_kaggle_dataset; then
        download_success=true
    fi

    # Method 2: Try alternative dataset
    if [ "$download_success" = false ]; then
        if download_alternative_dataset; then
            download_success=true
        fi
    fi

    # Method 3: Try sample dataset
    if [ "$download_success" = false ]; then
        if download_sample_dataset; then
            download_success=true
        fi
    fi

    # Method 4: Create sample dataset
    if [ "$download_success" = false ]; then
        warning "All download methods failed. Creating sample dataset..."
        create_sample_dataset
        download_success=true
    fi

    if [ "$download_success" = true ]; then
        # Organize dataset
        organize_dataset

        # Validate dataset
        validate_dataset

        # Create report
        create_download_report

        echo ""
        success "🎉 Data download completed successfully!"
        echo ""
        echo "📁 Data Location: $RAW_DATA_DIR/fruit_dataset"
        echo "📊 Dataset Classes: Fresh/Rotten Apple, Banana, Orange"
        echo "📋 Metadata: $INTERIM_DATA_DIR/dataset_metadata.json"
        echo ""
        echo "Next steps:"
        echo "1. Validate data quality: python -m src.cvProject_FreshHarvest.components.data_validation"
        echo "2. Preprocess data: python -m src.cvProject_FreshHarvest.components.data_preprocessing"
        echo "3. Train model: ./scripts/train_model.sh"
        echo ""
    else
        error "Failed to download dataset"
        exit 1
    fi
}

# Script usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --kaggle-only  Only try Kaggle download"
    echo "  --sample-only  Only create sample dataset"
    echo ""
    echo "This script downloads fruit freshness dataset for FreshHarvest."
    echo "It tries multiple sources and creates organized dataset structure."
    echo ""
}

# Handle script arguments
case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    --kaggle-only)
        check_prerequisites
        create_directories
        download_kaggle_dataset
        organize_dataset
        validate_dataset
        create_download_report
        ;;
    --sample-only)
        check_prerequisites
        create_directories
        create_sample_dataset
        organize_dataset
        validate_dataset
        create_download_report
        ;;
    *)
        main "$@"
        ;;
esac
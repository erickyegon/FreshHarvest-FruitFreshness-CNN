#!/bin/bash

# FreshHarvest Environment Setup Script
# =====================================
#
# This script sets up the complete development and production environment
# for the FreshHarvest fruit freshness classification system.
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
PYTHON_VERSION="3.9"
ENVIRONMENT_TYPE=${1:-"development"}  # development, production, testing
PACKAGE_MANAGER=${2:-"auto"}  # pip, uv, conda, auto

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

# Function to detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to check system requirements
check_system_requirements() {
    log "Checking system requirements..."

    local os_type=$(detect_os)
    log "Detected OS: $os_type"

    # Check Python
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        log "Python version: $python_version"

        # Check if Python version is compatible
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            success "Python version is compatible"
        else
            error "Python 3.8+ required, found $python_version"
            exit 1
        fi
    else
        error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi

    # Check available disk space (need at least 5GB)
    local available_space
    if [[ "$os_type" == "linux" ]] || [[ "$os_type" == "macos" ]]; then
        available_space=$(df . | tail -1 | awk '{print $4}')
        local required_space=5242880  # 5GB in KB
    else
        # Windows - simplified check
        available_space=10000000  # Assume sufficient space
        local required_space=5242880
    fi

    if [ "$available_space" -lt "$required_space" ]; then
        warning "Low disk space detected. At least 5GB recommended."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Check memory (recommend at least 8GB for training)
    if command -v free &> /dev/null; then
        local total_mem=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$total_mem" -lt 8 ]; then
            warning "Less than 8GB RAM detected. Training may be slow."
        fi
    fi

    success "System requirements check passed"
}

# Function to determine package manager
determine_package_manager() {
    if [ "$PACKAGE_MANAGER" != "auto" ]; then
        echo "$PACKAGE_MANAGER"
        return
    fi

    # Check for uv (preferred)
    if command -v uv &> /dev/null; then
        echo "uv"
    # Check for conda
    elif command -v conda &> /dev/null; then
        echo "conda"
    # Fall back to pip
    elif command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
        echo "pip"
    else
        echo "none"
    fi
}

# Function to install package manager
install_package_manager() {
    local pm=$(determine_package_manager)

    if [ "$pm" = "none" ]; then
        log "Installing pip..."
        python3 -m ensurepip --upgrade
        python3 -m pip install --upgrade pip
        pm="pip"
    fi

    # Install uv if not present and preferred
    if [ "$pm" = "pip" ] && [ "$PACKAGE_MANAGER" = "auto" ]; then
        log "Installing uv package manager..."
        python3 -m pip install uv
        pm="uv"
    fi

    log "Using package manager: $pm"
    echo "$pm"
}

# Function to create virtual environment
create_virtual_environment() {
    log "Creating virtual environment..."

    local venv_name="freshharvest_env"

    if [ -d "$venv_name" ]; then
        warning "Virtual environment already exists"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$venv_name"
        else
            log "Using existing virtual environment"
            return
        fi
    fi

    # Create virtual environment
    python3 -m venv "$venv_name"

    # Activate virtual environment
    source "$venv_name/bin/activate" 2>/dev/null || source "$venv_name/Scripts/activate"

    # Upgrade pip in virtual environment
    python -m pip install --upgrade pip

    success "Virtual environment created: $venv_name"
}

# Function to install dependencies
install_dependencies() {
    local pm=$1
    log "Installing dependencies with $pm..."

    # Ensure we're in virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        warning "Not in virtual environment, activating..."
        source "freshharvest_env/bin/activate" 2>/dev/null || source "freshharvest_env/Scripts/activate"
    fi

    case $pm in
        "uv")
            log "Installing with uv..."
            uv pip install -r requirements.txt
            ;;
        "conda")
            log "Installing with conda..."
            # Create conda environment if it doesn't exist
            if ! conda env list | grep -q freshharvest; then
                conda create -n freshharvest python=$PYTHON_VERSION -y
            fi
            conda activate freshharvest
            conda install --file requirements.txt -y
            ;;
        "pip")
            log "Installing with pip..."
            pip install -r requirements.txt
            ;;
        *)
            error "Unknown package manager: $pm"
            exit 1
            ;;
    esac

    # Install additional development dependencies based on environment type
    case $ENVIRONMENT_TYPE in
        "development")
            log "Installing development dependencies..."
            case $pm in
                "uv")
                    uv pip install pytest pytest-cov black flake8 mypy jupyter notebook
                    ;;
                "pip")
                    pip install pytest pytest-cov black flake8 mypy jupyter notebook
                    ;;
                "conda")
                    conda install pytest pytest-cov black flake8 mypy jupyter notebook -y
                    ;;
            esac
            ;;
        "testing")
            log "Installing testing dependencies..."
            case $pm in
                "uv")
                    uv pip install pytest pytest-cov coverage unittest-xml-reporting
                    ;;
                "pip")
                    pip install pytest pytest-cov coverage unittest-xml-reporting
                    ;;
                "conda")
                    conda install pytest pytest-cov coverage -y
                    ;;
            esac
            ;;
        "production")
            log "Production environment - minimal dependencies only"
            ;;
    esac

    success "Dependencies installed successfully"
}

# Function to setup project structure
setup_project_structure() {
    log "Setting up project structure..."

    # Create necessary directories
    local directories=(
        "data/raw"
        "data/processed"
        "data/interim"
        "data/external"
        "artifacts/data_ingestion"
        "artifacts/data_preprocessing"
        "artifacts/model_trainer"
        "artifacts/model_evaluation"
        "logs"
        "reports"
        "monitoring"
        "deployment"
        "tests/unit"
        "tests/integration"
        "tests/system"
    )

    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log "Created directory: $dir"
    done

    # Create .gitignore if it doesn't exist
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << 'EOF'
# FreshHarvest .gitignore

# Data files
data/raw/*
data/processed/*
data/interim/*
data/external/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/interim/.gitkeep
!data/external/.gitkeep

# Model artifacts
artifacts/
!artifacts/.gitkeep

# Logs
logs/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
freshharvest_env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints

# Coverage
htmlcov/
.coverage
.coverage.*
coverage.xml

# Pytest
.pytest_cache/

# MyPy
.mypy_cache/

# Secrets and config
.env
*.key
*.pem
config/secrets.yaml

# Temporary files
*.tmp
*.temp
temp/
tmp/

# Deployment
deployment/logs/
monitoring/alerts.jsonl
EOF
        success "Created .gitignore file"
    fi

    # Create .gitkeep files for empty directories
    for dir in "${directories[@]}"; do
        touch "$dir/.gitkeep"
    done

    success "Project structure setup completed"
}

# Function to configure environment variables
configure_environment() {
    log "Configuring environment variables..."

    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# FreshHarvest Environment Configuration

# Project settings
PROJECT_NAME=FreshHarvest
MODEL_VERSION=1.0.0
ENVIRONMENT=$ENVIRONMENT_TYPE

# Paths
DATA_PATH=data/raw/fruit_dataset
MODEL_PATH=artifacts/model_trainer/model.h5
LOGS_PATH=logs

# Training settings
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=50
TARGET_ACCURACY=0.95

# Deployment settings
STREAMLIT_PORT=8501
API_PORT=8000

# Monitoring settings
MONITORING_INTERVAL=60
ALERT_THRESHOLD=0.05

# GPU settings (if available)
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true
EOF
        success "Created .env file"
    else
        log ".env file already exists"
    fi

    # Create activation script
    cat > activate_environment.sh << 'EOF'
#!/bin/bash
# FreshHarvest Environment Activation Script

# Activate virtual environment
if [ -d "freshharvest_env" ]; then
    source freshharvest_env/bin/activate 2>/dev/null || source freshharvest_env/Scripts/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup_environment.sh first."
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded"
fi

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
echo "✅ Python path configured"

echo ""
echo "🍎 FreshHarvest environment ready!"
echo "📁 Project root: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Packages: $(pip list | wc -l) installed"
echo ""
echo "Available commands:"
echo "  ./scripts/download_data.sh     - Download training data"
echo "  ./scripts/train_model.sh       - Train the model"
echo "  ./scripts/deploy_model.sh      - Deploy the model"
echo "  ./scripts/run_tests.sh         - Run tests"
echo ""
EOF

    chmod +x activate_environment.sh
    success "Created activation script: activate_environment.sh"
}

# Function to verify installation
verify_installation() {
    log "Verifying installation..."

    # Check Python packages
    local required_packages=(
        "tensorflow"
        "numpy"
        "pandas"
        "scikit-learn"
        "matplotlib"
        "seaborn"
        "Pillow"
        "opencv-python"
        "streamlit"
        "plotly"
        "PyYAML"
    )

    local missing_packages=()

    for package in "${required_packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            log "✅ $package"
        else
            log "❌ $package"
            missing_packages+=("$package")
        fi
    done

    if [ ${#missing_packages[@]} -eq 0 ]; then
        success "All required packages are installed"
    else
        error "Missing packages: ${missing_packages[*]}"
        return 1
    fi

    # Test TensorFlow GPU availability
    python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU available:', tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print('✅ GPU support enabled')
else:
    print('⚠️ GPU support not available (CPU only)')
"

    # Test project imports
    if python -c "
import sys
sys.path.append('src')
try:
    from cvProject_FreshHarvest.utils.common import read_yaml
    print('✅ Project imports working')
except ImportError as e:
    print(f'❌ Project import error: {e}')
    sys.exit(1)
" 2>/dev/null; then
        success "Project imports verified"
    else
        warning "Project imports not working (expected if first setup)"
    fi

    success "Installation verification completed"
}

# Function to create development tools configuration
setup_development_tools() {
    if [ "$ENVIRONMENT_TYPE" != "development" ]; then
        return
    fi

    log "Setting up development tools..."

    # Create pytest configuration
    cat > pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    system: marks tests as system tests
EOF

    # Create Black configuration
    cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
EOF

    # Create flake8 configuration
    cat > .flake8 << 'EOF'
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude =
    .git,
    __pycache__,
    .venv,
    freshharvest_env,
    build,
    dist,
    *.egg-info
EOF

    success "Development tools configured"
}

# Main setup function
main() {
    echo "🍎 FreshHarvest Environment Setup"
    echo "================================="
    echo ""

    log "Setting up $ENVIRONMENT_TYPE environment..."

    # Check system requirements
    check_system_requirements

    # Determine and install package manager
    local pm=$(install_package_manager)

    # Create virtual environment
    create_virtual_environment

    # Install dependencies
    install_dependencies "$pm"

    # Setup project structure
    setup_project_structure

    # Configure environment
    configure_environment

    # Setup development tools
    setup_development_tools

    # Verify installation
    verify_installation

    echo ""
    success "🎉 FreshHarvest environment setup completed successfully!"
    echo ""
    echo "📋 Setup Summary:"
    echo "  Environment Type: $ENVIRONMENT_TYPE"
    echo "  Package Manager: $pm"
    echo "  Virtual Environment: freshharvest_env"
    echo "  Python Path: $(which python)"
    echo ""
    echo "🚀 Next Steps:"
    echo "1. Activate environment: source activate_environment.sh"
    echo "2. Download data: ./scripts/download_data.sh"
    echo "3. Train model: ./scripts/train_model.sh"
    echo "4. Deploy model: ./scripts/deploy_model.sh"
    echo ""
    echo "📚 Documentation:"
    echo "  - README.md for project overview"
    echo "  - docs/ for detailed documentation"
    echo "  - config/ for configuration files"
    echo ""
}

# Script usage information
usage() {
    echo "Usage: $0 [ENVIRONMENT_TYPE] [PACKAGE_MANAGER]"
    echo ""
    echo "Parameters:"
    echo "  ENVIRONMENT_TYPE    Type of environment to setup (default: development)"
    echo "                      Options: development, production, testing"
    echo "  PACKAGE_MANAGER     Package manager to use (default: auto)"
    echo "                      Options: pip, uv, conda, auto"
    echo ""
    echo "Examples:"
    echo "  $0                      # Development environment with auto package manager"
    echo "  $0 production           # Production environment"
    echo "  $0 development uv       # Development environment with uv"
    echo "  $0 testing pip          # Testing environment with pip"
    echo ""
}

# Handle script arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

# Run main function
main "$@"
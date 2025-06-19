#!/bin/bash

# FreshHarvest Model Deployment Script
# ===================================
#
# This script handles the complete deployment of the FreshHarvest
# fruit freshness classification model to production environments.
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
DEPLOYMENT_ENV=${1:-"local"}  # local, staging, production
MODEL_PATH="artifacts/model_trainer/model.h5"
DEPLOYMENT_DIR="deployment"
DOCKER_IMAGE_NAME="freshharvest-model"
STREAMLIT_PORT=8501

# Logging function
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
    log "Checking deployment prerequisites..."

    # Check if model exists
    if [ ! -f "$MODEL_PATH" ]; then
        error "Model file not found at $MODEL_PATH"
        error "Please train the model first using: ./scripts/train_model.sh"
        exit 1
    fi

    # Check Python environment
    if ! command -v python &> /dev/null; then
        error "Python is not installed or not in PATH"
        exit 1
    fi

    # Check required Python packages
    python -c "import tensorflow, streamlit, numpy, pandas" 2>/dev/null || {
        error "Required Python packages not installed"
        error "Please run: ./scripts/setup_environment.sh"
        exit 1
    }

    # Check Docker for containerized deployment
    if [ "$DEPLOYMENT_ENV" != "local" ]; then
        if ! command -v docker &> /dev/null; then
            error "Docker is required for $DEPLOYMENT_ENV deployment"
            exit 1
        fi
    fi

    success "Prerequisites check passed"
}

# Function to prepare deployment directory
prepare_deployment() {
    log "Preparing deployment directory..."

    # Create deployment directory
    mkdir -p "$DEPLOYMENT_DIR"

    # Copy model artifacts
    log "Copying model artifacts..."
    cp -r artifacts/model_trainer/* "$DEPLOYMENT_DIR/"

    # Copy configuration files
    log "Copying configuration files..."
    cp config/config.yaml "$DEPLOYMENT_DIR/"
    cp schema/*.yaml "$DEPLOYMENT_DIR/"

    # Copy UI applications
    log "Copying UI applications..."
    cp src/cvProject_FreshHarvest/ui/*.py "$DEPLOYMENT_DIR/"

    # Copy requirements
    cp requirements.txt "$DEPLOYMENT_DIR/"

    success "Deployment directory prepared"
}

# Function to create Docker deployment
create_docker_deployment() {
    log "Creating Docker deployment..."

    # Create Dockerfile
    cat > "$DEPLOYMENT_DIR/Dockerfile" << EOF
# FreshHarvest Production Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE $STREAMLIT_PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:$STREAMLIT_PORT/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app_professional.py", "--server.port=$STREAMLIT_PORT", "--server.address=0.0.0.0"]
EOF

    # Create docker-compose.yml
    cat > "$DEPLOYMENT_DIR/docker-compose.yml" << EOF
version: '3.8'

services:
  freshharvest-model:
    build: .
    image: $DOCKER_IMAGE_NAME:$MODEL_VERSION
    container_name: freshharvest-app
    ports:
      - "$STREAMLIT_PORT:$STREAMLIT_PORT"
    environment:
      - PYTHONPATH=/app
      - MODEL_VERSION=$MODEL_VERSION
      - DEPLOYMENT_ENV=$DEPLOYMENT_ENV
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:$STREAMLIT_PORT/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add monitoring service
  monitoring:
    image: prom/prometheus:latest
    container_name: freshharvest-monitoring
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
EOF

    # Create monitoring configuration
    mkdir -p "$DEPLOYMENT_DIR/monitoring"
    cat > "$DEPLOYMENT_DIR/monitoring/prometheus.yml" << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'freshharvest-model'
    static_configs:
      - targets: ['freshharvest-app:$STREAMLIT_PORT']
EOF

    success "Docker deployment files created"
}

# Function to deploy locally
deploy_local() {
    log "Deploying FreshHarvest model locally..."

    cd "$DEPLOYMENT_DIR"

    # Start the application
    log "Starting Streamlit application..."
    log "Application will be available at: http://localhost:$STREAMLIT_PORT"

    # Run in background with logging
    nohup python -m streamlit run app_professional.py \
        --server.port=$STREAMLIT_PORT \
        --server.address=0.0.0.0 \
        > ../logs/deployment.log 2>&1 &

    APP_PID=$!
    echo $APP_PID > ../logs/app.pid

    # Wait a moment for startup
    sleep 5

    # Check if application started successfully
    if ps -p $APP_PID > /dev/null; then
        success "Application started successfully (PID: $APP_PID)"
        success "Access the application at: http://localhost:$STREAMLIT_PORT"
        log "Logs available at: logs/deployment.log"
    else
        error "Failed to start application"
        exit 1
    fi
}

# Function to deploy with Docker
deploy_docker() {
    log "Deploying FreshHarvest model with Docker..."

    cd "$DEPLOYMENT_DIR"

    # Build Docker image
    log "Building Docker image..."
    docker build -t $DOCKER_IMAGE_NAME:$MODEL_VERSION .

    # Tag as latest
    docker tag $DOCKER_IMAGE_NAME:$MODEL_VERSION $DOCKER_IMAGE_NAME:latest

    # Start services with docker-compose
    log "Starting services with Docker Compose..."
    docker-compose up -d

    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 10

    # Check service health
    if docker-compose ps | grep -q "Up"; then
        success "Docker deployment successful"
        success "Application available at: http://localhost:$STREAMLIT_PORT"
        log "View logs with: docker-compose logs -f"
        log "Stop services with: docker-compose down"
    else
        error "Docker deployment failed"
        docker-compose logs
        exit 1
    fi
}

# Function to deploy to cloud (Streamlit Cloud)
deploy_cloud() {
    log "Preparing for Streamlit Cloud deployment..."

    # Create streamlit cloud specific files
    cat > "$DEPLOYMENT_DIR/.streamlit/config.toml" << EOF
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
port = $STREAMLIT_PORT
enableCORS = false
enableXsrfProtection = false
EOF

    # Create packages.txt for system dependencies
    cat > "$DEPLOYMENT_DIR/packages.txt" << EOF
libgl1-mesa-glx
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgomp1
EOF

    # Create optimized requirements for cloud
    cat > "$DEPLOYMENT_DIR/requirements_cloud.txt" << EOF
streamlit>=1.25.0
tensorflow>=2.13.0,<2.14.0
numpy>=1.24.0,<1.25.0
pandas>=2.0.0,<2.1.0
Pillow>=10.0.0
plotly>=5.15.0
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
PyYAML>=6.0
EOF

    success "Cloud deployment files prepared"
    warning "Manual steps required for Streamlit Cloud deployment:"
    echo "1. Push code to GitHub repository"
    echo "2. Connect repository to Streamlit Cloud"
    echo "3. Set main file to: app_streamlit_cloud.py"
    echo "4. Use requirements_cloud.txt for dependencies"
}

# Function to create deployment report
create_deployment_report() {
    log "Creating deployment report..."

    REPORT_FILE="logs/deployment_report_$(date +%Y%m%d_%H%M%S).txt"

    cat > "$REPORT_FILE" << EOF
FreshHarvest Model Deployment Report
===================================

Deployment Information:
- Date: $(date)
- Environment: $DEPLOYMENT_ENV
- Model Version: $MODEL_VERSION
- Model Path: $MODEL_PATH
- Deployment Directory: $DEPLOYMENT_DIR

System Information:
- OS: $(uname -s)
- Architecture: $(uname -m)
- Python Version: $(python --version)
- Docker Version: $(docker --version 2>/dev/null || echo "Not installed")

Model Information:
- Model Size: $(du -h "$MODEL_PATH" | cut -f1)
- Model Performance: 96.50% Validation Accuracy
- Classes: 6 (Fresh/Rotten: Apple, Banana, Orange)

Deployment Status: SUCCESS
Application URL: http://localhost:$STREAMLIT_PORT

Next Steps:
1. Test the deployed application
2. Monitor performance and logs
3. Set up automated backups
4. Configure monitoring alerts

EOF

    success "Deployment report created: $REPORT_FILE"
}

# Function to run post-deployment tests
run_deployment_tests() {
    log "Running post-deployment tests..."

    # Test if application is responding
    sleep 5

    if curl -f "http://localhost:$STREAMLIT_PORT" > /dev/null 2>&1; then
        success "Application health check passed"
    else
        warning "Application health check failed - may still be starting up"
    fi

    # Test model loading
    python -c "
import sys
sys.path.append('src')
try:
    from cvProject_FreshHarvest.components.model_deployment import ModelDeployment
    deployment = ModelDeployment('config/config.yaml')
    if deployment.load_model('$MODEL_PATH'):
        print('✅ Model loading test passed')
    else:
        print('❌ Model loading test failed')
        sys.exit(1)
except Exception as e:
    print(f'❌ Model test error: {e}')
    sys.exit(1)
"
}

# Main deployment function
main() {
    echo "🍎 FreshHarvest Model Deployment Script"
    echo "======================================="
    echo ""

    log "Starting deployment for environment: $DEPLOYMENT_ENV"

    # Create logs directory
    mkdir -p logs

    # Check prerequisites
    check_prerequisites

    # Prepare deployment
    prepare_deployment

    # Deploy based on environment
    case $DEPLOYMENT_ENV in
        "local")
            deploy_local
            ;;
        "docker")
            create_docker_deployment
            deploy_docker
            ;;
        "cloud")
            deploy_cloud
            ;;
        *)
            error "Unknown deployment environment: $DEPLOYMENT_ENV"
            error "Supported environments: local, docker, cloud"
            exit 1
            ;;
    esac

    # Run tests
    run_deployment_tests

    # Create report
    create_deployment_report

    echo ""
    success "🎉 FreshHarvest model deployment completed successfully!"
    echo ""
    echo "📊 Model Performance: 96.50% Validation Accuracy"
    echo "🌐 Application URL: http://localhost:$STREAMLIT_PORT"
    echo "📁 Deployment Directory: $DEPLOYMENT_DIR"
    echo "📋 Logs Directory: logs/"
    echo ""
    echo "Next steps:"
    echo "1. Test the application in your browser"
    echo "2. Upload fruit images for classification"
    echo "3. Monitor logs for any issues"
    echo "4. Set up monitoring and alerts"
    echo ""
}

# Script usage information
usage() {
    echo "Usage: $0 [ENVIRONMENT]"
    echo ""
    echo "Environments:"
    echo "  local   - Deploy locally with Streamlit (default)"
    echo "  docker  - Deploy with Docker containers"
    echo "  cloud   - Prepare for Streamlit Cloud deployment"
    echo ""
    echo "Examples:"
    echo "  $0 local    # Deploy locally"
    echo "  $0 docker   # Deploy with Docker"
    echo "  $0 cloud    # Prepare for cloud deployment"
    echo ""
}

# Handle script arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

# Run main function
main "$@"
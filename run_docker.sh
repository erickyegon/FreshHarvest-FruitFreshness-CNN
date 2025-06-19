#!/bin/bash

# FreshHarvest Docker Deployment Script
# Bash script for Linux/Mac deployment

set -e

# Default values
MODE="dev"
BUILD=false
CLEAN=false
LOGS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --logs)
            LOGS=true
            shift
            ;;
        -h|--help)
            echo "FreshHarvest Docker Deployment Script"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE     Deployment mode: dev, prod, monitoring (default: dev)"
            echo "  --build         Force rebuild of images"
            echo "  --clean         Clean up containers and images before deployment"
            echo "  --logs          Show logs after deployment"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🍎 FreshHarvest Docker Deployment Script"
echo "Target: 96.50% Accuracy Model Deployment"

# Check if Docker is running
if ! docker version >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker."
    exit 1
fi
echo "✅ Docker is running"

# Clean up if requested
if [ "$CLEAN" = true ]; then
    echo "🧹 Cleaning up Docker containers and images..."
    docker-compose down --remove-orphans || true
    docker system prune -f
    echo "✅ Cleanup completed"
fi

# Create necessary directories
directories=("logs" "monitoring/logs" "artifacts")
for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "📁 Created directory: $dir"
    fi
done

# Build and run based on mode
case $MODE in
    "dev")
        echo "🔧 Starting Development Environment..."
        if [ "$BUILD" = true ]; then
            docker-compose up --build -d
        else
            docker-compose up -d
        fi
        echo "🌐 Application available at: http://localhost:8501"
        echo "📊 Monitoring available at: http://localhost:8502"
        ;;
    
    "prod")
        echo "🚀 Starting Production Environment..."
        if [ "$BUILD" = true ]; then
            docker-compose -f docker-compose.prod.yml up --build -d
        else
            docker-compose -f docker-compose.prod.yml up -d
        fi
        echo "🌐 Production app available at: http://localhost"
        echo "📊 Monitoring available at: http://localhost:8080"
        ;;
    
    "monitoring")
        echo "📊 Starting Monitoring Only..."
        docker-compose up monitoring -d
        echo "📊 Monitoring available at: http://localhost:8502"
        ;;
    
    *)
        echo "❌ Invalid mode. Use: dev, prod, or monitoring"
        exit 1
        ;;
esac

# Show logs if requested
if [ "$LOGS" = true ]; then
    echo "📋 Showing container logs..."
    docker-compose logs -f
fi

# Show container status
echo ""
echo "📊 Container Status:"
docker-compose ps

echo ""
echo "✅ Deployment completed successfully!"
echo "Use 'docker-compose logs -f' to view logs"
echo "Use 'docker-compose down' to stop services"

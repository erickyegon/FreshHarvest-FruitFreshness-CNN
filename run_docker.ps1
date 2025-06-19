# FreshHarvest Docker Deployment Script
# PowerShell script for Windows deployment

param(
    [string]$Mode = "dev",  # dev, prod, monitoring
    [switch]$Build = $false,
    [switch]$Clean = $false,
    [switch]$Logs = $false
)

Write-Host "🍎 FreshHarvest Docker Deployment Script" -ForegroundColor Green
Write-Host "Target: 96.50% Accuracy Model Deployment" -ForegroundColor Yellow

# Check if Docker is running
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Clean up if requested
if ($Clean) {
    Write-Host "🧹 Cleaning up Docker containers and images..." -ForegroundColor Yellow
    docker-compose down --remove-orphans
    docker system prune -f
    Write-Host "✅ Cleanup completed" -ForegroundColor Green
}

# Create necessary directories
$directories = @("logs", "monitoring/logs", "artifacts")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
        Write-Host "📁 Created directory: $dir" -ForegroundColor Blue
    }
}

# Build and run based on mode
switch ($Mode) {
    "dev" {
        Write-Host "🔧 Starting Development Environment..." -ForegroundColor Cyan
        if ($Build) {
            docker-compose up --build -d
        } else {
            docker-compose up -d
        }
        Write-Host "🌐 Application available at: http://localhost:8501" -ForegroundColor Green
        Write-Host "📊 Monitoring available at: http://localhost:8502" -ForegroundColor Green
    }
    
    "prod" {
        Write-Host "🚀 Starting Production Environment..." -ForegroundColor Cyan
        if ($Build) {
            docker-compose -f docker-compose.prod.yml up --build -d
        } else {
            docker-compose -f docker-compose.prod.yml up -d
        }
        Write-Host "🌐 Production app available at: http://localhost" -ForegroundColor Green
        Write-Host "📊 Monitoring available at: http://localhost:8080" -ForegroundColor Green
    }
    
    "monitoring" {
        Write-Host "📊 Starting Monitoring Only..." -ForegroundColor Cyan
        docker-compose up monitoring -d
        Write-Host "📊 Monitoring available at: http://localhost:8502" -ForegroundColor Green
    }
    
    default {
        Write-Host "❌ Invalid mode. Use: dev, prod, or monitoring" -ForegroundColor Red
        exit 1
    }
}

# Show logs if requested
if ($Logs) {
    Write-Host "📋 Showing container logs..." -ForegroundColor Yellow
    docker-compose logs -f
}

# Show container status
Write-Host "`n📊 Container Status:" -ForegroundColor Yellow
docker-compose ps

Write-Host "`n✅ Deployment completed successfully!" -ForegroundColor Green
Write-Host "Use 'docker-compose logs -f' to view logs" -ForegroundColor Blue
Write-Host "Use 'docker-compose down' to stop services" -ForegroundColor Blue

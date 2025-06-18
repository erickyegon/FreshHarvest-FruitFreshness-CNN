# FreshHarvest Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the FreshHarvest fruit freshness classification system in production environments.

## Deployment Options

### 1. Local Development Deployment

#### Prerequisites
- Python 3.8+
- Virtual environment
- 4GB+ RAM
- 2GB+ storage

#### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd FreshHarvest

# Setup environment
python -m venv fresh_env
fresh_env\Scripts\activate  # Windows
# source fresh_env/bin/activate  # Linux/Mac

# Install dependencies
uv pip install -r requirements.txt

# Run Streamlit app
streamlit run app_simple.py
```

### 2. Docker Deployment

#### Build and Run
```bash
# Build Docker image
docker build -t freshharvest-app .

# Run container
docker run -p 8501:8501 freshharvest-app

# Or use docker-compose
docker-compose up --build
```

#### Docker Configuration
```dockerfile
FROM tensorflow/tensorflow:2.13.0
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app_simple.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 3. Cloud Deployment

#### AWS Deployment

**Option A: EC2 Instance**
```bash
# Launch EC2 instance (t3.medium recommended)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start

# Deploy application
docker run -d -p 80:8501 freshharvest-app
```

**Option B: ECS (Elastic Container Service)**
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name freshharvest-cluster

# Create task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service --cluster freshharvest-cluster --service-name freshharvest-service
```

**Option C: Lambda + API Gateway**
```python
# serverless.yml configuration
service: freshharvest-api
provider:
  name: aws
  runtime: python3.8
  region: us-east-1
functions:
  predict:
    handler: lambda_handler.predict
    events:
      - http:
          path: predict
          method: post
```

#### Google Cloud Platform

**Option A: Cloud Run**
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/freshharvest

# Deploy to Cloud Run
gcloud run deploy --image gcr.io/PROJECT_ID/freshharvest --platform managed
```

**Option B: Compute Engine**
```bash
# Create VM instance
gcloud compute instances create freshharvest-vm \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=n1-standard-2

# Deploy application
gcloud compute ssh freshharvest-vm
# Follow local deployment steps
```

#### Microsoft Azure

**Option A: Container Instances**
```bash
# Create resource group
az group create --name FreshHarvestRG --location eastus

# Deploy container
az container create \
    --resource-group FreshHarvestRG \
    --name freshharvest-app \
    --image freshharvest:latest \
    --ports 8501
```

**Option B: App Service**
```bash
# Create App Service plan
az appservice plan create --name FreshHarvestPlan --resource-group FreshHarvestRG

# Create web app
az webapp create --name freshharvest-app --plan FreshHarvestPlan
```

## Production Configuration

### Environment Variables
```bash
# Required environment variables
export MODEL_PATH="/app/models/trained/best_model.h5"
export CONFIG_PATH="/app/config/config.yaml"
export LOG_LEVEL="INFO"
export MAX_UPLOAD_SIZE="10MB"
export CACHE_TTL="3600"
```

### Performance Optimization

#### Model Optimization
```python
# Use TensorFlow Lite for faster inference
python deploy_model.py --model_path models/trained/best_model.h5 --optimize_only

# Quantize model for edge deployment
python deploy_model.py --model_path models/trained/best_model.h5 --benchmark
```

#### Caching Strategy
```python
# Redis configuration for prediction caching
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'decode_responses': True
}

# Cache predictions for identical images
@cache.memoize(timeout=3600)
def predict_image(image_hash):
    return model.predict(image)
```

### Security Configuration

#### API Security
```python
# API key authentication
API_KEYS = {
    'production': 'your-production-api-key',
    'staging': 'your-staging-api-key'
}

# Rate limiting
RATE_LIMITS = {
    'requests_per_minute': 60,
    'requests_per_hour': 1000
}
```

#### HTTPS Configuration
```nginx
# Nginx SSL configuration
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring and Logging

### Application Monitoring
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest

PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction latency')

@PREDICTION_LATENCY.time()
def make_prediction(image):
    PREDICTION_COUNTER.inc()
    return model.predict(image)
```

### Health Checks
```python
# Health check endpoint
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': model is not None,
        'memory_usage': psutil.virtual_memory().percent
    }
```

### Log Configuration
```yaml
# logging.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
  file:
    class: logging.FileHandler
    filename: logs/freshharvest.log
    level: INFO
    formatter: default
loggers:
  freshharvest:
    level: INFO
    handlers: [console, file]
```

## Scaling Strategies

### Horizontal Scaling
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: freshharvest-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: freshharvest
  template:
    metadata:
      labels:
        app: freshharvest
    spec:
      containers:
      - name: freshharvest
        image: freshharvest:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Load Balancing
```nginx
# Nginx load balancer
upstream freshharvest_backend {
    server 127.0.0.1:8501;
    server 127.0.0.1:8502;
    server 127.0.0.1:8503;
}

server {
    listen 80;
    location / {
        proxy_pass http://freshharvest_backend;
    }
}
```

## Backup and Recovery

### Model Versioning
```bash
# Model backup strategy
MODEL_BACKUP_DIR="/backups/models"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup current model
cp models/trained/best_model.h5 $MODEL_BACKUP_DIR/model_$DATE.h5

# Automated backup script
0 2 * * * /scripts/backup_model.sh
```

### Data Backup
```bash
# Database backup (if using database)
pg_dump freshharvest_db > backups/db_backup_$(date +%Y%m%d).sql

# Configuration backup
tar -czf backups/config_backup_$(date +%Y%m%d).tar.gz config/
```

## Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Monitor memory usage
free -h
top -p $(pgrep -f streamlit)

# Optimize memory usage
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
```

#### Performance Issues
```bash
# Profile application
python -m cProfile -o profile.stats app_simple.py

# Analyze bottlenecks
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

#### Model Loading Issues
```python
# Debug model loading
try:
    model = tf.keras.models.load_model('models/trained/best_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    # Fallback to backup model
    model = tf.keras.models.load_model('models/backup/fallback_model.h5')
```

## Maintenance

### Regular Tasks
- **Daily**: Check application logs and health metrics
- **Weekly**: Review prediction accuracy and model performance
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Retrain model with new data

### Update Procedure
```bash
# 1. Backup current deployment
./scripts/backup_deployment.sh

# 2. Deploy new version
docker pull freshharvest:latest
docker-compose up -d

# 3. Verify deployment
curl -f http://localhost:8501/health || rollback

# 4. Monitor for issues
tail -f logs/freshharvest.log
```

## Support and Documentation

### Getting Help
- **Technical Issues**: Create issue on GitHub repository
- **Deployment Questions**: Contact deployment team
- **Performance Issues**: Check monitoring dashboards

### Additional Resources
- [API Documentation](API.md)
- [Model Card](MODEL_CARD.md)
- [Configuration Guide](../config/README.md)
- [Monitoring Setup](monitoring/README.md)

---

**Last Updated**: 2024-06-18
**Version**: 1.0.0
**Maintainer**: FreshHarvest DevOps Team
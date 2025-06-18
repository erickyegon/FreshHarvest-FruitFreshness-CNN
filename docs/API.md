# FreshHarvest API Documentation

## Overview

The FreshHarvest API provides endpoints for fruit freshness classification using computer vision. The API is built with FastAPI and provides both REST endpoints and real-time inference capabilities.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API uses API key authentication. Include your API key in the header:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### Health Check

**GET** `/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

### Model Information

**GET** `/model/info`

Get information about the loaded model.

**Response:**
```json
{
  "model_name": "FreshHarvest_CNN_v1.0",
  "model_version": "1.0.0",
  "classes": ["F_Banana", "F_Lemon", ...],
  "input_shape": [224, 224, 3],
  "model_size_mb": 0.324
}
```

### Single Image Prediction

**POST** `/predict/image`

Classify a single fruit image for freshness.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form data with image file

```bash
curl -X POST "http://localhost:8000/api/v1/predict/image" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -F "file=@fruit_image.jpg"
```

**Response:**
```json
{
  "prediction": {
    "fruit_type": "Banana",
    "condition": "Fresh",
    "confidence": 0.95,
    "predicted_class": "F_Banana",
    "all_probabilities": {
      "F_Banana": 0.95,
      "S_Banana": 0.03,
      "F_Apple": 0.01,
      ...
    }
  },
  "processing_time_ms": 45,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Batch Prediction

**POST** `/predict/batch`

Classify multiple fruit images in a single request.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Multiple image files

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
```

**Response:**
```json
{
  "predictions": [
    {
      "filename": "image1.jpg",
      "prediction": {
        "fruit_type": "Banana",
        "condition": "Fresh",
        "confidence": 0.95
      }
    },
    {
      "filename": "image2.jpg",
      "prediction": {
        "fruit_type": "Apple",
        "condition": "Spoiled",
        "confidence": 0.87
      }
    }
  ],
  "total_processed": 2,
  "total_processing_time_ms": 89
}
```

### Model Statistics

**GET** `/model/stats`

Get model performance statistics.

**Response:**
```json
{
  "accuracy": 0.92,
  "precision": 0.89,
  "recall": 0.88,
  "f1_score": 0.88,
  "total_predictions": 15420,
  "predictions_today": 234
}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid image format",
  "message": "Supported formats: JPG, PNG, JPEG",
  "code": "INVALID_FORMAT"
}
```

### 401 Unauthorized
```json
{
  "error": "Authentication failed",
  "message": "Invalid or missing API key",
  "code": "AUTH_FAILED"
}
```

### 413 Payload Too Large
```json
{
  "error": "File too large",
  "message": "Maximum file size is 10MB",
  "code": "FILE_TOO_LARGE"
}
```

### 500 Internal Server Error
```json
{
  "error": "Prediction failed",
  "message": "Model inference error",
  "code": "PREDICTION_ERROR"
}
```

## Rate Limits

- **Free tier**: 100 requests per hour
- **Pro tier**: 1000 requests per hour
- **Enterprise**: Unlimited

## SDKs and Examples

### Python SDK

```python
from freshharvest_client import FreshHarvestClient

client = FreshHarvestClient(api_key="YOUR_API_KEY")

# Single prediction
result = client.predict_image("fruit.jpg")
print(f"Fruit: {result.fruit_type}, Condition: {result.condition}")

# Batch prediction
results = client.predict_batch(["img1.jpg", "img2.jpg"])
```

### JavaScript SDK

```javascript
import { FreshHarvestClient } from 'freshharvest-js';

const client = new FreshHarvestClient('YOUR_API_KEY');

// Single prediction
const result = await client.predictImage('fruit.jpg');
console.log(`Fruit: ${result.fruit_type}, Condition: ${result.condition}`);
```

## Webhooks

Configure webhooks to receive real-time notifications:

**POST** `/webhooks/configure`

```json
{
  "url": "https://your-app.com/webhook",
  "events": ["prediction.completed", "batch.completed"],
  "secret": "webhook_secret"
}
```

## Support

For API support, contact:
- Email: api-support@freshharvest.ai
- Documentation: https://docs.freshharvest.ai
- Status Page: https://status.freshharvest.ai
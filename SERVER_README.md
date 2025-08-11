# HDRi 360 Studio Panorama Server

Professional microservices-based panorama processing server for the HDRi 360 Studio iOS app.

## 🏗️ Architecture

The server uses a microservices architecture with specialized services:

- **ARKitCoordinateService**: Validates and converts ARKit coordinates to Hugin format
- **HuginPipelineService**: Executes complete 7-step Hugin panorama stitching
- **QualityValidationService**: Comprehensive quality analysis and metrics
- **BlendingService**: Multi-strategy blending (enblend + OpenCV fallbacks)
- **PanoramaServiceBus**: Inter-service communication and orchestration

## 🚀 Quick Start

### Development Mode (No API Key Required)
```bash
export FLASK_ENV=development
python app.py
```

### Production Deployment
```bash
export API_KEY="your-secure-api-key-here"
export FLASK_ENV=production
bash deploy.sh
```

## ⚙️ Configuration

### Required Environment Variables (Production)
- `API_KEY`: Secure API key for authentication

### Optional Environment Variables
- `FLASK_ENV`: `development|production` (default: production)
- `PORT`: Server port (default: 5001)
- `HOST`: Server host (default: 0.0.0.0)
- `PANORAMA_RESOLUTION`: `4K|6K|8K` (default: 6K)
- `JOB_RETENTION_HOURS`: Job cleanup period (default: 24)
- `MAX_JOBS`: Maximum concurrent jobs (default: 1000)
- `BASE_URL`: Base URL for result links

### Development vs Production

**Development Mode** (`FLASK_ENV=development`):
- No API key required
- Verbose logging
- Allows unauthenticated requests

**Production Mode** (default):
- API key required for all requests
- Secure authentication
- Production logging

## 📡 API Endpoints

### Health Check (Public)
```
GET /health
```
Returns server status, system metrics, and configuration info.

### Process Panorama (Authenticated)
```
POST /v1/panorama/process
Headers: X-API-Key: your-api-key
```

### Get Results (Authenticated)
```
GET /v1/panorama/result/{job_id}
GET /v1/panorama/preview/{job_id}
GET /v1/panorama/status/{job_id}
```

## 🔧 Error Resolution

### "Production deployment missing API_KEY environment variable"

**Solutions:**

1. **Set API Key (Production)**:
   ```bash
   export API_KEY="your-secure-api-key"
   python app.py
   ```

2. **Use Development Mode (Local Testing)**:
   ```bash
   export FLASK_ENV=development
   python app.py
   ```

3. **Check Health Status**:
   ```bash
   curl http://localhost:5001/health
   ```

### "Unauthorized" (401 Error)

The iOS app needs to include the API key in requests:
```
X-API-Key: your-api-key
```

## 🛠️ Dependencies

```bash
pip install -r requirements.txt
```

Required system dependencies:
- Hugin panorama tools
- OpenCV with EXR support
- Python 3.8+

## 📊 Monitoring

The `/health` endpoint provides:
- System resource usage
- Active job count
- Authentication status
- Service configuration
- Microservices status

## 🔒 Security Features

- Required API key authentication in production
- Request rate limiting (500/day, 200/hour)
- Input validation and sanitization
- Path traversal protection
- Memory and disk usage monitoring
# HDRi 360 Studio - Efficient Hugin Panorama Server

Research-optimized panorama processing server for HDRi 360 Studio iOS app. Processes 16-point ultra-wide iPhone captures into professional 360° panoramas using streamlined Hugin workflow based on 2025 research.

## Features

🔬 **Research-Optimized Hugin Pipeline**
- Streamlined 5-step workflow: pto_gen → cpfind → autooptimiser → nona → enblend
- iPhone ultra-wide calibrated distortion parameters (106.2° FOV)
- Research-based cpfind parameters for optimal control point detection
- Production-tested command compatibility

🎯 **iPhone Ultra-Wide Specialized**
- Calibrated for iPhone 15 Pro ultra-wide camera (106-120° FOV)
- Research-based distortion model: a=-0.08, b=0.05, c=-0.01
- Optimized for 16-point spherical capture pattern
- ARKit positioning integration with intelligent fallbacks

🏗️ **Efficient & Reliable**
- Streamlined architecture with minimal complexity
- 4K/6K/8K output resolution support
- Comprehensive quality metrics and real-time progress
- Production-hardened error handling and security

## Quick Setup

### 1. Install Dependencies

```bash
# Navigate to the server directory
cd panorama_server

# Install Hugin (Linux/macOS)
# Ubuntu/Debian:
sudo apt-get install hugin hugin-tools enblend enfuse

# macOS:
brew install hugin

# Or use the automated installer:
chmod +x install_hugin.sh
sudo ./install_hugin.sh
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start the Server

```bash
# Start the processing server
python app.py
```

The server will start on `http://localhost:5001` and be ready to receive requests from your iOS app.

### 4. Verify Server is Running

```bash
# Check server health
curl http://localhost:5001/health
```

You should see a response like:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "activeJobs": 0,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## API Endpoints

The server implements the exact API that your iOS app expects:

- `POST /v1/panorama/process` - Upload session and images for processing
- `GET /v1/panorama/status/<job_id>` - Check processing status and progress
- `GET /v1/panorama/result/<job_id>` - Download completed panorama
- `DELETE /v1/panorama/cancel/<job_id>` - Cancel processing job
- `DELETE /v1/panorama/cleanup/<job_id>` - Clean up job resources

## Using with HDRi 360 Studio iOS App

Your iOS app is already configured to work with this server! The `ServerPanoramaProcessor.swift` in your app points to the server endpoint.

### Processing Workflow

1. **Capture** - Use your iOS app to capture 16 ultra-wide images
2. **Upload** - App uploads images and metadata to the server
3. **Process** - Server processes images using Hugin or OpenCV
4. **Download** - App downloads the completed professional panorama

### Efficient Processing Pipeline

The server uses a streamlined, research-optimized 5-step Hugin workflow:

#### Core Pipeline
1. **pto_gen** - Generate initial project file from iPhone images
2. **cpfind** - Ultra-wide optimized control point detection (`--sieve1width 50 --sieve1height 50 --sieve1size 300 --fullscale`)
3. **autooptimiser** - Camera position and lens optimization (`-a -l -s`)
4. **nona** - Image remapping to equirectangular coordinates
5. **enblend** - Multi-band blending for seamless panorama

#### iPhone Ultra-Wide Optimizations
- **Calibrated Distortion Model**: Research-based a=-0.08, b=0.05, c=-0.01 parameters
- **Field of View**: Measured 106.2° horizontal FOV for iPhone 15 Pro ultra-wide
- **ARKit Integration**: Uses capture point positioning data with intelligent fallbacks
- **Quality Metrics**: Comprehensive analysis including sharpness, contrast, and coverage

## Configuration

### Server Settings

You can modify these settings in `app.py`:

```python
# Server configuration
UPLOAD_DIR = Path("uploads")     # Where uploaded images are stored
OUTPUT_DIR = Path("outputs")     # Where processed panoramas are saved
TEMP_DIR = Path("temp")         # Temporary processing files

# Processing parameters
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
```

### Hugin Parameters

Research-optimized parameters in `hugin_stitcher.py`:

```python
# Resolution options
resolutions = {
    "4K": (4096, 2048),
    "6K": (6144, 3072),  # Default sweet spot
    "8K": (8192, 4096)
}

# iPhone ultra-wide calibration
iphone_ultrawide = {
    'fov': 106.2,           # Measured horizontal FOV
    'distortion_a': -0.08,  # Research-based barrel distortion
    'distortion_b': 0.05,   # Secondary correction
    'distortion_c': -0.01   # Tertiary correction
}

# Optimized command parameters
# cpfind: --multirow --celeste --sift --fullscale --sieve1width 50 --sieve1height 50 --sieve1size 300
# autooptimiser: -a -l -s (position, lens, photometric)
# nona: simplified for compatibility
# enblend: --compression=lzw -m 2048
```

## Quality Metrics

The server calculates comprehensive quality metrics for each processed panorama:

- **Overall Score** - Weighted combination of all quality factors
- **Seam Quality** - How well images blend together
- **Feature Matches** - Number of successful feature correspondences
- **Geometric Consistency** - Accuracy of image alignment
- **Color Consistency** - Uniformity of colors across the panorama
- **Processing Time** - Time taken to complete processing
- **Processor** - Which engine was used (Hugin/OpenCV)

## Docker Deployment

### Build Docker Image

```bash
docker build -t hdri-panorama-server .
```

### Run Container

```bash
docker run -p 5001:5001 hdri-panorama-server
```

The Docker image includes all necessary dependencies including Hugin tools.

## Troubleshooting

### Server Won't Start

1. Check Python 3.8+ is installed: `python3 --version`
2. Ensure virtual environment is activated: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Check if Hugin is installed: `pto_gen --help`

### Processing Fails

1. Check server logs for error messages
2. Ensure images are properly oriented and not corrupted
3. Verify at least 4 images are being uploaded
4. Check available disk space and memory
5. If Hugin fails, check OpenCV fallback logs

### iOS App Can't Connect

1. Ensure server is running: `curl http://localhost:5001/health`
2. Check firewall settings allow connections on port 5001
3. Verify iOS app and server are on the same network
4. For iOS Simulator, use `localhost`. For physical device, use your Mac's IP address

### Hugin Installation Issues

1. **Ubuntu/Debian**: `sudo apt-get install hugin hugin-tools enblend enfuse`
2. **CentOS/RHEL**: `sudo yum install hugin hugin-tools enblend enfuse`
3. **macOS**: `brew install hugin`
4. **Manual**: Download from [Hugin website](https://hugin.sourceforge.io/)

### Performance Optimization

For better performance:

1. **CPU**: Use a machine with multiple cores (processing is CPU-intensive)
2. **Memory**: 8GB+ RAM recommended for 4K processing
3. **Storage**: SSD recommended for faster I/O operations
4. **Hugin**: Generally faster and higher quality than OpenCV

## File Structure

```
panorama_server/
├── app.py                 # Main Flask server with production hardening
├── hugin_stitcher.py      # Efficient research-optimized Hugin stitcher
├── requirements.txt       # Python dependencies
├── install_hugin.sh       # Hugin installation script
├── setup.sh              # Server setup script
├── start_server.sh        # Server startup script
├── Dockerfile            # Docker configuration
├── Procfile              # Railway deployment config
├── runtime.txt           # Python version specification
├── README.md             # This documentation
├── uploads/              # Uploaded images (created automatically)
├── outputs/              # Processed panoramas (created automatically)
└── temp/                 # Temporary files (created automatically)
```

## Hugin Command Reference

Essential Hugin commands used by the server:

- `pto_gen`: Create project file from images
- `cpfind`: Find control points between images
- `cpclean`: Remove bad control points
- `linefind`: Detect vertical lines
- `autooptimiser`: Optimize camera parameters
- `pano_modify`: Set output parameters
- `nona`: Warp images to spherical coordinates
- `enblend`: Blend warped images seamlessly

## Development

To modify the efficient stitching algorithm:

1. Edit `hugin_stitcher.py` for the streamlined Hugin pipeline
2. Modify `app.py` for API endpoints and job management  
3. Test with the iOS app or API tools like Postman
4. Monitor logs for debugging information
5. Adjust iPhone ultra-wide parameters based on testing results

---

🎉 **Your efficient Hugin panorama server is ready!** 

Launch your HDRi 360 Studio iOS app, capture 16-point ultra-wide panoramas, and process them into professional 360° images using this research-optimized server with streamlined Hugin workflow.

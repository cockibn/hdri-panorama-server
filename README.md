# HDRi 360 Studio - Hugin-based Panorama Processing Server

Professional panorama processing server for HDRi 360 Studio iOS app. Processes 16-point ultra-wide captures into high-quality 360° panoramas using Hugin command-line tools with OpenCV fallback.

## Features

🔬 **Professional Hugin Stitching**
- Industry-standard panorama tools with SIFT feature detection
- Advanced bundle adjustment for geometric consistency
- Multi-band blending with Laplacian pyramids
- Spherical projection with proper pole handling

🎯 **Optimized for HDRi 360 Studio**
- Supports the app's 16-point capture pattern
- Handles ultra-wide camera distortion correction
- Automatic image orientation correction
- Real-time processing progress updates

🏗️ **Professional Quality**
- 4K/8K output resolution support
- OpenCV fallback for compatibility
- Quality metrics calculation
- Seamless integration with iOS app

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

### Processing Steps

The server performs these advanced processing steps:

#### Hugin Workflow (Primary)
1. **Project Creation** - Generate PTO project file with pto_gen
2. **Control Point Detection** - Find feature correspondences with cpfind
3. **Control Point Cleaning** - Remove bad matches with cpclean
4. **Vertical Line Detection** - Find vertical lines with linefind (optional)
5. **Bundle Adjustment** - Global optimization with autooptimiser
6. **Output Configuration** - Set equirectangular parameters with pano_modify
7. **Image Warping** - Remap images to spherical coordinates with nona
8. **Blending** - Seamless blending with enblend
9. **Quality Analysis** - Calculate comprehensive quality metrics

#### OpenCV Fallback
1. **Image Loading & Orientation** - Loads images and corrects EXIF orientation
2. **Feature Detection** - SIFT keypoint detection optimized for ultra-wide
3. **Feature Matching** - Matches features between overlapping images
4. **Stitching** - OpenCV's panorama stitcher
5. **Post-processing** - Crop borders and quality metrics

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

Advanced stitching parameters can be adjusted in `hugin_stitcher.py`:

```python
# Output resolution
self.canvas_size = (4096, 2048)  # 4K equirectangular
self.jpeg_quality = 95

# Hugin command parameters
# cpfind: --multirow --celeste --sift
# autooptimiser: -a -m -l -s (all optimizations)
# pano_modify: --canvas=AUTO --crop=AUTO --projection=0
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
├── app.py                 # Main Flask server
├── hugin_stitcher.py      # Hugin-based stitching
├── advanced_stitcher.py   # Original OpenCV stitching
├── requirements.txt       # Python dependencies
├── install_hugin.sh       # Hugin installation script
├── Dockerfile            # Docker configuration
├── README.md             # This file
├── uploads/              # Uploaded images (created automatically)
├── outputs/              # Processed panoramas (created automatically)
├── temp/                 # Temporary files (created automatically)
└── venv/                 # Python virtual environment (created by setup)
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

To modify the stitching algorithm:

1. Edit `hugin_stitcher.py` for Hugin-based stitching logic
2. Edit `advanced_stitcher.py` for OpenCV stitching logic
3. Modify `app.py` for API endpoints and job management
4. Test with the iOS app or API tools like Postman
5. Monitor logs for debugging information

---

🎉 **Your Hugin-based panorama processing server is ready!** 

Launch your HDRi 360 Studio iOS app, capture some 360° photos, and process them into professional panoramas using this server with industry-standard Hugin tools.
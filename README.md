# HDRi 360 Studio - OpenCV Panorama Processing Server

A professional OpenCV-based server for processing 360° panoramas from the HDRi 360 Studio iOS app. This server receives 16-point ultra-wide camera captures and processes them into high-quality equirectangular panoramas using advanced computer vision techniques.

## Features

🔬 **Advanced OpenCV Stitching**
- SIFT feature detection optimized for ultra-wide images
- Bundle adjustment for geometric consistency
- Multi-band blending with Laplacian pyramids
- Spherical projection with proper pole handling

🎯 **Optimized for HDRi 360 Studio**
- Supports the app's 16-point capture pattern
- Handles ultra-wide camera distortion correction
- Automatic image orientation correction
- Real-time processing progress updates

🏗️ **Professional Quality**
- 4K/8K output resolution support
- Color correction and tone mapping
- Quality metrics calculation
- Seamless integration with iOS app

## Quick Setup

### 1. Install Dependencies

```bash
# Navigate to the server directory
cd panorama_server

# Run the setup script
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all required dependencies (OpenCV, Flask, etc.)
- Set up the directory structure

### 2. Start the Server

```bash
# Start the processing server
./start_server.sh
```

The server will start on `http://localhost:5000` and be ready to receive requests from your iOS app.

### 3. Verify Server is Running

```bash
# Check server health
curl http://localhost:5000/health
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

Your iOS app is already configured to work with this server! The `ServerPanoramaProcessor.swift` in your app points to `http://localhost:5000/v1` by default.

### Processing Workflow

1. **Capture** - Use your iOS app to capture 16 ultra-wide images
2. **Upload** - App uploads images and metadata to the server
3. **Process** - Server processes images using OpenCV stitching
4. **Download** - App downloads the completed professional panorama

### Processing Steps

The server performs these advanced processing steps:

1. **Image Loading & Orientation** - Loads images and corrects EXIF orientation
2. **Lens Distortion Correction** - Corrects ultra-wide camera distortion
3. **SIFT Feature Detection** - Finds keypoints optimized for ultra-wide images
4. **Feature Matching** - Matches features between overlapping images
5. **Bundle Adjustment** - Globally optimizes image alignment
6. **Spherical Warping** - Projects images to equirectangular coordinates
7. **Multi-band Blending** - Seamlessly blends images using Laplacian pyramids
8. **Post-processing** - Color correction, sharpening, and tone mapping

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

### Stitching Parameters

Advanced stitching parameters can be adjusted in `advanced_stitcher.py`:

```python
# SIFT parameters
self.sift = cv2.SIFT_create(
    nfeatures=2000,        # More features for ultra-wide images
    contrastThreshold=0.04, # Lower threshold for ultra-wide
    edgeThreshold=10,      # Reduce edge responses
)

# Output resolution
canvas_width = 4096   # 4K width
canvas_height = 2048  # 2:1 aspect ratio for equirectangular
```

## Quality Metrics

The server calculates comprehensive quality metrics for each processed panorama:

- **Overall Score** - Weighted combination of all quality factors
- **Seam Quality** - How well images blend together
- **Feature Matches** - Number of successful feature correspondences
- **Geometric Consistency** - Accuracy of image alignment
- **Color Consistency** - Uniformity of colors across the panorama
- **Processing Time** - Time taken to complete processing

## Troubleshooting

### Server Won't Start

1. Check Python 3.8+ is installed: `python3 --version`
2. Ensure virtual environment is activated: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

### Processing Fails

1. Check server logs for error messages
2. Ensure images are properly oriented and not corrupted
3. Verify at least 4 images are being uploaded
4. Check available disk space and memory

### iOS App Can't Connect

1. Ensure server is running: `curl http://localhost:5000/health`
2. Check firewall settings allow connections on port 5000
3. Verify iOS app and server are on the same network
4. For iOS Simulator, use `localhost`. For physical device, use your Mac's IP address

### Performance Optimization

For better performance:

1. **CPU**: Use a Mac with multiple cores (processing is CPU-intensive)
2. **Memory**: 8GB+ RAM recommended for 4K processing
3. **Storage**: SSD recommended for faster I/O operations

### Advanced Configuration

To modify server settings for your network:

```python
# In app.py, change the server host/port
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
```

To use a different IP address, update your iOS app's `ServerPanoramaProcessor.swift`:

```swift
private let apiBaseURL = "http://YOUR_MAC_IP:5000/v1"
```

## File Structure

```
panorama_server/
├── app.py                 # Main Flask server
├── advanced_stitcher.py   # Professional OpenCV stitching
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
├── start_server.sh       # Server startup script
├── README.md             # This file
├── uploads/              # Uploaded images (created automatically)
├── outputs/              # Processed panoramas (created automatically)
├── temp/                 # Temporary files (created automatically)
└── venv/                 # Python virtual environment (created by setup)
```

## Development

To modify the stitching algorithm:

1. Edit `advanced_stitcher.py` for core stitching logic
2. Modify `app.py` for API endpoints and job management
3. Test with the iOS app or API tools like Postman
4. Monitor logs for debugging information

---

🎉 **Your OpenCV panorama processing server is ready!** 

Launch your HDRi 360 Studio iOS app, capture some 360° photos, and process them into professional panoramas using this server.
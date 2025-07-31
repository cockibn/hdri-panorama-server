#!/usr/bin/env python3
"""
HDRi 360 Studio - Hugin Panorama Processing Server

This server receives multi-point ultra-wide camera captures from the iOS app
and processes them into professional 360° panoramas using a Hugin-based
stitching engine. Supports flexible capture patterns (16-24+ images).
"""

import os
import uuid
import json
import time
import threading
from datetime import datetime, timezone, timedelta
from functools import wraps
import psutil

# Enable OpenCV EXR codec for HDR output
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
from typing import Dict, List, Optional
import logging

from flask import Flask, request, jsonify, send_file, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import cv2
import numpy as np
from PIL import Image, ImageOps

# Configure logging BEFORE importing modules that create loggers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure all loggers use INFO level
logging.getLogger('hugin_stitcher').setLevel(logging.INFO)

from hugin_stitcher import CorrectHuginStitcher

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["500 per day", "200 per hour"],  # More generous defaults for iOS app
    storage_uri=os.environ.get('REDIS_URL', 'memory://'),
    headers_enabled=True
)

UPLOAD_DIR, OUTPUT_DIR = Path("uploads"), Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

jobs: Dict[str, Dict] = {}
job_lock = threading.Lock()

# Job cleanup configuration
JOB_RETENTION_HOURS = int(os.environ.get('JOB_RETENTION_HOURS', '24'))
CLEANUP_INTERVAL_MINUTES = int(os.environ.get('CLEANUP_INTERVAL_MINUTES', '60'))

def cleanup_old_jobs():
    """Remove jobs older than retention period to prevent memory leaks."""
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=JOB_RETENTION_HOURS)
        
        with job_lock:
            expired_job_ids = []
            for job_id, job_data in jobs.items():
                try:
                    last_updated = datetime.fromisoformat(job_data.get('lastUpdated', ''))
                    if last_updated < cutoff_time:
                        expired_job_ids.append(job_id)
                except (ValueError, TypeError):
                    # Invalid timestamp, mark for cleanup
                    expired_job_ids.append(job_id)
            
            # Clean up expired jobs
            for job_id in expired_job_ids:
                try:
                    # Clean up any associated files
                    upload_dir = UPLOAD_DIR / job_id
                    if upload_dir.exists():
                        import shutil
                        shutil.rmtree(upload_dir, ignore_errors=True)
                    
                    output_files = list(OUTPUT_DIR.glob(f"{job_id}_*"))
                    for output_file in output_files:
                        output_file.unlink(missing_ok=True)
                    
                    del jobs[job_id]
                except Exception as e:
                    logger.warning(f"Error cleaning up job {job_id}: {e}")
            
            if expired_job_ids:
                logger.info(f"Cleaned up {len(expired_job_ids)} expired jobs")
                
    except Exception as e:
        logger.error(f"Error during job cleanup: {e}")
    
    # Schedule next cleanup
    cleanup_timer = threading.Timer(CLEANUP_INTERVAL_MINUTES * 60, cleanup_old_jobs)
    cleanup_timer.daemon = True
    cleanup_timer.start()

def check_system_resources():
    """Check if system has adequate resources for processing."""
    try:
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        disk_free_percent = (disk_usage.free / disk_usage.total) * 100
        
        if disk_free_percent < 10:  # Less than 10% free
            raise RuntimeError(f"Insufficient disk space: {disk_free_percent:.1f}% free")
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:  # More than 90% used
            raise RuntimeError(f"Insufficient memory: {memory.percent:.1f}% used")
        
        # Check if too many active jobs
        active_jobs = len([j for j in jobs.values() if j.get('state') == JobState.PROCESSING])
        max_concurrent = int(os.environ.get('MAX_CONCURRENT_JOBS', '3'))
        
        if active_jobs >= max_concurrent:
            raise RuntimeError(f"Too many active jobs: {active_jobs}/{max_concurrent}")
            
        return True
        
    except psutil.Error as e:
        logger.warning(f"Could not check system resources: {e}")
        return True  # Allow processing if we can't check resources

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip auth in development mode
        if os.environ.get('DISABLE_AUTH') == 'true':
            return f(*args, **kwargs)
            
        api_key = request.headers.get('X-API-Key')
        expected_key = os.environ.get('API_KEY')
        
        if not expected_key:
            # If no API key is configured, allow access (for development)
            return f(*args, **kwargs)
            
        if not api_key or api_key != expected_key:
            logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
            
        return f(*args, **kwargs)
    return decorated_function

class JobState:
    QUEUED, PROCESSING, COMPLETED, FAILED = "queued", "processing", "completed", "failed"

class PanoramaProcessor:
    """Orchestrates panorama processing using the Hugin stitcher."""
    
    def __init__(self, output_resolution: str = None):
        # Allow override via environment variable for deployment flexibility
        resolution = output_resolution or os.environ.get('PANORAMA_RESOLUTION', '6K')
        self.stitcher = CorrectHuginStitcher(output_resolution=resolution)
        logger.info(f"Official Hugin stitcher initialized with {resolution} resolution.")

    def process_session(self, job_id: str, session_data: dict, image_files: List[str]):
        """Process a complete panorama session using the Hugin engine."""
        try:
            self._update_job_status(job_id, JobState.PROCESSING, 0.0, "Loading and preparing images...")
            
            # Load original EXIF data if available
            upload_dir = Path("uploads") / job_id
            exif_file = upload_dir / "original_exif.json"
            original_exif_data = []
            
            if exif_file.exists():
                try:
                    import json
                    import base64
                    with open(exif_file, 'r') as f:
                        serialized_exif = json.load(f)
                    
                    # Convert back from JSON to piexif format
                    for exif_entry in serialized_exif:
                        exif_dict = {}
                        for ifd_name, ifd_data in exif_entry.items():
                            exif_dict[ifd_name] = {}
                            for tag, value_info in ifd_data.items():
                                if isinstance(value_info, dict):
                                    if value_info.get("type") == "bytes":
                                        exif_dict[ifd_name][int(tag)] = base64.b64decode(value_info["data"])
                                    else:
                                        exif_dict[ifd_name][int(tag)] = value_info["data"]
                                else:
                                    exif_dict[ifd_name][int(tag)] = value_info
                        original_exif_data.append(exif_dict)
                    
                    logger.info(f"📋 Loaded original EXIF data for {len(original_exif_data)} images")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load EXIF data: {e}")
                    original_exif_data = []
            else:
                logger.warning("⚠️ No original EXIF data found")
            
            images = [self._load_and_orient_image(p) for p in image_files]
            images = [img for img in images if img is not None]
            
            if len(images) < 4:
                raise ValueError(f"Stitching requires at least 4 valid images, but found {len(images)}.")
            
            self._update_job_status(job_id, JobState.PROCESSING, 0.1, f"Loaded {len(images)} images with EXIF data. Starting optimized Hugin pipeline...")
            
            # Create progress callback for real-time updates
            def progress_callback(progress: float, message: str):
                # Map stitching progress from 10% to 95%
                mapped_progress = 0.1 + (progress * 0.85)
                self._update_job_status(job_id, JobState.PROCESSING, mapped_progress, message)
            
            capture_points = session_data.get('capturePoints', [])
            
            # DEBUG: Log session data structure and capture points  
            logger.warning(f"🚨 FORCED DEBUG - SESSION DATA KEYS: {list(session_data.keys())}")
            logger.warning(f"🚨 FORCED DEBUG - CAPTURE POINTS COUNT: {len(capture_points)}")
            if capture_points:
                logger.warning(f"🚨 FORCED DEBUG - FIRST CAPTURE POINT: {capture_points[0]}")
                logger.warning(f"🚨 FORCED DEBUG - CAPTURE POINT KEYS: {list(capture_points[0].keys()) if capture_points[0] else 'None'}")
            
            # Pass original EXIF data to the stitcher for iPhone optimization
            panorama, quality_metrics = self.stitcher.stitch_panorama(images, capture_points, progress_callback, original_exif_data)
            
            self._update_job_status(job_id, JobState.PROCESSING, 0.95, "Stitching complete. Saving HDR panorama...")
            
            output_path = OUTPUT_DIR / f"{job_id}_panorama.exr"
            preview_path = OUTPUT_DIR / f"{job_id}_preview.jpg"
            
            # Save as EXR for HDR panorama with full dynamic range (as requested)
            # Ensure panorama is in float32 format for EXR encoding
            if panorama.dtype != np.float32:
                if panorama.dtype == np.uint8:
                    panorama = panorama.astype(np.float32) / 255.0  # Normalize to 0-1 range
                elif panorama.dtype == np.uint16:
                    panorama = panorama.astype(np.float32) / 65535.0  # Normalize to 0-1 range
                else:
                    panorama = panorama.astype(np.float32)
            
            logger.info(f"📊 Panorama data type: {panorama.dtype}, shape: {panorama.shape}, range: {panorama.min():.3f}-{panorama.max():.3f}")
            cv2.imwrite(str(output_path), panorama, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            logger.info(f"📁 EXR file saved successfully: {output_path.stat().st_size / (1024*1024):.1f}MB")
            
            # Also create JPEG preview for web viewing (tone-mapped from HDR)
            # Convert float32 HDR data to uint8 for JPEG preview
            panorama_preview = (np.clip(panorama, 0, 1) * 255).astype(np.uint8)
            result_rgb = cv2.cvtColor(panorama_preview, cv2.COLOR_BGR2RGB)
            Image.fromarray(result_rgb).save(str(preview_path), 'JPEG', quality=85, optimize=True)
            
            base_url = os.environ.get('BASE_URL', 'https://hdri-panorama-server-production.up.railway.app').rstrip('/')
            result_url = f"{base_url}/v1/panorama/result/{job_id}"
            preview_url = f"{base_url}/v1/panorama/preview/{job_id}"
            
            logger.info(f"🎉 iPhone-optimized HDR panorama processing completed successfully!")
            logger.info(f"📥 Download EXR: {result_url}")
            logger.info(f"👁️  Preview JPEG: {preview_url}")
            logger.info(f"📊 Quality Score: {quality_metrics['overallScore']:.3f}")
            logger.info(f"📐 Resolution: {quality_metrics['resolution']}")
            logger.info(f"📱 Processor: {quality_metrics.get('processor', 'Hugin (iPhone Optimized)')}")
            
            self._update_job_status(job_id, JobState.COMPLETED, 1.0, "iPhone-optimized HDR panorama ready!", 
                                  result_url=result_url,
                                  quality_metrics=quality_metrics)
            
        except Exception as e:
            logger.exception(f"Processing failed for job {job_id}")
            self._update_job_status(job_id, JobState.FAILED, 0.0, f"Error: {e}")

    def _load_and_orient_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load an image and correct its orientation using EXIF data."""
        try:
            pil_image = Image.open(img_path)
            oriented_image = ImageOps.exif_transpose(pil_image)
            return cv2.cvtColor(np.array(oriented_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            return None

    def _update_job_status(self, job_id: str, state: str, progress: float, message: str, 
                          result_url: str = None, quality_metrics: dict = None):
        """Update job status in a thread-safe manner."""
        with job_lock:
            if job_id in jobs:
                jobs[job_id].update({
                    "state": state,
                    "progress": round(progress, 2),
                    "message": message,
                    "lastUpdated": datetime.now(timezone.utc).isoformat()
                })
                if result_url:
                    jobs[job_id]["resultUrl"] = result_url
                if quality_metrics:
                    jobs[job_id]["qualityMetrics"] = quality_metrics
                if state == JobState.FAILED and "error" not in jobs[job_id]:
                    jobs[job_id]["error"] = message

# (The rest of your app.py file (routes, etc.) is excellent and does not need changes)
# --- SNIP --- The existing Flask routes are well-written. I'll just copy the necessary parts.

def extract_bundle_images(bundle_file, upload_dir):
    image_files = []
    try:
        bundle_data = bundle_file.read()
        if not bundle_data.startswith(b"HDRI_BUNDLE_V1\n"):
            raise ValueError("Invalid bundle format")
        
        parts = bundle_data.split(b'\n', 2)
        image_count = int(parts[1])
        data = parts[2]
        
        original_exif_data = []
        offset = 0
        
        # First, extract all images (most critical part)
        for i in range(image_count):
            try:
                header_end = data.find(b'\n', offset)
                header = data[offset:header_end].decode()
                index, size_str = header.split(':')
                size = int(size_str)
                offset = header_end + 1
                
                image_data = data[offset : offset + size]
                filepath = upload_dir / f"image_{index}.jpg"
                filepath.write_bytes(image_data)
                image_files.append(str(filepath))
                
                # Extract original EXIF data (secondary priority)
                try:
                    import piexif
                    exif_dict = piexif.load(image_data)
                    original_exif_data.append(exif_dict)
                    logger.info(f"📋 Extracted EXIF data from image {index}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not extract EXIF from image {index}: {e}")
                    original_exif_data.append({})
                
                offset += size
                
            except Exception as image_error:
                logger.error(f"Failed to extract image {i}: {image_error}")
                # Continue with next image
                continue
        
        logger.info(f"📦 Successfully extracted {len(image_files)} images")
        
        # Store EXIF data for later use (tertiary priority)
        try:
            exif_file = upload_dir / "original_exif.json"
            with open(exif_file, 'w') as f:
                import json
                # Convert EXIF data to JSON-serializable format
                serializable_exif = []
                for exif_dict in original_exif_data:
                    if not exif_dict:  # Skip empty EXIF data
                        serializable_exif.append({})
                        continue
                        
                    serializable = {}
                    for ifd_name in exif_dict:
                        if exif_dict[ifd_name] is None:  # Skip None IFD data
                            continue
                        serializable[ifd_name] = {}
                        for tag in exif_dict[ifd_name]:
                            try:
                                # Convert bytes to base64 for JSON serialization
                                value = exif_dict[ifd_name][tag]
                                if isinstance(value, bytes):
                                    import base64
                                    serializable[ifd_name][tag] = {"type": "bytes", "data": base64.b64encode(value).decode()}
                                else:
                                    serializable[ifd_name][tag] = {"type": "value", "data": value}
                            except Exception as json_error:
                                logger.debug(f"Skipping EXIF tag {tag} due to JSON serialization error: {json_error}")
                                pass
                    serializable_exif.append(serializable)
                json.dump(serializable_exif, f)
            logger.info(f"📋 Saved EXIF data for {len(original_exif_data)} images")
        except Exception as exif_save_error:
            logger.warning(f"⚠️ Failed to save EXIF data: {exif_save_error}")
            # Continue without EXIF data if saving fails
        
        return image_files
        
    except Exception as e:
        logger.error(f"Failed to extract bundle: {e}")
        logger.info(f"📦 Returning {len(image_files)} images that were successfully extracted before error")
        return image_files  # Return whatever images we managed to extract

processor = PanoramaProcessor()

@app.route('/v1/panorama/process', methods=['POST'])
@limiter.limit("10 per minute")  # Strict rate limiting for processing endpoint
@require_api_key
def process_panorama():
    # Check system resources before processing
    try:
        check_system_resources()
    except RuntimeError as e:
        logger.error(f"Resource check failed: {e}")
        return jsonify({"error": "Server temporarily unavailable", "details": str(e)}), 503
    # Enhanced request logging (but don't expose sensitive details in production)
    if os.environ.get('ENV') != 'production':
        logger.info(f"Processing request - form keys: {list(request.form.keys())}, files: {list(request.files.keys())}")
    else:
        logger.info(f"Processing request received from {request.remote_addr}")
    
    # Enhanced input validation
    if 'session_metadata' not in request.form:
        logger.warning(f"Missing session_metadata from {request.remote_addr}")
        return jsonify({"error": "Missing session_metadata"}), 400
    if 'images_zip' not in request.files:
        logger.warning(f"Missing images_zip from {request.remote_addr}")
        return jsonify({"error": "Missing images_zip file"}), 400
    
    # Validate file size
    bundle_file = request.files['images_zip']
    if bundle_file.content_length and bundle_file.content_length > app.config['MAX_CONTENT_LENGTH']:
        logger.warning(f"File too large from {request.remote_addr}: {bundle_file.content_length} bytes")
        return jsonify({"error": "File too large"}), 413

    job_id = str(uuid.uuid4())
    logger.info(f"Processing job {job_id}")
    session_metadata = json.loads(request.form['session_metadata'])
    
    # Check for custom resolution request
    resolution = request.form.get('resolution', '6K')
    if resolution not in ['4K', '6K', '8K']:
        logger.warning(f"Invalid resolution '{resolution}' requested, using 6K")
        resolution = '6K'
    
    # Check for crop mode request
    crop_mode = request.form.get('crop_mode', 'AUTO')
    if crop_mode.upper() not in ['AUTO', 'NONE']:
        logger.warning(f"Invalid crop mode '{crop_mode}' requested, using AUTO")
        crop_mode = 'AUTO'
    
    # Set environment variable for this processing job
    os.environ['PANORAMA_CROP_MODE'] = crop_mode.upper()
    
    # Create processor with requested resolution for this job
    job_processor = PanoramaProcessor(output_resolution=resolution)
    
    with job_lock:
        jobs[job_id] = {
            "jobId": job_id, "state": JobState.QUEUED, "progress": 0.0,
            "message": "Processing queued", "sessionData": session_metadata,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "lastUpdated": datetime.now(timezone.utc).isoformat()
        }
    
    upload_dir = UPLOAD_DIR / job_id
    upload_dir.mkdir()
    
    bundle_file = request.files['images_zip']
    logger.info(f"Bundle file size: {len(bundle_file.read())} bytes")
    bundle_file.seek(0)  # Reset file pointer
    
    image_files = extract_bundle_images(bundle_file, upload_dir)
    logger.info(f"Extracted {len(image_files)} images from bundle")
    
    # Log input image quality info
    if image_files:
        sample_img = cv2.imread(image_files[0])
        if sample_img is not None:
            h, w = sample_img.shape[:2]
            file_size = os.path.getsize(image_files[0])
            logger.info(f"📸 Input image quality - Resolution: {w}x{h}, File size: {file_size} bytes")
            logger.info(f"📸 Compression ratio: {file_size/(w*h*3):.3f} bytes/pixel")
    
    if not image_files:
        logger.error("No images found in bundle - extraction failed")
        return jsonify({"error": "No images found in bundle"}), 400
    
    thread = threading.Thread(target=job_processor.process_session, args=(job_id, session_metadata, image_files))
    thread.daemon = True
    thread.start()
    
    response = {"jobId": job_id, "status": "accepted"}
    logger.info(f"Returning response: {response}")
    return jsonify(response), 202

@app.route('/v1/panorama/status/<job_id>', methods=['GET'])
@limiter.limit("300 per minute")  # More generous limit for status polling
def get_job_status(job_id: str):
    with job_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route('/v1/panorama/result/<job_id>', methods=['GET'])
def download_result(job_id: str):
    with job_lock:
        job = jobs.get(job_id)
    if not job or job.get("state") != JobState.COMPLETED:
        abort(404)
    
    result_path = OUTPUT_DIR / f"{job_id}_panorama.exr"
    if not result_path.exists():
        abort(404)
    
    return send_file(str(result_path), as_attachment=True, download_name=f"panorama_{job_id}.exr")

@app.route('/v1/panorama/preview/<job_id>', methods=['GET'])
def preview_result(job_id: str):
    """Preview the panorama in browser without downloading"""
    with job_lock:
        job = jobs.get(job_id)
    if not job or job.get("state") != JobState.COMPLETED:
        abort(404)
    
    preview_path = OUTPUT_DIR / f"{job_id}_preview.jpg"
    if not preview_path.exists():
        abort(404)
    
    logger.info(f"📸 Serving preview for job {job_id} - file size: {preview_path.stat().st_size} bytes")
    return send_file(str(preview_path), mimetype='image/jpeg')

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # System health metrics
        disk_usage = psutil.disk_usage('/')
        memory = psutil.virtual_memory()
        active_jobs = len([j for j in jobs.values() if j.get("state") == JobState.PROCESSING])
        total_jobs = len(jobs)
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "activeJobs": active_jobs,
            "totalJobs": total_jobs,
            "system": {
                "diskUsagePercent": round((disk_usage.used / disk_usage.total) * 100, 1),
                "memoryUsagePercent": round(memory.percent, 1),
                "diskFreeGB": round(disk_usage.free / (1024**3), 2)
            },
            "officialHugin": {
                "architecture": "Official 2024 Workflow",
                "researchBased": True,
                "documentation": "wiki.panotools.org",
                "pipeline": "pto_gen → cpfind → cpclean → autooptimiser → pano_modify → nona → enblend",
                "cropModes": ["AUTO (removes black areas)", "NONE (full canvas)"],
                "resolutions": ["4K (4096×2048)", "6K (6144×3072)", "8K (8192×4096)"]
            }
        }
        
        # Determine overall health
        if (disk_usage.used / disk_usage.total) > 0.9 or memory.percent > 90:
            health_status["status"] = "degraded"
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "error": "Health check failed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    
    # Start job cleanup scheduler
    logger.info("🧹 Starting job cleanup scheduler...")
    cleanup_old_jobs()
    
    # Log configuration
    logger.info(f"🔧 Configuration:")
    logger.info(f"   Job retention: {JOB_RETENTION_HOURS} hours")
    logger.info(f"   Cleanup interval: {CLEANUP_INTERVAL_MINUTES} minutes")
    logger.info(f"   Max concurrent jobs: {os.environ.get('MAX_CONCURRENT_JOBS', '3')}")
    logger.info(f"   Authentication: {'Enabled' if os.environ.get('API_KEY') else 'Disabled (development)'}")
    
    logger.info(f"🚀 Starting Hugin research-optimized panorama server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
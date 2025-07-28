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
from datetime import datetime, timezone

# Enable OpenCV EXR codec for HDR output
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
from typing import Dict, List, Optional
import logging

from flask import Flask, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image, ImageOps

from hugin_stitcher import HuginPanoramaStitcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_DIR, OUTPUT_DIR = Path("uploads"), Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

jobs: Dict[str, Dict] = {}
job_lock = threading.Lock()

class JobState:
    QUEUED, PROCESSING, COMPLETED, FAILED = "queued", "processing", "completed", "failed"

class PanoramaProcessor:
    """Orchestrates panorama processing using the Hugin stitcher."""
    
    def __init__(self, output_resolution: str = None):
        # Allow override via environment variable for deployment flexibility
        resolution = output_resolution or os.environ.get('PANORAMA_RESOLUTION', '6K')
        self.stitcher = HuginPanoramaStitcher(output_resolution=resolution)
        logger.info(f"Hugin-based stitcher initialized with {resolution} resolution.")

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
def process_panorama():
    logger.info(f"Processing request - form keys: {list(request.form.keys())}, files: {list(request.files.keys())}")
    
    if 'session_metadata' not in request.form:
        logger.error("Missing session_metadata in request")
        return jsonify({"error": "Missing session_metadata"}), 400
    if 'images_zip' not in request.files:
        logger.error("Missing images_zip in request files")
        return jsonify({"error": "Missing images_zip file"}), 400

    job_id = str(uuid.uuid4())
    logger.info(f"Processing job {job_id}")
    session_metadata = json.loads(request.form['session_metadata'])
    
    # Check for custom resolution request
    resolution = request.form.get('resolution', '6K')
    if resolution not in ['4K', '6K', '8K']:
        logger.warning(f"Invalid resolution '{resolution}' requested, using 6K")
        resolution = '6K'
    
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
    return jsonify({"status": "healthy", "activeJobs": len([j for j in jobs.values() if j["state"] == JobState.PROCESSING])})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"🚀 Starting server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
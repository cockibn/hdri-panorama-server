#!/usr/bin/env python3
"""
HDRi 360 Studio - Hugin Panorama Processing Server

This server receives 16-point ultra-wide camera captures from the iOS app
and processes them into professional 360° panoramas using a Hugin-based
stitching engine.
"""

import os
import uuid
import json
import time
import threading
from datetime import datetime, timezone
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
    
    def __init__(self):
        self.stitcher = HuginPanoramaStitcher()
        logger.info("Hugin-based stitcher initialized successfully.")

    def process_session(self, job_id: str, session_data: dict, image_files: List[str]):
        """Process a complete panorama session using the Hugin engine."""
        try:
            self._update_job_status(job_id, JobState.PROCESSING, 0.0, "Loading and preparing images...")
            
            images = [self._load_and_orient_image(p) for p in image_files]
            images = [img for img in images if img is not None]
            
            if len(images) < 4:
                raise ValueError(f"Stitching requires at least 4 valid images, but found {len(images)}.")
            
            self._update_job_status(job_id, JobState.PROCESSING, 0.1, f"Loaded {len(images)} images. Starting Hugin pipeline...")
            
            capture_points = session_data.get('capturePoints', [])
            panorama, quality_metrics = self.stitcher.stitch_panorama(images, capture_points)
            
            self._update_job_status(job_id, JobState.PROCESSING, 0.95, "Stitching complete. Saving panorama...")
            
            output_path = OUTPUT_DIR / f"{job_id}_panorama.jpg"
            # Use Pillow for high-quality JPEG saving
            result_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
            Image.fromarray(result_rgb).save(str(output_path), 'JPEG', quality=95, optimize=True)
            
            base_url = os.environ.get('BASE_URL', 'https://hdri-panorama-server-production.up.railway.app').rstrip('/')
            result_url = f"{base_url}/v1/panorama/result/{job_id}"
            preview_url = f"{base_url}/v1/panorama/preview/{job_id}"
            
            logger.info(f"🎉 Panorama processing completed successfully!")
            logger.info(f"📥 Download: {result_url}")
            logger.info(f"👁️  Preview: {preview_url}")
            logger.info(f"📊 Quality Score: {quality_metrics['overallScore']:.3f}")
            logger.info(f"📐 Resolution: {quality_metrics['resolution']}")
            
            self._update_job_status(job_id, JobState.COMPLETED, 1.0, "Professional panorama ready!", 
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
    try:
        bundle_data = bundle_file.read()
        if not bundle_data.startswith(b"HDRI_BUNDLE_V1\n"):
            raise ValueError("Invalid bundle format")
        
        parts = bundle_data.split(b'\n', 2)
        image_count = int(parts[1])
        data = parts[2]
        
        image_files = []
        offset = 0
        for i in range(image_count):
            header_end = data.find(b'\n', offset)
            header = data[offset:header_end].decode()
            index, size_str = header.split(':')
            size = int(size_str)
            offset = header_end + 1
            
            image_data = data[offset : offset + size]
            filepath = upload_dir / f"image_{index}.jpg"
            filepath.write_bytes(image_data)
            image_files.append(str(filepath))
            offset += size
        
        return image_files
    except Exception as e:
        logger.error(f"Failed to extract bundle: {e}")
        return []

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
    
    if not image_files:
        logger.error("No images found in bundle - extraction failed")
        return jsonify({"error": "No images found in bundle"}), 400
    
    thread = threading.Thread(target=processor.process_session, args=(job_id, session_metadata, image_files))
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
    
    result_path = OUTPUT_DIR / f"{job_id}_panorama.jpg"
    if not result_path.exists():
        abort(404)
    
    return send_file(str(result_path), as_attachment=True, download_name=f"panorama_{job_id}.jpg")

@app.route('/v1/panorama/preview/<job_id>', methods=['GET'])
def preview_result(job_id: str):
    """Preview the panorama in browser without downloading"""
    with job_lock:
        job = jobs.get(job_id)
    if not job or job.get("state") != JobState.COMPLETED:
        abort(404)
    
    result_path = OUTPUT_DIR / f"{job_id}_panorama.jpg"
    if not result_path.exists():
        abort(404)
    
    logger.info(f"📸 Serving preview for job {job_id} - file size: {result_path.stat().st_size} bytes")
    return send_file(str(result_path), mimetype='image/jpeg')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "activeJobs": len([j for j in jobs.values() if j["state"] == JobState.PROCESSING])})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"🚀 Starting server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
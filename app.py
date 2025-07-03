#!/usr/bin/env python3
"""
HDRi 360 Studio - Local OpenCV Panorama Processing Server

This server receives 16-point ultra-wide camera captures from the iOS app
and processes them into professional 360° panoramas using OpenCV with SIFT
feature detection and multi-band blending.

API Endpoints:
- POST /v1/panorama/process - Upload session and images for processing
- GET /v1/panorama/status/<job_id> - Check processing status
- GET /v1/panorama/result/<job_id> - Download completed panorama
- DELETE /v1/panorama/cancel/<job_id> - Cancel processing job
- DELETE /v1/panorama/cleanup/<job_id> - Clean up job resources
"""

import os
import uuid
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from flask import Flask, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import requests
from advanced_stitcher import AdvancedPanoramaStitcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Server configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs") 
TEMP_DIR = Path("temp")

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# Job tracking
jobs: Dict[str, Dict] = {}
job_lock = threading.Lock()

class JobState:
    QUEUED = "queued"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

class PanoramaProcessor:
    """Professional OpenCV panorama stitching with SIFT and multi-band blending"""
    
    def __init__(self):
        self.stitcher = AdvancedPanoramaStitcher()
        
    def process_session(self, job_id: str, session_data: dict, image_files: List[str]) -> dict:
        """Process a complete panorama session"""
        
        try:
            self._update_job_status(job_id, JobState.PROCESSING, 0.0, "Loading and preparing images...")
            
            # Load and orient images
            images = []
            for i, img_path in enumerate(image_files):
                img = self._load_and_orient_image(img_path)
                if img is not None:
                    images.append(img)
                    progress = (i + 1) / len(image_files) * 0.3
                    self._update_job_status(job_id, JobState.PROCESSING, progress, f"Loaded image {i+1}/{len(image_files)}")
                else:
                    logger.warning(f"Failed to load image: {img_path}")
            
            if len(images) < 4:
                raise Exception("Need at least 4 valid images for panorama stitching")
            
            self._update_job_status(job_id, JobState.PROCESSING, 0.3, "Starting advanced OpenCV stitching...")
            
            # Extract capture point data
            capture_points = session_data.get('capturePoints', [])
            
            # Use advanced stitcher
            result, quality_metrics = self.stitcher.stitch_panorama(images, capture_points)
            
            self._update_job_status(job_id, JobState.PROCESSING, 0.95, "Saving processed panorama...")
            
            # Save result
            output_path = OUTPUT_DIR / f"{job_id}_panorama.jpg"
            cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            self._update_job_status(job_id, JobState.COMPLETED, 1.0, "Professional panorama ready!", 
                                  result_url=f"http://localhost:5001/v1/panorama/result/{job_id}",
                                  quality_metrics=quality_metrics)
            
            return {"success": True, "output_path": str(output_path)}
            
        except Exception as e:
            logger.error(f"Processing failed for job {job_id}: {str(e)}")
            self._update_job_status(job_id, JobState.FAILED, 0.0, f"Processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _load_and_orient_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load image and correct orientation based on EXIF data"""
        try:
            # Load with PIL to handle EXIF orientation
            pil_image = Image.open(img_path)
            
            # Auto-orient based on EXIF
            try:
                for orientation in ExifTags.ORIENTATION.values():
                    if orientation in pil_image._getexif():
                        if pil_image._getexif()[orientation] == 3:
                            pil_image = pil_image.rotate(180, expand=True)
                        elif pil_image._getexif()[orientation] == 6:
                            pil_image = pil_image.rotate(270, expand=True)
                        elif pil_image._getexif()[orientation] == 8:
                            pil_image = pil_image.rotate(90, expand=True)
                        break
            except (AttributeError, KeyError, TypeError):
                pass  # No EXIF data or orientation info
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return opencv_image
            
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {str(e)}")
            return None
    
    
    def _update_job_status(self, job_id: str, state: str, progress: float, message: str, 
                          result_url: str = None, quality_metrics: dict = None):
        """Update job status in thread-safe manner"""
        with job_lock:
            if job_id in jobs:
                jobs[job_id].update({
                    "state": state,
                    "progress": progress,
                    "message": message,
                    "lastUpdated": datetime.now(timezone.utc).isoformat()
                })
                if result_url:
                    jobs[job_id]["resultUrl"] = result_url
                if quality_metrics:
                    jobs[job_id]["qualityMetrics"] = quality_metrics

# Global processor instance
processor = PanoramaProcessor()

@app.route('/v1/panorama/process', methods=['POST'])
def process_panorama():
    """Upload session and start processing"""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Get session metadata
        session_metadata = json.loads(request.form.get('session_metadata', '{}'))
        
        # Create job entry
        with job_lock:
            jobs[job_id] = {
                "jobId": job_id,
                "state": JobState.QUEUED,
                "progress": 0.0,
                "message": "Processing queued",
                "sessionData": session_metadata,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "lastUpdated": datetime.now(timezone.utc).isoformat()
            }
        
        # Save uploaded images
        image_files = []
        upload_dir = UPLOAD_DIR / job_id
        upload_dir.mkdir(exist_ok=True)
        
        for key in request.files:
            if key.startswith('image_'):
                file = request.files[key]
                if file and file.filename:
                    filename = secure_filename(f"{key}.jpg")
                    filepath = upload_dir / filename
                    file.save(str(filepath))
                    image_files.append(str(filepath))
        
        if not image_files:
            return jsonify({"error": "No images uploaded"}), 400
        
        logger.info(f"Job {job_id}: Received {len(image_files)} images")
        
        # Start processing in background thread
        def process_in_background():
            processor.process_session(job_id, session_metadata, image_files)
        
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        # Return job started response
        return jsonify({
            "jobId": job_id,
            "status": "accepted",
            "estimatedProcessingTime": 300,  # 5 minutes estimate
            "queuePosition": 1
        }), 202
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/v1/panorama/status/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """Get processing status for a job"""
    with job_lock:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404
        
        job = jobs[job_id]
        
        response = {
            "jobId": job_id,
            "state": job["state"],
            "progress": job["progress"],
            "message": job["message"]
        }
        
        if "resultUrl" in job:
            response["resultUrl"] = job["resultUrl"]
        if "qualityMetrics" in job:
            response["qualityMetrics"] = job["qualityMetrics"]
        if job["state"] == JobState.FAILED and "error" in job:
            response["error"] = job["error"]
            
        return jsonify(response)

@app.route('/v1/panorama/result/<job_id>', methods=['GET'])
def download_result(job_id: str):
    """Download the processed panorama"""
    with job_lock:
        if job_id not in jobs:
            abort(404)
        
        job = jobs[job_id]
        if job["state"] != JobState.COMPLETED:
            abort(404)
    
    result_path = OUTPUT_DIR / f"{job_id}_panorama.jpg"
    if not result_path.exists():
        abort(404)
    
    return send_file(str(result_path), as_attachment=True, download_name=f"panorama_{job_id}.jpg")

@app.route('/v1/panorama/cancel/<job_id>', methods=['DELETE'])
def cancel_job(job_id: str):
    """Cancel a processing job"""
    with job_lock:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404
        
        job = jobs[job_id]
        if job["state"] in [JobState.QUEUED, JobState.PROCESSING]:
            job["state"] = JobState.FAILED
            job["message"] = "Cancelled by user"
            job["error"] = "Cancelled"
    
    return jsonify({"message": "Job cancelled"})

@app.route('/v1/panorama/cleanup/<job_id>', methods=['DELETE'])
def cleanup_job(job_id: str):
    """Clean up job resources"""
    try:
        # Remove from jobs dict
        with job_lock:
            if job_id in jobs:
                del jobs[job_id]
        
        # Clean up files
        upload_dir = UPLOAD_DIR / job_id
        if upload_dir.exists():
            import shutil
            shutil.rmtree(str(upload_dir))
        
        result_file = OUTPUT_DIR / f"{job_id}_panorama.jpg"
        if result_file.exists():
            result_file.unlink()
        
        return jsonify({"message": "Job cleaned up"})
        
    except Exception as e:
        logger.error(f"Cleanup failed for job {job_id}: {str(e)}")
        return jsonify({"error": "Cleanup failed"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "activeJobs": len([j for j in jobs.values() if j["state"] == JobState.PROCESSING]),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

if __name__ == '__main__':
    print("🌍 HDRi 360 Studio - OpenCV Panorama Processing Server")
    print("🚀 Starting server on http://localhost:5001")
    print("📸 Ready to process ultra-wide 360° panoramas!")
    
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
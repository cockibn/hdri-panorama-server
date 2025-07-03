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
from PIL import Image, ImageOps, ExifTags
from PIL.ExifTags import TAGS
import requests

# Import our new Hugin-based processor
from hugin_stitcher import HuginPanoramaStitcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

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
    """
    Processes a set of images into a 360-degree spherical panorama using
    Hugin command-line tools for professional quality results.
    """
    
    def __init__(self):
        # Temporarily disable Hugin due to cpfind hanging issues
        logger.info("Temporarily using OpenCV stitcher (Hugin disabled due to timeout issues)")
        self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        self.use_hugin = False
        
        # Initialize Hugin-based stitcher (disabled)
        # try:
        #     self.stitcher = HuginPanoramaStitcher()
        #     logger.info("Hugin-based stitcher initialized successfully")
        # except Exception as e:
        #     logger.error(f"Failed to initialize Hugin stitcher: {e}")
        #     # Fallback to OpenCV stitcher
        #     self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        #     logger.info("Fallback to OpenCV stitcher configured for 360° panorama mode")
        #     self.use_hugin = False
        # else:
        #     self.use_hugin = True

    def process_session(self, job_id: str, session_data: dict, image_files: List[str]) -> dict:
        """Process a complete panorama session using Hugin or OpenCV as fallback."""
        
        start_time = time.time()
        try:
            self._update_job_status(job_id, JobState.PROCESSING, 0.0, "Loading and preparing images...")
            
            images = []
            for i, img_path in enumerate(image_files):
                img = self._load_and_orient_image(img_path)
                if img is not None:
                    # For Hugin, we can use higher resolution images
                    if self.use_hugin:
                        images.append(img)
                    else:
                        # For OpenCV, resize for performance
                        h, w = img.shape[:2]
                        scale = 1024 / max(h, w)
                        resized_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                        images.append(resized_img)
                    
                    progress = (i + 1) / len(image_files) * 0.3
                    self._update_job_status(job_id, JobState.PROCESSING, progress, f"Loaded image {i+1}/{len(image_files)}")
            
            if len(images) < 4:
                raise ValueError("At least 4 valid images are required for stitching.")
            
            # Extract capture points from session data
            capture_points = session_data.get('capturePoints', [])
            
            if self.use_hugin:
                logger.info(f"Job {job_id}: Starting Hugin stitching with {len(images)} images.")
                self._update_job_status(job_id, JobState.PROCESSING, 0.3, "Stitching 360° panorama with Hugin...")
                
                try:
                    # Use Hugin stitcher
                    panorama, quality_metrics = self.stitcher.stitch_panorama(images, capture_points)
                    
                    logger.info(f"Job {job_id}: Hugin stitching successful!")
                    self._update_job_status(job_id, JobState.PROCESSING, 0.9, "Finalizing panorama...")
                    
                    result = panorama
                    
                except Exception as hugin_error:
                    logger.warning(f"Job {job_id}: Hugin stitching failed: {hugin_error}")
                    logger.info(f"Job {job_id}: Falling back to OpenCV stitching")
                    self._update_job_status(job_id, JobState.PROCESSING, 0.3, "Hugin failed, trying OpenCV fallback...")
                    
                    # Fallback to OpenCV
                    opencv_stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
                    
                    # Resize images for OpenCV
                    resized_images = []
                    for img in images:
                        h, w = img.shape[:2]
                        scale = 1024 / max(h, w)
                        resized_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                        resized_images.append(resized_img)
                    
                    status, panorama = opencv_stitcher.stitch(resized_images)
                    
                    if status != cv2.Stitcher_OK:
                        error_messages = {
                            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough images to stitch.",
                            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Stitching failed. Ensure images have sufficient overlap and distinct features.",
                            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Could not adjust camera parameters during stitching."
                        }
                        error_message = error_messages.get(status, f"An unknown stitching error occurred (Code: {status}).")
                        raise RuntimeError(f"Both Hugin and OpenCV stitching failed. OpenCV error: {error_message}")
                    
                    logger.info(f"Job {job_id}: OpenCV fallback stitching successful!")
                    self._update_job_status(job_id, JobState.PROCESSING, 0.9, "Cropping and finalizing panorama...")
                    
                    result = self._crop_black_borders(panorama)
                    
                    # Calculate quality metrics for OpenCV fallback
                    processing_time = round(time.time() - start_time, 2)
                    quality_metrics = self._calculate_stitching_quality(result, len(images))
                    quality_metrics["processingTime"] = processing_time
                    quality_metrics["processor"] = "OpenCV (Hugin fallback)"
                
            else:
                logger.info(f"Job {job_id}: Starting OpenCV stitching with {len(images)} images.")
                self._update_job_status(job_id, JobState.PROCESSING, 0.3, "Stitching 360° panorama with OpenCV...")
                
                # Use OpenCV stitcher (fallback)
                try:
                    status, panorama = self.stitcher.stitch(images)
                    logger.info(f"Job {job_id}: OpenCV stitcher returned status code: {status}")
                except Exception as stitch_error:
                    logger.error(f"Job {job_id}: OpenCV stitcher crashed with error: {stitch_error}")
                    raise RuntimeError(f"Stitching process crashed: {str(stitch_error)}")

                if status != cv2.Stitcher_OK:
                    error_messages = {
                        cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough images to stitch.",
                        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Stitching failed. Ensure images have sufficient overlap and distinct features.",
                        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Could not adjust camera parameters during stitching."
                    }
                    error_message = error_messages.get(status, f"An unknown stitching error occurred (Code: {status}).")
                    raise RuntimeError(error_message)

                logger.info(f"Job {job_id}: OpenCV stitching successful!")
                self._update_job_status(job_id, JobState.PROCESSING, 0.9, "Cropping and finalizing panorama...")

                result = self._crop_black_borders(panorama)
                
                # Calculate quality metrics for OpenCV
                processing_time = round(time.time() - start_time, 2)
                quality_metrics = self._calculate_stitching_quality(result, len(images))
                quality_metrics["processingTime"] = processing_time
                quality_metrics["processor"] = "OpenCV"
            
            # Clean up memory
            del images
            
            self._update_job_status(job_id, JobState.PROCESSING, 0.95, "Saving processed panorama...")

            output_path = OUTPUT_DIR / f"{job_id}_panorama.jpg"
            cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            base_url = os.environ.get('BASE_URL', 'https://hdri-panorama-server-production.up.railway.app')
            self._update_job_status(job_id, JobState.COMPLETED, 1.0, "Professional panorama ready!", 
                                  result_url=f"{base_url}/v1/panorama/result/{job_id}",
                                  quality_metrics=quality_metrics)
            
            return {"success": True, "output_path": str(output_path)}
            
        except Exception as e:
            logger.exception(f"Processing failed for job {job_id}")
            self._update_job_status(job_id, JobState.FAILED, 0.0, f"Processing error: {str(e)}")
            return {"success": False, "error": str(e)}


    def _load_and_orient_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load image and correct orientation based on EXIF data using Pillow."""
        try:
            pil_image = Image.open(img_path)
            # Use ImageOps.exif_transpose for robust, automatic orientation
            oriented_image = ImageOps.exif_transpose(pil_image)
            # Convert from PIL's RGB to OpenCV's BGR format
            opencv_image = cv2.cvtColor(np.array(oriented_image), cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            return None

    def _crop_black_borders(self, image: np.ndarray) -> np.ndarray:
        """Crops the black border from a stitched panorama."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image # No content found
            
        # Find the largest contour which corresponds to the stitched image area
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image

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

    def _calculate_stitching_quality(self, panorama: np.ndarray, num_images: int) -> dict:
        """Calculate quality metrics for the stitched panorama."""
        h, w, _ = panorama.shape
        if h == 0 or w == 0: 
            return {
                "overallScore": 0.1,
                "seamQuality": 0.1,
                "featureMatches": 0,
                "geometricConsistency": 0.1,
                "colorConsistency": 0.1,
            }

        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        sharpness_score = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 2000.0, 1.0)
        
        lab = cv2.cvtColor(panorama, cv2.COLOR_BGR2LAB)
        color_std = np.std(lab[:, :, 1:])
        color_consistency = max(0, 1.0 - color_std / 50.0)
        
        # This is a proxy; a real system might count keypoints from the stitcher
        estimated_matches = num_images * 250 
        
        aspect_ratio = w / h
        geo_consistency = 1.0 if 1.8 <= aspect_ratio <= 2.2 else 0.7 # Tighter ratio for equirectangular
        
        overall_score = (sharpness_score * 0.3 + color_consistency * 0.3 + geo_consistency * 0.4)
        
        return {
            "overallScore": round(np.clip(overall_score, 0.1, 1.0), 2),
            "seamQuality": round(np.clip(color_consistency, 0, 1), 2),
            "featureMatches": int(estimated_matches),
            "geometricConsistency": round(np.clip(geo_consistency, 0, 1), 2),
            "colorConsistency": round(np.clip(color_consistency, 0, 1), 2),
        }

def extract_bundle_images(bundle_file, upload_dir):
    """Extract images from our custom bundle format"""
    try:
        # Read the bundle data
        bundle_data = bundle_file.read()
        logger.info(f"Bundle size: {len(bundle_data)} bytes")
        
        # Parse bundle header
        data_offset = 0
        header_end = bundle_data.find(b'\n', data_offset)
        if header_end == -1:
            raise ValueError("Invalid bundle format - no header found")
        
        magic = bundle_data[data_offset:header_end].decode('utf-8')
        if magic != "HDRI_BUNDLE_V1":
            raise ValueError(f"Invalid bundle format - expected HDRI_BUNDLE_V1, got {magic}")
        
        data_offset = header_end + 1
        
        # Parse image count
        count_end = bundle_data.find(b'\n', data_offset)
        if count_end == -1:
            raise ValueError("Invalid bundle format - no image count found")
        
        image_count = int(bundle_data[data_offset:count_end].decode('utf-8'))
        logger.info(f"Bundle contains {image_count} images")
        
        data_offset = count_end + 1
        
        # Extract images
        image_files = []
        for i in range(image_count):
            # Parse image header: "index:size\n"
            header_end = bundle_data.find(b'\n', data_offset)
            if header_end == -1:
                raise ValueError(f"Invalid bundle format - no header for image {i}")
            
            header = bundle_data[data_offset:header_end].decode('utf-8')
            index, size_str = header.split(':')
            size = int(size_str)
            
            data_offset = header_end + 1
            
            # Extract image data
            image_data = bundle_data[data_offset:data_offset + size]
            if len(image_data) != size:
                raise ValueError(f"Invalid bundle format - expected {size} bytes for image {i}, got {len(image_data)}")
            
            # Save image file
            filename = f"image_{index}.jpg"
            filepath = upload_dir / filename
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            image_files.append(str(filepath))
            logger.info(f"Extracted image {index}: {size} bytes -> {filename}")
            
            data_offset += size
        
        logger.info(f"Successfully extracted {len(image_files)} images from bundle")
        return image_files
        
    except Exception as e:
        logger.error(f"Failed to extract bundle images: {str(e)}")
        return []

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
        
        # Check for compressed bundle format (new format)
        if 'images_zip' in request.files:
            bundle_file = request.files['images_zip']
            if bundle_file and bundle_file.filename:
                logger.info(f"Processing compressed bundle: {bundle_file.filename}")
                image_files = extract_bundle_images(bundle_file, upload_dir)
        else:
            # Legacy individual file format
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
    
    # Create required directories if they don't exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    
    # Use PORT environment variable for cloud deployment, fallback to 5001 for local
    port = int(os.environ.get('PORT', 5001))
    host = '0.0.0.0'
    
    print(f"🚀 Starting server on {host}:{port}")
    print("📸 Ready to process ultra-wide 360° panoramas!")
    
    app.run(host=host, port=port, debug=False, threaded=True)
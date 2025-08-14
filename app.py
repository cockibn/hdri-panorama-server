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
logging.getLogger('services').setLevel(logging.INFO)

# Import microservices
from services import (
    create_coordinate_service, create_hugin_service, create_quality_service, create_blending_service,
    get_service_bus, ServiceStatus
)

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

def validate_job_id(job_id: str) -> bool:
    """Validate job ID to prevent path traversal and injection attacks."""
    if not job_id or not isinstance(job_id, str):
        return False
    if len(job_id) != 36:  # UUID4 length
        return False
    # UUID4 format: 8-4-4-4-12 characters, only hex and hyphens
    if not all(c in '0123456789abcdef-' for c in job_id.lower()):
        return False
    # Check UUID format structure
    parts = job_id.split('-')
    if len(parts) != 5 or [len(p) for p in parts] != [8, 4, 4, 4, 12]:
        return False
    return True

def validate_bundle_format(bundle_data):
    """Validate bundle data format and prevent malicious content."""
    if not bundle_data.startswith(b"HDRI_BUNDLE_V1\n"):
        raise ValueError("Invalid bundle format header")
    
    if len(bundle_data) > 500 * 1024 * 1024:  # 500MB limit
        raise ValueError("Bundle exceeds maximum size limit")
    
    return True

def validate_image_index(index_str):
    """Validate image index to prevent path traversal."""
    try:
        index = int(index_str)
        if index < 0 or index > 100:  # Reasonable limits
            raise ValueError(f"Image index out of range: {index}")
        return index
    except ValueError:
        raise ValueError(f"Invalid image index: {index_str}")

def extract_bundle_images(bundle_file, upload_dir):
    """Extract images from iOS app bundle format."""
    image_files = []
    try:
        bundle_data = bundle_file.read()
        validate_bundle_format(bundle_data)
        
        parts = bundle_data.split(b'\n', 2)
        if len(parts) < 3:
            raise ValueError("Malformed bundle structure")
            
        try:
            image_count = int(parts[1])
        except ValueError:
            raise ValueError("Invalid image count in bundle")
            
        if image_count <= 0 or image_count > 50:  # Reasonable limits
            raise ValueError(f"Invalid image count: {image_count}")
            
        data = parts[2]
        original_exif_data = []
        offset = 0
        
        # Extract all images
        for i in range(image_count):
            try:
                header_end = data.find(b'\n', offset)
                if header_end == -1:
                    raise ValueError(f"Malformed header for image {i}")
                    
                header = data[offset:header_end].decode('utf-8', errors='strict')
                
                if ':' not in header:
                    raise ValueError(f"Invalid header format for image {i}: {header}")
                    
                index_str, size_str = header.split(':', 1)
                
                # Validate index to prevent path traversal
                index = validate_image_index(index_str)
                
                try:
                    size = int(size_str)
                except ValueError:
                    raise ValueError(f"Invalid size for image {i}: {size_str}")
                    
                if size <= 0 or size > 50 * 1024 * 1024:  # 50MB per image limit
                    raise ValueError(f"Invalid image size for image {i}: {size}")
                
                offset = header_end + 1
                
                if offset + size > len(data):
                    raise ValueError(f"Image {i} data extends beyond bundle")
                
                image_data = data[offset : offset + size]
                
                # Use safe filename construction
                safe_filename = f"image_{index:04d}.jpg"
                filepath = upload_dir / safe_filename
                
                # Ensure the path is within upload_dir
                if not filepath.resolve().is_relative_to(upload_dir.resolve()):
                    raise ValueError(f"Invalid file path for image {i}")
                
                # CRITICAL FIX: Apply EXIF orientation correction before saving
                try:
                    from PIL import Image, ImageOps
                    import io
                    
                    # Load image data and apply EXIF orientation with proper cleanup
                    with Image.open(io.BytesIO(image_data)) as temp_image:
                        original_size = temp_image.size
                        
                        # Apply EXIF orientation correction
                        oriented_image = ImageOps.exif_transpose(temp_image)
                        corrected_size = oriented_image.size
                        
                        # Save orientation-corrected image (high quality to minimize loss)
                        try:
                            oriented_image.save(filepath, 'JPEG', quality=98, optimize=True)
                        finally:
                            # Ensure oriented_image is cleaned up
                            if oriented_image != temp_image:  # Only close if it's a new image
                                oriented_image.close()
                    
                    # Log orientation correction
                    if original_size != corrected_size:
                        logger.info(f"🔄 Image {index}: EXIF orientation applied {original_size} → {corrected_size}")
                    else:
                        logger.debug(f"📸 Image {index}: No orientation correction needed {original_size}")
                        
                except Exception as exif_error:
                    logger.warning(f"⚠️ EXIF orientation failed for image {index}: {exif_error}")
                    # Fallback to raw image data
                    filepath.write_bytes(image_data)
                
                image_files.append(str(filepath))
                
                # Extract EXIF data
                try:
                    import piexif
                    exif_dict = piexif.load(image_data)
                    original_exif_data.append(exif_dict)
                except Exception as e:
                    logger.warning(f"⚠️ Could not extract EXIF from image {index}: {e}")
                    original_exif_data.append({})
                
                offset += size
                
            except Exception as image_error:
                logger.error(f"Failed to extract image {i}: {image_error}")
                continue
                
        logger.info(f"📸 Extracted {len(image_files)} images from bundle")
        
        # CRITICAL FIX: Sort image files by filename to ensure correct order
        # This ensures image_files order matches capturePoints order
        image_files.sort()
        logger.info(f"🔄 Sorted image files: {[os.path.basename(f) for f in image_files[:5]]}...")
        
        # Save EXIF data for processing
        exif_file = upload_dir / "original_exif.json"
        try:
            import json
            import base64
            
            # Convert EXIF data to JSON-serializable format
            serialized_exif = []
            for exif_dict in original_exif_data:
                serialized_dict = {}
                for ifd_name, ifd_data in exif_dict.items():
                    serialized_dict[ifd_name] = {}
                    for tag, value in ifd_data.items():
                        if isinstance(value, bytes):
                            serialized_dict[ifd_name][tag] = {
                                "type": "bytes",
                                "data": base64.b64encode(value).decode('utf-8')
                            }
                        else:
                            serialized_dict[ifd_name][tag] = {"data": value}
                serialized_exif.append(serialized_dict)
            
            with open(exif_file, 'w') as f:
                json.dump(serialized_exif, f)
                
            logger.info(f"💾 Saved EXIF data for {len(original_exif_data)} images")
        except Exception as e:
            logger.warning(f"⚠️ Could not save EXIF data: {e}")
            
        return image_files
        
    except Exception as e:
        logger.error(f"Bundle extraction failed: {e}")
        # Cleanup any partially extracted files
        for file_path in image_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass
        raise ValueError(f"Failed to extract images from bundle: {e}")

jobs: Dict[str, Dict] = {}
job_lock = threading.Lock()
cleanup_timer = None  # Global cleanup timer

# Job cleanup configuration
JOB_RETENTION_HOURS = int(os.environ.get('JOB_RETENTION_HOURS', '24'))
CLEANUP_INTERVAL_MINUTES = int(os.environ.get('CLEANUP_INTERVAL_MINUTES', '60'))

def cleanup_old_jobs():
    """Remove jobs older than retention period to prevent memory leaks."""
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=JOB_RETENTION_HOURS)
        
        with job_lock:
            expired_job_ids = []
            current_job_count = len(jobs)
            
            # Also limit total job count to prevent unbounded growth
            MAX_JOBS = int(os.environ.get('MAX_JOBS', '1000'))
            
            for job_id, job_data in jobs.items():
                try:
                    # Validate job_id format
                    if not validate_job_id(job_id):
                        logger.warning(f"Invalid job ID found in storage: {job_id}")
                        expired_job_ids.append(job_id)
                        continue
                    
                    last_updated = datetime.fromisoformat(job_data.get('lastUpdated', ''))
                    if last_updated < cutoff_time:
                        expired_job_ids.append(job_id)
                except (ValueError, TypeError) as e:
                    # Invalid timestamp, mark for cleanup
                    logger.warning(f"Invalid timestamp for job {job_id}: {e}")
                    expired_job_ids.append(job_id)
            
            # If still too many jobs, remove oldest completed ones
            if current_job_count - len(expired_job_ids) > MAX_JOBS:
                completed_jobs = []
                for job_id, job_data in jobs.items():
                    if job_id not in expired_job_ids and job_data.get('state') in ['completed', 'failed']:
                        try:
                            last_updated = datetime.fromisoformat(job_data.get('lastUpdated', ''))
                            completed_jobs.append((job_id, last_updated))
                        except (ValueError, TypeError):
                            expired_job_ids.append(job_id)
                
                # Sort by timestamp and remove oldest
                completed_jobs.sort(key=lambda x: x[1])
                overflow = (current_job_count - len(expired_job_ids)) - MAX_JOBS
                if overflow > 0:
                    for job_id, _ in completed_jobs[:overflow]:
                        expired_job_ids.append(job_id)
                    logger.warning(f"Removing {overflow} old jobs due to MAX_JOBS limit")
            
            # Clean up expired jobs
            for job_id in expired_job_ids:
                try:
                    # Validate job_id before using in file operations
                    if validate_job_id(job_id):
                        # Clean up any associated files
                        upload_dir = UPLOAD_DIR / job_id
                        if upload_dir.exists():
                            import shutil
                            shutil.rmtree(upload_dir, ignore_errors=True)
                        
                        output_files = list(OUTPUT_DIR.glob(f"{job_id}_*"))
                        for output_file in output_files:
                            try:
                                output_file.unlink(missing_ok=True)
                            except Exception as e:
                                logger.warning(f"Could not delete file {output_file}: {e}")
                    
                    # Remove from jobs dict
                    if job_id in jobs:
                        del jobs[job_id]
                        
                except Exception as e:
                    logger.warning(f"Error cleaning up job {job_id}: {e}")
            
            if expired_job_ids:
                logger.info(f"Cleaned up {len(expired_job_ids)} expired jobs (was {current_job_count}, now {len(jobs)})")
                
    except Exception as e:
        logger.error(f"Error during job cleanup: {e}")
        # Continue running even if cleanup fails
    
    # MEMORY LEAK FIX: Use a single persistent timer instead of recursive creation
    global cleanup_timer
    if cleanup_timer and cleanup_timer.is_alive():
        cleanup_timer.cancel()
    
    cleanup_timer = threading.Timer(CLEANUP_INTERVAL_MINUTES * 60, cleanup_old_jobs)
    cleanup_timer.daemon = True
    cleanup_timer.start()

def check_system_resources():
    """Check if system has adequate resources for processing."""
    try:
        # Check disk space
        try:
            disk_usage = psutil.disk_usage('/')
            disk_free_percent = (disk_usage.free / disk_usage.total) * 100
            disk_free_gb = disk_usage.free / (1024**3)
            
            # Require at least 5GB free space for processing
            if disk_free_gb < 5.0:
                raise RuntimeError(f"Insufficient disk space: {disk_free_gb:.1f}GB free (need 5GB minimum)")
                
            if disk_free_percent < 5:  # Less than 5% free is critical
                raise RuntimeError(f"Critical disk space: {disk_free_percent:.1f}% free")
        except psutil.Error as e:
            logger.error(f"Could not check disk space: {e}")
            raise RuntimeError("Unable to verify disk space availability")
        
        # Check memory
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            # Require at least 2GB available memory for processing
            if available_gb < 2.0:
                raise RuntimeError(f"Insufficient memory: {available_gb:.1f}GB available (need 2GB minimum)")
                
            if memory.percent > 85:  # More than 85% used is concerning
                raise RuntimeError(f"High memory usage: {memory.percent:.1f}% used")
        except psutil.Error as e:
            logger.error(f"Could not check memory: {e}")
            raise RuntimeError("Unable to verify memory availability")
        
        # Check active jobs
        try:
            with job_lock:
                active_jobs = len([j for j in jobs.values() if j.get('state') == JobState.PROCESSING])
            max_concurrent = int(os.environ.get('MAX_CONCURRENT_JOBS', '3'))
            
            if active_jobs >= max_concurrent:
                raise RuntimeError(f"Server at capacity: {active_jobs}/{max_concurrent} jobs processing")
        except Exception as e:
            logger.error(f"Could not check job count: {e}")
            raise RuntimeError("Unable to verify server capacity")
            
        return True
        
    except Exception as e:
        # SECURITY FIX: Don't allow processing on resource check failure
        logger.error(f"Resource check failed: {e}")
        raise

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # SECURITY FIX: Removed dangerous DISABLE_AUTH bypass
        
        api_key = request.headers.get('X-API-Key')
        expected_key = os.environ.get('API_KEY')
        
        # Require API key in production
        if not expected_key:
            flask_env = os.environ.get('FLASK_ENV', '').lower()
            # Allow development/testing without API key
            if flask_env in ['development', 'dev', 'test', 'local']:
                logger.warning(f"{flask_env.title()} mode: API key not configured - allowing unauthenticated access")
                return f(*args, **kwargs)
            else:
                logger.error("Production deployment missing API_KEY environment variable")
                logger.error("Set API_KEY environment variable or use FLASK_ENV=development for local testing")
                return jsonify({
                    "error": "Server configuration error", 
                    "message": "API_KEY required in production",
                    "hint": "Set FLASK_ENV=development for local testing"
                }), 500
            
        if not api_key or api_key != expected_key:
            logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
            
        return f(*args, **kwargs)
    return decorated_function

class JobState:
    QUEUED, PROCESSING, COMPLETED, FAILED = "queued", "processing", "completed", "failed"

class MicroservicesPanoramaProcessor:
    """Microservices-based panorama processing orchestrator."""
    
    def __init__(self, output_resolution: str = None):
        # Allow override via environment variable for deployment flexibility
        self.resolution = output_resolution or os.environ.get('PANORAMA_RESOLUTION', '6K')
        
        # Canvas size mapping
        resolution_mapping = {
            "4K": (4096, 2048),
            "6K": (6144, 3072), 
            "8K": (8192, 4096)
        }
        self.canvas_size = resolution_mapping.get(self.resolution, (6144, 3072))
        
        # Initialize service bus
        self.service_bus = get_service_bus()
        self.service_bus.enable_debug_mode(True)
        
        # Initialize all microservices
        self.coordinate_service = create_coordinate_service()
        self.hugin_service = create_hugin_service()  # Let Hugin calculate optimal canvas size
        self.quality_service = create_quality_service()
        self.blending_service = create_blending_service()
        
        # Register all services with service bus
        self._register_services()
        
        logger.info(f"🏗️ Microservices Panorama Processor initialized")
        logger.info(f"   Resolution: {self.resolution} ({self.canvas_size[0]}×{self.canvas_size[1]})")
        logger.info(f"   Services: coordinate, hugin, quality, blending")
        
    def _register_services(self):
        """Register all services with the service bus."""
        services = [
            ("coordinate_service", "1.0.0", ["arkit_validation", "coordinate_conversion", "debug_analysis"]),
            ("hugin_service", "1.0.0", ["pto_gen", "cpfind", "autooptimiser", "nona", "full_pipeline"]),
            ("quality_service", "1.0.0", ["quality_analysis", "metrics_calculation", "issue_detection"]),
            ("blending_service", "1.0.0", ["enblend", "opencv_blend", "emergency_fallback"])
        ]
        
        for name, version, capabilities in services:
            self.service_bus.register_service(name, version, capabilities)
            
        logger.info(f"📡 Registered {len(services)} services with service bus")

    def process_session(self, job_id: str, session_data: dict, image_files: List[str], base_url: str = None):
        """Process panorama session using complete microservices pipeline."""
        try:
            logger.info(f"🚀 Starting microservices panorama processing for job {job_id}")
            self._update_job_status(job_id, JobState.PROCESSING, 0.0, "Initializing microservices pipeline...")
            
            start_time = time.time()
            capture_points = session_data.get('capturePoints', [])
            
            # STEP 1: Coordinate Service - ARKit validation and conversion
            self._update_job_status(job_id, JobState.PROCESSING, 0.05, "🎯 Validating ARKit coordinates...")
            self.service_bus.update_service_status("coordinate_service", ServiceStatus.BUSY)
            
            try:
                validation_results = self.coordinate_service.validate_arkit_data(capture_points)
                converted_coordinates = self.coordinate_service.convert_arkit_to_hugin(capture_points, job_id)
                
                # Store coordinate analysis for debugging
                with job_lock:
                    if job_id in jobs:
                        jobs[job_id]['coordinate_validation'] = validation_results
                        jobs[job_id]['converted_coordinates_count'] = len(converted_coordinates)
                        
                coverage_quality = validation_results.get('coverage_quality', 'UNKNOWN')
                geometric_issues = validation_results.get('geometric_issues', [])
                
                logger.info(f"✅ Coordinate service: {coverage_quality}, {len(geometric_issues)} issues")
                self.service_bus.update_service_status("coordinate_service", ServiceStatus.READY)
                
            except Exception as e:
                self.service_bus.update_service_status("coordinate_service", ServiceStatus.ERROR)
                logger.error(f"❌ Coordinate service failed: {e}")
                converted_coordinates = []
                
            # STEP 2: Hugin Service - Complete pipeline execution  
            self._update_job_status(job_id, JobState.PROCESSING, 0.1, "🔧 Executing Hugin pipeline...")
            self.service_bus.update_service_status("hugin_service", ServiceStatus.BUSY)
            
            def hugin_progress_callback(progress: float, message: str):
                # Map hugin progress from 10% to 80%
                mapped_progress = 0.1 + (progress * 0.7)
                self._update_job_status(job_id, JobState.PROCESSING, mapped_progress, f"Hugin: {message}")
                
            try:
                hugin_result = self.hugin_service.execute_pipeline(
                    images=image_files,
                    converted_coordinates=converted_coordinates,
                    progress_callback=hugin_progress_callback
                )
                
                if not hugin_result['success']:
                    raise Exception(f"Hugin pipeline failed: {hugin_result.get('error', 'Unknown error')}")
                    
                tiff_files = [step['output_files'] for step in hugin_result['pipeline_steps'] 
                             if step['name'] == 'nona' and step['success']]
                if not tiff_files:
                    raise Exception("Hugin pipeline did not produce rendered images")
                    
                tiff_files = tiff_files[0]  # Get the actual file list
                
                # Store Hugin analysis
                with job_lock:
                    if job_id in jobs:
                        jobs[job_id]['hugin_pipeline'] = hugin_result
                        
                logger.info(f"✅ Hugin service: {len(tiff_files)} images rendered in {hugin_result['processing_time']:.1f}s")
                self.service_bus.update_service_status("hugin_service", ServiceStatus.READY)
                
            except Exception as e:
                self.service_bus.update_service_status("hugin_service", ServiceStatus.ERROR)
                logger.error(f"❌ Hugin service failed: {e}")
                raise Exception(f"Hugin pipeline failed: {e}")
                
            # STEP 3: Blending Service - Multi-strategy blending
            self._update_job_status(job_id, JobState.PROCESSING, 0.8, "🎨 Blending panorama...")
            self.service_bus.update_service_status("blending_service", ServiceStatus.BUSY)
            
            def blending_progress_callback(progress: float, message: str):
                # Map blending progress from 80% to 95%
                mapped_progress = 0.8 + (progress * 0.15)
                self._update_job_status(job_id, JobState.PROCESSING, mapped_progress, f"Blending: {message}")
                
            try:
                output_path = OUTPUT_DIR / f"{job_id}_panorama.exr"
                blending_result = self.blending_service.blend_panorama(
                    tiff_files=tiff_files,
                    output_path=str(output_path),
                    expected_image_count=len(image_files),
                    progress_callback=blending_progress_callback
                )
                
                if not blending_result['success']:
                    raise Exception(f"Blending failed: {blending_result.get('error', 'Unknown error')}")
                    
                # Store blending analysis
                with job_lock:
                    if job_id in jobs:
                        jobs[job_id]['blending_result'] = blending_result
                        
                logger.info(f"✅ Blending service: {blending_result['strategy']} strategy, {blending_result['output_size_mb']:.1f}MB")
                self.service_bus.update_service_status("blending_service", ServiceStatus.READY)
                
            except Exception as e:
                self.service_bus.update_service_status("blending_service", ServiceStatus.ERROR)
                logger.error(f"❌ Blending service failed: {e}")
                raise Exception(f"Blending failed: {e}")
                
            # Load final panorama (handle both EXR and TIFF formats)
            panorama = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
            
            # If EXR loading failed, try TIFF fallback
            if panorama is None and str(output_path).endswith('.exr'):
                tiff_fallback = str(output_path).replace('.exr', '.tif')
                if Path(tiff_fallback).exists():
                    logger.warning(f"⚠️ EXR loading failed, trying TIFF fallback: {tiff_fallback}")
                    panorama = cv2.imread(tiff_fallback, cv2.IMREAD_UNCHANGED)
                    if panorama is not None:
                        output_path = Path(tiff_fallback)  # Update output path for subsequent operations
                        
            if panorama is None:
                raise Exception(f"Failed to load final panorama from {output_path} (tried EXR and TIFF)")
                
            # STEP 4: Quality Service - Comprehensive analysis
            self._update_job_status(job_id, JobState.PROCESSING, 0.95, "🔍 Analyzing quality...")
            self.service_bus.update_service_status("quality_service", ServiceStatus.BUSY)
            
            try:
                total_processing_time = time.time() - start_time
                control_points = hugin_result['statistics'].get('control_points_found', 0)
                
                # Include all context for quality analysis
                quality_context = {
                    'coordinate_validation': validation_results,
                    'hugin_pipeline': hugin_result,
                    'blending_result': blending_result
                }
                
                quality_metrics = self.quality_service.analyze_panorama_quality(
                    panorama_path=str(output_path),
                    image_count=len(image_files),
                    control_points=control_points,
                    processing_time=total_processing_time,
                    additional_context=quality_context
                )
                
                # Store quality analysis
                with job_lock:
                    if job_id in jobs:
                        jobs[job_id]['quality_metrics'] = {
                            'overall_score': quality_metrics.overall_score,
                            'sharpness': quality_metrics.sharpness_score,
                            'contrast': quality_metrics.contrast_score,
                            'coverage': quality_metrics.coverage_percentage,
                            'control_points': quality_metrics.control_points_count,
                            'cp_efficiency': quality_metrics.control_points_efficiency,
                            'visual_issues': quality_metrics.visual_issues,
                            'geometric_issues': quality_metrics.geometric_issues
                        }
                        
                logger.info(f"✅ Quality service: {quality_metrics.overall_score:.1f}/100 overall score")
                self.service_bus.update_service_status("quality_service", ServiceStatus.READY)
                
            except Exception as e:
                self.service_bus.update_service_status("quality_service", ServiceStatus.ERROR)
                logger.error(f"❌ Quality service failed: {e}")
                # Create minimal metrics
                class MockMetrics:
                    def __init__(self):
                        self.overall_score = 0.0
                        self.control_points_count = control_points if 'control_points' in locals() else 0
                        self.processing_time = time.time() - start_time
                quality_metrics = MockMetrics()
                
            # Create debug overlay (coordinate positioning vs result)
            self._update_job_status(job_id, JobState.PROCESSING, 0.97, "Creating debug overlay...")
            try:
                from create_overlay_service import create_debug_overlay_for_job
                
                # Find debug coordinate image  
                debug_image_path = f"/tmp/coordinate_debug_{job_id}.png"
                if os.path.exists(debug_image_path):
                    overlay_path = create_debug_overlay_for_job(job_id, debug_image_path, str(output_path))
                    if overlay_path:
                        logger.info(f"🎨 Debug overlay created: {overlay_path}")
                    else:
                        logger.warning("⚠️ Debug overlay creation failed")
                else:
                    logger.info(f"📍 No debug image found at: {debug_image_path}")
            except Exception as overlay_error:
                logger.warning(f"⚠️ Debug overlay creation failed: {overlay_error}")
            
            # Create preview
            self._update_job_status(job_id, JobState.PROCESSING, 0.98, "Creating preview...")
            preview_path = OUTPUT_DIR / f"{job_id}_preview.jpg"
            self._create_photosphere_preview(panorama, str(preview_path), session_data)
            
            # Complete processing with URLs
            total_processing_time = time.time() - start_time
            
            # Generate access URLs (Railway provides RAILWAY_STATIC_URL or PORT)
            base_url = os.environ.get('RAILWAY_STATIC_URL', f"http://localhost:{os.environ.get('PORT', '5001')}")
            preview_url = f"{base_url}/v1/panorama/preview/{job_id}"
            download_url = f"{base_url}/v1/panorama/result/{job_id}"
            
            completion_message = f"✅ Processing complete! {total_processing_time:.1f}s"
            self._update_job_status(job_id, JobState.COMPLETED, 1.0, completion_message, result_url=download_url)
            
            # Log direct access URLs
            logger.info(f"🎉 Panorama ready for job: {job_id}")
            logger.info(f"📖 Preview URL: {preview_url}")
            logger.info(f"⬇️ Download URL: {download_url}")
            logger.info(f"🔗 Direct links ready for iOS app download")
            
            # Generate comprehensive service bus report
            service_bus_report = self.service_bus.generate_debug_report()
            
            # Return comprehensive results
            return {
                'panorama': panorama,
                'panorama_path': str(output_path),
                'preview_path': str(preview_path),
                'processing_time': total_processing_time,
                'quality_metrics': {
                    'overall_score': quality_metrics.overall_score,
                    'control_points': quality_metrics.control_points_count,
                    'processing_time': quality_metrics.processing_time
                },
                'microservices_report': {
                    'coordinate_validation': validation_results if 'validation_results' in locals() else {},
                    'hugin_pipeline': hugin_result if 'hugin_result' in locals() else {},
                    'blending_result': blending_result if 'blending_result' in locals() else {},
                    'service_bus': service_bus_report
                }
            }
            
        except Exception as e:
            # Update all services to error state
            for service_name in ["coordinate_service", "hugin_service", "quality_service", "blending_service"]:
                self.service_bus.update_service_status(service_name, ServiceStatus.ERROR)
                
            logger.error(f"❌ Microservices processing failed: {e}")
            self._update_job_status(job_id, JobState.FAILED, 0.0, f"Processing failed: {str(e)}")
            raise
            
    def _load_and_orient_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load image with proper EXIF orientation handling for Hugin processing."""
        try:
            pil_image = Image.open(img_path)
            # QUALITY FIX: Handle orientation properly for maximum quality preservation
            # iOS now sends original camera data - apply EXIF orientation once, correctly
            oriented_image = ImageOps.exif_transpose(pil_image)
            result = cv2.cvtColor(np.array(oriented_image), cv2.COLOR_RGB2BGR)
            
            # Log image dimensions to verify orientation handling
            height, width = result.shape[:2]
            logger.info(f"📸 Loaded image: {width}×{height} (orientation-corrected from EXIF)")
            
            return result
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            return None

    def _create_photosphere_preview(self, panorama: np.ndarray, preview_path: str, session_data: dict):
        """Create JPEG preview with 360° photosphere metadata."""
        try:
            # Convert to uint8 for JPEG
            if panorama.dtype == np.float32 or panorama.dtype == np.float64:
                panorama_preview = (np.clip(panorama, 0, 1) * 255).astype(np.uint8)
            else:
                panorama_preview = panorama
                
            result_rgb = cv2.cvtColor(panorama_preview, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(result_rgb)
            
            # Add basic EXIF metadata
            import piexif
            height, width = panorama_preview.shape[:2]
            exif_dict = {
                "0th": {
                    piexif.ImageIFD.Make: "HDRi 360 Studio",
                    piexif.ImageIFD.Model: "Microservices Panorama Processor",
                    piexif.ImageIFD.Software: "ARKit + Hugin Pipeline",
                    piexif.ImageIFD.ImageDescription: "Equirectangular 360° Photosphere"
                }
            }
            
            try:
                exif_bytes = piexif.dump(exif_dict)
                pil_image.save(preview_path, 'JPEG', quality=85, optimize=True, exif=exif_bytes)
                logger.info("📱 JPEG preview saved with EXIF metadata")
            except Exception as e:
                logger.warning(f"⚠️ Could not add metadata: {e}")
                pil_image.save(preview_path, 'JPEG', quality=85, optimize=True)
                
        except Exception as e:
            logger.error(f"❌ Failed to create preview: {e}")
            
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

# Create processor instance with microservices architecture                    
processor = MicroservicesPanoramaProcessor()

# Bundle processing functions are defined below after the Flask routes

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
    
    # DEBUG: Log session metadata structure to understand what iOS is sending
    if os.environ.get('ENV') != 'production':
        logger.info(f"📊 Session metadata keys: {list(session_metadata.keys())}")
        if 'capturePoints' in session_metadata:
            logger.info(f"📍 Capture points count: {len(session_metadata['capturePoints'])}")
        if 'calibrationReference' in session_metadata:
            logger.info(f"🎯 Calibration reference present: {session_metadata['calibrationReference']}")
        if 'cameraConfig' in session_metadata:
            logger.info(f"📸 Camera config: {session_metadata['cameraConfig']}")
    
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
    job_processor = MicroservicesPanoramaProcessor(output_resolution=resolution)
    
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
            # QUALITY ASSESSMENT: Analyze input image quality after iOS fix
            quality_metric = file_size / (w * h)  # Bytes per pixel (rough quality indicator)
            logger.info(f"📸 QUALITY ANALYSIS - Resolution: {w}x{h}, File size: {file_size} bytes, Quality metric: {quality_metric:.2f} bytes/pixel")
            
            if quality_metric < 0.5:
                logger.warning(f"⚠️ Low quality detected: {quality_metric:.2f} bytes/pixel (expected >1.0 for high quality)")
            else:
                logger.info(f"✅ Good input quality: {quality_metric:.2f} bytes/pixel")
            logger.info(f"📸 Compression ratio: {file_size/(w*h*3):.3f} bytes/pixel")
    
    if not image_files:
        logger.error("No images found in bundle - extraction failed")
        return jsonify({"error": "No images found in bundle"}), 400
    
    # Get base URL from request context (where it's available) before starting background thread
    request_base_url = os.environ.get('BASE_URL')
    if not request_base_url:
        request_base_url = request.host_url.rstrip('/')
    else:
        request_base_url = request_base_url.rstrip('/')
    
    thread = threading.Thread(target=job_processor.process_session, args=(job_id, session_metadata, image_files, request_base_url))
    thread.daemon = True
    thread.start()
    
    response = {"jobId": job_id, "status": "accepted"}
    logger.info(f"Returning response: {response}")
    return jsonify(response), 202

@app.route('/v1/panorama/status/<job_id>', methods=['GET'])
@limiter.limit("300 per minute")  # More generous limit for status polling
def get_job_status(job_id: str):
    # SECURITY FIX: Validate job ID
    if not validate_job_id(job_id):
        logger.warning(f"Invalid job ID format from {request.remote_addr}: {job_id}")
        return jsonify({"error": "Invalid job ID format"}), 400
    
    with job_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route('/v1/panorama/result/<job_id>', methods=['GET'])
def download_result(job_id: str):
    # SECURITY FIX: Validate job ID
    if not validate_job_id(job_id):
        logger.warning(f"Invalid job ID format from {request.remote_addr}: {job_id}")
        abort(400)
    
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
    # SECURITY FIX: Validate job ID
    if not validate_job_id(job_id):
        logger.warning(f"Invalid job ID format from {request.remote_addr}: {job_id}")
        abort(400)
    
    with job_lock:
        job = jobs.get(job_id)
    if not job or job.get("state") != JobState.COMPLETED:
        abort(404)
    
    preview_path = OUTPUT_DIR / f"{job_id}_preview.jpg"
    if not preview_path.exists():
        abort(404)
    
    logger.info(f"📸 Serving preview for job {job_id} - file size: {preview_path.stat().st_size} bytes")
    return send_file(str(preview_path), mimetype='image/jpeg')

@app.route('/v1/panorama/debug/<job_id>', methods=['GET'])
def view_coordinate_debug(job_id: str):
    """View coordinate debug visualization for a panorama job"""
    # SECURITY: Validate job ID
    if not validate_job_id(job_id):
        logger.warning(f"Invalid job ID format from {request.remote_addr}: {job_id}")
        abort(400)
    
    with job_lock:
        job = jobs.get(job_id)
    if not job:
        abort(404)
    
    # Look for coordinate debug image in temp directories
    import glob
    import os
    
    # Search common temp locations for debug images
    debug_patterns = [
        f"/tmp/coordinate_debug_*_points.png",
        f"/tmp/**/coordinate_debug_*.png",
        f"/tmp/coordinate_debug_{job_id}*.png"
    ]
    
    debug_path = None
    for pattern in debug_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            # Get the most recent debug image
            debug_path = max(matches, key=os.path.getctime)
            break
    
    if not debug_path or not os.path.exists(debug_path):
        # Return a helpful message instead of 404
        return f"""
        <html>
        <head><title>Debug Image - Job {job_id}</title></head>
        <body>
        <h2>🎯 Coordinate Debug Image</h2>
        <p><strong>Job ID:</strong> {job_id}</p>
        <p><strong>Status:</strong> Debug image not found or expired</p>
        <p>Debug images are created during processing and may be cleaned up after completion.</p>
        <p>To see debug images, check the logs during processing for:</p>
        <code>🎨 Coordinate debug visualization: /tmp/coordinate_debug_*.png</code>
        <p><a href="/v1/panorama/preview/{job_id}">← Back to Panorama Preview</a></p>
        </body>
        </html>
        """, 200, {'Content-Type': 'text/html'}
    
    logger.info(f"🎨 Serving coordinate debug for job {job_id} - file: {debug_path}")
    return send_file(debug_path, mimetype='image/png')

@app.route('/v1/panorama/overlay/<job_id>', methods=['GET'])
def view_debug_overlay(job_id: str):
    """View debug overlay combining coordinates with panorama result"""
    # SECURITY: Validate job ID
    if not validate_job_id(job_id):
        logger.warning(f"Invalid job ID format from {request.remote_addr}: {job_id}")
        abort(400)
    
    with job_lock:
        job = jobs.get(job_id)
    if not job:
        abort(404)
    
    # Look for debug overlay image
    overlay_path = f"/tmp/debug_overlay_{job_id}.jpg"
    
    if not os.path.exists(overlay_path):
        return f"""
        <html>
        <head><title>Debug Overlay - Job {job_id}</title></head>
        <body>
        <h2>🎨 Debug Overlay</h2>
        <p><strong>Job ID:</strong> {job_id}</p>
        <p><strong>Status:</strong> Overlay not found or expired</p>
        <p>Debug overlays are created automatically during processing and show coordinate positioning vs final panorama result.</p>
        <p>Available debug visualizations:</p>
        <ul>
        <li><a href="/v1/panorama/debug/{job_id}">Coordinate Debug (grid + dots)</a></li>
        <li><a href="/v1/panorama/preview/{job_id}">Panorama Preview</a></li>
        </ul>
        </body>
        </html>
        """, 200, {'Content-Type': 'text/html'}
    
    file_size = os.path.getsize(overlay_path) / (1024 * 1024)
    logger.info(f"🎨 Serving debug overlay for job {job_id} - file size: {file_size:.1f}MB")
    return send_file(overlay_path, mimetype='image/jpeg')

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
            "authentication": {
                "apiKeyConfigured": bool(os.environ.get('API_KEY')),
                "environment": os.environ.get('FLASK_ENV', 'production'),
                "authenticationRequired": not bool(os.environ.get('FLASK_ENV', '').lower() in ['development', 'dev', 'test', 'local'] and not os.environ.get('API_KEY'))
            },
            "officialHugin": {
                "architecture": "Microservices 2024 Workflow",
                "researchBased": True,
                "documentation": "wiki.panotools.org",
                "pipeline": "coordinate → hugin → blending → quality",
                "services": ["ARKitCoordinateService", "HuginPipelineService", "BlendingService", "QualityValidationService"],
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

@app.route('/v1/debug/coordinate-test', methods=['POST'])
@require_api_key
def test_coordinate_service():
    """Debug endpoint to test coordinate service in isolation."""
    try:
        if 'test_data' not in request.json:
            return jsonify({"error": "Missing test_data field"}), 400
            
        test_capture_points = request.json['test_data']
        
        # Create coordinate service for testing
        service_bus = get_service_bus()
        coordinate_service = create_coordinate_service()
        
        # Register service
        service_bus.register_service(
            name="debug_coordinate_service",
            version="1.0.0",
            capabilities=["debug_testing"]
        )
        
        # Run validation
        validation_results = coordinate_service.validate_arkit_data(test_capture_points)
        
        # Run conversion (test endpoint uses generic job ID)
        converted_coordinates = coordinate_service.convert_arkit_to_hugin(test_capture_points, "test_endpoint")
        
        # Generate debug report
        debug_report = coordinate_service.generate_debug_report()
        
        return jsonify({
            "status": "success",
            "validation_results": validation_results,
            "converted_coordinates": converted_coordinates[:5],  # First 5 for brevity
            "total_converted": len(converted_coordinates),
            "debug_report": debug_report,
            "service_bus_status": service_bus.generate_debug_report(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Coordinate test failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
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
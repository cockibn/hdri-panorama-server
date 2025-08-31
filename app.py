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
import shutil
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
    create_hugin_service
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

def create_enhanced_bundle_with_metadata(original_bundle_data: bytes, session_metadata: dict, output_path):
    """
    Create an enhanced bundle that includes both the original image data and session positioning metadata.
    
    Enhanced Format:
    HDRI_BUNDLE_V2_WITH_METADATA
    {image_count}
    SESSION_METADATA:{json_length}
    {session_json_data}
    ORIGINAL_IMAGES:
    {original_image_data...}
    """
    logger.info("📊 Creating enhanced bundle with session metadata...")
    
    # Extract session positioning data
    session_json = json.dumps({
        "bundleVersion": "HDRI_BUNDLE_V2_WITH_METADATA",
        "sessionData": session_metadata,
        "enhancedAt": datetime.now(timezone.utc).isoformat(),
        "format": {
            "description": "Enhanced HDRI bundle with positioning metadata for debugging",
            "originalFormat": "HDRI_BUNDLE_V1", 
            "imageCount": session_metadata.get("totalPoints", 0),
            "capturePoints": session_metadata.get("capturePoints", [])
        }
    }, indent=2)
    
    session_json_bytes = session_json.encode('utf-8')
    
    # Create enhanced bundle
    with open(output_path, 'wb') as f:
        # New header format
        f.write(b'HDRI_BUNDLE_V2_WITH_METADATA\n')
        
        # Image count (from session metadata)
        image_count = session_metadata.get("totalPoints", 0)
        f.write(f'{image_count}\n'.encode('utf-8'))
        
        # Session metadata section  
        f.write(f'SESSION_METADATA:{len(session_json_bytes)}\n'.encode('utf-8'))
        f.write(session_json_bytes)
        f.write(b'\n')
        
        # Original bundle data (images)
        # Skip the original header and just include the image data
        original_header_end = original_bundle_data.find(b'\xff\xd8')  # First JPEG marker
        if original_header_end > 0:
            f.write(b'ORIGINAL_IMAGES:\n')
            f.write(original_bundle_data[original_header_end:])  # All image data
        else:
            f.write(b'ORIGINAL_BUNDLE:\n')
            f.write(original_bundle_data)  # Include full original if can't parse
    
    file_size = output_path.stat().st_size
    logger.info(f"📦 Enhanced bundle created: {file_size / 1024 / 1024:.1f}MB with positioning metadata")

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
        
        # Initialize Hugin service for panorama stitching
        self.hugin_service = create_hugin_service()
        
        logger.info(f"🏗️ Simple Panorama Processor initialized")
        logger.info(f"   Resolution: {self.resolution} ({self.canvas_size[0]}×{self.canvas_size[1]})")
        logger.info(f"   Using proven Hugin 7-step workflow")
        

    def process_session(self, job_id: str, session_data: dict, image_files: List[str], base_url: str = None):
        """Process panorama session using proven Hugin workflow."""
        try:
            logger.info(f"🚀 Starting panorama processing for job {job_id}")
            self._update_job_status(job_id, JobState.PROCESSING, 0.0, "Initializing Hugin panorama stitching...")
            
            start_time = time.time()
            capture_points = session_data.get('capturePoints', [])
            logger.info(f"📊 Processing {len(capture_points)} capture points using proven Hugin workflow")
                
            # Complete Hugin panorama stitching workflow
            self._update_job_status(job_id, JobState.PROCESSING, 0.1, "🎯 Starting Hugin panorama stitching...")
            
            def hugin_progress_callback(progress: float, message: str):
                # Map hugin progress from 10% to 90%
                mapped_progress = 0.1 + (progress * 0.8)
                self._update_job_status(job_id, JobState.PROCESSING, mapped_progress, message)
                
            try:
                # Use the proven workflow to create panorama
                output_file = f"panorama_{job_id}.jpg"
                panorama_path = self.hugin_service.stitch_panorama(
                    images=image_files,
                    output_file=output_file,
                    session_metadata=session_data,
                    progress_callback=hugin_progress_callback
                )
                
                # Move panorama to output directory
                final_output_path = os.path.join(OUTPUT_DIR, f"{job_id}.jpg")
                shutil.move(panorama_path, final_output_path)
                
                # Calculate file size
                output_size_mb = os.path.getsize(final_output_path) / (1024 * 1024)
                
                processing_time = time.time() - start_time
                logger.info(f"✅ Panorama processing completed in {processing_time:.1f}s: {output_size_mb:.1f}MB")
                
                # Create preview with photosphere metadata
                logger.info("🖼️ Creating panorama preview with photosphere metadata...")
                preview_path = os.path.join(OUTPUT_DIR, f"{job_id}_preview.jpg")
                panorama_image = cv2.imread(final_output_path)
                if panorama_image is not None:
                    self._create_photosphere_preview(panorama_image, preview_path, session_data)
                    logger.info(f"📱 Preview created: {preview_path}")
                else:
                    logger.warning("⚠️ Could not load panorama for preview creation")
                
                # Generate result URLs
                result_url = f"/v1/panorama/result/{job_id}" if base_url else None
                preview_url = f"/v1/panorama/preview/{job_id}" if base_url else None
                
                if result_url:
                    logger.info(f"🔗 Download URL: {base_url}{result_url}")
                if preview_url:
                    logger.info(f"👁️ Preview URL: {base_url}{preview_url}")
                
                # Log bundle download information
                bundle_url = f"/v1/panorama/original/{job_id}"
                logger.info(f"📦 Original Bundle URL: {base_url}{bundle_url}")
                logger.info(f"📊 Enhanced Bundle: Available (includes positioning metadata)")
                logger.info(f"💡 Use bundle URL to download original images with metadata for debugging")
                
                # Update job status with success
                self._update_job_status(job_id, JobState.COMPLETED, 1.0, "Panorama completed successfully", result_url)
                
                # Store results
                with job_lock:
                    if job_id in jobs:
                        jobs[job_id]['output_file'] = final_output_path
                        jobs[job_id]['output_size_mb'] = output_size_mb
                        jobs[job_id]['processing_time'] = processing_time
                        jobs[job_id]['completed_at'] = datetime.now(timezone.utc).isoformat()
                        
                return {
                    'success': True,
                    'output_file': final_output_path,
                    'output_size_mb': output_size_mb,
                    'processing_time': processing_time
                }
                
            except Exception as e:
                logger.error(f"❌ Panorama stitching failed: {e}")
                self._update_job_status(job_id, JobState.FAILED, 0.0, f"Stitching failed: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"❌ Panorama processing failed: {e}")
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
    bundle_data = bundle_file.read()
    logger.info(f"Bundle file size: {len(bundle_data)} bytes")
    
    # Save the original bundle file for download/debugging
    original_bundle_path = upload_dir / 'original_bundle.zip'
    with open(original_bundle_path, 'wb') as f:
        f.write(bundle_data)
    logger.info(f"💾 Saved original bundle: {original_bundle_path}")
    
    # Create enhanced bundle with session positioning data for debugging
    enhanced_bundle_path = upload_dir / 'original_bundle_with_metadata.zip'
    try:
        create_enhanced_bundle_with_metadata(
            original_bundle_data=bundle_data,
            session_metadata=session_metadata,
            output_path=enhanced_bundle_path
        )
        logger.info(f"📊 Created enhanced bundle with positioning data: {enhanced_bundle_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not create enhanced bundle: {e}")
        # Fall back to original bundle only
    
    # Reset file pointer for extraction
    bundle_file.seek(0)
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
    
    # Enhance status response with bundle download URL for completed jobs
    enhanced_job = job.copy()
    if job.get("state") == JobState.COMPLETED:
        # Add bundle download URLs to completed job status
        base_url = f"{request.scheme}://{request.host}"
        enhanced_job["bundleDownloadUrl"] = f"{base_url}/v1/panorama/original/{job_id}"
        enhanced_job["bundleInfo"] = {
            "description": "Download original images with positioning metadata",
            "format": "ZIP archive with enhanced metadata",
            "includes": ["Original JPEG images", "Session positioning data", "Coordinate system metadata"]
        }
        logger.info(f"📦 Status request for completed job {job_id} - bundle download available")
    
    return jsonify(enhanced_job)

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
        <li><a href="/v1/panorama/debug-preview/{job_id}">Debug Overlay Preview</a></li>
        <li><a href="/v1/panorama/preview/{job_id}">Panorama Preview</a></li>
        <li><a href="/v1/panorama/original/{job_id}">Download Original Bundle</a></li>
        </ul>
        </body>
        </html>
        """, 200, {'Content-Type': 'text/html'}
    
    file_size = os.path.getsize(overlay_path) / (1024 * 1024)
    logger.info(f"🎨 Serving debug overlay for job {job_id} - file size: {file_size:.1f}MB")
    return send_file(overlay_path, mimetype='image/jpeg')

@app.route('/v1/panorama/debug-preview/<job_id>', methods=['GET'])
def debug_overlay_preview(job_id: str):
    """View debug overlay as an HTML preview with analysis information"""
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
    coordinate_debug_path = f"/tmp/coordinate_debug_{job_id}.png"
    
    # Get quality score and status
    quality_score = job.get('quality_score', 'Unknown')
    job_state = job.get('state', 'Unknown')
    processing_info = job.get('processing_info', {})
    
    # Create HTML preview with embedded image and analysis
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Debug Analysis - Job {job_id}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0; 
                padding: 20px; 
                background: #1a1a1a; 
                color: #fff;
                line-height: 1.6;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: #2a2a2a; 
                border-radius: 12px; 
                padding: 30px; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }}
            .header {{ 
                text-align: center; 
                margin-bottom: 30px; 
                border-bottom: 2px solid #444; 
                padding-bottom: 20px;
            }}
            .status-card {{ 
                background: #333; 
                border-radius: 8px; 
                padding: 20px; 
                margin: 20px 0;
                border-left: 4px solid {"#e74c3c" if quality_score != "Unknown" and float(str(quality_score).split("/")[0]) < 50 else "#27ae60"};
            }}
            .grid {{ 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 20px; 
                margin: 20px 0;
            }}
            .image-container {{ 
                text-align: center; 
                background: #333; 
                border-radius: 8px; 
                padding: 20px;
                overflow: hidden;
            }}
            .debug-image {{ 
                max-width: 100%; 
                height: auto; 
                border-radius: 8px; 
                box-shadow: 0 4px 16px rgba(0,0,0,0.3);
                transition: transform 0.3s ease;
            }}
            .debug-image:hover {{ 
                transform: scale(1.05); 
                cursor: pointer;
            }}
            .info-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 15px;
                margin: 20px 0;
            }}
            .info-item {{ 
                background: #333; 
                padding: 15px; 
                border-radius: 8px; 
                text-align: center;
            }}
            .quality-score {{ 
                font-size: 2em; 
                font-weight: bold; 
                color: {"#e74c3c" if quality_score != "Unknown" and float(str(quality_score).split("/")[0]) < 50 else "#27ae60"};
            }}
            .links {{ 
                margin-top: 30px; 
                text-align: center;
            }}
            .links a {{ 
                display: inline-block; 
                margin: 10px; 
                padding: 12px 24px; 
                background: #007acc; 
                color: white; 
                text-decoration: none; 
                border-radius: 6px; 
                transition: background 0.3s;
            }}
            .links a:hover {{ 
                background: #005fa3; 
            }}
            .timestamp {{ 
                color: #888; 
                font-size: 0.9em;
            }}
            @media (max-width: 768px) {{ 
                .grid {{ 
                    grid-template-columns: 1fr; 
                }}
                .info-grid {{
                    grid-template-columns: 1fr 1fr;
                }}
                .container {{ 
                    padding: 15px; 
                    margin: 10px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Panorama Debug Analysis</h1>
                <div class="timestamp">Job ID: {job_id} • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
            </div>
            
            <div class="status-card">
                <div class="info-grid">
                    <div class="info-item">
                        <div class="quality-score">{quality_score}</div>
                        <div>Quality Score</div>
                    </div>
                    <div class="info-item">
                        <div style="font-size: 1.5em; color: #f39c12;">{job_state}</div>
                        <div>Status</div>
                    </div>
                    <div class="info-item">
                        <div style="font-size: 1.5em; color: #3498db;">{processing_info.get('arkit_mode', 'N/A')}</div>
                        <div>ARKit Mode</div>
                    </div>
                    <div class="info-item">
                        <div style="font-size: 1.5em; color: #9b59b6;">{processing_info.get('control_points_found', 'N/A')}</div>
                        <div>Control Points</div>
                    </div>
                </div>
            </div>
    """
    
    # Add debug images if they exist
    if os.path.exists(overlay_path) or os.path.exists(coordinate_debug_path):
        html_content += '<div class="grid">'
        
        if os.path.exists(overlay_path):
            overlay_size = os.path.getsize(overlay_path) / (1024 * 1024)
            html_content += f"""
            <div class="image-container">
                <h3>🎨 Debug Overlay ({overlay_size:.1f}MB)</h3>
                <p>Coordinate positioning vs panorama result</p>
                <img src="/v1/panorama/overlay/{job_id}" class="debug-image" 
                     onclick="window.open('/v1/panorama/overlay/{job_id}', '_blank')"
                     alt="Debug overlay showing coordinate accuracy">
            </div>
            """
        
        if os.path.exists(coordinate_debug_path):
            coord_size = os.path.getsize(coordinate_debug_path) / (1024 * 1024)
            html_content += f"""
            <div class="image-container">
                <h3>📍 Coordinate Debug ({coord_size:.1f}MB)</h3>
                <p>ARKit positioning visualization</p>
                <img src="/v1/panorama/debug/{job_id}" class="debug-image"
                     onclick="window.open('/v1/panorama/debug/{job_id}', '_blank')"
                     alt="Coordinate debug showing ARKit positions">
            </div>
            """
        
        html_content += '</div>'
    else:
        html_content += f"""
        <div class="status-card">
            <h3>⚠️ Debug Images Not Available</h3>
            <p>Debug visualizations may not have been generated yet or have expired.</p>
            <p>Debug images are automatically created during processing and show:</p>
            <ul>
                <li><strong>Debug Overlay:</strong> Coordinate positioning accuracy vs final result</li>
                <li><strong>Coordinate Debug:</strong> ARKit positioning visualization</li>
            </ul>
        </div>
        """
    
    # Add navigation links
    html_content += f"""
            <div class="links">
                <a href="/v1/panorama/preview/{job_id}">📸 View Panorama</a>
                <a href="/v1/panorama/result/{job_id}">⬇️ Download Result</a>
                <a href="/v1/panorama/original/{job_id}">📦 Download Original</a>
                <a href="/v1/panorama/status/{job_id}">📊 Status API</a>
            </div>
        </div>
    </body>
    </html>
    """
    
    logger.info(f"🎨 Serving debug preview page for job {job_id}")
    return html_content, 200, {'Content-Type': 'text/html'}

@app.route('/v1/panorama/original/<job_id>', methods=['GET'])
def download_original_bundle(job_id: str):
    """Download the original bundle file uploaded from the iOS app"""
    # SECURITY: Validate job ID
    if not validate_job_id(job_id):
        logger.warning(f"Invalid job ID format from {request.remote_addr}: {job_id}")
        abort(400)
    
    with job_lock:
        job = jobs.get(job_id)
    if not job:
        abort(404)
    
    # Look for enhanced bundle first (with metadata), fallback to original
    upload_dir = UPLOAD_DIR / job_id
    enhanced_bundle_path = upload_dir / 'original_bundle_with_metadata.zip'
    original_bundle_path = upload_dir / 'original_bundle.zip'
    
    # Prioritize enhanced bundle with positioning metadata for debugging
    if enhanced_bundle_path.exists():
        bundle_path = enhanced_bundle_path
        bundle_type = "enhanced"
        logger.info(f"📊 Serving enhanced bundle with session positioning data for job {job_id}")
    elif original_bundle_path.exists():
        bundle_path = original_bundle_path
        bundle_type = "original"
        logger.info(f"📦 Serving original bundle (no metadata) for job {job_id}")
    else:
        return jsonify({
            "error": "Bundle not found",
            "message": f"No bundle files found for job {job_id}. This may occur if the job is very old or if there was an issue during upload.",
            "job_id": job_id
        }), 404
    
    # Check file size and log comprehensive download info
    file_size = bundle_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    # Get additional job metadata for enhanced logging
    session_data = job.get('session_data', {})
    image_count = len(session_data.get('capturePoints', []))
    processing_status = job.get('status', 'unknown')
    client_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', 'unknown')
    
    logger.info(f"📦 BUNDLE DOWNLOAD REQUEST:")
    logger.info(f"   Job ID: {job_id}")
    logger.info(f"   Bundle Type: {bundle_type} {'(includes positioning metadata)' if bundle_type == 'enhanced' else '(original images only)'}")
    logger.info(f"   File Size: {file_size_mb:.1f}MB ({file_size:,} bytes)")
    logger.info(f"   Image Count: {image_count} images")
    logger.info(f"   Processing Status: {processing_status}")
    logger.info(f"   Client: {client_ip} ({user_agent})")
    logger.info(f"   Bundle Path: {bundle_path}")
    
    # Generate a descriptive filename with timestamp
    job_timestamp = job.get('created_at', 'unknown')
    if isinstance(job_timestamp, str):
        try:
            from datetime import datetime
            parsed_time = datetime.fromisoformat(job_timestamp.replace('Z', '+00:00'))
            timestamp_str = parsed_time.strftime('%Y%m%d_%H%M%S')
        except:
            timestamp_str = 'unknown'
    else:
        timestamp_str = 'unknown'
    
    # Include bundle type in filename for clarity
    bundle_suffix = "_with_metadata" if bundle_type == "enhanced" else ""
    descriptive_filename = f"hdri360_original{bundle_suffix}_{timestamp_str}_{job_id[:8]}.zip"
    
    logger.info(f"   Download Filename: {descriptive_filename}")
    logger.info(f"✅ Serving bundle download for job {job_id}")
    
    return send_file(
        bundle_path, 
        mimetype='application/zip',
        as_attachment=True,
        download_name=descriptive_filename
    )

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
                "pipeline": "hugin → blending → quality",
                "services": ["HuginPipelineService", "BlendingService", "QualityValidationService"],
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
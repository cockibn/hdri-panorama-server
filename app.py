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

# Health check endpoint is defined later in the file with comprehensive metrics

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["500 per day", "200 per hour"],  # More generous defaults for iOS app
    storage_uri=os.environ.get('REDIS_URL', 'memory://'),
    headers_enabled=True
)

UPLOAD_DIR, OUTPUT_DIR = Path("uploads"), Path("outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    
    # Check size limit first
    if len(bundle_data) > 500 * 1024 * 1024:  # 500MB limit
        raise ValueError("Bundle exceeds maximum size limit")
    
    # DEBUG: Log first 100 bytes of bundle to see what we're receiving
    header_bytes = bundle_data[:100]
    logger.info(f"🔍 DEBUG: Bundle header (first 100 bytes): {header_bytes}")
    
    # Support both V2 and V3 HDR bundle formats
    if bundle_data.startswith(b"HDRI_BUNDLE_V3_WITH_HDR_BRACKETS\n"):
        logger.info("🌈 Processing V3 HDR bundle with bracket support")
        return "V3_HDR"
    elif bundle_data.startswith(b"HDRI_BUNDLE_V2_WITH_METADATA\n"):
        logger.info("📸 Processing V2 bundle with single images")
        return "V2"
    else:
        # More detailed error with actual header
        actual_header = bundle_data[:50].decode('utf-8', errors='ignore')
        logger.error(f"❌ Invalid bundle header received: '{actual_header}'")
        raise ValueError(f"Invalid bundle format header - got '{actual_header}', expecting HDRI_BUNDLE_V2_WITH_METADATA or HDRI_BUNDLE_V3_WITH_HDR_BRACKETS")

def parse_hdr_bundle_v3(bundle_data):
    """
    Parse V3 HDR bundle format with bracket support.
    
    V3 Format:
    HDRI_BUNDLE_V3_WITH_HDR_BRACKETS
    SESSION_METADATA:{json_length}
    {session_json_data}
    ORIGINAL_IMAGES:
    BRACKET_{dot}_{bracket}_EV{exposure}:{size}
    {image_data}
    ...
    
    Returns:
    - session_metadata: dict with session info
    - hdr_brackets: dict with structure {dot_index: [{exposure: float, data: bytes}, ...]}
    """
    logger.info("🌈 Parsing V3 HDR bundle...")
    
    try:
        bundle_str = bundle_data.decode('utf-8', errors='ignore')
    except UnicodeDecodeError:
        # Handle binary data - extract text portions only
        bundle_str = bundle_data[:10000].decode('utf-8', errors='ignore')
    
    lines = bundle_str.split('\n')
    
    # Parse session metadata
    session_metadata = {}
    hdr_brackets = {}
    current_pos = 0
    
    # Skip header
    if lines[0] != "HDRI_BUNDLE_V3_WITH_HDR_BRACKETS":
        raise ValueError("Invalid V3 HDR bundle header")
    current_pos += len(lines[0]) + 1
    
    # Find SESSION_METADATA section
    for i, line in enumerate(lines[1:], 1):
        if line.startswith("SESSION_METADATA:"):
            metadata_size = int(line.split(':')[1])
            
            # Calculate position more carefully
            header_end = bundle_data.find(f"SESSION_METADATA:{metadata_size}\n".encode('utf-8'))
            if header_end == -1:
                raise ValueError("Cannot find SESSION_METADATA header")
            
            metadata_start = header_end + len(f"SESSION_METADATA:{metadata_size}\n")
            metadata_bytes = bundle_data[metadata_start:metadata_start + metadata_size]
            
            session_metadata = json.loads(metadata_bytes.decode('utf-8'))
            current_pos = metadata_start + metadata_size
            break
    
    # Find ORIGINAL_IMAGES section
    remaining_data = bundle_data[current_pos:]
    remaining_str = remaining_data.decode('utf-8', errors='ignore')
    
    if "ORIGINAL_IMAGES:" in remaining_str:
        images_start = remaining_data.find(b"ORIGINAL_IMAGES:") + len(b"ORIGINAL_IMAGES:\n")
        image_data = remaining_data[images_start:]
    else:
        image_data = remaining_data
    
    # Parse HDR bracket images
    data_pos = 0
    bracket_count = 0
    
    while data_pos < len(image_data):
        # Look for bracket header pattern
        remaining = image_data[data_pos:]
        text_portion = remaining[:200].decode('utf-8', errors='ignore')
        
        if text_portion.startswith('BRACKET_'):
            # Parse bracket header: BRACKET_0_0_EV-1.0:1028
            header_end = text_portion.find('\n')
            if header_end == -1:
                break
                
            header = text_portion[:header_end]
            parts = header.split('_')
            
            if len(parts) >= 4:
                try:
                    dot_index = int(parts[1])
                    bracket_index = int(parts[2])
                    
                    # Extract EV and size
                    ev_size_part = parts[3]  # "EV-1.0:1028"
                    ev_part, size_part = ev_size_part.split(':')
                    exposure = float(ev_part.replace('EV', ''))
                    image_size = int(size_part)
                    
                    # Extract image data
                    header_bytes = len(header) + 1  # +1 for newline
                    image_bytes = remaining[header_bytes:header_bytes + image_size]
                    
                    # Store bracket data
                    if dot_index not in hdr_brackets:
                        hdr_brackets[dot_index] = []
                    
                    hdr_brackets[dot_index].append({
                        'bracket_index': bracket_index,
                        'exposure': exposure,
                        'data': image_bytes,
                        'size': image_size
                    })
                    
                    bracket_count += 1
                    data_pos += header_bytes + image_size
                    
                    if bracket_count % 10 == 0:
                        logger.info(f"📸 Parsed {bracket_count} HDR brackets...")
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"⚠️ Failed to parse bracket header '{header}': {e}")
                    data_pos += header_end + 1
            else:
                data_pos += header_end + 1
        else:
            # No more bracket headers found
            break
    
    # Sort brackets by bracket_index within each dot
    for dot_index in hdr_brackets:
        hdr_brackets[dot_index].sort(key=lambda x: x['bracket_index'])
    
    logger.info(f"✅ Parsed V3 HDR bundle: {len(hdr_brackets)} dots, {bracket_count} total brackets")
    
    return session_metadata, hdr_brackets

def merge_hdr_brackets(hdr_brackets, output_dir):
    """
    Merge HDR brackets for each capture point using OpenCV HDR processing.
    
    Args:
        hdr_brackets: dict with structure {dot_index: [{exposure: float, data: bytes}, ...]}
        output_dir: directory to save merged HDR images
    
    Returns:
        merged_images: dict with structure {dot_index: merged_image_path}
    """
    logger.info("🌈 Starting HDR bracket merging...")
    
    merged_images = {}
    
    for dot_index, brackets in hdr_brackets.items():
        if len(brackets) < 2:
            # Not enough brackets for HDR - use single image
            logger.warning(f"⚠️ Dot {dot_index}: Only {len(brackets)} brackets, skipping HDR merge")
            if brackets:
                # Save single image
                single_image_path = os.path.join(output_dir, f"merged_dot_{dot_index}.jpg")
                with open(single_image_path, 'wb') as f:
                    f.write(brackets[0]['data'])
                merged_images[dot_index] = single_image_path
            continue
        
        try:
            # Load bracket images into OpenCV format
            cv_images = []
            exposures = []
            
            for bracket in brackets:
                # Convert bytes to numpy array
                nparr = np.frombuffer(bracket['data'], np.uint8)
                cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if cv_img is not None:
                    cv_images.append(cv_img)
                    # Convert exposure bias to exposure time (assuming base shutter speed of 1/60s)
                    # EV = -2 means longer exposure (brighter), EV = +2 means shorter exposure (darker)
                    exposure_time = (1.0 / 60.0) * (2.0 ** -bracket['exposure'])
                    exposures.append(exposure_time)
                    logger.info(f"📸 Dot {dot_index}: Loaded bracket EV={bracket['exposure']}, exposure_time={exposure_time:.6f}s")
                else:
                    logger.warning(f"⚠️ Dot {dot_index}: Failed to decode bracket EV={bracket['exposure']}")
            
            if len(cv_images) < 2:
                logger.warning(f"⚠️ Dot {dot_index}: Only {len(cv_images)} valid images after loading, skipping HDR merge")
                continue
            
            # Create HDR merge object
            merge_debevec = cv2.createMergeDebevec()
            
            # Convert to numpy arrays
            exposures_np = np.array(exposures, dtype=np.float32)
            
            logger.info(f"🌈 Dot {dot_index}: Merging {len(cv_images)} brackets with exposures {exposures}")
            
            # Merge HDR - preserve true HDR data
            hdr_image = merge_debevec.process(cv_images, exposures_np)
            
            # **HDR MERGE VALIDATION**: Verify we created true HDR data
            hdr_min, hdr_max = hdr_image.min(), hdr_image.max()
            hdr_values_above_1 = np.sum(hdr_image > 1.0)
            hdr_total_pixels = hdr_image.size
            hdr_percentage = (hdr_values_above_1 / hdr_total_pixels) * 100
            
            logger.info(f"🔍 HDR Merge Dot {dot_index}: range=[{hdr_min:.6f}, {hdr_max:.6f}]")
            logger.info(f"🔍 HDR Merge: {hdr_values_above_1}/{hdr_total_pixels} ({hdr_percentage:.2f}%) pixels above 1.0")
            
            if hdr_max > 1.0:
                logger.info(f"✅ Dot {dot_index}: Authentic HDR data created (max={hdr_max:.3f})")
            else:
                logger.warning(f"⚠️ Dot {dot_index}: HDR merge may have failed - no values > 1.0")
            
            # Save HDR merged image as 32-bit TIFF for Hugin HDR stitching
            hdr_merged_path = os.path.join(output_dir, f"merged_dot_{dot_index}_hdr.tif")
            hdr_success = cv2.imwrite(hdr_merged_path, hdr_image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            
            # **EXIF PRESERVATION**: Save EXIF metadata as sidecar file for HDR TIFF
            # NOTE: We don't embed EXIF directly in 32-bit TIFF to avoid data corruption
            if hdr_success and brackets:
                try:
                    # Find the original JPG for this dot (middle exposure is most representative)
                    middle_bracket_idx = len(brackets) // 2
                    original_jpg_data = brackets[middle_bracket_idx].get('data')
                    
                    if original_jpg_data:
                        # Extract EXIF from original JPG using piexif
                        import piexif
                        import json
                        from io import BytesIO
                        
                        # Load EXIF from the JPG data in memory
                        try:
                            # piexif.load expects bytes or filename, not BytesIO
                            exif_dict = piexif.load(original_jpg_data)
                            
                            # Extract key FOV-related EXIF data
                            exif_metadata = {}
                            if "Exif" in exif_dict:
                                exif = exif_dict["Exif"]
                                # Focal length
                                if piexif.ExifIFD.FocalLength in exif:
                                    fl = exif[piexif.ExifIFD.FocalLength]
                                    exif_metadata['focal_length'] = fl[0] / fl[1] if isinstance(fl, tuple) else fl
                                # F-number
                                if piexif.ExifIFD.FNumber in exif:
                                    fn = exif[piexif.ExifIFD.FNumber]
                                    exif_metadata['f_number'] = fn[0] / fn[1] if isinstance(fn, tuple) else fn
                                # ISO
                                if piexif.ExifIFD.ISOSpeedRatings in exif:
                                    exif_metadata['iso'] = exif[piexif.ExifIFD.ISOSpeedRatings]
                            
                            # Save as JSON sidecar file (safe for 32-bit TIFF workflow)
                            sidecar_path = hdr_merged_path.replace('.tif', '_exif.json')
                            logger.info(f"🔍 Saving EXIF to: {sidecar_path}")
                            logger.info(f"🔍 EXIF data: {exif_metadata}")
                            
                            with open(sidecar_path, 'w') as f:
                                json.dump(exif_metadata, f, indent=2)
                            
                            # Verify file was created
                            if os.path.exists(sidecar_path):
                                file_size = os.path.getsize(sidecar_path)
                                logger.info(f"✅ Dot {dot_index}: EXIF saved as sidecar {os.path.basename(sidecar_path)} ({file_size} bytes)")
                            else:
                                logger.error(f"❌ Dot {dot_index}: Failed to create sidecar file!")
                        except Exception as exif_error:
                            logger.warning(f"⚠️ Dot {dot_index}: Could not extract EXIF: {exif_error}")
                except Exception as e:
                    logger.warning(f"⚠️ Dot {dot_index}: EXIF extraction failed: {e}")
            
            # Also create LDR version for preview/fallback
            tonemap = cv2.createTonemapDrago(gamma=1.0, saturation=1.0, bias=0.85)
            ldr_image = tonemap.process(hdr_image)
            ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)
            
            merged_image_path = os.path.join(output_dir, f"merged_dot_{dot_index}.jpg")
            ldr_success = cv2.imwrite(merged_image_path, ldr_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Use HDR TIFF for stitching if successful, fallback to LDR
            if hdr_success:
                merged_images[dot_index] = hdr_merged_path
                logger.info(f"✅ Dot {dot_index}: True HDR merge saved as 32-bit TIFF for stitching")
                success = True
            elif ldr_success:
                merged_images[dot_index] = merged_image_path
                logger.warning(f"⚠️ Dot {dot_index}: HDR TIFF failed, using LDR JPG fallback")
                success = True
            else:
                logger.error(f"❌ Dot {dot_index}: Both HDR and LDR saves failed")
                success = False
                
        except Exception as e:
            logger.error(f"❌ Dot {dot_index}: HDR merge failed - {str(e)}")
            # Fallback to middle exposure bracket
            try:
                middle_bracket = brackets[len(brackets) // 2]
                fallback_path = os.path.join(output_dir, f"merged_dot_{dot_index}.jpg")
                with open(fallback_path, 'wb') as f:
                    f.write(middle_bracket['data'])
                merged_images[dot_index] = fallback_path
                logger.info(f"📸 Dot {dot_index}: Used fallback middle exposure (EV={middle_bracket['exposure']})")
            except Exception as fallback_error:
                logger.error(f"❌ Dot {dot_index}: Fallback also failed - {str(fallback_error)}")
    
    logger.info(f"✅ HDR merging complete: {len(merged_images)}/{len(hdr_brackets)} dots processed")
    return merged_images

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
    """Extract images from iOS app bundle V2 format with metadata."""
    image_files = []
    try:
        bundle_data = bundle_file.read()
        validate_bundle_format(bundle_data)
        
        # Parse V2 format: HDRI_BUNDLE_V2_WITH_METADATA\n{count}\nSESSION_METADATA:{size}\n{json}\nORIGINAL_IMAGES:\n{images}
        lines = bundle_data.split(b'\n', 4)
        if len(lines) < 5:
            raise ValueError("Malformed V2 bundle structure")
            
        # Parse image count
        try:
            image_count = int(lines[1])
        except ValueError:
            raise ValueError("Invalid image count in bundle")
            
        if image_count <= 0 or image_count > 50:  # Reasonable limits
            raise ValueError(f"Invalid image count: {image_count}")
        
        # Parse metadata section
        metadata_header = lines[2].decode('utf-8')
        if not metadata_header.startswith('SESSION_METADATA:'):
            raise ValueError("Missing SESSION_METADATA section")
        
        metadata_size = int(metadata_header.split(':')[1])
        
        # Extract metadata JSON (for future use)
        remaining_data = b'\n'.join(lines[3:])
        metadata_json = remaining_data[:metadata_size]
        try:
            import json
            metadata = json.loads(metadata_json)
            logger.info(f"📊 Extracted session metadata: {metadata.get('sessionId', 'unknown')}")
        except Exception as e:
            logger.warning(f"Could not parse metadata JSON: {e}")
        
        # Find ORIGINAL_IMAGES marker
        images_marker = b'ORIGINAL_IMAGES:\n'
        images_start = remaining_data.find(images_marker)
        if images_start == -1:
            raise ValueError("Missing ORIGINAL_IMAGES section")
        
        # Extract image data (all images concatenated as JPEGs)
        image_data = remaining_data[images_start + len(images_marker):]
        
        # AGGRESSIVE FIX: Find all JPEG markers then filter by size to get only main images
        all_jpeg_starts = []
        i = 0
        while i < len(image_data) - 1:
            if image_data[i:i+2] == b'\xff\xd8':  # JPEG SOI marker
                all_jpeg_starts.append(i)
            i += 1
        
        logger.info(f"Found {len(all_jpeg_starts)} total JPEG markers, filtering for main images...")
        
        # Calculate sizes and filter out small embedded thumbnails
        jpeg_starts = []
        for idx, start in enumerate(all_jpeg_starts):
            end = all_jpeg_starts[idx + 1] if idx + 1 < len(all_jpeg_starts) else len(image_data)
            size = end - start
            
            logger.debug(f"JPEG {idx} at position {start}: {size} bytes")
            
            # Only keep large images (> 1MB = main images, skip < 1MB = thumbnails)
            if size >= 1000000:  # 1MB threshold
                jpeg_starts.append(start)
                logger.info(f"✅ Keeping main JPEG {len(jpeg_starts)-1}: {size} bytes")
            else:
                logger.info(f"❌ Skipping thumbnail JPEG: {size} bytes")
        
        logger.info(f"Filtered to {len(jpeg_starts)} main images (expected: {image_count})")
        
        # Extract all valid images - iOS now only sends valid full-resolution images
        extracted_count = 0
        for i in range(min(len(jpeg_starts), 16)):  # Max 16 images expected
            try:
                start = jpeg_starts[i]
                end = jpeg_starts[i+1] if i+1 < len(jpeg_starts) else len(image_data)
                img_data = image_data[start:end]
                
                # Skip tiny placeholder images (< 50KB - should only be valid full-res images)
                if len(img_data) < 50000:
                    logger.warning(f"⚠️ Skipping small image {i}: {len(img_data)} bytes")
                    continue
                
                # Save image file with EXIF orientation correction
                filename = f'image_{i:04d}.jpg'
                filepath = Path(upload_dir) / filename
                
                # Save image with EXIF preservation
                try:
                    from PIL import Image, ImageOps
                    import io
                    
                    with Image.open(io.BytesIO(img_data)) as img:
                        # Preserve EXIF data before orientation correction
                        exif_data = img.info.get('exif')
                        
                        # Apply EXIF orientation correction
                        oriented_img = ImageOps.exif_transpose(img)
                        if oriented_img and oriented_img != img:
                            # Orientation was applied, save with preserved EXIF
                            oriented_img.save(filepath, 'JPEG', quality=98, optimize=True, exif=exif_data)
                            oriented_img.close()
                        else:
                            # No orientation change needed, save original with EXIF
                            img.save(filepath, 'JPEG', quality=98, optimize=True, exif=exif_data)
                except Exception as e:
                    logger.warning(f"EXIF-preserving image save failed for image {i}: {e}")
                    # Fallback to raw save (preserves original EXIF)
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                
                image_files.append(str(filepath))
                logger.info(f"✅ Extracted image {i}: {len(img_data)} bytes")
            
            except Exception as e:
                logger.error(f"Failed to extract image {i}: {e}")
                continue
        
        logger.info(f"📸 Successfully extracted {len(image_files)} valid images from V2 bundle")
        return image_files
    
    except Exception as e:
        logger.error(f"Bundle extraction failed: {e}")
        raise ValueError(f"Failed to extract bundle: {e}")

# Job management
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
                # Check if we have HDR TIFF inputs for native HDR processing
                hdr_tiff_count = sum(1 for img_path in image_files if img_path.endswith('_hdr.tif'))
                has_hdr_inputs = hdr_tiff_count > 0
                
                if has_hdr_inputs:
                    # Native HDR workflow: Create true HDR EXR panorama
                    logger.info(f"🌈 Processing {hdr_tiff_count} HDR TIFF inputs for native HDR panorama")
                    exr_output_file = f"panorama_{job_id}.exr"
                    panorama_path = self.hugin_service.stitch_panorama(
                        images=image_files,
                        output_file=exr_output_file,
                        session_metadata=session_data,
                        progress_callback=hugin_progress_callback
                    )
                    
                    # Move HDR EXR to output directory
                    exr_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_panorama.exr")
                    shutil.move(panorama_path, exr_output_path)
                    exr_size_mb = os.path.getsize(exr_output_path) / (1024 * 1024)
                    logger.info(f"✅ Native HDR EXR panorama created: {exr_size_mb:.1f}MB")
                    
                    # Create tone-mapped JPG for preview/compatibility
                    logger.info("🖼️ Creating tone-mapped JPG preview from HDR EXR...")
                    jpg_output_path = os.path.join(OUTPUT_DIR, f"{job_id}.jpg")
                    try:
                        hdr_panorama = cv2.imread(exr_output_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                        if hdr_panorama is not None:
                            # Tone mapping for JPG preview
                            tonemap = cv2.createTonemapDrago(gamma=1.0, saturation=1.0, bias=0.85)
                            ldr_preview = tonemap.process(hdr_panorama)
                            ldr_preview = np.clip(ldr_preview * 255, 0, 255).astype(np.uint8)
                            
                            cv2.imwrite(jpg_output_path, ldr_preview, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            jpg_size_mb = os.path.getsize(jpg_output_path) / (1024 * 1024)
                            logger.info(f"✅ Tone-mapped JPG preview: {jpg_size_mb:.1f}MB")
                        else:
                            logger.warning("⚠️ Failed to load EXR for JPG preview creation")
                    except Exception as e:
                        logger.error(f"❌ JPG preview creation failed: {e}")
                    
                    final_output_path = jpg_output_path
                    output_size_mb = exr_size_mb
                    
                else:
                    # Standard LDR workflow: Create JPG panorama
                    logger.info("📷 Processing LDR inputs for standard JPG panorama")
                    jpg_output_file = f"panorama_{job_id}.jpg"
                    panorama_path = self.hugin_service.stitch_panorama(
                        images=image_files,
                        output_file=jpg_output_file,
                        session_metadata=session_data,
                        progress_callback=hugin_progress_callback
                    )
                    
                    # Move JPG to output directory
                    final_output_path = os.path.join(OUTPUT_DIR, f"{job_id}.jpg")
                    shutil.move(panorama_path, final_output_path)
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
        """Create optimized JPEG preview with 360° photosphere metadata for mobile viewing."""
        try:
            # Convert to uint8 for JPEG
            if panorama.dtype == np.float32 or panorama.dtype == np.float64:
                panorama_preview = (np.clip(panorama, 0, 1) * 255).astype(np.uint8)
            else:
                panorama_preview = panorama
                
            height, width = panorama_preview.shape[:2]
            original_size_mb = (height * width * 3) / (1024 * 1024)
            logger.info(f"📱 Creating preview from {width}×{height} panorama ({original_size_mb:.1f}MB uncompressed)")
                
            # Optimize preview size for mobile viewing
            # Target: 2K width (2048px) for good quality without excessive file size
            max_preview_width = 2048
            if width > max_preview_width:
                scale_factor = max_preview_width / width
                new_width = max_preview_width
                new_height = int(height * scale_factor)
                
                logger.info(f"🔄 Scaling preview from {width}×{height} to {new_width}×{new_height} (scale: {scale_factor:.3f})")
                panorama_preview = cv2.resize(panorama_preview, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                
            result_rgb = cv2.cvtColor(panorama_preview, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(result_rgb)
            
            # Enhanced EXIF metadata with Google Photosphere support
            import piexif
            final_height, final_width = panorama_preview.shape[:2]
            
            exif_dict = {
                "0th": {
                    piexif.ImageIFD.Make: "HDRi 360 Studio",
                    piexif.ImageIFD.Model: "Microservices Panorama Processor",
                    piexif.ImageIFD.Software: "ARKit + Hugin Pipeline",
                    piexif.ImageIFD.ImageDescription: "Equirectangular 360° Photosphere Preview",
                    piexif.ImageIFD.ImageWidth: final_width,
                    piexif.ImageIFD.ImageLength: final_height
                },
                "Exif": {
                    piexif.ExifIFD.PixelXDimension: final_width,
                    piexif.ExifIFD.PixelYDimension: final_height,
                    piexif.ExifIFD.ColorSpace: 1  # sRGB
                }
            }
            
            try:
                exif_bytes = piexif.dump(exif_dict)
                # Use high quality (95%) for preview to maintain visual quality despite scaling
                pil_image.save(preview_path, 'JPEG', quality=95, optimize=True, exif=exif_bytes)
                
                # Log preview file size
                preview_size_mb = os.path.getsize(preview_path) / (1024 * 1024)
                logger.info(f"✅ Optimized preview saved: {final_width}×{final_height}, {preview_size_mb:.1f}MB (JPEG 95%)")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not add metadata: {e}")
                # Fallback without metadata but still high quality
                pil_image.save(preview_path, 'JPEG', quality=95, optimize=True)
                preview_size_mb = os.path.getsize(preview_path) / (1024 * 1024)
                logger.info(f"✅ Preview saved (no metadata): {final_width}×{final_height}, {preview_size_mb:.1f}MB")
                
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
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    bundle_file = request.files['images_zip']
    bundle_data = bundle_file.read()
    logger.info(f"Bundle file size: {len(bundle_data)} bytes")
    
    # Save the original bundle file for download/debugging
    original_bundle_path = upload_dir / 'original_bundle.zip'
    with open(original_bundle_path, 'wb') as f:
        f.write(bundle_data)
    logger.info(f"💾 Saved original bundle: {original_bundle_path}")
    
    # Validate and determine bundle format
    try:
        bundle_format = validate_bundle_format(bundle_data)
    except ValueError as e:
        logger.error(f"❌ Bundle validation failed: {e}")
        return jsonify({"error": "Invalid bundle format", "details": str(e)}), 400
    
    # Handle HDR vs standard processing
    image_files = []
    hdr_merge_dir = None
    
    if bundle_format == "V3_HDR":
        logger.info("🌈 Processing V3 HDR bundle with bracket merging")
        
        try:
            # Parse HDR bundle
            hdr_metadata, hdr_brackets = parse_hdr_bundle_v3(bundle_data)
            logger.info(f"📊 HDR Bundle contains {len(hdr_brackets)} dots with brackets")
            
            # Create directory for HDR merge results
            hdr_merge_dir = upload_dir / 'hdr_merged'
            hdr_merge_dir.mkdir()
            
            # Merge HDR brackets for each capture point
            merged_images = merge_hdr_brackets(hdr_brackets, str(hdr_merge_dir))
            
            # Create image file list from merged results (sorted by dot index)
            image_files = []
            for dot_index in sorted(merged_images.keys()):
                image_files.append(merged_images[dot_index])
            
            logger.info(f"✅ HDR processing complete: {len(image_files)} merged images ready for stitching")
            
            # Create enhanced bundle with HDR info
            enhanced_bundle_path = upload_dir / 'original_bundle_with_metadata.zip'
            shutil.copy2(original_bundle_path, enhanced_bundle_path)
            
        except Exception as e:
            logger.error(f"❌ HDR processing failed: {e}")
            return jsonify({"error": "HDR processing failed", "details": str(e)}), 500
            
    else:
        logger.info("📸 Processing V2 standard bundle")
        
        # Since iOS sends V2 bundles with metadata, just copy the original as enhanced
        enhanced_bundle_path = upload_dir / 'original_bundle_with_metadata.zip'
        try:
            shutil.copy2(original_bundle_path, enhanced_bundle_path)
            logger.info(f"📊 V2 bundle already contains metadata - copied as enhanced bundle")
        except Exception as e:
            logger.warning(f"⚠️ Could not create enhanced bundle: {e}")
        
        # Extract images using standard method
        bundle_file.seek(0)
        image_files = extract_bundle_images(bundle_file, upload_dir)
    
    logger.info(f"Extracted/processed {len(image_files)} images for stitching")
    
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
        # Force HTTPS for Railway deployment
        if 'railway.app' in request_base_url:
            request_base_url = request_base_url.replace('http://', 'https://')
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

@app.route('/v1/panorama/original/<job_id>', methods=['GET'])
def download_original_bundle(job_id: str):
    """Download original HDR bundle with all bracket images and metadata"""
    # SECURITY: Validate job ID
    if not validate_job_id(job_id):
        logger.warning(f"Invalid job ID format from {request.remote_addr}: {job_id}")
        abort(400)
    
    with job_lock:
        job = jobs.get(job_id)
    if not job:
        abort(404)
    
    # Check if job is completed (allow download even for failed jobs for debugging)
    if job.get("state") not in [JobState.COMPLETED, JobState.FAILED]:
        return jsonify({"error": "Job not ready for download"}), 202
    
    # Find original bundle file
    upload_dir = UPLOAD_DIR / job_id
    original_bundle_path = upload_dir / 'original_bundle.zip'
    enhanced_bundle_path = upload_dir / 'original_bundle_with_metadata.zip'
    
    # Prefer enhanced bundle if available, fallback to original
    if enhanced_bundle_path.exists():
        bundle_path = enhanced_bundle_path
        filename = f"hdr_bundle_{job_id}_enhanced.zip"
        logger.info(f"📦 Serving enhanced HDR bundle: {bundle_path}")
    elif original_bundle_path.exists():
        bundle_path = original_bundle_path
        filename = f"hdr_bundle_{job_id}_original.zip"
        logger.info(f"📦 Serving original HDR bundle: {bundle_path}")
    else:
        logger.error(f"❌ No bundle file found for job {job_id}")
        abort(404)
    
    try:
        return send_file(
            bundle_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/zip'
        )
    except Exception as e:
        logger.error(f"❌ Failed to serve bundle {job_id}: {e}")
        abort(500)

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
    
    # Try EXR first, then fallback to JPG (current server creates JPG files)
    exr_path = OUTPUT_DIR / f"{job_id}_panorama.exr"
    jpg_path = OUTPUT_DIR / f"{job_id}.jpg"
    
    if exr_path.exists():
        logger.info(f"📸 Serving EXR panorama: {exr_path}")
        return send_file(str(exr_path), as_attachment=True, download_name=f"panorama_{job_id}.exr")
    elif jpg_path.exists():
        logger.info(f"📸 Serving JPG panorama: {jpg_path}")
        return send_file(str(jpg_path), as_attachment=True, download_name=f"panorama_{job_id}.jpg")
    else:
        logger.error(f"❌ No panorama file found for job {job_id} (checked EXR and JPG)")
        abort(404)

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
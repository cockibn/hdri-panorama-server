#!/usr/bin/env python3
"""
Create overlay visualization combining coordinate debug and panorama result.

This tool creates a composite image showing:
1. Background: Final panorama result (dimmed)
2. Overlay: Coordinate debug dots and grid lines
3. Transparency: Allows seeing positioning accuracy vs actual result

Usage:
    python create_debug_overlay.py <job_id>
    python create_debug_overlay.py --help
"""

import os
import cv2
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_job_files(job_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Find debug image and panorama result for a job ID."""
    
    # Common locations to search
    search_paths = [
        "/tmp",
        "/tmp/uploads",
        "/tmp/results", 
        "/tmp/blending_service*",
        "/tmp/hugin_pipeline*"
    ]
    
    debug_image = None
    panorama_result = None
    
    # Search for debug image
    debug_patterns = [
        f"coordinate_debug_{job_id}.png",
        f"coordinate_debug_{job_id}*.png"
    ]
    
    # Search for panorama result  
    result_patterns = [
        f"{job_id}_panorama.exr",
        f"{job_id}_panorama.tif", 
        f"{job_id}_panorama.jpg",
        f"panorama_result_{job_id}.*",
        f"blended_panorama_{job_id}.*"
    ]
    
    import glob
    
    for search_path in search_paths:
        # Find debug image
        for pattern in debug_patterns:
            matches = glob.glob(f"{search_path}/**/{pattern}", recursive=True)
            if matches:
                debug_image = matches[0]
                logger.info(f"📍 Found debug image: {debug_image}")
                break
                
        # Find panorama result
        for pattern in result_patterns:
            matches = glob.glob(f"{search_path}/**/{pattern}", recursive=True)
            if matches:
                panorama_result = matches[0]
                logger.info(f"🖼️ Found panorama result: {panorama_result}")
                break
                
        if debug_image and panorama_result:
            break
    
    return debug_image, panorama_result

def load_and_normalize_images(debug_path: str, panorama_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and normalize both images to same size and format."""
    
    # Load debug image (should be PNG with coordinate grid)
    debug_img = cv2.imread(debug_path, cv2.IMREAD_COLOR)
    if debug_img is None:
        raise ValueError(f"Could not load debug image: {debug_path}")
    
    logger.info(f"📍 Debug image: {debug_img.shape} ({debug_img.dtype})")
    
    # Load panorama result (could be EXR, TIFF, or JPEG)
    panorama_img = cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED)
    if panorama_img is None:
        raise ValueError(f"Could not load panorama result: {panorama_path}")
    
    logger.info(f"🖼️ Panorama image: {panorama_img.shape} ({panorama_img.dtype})")
    
    # Convert panorama to 8-bit if needed (EXR/16-bit TIFF handling)
    if panorama_img.dtype == np.float32:
        # EXR format - normalize to 0-255
        panorama_img = np.clip(panorama_img * 255.0, 0, 255).astype(np.uint8)
        logger.info("🔄 Converted EXR float32 to uint8")
    elif panorama_img.dtype == np.uint16:
        # 16-bit TIFF - scale down to 8-bit
        panorama_img = (panorama_img / 256).astype(np.uint8)
        logger.info("🔄 Converted 16-bit to 8-bit")
    
    # Ensure both images are RGB (3 channels)
    if len(debug_img.shape) == 3 and debug_img.shape[2] == 3:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
    if len(panorama_img.shape) == 3 and panorama_img.shape[2] >= 3:
        panorama_img = cv2.cvtColor(panorama_img, cv2.COLOR_BGR2RGB)
    elif len(panorama_img.shape) == 2:
        panorama_img = cv2.cvtColor(panorama_img, cv2.COLOR_GRAY2RGB)
    
    # Get target size (use panorama size as reference, typically 6144x3072)
    target_height, target_width = panorama_img.shape[:2]
    
    # Resize debug image to match panorama
    if debug_img.shape[:2] != (target_height, target_width):
        debug_img = cv2.resize(debug_img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        logger.info(f"🔄 Resized debug image to {target_width}×{target_height}")
    
    logger.info(f"✅ Normalized images: {target_width}×{target_height}")
    return debug_img, panorama_img

def create_overlay(debug_img: np.ndarray, panorama_img: np.ndarray, 
                  panorama_opacity: float = 0.7, debug_opacity: float = 0.8) -> np.ndarray:
    """Create overlay combining debug coordinates with panorama result."""
    
    # Create dimmed panorama as background
    background = (panorama_img * panorama_opacity).astype(np.uint8)
    
    # Extract coordinate elements from debug image
    # Look for non-white pixels (coordinate dots and grid lines are colored)
    debug_mask = np.any(debug_img < [240, 240, 240], axis=2)  # Non-white pixels
    
    # Create overlay
    overlay = background.copy()
    
    # Apply debug elements where they exist
    overlay[debug_mask] = (
        background[debug_mask] * (1 - debug_opacity) + 
        debug_img[debug_mask] * debug_opacity
    ).astype(np.uint8)
    
    # Enhance coordinate dots and grid lines for visibility
    # Look for very bright/saturated colors in debug image (coordinate elements)
    bright_mask = np.any(debug_img > [200, 100, 100], axis=2) | \
                  np.any(debug_img > [100, 200, 100], axis=2) | \
                  np.any(debug_img > [100, 100, 200], axis=2)
    
    # Make coordinate elements more visible
    overlay[bright_mask] = debug_img[bright_mask]
    
    return overlay

def add_legend(overlay: np.ndarray, job_id: str) -> np.ndarray:
    """Add legend explaining the overlay visualization."""
    
    height, width = overlay.shape[:2]
    legend_height = 120
    
    # Create legend area at bottom
    legend_overlay = overlay.copy()
    
    # Dark semi-transparent background for legend
    legend_bg = np.zeros((legend_height, width, 3), dtype=np.uint8)
    legend_bg[:, :] = [40, 40, 40]  # Dark gray
    
    # Blend legend background
    y_start = height - legend_height
    alpha = 0.8
    legend_overlay[y_start:, :] = (
        legend_overlay[y_start:, :] * (1 - alpha) + 
        legend_bg * alpha
    ).astype(np.uint8)
    
    # Add legend text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color_white = (255, 255, 255)
    color_cyan = (0, 255, 255)
    color_green = (0, 255, 0)
    color_blue = (255, 100, 100)
    
    # Legend text
    cv2.putText(legend_overlay, f"Coordinate Debug Overlay - Job: {job_id}", 
                (20, y_start + 30), font, font_scale, color_white, thickness)
    
    cv2.putText(legend_overlay, "Grid Lines: ARKit coordinate system", 
                (20, y_start + 60), font, font_scale * 0.7, color_cyan, thickness)
    
    cv2.putText(legend_overlay, "Green Dots: Capture positions", 
                (400, y_start + 60), font, font_scale * 0.7, color_green, thickness)
    
    cv2.putText(legend_overlay, "Blue Dot: Calibration reference", 
                (750, y_start + 60), font, font_scale * 0.7, color_blue, thickness)
    
    cv2.putText(legend_overlay, "Background: Dimmed panorama result", 
                (20, y_start + 90), font, font_scale * 0.7, color_white, thickness)
    
    return legend_overlay

def create_debug_overlay(job_id: str, output_path: Optional[str] = None) -> str:
    """Create complete debug overlay for a job."""
    
    logger.info(f"🎯 Creating debug overlay for job: {job_id}")
    
    # Find job files
    debug_path, panorama_path = find_job_files(job_id)
    
    if not debug_path:
        raise ValueError(f"Could not find debug image for job: {job_id}")
    if not panorama_path:
        raise ValueError(f"Could not find panorama result for job: {job_id}")
    
    # Load and normalize images
    debug_img, panorama_img = load_and_normalize_images(debug_path, panorama_path)
    
    # Create overlay
    logger.info("🎨 Creating coordinate overlay...")
    overlay = create_overlay(debug_img, panorama_img, panorama_opacity=0.6, debug_opacity=0.9)
    
    # Add legend
    logger.info("📝 Adding legend...")
    final_overlay = add_legend(overlay, job_id)
    
    # Save result
    if not output_path:
        output_path = f"/tmp/debug_overlay_{job_id}.jpg"
    
    # Convert RGB back to BGR for OpenCV saving
    final_bgr = cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR)
    
    success = cv2.imwrite(output_path, final_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError(f"Failed to save overlay image: {output_path}")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"✅ Debug overlay created: {output_path} ({file_size:.1f}MB)")
    
    return output_path

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Create debug overlay for panorama job')
    parser.add_argument('job_id', help='Job ID to create overlay for')
    parser.add_argument('-o', '--output', help='Output path for overlay image')
    parser.add_argument('--panorama-opacity', type=float, default=0.6, 
                       help='Opacity of panorama background (0.0-1.0)')
    parser.add_argument('--debug-opacity', type=float, default=0.9,
                       help='Opacity of debug elements (0.0-1.0)')
    
    args = parser.parse_args()
    
    try:
        output_path = create_debug_overlay(args.job_id, args.output)
        print(f"✅ Debug overlay created: {output_path}")
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to create debug overlay: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
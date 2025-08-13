#!/usr/bin/env python3
"""
Debug Overlay Service for Panorama Server

Integrates into the main panorama processing pipeline to automatically
create debug overlays showing coordinate positioning vs final results.
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class DebugOverlayService:
    """Service for creating debug overlays during panorama processing."""
    
    def __init__(self):
        pass
        
    def create_overlay_for_job(self, job_id: str, 
                             debug_image_path: str, 
                             panorama_result_path: str,
                             output_dir: str = "/tmp") -> Optional[str]:
        """
        Create debug overlay for a completed panorama job.
        
        Args:
            job_id: Panorama job ID
            debug_image_path: Path to coordinate debug image
            panorama_result_path: Path to final panorama result
            output_dir: Directory to save overlay
            
        Returns:
            Path to created overlay image, or None if failed
        """
        try:
            logger.info(f"🎨 Creating debug overlay for job: {job_id}")
            
            # Validate input files
            if not os.path.exists(debug_image_path):
                logger.warning(f"⚠️ Debug image not found: {debug_image_path}")
                return None
                
            if not os.path.exists(panorama_result_path):
                logger.warning(f"⚠️ Panorama result not found: {panorama_result_path}")
                return None
                
            # Load and process images
            debug_img, panorama_img = self._load_and_normalize_images(
                debug_image_path, panorama_result_path
            )
            
            if debug_img is None or panorama_img is None:
                return None
                
            # Create overlay
            overlay = self._create_overlay(debug_img, panorama_img)
            
            # Add informative legend
            final_overlay = self._add_legend(overlay, job_id)
            
            # Save result
            output_path = os.path.join(output_dir, f"debug_overlay_{job_id}.jpg")
            success = cv2.imwrite(output_path, final_overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success and os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"✅ Debug overlay created: {output_path} ({file_size:.1f}MB)")
                return output_path
            else:
                logger.error(f"❌ Failed to save overlay: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Debug overlay creation failed: {e}")
            return None
            
    def _load_and_normalize_images(self, debug_path: str, panorama_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load and normalize both images to same size and format."""
        try:
            # Load debug image (PNG with coordinate grid)
            debug_img = cv2.imread(debug_path, cv2.IMREAD_COLOR)
            if debug_img is None:
                logger.error(f"❌ Could not load debug image: {debug_path}")
                return None, None
                
            # Load panorama result (EXR, TIFF, or JPEG)
            panorama_img = cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED)
            if panorama_img is None:
                logger.error(f"❌ Could not load panorama: {panorama_path}")
                return None, None
                
            logger.debug(f"📍 Debug: {debug_img.shape} ({debug_img.dtype})")
            logger.debug(f"🖼️ Panorama: {panorama_img.shape} ({panorama_img.dtype})")
            
            # Normalize panorama to 8-bit RGB
            panorama_img = self._normalize_panorama_image(panorama_img)
            
            # Get target size from panorama (typically 6144x3072)
            target_height, target_width = panorama_img.shape[:2]
            
            # Resize debug image to match panorama
            if debug_img.shape[:2] != (target_height, target_width):
                debug_img = cv2.resize(debug_img, (target_width, target_height), 
                                     interpolation=cv2.INTER_LANCZOS4)
                logger.debug(f"🔄 Resized debug image to {target_width}×{target_height}")
                
            return debug_img, panorama_img
            
        except Exception as e:
            logger.error(f"❌ Image loading failed: {e}")
            return None, None
            
    def _normalize_panorama_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize panorama image to 8-bit BGR format."""
        
        # Handle different input formats
        if img.dtype == np.float32:
            # EXR format - normalize and convert
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            logger.debug("🔄 Converted EXR float32 to uint8")
        elif img.dtype == np.uint16:
            # 16-bit TIFF - scale down
            img = (img / 256).astype(np.uint8)
            logger.debug("🔄 Converted 16-bit to 8-bit")
            
        # Ensure 3-channel BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        return img
        
    def _create_overlay(self, debug_img: np.ndarray, panorama_img: np.ndarray) -> np.ndarray:
        """Create overlay combining debug coordinates with panorama result."""
        
        # Create dimmed panorama as background (70% opacity)
        background = (panorama_img * 0.7).astype(np.uint8)
        
        # Find coordinate elements in debug image
        # Look for non-white pixels (coordinate dots and grid lines are colored)
        gray_debug = cv2.cvtColor(debug_img, cv2.COLOR_BGR2GRAY)
        coordinate_mask = gray_debug < 240  # Non-white pixels
        
        # Create overlay starting with dimmed background
        overlay = background.copy()
        
        # Blend coordinate elements over background
        alpha = 0.9  # High opacity for coordinate elements
        for c in range(3):  # For each BGR channel
            overlay[:, :, c][coordinate_mask] = (
                background[:, :, c][coordinate_mask] * (1 - alpha) + 
                debug_img[:, :, c][coordinate_mask] * alpha
            ).astype(np.uint8)
            
        # Make coordinate dots extra visible by finding bright/saturated colors
        hsv_debug = cv2.cvtColor(debug_img, cv2.COLOR_BGR2HSV)
        
        # Find green dots (capture points) - high saturation in green range
        green_mask = (hsv_debug[:, :, 0] >= 40) & (hsv_debug[:, :, 0] <= 80) & (hsv_debug[:, :, 1] > 150)
        
        # Find blue dots (calibration) - high saturation in blue range  
        blue_mask = (hsv_debug[:, :, 0] >= 100) & (hsv_debug[:, :, 0] <= 130) & (hsv_debug[:, :, 1] > 150)
        
        # Find cyan grid lines
        cyan_mask = (hsv_debug[:, :, 0] >= 80) & (hsv_debug[:, :, 0] <= 100) & (hsv_debug[:, :, 1] > 100)
        
        # Make these elements fully visible
        overlay[green_mask] = debug_img[green_mask]
        overlay[blue_mask] = debug_img[blue_mask] 
        overlay[cyan_mask] = debug_img[cyan_mask]
        
        return overlay
        
    def _add_legend(self, overlay: np.ndarray, job_id: str) -> np.ndarray:
        """Add legend explaining the overlay visualization."""
        
        height, width = overlay.shape[:2]
        legend_height = 100
        
        # Create legend area at bottom
        legend_overlay = overlay.copy()
        
        # Semi-transparent dark background for legend
        cv2.rectangle(legend_overlay, 
                     (0, height - legend_height), 
                     (width, height), 
                     (30, 30, 30), -1)
        
        # Apply transparency
        alpha = 0.85
        legend_overlay[height - legend_height:, :] = (
            overlay[height - legend_height:, :] * (1 - alpha) + 
            legend_overlay[height - legend_height:, :] * alpha
        ).astype(np.uint8)
        
        # Add legend text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        y_start = height - legend_height + 25
        
        # Title
        cv2.putText(legend_overlay, f"Debug Overlay - Job: {job_id[:8]}...", 
                    (15, y_start), font, font_scale, (255, 255, 255), thickness)
        
        # Legend items
        cv2.putText(legend_overlay, "Grid: ARKit coordinates", 
                    (15, y_start + 30), font, 0.5, (0, 255, 255), 1)
        
        cv2.putText(legend_overlay, "Green: Capture points", 
                    (250, y_start + 30), font, 0.5, (0, 255, 0), 1)
        
        cv2.putText(legend_overlay, "Blue: Calibration", 
                    (450, y_start + 30), font, 0.5, (255, 100, 100), 1)
        
        cv2.putText(legend_overlay, "Background: Panorama result (dimmed)", 
                    (15, y_start + 55), font, 0.5, (200, 200, 200), 1)
        
        return legend_overlay

# Service instance
debug_overlay_service = DebugOverlayService()

def create_debug_overlay_for_job(job_id: str, debug_image_path: str, 
                                panorama_result_path: str) -> Optional[str]:
    """
    Convenience function for creating debug overlays.
    
    Returns path to overlay image or None if failed.
    """
    return debug_overlay_service.create_overlay_for_job(
        job_id, debug_image_path, panorama_result_path
    )
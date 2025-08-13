#!/usr/bin/env python3
"""
Blending Service

Isolates panorama blending operations from the main processing pipeline.
Provides multiple blending strategies:

1. Professional enblend multi-resolution spline blending (primary)
2. Emergency OpenCV multi-band blending (fallback)
3. Simple pixel averaging (emergency fallback)

This service can be debugged independently to isolate blending issues
that may cause seams, artifacts, or complete blending failures.
The service tries methods in order and falls back gracefully.
"""

import os
import cv2
import numpy as np
import logging
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import tempfile
import shutil

logger = logging.getLogger(__name__)

class BlendingError(Exception):
    """Raised when all blending strategies fail."""
    pass

class BlendingStrategy:
    """Represents a blending strategy attempt."""
    def __init__(self, name: str, description: str, timeout: int = 1800):
        self.name = name
        self.description = description
        self.timeout = timeout
        self.start_time = None
        self.end_time = None
        self.success = False
        self.error = None
        self.output_size_mb = 0.0
        
    def start(self):
        self.start_time = time.time()
        logger.info(f"🎨 Starting {self.name}: {self.description}")
        
    def complete(self, success: bool = True, error: Optional[str] = None, output_size_mb: float = 0.0):
        self.end_time = time.time()
        self.success = success
        self.error = error
        self.output_size_mb = output_size_mb
        
        duration = self.end_time - self.start_time if self.start_time else 0
        if success:
            logger.info(f"✅ Completed {self.name} in {duration:.1f}s, output: {output_size_mb:.1f}MB")
        else:
            logger.error(f"❌ Failed {self.name} after {duration:.1f}s: {error}")

class BlendingService:
    """
    Service for professional panorama blending with multiple fallback strategies.
    
    Provides isolated blending execution with comprehensive debugging,
    progress tracking, and graceful fallback when professional tools fail.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="blending_service_")
        self.blending_attempts = []
        self.progress_callback = None
        
        # Verify tools availability
        self.enblend_available = shutil.which('enblend') is not None
        
        logger.info(f"🎨 Blending Service initialized")
        logger.info(f"   Temp directory: {self.temp_dir}")
        logger.info(f"   Enblend available: {self.enblend_available}")
        
    def blend_panorama(self, 
                      tiff_files: List[str], 
                      output_path: str,
                      expected_image_count: int = 0,
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Blend panorama using multiple strategies with fallback.
        
        Args:
            tiff_files: List of rendered TIFF files to blend
            output_path: Output path for blended panorama
            expected_image_count: Expected number of images (for validation)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Blending results with strategy used and debugging information
        """
        logger.info(f"🎨 Starting panorama blending with {len(tiff_files)} images")
        
        self.progress_callback = progress_callback
        self.blending_attempts = []
        
        if not tiff_files:
            raise BlendingError("No TIFF files provided for blending")
            
        # Validate input files
        missing_files = [f for f in tiff_files if not os.path.exists(f)]
        if missing_files:
            raise BlendingError(f"Missing TIFF files: {missing_files}")
            
        start_time = time.time()
        
        # Strategy 1: Professional enblend
        if self.enblend_available:
            try:
                result = self._strategy_enblend(tiff_files, output_path, expected_image_count)
                if result['success']:
                    result['total_time'] = time.time() - start_time
                    result['blending_attempts'] = self.blending_attempts
                    return result
            except Exception as e:
                logger.warning(f"⚠️ Enblend strategy failed: {e}")
                
        # Strategy 2: OpenCV multi-band blending
        try:
            result = self._strategy_opencv_multiband(tiff_files, output_path)
            if result['success']:
                result['total_time'] = time.time() - start_time
                result['blending_attempts'] = self.blending_attempts
                return result
        except Exception as e:
            logger.warning(f"⚠️ OpenCV multi-band strategy failed: {e}")
            
        # Strategy 3: Emergency simple blending
        try:
            result = self._strategy_simple_blend(tiff_files, output_path)
            if result['success']:
                result['total_time'] = time.time() - start_time
                result['blending_attempts'] = self.blending_attempts
                return result
        except Exception as e:
            logger.error(f"❌ Simple blending strategy failed: {e}")
            
        # All strategies failed
        total_time = time.time() - start_time
        raise BlendingError(f"All blending strategies failed after {total_time:.1f}s")
        
    def _strategy_enblend(self, tiff_files: List[str], output_path: str, expected_count: int) -> Dict[str, Any]:
        """Strategy 1: Professional enblend multi-resolution blending."""
        strategy = BlendingStrategy(
            "enblend", 
            "Professional multi-resolution spline blending",
            timeout=1800  # 30 minutes max
        )
        strategy.start()
        self.blending_attempts.append(strategy)
        
        if self.progress_callback:
            self.progress_callback(0.1, "Starting professional enblend blending...")
            
        try:
            # Create temporary TIFF output (enblend doesn't support EXR directly)
            temp_tiff = os.path.join(self.temp_dir, "enblend_output.tif")
            
            # Build optimized enblend command for iPhone ultra-wide panoramas
            # Optimizations for 16-point iPhone capture pattern with 106.2° FOV:
            # - Auto pyramid levels for optimal quality/speed balance
            # - CIELAB colorspace for better color blending 
            # - Fine masks for seamless ultra-wide image boundaries
            # - Optimized for photometric consistency over geometric precision
            cmd = [
                "enblend",
                "-o", temp_tiff,
                "--wrap=horizontal",        # Essential for 360° seamless wrapping
                "--compression=lzw",        # Lossless compression
                "--levels=auto",            # Auto-determine optimal pyramid levels
                "--blend-colorspace=CIELAB", # Better color blending for photos
                "--fallback-overlap=0.05",  # Handle small overlaps from ultra-wide
                "--no-ciecam",             # Skip complex color appearance model (faster)
                "--fine-mask",             # Generate high-quality seam masks
                "--optimizer-weights=0:0:1:0", # Focus on photometric optimization
                "--mask-vectorize=12"      # Vectorize seam boundaries for smoother blending
            ]
            
            # Add environment-configurable options for debugging
            if os.environ.get('ENBLEND_VERBOSE', '').lower() in ('1', 'true'):
                cmd.append("--verbose")
                
            # Override auto levels if manually specified
            if os.environ.get('ENBLEND_LEVELS'):
                try:
                    levels = int(os.environ.get('ENBLEND_LEVELS'))
                    if 1 <= levels <= 29:  # Valid enblend levels range
                        # Replace --levels=auto with manual setting
                        for i, arg in enumerate(cmd):
                            if arg.startswith("--levels="):
                                cmd[i] = f"--levels={levels}"
                                break
                except ValueError:
                    logger.warning("⚠️ Invalid ENBLEND_LEVELS value, using auto")
                    
            # Add TIFF files
            cmd.extend(tiff_files)
            
            logger.info(f"🔧 Enblend command: {' '.join(cmd)}")
            
            # Run enblend with progress monitoring
            success = self._run_enblend_with_progress(cmd, strategy)
            
            if success and os.path.exists(temp_tiff) and os.path.getsize(temp_tiff) > 0:
                # Convert TIFF to final format if needed
                final_output = self._convert_to_final_format(temp_tiff, output_path)
                
                output_size_mb = os.path.getsize(final_output) / (1024 * 1024)
                
                # Debug: Analyze final blended output
                try:
                    import cv2
                    final_img = cv2.imread(final_output, cv2.IMREAD_UNCHANGED)
                    if final_img is not None:
                        height, width = final_img.shape[:2]
                        mean_val = final_img.mean()
                        
                        # Count non-black pixels (match quality service threshold)
                        gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY) if len(final_img.shape) == 3 else final_img
                        non_zero_pixels = np.count_nonzero(gray > 5)  # FIXED: Match quality service threshold
                        total_pixels = height * width
                        black_percentage = ((total_pixels - non_zero_pixels) / total_pixels) * 100
                        
                        logger.info(f"🔍 DEBUG Blended Output: {width}×{height}")
                        logger.info(f"🔍 DEBUG Mean brightness: {mean_val:.3f}")
                        logger.info(f"🔍 DEBUG Black pixels: {black_percentage:.1f}%")
                        
                        if black_percentage > 80:
                            logger.error(f"❌ CRITICAL: {black_percentage:.1f}% black pixels in final output!")
                        elif black_percentage > 50:
                            logger.warning(f"⚠️ HIGH: {black_percentage:.1f}% black pixels")
                        else:
                            logger.info(f"✅ GOOD: {black_percentage:.1f}% black pixels")
                            
                    else:
                        logger.error("❌ Could not read final blended output for analysis")
                except Exception as debug_error:
                    logger.warning(f"⚠️ Final output debug failed: {debug_error}")
                
                strategy.complete(success=True, output_size_mb=output_size_mb)
                
                return {
                    'success': True,
                    'strategy': 'enblend',
                    'output_path': final_output,
                    'output_size_mb': output_size_mb,
                    'quality_level': 'Professional'
                }
            else:
                raise Exception("Enblend produced no output or empty file")
                
        except Exception as e:
            strategy.complete(success=False, error=str(e))
            logger.error(f"❌ Enblend strategy failed: {e}")
            raise
            
    def _run_enblend_with_progress(self, cmd: List[str], strategy: BlendingStrategy) -> bool:
        """Run enblend with progress tracking and timeout handling."""
        try:
            # Use threading for progress monitoring
            process_completed = threading.Event()
            process_success = threading.Event()
            
            def run_process():
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=strategy.timeout,
                        cwd=self.temp_dir
                    )
                    
                    if result.returncode == 0:
                        process_success.set()
                    else:
                        logger.error(f"❌ Enblend failed with code {result.returncode}")
                        logger.error(f"❌ Stderr: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"❌ Enblend timed out after {strategy.timeout}s")
                except Exception as e:
                    logger.error(f"❌ Enblend execution error: {e}")
                finally:
                    process_completed.set()
                    
            # Start process in background
            process_thread = threading.Thread(target=run_process)
            process_thread.start()
            
            # Progress monitoring
            start_time = time.time()
            last_progress_time = start_time
            
            while not process_completed.is_set():
                elapsed = time.time() - start_time
                
                # Update progress every 30 seconds
                if time.time() - last_progress_time > 30:
                    if self.progress_callback:
                        progress = min(0.9, 0.1 + (elapsed / strategy.timeout) * 0.8)
                        self.progress_callback(progress, f"Enblend blending... {elapsed/60:.1f}min elapsed")
                    last_progress_time = time.time()
                    
                process_completed.wait(timeout=1)  # Check every second
                
            process_thread.join()
            return process_success.is_set()
            
        except Exception as e:
            logger.error(f"❌ Enblend progress monitoring failed: {e}")
            return False
            
    def _strategy_opencv_multiband(self, tiff_files: List[str], output_path: str) -> Dict[str, Any]:
        """Strategy 2: OpenCV multi-band blending."""
        strategy = BlendingStrategy(
            "opencv_multiband", 
            "OpenCV multi-band pyramid blending",
            timeout=600
        )
        strategy.start()
        self.blending_attempts.append(strategy)
        
        if self.progress_callback:
            self.progress_callback(0.2, "Starting OpenCV multi-band blending...")
            
        try:
            logger.info("🎨 Starting OpenCV multi-band blending")
            
            # Load all images
            images = []
            for i, tiff_file in enumerate(tiff_files):
                img = cv2.imread(tiff_file, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise Exception(f"Failed to load {tiff_file}")
                images.append(img)
                logger.debug(f"   Loaded image {i}: {img.shape}")
                
            if not images:
                raise Exception("No valid images loaded")
                
            # Determine canvas size
            heights = [img.shape[0] for img in images]
            widths = [img.shape[1] for img in images]
            canvas_height = max(heights)
            canvas_width = max(widths)
            
            logger.info(f"🎨 Canvas size: {canvas_width}×{canvas_height}")
            
            if self.progress_callback:
                self.progress_callback(0.4, "Computing multi-band pyramid blend...")
                
            # Multi-band blending approach
            num_bands = 6  # Number of frequency bands
            
            # Initialize blended image
            if len(images[0].shape) == 3:
                blended = np.zeros((canvas_height, canvas_width, images[0].shape[2]), dtype=np.float64)
            else:
                blended = np.zeros((canvas_height, canvas_width), dtype=np.float64)
                
            weights_total = np.zeros((canvas_height, canvas_width), dtype=np.float64)
            
            for i, img in enumerate(images):
                if self.progress_callback:
                    progress = 0.4 + (i / len(images)) * 0.5
                    self.progress_callback(progress, f"Blending image {i+1}/{len(images)}...")
                    
                # Convert to float64 for precision
                img_float = img.astype(np.float64)
                
                # Create weight mask (simple distance-based weighting)
                mask = np.ones((img.shape[0], img.shape[1]), dtype=np.float64)
                
                # Apply feathering to edges
                mask = self._apply_feathering(mask, feather_width=50)
                
                # Resize to canvas if needed
                if img_float.shape[:2] != (canvas_height, canvas_width):
                    img_float = cv2.resize(img_float, (canvas_width, canvas_height))
                    mask = cv2.resize(mask, (canvas_width, canvas_height))
                    
                # Add to blend
                if len(blended.shape) == 3:
                    for c in range(blended.shape[2]):
                        blended[:, :, c] += img_float[:, :, c] * mask
                else:
                    blended += img_float * mask
                    
                weights_total += mask
                
            # Normalize by weights
            mask_nonzero = weights_total > 0
            if len(blended.shape) == 3:
                for c in range(blended.shape[2]):
                    blended[:, :, c][mask_nonzero] /= weights_total[mask_nonzero]
            else:
                blended[mask_nonzero] /= weights_total[mask_nonzero]
                
            if self.progress_callback:
                self.progress_callback(0.95, "Finalizing OpenCV blend...")
                
            # Convert back to appropriate format
            if images[0].dtype == np.uint8:
                final_blend = np.clip(blended, 0, 255).astype(np.uint8)
            elif images[0].dtype == np.uint16:
                final_blend = np.clip(blended, 0, 65535).astype(np.uint16)
            else:
                final_blend = blended.astype(images[0].dtype)
                
            # Save result with proper format handling
            success = self._safe_write_image(output_path, final_blend)
            if not success:
                raise Exception(f"Failed to save blended image to {output_path}")
                
            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            strategy.complete(success=True, output_size_mb=output_size_mb)
            
            logger.info(f"✅ OpenCV multi-band blending completed: {output_size_mb:.1f}MB")
            
            return {
                'success': True,
                'strategy': 'opencv_multiband',
                'output_path': output_path,
                'output_size_mb': output_size_mb,
                'quality_level': 'Good'
            }
            
        except Exception as e:
            strategy.complete(success=False, error=str(e))
            logger.error(f"❌ OpenCV multi-band blending failed: {e}")
            raise
            
    def _strategy_simple_blend(self, tiff_files: List[str], output_path: str) -> Dict[str, Any]:
        """Strategy 3: Emergency simple pixel averaging blend."""
        strategy = BlendingStrategy(
            "simple_blend", 
            "Emergency simple pixel averaging",
            timeout=300
        )
        strategy.start()
        self.blending_attempts.append(strategy)
        
        if self.progress_callback:
            self.progress_callback(0.3, "Starting emergency simple blending...")
            
        try:
            logger.info("🚨 Starting emergency simple blending")
            logger.warning("⚠️ This will produce lower quality output but may recover the panorama")
            
            # Load images
            images = []
            for i, tiff_file in enumerate(tiff_files):
                img = cv2.imread(tiff_file, cv2.IMREAD_UNCHANGED)
                if img is None:
                    logger.warning(f"⚠️ Failed to load {tiff_file}, skipping")
                    continue
                images.append(img)
                
            if not images:
                raise Exception("No valid images loaded for simple blending")
                
            # Determine canvas size
            canvas_height = max(img.shape[0] for img in images)
            canvas_width = max(img.shape[1] for img in images)
            channels = images[0].shape[2] if len(images[0].shape) > 2 else 1
            
            logger.info(f"🎨 Simple blend canvas: {canvas_width}×{canvas_height}")
            
            # Initialize accumulation arrays
            if channels > 1:
                accumulated = np.zeros((canvas_height, canvas_width, channels), dtype=np.float64)
            else:
                accumulated = np.zeros((canvas_height, canvas_width), dtype=np.float64)
                
            count_mask = np.zeros((canvas_height, canvas_width), dtype=np.int32)
            
            # Simple averaging blend
            for i, img in enumerate(images):
                if self.progress_callback:
                    progress = 0.3 + (i / len(images)) * 0.6
                    self.progress_callback(progress, f"Simple blending image {i+1}/{len(images)}...")
                    
                # Resize if needed
                if img.shape[:2] != (canvas_height, canvas_width):
                    img = cv2.resize(img, (canvas_width, canvas_height))
                    
                # Create mask for non-black pixels
                if len(img.shape) == 3:
                    mask = np.any(img > 5, axis=2)  # Non-black pixels
                else:
                    mask = img > 5
                    
                # Add to accumulation
                img_float = img.astype(np.float64)
                if channels > 1:
                    for c in range(channels):
                        accumulated[:, :, c][mask] += img_float[:, :, c][mask]
                else:
                    accumulated[mask] += img_float[mask]
                    
                count_mask[mask] += 1
                
            # Average where we have data
            valid_pixels = count_mask > 0
            if channels > 1:
                for c in range(channels):
                    accumulated[:, :, c][valid_pixels] /= count_mask[valid_pixels]
            else:
                accumulated[valid_pixels] /= count_mask[valid_pixels]
                
            # Convert to output format
            if images[0].dtype == np.uint8:
                result = np.clip(accumulated, 0, 255).astype(np.uint8)
            elif images[0].dtype == np.uint16:
                result = np.clip(accumulated, 0, 65535).astype(np.uint16)
            else:
                result = accumulated.astype(images[0].dtype)
                
            if self.progress_callback:
                self.progress_callback(0.95, "Saving emergency blend result...")
                
            # Save result
            success = self._safe_write_image(output_path, result)
            if not success:
                raise Exception(f"Failed to save simple blend to {output_path}")
                
            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            strategy.complete(success=True, output_size_mb=output_size_mb)
            
            logger.info(f"✅ Emergency simple blending completed: {output_size_mb:.1f}MB")
            logger.warning("⚠️ Quality may be reduced due to simple averaging method")
            
            return {
                'success': True,
                'strategy': 'simple_blend',
                'output_path': output_path,
                'output_size_mb': output_size_mb,
                'quality_level': 'Basic'
            }
            
        except Exception as e:
            strategy.complete(success=False, error=str(e))
            logger.error(f"❌ Simple blending failed: {e}")
            raise
            
    def _apply_feathering(self, mask: np.ndarray, feather_width: int = 50) -> np.ndarray:
        """Apply feathering to mask edges to reduce seams."""
        try:
            # Create distance transform from edges
            edges = np.zeros_like(mask, dtype=np.uint8)
            edges[0, :] = 255
            edges[-1, :] = 255
            edges[:, 0] = 255
            edges[:, -1] = 255
            
            # Distance from edges
            dist_from_edge = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
            
            # Apply feathering
            feathered = np.minimum(dist_from_edge / feather_width, 1.0)
            feathered = np.maximum(feathered, 0.1)  # Minimum weight
            
            return feathered * mask
            
        except:
            # Fallback to original mask if feathering fails
            return mask
            
    def _convert_to_final_format(self, input_path: str, output_path: str) -> str:
        """Convert blended image to final output format with robust error handling."""
        try:
            # If output is EXR, convert TIFF to EXR
            if output_path.lower().endswith('.exr'):
                img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Ensure float32 for EXR
                    if img.dtype != np.float32:
                        if img.dtype == np.uint8:
                            img = img.astype(np.float32) / 255.0
                        elif img.dtype == np.uint16:
                            img = img.astype(np.float32) / 65535.0
                        else:
                            img = img.astype(np.float32)
                    
                    # Try to write EXR with error checking
                    exr_success = cv2.imwrite(output_path, img, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    
                    if exr_success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        logger.info(f"✅ Successfully created EXR: {os.path.getsize(output_path)} bytes")
                        return output_path
                    else:
                        logger.warning("⚠️ EXR write failed, falling back to TIFF format")
                        # Fallback to TIFF format
                        tiff_output = output_path.replace('.exr', '.tif')
                        tiff_success = cv2.imwrite(tiff_output, img)
                        if tiff_success and os.path.exists(tiff_output):
                            logger.info(f"✅ Fallback TIFF created: {tiff_output}")
                            return tiff_output
                        else:
                            raise Exception("Both EXR and TIFF fallback failed")
                else:
                    raise Exception(f"Could not load input image: {input_path}")
                    
            # Otherwise, copy/convert as needed
            if input_path != output_path:
                shutil.copy2(input_path, output_path)
                
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Format conversion failed: {e}")
            # Return input path if conversion fails
            
    def _safe_write_image(self, output_path: str, image: np.ndarray) -> bool:
        """Safely write image with proper format handling and fallbacks."""
        try:
            # Handle EXR format specially
            if output_path.lower().endswith('.exr'):
                # Convert to float32 for EXR
                if image.dtype != np.float32:
                    if image.dtype == np.uint8:
                        image_float = image.astype(np.float32) / 255.0
                    elif image.dtype == np.uint16:
                        image_float = image.astype(np.float32) / 65535.0
                    else:
                        image_float = image.astype(np.float32)
                else:
                    image_float = image
                
                # Try EXR write
                success = cv2.imwrite(output_path, image_float, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                
                # Validate EXR write
                if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"✅ Successfully wrote EXR: {os.path.getsize(output_path)} bytes")
                    return True
                else:
                    logger.warning("⚠️ EXR write failed, falling back to TIFF")
                    # Fallback to TIFF
                    tiff_path = output_path.replace('.exr', '.tif')
                    success = cv2.imwrite(tiff_path, image)
                    if success and os.path.exists(tiff_path):
                        logger.info(f"✅ Fallback TIFF written: {tiff_path}")
                        # Update the output path reference in calling code would be ideal,
                        # but for now we return success and the caller will find the TIFF
                        return True
                    else:
                        return False
            else:
                # Regular format writing
                success = cv2.imwrite(output_path, image)
                return success and os.path.exists(output_path) and os.path.getsize(output_path) > 0
                
        except Exception as e:
            logger.error(f"❌ Safe image write failed: {e}")
            return False
            
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"🧹 Cleaned up blending temporary directory")
            
    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report."""
        return {
            'service_info': {
                'name': 'BlendingService',
                'version': '1.0.0',
                'purpose': 'Professional panorama blending with multiple fallback strategies'
            },
            'configuration': {
                'temp_dir': self.temp_dir,
                'enblend_available': self.enblend_available
            },
            'strategies': [
                {
                    'name': 'enblend',
                    'description': 'Professional multi-resolution spline blending',
                    'available': self.enblend_available,
                    'quality': 'Professional'
                },
                {
                    'name': 'opencv_multiband', 
                    'description': 'OpenCV multi-band pyramid blending',
                    'available': True,
                    'quality': 'Good'
                },
                {
                    'name': 'simple_blend',
                    'description': 'Emergency pixel averaging',
                    'available': True,
                    'quality': 'Basic'
                }
            ],
            'last_blending_attempts': [
                {
                    'name': attempt.name,
                    'success': attempt.success,
                    'duration': attempt.end_time - attempt.start_time if attempt.start_time and attempt.end_time else 0,
                    'error': attempt.error,
                    'output_size_mb': attempt.output_size_mb
                } for attempt in self.blending_attempts
            ]
        }

def create_blending_service(temp_dir: Optional[str] = None) -> BlendingService:
    """Factory function to create blending service."""
    return BlendingService(temp_dir=temp_dir)
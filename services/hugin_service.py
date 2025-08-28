#!/usr/bin/env python3
"""
Hugin Pipeline Service

Implements the proven 7-step Hugin panorama stitching workflow:

1. pto_gen: Generate project file from images
2. cpfind --multirow: Find control points (multirow for spherical)
3. celeste_standalone + cpclean: Clean control points
4. linefind: Find vertical lines for better alignment
5. autooptimiser -a -m -l -s: Optimize everything
6. pano_modify: Set equirectangular projection
7. hugin_executor --stitching: Final stitching with enblend

This follows the proven workflow for perfect equirectangular panoramas.
"""

import os
import time
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import tempfile

logger = logging.getLogger(__name__)

class HuginPipelineError(Exception):
    """Raised when Hugin pipeline step fails."""
    pass

class HuginPipelineService:
    """
    Service for executing the proven Hugin panorama stitching pipeline.
    
    Uses the established 7-step workflow that's proven to work for
    professional panorama stitching.
    """
    
    def __init__(self, temp_dir: Optional[str] = None, canvas_size: Tuple[int, int] = (8192, 4096)):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="hugin_pipeline_")
        self.canvas_size = canvas_size
        
        # Verify Hugin installation
        self._verify_hugin_installation()
        
        logger.info(f"🏗️ Hugin Pipeline Service initialized")
        logger.info(f"   Temp directory: {self.temp_dir}")
        logger.info(f"   Canvas size: {self.canvas_size[0]}×{self.canvas_size[1]}")

    def _verify_hugin_installation(self):
        """Verify that all required Hugin tools are available."""
        required_tools = [
            'pto_gen', 'cpfind', 'celeste_standalone', 'cpclean', 
            'linefind', 'autooptimiser', 'pano_modify', 'hugin_executor'
        ]
        
        missing_tools = []
        for tool in required_tools:
            try:
                subprocess.run([tool, '--help'], capture_output=True, timeout=5)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                missing_tools.append(tool)
        
        if missing_tools:
            raise HuginPipelineError(f"Missing Hugin tools: {', '.join(missing_tools)}")
            
        logger.info(f"✅ Hugin installation verified: {len(required_tools)} tools available")
        
    def _run_command(self, cmd: List[str], step_name: str, timeout: int = 300) -> Tuple[str, str]:
        """Execute command with logging and error handling."""
        logger.info(f"🔧 Executing {step_name}: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.stdout:
                logger.debug(f"📝 {step_name} stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"📝 {step_name} stderr: {result.stderr}")
                
            if result.returncode != 0:
                raise HuginPipelineError(
                    f"{step_name} failed with code {result.returncode}: {result.stderr}"
                )
                
            return result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            raise HuginPipelineError(f"{step_name} timed out after {timeout}s")
        except FileNotFoundError:
            raise HuginPipelineError(f"{step_name} command not found: {cmd[0]}")
            
    def stitch_panorama(self, images: List[str], output_file: str = 'panorama.jpg', progress_callback: Optional[Callable] = None) -> str:
        """
        Complete Hugin panorama stitching using proven 7-step workflow.
        
        Args:
            images: List of image file paths (expects 16 images)
            output_file: Output panorama filename
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to completed panorama
        """
        logger.info(f"🎯 Starting Hugin panorama stitching for {len(images)} images")
        
        if len(images) != 16:
            raise HuginPipelineError(f"Expected exactly 16 images, got {len(images)}")
        
        start_time = time.time()
        
        # Always validate image files first
        self._validate_input_images(images)
        
        # Convert to absolute paths before changing directory
        absolute_images = [os.path.abspath(img_path) for img_path in images]
        
        # Change to temp directory for processing
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Copy images to temp directory using pre-resolved absolute paths
            local_images = []
            for i, abs_img_path in enumerate(absolute_images):
                local_name = f'img{i+1:02d}.jpg'
                shutil.copy2(abs_img_path, local_name)
                local_images.append(local_name)
                
            if progress_callback:
                progress_callback(0.1, "Copied images to working directory")
            
            pto_file = 'project.pto'
            
            # Step 1: Generate .pto project file with iPhone ultra-wide settings
            logger.info("🚀 Step 1: Generating project file (iPhone ultra-wide optimized)")
            self._run_command([
                'pto_gen', 
                '-f', '120',           # iPhone ultra-wide: 120° horizontal FOV
                '-p', '0',             # Rectilinear projection (standard lens model)
                '-o', pto_file
            ] + local_images, "pto_gen")
            if progress_callback:
                progress_callback(0.2, "Generated project file")
            
            # Step 2: Find control points (balanced for ultra-wide + cloud resources)
            logger.info("🔍 Step 2: Finding control points (cloud-optimized)")
            self._run_command([
                'cpfind', 
                '--multirow',                    # Multi-row algorithm for spherical
                '--sieve1width', '30',           # Moderate sieve width for wide-angle
                '--sieve1height', '30',          # Moderate sieve height for wide-angle  
                '--sieve1size', '200',           # Balanced keypoints per bucket
                '--ransaciter', '1000',          # Standard RANSAC iterations
                '-o', pto_file, pto_file
            ], "cpfind", timeout=600)
            if progress_callback:
                progress_callback(0.4, "Found control points")
            
            # Step 3: Clean control points
            logger.info("🧹 Step 3: Cleaning control points")
            self._run_command(['celeste_standalone', '-i', pto_file, '-o', pto_file], "celeste_standalone")
            self._run_command(['cpclean', '-o', pto_file, pto_file], "cpclean")
            if progress_callback:
                progress_callback(0.5, "Cleaned control points")
            
            # Step 4: Find vertical lines
            logger.info("📐 Step 4: Finding vertical lines")
            self._run_command(['linefind', '-o', pto_file, pto_file], "linefind")
            if progress_callback:
                progress_callback(0.6, "Found vertical lines")
            
            # Step 5: Optimize geometry and photometry (enhanced for spherical 360°)
            logger.info("⚙️ Step 5: Optimizing spherical panorama")
            self._run_command([
                'autooptimiser', 
                '-a',              # Automatic position optimization
                '-m',              # Photometric optimization (exposure, vignetting)
                '-l',              # Level horizon (straighten)
                '-s',              # Smart output projection selection
                '-o', pto_file, pto_file
            ], "autooptimiser")
            if progress_callback:
                progress_callback(0.7, "Optimized panorama")
            
            # Step 6: Set equirectangular projection (optimized for spherical 360°)
            logger.info("🌐 Step 6: Setting spherical equirectangular projection")
            self._run_command([
                'pano_modify', 
                '--projection=2',           # Equirectangular (spherical)
                '--fov=360x180',           # Full 360° horizontal × 180° vertical 
                '--canvas=8192x4096',      # High resolution 2:1 aspect ratio
                '--crop=AUTO',             # Autocrop to maximal panorama size
                '--center',                # Center the panorama
                '--straighten',            # Level the horizon
                '-o', pto_file, pto_file
            ], "pano_modify")
            if progress_callback:
                progress_callback(0.8, "Set equirectangular projection")
            
            # Step 7: Final stitching with progressive fallback strategy
            logger.info("🎨 Step 7: Final stitching with memory optimization")
            
            # Try optimized stitching first (reduced memory, faster processing)
            success = self._try_optimized_stitching(pto_file, progress_callback)
            if not success:
                logger.warning("⚠️ Optimized stitching failed, trying standard stitching")
                success = self._try_standard_stitching(pto_file, progress_callback)
            
            if not success:
                logger.error("❌ All stitching strategies failed")
                raise HuginPipelineError("All stitching methods failed - enblend cannot process this image set")
            if progress_callback:
                progress_callback(0.95, "Completed stitching")
            
            # Check for output
            stitched_tif = 'stitched.tif'
            if not os.path.exists(stitched_tif):
                raise HuginPipelineError("Stitching failed; no output file created")
                
            # Convert TIFF to final format if needed
            final_output_path = os.path.join(self.temp_dir, output_file)
            if output_file.lower().endswith('.tif'):
                shutil.copy2(stitched_tif, final_output_path)
            else:
                # Convert TIFF to JPG using Pillow
                from PIL import Image
                with Image.open(stitched_tif) as img:
                    img.convert('RGB').save(final_output_path, 'JPEG', quality=95)
            
            if progress_callback:
                progress_callback(1.0, "Panorama completed successfully")
                
            processing_time = time.time() - start_time
            logger.info(f"✅ Panorama stitching completed in {processing_time:.1f}s: {final_output_path}")
            
            return final_output_path
            
        except HuginPipelineError:
            # Re-raise Hugin-specific errors without modification
            raise
        except Exception as e:
            raise HuginPipelineError(f"Panorama stitching failed: {e}")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def _try_optimized_stitching(self, pto_file: str, progress_callback: Optional[Callable] = None) -> bool:
        """Try optimized stitching designed specifically for iPhone ultra-wide excessive overlap."""
        try:
            logger.info("🚀 Attempting iPhone ultra-wide optimized stitching")
            
            # Set environment variables for memory optimization
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '2'  # Limit CPU threads
            env['TMPDIR'] = '/tmp'  # Use fast temporary directory
            
            # Use nona + research-based enblend parameters for iPhone ultra-wide
            # This bypasses hugin_executor's overhead
            logger.info("📸 Step 7a: Generating panorama tiles for iPhone ultra-wide")
            
            result = subprocess.run([
                'nona', '-o', 'img_', pto_file
            ], capture_output=True, text=True, timeout=600, env=env)
            
            if result.returncode != 0:
                logger.warning(f"nona failed: {result.stderr}")
                return False
                
            if progress_callback:
                progress_callback(0.9, "Generated panorama tiles")
            
            # Find generated images  
            img_files = sorted([f for f in os.listdir('.') if f.startswith('img_') and f.endswith('.tif')])
            if not img_files:
                logger.warning("No nona output images found")
                return False
                
            logger.info(f"🎨 Step 7b: iPhone ultra-wide blending - handling excessive overlap")
            
            # RESEARCH-BASED FIX: iPhone 13mm ultra-wide (120° FOV) creates ~50-60% overlap
            # which triggers enblend's "excessive overlap" error. Solutions from research:
            
            # Try Method 1: enblend with research-verified parameters for excessive overlap
            logger.info("🔬 Method 1: Research-based enblend excessive overlap handling")
            result = subprocess.run([
                'enblend', 
                '--no-optimize',         # CRITICAL: Skip problematic overlap optimization
                '--fine-mask',           # Higher resolution masks for overlap issues
                '-l', '10',              # Reduce blending levels for excessive overlap
                '--compression=lzw',     # Maintain compression
                '--primary-seam-generator=nearest-feature-transform',  # Alternative seam method
                '-o', 'stitched.tif'
            ] + img_files, capture_output=True, text=True, timeout=900, env=env)
            
            if result.returncode == 0:
                logger.info("✅ Method 1 successful: enblend with excessive overlap handling")
                # Clean up intermediate files
                for img_file in img_files:
                    try:
                        os.remove(img_file)
                    except:
                        pass
                        
                if progress_callback:
                    progress_callback(0.95, "Completed ultra-wide optimized stitching")
                    
                return os.path.exists('stitched.tif')
            else:
                logger.warning(f"Method 1 failed: {result.stderr}")
                
            # Method 2: Simple averaging blending (emergency fallback)
            logger.info("🛟 Method 2: Emergency averaging blend for excessive overlap")
            return self._emergency_average_blend(img_files, progress_callback)
            
        except subprocess.TimeoutExpired:
            logger.warning("⏰ Ultra-wide stitching timed out")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Ultra-wide stitching failed: {e}")
            return False

    def _emergency_average_blend(self, img_files: list, progress_callback: Optional[Callable] = None) -> bool:
        """Emergency fallback: Simple average blending for excessive overlap scenarios."""
        try:
            import cv2
            import numpy as np
            
            logger.info(f"🚨 Emergency blending {len(img_files)} images with simple averaging")
            
            # Load first image to get dimensions
            first_img = cv2.imread(img_files[0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if first_img is None:
                return False
                
            height, width = first_img.shape[:2]
            logger.info(f"📐 Processing images at {width}x{height}")
            
            # Initialize accumulation arrays
            accumulated = np.zeros((height, width, 3), dtype=np.float64)
            count = np.zeros((height, width), dtype=np.uint16)
            
            # Process each image
            for i, img_file in enumerate(img_files):
                img = cv2.imread(img_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if img is None:
                    continue
                    
                # Convert to float64 for accumulation
                img_float = img.astype(np.float64)
                
                # Create mask for valid pixels (non-black areas)
                mask = np.any(img_float > 0, axis=2)
                
                # Accumulate valid pixels
                accumulated[mask] += img_float[mask]
                count[mask] += 1
                
                logger.info(f"📊 Processed image {i+1}/{len(img_files)}: {img_file}")
            
            # Average the accumulated values
            mask = count > 0
            result = np.zeros_like(accumulated)
            result[mask] = accumulated[mask] / count[mask, np.newaxis]
            
            # Convert back to appropriate data type and save
            if first_img.dtype == np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)
            else:  # Assume 16-bit
                result = np.clip(result, 0, 65535).astype(np.uint16)
            
            cv2.imwrite('stitched.tif', result)
            logger.info("✅ Emergency blend completed successfully")
            
            # Clean up intermediate files
            for img_file in img_files:
                try:
                    os.remove(img_file)
                except:
                    pass
                    
            if progress_callback:
                progress_callback(0.95, "Completed emergency blend")
                
            return os.path.exists('stitched.tif')
            
        except Exception as e:
            logger.error(f"❌ Emergency blend failed: {e}")
            return False

    def _try_standard_stitching(self, pto_file: str, progress_callback: Optional[Callable] = None) -> bool:
        """Try standard hugin_executor stitching with timeout protection."""
        try:
            logger.info("🔄 Attempting standard stitching (hugin_executor)")
            
            # Set environment for reduced resource usage
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'  # Single thread for stability
            env['TMPDIR'] = '/tmp'
            
            # Shorter timeout for standard method (10 minutes max)
            result = subprocess.run([
                'hugin_executor', '--stitching', '--prefix=stitched', pto_file
            ], capture_output=True, text=True, timeout=600, env=env)
            
            if result.returncode != 0:
                logger.warning(f"hugin_executor failed: {result.stderr}")
                return False
                
            if progress_callback:
                progress_callback(0.95, "Completed standard stitching")
                
            return os.path.exists('stitched.tif')
            
        except subprocess.TimeoutExpired:
            logger.warning("⏰ Standard stitching timed out after 10 minutes")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Standard stitching failed: {e}")
            return False

    def _validate_input_images(self, images: List[str]):
        """Validate that all input image files exist and are readable."""
        logger.info(f"🔍 Validating {len(images)} input images...")
        
        for i, img_path in enumerate(images):
            if not os.path.exists(img_path):
                raise HuginPipelineError(f"Image {i} missing: {img_path}")
                
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
                    logger.info(f"📸 Image {i}: {os.path.basename(img_path)} {width}×{height} ({mode})")
            except Exception as e:
                raise HuginPipelineError(f"Failed to read image {img_path}: {e}")
                
        logger.info("✅ All input images validated successfully")


def create_hugin_service(temp_dir: Optional[str] = None, canvas_size: Tuple[int, int] = (8192, 4096)) -> HuginPipelineService:
    """Factory function to create Hugin pipeline service."""
    return HuginPipelineService(temp_dir=temp_dir, canvas_size=canvas_size)
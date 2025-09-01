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
import json
import math

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
            'linefind', 'autooptimiser', 'pano_modify', 'hugin_executor',
            'pto_var', 'geocpset'  # Added for sensor-guided stitching
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
    
    def _convert_ios_to_hugin_coordinates(self, capture_points: List[Dict]) -> List[Dict]:
        """
        Convert iOS ARKit coordinates to Hugin coordinate system.
        
        Key insight: iOS sends absolute compass directions, but Hugin needs
        coordinates relative to the calibration point (first image) as front center.
        
        Strategy: Make first capture point (calibration dot) face forward (yaw=0°)
        and all other points relative to that reference.
        """
        hugin_poses = []
        
        if not capture_points:
            return hugin_poses
            
        # Get calibration reference point (first capture)
        calibration_azimuth = capture_points[0].get('azimuth', 0.0)
        logger.info(f"🎯 Calibration reference: iOS azimuth={calibration_azimuth:.1f}°")
        
        for i, point in enumerate(capture_points):
            ios_azimuth = point.get('azimuth', 0.0)
            ios_elevation = point.get('elevation', 0.0) 
            ios_roll = point.get('roll', 0.0)  # Device roll angle
            
            # DEBUG: Log original iOS values
            logger.info(f"   📍 Point {i}: iOS azimuth={ios_azimuth:.1f}°, elevation={ios_elevation:.1f}°")
            
            # Convert to relative coordinates with calibration point as front (yaw=0°)
            relative_azimuth = (ios_azimuth - calibration_azimuth) % 360
            
            # Convert to Hugin coordinates where calibration point faces front center (yaw=0°)
            # For the calibration point: relative_azimuth=0° should map to hugin_yaw=0° (front)
            # iOS coordinates are clockwise from East, Hugin coordinates are clockwise from North
            # But we want calibration point at front (0°), not North
            hugin_yaw = (360 - relative_azimuth) % 360
            hugin_pitch = ios_elevation  # Direct mapping
            hugin_roll = ios_roll       # Direct mapping
            
            # DEBUG: Log converted values
            logger.info(f"      → Relative: {relative_azimuth:.1f}°, Hugin yaw={hugin_yaw:.1f}°, pitch={hugin_pitch:.1f}°")
            
            hugin_poses.append({
                'image_index': i,
                'yaw': hugin_yaw,
                'pitch': hugin_pitch, 
                'roll': hugin_roll,
                'ios_source': {
                    'azimuth': ios_azimuth,
                    'elevation': ios_elevation,
                    'roll': ios_roll
                }
            })
            
        logger.info(f"🧭 Converted {len(hugin_poses)} iOS poses to Hugin coordinates")
        return hugin_poses
    
    def _inject_poses_into_pto(self, pto_file: str, poses: List[Dict]) -> None:
        """
        Inject yaw/pitch/roll poses into PTO file using pto_var.
        This provides Hugin with accurate starting positions from iPhone sensors.
        """
        logger.info("📍 Injecting sensor poses into project file")
        
        # Build pto_var command with all pose parameters
        pto_var_cmd = ['pto_var']
        
        for pose in poses:
            i = pose['image_index']
            pto_var_cmd.extend([
                f'--set', f'y{i}={pose["yaw"]:.2f}',     # Yaw (azimuth)  
                f'--set', f'p{i}={pose["pitch"]:.2f}',   # Pitch (elevation)
                f'--set', f'r{i}={pose["roll"]:.2f}'     # Roll
            ])
        
        # Output to temporary file, then replace original
        temp_pto = pto_file + '.tmp'
        pto_var_cmd.extend(['-o', temp_pto, pto_file])
        
        self._run_command(pto_var_cmd, "pto_var pose injection")
        
        # Replace original with pose-injected version
        shutil.move(temp_pto, pto_file)
        
        logger.info(f"✅ Injected {len(poses)} sensor poses into {pto_file}")
        
    def _get_first_image_yaw(self, pto_file: str) -> float:
        """Extract the yaw angle of the first image from PTO file."""
        try:
            with open(pto_file, 'r') as f:
                content = f.read()
                
            # Look for first image line (i n"..." y... p... r...)
            import re
            match = re.search(r'i n"[^"]*"\s+.*?\by([-\d\.]+)', content)
            if match:
                return float(match.group(1))
                
        except Exception as e:
            logger.warning(f"Could not extract first image yaw: {e}")
            
        return 0.0
        
    def _get_image_yaw(self, pto_file: str, image_index: int) -> float:
        """Extract the yaw angle of a specific image from PTO file."""
        try:
            with open(pto_file, 'r') as f:
                lines = f.readlines()
                
            # Find the image line for the specific index
            image_line_count = 0
            for line in lines:
                if line.startswith('i '):
                    if image_line_count == image_index:
                        import re
                        match = re.search(r'\by([-\d\.]+)', line)
                        if match:
                            return float(match.group(1))
                    image_line_count += 1
                        
        except Exception as e:
            logger.warning(f"Could not extract image {image_index} yaw: {e}")
            
        return 0.0
        
    def _calculate_fov_from_exif(self, image_path: str) -> float:
        """Calculate field of view from EXIF focal length data."""
        try:
            # Import piexif locally like the rest of the server code
            import piexif
            
            # Read EXIF data from the image
            exif_dict = piexif.load(image_path)
            
            # Look for focal length in EXIF data
            focal_length = None
            if "Exif" in exif_dict and piexif.ExifIFD.FocalLength in exif_dict["Exif"]:
                focal_length_rational = exif_dict["Exif"][piexif.ExifIFD.FocalLength]
                if isinstance(focal_length_rational, tuple) and len(focal_length_rational) == 2:
                    focal_length = focal_length_rational[0] / focal_length_rational[1]
                else:
                    focal_length = float(focal_length_rational)
            
            if focal_length:
                # iPhone sensor specs for FOV calculation
                # iPhone ultra-wide: ~2.2mm focal length = ~106.2° FOV
                # Standard calculation: FOV = 2 * atan(sensor_width / (2 * focal_length))
                # For iPhone ultra-wide sensor: approximate sensor width = 4.3mm
                sensor_width = 4.3  # mm (iPhone ultra-wide sensor width)
                fov_radians = 2 * math.atan(sensor_width / (2 * focal_length))
                fov_degrees = math.degrees(fov_radians)
                
                logger.info(f"📸 EXIF Focal Length: {focal_length:.2f}mm → FOV: {fov_degrees:.1f}°")
                
                # Sanity check: iPhone ultra-wide should be between 100-120°
                if 95 <= fov_degrees <= 130:
                    return fov_degrees
                else:
                    logger.warning(f"⚠️ Calculated FOV {fov_degrees:.1f}° outside expected range (95-130°)")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not extract FOV from EXIF: {e}")
            
        # Fallback to iPhone ultra-wide measured FOV
        logger.info("📸 Using measured iPhone 15 Pro ultra-wide FOV: 106.2°")
        return 106.2
        
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
            
    def stitch_panorama(self, images: List[str], output_file: str = 'panorama.jpg', session_metadata: Optional[Dict] = None, progress_callback: Optional[Callable] = None) -> str:
        """
        Complete Hugin panorama stitching using proven 7-step workflow.
        
        Args:
            images: List of image file paths (expects 16 images)
            output_file: Output panorama filename
            session_metadata: Optional iOS session metadata with capture points
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
            
            # Calculate FOV from EXIF data instead of using hardcoded value
            calculated_fov = self._calculate_fov_from_exif(local_images[0])
            fov_str = str(int(round(calculated_fov)))
            
            # Step 1: Generate .pto project file with EXIF-based camera settings
            logger.info(f"🚀 Step 1: Generating project file (EXIF-based FOV: {calculated_fov:.1f}°)")
            self._run_command([
                'pto_gen', 
                '-f', fov_str,         # Dynamic FOV from EXIF focal length
                '-p', '0',             # Rectilinear projection (standard lens model)
                '-o', pto_file
            ] + local_images, "pto_gen")
            if progress_callback:
                progress_callback(0.2, "Generated project file")
            
            # Check if we have sensor data for guided stitching
            use_sensor_guidance = (session_metadata and 
                                 'capturePoints' in session_metadata and 
                                 len(session_metadata['capturePoints']) == len(images))
                                 
            # Debug logging for workflow selection
            if session_metadata:
                logger.info(f"📊 Session metadata available: {list(session_metadata.keys())}")
                if 'capturePoints' in session_metadata:
                    logger.info(f"📍 Capture points: {len(session_metadata['capturePoints'])}, Images: {len(images)}")
                else:
                    logger.warning("❌ No capturePoints in session metadata")
            else:
                logger.warning("❌ No session metadata provided")
                                 
            if use_sensor_guidance:
                logger.info("🧭 Using sensor-guided stitching workflow")
                
                try:
                    # Step 1.5: Inject sensor poses from iOS app
                    capture_points = session_metadata['capturePoints']
                    hugin_poses = self._convert_ios_to_hugin_coordinates(capture_points)
                    self._inject_poses_into_pto(pto_file, hugin_poses)
                    
                    if progress_callback:
                        progress_callback(0.25, "Injected sensor poses")
                    
                    # Step 2: Find control points with prealigned optimization
                    logger.info("🔍 Step 2: Finding control points (sensor-guided --prealigned)")
                    self._run_command([
                        'cpfind', 
                        '--prealigned',                  # Use sensor poses as starting point
                        '--sieve1width', '50',           # Maximum recommended for ultra-wide
                        '--sieve1height', '50',          # Maximum recommended for ultra-wide  
                        '--sieve1size', '300',           # Moderate keypoints (sensors provide positioning)
                        '--ransaciter', '2000',          # Reduced RANSAC (sensors guide matching)
                        '-o', pto_file, pto_file
                    ], "cpfind", timeout=800)
                    
                    if progress_callback:
                        progress_callback(0.35, "Found control points (sensor-guided)")
                        
                    # Step 2.5: Add geometric safety net for sparse areas
                    logger.info("🛡️ Adding geometric safety net")
                    self._run_command(['geocpset', '-o', pto_file, pto_file], "geocpset")
                    
                    sensor_guided_success = True
                    
                except Exception as e:
                    logger.warning(f"⚠️ Sensor-guided stitching failed: {e}")
                    logger.info("🔄 Falling back to traditional feature-based stitching")
                    use_sensor_guidance = False
                    sensor_guided_success = False
                
            if not use_sensor_guidance:
                logger.info("🔍 Using traditional feature-based stitching")
                
                # Step 2: Find control points (balanced iPhone ultra-wide ground detection)
                logger.info("🔍 Step 2: Finding control points (balanced ultra-wide ground detection)")
                self._run_command([
                    'cpfind', 
                    '--multirow',                    # Multi-row algorithm for spherical 360°
                    '--sieve1width', '50',           # Maximum recommended for ultra-wide
                    '--sieve1height', '50',          # Maximum recommended for ultra-wide  
                    '--sieve1size', '500',           # 1.25M keypoints per image (balanced)
                    '--sieve2size', '3',             # Keep 3 points per region (vs default 2)
                    '--ransaciter', '3000',          # Enhanced RANSAC for low-texture
                    '--ransacdist', '20',            # Balanced threshold for precision
                    '--kdtreesteps', '300',          # More feature matching iterations
                    '--kdtreeseconddist', '0.35',    # Slightly relaxed for ground textures
                    '-o', pto_file, pto_file
                ], "cpfind", timeout=1000)
            if progress_callback:
                progress_callback(0.4, "Found control points")
            
            # Step 3: Clean control points (skip celeste for indoor/ground panoramas)
            logger.info("🧹 Step 3: Cleaning control points (preserving ground features)")
            # Skip celeste_standalone - it removes ground textures mistaken for clouds
            # For indoor/360° panoramas, celeste is counterproductive
            logger.info("   Skipping celeste (preserves ground control points)")
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
            ], "autooptimiser", timeout=700)  # Extended timeout for enhanced control points
            if progress_callback:
                progress_callback(0.7, "Optimized panorama")
            
            # Step 6: Set equirectangular projection and handle calibration dot facing
            logger.info("🌐 Step 6: Setting spherical equirectangular projection")
            
            if use_sensor_guidance:
                # For sensor-guided mode, face the calibration dot (first image) front
                logger.info("🎯 Positioning calibration dot at front center")
                
                # Get current yaw of first image after optimization
                first_image_yaw = self._get_first_image_yaw(pto_file)
                logger.info(f"   First image current yaw: {first_image_yaw:.1f}°")
                
                # Calculate rotation needed to put first image at front (yaw=0°)
                rotation_needed = -first_image_yaw
                
                # Apply rotation to all images to center calibration dot
                if abs(rotation_needed) > 1.0:  # Only if significant rotation needed
                    logger.info(f"   Applying {rotation_needed:.1f}° rotation to center calibration dot")
                    
                    # Build pto_var command to rotate all images
                    pto_var_cmd = ['pto_var']
                    for i in range(len(images)):
                        current_yaw = self._get_image_yaw(pto_file, i) if i > 0 else first_image_yaw
                        new_yaw = (current_yaw + rotation_needed) % 360
                        pto_var_cmd.extend(['--set', f'y{i}={new_yaw:.2f}'])
                    
                    temp_pto = pto_file + '.rotation'
                    pto_var_cmd.extend(['-o', temp_pto, pto_file])
                    self._run_command(pto_var_cmd, "calibration dot centering")
                    shutil.move(temp_pto, pto_file)
                
                # Set projection without --center to preserve our calibration dot positioning
                self._run_command([
                    'pano_modify', 
                    '--projection=2',           # Equirectangular (spherical)
                    '--fov=360x180',           # Full 360° horizontal × 180° vertical 
                    '--canvas=8192x4096',      # High resolution 2:1 aspect ratio
                    '--crop=AUTO',             # Autocrop to maximal panorama size
                    '--straighten',            # Level the horizon (no --center)
                    '-o', pto_file, pto_file
                ], "pano_modify")
            else:
                # Traditional mode with centering
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
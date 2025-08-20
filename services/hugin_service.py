#!/usr/bin/env python3
"""
Hugin Pipeline Service

Extracts the complete 7-step Hugin panorama stitching workflow into an independent service.
This allows isolated debugging of each Hugin processing stage:

1. pto_gen: Generate project file with ARKit positioning
2. cpfind: Find control points using multirow strategy
3. cpclean: Clean and validate control points
4. autooptimiser: Optimize geometry and photometrics
5. pano_modify: Set output parameters (canvas, crop, projection)
6. nona: Render images to equirectangular coordinates
7. enblend: Multi-band blending for seamless panorama

Each step can be debugged independently to isolate processing issues.
"""

import os
import time
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import tempfile
import math

logger = logging.getLogger(__name__)

class HuginPipelineError(Exception):
    """Raised when Hugin pipeline step fails."""
    pass

class HuginStep:
    """Represents a single step in the Hugin pipeline."""
    def __init__(self, name: str, description: str, timeout: int = 300):
        self.name = name
        self.description = description
        self.timeout = timeout
        self.start_time = None
        self.end_time = None
        self.success = False
        self.error = None
        self.output_files = []
        
    def start(self):
        self.start_time = time.time()
        logger.info(f"🚀 Starting {self.name}: {self.description}")
        
    def complete(self, success: bool = True, error: Optional[str] = None, output_files: List[str] = None):
        self.end_time = time.time()
        self.success = success
        self.error = error
        self.output_files = output_files or []
        
        duration = self.end_time - self.start_time if self.start_time else 0
        if success:
            logger.info(f"✅ Completed {self.name} in {duration:.1f}s")
        else:
            logger.error(f"❌ Failed {self.name} after {duration:.1f}s: {error}")

class HuginPipelineService:
    """
    Service for executing the complete Hugin panorama stitching pipeline.
    
    Provides isolated execution of each pipeline stage with comprehensive
    debugging, validation, and error handling.
    """
    
    def __init__(self, temp_dir: Optional[str] = None, canvas_size: Tuple[int, int] = (8192, 4096)):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="hugin_pipeline_")
        self.canvas_size = canvas_size
        self.pipeline_steps = []
        self.arkit_mode = False
        self.geocpset_used = False
        self.control_points_found = 0
        
        # Verify Hugin installation
        self._verify_hugin_installation()
        
        logger.info(f"🏗️ Hugin Pipeline Service initialized")
        logger.info(f"   Temp directory: {self.temp_dir}")
        logger.info(f"   Canvas size: {canvas_size[0]}×{canvas_size[1]}")
        
    def _verify_hugin_installation(self):
        """Verify all required Hugin tools are available."""
        required_tools = [
            'pto_gen', 'cpfind', 'cpclean', 'linefind', 
            'autooptimiser', 'pano_modify', 'nona', 'enblend', 'geocpset'
        ]
        
        missing_tools = []
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)
                
        if missing_tools:
            raise HuginPipelineError(f"Missing Hugin tools: {missing_tools}")
            
        logger.info(f"✅ Hugin installation verified: {len(required_tools)} tools available")
        
    def _run_command(self, cmd: List[str], step_name: str, timeout: int = 300) -> Tuple[str, str]:
        """Execute command with logging and error handling."""
        logger.info(f"🔧 Executing {step_name}: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.temp_dir
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
            
    def execute_pipeline(self, 
                        images: List[str], 
                        converted_coordinates: List[Dict] = None,
                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute complete Hugin pipeline with converted coordinates.
        
        Args:
            images: List of image file paths
            converted_coordinates: Converted coordinates from coordinate service
            progress_callback: Optional callback for progress updates
            
        Returns:
            Pipeline execution results with debugging information
        """
        logger.info(f"🎯 Starting Hugin pipeline for {len(images)} images")
        
        self.arkit_mode = converted_coordinates is not None and len(converted_coordinates) > 0
        self.pipeline_steps = []
        
        start_time = time.time()
        
        try:
            # Step 1: Generate project file
            project_file = self._step_1_generate_project(images, converted_coordinates, progress_callback)
            
            # Step 2: Find control points
            cp_project = self._step_2_find_control_points(project_file, len(images), progress_callback)
            
            # Step 3: Clean control points (conditional)
            clean_project = self._step_3_clean_control_points(cp_project, progress_callback)
            
            # Step 3.5: Detect lines
            line_project = self._step_3_5_detect_lines(clean_project, progress_callback)
            
            # Step 4: Optimize panorama
            opt_project = self._step_4_optimize(line_project, progress_callback)
            
            # Step 5: Set output parameters
            final_project = self._step_5_set_output(opt_project, progress_callback)
            
            # Step 6: Render images
            tiff_files = self._step_6_render_images(final_project, len(images), progress_callback)
            
            # Step 7: Export TIFF files for blending service (microservices architecture)
            validated_tiffs = self._step_7_return_tiff_files(tiff_files, len(images), progress_callback)
            
            total_time = time.time() - start_time
            
            # Compilation results
            results = {
                'success': True,
                'tiff_files': validated_tiffs,
                'processing_time': total_time,
                'pipeline_steps': [
                    {
                        'name': step.name,
                        'description': step.description,
                        'success': step.success,
                        'duration': step.end_time - step.start_time if step.start_time and step.end_time else 0,
                        'error': step.error,
                        'output_files': step.output_files
                    } for step in self.pipeline_steps
                ],
                'statistics': {
                    'arkit_mode': self.arkit_mode,
                    'geocpset_used': self.geocpset_used,
                    'control_points_found': self.control_points_found,
                    'rendered_images': len(tiff_files) if 'tiff_files' in locals() else 0
                }
            }
            
            logger.info(f"🎉 Hugin pipeline completed successfully in {total_time:.1f}s")
            return results
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Hugin pipeline failed after {total_time:.1f}s: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': total_time,
                'pipeline_steps': [
                    {
                        'name': step.name,
                        'description': step.description,
                        'success': step.success,
                        'duration': step.end_time - step.start_time if step.start_time and step.end_time else 0,
                        'error': step.error,
                        'output_files': step.output_files
                    } for step in self.pipeline_steps
                ]
            }
            
    def _step_1_generate_project(self, images: List[str], converted_coordinates: List[Dict], progress_callback: Optional[Callable]) -> str:
        """Step 1: Generate project file with ARKit positioning data."""
        step = HuginStep("pto_gen", "Generate project file with ARKit positioning")
        step.start()
        self.pipeline_steps.append(step)
        
        if progress_callback:
            progress_callback(0.1, "Generating project file...")
            
        project_file = os.path.join(self.temp_dir, "project.pto")
        
        try:
            if self.arkit_mode:
                # Generate positioned project with ARKit data
                self._generate_positioned_project(images, converted_coordinates, project_file)
            else:
                # Basic project generation
                cmd = ["pto_gen", "-o", project_file] + images
                self._run_command(cmd, "pto_gen")
                
            step.complete(success=True, output_files=[project_file])
            logger.info(f"✅ Generated project file with {len(images)} images")
            return project_file
            
        except Exception as e:
            step.complete(success=False, error=str(e))
            raise HuginPipelineError(f"Project generation failed: {e}")
            
    def _generate_positioned_project(self, images: List[str], converted_coordinates: List[Dict], project_file: str):
        """Generate PTO file with converted ARKit coordinates."""
        logger.info(f"🎯 Generating positioned project with {len(converted_coordinates)} converted coordinates")
        
        # Debug: Verify image files exist and get dimensions
        logger.info(f"🔍 DEBUG: Verifying {len(images)} input images")
        for i, img_path in enumerate(images):
            if not os.path.exists(img_path):
                logger.error(f"❌ Image {i} missing: {img_path}")
                continue
                
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
                    logger.info(f"📸 Image {i}: {os.path.basename(img_path)} {width}×{height} ({mode})")
            except Exception as e:
                logger.error(f"❌ Could not read image {i}: {e}")
                raise HuginPipelineError(f"Failed to read image {img_path}: {e}")  # Propagate error
                
        logger.info(f"🔍 DEBUG: Processing {len(converted_coordinates)} coordinate mappings")
        
        # iPhone ultra-wide lens parameters (corrected based on expert analysis)
        # iPhone 11 Pro and newer: ~120° actual FOV measured
        iphone_ultrawide_fov = 120.0  # Horizontal FOV in degrees (iPhone 11+ ultra-wide)
        iphone_projection = 0  # Rectilinear projection (f0 in PTO) - CRITICAL FIX
        
        with open(project_file, 'w') as f:
            # Write PTO header with panorama parameters
            f.write("# Generated by Hugin Pipeline Service with ARKit positioning\n")
            f.write(f"p f0 w{self.canvas_size[0]} h{self.canvas_size[1]} v360 E0 R0 n\"TIFF_m\"\n")
            f.write("m g1 i0 f0 m2 p0.00784314\n")
            f.write("\n")
            
            # Write image lines with converted coordinates and actual dimensions
            for i, (image_path, coord_data) in enumerate(zip(images, converted_coordinates)):
                hugin_output = coord_data['hugin_output']
                yaw = hugin_output['yaw']
                pitch = hugin_output['pitch']
                roll = hugin_output['roll']
                
                # Get actual image dimensions instead of hard-coding
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        actual_width, actual_height = img.size
                except Exception as e:
                    logger.error(f"❌ Could not get dimensions for {image_path}: {e}")
                    raise HuginPipelineError(f"Failed to read image dimensions: {e}")  # Propagate error
                
                # Write image line with rectilinear projection, accurate FOV, and distortion correction
                # iPhone ultra-wide lens distortion parameters (typical values for barrel distortion)
                barrel_a = -0.05   # Barrel distortion coefficient (negative for barrel, positive for pincushion)
                barrel_b = 0.01    # Secondary distortion coefficient
                barrel_c = 0.001   # Tertiary distortion coefficient
                
                f.write(f'i w{actual_width} h{actual_height} f{iphone_projection} v{iphone_ultrawide_fov:.1f} '
                       f'Ra0 Rb0 Rc0 Rd0 Re0 Ef0 Er1 Eb1 '
                       f'r{roll:.6f} p{pitch:.6f} y{yaw:.6f} '
                       f'TrX0 TrY0 TrZ0 Tpy0 Tpp0 '
                       f'j0 a{barrel_a} b{barrel_b} c{barrel_c} '  # FIXED: Add distortion correction
                       f'd0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 '
                       f'n"{os.path.abspath(image_path)}"\n')
                       
                logger.debug(f"   📍 Image {i:2d}: y={yaw:.1f}°, p={pitch:.1f}°, r={roll:.1f}°")
            
            # Add optimization variables (for lens/FOV only, to keep positions fixed)
            f.write("\n# Optimization variables (lens distortion and FOV, shared across images)\n")
            f.write("v v0\n")  # Optimize FOV if needed
            f.write("v a0\n")  # Optimize distortion parameter a
            f.write("v b0\n")  # Optimize distortion parameter b  
            f.write("v c0\n")  # Optimize distortion parameter c
            f.write("\n")
            
        logger.info(f"✅ Generated positioned PTO with ARKit data covering {len(converted_coordinates)} viewpoints")
        
    def _step_2_find_control_points(self, project_file: str, image_count: int, progress_callback: Optional[Callable]) -> str:
        """Step 2: Find control points using multirow strategy."""
        step = HuginStep("cpfind", "Find control points with multirow strategy")
        step.start()
        self.pipeline_steps.append(step)
        
        if progress_callback:
            progress_callback(0.3, "Finding control points...")
            
        cp_project = os.path.join(self.temp_dir, "project_cp.pto")
        
        try:
            # iPhone ultra-wide optimized strategy for better control point detection
            cmd = [
                "cpfind",
                "--multirow",  # Essential for spherical coverage
                "--fullscale", # Use full resolution for better feature detection
                "--sieve1-width=40",   # Reduced for more sensitive detection
                "--sieve1-height=40",  # Reduced for more sensitive detection  
                "--sieve1-size=100",   # Lower threshold for more control points
                "--ransac-dist=50",    # More lenient for ultra-wide distortion
                "--kall",             # Keep all control points (less filtering)
                # Removed --celeste to preserve horizon features
                "-o", cp_project,
                project_file
            ]
            
            self._run_command(cmd, "cpfind", timeout=600)
            
            # Check if cpfind created output file
            if not os.path.exists(cp_project):
                # Primary strategy failed - try fallback with even more lenient parameters
                logger.warning("⚠️ Primary cpfind strategy failed - trying fallback...")
                
                fallback_cmd = [
                    "cpfind",
                    "--multirow",
                    "--sieve1-width=30",   # Even more sensitive
                    "--sieve1-height=30", 
                    "--sieve1-size=50",    # Very low threshold
                    "--ransac-dist=100",   # Very lenient for ultra-wide
                    "--kall",             # Keep all points
                    "--downscale=2",      # Use smaller images for faster processing
                    "-o", cp_project,
                    project_file
                ]
                
                try:
                    self._run_command(fallback_cmd, "cpfind-fallback", timeout=300)
                except Exception as fallback_error:
                    logger.warning(f"⚠️ Fallback cpfind also failed: {fallback_error}")
            
            # Final check for output file
            if not os.path.exists(cp_project):
                # Both strategies failed - create empty project file
                logger.warning("⚠️ All cpfind strategies failed - no control points found")
                shutil.copy2(project_file, cp_project)
                self.control_points_found = 0
            else:
                # Count control points found
                self.control_points_found = self._count_control_points(cp_project)
            
            step.complete(success=True, output_files=[cp_project])
            logger.info(f"✅ Found {self.control_points_found} control points")
            
            # Enhanced geocpset fallback - trigger earlier for ultra-wide challenges
            if self.control_points_found < 150:  # Increased threshold for ultra-wide reliability
                logger.warning(f"⚠️ Low control points ({self.control_points_found}) - attempting geocpset rescue")
                cp_project = self._fix_connectivity_with_geocpset(cp_project)
            
            return cp_project
            
        except Exception as e:
            step.complete(success=False, error=str(e))
            raise HuginPipelineError(f"Control point detection failed: {e}")
            
    def _count_control_points(self, project_file: str) -> int:
        """Count control points in PTO file."""
        try:
            with open(project_file, 'r') as f:
                content = f.read()
            cp_lines = [line for line in content.split('\n') if line.startswith('c ')]
            return len(cp_lines)
        except:
            return 0
            
    def _fix_connectivity_with_geocpset(self, project_file: str) -> str:
        """Use geocpset to connect isolated images based on ARKit positioning - from backup."""
        try:
            geocp_project = os.path.join(self.temp_dir, "project_geocp.pto")
            
            # geocpset generates control points based on image positions
            # This is perfect for ARKit data where we have accurate positioning
            cmd = ["geocpset", "-o", geocp_project, project_file]
            
            logger.info("🌍 Running geocpset to connect isolated images using ARKit positioning...")
            self._run_command(cmd, "geocpset", timeout=300)
            
            if os.path.exists(geocp_project):
                new_cp_count = self._count_control_points(geocp_project)
                old_cp_count = self._count_control_points(project_file)
                
                logger.info(f"📊 geocpset result: {old_cp_count} → {new_cp_count} control points")
                
                if new_cp_count > old_cp_count:
                    logger.info("✅ geocpset successfully improved connectivity!")
                    self.geocpset_used = True  # Mark that geocpset was used
                    self.control_points_found = new_cp_count
                    return geocp_project
                else:
                    logger.warning("⚠️ geocpset didn't improve connectivity")
                    return project_file
            else:
                logger.warning("⚠️ geocpset failed to create output file")
                return project_file
                
        except Exception as e:
            logger.warning(f"⚠️ geocpset failed: {e}")
            logger.info("📝 Continuing with original control points")
            return project_file
            
    def _step_3_clean_control_points(self, project_file: str, progress_callback: Optional[Callable]) -> str:
        """Step 3: Clean control points (skip if geocpset used)."""
        if self.geocpset_used:
            step = HuginStep("cpclean", "Skipped - preserving geocpset control points")
            step.start()
            step.complete(success=True, output_files=[project_file])
            self.pipeline_steps.append(step)
            
            if progress_callback:
                progress_callback(0.4, "Preserving geocpset control points...")
                
            logger.info("⚠️ Skipping cpclean - geocpset control points need preservation")
            return project_file
            
        step = HuginStep("cpclean", "Clean and validate control points")
        step.start()
        self.pipeline_steps.append(step)
        
        if progress_callback:
            progress_callback(0.4, "Cleaning control points...")
            
        clean_project = os.path.join(self.temp_dir, "project_clean.pto")
        
        try:
            cmd = ["cpclean", "-o", clean_project, project_file]
            self._run_command(cmd, "cpclean")
            
            step.complete(success=True, output_files=[clean_project])
            logger.info("✅ Control points cleaned")
            return clean_project
            
        except Exception as e:
            step.complete(success=False, error=str(e))
            raise HuginPipelineError(f"Control point cleaning failed: {e}")
            
    def _step_3_5_detect_lines(self, project_file: str, progress_callback: Optional[Callable]) -> str:
        """Step 3.5: Detect vertical/horizontal lines (skip for ARKit mode)."""
        
        if self.arkit_mode:
            step = HuginStep("linefind", "Skipped - ARKit provides superior geometric accuracy")
            step.start()
            step.complete(success=True, output_files=[project_file])
            self.pipeline_steps.append(step)
            
            if progress_callback:
                progress_callback(0.5, "Preserving ARKit geometric accuracy...")
                
            logger.info("⚠️ Skipping linefind - ARKit positioning provides superior geometric constraints")
            return project_file
        
        # Original linefind logic for non-ARKit mode
        step = HuginStep("linefind", "Detect horizon and vertical lines")
        step.start()
        self.pipeline_steps.append(step)
        
        if progress_callback:
            progress_callback(0.5, "Detecting lines...")
            
        line_project = os.path.join(self.temp_dir, "project_lines.pto")
        
        try:
            cmd = ["linefind", "-o", line_project, project_file]
            self._run_command(cmd, "linefind")
            
            step.complete(success=True, output_files=[line_project])
            logger.info("✅ Line detection completed")
            return line_project
            
        except Exception as e:
            step.complete(success=False, error=str(e))
            # Line detection failure is not critical
            logger.warning(f"⚠️ Line detection failed: {e}")
            return project_file  # Continue with previous project file
            
    def _step_4_optimize(self, project_file: str, progress_callback: Optional[Callable]) -> str:
        """Step 4: Optimize geometry and photometrics."""
        step = HuginStep("autooptimiser", "Optimize geometry and photometrics")
        step.start()
        self.pipeline_steps.append(step)
        
        if progress_callback:
            progress_callback(0.6, "Optimizing parameters...")
            
        opt_project = os.path.join(self.temp_dir, "project_opt.pto")
        
        try:
            if self.arkit_mode:
                # ARKit mode: Photometric + lens optimization for ultra-wide distortion - CRITICAL FIX
                cmd = [
                    "autooptimiser",
                    "-m",  # Photometric optimization
                    "-l",  # Lens optimization for ultra-wide distortion
                    "-o", opt_project,
                    project_file
                ]
                logger.info("📋 Using ARKit photometric + lens optimization for ultra-wide distortion")
            else:
                # Full optimization for non-ARKit data
                cmd = [
                    "autooptimiser",
                    "-a",  # Auto align mode
                    "-m",  # Photometric optimization
                    "-l",  # Lens optimization
                    "-s",  # Sky optimization
                    "-o", opt_project,
                    project_file
                ]
                logger.info("📋 Using full optimization")
                
            self._run_command(cmd, "autooptimiser", timeout=600)
            
            step.complete(success=True, output_files=[opt_project])
            logger.info("✅ Parameter optimization completed")
            return opt_project
            
        except Exception as e:
            step.complete(success=False, error=str(e))
            raise HuginPipelineError(f"Parameter optimization failed: {e}")
            
    def _step_5_set_output(self, project_file: str, progress_callback: Optional[Callable]) -> str:
        """Step 5: Set output parameters with optimal canvas size."""
        step = HuginStep("pano_modify", "Calculate optimal output parameters")
        step.start()
        self.pipeline_steps.append(step)
        
        if progress_callback:
            progress_callback(0.7, "Calculating optimal canvas size...")
            
        final_project = os.path.join(self.temp_dir, "project_final.pto")
        
        try:
            crop_mode = os.environ.get('PANORAMA_CROP_MODE', 'AUTO')
            
            # CRITICAL FIX: Use AUTO canvas sizing based on input image resolution
            # This prevents upscaling and preserves iPhone image quality
            cmd = [
                "pano_modify",
                "--canvas=AUTO",  # Let Hugin calculate optimal size from input images
                f"--crop={crop_mode}",
                "--projection=2",  # Equirectangular
                "--fov=360x180",
                "-o", final_project,
                project_file
            ]
            
            self._run_command(cmd, "pano_modify")
            
            # Read back the calculated optimal canvas size for logging
            try:
                with open(final_project, 'r') as f:
                    content = f.read()
                    # Extract canvas size from PTO file
                    import re
                    canvas_match = re.search(r'p f\d+ w(\d+) h(\d+)', content)
                    if canvas_match:
                        optimal_width = int(canvas_match.group(1))
                        optimal_height = int(canvas_match.group(2))
                        logger.info(f"📐 Optimal canvas calculated: {optimal_width}×{optimal_height} (preserves input resolution)")
                    else:
                        logger.info("📐 Optimal canvas calculated (size detection failed)")
            except Exception as size_error:
                logger.debug(f"Could not detect optimal canvas size: {size_error}")
            
            step.complete(success=True, output_files=[final_project])
            logger.info(f"✅ Output parameters set with optimal canvas sizing")
            return final_project
            
        except Exception as e:
            step.complete(success=False, error=str(e))
            raise HuginPipelineError(f"Output parameter setting failed: {e}")
            
    def _step_6_render_images(self, project_file: str, expected_count: int, progress_callback: Optional[Callable]) -> List[str]:
        """Step 6: Render images with nona."""
        step = HuginStep("nona", "Render images to equirectangular")
        step.start()
        self.pipeline_steps.append(step)
        
        if progress_callback:
            progress_callback(0.8, "Rendering images...")
            
        output_prefix = os.path.join(self.temp_dir, "rendered")
        
        try:
            # Validate image dimensions before rendering
            self._validate_image_dimensions(project_file)
            
            # Render images with proper TIFF format
            cmd = ["nona", "-m", "TIFF_m", "-o", output_prefix, project_file]
            self._run_command(cmd, "nona", timeout=600)
            
            # Find generated TIFF files
            tiff_files = list(Path(self.temp_dir).glob("rendered*.tif"))
            tiff_paths = [str(f) for f in tiff_files]
            
            # Debug: Check generated TIFF files
            logger.info(f"🔍 DEBUG: Found {len(tiff_files)} TIFF files after nona")
            for tiff_path in tiff_paths:
                try:
                    file_size = Path(tiff_path).stat().st_size
                    logger.info(f"📁 TIFF: {os.path.basename(tiff_path)} ({file_size/1024/1024:.1f}MB)")
                    
                    # Check if file has actual image data (not empty/black)
                    import cv2
                    img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        height, width = img.shape[:2]
                        mean_val = img.mean() if len(img.shape) == 2 else img.mean()
                        logger.info(f"📊 TIFF: {width}×{height}, mean value: {mean_val:.3f}")
                        if mean_val < 0.01:
                            logger.warning(f"⚠️ TIFF appears very dark/black: {os.path.basename(tiff_path)}")
                    else:
                        logger.error(f"❌ Could not read TIFF: {os.path.basename(tiff_path)}")
                except Exception as debug_error:
                    logger.warning(f"⚠️ Debug failed for {os.path.basename(tiff_path)}: {debug_error}")
            
            if not tiff_paths:
                raise HuginPipelineError("nona failed to generate TIFF files")
                
            step.complete(success=True, output_files=tiff_paths)
            logger.info(f"✅ Rendered {len(tiff_paths)} images")
            return tiff_paths
            
        except Exception as e:
            step.complete(success=False, error=str(e))
            raise HuginPipelineError(f"Image rendering failed: {e}")
            
    def _step_7_return_tiff_files(self, tiff_files: List[str], expected_count: int, progress_callback: Optional[Callable]) -> List[str]:
        """Step 7: Return TIFF files for external blending (microservices architecture)."""
        step = HuginStep("tiff_export", "Exporting rendered TIFFs for blending service")
        step.start()
        self.pipeline_steps.append(step)
        
        if progress_callback:
            progress_callback(0.9, "Preparing for blending...")
            
        try:
            # Validate TIFF files exist and are non-empty
            valid_tiffs = []
            for tiff_path in tiff_files:
                if os.path.exists(tiff_path) and os.path.getsize(tiff_path) > 0:
                    valid_tiffs.append(tiff_path)
                else:
                    logger.warning(f"⚠️ Invalid TIFF file: {tiff_path}")
                    
            if not valid_tiffs:
                raise HuginPipelineError("No valid TIFF files for blending")
                
            step.complete(success=True, output_files=valid_tiffs)
            logger.info(f"✅ {len(valid_tiffs)} TIFF files ready for blending service")
            return valid_tiffs
            
        except Exception as e:
            step.complete(success=False, error=str(e))
            raise HuginPipelineError(f"TIFF export failed: {e}")
            
    def _validate_image_dimensions(self, project_file: str):
        """Validate that all images in project have consistent dimensions with PTO file."""
        try:
            import re
            
            with open(project_file, 'r') as f:
                content = f.read()
            
            # Extract image lines from PTO file
            image_lines = [line for line in content.split('\n') if line.startswith('i ')]
            
            for i, line in enumerate(image_lines):
                # Extract width and height from PTO line (w and h parameters)
                w_match = re.search(r'w(\d+)', line)
                h_match = re.search(r'h(\d+)', line)
                
                if not w_match or not h_match:
                    logger.warning(f"⚠️ Could not extract dimensions from PTO line {i}")
                    continue
                    
                pto_width = int(w_match.group(1))
                pto_height = int(h_match.group(1))
                
                # Extract filename (n parameter)
                n_match = re.search(r'n"([^"]+)"', line)
                if not n_match:
                    logger.warning(f"⚠️ Could not extract filename from PTO line {i}")
                    continue
                    
                image_path = n_match.group(1)
                
                # Check if file exists and get actual dimensions
                if not os.path.exists(image_path):
                    logger.warning(f"⚠️ Image file not found: {image_path}")
                    continue
                
                try:
                    # Use PIL to get actual image dimensions
                    from PIL import Image
                    with Image.open(image_path) as img:
                        actual_width, actual_height = img.size
                        
                    if actual_width != pto_width or actual_height != pto_height:
                        logger.warning(f"⚠️ Dimension mismatch for {os.path.basename(image_path)}")
                        logger.warning(f"   PTO: {pto_width}×{pto_height}, Actual: {actual_width}×{actual_height}")
                        
                        # Update PTO file with correct dimensions
                        old_w = f"w{pto_width}"
                        old_h = f"h{pto_height}"
                        new_w = f"w{actual_width}"  
                        new_h = f"h{actual_height}"
                        
                        content = content.replace(f"{old_w} {old_h}", f"{new_w} {new_h}")
                        logger.info(f"✅ Updated PTO dimensions for {os.path.basename(image_path)}: {actual_width}×{actual_height}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not validate dimensions for {image_path}: {e}")
                    
            # Write back updated PTO file
            with open(project_file, 'w') as f:
                f.write(content)
                
            logger.info("✅ Image dimension validation completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Image dimension validation failed: {e}")

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"🧹 Cleaned up temporary directory")
            
    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report."""
        return {
            'service_info': {
                'name': 'HuginPipelineService',
                'version': '1.0.0',
                'purpose': 'Execute complete 7-step Hugin panorama pipeline'
            },
            'configuration': {
                'temp_dir': self.temp_dir,
                'canvas_size': self.canvas_size,
                'arkit_mode': self.arkit_mode,
                'geocpset_used': self.geocpset_used
            },
            'statistics': {
                'control_points_found': self.control_points_found,
                'pipeline_steps_completed': len([s for s in self.pipeline_steps if s.success]),
                'total_steps': len(self.pipeline_steps)
            },
            'pipeline_steps': [
                {
                    'name': step.name,
                    'description': step.description,
                    'success': step.success,
                    'duration': step.end_time - step.start_time if step.start_time and step.end_time else 0,
                    'error': step.error,
                    'output_files_count': len(step.output_files)
                } for step in self.pipeline_steps
            ]
        }

def create_hugin_service(temp_dir: Optional[str] = None, canvas_size: Tuple[int, int] = (8192, 4096)) -> HuginPipelineService:
    """Factory function to create Hugin pipeline service."""
    return HuginPipelineService(temp_dir=temp_dir, canvas_size=canvas_size)
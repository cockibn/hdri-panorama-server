#!/usr/bin/env python3
"""
Correct Hugin Panorama Stitcher - Based on 2024 Documentation
Research-based implementation following official Hugin workflow and best practices.
"""

import os
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import time
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class CorrectHuginStitcher:
    """Hugin panorama stitcher following official 2024 documentation and best practices."""
    
    def __init__(self, output_resolution: str = "6K"):
        self.temp_dir = None
        self._verify_hugin_installation()
        
        # Standard resolution options
        self.resolutions = {
            "4K": (4096, 2048),
            "6K": (6144, 3072),
            "8K": (8192, 4096)
        }
        
        self.canvas_size = self.resolutions.get(output_resolution, self.resolutions["6K"])
        logger.info(f"🎨 Hugin stitcher initialized: {self.canvas_size[0]}×{self.canvas_size[1]}")
    
    def _verify_hugin_installation(self):
        """Verify required Hugin tools are available."""
        required_tools = ['pto_gen', 'cpfind', 'cpclean', 'autooptimiser', 'pano_modify', 'nona', 'enblend']
        
        for tool in required_tools:
            try:
                subprocess.run([tool, '--help'], capture_output=True, timeout=5)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                raise RuntimeError(f"Hugin tool '{tool}' not found. Please install complete Hugin package.")
        
        logger.info("✅ Hugin installation verified")
    
    def stitch_panorama(self, images: List[np.ndarray], capture_points: List[Dict], 
                       progress_callback: Optional[Callable] = None,
                       exif_data: List[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Official Hugin panorama workflow based on 2024 documentation.
        """
        start_time = time.time()
        
        try:
            with tempfile.TemporaryDirectory(prefix="hugin_correct_") as temp_dir:
                self.temp_dir = temp_dir
                logger.info(f"🚀 Starting official Hugin workflow in {temp_dir}")
                
                if progress_callback:
                    progress_callback(0.05, "Preparing images...")
                
                # Step 1: Save images and generate project file
                image_paths = self._save_images(images)
                if progress_callback:
                    progress_callback(0.15, "Generating project file...")
                
                project_file = self._generate_project_file(image_paths, capture_points)
                
                # Step 2: Find control points (2024 multirow default)
                if progress_callback:
                    progress_callback(0.30, "Finding control points with multirow strategy...")
                
                cp_project = self._find_control_points(project_file)
                self.control_points_found = self._count_control_points(cp_project)
                
                # Step 3: Clean control points
                if progress_callback:
                    progress_callback(0.45, "Cleaning control points...")
                
                clean_project = self._clean_control_points(cp_project)
                
                # Step 4: Optimize geometry and photometrics
                if progress_callback:
                    progress_callback(0.60, "Optimizing geometry and photometrics...")
                
                opt_project = self._optimize_panorama(clean_project)
                
                # Step 5: Set output parameters (canvas, crop, projection)
                if progress_callback:
                    progress_callback(0.70, "Setting output parameters...")
                
                final_project = self._set_output_parameters(opt_project)
                
                # Step 6: Render images with nona
                if progress_callback:
                    progress_callback(0.80, "Rendering images...")
                
                tiff_files = self._render_images(final_project)
                
                # Step 7: Blend with enblend
                if progress_callback:
                    progress_callback(0.95, "Blending final panorama...")
                
                panorama_path = self._blend_images(tiff_files)
                
                # Load and return result
                logger.info(f"📁 Loading final panorama from: {panorama_path}")
                
                # Check if file exists and get info
                import os
                if os.path.exists(panorama_path):
                    file_size = os.path.getsize(panorama_path)
                    logger.info(f"📊 Final panorama file: {file_size} bytes")
                else:
                    logger.error(f"❌ Final panorama file not found: {panorama_path}")
                    raise RuntimeError(f"Final panorama file not found: {panorama_path}")
                
                panorama = cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED)
                if panorama is None:
                    # Try different loading methods
                    logger.warning("⚠️ Standard CV2 loading failed, trying alternatives...")
                    
                    # Try loading as TIFF with PIL
                    try:
                        from PIL import Image
                        import numpy as np
                        pil_image = Image.open(panorama_path)
                        panorama = np.array(pil_image)
                        if len(panorama.shape) == 3:
                            panorama = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
                        logger.info(f"✅ Loaded with PIL: {panorama.shape}")
                    except Exception as e:
                        logger.error(f"❌ PIL loading also failed: {e}")
                        raise RuntimeError(f"Failed to load final panorama with both CV2 and PIL: {e}")
                
                processing_time = time.time() - start_time
                quality_metrics = self._calculate_quality_metrics(panorama, len(images), processing_time)
                
                # Add control point information to metrics
                quality_metrics['controlPoints'] = getattr(self, 'control_points_found', 0)
                
                logger.info(f"🎉 Official Hugin workflow completed in {processing_time:.1f}s")
                
                return panorama, quality_metrics
                
        except Exception as e:
            logger.error(f"❌ Hugin workflow failed: {e}")
            raise
    
    def _save_images(self, images: List[np.ndarray]) -> List[str]:
        """Save images to temporary directory."""
        image_paths = []
        
        for i, img in enumerate(images):
            path = os.path.join(self.temp_dir, f"img_{i:04d}.jpg")
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_paths.append(path)
        
        logger.info(f"📁 Saved {len(image_paths)} images")
        return image_paths
    
    def _generate_project_file(self, image_paths: List[str], capture_points: List[Dict] = None) -> str:
        """Step 1: Generate project file with ARKit positioning data."""
        project_file = os.path.join(self.temp_dir, "project.pto")
        
        if capture_points and len(capture_points) == len(image_paths):
            logger.info(f"🎯 Using ARKit positioning data for {len(capture_points)} images")
            # Debug: Log first few capture points to understand data structure
            for i, cp in enumerate(capture_points[:3]):
                logger.info(f"🔍 Capture point {i}: {cp}")
            
            # Check if we have valid positioning data
            has_positioning = any(
                cp.get('azimuth') is not None and cp.get('elevation') is not None 
                for cp in capture_points
            )
            
            if has_positioning:
                self._generate_positioned_project(image_paths, capture_points, project_file)
            else:
                logger.warning(f"⚠️ No valid azimuth/elevation data - falling back to basic pto_gen")
                cmd = ["pto_gen", "-o", project_file] + image_paths
                self._run_command(cmd, "pto_gen")
        else:
            logger.warning(f"⚠️ No positioning data ({len(capture_points) if capture_points else 0} points vs {len(image_paths)} images) - falling back to basic pto_gen")
            # Fallback to basic pto_gen
            cmd = ["pto_gen", "-o", project_file] + image_paths
            self._run_command(cmd, "pto_gen")
        
        logger.info(f"✅ Generated project file with {len(image_paths)} images")
        return project_file
    
    def _generate_positioned_project(self, image_paths: List[str], capture_points: List[Dict], project_file: str):
        """Generate PTO file with ARKit positioning data."""
        logger.info(f"🎯 Generating positioned project with {len(capture_points)} ARKit positions")
        
        # iPhone ultra-wide camera parameters (106.2° FOV measured)
        fov = 106.2
        
        with open(project_file, 'w') as f:
            # Write PTO header
            f.write("# hugin project file\n")
            f.write(f"p f2 w{self.canvas_size[0]} h{self.canvas_size[1]} v360 n\"TIFF_m c:LZW\"\n")
            f.write("m g1 i0 f0 m2 p0.00784314\n")
            
            # Write image lines with ARKit positioning
            for i, (img_path, capture_point) in enumerate(zip(image_paths, capture_points)):
                # Convert ARKit coordinates to Hugin angles
                azimuth = capture_point.get('azimuth', 0.0)  # ARKit azimuth
                elevation = capture_point.get('elevation', 0.0)  # ARKit elevation  
                position = capture_point.get('position', [0.0, 0.0, 0.0])  # 3D position [x,y,z]
                
                # For now, use zero roll (could be derived from position if needed)
                roll = 0.0
                
                # Convert to Hugin coordinate system
                # ARKit: azimuth (0-360°), elevation (-90 to +90°)
                # Hugin: yaw (-180 to +180°), pitch (-90 to +90°), roll (-180 to +180°)
                yaw = azimuth if azimuth <= 180 else azimuth - 360
                pitch = elevation
                roll_hugin = roll if roll <= 180 else roll - 360
                
                # Write image line with positioning
                f.write(f'i w4032 h3024 f0 v{fov} Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r{roll_hugin:.6f} p{pitch:.6f} y{yaw:.6f} TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5 n"{img_path}"\n')
                
                logger.info(f"📍 Image {i}: azimuth={azimuth:.1f}°, elevation={elevation:.1f}°, roll={roll:.1f}° → yaw={yaw:.1f}°, pitch={pitch:.1f}°")
        
        logger.info(f"✅ Generated positioned PTO with ARKit data covering {len(capture_points)} viewpoints")
    
    def _find_control_points(self, project_file: str) -> str:
        """Step 2: Find control points using cpfind with 2024 multirow default."""
        cp_project = os.path.join(self.temp_dir, "project_cp.pto")
        
        # Official 2024 cpfind command (multirow is default)
        cmd = [
            "cpfind",
            "--multirow",      # 2024 default strategy
            "--celeste",       # Sky detection
            "-o", cp_project,
            project_file
        ]
        
        self._run_command(cmd, "cpfind")
        
        # Verify output
        if not os.path.exists(cp_project):
            raise RuntimeError("cpfind failed to create output file")
        
        cp_count = self._count_control_points(cp_project)
        logger.info(f"🎯 Found {cp_count} control points using multirow strategy")
        
        return cp_project
    
    def _clean_control_points(self, project_file: str) -> str:
        """Step 3: Clean control points using cpclean."""
        clean_project = os.path.join(self.temp_dir, "project_clean.pto")
        
        # Official cpclean command
        cmd = ["cpclean", "-o", clean_project, project_file]
        self._run_command(cmd, "cpclean")
        
        logger.info("✅ Control points cleaned")
        return clean_project
    
    def _optimize_panorama(self, project_file: str) -> str:
        """Step 4: Optimize using autooptimiser."""
        opt_project = os.path.join(self.temp_dir, "project_opt.pto")
        
        # Official autooptimiser command
        cmd = [
            "autooptimiser",
            "-a",  # Optimize positions (yaw, pitch, roll)
            "-m",  # Optimize photometric parameters
            "-l",  # Optimize lens parameters
            "-s",  # Optimize exposure
            "-o", opt_project,
            project_file
        ]
        
        self._run_command(cmd, "autooptimiser")
        
        logger.info("✅ Panorama optimization completed")
        return opt_project
    
    def _set_output_parameters(self, project_file: str) -> str:
        """Step 5: Set output parameters using pano_modify."""
        final_project = os.path.join(self.temp_dir, "project_final.pto")
        
        # Check for crop mode preference (AUTO removes black areas, NONE keeps full canvas)
        crop_mode = os.environ.get('PANORAMA_CROP_MODE', 'AUTO')
        
        if crop_mode.upper() == 'NONE':
            crop_param = "--crop=NONE"
            logger.info(f"📐 Using full canvas mode: {self.canvas_size[0]}×{self.canvas_size[1]} (no cropping)")
        else:
            crop_param = "--crop=AUTO"
            logger.info(f"📐 Using auto-crop mode: will crop to content area from {self.canvas_size[0]}×{self.canvas_size[1]} canvas")
        
        # Official pano_modify command
        cmd = [
            "pano_modify",
            f"--canvas={self.canvas_size[0]}x{self.canvas_size[1]}",  # Set canvas size
            crop_param,                                                # Crop mode
            "--projection=0",                                          # Equirectangular
            "-o", final_project,
            project_file
        ]
        
        self._run_command(cmd, "pano_modify")
        
        # Log the actual output parameters by reading the final project file
        self._log_final_output_params(final_project)
        
        return final_project
    
    def _render_images(self, project_file: str) -> List[str]:
        """Step 6: Render images using nona."""
        output_prefix = os.path.join(self.temp_dir, "rendered")
        
        # Use uncompressed TIFF for better enblend compatibility
        cmd = ["nona", "-m", "TIFF", "-o", output_prefix, project_file]
        self._run_command(cmd, "nona")
        
        # Find generated TIFF files
        tiff_files = sorted(Path(self.temp_dir).glob("rendered*.tif"))
        tiff_paths = [str(f) for f in tiff_files]
        
        if not tiff_paths:
            raise RuntimeError("nona failed to generate TIFF files")
        
        logger.info(f"🗺️ Rendered {len(tiff_paths)} images")
        return tiff_paths
    
    def _blend_images(self, tiff_files: List[str]) -> str:
        """Step 7: Blend images using enblend, then convert to EXR."""
        tiff_output = os.path.join(self.temp_dir, "final_panorama.tif")
        exr_output = os.path.join(self.temp_dir, "final_panorama.exr")
        
        logger.info(f"🎨 Blending {len(tiff_files)} images with optimized enblend...")
        
        # Fast enblend for large panoramas
        cmd = [
            "enblend", 
            "-o", tiff_output,
            "--levels=10",      # Much fewer levels for speed
        ] + tiff_files
        
        try:
            self._run_command(cmd, "enblend", timeout=600)  # 10 minute timeout for large panoramas
        except RuntimeError as e:
            if "timed out" in str(e):
                # If timeout, try even faster settings
                logger.warning("⚠️ Fast enblend timed out, trying ultra-fast version...")
                ultrafast_cmd = [
                    "enblend", 
                    "-o", tiff_output,
                    "--levels=5"  # Minimal levels
                ] + tiff_files
                
                try:
                    self._run_command(ultrafast_cmd, "enblend", timeout=900)  # 15 minutes
                except RuntimeError:
                    # Last resort: basic enblend with long timeout
                    logger.warning("⚠️ Ultra-fast enblend failed, trying basic version...")
                    basic_cmd = ["enblend", "-o", tiff_output] + tiff_files
                    self._run_command(basic_cmd, "enblend", timeout=1200)  # 20 minutes
            else:
                # Non-timeout error, try basic version
                logger.warning("⚠️ Fast enblend failed with error, trying basic version...")
                basic_cmd = ["enblend", "-o", tiff_output] + tiff_files
                self._run_command(basic_cmd, "enblend", timeout=900)
        
        if not os.path.exists(tiff_output):
            raise RuntimeError("enblend failed to create final panorama")
        
        tiff_size = os.path.getsize(tiff_output)
        logger.info(f"🎨 Images blended successfully - TIFF: {tiff_size} bytes")
        
        # Convert TIFF to EXR for HDR output
        logger.info("🌟 Converting to EXR for HDR output...")
        tiff_image = cv2.imread(tiff_output, cv2.IMREAD_UNCHANGED)
        if tiff_image is None:
            raise RuntimeError("Failed to load blended TIFF for EXR conversion")
        
        # Convert to float32 for EXR
        if tiff_image.dtype != np.float32:
            if tiff_image.dtype == np.uint8:
                tiff_image = tiff_image.astype(np.float32) / 255.0
            elif tiff_image.dtype == np.uint16:
                tiff_image = tiff_image.astype(np.float32) / 65535.0
            else:
                tiff_image = tiff_image.astype(np.float32)
        
        # Save as EXR with HDR format
        cv2.imwrite(exr_output, tiff_image, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
        
        if not os.path.exists(exr_output):
            raise RuntimeError("Failed to create EXR output")
        
        exr_size = os.path.getsize(exr_output)
        logger.info(f"🌟 EXR conversion complete - output: {exr_size} bytes")
        
        return exr_output
    
    def _run_command(self, cmd: List[str], tool_name: str, timeout: int = 300):
        """Run Hugin command with error handling."""
        logger.debug(f"🔧 Running {tool_name}: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
                cwd=self.temp_dir
            )
            return result.stdout, result.stderr
            
        except subprocess.CalledProcessError as e:
            error_msg = f"{tool_name} failed (return code {e.returncode})"
            if e.stderr:
                error_msg += f": {e.stderr[:200]}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{tool_name} timed out after {timeout}s")
    
    def _count_control_points(self, project_file: str) -> int:
        """Count control points in PTO file."""
        try:
            with open(project_file, 'r') as f:
                content = f.read()
            return content.count('\nc ')
        except:
            return 0
    
    def _log_final_output_params(self, project_file: str):
        """Log the actual output parameters from the final project file."""
        try:
            with open(project_file, 'r') as f:
                content = f.read()
            
            # Look for panorama line (starts with 'p')
            for line in content.split('\n'):
                if line.startswith('p '):
                    # Extract parameters from panorama line
                    parts = line.split()
                    width = height = None
                    crop_info = None
                    
                    for part in parts:
                        if part.startswith('w'):
                            width = part[1:]
                        elif part.startswith('h'):
                            height = part[1:]
                        elif part.startswith('S'):
                            crop_info = part[1:]  # Crop bounds
                    
                    if width and height:
                        logger.info(f"📊 Final output will be: {width}×{height}")
                        if crop_info:
                            logger.info(f"📐 Crop bounds: {crop_info}")
                        break
            
        except Exception as e:
            logger.warning(f"⚠️ Could not read final project parameters: {e}")
    
    def _calculate_quality_metrics(self, panorama: np.ndarray, input_count: int,
                                 processing_time: float) -> Dict:
        """Calculate quality metrics."""
        height, width = panorama.shape[:2]
        
        metrics = {
            'resolution': f"{width}×{height}",
            'aspectRatio': round(width / height, 2),
            'inputImages': input_count,
            'processingTime': round(processing_time, 1),
            'processor': 'Hugin (Official 2024 Workflow)',
            'pipeline': 'pto_gen → cpfind → cpclean → autooptimiser → pano_modify → nona → enblend'
        }
        
        # Quality analysis - convert to uint8 for OpenCV compatibility
        if len(panorama.shape) == 3:
            # Convert float32 EXR to uint8 for analysis
            if panorama.dtype == np.float32:
                display_img = (np.clip(panorama, 0, 1) * 255).astype(np.uint8)
            else:
                display_img = panorama
            gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        else:
            if panorama.dtype == np.float32:
                gray = (np.clip(panorama, 0, 1) * 255).astype(np.uint8)
            else:
                gray = panorama
        
        # Sharpness analysis on uint8 version
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = round(laplacian_var, 2)
        
        # Contrast
        metrics['contrast'] = round(float(np.std(gray)), 2)
        
        # Coverage
        non_black = np.sum(gray > 10)
        total_pixels = gray.size
        metrics['coverage'] = round((non_black / total_pixels) * 100, 1)
        
        # Control point quality assessment
        theoretical_max_pairs = input_count * (input_count - 1) // 2
        cp_efficiency = metrics.get('controlPoints', 0) / theoretical_max_pairs if theoretical_max_pairs > 0 else 0
        metrics['controlPointEfficiency'] = round(cp_efficiency * 100, 1)
        
        # Overall score including control point quality
        quality_score = (
            min(laplacian_var / 500, 1.0) * 0.25 +
            min(metrics['contrast'] / 50, 1.0) * 0.2 +
            (metrics['coverage'] / 100) * 0.25 +
            min(input_count / 16, 1.0) * 0.15 +
            cp_efficiency * 0.15
        )
        
        metrics['overallScore'] = round(quality_score, 3)
        
        # Add analysis comments
        if metrics.get('controlPoints', 0) > 0:
            if cp_efficiency > 0.8:
                metrics['controlPointAnalysis'] = "Excellent feature matching"
            elif cp_efficiency > 0.6:
                metrics['controlPointAnalysis'] = "Good feature matching"
            elif cp_efficiency > 0.4:
                metrics['controlPointAnalysis'] = "Adequate feature matching"
            else:
                metrics['controlPointAnalysis'] = "Limited feature matching"
        
        return metrics

# Compatibility alias
EfficientHuginStitcher = CorrectHuginStitcher
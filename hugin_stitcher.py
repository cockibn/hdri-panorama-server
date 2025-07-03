#!/usr/bin/env python3
"""
Hugin-based Panorama Stitching Module

This module implements professional-grade panorama stitching using Hugin command-line tools,
specifically optimized for the 16-point ultra-wide capture pattern used by HDRi 360 Studio.

Key Features:
- Hugin command-line tool integration (cpfind, autooptimiser, nona, etc.)
- Professional quality control point detection
- Advanced bundle adjustment optimization
- Equirectangular projection output
- Quality metrics calculation
- Robust error handling and progress tracking
"""

import os
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import json
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class HuginPanoramaStitcher:
    """Professional panorama stitcher using Hugin command-line tools"""
    
    def __init__(self, hugin_path: str = None):
        """
        Initialize the Hugin stitcher
        
        Args:
            hugin_path: Path to Hugin installation directory (auto-detected if None)
        """
        self.hugin_path = hugin_path or self._find_hugin_path()
        self.temp_dir = None
        
        # Verify Hugin installation
        self._verify_hugin_installation()
        
        # Processing parameters
        self.canvas_size = (4096, 2048)  # 4K equirectangular
        self.jpeg_quality = 95
        
        # iPhone 15 Pro Ultra-Wide Camera Specifications
        self.iphone_15_pro_ultrawide = {
            'focal_length_mm': 13.0,  # 13mm equivalent
            'sensor_width_mm': 5.7,   # 1/2.55" sensor physical width
            'sensor_height_mm': 4.28,  # 1/2.55" sensor physical height
            'fov_diagonal': 120.0,     # 120° diagonal field of view
            'fov_horizontal': 103.0,   # Approximate horizontal FOV
            'fov_vertical': 77.0,      # Approximate vertical FOV
            'image_width': 4032,       # Native resolution width
            'image_height': 3024,      # Native resolution height
            # Lens distortion parameters (estimated for ultra-wide)
            'distortion_k1': -0.28,   # Primary radial distortion
            'distortion_k2': 0.15,    # Secondary radial distortion
            'distortion_k3': -0.05,   # Tertiary radial distortion
            'distortion_p1': 0.001,   # Tangential distortion
            'distortion_p2': 0.001,   # Tangential distortion
        }
        
    def _find_hugin_path(self) -> str:
        """Auto-detect Hugin installation path"""
        possible_paths = [
            "/usr/bin",
            "/usr/local/bin", 
            "/opt/hugin/bin",
            "C:\\Program Files\\Hugin\\bin",
            "C:\\Program Files (x86)\\Hugin\\bin"
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "pto_gen")):
                return path
        
        # Try system PATH
        try:
            subprocess.run(["pto_gen", "--help"], capture_output=True, check=True)
            return ""  # Available in PATH
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        raise RuntimeError("Hugin not found. Please install Hugin or provide hugin_path")
    
    def _verify_hugin_installation(self):
        """Verify required Hugin tools are available"""
        required_tools = [
            "pto_gen", "cpfind", "autooptimiser", "pano_modify", 
            "nona", "enblend", "cpclean", "linefind"
        ]
        
        missing_tools = []
        for tool in required_tools:
            tool_path = os.path.join(self.hugin_path, tool) if self.hugin_path else tool
            try:
                subprocess.run([tool_path, "--help"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(tool)
        
        if missing_tools:
            raise RuntimeError(f"Missing Hugin tools: {', '.join(missing_tools)}")
        
        logger.info(f"Hugin installation verified at: {self.hugin_path or 'system PATH'}")
    
    def _run_hugin_command(self, command: List[str], timeout: int = 300) -> Tuple[str, str]:
        """
        Run a Hugin command and return output
        
        Args:
            command: Command and arguments to run
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (stdout, stderr)
        """
        if self.hugin_path:
            command[0] = os.path.join(self.hugin_path, command[0])
        
        logger.debug(f"Running command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Command timed out after {timeout} seconds: {' '.join(command)}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {' '.join(command)}\nError: {e.stderr}")
    
    def stitch_panorama(self, images: List[np.ndarray], capture_points: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Main stitching pipeline using Hugin
        
        Args:
            images: List of input images (BGR format)
            capture_points: List of capture point metadata with azimuth/elevation
            
        Returns:
            Tuple of (stitched_panorama, quality_metrics)
        """
        start_time = time.time()
        
        if len(images) < 4:
            raise ValueError("Need at least 4 images for panorama stitching")
        
        logger.info(f"Starting Hugin panorama stitching with {len(images)} images")
        
        # Create temporary working directory
        self.temp_dir = tempfile.mkdtemp(prefix="hugin_stitch_")
        
        try:
            # Step 1: Save images to temporary directory
            image_paths = self._save_images_to_temp(images)
            
            # Step 2: Create PTO project file
            project_file = self._create_pto_project(image_paths, capture_points)
            
            # Step 3: Find control points
            project_file = self._find_control_points(project_file)
            
            # Check if we have any control points at all
            with open(project_file, 'r') as f:
                content = f.read()
                if 'c n' not in content:
                    logger.warning("No control points found after all attempts")
                    raise RuntimeError("Insufficient control points for Hugin stitching")
            
            # Step 4: Clean control points
            project_file = self._clean_control_points(project_file)
            
            # Step 5: Find vertical lines (if any)
            project_file = self._find_vertical_lines(project_file)
            
            # Step 6: Optimize panorama
            project_file = self._optimize_panorama(project_file)
            
            # Step 7: Set output parameters
            project_file = self._set_output_parameters(project_file)
            
            # Step 8: Stitch panorama
            panorama_path = self._stitch_with_nona(project_file)
            
            # Step 9: Load result and calculate metrics
            final_panorama = cv2.imread(panorama_path)
            if final_panorama is None:
                raise RuntimeError(f"Failed to load stitched panorama: {panorama_path}")
            
            processing_time = time.time() - start_time
            quality_metrics = self._calculate_quality_metrics(
                final_panorama, project_file, processing_time
            )
            
            logger.info(f"Hugin panorama stitching completed in {processing_time:.2f}s")
            
            return final_panorama, quality_metrics
            
        finally:
            # Cleanup temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _save_images_to_temp(self, images: List[np.ndarray]) -> List[str]:
        """Save images to temporary directory"""
        image_paths = []
        
        for i, img in enumerate(images):
            image_path = os.path.join(self.temp_dir, f"image_{i:04d}.jpg")
            cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            image_paths.append(image_path)
        
        logger.info(f"Saved {len(image_paths)} images to temporary directory")
        return image_paths
    
    def _create_pto_project(self, image_paths: List[str], capture_points: List[Dict]) -> str:
        """Create initial PTO project file with iPhone 15 Pro ultra-wide optimization"""
        project_file = os.path.join(self.temp_dir, "project.pto")
        
        # Use pto_gen to create basic project
        command = ["pto_gen", "-o", project_file] + image_paths
        self._run_hugin_command(command)
        
        # Apply iPhone 15 Pro ultra-wide specific optimizations
        self._optimize_pto_for_iphone_ultrawide(project_file, image_paths)
        
        # Set initial camera positions based on capture points
        if capture_points and len(capture_points) == len(image_paths):
            self._set_initial_positions(project_file, capture_points)
        
        logger.info(f"Created optimized PTO project for iPhone 15 Pro ultra-wide: {project_file}")
        return project_file
    
    def _optimize_pto_for_iphone_ultrawide(self, project_file: str, image_paths: List[str]):
        """Optimize PTO file specifically for iPhone 15 Pro ultra-wide camera"""
        
        # Read current PTO file
        with open(project_file, 'r') as f:
            pto_content = f.read()
        
        lines = pto_content.split('\n')
        
        # Calculate focal length in pixels
        image_width = self.iphone_15_pro_ultrawide['image_width']
        image_height = self.iphone_15_pro_ultrawide['image_height']
        fov_horizontal = self.iphone_15_pro_ultrawide['fov_horizontal']
        
        # Calculate focal length in pixels: f = (width/2) / tan(fov/2)
        import math
        focal_length_pixels = (image_width / 2) / math.tan(math.radians(fov_horizontal / 2))
        
        # Hugin's HFOV parameter
        hfov = fov_horizontal
        
        modified_lines = []
        
        for line in lines:
            if line.startswith('i '):
                # Modify image parameters for ultra-wide
                # Add ultra-wide specific parameters
                parts = line.split()
                
                # Build new image line with ultra-wide optimizations
                new_parts = ['i']
                
                # Add focal length and FOV
                new_parts.append(f'f{focal_length_pixels:.1f}')
                new_parts.append(f'v{hfov:.1f}')
                
                # Add lens distortion correction for ultra-wide
                # a, b, c parameters for lens distortion
                new_parts.append(f'a{self.iphone_15_pro_ultrawide["distortion_k1"]:.6f}')
                new_parts.append(f'b{self.iphone_15_pro_ultrawide["distortion_k2"]:.6f}')
                new_parts.append(f'c{self.iphone_15_pro_ultrawide["distortion_k3"]:.6f}')
                
                # Add tangential distortion
                new_parts.append(f'd{self.iphone_15_pro_ultrawide["distortion_p1"]:.6f}')
                new_parts.append(f'e{self.iphone_15_pro_ultrawide["distortion_p2"]:.6f}')
                
                # Set projection type for ultra-wide (rectilinear)
                new_parts.append('f0')  # Rectilinear projection
                
                # Keep existing parameters that don't conflict
                for part in parts[1:]:
                    param_type = part[0] if part else ''
                    if param_type not in ['f', 'v', 'a', 'b', 'c', 'd', 'e']:
                        new_parts.append(part)
                
                modified_lines.append(' '.join(new_parts))
                
            elif line.startswith('p '):
                # Modify project parameters for ultra-wide panorama
                # Set optimal output parameters for ultra-wide input
                modified_lines.append(f'p f2 w{self.canvas_size[0]} h{self.canvas_size[1]} v360 u10 n"JPEG q95"')
                
            else:
                modified_lines.append(line)
        
        # Write modified PTO file
        with open(project_file, 'w') as f:
            f.write('\n'.join(modified_lines))
        
        logger.info("Applied iPhone 15 Pro ultra-wide optimizations to PTO file")
    
    def _set_initial_positions(self, project_file: str, capture_points: List[Dict]):
        """Set initial camera positions based on capture metadata"""
        # Read current PTO file
        with open(project_file, 'r') as f:
            pto_content = f.read()
        
        # Parse and modify image lines with yaw/pitch from capture points
        lines = pto_content.split('\n')
        image_index = 0
        
        for i, line in enumerate(lines):
            if line.startswith('i '):
                if image_index < len(capture_points):
                    point = capture_points[image_index]
                    azimuth = point.get('azimuth', 0)
                    elevation = point.get('elevation', 0)
                    
                    # Convert to Hugin coordinates
                    yaw = azimuth
                    pitch = elevation
                    
                    # Add yaw and pitch to image line
                    if 'y' not in line:
                        line += f' y{yaw}'
                    if 'p' not in line:
                        line += f' p{pitch}'
                    
                    lines[i] = line
                    image_index += 1
        
        # Write modified PTO file
        with open(project_file, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Set initial positions for {image_index} images")
    
    def _find_control_points(self, project_file: str) -> str:
        """Find control points using cpfind optimized for iPhone 15 Pro ultra-wide"""
        output_file = os.path.join(self.temp_dir, "project_cp.pto")
        
        # Ultra-wide optimized cpfind parameters
        command = [
            "cpfind",
            "-o", output_file,
            "--multirow",                    # Enable multirow detection
            "--celeste",                     # Use celeste for sky detection  
            "--sift",                        # Use SIFT detector
            "--ransac-mode", "auto",         # Automatic RANSAC
            "--ransac-threshold", "15",      # Tighter threshold for ultra-wide precision
            "--min-matches", "6",            # Higher minimum for ultra-wide reliability
            "--sift-features", "3000",       # More features for ultra-wide content
            "--kdtree-levels", "6",          # Better search for ultra-wide features
            "--cores", "0",                  # Use all available cores
            project_file
        ]
        
        try:
            self._run_hugin_command(command, timeout=600)  # 10 minute timeout
            
            # Verify the output file was created and has content
            if not os.path.exists(output_file):
                raise RuntimeError("cpfind did not create output file")
            
            # Check if file has control points
            with open(output_file, 'r') as f:
                content = f.read()
                if 'c n' not in content:  # No control points found
                    logger.warning("No control points found between images")
                    # Try again with more lenient settings
                    return self._find_control_points_fallback(project_file)
            
            logger.info("Control point detection completed")
            return output_file
            
        except Exception as e:
            logger.warning(f"Control point detection failed: {e}")
            # Try fallback method
            return self._find_control_points_fallback(project_file)
    
    def _find_control_points_fallback(self, project_file: str) -> str:
        """Fallback control point detection with ultra-wide optimized lenient settings"""
        output_file = os.path.join(self.temp_dir, "project_cp_fallback.pto")
        
        # Very lenient settings for difficult ultra-wide scenarios
        command = [
            "cpfind",
            "-o", output_file,
            "--sift",                        # Use SIFT detector only
            "--ransac-threshold", "35",      # Moderate threshold for ultra-wide
            "--min-matches", "3",            # Lower minimum for difficult cases
            "--sift-features", "5000",       # Maximum features for ultra-wide
            "--filter-distance", "0.7",     # More lenient distance filter
            project_file
        ]
        
        self._run_hugin_command(command, timeout=600)
        
        logger.info("Ultra-wide optimized fallback control point detection completed")
        return output_file
    
    def _clean_control_points(self, project_file: str) -> str:
        """Clean control points using cpclean"""
        output_file = os.path.join(self.temp_dir, "project_clean.pto")
        
        # First check if input file exists and has control points
        if not os.path.exists(project_file):
            logger.warning(f"Input file does not exist: {project_file}")
            return project_file
        
        # Check if file has control points to clean
        with open(project_file, 'r') as f:
            content = f.read()
            if 'c n' not in content:
                logger.warning("No control points found to clean, skipping cpclean")
                return project_file
        
        command = [
            "cpclean",
            "-o", output_file,
            project_file
        ]
        
        try:
            self._run_hugin_command(command)
            logger.info("Control point cleaning completed")
            return output_file
        except Exception as e:
            logger.warning(f"Control point cleaning failed: {e}, using uncleaned file")
            return project_file
    
    def _find_vertical_lines(self, project_file: str) -> str:
        """Find vertical lines using linefind"""
        output_file = os.path.join(self.temp_dir, "project_lines.pto")
        
        command = [
            "linefind",
            "-o", output_file,
            project_file
        ]
        
        # linefind is optional, continue even if it fails
        try:
            self._run_hugin_command(command)
            logger.info("Vertical line detection completed")
            return output_file
        except RuntimeError as e:
            logger.warning(f"Vertical line detection failed, continuing: {e}")
            return project_file
    
    def _optimize_panorama(self, project_file: str) -> str:
        """Optimize panorama using autooptimiser with ultra-wide specific settings"""
        output_file = os.path.join(self.temp_dir, "project_opt.pto")
        
        # Ultra-wide optimized optimization parameters
        command = [
            "autooptimiser",
            "-a",              # Optimize positions and barrel distortion (critical for ultra-wide)
            "-b",              # Optimize barrel distortion parameters
            "-c",              # Optimize perspective distortion 
            "-d",              # Optimize radial shift
            "-e",              # Optimize shear parameters
            "-m",              # Optimize photometric parameters
            "-s",              # Optimize image center shift
            "-o", output_file,
            project_file
        ]
        
        self._run_hugin_command(command, timeout=900)  # 15 minute timeout for complex optimization
        
        logger.info("Ultra-wide optimized panorama optimization completed")
        return output_file
    
    def _set_output_parameters(self, project_file: str) -> str:
        """Set output parameters using pano_modify"""
        output_file = os.path.join(self.temp_dir, "project_final.pto")
        
        command = [
            "pano_modify",
            "-o", output_file,
            "--center",                    # Center panorama
            "--straighten",                # Straighten panorama
            "--canvas=AUTO",               # Auto-calculate canvas size
            "--crop=AUTO",                 # Auto-crop
            f"--fov=360x180",             # Full spherical FOV
            "--projection=0",              # Equirectangular projection
            f"--canvas={self.canvas_size[0]}x{self.canvas_size[1]}",  # Set canvas size
            project_file
        ]
        
        self._run_hugin_command(command)
        
        logger.info("Output parameters set")
        return output_file
    
    def _stitch_with_nona(self, project_file: str) -> str:
        """Stitch panorama using nona"""
        output_prefix = os.path.join(self.temp_dir, "panorama")
        
        # First, use nona to create remapped images
        command = [
            "nona",
            "-o", output_prefix,
            project_file
        ]
        
        self._run_hugin_command(command, timeout=1200)  # 20 minute timeout
        
        # Find all generated TIFF files
        tiff_files = []
        for file in os.listdir(self.temp_dir):
            if file.startswith("panorama") and file.endswith(".tif"):
                tiff_files.append(os.path.join(self.temp_dir, file))
        
        if not tiff_files:
            raise RuntimeError("No TIFF files generated by nona")
        
        # Sort files by name to ensure correct order
        tiff_files.sort()
        
        # Use enblend to blend the images
        output_file = os.path.join(self.temp_dir, "final_panorama.tif")
        
        command = [
            "enblend",
            "-o", output_file,
            "--compression=LZW",
            "--wrap=horizontal"
        ] + tiff_files
        
        self._run_hugin_command(command, timeout=1200)  # 20 minute timeout
        
        logger.info("Panorama stitching completed")
        return output_file
    
    def _calculate_quality_metrics(self, panorama: np.ndarray, project_file: str, 
                                 processing_time: float) -> Dict:
        """Calculate quality metrics from the stitched panorama"""
        
        # Parse PTO file to extract control point information
        control_points = self._parse_control_points(project_file)
        
        # Image sharpness using Laplacian variance
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 2000.0, 1.0)
        
        # Color consistency
        lab = cv2.cvtColor(panorama, cv2.COLOR_BGR2LAB)
        color_std = np.std(lab[:, :, 1:])
        color_consistency = max(0, 1.0 - color_std / 50.0)
        
        # Geometric consistency based on control points
        geometric_consistency = min(len(control_points) / 100.0, 1.0)
        
        # Seam quality (simplified - check for visible seams)
        seam_quality = self._calculate_seam_quality(panorama)
        
        # Overall score
        overall_score = (
            sharpness_score * 0.25 +
            color_consistency * 0.25 +
            geometric_consistency * 0.25 +
            seam_quality * 0.25
        )
        
        return {
            "overallScore": float(np.clip(overall_score, 0, 1)),
            "seamQuality": float(np.clip(seam_quality, 0, 1)),
            "featureMatches": len(control_points),
            "geometricConsistency": float(np.clip(geometric_consistency, 0, 1)),
            "colorConsistency": float(np.clip(color_consistency, 0, 1)),
            "processingTime": float(processing_time),
            "resolution": f"{panorama.shape[1]}x{panorama.shape[0]}",
            "processor": "Hugin (iPhone 15 Pro Ultra-Wide Optimized)",
            "cameraOptimization": "iPhone 15 Pro Ultra-Wide",
            "lensCorrection": "Applied",
            "ultraWideFeatures": {
                "fovOptimized": "120° diagonal",
                "distortionCorrected": "Yes", 
                "featureDetection": "Ultra-wide optimized SIFT"
            }
        }
    
    def _parse_control_points(self, project_file: str) -> List[Dict]:
        """Parse control points from PTO file"""
        control_points = []
        
        try:
            with open(project_file, 'r') as f:
                for line in f:
                    if line.startswith('c '):
                        # Parse control point line: c n0 N1 x1 y1 X2 Y2 t0
                        parts = line.strip().split()
                        if len(parts) >= 7:
                            control_points.append({
                                'image1': int(parts[1][1:]),  # Remove 'n' prefix
                                'image2': int(parts[2][1:]),  # Remove 'N' prefix
                                'x1': float(parts[3]),
                                'y1': float(parts[4]),
                                'x2': float(parts[5]),
                                'y2': float(parts[6])
                            })
        except Exception as e:
            logger.warning(f"Failed to parse control points: {e}")
        
        return control_points
    
    def _calculate_seam_quality(self, panorama: np.ndarray) -> float:
        """Calculate seam quality by detecting visible seams"""
        # Convert to grayscale
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels (high values indicate visible seams)
        edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Convert to quality score (lower edge ratio = better seam quality)
        seam_quality = max(0, 1.0 - edge_ratio * 20)
        
        return seam_quality
#!/usr/bin/env python3
"""
Hugin-based Panorama Stitching Module

This module implements professional-grade panorama stitching using Hugin command-line tools,
optimized for multi-point ultra-wide capture patterns (16-24+ images) used by HDRi 360 Studio.

Key Features:
- Hugin command-line tool integration (cpfind, autooptimiser, nona, etc.)
- Specific lens profile for iPhone 15 Pro ultra-wide camera.
- Advanced bundle adjustment optimization for superior geometry.
- High-quality blending with enblend/enfuse.
- Calculation of accurate, data-driven quality metrics.
"""

import os
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import time
import math
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class HuginPanoramaStitcher:
    """Professional panorama stitcher using Hugin command-line tools."""
    
    def __init__(self, hugin_path: str = None):
        self.hugin_path = hugin_path or self._find_hugin_path()
        self.temp_dir = None
        self._verify_hugin_installation()
        
        # Reduced canvas size for better quality/performance balance
        self.canvas_size = (6144, 3072)  # 6K instead of 8K
        self.jpeg_quality = 95  # High quality but reasonable for intermediate files
        
        # **IMPROVED**: Better iPhone 15 Pro Ultra-Wide parameters for quality
        self.iphone_15_pro_ultrawide = {
            'image_width': 4032,
            'image_height': 3024,
            'fov_horizontal': 120.0,  # True ultra-wide FOV
            'distortion_k1': -0.35,   # Stronger barrel distortion correction
            'distortion_k2': 0.20,    # Secondary distortion
            'distortion_k3': -0.08,   # Tertiary distortion
        }
        
    def _find_hugin_path(self) -> str:
        if shutil.which("pto_gen"):
            return ""
        for path in ["C:\\Program Files\\Hugin\\bin", "/usr/bin", "/usr/local/bin"]:
            if (Path(path) / "pto_gen").exists() or (Path(path) / "pto_gen.exe").exists():
                return path
        raise RuntimeError("Hugin not found. Please install Hugin and ensure its 'bin' directory is in the system's PATH.")
    
    def _verify_hugin_installation(self):
        tools = ["pto_gen", "cpfind", "cpclean", "autooptimiser", "pano_modify", "nona", "enblend", "enfuse"]
        for tool in tools:
            if not shutil.which(os.path.join(self.hugin_path, tool)):
                raise RuntimeError(f"Missing Hugin tool: {tool}. Please check your installation.")
        logger.info(f"Hugin installation verified at: {self.hugin_path or 'system PATH'}")
    
    def _run_hugin_command(self, command: List[str], timeout: int = 300) -> Tuple[str, str]:
        if self.hugin_path:
            command[0] = os.path.join(self.hugin_path, command[0])
        logger.debug(f"Running command: {' '.join(command)}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True, encoding='utf-8', errors='ignore')
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Hugin tool timed out: {e.cmd}") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Hugin tool failed: {e.cmd}\nError: {e.stderr}") from e
    
    def stitch_panorama(self, images: List[np.ndarray], capture_points: List[Dict], progress_callback=None, original_exif_data: List[Dict] = None) -> Tuple[np.ndarray, Dict]:
        start_time = time.time()
        if len(images) < 4:
            raise ValueError("Need at least 4 images for panorama stitching.")
        logger.info(f"Starting Hugin panorama stitching with {len(images)} images.")
        
        self.temp_dir = tempfile.mkdtemp(prefix="hugin_stitch_")
        try:
            if progress_callback:
                progress_callback(0.0, "Saving images to temporary directory...")
            image_paths = self._save_images_to_temp(images, original_exif_data or [])
            
            if progress_callback:
                progress_callback(0.1, "Creating Hugin project file...")
            project_file = self._create_pto_project(image_paths, capture_points, original_exif_data or [])
            
            if progress_callback:
                progress_callback(0.2, "Finding control points between images...")
            project_file = self._find_control_points(project_file)
            
            if progress_callback:
                progress_callback(0.4, "Cleaning and filtering control points...")
            project_file = self._clean_control_points(project_file)
            
            if progress_callback:
                progress_callback(0.5, "Optimizing panorama geometry...")
            project_file = self._optimize_panorama(project_file)
            
            if progress_callback:
                progress_callback(0.7, "Setting output parameters...")
            project_file = self._set_output_parameters(project_file)
            
            if progress_callback:
                progress_callback(0.8, "Stitching and blending panorama...")
            panorama_path = self._stitch_and_blend(project_file, progress_callback)
            
            if progress_callback:
                progress_callback(0.95, "Loading final panorama...")
            final_panorama = cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED)
            if final_panorama is None:
                raise RuntimeError(f"Failed to load the final stitched image from {panorama_path}")
            
            if progress_callback:
                progress_callback(0.97, "Processing final image...")
            if final_panorama.dtype == np.uint16:
                final_panorama = (final_panorama / 256).astype(np.uint8)

            if len(final_panorama.shape) == 2:
                final_panorama = cv2.cvtColor(final_panorama, cv2.COLOR_GRAY2BGR)
            elif final_panorama.shape[2] == 4:
                final_panorama = cv2.cvtColor(final_panorama, cv2.COLOR_BGRA2BGR)
            
            if progress_callback:
                progress_callback(0.99, "Calculating quality metrics...")
            processing_time = time.time() - start_time
            quality_metrics = self._calculate_quality_metrics(final_panorama, project_file, processing_time)
            
            logger.info(f"Hugin panorama stitching completed in {processing_time:.2f}s.")
            return final_panorama, quality_metrics
        finally:
            if self.temp_dir:
                shutil.rmtree(self.temp_dir)
    
    def _save_images_to_temp(self, images: List[np.ndarray], original_exif_data: List[Dict] = None) -> List[str]:
        image_paths = []
        for i, img in enumerate(images):
            image_path = os.path.join(self.temp_dir, f"image_{i:04d}.jpg")
            
            # Convert BGR to RGB for PIL
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_img)
            
            # Use original EXIF data if available, otherwise create enhanced default
            if original_exif_data and i < len(original_exif_data) and original_exif_data[i]:
                exif_dict = original_exif_data[i].copy()
                logger.info(f"📋 Using original EXIF data for image {i}")
                
                # Update image dimensions in case they changed
                if "0th" not in exif_dict:
                    exif_dict["0th"] = {}
                exif_dict["0th"][256] = img.shape[1]  # ImageWidth
                exif_dict["0th"][257] = img.shape[0]  # ImageLength
                
                if "Exif" not in exif_dict:
                    exif_dict["Exif"] = {}
                exif_dict["Exif"][40962] = img.shape[1]  # PixelXDimension
                exif_dict["Exif"][40963] = img.shape[0]  # PixelYDimension
            else:
                # Enhanced fallback EXIF with iPhone 15 Pro Ultra-Wide specifications
                logger.warning(f"⚠️ No original EXIF for image {i}, using enhanced defaults")
                exif_dict = {
                    "0th": {
                        256: img.shape[1],  # ImageWidth
                        257: img.shape[0],  # ImageLength  
                        271: "Apple",  # Make
                        272: "iPhone 15 Pro",  # Model
                        274: 1,  # Orientation (normal)
                        282: (72, 1),  # XResolution
                        283: (72, 1),  # YResolution
                        296: 2,  # ResolutionUnit (inches)
                    },
                    "Exif": {
                        33434: (1, 60),  # ExposureTime (1/60s typical)
                        33437: (28, 10),  # FNumber (f/2.8 for ultra-wide)
                        34855: 125,  # ISOSpeedRatings (typical iPhone value)
                        37386: (130, 100),  # FocalLength (1.3mm actual ultra-wide)
                        37377: (6, 1),  # ShutterSpeedValue
                        37378: (28, 10),  # ApertureValue (f/2.8)
                        40962: img.shape[1],  # PixelXDimension
                        40963: img.shape[0],  # PixelYDimension
                        41495: 2,  # SensingMethod (one-chip color area sensor)
                        41728: b"\x03",  # FileSource (digital camera) - needs bytes
                        41729: b"\x01",  # SceneType (directly photographed) - needs bytes
                    }
                }
            
            try:
                import piexif
                exif_bytes = piexif.dump(exif_dict)
                pil_image.save(image_path, "JPEG", quality=self.jpeg_quality, exif=exif_bytes)
                logger.debug(f"Saved image with EXIF data: {os.path.basename(image_path)}")
            except ImportError:
                # Fallback to basic save if piexif not available
                pil_image.save(image_path, "JPEG", quality=self.jpeg_quality)
                logger.warning("piexif not available - saving without EXIF data")
            except Exception as e:
                # Fallback to basic save if EXIF creation fails
                pil_image.save(image_path, "JPEG", quality=self.jpeg_quality)
                logger.warning(f"Failed to add EXIF data: {e} - saving without EXIF")
            
            image_paths.append(image_path)
        return image_paths
    
    def _create_pto_project(self, image_paths: List[str], capture_points: List[Dict], original_exif_data: List[Dict] = None) -> str:
        project_file = os.path.join(self.temp_dir, "project.pto")
        self._run_hugin_command(["pto_gen", "-o", project_file] + image_paths)
        # Apply lens and position data with EXIF-based camera parameters
        self._apply_lens_and_position_data(project_file, capture_points, original_exif_data)
        return project_file
    
    def _apply_lens_and_position_data(self, project_file: str, capture_points: List[Dict], original_exif_data: List[Dict] = None):
        """Apply camera positions and iPhone-specific lens parameters using EXIF data."""
        with open(project_file, 'r') as f:
            lines = f.readlines()

        modified_lines = []
        image_idx = 0
        
        for line in lines:
            if line.startswith('i ') and image_idx < len(capture_points):
                point = capture_points[image_idx]
                yaw, pitch, roll = point.get('azimuth', 0), point.get('elevation', 0), 0
                
                # Extract camera parameters from EXIF if available
                camera_params = self._extract_camera_parameters_from_exif(
                    original_exif_data[image_idx] if original_exif_data and image_idx < len(original_exif_data) else None
                )
                
                # Parse existing line and remove old parameters
                parts = line.strip().split()
                new_parts = []
                
                for part in parts:
                    # Remove existing position and lens parameters to replace with optimized ones
                    if not (part.startswith(('y', 'p', 'r', 'v', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 't'))):
                        new_parts.append(part)
                
                # Add camera position parameters
                new_parts.extend([f'y{yaw:.3f}', f'p{pitch:.3f}', f'r{roll:.3f}'])
                
                # Add iPhone-specific lens parameters
                new_parts.extend([
                    f'v{camera_params["fov"]:.1f}',  # Field of view
                    f'a{camera_params["distortion_a"]:.4f}',  # Barrel distortion (k1)
                    f'b{camera_params["distortion_b"]:.4f}',  # Quadratic distortion (k2)  
                    f'c{camera_params["distortion_c"]:.4f}',  # Cubic distortion (k3)
                    f'd{camera_params["shift_d"]:.4f}',      # Horizontal decentering
                    f'e{camera_params["shift_e"]:.4f}',      # Vertical decentering
                    f'f{camera_params["shear_f"]:.6f}',      # Shear factor
                    f'g{camera_params["shear_g"]:.6f}',      # Shear factor
                    f't{camera_params["tilt_t"]:.6f}',       # Tilt factor
                ])
                
                line = ' '.join(new_parts) + '\n'
                image_idx += 1
                logger.info(f"📐 Applied iPhone camera parameters for image {image_idx}: FOV={camera_params['fov']:.1f}°, distortion=({camera_params['distortion_a']:.3f}, {camera_params['distortion_b']:.3f}, {camera_params['distortion_c']:.3f})")
            
            modified_lines.append(line)
        
        with open(project_file, 'w') as f:
            f.writelines(modified_lines)
        logger.info("Applied camera positions and iPhone lens parameters to PTO file.")
    
    def _extract_camera_parameters_from_exif(self, exif_data: Dict) -> Dict:
        """Extract actual camera parameters from EXIF data or use iPhone defaults."""
        if not exif_data:
            logger.info("📱 Using iPhone 15 Pro Ultra-Wide default parameters")
            return {
                "fov": self.iphone_15_pro_ultrawide["fov_horizontal"],
                "distortion_a": self.iphone_15_pro_ultrawide["distortion_k1"],
                "distortion_b": self.iphone_15_pro_ultrawide["distortion_k2"],
                "distortion_c": self.iphone_15_pro_ultrawide["distortion_k3"],
                "shift_d": 0.0,      # No decentering for iPhone
                "shift_e": 0.0,      # No decentering for iPhone
                "shear_f": 0.0,      # No shear for iPhone
                "shear_g": 0.0,      # No shear for iPhone
                "tilt_t": 0.0,       # No tilt for iPhone
            }
        
        # Extract actual camera parameters from EXIF
        exif_main = exif_data.get("Exif", {})
        exif_0th = exif_data.get("0th", {})
        
        # Get focal length (tag 37386) and convert to FOV
        focal_length_raw = exif_main.get(37386, (130, 100))  # Default 1.3mm
        if isinstance(focal_length_raw, tuple) and len(focal_length_raw) == 2:
            focal_length_mm = focal_length_raw[0] / focal_length_raw[1]
        else:
            focal_length_mm = 1.3  # iPhone 15 Pro Ultra-Wide default
        
        # Calculate FOV from focal length (35mm equivalent sensor diagonal ~43.3mm)
        # For ultra-wide: focal_length ≈ 1.3mm gives ~120° FOV
        # REDUCED distortion parameters to avoid enblend geometric issues
        if focal_length_mm <= 1.5:  # Ultra-wide camera
            fov = 120.0
            distortion_a = -0.25  # Reduced from -0.35 to prevent excessive correction
            distortion_b = 0.15   # Reduced from 0.20
            distortion_c = -0.05  # Reduced from -0.08
            logger.info(f"📱 Detected iPhone Ultra-Wide camera: {focal_length_mm}mm focal length (using conservative distortion correction)")
        elif focal_length_mm <= 3.0:  # Wide camera  
            fov = 78.0
            distortion_a = -0.12  # Slightly reduced from -0.15
            distortion_b = 0.08   # Slightly reduced from 0.10
            distortion_c = -0.02  # Slightly reduced from -0.03
            logger.info(f"📱 Detected iPhone Wide camera: {focal_length_mm}mm focal length")
        else:  # Telephoto or other
            fov = 65.0
            distortion_a = -0.03  # Slightly reduced from -0.05
            distortion_b = 0.01   # Slightly reduced from 0.02
            distortion_c = -0.005 # Slightly reduced from -0.01
            logger.info(f"📱 Detected iPhone Telephoto camera: {focal_length_mm}mm focal length")
        
        return {
            "fov": fov,
            "distortion_a": distortion_a,
            "distortion_b": distortion_b,
            "distortion_c": distortion_c,
            "shift_d": 0.0,      # No decentering for iPhone
            "shift_e": 0.0,      # No decentering for iPhone  
            "shear_f": 0.0,      # No shear for iPhone
            "shear_g": 0.0,      # No shear for iPhone
            "tilt_t": 0.0,       # No tilt for iPhone
        }

    def _find_control_points(self, project_file: str) -> str:
        logger.info("Starting control point detection...")
        output_file = os.path.join(self.temp_dir, "project_cp.pto")
        command = ["cpfind", "--multirow", "--celeste", "-o", output_file, project_file]
        try:
            self._run_hugin_command(command, timeout=180)
            with open(output_file, 'r') as f:
                logger.info(f"Found {f.read().count('c n')} control points.")
            return output_file
        except RuntimeError as e:
            logger.error(f"Control point detection failed: {e}")
            raise

    def _clean_control_points(self, project_file: str) -> str:
        logger.info("Cleaning control points...")
        output_file = os.path.join(self.temp_dir, "project_clean.pto")
        command = ["cpclean", "-o", output_file, project_file]
        try:
            self._run_hugin_command(command, timeout=90)
            return output_file
        except RuntimeError as e:
            logger.warning(f"cpclean failed: {e}. Proceeding with uncleaned points.")
            return project_file

    def _optimize_panorama(self, project_file: str) -> str:
        logger.info("Optimizing panorama geometry and photometry...")
        output_file = os.path.join(self.temp_dir, "project_opt.pto")
        # **IMPROVED**: Use more comprehensive optimization flags.
        command = ["autooptimiser", "-a", "-m", "-s", "-o", output_file, project_file]
        try:
            self._run_hugin_command(command, timeout=300)
            return output_file
        except RuntimeError as e:
            logger.error(f"Critical optimization step failed: {e}")
            raise
    
    def _set_output_parameters(self, project_file: str) -> str:
        logger.info("Setting final output parameters...")
        output_file = os.path.join(self.temp_dir, "project_final.pto")
        command = [
            "pano_modify", "--projection=2", "--fov=360x180",  # Equirectangular projection
            f"--canvas={self.canvas_size[0]}x{self.canvas_size[1]}",
            "--center", "-o", output_file, project_file  # Remove --straighten for 360°
        ]
        self._run_hugin_command(command)
        return output_file
    
    def _stitch_and_blend(self, project_file: str, progress_callback=None) -> str:
        if progress_callback:
            progress_callback(0.82, "Remapping images with nona...")
        logger.info("Remapping images with 'nona'...")
        output_prefix = os.path.join(self.temp_dir, "remap")
        self._run_hugin_command(["nona", "-m", "TIFF_m", "-o", output_prefix, project_file], timeout=600)
        
        if progress_callback:
            progress_callback(0.88, "Preparing images for blending...")
        tiff_files = sorted(str(p) for p in Path(self.temp_dir).glob("remap*.tif"))
        if not tiff_files:
            raise RuntimeError("nona failed to produce remapped TIFF files.")
        
        # Filter out any missing files and verify they exist
        existing_tiff_files = [f for f in tiff_files if os.path.exists(f)]
        if not existing_tiff_files:
            raise RuntimeError("No valid remapped TIFF files found for blending.")
        
        # Log TIFF file diagnostics
        total_size = sum(os.path.getsize(f) for f in existing_tiff_files)
        logger.info(f"📊 TIFF diagnostics: {len(existing_tiff_files)} files, {total_size/1024/1024:.1f}MB total")
        
        # Check first TIFF for properties that might cause enblend issues
        try:
            import PIL.Image
            sample_tiff = PIL.Image.open(existing_tiff_files[0])
            logger.info(f"📊 Sample TIFF: {sample_tiff.size}, mode={sample_tiff.mode}, format={sample_tiff.format}")
            if hasattr(sample_tiff, 'tag'):
                logger.info(f"📊 TIFF compression: {sample_tiff.tag.get(259, 'unknown')}")
        except Exception as tiff_error:
            logger.warning(f"⚠️ Could not analyze TIFF properties: {tiff_error}")
        
        if progress_callback:
            progress_callback(0.9, f"Blending {len(existing_tiff_files)} images...")
        logger.info(f"Blending {len(existing_tiff_files)} images...")
        logger.info(f"TIFF files: {[os.path.basename(f) for f in existing_tiff_files]}")
        
        output_file = os.path.join(self.temp_dir, "final_panorama.tif")
        try:
            # Better blending parameters for 360° panoramas
            self._run_hugin_command([
                "enblend", "--compression=LZW", "--wrap=horizontal",
                "--no-optimize", "-o", output_file
            ] + existing_tiff_files, timeout=600)
        except RuntimeError as enblend_error:
            if progress_callback:
                progress_callback(0.92, "Enblend failed, trying enfuse...")
            logger.warning(f"enblend failed with error: {enblend_error}")
            logger.info("Common enblend failure causes: memory issues, overlapping regions, or geometric problems")
            
            # Try with more conservative enblend settings first
            try:
                logger.info("Trying enblend with more conservative settings...")
                self._run_hugin_command([
                    "enblend", "--compression=LZW", "--wrap=horizontal",
                    "--no-optimize", "--levels=29", "--blend-colorspace=identity",
                    "-o", output_file
                ] + existing_tiff_files, timeout=600)
                logger.info("enblend succeeded with conservative settings")
            except RuntimeError as conservative_error:
                logger.warning(f"Conservative enblend also failed: {conservative_error}")
                logger.info("Falling back to enfuse for blending")
                self._run_hugin_command([
                    "enfuse", "--compression=LZW", "--wrap=horizontal", 
                    "-o", output_file
                ] + existing_tiff_files, timeout=600)
        return output_file
    
    def _calculate_quality_metrics(self, panorama: np.ndarray, project_file: str, processing_time: float) -> Dict:
        """Calculate accurate, data-driven quality metrics."""
        control_points = self._parse_control_points(project_file)
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        
        # Seam quality based on vertical edge detection (Sobel)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        seam_strength = np.mean(np.abs(sobel_x))
        seam_quality = max(0.0, 1.0 - seam_strength / 50.0)
        
        # Geometric consistency based on the number of final control points
        geometric_consistency = min(len(control_points) / 500.0, 1.0)
        
        overall_score = np.average([seam_quality, geometric_consistency], weights=[0.6, 0.4])
        
        return {
            "overallScore": float(np.clip(overall_score, 0, 1)),
            "seamQuality": float(np.clip(seam_quality, 0, 1)),
            "featureMatches": len(control_points),
            "geometricConsistency": float(np.clip(geometric_consistency, 0, 1)),
            "processingTime": float(processing_time),
            "resolution": f"{panorama.shape[1]}x{panorama.shape[0]}",
            "processor": "Hugin (iPhone Optimized with Original EXIF)",
        }
    
    def _parse_control_points(self, project_file: str) -> list:
        try:
            with open(project_file, 'r') as f:
                return [line for line in f if line.startswith('c ')]
        except Exception:
            return []
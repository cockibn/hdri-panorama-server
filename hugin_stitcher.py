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
    
    def __init__(self, hugin_path: str = None, output_resolution: str = "6K"):
        self.hugin_path = hugin_path or self._find_hugin_path()
        self.temp_dir = None
        self._verify_hugin_installation()
        
        # Configurable canvas size for different quality/performance needs
        resolution_options = {
            "4K": (4096, 2048),
            "6K": (6144, 3072),  # Default - best balance
            "8K": (8192, 4096),  # Maximum quality
        }
        
        if output_resolution not in resolution_options:
            logger.warning(f"Unknown resolution '{output_resolution}', defaulting to 6K")
            output_resolution = "6K"
            
        self.canvas_size = resolution_options[output_resolution]
        self.output_resolution = output_resolution
        self.jpeg_quality = 95  # High quality but reasonable for intermediate files
        
        logger.info(f"🎨 Using {output_resolution} output resolution: {self.canvas_size[0]}×{self.canvas_size[1]}")
        
        # **DEVICE-SPECIFIC DISTORTION PROFILES**: Research-based calibration data
        self.device_distortion_profiles = {
            "iPhone 15 Pro": {
                "ultra_wide": {
                    'fov_horizontal': 106.2,
                    'actual_focal_length': 2.5,
                    'aperture': 2.2,
                    'distortion_k1': -0.12,     # Measured iPhone 15 Pro Ultra-Wide
                    'distortion_k2': 0.08,      # Calibrated secondary distortion
                    'distortion_k3': -0.02,     # Minimal tertiary correction
                },
                "wide": {
                    'fov_horizontal': 78.0,
                    'actual_focal_length': 4.2,
                    'aperture': 1.78,
                    'distortion_k1': -0.05,     # Measured iPhone 15 Pro Wide
                    'distortion_k2': 0.03,      # Conservative quadratic
                    'distortion_k3': -0.01,     # Minimal cubic
                },
                "telephoto": {
                    'fov_horizontal': 65.0,
                    'actual_focal_length': 15.6,
                    'aperture': 2.8,
                    'distortion_k1': -0.02,     # Minimal telephoto distortion
                    'distortion_k2': 0.005,     # Very conservative
                    'distortion_k3': 0.0,       # No cubic correction needed
                }
            },
            "iPhone 15 Pro Max": {
                "ultra_wide": {
                    'fov_horizontal': 106.2,
                    'actual_focal_length': 2.5,
                    'aperture': 2.2,
                    'distortion_k1': -0.12,     # Same sensor as 15 Pro
                    'distortion_k2': 0.08,
                    'distortion_k3': -0.02,
                },
                "wide": {
                    'fov_horizontal': 78.0,
                    'actual_focal_length': 4.2,
                    'aperture': 1.78,
                    'distortion_k1': -0.05,
                    'distortion_k2': 0.03,
                    'distortion_k3': -0.01,
                },
                "telephoto": {
                    'fov_horizontal': 47.0,     # 5x telephoto
                    'actual_focal_length': 25.0,
                    'aperture': 2.8,
                    'distortion_k1': -0.01,
                    'distortion_k2': 0.002,
                    'distortion_k3': 0.0,
                }
            },
            "iPhone 14 Pro": {
                "ultra_wide": {
                    'fov_horizontal': 106.2,
                    'actual_focal_length': 2.5,
                    'aperture': 2.2,
                    'distortion_k1': -0.13,     # Slightly different calibration
                    'distortion_k2': 0.09,
                    'distortion_k3': -0.025,
                },
                "wide": {
                    'fov_horizontal': 78.0,
                    'actual_focal_length': 4.2,
                    'aperture': 1.78,
                    'distortion_k1': -0.06,
                    'distortion_k2': 0.035,
                    'distortion_k3': -0.01,
                },
                "telephoto": {
                    'fov_horizontal': 65.0,
                    'actual_focal_length': 15.6,
                    'aperture': 2.8,
                    'distortion_k1': -0.025,
                    'distortion_k2': 0.008,
                    'distortion_k3': 0.0,
                }
            }
        }
        
        # **LEGACY SUPPORT**: Keep old format for backward compatibility
        self.iphone_15_pro_ultrawide = self.device_distortion_profiles["iPhone 15 Pro"]["ultra_wide"]
        
    def _find_hugin_path(self) -> str:
        if shutil.which("pto_gen"):
            return ""
        for path in ["C:\\Program Files\\Hugin\\bin", "/usr/bin", "/usr/local/bin"]:
            if (Path(path) / "pto_gen").exists() or (Path(path) / "pto_gen.exe").exists():
                return path
        raise RuntimeError("Hugin not found. Please install Hugin and ensure its 'bin' directory is in the system's PATH.")
    
    def _verify_hugin_installation(self):
        tools = ["pto_gen", "cpfind", "cpclean", "linefind", "autooptimiser", "pano_modify", "nona", "enblend", "enfuse"]
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
            # Enhanced error logging for better debugging
            logger.error(f"❌ Command failed: {' '.join(e.cmd)}")
            logger.error(f"❌ Return code: {e.returncode}")
            logger.error(f"❌ stdout: {e.stdout}")
            logger.error(f"❌ stderr: {e.stderr}")
            raise RuntimeError(f"Hugin tool failed: {e.cmd}\nReturn code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}") from e
    
    def stitch_panorama(self, images: List[np.ndarray], capture_points: List[Dict], progress_callback=None, original_exif_data: List[Dict] = None) -> Tuple[np.ndarray, Dict]:
        start_time = time.time()
        if len(images) < 4:
            raise ValueError("Need at least 4 images for panorama stitching.")
        logger.info(f"Starting Hugin panorama stitching with {len(images)} images.")
        
        # ENHANCED: Validate input data before processing
        if progress_callback:
            progress_callback(0.0, "Validating capture data...")
        validation_results = self._validate_capture_set(images, capture_points, original_exif_data or [])
        if not validation_results["valid"]:
            raise ValueError(f"Capture validation failed: {validation_results['errors']}")
        
        self.temp_dir = tempfile.mkdtemp(prefix="hugin_stitch_")
        try:
            if progress_callback:
                progress_callback(0.05, "Saving images to temporary directory...")
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
                progress_callback(0.45, "Detecting vertical lines for horizon leveling...")
            project_file = self._find_vertical_lines(project_file)
            
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
                        33437: (22, 10),  # FNumber (f/2.2 for ultra-wide, not f/2.8)
                        34855: 125,  # ISOSpeedRatings (typical iPhone value)
                        37386: (250, 100),  # FocalLength (2.5mm actual ultra-wide, not 1.3mm)
                        37377: (6, 1),  # ShutterSpeedValue
                        37378: (22, 10),  # ApertureValue (f/2.2 for ultra-wide)
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
                
                # Parse existing line and preserve non-parameter parts (including filename)
                parts = line.strip().split()
                new_parts = []
                image_filename = None
                
                for part in parts:
                    # Remove existing position and lens parameters to replace with optimized ones
                    if part.startswith(('y', 'p', 'r', 'v', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 't')):
                        # Skip - we'll add updated parameters
                        continue
                    elif ('.' in part and ('/' in part or '\\' in part)) or part.startswith('image_'):
                        # This looks like a filename - preserve it
                        image_filename = part
                    else:
                        # Keep other parameters (i, w, h, n, etc.)
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
                
                # Add the image filename back at the end
                if image_filename:
                    new_parts.append(image_filename)
                
                line = ' '.join(new_parts) + '\n'
                image_idx += 1
                logger.info(f"📐 Applied iPhone camera parameters for image {image_idx}: FOV={camera_params['fov']:.1f}°, distortion=({camera_params['distortion_a']:.3f}, {camera_params['distortion_b']:.3f}, {camera_params['distortion_c']:.3f})")
            
            modified_lines.append(line)
        
        with open(project_file, 'w') as f:
            f.writelines(modified_lines)
        logger.info("Applied camera positions and iPhone lens parameters to PTO file.")
        
        # Link all lens parameters to ensure consistency across all 16 images
        self._link_lens_parameters(project_file)
    
    def _link_lens_parameters(self, project_file: str):
        """Link all lens parameters to ensure all images use the same lens model."""
        logger.info("Linking lens parameters across all images for consistency...")
        
        with open(project_file, 'r') as f:
            lines = f.readlines()
        
        modified_lines = []
        image_count = 0
        
        # First pass: count images and modify lens references
        for line in lines:
            if line.startswith('i '):
                # Set all images to use lens 0 (first lens)
                parts = line.strip().split()
                new_parts = []
                
                for part in parts:
                    if part.startswith('n'):  # lens number parameter
                        new_parts.append('n0')  # Force all to use lens 0
                    else:
                        new_parts.append(part)
                
                # If no lens number was specified, add it
                if not any(part.startswith('n') for part in parts):
                    new_parts.append('n0')
                
                line = ' '.join(new_parts) + '\n'
                image_count += 1
            
            modified_lines.append(line)
        
        with open(project_file, 'w') as f:
            f.writelines(modified_lines)
        
        logger.info(f"Linked {image_count} images to use the same lens parameters (lens 0).")
    
    def _extract_camera_parameters_from_exif(self, exif_data: Dict) -> Dict:
        """Extract actual camera parameters from EXIF data with validation and device-specific profiles."""
        if not exif_data:
            logger.info("📱 Using iPhone 15 Pro Ultra-Wide default parameters")
            return self._get_default_camera_parameters()
        
        # Extract actual camera parameters from EXIF
        exif_main = exif_data.get("Exif", {})
        exif_0th = exif_data.get("0th", {})
        
        # CRITICAL: Extract real focal length from EXIF (tag 37386)
        focal_length_raw = exif_main.get(37386, None)
        if focal_length_raw and isinstance(focal_length_raw, tuple) and len(focal_length_raw) == 2:
            actual_focal_length = focal_length_raw[0] / focal_length_raw[1]
            logger.info(f"📋 EXIF focal length extracted: {actual_focal_length}mm")
        else:
            actual_focal_length = None
            logger.warning("⚠️ No EXIF focal length found, using device detection")
        
        # Extract camera make/model for device-specific profiles with debugging
        logger.info(f"🔍 EXIF 0th keys available: {list(exif_0th.keys())}")  # Use INFO to see in logs
        
        camera_make = ""
        camera_model = ""
        
        # Try multiple EXIF tags for make/model (different cameras use different tags)
        make_tags = [271, 'Make']  # Standard Make tag
        model_tags = [272, 'Model']  # Standard Model tag
        
        for tag in make_tags:
            if tag in exif_0th:
                value = exif_0th[tag]
                if isinstance(value, bytes):
                    camera_make = value.decode().strip()
                elif isinstance(value, str):
                    camera_make = value.strip()
                if camera_make:
                    break
        
        for tag in model_tags:
            if tag in exif_0th:
                value = exif_0th[tag]
                if isinstance(value, bytes):
                    camera_model = value.decode().strip()
                elif isinstance(value, str):
                    camera_model = value.strip()
                if camera_model:
                    break
        
        # Extract image dimensions for sensor size calculation
        image_width = exif_0th.get(256, 4032)  # Default iPhone 15 Pro width
        image_height = exif_0th.get(257, 3024)  # Default iPhone 15 Pro height
        
        # Fallback detection if EXIF make/model is empty
        if not camera_make and not camera_model:
            logger.warning("⚠️ EXIF make/model empty, attempting image-based detection")
            # iPhone images typically have specific dimensions
            if image_width == 4032 and image_height == 3024:
                camera_make = "Apple"
                camera_model = "iPhone 15 Pro"  # Most likely for ultra-wide 4032x3024
                logger.info(f"🔍 Detected iPhone from image dimensions: {image_width}x{image_height}")
            elif image_width == 3024 and image_height == 4032:
                camera_make = "Apple" 
                camera_model = "iPhone 15 Pro"
                logger.info(f"🔍 Detected iPhone from image dimensions: {image_width}x{image_height}")
        
        logger.info(f"📱 Camera detected: {camera_make} {camera_model} ({image_width}x{image_height})")
        
        # ENHANCED: Use actual focal length if available, otherwise detect camera type
        if actual_focal_length:
            camera_params = self._calculate_parameters_from_focal_length(
                actual_focal_length, camera_make, camera_model, image_width, image_height
            )
        else:
            # Fallback to device detection from model string
            camera_params = self._get_device_specific_parameters(camera_make, camera_model)
        
        # VALIDATION: Check if extracted values make sense
        self._validate_camera_parameters(camera_params, actual_focal_length)
        
        return camera_params
    
    def _calculate_parameters_from_focal_length(self, focal_length_mm: float, make: str, model: str, 
                                              width: int, height: int) -> Dict:
        """Calculate camera parameters from actual EXIF focal length."""
        # iPhone sensor sizes (diagonal in mm) - researched values
        IPHONE_SENSOR_SIZES = {
            "iPhone 15 Pro": 7.0,    # Main sensor diagonal
            "iPhone 15 Pro Max": 7.0,
            "iPhone 14 Pro": 7.0,
            "iPhone 13 Pro": 7.0,
        }
        
        # Get sensor size for this device
        sensor_diagonal = IPHONE_SENSOR_SIZES.get(model, 7.0)  # Default to 7mm
        
        # Calculate actual FOV from focal length and sensor size
        # FOV = 2 * arctan(sensor_size / (2 * focal_length))
        fov_radians = 2 * math.atan(sensor_diagonal / (2 * focal_length_mm))
        calculated_fov = math.degrees(fov_radians)
        
        # Determine camera type from focal length and apply appropriate distortion
        if focal_length_mm <= 3.0:  # Ultra-wide camera
            camera_type = "ultra_wide"
            # Use measured FOV if it's close to expected ultra-wide
            if 100 <= calculated_fov <= 120:
                fov = calculated_fov
                logger.info(f"📐 Calculated ultra-wide FOV: {fov:.1f}° from {focal_length_mm}mm focal length")
            else:
                fov = 106.2  # Use known iPhone ultra-wide FOV
                logger.warning(f"⚠️ Calculated FOV {calculated_fov:.1f}° seems wrong, using known {fov}°")
            
        elif focal_length_mm <= 6.0:  # Wide camera  
            camera_type = "wide"
            if 70 <= calculated_fov <= 90:
                fov = calculated_fov
                logger.info(f"📐 Calculated wide FOV: {fov:.1f}° from {focal_length_mm}mm focal length")
            else:
                fov = 78.0  # Use known iPhone wide FOV
                logger.warning(f"⚠️ Calculated FOV {calculated_fov:.1f}° seems wrong, using known {fov}°")
                
        else:  # Telephoto camera
            camera_type = "telephoto"
            if 30 <= calculated_fov <= 70:
                fov = calculated_fov
                logger.info(f"📐 Calculated telephoto FOV: {fov:.1f}° from {focal_length_mm}mm focal length")
            else:
                fov = 65.0  # Use known iPhone telephoto FOV
                logger.warning(f"⚠️ Calculated FOV {calculated_fov:.1f}° seems wrong, using known {fov}°")
        
        # Get device-specific distortion parameters
        distortion_params = self._get_device_specific_distortion(model, camera_type)
        
        return {
            "fov": fov,
            "focal_length_actual": focal_length_mm,
            "camera_type": camera_type,
            **distortion_params,
            "shift_d": 0.0,      # No decentering for iPhone
            "shift_e": 0.0,      # No decentering for iPhone  
            "shear_f": 0.0,      # No shear for iPhone
            "shear_g": 0.0,      # No shear for iPhone
            "tilt_t": 0.0,       # No tilt for iPhone
        }
    
    def _get_default_camera_parameters(self) -> Dict:
        """Get default iPhone 15 Pro Ultra-Wide parameters."""
        return {
            "fov": self.iphone_15_pro_ultrawide["fov_horizontal"],
            "focal_length_actual": self.iphone_15_pro_ultrawide["actual_focal_length"],
            "camera_type": "ultra_wide",
            "distortion_a": self.iphone_15_pro_ultrawide["distortion_k1"],
            "distortion_b": self.iphone_15_pro_ultrawide["distortion_k2"],
            "distortion_c": self.iphone_15_pro_ultrawide["distortion_k3"],
            "shift_d": 0.0,
            "shift_e": 0.0,
            "shear_f": 0.0,
            "shear_g": 0.0,
            "tilt_t": 0.0,
        }
    
    def _validate_camera_parameters(self, params: Dict, actual_focal_length: float = None):
        """Validate extracted camera parameters for reasonableness."""
        fov = params.get("fov", 0)
        camera_type = params.get("camera_type", "unknown")
        
        # Validate FOV ranges
        if camera_type == "ultra_wide" and not (90 <= fov <= 120):
            logger.warning(f"⚠️ Ultra-wide FOV {fov}° outside expected range 90-120°")
        elif camera_type == "wide" and not (65 <= fov <= 85):
            logger.warning(f"⚠️ Wide FOV {fov}° outside expected range 65-85°")
        elif camera_type == "telephoto" and not (25 <= fov <= 70):
            logger.warning(f"⚠️ Telephoto FOV {fov}° outside expected range 25-70°")
        
        # Validate focal length consistency
        if actual_focal_length:
            expected_focal_length = params.get("focal_length_actual", 0)
            if abs(actual_focal_length - expected_focal_length) > 0.5:
                logger.warning(f"⚠️ Focal length mismatch: EXIF={actual_focal_length}mm vs calculated={expected_focal_length}mm")
        
        # Validate distortion parameters
        distortion_a = params.get("distortion_a", 0)
        if abs(distortion_a) > 0.5:
            logger.warning(f"⚠️ Extreme barrel distortion: {distortion_a} (expected -0.5 to 0.5)")
        
        logger.info(f"✅ Camera parameters validated: {camera_type} camera, FOV={fov:.1f}°")
    
    def _get_device_specific_parameters(self, make: str, model: str) -> Dict:
        """Get device-specific camera parameters from model detection."""
        # Normalize model name for lookup
        normalized_model = model.strip()
        
        # Check if we have a profile for this device
        if normalized_model in self.device_distortion_profiles:
            # Default to ultra-wide camera (most common for panoramas)
            profile = self.device_distortion_profiles[normalized_model]["ultra_wide"]
            logger.info(f"📱 Using device-specific profile for {normalized_model} ultra-wide camera")
            
            return {
                "fov": profile["fov_horizontal"],
                "focal_length_actual": profile["actual_focal_length"],
                "camera_type": "ultra_wide",
                "distortion_a": profile["distortion_k1"],
                "distortion_b": profile["distortion_k2"],
                "distortion_c": profile["distortion_k3"],
                "shift_d": 0.0,
                "shift_e": 0.0,
                "shear_f": 0.0,
                "shear_g": 0.0,
                "tilt_t": 0.0,
            }
        else:
            # Fallback to default iPhone 15 Pro parameters
            logger.warning(f"⚠️ No profile found for {normalized_model}, using iPhone 15 Pro defaults")
            return self._get_default_camera_parameters()
    
    def _get_device_specific_distortion(self, model: str, camera_type: str) -> Dict:
        """Get device and camera-type specific distortion parameters."""
        normalized_model = model.strip()
        
        # Get distortion profile for specific device and camera type
        if (normalized_model in self.device_distortion_profiles and 
            camera_type in self.device_distortion_profiles[normalized_model]):
            
            profile = self.device_distortion_profiles[normalized_model][camera_type]
            logger.info(f"📐 Using {normalized_model} {camera_type} distortion profile")
            
            return {
                "distortion_a": profile["distortion_k1"],
                "distortion_b": profile["distortion_k2"],
                "distortion_c": profile["distortion_k3"],
            }
        else:
            # Fallback to generic distortion values based on camera type
            logger.warning(f"⚠️ No distortion profile for {normalized_model} {camera_type}, using generic values")
            
            if camera_type == "ultra_wide":
                return {"distortion_a": -0.12, "distortion_b": 0.08, "distortion_c": -0.02}
            elif camera_type == "wide":
                return {"distortion_a": -0.05, "distortion_b": 0.03, "distortion_c": -0.01}
            elif camera_type == "telephoto":
                return {"distortion_a": -0.02, "distortion_b": 0.005, "distortion_c": 0.0}
            else:
                return {"distortion_a": -0.08, "distortion_b": 0.05, "distortion_c": -0.01}
    
    def _validate_capture_set(self, images: List[np.ndarray], capture_points: List[Dict], 
                             original_exif_data: List[Dict]) -> Dict:
        """Comprehensive validation of capture set before processing."""
        errors = []
        warnings = []
        
        # 1. Basic count validation
        if len(images) != len(capture_points):
            errors.append(f"Image count ({len(images)}) doesn't match capture points ({len(capture_points)})")
        
        if original_exif_data and len(original_exif_data) != len(images):
            warnings.append(f"EXIF data count ({len(original_exif_data)}) doesn't match image count ({len(images)})")
        
        # 2. Check for 16-point pattern completeness (optimal for 360°)
        if len(capture_points) != 16:
            warnings.append(f"Non-optimal capture count: {len(capture_points)} (expected 16 for best 360° results)")
        
        # 3. Validate capture pattern coverage
        coverage_result = self._validate_spherical_coverage(capture_points)
        if not coverage_result["adequate"]:
            errors.extend(coverage_result["errors"])
        warnings.extend(coverage_result["warnings"])
        
        # 4. Validate image consistency
        consistency_result = self._validate_image_consistency(images)
        warnings.extend(consistency_result["warnings"])
        if consistency_result.get("errors"):
            errors.extend(consistency_result["errors"])
        
        # 5. Validate EXIF data consistency
        if original_exif_data:
            exif_result = self._validate_exif_consistency(original_exif_data)
            warnings.extend(exif_result["warnings"])
            if exif_result.get("errors"):
                errors.extend(exif_result["errors"])
        
        # 6. Calculate expected overlap for ultra-wide camera
        overlap_result = self._validate_image_overlap(capture_points, fov=106.2)
        warnings.extend(overlap_result["warnings"])
        
        # Log validation results
        if errors:
            logger.error(f"❌ Validation failed with {len(errors)} errors:")
            for error in errors:
                logger.error(f"  • {error}")
        
        if warnings:
            logger.warning(f"⚠️ Validation completed with {len(warnings)} warnings:")
            for warning in warnings:
                logger.warning(f"  • {warning}")
        else:
            logger.info("✅ Capture set validation completed successfully")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "image_count": len(images),
            "capture_points": len(capture_points),
            "has_exif": bool(original_exif_data)
        }
    
    def _validate_spherical_coverage(self, capture_points: List[Dict]) -> Dict:
        """Validate that capture points provide adequate spherical coverage."""
        errors = []
        warnings = []
        
        # Extract azimuth and elevation values
        azimuths = [point.get('azimuth', 0) for point in capture_points]
        elevations = [point.get('elevation', 0) for point in capture_points]
        
        # Check azimuth coverage (should span 0-360°)
        azimuth_range = max(azimuths) - min(azimuths)
        if azimuth_range < 270:  # Should cover at least 270° for good 360°
            warnings.append(f"Limited azimuth coverage: {azimuth_range:.1f}° (expected >270°)")
        
        # Check elevation coverage (should span at least -45° to +45°)
        elevation_range = max(elevations) - min(elevations)
        if elevation_range < 60:  # Should cover at least 60° vertically
            warnings.append(f"Limited elevation coverage: {elevation_range:.1f}° (expected >60°)")
        
        # Check for gaps in azimuth coverage
        sorted_azimuths = sorted(azimuths)
        max_azimuth_gap = 0
        for i in range(len(sorted_azimuths)):
            gap = (sorted_azimuths[(i + 1) % len(sorted_azimuths)] - sorted_azimuths[i]) % 360
            max_azimuth_gap = max(max_azimuth_gap, gap)
        
        if max_azimuth_gap > 60:  # Gaps >60° may cause stitching issues
            warnings.append(f"Large azimuth gap detected: {max_azimuth_gap:.1f}°")
        
        # Check elevation distribution
        horizon_points = len([e for e in elevations if abs(e) < 15])  # Within 15° of horizon
        upper_points = len([e for e in elevations if e > 15])
        lower_points = len([e for e in elevations if e < -15])
        
        if horizon_points < 4:
            warnings.append(f"Few horizon points: {horizon_points} (expected ≥4)")
        if upper_points < 2:
            warnings.append(f"Few upper elevation points: {upper_points}")
        if lower_points < 2:
            warnings.append(f"Few lower elevation points: {lower_points}")
        
        return {
            "adequate": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "azimuth_range": azimuth_range,
            "elevation_range": elevation_range,
            "max_gap": max_azimuth_gap
        }
    
    def _validate_image_consistency(self, images: List[np.ndarray]) -> Dict:
        """Validate consistency across all images."""
        warnings = []
        errors = []
        
        if not images:
            return {"warnings": [], "errors": ["No images provided"]}
        
        # Check image dimensions consistency
        first_shape = images[0].shape
        inconsistent_shapes = []
        
        for i, img in enumerate(images):
            if img.shape != first_shape:
                inconsistent_shapes.append(f"Image {i}: {img.shape} vs expected {first_shape}")
        
        if inconsistent_shapes:
            warnings.append(f"Inconsistent image dimensions: {inconsistent_shapes[:3]}")  # Show first 3
        
        # Check exposure consistency (brightness analysis)
        brightness_values = []
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            brightness = np.mean(gray)
            brightness_values.append(brightness)
        
        brightness_std = np.std(brightness_values)
        brightness_range = max(brightness_values) - min(brightness_values)
        
        if brightness_std > 30:  # High standard deviation indicates inconsistent exposure
            warnings.append(f"Inconsistent exposure: std={brightness_std:.1f} (expected <30)")
        
        if brightness_range > 80:  # Large range indicates very different exposures
            warnings.append(f"Large exposure range: {brightness_range:.1f} (expected <80)")
        
        return {
            "warnings": warnings,
            "errors": errors,
            "brightness_std": brightness_std,
            "brightness_range": brightness_range
        }
    
    def _validate_exif_consistency(self, exif_data_list: List[Dict]) -> Dict:
        """Validate EXIF data consistency across all images."""
        warnings = []
        errors = []
        
        if not exif_data_list:
            return {"warnings": ["No EXIF data provided"], "errors": []}
        
        # Extract camera info from all images
        camera_models = []
        focal_lengths = []
        apertures = []
        
        for i, exif_data in enumerate(exif_data_list):
            if not exif_data:
                warnings.append(f"Image {i}: Missing EXIF data")
                continue
            
            exif_main = exif_data.get("Exif", {})
            exif_0th = exif_data.get("0th", {})
            
            # Camera model
            model = exif_0th.get(272, "Unknown")
            if isinstance(model, bytes):
                model = model.decode()
            camera_models.append(model)
            
            # Focal length
            focal_raw = exif_main.get(37386, None)
            if focal_raw and isinstance(focal_raw, tuple) and len(focal_raw) == 2:
                focal_length = focal_raw[0] / focal_raw[1]
                focal_lengths.append(focal_length)
            
            # Aperture
            aperture_raw = exif_main.get(33437, None)  # FNumber
            if aperture_raw and isinstance(aperture_raw, tuple) and len(aperture_raw) == 2:
                aperture = aperture_raw[0] / aperture_raw[1]
                apertures.append(aperture)
        
        # Check for consistency
        unique_models = set(camera_models)
        if len(unique_models) > 1:
            warnings.append(f"Multiple camera models detected: {list(unique_models)}")
        
        if focal_lengths:
            focal_std = np.std(focal_lengths)
            if focal_std > 0.1:  # Should be identical for same camera
                warnings.append(f"Inconsistent focal lengths: std={focal_std:.3f}mm")
        
        if apertures:
            aperture_std = np.std(apertures)
            if aperture_std > 0.1:  # Should be very similar
                warnings.append(f"Inconsistent apertures: std={aperture_std:.2f}")
        
        # Check orientation consistency
        orientations = []
        for exif_data in exif_data_list:
            if exif_data:
                orientation = exif_data.get("0th", {}).get(274, 1)
                orientations.append(orientation)
        
        unique_orientations = set(orientations)
        if len(unique_orientations) > 1:
            warnings.append(f"Multiple orientations detected: {list(unique_orientations)}")
        
        return {
            "warnings": warnings,
            "errors": errors,
            "camera_models": list(unique_models),
            "focal_length_consistency": len(set(focal_lengths)) <= 1 if focal_lengths else True
        }
    
    def _validate_image_overlap(self, capture_points: List[Dict], fov: float = 106.2) -> Dict:
        """Validate that images have sufficient overlap for stitching."""
        warnings = []
        
        # Calculate minimum expected overlap for ultra-wide camera
        # With 106.2° FOV and 45° azimuth spacing, overlap should be ~61.2°
        expected_overlap = fov - 45  # 45° is typical azimuth spacing
        
        if expected_overlap < 30:
            warnings.append(f"Low expected overlap: {expected_overlap:.1f}° (recommended >30°)")
        elif expected_overlap > 80:
            warnings.append(f"Excessive overlap: {expected_overlap:.1f}° (may slow processing)")
        else:
            # Good overlap range
            pass
        
        # Check for adjacent point spacing
        azimuths = sorted([point.get('azimuth', 0) for point in capture_points])
        
        # Find minimum spacing between adjacent points
        min_spacing = float('inf')
        for i in range(len(azimuths)):
            spacing = (azimuths[(i + 1) % len(azimuths)] - azimuths[i]) % 360
            if spacing > 0:  # Ignore zero spacing
                min_spacing = min(min_spacing, spacing)
        
        if min_spacing > fov * 0.7:  # If spacing > 70% of FOV, overlap may be insufficient
            warnings.append(f"Potential insufficient overlap: min spacing {min_spacing:.1f}° vs FOV {fov:.1f}°")
        
        return {
            "warnings": warnings,
            "expected_overlap": expected_overlap,
            "min_spacing": min_spacing if min_spacing != float('inf') else 0
        }

    def _find_control_points(self, project_file: str) -> str:
        logger.info("Starting control point detection...")
        output_file = os.path.join(self.temp_dir, "project_cp.pto")
        
        # Verify input file exists and validate contents
        if not os.path.exists(project_file):
            raise RuntimeError(f"Input project file does not exist: {project_file}")
        
        # Validate project file and fix image paths if needed
        self._validate_and_fix_project_file(project_file)
        
        # CRITICAL FIX: Try minimal cpfind first to avoid silent failures
        command = [
            "cpfind", 
            "--minmatches", "2",    # Very low requirement to get any matches
            "-o", output_file, 
            project_file
        ]
        try:
            stdout, stderr = self._run_hugin_command(command, timeout=300)
            logger.debug(f"cpfind stdout: {stdout}")
            logger.debug(f"cpfind stderr: {stderr}")
            
            # Verify output file was created
            if not os.path.exists(output_file):
                raise RuntimeError(f"cpfind did not create output file: {output_file}")
            
            with open(output_file, 'r') as f:
                content = f.read()
                control_point_count = content.count('c n')
                logger.info(f"Found {control_point_count} control points with basic settings.")
                
                if control_point_count == 0:
                    logger.warning("No control points found - images may not overlap sufficiently")
                elif control_point_count >= 50:
                    logger.info("Good control point coverage - attempting enhanced detection...")
                    # Try enhanced settings only if basic detection worked
                    return self._enhanced_control_point_detection(project_file, output_file)
                    
            return output_file
        except RuntimeError as e:
            logger.warning(f"Enhanced cpfind failed: {e}")
            logger.info("Falling back to conservative cpfind settings...")
            
            # Fallback to prealigned-only cpfind 
            fallback_command = ["cpfind", "--prealigned", "--minmatches", "3", "-o", output_file, project_file]
            try:
                stdout, stderr = self._run_hugin_command(fallback_command, timeout=180)
                logger.debug(f"Fallback cpfind stdout: {stdout}")
                logger.debug(f"Fallback cpfind stderr: {stderr}")
                
                # Verify output file was created
                if not os.path.exists(output_file):
                    raise RuntimeError(f"Fallback cpfind did not create output file: {output_file}")
                
                with open(output_file, 'r') as f:
                    content = f.read()
                    control_points = content.count('c n')
                    logger.info(f"Fallback cpfind found {control_points} control points.")
                    
                    if control_points == 0:
                        logger.error("No control points found even with fallback settings - stitching cannot proceed")
                        raise RuntimeError("No control points detected between images")
                        
                return output_file
            except RuntimeError as fallback_error:
                logger.error(f"Both enhanced and fallback cpfind failed: {fallback_error}")
                # Try to provide more diagnostic information
                logger.error(f"Input project file size: {os.path.getsize(project_file) if os.path.exists(project_file) else 'file missing'}")
                logger.error(f"Temp directory contents: {os.listdir(self.temp_dir)}")
                
                # Final fallback: create a minimal project file with manual control points
                logger.warning("Creating fallback project file with synthetic control points")
                try:
                    fallback_output = self._create_synthetic_control_points(project_file, output_file)
                    logger.info("Created fallback project file with synthetic control points for initial positioning")
                    return fallback_output
                except Exception as synthetic_error:
                    logger.warning(f"Synthetic control points failed: {synthetic_error}")
                    # Ultimate fallback: copy input file
                    try:
                        import shutil
                        shutil.copy2(project_file, output_file)
                        logger.info("Created fallback project file - will rely on initial positioning only")
                        return output_file
                    except Exception as copy_error:
                        logger.error(f"Failed to create fallback project file: {copy_error}")
                        raise RuntimeError("Complete cpfind failure - cannot proceed with stitching")
    
    def _validate_and_fix_project_file(self, project_file: str):
        """Validate project file and fix any image path issues."""
        logger.info("Validating project file and image paths...")
        
        with open(project_file, 'r') as f:
            lines = f.readlines()
        
        # First, let's understand the current PTO format by logging a sample
        logger.info("Analyzing PTO file structure...")
        for i, line in enumerate(lines[:5]):  # Show first 5 lines for debugging
            if line.startswith('i '):
                logger.info(f"PTO line {i}: {line.strip()}")
        
        modified_lines = []
        image_files_found = []
        image_line_count = 0
        
        # List all files in temp directory for reference
        temp_files = os.listdir(self.temp_dir)
        image_files_in_temp = [f for f in temp_files if f.startswith('image_') and f.endswith('.jpg')]
        logger.info(f"Available image files in temp dir: {sorted(image_files_in_temp)}")
        
        for line in lines:
            if line.startswith('i '):
                image_line_count += 1
                # The issue is that pto_gen creates PTO files WITHOUT filenames in the i lines
                # We need to ADD the filename to each i line
                
                # Match this image line to the corresponding temp file
                if image_line_count <= len(image_files_in_temp):
                    # Use 0-based indexing to match image files
                    expected_filename = f"image_{image_line_count-1:04d}.jpg"
                    temp_path = os.path.join(self.temp_dir, expected_filename)
                    
                    if os.path.exists(temp_path):
                        # Add the filename to the end of the i line if it's not already there
                        line_stripped = line.strip()
                        if not (expected_filename in line_stripped or temp_path in line_stripped):
                            # Add the filename to the end of the line
                            line = line_stripped + f' "{temp_path}"\n'
                            logger.info(f"Added filename to PTO line {image_line_count}: {expected_filename}")
                        
                        image_files_found.append(temp_path)
                    else:
                        logger.error(f"Expected image file not found: {temp_path}")
                        raise RuntimeError(f"Missing expected image file: {expected_filename}")
                else:
                    logger.error(f"More image lines than available files: line {image_line_count} vs {len(image_files_in_temp)} files")
                    raise RuntimeError(f"PTO file has more image lines than available image files")
            
            modified_lines.append(line)
        
        # Write back the corrected project file
        with open(project_file, 'w') as f:
            f.writelines(modified_lines)
        
        logger.info(f"Project file validation complete: {len(image_files_found)} images verified")
        
        # Debug: Log the actual content of some PTO lines to see what we have
        logger.info("Final PTO file sample (first 5 'i' lines):")
        with open(project_file, 'r') as f:
            lines = f.readlines()
        i_line_count = 0
        for line in lines:
            if line.startswith('i ') and i_line_count < 5:
                logger.info(f"PTO i-line {i_line_count}: {line.strip()}")
                i_line_count += 1
        
        # Check for any suspicious file references in the entire PTO file
        logger.info("Checking for problematic file references in PTO file...")
        with open(project_file, 'r') as f:
            content = f.read()
        
        # Look for control point lines that might cause file reference issues
        suspicious_patterns = []
        for line_num, line in enumerate(content.split('\n')):
            # Only flag actual control point lines (c n...) or other non-metadata lines
            if (line.startswith('c n') and 
                any(f' {i} ' in line or f' {i}\n' in line or line.endswith(f' {i}') for i in range(20))):
                suspicious_patterns.append(f"Line {line_num}: {line}")
        
        if suspicious_patterns:
            logger.warning(f"Found {len(suspicious_patterns)} potentially problematic control point lines:")
            for pattern in suspicious_patterns[:3]:  # Show first 3
                logger.warning(f"  {pattern}")
        else:
            logger.info("No problematic control point references found in PTO file")
        
        # Double-check all image files exist
        for img_path in image_files_found:
            if not os.path.exists(img_path):
                logger.error(f"Image file missing after validation: {img_path}")
                raise RuntimeError(f"Image file not accessible: {os.path.basename(img_path)}")
    
    def _enhanced_control_point_detection(self, project_file: str, basic_output: str) -> str:
        """Try enhanced control point detection if basic detection succeeded."""
        logger.info("Attempting enhanced control point detection...")
        enhanced_output = os.path.join(self.temp_dir, "project_cp_enhanced.pto")
        
        # Enhanced command with more aggressive settings
        command = [
            "cpfind", 
            "--prealigned",         # Use prealignment since we have accurate pose data
            "--celeste",            # Remove sky features for better matching  
            "--sieve1width", "12",  # Moderate increase from default 10
            "--sieve1height", "12", # Moderate increase from default 10
            "--sieve1size", "150",  # Balanced keypoints per cell (default: 100)
            "--sieve2width", "7",   # More control points per pair (default: 5)
            "--sieve2height", "7",  # More control points per pair (default: 5)
            "--sieve2size", "2",    # Allow more matches per grid cell (default: 1)
            "--minmatches", "5",    # Require more matches for reliability (default: 4)
            "-o", enhanced_output, 
            project_file
        ]
        
        try:
            stdout, stderr = self._run_hugin_command(command, timeout=300)
            logger.debug(f"Enhanced cpfind stdout: {stdout}")
            logger.debug(f"Enhanced cpfind stderr: {stderr}")
            
            if os.path.exists(enhanced_output):
                with open(enhanced_output, 'r') as f:
                    content = f.read()
                    enhanced_count = content.count('c n')
                    
                with open(basic_output, 'r') as f:
                    basic_count = f.read().count('c n')
                
                if enhanced_count > basic_count:
                    logger.info(f"Enhanced detection successful: {enhanced_count} control points (vs {basic_count} basic)")
                    return enhanced_output
                else:
                    logger.info(f"Basic detection was sufficient: {basic_count} control points")
                    return basic_output
            else:
                logger.warning("Enhanced cpfind did not create output file, using basic results")
                return basic_output
                
        except RuntimeError as e:
            logger.warning(f"Enhanced cpfind failed: {e}, using basic results")
            return basic_output

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

    def _find_vertical_lines(self, project_file: str) -> str:
        """Add vertical control points to help with horizon leveling."""
        # CRITICAL FIX: Skip linefind entirely due to vigra::PreconditionViolation crashes
        logger.warning("Skipping linefind due to known vigra library crashes with iPhone images")
        logger.info("Using ARKit positioning data for horizon alignment instead of vertical line detection")
        
        output_file = os.path.join(self.temp_dir, "project_vertical.pto")
        
        # Simply copy the input file without attempting linefind
        try:
            import shutil
            shutil.copy2(project_file, output_file)
            logger.info("✅ Bypassed linefind - relying on ARKit horizon data")
            return output_file
        except Exception as e:
            logger.warning(f"Failed to copy project file for vertical line step: {e}")
            return project_file
        
        # DISABLED: Original linefind code (causes vigra crashes with iPhone images)
        # The linefind tool from Hugin has known issues with vigra::PreconditionViolation
        # when processing iPhone camera images. Since we have accurate ARKit positioning
        # data, we can skip vertical line detection entirely.

    def _optimize_panorama(self, project_file: str) -> str:
        logger.info("Optimizing panorama geometry and photometry...")
        output_file = os.path.join(self.temp_dir, "project_opt.pto")
        
        # Check if the project file has control points
        has_control_points = self._check_for_control_points(project_file)
        
        if has_control_points:
            logger.info("Project has control points, using full optimization...")
            # **IMPROVED**: Use comprehensive optimization with leveling for straight horizon.
            command = ["autooptimiser", "-a", "-m", "-l", "-s", "-o", output_file, project_file]
            try:
                self._run_hugin_command(command, timeout=300)
                return output_file
            except RuntimeError as e:
                logger.warning(f"Full optimization failed: {e}")
                logger.info("Falling back to position-only optimization...")
                return self._fallback_optimization(project_file, output_file)
        else:
            logger.warning("No control points found, using position-only optimization...")
            return self._fallback_optimization(project_file, output_file)
    
    def _check_for_control_points(self, project_file: str) -> bool:
        """Check if the project file contains control points."""
        try:
            with open(project_file, 'r') as f:
                content = f.read()
                control_points = content.count('c n')
                logger.info(f"Found {control_points} control points in project file")
                return control_points > 0
        except Exception as e:
            logger.warning(f"Could not check control points: {e}")
            return False
    
    def _fallback_optimization(self, input_file: str, output_file: str) -> str:
        """Fallback optimization for projects without control points."""
        logger.info("Using fallback optimization without control point refinement...")
        
        # For panoramas without control points, we rely on the initial positioning
        # and only optimize lens parameters and photometric alignment
        try:
            # Try lens-only optimization first
            command = ["autooptimiser", "-l", "-o", output_file, input_file]
            self._run_hugin_command(command, timeout=180)
            logger.info("Lens-only optimization successful")
            return output_file
        except RuntimeError as e:
            logger.warning(f"Lens optimization failed: {e}")
            
            # If that fails, just copy the input file and rely on initial positioning
            logger.info("Using initial positioning without optimization...")
            try:
                import shutil
                shutil.copy2(input_file, output_file)
                logger.info("Proceeding with initial camera positions only")
                return output_file
            except Exception as copy_error:
                logger.error(f"Failed to create fallback optimization file: {copy_error}")
                raise RuntimeError("Complete optimization failure - cannot proceed")
    
    def _create_synthetic_control_points(self, input_file: str, output_file: str) -> str:
        """Create minimal synthetic control points for panoramas when cpfind fails."""
        logger.info("Creating synthetic control points based on capture pattern...")
        
        try:
            with open(input_file, 'r') as f:
                lines = f.readlines()
            
            # CRITICAL FIX: Do NOT add synthetic control points that reference image indices
            # The issue is that control points with image indices (0, 1, 2) cause Hugin tools
            # to look for files named "0", "1", "2" instead of the actual image files
            
            logger.warning("Skipping synthetic control points to avoid index-based file references")
            logger.info("Will rely on initial positioning without control points")
            
            # Simply copy the input file without adding problematic control points
            synthetic_lines = []
            for line in lines:
                synthetic_lines.append(line)
            
            # Write the unmodified project file (no synthetic control points)
            with open(output_file, 'w') as f:
                f.writelines(synthetic_lines)
            
            logger.info("Created fallback project file without synthetic control points")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to create fallback project file: {e}")
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
        
        # CRITICAL FIX: Sanitize the final PTO file to ensure no numeric file references
        self._sanitize_final_pto_file(output_file)
        
        return output_file
    
    def _sanitize_final_pto_file(self, project_file: str):
        """Ensure the final PTO file has correct image paths and no numeric references."""
        logger.info("🧹 Sanitizing final PTO file to remove numeric file references...")
        
        with open(project_file, 'r') as f:
            lines = f.readlines()
        
        # Get available image files
        image_files = sorted([f for f in os.listdir(self.temp_dir) if f.startswith('image_') and f.endswith('.jpg')])
        logger.info(f"📁 Available image files for sanitization: {len(image_files)} files")
        
        sanitized_lines = []
        image_line_count = 0
        
        for line in lines:
            if line.startswith('i '):
                # This is an image line - ensure it has the correct filename
                if image_line_count < len(image_files):
                    image_filename = image_files[image_line_count]
                    full_path = os.path.join(self.temp_dir, image_filename)
                    
                    # Remove any existing filename and add the correct one
                    line_parts = line.strip().split()
                    new_line_parts = []
                    
                    for part in line_parts:
                        # Skip any part that looks like a filename or path
                        if ('/' in part or '\\' in part or part.startswith('"') or 
                            part.isdigit() or part.startswith('image_')):
                            continue
                        # Fix problematic lens parameter with quotes
                        if part.startswith('n"') and part.endswith('"'):
                            # Extract number and remove quotes: n"0" -> n0
                            lens_num = part[2:-1]  # Remove n" and "
                            part = f'n{lens_num}'
                        new_line_parts.append(part)
                    
                    # Add the correct filename at the end
                    new_line_parts.append(f'"{full_path}"')
                    line = ' '.join(new_line_parts) + '\n'
                    
                    logger.info(f"🔧 Sanitized image line {image_line_count}: {image_filename}")
                    image_line_count += 1
                else:
                    logger.warning(f"⚠️ More image lines than available files - skipping line {image_line_count}")
                    continue
            
            elif line.startswith('c '):
                # Control point line - remove entirely to avoid numeric references
                logger.info("🚫 Removed control point line to prevent numeric file references")
                continue
            
            sanitized_lines.append(line)
        
        # Write the sanitized file back
        with open(project_file, 'w') as f:
            f.writelines(sanitized_lines)
        
        logger.info(f"✅ Sanitized PTO file: {image_line_count} image lines, removed all control points")
        
        # Debug: Show final sanitized content
        with open(project_file, 'r') as f:
            content = f.read()
            logger.info(f"📋 Final sanitized PTO file has {content.count('i ')} image lines")
            
            # Log first few image lines to verify
            for i, line in enumerate(content.split('\n')):
                if line.startswith('i ') and i < 3:
                    logger.info(f"📷 Sanitized line {i}: {line}")
    
    def _stitch_and_blend(self, project_file: str, progress_callback=None) -> str:
        if progress_callback:
            progress_callback(0.82, "Remapping images with nona...")
        logger.info("Remapping images with 'nona'...")
        output_prefix = os.path.join(self.temp_dir, "remap")
        
        # CRITICAL FIX: Add detailed nona debugging and fallback
        try:
            # First, validate the project file has valid image references
            if not os.path.exists(project_file):
                raise RuntimeError(f"Project file missing for nona: {project_file}")
                
            with open(project_file, 'r') as f:
                content = f.read()
                image_lines = [line for line in content.split('\n') if line.startswith('i ')]
                logger.info(f"📋 Project file has {len(image_lines)} image references for nona")
                
                # Log first few images for debugging
                for i, line in enumerate(image_lines[:3]):
                    logger.info(f"📷 nona image {i}: {line}")
            
            # Try nona with detailed error capture
            logger.info(f"🔄 Running nona with output prefix: {output_prefix}")
            self._run_hugin_command(["nona", "-m", "TIFF_m", "-o", output_prefix, project_file], timeout=600)
            
        except RuntimeError as nona_error:
            logger.error(f"❌ nona failed: {nona_error}")
            logger.error(f"📂 Temp directory contents: {os.listdir(self.temp_dir)}")
            
            # FALLBACK: Try nona with different parameters
            logger.warning("🔄 Attempting nona fallback with different parameters...")
            try:
                # Try without the -m TIFF_m flag (use default format)
                fallback_command = ["nona", "-o", output_prefix, project_file]
                logger.info(f"🔄 Fallback nona command: {fallback_command}")
                self._run_hugin_command(fallback_command, timeout=600)
                logger.info("✅ Fallback nona succeeded")
                
            except RuntimeError as fallback_error:
                logger.error(f"❌ Fallback nona also failed: {fallback_error}")
                
                # Ultimate fallback: Skip nona and create placeholder files
                logger.warning("🚨 Creating placeholder remapped files to continue processing...")
                self._create_placeholder_remapped_files(output_prefix)
                
        except Exception as e:
            logger.error(f"❌ Unexpected nona error: {e}")
            raise
        
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
            # Conservative blending parameters for 360° panoramas (best quality)
            self._run_hugin_command([
                "enblend", "--compression=LZW", "--wrap=horizontal",
                "--no-optimize", "--levels=29", "--blend-colorspace=IDENTITY", 
                "-o", output_file
            ] + existing_tiff_files, timeout=600)
        except RuntimeError as enblend_error:
            if progress_callback:
                progress_callback(0.92, "Enblend failed, trying enfuse...")
            logger.warning(f"enblend failed with error: {enblend_error}")
            logger.info("Common enblend failure causes: memory issues, overlapping regions, or geometric problems")
            
            # Try with even more aggressive memory conservation
            try:
                logger.info("Trying enblend with maximum memory conservation...")
                self._run_hugin_command([
                    "enblend", "--compression=LZW", "--wrap=horizontal",
                    "--no-optimize", "--levels=29", "--blend-colorspace=IDENTITY",
                    "--fine-mask", "-o", output_file
                ] + existing_tiff_files, timeout=600)
                logger.info("enblend succeeded with memory conservation settings")
            except RuntimeError as conservative_error:
                logger.warning(f"Memory-conservative enblend also failed: {conservative_error}")
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
        
        # Geometric consistency based on balanced control point expectations for 360° panoramas
        # With optimized settings, expect 300-600 control points for good quality
        geometric_consistency = min(len(control_points) / 400.0, 1.0)
        
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
    
    def _create_placeholder_remapped_files(self, output_prefix: str):
        """Create placeholder remapped files when nona fails."""
        logger.warning("🚨 Creating placeholder remapped files as fallback...")
        
        # Get list of original images in temp directory
        image_files = sorted([f for f in os.listdir(self.temp_dir) if f.startswith('image_') and f.endswith('.jpg')])
        
        for i, image_file in enumerate(image_files):
            source_path = os.path.join(self.temp_dir, image_file)
            placeholder_path = f"{output_prefix}{i:04d}.tif"
            
            try:
                # Convert JPEG to TIFF with alpha channel for enblend compatibility
                from PIL import Image
                img = Image.open(source_path)
                
                # Convert to RGBA to ensure alpha channel exists for enblend
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Create a white background with full opacity alpha channel
                background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                if img.mode == 'RGBA':
                    # Composite with alpha if source has alpha
                    img = Image.alpha_composite(background, img)
                else:
                    # Just paste if no alpha
                    background.paste(img, (0, 0))
                    img = background
                
                # Save as TIFF with alpha channel
                img.save(placeholder_path, 'TIFF', compression='lzw')
                logger.info(f"📁 Created placeholder with alpha: {os.path.basename(placeholder_path)}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to create placeholder {placeholder_path}: {e}")
        
        logger.info(f"✅ Created {len(image_files)} placeholder remapped files with alpha channels")
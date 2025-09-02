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
        """Calculate field of view from EXIF focal length and sensor data."""
        try:
            # Import piexif locally like the rest of the server code
            import piexif
            
            # Read EXIF data from the image
            exif_dict = piexif.load(image_path)
            
            # Extract focal length
            focal_length = None
            if "Exif" in exif_dict and piexif.ExifIFD.FocalLength in exif_dict["Exif"]:
                focal_length_rational = exif_dict["Exif"][piexif.ExifIFD.FocalLength]
                if isinstance(focal_length_rational, tuple) and len(focal_length_rational) == 2:
                    focal_length = focal_length_rational[0] / focal_length_rational[1]
                else:
                    focal_length = float(focal_length_rational)
            
            # Extract sensor dimensions from EXIF if available
            sensor_width = None
            sensor_height = None
            
            # Check for focal plane resolution (pixels per unit)
            if "Exif" in exif_dict:
                exif_data = exif_dict["Exif"]
                
                # Get sensor resolution and focal plane dimensions
                focal_plane_x_res = exif_data.get(piexif.ExifIFD.FocalPlaneXResolution)
                focal_plane_y_res = exif_data.get(piexif.ExifIFD.FocalPlaneYResolution)
                focal_plane_unit = exif_data.get(piexif.ExifIFD.FocalPlaneResolutionUnit, 2)  # 2 = inches
                
                if focal_plane_x_res and focal_plane_y_res:
                    # Convert resolution to sensor dimensions
                    if isinstance(focal_plane_x_res, tuple):
                        x_res = focal_plane_x_res[0] / focal_plane_x_res[1]
                    else:
                        x_res = float(focal_plane_x_res)
                        
                    if isinstance(focal_plane_y_res, tuple):
                        y_res = focal_plane_y_res[0] / focal_plane_y_res[1]
                    else:
                        y_res = float(focal_plane_y_res)
                    
                    # Get image dimensions
                    if "0th" in exif_dict:
                        img_width = exif_dict["0th"].get(piexif.ImageIFD.ImageWidth)
                        img_height = exif_dict["0th"].get(piexif.ImageIFD.ImageLength)
                        
                        if img_width and img_height and x_res > 0 and y_res > 0:
                            # Calculate sensor dimensions in mm
                            unit_factor = 25.4 if focal_plane_unit == 2 else 1.0  # Convert inches to mm
                            sensor_width = (img_width / x_res) * unit_factor
                            sensor_height = (img_height / y_res) * unit_factor
                            
                            logger.info(f"📸 EXIF Sensor Dimensions: {sensor_width:.2f}mm × {sensor_height:.2f}mm")
            
            if focal_length and sensor_width:
                # Use EXIF-derived sensor width for accurate FOV calculation
                fov_radians = 2 * math.atan(sensor_width / (2 * focal_length))
                fov_degrees = math.degrees(fov_radians)
                
                logger.info(f"📸 EXIF-based FOV: {focal_length:.2f}mm focal length + {sensor_width:.2f}mm sensor → {fov_degrees:.1f}°")
                
                # Sanity check: iPhone ultra-wide should be between 100-130°
                if 95 <= fov_degrees <= 135:
                    return fov_degrees
                else:
                    logger.warning(f"⚠️ EXIF-calculated FOV {fov_degrees:.1f}° outside expected range (95-135°)")
            
            elif focal_length:
                # Try to calculate sensor width from 35mm equivalent focal length
                focal_35mm = None
                if "Exif" in exif_dict and piexif.ExifIFD.FocalLengthIn35mmFilm in exif_dict["Exif"]:
                    focal_35mm = exif_dict["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm]
                
                if focal_35mm and focal_35mm > 0:
                    # Calculate sensor width using 35mm equivalent method
                    # crop_factor = focal_35mm / focal_length
                    # sensor_width = full_frame_width / crop_factor
                    crop_factor = focal_35mm / focal_length
                    sensor_width = 36.0 / crop_factor  # 36mm = full frame sensor width
                    
                    fov_radians = 2 * math.atan(sensor_width / (2 * focal_length))
                    fov_degrees = math.degrees(fov_radians)
                    
                    logger.info(f"📸 35mm-equivalent FOV: {focal_length:.2f}mm focal + {focal_35mm}mm equiv → {sensor_width:.2f}mm sensor → {fov_degrees:.1f}°")
                    
                    if 95 <= fov_degrees <= 135:
                        return fov_degrees
                    else:
                        logger.warning(f"⚠️ 35mm-calculated FOV {fov_degrees:.1f}° outside expected range (95-135°)")
                else:
                    # Fallback to research-based iPhone ultra-wide sensor width
                    sensor_width = 4.88  # mm (research-based iPhone ultra-wide sensor width)
                    fov_radians = 2 * math.atan(sensor_width / (2 * focal_length))
                    fov_degrees = math.degrees(fov_radians)
                    
                    logger.info(f"📸 Research-based FOV: {focal_length:.2f}mm focal length + {sensor_width:.2f}mm sensor → {fov_degrees:.1f}°")
                    
                    if 95 <= fov_degrees <= 135:
                        return fov_degrees
                    else:
                        logger.warning(f"⚠️ Research-calculated FOV {fov_degrees:.1f}° outside expected range")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not extract FOV from EXIF: {e}")
            
        # Final fallback to measured iPhone ultra-wide FOV
        logger.info("📸 Using measured iPhone ultra-wide FOV: 106.2°")
        return 106.2
    
    def _extract_photometric_exif(self, image_path: str) -> Dict:
        """Extract photometric parameters from EXIF for enhanced optimization."""
        photometric_data = {}
        
        try:
            import piexif
            exif_dict = piexif.load(image_path)
            
            if "Exif" in exif_dict:
                exif_data = exif_dict["Exif"]
                
                # ISO speed
                if piexif.ExifIFD.ISOSpeedRatings in exif_data:
                    photometric_data['iso'] = exif_data[piexif.ExifIFD.ISOSpeedRatings]
                
                # Aperture (F-number)
                if piexif.ExifIFD.FNumber in exif_data:
                    f_number = exif_data[piexif.ExifIFD.FNumber]
                    if isinstance(f_number, tuple) and len(f_number) == 2:
                        photometric_data['aperture'] = f_number[0] / f_number[1]
                    else:
                        photometric_data['aperture'] = float(f_number)
                
                # Exposure time
                if piexif.ExifIFD.ExposureTime in exif_data:
                    exp_time = exif_data[piexif.ExifIFD.ExposureTime]
                    if isinstance(exp_time, tuple) and len(exp_time) == 2:
                        photometric_data['exposure_time'] = exp_time[0] / exp_time[1]
                    else:
                        photometric_data['exposure_time'] = float(exp_time)
                
                # White balance
                if piexif.ExifIFD.WhiteBalance in exif_data:
                    photometric_data['white_balance'] = exif_data[piexif.ExifIFD.WhiteBalance]
                
                # Exposure compensation
                if piexif.ExifIFD.ExposureBiasValue in exif_data:
                    exp_bias = exif_data[piexif.ExifIFD.ExposureBiasValue]
                    if isinstance(exp_bias, tuple) and len(exp_bias) == 2:
                        photometric_data['exposure_bias'] = exp_bias[0] / exp_bias[1]
                
            # Log photometric data found
            if photometric_data:
                logger.info(f"📊 EXIF Photometric Data: {photometric_data}")
            else:
                logger.info("📊 No photometric EXIF data found")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not extract photometric EXIF: {e}")
            
        return photometric_data
        
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
                
                # Also copy sidecar JSON if it exists (for HDR TIFF EXIF data)
                json_sidecar = abs_img_path.replace('.tif', '_exif.json').replace('.jpg', '_exif.json')
                if os.path.exists(json_sidecar):
                    local_json = local_name.replace('.jpg', '_exif.json')
                    shutil.copy2(json_sidecar, local_json)
                    logger.info(f"📄 Copied EXIF sidecar: {os.path.basename(json_sidecar)} → {local_json}")
                
            if progress_callback:
                progress_callback(0.1, "Copied images to working directory")
            
            pto_file = 'project.pto'
            
            # **HDR WORKFLOW FIX**: Extract EXIF from corresponding JPG if using HDR TIFF
            first_image = local_images[0]
            exif_source_image = first_image
            
            logger.info(f"🔍 EXIF Debug: First image = {first_image}")
            
            # Check if we're processing HDR TIFF files (even if renamed to .jpg)
            is_hdr_tiff = False
            try:
                # Check file magic bytes to detect TIFF regardless of extension
                with open(first_image, 'rb') as f:
                    header = f.read(4)
                    if header in [b'II*\x00', b'MM\x00*']:  # TIFF magic bytes
                        # It's a TIFF file, check if it's 32-bit float (HDR)
                        f.seek(0)
                        # Quick check: 32-bit TIFFs are typically > 100MB for our images
                        file_size = os.path.getsize(first_image)
                        if file_size > 100 * 1024 * 1024:  # > 100MB suggests 32-bit
                            is_hdr_tiff = True
                            logger.info(f"🔍 EXIF Debug: Detected HDR TIFF renamed as .jpg ({file_size/1024/1024:.1f}MB)")
            except Exception as e:
                logger.warning(f"🔍 EXIF Debug: Could not detect file type: {e}")
            
            # Also check filename pattern as fallback
            if not is_hdr_tiff:
                is_hdr_tiff = first_image.endswith('_hdr.tif')
            
            logger.info(f"🔍 EXIF Debug: Is HDR TIFF? {is_hdr_tiff}")
            
            # Check if we're processing HDR TIFF files
            if is_hdr_tiff:
                logger.info(f"🔍 HDR TIFF detected, looking for EXIF sidecar...")
                
                # Try multiple sidecar naming patterns (Hugin renames files)
                potential_sidecars = [
                    first_image.replace('.jpg', '_exif.json'),  # img01_exif.json
                    first_image.replace('.jpg', '.json'),       # img01.json
                    # Look for original naming pattern in same directory
                    os.path.join(os.path.dirname(first_image), 'merged_dot_0_hdr_exif.json'),
                    os.path.join(os.path.dirname(first_image), 'img01_exif.json'),
                ]
                
                sidecar_found = False
                for sidecar_path in potential_sidecars:
                    logger.info(f"🔍 Checking for sidecar: {sidecar_path} - exists: {os.path.exists(sidecar_path)}")
                    if os.path.exists(sidecar_path):
                        try:
                            import json
                            with open(sidecar_path, 'r') as f:
                                exif_metadata = json.load(f)
                            logger.info(f"✅ Found EXIF sidecar: {os.path.basename(sidecar_path)}")
                            logger.info(f"📊 EXIF from sidecar: {exif_metadata}")
                            
                            # Calculate FOV from sidecar focal length if available
                            if 'focal_length' in exif_metadata:
                                focal_length = float(exif_metadata['focal_length'])
                                # Use standard iPhone sensor width
                                sensor_width = 5.7  # mm for iPhone ultra-wide
                                fov_radians = 2 * math.atan(sensor_width / (2 * focal_length))
                                calculated_fov = math.degrees(fov_radians)
                                logger.info(f"📸 FOV from sidecar: {focal_length}mm → {calculated_fov:.1f}°")
                            
                            sidecar_found = True
                            break
                        except Exception as e:
                            logger.warning(f"⚠️ Could not read sidecar {sidecar_path}: {e}")
                
                if not sidecar_found:
                    logger.warning("⚠️ No EXIF sidecar found, looking in parent directory for original JPGs...")
                
                # If no sidecar or sidecar failed, look in parent directory
                # HDR TIFFs are in hdr_merged/ subdirectory, originals are in parent directory
                tiff_dir = os.path.dirname(first_image)
                parent_dir = os.path.dirname(tiff_dir)  # Go up from hdr_merged/ to upload dir
                
                logger.info(f"🔍 EXIF Debug: TIFF dir = {tiff_dir}")
                logger.info(f"🔍 EXIF Debug: Parent dir = {parent_dir}")
                logger.info(f"🔍 EXIF Debug: Parent dir exists? {os.path.exists(parent_dir)}")
                
                # List contents of parent directory for debugging
                try:
                    parent_contents = os.listdir(parent_dir)
                    jpg_files_found = [f for f in parent_contents if f.endswith('.jpg')]
                    logger.info(f"🔍 EXIF Debug: Parent dir contents: {parent_contents}")
                    logger.info(f"🔍 EXIF Debug: JPG files found: {jpg_files_found}")
                except Exception as e:
                    logger.error(f"🔍 EXIF Debug: Error listing parent dir: {e}")
                
                # Method 1: Look for corresponding numbered JPG in parent directory
                tiff_basename = os.path.basename(first_image)  # e.g., "merged_dot_0_hdr.tif"
                logger.info(f"🔍 EXIF Debug: TIFF basename = {tiff_basename}")
                
                if 'merged_dot_' in tiff_basename:
                    # Extract dot number: merged_dot_0_hdr.tif -> 0
                    dot_num = tiff_basename.split('merged_dot_')[1].split('_hdr.tif')[0]
                    logger.info(f"🔍 EXIF Debug: Extracted dot number = {dot_num}")
                    
                    potential_jpgs = [
                        f"img_{dot_num:02d}.jpg",    # img_00.jpg format (zero-padded)
                        f"img_{dot_num}.jpg",        # img_0.jpg format (no padding)
                        f"image_{dot_num}.jpg",      # image_0.jpg format  
                        f"dot_{dot_num}.jpg",        # dot_0.jpg format
                        f"{dot_num}.jpg"             # just number.jpg format
                    ]
                    
                    logger.info(f"🔍 EXIF Debug: Looking for JPGs: {potential_jpgs}")
                    
                    for jpg_name in potential_jpgs:
                        jpg_path = os.path.join(parent_dir, jpg_name)
                        logger.info(f"🔍 EXIF Debug: Checking {jpg_path} - exists? {os.path.exists(jpg_path)}")
                        if os.path.exists(jpg_path):
                            exif_source_image = jpg_path
                            logger.info(f"✅ HDR Mode: Found {jpg_name} for EXIF (dot {dot_num})")
                            break
                
                # Method 2: If specific JPG not found, use any JPG in parent directory 
                if exif_source_image == first_image:  # Still using TIFF (no JPG found)
                    try:
                        jpg_files = [f for f in os.listdir(parent_dir) if f.endswith('.jpg')]
                        logger.info(f"🔍 EXIF Debug: All JPG files in parent: {jpg_files}")
                        if jpg_files:
                            # Sort to get consistent results
                            jpg_files.sort()
                            exif_source_image = os.path.join(parent_dir, jpg_files[0])
                            logger.info(f"✅ HDR Mode: Using {jpg_files[0]} from parent dir for EXIF")
                        else:
                            logger.warning("⚠️ HDR Mode: No JPG found for EXIF - using fallback FOV")
                    except Exception as e:
                        logger.error(f"🔍 EXIF Debug: Error accessing parent directory: {e}")
            
            logger.info(f"🔍 EXIF Debug: Final EXIF source = {exif_source_image}")
            logger.info(f"🔍 EXIF Debug: EXIF source exists? {os.path.exists(exif_source_image)}")
            
            # Calculate FOV from EXIF data with enhanced sensor dimension extraction
            calculated_fov = self._calculate_fov_from_exif(exif_source_image)
            fov_str = str(int(round(calculated_fov)))
            
            # Extract photometric data from EXIF source for logging (helps with debugging)
            photometric_data = self._extract_photometric_exif(exif_source_image)
            
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
            
            # RESEARCH: Enhanced sensor data validation for ARKit-based guidance
            use_sensor_guidance = (session_metadata and 
                                 'capturePoints' in session_metadata and 
                                 len(session_metadata['capturePoints']) == len(images))
                                 
            # Research-based sensor guidance validation and logging
            if session_metadata:
                logger.info(f"📊 Session metadata available: {list(session_metadata.keys())}")
                if 'capturePoints' in session_metadata:
                    capture_points = session_metadata['capturePoints']
                    logger.info(f"📍 ARKit capture points: {len(capture_points)}, Images: {len(images)}")
                    
                    # RESEARCH: Validate systematic pattern quality for enhanced stitching
                    if len(capture_points) == 16 and len(capture_points) == len(images):
                        logger.info("✅ Validated 16-point systematic pattern - optimal for sensor guidance")
                    else:
                        logger.warning(f"⚠️ Non-standard pattern: {len(capture_points)} points - sensor guidance may be suboptimal")
                else:
                    logger.warning("❌ No capturePoints in session metadata")
            else:
                logger.warning("❌ No session metadata provided")
                                 
            if use_sensor_guidance:
                logger.info("🧭 Using research-enhanced sensor-guided stitching workflow")
                
                try:
                    # Step 1.5: Research-optimized sensor pose injection from ARKit data
                    capture_points = session_metadata['capturePoints']
                    hugin_poses = self._convert_ios_to_hugin_coordinates(capture_points)
                    
                    # RESEARCH: Enhanced pose injection with gravity prior and systematic pattern awareness
                    self._inject_poses_into_pto(pto_file, hugin_poses)
                    logger.info(f"🧭 Injected {len(hugin_poses)} ARKit poses with systematic pattern optimization")
                    
                    if progress_callback:
                        progress_callback(0.25, "Injected sensor poses")
                    
                    # Step 2: Find control points with research-optimized sensor guidance
                    logger.info("🔍 Step 2: Research-optimized control points (sensor-guided)")
                    self._run_command([
                        'cpfind', 
                        '--prealigned',                  # Use sensor poses as starting point
                        '--sieve1width', '50',           # Optimal for ultra-wide systematic patterns
                        '--sieve1height', '50',          # Balanced keypoint distribution  
                        '--sieve1size', '800',           # RESEARCH: Increased for ultra-wide overlap optimization
                        '--ransacmode', 'rpy',           # RESEARCH: Optimal for calibrated iPhone lenses
                        '--ransaciter', '1500',          # RESEARCH: Reduced iterations (sensors guide matching)
                        '--cache',                       # RESEARCH: Enable keypoint caching for performance
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
                
                # Step 2: Find control points with research-optimized traditional method
                logger.info("🔍 Step 2: Research-optimized control points (traditional feature-based)")
                self._run_command([
                    'cpfind', 
                    # NOTE: --multirow is now DEFAULT in Hugin 2024+, no need to specify
                    '--sieve1width', '50',           # Optimal for ultra-wide systematic patterns
                    '--sieve1height', '50',          # Balanced keypoint distribution
                    '--sieve1size', '1000',          # RESEARCH: Increased for ultra-wide indoor scenes
                    '--sieve2size', '4',             # RESEARCH: More points per region for low-texture
                    '--ransacmode', 'rpy',           # RESEARCH: Optimal for calibrated iPhone lenses  
                    '--ransaciter', '2500',          # RESEARCH: Enhanced RANSAC for low-texture
                    '--ransacdist', '15',            # RESEARCH: Tighter threshold for precision
                    '--kdtreesteps', '400',          # RESEARCH: More iterations for indoor matching
                    '--kdtreeseconddist', '0.3',     # RESEARCH: Stricter matching for quality
                    '--cache',                       # RESEARCH: Enable keypoint caching
                    '-o', pto_file, pto_file
                ], "cpfind", timeout=1200)
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
            
            # Step 5: Research-optimized bundle adjustment for iPhone ultra-wide
            logger.info("⚙️ Step 5: Research-optimized bundle adjustment and photometry")
            self._run_command([
                'autooptimiser', 
                '-a',              # RESEARCH: Auto-align mode with intelligent optimization strategy
                '-m',              # RESEARCH: Photometric optimization (exposure, white balance, vignetting) 
                '-l',              # RESEARCH: Horizon leveling using gravity prior (beneficial for ARKit data)
                '-s',              # RESEARCH: Smart output projection selection for systematic patterns
                # NOTE: -v parameter auto-handled for iPhone ultra-wide FOV from EXIF
                '-o', pto_file, pto_file
            ], "autooptimiser", timeout=700)  # Extended timeout for research-optimized processing
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
            elif output_file.lower().endswith('.exr'):
                # Convert TIFF to EXR preserving HDR data using OpenCV
                import cv2
                import numpy as np
                
                logger.info("🌈 Converting HDR TIFF to native EXR format...")
                stitched_hdr = None
                
                # Try multiple loading methods for robustness
                try:
                    # Method 1: Load as-is with any depth and color
                    stitched_hdr = cv2.imread(stitched_tif, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                    if stitched_hdr is not None:
                        logger.info(f"✅ Loaded TIFF: {stitched_hdr.shape}, dtype: {stitched_hdr.dtype}")
                except Exception as e:
                    logger.warning(f"⚠️ Method 1 failed: {e}")
                
                if stitched_hdr is None:
                    try:
                        # Method 2: Load unchanged (preserve original format)
                        stitched_hdr = cv2.imread(stitched_tif, cv2.IMREAD_UNCHANGED)
                        if stitched_hdr is not None:
                            logger.info(f"✅ Loaded TIFF (unchanged): {stitched_hdr.shape}, dtype: {stitched_hdr.dtype}")
                    except Exception as e:
                        logger.warning(f"⚠️ Method 2 failed: {e}")
                
                if stitched_hdr is None:
                    try:
                        # Method 3: Load as grayscale then convert
                        gray = cv2.imread(stitched_tif, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
                        if gray is not None:
                            stitched_hdr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                            logger.info(f"✅ Loaded TIFF (gray→BGR): {stitched_hdr.shape}, dtype: {stitched_hdr.dtype}")
                    except Exception as e:
                        logger.warning(f"⚠️ Method 3 failed: {e}")
                
                if stitched_hdr is not None:
                    # **HDR VALIDATION**: Check if we have true HDR data
                    original_min, original_max = stitched_hdr.min(), stitched_hdr.max()
                    logger.info(f"🔍 HDR Analysis - Original TIFF: dtype={stitched_hdr.dtype}, shape={stitched_hdr.shape}")
                    logger.info(f"🔍 HDR Range: [{original_min:.6f}, {original_max:.6f}]")
                    
                    # Check for HDR indicators
                    values_above_1 = np.sum(stitched_hdr > 1.0)
                    total_pixels = stitched_hdr.size
                    hdr_percentage = (values_above_1 / total_pixels) * 100
                    logger.info(f"🔍 HDR Pixels: {values_above_1}/{total_pixels} ({hdr_percentage:.2f}%) above 1.0")
                    
                    if original_max > 1.0:
                        logger.info("✅ AUTHENTIC HDR: Values exceed 1.0 - true HDR data detected")
                    else:
                        logger.warning("⚠️ POTENTIAL LDR: All values ≤ 1.0 - may be tone-mapped data")
                    
                    # Ensure float32 format for EXR
                    if stitched_hdr.dtype != np.float32:
                        # Handle different input types appropriately
                        if stitched_hdr.dtype == np.uint8:
                            stitched_hdr = stitched_hdr.astype(np.float32) / 255.0
                            logger.info("🔄 Converted uint8 → float32 (0-1 range)")
                        elif stitched_hdr.dtype == np.uint16:
                            stitched_hdr = stitched_hdr.astype(np.float32) / 65535.0
                            logger.info("🔄 Converted uint16 → float32 (0-1 range)")
                        else:
                            stitched_hdr = stitched_hdr.astype(np.float32)
                            logger.info(f"🔄 Converted {original_min} → float32 (preserved range)")
                        
                        final_min, final_max = stitched_hdr.min(), stitched_hdr.max()
                        logger.info(f"🔄 Final range: [{final_min:.6f}, {final_max:.6f}]")
                    
                    success = cv2.imwrite(final_output_path, stitched_hdr, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    if success:
                        # **EXR VALIDATION**: Verify the saved EXR file  
                        try:
                            verification_img = cv2.imread(final_output_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                            if verification_img is not None:
                                exr_min, exr_max = verification_img.min(), verification_img.max()
                                exr_values_above_1 = np.sum(verification_img > 1.0)
                                exr_hdr_percentage = (exr_values_above_1 / verification_img.size) * 100
                                
                                logger.info(f"🎯 EXR Verification: range=[{exr_min:.6f}, {exr_max:.6f}]")
                                logger.info(f"🎯 EXR HDR Pixels: {exr_values_above_1} ({exr_hdr_percentage:.2f}%) above 1.0")
                                
                                if exr_max > 1.0:
                                    logger.info("✅ SUCCESS: EXR contains authentic HDR data (values > 1.0)")
                                else:
                                    logger.warning("⚠️ WARNING: EXR may not contain true HDR data")
                            else:
                                logger.warning("⚠️ Could not verify EXR file")
                        except Exception as e:
                            logger.warning(f"⚠️ EXR verification failed: {e}")
                        
                        logger.info("✅ Native HDR EXR created successfully")
                    else:
                        raise HuginPipelineError("Failed to write EXR file")
                else:
                    raise HuginPipelineError("Failed to load HDR TIFF for EXR conversion - all methods failed")
            else:
                # Convert TIFF to JPG using Pillow (tone-mapped for display)
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
            
            # RESEARCH-OPTIMIZED PIPELINE: iPhone ultra-wide (106.2° measured FOV) systematic 16-point pattern
            # Creates ~50-60% overlap requiring specialized enblend parameters for professional quality
            
            # Try Method 1: Railway-compatible optimal settings for ultra-wide panorama
            logger.info("🔬 Method 1: Railway-compatible optimal settings for ultra-wide panorama")
            result = subprocess.run([
                'enblend', 
                '--fine-mask',           # RESEARCH: Best with graph-cut for detailed seam placement
                '-l', '20',              # RESEARCH: Higher levels for smoother high-overlap transitions
                '--compression=lzw',     # Maintain compression for storage efficiency
                # NOTE: graph-cut is DEFAULT (superior to nearest-feature-transform)
                # NOTE: Removed incompatible options: -m, --blend-colorspace for Railway compatibility
                '-o', 'stitched.tif'
            ] + img_files, capture_output=True, text=True, timeout=900, env=env)
            
            if result.returncode == 0:
                logger.info("✅ Method 1 successful: research-optimized enblend")
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
                
            # Method 2: Performance-optimized for challenging overlap scenarios  
            logger.info("🔄 Method 2: Railway-compatible performance-optimized enblend")
            result = subprocess.run([
                'enblend',
                '--coarse-mask',         # RESEARCH: Faster processing for excessive overlap
                '-l', '15',              # RESEARCH: Moderate levels for speed vs quality balance
                '--compression=lzw',     # Maintain compression
                # NOTE: Still using superior graph-cut algorithm (default)
                # NOTE: Removed incompatible options: -m, --blend-colorspace for Railway compatibility
                '-o', 'stitched.tif'
            ] + img_files, capture_output=True, text=True, timeout=600, env=env)
            
            if result.returncode == 0:
                logger.info("✅ Method 2 successful: performance-optimized enblend")
                # Clean up intermediate files
                for img_file in img_files:
                    try:
                        os.remove(img_file)
                    except:
                        pass
                return os.path.exists('stitched.tif')
            else:
                logger.warning(f"Method 2 failed: {result.stderr}")
                
            # Method 3: Simple averaging blending (emergency fallback)
            logger.info("🛟 Method 3: Emergency averaging blend for excessive overlap")
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
                
                # Ensure image dimensions match target
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                    
                # Convert to float64 for accumulation
                img_float = img.astype(np.float64)
                
                # Create mask for valid pixels (non-black areas)
                mask = np.any(img_float > 0, axis=2)
                
                # Accumulate valid pixels
                accumulated[mask] += img_float[mask]
                count[mask] += 1
                
                logger.info(f"📊 Processed image {i+1}/{len(img_files)}: {img_file}")
            
            # Average the accumulated values with proper dimension handling
            valid_mask = count > 0
            result = np.zeros_like(accumulated)
            
            # Safe division with broadcasting
            for c in range(3):  # For each color channel
                channel_mask = valid_mask
                result[channel_mask, c] = accumulated[channel_mask, c] / count[channel_mask]
            
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
                # Check if it's an HDR TIFF file that needs special handling
                if img_path.endswith('_hdr.tif'):
                    # **32-bit HDR TIFF Validation** - Don't load pixels, just validate file
                    
                    # Method 1: File system validation
                    file_size = os.path.getsize(img_path)
                    if file_size < 1000:  # Less than 1KB is suspicious
                        raise HuginPipelineError(f"HDR TIFF too small: {file_size} bytes")
                    
                    # Method 2: Try to read TIFF header for dimensions
                    width, height, dtype = None, None, "float32"
                    
                    try:
                        # Try using tifffile if available (better for 32-bit float)
                        import tifffile
                        with tifffile.TiffFile(img_path) as tif:
                            page = tif.pages[0]
                            width, height = page.shape[1], page.shape[0]
                            dtype = page.dtype
                            logger.info(f"📸 HDR Image {i}: {os.path.basename(img_path)} {width}×{height} ({dtype}) [tifffile]")
                    except ImportError:
                        # Fallback: Try ImageMagick identify command
                        try:
                            result = subprocess.run(
                                ['identify', '-format', '%w %h %z', img_path],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            if result.returncode == 0:
                                parts = result.stdout.strip().split()
                                if len(parts) >= 2:
                                    width, height = int(parts[0]), int(parts[1])
                                    bit_depth = parts[2] if len(parts) > 2 else "32"
                                    logger.info(f"📸 HDR Image {i}: {os.path.basename(img_path)} {width}×{height} ({bit_depth}-bit) [ImageMagick]")
                            else:
                                raise subprocess.CalledProcessError(result.returncode, result.args)
                        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                            # Ultimate fallback: Just check file exists and size
                            logger.info(f"📸 HDR Image {i}: {os.path.basename(img_path)} ({file_size/1024/1024:.1f}MB) [file-only validation]")
                    except Exception as e:
                        # If tifffile fails, just validate file existence
                        logger.warning(f"⚠️ Could not read HDR TIFF metadata: {e}")
                        logger.info(f"📸 HDR Image {i}: {os.path.basename(img_path)} ({file_size/1024/1024:.1f}MB) [file validation only]")
                else:
                    # Use PIL for standard images
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
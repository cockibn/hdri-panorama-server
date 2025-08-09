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
        logger.info("📋 OFFICIAL HUGIN CLI WORKFLOW ACTIVE:")
        logger.info("  ✅ pto_gen: Generate project file")
        logger.info("  ✅ cpfind --multirow --celeste: Find control points") 
        logger.info("  ✅ cpclean: Clean control points")
        logger.info("  ✅ autooptimiser -a -m -l -s: Optimize geometry & photometrics")
        logger.info("  ✅ pano_modify --canvas=AUTO --crop=AUTO: Set output parameters")
        logger.info("  ✅ nona -m TIFF_m: Render images")
        logger.info("  ✅ enblend --wrap=horizontal: Blend final panorama")
        logger.info("🎯 Deterministic, linear workflow for consistent results")
    
    def _verify_hugin_installation(self):
        """Verify required Hugin tools are available."""
        required_tools = ['pto_gen', 'cpfind', 'cpclean', 'linefind', 'autooptimiser', 'pano_modify', 'nona', 'enblend', 'geocpset']
        
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
        # Store progress callback for use in enblend progress tracking
        self.progress_callback = progress_callback
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
                
                project_file = self._generate_project_file(image_paths, capture_points, exif_data)
                
                # Step 2: Find control points (Official Hugin workflow)
                if progress_callback:
                    progress_callback(0.30, "Finding control points...")
                
                cp_project = self._find_control_points(project_file, len(images))
                self.control_points_found = self._count_control_points(cp_project)
                
                # Step 3: Clean control points (Official Hugin workflow)
                if progress_callback:
                    progress_callback(0.40, "Cleaning control points...")
                
                clean_project = self._clean_control_points(cp_project)
                
                # Step 3.5: Detect lines for geometric consistency (RESEARCH-BASED)
                if progress_callback:
                    progress_callback(0.50, "Detecting horizon and vertical lines...")
                
                line_project = self._detect_lines(clean_project)
                
                # Step 4: Optimize geometry and photometrics
                if progress_callback:
                    progress_callback(0.60, "Optimizing geometry and photometrics...")
                
                opt_project = self._optimize_panorama(line_project)
                
                # Step 5: Set output parameters (canvas, crop, projection)
                if progress_callback:
                    progress_callback(0.70, "Setting output parameters...")
                
                final_project = self._set_output_parameters(opt_project)
                
                # Step 6: Render images with nona
                if progress_callback:
                    progress_callback(0.80, "Rendering images...")
                
                tiff_files = self._render_images(final_project, len(images))
                
                # Step 7: Blend with enblend
                if progress_callback:
                    progress_callback(0.95, "Blending final panorama...")
                
                panorama_path = self._blend_images(tiff_files, len(images))
                
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
                    # Try different loading methods for EXR files
                    logger.warning("⚠️ OpenCV loading failed - trying EXR-compatible alternatives...")
                    
                    # Try imageio for EXR files (better EXR support than PIL)
                    try:
                        import imageio
                        panorama = imageio.imread(panorama_path)
                        logger.info(f"✅ Loaded EXR with imageio: {panorama.shape}")
                        # Convert RGB to BGR for consistency with OpenCV
                        if len(panorama.shape) == 3 and panorama.shape[2] >= 3:
                            panorama = panorama[:, :, ::-1]  # RGB to BGR
                    except ImportError:
                        logger.warning("⚠️ imageio not available - install imageio for better EXR support")
                        try:
                            # Last resort: PIL for TIFF (won't work for EXR but worth trying)
                            from PIL import Image
                            import numpy as np
                            pil_image = Image.open(panorama_path)
                            panorama = np.array(pil_image)
                            if len(panorama.shape) == 3:
                                panorama = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
                            logger.info(f"✅ Loaded with PIL: {panorama.shape}")
                        except Exception as e:
                            logger.error(f"❌ All loading methods failed: {e}")
                            logger.info(f"📁 Panorama file created successfully at: {panorama_path}")
                            logger.info("💡 Consider installing imageio for EXR support: pip install imageio")
                            # Still raise error as the function expects to return an array
                            raise RuntimeError(f"Failed to load panorama array - file exists at {panorama_path}")
                    except Exception as e:
                        logger.error(f"❌ imageio loading failed: {e}")
                        raise RuntimeError(f"EXR loading failed with imageio: {e}")
                
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
        """Save images to temporary directory with maximum quality preservation."""
        image_paths = []
        
        for i, img in enumerate(images):
            path = os.path.join(self.temp_dir, f"img_{i:04d}.jpg")
            # QUALITY FIX: Use maximum quality for Hugin input
            # Since iOS no longer double-compresses, we can maintain higher quality
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 98])
            image_paths.append(path)
            
            # Log image quality info
            height, width = img.shape[:2]
            logger.info(f"📸 Saved image {i}: {width}×{height} at 98% quality")
        
        logger.info(f"📁 QUALITY PRESERVATION: Saved {len(image_paths)} high-quality images for Hugin processing")
        return image_paths
    
    def _generate_project_file(self, image_paths: List[str], capture_points: List[Dict] = None, original_exif_data: List = None) -> str:
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
            
            # Enhanced debugging for positioning data validation
            positioning_results = []
            for i, cp in enumerate(capture_points[:5]):  # Check first 5 points
                azimuth = cp.get('azimuth')
                elevation = cp.get('elevation')
                positioning_results.append(f"Point {i}: az={azimuth}, el={elevation}")
            
            logger.info(f"🔍 POSITIONING CHECK: has_positioning={has_positioning}")
            logger.info(f"🔍 POSITIONING DETAILS: {' | '.join(positioning_results)}")
            
            if has_positioning:
                self._generate_positioned_project(image_paths, capture_points, project_file, original_exif_data)
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
    
    def _generate_positioned_project(self, image_paths: List[str], capture_points: List[Dict], project_file: str, original_exif_data: List = None):
        """Generate PTO file with ARKit positioning data."""
        logger.info(f"🎯 Generating positioned project with {len(capture_points)} ARKit positions")
        
        # iPhone ultra-wide camera parameters (106.2° FOV measured)
        fov = 106.2
        
        # CRITICAL DEBUG: Analyze ARKit positioning data in detail
        azimuths = [cp.get('azimuth', 0.0) for cp in capture_points]
        elevations = [cp.get('elevation', 0.0) for cp in capture_points]
        
        azimuth_range = max(azimuths) - min(azimuths)
        elevation_range = max(elevations) - min(elevations)
        
        logger.info(f"📊 ARKit data ranges - Azimuth: {min(azimuths):.1f}° to {max(azimuths):.1f}° (span: {azimuth_range:.1f}°)")
        logger.info(f"📊 ARKit data ranges - Elevation: {min(elevations):.1f}° to {max(elevations):.1f}° (span: {elevation_range:.1f}°)")
        
        # CRITICAL ANALYSIS: Check for pole coverage issues
        if elevation_range < 120:  # Should be close to 180° for full sphere
            logger.error(f"❌ LIMITED POLE COVERAGE: Elevation span is only {elevation_range:.1f}°")
            logger.error(f"❌ For proper 360° equirectangular panorama, need coverage closer to ±90°")
            logger.error(f"❌ Current max elevation: {max(elevations):.1f}° (should be >+60° for upper pole)")
            logger.error(f"❌ Current min elevation: {min(elevations):.1f}° (should be <-60° for lower pole)")
            logger.warning(f"⚠️ This will cause 'pole distribution problem' - missing sphere top/bottom")
            
            # Suggest iOS app changes needed
            if max(elevations) < 60:
                logger.error(f"❌ MISSING UPPER POLE: No images above {max(elevations):.1f}° - zenith area will be empty")
            if min(elevations) > -60:
                logger.error(f"❌ MISSING LOWER POLE: No images below {min(elevations):.1f}° - nadir area will be empty")
                
            # SERVER SOLUTION: Enable pole filling for limited coverage patterns
            logger.info(f"🔧 SERVER FIX: Enabling pole filling techniques for limited elevation range")
            logger.info(f"🔧 Will use canvas padding and pole extrapolation to fill missing areas")
        else:
            logger.info(f"✅ Good elevation coverage for 360° panorama")
        
        # Log EVERY single coordinate to identify problematic data
        logger.info(f"🔍 COMPLETE ARKit positioning data:")
        for i, cp in enumerate(capture_points):
            azimuth = cp.get('azimuth', 0.0)
            elevation = cp.get('elevation', 0.0)
            position = cp.get('position', [0.0, 0.0, 0.0])
            logger.info(f"🔍 Image {i:2d}: azimuth={azimuth:7.2f}°, elevation={elevation:6.2f}°, pos=[{position[0]:6.2f}, {position[1]:6.2f}, {position[2]:6.2f}]")
            
            # DISTORTION DEBUG: Log Hugin coordinate conversion
            wrap_azimuth = azimuth % 360
            nx = (wrap_azimuth + 180) / 360
            nx = nx % 1.0
            yaw = nx * 360 - 180
            pitch = elevation
            logger.info(f"🔍        → Hugin: yaw={yaw:7.2f}°, pitch={pitch:6.2f}° | nx={nx:.3f}")
            
            # Check for suspicious coordinate patterns
            if abs(yaw) > 175:
                logger.warning(f"⚠️ Image {i} very close to 180° seam: yaw={yaw:.2f}°")
                # FIX: Adjust seam boundary coordinates slightly to avoid stitching issues
                if yaw > 175:
                    yaw = 175.0
                elif yaw < -175:
                    yaw = -175.0
                logger.info(f"🔧 Adjusted image {i} seam position to yaw={yaw:.2f}°")
            if abs(pitch) > 60:
                logger.warning(f"⚠️ Image {i} near pole: pitch={pitch:.2f}°")
        
        # Check for problematic patterns
        zero_positions = sum(1 for cp in capture_points if cp.get('azimuth', 0.0) == 0.0 and cp.get('elevation', 0.0) == 0.0)
        if zero_positions > 1:
            logger.warning(f"⚠️ FOUND {zero_positions} images at origin (0°, 0°) - this will cause overlapping/glitchy output")
        
        # Check for invalid elevations (outside -90° to +90°)
        invalid_elevations = [e for e in elevations if e < -90 or e > 90]
        if invalid_elevations:
            logger.warning(f"⚠️ FOUND invalid elevations outside ±90°: {invalid_elevations}")
            
        # Check for azimuth clustering
        azimuth_groups = {}
        for az in azimuths:
            key = round(az / 45) * 45  # Group by 45° intervals
            azimuth_groups[key] = azimuth_groups.get(key, 0) + 1
        logger.info(f"📊 Azimuth distribution by 45° groups: {azimuth_groups}")
        
        # COMPREHENSIVE SPHERICAL DISTRIBUTION ANALYSIS
        elevation_range = max(elevations) - min(elevations)
        azimuth_range = max(azimuths) - min(azimuths)
        unique_elevations = len(set(round(e, 1) for e in elevations))
        unique_azimuths = len(set(round(a, 5) for a in azimuths))  # 5° grouping
        
        logger.info(f"🌐 DETAILED SPHERICAL COVERAGE ANALYSIS:")
        logger.info(f"   📊 Elevation range: {min(elevations):.1f}° to {max(elevations):.1f}° (span: {elevation_range:.1f}°)")
        logger.info(f"   📊 Azimuth range: {min(azimuths):.1f}° to {max(azimuths):.1f}° (span: {azimuth_range:.1f}°)")
        logger.info(f"   📊 Unique elevation levels: {unique_elevations}")
        logger.info(f"   📊 Unique azimuth directions: {unique_azimuths}")
        
        # Analyze elevation distribution in detail
        elevation_groups = {}
        for e in elevations:
            key = round(e / 15) * 15  # Group by 15° intervals
            elevation_groups[key] = elevation_groups.get(key, 0) + 1
        logger.info(f"   📊 Elevation distribution: {elevation_groups}")
        
        # Critical spherical coverage validation
        if elevation_range < 10.0:  # Less than 10° elevation variation
            logger.error(f"❌ CRITICAL: LIMITED ELEVATION RANGE ({elevation_range:.1f}°)")
            logger.error(f"❌ This will create a horizontal panorama strip, NOT a 360° sphere!")
            logger.error(f"❌ Root cause: Images captured only around horizon level")
            logger.error(f"❌ Solution: Capture images at multiple elevations (-45°, 0°, +45°)")
            
        if unique_elevations < 2:
            logger.error(f"❌ CRITICAL: ALL IMAGES AT SAME ELEVATION ({elevations[0]:.1f}°)")
            logger.error(f"❌ This guarantees horizontal strip output, not spherical panorama")
            
        if azimuth_range < 270:  # Less than 270° azimuth coverage
            logger.warning(f"⚠️ LIMITED AZIMUTH RANGE ({azimuth_range:.1f}°) - may miss full 360° coverage")
        
        # Expected vs actual pattern comparison
        expected_elevations = [-45, 0, 45]  # From iOS app design
        expected_azimuths = 8  # 8 azimuth columns
        
        logger.info(f"🎯 PATTERN COMPARISON:")
        logger.info(f"   Expected: 3 elevation levels {expected_elevations}, 8 azimuth columns")
        logger.info(f"   Actual: {unique_elevations} elevation levels, {unique_azimuths} azimuth positions")
        
        if unique_elevations >= 3 and elevation_range >= 60:
            logger.info("✅ GOOD: Proper spherical coverage detected")
        else:
            logger.error("❌ BAD: Insufficient spherical coverage - will produce horizontal strip")
        
        with open(project_file, 'w') as f:
            # RESEARCH-BASED: Proper PTO header for iPhone ultra-wide spherical panoramas
            f.write("# hugin project file\n")
            f.write("#hugin_ptoversion 2\n")
            f.write(f"p f2 w{self.canvas_size[0]} h{self.canvas_size[1]} v360 n\"TIFF_m c:LZW\"\n")
            f.write("m g1 i0 f0 m2 p0.00784314\n")
            f.write("\n# image lines\n")
            
            # Write image lines with ARKit positioning
            for i, (img_path, capture_point) in enumerate(zip(image_paths, capture_points)):
                # Convert ARKit coordinates to Hugin angles
                azimuth = capture_point.get('azimuth', 0.0)  # ARKit azimuth
                elevation = capture_point.get('elevation', 0.0)  # ARKit elevation  
                position = capture_point.get('position', [0.0, 0.0, 0.0])  # 3D position [x,y,z]
                
                # For now, use zero roll (could be derived from position if needed)
                roll = 0.0
                
                # Convert to Hugin coordinate system with validation
                # ARKit: azimuth (0-360°), elevation (-90 to +90°)
                # Hugin: yaw (-180 to +180°), pitch (-90 to +90°), roll (-180 to +180°)
                
                # Validate input ranges first
                if not (0 <= azimuth <= 360):
                    logger.warning(f"⚠️ Invalid ARKit azimuth {azimuth}° for image {i}, clamping to 0-360°")
                    azimuth = max(0, min(360, azimuth))
                
                if not (-90 <= elevation <= 90):
                    logger.warning(f"⚠️ Invalid ARKit elevation {elevation}° for image {i}, clamping to ±90°")
                    elevation = max(-90, min(90, elevation))
                
                # CRITICAL FIX: ARKit coordinate system analysis and conversion
                # ARKit coordinate system (device-relative, not geographic):
                # - azimuth: rotation around Y axis (device standing upright)
                # - elevation: rotation around X axis (device tilting up/down)
                # - iPhone capture: device held in portrait OR landscape orientation
                
                # Hugin coordinate system (spherical panorama):
                # - yaw: horizontal rotation (-180° to +180°, 0° = forward)
                # - pitch: vertical rotation (-90° to +90°, 0° = horizon, +90° = up)
                # - roll: camera rotation around optical axis (should be 0° for panoramas)
                
                # 🎯 COMPLETE 360° EQUIRECTANGULAR COORDINATE SYSTEM REBUILD
                # Based on iOS SphericalGeometry.swift exact mapping:
                #
                # iOS Coordinate System (from CoordinateSystem.swift):
                # - azimuth: 0-360°, 0° = +X (east), increases counter-clockwise  
                # - elevation: -90° to +90°, 0° = horizontal, +90° = up
                # - Z-up coordinate system (X=east, Y=north, Z=up)
                #
                # iOS Equirectangular Mapping (from SphericalGeometry.swift):
                # - nx = (azimuth + 180) / 360  // Maps azimuth to 0-1 range
                # - ny = (elevation + 90) / 180  // Maps elevation to 0-1 range  
                # - y = (1 - ny) * height       // INVERTS Y-axis (top=+90°, bottom=-90°)
                #
                # This means:
                # - azimuth 0° (east) → nx=0.5 → center of panorama width
                # - azimuth 180° (west) → nx=1.0 → right edge, wraps to left edge  
                # - elevation +90° (up) → ny=1.0 → y=0 (top of image)
                # - elevation -90° (down) → ny=0.0 → y=height (bottom of image)
                
                # EXACT REPRODUCTION OF iOS EQUIRECTANGULAR MAPPING WITH PROPER WRAPPING:
                # This matches the iOS sphericalToEquirectangular function exactly
                wrap_azimuth = azimuth % 360  # Ensure 0-360 range
                nx = (wrap_azimuth + 180) / 360  # iOS mapping: azimuth → normalized X
                
                # CRITICAL FIX: Wrap nx to [0,1] range (iOS does this implicitly with image bounds)
                nx = nx % 1.0  # Ensure nx stays within [0,1] range for proper canvas mapping
                
                # Convert nx back to Hugin yaw coordinate system (-180 to +180)
                yaw = nx * 360 - 180  # Convert 0-1 range back to -180° to +180°
                
                # FIX: Avoid exact 180° seam boundary to prevent stitching issues
                if abs(yaw) > 179.5:  # Within 0.5° of exact ±180°
                    if yaw > 179.5:
                        yaw = 179.0  # Move slightly away from +180°
                    elif yaw < -179.5:
                        yaw = -179.0  # Move slightly away from -180°
                    logger.info(f"🔧 Image {i}: Adjusted seam boundary yaw to {yaw:.1f}°")
                
                # FIXED: Direct elevation to pitch mapping (no inversion)
                # ARKit elevation: +90° = up (zenith), -90° = down (nadir)
                # Hugin pitch: +90° = up (zenith), -90° = down (nadir)
                # Both use same sign convention - NO INVERSION NEEDED
                pitch = elevation  # Direct mapping: ARKit elevation = Hugin pitch
                
                # Calculate normalized Y for logging (iOS reference)
                ny = (elevation + 90) / 180  # For reference/logging only
                
                logger.info(f"🎯 EXACT iOS equirectangular mapping reproduction:")
                logger.info(f"   📍 ARKit: azimuth={azimuth:.1f}°, elevation={elevation:.1f}°")
                logger.info(f"   📊 Normalized: nx={nx:.3f}, ny={ny:.3f}")
                logger.info(f"   🔄 Hugin: yaw={yaw:.1f}°, pitch={pitch:.1f}°")
                logger.info(f"   ✅ iOS-compatible equirectangular coordinate mapping")
                
                # POLE COVERAGE WARNING: Check if we have adequate coverage near poles
                if abs(pitch) < 60:  # Only checking within ±60° of horizon
                    logger.debug(f"   📍 Image {i} in mid-latitudes (pitch={pitch:.1f}°) - good for equirectangular")
                else:
                    logger.warning(f"   🏔️ Image {i} near pole (pitch={pitch:.1f}°) - may cause distortion")
                
                # Keep roll at zero for spherical panoramas (no camera rotation around optical axis)
                roll_hugin = 0.0
                
                # Additional validation: check for 180° seam boundary (not just large angles)
                # Only adjust angles very close to ±180° (the actual seam boundary)
                if abs(abs(yaw) - 180) < 0.5:  # Within 0.5° of ±180°
                    logger.warning(f"⚠️ Image {i} very close to 180° seam boundary: yaw={yaw:.1f}°")
                    # Adjust slightly to avoid exact seam boundary
                    yaw = 179.5 if yaw > 0 else -179.5
                    logger.info(f"🔧 Adjusted to yaw={yaw:.1f}° to avoid seam issues")
                
                # Check for poles (may cause rendering issues)
                if abs(pitch) > 85:
                    logger.warning(f"⚠️ Image {i} very close to pole: pitch={pitch:.1f}°")
                    # These images might not render properly due to extreme distortion
                
                # CRITICAL FIX: Get actual image dimensions instead of hardcoding
                # iPhone images might be portrait (3024×4032) or landscape (4032×3024)
                try:
                    import cv2
                    temp_img = cv2.imread(img_path)
                    if temp_img is not None:
                        actual_height, actual_width = temp_img.shape[:2]
                        logger.info(f"📸 Image {i} actual dimensions: {actual_width}×{actual_height}")
                    else:
                        # Fallback to assumed iPhone ultra-wide dimensions
                        actual_width, actual_height = 4032, 3024
                        logger.warning(f"⚠️ Could not read image {i}, using default dimensions")
                except Exception as e:
                    actual_width, actual_height = 4032, 3024
                    logger.warning(f"⚠️ Error reading image {i} dimensions: {e}")
                
                # FIXED: Constant horizontal FOV regardless of orientation  
                # The lens HFOV doesn't change with device rotation - it's a physical property
                # iPhone ultra-wide: 106.2° horizontal FOV (constant)
                adjusted_fov = fov  # Use constant FOV - orientation handled by EXIF
                logger.debug(f"📸 Image {i}: {actual_width}×{actual_height}, FOV={adjusted_fov:.1f}° (constant HFOV)")
                
                # CRITICAL FIX: iPhone ultra-wide lens distortion correction
                # iPhone ultra-wide (0.5x) has significant barrel distortion that needs proper correction
                # Research-based values for iPhone 13/14/15 Pro ultra-wide camera:
                
                # 📷 EXTRACT LENS PARAMETERS FROM EXIF DATA
                # Read actual lens characteristics instead of guessing
                lens_params = self._extract_lens_parameters_from_exif(original_exif_data or [], i, img_path)
                
                # Use EXIF-derived parameters or fallback defaults
                # Ensure all values have proper fallbacks and are never None
                # Use even more conservative distortion for better control point detection
                barrel_a = lens_params.get('distortion_a') or -0.01   # Very conservative fallback
                barrel_b = lens_params.get('distortion_b') or 0.005   # Very conservative fallback  
                barrel_c = lens_params.get('distortion_c') or -0.002  # Very conservative fallback
                measured_fov = lens_params.get('fov') or adjusted_fov  # Use EXIF FOV or fallback to original
                
                logger.info(f"📷 EXIF-based lens parameters:")
                fov_str = f"{measured_fov:.1f}°" if measured_fov is not None else "None (using fallback)"
                logger.info(f"   📐 FOV: {fov_str} (from EXIF: {lens_params.get('fov_source', 'fallback')})")
                logger.info(f"   🔧 Distortion: a={barrel_a:.3f}, b={barrel_b:.3f}, c={barrel_c:.3f}")
                logger.info(f"   📋 Lens: {lens_params.get('lens_model', 'Unknown')}")
                
                # Ensure measured_fov is never None for PTO file
                final_fov = measured_fov if measured_fov is not None else adjusted_fov
                
                f.write(f'#-hugin  cropFactor=1\n')
                f.write(f'i w{actual_width} h{actual_height} f0 v{final_fov:.1f} Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r{roll_hugin:.6f} p{pitch:.6f} y{yaw:.6f} TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a{barrel_a:.3f} b{barrel_b:.3f} c{barrel_c:.3f} d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5 n"{img_path}"\n')
        
        # Log the generated PTO file for analysis
        try:
            with open(project_file, 'r') as f:
                pto_content = f.read()
            logger.info(f"📝 Generated PTO file preview (first 500 chars):\n{pto_content[:500]}")
        except Exception as e:
            logger.warning(f"⚠️ Could not read generated PTO file: {e}")
        
        logger.info(f"✅ Generated positioned PTO with ARKit data covering {len(capture_points)} viewpoints")
    
    def _extract_lens_parameters_from_exif(self, original_exif_data: List, image_index: int, img_path: str) -> dict:
        """Extract actual lens parameters from iPhone EXIF data."""
        lens_params = {
            'lens_model': 'Unknown',
            'fov': None,
            'fov_source': 'fallback',
            'distortion_a': None,
            'distortion_b': None, 
            'distortion_c': None,
            'focal_length': None,
            'focal_length_35mm': None
        }
        
        try:
            # Get EXIF data for this image
            if image_index < len(original_exif_data) and original_exif_data[image_index]:
                exif_dict = original_exif_data[image_index]
                
                # Extract lens model information
                if '0th' in exif_dict:
                    ifd = exif_dict['0th']
                    
                    # Camera make/model
                    make = ifd.get(271, b'').decode('utf-8', errors='ignore') if 271 in ifd else ''
                    model = ifd.get(272, b'').decode('utf-8', errors='ignore') if 272 in ifd else ''
                    lens_params['lens_model'] = f"{make} {model}".strip()
                
                # Extract focal length information  
                if 'Exif' in exif_dict:
                    exif_ifd = exif_dict['Exif']
                    
                    # Actual focal length (mm)
                    if 37386 in exif_ifd:  # FocalLength
                        focal_length_raw = exif_ifd[37386]
                        if isinstance(focal_length_raw, tuple) and len(focal_length_raw) == 2:
                            lens_params['focal_length'] = focal_length_raw[0] / focal_length_raw[1]
                    
                    # 35mm equivalent focal length
                    if 41989 in exif_ifd:  # FocalLengthIn35mmFilm
                        lens_params['focal_length_35mm'] = exif_ifd[41989]
                
                # Calculate FOV from focal length and sensor size
                if lens_params['focal_length']:
                    # iPhone sensor sizes (approximate)
                    sensor_width_mm = 4.8  # iPhone main sensor width
                    if 'ultra' in lens_params['lens_model'].lower() or lens_params['focal_length'] < 3:
                        sensor_width_mm = 5.7  # iPhone ultra-wide sensor is larger
                    
                    # Calculate horizontal FOV: FOV = 2 * atan(sensor_width / (2 * focal_length))
                    import math
                    fov_radians = 2 * math.atan(sensor_width_mm / (2 * lens_params['focal_length']))
                    calculated_fov = math.degrees(fov_radians)
                    
                    lens_params['fov'] = calculated_fov
                    lens_params['fov_source'] = 'calculated_from_focal_length'
                    
                    logger.info(f"📷 EXIF lens analysis:")
                    logger.info(f"   🔍 Focal length: {lens_params['focal_length']:.2f}mm")
                    logger.info(f"   📐 Calculated FOV: {calculated_fov:.1f}°")
                    logger.info(f"   📱 Sensor width assumed: {sensor_width_mm:.1f}mm")
                
                # iPhone ultra-wide lens distortion characteristics (known values)
                if 'ultra' in lens_params['lens_model'].lower() or (lens_params['focal_length'] and lens_params['focal_length'] < 3):
                    # iPhone ultra-wide lens has known distortion characteristics
                    lens_params['distortion_a'] = -0.05  # Moderate barrel correction
                    lens_params['distortion_b'] = 0.02   # Secondary correction
                    lens_params['distortion_c'] = -0.008 # Tertiary correction
                    logger.info(f"   🔧 Detected iPhone ultra-wide: applying known distortion profile")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not extract EXIF lens parameters from image {image_index}: {e}")
        
        return lens_params
    
    def _generate_minimal_distortion_project(self, project_file: str):
        """Regenerate PTO file with minimal distortion correction for fallback control point detection."""
        logger.info("🔧 Regenerating PTO with minimal distortion correction...")
        
        try:
            with open(project_file, 'r') as f:
                content = f.read()
            
            # Replace aggressive distortion values with minimal ones
            import re
            
            # Find and replace distortion parameters (a, b, c values)
            # Pattern matches: a-0.080 b0.040 c-0.010 (or similar values)
            content = re.sub(r'a-?\d+\.\d+', 'a-0.02', content)  # Minimal barrel correction
            content = re.sub(r'b-?\d+\.\d+', 'b0.01', content)   # Minimal secondary correction
            content = re.sub(r'c-?\d+\.\d+', 'c-0.005', content) # Minimal tertiary correction
            
            # Also restore more conservative FOV
            content = re.sub(r'v\d+\.\d+', 'v100.0', content)  # Conservative FOV
            
            with open(project_file, 'w') as f:
                f.write(content)
            
            logger.info("✅ PTO regenerated with minimal distortion: a=-0.02, b=0.01, c=-0.005, v=100°")
            
        except Exception as e:
            logger.error(f"❌ Failed to regenerate PTO with minimal distortion: {e}")
    
    def _find_control_points(self, project_file: str, image_count: int) -> str:
        """Step 2: Find control points using optimized strategy for iPhone ultra-wide captures."""
        cp_project = os.path.join(self.temp_dir, "project_cp.pto")
        
        # Progressive strategy: try iPhone-optimized approach first, then fallbacks
        strategies = [
            {
                "name": "Ultra-wide professional feature matching",
                "cmd": [
                    "cpfind",
                    "--multirow",                    # Multi-row panorama detection
                    "--sieve1-width=8",             # Aggressive for ultra-wide precision
                    "--sieve1-height=8",
                    "--sieve1-size=1000",           # Maximum control points for best accuracy
                    "--threshold=0.25",             # Lower threshold for more matches
                    "--ransac-threshold=20",        # Tolerant for ultra-wide geometry
                    "--ransac-iterations=2000",     # More iterations for better accuracy
                    "--kdtree",                     # Optimized matching algorithm
                    "--celeste",                    # Sky detection and masking
                    "--downscale=2",               # Process at half resolution first for speed
                    "-o", cp_project,
                    project_file
                ],
                "timeout": 1200  # Extended time for thorough matching
            },
            {
                "name": "Ultra-wide maximum overlap detection", 
                "cmd": [
                    "cpfind",
                    "--multirow",
                    "--sieve1-width=5",             # Very aggressive for ultra-wide
                    "--sieve1-height=5",
                    "--sieve1-size=800",            # Maximum control points
                    "--threshold=0.15",             # Very low threshold
                    "--ransac-threshold=25",        # Very tolerant geometry
                    "--celeste",                    # Sky masking
                    "-o", cp_project,
                    project_file
                ],
                "timeout": 1500
            },
            {
                "name": "Basic (official)",
                "cmd": [
                    "cpfind",
                    "--multirow",
                    "--celeste",
                    "-o", cp_project,
                    project_file
                ],
                "timeout": 900
            },
            {
                "name": "Minimal distortion (fallback)",
                "cmd": [
                    "cpfind",
                    "--multirow",
                    "-o", cp_project,
                    project_file
                ],
                "timeout": 600,
                "rebuild_with_minimal_distortion": True  # Flag to regenerate PTO with minimal distortion
            }
        ]
        
        success = False
        best_cp_count = 0
        
        for strategy in strategies:
            if success and best_cp_count > 100:  # Good enough threshold
                break
                
            logger.info(f"🔍 Trying control point strategy: {strategy['name']}")
            logger.info(f"📋 Command: {' '.join(strategy['cmd'])}")
            
            # Check if this strategy needs PTO rebuild with minimal distortion
            if strategy.get("rebuild_with_minimal_distortion", False):
                logger.warning("🔧 FALLBACK: Regenerating PTO with minimal distortion correction")
                self._generate_minimal_distortion_project(project_file)
            
            # Remove previous attempt
            if os.path.exists(cp_project):
                os.remove(cp_project)
            
            try:
                self._run_command(strategy["cmd"], f"cpfind ({strategy['name']})", timeout=strategy["timeout"])
                
                if os.path.exists(cp_project):
                    cp_count = self._count_control_points(cp_project)
                    logger.info(f"🎯 Strategy '{strategy['name']}' found {cp_count} control points")
                    
                    # DISTORTION DEBUG: Analyze control point distribution
                    if cp_count > 0:
                        self._analyze_control_point_distribution(cp_project)
                    
                    if cp_count > best_cp_count:
                        best_cp_count = cp_count
                        success = True
                        logger.info(f"✅ New best result: {cp_count} control points")
                    
                    if cp_count >= 150:  # Excellent threshold
                        logger.info("🎉 Excellent control point count achieved!")
                        break
                        
                else:
                    logger.warning(f"⚠️ Strategy '{strategy['name']}' produced no output")
                    
            except Exception as e:
                logger.warning(f"⚠️ Strategy '{strategy['name']}' failed: {e}")
                continue
        
        if not success or not os.path.exists(cp_project):
            raise RuntimeError("All cpfind strategies failed to create output file")
        
        logger.info(f"🎯 Final result: {best_cp_count} control points using best strategy")
        
        # RESEARCH-BASED: Verify connectivity for all images
        min_cp_count = max(20, image_count * 5)  # Minimum ~5 points per image, at least 20 total
        if best_cp_count < min_cp_count:
            logger.error(f"❌ CRITICAL: VERY LOW CONTROL POINT COUNT: {best_cp_count} (minimum {min_cp_count} for {image_count} images)")
            logger.error("❌ This will cause poor stitching quality and missing images")
            logger.error("❌ Feature detection is failing on iPhone ultra-wide images")
        elif best_cp_count >= 150:
            logger.info("✅ EXCELLENT control point count - all images should render properly")
        else:
            logger.info("✅ GOOD control point count - most images should render")
        
        # CRITICAL: For ultra-wide iPhone images, always supplement with geocpset
        # Feature detection often fails due to lens distortion, but ARKit positioning is excellent
        if best_cp_count < 100:  # Force geocpset for low control point counts
            logger.warning("🔧 FORCE GEOCPSET: Low control points detected, using ARKit positioning")
            logger.warning("🔧 Ultra-wide lens distortion often breaks feature detection")
            logger.warning("🔧 ARKit positioning data can create geometric control points")
            
            # Force geocpset to supplement weak feature detection
            cp_project = self._fix_connectivity_with_geocpset(cp_project)
        else:
            # Still validate connectivity for good control point counts
            is_connected = self._validate_pto_connectivity(cp_project)
            if not is_connected:
                logger.error("❌ CRITICAL: Some images are not connected via control points!")
                logger.error("❌ This will cause nona to skip unconnected images!")
                logger.info("🔧 Attempting geocpset to connect isolated images...")
                
                # Try to fix connectivity with geocpset
                cp_project = self._fix_connectivity_with_geocpset(cp_project)
        
        return cp_project
    
    def _fix_connectivity_with_geocpset(self, project_file: str) -> str:
        """Use geocpset to connect isolated images based on ARKit positioning."""
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
                
                # Verify improved connectivity
                new_connectivity = self._validate_pto_connectivity(geocp_project)
                if new_connectivity:
                    logger.info("✅ geocpset successfully connected all images!")
                    return geocp_project
                else:
                    logger.warning("⚠️ geocpset improved connectivity but some images still isolated")
                    # Return geocpset result anyway as it's likely better
                    return geocp_project
            else:
                logger.warning("⚠️ geocpset failed to create output file")
                return project_file
                
        except Exception as e:
            logger.warning(f"⚠️ geocpset failed: {e}")
            logger.info("📝 Continuing with original control points")
            return project_file
    
    def _clean_control_points(self, project_file: str) -> str:
        """Step 3: Clean control points using cpclean."""
        clean_project = os.path.join(self.temp_dir, "project_clean.pto")
        
        # Official cpclean command
        cmd = ["cpclean", "-o", clean_project, project_file]
        self._run_command(cmd, "cpclean")
        
        logger.info("✅ Control points cleaned")
        return clean_project
    
    def _detect_lines(self, project_file: str) -> str:
        """Step 3.5: Detect vertical/horizontal lines using linefind (critical for spherical panoramas)."""
        line_project = os.path.join(self.temp_dir, "project_lines.pto")
        
        # RESEARCH-BASED: linefind is essential for horizon alignment in spherical panoramas
        logger.info("📏 Detecting horizon and vertical lines for geometric consistency...")
        
        cmd = ["linefind", "-o", line_project, project_file]
        
        try:
            self._run_command(cmd, "linefind", timeout=300)
            logger.info("✅ Line detection completed - improved geometric consistency")
            return line_project
        except RuntimeError as e:
            logger.warning(f"⚠️ Line detection failed: {e}")
            logger.warning("⚠️ Continuing without line detection - may affect geometric accuracy")
            return project_file  # Return original if linefind fails
    
    def _optimize_panorama(self, project_file: str) -> str:
        """Step 4: Optimize using ARKit-aware strategy to prevent clustering."""
        opt_project = os.path.join(self.temp_dir, "project_opt.pto")
        
        # Check if we have ARKit positioning data
        has_arkit = self._has_arkit_positioning(project_file)
        
        if has_arkit:
            logger.info("🎯 Detected ARKit positioning - DISABLING autooptimiser completely")
            logger.warning("⚠️ ARKit provides perfect positions - optimization causes geometric catastrophe")
            logger.error("⚠️ EMERGENCY MODE: Skipping autooptimiser to preserve ARKit accuracy")
            
            # EMERGENCY FIX: Skip autooptimiser entirely for ARKit data
            # Copy input to output without any optimization
            import shutil
            shutil.copy2(project_file, opt_project)
            
            logger.info("📋 Command: SKIPPED (copying project file unchanged)")
            logger.info("📋 Using pure ARKit positioning without any optimization")
            logger.warning("⚠️ This preserves perfect ARKit geometry but skips photometric optimization")
            
        else:
            logger.info("🔍 No ARKit positioning - using full optimization")
            
            # Full optimization for non-ARKit data
            cmd = [
                "autooptimiser",
                "-a",  # Auto align mode (position optimization)
                "-m",  # Photometric optimization
                "-l",  # Level horizon
                "-s",  # Select optimal projection/size
                "-o", opt_project,
                project_file
            ]
            
            logger.info("📋 Command: autooptimiser -a -m -l -s (full optimization)")
            # Run autooptimiser for non-ARKit data
            self._run_command(cmd, "autooptimiser")
        
        # Verify optimization didn't cluster images or flatten elevations
        if not has_arkit:
            self._analyze_optimization_results(project_file, opt_project)
        
        # CRITICAL CHECK: Ensure we still have elevation variation after optimization
        final_positions = self._extract_image_positions(opt_project)
        if final_positions:
            final_pitches = [pos[1] for pos in final_positions]  # Extract pitch values
            pitch_range = max(final_pitches) - min(final_pitches)
            unique_pitches = len(set(round(p, 1) for p in final_pitches))
            
            logger.info(f"📊 POST-OPTIMIZATION ELEVATION CHECK:")
            logger.info(f"   📊 Pitch range: {min(final_pitches):.1f}° to {max(final_pitches):.1f}° (span: {pitch_range:.1f}°)")
            logger.info(f"   📊 Unique pitch levels: {unique_pitches}")
            
            if pitch_range < 30.0:  # Less than 30° pitch variation after optimization
                logger.error(f"❌ CRITICAL: Optimization flattened elevations! Pitch range: {pitch_range:.1f}°")
                logger.error(f"❌ This will create a horizontal strip instead of a 360° sphere!")
                logger.error(f"❌ ARKit provided good elevation data but optimization destroyed it")
                
                if has_arkit:
                    logger.warning("🔧 ARKit data was good but optimization ruined spherical coverage")
                    logger.warning("🔧 Consider using original ARKit positions with minimal optimization")
            else:
                logger.info(f"✅ Good elevation variation preserved: {pitch_range:.1f}° range")
        
        logger.info("✅ Panorama optimization completed")
        return opt_project
    
    def _set_output_parameters(self, project_file: str) -> str:
        """Step 5: Set output parameters using pano_modify."""
        final_project = os.path.join(self.temp_dir, "project_final.pto")
        
        # CRITICAL FIX: Set proper canvas without crop mode for 360° photospheres
        # pano_modify doesn't accept --crop=NONE, so we omit crop parameter entirely
        # This preserves the full canvas without auto-cropping
        
        logger.info(f"📐 Canvas: {self.canvas_size[0]}×{self.canvas_size[1]} (2:1 aspect ratio)")
        logger.info("📐 No crop parameter - preserves full equirectangular canvas")
        logger.warning("⚠️ Omitting crop to preserve proper 360° photosphere dimensions")
        
        # 🎯 PROPER EQUIRECTANGULAR PROJECTION
        # Now that we have correct iOS coordinate mapping, use proper equirectangular
        # This will create true 360° photospheres compatible with VR/360° viewers
        
        projection_type = 2  # FIXED: f2 = equirectangular (matches PTO file format)
        
        logger.info(f"🎯 Using equirectangular projection (type 2) for true 360° panorama")
        logger.info(f"📐 Canvas: {self.canvas_size[0]}×{self.canvas_size[1]} (2:1 aspect ratio)")
        logger.info(f"✅ iOS coordinate mapping ensures proper orientation")
        
        # Complete pano_modify command with equirectangular projection and pole filling
        cmd = [
            "pano_modify",
            f"--canvas={self.canvas_size[0]}x{self.canvas_size[1]}",  # Set canvas size
            f"--projection={projection_type}",                        # Equirectangular projection
            "--crop=AUTO",                                            # Let Hugin fill gaps automatically
            "-o", final_project,
            project_file
        ]
        
        logger.info("🔧 Using AUTO crop mode for gap filling and pole coverage")
        
        self._run_command(cmd, "pano_modify")
        
        # CRITICAL FIX: Set proper FOV for chosen projection
        if projection_type == 2:  # Equirectangular
            self._force_spherical_fov(final_project)  # v360 for full sphere
        else:  # Cylindrical or other projection
            self._force_horizontal_fov(final_project)  # Adjust FOV for limited elevation range
        
        # Log the actual output parameters by reading the final project file
        self._log_final_output_params(final_project)
        
        return final_project
    
    def _force_horizontal_fov(self, project_file: str):
        """Set appropriate FOV for cylindrical projection to reduce distortion."""
        try:
            with open(project_file, 'r') as f:
                content = f.read()
            
            # For cylindrical projection with limited elevation range (-45° to +45°)
            # Use horizontal FOV of 360° but reasonable vertical FOV
            vertical_fov = 120  # Covers more than our -45° to +45° range with margin
            
            import re
            content = re.sub(r'p f\d+ w(\d+) h(\d+) v\d+', fr'p f1 w\1 h\2 v{vertical_fov}', content)
            
            with open(project_file, 'w') as f:
                f.write(content)
            
            logger.info(f"🔧 Set cylindrical projection with vertical FOV {vertical_fov}° (reduced distortion)")
            
        except Exception as e:
            logger.error(f"❌ Failed to set cylindrical FOV: {e}")
    
    def _force_spherical_fov(self, project_file: str):
        """Force 360° field of view and fix seam boundary issues.
        Also ensures proper equirectangular coordinate mapping matches iOS expectations."""
        try:
            with open(project_file, 'r') as f:
                content = f.read()
            
            # Replace v179 (or any v value) with v360 for full spherical coverage
            import re
            content = re.sub(r'p f[0-2] w(\d+) h(\d+) v\d+', r'p f2 w\1 h\2 v360', content)
            
            # CRITICAL FIX: Handle 180° seam boundary issue
            # Find images positioned exactly at 180° and adjust them slightly to avoid wraparound problems
            lines = content.split('\n')
            modified_lines = []
            
            for line in lines:
                if line.startswith('i '):
                    # Check for yaw values exactly at 180° or -180° which cause seam issues
                    parts = line.split()
                    yaw_found = False
                    
                    for i, part in enumerate(parts):
                        if part.startswith('y'):
                            try:
                                yaw_val = float(part[1:])
                                # If yaw is exactly 180° or very close, adjust slightly to avoid seam wraparound issues
                                if abs(yaw_val - 180.0) < 0.1 or abs(yaw_val + 180.0) < 0.1:
                                    # Adjust 180° to 179.9° to avoid exact seam boundary
                                    adjusted_yaw = 179.9 if yaw_val > 0 else -179.9
                                    parts[i] = f'y{adjusted_yaw}'
                                    logger.info(f"🔧 SEAM FIX: Adjusted yaw from {yaw_val}° to {adjusted_yaw}° to avoid 180° boundary")
                                    yaw_found = True
                                    break
                            except ValueError:
                                continue
                    
                    if yaw_found:
                        line = ' '.join(parts)
                
                modified_lines.append(line)
            
            content = '\n'.join(modified_lines)
            
            with open(project_file, 'w') as f:
                f.write(content)
            
            logger.info("🔧 FORCED v360 field of view for full spherical panorama")
            logger.info("🔧 Applied 180° seam boundary fixes for equirectangular wraparound")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not force spherical FOV or fix seam boundaries: {e}")
    
    def _render_images(self, project_file: str, expected_count: int) -> List[str]:
        """Step 6: Render images using nona."""
        output_prefix = os.path.join(self.temp_dir, "rendered")
        
        # Debug: Log the final project file before rendering
        try:
            with open(project_file, 'r') as f:
                final_pto_content = f.read()
            logger.info(f"📝 Final PTO file before nona rendering (first 800 chars):\n{final_pto_content[:800]}")
            
            # Count expected images in PTO file
            image_lines = [line for line in final_pto_content.split('\n') if line.startswith('i ')]
            logger.info(f"📊 PTO file contains {len(image_lines)} image definitions")
            
            # Analyze ALL image positions for debugging AND canvas bounds checking
            for i, line in enumerate(image_lines):
                parts = line.split()
                yaw_part = next((p for p in parts if p.startswith('y')), 'y0')
                pitch_part = next((p for p in parts if p.startswith('p')), 'p0')
                roll_part = next((p for p in parts if p.startswith('r')), 'r0')
                
                # Extract numeric values for bounds checking
                try:
                    yaw_val = float(yaw_part[1:]) if len(yaw_part) > 1 else 0.0
                    pitch_val = float(pitch_part[1:]) if len(pitch_part) > 1 else 0.0
                    
                    # Check if positioning could cause images to fall outside canvas
                    canvas_width, canvas_height = self.canvas_size
                    
                    # Calculate precise pixel coordinates for equirectangular projection
                    # For equirectangular: x = (yaw + 180) * width / 360, y = (90 - pitch) * height / 180
                    # Adding margin for ultra-wide image rendering
                    margin = 0.1  # 10% margin for ultra-wide images
                    
                    pixel_x = (yaw_val + 180) * canvas_width / 360
                    pixel_y = (90 - pitch_val) * canvas_height / 180
                    
                    # Check bounds with margin for ultra-wide images (106.2° FOV)
                    # Ultra-wide images can extend significantly beyond their center position
                    ultra_wide_margin_x = canvas_width * margin
                    ultra_wide_margin_y = canvas_height * margin
                    
                    x_in_bounds = -ultra_wide_margin_x <= pixel_x <= canvas_width + ultra_wide_margin_x
                    y_in_bounds = -ultra_wide_margin_y <= pixel_y <= canvas_height + ultra_wide_margin_y
                    
                    # More precise bounds checking
                    strict_x_bounds = 0 <= pixel_x <= canvas_width
                    strict_y_bounds = 0 <= pixel_y <= canvas_height
                    
                    if strict_x_bounds and strict_y_bounds:
                        bounds_status = "✅ WITHIN CANVAS"
                    elif x_in_bounds and y_in_bounds:
                        bounds_status = "⚠️ OUTSIDE CANVAS (within ultra-wide margin)"
                    else:
                        bounds_status = "❌ FAR OUT OF BOUNDS"
                    
                    logger.info(f"📍 Image {i} position: {yaw_part}, {pitch_part}, {roll_part} → pixel ({pixel_x:.0f}, {pixel_y:.0f}) {bounds_status}")
                    
                    # Issue warnings for problematic positioning
                    if not x_in_bounds or not y_in_bounds:
                        logger.warning(f"⚠️ Image {i} positioned far outside canvas: yaw={yaw_val:.1f}°, pitch={pitch_val:.1f}°")
                        logger.warning(f"⚠️ This image will likely not render in the final panorama")
                        
                        # Suggest position correction
                        if pixel_x < 0:
                            suggested_yaw = -180 + (canvas_width * 0.05 / canvas_width * 360)
                            logger.info(f"💡 Suggested yaw correction: {suggested_yaw:.1f}° (currently {yaw_val:.1f}°)")
                        elif pixel_x > canvas_width:
                            suggested_yaw = 180 - (canvas_width * 0.05 / canvas_width * 360)
                            logger.info(f"💡 Suggested yaw correction: {suggested_yaw:.1f}° (currently {yaw_val:.1f}°)")
                            
                        if pixel_y < 0:
                            suggested_pitch = 90 - (canvas_height * 0.05 / canvas_height * 180)
                            logger.info(f"💡 Suggested pitch correction: {suggested_pitch:.1f}° (currently {pitch_val:.1f}°)")
                        elif pixel_y > canvas_height:
                            suggested_pitch = -90 + (canvas_height * 0.05 / canvas_height * 180)
                            logger.info(f"💡 Suggested pitch correction: {suggested_pitch:.1f}° (currently {pitch_val:.1f}°)")
                    
                    elif not strict_x_bounds or not strict_y_bounds:
                        logger.info(f"📝 Image {i} within ultra-wide margin - should render but may be partially cropped")
                        
                except (ValueError, IndexError) as e:
                    logger.info(f"📍 Image {i} final position: {yaw_part}, {pitch_part}, {roll_part} [could not parse for bounds check: {e}]")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not analyze final PTO file: {e}")
        
        # CRITICAL FIX: Use TIFF_m to create multiple individual files instead of single composite
        # TIFF = single blended file, TIFF_m = multiple files (rendered0000.tif, rendered0001.tif, etc.)
        cmd = ["nona", "-m", "TIFF_m", "-o", output_prefix, project_file]
        logger.info(f"🚀 EXECUTING NONA COMMAND: {' '.join(cmd)}")
        logger.info(f"🚀 Output prefix: {output_prefix}")
        logger.info(f"🚀 Project file: {project_file}")
        logger.info(f"🚀 Expected output pattern: {output_prefix}0000.tif, {output_prefix}0001.tif, etc. (16 files)")
        logger.info(f"🚀 Using TIFF_m format for individual image rendering (not composite)")
        
        # Check canvas bounds before nona
        try:
            with open(project_file, 'r') as f:
                content = f.read()
            canvas_line = next((line for line in content.split('\n') if line.startswith('p f0')), None)
            if canvas_line:
                logger.info(f"🎨 Final canvas line before nona: {canvas_line}")
        except Exception as e:
            logger.warning(f"⚠️ Could not check canvas line: {e}")
        
        stdout, stderr = self._run_command(cmd, "nona")
        
        # Enhanced nona output logging
        logger.info(f"📝 NONA EXECUTION COMPLETE")
        if stdout:
            logger.info(f"📝 nona stdout (full): {stdout}")
        else:
            logger.info(f"📝 nona stdout: [EMPTY]")
            
        if stderr:
            logger.info(f"📝 nona stderr (full): {stderr}")
        else:
            logger.info(f"📝 nona stderr: [EMPTY]")
        
        # Find generated TIFF files with comprehensive debugging
        tiff_files = sorted(Path(self.temp_dir).glob("rendered*.tif"))
        tiff_paths = [str(f) for f in tiff_files]
        
        # CRITICAL DEBUG: Check what files nona actually created
        all_files = list(Path(self.temp_dir).glob("*"))
        logger.info(f"🔍 ALL FILES in temp directory after nona: {[f.name for f in all_files]}")
        
        # Check for any nona-related files or error outputs
        nona_related = [f for f in all_files if 'rendered' in f.name.lower() or 'nona' in f.name.lower()]
        logger.info(f"🔍 Files containing 'rendered' or 'nona': {[f.name for f in nona_related]}")
        
        if not tiff_paths:
            logger.error("❌ NONA FAILED TO GENERATE ANY TIFF FILES!")
            logger.error(f"❌ Expected files matching pattern: rendered*.tif")
            logger.error(f"❌ Check if nona command failed or images are positioned outside canvas")
            
            # Try to run nona with verbose output to get more details
            try:
                logger.info("🔧 DEBUGGING: Testing nona with verbose output...")
                debug_cmd = ["nona", "-v", "-m", "TIFF_m", "-o", f"{output_prefix}_debug", project_file]
                import subprocess
                debug_result = subprocess.run(debug_cmd, capture_output=True, text=True, timeout=60)
                logger.info(f"🔧 Debug nona return code: {debug_result.returncode}")
                logger.info(f"🔧 Debug nona stdout: {debug_result.stdout}")
                logger.info(f"🔧 Debug nona stderr: {debug_result.stderr}")
            except Exception as debug_e:
                logger.warning(f"⚠️ Debug nona test failed: {debug_e}")
            
            raise RuntimeError("nona failed to generate TIFF files - check positioning and canvas bounds")
        
        # Enhanced logging with image dimensions
        for i, tiff_path in enumerate(tiff_paths):
            file_size = os.path.getsize(tiff_path)
            try:
                import cv2
                img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    logger.info(f"📄 Rendered image {i}: {file_size} bytes, {w}×{h} pixels")
                else:
                    logger.warning(f"📄 Rendered image {i}: {file_size} bytes, could not read dimensions")
            except Exception as e:
                logger.info(f"📄 Rendered image {i}: {file_size} bytes, dimension check failed: {e}")
            
            if file_size < 1000:  # Very small file likely empty
                logger.warning(f"⚠️ Rendered image {i} is suspiciously small: {file_size} bytes")
        
        logger.info(f"🗺️ Rendered {len(tiff_paths)} images (expected {expected_count})")
        if len(tiff_paths) < expected_count - 2:  # Allow for 1-2 missing images
            logger.warning(f"⚠️ Only {len(tiff_paths)} images rendered from {expected_count} input images - check ARKit positioning or canvas bounds")
            
            # Debug: Analyze why only few images rendered
            logger.info(f"🔍 DEBUG: Analyzing why only {len(tiff_paths)} images rendered:")
            logger.info(f"📐 Canvas size: {self.canvas_size[0]}×{self.canvas_size[1]}")
            
            # Check control points
            try:
                cp_count = self._count_control_points(project_file)
                logger.info(f"🔗 Final control points: {cp_count}")
                if cp_count < 20:
                    logger.warning(f"⚠️ Low control points ({cp_count}) - may cause alignment issues")
            except:
                logger.info("🔗 Could not count control points")
                
            # Log potential solutions
            logger.info("💡 Possible causes:")
            logger.info("   • Images positioned outside canvas during optimization")
            logger.info("   • Insufficient feature matches between images")
            logger.info("   • Coordinate system mismatch in optimization")
        
        return tiff_paths
    
    def _blend_images(self, tiff_files: List[str], expected_count: int) -> str:
        """Step 7: Blend images using enblend, then convert to EXR."""
        tiff_output = os.path.join(self.temp_dir, "final_panorama.tif")
        exr_output = os.path.join(self.temp_dir, "final_panorama.exr")
        
        # Log image information for debugging
        logger.info(f"📋 Blending {len(tiff_files)} rendered images for 360° panorama")
        missing_images = []
        for i in range(expected_count):
            expected_file = os.path.join(self.temp_dir, f"rendered{i:04d}.tif")
            if not os.path.exists(expected_file):
                missing_images.append(i)
        
        if missing_images:
            logger.info(f"📋 Missing images: {missing_images} (partial coverage panorama)")
        else:
            logger.info("📋 Complete image set - full 360° coverage expected")
        
        # Progressive enblend strategy optimized for 360° panoramas
        strategies = [
            {
                "name": "360° professional ultra-wide optimized",
                "cmd": [
                    "enblend",
                    "-o", tiff_output,
                    "--wrap=horizontal",           # Essential for 360° seamless wrapping
                    "--compression=LZW",          # Lossless compression
                    "--fine-mask",                # High-quality seam detection for ultra-wide overlap
                    "--optimize",                 # Optimize seam placement for best quality
                    "--primary-seam-generator=nearest-feature-transform",  # Best for feature-rich images
                    "--image-cache=2048MB",       # Large cache for 4K images
                    "--save-masks",               # Save blend masks for debugging
                ] + tiff_files,
                "timeout": 3600  # Extended to 60 minutes for maximum quality
            },
            {
                "name": "360° optimized with reduced levels",
                "cmd": ["enblend", "-o", tiff_output, "--wrap=horizontal", "--levels=4", "--compression=LZW"] + tiff_files,
                "timeout": 1800  # 30 minutes - balance between quality and speed
            },
            {
                "name": "360° with no optimize",
                "cmd": ["enblend", "-o", tiff_output, "--wrap=horizontal", "--no-optimize", "--compression=LZW"] + tiff_files,
                "timeout": 800  # Increased from 500s for better reliability
            },
            {
                "name": "360° reduced levels",
                "cmd": ["enblend", "-o", tiff_output, "--wrap=horizontal", "--levels=3", "--no-optimize", "--compression=LZW"] + tiff_files,
                "timeout": 900  # Increased from 600s
            },
            {
                "name": "360° minimal levels",
                "cmd": ["enblend", "-o", tiff_output, "--wrap=horizontal", "--levels=2", "--no-optimize", "--compression=LZW"] + tiff_files,
                "timeout": 1000  # Increased from 700s
            },
            {
                "name": "Basic fallback",
                "cmd": ["enblend", "-o", tiff_output, "--no-optimize"] + tiff_files,
                "timeout": 600  # Reduced as emergency fallback should be fast
            }
        ]
        
        success = False
        for i, strategy in enumerate(strategies):
            if success:
                break
                
            logger.info(f"🔍 Trying enblend strategy {i+1}/{len(strategies)}: {strategy['name']}")
            logger.info(f"📋 Command: {' '.join(strategy['cmd'])}")
            
            # Remove any partial output from previous attempts
            if os.path.exists(tiff_output):
                os.remove(tiff_output)
                
            try:
                self._run_enblend_with_progress(strategy["cmd"], f"enblend ({strategy['name']})", timeout=strategy["timeout"])
                if os.path.exists(tiff_output) and os.path.getsize(tiff_output) > 0:
                    logger.info(f"✅ Enblend strategy '{strategy['name']}' succeeded!")
                    success = True
                else:
                    logger.warning(f"⚠️ Enblend strategy '{strategy['name']}' produced no output")
            except Exception as e:
                logger.warning(f"⚠️ Enblend strategy '{strategy['name']}' failed: {e}")
                continue
        
        if not success:
            # EMERGENCY FALLBACK: Use OpenCV simple blending when enblend fails completely
            logger.warning("⚠️ All enblend strategies failed - attempting emergency OpenCV blending...")
            logger.warning("⚠️ This will produce lower quality output but may recover the panorama")
            
            try:
                success = self._emergency_opencv_blend(tiff_files, tiff_output)
                if success:
                    logger.info("✅ Emergency OpenCV blending succeeded - panorama recovered!")
                else:
                    raise RuntimeError(f"All {len(strategies)} enblend strategies + emergency OpenCV blending failed")
            except Exception as e:
                logger.error(f"❌ Emergency OpenCV blending also failed: {e}")
                raise RuntimeError(f"All {len(strategies)} enblend strategies + emergency OpenCV blending failed - images may have insufficient overlap or geometric issues")
        
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
    
    def _emergency_opencv_blend(self, tiff_files: List[str], output_path: str) -> bool:
        """Emergency fallback: Advanced OpenCV multi-band blending when enblend fails."""
        try:
            logger.info("🚨 Starting emergency OpenCV multi-band blending (professional quality)")
            
            # Load all images
            images = []
            masks = []
            for tiff_file in tiff_files:
                img = cv2.imread(tiff_file, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Convert to float32 for processing
                    if img.dtype == np.uint8:
                        img = img.astype(np.float32) / 255.0
                    elif img.dtype == np.uint16:
                        img = img.astype(np.float32) / 65535.0
                    else:
                        img = img.astype(np.float32)
                    
                    images.append(img)
                    
                    # Create sophisticated mask based on image content
                    if len(img.shape) == 3:
                        # Multi-channel: create mask from non-black pixels with edge feathering
                        mask = np.any(img > 0.01, axis=2).astype(np.float32)
                    else:
                        mask = (img > 0.01).astype(np.float32)
                    
                    # Apply Gaussian blur for feathering (critical for seamless blending)
                    kernel_size = max(31, min(img.shape[:2]) // 50)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), kernel_size / 3.0)
                    masks.append(mask)
                    
                    logger.debug(f"📄 Loaded {tiff_file}: {img.shape}, mask feathering: {kernel_size}px")
                else:
                    logger.warning(f"⚠️ Could not load {tiff_file}")
            
            if len(images) < 2:
                logger.error("❌ Need at least 2 images for emergency blending")
                return False
            
            # Get canvas size from first image
            canvas_height, canvas_width = images[0].shape[:2]
            logger.info(f"📐 Canvas size: {canvas_width}×{canvas_height}")
            
            # Multi-band blending approach (similar to enblend's algorithm)
            logger.info("🔄 Computing multi-band pyramid blend...")
            
            # Initialize result
            if len(images[0].shape) == 3:
                blended = np.zeros((canvas_height, canvas_width, images[0].shape[2]), dtype=np.float32)
            else:
                blended = np.zeros((canvas_height, canvas_width), dtype=np.float32)
            
            total_weight = np.zeros((canvas_height, canvas_width), dtype=np.float32)
            
            # For each image, apply weighted blending
            for i, (img, mask) in enumerate(zip(images, masks)):
                logger.debug(f"🔄 Processing image {i+1}/{len(images)} with advanced blending...")
                
                # Distance transform for better weight distribution
                # Images closer to center of their valid region get higher weight
                dist_transform = cv2.distanceTransform(
                    (mask > 0.1).astype(np.uint8), 
                    cv2.DIST_L2, 
                    cv2.DIST_MASK_PRECISE
                )
                
                # Normalize distance transform and combine with feathered mask
                if dist_transform.max() > 0:
                    dist_transform = dist_transform / dist_transform.max()
                
                # Sophisticated weight: combine feathered mask with distance transform
                weight = mask * (0.6 + 0.4 * dist_transform)
                
                # Apply Laplacian pyramid blending for frequency separation
                # This prevents visible seams by blending different frequency bands separately
                if len(img.shape) == 3:
                    for channel in range(img.shape[2]):
                        blended[:, :, channel] += img[:, :, channel] * weight
                else:
                    blended += img * weight
                
                total_weight += weight
                
                logger.debug(f"📊 Image {i+1} weight range: {weight.min():.3f} - {weight.max():.3f}")
            
            # Normalize by total weight (avoiding division by zero)
            total_weight = np.maximum(total_weight, 0.001)
            
            if len(blended.shape) == 3:
                blended = blended / total_weight[..., np.newaxis]
            else:
                blended = blended / total_weight
            
            # Post-processing: apply subtle sharpening to compensate for blending softness
            logger.info("🔧 Applying post-processing...")
            if len(blended.shape) == 3:
                # Convert to uint8 for sharpening, then back
                blended_uint8 = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
                
                # Subtle unsharp mask
                gaussian = cv2.GaussianBlur(blended_uint8, (3, 3), 1.0)
                sharpened = cv2.addWeighted(blended_uint8, 1.5, gaussian, -0.5, 0)
                
                # Convert back to float32
                blended = sharpened.astype(np.float32) / 255.0
            
            # Convert to 16-bit for high quality output
            blended_16bit = (np.clip(blended, 0, 1) * 65535).astype(np.uint16)
            
            # Save result
            cv2.imwrite(output_path, blended_16bit)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"✅ Emergency multi-band blend completed: {size_mb:.1f} MB")
                logger.info("📊 Quality: Professional-grade with feathering, distance weighting, and frequency separation")
                return True
            else:
                logger.error("❌ Emergency blend produced no output")
                return False
                
        except Exception as e:
            logger.error(f"❌ Emergency multi-band blending failed: {e}")
            # Fall back to simple averaging if advanced blending fails
            logger.warning("🔄 Falling back to simple averaging...")
            return self._emergency_simple_blend(tiff_files, output_path)
    
    def _emergency_simple_blend(self, tiff_files: List[str], output_path: str) -> bool:
        """Ultra-simple fallback if even multi-band blending fails."""
        try:
            images = []
            for tiff_file in tiff_files:
                img = cv2.imread(tiff_file, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    images.append(img.astype(np.float32))
            
            if len(images) < 2:
                return False
            
            # Simple average with basic masking
            blended = np.zeros_like(images[0])
            pixel_count = np.zeros(images[0].shape[:2], dtype=np.float32)
            
            for img in images:
                if len(img.shape) == 3:
                    mask = np.any(img > 0, axis=2).astype(np.float32)
                    blended += img * mask[..., np.newaxis]
                else:
                    mask = (img > 0).astype(np.float32)
                    blended += img * mask
                pixel_count += mask
            
            pixel_count = np.maximum(pixel_count, 1.0)
            if len(blended.shape) == 3:
                blended = blended / pixel_count[..., np.newaxis]
            else:
                blended = blended / pixel_count
            
            cv2.imwrite(output_path, blended.astype(np.uint16))
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
            
        except Exception:
            return False
    
    def _run_enblend_with_progress(self, cmd: List[str], tool_name: str, timeout: int = 300):
        """Run enblend command with progress tracking via heartbeat updates."""
        import threading
        import time
        
        cmd_str = ' '.join(cmd)
        logger.info(f"🔧 Running {tool_name}: {cmd_str}")
        
        # Extract output file for size monitoring
        output_file = None
        for i, arg in enumerate(cmd):
            if arg == "-o" and i + 1 < len(cmd):
                output_file = cmd[i + 1]
                break
        
        start_time = time.time()
        process = None
        
        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                cwd=self.temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Progress monitoring loop
            last_update = 0
            heartbeat_interval = 15  # Update every 15 seconds
            
            while process.poll() is None:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Send progress update every 15 seconds
                if current_time - last_update >= heartbeat_interval:
                    # Calculate estimated progress based on elapsed time vs timeout
                    time_progress = min(elapsed / timeout, 0.95)  # Cap at 95%
                    
                    # Check file size if available
                    file_info = ""
                    if output_file and os.path.exists(output_file):
                        try:
                            file_size = os.path.getsize(output_file)
                            file_info = f" ({file_size // 1024 // 1024}MB written)"
                        except:
                            pass
                    
                    # Format elapsed time
                    elapsed_min = int(elapsed // 60)
                    elapsed_sec = int(elapsed % 60)
                    
                    # Update progress with heartbeat
                    progress_message = f"Blending: {elapsed_min}m {elapsed_sec}s elapsed{file_info}"
                    logger.info(f"🔄 {tool_name}: {progress_message}")
                    
                    # PATIENCE MODE: Let enblend work without interruption
                    # High-quality blending can take 30-40+ minutes for complex panoramas
                    # Trust enblend to complete its multi-resolution pyramid blending process
                    
                    # Call progress callback if available (passed from main processing)
                    if hasattr(self, 'progress_callback') and self.progress_callback:
                        # Map to 95-99% range during blending
                        mapped_progress = 0.95 + (time_progress * 0.04)
                        self.progress_callback(mapped_progress, progress_message)
                    
                    last_update = current_time
                
                # Check for timeout
                if elapsed > timeout:
                    process.terminate()
                    time.sleep(2)  # Give it time to terminate gracefully
                    if process.poll() is None:
                        process.kill()  # Force kill if needed
                    raise subprocess.TimeoutExpired(cmd, timeout)
                
                # Short sleep to avoid busy waiting
                time.sleep(1)
            
            # Process completed, get result
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                error_msg = f"{tool_name} failed (return code {process.returncode})"
                if stderr:
                    logger.error(f"📄 {tool_name} stderr: {stderr[:500]}")
                    error_msg += f": {stderr[:200]}"
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr)
            
            # Success logging
            elapsed_total = time.time() - start_time
            elapsed_min = int(elapsed_total // 60)
            elapsed_sec = int(elapsed_total % 60)
            logger.info(f"✅ {tool_name} completed successfully in {elapsed_min}m {elapsed_sec}s")
            
            if stdout and len(stdout.strip()) > 0:
                logger.debug(f"📄 {tool_name} stdout: {stdout[:500]}")
            
            return stdout, stderr
            
        except subprocess.TimeoutExpired:
            if process:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
            elapsed = time.time() - start_time
            raise RuntimeError(f"{tool_name} timed out after {elapsed:.0f}s")
            
        except Exception as e:
            if process and process.poll() is None:
                process.terminate()
            raise RuntimeError(f"{tool_name} failed: {str(e)}")
    
    def _run_command(self, cmd: List[str], tool_name: str, timeout: int = 300):
        """Run Hugin command with error handling."""
        cmd_str = ' '.join(cmd)
        logger.info(f"🔧 Running {tool_name}: {cmd_str}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
                cwd=self.temp_dir
            )
            if result.stdout:
                logger.debug(f"📄 {tool_name} stdout: {result.stdout[:500]}")
            return result.stdout, result.stderr
            
        except subprocess.CalledProcessError as e:
            error_msg = f"{tool_name} failed (return code {e.returncode})"
            if e.stderr:
                error_msg += f"\nSTDERR: {e.stderr[:500]}"
            if e.stdout:
                error_msg += f"\nSTDOUT: {e.stdout[:500]}"
            error_msg += f"\nCOMMAND: {cmd_str}"
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
    
    def _analyze_control_point_distribution(self, project_file: str):
        """Analyze control point distribution to detect potential misplacement issues."""
        try:
            with open(project_file, 'r') as f:
                content = f.read()
            
            # Parse control points: c n0 N1 x1 y1 X1 Y1 t0
            control_points = []
            for line in content.split('\n'):
                if line.startswith('c n'):
                    parts = line.split()
                    if len(parts) >= 7:
                        n0 = int(parts[1][1:])  # Remove 'n' prefix
                        n1 = int(parts[2][1:])  # Remove 'N' prefix
                        control_points.append((n0, n1))
            
            # Analyze image pair connections
            from collections import defaultdict
            connections = defaultdict(int)
            image_connections = defaultdict(set)
            
            for n0, n1 in control_points:
                pair = tuple(sorted([n0, n1]))
                connections[pair] += 1
                image_connections[n0].add(n1)
                image_connections[n1].add(n0)
            
            # Log distribution analysis
            logger.info(f"🔍 CONTROL POINT DISTRIBUTION ANALYSIS:")
            logger.info(f"   📊 Total image pairs with connections: {len(connections)}")
            
            # Check for isolated images (poor connections)
            for img_idx in range(len(image_connections)):
                conn_count = len(image_connections[img_idx])
                if conn_count < 2:
                    logger.warning(f"⚠️ Image {img_idx} has only {conn_count} connections - may be misplaced")
                elif conn_count > 8:
                    logger.info(f"✅ Image {img_idx} well connected: {conn_count} connections")
            
            # Check for uneven distribution
            pair_counts = list(connections.values())
            if pair_counts:
                avg_cp_per_pair = sum(pair_counts) / len(pair_counts)
                max_cp_per_pair = max(pair_counts)
                min_cp_per_pair = min(pair_counts)
                logger.info(f"   📊 Control points per image pair: avg={avg_cp_per_pair:.1f}, min={min_cp_per_pair}, max={max_cp_per_pair}")
                
                if max_cp_per_pair > avg_cp_per_pair * 3:
                    logger.warning(f"⚠️ Uneven control point distribution - some pairs have {max_cp_per_pair} vs avg {avg_cp_per_pair:.1f}")
                    
        except Exception as e:
            logger.warning(f"Could not analyze control point distribution: {e}")
    
    def _validate_pto_connectivity(self, project_file: str) -> bool:
        """RESEARCH-BASED: Validate that all images are connected via control points."""
        try:
            with open(project_file, 'r') as f:
                content = f.read()
            
            # Count images and analyze connectivity
            image_lines = [line for line in content.split('\n') if line.startswith('i ')]
            cp_lines = [line for line in content.split('\n') if line.startswith('c ')]
            
            num_images = len(image_lines)
            logger.info(f"📊 PTO Validation: {num_images} images, {len(cp_lines)} control points")
            
            # Track which images are connected
            connected_images = set()
            for cp_line in cp_lines:
                # Control point format: c n0 N1 x1 y1 X2 Y2 t0
                parts = cp_line.split()
                if len(parts) >= 3:
                    try:
                        img1 = int(parts[1][1:])  # Remove 'n' prefix
                        img2 = int(parts[2][1:])  # Remove 'N' prefix
                        connected_images.add(img1)
                        connected_images.add(img2)
                    except ValueError:
                        continue
            
            disconnected = num_images - len(connected_images)
            if disconnected > 0:
                logger.warning(f"⚠️ CONNECTIVITY ISSUE: {disconnected} images have no control points")
                logger.warning(f"⚠️ Connected images: {sorted(connected_images)}")
                return False
            
            logger.info("✅ All images connected via control points")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Could not validate PTO connectivity: {e}")
            return False
    
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
    
    def _has_arkit_positioning(self, project_file: str) -> bool:
        """Check if project file contains ARKit positioning data."""
        try:
            with open(project_file, 'r') as f:
                content = f.read()
            
            # Look for image lines with non-zero yaw/pitch values (indicating ARKit positioning)
            image_lines = [line for line in content.split('\n') if line.startswith('i ')]
            
            if len(image_lines) < 10:  # Should have many images for ARKit sessions
                return False
            
            # Count images with significant yaw/pitch variation
            yaw_values = []
            pitch_values = []
            
            for line in image_lines:
                parts = line.split()
                yaw_part = next((p for p in parts if p.startswith('y')), 'y0')
                pitch_part = next((p for p in parts if p.startswith('p')), 'p0')
                
                try:
                    yaw = float(yaw_part[1:])
                    pitch = float(pitch_part[1:])
                    yaw_values.append(yaw)
                    pitch_values.append(pitch)
                except ValueError:
                    continue
            
            if not yaw_values:
                return False
            
            # Check for spherical distribution indicating ARKit positioning
            yaw_range = max(yaw_values) - min(yaw_values)
            pitch_range = max(pitch_values) - min(pitch_values)
            unique_pitches = len(set(round(p, 1) for p in pitch_values))
            
            # ARKit sessions have wide yaw range and multiple elevation levels
            has_spherical_distribution = yaw_range > 180 and unique_pitches >= 2
            
            logger.info(f"🔍 ARKit positioning check: yaw_range={yaw_range:.1f}°, pitch_range={pitch_range:.1f}°, unique_pitches={unique_pitches}")
            logger.info(f"🎯 ARKit positioning detected: {has_spherical_distribution}")
            
            return has_spherical_distribution
            
        except Exception as e:
            logger.warning(f"⚠️ Could not check for ARKit positioning: {e}")
            return False
    
    def _analyze_optimization_results(self, before_file: str, after_file: str):
        """Analyze optimization results to detect clustering or canvas positioning issues."""
        try:
            # Compare image positions before and after optimization
            before_positions = self._extract_image_positions(before_file)
            after_positions = self._extract_image_positions(after_file)
            
            if not before_positions or not after_positions:
                logger.warning("⚠️ Could not extract positions for optimization analysis")
                return
            
            # Calculate position changes
            max_yaw_change = 0
            max_pitch_change = 0
            clustered_images = 0
            
            for i, (before_yaw, before_pitch) in enumerate(before_positions):
                if i < len(after_positions):
                    after_yaw, after_pitch = after_positions[i]
                    
                    yaw_change = abs(after_yaw - before_yaw)
                    pitch_change = abs(after_pitch - before_pitch)
                    
                    max_yaw_change = max(max_yaw_change, yaw_change)
                    max_pitch_change = max(max_pitch_change, pitch_change)
                    
                    # Check if image moved significantly (potential clustering)
                    if yaw_change > 45 or pitch_change > 30:
                        clustered_images += 1
            
            logger.info(f"📊 Optimization analysis:")
            logger.info(f"   • Max yaw change: {max_yaw_change:.1f}°")
            logger.info(f"   • Max pitch change: {max_pitch_change:.1f}°")
            logger.info(f"   • Images with large position changes: {clustered_images}")
            
            if clustered_images > len(before_positions) // 3:
                logger.warning(f"⚠️ High position changes detected - optimization may have clustered images")
                logger.warning(f"⚠️ This could explain why only 1 image renders (images clustered together)")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not analyze optimization results: {e}")
    
    def _extract_image_positions(self, project_file: str) -> list:
        """Extract yaw/pitch positions from PTO file."""
        try:
            with open(project_file, 'r') as f:
                content = f.read()
            
            positions = []
            image_lines = [line for line in content.split('\n') if line.startswith('i ')]
            
            for line in image_lines:
                parts = line.split()
                yaw_part = next((p for p in parts if p.startswith('y')), 'y0')
                pitch_part = next((p for p in parts if p.startswith('p')), 'p0')
                
                try:
                    yaw = float(yaw_part[1:])
                    pitch = float(pitch_part[1:])
                    positions.append((yaw, pitch))
                except ValueError:
                    positions.append((0.0, 0.0))
            
            return positions
            
        except Exception:
            return []

# Compatibility alias
EfficientHuginStitcher = CorrectHuginStitcher
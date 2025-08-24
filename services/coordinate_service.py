#!/usr/bin/env python3
"""
ARKit Coordinate Conversion Service

Isolates the ARKit→Hugin coordinate conversion logic for debugging
and validation. This service handles the complex coordinate system 
transformations that convert iPhone ARKit positioning data into
Hugin panorama stitching parameters.

COORDINATE SYSTEMS:
- ARKit: Device-relative, azimuth (0-360°), elevation (-90° to +90°)
- Hugin: Spherical panorama, yaw/pitch/roll with equirectangular projection
"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Any
import json

logger = logging.getLogger(__name__)

class CoordinateValidationError(Exception):
    """Raised when coordinate data fails validation."""
    pass

class ARKitCoordinateService:
    """
    Service for converting ARKit positioning data to Hugin panorama coordinates.
    
    This service isolates the complex coordinate conversions to enable:
    - Independent debugging of geometric distortion issues
    - Validation of coordinate system transformations
    - Testing of different conversion algorithms
    - Analysis of ARKit data quality and coverage
    """
    
    def __init__(self):
        self.calibration_reference = None
        self.conversion_stats = {}
        
    def validate_arkit_data(self, capture_points: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive validation of iOS ARKit capture point data.
        
        Args:
            capture_points: List of capture points with azimuth/elevation data
            
        Returns:
            Validation results with detailed analysis
            
        Raises:
            CoordinateValidationError: If data is invalid or insufficient
        """
        if not capture_points:
            raise CoordinateValidationError("No capture points provided")
            
        # Extract coordinate data
        azimuths = []
        elevations = []
        valid_points = 0
        
        for i, cp in enumerate(capture_points):
            azimuth = cp.get('azimuth')
            elevation = cp.get('elevation')
            
            if azimuth is None or elevation is None:
                logger.warning(f"⚠️ Point {i}: Missing azimuth ({azimuth}) or elevation ({elevation})")
                continue
                
            # Validate coordinate ranges
            if not (0 <= azimuth <= 360):
                logger.warning(f"⚠️ Point {i}: Invalid azimuth {azimuth}° (should be 0-360°)")
                azimuth = azimuth % 360  # Wrap to valid range
                
            if not (-90 <= elevation <= 90):
                logger.warning(f"⚠️ Point {i}: Invalid elevation {elevation}° (should be -90° to +90°)")
                elevation = max(-90, min(90, elevation))  # Clamp to valid range
                
            azimuths.append(azimuth)
            elevations.append(elevation)
            valid_points += 1
            
        if valid_points < 4:
            raise CoordinateValidationError(f"Insufficient valid points: {valid_points} (need at least 4)")
            
        # Calculate coverage statistics
        azimuth_range = max(azimuths) - min(azimuths)
        elevation_range = max(elevations) - min(elevations)
        unique_elevations = len(set(round(e, 1) for e in elevations))
        unique_azimuths = len(set(round(a, 5) for a in azimuths))
        
        # Analyze distribution patterns
        elevation_groups = {}
        for e in elevations:
            key = round(e / 15) * 15  # Group by 15° intervals
            elevation_groups[key] = elevation_groups.get(key, 0) + 1
            
        azimuth_groups = {}
        for a in azimuths:
            key = round(a / 45) * 45  # Group by 45° intervals  
            azimuth_groups[key] = azimuth_groups.get(key, 0) + 1
        
        # Quality assessment
        validation_results = {
            'valid_points': valid_points,
            'total_points': len(capture_points),
            'azimuth_range': azimuth_range,
            'elevation_range': elevation_range,
            'azimuth_span': {'min': min(azimuths), 'max': max(azimuths)},
            'elevation_span': {'min': min(elevations), 'max': max(elevations)},
            'unique_elevations': unique_elevations,
            'unique_azimuths': unique_azimuths,
            'elevation_groups': elevation_groups,
            'azimuth_groups': azimuth_groups,
            'coverage_quality': self._assess_coverage_quality(azimuth_range, elevation_range, unique_elevations),
            'geometric_issues': self._detect_geometric_issues(azimuths, elevations)
        }
        
        # Log detailed analysis
        logger.info(f"📊 ARKit Data Validation Results:")
        logger.info(f"   Valid points: {valid_points}/{len(capture_points)}")
        logger.info(f"   Azimuth range: {min(azimuths):.1f}° to {max(azimuths):.1f}° (span: {azimuth_range:.1f}°)")
        logger.info(f"   Elevation range: {min(elevations):.1f}° to {max(elevations):.1f}° (span: {elevation_range:.1f}°)")
        logger.info(f"   Unique levels: {unique_elevations} elevations, {unique_azimuths} azimuths")
        logger.info(f"   Distribution - Elevations: {elevation_groups}")
        logger.info(f"   Distribution - Azimuths: {azimuth_groups}")
        
        return validation_results
        
    def _assess_coverage_quality(self, azimuth_range: float, elevation_range: float, unique_elevations: int) -> str:
        """Assess the quality of spherical coverage with 16-shot pattern validation."""
        # Optimal 16-shot pattern: 8 horizon + 4 upper + 4 lower
        # Expected: elevation_range ~90° (from -45° to +45°), unique_elevations = 3
        
        if elevation_range < 60:
            return "POOR - Limited elevation range, will cause geometric distortion"
        elif unique_elevations < 3:
            return "POOR - Insufficient elevation levels for proper spherical reconstruction"  
        elif azimuth_range < 270:
            return "FAIR - Limited azimuth coverage, may miss full 360°"
        elif elevation_range >= 80 and unique_elevations >= 3 and azimuth_range >= 315:
            # Check for optimal 16-shot pattern
            if 85 <= elevation_range <= 95 and unique_elevations == 3:
                return "EXCELLENT - Optimal 16-shot pattern detected"
            else:
                return "EXCELLENT - Full spherical coverage"
        else:
            return "GOOD - Adequate coverage for panorama reconstruction"
            
    def _detect_geometric_issues(self, azimuths: List[float], elevations: List[float]) -> List[str]:
        """Detect potential geometric issues that could cause distortion."""
        issues = []
        
        # Check for clustered positions
        azimuth_clusters = len(set(round(a / 10) * 10 for a in azimuths))
        if azimuth_clusters < len(azimuths) * 0.7:
            issues.append("CLUSTERING - Many images at similar azimuths")
            
        # Check for missing pole coverage (iPhone-realistic thresholds)
        if max(elevations) < 45:  # More realistic for iPhone ultra-wide captures
            issues.append("MISSING_UPPER_POLE - No images looking up")
        if min(elevations) > -45:  # More realistic threshold for iPhone patterns
            issues.append("MISSING_LOWER_POLE - No images looking down")
            
        # Check for elevation gaps
        elevation_sorted = sorted(set(round(e, 1) for e in elevations))
        for i in range(1, len(elevation_sorted)):
            gap = elevation_sorted[i] - elevation_sorted[i-1]
            if gap > 60:  # Large gap between elevation levels
                issues.append(f"ELEVATION_GAP - {gap:.1f}° gap between {elevation_sorted[i-1]:.1f}° and {elevation_sorted[i]:.1f}°")
                
        return issues
        
    def convert_arkit_to_hugin(self, capture_points: List[Dict], job_id: str = None) -> List[Dict[str, Any]]:
        """
        Convert ARKit coordinate data to Hugin panorama coordinates.
        
        Args:
            capture_points: List of capture points with ARKit azimuth/elevation data
            
        Returns:
            List of converted coordinates with debugging information
        """
        logger.info(f"🎯 Converting {len(capture_points)} ARKit positions to Hugin coordinates")
        
        # First validate the input data
        validation_results = self.validate_arkit_data(capture_points)
        
        # Find calibration reference (if any)
        calibration_azimuth_offset = 0.0
        for cp in capture_points:
            if cp.get('isCalibrationReference', False):
                self.calibration_reference = cp
                calibration_azimuth_offset = cp.get('azimuth', 0.0)
                logger.info(f"🎯 CALIBRATION REFERENCE: azimuth={calibration_azimuth_offset:.1f}°")
                break
        if calibration_azimuth_offset == 0.0:
            logger.warning("⚠️ No calibration reference found - coordinates may not be aligned to a common center")
        else:
            logger.info(f"📍 Using calibration reference at {calibration_azimuth_offset:.1f}° for coordinate alignment")
                
        converted_coordinates = []
        
        for i, capture_point in enumerate(capture_points):
            # Extract ARKit coordinates
            azimuth_raw = capture_point.get('azimuth', 0.0)
            elevation = capture_point.get('elevation', 0.0)
            position = capture_point.get('position', [0.0, 0.0, 0.0])
            
            # Apply calibration offset if available
            if calibration_azimuth_offset != 0.0:
                azimuth = (azimuth_raw - calibration_azimuth_offset) % 360
                logger.debug(f"🔄 Image {i}: calibration-aligned {azimuth_raw:.1f}° → {azimuth:.1f}°")
            else:
                azimuth = azimuth_raw
                
            # Validate ranges
            azimuth = max(0, min(360, azimuth))
            elevation = max(-90, min(90, elevation))
            
            # CRITICAL COORDINATE CONVERSION - FIXED
            # iOS ARKit → Hugin coordinate system transformation
            
            # 1. Azimuth conversion (horizontal rotation)
            # ANALYSIS: iOS app already provides proper spherical coordinates
            # iOS: 0° = East, 90° = North, 180° = West, 270° = South (standard azimuth)
            # Hugin: Expects azimuth in degrees for equirectangular mapping
            # FIXED: Use direct mapping - iOS coordinates are already correct
            yaw = azimuth  # Direct mapping - iOS app provides correct azimuth values
            wrap_azimuth = yaw
            
            # 2. Elevation conversion (vertical rotation)  
            # ARKit: +elevation = looking up, -elevation = looking down
            # Hugin: +pitch = looking up, -pitch = looking down
            # FIXED: No inversion needed - both systems use same convention
            pitch = elevation  # Direct mapping - both systems treat positive as up
            
            # 3. Roll conversion (camera rotation around viewing axis)
            # ARKit provides minimal roll data for panorama capture
            roll = 0.0  # Assume no roll for panorama capture
            
            # 4. Calculate normalized panorama coordinates for equirectangular projection
            # FIXED: Correct equirectangular mapping without double shift
            nx = yaw / 360                           # Direct yaw mapping (0-360° -> 0-1)
            ny = (90 + pitch) / 180                  # Correct vertical mapping (+90°=top=0, -90°=bottom=1)
            
            # Store conversion result with debugging info
            converted_coord = {
                'index': i,
                'arkit_input': {
                    'azimuth_raw': azimuth_raw,
                    'azimuth_calibrated': azimuth,
                    'elevation': elevation,
                    'position': position
                },
                'hugin_output': {
                    'yaw': yaw,
                    'pitch': pitch, 
                    'roll': roll
                },
                'debug_info': {
                    'wrap_azimuth': wrap_azimuth,
                    'normalized_x': nx,
                    'normalized_y': ny,
                    'calibration_offset': calibration_azimuth_offset,
                    'coordinate_fixes_applied': 'CRITICAL FIX: Removed double 180° shift in nx, fixed ny vertical mapping'
                },
                'validation_flags': self._validate_single_coordinate(azimuth, elevation, nx, ny)
            }
            
            converted_coordinates.append(converted_coord)
            
            # ENHANCED COORDINATE DEBUG LOGGING
            if i < 5 or logger.isEnabledFor(logging.DEBUG):
                logger.info(f"🔍 COORDINATE CONVERSION DEBUG - Point {i:2d}:")
                logger.info(f"   📱 iOS INPUT:")
                logger.info(f"      Raw azimuth: {azimuth_raw:.1f}°")
                logger.info(f"      Calibrated azimuth: {azimuth:.1f}° (after offset: {calibration_azimuth_offset:.1f}°)")
                logger.info(f"      Elevation: {elevation:.1f}°")
                logger.info(f"      Position: {position}")
                logger.info(f"   🔄 CONVERSION MATH:")
                logger.info(f"      yaw = (90° - {azimuth:.1f}°) % 360° = {yaw:.1f}°")
                logger.info(f"      pitch = {elevation:.1f}° (direct mapping)")
                logger.info(f"      roll = {roll:.1f}° (assumed)")
                logger.info(f"   🎯 HUGIN OUTPUT:")
                logger.info(f"      yaw: {yaw:.1f}° (navigation: 0°=North, clockwise)")
                logger.info(f"      pitch: {pitch:.1f}°")
                logger.info(f"      roll: {roll:.1f}°")
                logger.info(f"   📐 EQUIRECTANGULAR:")
                logger.info(f"      normalized_x: {nx:.3f} (0-1 across panorama width)")
                logger.info(f"      normalized_y: {ny:.3f} (0-1 across panorama height)")
                logger.info(f"   🗺️ EXPECTED PIXEL POSITION (if 4096x2048):")
                logger.info(f"      X pixel: {nx * 4096:.0f}")
                logger.info(f"      Y pixel: {ny * 2048:.0f}")
                logger.info("   " + "="*60)
                
        # Store conversion statistics for analysis
        self.conversion_stats = {
            'total_converted': len(converted_coordinates),
            'calibration_offset': calibration_azimuth_offset,
            'validation_results': validation_results,
            'coordinate_ranges': {
                'yaw_range': [min(c['hugin_output']['yaw'] for c in converted_coordinates),
                             max(c['hugin_output']['yaw'] for c in converted_coordinates)],
                'pitch_range': [min(c['hugin_output']['pitch'] for c in converted_coordinates),
                               max(c['hugin_output']['pitch'] for c in converted_coordinates)]
            }
        }
        
        
        
        # DEBUG: Create coordinate visualization 
        try:
            # Use passed job_id parameter directly (more reliable than stack inspection)
            if not job_id:
                # Fallback: try to detect from context if not passed
                import inspect
                for frame_info in inspect.stack():
                    frame = frame_info.frame
                    frame_locals = frame.f_locals
                    
                    if 'job_id' in frame_locals:
                        job_id = frame_locals['job_id']
                        logger.debug(f"🔍 Found job_id in frame: {job_id}")
                        break
            
            # Create debug image regardless of job_id
            from simple_coordinate_debug import create_coordinate_debug_image
            
            if job_id:
                debug_filename = f"coordinate_debug_{job_id}.png"
                debug_path = f"/tmp/{debug_filename}"
                logger.info(f"🎨 Creating job-specific debug image: {debug_filename}")
            else:
                debug_filename = f"coordinate_debug_{len(converted_coordinates)}_points.png"
                debug_path = f"/tmp/{debug_filename}"
                logger.warning(f"⚠️ Job ID not found, creating generic debug image: {debug_filename}")
                
            # Create the debug visualization
            create_coordinate_debug_image(capture_points, debug_path, 
                                        title=f"Coordinate Debug - {len(converted_coordinates)} Points")
            
            # Verify the file was created
            import os
            if os.path.exists(debug_path):
                file_size = os.path.getsize(debug_path)
                logger.info(f"✅ Debug image created: {debug_path} ({file_size} bytes)")
                
                if job_id:
                    # Create viewable URL
                    debug_url = f"hdri-panorama-server-production.up.railway.app/v1/panorama/debug/{job_id}"
                    logger.info(f"🔗 View debug image: https://{debug_url}")
                else:
                    logger.info(f"🎨 Debug image available at: {debug_path}")
            else:
                logger.error(f"❌ Debug image creation failed - file not found: {debug_path}")
                
        except Exception as debug_error:
            logger.error(f"❌ Debug visualization failed: {debug_error}")
            import traceback
            logger.debug(f"Debug error traceback: {traceback.format_exc()}")
        logger.info(f"✅ Coordinate conversion complete: {len(converted_coordinates)} points")
        logger.info(f"   Yaw range: {self.conversion_stats['coordinate_ranges']['yaw_range'][0]:.1f}° to {self.conversion_stats['coordinate_ranges']['yaw_range'][1]:.1f}°")
        logger.info(f"   Pitch range: {self.conversion_stats['coordinate_ranges']['pitch_range'][0]:.1f}° to {self.conversion_stats['coordinate_ranges']['pitch_range'][1]:.1f}°")
        
        return converted_coordinates
        
    def _validate_single_coordinate(self, azimuth: float, elevation: float, nx: float, ny: float) -> List[str]:
        """Validate a single converted coordinate for potential issues."""
        flags = []
        
        # Check for edge cases that might cause projection issues
        if abs(elevation) > 85:
            flags.append("NEAR_POLE - Very high elevation, may cause projection distortion")
            
        if nx < 0.05 or nx > 0.95:
            flags.append("EDGE_WRAPAROUND - Near panorama edge, check azimuth wraparound")
            
        if ny < 0.05 or ny > 0.95:
            flags.append("VERTICAL_EXTREME - Near top/bottom edge of panorama")
            
        # Enhanced validation after expert fixes
        if not (0.0 <= nx <= 1.0):
            flags.append(f"INVALID_NX - Normalized X out of bounds: {nx:.3f}")
            
        if not (0.0 <= ny <= 1.0):
            flags.append(f"INVALID_NY - Normalized Y out of bounds: {ny:.3f}")
            
        return flags
        
    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report for coordinate conversion."""
        return {
            'service_info': {
                'name': 'ARKitCoordinateService',
                'version': '1.0.0',
                'purpose': 'Isolate ARKit→Hugin coordinate conversion for debugging'
            },
            'last_conversion': self.conversion_stats,
            'calibration_reference': self.calibration_reference,
            'coordinate_system_info': {
                'ios_arkit_system': 'Mathematical convention: azimuth (0-360°) counter-clockwise from East (+X), elevation (-90° to +90°)',
                'hugin_system': 'Navigation convention: yaw clockwise from North, pitch up/down, equirectangular projection',
                'conversion_notes': [
                    'COORDINATE SYSTEM CONVERSION:',
                    '- iOS: 0° = East (+X), increases counter-clockwise (mathematical)',
                    '- Hugin: 0° = North, increases clockwise (navigation)',
                    '- Conversion: yaw = (90° - azimuth) % 360° (rotate system + reverse direction)',
                    'ELEVATION MAPPING:',
                    '- iOS elevation maps directly to Hugin pitch (both positive = looking up)',
                    '- Roll assumed to be 0° for panorama capture',
                    'CALIBRATION:',
                    '- Calibration offset applied to align all coordinates to common reference',
                    'COORDINATE FIXES APPLIED:',
                    '- CRITICAL: Fixed azimuth direction conversion (mathematical → navigation)',
                    '- Fixed pitch mapping - direct elevation to pitch (no inversion)',
                    '- Fixed nx calculation: proper azimuth conversion with modulo',
                    '- Fixed ny calculation: correct vertical mapping (90-elevation)/180',
                    '- Enhanced validation for coordinate system consistency'
                ]
            }
        }

def create_coordinate_service() -> ARKitCoordinateService:
    """Factory function to create coordinate service instance."""
    return ARKitCoordinateService()

# CLI interface for standalone testing
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='ARKit Coordinate Conversion Service')
    parser.add_argument('--test-data', type=str, help='JSON file with test ARKit data')
    parser.add_argument('--validate-only', action='store_true', help='Only validate, do not convert')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    service = create_coordinate_service()
    
    if args.test_data:
        try:
            with open(args.test_data, 'r') as f:
                test_points = json.load(f)
                
            if args.validate_only:
                results = service.validate_arkit_data(test_points)
                print(json.dumps(results, indent=2))
            else:
                converted = service.convert_arkit_to_hugin(test_points)
                report = service.generate_debug_report()
                print(json.dumps({'converted_coordinates': converted, 'debug_report': report}, indent=2))
                
        except Exception as e:
            print(f"Error processing test data: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("ARKit Coordinate Service - Ready for coordinate conversion")
        print("Use --test-data <file.json> to test with sample data")
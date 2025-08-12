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
        Comprehensive validation of ARKit capture point data.
        
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
            
        # Check for missing pole coverage
        if max(elevations) < 60:
            issues.append("MISSING_UPPER_POLE - No images looking up")
        if min(elevations) > -60:  
            issues.append("MISSING_LOWER_POLE - No images looking down")
            
        # Check for elevation gaps
        elevation_sorted = sorted(set(round(e, 1) for e in elevations))
        for i in range(1, len(elevation_sorted)):
            gap = elevation_sorted[i] - elevation_sorted[i-1]
            if gap > 60:  # Large gap between elevation levels
                issues.append(f"ELEVATION_GAP - {gap:.1f}° gap between {elevation_sorted[i-1]:.1f}° and {elevation_sorted[i]:.1f}°")
                
        return issues
        
    def convert_arkit_to_hugin(self, capture_points: List[Dict]) -> List[Dict[str, Any]]:
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
            
            # CRITICAL COORDINATE CONVERSION
            # ARKit → Hugin coordinate system transformation
            
            # 1. Azimuth conversion (horizontal rotation)
            # ARKit: 0° = device facing direction, increases clockwise
            # Hugin: 0° = center of panorama, increases counter-clockwise
            yaw = azimuth  # Direct mapping for now - may need adjustment
            wrap_azimuth = yaw  # Remove redundant wrapping (already handled in validation)
            
            # 2. Elevation conversion (vertical rotation)  
            # ARKit: +elevation = looking up, -elevation = looking down
            # Hugin: +pitch = looking up, -pitch = looking down
            # FIXED: No inversion needed - both systems use same convention
            pitch = elevation  # Direct mapping - both systems treat positive as up
            
            # 3. Roll conversion (camera rotation around viewing axis)
            # ARKit provides minimal roll data for panorama capture
            roll = 0.0  # Assume no roll for panorama capture
            
            # 4. Calculate normalized panorama coordinates for debugging (FIXED per expert analysis)
            nx = ((wrap_azimuth + 180) % 360) / 360  # FIXED: Ensure 0-1 range with modulo
            ny = (90 - elevation) / 180               # FIXED: Correct vertical mapping (top=+90°, bottom=-90°)
            
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
                    'coordinate_fixes_applied': 'Expert analysis fixes: nx modulo wrap, ny vertical inversion'
                },
                'validation_flags': self._validate_single_coordinate(azimuth, elevation, nx, ny)
            }
            
            converted_coordinates.append(converted_coord)
            
            # Detailed logging for first few coordinates
            if i < 5 or logger.isEnabledFor(logging.DEBUG):
                logger.info(f"   📍 Point {i:2d}: ARKit({azimuth:.1f}°,{elevation:.1f}°) → Hugin({yaw:.1f}°,{pitch:.1f}°,{roll:.1f}°) → norm({nx:.3f},{ny:.3f})")
                
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
                'arkit_system': 'Device-relative, azimuth (0-360°), elevation (-90° to +90°)',
                'hugin_system': 'Spherical panorama, yaw/pitch/roll with equirectangular projection',
                'conversion_notes': [
                    'ARKit azimuth maps directly to Hugin yaw',
                    'ARKit elevation maps directly to Hugin pitch (both positive = looking up)',
                    'Roll assumed to be 0° for panorama capture',
                    'Calibration offset applied to align all coordinates',
                    'EXPERT FIXES APPLIED:',
                    '- CRITICAL: Fixed pitch inversion - removed unnecessary -elevation',
                    '- Fixed nx calculation: added modulo 360 to ensure 0-1 range',
                    '- Fixed ny calculation: inverted vertical mapping (90-elevation)/180',
                    '- Enhanced validation for out-of-bounds normalized coordinates'
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
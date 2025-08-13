#!/usr/bin/env python3
"""
End-to-end test of coordinate conversion through the actual server pipeline.
"""

import sys
import json
import tempfile
import os
from services.coordinate_service import create_coordinate_service

def test_coordinate_service_pipeline():
    """Test the coordinate service with realistic iOS data patterns."""
    
    print("🧪 TESTING COORDINATE SERVICE PIPELINE")
    print("=" * 50)
    
    # Realistic 16-point iOS pattern (Hugin optimized)
    ios_pattern = [
        # Even columns (0°, 90°, 180°, 270°): 3 elevations each
        {'azimuth': 0,   'elevation': 45,  'isCalibrationReference': True},  # East, Up
        {'azimuth': 0,   'elevation': 0,   'isCalibrationReference': False}, # East, Horizon  
        {'azimuth': 0,   'elevation': -45, 'isCalibrationReference': False}, # East, Down
        
        {'azimuth': 90,  'elevation': 45,  'isCalibrationReference': False}, # North, Up
        {'azimuth': 90,  'elevation': 0,   'isCalibrationReference': False}, # North, Horizon
        {'azimuth': 90,  'elevation': -45, 'isCalibrationReference': False}, # North, Down
        
        {'azimuth': 180, 'elevation': 45,  'isCalibrationReference': False}, # West, Up
        {'azimuth': 180, 'elevation': 0,   'isCalibrationReference': False}, # West, Horizon
        {'azimuth': 180, 'elevation': -45, 'isCalibrationReference': False}, # West, Down
        
        {'azimuth': 270, 'elevation': 45,  'isCalibrationReference': False}, # South, Up  
        {'azimuth': 270, 'elevation': 0,   'isCalibrationReference': False}, # South, Horizon
        {'azimuth': 270, 'elevation': -45, 'isCalibrationReference': False}, # South, Down
        
        # Odd columns (45°, 135°, 225°, 315°): horizon only
        {'azimuth': 45,  'elevation': 0,   'isCalibrationReference': False}, # Northeast
        {'azimuth': 135, 'elevation': 0,   'isCalibrationReference': False}, # Northwest
        {'azimuth': 225, 'elevation': 0,   'isCalibrationReference': False}, # Southwest  
        {'azimuth': 315, 'elevation': 0,   'isCalibrationReference': False}, # Southeast
    ]
    
    try:
        # Create coordinate service
        service = create_coordinate_service()
        
        print(f"📊 Testing with {len(ios_pattern)} point iOS pattern...")
        
        # Step 1: Validate iOS data
        validation_result = service.validate_arkit_data(ios_pattern)
        
        print(f"\n✅ Validation Results:")
        print(f"   Valid points: {validation_result['valid_points']}/{validation_result['total_points']}")
        print(f"   Azimuth range: {validation_result['azimuth_range']:.1f}°")
        print(f"   Elevation range: {validation_result['elevation_range']:.1f}°")
        print(f"   Coverage quality: {validation_result['coverage_quality']}")
        
        if validation_result['geometric_issues']:
            print(f"   ⚠️ Issues: {validation_result['geometric_issues']}")
        
        # Step 2: Convert coordinates
        converted_coords = service.convert_arkit_to_hugin(ios_pattern)
        
        print(f"\n🔄 Coordinate Conversion Results:")
        
        # Analyze key compass directions
        key_directions = [
            (0, "EAST"),
            (90, "NORTH"), 
            (180, "WEST"),
            (270, "SOUTH")
        ]
        
        for ios_azimuth, direction in key_directions:
            # Find matching point in converted data
            for coord in converted_coords:
                if coord['arkit_input']['azimuth_calibrated'] == ios_azimuth and coord['arkit_input']['elevation'] == 0:
                    hugin_yaw = coord['hugin_output']['yaw']
                    nx = coord['debug_info']['normalized_x']
                    print(f"   📍 {direction:5s}: iOS {ios_azimuth:3.0f}°↺ → Hugin {hugin_yaw:3.0f}°↻ → equirect x={nx:.3f}")
                    break
        
        # Step 3: Generate debug report
        debug_report = service.generate_debug_report()
        
        print(f"\n📋 Coordinate System Info:")
        coord_info = debug_report['coordinate_system_info']
        print(f"   iOS: {coord_info['ios_arkit_system']}")
        print(f"   Hugin: {coord_info['hugin_system']}")
        
        # Step 4: Validate conversion makes sense
        print(f"\n🔍 Conversion Validation:")
        
        # Check if North (iOS 90°) maps to center (Hugin 0°, x=0.5)
        north_point = next((c for c in converted_coords if c['arkit_input']['azimuth_calibrated'] == 90), None)
        if north_point:
            north_yaw = north_point['hugin_output']['yaw']
            north_x = north_point['debug_info']['normalized_x']
            
            if north_yaw == 0 and abs(north_x - 0.5) < 0.01:
                print("   ✅ NORTH correctly maps to panorama center")
            else:
                print(f"   ❌ NORTH mapping error: yaw={north_yaw}°, x={north_x} (should be 0°, 0.5)")
        
        # Check if coordinate ranges make sense
        yaw_range = [c['hugin_output']['yaw'] for c in converted_coords]
        if min(yaw_range) >= 0 and max(yaw_range) <= 360:
            print("   ✅ Yaw values in valid range [0°, 360°]")
        else:
            print(f"   ❌ Yaw range error: {min(yaw_range):.1f}° to {max(yaw_range):.1f}°")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coordinate_service_pipeline()
    sys.exit(0 if success else 1)
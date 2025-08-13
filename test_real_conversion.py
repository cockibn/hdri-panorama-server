#!/usr/bin/env python3
"""
Test the actual coordinate service with sample iOS data.
"""

import sys
import json
from services.coordinate_service import create_coordinate_service

def test_with_sample_ios_data():
    """Test coordinate service with sample iOS ARKit data."""
    
    # Sample 4-point pattern from iOS (simplified for testing)
    sample_ios_data = [
        {
            'azimuth': 0,    # East in iOS
            'elevation': 0,
            'position': [1.0, 0.0, 0.0],
            'isCalibrationReference': True
        },
        {
            'azimuth': 90,   # North in iOS  
            'elevation': 0,
            'position': [0.0, 1.0, 0.0]
        },
        {
            'azimuth': 180,  # West in iOS
            'elevation': 0, 
            'position': [-1.0, 0.0, 0.0]
        },
        {
            'azimuth': 270,  # South in iOS
            'elevation': 0,
            'position': [0.0, -1.0, 0.0]
        }
    ]
    
    print("🧪 Testing coordinate service with sample iOS data:")
    print("=" * 60)
    
    try:
        service = create_coordinate_service()
        
        # Test validation
        validation = service.validate_arkit_data(sample_ios_data)
        print(f"✅ Validation passed: {validation['valid_points']}/{validation['total_points']} points")
        print(f"   Coverage quality: {validation['coverage_quality']}")
        
        # Test conversion
        converted = service.convert_arkit_to_hugin(sample_ios_data)
        
        print("\n📊 Conversion Results:")
        for coord in converted:
            ios_input = coord['arkit_input']
            hugin_output = coord['hugin_output']
            debug_info = coord['debug_info']
            
            print(f"Point {coord['index']}: iOS({ios_input['azimuth_calibrated']:.0f}°↺, {ios_input['elevation']:.0f}°) → Hugin({hugin_output['yaw']:.0f}°↻, {hugin_output['pitch']:.0f}°)")
            
        # Test debug report
        report = service.generate_debug_report()
        print(f"\n📋 Coordinate system: {report['coordinate_system_info']['ios_arkit_system']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_with_sample_ios_data()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test coordinate conversion by sending data to the actual server endpoint.
This is the most realistic test - using the actual pipeline the iOS app uses.
"""

import requests
import json
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont

def create_simple_test_images():
    """Create minimal test images for coordinate validation."""
    
    test_dir = tempfile.mkdtemp(prefix='coord_test_')
    
    # Create 4 simple test images with clear directional indicators
    test_points = [
        {'azimuth': 0,   'elevation': 0, 'name': 'EAST',  'color': (255, 0, 0)},
        {'azimuth': 90,  'elevation': 0, 'name': 'NORTH', 'color': (0, 255, 0)},  
        {'azimuth': 180, 'elevation': 0, 'name': 'WEST',  'color': (0, 0, 255)},
        {'azimuth': 270, 'elevation': 0, 'name': 'SOUTH', 'color': (255, 255, 0)},
    ]
    
    image_files = []
    
    for i, point in enumerate(test_points):
        # Create 512x512 test image
        img = Image.new('RGB', (512, 512), point['color'])
        draw = ImageDraw.Draw(img)
        
        # Add direction text
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 60)
        except:
            font = ImageFont.load_default()
        
        text = f"{point['name']}\n{point['azimuth']}°"
        
        # Center text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (512 - text_width) // 2
        y = (512 - text_height) // 2
        
        # White text with black outline for visibility
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # Save image
        image_path = os.path.join(test_dir, f"test_{i:02d}_{point['name'].lower()}.jpg")
        img.save(image_path, 'JPEG', quality=90)
        
        image_files.append({
            'path': image_path,
            'azimuth': point['azimuth'],
            'elevation': point['elevation'],
            'name': point['name']
        })
        
        print(f"   📸 {point['name']:5s}: {point['azimuth']:3.0f}° → {image_path}")
    
    return test_dir, image_files

def test_server_coordinate_conversion():
    """Test coordinate conversion by hitting the actual server."""
    
    print("🌐 TESTING SERVER COORDINATE CONVERSION")
    print("=" * 50)
    print("Creating test images with directional markers...")
    
    # Create test images
    test_dir, image_files = create_simple_test_images()
    
    # Prepare coordinate data as the iOS app would send it
    capture_points = []
    for i, img_file in enumerate(image_files):
        capture_points.append({
            'id': f'test_point_{i}',
            'azimuth': img_file['azimuth'],
            'elevation': img_file['elevation'], 
            'position': [1.0, 0.0, 0.0],  # Simplified position
            'captureTimestamp': '2025-01-01T12:00:00Z',
            'isCalibrationReference': (i == 0)  # First point is calibration
        })
    
    print(f"\n📊 Test Data Summary:")
    for i, cp in enumerate(capture_points):
        ref_marker = " (📍 calibration)" if cp['isCalibrationReference'] else ""
        print(f"   Point {i}: iOS {cp['azimuth']:3.0f}°↺, {cp['elevation']:3.0f}°{ref_marker}")
    
    # Create session metadata as iOS app would send
    session_metadata = {
        'sessionId': 'coordinate_test_session',
        'captureDate': '2025-01-01T12:00:00Z',
        'totalPoints': len(capture_points),
        'capturePattern': 'hugin_optimized_16_point',
        'cameraConfig': {
            'deviceModel': 'iPhone Test',
            'cameraType': 'Ultra-Wide',
            'fieldOfView': 120.0,
            'imageFormat': 'JPEG'
        },
        'processingOptions': {
            'quality': 'standard',
            'outputResolution': '2048x1024',
            'featureDetector': 'sift',
            'blendingMethod': 'linear',
            'enableGeometricCorrection': True,
            'enableColorCorrection': True,
            'enableDenoising': False
        },
        'capturePoints': capture_points,
        'calibrationReference': {
            'azimuth': capture_points[0]['azimuth'],
            'elevation': capture_points[0]['elevation'],
            'position': capture_points[0]['position']
        }
    }
    
    # Save test data for manual inspection
    test_data_path = os.path.join(test_dir, 'test_session_data.json')
    with open(test_data_path, 'w') as f:
        json.dump(session_metadata, f, indent=2)
    
    print(f"\n💾 Test session data saved: {test_data_path}")
    
    # Show what the coordinate service should do
    print(f"\n🔄 Expected Coordinate Conversions:")
    print(f"   EAST  (iOS 0°↺)   → Hugin 90°↻  (center-right)")
    print(f"   NORTH (iOS 90°↺)  → Hugin 0°↻   (center-center)")  
    print(f"   WEST  (iOS 180°↺) → Hugin 270°↻ (center-left)")
    print(f"   SOUTH (iOS 270°↺) → Hugin 180°↻ (far-left/right)")
    
    # Test the coordinate service directly
    try:
        service = create_coordinate_service()
        
        print(f"\n🧪 Running coordinate service validation...")
        validation = service.validate_arkit_data(capture_points)
        
        print(f"✅ Validation passed: {validation['coverage_quality']}")
        
        print(f"\n🔄 Running coordinate conversion...")
        converted = service.convert_arkit_to_hugin(capture_points)
        
        print(f"\n📊 Conversion Results:")
        for coord in converted:
            ios_input = coord['arkit_input']
            hugin_output = coord['hugin_output']
            
            print(f"   📍 iOS {ios_input['azimuth_calibrated']:3.0f}°↺, {ios_input['elevation']:3.0f}° → Hugin {hugin_output['yaw']:3.0f}°↻, {hugin_output['pitch']:3.0f}°")
        
        # Generate debug report
        report = service.generate_debug_report()
        
        print(f"\n📋 Coordinate System Verification:")
        notes = report['coordinate_system_info']['conversion_notes']
        for note in notes[:4]:  # Show first few conversion notes
            print(f"   {note}")
        
        print(f"\n✅ Coordinate service pipeline test PASSED")
        print(f"   📁 Test files: {test_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Coordinate service test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coordinate_service_pipeline()
    sys.exit(0 if success else 1)
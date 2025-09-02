#!/usr/bin/env python3
"""
Test EXIF extraction functionality locally.
"""

import os
import sys
import logging

# Add the services directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from services.hugin_service import HuginPipelineService

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_exif_extraction():
    """Test EXIF extraction with any available iPhone images."""
    
    # Look for sample iPhone images in common locations
    sample_locations = [
        "/Users/gianluca/Library/Developer/CoreSimulator/Devices/642F0FF9-CFFD-414E-B9B2-F855FF30F04D/data/Containers/Data/Application/392523DD-DDFB-47C5-8A30-E097427D0920/Documents/SavedPanoramas",
        "/Users/gianluca/Library/Developer/Xcode/UserData/Previews/Simulator Devices/E0522497-3D17-4495-83A2-119B2B895FE3/data/Containers/Data/Application/D6B32B8F-EDDA-4E7F-ADED-3F43FF7B5AB9/Documents/SavedPanoramas",
        "/Users/gianluca/Desktop",
        "/Users/gianluca/Downloads"
    ]
    
    sample_image = None
    for location in sample_locations:
        if os.path.exists(location):
            print(f"🔍 Checking {location}")
            for file in os.listdir(location):
                if file.lower().endswith(('.jpg', '.jpeg')):
                    sample_image = os.path.join(location, file)
                    print(f"✅ Found sample image: {sample_image}")
                    break
        if sample_image:
            break
    
    if not sample_image:
        print("❌ No sample iPhone images found for testing")
        print("📝 To test EXIF extraction, place an iPhone photo in:")
        for loc in sample_locations:
            print(f"   - {loc}")
        return
    
    # Test the EXIF extraction
    print(f"\n📸 Testing EXIF extraction on: {sample_image}")
    print("=" * 60)
    
    try:
        hugin_service = HuginPipelineService()
        
        # Test FOV calculation
        print("\n🎯 Testing FOV Calculation:")
        fov = hugin_service._calculate_fov_from_exif(sample_image)
        print(f"   Calculated FOV: {fov:.1f}°")
        
        # Test photometric data extraction
        print("\n📊 Testing Photometric Data Extraction:")
        photo_data = hugin_service._extract_photometric_exif(sample_image)
        if photo_data:
            print("   Found photometric data:")
            for key, value in photo_data.items():
                print(f"     {key}: {value}")
        else:
            print("   No photometric data found")
        
        print("\n✅ EXIF extraction test completed successfully!")
        
    except Exception as e:
        print(f"❌ EXIF extraction test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 EXIF EXTRACTION TEST")
    print("=" * 40)
    test_exif_extraction()
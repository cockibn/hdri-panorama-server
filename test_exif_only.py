#!/usr/bin/env python3
"""
Test EXIF extraction methods only (without Hugin dependencies).
"""

import os
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_fov_from_exif(image_path: str) -> float:
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

def extract_photometric_exif(image_path: str):
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

def test_exif_extraction():
    """Test EXIF extraction with any available iPhone images."""
    
    # Look for sample iPhone images
    sample_locations = [
        "/Users/gianluca/Downloads",
        "/Users/gianluca/Desktop",
        "/Users/gianluca/Library/Developer/CoreSimulator/Devices/642F0FF9-CFFD-414E-B9B2-F855FF30F04D/data/Containers/Data/Application/392523DD-DDFB-47C5-8A30-E097427D0920/Documents/SavedPanoramas"
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
        return
    
    # Test the EXIF extraction
    print(f"\n📸 Testing EXIF extraction on: {sample_image}")
    print("=" * 60)
    
    try:
        # Test FOV calculation
        print("\n🎯 Testing FOV Calculation:")
        fov = calculate_fov_from_exif(sample_image)
        print(f"   Final calculated FOV: {fov:.1f}°")
        
        # Test photometric data extraction
        print("\n📊 Testing Photometric Data Extraction:")
        photo_data = extract_photometric_exif(sample_image)
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
    print("🧪 EXIF EXTRACTION TEST (Hugin-Independent)")
    print("=" * 50)
    test_exif_extraction()
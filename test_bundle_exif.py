#!/usr/bin/env python3
"""
Test EXIF extraction on images from a real HDRi 360 Studio bundle.
"""

import os
import struct
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_images_from_bundle(bundle_path, extract_dir):
    """Extract images from HDRi bundle format using correct parsing logic."""
    print(f"📦 Extracting bundle: {bundle_path}")
    
    os.makedirs(extract_dir, exist_ok=True)
    
    with open(bundle_path, 'rb') as f:
        uint8_data = f.read()
    
    # Parse header: find first JPEG marker (0xFF 0xD8)
    header_text = ''
    offset = 0
    
    for i in range(len(uint8_data) - 1):
        if uint8_data[i] == 0xFF and uint8_data[i + 1] == 0xD8:
            header_text = uint8_data[:i].decode('utf-8', errors='ignore')
            offset = i
            break
    
    lines = header_text.strip().split('\n')
    format_line = lines[0] if lines else 'HDRI_BUNDLE_V2_WITH_METADATA'
    image_count = int(lines[1]) if len(lines) > 1 else 16
    
    print(f"📋 Bundle format: {format_line}")
    print(f"📷 Expected images: {image_count}")
    
    # Extract images starting from offset
    extracted_images = []
    current_offset = offset
    
    # Look for JPEG markers and extract images
    i = 0
    while current_offset < len(uint8_data) - 1 and i < image_count:
        # Find JPEG start (0xFF 0xD8)
        jpeg_start = current_offset
        if uint8_data[jpeg_start] != 0xFF or uint8_data[jpeg_start + 1] != 0xD8:
            # Find next JPEG start
            found = False
            for j in range(current_offset, len(uint8_data) - 1):
                if uint8_data[j] == 0xFF and uint8_data[j + 1] == 0xD8:
                    jpeg_start = j
                    found = True
                    break
            if not found:
                break
        
        # Find JPEG end (0xFF 0xD9)
        jpeg_end = None
        for j in range(jpeg_start + 2, len(uint8_data) - 1):
            if uint8_data[j] == 0xFF and uint8_data[j + 1] == 0xD9:
                jpeg_end = j + 2  # Include the end marker
                break
        
        if jpeg_end is None:
            print(f"   ⚠️ Could not find end marker for image {i+1}")
            break
        
        # Extract image data
        image_data = uint8_data[jpeg_start:jpeg_end]
        image_size = len(image_data)
        
        # Save image
        image_filename = f"img_{i+1:02d}.jpg"
        image_path = os.path.join(extract_dir, image_filename)
        
        with open(image_path, 'wb') as img_file:
            img_file.write(image_data)
        
        extracted_images.append(image_path)
        print(f"   ✅ Extracted: {image_filename} ({image_size:,} bytes)")
        
        current_offset = jpeg_end
        i += 1
    
    print(f"📊 Total extracted: {len(extracted_images)} images")
    return extracted_images

def calculate_fov_from_exif(image_path: str) -> float:
    """Calculate field of view from EXIF focal length and sensor data."""
    try:
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
            
            return fov_degrees
        
        elif focal_length:
            # Fallback to research-based iPhone ultra-wide sensor width
            sensor_width = 4.88  # mm (research-based iPhone ultra-wide sensor width)
            fov_radians = 2 * math.atan(sensor_width / (2 * focal_length))
            fov_degrees = math.degrees(fov_radians)
            
            logger.info(f"📸 Research-based FOV: {focal_length:.2f}mm focal length + {sensor_width:.2f}mm sensor → {fov_degrees:.1f}°")
            
            return fov_degrees
                
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

def test_bundle_exif():
    """Test EXIF extraction on a real HDRi bundle."""
    
    bundle_path = "/Users/gianluca/Downloads/hdri360_original_with_metadata_unknown_efa6d58b.zip"
    extract_dir = "/tmp/bundle_exif_test"
    
    if not os.path.exists(bundle_path):
        print(f"❌ Bundle not found: {bundle_path}")
        return
    
    try:
        # Extract images from bundle
        extracted_images = extract_images_from_bundle(bundle_path, extract_dir)
        
        if not extracted_images:
            print("❌ No images extracted from bundle")
            return
        
        # Test EXIF extraction on first few images
        test_images = extracted_images[:3]  # Test first 3 images
        
        print(f"\n🧪 Testing EXIF extraction on {len(test_images)} images:")
        print("=" * 60)
        
        for i, image_path in enumerate(test_images):
            print(f"\n📷 Image {i+1}: {os.path.basename(image_path)}")
            
            # Test FOV calculation
            fov = calculate_fov_from_exif(image_path)
            print(f"   Calculated FOV: {fov:.1f}°")
            
            # Test photometric data extraction
            photo_data = extract_photometric_exif(image_path)
            if photo_data:
                print("   Photometric data:")
                for key, value in photo_data.items():
                    print(f"     {key}: {value}")
        
        print(f"\n✅ Bundle EXIF extraction test completed!")
        print(f"📁 Extracted images available in: {extract_dir}")
        
    except Exception as e:
        print(f"❌ Bundle EXIF test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 BUNDLE EXIF EXTRACTION TEST")
    print("=" * 40)
    test_bundle_exif()
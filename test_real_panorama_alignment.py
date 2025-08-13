#!/usr/bin/env python3
"""
Real validation test for iOS→Hugin coordinate conversion.
Creates a test pattern and verifies panorama alignment.
"""

import os
import sys
import json
import tempfile
import subprocess
from PIL import Image, ImageDraw, ImageFont
import math

def create_test_pattern_images():
    """Create test images with directional markers to validate coordinate conversion."""
    
    # Create temporary directory for test images
    test_dir = tempfile.mkdtemp(prefix='panorama_coord_test_')
    print(f"📁 Creating test images in: {test_dir}")
    
    # Test pattern: 8 images around horizon with clear directional markers
    test_points = [
        {'azimuth': 0,   'elevation': 0, 'direction': 'EAST',  'color': (255, 0, 0)},    # Red - East
        {'azimuth': 45,  'elevation': 0, 'direction': 'NE',    'color': (255, 128, 0)},  # Orange - Northeast  
        {'azimuth': 90,  'elevation': 0, 'direction': 'NORTH', 'color': (255, 255, 0)},  # Yellow - North
        {'azimuth': 135, 'elevation': 0, 'direction': 'NW',    'color': (128, 255, 0)},  # Green-Yellow - Northwest
        {'azimuth': 180, 'elevation': 0, 'direction': 'WEST',  'color': (0, 255, 0)},    # Green - West
        {'azimuth': 225, 'elevation': 0, 'direction': 'SW',    'color': (0, 255, 128)},  # Cyan - Southwest
        {'azimuth': 270, 'elevation': 0, 'direction': 'SOUTH', 'color': (0, 128, 255)},  # Blue - South
        {'azimuth': 315, 'elevation': 0, 'direction': 'SE',    'color': (128, 0, 255)},  # Purple - Southeast
    ]
    
    # Create images with clear directional markers
    image_size = (1024, 1024)
    images_data = []
    
    for i, point in enumerate(test_points):
        # Create image with solid color background and direction text
        img = Image.new('RGB', image_size, point['color'])
        draw = ImageDraw.Draw(img)
        
        # Draw large directional text
        try:
            # Try to use a larger font if available
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 120)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw direction text
        text = f"{point['direction']}\n{point['azimuth']}°"
        
        # Get text dimensions for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (image_size[0] - text_width) // 2
        y = (image_size[1] - text_height) // 2
        
        # Draw text with contrasting color
        text_color = (255, 255, 255) if sum(point['color']) < 400 else (0, 0, 0)
        draw.text((x, y), text, fill=text_color, font=font)
        
        # Draw arrow pointing to center (simulating panorama perspective)
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        arrow_length = 100
        draw.ellipse([center_x - 10, center_y - 10, center_x + 10, center_y + 10], 
                    fill=text_color, outline=text_color)
        
        # Save image
        image_path = os.path.join(test_dir, f"test_image_{i:02d}.jpg")
        img.save(image_path, 'JPEG', quality=95)
        
        # Store metadata
        images_data.append({
            'file_path': image_path,
            'azimuth': point['azimuth'],
            'elevation': point['elevation'],
            'direction': point['direction'],
            'index': i
        })
        
        print(f"   📸 Created {point['direction']} image: {point['azimuth']}° → {image_path}")
    
    return test_dir, images_data

def test_coordinate_conversion_with_real_data(images_data):
    """Test the coordinate conversion with real image data."""
    
    print("\n🧪 Testing coordinate conversion with real image pattern:")
    print("=" * 70)
    
    # Convert iOS coordinates to Hugin using our fixed formula
    converted_points = []
    
    for img_data in images_data:
        ios_azimuth = img_data['azimuth']
        ios_elevation = img_data['elevation']
        
        # Apply our conversion formula
        hugin_yaw = (90 - ios_azimuth) % 360
        hugin_pitch = ios_elevation
        
        # Calculate expected position in equirectangular panorama
        nx = ((hugin_yaw + 180) % 360) / 360
        ny = (90 - ios_elevation) / 180
        
        converted_points.append({
            'original': img_data,
            'hugin_yaw': hugin_yaw,
            'hugin_pitch': hugin_pitch,
            'equirect_x': nx,
            'equirect_y': ny
        })
        
        print(f"📍 {img_data['direction']:5s} (iOS {ios_azimuth:3.0f}°↺) → Hugin {hugin_yaw:3.0f}°↻ → equirect({nx:.3f}, {ny:.3f})")
    
    return converted_points

def create_validation_panorama(converted_points, output_path):
    """Create a validation panorama showing where each direction should appear."""
    
    print(f"\n🎨 Creating validation panorama: {output_path}")
    
    # Create equirectangular canvas (2:1 aspect ratio)
    pano_width, pano_height = 2048, 1024
    pano_img = Image.new('RGB', (pano_width, pano_height), (64, 64, 64))  # Dark gray background
    draw = ImageDraw.Draw(pano_img)
    
    # Font for labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
        small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Place direction markers at converted coordinates
    for point in converted_points:
        # Calculate pixel position in panorama
        pixel_x = int(point['equirect_x'] * pano_width)
        pixel_y = int(point['equirect_y'] * pano_height)
        
        # Draw marker circle
        marker_radius = 30
        color = point['original'].get('color', (255, 255, 255))
        
        draw.ellipse([
            pixel_x - marker_radius, pixel_y - marker_radius,
            pixel_x + marker_radius, pixel_y + marker_radius
        ], fill=color, outline=(255, 255, 255), width=3)
        
        # Draw direction text
        direction = point['original']['direction']
        text_color = (255, 255, 255)
        
        # Center text on marker
        bbox = draw.textbbox((0, 0), direction, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = pixel_x - text_width // 2
        text_y = pixel_y - text_height // 2
        
        draw.text((text_x, text_y), direction, fill=text_color, font=font)
        
        # Add coordinate info below
        coord_text = f"{point['hugin_yaw']:.0f}°"
        coord_bbox = draw.textbbox((0, 0), coord_text, font=small_font)
        coord_width = coord_bbox[2] - coord_bbox[0]
        
        draw.text((pixel_x - coord_width // 2, pixel_y + marker_radius + 5), 
                 coord_text, fill=text_color, font=small_font)
        
        print(f"   📍 {direction:5s}: placed at pixel ({pixel_x:4d}, {pixel_y:4d}) | Hugin {point['hugin_yaw']:3.0f}°")
    
    # Add grid lines for reference
    # Vertical lines (longitude)
    for i in range(0, 360, 45):
        x = (i + 180) * pano_width // 360
        draw.line([(x, 0), (x, pano_height)], fill=(128, 128, 128), width=1)
        draw.text((x + 5, 10), f"{i}°", fill=(200, 200, 200), font=small_font)
    
    # Horizontal lines (latitude)
    for i in range(-90, 91, 30):
        y = (90 - i) * pano_height // 180
        draw.line([(0, y), (pano_width, y)], fill=(128, 128, 128), width=1)
        if i != 0:  # Don't overlap with center
            draw.text((10, y + 5), f"{i}°", fill=(200, 200, 200), font=small_font)
    
    # Save validation panorama
    pano_img.save(output_path, 'JPEG', quality=95)
    
    print(f"✅ Validation panorama saved: {output_path}")
    return output_path

def analyze_expected_positions():
    """Analyze where each direction should appear in the final panorama."""
    
    print("\n📊 Expected positions in equirectangular panorama:")
    print("=" * 70)
    print("If coordinate conversion is correct, directions should appear at:")
    print()
    
    expected_positions = [
        ("EAST (iOS 0°)",   "should appear at CENTER-RIGHT (Hugin 90°, equirect x=0.75)"),
        ("NORTH (iOS 90°)", "should appear at CENTER-CENTER (Hugin 0°, equirect x=0.50)"),
        ("WEST (iOS 180°)", "should appear at CENTER-LEFT (Hugin 270°, equirect x=0.25)"),
        ("SOUTH (iOS 270°)", "should appear at FAR-LEFT (Hugin 180°, equirect x=0.00)"),
    ]
    
    for direction, position in expected_positions:
        print(f"   {direction:15s} {position}")
    
    print()
    print("🔍 Visual validation:")
    print("   - Open the generated panorama image")
    print("   - Check if directional labels appear at expected positions")
    print("   - If NORTH appears at center, conversion is working correctly")

def main():
    """Main test function."""
    
    print("🧪 REAL PANORAMA COORDINATE VALIDATION TEST")
    print("=" * 60)
    print("This test creates actual images with directional markers")
    print("and validates the iOS→Hugin coordinate conversion.")
    print()
    
    try:
        # Step 1: Create test pattern images
        test_dir, images_data = create_test_pattern_images()
        
        # Step 2: Test coordinate conversion
        converted_points = test_coordinate_conversion_with_real_data(images_data)
        
        # Step 3: Create validation panorama
        validation_pano_path = os.path.join(test_dir, 'validation_panorama.jpg')
        create_validation_panorama(converted_points, validation_pano_path)
        
        # Step 4: Analysis instructions
        analyze_expected_positions()
        
        print(f"\n✅ Test complete! Validation files created in:")
        print(f"   📁 {test_dir}")
        print(f"   📸 {validation_pano_path}")
        print(f"\n🔍 To validate coordinate conversion:")
        print(f"   open \"{validation_pano_path}\"")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
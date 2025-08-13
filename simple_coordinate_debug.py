#!/usr/bin/env python3
"""
Simple coordinate debug visualization using only PIL.
Creates visual maps showing where images are positioned.
"""

import os
import json
from PIL import Image, ImageDraw, ImageFont

def create_coordinate_debug_image(capture_points, output_path, title="Coordinate Debug"):
    """Create a simple visual debug image showing image positions."""
    
    print(f"🎨 Creating coordinate debug visualization: {output_path}")
    
    # Create canvas
    width, height = 1600, 800
    img = Image.new('RGB', (width, height), (40, 40, 40))  # Dark gray background
    draw = ImageDraw.Draw(img)
    
    # Try to load font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw title
    draw.text((width//2 - 100, 20), title, fill=(255, 255, 255), font=title_font)
    
    # Draw equirectangular grid (360° x 180°)
    grid_left, grid_top = 50, 80
    grid_width, grid_height = width - 100, height - 160
    
    # Grid background
    draw.rectangle([grid_left, grid_top, grid_left + grid_width, grid_top + grid_height], 
                   fill=(60, 60, 60), outline=(100, 100, 100), width=2)
    
    # Draw grid lines
    # Vertical lines (longitude/azimuth)
    for deg in range(0, 361, 45):
        x = grid_left + (deg / 360.0) * grid_width
        draw.line([(x, grid_top), (x, grid_top + grid_height)], fill=(80, 80, 80), width=1)
        if deg <= 315:  # Don't overlap labels
            draw.text((x + 2, grid_top - 15), f"{deg}°", fill=(200, 200, 200), font=font)
    
    # Horizontal lines (latitude/elevation)
    for deg in range(-90, 91, 30):
        y = grid_top + ((90 - deg) / 180.0) * grid_height
        draw.line([(grid_left, y), (grid_left + grid_width, y)], fill=(80, 80, 80), width=1)
        draw.text((5, y - 8), f"{deg}°", fill=(200, 200, 200), font=font)
    
    # Draw reference lines
    # North (should be center after conversion)
    north_x = grid_left + (0.5 * grid_width)  # Center
    draw.line([(north_x, grid_top), (north_x, grid_top + grid_height)], 
              fill=(0, 255, 0), width=3)
    draw.text((north_x + 5, grid_top + 10), "NORTH\n(Center)", fill=(0, 255, 0), font=font)
    
    # East (should be right side after conversion)
    east_x = grid_left + (0.75 * grid_width)  # Right side
    draw.line([(east_x, grid_top), (east_x, grid_top + grid_height)], 
              fill=(255, 0, 0), width=3)
    draw.text((east_x + 5, grid_top + 10), "EAST\n(Right)", fill=(255, 0, 0), font=font)
    
    # West (should be left side after conversion)
    west_x = grid_left + (0.25 * grid_width)  # Left side
    draw.line([(west_x, grid_top), (west_x, grid_top + grid_height)], 
              fill=(0, 0, 255), width=3)
    draw.text((west_x + 5, grid_top + 10), "WEST\n(Left)", fill=(0, 0, 255), font=font)
    
    # Horizon line
    horizon_y = grid_top + (0.5 * grid_height)
    draw.line([(grid_left, horizon_y), (grid_left + grid_width, horizon_y)], 
              fill=(255, 255, 0), width=2)
    draw.text((grid_left + 10, horizon_y + 5), "HORIZON", fill=(255, 255, 0), font=font)
    
    # Colors for different points
    colors = [
        (255, 100, 100),  # Red variations
        (100, 255, 100),  # Green variations  
        (100, 100, 255),  # Blue variations
        (255, 255, 100),  # Yellow variations
        (255, 100, 255),  # Magenta variations
        (100, 255, 255),  # Cyan variations
        (255, 200, 100),  # Orange variations
        (200, 100, 255),  # Purple variations
        (255, 150, 150),  # Light red
        (150, 255, 150),  # Light green
        (150, 150, 255),  # Light blue
        (255, 255, 150),  # Light yellow
        (255, 150, 255),  # Light magenta
        (150, 255, 255),  # Light cyan
        (255, 220, 150),  # Light orange
        (220, 150, 255),  # Light purple
    ]
    
    print(f"\n📊 Processing {len(capture_points)} capture points:")
    
    for i, point in enumerate(capture_points):
        ios_azimuth = point.get('azimuth', 0)
        ios_elevation = point.get('elevation', 0)
        
        # Apply coordinate conversion (our fix)
        hugin_yaw = (90 - ios_azimuth) % 360
        hugin_pitch = ios_elevation
        
        # Calculate equirectangular position
        nx = ((hugin_yaw + 180) % 360) / 360
        ny = (90 - ios_elevation) / 180
        
        # Convert to pixel coordinates
        px = grid_left + (nx * grid_width)
        py = grid_top + (ny * grid_height)
        
        # Use color based on index
        color = colors[i % len(colors)]
        
        # Draw point
        radius = 12
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], 
                    fill=color, outline=(255, 255, 255), width=2)
        
        # Draw point number
        draw.text((px - 6, py - 8), str(i), fill=(255, 255, 255), font=font)
        
        # Log the conversion
        print(f"   📍 Point {i:2d}: iOS({ios_azimuth:6.1f}°↺, {ios_elevation:5.1f}°) → "
              f"Hugin({hugin_yaw:6.1f}°↻, {hugin_pitch:5.1f}°) → "
              f"equirect({nx:.3f}, {ny:.3f}) → pixel({px:.0f}, {py:.0f})")
    
    # Add legend
    legend_y = grid_top + grid_height + 20
    draw.text((grid_left, legend_y), 
              "Legend: Points show CONVERTED positions (after iOS→Hugin coordinate fix)", 
              fill=(255, 255, 255), font=font)
    draw.text((grid_left, legend_y + 20), 
              "✅ If NORTH points (iOS 90°) appear at CENTER (green line), conversion is working", 
              fill=(255, 255, 255), font=font)
    draw.text((grid_left, legend_y + 40), 
              "✅ If EAST points (iOS 0°) appear at RIGHT (red line), conversion is working", 
              fill=(255, 255, 255), font=font)
    
    # Save image
    img.save(output_path, 'PNG')
    print(f"✅ Debug visualization saved: {output_path}")
    return output_path

def integrate_debug_into_coordinate_service():
    """Add debug visualization to coordinate service."""
    
    coord_service_path = "/Users/gianluca/Desktop/HDRi 360 Studio/panorama_server/services/coordinate_service.py"
    
    # Read current service
    with open(coord_service_path, 'r') as f:
        content = f.read()
    
    # Check if debug is already integrated
    if "create_coordinate_debug_image" in content:
        print("✅ Debug already integrated in coordinate_service.py")
        return True
    
    # Find where to insert debug code (after conversion completion)
    insert_point = content.find("logger.info(f\"✅ Coordinate conversion complete: {len(converted_coordinates)} points\")")
    
    if insert_point == -1:
        print("❌ Could not find insertion point in coordinate_service.py")
        return False
    
    # Debug code to insert
    debug_code = '''
        
        # DEBUG: Create coordinate visualization
        try:
            from simple_coordinate_debug import create_coordinate_debug_image
            debug_filename = f"coordinate_debug_{len(converted_coordinates)}_points.png"
            debug_path = f"/tmp/{debug_filename}"
            create_coordinate_debug_image(capture_points, debug_path, 
                                        title=f"Coordinate Debug - {len(converted_coordinates)} Points")
            logger.info(f"🎨 Coordinate debug visualization: {debug_path}")
        except Exception as debug_error:
            logger.warning(f"⚠️ Debug visualization failed: {debug_error}")'''
    
    # Insert debug code
    new_content = content[:insert_point] + debug_code + "\n        " + content[insert_point:]
    
    # Write back to file
    with open(coord_service_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Debug visualization integrated into coordinate_service.py")
    return True

def test_coordinate_debug():
    """Test the coordinate debug visualization with sample data."""
    
    print("🧪 Testing coordinate debug visualization...")
    
    # Sample iOS capture points (16-point Hugin pattern)
    test_points = [
        {'azimuth': 0,   'elevation': 45},   # East Up
        {'azimuth': 0,   'elevation': 0},    # East Horizon  
        {'azimuth': 0,   'elevation': -45},  # East Down
        {'azimuth': 45,  'elevation': 0},    # Northeast
        {'azimuth': 90,  'elevation': 45},   # North Up
        {'azimuth': 90,  'elevation': 0},    # North Horizon
        {'azimuth': 90,  'elevation': -45},  # North Down
        {'azimuth': 135, 'elevation': 0},    # Northwest
        {'azimuth': 180, 'elevation': 45},   # West Up
        {'azimuth': 180, 'elevation': 0},    # West Horizon
        {'azimuth': 180, 'elevation': -45},  # West Down
        {'azimuth': 225, 'elevation': 0},    # Southwest
        {'azimuth': 270, 'elevation': 45},   # South Up
        {'azimuth': 270, 'elevation': 0},    # South Horizon
        {'azimuth': 270, 'elevation': -45},  # South Down
        {'azimuth': 315, 'elevation': 0},    # Southeast
    ]
    
    # Create test debug image
    test_output = "/Users/gianluca/Desktop/HDRi 360 Studio/panorama_server/test_coordinate_debug.png"
    create_coordinate_debug_image(test_points, test_output, 
                                 title="Test iOS→Hugin Coordinate Conversion")
    
    print(f"\n🔍 Validation Instructions:")
    print(f"   Open: {test_output}")
    print(f"   ✅ Points 4,5,6 (North) should appear at GREEN line (center)")
    print(f"   ✅ Points 0,1,2 (East) should appear at RED line (right side)")
    print(f"   ✅ Points 8,9,10 (West) should appear at BLUE line (left side)")
    print(f"   ✅ Points 12,13,14 (South) should appear at far left/right edge")
    
    return test_output

if __name__ == "__main__":
    print("🎨 SIMPLE COORDINATE DEBUG TOOL")
    print("=" * 40)
    
    # Test the visualization
    test_image = test_coordinate_debug()
    
    # Integrate into coordinate service
    integrate_debug_into_coordinate_service()
    
    print(f"\n✅ Debug tool ready!")
    print(f"   📸 Test image: {test_image}")
    print(f"   🔧 Integration: Added to coordinate_service.py")
    print(f"\nNext time you process a panorama, check Railway logs for:")
    print(f"   '🎨 Coordinate debug visualization: /tmp/coordinate_debug_*.png'")
    print(f"   You can download this file from the server to verify positioning!")
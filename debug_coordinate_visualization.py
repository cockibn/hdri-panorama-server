#!/usr/bin/env python3
"""
Debug visualization tool for coordinate positioning.
Creates visual maps showing where images are positioned in the panorama.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_coordinate_debug_image(capture_points, output_path, title="Coordinate Debug"):
    """Create a visual debug image showing image positions in equirectangular space."""
    
    print(f"🎨 Creating coordinate debug visualization: {output_path}")
    
    # Create equirectangular canvas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f'{title} - Image Position Debug', fontsize=16, fontweight='bold')
    
    # === TOP PLOT: Spherical coordinate view ===
    ax1.set_xlim(0, 360)
    ax1.set_ylim(-90, 90)
    ax1.set_xlabel('Azimuth (degrees)', fontsize=12)
    ax1.set_ylabel('Elevation (degrees)', fontsize=12)
    ax1.set_title('Spherical Coordinate Positions (iOS Convention)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add compass labels
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='East (0°)')
    ax1.axvline(x=90, color='green', linestyle='--', alpha=0.5, label='North (90°)')
    ax1.axvline(x=180, color='blue', linestyle='--', alpha=0.5, label='West (180°)')
    ax1.axvline(x=270, color='orange', linestyle='--', alpha=0.5, label='South (270°)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Horizon')
    
    # === BOTTOM PLOT: Equirectangular projection ===
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Normalized X (0=left, 1=right)', fontsize=12)
    ax2.set_ylabel('Normalized Y (0=top, 1=bottom)', fontsize=12)
    ax2.set_title('Equirectangular Projection (After Coordinate Conversion)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add reference lines for equirectangular
    ax2.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Center (North)')
    ax2.axvline(x=0.25, color='blue', linestyle='--', alpha=0.5, label='West')
    ax2.axvline(x=0.75, color='red', linestyle='--', alpha=0.5, label='East')
    ax2.axvline(x=0.0, color='orange', linestyle='--', alpha=0.5, label='South (wrap)')
    ax2.axhline(y=0.5, color='black', linestyle='-', alpha=0.3, label='Horizon')
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(capture_points)))
    
    for i, point in enumerate(capture_points):
        ios_azimuth = point.get('azimuth', 0)
        ios_elevation = point.get('elevation', 0)
        
        # Apply coordinate conversion (our fix)
        hugin_yaw = (90 - ios_azimuth) % 360
        hugin_pitch = ios_elevation
        
        # Calculate equirectangular coordinates
        nx = ((hugin_yaw + 180) % 360) / 360
        ny = (90 - ios_elevation) / 180
        
        # Plot on spherical view (iOS coordinates)
        ax1.scatter(ios_azimuth, ios_elevation, c=[colors[i]], s=200, alpha=0.8, 
                   edgecolors='black', linewidth=2)
        ax1.annotate(f'{i}', (ios_azimuth, ios_elevation), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold', color='white')
        
        # Plot on equirectangular view (converted coordinates)
        ax2.scatter(nx, ny, c=[colors[i]], s=200, alpha=0.8, 
                   edgecolors='black', linewidth=2)
        ax2.annotate(f'{i}', (nx, ny), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold', color='white')
        
        print(f"   📍 Point {i:2d}: iOS({ios_azimuth:6.1f}°, {ios_elevation:5.1f}°) → "
              f"Hugin({hugin_yaw:6.1f}°, {hugin_pitch:5.1f}°) → "
              f"equirect({nx:.3f}, {ny:.3f})")
    
    # Add legends
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add coordinate conversion info
    conversion_text = (
        "Coordinate Conversion:\n"
        "iOS: counter-clockwise from East\n"
        "Hugin: clockwise from North\n"
        "Formula: yaw = (90° - azimuth) % 360°"
    )
    ax2.text(1.05, 0.5, conversion_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
             fontsize=9, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Debug visualization saved: {output_path}")
    return output_path

def add_coordinate_debug_to_pipeline():
    """Add coordinate debugging to the Hugin service."""
    
    hugin_service_path = "/Users/gianluca/Desktop/HDRi 360 Studio/panorama_server/services/hugin_service.py"
    
    # Check if debug code already exists
    try:
        with open(hugin_service_path, 'r') as f:
            content = f.read()
            
        if "create_coordinate_debug_image" in content:
            print("✅ Coordinate debug already integrated in hugin_service.py")
            return True
            
    except FileNotFoundError:
        print("❌ hugin_service.py not found")
        return False
    
    # Add debug import and function call
    debug_code = '''
# Add after coordinate conversion in pto_gen
try:
    from debug_coordinate_visualization import create_coordinate_debug_image
    debug_path = os.path.join(work_dir, 'coordinate_debug.png')
    create_coordinate_debug_image(converted_coordinates, debug_path, 
                                 title=f"Job {job_id}")
    logger.info(f"🎨 Coordinate debug image: {debug_path}")
except Exception as e:
    logger.warning(f"⚠️ Debug visualization failed: {e}")
'''
    
    print("📝 Manual integration needed:")
    print("   Add coordinate debug visualization to hugin_service.py")
    print("   Insert after coordinate conversion in pto_gen function")
    print(f"   Debug code: {debug_code}")
    
    return False

def test_coordinate_debug():
    """Test the coordinate debug visualization with sample data."""
    
    # Sample iOS capture points (typical pattern)
    test_points = [
        {'azimuth': 0,   'elevation': 0,   'index': 0},   # East
        {'azimuth': 45,  'elevation': 0,   'index': 1},   # Northeast
        {'azimuth': 90,  'elevation': 0,   'index': 2},   # North
        {'azimuth': 135, 'elevation': 0,   'index': 3},   # Northwest
        {'azimuth': 180, 'elevation': 0,   'index': 4},   # West
        {'azimuth': 225, 'elevation': 0,   'index': 5},   # Southwest
        {'azimuth': 270, 'elevation': 0,   'index': 6},   # South
        {'azimuth': 315, 'elevation': 0,   'index': 7},   # Southeast
        {'azimuth': 0,   'elevation': 45,  'index': 8},   # East Up
        {'azimuth': 90,  'elevation': 45,  'index': 9},   # North Up
        {'azimuth': 180, 'elevation': 45,  'index': 10},  # West Up
        {'azimuth': 270, 'elevation': 45,  'index': 11},  # South Up
        {'azimuth': 0,   'elevation': -45, 'index': 12},  # East Down
        {'azimuth': 90,  'elevation': -45, 'index': 13},  # North Down
        {'azimuth': 180, 'elevation': -45, 'index': 14},  # West Down
        {'azimuth': 270, 'elevation': -45, 'index': 15},  # South Down
    ]
    
    # Create test debug image
    test_output = "/Users/gianluca/Desktop/HDRi 360 Studio/panorama_server/test_coordinate_debug.png"
    create_coordinate_debug_image(test_points, test_output, 
                                 title="Test Coordinate Conversion")
    
    print(f"\n🔍 Test coordinate debug created: {test_output}")
    print("   Open this image to verify:")
    print("   - North (90°) should appear at center (x=0.5) in bottom plot")
    print("   - East (0°) should appear at right side (x=0.75) in bottom plot")
    print("   - West (180°) should appear at left side (x=0.25) in bottom plot")
    
    return test_output

if __name__ == "__main__":
    print("🧪 COORDINATE DEBUG VISUALIZATION TOOL")
    print("=" * 50)
    
    # Test the visualization
    test_image = test_coordinate_debug()
    
    # Try to add to pipeline
    add_coordinate_debug_to_pipeline()
    
    print(f"\n✅ Debug tool ready!")
    print(f"   Test image: {test_image}")
    print(f"   Integration: Manual step needed in hugin_service.py")
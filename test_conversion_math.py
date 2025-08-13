#!/usr/bin/env python3
"""
Test the coordinate conversion math directly.
"""

def test_ios_to_hugin_conversion():
    """Test the iOS → Hugin coordinate conversion formula."""
    
    # Test cases: (ios_azimuth, ios_elevation, expected_description)
    test_cases = [
        (0,    0,   "East (iOS) → North (Hugin)"),
        (90,   0,   "North (iOS) → East (Hugin)"),  
        (180,  0,   "West (iOS) → South (Hugin)"),
        (270,  0,   "South (iOS) → West (Hugin)"),
        (45,   0,   "Northeast (iOS) → Southeast (Hugin)"),
        (135,  0,   "Northwest (iOS) → Southwest (Hugin)"),
        (0,    45,  "East+Up (iOS) → North+Up (Hugin)"),
        (0,    -45, "East+Down (iOS) → North+Down (Hugin)"),
    ]
    
    print("🧪 Testing iOS → Hugin coordinate conversion formula:")
    print("Formula: yaw = (90° - azimuth) % 360°")
    print("=" * 70)
    
    for ios_azimuth, ios_elevation, description in test_cases:
        # Apply the conversion formula
        hugin_yaw = (90 - ios_azimuth) % 360
        hugin_pitch = ios_elevation  # Direct mapping
        
        # Calculate normalized equirectangular coordinates
        nx = ((hugin_yaw + 180) % 360) / 360
        ny = (90 - ios_elevation) / 180
        
        print(f"📍 {description}")
        print(f"   iOS: {ios_azimuth:3.0f}°↺, {ios_elevation:3.0f}° → Hugin: {hugin_yaw:3.0f}°↻, {hugin_pitch:3.0f}°")
        print(f"   Equirectangular: ({nx:.3f}, {ny:.3f})")
        print()
    
    print("Legend:")
    print("  ↺ = counter-clockwise from East (iOS mathematical convention)")
    print("  ↻ = clockwise from North (Hugin navigation convention)")
    print()
    
    # Test wraparound cases
    print("🔄 Testing wraparound cases:")
    wraparound_cases = [
        (350, "iOS 350° → Hugin 100°"),
        (10,  "iOS 10° → Hugin 80°"),
        (0,   "iOS 0° → Hugin 90°"),
        (360, "iOS 360° → Hugin 90°"),
    ]
    
    for ios_az, desc in wraparound_cases:
        hugin_yaw = (90 - ios_az) % 360
        print(f"   {desc}: {hugin_yaw}°")

if __name__ == "__main__":
    test_ios_to_hugin_conversion()
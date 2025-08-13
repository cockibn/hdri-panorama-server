#!/usr/bin/env python3
"""
Test script to verify the coordinate system fix.
Tests the conversion from iOS mathematical coordinates to Hugin navigation coordinates.
"""

def test_coordinate_conversion():
    """Test the coordinate conversion with known patterns."""
    
    # Test cases: iOS mathematical coordinates
    test_cases = [
        # (ios_azimuth, ios_elevation, expected_hugin_yaw, description)
        (0,    0,   90,  "East → North"),      # East in iOS → North in Hugin
        (90,   0,   0,   "North → East"),     # North in iOS → East in Hugin  
        (180,  0,   270, "West → South"),     # West in iOS → South in Hugin
        (270,  0,   180, "South → West"),     # South in iOS → West in Hugin
        (360,  0,   90,  "East (360°) → North"), # Wraparound case
        (45,   0,   45,  "Northeast → Southeast"),
        (135,  0,   315, "Northwest → Southwest"),
    ]
    
    print("🧪 Testing iOS → Hugin coordinate conversion:")
    print("=" * 70)
    
    for ios_azimuth, ios_elevation, expected_yaw, description in test_cases:
        # Apply the conversion formula
        hugin_yaw = (90 - ios_azimuth) % 360
        hugin_pitch = ios_elevation  # Direct mapping
        
        # Calculate normalized coordinates
        nx = ((hugin_yaw + 180) % 360) / 360
        ny = (90 - ios_elevation) / 180
        
        # Check if conversion matches expected
        is_correct = hugin_yaw == expected_yaw
        status = "✅" if is_correct else "❌"
        
        print(f"{status} {description}")
        print(f"   iOS: {ios_azimuth:3.0f}°↺, {ios_elevation:3.0f}° → Hugin: {hugin_yaw:3.0f}°↻, {hugin_pitch:3.0f}° → norm({nx:.3f}, {ny:.3f})")
        if not is_correct:
            print(f"   Expected: {expected_yaw}°, Got: {hugin_yaw}°")
        print()
    
    print("Legend: ↺ = counter-clockwise (iOS), ↻ = clockwise (Hugin)")

if __name__ == "__main__":
    test_coordinate_conversion()
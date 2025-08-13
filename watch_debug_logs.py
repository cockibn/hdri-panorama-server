#!/usr/bin/env python3
"""
Watch Railway logs for coordinate debug information.
Helps monitor coordinate conversion in real-time.
"""

import subprocess
import re
import time

def watch_coordinate_debug():
    """Watch Railway logs for coordinate debug messages."""
    
    print("🔍 WATCHING RAILWAY LOGS FOR COORDINATE DEBUG")
    print("=" * 50)
    print("Monitoring for coordinate conversion activity...")
    print("Upload a panorama from iOS app to see debug output!")
    print()
    
    try:
        # Start watching logs
        proc = subprocess.Popen(['railway', 'logs', '--follow'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True, 
                               bufsize=1)
        
        debug_image_pattern = re.compile(r'🎨 Coordinate debug visualization: (.+\.png)')
        coordinate_pattern = re.compile(r'📍 Point\s+(\d+): iOS\(([^)]+)\) → Hugin\(([^)]+)\)')
        conversion_pattern = re.compile(r'🔄 Conversion: (.+)')
        
        for line in proc.stdout:
            line = line.strip()
            
            # Check for coordinate debug image
            debug_match = debug_image_pattern.search(line)
            if debug_match:
                debug_path = debug_match.group(1)
                print(f"🎨 DEBUG IMAGE CREATED: {debug_path}")
                print(f"   To download: railway run 'cp {debug_path} /tmp/debug.png && cat /tmp/debug.png > /dev/stdout' > debug_download.png")
                print()
            
            # Check for coordinate conversion details
            coord_match = coordinate_pattern.search(line)
            if coord_match:
                point_num = coord_match.group(1)
                ios_coords = coord_match.group(2)
                hugin_coords = coord_match.group(3)
                print(f"📍 Point {point_num}: {ios_coords} → {hugin_coords}")
            
            # Check for conversion formula details
            conv_match = conversion_pattern.search(line)
            if conv_match:
                conversion_detail = conv_match.group(1)
                print(f"🔄 {conversion_detail}")
            
            # Show coordinate validation results
            if "✅ Coordinate conversion complete" in line:
                print(f"✅ {line}")
                print()
            
            # Show validation quality
            if "📊 ARKit Data Validation Results" in line:
                print(f"📊 VALIDATION START")
            
            if "Coverage quality:" in line:
                print(f"📊 {line}")
                print()
            
            # Show any coordinate-related warnings
            if "⚠️" in line and ("azimuth" in line.lower() or "coordinate" in line.lower()):
                print(f"⚠️  {line}")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopped watching logs")
        proc.terminate()
    except Exception as e:
        print(f"❌ Error watching logs: {e}")

def quick_log_check():
    """Quick check of recent logs for coordinate activity."""
    
    print("🔍 QUICK LOG CHECK FOR COORDINATE ACTIVITY")
    print("=" * 45)
    
    try:
        # Get recent logs
        result = subprocess.run(['railway', 'logs'], 
                               capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print("❌ Failed to get logs")
            return
        
        lines = result.stdout.split('\n')
        
        # Look for coordinate-related activity in recent logs
        coord_activity = []
        debug_images = []
        
        for line in lines:
            if '📍 Point' in line and 'iOS' in line and 'Hugin' in line:
                coord_activity.append(line.strip())
            elif '🎨 Coordinate debug visualization:' in line:
                debug_images.append(line.strip())
        
        if debug_images:
            print("🎨 Recent debug images created:")
            for img_line in debug_images[-3:]:  # Show last 3
                print(f"   {img_line}")
            print()
        
        if coord_activity:
            print("📊 Recent coordinate conversions:")
            for coord_line in coord_activity[-5:]:  # Show last 5
                print(f"   {coord_line}")
            print()
        
        if not debug_images and not coord_activity:
            print("ℹ️  No recent coordinate activity found")
            print("   Try uploading a panorama from the iOS app!")
        
    except Exception as e:
        print(f"❌ Error checking logs: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_log_check()
    else:
        watch_coordinate_debug()
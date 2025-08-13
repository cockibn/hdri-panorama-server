#!/usr/bin/env python3
"""
Download coordinate debug images from Railway server.
"""

import subprocess
import sys
import os

def download_latest_debug_image():
    """Download the latest coordinate debug image from Railway server."""
    
    print("📥 DOWNLOADING COORDINATE DEBUG IMAGE FROM RAILWAY")
    print("=" * 55)
    
    # First, get recent logs to find the debug image path
    print("🔍 Finding latest debug image path...")
    
    try:
        result = subprocess.run(['railway', 'logs'], 
                               capture_output=True, text=True, timeout=15)
        
        if result.returncode != 0:
            print("❌ Failed to get Railway logs")
            return False
        
        # Find the most recent debug image path
        debug_paths = []
        for line in result.stdout.split('\n'):
            if '🎨 Coordinate debug visualization:' in line and '.png' in line:
                # Extract the path
                path_start = line.find('/tmp/')
                if path_start != -1:
                    path_end = line.find('.png', path_start) + 4
                    debug_path = line[path_start:path_end]
                    debug_paths.append(debug_path)
        
        if not debug_paths:
            print("❌ No debug images found in recent logs")
            print("   Try uploading a panorama first!")
            return False
        
        # Use the most recent one
        latest_debug_path = debug_paths[-1]
        print(f"✅ Found latest debug image: {latest_debug_path}")
        
        # Download the file using Railway run
        local_filename = f"coordinate_debug_downloaded_{int(__import__('time').time())}.png"
        
        print(f"📥 Downloading to: {local_filename}")
        
        # Use Railway run to copy the file and output it
        download_cmd = [
            'railway', 'run', 
            f'if [ -f "{latest_debug_path}" ]; then cat "{latest_debug_path}"; else echo "File not found: {latest_debug_path}"; exit 1; fi'
        ]
        
        print(f"🔧 Running: {' '.join(download_cmd)}")
        
        result = subprocess.run(download_cmd, 
                               capture_output=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ Download failed:")
            print(f"   stdout: {result.stdout.decode() if result.stdout else 'None'}")
            print(f"   stderr: {result.stderr.decode() if result.stderr else 'None'}")
            return False
        
        # Save the binary data
        with open(local_filename, 'wb') as f:
            f.write(result.stdout)
        
        # Check if we got a valid PNG file
        if len(result.stdout) < 1000:
            print(f"❌ Downloaded file seems too small ({len(result.stdout)} bytes)")
            print(f"   Content: {result.stdout[:200]}")
            return False
        
        print(f"✅ Downloaded successfully!")
        print(f"   File: {local_filename}")
        print(f"   Size: {len(result.stdout):,} bytes")
        
        # Try to open it
        try:
            subprocess.run(['open', local_filename], timeout=5)
            print(f"🖼️ Opening {local_filename}")
        except:
            print(f"📁 Saved as {local_filename} (couldn't auto-open)")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout while downloading")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def alternative_download_method():
    """Alternative method using railway shell."""
    
    print("\n🔄 TRYING ALTERNATIVE METHOD")
    print("=" * 30)
    
    print("Run this command manually:")
    print("railway run 'find /tmp -name \"coordinate_debug*.png\" -exec ls -la {} \\;'")
    print()
    print("Then download with:")
    print("railway run 'cat /tmp/coordinate_debug_16_points.png' > debug_image.png")

if __name__ == "__main__":
    success = download_latest_debug_image()
    
    if not success:
        alternative_download_method()
    
    print(f"\n🎯 COORDINATE VALIDATION:")
    print("   Look for these patterns in the debug image:")
    print("   ✅ North (iOS 270°) → Hugin 180° → LEFT EDGE (equirect x=0.000)")
    print("   ✅ East (iOS 0°) → Hugin 90° → RIGHT SIDE (equirect x=0.750)") 
    print("   ✅ Points should form organized pattern, not clustered")
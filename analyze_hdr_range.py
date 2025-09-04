#!/usr/bin/env python3
"""
Analyze HDR range and brightness distribution of EXR files.
"""

import numpy as np
import OpenEXR
import Imath
import os
import sys

def analyze_hdr_range(exr_path: str):
    """Analyze the HDR range and distribution of an EXR file."""
    if not os.path.exists(exr_path):
        print(f"❌ File not found: {exr_path}")
        return
        
    print(f"🔍 Analyzing HDR range: {exr_path}")
    print("=" * 60)
    
    try:
        # Load EXR file using OpenEXR
        exr_file = OpenEXR.InputFile(exr_path)
        header = exr_file.header()
        
        # Get image dimensions
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        print(f"📐 Image dimensions: {height} x {width}")
        
        # Read RGB channels
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = exr_file.channels(['R', 'G', 'B'], FLOAT)
        
        # Convert to numpy arrays
        r_channel = np.frombuffer(channels[0], dtype=np.float32).reshape(height, width)
        g_channel = np.frombuffer(channels[1], dtype=np.float32).reshape(height, width)  
        b_channel = np.frombuffer(channels[2], dtype=np.float32).reshape(height, width)
        
        # Stack into RGB image
        hdr_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
        
        print(f"📊 Data type: {hdr_rgb.dtype}")
        print(f"📐 Final shape: {hdr_rgb.shape}")
        
        exr_file.close()
            
        # Analyze HDR range
        min_val = hdr_rgb.min()
        max_val = hdr_rgb.max()
        mean_val = hdr_rgb.mean()
        median_val = np.median(hdr_rgb)
        
        print(f"\n🌈 HDR Range Analysis:")
        print(f"   Minimum value: {min_val:.6f}")
        print(f"   Maximum value: {max_val:.6f}")
        print(f"   Mean value: {mean_val:.6f}")
        print(f"   Median value: {median_val:.6f}")
        print(f"   Dynamic range: {max_val/max(min_val, 1e-6):.1f}x")
        
        # Check HDR characteristics
        pixels_above_1 = np.sum(hdr_rgb > 1.0)
        total_pixels = hdr_rgb.size
        hdr_percentage = (pixels_above_1 / total_pixels) * 100
        
        print(f"\n💡 HDR Characteristics:")
        print(f"   Total pixels: {total_pixels:,}")
        print(f"   Pixels above 1.0: {pixels_above_1:,} ({hdr_percentage:.2f}%)")
        
        # Brightness distribution analysis
        print(f"\n📊 Brightness Distribution:")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(hdr_rgb, p)
            print(f"   {p:2d}th percentile: {val:.6f}")
            
        # Check if values are being clamped
        clamped_pixels = np.sum(hdr_rgb >= 65504)  # Half-float max value
        if clamped_pixels > 0:
            clamp_percentage = (clamped_pixels / total_pixels) * 100
            print(f"\n⚠️ Potential Clamping Detected:")
            print(f"   Pixels at/near max value (65504): {clamped_pixels:,} ({clamp_percentage:.2f}%)")
        else:
            print(f"\n✅ No clamping detected (max value {max_val:.1f} < 65504)")
            
        # Compare to typical HDR ranges
        print(f"\n🏆 HDR Quality Assessment:")
        if max_val > 10000:
            print("   🌟 Excellent HDR range (very bright highlights)")
        elif max_val > 1000:
            print("   🔆 Good HDR range (bright highlights)")  
        elif max_val > 100:
            print("   💡 Moderate HDR range (some bright areas)")
        elif max_val > 10:
            print("   📱 Limited HDR range (smartphone-like)")
        else:
            print("   🔕 Poor HDR range (may be tone-mapped)")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for command line argument
    if len(sys.argv) > 1:
        exr_file = sys.argv[1]
    else:
        # Default to the latest downloaded file
        exr_file = "/Users/gianluca/Downloads/panorama_ee7a68ca-c1f2-49f9-bd88-102c1b7e7e9b.exr"
        
    analyze_hdr_range(exr_file)
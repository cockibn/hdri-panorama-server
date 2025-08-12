#!/usr/bin/env python3
"""
Panorama Recovery Script

Recovers completed panorama from temporary files when the final copy/conversion step failed.
Based on the enblend command from your logs, the panorama should exist in the temp directory.
"""

import os
import cv2
import shutil
from pathlib import Path

def recover_panorama():
    """Recover the completed panorama from temporary files."""
    job_id = "793a3dce-88c1-44b5-b241-1348bf3cd9d5"
    
    # From your logs, the enblend output was: /tmp/blending_service_gna09trm/enblend_output.tif
    possible_locations = [
        "/tmp/blending_service_gna09trm/enblend_output.tif",
        f"/tmp/hugin_pipeline_*/rendered*.tif",
        f"/tmp/*/enblend_output.tif",
        f"/tmp/*/{job_id}*.tif",
        f"/tmp/*/{job_id}*.exr"
    ]
    
    print(f"🔍 Searching for completed panorama for job: {job_id}")
    
    # Search in /tmp directories
    import glob
    
    for pattern in possible_locations:
        files = glob.glob(pattern)
        if files:
            print(f"📁 Found files matching {pattern}:")
            for file_path in files:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                    print(f"   {file_path} ({file_size:.1f}MB)")
                    
                    # Try to load the image to verify it's valid
                    try:
                        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            height, width = img.shape[:2]
                            print(f"   ✅ Valid image: {width}×{height}")
                            
                            # Copy to outputs directory
                            outputs_dir = Path("outputs")
                            outputs_dir.mkdir(exist_ok=True)
                            
                            # Copy as EXR
                            output_exr = outputs_dir / f"{job_id}_panorama.exr"
                            output_preview = outputs_dir / f"{job_id}_preview.jpg"
                            
                            # Save as EXR (high quality)
                            if img.dtype != cv2.CV_32F:
                                if img.dtype == cv2.CV_8U:
                                    img_float = img.astype('float32') / 255.0
                                else:
                                    img_float = img.astype('float32')
                            else:
                                img_float = img
                                
                            success = cv2.imwrite(str(output_exr), img_float, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                            if success:
                                print(f"✅ Saved EXR: {output_exr}")
                            
                            # Create JPEG preview
                            if img.dtype == cv2.CV_32F:
                                preview_img = (img * 255).astype('uint8')
                            else:
                                preview_img = img
                                
                            # Convert BGR to RGB for JPEG
                            if len(preview_img.shape) == 3:
                                preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                                preview_img = cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR)
                            
                            cv2.imwrite(str(output_preview), preview_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            print(f"✅ Saved preview: {output_preview}")
                            
                            return str(output_exr), str(output_preview)
                            
                    except Exception as e:
                        print(f"   ❌ Could not load image: {e}")
    
    print("❌ No recoverable panorama found")
    return None, None

if __name__ == "__main__":
    exr_path, preview_path = recover_panorama()
    if exr_path:
        print("\n🎉 Panorama recovered successfully!")
        print(f"Full quality: {exr_path}")
        print(f"Preview: {preview_path}")
        print(f"\nDirect links:")
        print(f"Preview: http://localhost:5001/v1/panorama/preview/793a3dce-88c1-44b5-b241-1348bf3cd9d5")
        print(f"Download: http://localhost:5001/v1/panorama/result/793a3dce-88c1-44b5-b241-1348bf3cd9d5")
    else:
        print("\n❌ Could not recover panorama - temp files may have been cleaned up")
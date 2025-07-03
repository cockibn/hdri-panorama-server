#!/usr/bin/env python3
"""
Check server logs to see if Hugin is properly initialized
"""

import requests
import json

def check_hugin_status():
    """Check if Hugin is working by looking at server behavior"""
    
    # Test with a request that will trigger processor initialization logging
    try:
        from PIL import Image
        import io
        
        # Create 4 small test images to pass the minimum requirement
        images = []
        for i in range(4):
            test_image = Image.new('RGB', (100, 100), color=('red', 'green', 'blue', 'yellow')[i])
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG', quality=95)
            img_buffer.seek(0)
            images.append(('test_image', ('test.jpg', img_buffer, 'image/jpeg')))
        
        # Create files dict with 4 test images
        files = {
            'session_metadata': (None, json.dumps({
                "test": "hugin_check",
                "capturePoints": [
                    {"azimuth": 0, "elevation": 0},
                    {"azimuth": 90, "elevation": 0},
                    {"azimuth": 180, "elevation": 0},
                    {"azimuth": 270, "elevation": 0}
                ]
            }))
        }
        
        # Add images to files
        for i, (key, value) in enumerate(images):
            files[f'image_{i}'] = value
        
        print("🧪 Testing with 4 test images to trigger processing...")
        
        response = requests.post(
            "https://hdri-panorama-server-production.up.railway.app/v1/panorama/process",
            files=files,
            timeout=30
        )
        
        if response.status_code == 202:
            job_data = response.json()
            job_id = job_data.get('jobId')
            print(f"✅ Job created: {job_id}")
            
            # Monitor processing for a few seconds
            import time
            for i in range(10):
                time.sleep(2)
                
                status_response = requests.get(
                    f"https://hdri-panorama-server-production.up.railway.app/v1/panorama/status/{job_id}",
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    state = status_data.get('state')
                    message = status_data.get('message', '')
                    
                    print(f"   [{i*2}s] {state}: {message}")
                    
                    # Check for Hugin vs OpenCV indicators
                    if "Hugin" in message:
                        print("✅ HUGIN PROCESSOR DETECTED!")
                        if "successful" in message.lower():
                            print("✅ Hugin processing completed successfully")
                        return True
                    elif "OpenCV" in message:
                        print("⚠️  OpenCV fallback detected")
                        return False
                    elif state in ['completed', 'failed']:
                        break
                else:
                    print(f"   Status check failed: {status_response.status_code}")
                    break
            
            # Final status check
            print(f"\n📋 Final status for job {job_id}:")
            final_response = requests.get(
                f"https://hdri-panorama-server-production.up.railway.app/v1/panorama/status/{job_id}",
                timeout=10
            )
            
            if final_response.status_code == 200:
                final_data = final_response.json()
                print(f"   State: {final_data.get('state')}")
                print(f"   Message: {final_data.get('message')}")
                
                if 'qualityMetrics' in final_data:
                    metrics = final_data['qualityMetrics']
                    processor = metrics.get('processor', 'Unknown')
                    print(f"   Processor Used: {processor}")
                    
                    if processor == 'Hugin':
                        print("🎉 HUGIN IS WORKING!")
                        return True
                    elif processor == 'OpenCV':
                        print("⚠️  OpenCV fallback is being used")
                        return False
            
            return None
            
        else:
            print(f"❌ Job creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    print("🔍 Checking Hugin Implementation Status")
    print("=" * 50)
    
    result = check_hugin_status()
    
    print("\n" + "=" * 50)
    if result is True:
        print("✅ HUGIN IS WORKING CORRECTLY!")
        print("   The server is using Hugin for panorama processing")
    elif result is False:
        print("⚠️  HUGIN FALLBACK TO OPENCV")
        print("   The server fell back to OpenCV (Hugin may not be installed)")
    else:
        print("❓ UNCLEAR RESULT")
        print("   Could not determine which processor is being used")

if __name__ == "__main__":
    main()
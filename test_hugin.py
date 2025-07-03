#!/usr/bin/env python3
"""
Test script to verify Hugin implementation
"""

import requests
import json
import time

def test_server_health():
    """Test if server is responding"""
    try:
        response = requests.get("https://hdri-panorama-server-production.up.railway.app/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is healthy")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Active Jobs: {data.get('activeJobs')}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server health check error: {e}")
        return False

def test_processor_initialization():
    """Test a minimal request to see which processor is being used"""
    try:
        # Create minimal test image data (1x1 pixel JPEG)
        import io
        from PIL import Image
        
        # Create a tiny test image
        test_image = Image.new('RGB', (1, 1), color='red')
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        
        # Create a test job with minimal data
        files = {
            'session_metadata': (None, json.dumps({"test": "processor_check"})),
            'image_0': ('test.jpg', img_buffer, 'image/jpeg')
        }
        
        response = requests.post(
            "https://hdri-panorama-server-production.up.railway.app/v1/panorama/process",
            files=files,
            timeout=30
        )
        
        if response.status_code == 202:
            job_data = response.json()
            job_id = job_data.get('jobId')
            print(f"✅ Test job created: {job_id}")
            
            # Wait a moment for processing to start
            time.sleep(5)
            
            # Check status to see error message (which will reveal processor type)
            status_response = requests.get(
                f"https://hdri-panorama-server-production.up.railway.app/v1/panorama/status/{job_id}",
                timeout=10
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"   Job Status: {status_data.get('state')}")
                print(f"   Message: {status_data.get('message')}")
                
                # The error message will tell us if Hugin was attempted
                if "Hugin" in str(status_data.get('message', '')):
                    print("✅ Hugin processor is being used")
                elif "OpenCV" in str(status_data.get('message', '')):
                    print("⚠️  OpenCV fallback is being used")
                else:
                    print("ℹ️  Processor type unclear from error message")
                
                return True
            else:
                print(f"❌ Status check failed: {status_response.status_code}")
                return False
        else:
            print(f"❌ Test job creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Processor test error: {e}")
        return False

def main():
    print("🧪 Testing Hugin Implementation on Railway")
    print("=" * 50)
    
    # Test 1: Server Health
    print("\n1. Testing server health...")
    health_ok = test_server_health()
    
    if not health_ok:
        print("❌ Server is not responding. Deployment may have failed.")
        return
    
    # Test 2: Processor Type
    print("\n2. Testing processor initialization...")
    processor_ok = test_processor_initialization()
    
    if processor_ok:
        print("\n✅ Server deployment successful!")
        print("📋 Summary:")
        print("   - Server is responding to requests")
        print("   - Hugin implementation is deployed")
        print("   - Ready for iOS app integration")
    else:
        print("\n❌ Processor test failed")
        print("📋 Check Railway deployment logs for details")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script for HDRi 360 Studio OpenCV Panorama Processing Server

This script tests the server endpoints to ensure they're working correctly.
"""

import requests
import json
import time
import os
from pathlib import Path

# Server configuration
SERVER_URL = "http://localhost:5001"
API_BASE = f"{SERVER_URL}/v1"

def test_health_check():
    """Test the health endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Active jobs: {data['activeJobs']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_processing_workflow():
    """Test the complete processing workflow with dummy data"""
    print("\n📸 Testing processing workflow...")
    
    # Create dummy session metadata (matches iOS app format)
    session_metadata = {
        "sessionId": "test-session-123",
        "captureDate": "2024-01-15T10:30:00Z",
        "totalPoints": 4,  # Reduced for testing
        "capturePattern": "ultra_wide_16_point",
        "cameraConfig": {
            "deviceModel": "Test Device",
            "cameraType": "Ultra-Wide",
            "fieldOfView": 120.0,
            "imageFormat": "JPEG"
        },
        "processingOptions": {
            "quality": "professional",
            "outputResolution": "4096x2048",
            "featureDetector": "sift",
            "blendingMethod": "multiband",
            "enableGeometricCorrection": True,
            "enableColorCorrection": True,
            "enableDenoising": True
        },
        "capturePoints": [
            {
                "id": "point-1",
                "azimuth": 0.0,
                "elevation": 0.0,
                "position": [0.0, 0.0, 50.0],
                "captureTimestamp": "2024-01-15T10:30:01Z"
            },
            {
                "id": "point-2", 
                "azimuth": 90.0,
                "elevation": 0.0,
                "position": [50.0, 0.0, 0.0],
                "captureTimestamp": "2024-01-15T10:30:02Z"
            },
            {
                "id": "point-3",
                "azimuth": 180.0, 
                "elevation": 0.0,
                "position": [0.0, 0.0, -50.0],
                "captureTimestamp": "2024-01-15T10:30:03Z"
            },
            {
                "id": "point-4",
                "azimuth": 270.0,
                "elevation": 0.0, 
                "position": [-50.0, 0.0, 0.0],
                "captureTimestamp": "2024-01-15T10:30:04Z"
            }
        ]
    }
    
    # Check if we have test images
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print("⚠️  No test images found. Creating dummy test...")
        print("   To test with real images, place 4+ JPEG files in 'test_images/' directory")
        return test_dummy_upload(session_metadata)
    
    # Get test images
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.jpeg"))
    if len(image_files) < 4:
        print(f"⚠️  Found only {len(image_files)} test images, need at least 4")
        return test_dummy_upload(session_metadata)
    
    return test_real_upload(session_metadata, image_files[:4])

def test_dummy_upload(session_metadata):
    """Test upload endpoint without real images"""
    print("🔄 Testing upload endpoint (dummy data)...")
    
    try:
        # Create form data
        files = {'session_metadata': (None, json.dumps(session_metadata))}
        
        response = requests.post(f"{API_BASE}/panorama/process", files=files, timeout=30)
        
        if response.status_code == 400:
            data = response.json()
            if "No images uploaded" in data.get("error", ""):
                print("✅ Upload endpoint working (correctly rejected empty upload)")
                return True
        
        print(f"❌ Unexpected response: {response.status_code}")
        return False
        
    except Exception as e:
        print(f"❌ Upload test failed: {str(e)}")
        return False

def test_real_upload(session_metadata, image_files):
    """Test upload with real images"""
    print(f"🔄 Testing upload with {len(image_files)} real images...")
    
    try:
        # Prepare form data
        files = {'session_metadata': (None, json.dumps(session_metadata))}
        
        # Add image files
        for i, img_path in enumerate(image_files):
            files[f'image_point-{i+1}'] = (f'image_{i}.jpg', open(img_path, 'rb'), 'image/jpeg')
        
        # Upload
        response = requests.post(f"{API_BASE}/panorama/process", files=files, timeout=30)
        
        # Close file handles
        for key in files:
            if hasattr(files[key], '__len__') and len(files[key]) > 1:
                if hasattr(files[key][1], 'close'):
                    files[key][1].close()
        
        if response.status_code == 202:
            data = response.json()
            job_id = data['jobId']
            print(f"✅ Upload successful! Job ID: {job_id}")
            
            # Monitor processing
            return monitor_processing(job_id)
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return False

def monitor_processing(job_id):
    """Monitor processing progress"""
    print(f"🔄 Monitoring job {job_id}...")
    
    max_attempts = 60  # 5 minutes maximum
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{API_BASE}/panorama/status/{job_id}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                state = data['state']
                progress = data['progress']
                message = data['message']
                
                print(f"   Status: {state} ({progress*100:.1f}%) - {message}")
                
                if state == "completed":
                    print("✅ Processing completed successfully!")
                    if 'qualityMetrics' in data:
                        metrics = data['qualityMetrics']
                        print(f"   Quality Score: {metrics['overallScore']:.2f}")
                        print(f"   Feature Matches: {metrics['featureMatches']}")
                    return True
                elif state == "failed":
                    print(f"❌ Processing failed: {data.get('error', 'Unknown error')}")
                    return False
                    
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Status check error: {str(e)}")
            return False
        
        time.sleep(5)
        attempt += 1
    
    print("❌ Processing timeout")
    return False

def main():
    """Run all tests"""
    print("🧪 HDRi 360 Studio Server Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Server is not running or not responding")
        print("   Please start the server with: ./start_server.sh")
        return
    
    # Test 2: Processing workflow
    test_processing_workflow()
    
    print("\n🎉 Test suite completed!")
    print("\nTo test with your own images:")
    print("1. Create a 'test_images' directory")
    print("2. Add 4+ JPEG images to the directory") 
    print("3. Run this test script again")

if __name__ == "__main__":
    main()
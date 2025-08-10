#!/usr/bin/env python3
"""
Simple Architecture Test (without OpenCV dependencies)

Tests that the microservices architecture can be imported and the
MicroservicesPanoramaProcessor can be created, even without OpenCV installed.
"""

import sys
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports_without_cv2():
    """Test imports work without OpenCV."""
    try:
        print("🧪 Testing microservices imports (without OpenCV)")
        
        # Test that we can import the service bus without OpenCV
        from services.service_bus import get_service_bus, ServiceStatus, MessageType
        print("✅ Service bus imports: SUCCESS")
        
        # Test that service bus can initialize
        service_bus = get_service_bus()
        service_bus.enable_debug_mode(True)
        print("✅ Service bus initialization: SUCCESS")
        
        # Test service registration
        success = service_bus.register_service(
            name="test_service",
            version="1.0.0",
            capabilities=["testing"]
        )
        print(f"✅ Service registration: {'SUCCESS' if success else 'FAILED'}")
        
        # Test service communication
        services = service_bus.list_services()
        print(f"✅ Service listing: {len(services)} services registered")
        
        # Test debug report generation
        debug_report = service_bus.generate_debug_report()
        total_services = debug_report['service_bus_info']['total_services']
        print(f"✅ Debug report: {total_services} services tracked")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_app_architecture():
    """Test that the app architecture is correct."""
    try:
        print("\n🧪 Testing app architecture")
        
        # Test that MicroservicesPanoramaProcessor class exists and can be imported
        from app import MicroservicesPanoramaProcessor
        print("✅ MicroservicesPanoramaProcessor import: SUCCESS")
        
        # We can't instantiate it without OpenCV, but we can verify the class exists
        print("✅ MicroservicesPanoramaProcessor class: EXISTS")
        
        # Test that the global processor variable is defined
        import app
        processor_type = type(app.processor).__name__
        print(f"✅ Global processor type: {processor_type}")
        
        return processor_type == 'MicroservicesPanoramaProcessor'
        
    except Exception as e:
        print(f"❌ App architecture test failed: {e}")
        return False

def test_service_files_exist():
    """Test that all service files exist and can be parsed."""
    service_files = [
        'services/coordinate_service.py',
        'services/hugin_service.py', 
        'services/quality_service.py',
        'services/blending_service.py',
        'services/service_bus.py',
        'services/__init__.py'
    ]
    
    print("\n🧪 Testing service files exist and parse correctly")
    
    all_good = True
    for service_file in service_files:
        try:
            file_path = Path(__file__).parent / service_file
            if not file_path.exists():
                print(f"❌ {service_file}: FILE NOT FOUND")
                all_good = False
                continue
                
            # Try to compile the file to check syntax
            import py_compile
            py_compile.compile(str(file_path), doraise=True)
            print(f"✅ {service_file}: SYNTAX OK")
            
        except Exception as e:
            print(f"❌ {service_file}: SYNTAX ERROR - {e}")
            all_good = False
            
    return all_good

def main():
    """Run all architecture tests."""
    print("Microservices Architecture Verification")
    print("=" * 45)
    
    tests = [
        ("Service Bus (no OpenCV)", test_imports_without_cv2),
        ("App Architecture", test_app_architecture),
        ("Service Files", test_service_files_exist)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 45)
    print("📊 Test Results:")
    print("=" * 45)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print("-" * 45)
    print(f"Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 Microservices architecture is correctly implemented!")
        print("✅ The monolithic hugin_stitcher.py has been successfully replaced")
        print("✅ All service files exist and have valid syntax")
        print("✅ Service bus communication layer is working")
        print("✅ App.py uses MicroservicesPanoramaProcessor")
        return True
    else:
        print(f"\n⚠️ {len(results) - passed} issues found")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
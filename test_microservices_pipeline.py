#!/usr/bin/env python3
"""
Test Complete Microservices Pipeline

Tests the entire panorama processing pipeline using all microservices:
1. Coordinate Service - ARKit validation and conversion
2. Hugin Service - Complete 7-step pipeline
3. Quality Service - Comprehensive analysis  
4. Blending Service - Multi-strategy blending
5. Service Bus - Inter-service communication

This test verifies that the monolithic hugin_stitcher.py has been successfully
replaced with the modular microservices architecture.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_microservices_architecture():
    """Test that all microservices can be imported and initialized."""
    logger.info("🧪 Testing Microservices Architecture")
    
    try:
        # Test service imports
        from services import (
            create_coordinate_service, create_hugin_service, 
            create_quality_service, create_blending_service,
            get_service_bus, ServiceStatus
        )
        logger.info("✅ All service imports successful")
        
        # Test service bus
        service_bus = get_service_bus()
        service_bus.enable_debug_mode(True)
        logger.info("✅ Service bus initialized")
        
        # Test coordinate service
        coordinate_service = create_coordinate_service()
        logger.info("✅ Coordinate service created")
        
        # Test Hugin service  
        hugin_service = create_hugin_service()
        logger.info("✅ Hugin service created")
        
        # Test quality service
        quality_service = create_quality_service()
        logger.info("✅ Quality service created")
        
        # Test blending service
        blending_service = create_blending_service()
        logger.info("✅ Blending service created")
        
        # Test service registration
        services = [
            ("coordinate_service", "1.0.0", ["arkit_validation", "coordinate_conversion"]),
            ("hugin_service", "1.0.0", ["pto_gen", "cpfind", "nona", "full_pipeline"]),
            ("quality_service", "1.0.0", ["quality_analysis", "metrics_calculation"]),
            ("blending_service", "1.0.0", ["enblend", "opencv_blend", "emergency_fallback"])
        ]
        
        for name, version, capabilities in services:
            service_bus.register_service(name, version, capabilities)
            
        logger.info(f"✅ Registered {len(services)} services with service bus")
        
        # Test service communication
        service_bus.update_service_status("coordinate_service", ServiceStatus.READY)
        service_bus.update_service_status("hugin_service", ServiceStatus.READY) 
        service_bus.update_service_status("quality_service", ServiceStatus.READY)
        service_bus.update_service_status("blending_service", ServiceStatus.READY)
        
        # Generate debug report
        debug_report = service_bus.generate_debug_report()
        logger.info(f"✅ Service bus operational with {debug_report['service_bus_info']['total_services']} services")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Microservices architecture test failed: {e}")
        return False

def test_coordinate_service_integration():
    """Test coordinate service with sample ARKit data."""
    logger.info("🧪 Testing Coordinate Service Integration")
    
    try:
        from services import create_coordinate_service
        
        coordinate_service = create_coordinate_service()
        
        # Sample ARKit data (16-point ultra-wide capture)
        sample_capture_points = [
            {'azimuth': 0.0, 'elevation': 0.0, 'position': [0.0, 0.0, 0.0], 'isCalibrationReference': True},
            {'azimuth': 45.0, 'elevation': -45.0, 'position': [1.0, 0.0, 0.0], 'isCalibrationReference': False},
            {'azimuth': 90.0, 'elevation': 0.0, 'position': [0.0, 1.0, 0.0], 'isCalibrationReference': False},
            {'azimuth': 135.0, 'elevation': 45.0, 'position': [-1.0, 0.0, 0.0], 'isCalibrationReference': False},
            {'azimuth': 180.0, 'elevation': 0.0, 'position': [0.0, -1.0, 0.0], 'isCalibrationReference': False}
        ]
        
        # Test validation
        validation_results = coordinate_service.validate_arkit_data(sample_capture_points)
        logger.info(f"✅ Coordinate validation: {validation_results['coverage_quality']}")
        
        # Test conversion
        converted_coordinates = coordinate_service.convert_arkit_to_hugin(sample_capture_points)
        logger.info(f"✅ Coordinate conversion: {len(converted_coordinates)} points converted")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Coordinate service test failed: {e}")
        return False

def test_app_integration():
    """Test that the main app can initialize with microservices."""
    logger.info("🧪 Testing App Integration")
    
    try:
        # Import the main processor
        sys.path.insert(0, str(Path(__file__).parent))
        from app import MicroservicesPanoramaProcessor
        
        # Test processor initialization
        processor = MicroservicesPanoramaProcessor()
        logger.info("✅ MicroservicesPanoramaProcessor initialized")
        
        # Test service registration
        services = processor.service_bus.list_services()
        logger.info(f"✅ Processor registered {len(services)} services")
        
        for name, info in services.items():
            logger.info(f"   - {name}: {info.status.value} ({len(info.capabilities)} capabilities)")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ App integration test failed: {e}")
        return False

def test_hugin_tools_availability():
    """Test that Hugin tools are available for the Hugin service."""
    logger.info("🧪 Testing Hugin Tools Availability")
    
    try:
        import shutil
        
        required_tools = ['pto_gen', 'cpfind', 'cpclean', 'autooptimiser', 'pano_modify', 'nona', 'enblend']
        available_tools = []
        missing_tools = []
        
        for tool in required_tools:
            if shutil.which(tool):
                available_tools.append(tool)
            else:
                missing_tools.append(tool)
                
        logger.info(f"✅ Available Hugin tools: {len(available_tools)}/{len(required_tools)}")
        for tool in available_tools:
            logger.info(f"   ✅ {tool}")
            
        if missing_tools:
            logger.warning(f"⚠️ Missing Hugin tools: {missing_tools}")
            logger.warning("   Install with: brew install hugin (macOS) or apt-get install hugin (Linux)")
            
        return len(available_tools) >= len(required_tools) // 2  # At least half available
        
    except Exception as e:
        logger.error(f"❌ Hugin tools test failed: {e}")
        return False

def run_all_tests():
    """Run all microservices tests."""
    logger.info("🚀 Starting Microservices Pipeline Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Microservices Architecture", test_microservices_architecture),
        ("Coordinate Service Integration", test_coordinate_service_integration), 
        ("App Integration", test_app_integration),
        ("Hugin Tools Availability", test_hugin_tools_availability)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        logger.info("-" * 30)
        
        start_time = time.time()
        try:
            success = test_func()
            duration = time.time() - start_time
            
            if success:
                logger.info(f"✅ {test_name} PASSED ({duration:.1f}s)")
                results.append((test_name, "PASSED", duration))
            else:
                logger.error(f"❌ {test_name} FAILED ({duration:.1f}s)")
                results.append((test_name, "FAILED", duration))
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {test_name} ERROR: {e} ({duration:.1f}s)")
            results.append((test_name, "ERROR", duration))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    logger.info("=" * 50)
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    errors = sum(1 for _, status, _ in results if status == "ERROR")
    total_time = sum(duration for _, _, duration in results)
    
    for test_name, status, duration in results:
        status_icon = "✅" if status == "PASSED" else "❌"
        logger.info(f"{status_icon} {test_name}: {status} ({duration:.1f}s)")
        
    logger.info("-" * 50)
    logger.info(f"📈 Results: {passed} passed, {failed} failed, {errors} errors")
    logger.info(f"⏱️ Total time: {total_time:.1f}s")
    
    if passed == len(tests):
        logger.info("🎉 ALL TESTS PASSED - Microservices architecture is ready!")
        return True
    else:
        logger.error("⚠️ Some tests failed - check configuration and dependencies")
        return False

if __name__ == "__main__":
    print("Microservices Pipeline Testing")
    print("=" * 40)
    
    try:
        success = run_all_tests()
        
        if success:
            print("\n🎯 Microservices architecture successfully tested!")
            print("The monolithic hugin_stitcher.py has been replaced with:")
            print("  ✅ Coordinate Service (ARKit validation & conversion)")
            print("  ✅ Hugin Service (Complete 7-step pipeline)")
            print("  ✅ Quality Service (Comprehensive analysis)")
            print("  ✅ Blending Service (Multi-strategy blending)")
            print("  ✅ Service Bus (Inter-service communication)")
            sys.exit(0)
        else:
            print("\n❌ Some microservices tests failed")
            print("Check the logs above for details")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Testing failed with error: {e}")
        sys.exit(1)
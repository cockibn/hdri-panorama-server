#!/usr/bin/env python3
"""
Test script for ARKit Coordinate Service

Tests the coordinate service in isolation to debug geometric distortion issues.
This script simulates the exact coordinate data that would come from the iOS app
and validates/converts it using the new microservices architecture.
"""

import json
import logging
import sys
from pathlib import Path

# Add the server directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from services.coordinate_service import create_coordinate_service
from services.service_bus import get_service_bus, ServiceStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data_16_point():
    """Create test data simulating typical 16-point ARKit capture."""
    # Simulates the dynamic capture pattern from iOS app:
    # 3 elevation levels (-45°, 0°, +45°) × 8 azimuth columns (45° spacing)
    # Plus 1 calibration reference point at (0°, 0°)
    
    capture_points = []
    
    # Calibration reference point (horizontal calibration)
    capture_points.append({
        'azimuth': 0.0,
        'elevation': 0.0,
        'position': [0.0, 0.0, 0.0],
        'isCalibrationReference': True,
        'captureIndex': 0
    })
    
    # Generate 15 additional points around calibration
    index = 1
    elevations = [-45.0, 0.0, 45.0]  # 3 levels
    azimuths_per_level = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 directions
    
    for elevation in elevations:
        for azimuth in azimuths_per_level:
            if elevation == 0.0 and azimuth == 0.0:
                continue  # Skip - already added as calibration reference
                
            # Add some realistic variation
            az_variation = azimuth + (index % 3 - 1) * 2.0  # ±2° variation
            el_variation = elevation + (index % 2 - 0.5) * 1.0  # ±0.5° variation
            
            capture_points.append({
                'azimuth': az_variation,
                'elevation': el_variation,
                'position': [
                    2.0 * (index % 3 - 1),  # Some position variation
                    1.5 * (index % 2),
                    0.1 * index
                ],
                'isCalibrationReference': False,
                'captureIndex': index
            })
            index += 1
            
            if len(capture_points) >= 16:  # Stop at 16 points
                break
        if len(capture_points) >= 16:
            break
    
    return capture_points

def create_test_data_problematic():
    """Create test data that simulates known geometric problems."""
    capture_points = []
    
    # Problem 1: All points at same elevation (no spherical coverage)
    for i, azimuth in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
        capture_points.append({
            'azimuth': azimuth,
            'elevation': 0.0,  # All at horizon - BAD
            'position': [i * 0.5, 0.0, 0.0],
            'isCalibrationReference': i == 0,
            'captureIndex': i
        })
    
    # Problem 2: Limited azimuth range (not full 360°)
    for i, azimuth in enumerate([10, 20, 30, 40, 50, 60, 70, 80]):
        capture_points.append({
            'azimuth': azimuth,  # Only 70° range - BAD
            'elevation': 15.0 if i % 2 == 0 else -15.0,
            'position': [i * 0.3, 0.0, 0.0],
            'isCalibrationReference': False,
            'captureIndex': i + 8
        })
    
    return capture_points

def create_test_data_extreme_distortion():
    """Create test data that would cause extreme geometric distortion."""
    capture_points = []
    
    # Extreme problem: Most points clustered, with outliers
    # This simulates what might happen if ARKit coordinate conversion is wrong
    
    # Cluster most points in small area
    for i in range(12):
        capture_points.append({
            'azimuth': 5.0 + i * 2.0,  # Tight cluster: 5° to 27°
            'elevation': -2.0 + i * 0.5,  # Small elevation range: -2° to +3.5°
            'position': [i * 0.1, 0.0, 0.0],
            'isCalibrationReference': i == 0,
            'captureIndex': i
        })
    
    # Add few outliers
    outliers = [
        {'azimuth': 180.0, 'elevation': 80.0},   # Far outlier
        {'azimuth': 270.0, 'elevation': -80.0},  # Another far outlier
        {'azimuth': 350.0, 'elevation': 45.0},   # Different area
        {'azimuth': 90.0, 'elevation': -45.0}    # Yet another area
    ]
    
    for i, outlier in enumerate(outliers):
        capture_points.append({
            'azimuth': outlier['azimuth'],
            'elevation': outlier['elevation'], 
            'position': [10.0 * i, 5.0, 0.0],
            'isCalibrationReference': False,
            'captureIndex': 12 + i
        })
    
    return capture_points

def test_coordinate_service():
    """Test the coordinate service with different scenarios."""
    logger.info("🧪 Testing ARKit Coordinate Service")
    
    # Initialize services
    service_bus = get_service_bus()
    service_bus.enable_debug_mode(True)
    
    coordinate_service = create_coordinate_service()
    
    # Register service
    service_bus.register_service(
        name="test_coordinate_service",
        version="1.0.0", 
        capabilities=["testing", "debugging", "validation"]
    )
    
    test_scenarios = [
        ("16-Point Normal", create_test_data_16_point()),
        ("Problematic Coverage", create_test_data_problematic()),
        ("Extreme Distortion Case", create_test_data_extreme_distortion())
    ]
    
    results = {}
    
    for scenario_name, test_data in test_scenarios:
        logger.info(f"\n🔍 Testing scenario: {scenario_name}")
        logger.info(f"   Test data points: {len(test_data)}")
        
        try:
            service_bus.update_service_status("test_coordinate_service", ServiceStatus.BUSY)
            
            # Step 1: Validation
            logger.info(f"   Running validation...")
            validation_results = coordinate_service.validate_arkit_data(test_data)
            
            # Step 2: Conversion
            logger.info(f"   Running coordinate conversion...")
            converted_coordinates = coordinate_service.convert_arkit_to_hugin(test_data)
            
            # Step 3: Analysis
            coverage_quality = validation_results.get('coverage_quality', 'UNKNOWN')
            geometric_issues = validation_results.get('geometric_issues', [])
            
            logger.info(f"   Results:")
            logger.info(f"     Coverage quality: {coverage_quality}")
            logger.info(f"     Geometric issues: {len(geometric_issues)}")
            logger.info(f"     Converted points: {len(converted_coordinates)}")
            
            if geometric_issues:
                logger.warning(f"   ⚠️ Issues found:")
                for issue in geometric_issues:
                    logger.warning(f"     - {issue}")
                    
            # Store results
            results[scenario_name] = {
                'validation': validation_results,
                'conversion_count': len(converted_coordinates),
                'issues': geometric_issues,
                'coverage_quality': coverage_quality,
                'sample_conversion': converted_coordinates[:3] if converted_coordinates else []
            }
            
            service_bus.update_service_status("test_coordinate_service", ServiceStatus.READY)
            
        except Exception as e:
            logger.error(f"   ❌ Test failed for {scenario_name}: {e}")
            service_bus.update_service_status("test_coordinate_service", ServiceStatus.ERROR)
            results[scenario_name] = {'error': str(e)}
    
    # Generate comprehensive report
    logger.info(f"\n📊 Test Summary:")
    for scenario, result in results.items():
        if 'error' in result:
            logger.info(f"   {scenario}: FAILED - {result['error']}")
        else:
            quality = result['coverage_quality']
            issues = len(result['issues'])
            logger.info(f"   {scenario}: {quality} quality, {issues} issues")
    
    # Service bus report
    logger.info(f"\n📡 Service Bus Report:")
    bus_report = service_bus.generate_debug_report()
    logger.info(f"   Total services: {bus_report['service_bus_info']['total_services']}")
    logger.info(f"   Messages logged: {bus_report['service_bus_info']['total_messages_logged']}")
    
    return results

def save_test_results(results, output_file="coordinate_test_results.json"):
    """Save test results to file for analysis."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"💾 Test results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    print("ARKit Coordinate Service Testing")
    print("=" * 40)
    
    try:
        results = test_coordinate_service()
        save_test_results(results)
        
        # Summary
        print(f"\n🎯 Testing complete!")
        print(f"   Check the logs above for detailed analysis")
        print(f"   Results saved to coordinate_test_results.json")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)
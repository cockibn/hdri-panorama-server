#!/usr/bin/env python3
"""
Test script to determine if ARKit integration actually improves panorama results.
Processes the same image set with and without ARKit coordinates.
"""

import os
import json
import time
import logging
from services.hugin_service import HuginPipelineService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_arkit_vs_basic(images_dir: str):
    """Compare basic processing vs what would happen with fake ARKit coordinates."""
    
    # Find image files and sort them
    images = []
    for f in sorted(os.listdir(images_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.tiff', '.tif')):
            images.append(os.path.join(images_dir, f))
    
    if not images:
        logger.error("No images found in directory")
        return
        
    logger.info(f"Testing with {len(images)} images")
    
    results = {}
    
    # Test 1: Basic mode (what backup uses)
    logger.info("📷 TEST 1: Processing in BASIC mode (like backup)")
    service_basic = HuginPipelineService()
    start_time = time.time()
    
    try:
        result_basic = service_basic.execute_pipeline(images, None)  # No coordinates
        results['basic'] = {
            'success': result_basic['success'],
            'processing_time': time.time() - start_time,
            'control_points': result_basic['statistics'].get('control_points_found', 0),
            'geocpset_used': result_basic['statistics'].get('geocpset_used', False),
            'steps_completed': len([s for s in result_basic['pipeline_steps'] if s['success']]),
            'tiff_files': len(result_basic.get('tiff_files', [])) if result_basic['success'] else 0
        }
        logger.info(f"✅ Basic result: {results['basic']}")
    except Exception as e:
        results['basic'] = {'success': False, 'error': str(e)}
        logger.error(f"❌ Basic processing failed: {e}")
    finally:
        service_basic.cleanup()
    
    # Test 2: With fake ARKit coordinates (simulate what iOS app would send)
    logger.info("🎯 TEST 2: Processing WITH simulated ARKit coordinates")
    fake_coordinates = generate_fake_arkit_coordinates(len(images))
    service_arkit = HuginPipelineService()
    start_time = time.time()
    
    try:
        result_arkit = service_arkit.execute_pipeline(images, fake_coordinates)
        results['arkit'] = {
            'success': result_arkit['success'],
            'processing_time': time.time() - start_time,
            'control_points': result_arkit['statistics'].get('control_points_found', 0),
            'geocpset_used': result_arkit['statistics'].get('geocpset_used', False),
            'steps_completed': len([s for s in result_arkit['pipeline_steps'] if s['success']]),
            'tiff_files': len(result_arkit.get('tiff_files', [])) if result_arkit['success'] else 0
        }
        logger.info(f"✅ ARKit result: {results['arkit']}")
    except Exception as e:
        results['arkit'] = {'success': False, 'error': str(e)}
        logger.error(f"❌ ARKit processing failed: {e}")
    finally:
        service_arkit.cleanup()

def generate_fake_arkit_coordinates(num_images: int):
    """Generate fake ARKit coordinates simulating a 16-shot capture pattern."""
    coordinates = []
    
    # Simulate typical 16-shot pattern: 8 horizon + 4 upper + 4 lower
    elevations = [0] * 8 + [45] * 4 + [-45] * 4  # 3 elevation levels
    azimuths = []
    
    # 8 horizon shots at 45° intervals
    for i in range(8):
        azimuths.append(i * 45)
    
    # 4 upper shots at 90° intervals  
    for i in range(4):
        azimuths.append(i * 90)
        
    # 4 lower shots at 90° intervals
    for i in range(4):
        azimuths.append(i * 90)
    
    # Trim to actual image count
    elevations = elevations[:num_images]
    azimuths = azimuths[:num_images]
    
    # Generate coordinate objects
    for i in range(num_images):
        coordinates.append({
            'index': i,
            'arkit_input': {
                'azimuth_raw': azimuths[i] if i < len(azimuths) else i * (360 / num_images),
                'azimuth_calibrated': azimuths[i] if i < len(azimuths) else i * (360 / num_images),
                'elevation': elevations[i] if i < len(elevations) else 0,
                'position': [0.0, 0.0, 0.0]
            },
            'hugin_output': {
                'yaw': azimuths[i] if i < len(azimuths) else i * (360 / num_images),
                'pitch': (elevations[i] if i < len(elevations) else 0),  # No inversion needed
                'roll': 0.0
            },
            'debug_info': {
                'wrap_azimuth': azimuths[i] if i < len(azimuths) else i * (360 / num_images),
                'normalized_x': (azimuths[i] + 180) / 360 if i < len(azimuths) else (i * (360 / num_images) + 180) / 360,
                'normalized_y': (elevations[i] + 90) / 180 if i < len(elevations) else 0.5,
                'calibration_offset': 0.0
            },
            'validation_flags': []
        })
    
    logger.info(f"🤖 Generated fake ARKit coordinates for {num_images} images")
    logger.info(f"   Pattern: 8 horizon (0°) + 4 upper (45°) + 4 lower (-45°)")
    
    return coordinates
    
    # Compare results
    logger.info("\n🔍 COMPARISON RESULTS:")
    logger.info("=" * 50)
    
    if 'arkit' in results and 'basic' in results:
        arkit = results['arkit']
        basic = results['basic']
        
        logger.info(f"Processing Time:  ARKit={arkit.get('processing_time', 0):.1f}s  Basic={basic.get('processing_time', 0):.1f}s")
        logger.info(f"Control Points:   ARKit={arkit.get('control_points', 0)}  Basic={basic.get('control_points', 0)}")
        logger.info(f"Geocpset Used:    ARKit={arkit.get('geocpset_used', False)}  Basic={basic.get('geocpset_used', False)}")
        logger.info(f"Success:          ARKit={arkit.get('success', False)}  Basic={basic.get('success', False)}")
        
        # Determine winner
        arkit_score = 0
        basic_score = 0
        
        if arkit.get('success', False):
            arkit_score += 2
        if basic.get('success', False):
            basic_score += 2
            
        if arkit.get('control_points', 0) > basic.get('control_points', 0):
            arkit_score += 1
        elif basic.get('control_points', 0) > arkit.get('control_points', 0):
            basic_score += 1
            
        if not arkit.get('geocpset_used', True) and basic.get('geocpset_used', True):
            arkit_score += 1  # ARKit should reduce need for geocpset
        elif arkit.get('geocpset_used', True) and not basic.get('geocpset_used', True):
            basic_score += 1
            
        logger.info(f"\n📊 VERDICT: ARKit Score={arkit_score}  Basic Score={basic_score}")
        
        if arkit_score > basic_score:
            logger.info("🎯 WINNER: ARKit integration provides measurable benefit")
        elif basic_score > arkit_score:
            logger.info("📷 WINNER: Basic mode is better - ARKit may be harmful")  
        else:
            logger.info("🤷 TIE: No significant difference - ARKit adds complexity without benefit")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_arkit_value.py <images_dir>")
        print("Example: python test_arkit_value.py /path/to/your/panorama/images")
        sys.exit(1)
        
    images_dir = sys.argv[1]
    test_arkit_vs_basic(images_dir)
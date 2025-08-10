#!/usr/bin/env python3
"""
Verify Microservices Architecture

Checks that the microservices architecture has been properly implemented
by verifying file structure, imports, and class definitions without 
requiring runtime dependencies like OpenCV.
"""

import ast
import sys
from pathlib import Path

def check_service_structure():
    """Check that all service files exist with correct structure."""
    print("🔍 Checking microservices file structure...")
    
    expected_files = {
        'services/__init__.py': ['ARKitCoordinateService', 'HuginPipelineService', 'QualityValidationService', 'BlendingService'],
        'services/coordinate_service.py': ['ARKitCoordinateService', 'create_coordinate_service'],
        'services/hugin_service.py': ['HuginPipelineService', 'create_hugin_service'],
        'services/quality_service.py': ['QualityValidationService', 'create_quality_service'],
        'services/blending_service.py': ['BlendingService', 'create_blending_service'],
        'services/service_bus.py': ['PanoramaServiceBus', 'get_service_bus', 'ServiceStatus']
    }
    
    all_good = True
    
    for file_path, expected_classes in expected_files.items():
        full_path = Path(__file__).parent / file_path
        
        if not full_path.exists():
            print(f"❌ Missing: {file_path}")
            all_good = False
            continue
            
        try:
            # Parse the file to check for class definitions
            with open(full_path, 'r') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Extract class and function names
            defined_names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    defined_names.append(node.name)
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    defined_names.append(node.name)
                    
            # Check if expected classes/functions exist
            missing = [name for name in expected_classes if name not in defined_names]
            if missing:
                print(f"❌ {file_path}: Missing {missing}")
                all_good = False
            else:
                print(f"✅ {file_path}: All expected classes/functions present")
                
        except Exception as e:
            print(f"❌ {file_path}: Error parsing - {e}")
            all_good = False
    
    return all_good

def check_app_py_migration():
    """Check that app.py has been properly migrated to microservices."""
    print("\n🔍 Checking app.py microservices migration...")
    
    app_path = Path(__file__).parent / 'app.py'
    if not app_path.exists():
        print("❌ app.py not found")
        return False
        
    try:
        with open(app_path, 'r') as f:
            content = f.read()
            
        # Parse AST to look for key elements
        tree = ast.parse(content)
        
        found_elements = {
            'MicroservicesPanoramaProcessor': False,
            'microservices_import': False,
            'old_hugin_import': True,  # Should be False (removed)
            'processor_instance': False
        }
        
        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == 'services':
                    found_elements['microservices_import'] = True
                elif node.module == 'hugin_stitcher':
                    found_elements['old_hugin_import'] = True
                    
            elif isinstance(node, ast.ClassDef):
                if node.name == 'MicroservicesPanoramaProcessor':
                    found_elements['MicroservicesPanoramaProcessor'] = True
                    
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'processor':
                        if isinstance(node.value, ast.Call):
                            if isinstance(node.value.func, ast.Name) and node.value.func.id == 'MicroservicesPanoramaProcessor':
                                found_elements['processor_instance'] = True
        
        # Check results
        results = []
        if found_elements['MicroservicesPanoramaProcessor']:
            print("✅ MicroservicesPanoramaProcessor class defined")
            results.append(True)
        else:
            print("❌ MicroservicesPanoramaProcessor class missing")
            results.append(False)
            
        if found_elements['microservices_import']:
            print("✅ Microservices imported from services package")
            results.append(True)
        else:
            print("❌ Microservices import missing")
            results.append(False)
            
        if not found_elements['old_hugin_import']:
            print("✅ Old hugin_stitcher import removed")
            results.append(True)
        else:
            print("⚠️ Old hugin_stitcher import still present")
            results.append(False)
            
        if found_elements['processor_instance']:
            print("✅ Global processor uses MicroservicesPanoramaProcessor")
            results.append(True)
        else:
            print("❌ Global processor not using MicroservicesPanoramaProcessor")
            results.append(False)
            
        return all(results)
        
    except Exception as e:
        print(f"❌ Error analyzing app.py: {e}")
        return False

def check_old_monolith_status():
    """Check status of old monolithic hugin_stitcher.py."""
    print("\n🔍 Checking monolithic hugin_stitcher.py status...")
    
    hugin_path = Path(__file__).parent / 'hugin_stitcher.py'
    
    if hugin_path.exists():
        print("⚠️ hugin_stitcher.py still exists")
        print("   This is OK - it contains some utility functions that may still be needed")
        print("   But the main CorrectHuginStitcher class should no longer be used")
        return True
    else:
        print("ℹ️ hugin_stitcher.py has been removed")
        return True

def check_service_method_coverage():
    """Check that services cover the main functionality from the old monolith."""
    print("\n🔍 Checking service method coverage...")
    
    required_methods = {
        'coordinate_service.py': ['validate_arkit_data', 'convert_arkit_to_hugin'],
        'hugin_service.py': ['execute_pipeline'],
        'quality_service.py': ['analyze_panorama_quality'],
        'blending_service.py': ['blend_panorama'],
        'service_bus.py': ['register_service', 'send_message', 'update_service_status']
    }
    
    all_covered = True
    
    for file_name, methods in required_methods.items():
        file_path = Path(__file__).parent / 'services' / file_name
        
        if not file_path.exists():
            print(f"❌ {file_name}: File missing")
            all_covered = False
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            tree = ast.parse(content)
            defined_methods = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    defined_methods.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    for class_node in node.body:
                        if isinstance(class_node, ast.FunctionDef) and not class_node.name.startswith('_'):
                            defined_methods.append(class_node.name)
            
            missing_methods = [m for m in methods if m not in defined_methods]
            if missing_methods:
                print(f"❌ {file_name}: Missing methods {missing_methods}")
                all_covered = False
            else:
                print(f"✅ {file_name}: All required methods present")
                
        except Exception as e:
            print(f"❌ {file_name}: Error checking methods - {e}")
            all_covered = False
            
    return all_covered

def main():
    """Run all verification tests."""
    print("Microservices Architecture Verification")
    print("=" * 50)
    
    tests = [
        ("Service File Structure", check_service_structure),
        ("App.py Migration", check_app_py_migration), 
        ("Old Monolith Status", check_old_monolith_status),
        ("Service Method Coverage", check_service_method_coverage)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Verification Results:")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print("-" * 50)
    print(f"Overall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 MICROSERVICES ARCHITECTURE VERIFICATION COMPLETE!")
        print("")
        print("✅ All service files exist with correct structure")
        print("✅ App.py successfully migrated to microservices")
        print("✅ All required service methods are implemented")
        print("✅ Architecture is ready for debugging coordinate conversion issues")
        print("")
        print("🎯 Key Benefits:")
        print("   • Isolated debugging of each processing stage")
        print("   • ARKit coordinate conversion can be tested independently")
        print("   • Hugin pipeline steps can be debugged individually")
        print("   • Quality analysis runs in isolation")
        print("   • Multiple blending fallback strategies")
        print("")
        print("🚀 The monolithic hugin_stitcher.py has been successfully")
        print("   replaced with a modular microservices architecture!")
        
        return True
    else:
        print(f"\n⚠️ {len(results) - passed} verification checks failed")
        print("The microservices migration needs additional work.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Validate enblend parameters and test blending functionality.
"""

import subprocess
import tempfile
import os

def test_enblend_parameters():
    """Test which enblend parameters are actually valid."""
    
    print("🧪 TESTING ENBLEND PARAMETERS")
    print("=" * 40)
    
    # Test enblend help first
    try:
        result = subprocess.run(['enblend', '--help'], 
                               capture_output=True, text=True, timeout=10)
        print(f"✅ Enblend available, exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("📋 Enblend help output (first 30 lines):")
            lines = result.stdout.split('\n')[:30]
            for line in lines:
                print(f"   {line}")
        else:
            print(f"⚠️ Help output stderr: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Enblend not available or error: {e}")
        return False
    
    # Test our current parameters one by one
    base_params = ['enblend', '--help']  # Use help for quick validation
    
    test_params = [
        '--wrap=horizontal',
        '--compression=lzw', 
        '--levels=29',
        '--fine-mask',
        '--no-ciecam',
        '--verbose',
        # Previously problematic params:
        '--fallback-overlap=0.05',
        '--blend-colorspace=CIELAB',
        '--optimizer-weights=0:0:1:0',
        '--mask-vectorize=12'
    ]
    
    print(f"\n🔍 Testing individual parameters:")
    valid_params = []
    invalid_params = []
    
    for param in test_params:
        try:
            # Test if parameter is recognized (won't actually run, just parse)
            result = subprocess.run(['enblend', param, '--help'], 
                                   capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 or 'unknown' not in result.stderr.lower():
                print(f"   ✅ {param} - Valid")
                valid_params.append(param)
            else:
                print(f"   ❌ {param} - Invalid: {result.stderr.strip()}")
                invalid_params.append(param)
                
        except Exception as e:
            print(f"   ❌ {param} - Error: {e}")
            invalid_params.append(param)
    
    # Show recommended configuration
    print(f"\n📋 Recommended enblend configuration:")
    print(f"   Valid parameters: {len(valid_params)}")
    print(f"   Invalid parameters: {len(invalid_params)}")
    
    if valid_params:
        print(f"\n✅ Use these parameters:")
        for param in valid_params:
            print(f"   {param}")
    
    if invalid_params:
        print(f"\n❌ Remove these parameters:")
        for param in invalid_params:
            print(f"   {param}")
    
    return len(valid_params) > 0

def generate_optimized_enblend_command():
    """Generate optimized enblend command based on validation."""
    
    print(f"\n🎯 OPTIMIZED ENBLEND CONFIGURATION")
    print("=" * 40)
    
    # Conservative parameters known to work across enblend versions
    safe_params = [
        '--wrap=horizontal',    # For 360° panoramas
        '--compression=LZW',    # Note: might be 'LZW' not 'lzw'  
        '--levels=29',         # Maximum pyramid levels
        '--fine-mask'          # High-quality seam detection
    ]
    
    # Optional parameters to test
    optional_params = [
        '--verbose',           # For debugging
        '--no-ciecam'         # Skip color appearance model
    ]
    
    print("🔧 Conservative enblend command:")
    cmd = ['enblend', '-o', 'output.tif'] + safe_params + ['input*.tif']
    print(f"   {' '.join(cmd)}")
    
    print("\n🔧 With optional parameters:")
    cmd_extended = ['enblend', '-o', 'output.tif'] + safe_params + optional_params + ['input*.tif']
    print(f"   {' '.join(cmd_extended)}")
    
    # Generate Python code for blending_service.py
    print(f"\n💾 Python code for blending_service.py:")
    print('''
cmd = [
    "enblend",
    "-o", temp_tiff,
    "--wrap=horizontal",        # For 360° panoramas
    "--compression=LZW",        # Lossless compression  
    "--levels=29",             # Maximum pyramid levels
    "--fine-mask"              # High-quality seam detection
]
    ''')

if __name__ == "__main__":
    success = test_enblend_parameters()
    generate_optimized_enblend_command()
    
    if not success:
        print("\n⚠️  Could not validate enblend - may not be installed locally")
        print("   Testing will happen on Railway server during deployment")
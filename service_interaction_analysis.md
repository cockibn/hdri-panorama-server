# Service Interaction Analysis

## Current Processing Pipeline Order

### 1. Coordinate Service (0-5%)
- **Input**: Raw iOS ARKit capture points with azimuth/elevation
- **Processing**: 
  - Validates ARKit data (coverage, geometric issues)
  - Converts iOS coordinates to Hugin format using `yaw = (90° - azimuth) % 360°`
  - Creates debug visualization PNG
- **Output**: Converted coordinates for Hugin, validation metrics
- **Status**: ✅ Working correctly with coordinate fix

### 2. Hugin Service (10-80%)  
- **Input**: Image files + converted coordinates from step 1
- **Processing**: 7-step Hugin pipeline
  - `pto_gen`: Generate project file with positioning
  - `cpfind`: Find control points between images
  - `geocpset`: Add geometric control points using ARKit data
  - `linefind`: Detect horizon/vertical lines
  - `autooptimiser`: Optimize geometry and photometrics
  - `pano_modify`: Set canvas size and projection
  - `nona`: Render images to equirectangular coordinates
- **Output**: Rendered TIFF files for blending
- **Status**: ✅ Working well, produces 16 rendered images

### 3. Blending Service (80-95%)
- **Input**: Rendered TIFF files from step 2
- **Processing**: Multi-strategy blending
  - Primary: `enblend` professional multi-resolution blending
  - Fallback: OpenCV multi-band blending
  - Emergency: Simple pixel averaging
- **Output**: Final panorama in EXR format
- **Status**: ❌ **ISSUE**: Invalid enblend parameters causing failure

### 4. Quality Service (95-100%)
- **Input**: Final panorama image + processing context
- **Processing**: 
  - Image quality analysis (sharpness, contrast, coverage)
  - Geometric validation
  - Visual issue detection
- **Output**: Quality metrics and recommendations
- **Status**: ✅ Works when blending succeeds

## Issues Identified

### 🚨 Critical Issue: Enblend Parameters
**Problem**: Using invalid enblend options:
- `--fallback-overlap=0.05` (not a valid option)
- `--blend-colorspace=CIELAB` (not always supported)  
- `--optimizer-weights=0:0:1:0` (not a valid option)
- `--mask-vectorize=12` (not a valid option)

**Impact**: Enblend fails immediately, causing entire pipeline failure

**Fix**: Use only valid enblend parameters

### ⚠️ Service Dependency Issues

#### 1. Hard Dependencies
- Hugin Service REQUIRES coordinate conversion from Step 1
- Blending Service REQUIRES rendered TIFFs from Step 2  
- Quality Service REQUIRES final panorama from Step 3

#### 2. Error Propagation
- If any service fails, entire pipeline stops
- No graceful degradation for non-critical failures

#### 3. Resource Management
- Services create temp files but may not clean up on failure
- No timeout handling for long-running operations

## Recommendations for Service Interaction

### 1. Fix Enblend Parameters (Immediate)
Use only verified enblend options:
```bash
enblend -o output.tif --wrap=horizontal --compression=lzw --levels=29 --fine-mask input*.tif
```

### 2. Add Service Resilience
- Coordinate Service: Should work even with partial data
- Hugin Service: Should handle geocpset fallback gracefully
- Blending Service: Should fall back to OpenCV when enblend fails
- Quality Service: Should provide basic metrics even with missing context

### 3. Improve Error Handling
- Services should clean up temp files on failure
- Add timeout protection for long operations
- Better error messages for debugging

### 4. Optimize Service Order
**Current order is CORRECT** for panorama processing:
1. Coordinate conversion must happen first (provides positioning)
2. Hugin pipeline requires coordinates (geometric setup)
3. Blending requires rendered images (final stitching)
4. Quality analysis on final result (validation)

The service interaction order is optimal - the issue is in implementation details, not architecture.
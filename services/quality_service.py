#!/usr/bin/env python3
"""
Quality Validation Service

Isolates quality metrics calculation and validation from the main processing pipeline.
Provides comprehensive analysis of panorama quality including:

- Image sharpness and contrast analysis
- Control point efficiency metrics
- Coverage analysis and geometric validation
- Processing time and performance metrics
- Visual quality scoring algorithms

This service can be used independently to evaluate panorama quality at any stage
of the processing pipeline, enabling detailed debugging of quality issues.
"""

import cv2
import numpy as np
import logging
import time
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Container for comprehensive quality metrics."""
    overall_score: float
    sharpness_score: float
    contrast_score: float
    coverage_percentage: float
    control_points_count: int
    control_points_efficiency: float
    processing_time: float
    resolution: Tuple[int, int]
    file_size_mb: float
    visual_issues: List[str]
    geometric_issues: List[str]
    technical_details: Dict[str, Any]

class QualityValidationService:
    """
    Service for comprehensive panorama quality analysis and validation.
    
    Provides isolated quality assessment capabilities that can be used
    at any stage of the processing pipeline to identify issues early.
    """
    
    def __init__(self):
        self.analysis_cache = {}
        logger.info(f"🔍 Quality Validation Service initialized")
        
    def analyze_panorama_quality(self, 
                                panorama_path: str, 
                                image_count: int,
                                control_points: int = 0,
                                processing_time: float = 0.0,
                                additional_context: Dict[str, Any] = None) -> QualityMetrics:
        """
        Comprehensive quality analysis of a panorama image.
        
        Args:
            panorama_path: Path to panorama image file
            image_count: Number of source images used
            control_points: Number of control points found
            processing_time: Total processing time in seconds
            additional_context: Additional context for analysis
            
        Returns:
            Comprehensive quality metrics
        """
        logger.info(f"🔍 Analyzing panorama quality: {os.path.basename(panorama_path)}")
        
        start_time = time.time()
        context = additional_context or {}
        
        # Load panorama image
        panorama = self._load_panorama_safe(panorama_path)
        if panorama is None:
            return self._create_error_metrics("Failed to load panorama image")
            
        height, width = panorama.shape[:2]
        file_size_mb = os.path.getsize(panorama_path) / (1024 * 1024)
        
        logger.info(f"   📊 Image: {width}×{height}, {file_size_mb:.1f}MB")
        
        # Core quality analysis
        sharpness_score = self._analyze_sharpness(panorama)
        contrast_score = self._analyze_contrast(panorama)
        coverage_percentage = self._analyze_coverage(panorama)
        
        # Control point analysis
        max_possible_cp = self._calculate_theoretical_max_control_points(image_count)
        cp_efficiency = (control_points / max_possible_cp * 100) if max_possible_cp > 0 else 0
        
        # Issue detection
        visual_issues = self._detect_visual_issues(panorama)
        geometric_issues = self._detect_geometric_issues(panorama, context)
        
        # Overall scoring
        overall_score = self._calculate_overall_score(
            sharpness_score, contrast_score, coverage_percentage, cp_efficiency, visual_issues, geometric_issues
        )
        
        analysis_time = time.time() - start_time
        
        # Technical details for debugging
        technical_details = {
            'aspect_ratio': width / height,
            'pixel_count': width * height,
            'channels': panorama.shape[2] if len(panorama.shape) > 2 else 1,
            'data_type': str(panorama.dtype),
            'dynamic_range': self._calculate_dynamic_range(panorama),
            'analysis_time': analysis_time,
            'theoretical_max_cp': max_possible_cp,
            'has_alpha_channel': len(panorama.shape) > 2 and panorama.shape[2] > 3
        }
        
        metrics = QualityMetrics(
            overall_score=overall_score,
            sharpness_score=sharpness_score,
            contrast_score=contrast_score,
            coverage_percentage=coverage_percentage,
            control_points_count=control_points,
            control_points_efficiency=cp_efficiency,
            processing_time=processing_time,
            resolution=(width, height),
            file_size_mb=file_size_mb,
            visual_issues=visual_issues,
            geometric_issues=geometric_issues,
            technical_details=technical_details
        )
        
        # Log comprehensive results
        self._log_quality_results(metrics)
        
        return metrics
        
    def _load_panorama_safe(self, panorama_path: str) -> Optional[np.ndarray]:
        """Safely load panorama with multiple fallback methods."""
        try:
            # Try OpenCV first
            panorama = cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED)
            if panorama is not None:
                logger.debug(f"✅ Loaded with OpenCV: {panorama.shape}")
                return panorama
                
            # Try imageio for EXR files
            try:
                import imageio
                panorama = imageio.imread(panorama_path)
                if len(panorama.shape) == 3 and panorama.shape[2] >= 3:
                    panorama = panorama[:, :, ::-1]  # RGB to BGR
                logger.debug(f"✅ Loaded with imageio: {panorama.shape}")
                return panorama
            except ImportError:
                logger.warning("⚠️ imageio not available")
                
            # Try PIL as last resort
            try:
                from PIL import Image
                pil_image = Image.open(panorama_path)
                panorama = np.array(pil_image)
                if len(panorama.shape) == 3:
                    panorama = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
                logger.debug(f"✅ Loaded with PIL: {panorama.shape}")
                return panorama
            except Exception as e:
                logger.error(f"❌ PIL loading failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load panorama: {e}")
            
        return None
        
    def _analyze_sharpness(self, panorama: np.ndarray) -> float:
        """Analyze image sharpness using Laplacian variance method."""
        try:
            # Convert to grayscale for analysis
            if len(panorama.shape) == 3:
                gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            else:
                gray = panorama
                
            # Normalize to 0-255 range if needed
            if gray.dtype == np.float32 or gray.dtype == np.float64:
                gray = (gray * 255).astype(np.uint8)
                
            # Calculate Laplacian variance (higher = sharper)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-100 scale (empirical scaling)
            sharpness_score = min(100.0, laplacian_var / 1000.0 * 100)
            
            logger.debug(f"🔍 Sharpness analysis: Laplacian variance = {laplacian_var:.1f}, Score = {sharpness_score:.1f}")
            return sharpness_score
            
        except Exception as e:
            logger.error(f"❌ Sharpness analysis failed: {e}")
            return 0.0
            
    def _analyze_contrast(self, panorama: np.ndarray) -> float:
        """Analyze image contrast using standard deviation method."""
        try:
            # Convert to grayscale
            if len(panorama.shape) == 3:
                gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            else:
                gray = panorama
                
            # Normalize to 0-255 if needed
            if gray.dtype == np.float32 or gray.dtype == np.float64:
                gray = (gray * 255).astype(np.uint8)
                
            # Calculate standard deviation (higher = more contrast)
            std_dev = np.std(gray)
            
            # Normalize to 0-100 scale
            contrast_score = min(100.0, std_dev / 64.0 * 100)  # 64 is theoretical max std dev for 8-bit
            
            logger.debug(f"🔍 Contrast analysis: Std dev = {std_dev:.1f}, Score = {contrast_score:.1f}")
            return contrast_score
            
        except Exception as e:
            logger.error(f"❌ Contrast analysis failed: {e}")
            return 0.0
            
    def _analyze_coverage(self, panorama: np.ndarray) -> float:
        """Analyze panorama coverage (percentage of non-black pixels)."""
        try:
            # Convert to grayscale for analysis
            if len(panorama.shape) == 3:
                gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            else:
                gray = panorama
                
            # Count non-black pixels (threshold at 5 to account for compression artifacts)
            non_black_pixels = np.count_nonzero(gray > 5)
            total_pixels = gray.size
            
            coverage_percentage = (non_black_pixels / total_pixels) * 100
            
            logger.debug(f"🔍 Coverage analysis: {non_black_pixels:,}/{total_pixels:,} = {coverage_percentage:.1f}%")
            return coverage_percentage
            
        except Exception as e:
            logger.error(f"❌ Coverage analysis failed: {e}")
            return 0.0
            
    def _calculate_theoretical_max_control_points(self, image_count: int) -> int:
        """Calculate theoretical maximum control points for given image count."""
        if image_count < 2:
            return 0
        # Each image can theoretically connect to every other image
        # Formula: n * (n-1) / 2 for full connectivity
        return image_count * (image_count - 1) // 2
        
    def _detect_visual_issues(self, panorama: np.ndarray) -> List[str]:
        """Detect common visual issues in panorama."""
        issues = []
        
        try:
            height, width = panorama.shape[:2]
            
            # Check aspect ratio (should be close to 2:1 for 360° panoramas)
            aspect_ratio = width / height
            if abs(aspect_ratio - 2.0) > 0.2:
                issues.append(f"ASPECT_RATIO_ISSUE - {aspect_ratio:.2f}:1 (expected ~2:1)")
                
            # Check for excessive black areas
            if len(panorama.shape) == 3:
                gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            else:
                gray = panorama
                
            black_percentage = (np.count_nonzero(gray <= 5) / gray.size) * 100
            if black_percentage > 20:
                issues.append(f"EXCESSIVE_BLACK_AREAS - {black_percentage:.1f}% black pixels")
                
            # Check for oversaturation
            if len(panorama.shape) == 3:
                if panorama.dtype == np.uint8:
                    oversaturated = np.count_nonzero(panorama >= 250)
                    total_pixels = panorama.size
                    if (oversaturated / total_pixels) > 0.05:  # More than 5% oversaturated
                        issues.append("OVERSATURATION - Excessive bright pixels detected")
                        
            # Check for very low brightness
            mean_brightness = np.mean(gray)
            if mean_brightness < 30:  # Very dark overall
                issues.append(f"LOW_BRIGHTNESS - Mean brightness {mean_brightness:.1f}/255")
                
        except Exception as e:
            logger.error(f"❌ Visual issue detection failed: {e}")
            issues.append("ANALYSIS_ERROR - Could not complete visual analysis")
            
        return issues
        
    def _detect_geometric_issues(self, panorama: np.ndarray, context: Dict[str, Any]) -> List[str]:
        """Detect geometric distortion issues."""
        issues = []
        
        try:
            height, width = panorama.shape[:2]
            
            # Check for extreme distortion patterns (placeholder - would need more sophisticated analysis)
            # This is where we could analyze for curved lines, warped geometry, etc.
            
            # Check context for coordinate validation issues
            if 'coordinate_validation' in context:
                coord_validation = context['coordinate_validation']
                if coord_validation.get('coverage_quality', '').startswith('POOR'):
                    issues.append("COORDINATE_COVERAGE_POOR - Insufficient spherical coverage")
                    
                geometric_coord_issues = coord_validation.get('geometric_issues', [])
                for issue in geometric_coord_issues:
                    if 'POLE' in issue:
                        issues.append(f"MISSING_POLE_COVERAGE - {issue}")
                    elif 'CLUSTERING' in issue:
                        issues.append(f"IMAGE_CLUSTERING - {issue}")
                        
            # Check for resolution consistency
            if width < 1000 or height < 500:
                issues.append(f"LOW_RESOLUTION - {width}×{height} may be insufficient")
                
        except Exception as e:
            logger.error(f"❌ Geometric issue detection failed: {e}")
            issues.append("GEOMETRIC_ANALYSIS_ERROR - Could not complete geometric analysis")
            
        return issues
        
    def _calculate_dynamic_range(self, panorama: np.ndarray) -> Dict[str, float]:
        """Calculate dynamic range statistics."""
        try:
            if panorama.dtype == np.float32 or panorama.dtype == np.float64:
                min_val = float(np.min(panorama))
                max_val = float(np.max(panorama))
                mean_val = float(np.mean(panorama))
            else:
                min_val = float(np.min(panorama))
                max_val = float(np.max(panorama))
                mean_val = float(np.mean(panorama))
                
            return {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'range': max_val - min_val
            }
        except:
            return {'min': 0, 'max': 0, 'mean': 0, 'range': 0}
            
    def _calculate_overall_score(self, 
                               sharpness: float, 
                               contrast: float, 
                               coverage: float, 
                               cp_efficiency: float,
                               visual_issues: List[str],
                               geometric_issues: List[str]) -> float:
        """Calculate weighted overall quality score."""
        
        # Base scores (0-100 scale)
        base_score = (
            sharpness * 0.25 +      # 25% weight on sharpness
            contrast * 0.20 +       # 20% weight on contrast  
            coverage * 0.30 +       # 30% weight on coverage
            cp_efficiency * 0.25    # 25% weight on control point efficiency
        )
        
        # Apply penalties for issues
        issue_penalty = 0
        issue_penalty += len(visual_issues) * 5      # 5 points per visual issue
        issue_penalty += len(geometric_issues) * 10  # 10 points per geometric issue (more severe)
        
        # Apply penalties for critical issues
        critical_keywords = ['POOR', 'CRITICAL', 'MISSING_POLE', 'LOW_RESOLUTION']
        for issue in visual_issues + geometric_issues:
            for keyword in critical_keywords:
                if keyword in issue:
                    issue_penalty += 15  # Additional penalty for critical issues
                    break
                    
        final_score = max(0.0, base_score - issue_penalty)
        
        return final_score
        
    def _log_quality_results(self, metrics: QualityMetrics):
        """Log comprehensive quality results."""
        logger.info(f"📊 Quality Analysis Results:")
        logger.info(f"   Overall Score: {metrics.overall_score:.1f}/100")
        logger.info(f"   Sharpness: {metrics.sharpness_score:.1f}/100")
        logger.info(f"   Contrast: {metrics.contrast_score:.1f}/100") 
        logger.info(f"   Coverage: {metrics.coverage_percentage:.1f}%")
        logger.info(f"   Control Points: {metrics.control_points_count} ({metrics.control_points_efficiency:.1f}% efficiency)")
        logger.info(f"   Resolution: {metrics.resolution[0]}×{metrics.resolution[1]}")
        logger.info(f"   File Size: {metrics.file_size_mb:.1f}MB")
        
        if metrics.visual_issues:
            logger.warning(f"⚠️ Visual Issues ({len(metrics.visual_issues)}):")
            for issue in metrics.visual_issues:
                logger.warning(f"   - {issue}")
                
        if metrics.geometric_issues:
            logger.warning(f"⚠️ Geometric Issues ({len(metrics.geometric_issues)}):")
            for issue in metrics.geometric_issues:
                logger.warning(f"   - {issue}")
                
        if metrics.overall_score >= 80:
            logger.info("✅ EXCELLENT quality panorama")
        elif metrics.overall_score >= 60:
            logger.info("🟡 GOOD quality panorama")
        elif metrics.overall_score >= 40:
            logger.warning("🟠 FAIR quality panorama - consider improvements")
        else:
            logger.error("❌ POOR quality panorama - significant issues detected")
            
    def _create_error_metrics(self, error_message: str) -> QualityMetrics:
        """Create error metrics when analysis fails."""
        return QualityMetrics(
            overall_score=0.0,
            sharpness_score=0.0,
            contrast_score=0.0,
            coverage_percentage=0.0,
            control_points_count=0,
            control_points_efficiency=0.0,
            processing_time=0.0,
            resolution=(0, 0),
            file_size_mb=0.0,
            visual_issues=[f"ANALYSIS_ERROR - {error_message}"],
            geometric_issues=[],
            technical_details={'error': error_message}
        )
        
    def validate_quality_thresholds(self, metrics: QualityMetrics, thresholds: Dict[str, float] = None) -> Dict[str, bool]:
        """Validate quality metrics against thresholds."""
        default_thresholds = {
            'min_overall_score': 40.0,
            'min_sharpness': 20.0,
            'min_contrast': 15.0,
            'min_coverage': 70.0,
            'min_cp_efficiency': 30.0,
            'max_visual_issues': 3,
            'max_geometric_issues': 2
        }
        
        thresholds = thresholds or default_thresholds
        
        validation_results = {
            'overall_score_pass': metrics.overall_score >= thresholds['min_overall_score'],
            'sharpness_pass': metrics.sharpness_score >= thresholds['min_sharpness'],
            'contrast_pass': metrics.contrast_score >= thresholds['min_contrast'],
            'coverage_pass': metrics.coverage_percentage >= thresholds['min_coverage'],
            'cp_efficiency_pass': metrics.control_points_efficiency >= thresholds['min_cp_efficiency'],
            'visual_issues_pass': len(metrics.visual_issues) <= thresholds['max_visual_issues'],
            'geometric_issues_pass': len(metrics.geometric_issues) <= thresholds['max_geometric_issues']
        }
        
        validation_results['overall_pass'] = all(validation_results.values())
        
        return validation_results
        
    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate debug report for quality service."""
        return {
            'service_info': {
                'name': 'QualityValidationService',
                'version': '1.0.0',
                'purpose': 'Comprehensive panorama quality analysis and validation'
            },
            'analysis_capabilities': [
                'Image sharpness analysis (Laplacian variance)',
                'Contrast analysis (standard deviation)',
                'Coverage analysis (non-black pixel percentage)',
                'Control point efficiency calculation',
                'Visual issue detection',
                'Geometric distortion detection',
                'Dynamic range analysis',
                'Quality threshold validation'
            ],
            'cache_status': {
                'cached_analyses': len(self.analysis_cache)
            }
        }

def create_quality_service() -> QualityValidationService:
    """Factory function to create quality validation service."""
    return QualityValidationService()
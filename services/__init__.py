"""
Panorama Processing Service

Simple, proven Hugin-based panorama stitching service following
the established 7-step workflow for professional 360° panoramas.

- hugin_service: Complete 7-step Hugin pipeline with progressive optimization
"""

from .hugin_service import HuginPipelineService, create_hugin_service

__all__ = [
    'HuginPipelineService', 
    'create_hugin_service'
]
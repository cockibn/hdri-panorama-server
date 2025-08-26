"""
Panorama Processing Microservices

This package contains specialized services for panorama processing:

- hugin_service: Complete 7-step Hugin pipeline execution with native positioning
- quality_service: Quality validation and comprehensive metrics analysis
- blending_service: Multi-strategy blending (enblend + OpenCV fallbacks)
- service_bus: Inter-service communication and orchestration

Each service is designed to be independently testable and debuggable,
allowing isolation of specific processing stages that may cause issues
like geometric distortion in panorama results.
"""

from .hugin_service import HuginPipelineService, create_hugin_service
from .quality_service import QualityValidationService, create_quality_service
from .blending_service import BlendingService, create_blending_service
from .service_bus import PanoramaServiceBus, get_service_bus, ServiceStatus, MessageType

__all__ = [
    'HuginPipelineService', 
    'create_hugin_service',
    'QualityValidationService',
    'create_quality_service',
    'BlendingService',
    'create_blending_service',
    'PanoramaServiceBus',
    'get_service_bus',
    'ServiceStatus',
    'MessageType'
]
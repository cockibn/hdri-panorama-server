#!/usr/bin/env python3
"""
Service Bus for Panorama Processing Microservices

Handles communication between microservices and provides:
- Service discovery and registration
- Message passing between services  
- Error propagation and debugging
- Processing pipeline orchestration
- State management across services
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service operational status."""
    READY = "ready"
    BUSY = "busy" 
    ERROR = "error"
    OFFLINE = "offline"

class MessageType(Enum):
    """Types of inter-service messages."""
    REQUEST = "request"
    RESPONSE = "response" 
    ERROR = "error"
    STATUS_UPDATE = "status_update"
    DEBUG_INFO = "debug_info"

@dataclass
class ServiceMessage:
    """Message passed between services."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    from_service: str = ""
    to_service: str = ""
    operation: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None  # For tracking request/response pairs
    debug_context: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ServiceInfo:
    """Information about a registered service."""
    name: str
    version: str
    status: ServiceStatus = ServiceStatus.OFFLINE
    capabilities: List[str] = field(default_factory=list)
    endpoint: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)
    debug_enabled: bool = False

class PanoramaServiceBus:
    """
    Central communication hub for panorama processing microservices.
    
    Enables the monolithic hugin_stitcher.py to be broken down into
    independent services that can be debugged in isolation.
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.message_handlers: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        self.message_log: List[ServiceMessage] = []
        self.debug_mode = False
        self._lock = threading.Lock()
        
    def register_service(self, name: str, version: str, capabilities: List[str], 
                        endpoint: Optional[str] = None) -> bool:
        """Register a service with the bus."""
        with self._lock:
            self.services[name] = ServiceInfo(
                name=name,
                version=version,
                status=ServiceStatus.READY,
                capabilities=capabilities,
                endpoint=endpoint,
                debug_enabled=self.debug_mode
            )
            
        logger.info(f"📡 Service registered: {name} v{version} with capabilities: {capabilities}")
        return True
        
    def unregister_service(self, name: str) -> bool:
        """Unregister a service from the bus."""
        with self._lock:
            if name in self.services:
                del self.services[name]
                logger.info(f"📡 Service unregistered: {name}")
                return True
        return False
        
    def update_service_status(self, name: str, status: ServiceStatus, debug_info: Optional[Dict] = None):
        """Update service status and broadcast to other services."""
        with self._lock:
            if name in self.services:
                self.services[name].status = status
                self.services[name].last_heartbeat = time.time()
                
        # Broadcast status update
        message = ServiceMessage(
            type=MessageType.STATUS_UPDATE,
            from_service=name,
            to_service="*",  # Broadcast to all
            operation="status_change",
            data={"status": status.value, "debug_info": debug_info or {}}
        )
        self._log_message(message)
        
        logger.debug(f"📡 Service {name} status updated to {status.value}")
        
    def register_handler(self, service_name: str, operation: str, handler: Callable):
        """Register a message handler for a specific service operation."""
        self.message_handlers[service_name][operation] = handler
        logger.debug(f"📡 Handler registered: {service_name}.{operation}")
        
    def send_message(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        """Send message between services with optional response."""
        self._log_message(message)
        
        # Check if target service exists
        if message.to_service != "*" and message.to_service not in self.services:
            error_msg = f"Target service '{message.to_service}' not found"
            logger.error(f"📡 {error_msg}")
            return ServiceMessage(
                type=MessageType.ERROR,
                from_service="service_bus", 
                to_service=message.from_service,
                operation="service_not_found",
                data={"error": error_msg},
                correlation_id=message.id
            )
            
        # Route message to handler if available
        if (message.to_service in self.message_handlers and 
            message.operation in self.message_handlers[message.to_service]):
            
            try:
                handler = self.message_handlers[message.to_service][message.operation]
                response_data = handler(message.data)
                
                # Create response message
                response = ServiceMessage(
                    type=MessageType.RESPONSE,
                    from_service=message.to_service,
                    to_service=message.from_service,
                    operation=f"{message.operation}_response",
                    data=response_data or {},
                    correlation_id=message.id
                )
                
                self._log_message(response)
                return response
                
            except Exception as e:
                logger.error(f"📡 Handler error in {message.to_service}.{message.operation}: {e}")
                error_response = ServiceMessage(
                    type=MessageType.ERROR,
                    from_service=message.to_service,
                    to_service=message.from_service, 
                    operation="handler_error",
                    data={"error": str(e), "operation": message.operation},
                    correlation_id=message.id
                )
                self._log_message(error_response)
                return error_response
                
        else:
            logger.warning(f"📡 No handler for {message.to_service}.{message.operation}")
            
        return None
        
    def call_service(self, to_service: str, operation: str, data: Dict[str, Any], 
                    from_service: str = "unknown", timeout: float = 30.0) -> Dict[str, Any]:
        """Synchronous service call with timeout."""
        message = ServiceMessage(
            type=MessageType.REQUEST,
            from_service=from_service,
            to_service=to_service,
            operation=operation,
            data=data
        )
        
        response = self.send_message(message)
        if response and response.type == MessageType.RESPONSE:
            return response.data
        elif response and response.type == MessageType.ERROR:
            raise RuntimeError(f"Service call failed: {response.data.get('error', 'Unknown error')}")
        else:
            raise TimeoutError(f"Service call to {to_service}.{operation} timed out")
            
    def get_service_info(self, name: str) -> Optional[ServiceInfo]:
        """Get information about a registered service."""
        return self.services.get(name)
        
    def list_services(self) -> Dict[str, ServiceInfo]:
        """Get list of all registered services."""
        return dict(self.services)
        
    def enable_debug_mode(self, enabled: bool = True):
        """Enable/disable debug mode for detailed message logging."""
        self.debug_mode = enabled
        for service in self.services.values():
            service.debug_enabled = enabled
        logger.info(f"📡 Debug mode {'enabled' if enabled else 'disabled'}")
        
    def get_message_log(self, limit: Optional[int] = None, 
                       service_filter: Optional[str] = None) -> List[ServiceMessage]:
        """Get message log for debugging."""
        messages = self.message_log
        
        if service_filter:
            messages = [m for m in messages if service_filter in [m.from_service, m.to_service]]
            
        if limit:
            messages = messages[-limit:]
            
        return messages
        
    def clear_message_log(self):
        """Clear message log."""
        with self._lock:
            self.message_log.clear()
        logger.info("📡 Message log cleared")
        
    def _log_message(self, message: ServiceMessage):
        """Log message for debugging and auditing."""
        with self._lock:
            self.message_log.append(message)
            
            # Keep log size manageable
            if len(self.message_log) > 1000:
                self.message_log = self.message_log[-500:]
                
        if self.debug_mode:
            logger.debug(f"📡 {message.type.value}: {message.from_service} → {message.to_service} [{message.operation}]")
            
    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report for service bus state."""
        recent_messages = self.get_message_log(limit=50)
        
        # Analyze message patterns
        message_stats = defaultdict(int)
        error_count = 0
        for msg in recent_messages:
            message_stats[f"{msg.from_service}→{msg.to_service}"] += 1
            if msg.type == MessageType.ERROR:
                error_count += 1
                
        return {
            'service_bus_info': {
                'total_services': len(self.services),
                'debug_mode': self.debug_mode,
                'total_messages_logged': len(self.message_log)
            },
            'registered_services': {name: {
                'status': service.status.value,
                'capabilities': service.capabilities,
                'last_heartbeat': service.last_heartbeat,
                'uptime': time.time() - service.last_heartbeat
            } for name, service in self.services.items()},
            'recent_activity': {
                'message_count': len(recent_messages),
                'error_count': error_count,
                'message_patterns': dict(message_stats)
            },
            'recent_messages': [
                {
                    'timestamp': msg.timestamp,
                    'type': msg.type.value,
                    'from': msg.from_service,
                    'to': msg.to_service,
                    'operation': msg.operation,
                    'has_error': msg.type == MessageType.ERROR
                } for msg in recent_messages[-10:]  # Last 10 messages
            ]
        }

# Global service bus instance
_service_bus_instance = None

def get_service_bus() -> PanoramaServiceBus:
    """Get the global service bus instance."""
    global _service_bus_instance
    if _service_bus_instance is None:
        _service_bus_instance = PanoramaServiceBus()
    return _service_bus_instance

def create_service_bus() -> PanoramaServiceBus:
    """Create a new service bus instance (for testing)."""
    return PanoramaServiceBus()
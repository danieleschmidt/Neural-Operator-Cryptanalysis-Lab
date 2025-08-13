"""Enhanced security utilities with comprehensive protection measures."""

import hashlib
import hmac
import time
import secrets
import re
import os
import ipaddress
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import threading
from collections import defaultdict, deque
import json

from .logging_utils import get_logger, log_security_event
from .errors import SecurityError, ValidationError, AuthorizationError, create_error_context


logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "auth_violation"
    INPUT_VALIDATION_FAILURE = "input_validation"
    RATE_LIMIT_EXCEEDED = "rate_limit"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    MALICIOUS_INPUT = "malicious_input"
    PATH_TRAVERSAL = "path_traversal"
    INJECTION_ATTEMPT = "injection_attempt"


@dataclass
class SecurityEvent:
    """Security event information."""
    event_type: SecurityEventType
    threat_level: ThreatLevel
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: float = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.context is None:
            self.context = {}


class InputSanitizer:
    """Comprehensive input sanitization and validation."""
    
    # Patterns for detecting malicious input
    INJECTION_PATTERNS = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',  # SQL injection
        r'(<script[^>]*>.*?</script>)',  # XSS
        r'(javascript:|data:|vbscript:)',  # Script injections
        r'(\.\./|\.\.\\)',  # Path traversal
        r'(\b(eval|exec|system|shell_exec|passthru)\s*\()',  # Code execution
        r'(\$\{.*\})',  # Expression language injection
        r'(#{.*})',  # SpEL injection
        r'(__import__|getattr|setattr|delattr)',  # Python introspection
        r'(\b(and|or)\s+\d+\s*=\s*\d+)',  # SQL boolean injection
    ]
    
    # Compiled regex patterns for efficiency
    COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS]
    
    def __init__(self):
        self.blocked_strings = {
            'script', 'javascript', 'vbscript', 'onload', 'onerror', 'eval', 'exec',
            'system', 'shell', 'cmd', 'powershell', 'bash', '../', '..\\',
            'union', 'select', 'insert', 'update', 'delete', 'drop', 'create',
            'alter', 'truncate', 'replace', '__import__', 'getattr', 'setattr'
        }
        
        self.max_input_length = 10000
        self.max_nested_depth = 10
    
    def sanitize_string(self, input_str: str, 
                       field_name: str = "input",
                       allow_html: bool = False,
                       max_length: Optional[int] = None) -> str:
        """Sanitize string input.
        
        Args:
            input_str: Input string to sanitize
            field_name: Name of the field for error reporting
            allow_html: Whether to allow HTML content
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If malicious content is detected
            ValidationError: If input is invalid
        """
        if not isinstance(input_str, str):
            raise ValidationError(
                f"Expected string input for {field_name}",
                field=field_name,
                value=type(input_str)
            )
        
        # Length validation
        max_len = max_length or self.max_input_length
        if len(input_str) > max_len:
            raise ValidationError(
                f"Input too long for {field_name}: {len(input_str)} > {max_len}",
                field=field_name,
                value=len(input_str)
            )
        
        # Check for null bytes
        if '\x00' in input_str:
            raise SecurityError(
                f"Null byte detected in {field_name}",
                context=create_error_context("InputSanitizer", "sanitize_string")
            )
        
        # Check for malicious patterns
        for pattern in self.COMPILED_PATTERNS:
            match = pattern.search(input_str)
            if match:
                logger.warning(f"Malicious pattern detected in {field_name}: {match.group()}")
                raise SecurityError(
                    f"Potentially malicious content detected in {field_name}",
                    context=create_error_context("InputSanitizer", "sanitize_string", 
                                               pattern=match.group(), field=field_name)
                )
        
        # Check for blocked strings
        input_lower = input_str.lower()
        for blocked in self.blocked_strings:
            if blocked in input_lower:
                logger.warning(f"Blocked string '{blocked}' found in {field_name}")
                raise SecurityError(
                    f"Forbidden content detected in {field_name}",
                    context=create_error_context("InputSanitizer", "sanitize_string",
                                               blocked_string=blocked, field=field_name)
                )
        
        # HTML sanitization
        if not allow_html:
            # Remove HTML tags
            import html
            sanitized = html.escape(input_str)
        else:
            # Basic HTML sanitization - remove dangerous tags
            dangerous_tags = ['script', 'object', 'embed', 'form', 'iframe', 'meta']
            sanitized = input_str
            for tag in dangerous_tags:
                sanitized = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', sanitized, flags=re.IGNORECASE)
                sanitized = re.sub(f'<{tag}[^>]*/?>', '', sanitized, flags=re.IGNORECASE)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def sanitize_path(self, path_str: str, 
                     base_path: Optional[Path] = None,
                     allow_absolute: bool = False) -> Path:
        """Sanitize file path input.
        
        Args:
            path_str: Path string to sanitize
            base_path: Base path to resolve relative paths
            allow_absolute: Whether to allow absolute paths
            
        Returns:
            Sanitized Path object
            
        Raises:
            SecurityError: If path traversal is detected
        """
        if not isinstance(path_str, str):
            raise ValidationError("Path must be a string", value=type(path_str))
        
        # Check for path traversal
        if '..' in path_str or '~' in path_str:
            raise SecurityError(
                "Path traversal detected in path",
                context=create_error_context("InputSanitizer", "sanitize_path", path=path_str)
            )
        
        # Check for null bytes
        if '\x00' in path_str:
            raise SecurityError(
                "Null byte in path",
                context=create_error_context("InputSanitizer", "sanitize_path", path=path_str)
            )
        
        # Normalize path
        try:
            path = Path(path_str)
            
            # Check if absolute path is allowed
            if path.is_absolute() and not allow_absolute:
                raise SecurityError(
                    "Absolute paths not allowed",
                    context=create_error_context("InputSanitizer", "sanitize_path", path=path_str)
                )
            
            # Resolve relative to base path
            if base_path and not path.is_absolute():
                path = base_path / path
            
            # Resolve and check for traversal
            resolved = path.resolve()
            if base_path:
                base_resolved = base_path.resolve()
                if not str(resolved).startswith(str(base_resolved)):
                    raise SecurityError(
                        "Path traversal outside base directory",
                        context=create_error_context("InputSanitizer", "sanitize_path",
                                                   path=path_str, base_path=str(base_path))
                    )
            
            return resolved
            
        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            raise ValidationError(
                f"Invalid path format: {e}",
                field="path",
                value=path_str
            )
    
    def sanitize_numeric(self, value: Any,
                        field_name: str = "value",
                        min_value: Optional[float] = None,
                        max_value: Optional[float] = None,
                        allow_negative: bool = True) -> Union[int, float]:
        """Sanitize numeric input.
        
        Args:
            value: Value to sanitize
            field_name: Field name for error reporting
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_negative: Whether negative values are allowed
            
        Returns:
            Sanitized numeric value
        """
        if not isinstance(value, (int, float)):
            try:
                if isinstance(value, str):
                    # Remove whitespace
                    value = value.strip()
                    # Check for non-numeric characters
                    if not re.match(r'^-?\d*\.?\d+([eE][-+]?\d+)?$', value):
                        raise ValueError("Invalid numeric format")
                    value = float(value) if '.' in value or 'e' in value.lower() else int(value)
                else:
                    value = float(value)
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Invalid numeric value for {field_name}",
                    field=field_name,
                    value=value
                )
        
        # Check for special float values
        if isinstance(value, float):
            if not (value == value):  # NaN check
                raise ValidationError(f"NaN not allowed for {field_name}", field=field_name)
            if abs(value) == float('inf'):
                raise ValidationError(f"Infinity not allowed for {field_name}", field=field_name)
        
        # Range validation
        if not allow_negative and value < 0:
            raise ValidationError(
                f"Negative values not allowed for {field_name}",
                field=field_name,
                value=value
            )
        
        if min_value is not None and value < min_value:
            raise ValidationError(
                f"{field_name} below minimum: {value} < {min_value}",
                field=field_name,
                value=value
            )
        
        if max_value is not None and value > max_value:
            raise ValidationError(
                f"{field_name} above maximum: {value} > {max_value}",
                field=field_name,
                value=value
            )
        
        return value
    
    def sanitize_dict(self, data: Dict[str, Any],
                     allowed_keys: Optional[List[str]] = None,
                     max_depth: int = None) -> Dict[str, Any]:
        """Sanitize dictionary input.
        
        Args:
            data: Dictionary to sanitize
            allowed_keys: List of allowed keys
            max_depth: Maximum nesting depth
            
        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            raise ValidationError("Expected dictionary input", value=type(data))
        
        max_depth = max_depth or self.max_nested_depth
        
        def _sanitize_recursive(obj, depth=0):
            if depth > max_depth:
                raise SecurityError(
                    f"Maximum nesting depth exceeded: {depth} > {max_depth}",
                    context=create_error_context("InputSanitizer", "sanitize_dict")
                )
            
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    # Sanitize key
                    if not isinstance(key, str):
                        key = str(key)
                    sanitized_key = self.sanitize_string(key, f"key_{depth}")
                    
                    # Check allowed keys
                    if allowed_keys and sanitized_key not in allowed_keys:
                        logger.warning(f"Disallowed key filtered: {sanitized_key}")
                        continue
                    
                    # Recursively sanitize value
                    result[sanitized_key] = _sanitize_recursive(value, depth + 1)
                
                return result
            
            elif isinstance(obj, list):
                return [_sanitize_recursive(item, depth + 1) for item in obj[:100]]  # Limit list size
            
            elif isinstance(obj, str):
                return self.sanitize_string(obj, f"value_{depth}")
            
            elif isinstance(obj, (int, float)):
                return self.sanitize_numeric(obj, f"value_{depth}")
            
            elif obj is None or isinstance(obj, bool):
                return obj
            
            else:
                # Convert unknown types to string and sanitize
                return self.sanitize_string(str(obj), f"value_{depth}")
        
        return _sanitize_recursive(data)


class RateLimiter:
    """Advanced rate limiting with multiple strategies."""
    
    def __init__(self):
        self.requests = defaultdict(lambda: deque())
        self.blocked_ips = {}
        self.lock = threading.RLock()
        
        # Rate limits per endpoint/user
        self.limits = {
            'default': {'requests': 100, 'window': 3600},  # 100 requests per hour
            'auth': {'requests': 5, 'window': 300},        # 5 auth attempts per 5 minutes
            'attack': {'requests': 10, 'window': 3600},    # 10 attacks per hour
            'training': {'requests': 5, 'window': 3600},   # 5 training sessions per hour
        }
    
    def is_allowed(self, identifier: str, 
                   endpoint: str = 'default',
                   ip_address: Optional[str] = None) -> bool:
        """Check if request is allowed under rate limits.
        
        Args:
            identifier: User/session identifier
            endpoint: API endpoint category
            ip_address: Source IP address
            
        Returns:
            True if request is allowed
        """
        current_time = time.time()
        
        with self.lock:
            # Check if IP is blocked
            if ip_address and ip_address in self.blocked_ips:
                if current_time < self.blocked_ips[ip_address]:
                    logger.warning(f"Blocked IP attempted access: {ip_address}")
                    return False
                else:
                    # Unblock expired blocks
                    del self.blocked_ips[ip_address]
            
            # Get rate limit config
            limit_config = self.limits.get(endpoint, self.limits['default'])
            max_requests = limit_config['requests']
            window = limit_config['window']
            
            # Clean old requests
            cutoff_time = current_time - window
            request_times = self.requests[identifier]
            
            while request_times and request_times[0] < cutoff_time:
                request_times.popleft()
            
            # Check if limit exceeded
            if len(request_times) >= max_requests:
                # Block IP temporarily if too many requests
                if ip_address:
                    self.blocked_ips[ip_address] = current_time + (window * 2)
                
                log_security_event(
                    logger,
                    SecurityEventType.RATE_LIMIT_EXCEEDED.value,
                    f"Rate limit exceeded for {identifier} on {endpoint}",
                    identifier=identifier,
                    endpoint=endpoint,
                    ip_address=ip_address
                )
                return False
            
            # Add current request
            request_times.append(current_time)
            return True
    
    def block_identifier(self, identifier: str, duration: int = 3600):
        """Manually block an identifier.
        
        Args:
            identifier: Identifier to block
            duration: Block duration in seconds
        """
        with self.lock:
            # Clear existing requests
            self.requests[identifier].clear()
            # Add to blocked list
            self.blocked_ips[identifier] = time.time() + duration
            
        logger.warning(f"Identifier blocked: {identifier} for {duration} seconds")


class SecurityMonitor:
    """Monitor for security events and threats."""
    
    def __init__(self):
        self.events = deque(maxlen=10000)
        self.threat_scores = defaultdict(float)
        self.lock = threading.RLock()
        
        # Threat scoring weights
        self.threat_weights = {
            SecurityEventType.AUTHENTICATION_FAILURE: 1.0,
            SecurityEventType.AUTHORIZATION_VIOLATION: 2.0,
            SecurityEventType.INPUT_VALIDATION_FAILURE: 1.5,
            SecurityEventType.MALICIOUS_INPUT: 3.0,
            SecurityEventType.PATH_TRAVERSAL: 2.5,
            SecurityEventType.INJECTION_ATTEMPT: 3.0,
            SecurityEventType.PRIVILEGE_ESCALATION: 4.0,
            SecurityEventType.DATA_EXFILTRATION: 5.0,
        }
    
    def record_event(self, event: SecurityEvent):
        """Record a security event."""
        with self.lock:
            self.events.append(event)
            
            # Update threat score
            if event.source_ip:
                weight = self.threat_weights.get(event.event_type, 1.0)
                self.threat_scores[event.source_ip] += weight
                
                # Decay old scores
                self.threat_scores[event.source_ip] *= 0.99
            
        # Log critical events immediately
        if event.threat_level == ThreatLevel.CRITICAL:
            logger.critical(f"CRITICAL SECURITY EVENT: {event.description}")
        
        logger.warning(f"Security event: {event.event_type.value} - {event.description}")
    
    def get_threat_level(self, identifier: str) -> ThreatLevel:
        """Get current threat level for identifier."""
        score = self.threat_scores.get(identifier, 0.0)
        
        if score >= 20.0:
            return ThreatLevel.CRITICAL
        elif score >= 10.0:
            return ThreatLevel.HIGH
        elif score >= 5.0:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def get_recent_events(self, event_type: Optional[SecurityEventType] = None,
                         hours: int = 24) -> List[SecurityEvent]:
        """Get recent security events."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            events = [
                event for event in self.events
                if event.timestamp >= cutoff_time and
                (event_type is None or event.event_type == event_type)
            ]
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        with self.lock:
            recent_events = self.get_recent_events(hours=24)
            
            # Count by type
            event_counts = defaultdict(int)
            for event in recent_events:
                event_counts[event.event_type.value] += 1
            
            # Top threat sources
            top_threats = sorted(
                self.threat_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Threat level distribution
            threat_levels = defaultdict(int)
            for score in self.threat_scores.values():
                if score >= 20.0:
                    threat_levels['critical'] += 1
                elif score >= 10.0:
                    threat_levels['high'] += 1
                elif score >= 5.0:
                    threat_levels['medium'] += 1
                else:
                    threat_levels['low'] += 1
        
        return {
            'report_timestamp': time.time(),
            'total_events_24h': len(recent_events),
            'event_counts_by_type': dict(event_counts),
            'top_threat_sources': top_threats,
            'threat_level_distribution': dict(threat_levels),
            'critical_events': [
                {
                    'type': event.event_type.value,
                    'description': event.description,
                    'timestamp': event.timestamp,
                    'source_ip': event.source_ip
                }
                for event in recent_events
                if event.threat_level == ThreatLevel.CRITICAL
            ]
        }


# Global security components
input_sanitizer = InputSanitizer()
rate_limiter = RateLimiter()
security_monitor = SecurityMonitor()


def secure_operation(operation_type: str = 'default',
                    require_auth: bool = True,
                    rate_limit: bool = True,
                    sanitize_inputs: bool = True):
    """Decorator for securing operations.
    
    Args:
        operation_type: Type of operation for rate limiting
        require_auth: Whether authentication is required
        rate_limit: Whether to apply rate limiting
        sanitize_inputs: Whether to sanitize inputs
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract context
            request_ip = kwargs.get('_request_ip', 'unknown')
            user_id = kwargs.get('_user_id', 'anonymous')
            
            # Rate limiting
            if rate_limit:
                if not rate_limiter.is_allowed(user_id, operation_type, request_ip):
                    security_monitor.record_event(SecurityEvent(
                        event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                        threat_level=ThreatLevel.MEDIUM,
                        description=f"Rate limit exceeded for {operation_type}",
                        source_ip=request_ip,
                        user_id=user_id
                    ))
                    raise SecurityError("Rate limit exceeded")
            
            # Authentication check
            if require_auth:
                auth_token = kwargs.get('_auth_token')
                if not auth_token:
                    security_monitor.record_event(SecurityEvent(
                        event_type=SecurityEventType.AUTHENTICATION_FAILURE,
                        threat_level=ThreatLevel.HIGH,
                        description="Missing authentication token",
                        source_ip=request_ip,
                        user_id=user_id
                    ))
                    raise AuthorizationError("Authentication required")
            
            # Input sanitization
            if sanitize_inputs:
                sanitized_kwargs = {}
                for key, value in kwargs.items():
                    if key.startswith('_'):  # Skip internal parameters
                        sanitized_kwargs[key] = value
                        continue
                    
                    try:
                        if isinstance(value, str):
                            sanitized_kwargs[key] = input_sanitizer.sanitize_string(value, key)
                        elif isinstance(value, dict):
                            sanitized_kwargs[key] = input_sanitizer.sanitize_dict(value)
                        else:
                            sanitized_kwargs[key] = value
                    except SecurityError as e:
                        security_monitor.record_event(SecurityEvent(
                            event_type=SecurityEventType.MALICIOUS_INPUT,
                            threat_level=ThreatLevel.HIGH,
                            description=f"Malicious input detected in {key}",
                            source_ip=request_ip,
                            user_id=user_id,
                            context={'parameter': key, 'error': str(e)}
                        ))
                        raise
                
                kwargs = sanitized_kwargs
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def get_security_status() -> Dict[str, Any]:
    """Get current security status."""
    return {
        'timestamp': time.time(),
        'rate_limiter_status': {
            'active_limiters': len(rate_limiter.requests),
            'blocked_ips': len(rate_limiter.blocked_ips)
        },
        'security_monitor_status': {
            'total_events': len(security_monitor.events),
            'threat_sources': len(security_monitor.threat_scores)
        },
        'recent_threats': security_monitor.get_recent_events(hours=1)[:5]
    }
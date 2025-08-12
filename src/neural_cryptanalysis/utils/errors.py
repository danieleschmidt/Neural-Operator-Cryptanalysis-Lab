"""Comprehensive error handling framework for neural cryptanalysis."""

import sys
import traceback
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from .logging_utils import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    DATA = "data"
    MODEL = "model"
    SYSTEM = "system"
    NETWORK = "network"
    HARDWARE = "hardware"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    PERMISSION = "permission"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"


@dataclass
class ErrorContext:
    """Context information for errors."""
    component: str
    operation: str
    parameters: Dict[str, Any] = None
    stack_trace: str = None
    system_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.system_info is None:
            self.system_info = self._collect_system_info()
        if self.stack_trace is None:
            self.stack_trace = ''.join(traceback.format_stack())
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect relevant system information."""
        try:
            import psutil
            import platform
            
            return {
                'platform': platform.platform(),
                'python_version': sys.version,
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(),
                'disk_percent': psutil.disk_usage('/').percent
            }
        except ImportError:
            return {
                'platform': sys.platform,
                'python_version': sys.version
            }


class NeuralCryptanalysisError(Exception):
    """Base exception for neural cryptanalysis framework."""
    
    def __init__(self, 
                 message: str,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None,
                 recoverable: bool = True,
                 user_message: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context
        self.cause = cause
        self.recoverable = recoverable
        self.user_message = user_message or self._generate_user_message()
        
        # Log the error automatically
        self._log_error()
    
    def _generate_user_message(self) -> str:
        """Generate user-friendly error message."""
        if self.category == ErrorCategory.VALIDATION:
            return f"Input validation failed: {self.message}"
        elif self.category == ErrorCategory.SECURITY:
            return "Security policy violation detected. Please check your authorization."
        elif self.category == ErrorCategory.CONFIGURATION:
            return f"Configuration error: {self.message}"
        elif self.category == ErrorCategory.DATA:
            return f"Data processing error: {self.message}"
        elif self.category == ErrorCategory.MODEL:
            return f"Model error: {self.message}"
        elif self.category == ErrorCategory.MEMORY:
            return "Insufficient memory for operation. Try reducing batch size or data size."
        elif self.category == ErrorCategory.TIMEOUT:
            return "Operation timed out. Try reducing complexity or increasing timeout."
        else:
            return f"An error occurred: {self.message}"
    
    def _log_error(self):
        """Log the error with appropriate level."""
        log_level = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[self.severity]
        
        context_info = ""
        if self.context:
            context_info = f" [Component: {self.context.component}, Operation: {self.context.operation}]"
        
        logger.log(log_level, f"{self.category.value.upper()} ERROR: {self.message}{context_info}")
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR DETAILS:\n{self}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        result = {
            'type': self.__class__.__name__,
            'message': self.message,
            'user_message': self.user_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'recoverable': self.recoverable
        }
        
        if self.context:
            result['context'] = {
                'component': self.context.component,
                'operation': self.context.operation,
                'parameters': self.context.parameters,
                'system_info': self.context.system_info
            }
        
        if self.cause:
            result['cause'] = str(self.cause)
        
        return result
    
    def __str__(self) -> str:
        """String representation of error."""
        result = f"{self.__class__.__name__}: {self.message}"
        result += f"\nSeverity: {self.severity.value}"
        result += f"\nCategory: {self.category.value}"
        result += f"\nRecoverable: {self.recoverable}"
        
        if self.context:
            result += f"\nComponent: {self.context.component}"
            result += f"\nOperation: {self.context.operation}"
            if self.context.parameters:
                result += f"\nParameters: {self.context.parameters}"
        
        if self.cause:
            result += f"\nCaused by: {self.cause}"
        
        return result


class ValidationError(NeuralCryptanalysisError):
    """Validation-related errors."""
    
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        self.field = field
        self.value = value
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            **kwargs
        )


class SecurityError(NeuralCryptanalysisError):
    """Security-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SECURITY,
            recoverable=False,
            **kwargs
        )


class ConfigurationError(NeuralCryptanalysisError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        self.config_key = config_key
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )


class DataError(NeuralCryptanalysisError):
    """Data processing errors."""
    
    def __init__(self, message: str, data_type: str = None, **kwargs):
        self.data_type = data_type
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA,
            **kwargs
        )


class ModelError(NeuralCryptanalysisError):
    """Model-related errors."""
    
    def __init__(self, message: str, model_component: str = None, **kwargs):
        self.model_component = model_component
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MODEL,
            **kwargs
        )


class ResourceError(NeuralCryptanalysisError):
    """Resource-related errors (memory, disk, etc.)."""
    
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        self.resource_type = resource_type
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY if resource_type == 'memory' else ErrorCategory.SYSTEM,
            **kwargs
        )


class TimeoutError(NeuralCryptanalysisError):
    """Timeout-related errors."""
    
    def __init__(self, message: str, timeout_duration: float = None, **kwargs):
        self.timeout_duration = timeout_duration
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TIMEOUT,
            **kwargs
        )


class AuthenticationError(NeuralCryptanalysisError):
    """Authentication-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHENTICATION,
            recoverable=False,
            **kwargs
        )


class AuthorizationError(NeuralCryptanalysisError):
    """Authorization-related errors."""
    
    def __init__(self, message: str, required_permission: str = None, **kwargs):
        self.required_permission = required_permission
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHORIZATION,
            recoverable=False,
            **kwargs
        )


def error_handler(
    default_return: Any = None,
    catch_exceptions: Union[Type[Exception], tuple] = Exception,
    log_level: int = logging.ERROR,
    re_raise: bool = False,
    cleanup_func: Optional[Callable] = None
):
    """Decorator for comprehensive error handling.
    
    Args:
        default_return: Default value to return on error
        catch_exceptions: Exception types to catch
        log_level: Logging level for caught exceptions
        re_raise: Whether to re-raise after handling
        cleanup_func: Cleanup function to call on error
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except catch_exceptions as e:
                # Create error context
                context = ErrorContext(
                    component=func.__module__,
                    operation=func.__name__,
                    parameters={'args': str(args)[:500], 'kwargs': str(kwargs)[:500]}
                )
                
                # Convert to framework error if needed
                if not isinstance(e, NeuralCryptanalysisError):
                    if isinstance(e, ValueError):
                        framework_error = ValidationError(str(e), context=context, cause=e)
                    elif isinstance(e, PermissionError):
                        framework_error = AuthorizationError(str(e), context=context, cause=e)
                    elif isinstance(e, MemoryError):
                        framework_error = ResourceError(str(e), resource_type='memory', context=context, cause=e)
                    elif isinstance(e, OSError):
                        framework_error = NeuralCryptanalysisError(
                            str(e), 
                            category=ErrorCategory.SYSTEM, 
                            context=context, 
                            cause=e
                        )
                    else:
                        framework_error = NeuralCryptanalysisError(str(e), context=context, cause=e)
                else:
                    framework_error = e
                
                # Call cleanup function
                if cleanup_func:
                    try:
                        cleanup_func()
                    except Exception as cleanup_error:
                        logger.error(f"Cleanup function failed: {cleanup_error}")
                
                # Log the error
                logger.log(log_level, f"Error in {func.__name__}: {framework_error}")
                
                if re_raise:
                    raise framework_error
                
                return default_return
        
        return wrapper
    return decorator


def validate_input(validation_func: Callable[[Any], bool], 
                  error_message: str = "Validation failed",
                  field_name: str = None):
    """Decorator for input validation.
    
    Args:
        validation_func: Function that returns True if valid
        error_message: Error message for validation failure
        field_name: Name of the field being validated
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate each argument
            for i, arg in enumerate(args):
                if not validation_func(arg):
                    raise ValidationError(
                        f"{error_message} for argument {i}",
                        field=field_name or f"arg_{i}",
                        value=arg,
                        context=ErrorContext(
                            component=func.__module__,
                            operation=func.__name__
                        )
                    )
            
            # Validate keyword arguments
            for key, value in kwargs.items():
                if not validation_func(value):
                    raise ValidationError(
                        f"{error_message} for parameter {key}",
                        field=field_name or key,
                        value=value,
                        context=ErrorContext(
                            component=func.__module__,
                            operation=func.__name__
                        )
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_authorization(required_permission: str):
    """Decorator to require authorization for operations.
    
    Args:
        required_permission: Required permission string
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # In a real implementation, this would check actual authorization
            # For now, just log the requirement
            logger.info(f"Authorization required for {func.__name__}: {required_permission}")
            
            # Check if authorization token is provided
            auth_token = kwargs.get('auth_token')
            if not auth_token:
                raise AuthorizationError(
                    f"Authorization required for operation: {func.__name__}",
                    required_permission=required_permission,
                    context=ErrorContext(
                        component=func.__module__,
                        operation=func.__name__
                    )
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class ErrorCollector:
    """Collects and manages multiple errors during batch operations."""
    
    def __init__(self, max_errors: int = 100):
        self.max_errors = max_errors
        self.errors: List[NeuralCryptanalysisError] = []
        self.warnings: List[str] = []
    
    def add_error(self, error: Union[Exception, NeuralCryptanalysisError]):
        """Add an error to the collection."""
        if len(self.errors) >= self.max_errors:
            logger.warning(f"Error collection limit reached ({self.max_errors})")
            return
        
        if isinstance(error, NeuralCryptanalysisError):
            self.errors.append(error)
        else:
            framework_error = NeuralCryptanalysisError(str(error), cause=error)
            self.errors.append(framework_error)
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)
    
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0
    
    def has_critical_errors(self) -> bool:
        """Check if any critical errors were collected."""
        return any(e.severity == ErrorSeverity.CRITICAL for e in self.errors)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of collected errors."""
        if not self.errors:
            return {'error_count': 0, 'warning_count': len(self.warnings)}
        
        severity_counts = {}
        category_counts = {}
        
        for error in self.errors:
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
        
        return {
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'severity_counts': severity_counts,
            'category_counts': category_counts,
            'recoverable_count': sum(1 for e in self.errors if e.recoverable),
            'first_error': self.errors[0].to_dict() if self.errors else None
        }
    
    def raise_if_errors(self, message: str = "Multiple errors occurred"):
        """Raise an exception if any errors were collected."""
        if self.has_errors():
            summary = self.get_error_summary()
            
            if self.has_critical_errors():
                raise NeuralCryptanalysisError(
                    f"{message}: {summary['error_count']} errors including critical errors",
                    severity=ErrorSeverity.CRITICAL,
                    context=ErrorContext(
                        component="ErrorCollector",
                        operation="batch_validation",
                        parameters=summary
                    )
                )
            else:
                raise NeuralCryptanalysisError(
                    f"{message}: {summary['error_count']} errors",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(
                        component="ErrorCollector",
                        operation="batch_validation",
                        parameters=summary
                    )
                )


def graceful_degradation(fallback_func: Callable, 
                        fallback_message: str = "Using fallback implementation"):
    """Decorator for graceful degradation on errors.
    
    Args:
        fallback_func: Fallback function to call on error
        fallback_message: Message to log when using fallback
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"{fallback_message}: {e}")
                
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    raise NeuralCryptanalysisError(
                        f"Both primary and fallback implementations failed. "
                        f"Primary: {e}, Fallback: {fallback_error}",
                        severity=ErrorSeverity.CRITICAL,
                        context=ErrorContext(
                            component=func.__module__,
                            operation=func.__name__
                        )
                    )
        return wrapper
    return decorator


def create_error_context(component: str, operation: str, **parameters) -> ErrorContext:
    """Helper function to create error context."""
    return ErrorContext(
        component=component,
        operation=operation,
        parameters=parameters
    )
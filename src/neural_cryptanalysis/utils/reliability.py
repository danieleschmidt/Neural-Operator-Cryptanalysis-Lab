"""Reliability utilities including retry mechanisms, circuit breakers, and resource management."""

import time
import random
import threading
import asyncio
import functools
import gc
import os
import signal
import psutil
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from contextlib import contextmanager

from .logging_utils import get_logger
from .errors import (
    NeuralCryptanalysisError, TimeoutError, ResourceError, 
    create_error_context, ErrorSeverity
)

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit open, rejecting requests
    HALF_OPEN = "half_open"  # Testing if circuit can close


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_factor: float = 2.0
    jitter: bool = True
    exceptions: tuple = (Exception,)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    expected_exception: Type[Exception] = Exception


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = threading.RLock()
        
        logger.debug(f"Circuit breaker initialized with config: {config}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            NeuralCryptanalysisError: If circuit is open or function fails
        """
        with self.lock:
            current_time = time.time()
            
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if current_time - self.last_failure_time >= self.config.timeout:
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise NeuralCryptanalysisError(
                        "Circuit breaker is OPEN - rejecting request",
                        severity=ErrorSeverity.HIGH,
                        context=create_error_context("CircuitBreaker", "call")
                    )
            
            try:
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record success
                self._record_success(execution_time)
                return result
                
            except self.config.expected_exception as e:
                # Record failure
                self._record_failure(current_time)
                raise NeuralCryptanalysisError(
                    f"Function failed in circuit breaker: {e}",
                    severity=ErrorSeverity.MEDIUM,
                    context=create_error_context("CircuitBreaker", "call"),
                    cause=e
                )
    
    def _record_success(self, execution_time: float):
        """Record successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                logger.info("Circuit breaker transitioning to CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
        
        logger.debug(f"Circuit breaker success - execution time: {execution_time:.3f}s")
    
    def _record_failure(self, failure_time: float):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = failure_time
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                logger.warning("Circuit breaker transitioning to OPEN")
                self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker transitioning back to OPEN")
            self.state = CircuitState.OPEN
            self.success_count = 0
        
        logger.debug(f"Circuit breaker failure - count: {self.failure_count}")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current circuit breaker state information."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }


class RetryManager:
    """Advanced retry mechanism with multiple strategies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            NeuralCryptanalysisError: If all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if attempt > 0:
                    logger.info(f"Function succeeded on attempt {attempt + 1} after {execution_time:.3f}s")
                
                return result
                
            except self.config.exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")
        
        # All retries exhausted
        raise NeuralCryptanalysisError(
            f"Function failed after {self.config.max_attempts} attempts",
            severity=ErrorSeverity.HIGH,
            context=create_error_context("RetryManager", "retry", 
                                       attempts=self.config.max_attempts),
            cause=last_exception
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_factor ** attempt)
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._fibonacci(attempt + 1)
        else:
            delay = self.config.base_delay
        
        # Apply jitter
        if self.config.jitter:
            jitter_factor = 0.1
            jitter = random.uniform(-jitter_factor, jitter_factor) * delay
            delay += jitter
        
        # Ensure delay is within bounds
        return min(delay, self.config.max_delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


class ResourceManager:
    """Manage system resources and prevent resource exhaustion."""
    
    def __init__(self):
        self.resource_limits = {
            'memory_percent': 85.0,      # Max memory usage percentage
            'cpu_percent': 90.0,         # Max CPU usage percentage
            'disk_percent': 90.0,        # Max disk usage percentage
            'max_open_files': 1000,      # Max open file descriptors
            'max_threads': 100,          # Max threads
        }
        
        self.resource_monitors = {}
        self.cleanup_callbacks = []
        
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage.
        
        Returns:
            Resource usage information
        """
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Process info
            process = psutil.Process()
            open_files = len(process.open_files())
            num_threads = process.num_threads()
            
            return {
                'memory_percent': memory_percent,
                'memory_available_gb': memory.available / (1024**3),
                'cpu_percent': cpu_percent,
                'disk_percent': disk_percent,
                'disk_free_gb': disk.free / (1024**3),
                'open_files': open_files,
                'num_threads': num_threads,
                'limits': self.resource_limits.copy()
            }
            
        except Exception as e:
            logger.warning(f"Failed to check resources: {e}")
            return {}
    
    def enforce_limits(self) -> bool:
        """Enforce resource limits.
        
        Returns:
            True if within limits, False otherwise
        """
        resources = self.check_resources()
        
        # Check memory limit
        if resources.get('memory_percent', 0) > self.resource_limits['memory_percent']:
            logger.warning(f"Memory usage ({resources['memory_percent']:.1f}%) exceeds limit")
            self._trigger_cleanup()
            return False
        
        # Check CPU limit
        if resources.get('cpu_percent', 0) > self.resource_limits['cpu_percent']:
            logger.warning(f"CPU usage ({resources['cpu_percent']:.1f}%) exceeds limit")
            return False
        
        # Check disk limit
        if resources.get('disk_percent', 0) > self.resource_limits['disk_percent']:
            logger.warning(f"Disk usage ({resources['disk_percent']:.1f}%) exceeds limit")
            return False
        
        # Check file descriptors
        if resources.get('open_files', 0) > self.resource_limits['max_open_files']:
            logger.warning(f"Open files ({resources['open_files']}) exceeds limit")
            return False
        
        # Check thread count
        if resources.get('num_threads', 0) > self.resource_limits['max_threads']:
            logger.warning(f"Thread count ({resources['num_threads']}) exceeds limit")
            return False
        
        return True
    
    def register_cleanup_callback(self, callback: Callable):
        """Register cleanup callback for resource management."""
        self.cleanup_callbacks.append(callback)
    
    def _trigger_cleanup(self):
        """Trigger cleanup procedures."""
        logger.info("Triggering resource cleanup")
        
        # Run registered cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Resource cleanup completed")


class TimeoutManager:
    """Manage operation timeouts."""
    
    @staticmethod
    def timeout(seconds: float, error_message: str = "Operation timed out"):
        """Decorator to add timeout to function execution.
        
        Args:
            seconds: Timeout in seconds
            error_message: Error message for timeout
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return TimeoutManager.execute_with_timeout(
                    func, seconds, error_message, *args, **kwargs
                )
            return wrapper
        return decorator
    
    @staticmethod
    def execute_with_timeout(func: Callable, timeout_seconds: float,
                           error_message: str = "Operation timed out",
                           *args, **kwargs) -> Any:
        """Execute function with timeout.
        
        Args:
            func: Function to execute
            timeout_seconds: Timeout in seconds
            error_message: Error message for timeout
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If operation times out
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(
                error_message,
                timeout_duration=timeout_seconds,
                context=create_error_context("TimeoutManager", "execute_with_timeout")
            )
        
        # Set up signal handler for timeout
        if hasattr(signal, 'SIGALRM'):  # Unix systems
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
        
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(f"Function completed in {execution_time:.3f}s (timeout: {timeout_seconds}s)")
            return result
            
        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.warning(f"Function timed out after {timeout_seconds}s")
            raise
        finally:
            # Clean up signal handler
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


@contextmanager
def resource_context(max_memory_mb: Optional[float] = None,
                    max_cpu_percent: Optional[float] = None,
                    cleanup_on_exit: bool = True):
    """Context manager for resource-aware operations.
    
    Args:
        max_memory_mb: Maximum memory usage in MB
        max_cpu_percent: Maximum CPU usage percentage
        cleanup_on_exit: Whether to trigger cleanup on exit
    """
    resource_manager = ResourceManager()
    
    # Set custom limits if provided
    if max_memory_mb:
        # Convert MB to percentage (approximate)
        total_memory = psutil.virtual_memory().total
        memory_percent = (max_memory_mb * 1024 * 1024 / total_memory) * 100
        resource_manager.resource_limits['memory_percent'] = min(memory_percent, 95.0)
    
    if max_cpu_percent:
        resource_manager.resource_limits['cpu_percent'] = max_cpu_percent
    
    initial_resources = resource_manager.check_resources()
    logger.info(f"Resource context entered - initial usage: "
               f"Memory: {initial_resources.get('memory_percent', 0):.1f}%, "
               f"CPU: {initial_resources.get('cpu_percent', 0):.1f}%")
    
    try:
        yield resource_manager
    finally:
        if cleanup_on_exit:
            resource_manager._trigger_cleanup()
        
        final_resources = resource_manager.check_resources()
        logger.info(f"Resource context exited - final usage: "
                   f"Memory: {final_resources.get('memory_percent', 0):.1f}%, "
                   f"CPU: {final_resources.get('cpu_percent', 0):.1f}%")


def resilient_operation(retry_config: Optional[RetryConfig] = None,
                       circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                       timeout_seconds: Optional[float] = None):
    """Decorator combining retry, circuit breaker, and timeout patterns.
    
    Args:
        retry_config: Retry configuration
        circuit_breaker_config: Circuit breaker configuration  
        timeout_seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        # Create managers
        retry_manager = RetryManager(retry_config) if retry_config else None
        circuit_breaker = CircuitBreaker(circuit_breaker_config) if circuit_breaker_config else None
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Wrap function with timeout if specified
            target_func = func
            if timeout_seconds:
                target_func = lambda *a, **kw: TimeoutManager.execute_with_timeout(
                    func, timeout_seconds, f"Operation timed out after {timeout_seconds}s", *a, **kw
                )
            
            # Wrap with circuit breaker if specified
            if circuit_breaker:
                target_func = lambda *a, **kw: circuit_breaker.call(target_func, *a, **kw)
            
            # Wrap with retry if specified
            if retry_manager:
                return retry_manager.retry(target_func, *args, **kwargs)
            else:
                return target_func(*args, **kwargs)
        
        # Add methods to access internal state
        wrapper._retry_manager = retry_manager
        wrapper._circuit_breaker = circuit_breaker
        
        return wrapper
    return decorator


# Global resource manager instance
global_resource_manager = ResourceManager()


def check_system_health() -> Dict[str, Any]:
    """Check overall system health.
    
    Returns:
        System health information
    """
    resources = global_resource_manager.check_resources()
    
    # Determine health status
    health_status = "healthy"
    issues = []
    
    if resources.get('memory_percent', 0) > 80:
        health_status = "degraded"
        issues.append(f"High memory usage: {resources['memory_percent']:.1f}%")
    
    if resources.get('cpu_percent', 0) > 80:
        health_status = "degraded"
        issues.append(f"High CPU usage: {resources['cpu_percent']:.1f}%")
    
    if resources.get('disk_percent', 0) > 85:
        health_status = "degraded"
        issues.append(f"High disk usage: {resources['disk_percent']:.1f}%")
    
    if resources.get('memory_percent', 0) > 95 or resources.get('cpu_percent', 0) > 95:
        health_status = "critical"
    
    return {
        'status': health_status,
        'timestamp': time.time(),
        'resources': resources,
        'issues': issues
    }
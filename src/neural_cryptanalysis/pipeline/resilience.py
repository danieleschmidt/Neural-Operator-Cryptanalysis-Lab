"""Resilience and Fault Tolerance Module for Self-Healing Pipelines.

This module provides advanced fault tolerance patterns including circuit breakers,
bulkheads, retry mechanisms, graceful degradation, and disaster recovery capabilities.

Key Features:
- Circuit breaker pattern implementation
- Bulkhead isolation for component failures
- Intelligent retry with exponential backoff
- Graceful degradation strategies
- Health checks and service discovery
- Backup and recovery mechanisms
- Chaos engineering integration
"""

import asyncio
import json
import logging
import pickle
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Generic, TypeVar
from dataclasses import dataclass, field
import queue
import statistics

# Mock imports for dependencies that may not be available
try:
    import numpy as np
except ImportError:
    class np:
        @staticmethod
        def array(x): return x
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0
        @staticmethod
        def std(x): return statistics.stdev(x) if len(x) > 1 else 0

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    RANDOM_JITTER = "random_jitter"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0
    monitor_failures: bool = True


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    backoff_multiplier: float = 2.0


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""
    max_concurrent_calls: int = 10
    queue_size: int = 100
    timeout: float = 30.0
    rejection_strategy: str = "fail_fast"  # or "queue_overflow"


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    check_interval: float = 30.0
    timeout: float = 10.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class BulkheadRejectError(Exception):
    """Raised when bulkhead rejects a request."""
    pass


class CircuitBreaker:
    """Implementation of circuit breaker pattern."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_changed_time = datetime.now()
        
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.total_calls += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.rejected_calls += 1
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
            
            elif self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self.successful_calls += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            else:
                self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self.failed_calls += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        self.state_changed_time = datetime.now()
        self.logger.warning(f"Circuit breaker {self.name} transitioned to OPEN")
    
    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.state_changed_time = datetime.now()
        self.success_count = 0
        self.logger.info(f"Circuit breaker {self.name} transitioned to HALF_OPEN")
    
    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.state_changed_time = datetime.now()
        self.failure_count = 0
        self.success_count = 0
        self.logger.info(f"Circuit breaker {self.name} transitioned to CLOSED")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            success_rate = self.successful_calls / self.total_calls if self.total_calls > 0 else 0
            return {
                'name': self.name,
                'state': self.state.value,
                'total_calls': self.total_calls,
                'successful_calls': self.successful_calls,
                'failed_calls': self.failed_calls,
                'rejected_calls': self.rejected_calls,
                'success_rate': success_rate,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'state_changed_time': self.state_changed_time.isoformat()
            }
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition_to_closed()
            self.logger.info(f"Circuit breaker {self.name} manually reset")


class Bulkhead:
    """Implementation of bulkhead isolation pattern."""
    
    def __init__(self, name: str, config: BulkheadConfig = None):
        self.name = name
        self.config = config or BulkheadConfig()
        
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_calls)
        self.semaphore = threading.Semaphore(self.config.max_concurrent_calls)
        self.request_queue = queue.Queue(maxsize=self.config.queue_size)
        
        # Metrics
        self.active_requests = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rejected_requests = 0
        
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def submit(self, func: Callable[..., T], *args, **kwargs) -> Future[T]:
        """Submit a function for execution with bulkhead protection."""
        with self._lock:
            self.total_requests += 1
            
            if not self.semaphore.acquire(blocking=False):
                self.rejected_requests += 1
                if self.config.rejection_strategy == "fail_fast":
                    raise BulkheadRejectError(f"Bulkhead {self.name} at capacity")
                else:
                    # Try to queue the request
                    try:
                        self.request_queue.put((func, args, kwargs), block=False)
                        return self._create_queued_future()
                    except queue.Full:
                        self.rejected_requests += 1
                        raise BulkheadRejectError(f"Bulkhead {self.name} queue full")
            
            self.active_requests += 1
        
        # Submit to executor
        future = self.executor.submit(self._execute_with_tracking, func, *args, **kwargs)
        return future
    
    def _execute_with_tracking(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with tracking."""
        try:
            result = func(*args, **kwargs)
            with self._lock:
                self.successful_requests += 1
            return result
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
            raise
        finally:
            with self._lock:
                self.active_requests -= 1
            self.semaphore.release()
    
    def _create_queued_future(self) -> Future[T]:
        """Create a future for a queued request."""
        # This is a simplified implementation
        # In practice, you'd want a more sophisticated queuing mechanism
        future = Future()
        future.set_exception(BulkheadRejectError("Request queued but not implemented"))
        return future
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics."""
        with self._lock:
            success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 0
            return {
                'name': self.name,
                'active_requests': self.active_requests,
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'rejected_requests': self.rejected_requests,
                'success_rate': success_rate,
                'max_concurrent': self.config.max_concurrent_calls,
                'queue_size': self.config.queue_size
            }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the bulkhead."""
        self.executor.shutdown(wait=wait)


class RetryMechanism:
    """Intelligent retry mechanism with various strategies."""
    
    def __init__(self, name: str, config: RetryConfig = None):
        self.name = name
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.total_attempts = 0
        self.successful_retries = 0
        self.failed_retries = 0
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            self.total_attempts += 1
            
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.successful_retries += 1
                    self.logger.info(f"Retry {self.name} succeeded on attempt {attempt + 1}")
                return result
            
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(f"Retry {self.name} attempt {attempt + 1} failed, retrying in {delay:.2f}s: {str(e)}")
                    time.sleep(delay)
                else:
                    self.failed_retries += 1
                    self.logger.error(f"Retry {self.name} exhausted all {self.config.max_attempts} attempts")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry strategy."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (attempt + 1)
        
        elif self.config.strategy == RetryStrategy.RANDOM_JITTER:
            delay = self.config.base_delay + random.uniform(0, self.config.base_delay)
        
        else:
            delay = self.config.base_delay
        
        # Apply jitter if enabled
        if self.config.jitter and self.config.strategy != RetryStrategy.RANDOM_JITTER:
            jitter = random.uniform(0.1, 0.2) * delay
            delay += jitter
        
        # Respect max delay
        return min(delay, self.config.max_delay)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry metrics."""
        return {
            'name': self.name,
            'total_attempts': self.total_attempts,
            'successful_retries': self.successful_retries,
            'failed_retries': self.failed_retries,
            'success_rate': self.successful_retries / max(1, self.total_attempts - self.failed_retries),
            'strategy': self.config.strategy.value
        }


class HealthChecker:
    """Health checker for service monitoring."""
    
    def __init__(self, name: str, health_check_func: Callable[[], bool], config: HealthCheckConfig = None):
        self.name = name
        self.health_check_func = health_check_func
        self.config = config or HealthCheckConfig()
        
        self.current_status = HealthStatus.UNKNOWN
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_check_time: Optional[datetime] = None
        self.check_history: List[bool] = []
        
        self.is_running = False
        self.check_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self) -> None:
        """Start health check monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self.check_thread.start()
        self.logger.info(f"Health checker {self.name} started")
    
    def stop_monitoring(self) -> None:
        """Stop health check monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        if self.check_thread:
            self.check_thread.join(timeout=5)
        self.logger.info(f"Health checker {self.name} stopped")
    
    def _check_loop(self) -> None:
        """Health check monitoring loop."""
        while not self.stop_event.is_set():
            try:
                self._perform_health_check()
                self.stop_event.wait(self.config.check_interval)
            except Exception as e:
                self.logger.error(f"Health check loop error for {self.name}: {e}")
                self.stop_event.wait(self.config.check_interval)
    
    def _perform_health_check(self) -> None:
        """Perform a single health check."""
        try:
            # Execute health check with timeout
            start_time = time.time()
            is_healthy = self.health_check_func()
            check_duration = time.time() - start_time
            
            if check_duration > self.config.timeout:
                is_healthy = False
                self.logger.warning(f"Health check {self.name} timed out ({check_duration:.2f}s)")
            
            with self._lock:
                self.last_check_time = datetime.now()
                self.check_history.append(is_healthy)
                
                # Keep only recent history
                if len(self.check_history) > 100:
                    self.check_history.pop(0)
                
                if is_healthy:
                    self.consecutive_failures = 0
                    self.consecutive_successes += 1
                    
                    if self.current_status != HealthStatus.HEALTHY:
                        if self.consecutive_successes >= self.config.healthy_threshold:
                            self._transition_to_healthy()
                else:
                    self.consecutive_successes = 0
                    self.consecutive_failures += 1
                    
                    if self.consecutive_failures >= self.config.unhealthy_threshold:
                        self._transition_to_unhealthy()
        
        except Exception as e:
            self.logger.error(f"Health check {self.name} failed with exception: {e}")
            with self._lock:
                self.consecutive_successes = 0
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.config.unhealthy_threshold:
                    self._transition_to_unhealthy()
    
    def _transition_to_healthy(self) -> None:
        """Transition to healthy status."""
        old_status = self.current_status
        self.current_status = HealthStatus.HEALTHY
        self.logger.info(f"Health checker {self.name}: {old_status.value} -> {self.current_status.value}")
    
    def _transition_to_unhealthy(self) -> None:
        """Transition to unhealthy status."""
        old_status = self.current_status
        self.current_status = HealthStatus.UNHEALTHY
        self.logger.warning(f"Health checker {self.name}: {old_status.value} -> {self.current_status.value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            recent_checks = self.check_history[-10:] if self.check_history else []
            success_rate = sum(recent_checks) / len(recent_checks) if recent_checks else 0
            
            return {
                'name': self.name,
                'status': self.current_status.value,
                'consecutive_failures': self.consecutive_failures,
                'consecutive_successes': self.consecutive_successes,
                'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
                'recent_success_rate': success_rate,
                'total_checks': len(self.check_history),
                'is_monitoring': self.is_running
            }


class GracefulDegradation:
    """Manages graceful degradation strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.degradation_levels: Dict[str, Callable] = {}
        self.current_level = "normal"
        self.fallback_strategies: Dict[str, Callable] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def register_degradation_level(self, level: str, strategy_func: Callable) -> None:
        """Register a degradation level with its strategy."""
        self.degradation_levels[level] = strategy_func
        self.logger.info(f"Registered degradation level '{level}' for {self.name}")
    
    def register_fallback(self, operation: str, fallback_func: Callable) -> None:
        """Register a fallback strategy for an operation."""
        self.fallback_strategies[operation] = fallback_func
        self.logger.info(f"Registered fallback for operation '{operation}' in {self.name}")
    
    def degrade_to_level(self, level: str) -> bool:
        """Degrade to a specific level."""
        if level not in self.degradation_levels:
            self.logger.error(f"Unknown degradation level: {level}")
            return False
        
        try:
            old_level = self.current_level
            self.degradation_levels[level]()
            self.current_level = level
            self.logger.warning(f"Degraded {self.name} from '{old_level}' to '{level}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to degrade to level '{level}': {e}")
            return False
    
    def execute_with_fallback(self, operation: str, primary_func: Callable[..., T], *args, **kwargs) -> T:
        """Execute operation with fallback if it fails."""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary operation '{operation}' failed: {e}")
            
            if operation in self.fallback_strategies:
                try:
                    self.logger.info(f"Executing fallback for operation '{operation}'")
                    return self.fallback_strategies[operation](*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback for operation '{operation}' also failed: {fallback_error}")
                    raise
            else:
                self.logger.error(f"No fallback strategy for operation '{operation}'")
                raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get degradation status."""
        return {
            'name': self.name,
            'current_level': self.current_level,
            'available_levels': list(self.degradation_levels.keys()),
            'fallback_operations': list(self.fallback_strategies.keys())
        }


class DisasterRecovery:
    """Disaster recovery and backup management."""
    
    def __init__(self, name: str, backup_dir: Path = Path("backups")):
        self.name = name
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(exist_ok=True)
        
        self.backup_strategies: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def register_backup_strategy(self, data_type: str, backup_func: Callable) -> None:
        """Register a backup strategy for a data type."""
        self.backup_strategies[data_type] = backup_func
        self.logger.info(f"Registered backup strategy for '{data_type}'")
    
    def register_recovery_strategy(self, data_type: str, recovery_func: Callable) -> None:
        """Register a recovery strategy for a data type."""
        self.recovery_strategies[data_type] = recovery_func
        self.logger.info(f"Registered recovery strategy for '{data_type}'")
    
    def create_backup(self, data_type: str, data: Any, backup_id: str = None) -> str:
        """Create a backup of specified data."""
        if data_type not in self.backup_strategies:
            raise ValueError(f"No backup strategy for data type: {data_type}")
        
        backup_id = backup_id or f"{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / f"{backup_id}.backup"
        
        try:
            backup_data = self.backup_strategies[data_type](data)
            
            with open(backup_path, 'wb') as f:
                pickle.dump({
                    'data_type': data_type,
                    'backup_id': backup_id,
                    'created_at': datetime.now(),
                    'data': backup_data
                }, f)
            
            self.logger.info(f"Created backup '{backup_id}' at {backup_path}")
            return backup_id
            
        except Exception as e:
            self.logger.error(f"Failed to create backup '{backup_id}': {e}")
            raise
    
    def restore_backup(self, backup_id: str) -> Any:
        """Restore data from a backup."""
        backup_path = self.backup_dir / f"{backup_id}.backup"
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup '{backup_id}' not found")
        
        try:
            with open(backup_path, 'rb') as f:
                backup_info = pickle.load(f)
            
            data_type = backup_info['data_type']
            backup_data = backup_info['data']
            
            if data_type not in self.recovery_strategies:
                raise ValueError(f"No recovery strategy for data type: {data_type}")
            
            restored_data = self.recovery_strategies[data_type](backup_data)
            self.logger.info(f"Restored backup '{backup_id}' (type: {data_type})")
            return restored_data
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup '{backup_id}': {e}")
            raise
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        backups = []
        
        for backup_file in self.backup_dir.glob("*.backup"):
            try:
                with open(backup_file, 'rb') as f:
                    backup_info = pickle.load(f)
                
                backups.append({
                    'backup_id': backup_info['backup_id'],
                    'data_type': backup_info['data_type'],
                    'created_at': backup_info['created_at'],
                    'file_size': backup_file.stat().st_size
                })
            except Exception as e:
                self.logger.warning(f"Failed to read backup {backup_file}: {e}")
        
        return sorted(backups, key=lambda x: x['created_at'], reverse=True)
    
    def cleanup_old_backups(self, max_age_days: int = 30, max_count: int = 50) -> int:
        """Clean up old backups."""
        backups = self.list_backups()
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0
        
        # Remove old backups
        for backup in backups:
            if backup['created_at'] < cutoff_date or len(backups) - removed_count > max_count:
                backup_path = self.backup_dir / f"{backup['backup_id']}.backup"
                try:
                    backup_path.unlink()
                    removed_count += 1
                    self.logger.info(f"Removed old backup: {backup['backup_id']}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove backup {backup['backup_id']}: {e}")
        
        return removed_count


class ResilienceManager:
    """Main manager for all resilience patterns."""
    
    def __init__(self, name: str):
        self.name = name
        
        # Pattern instances
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.retry_mechanisms: Dict[str, RetryMechanism] = {}
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.degradation_managers: Dict[str, GracefulDegradation] = {}
        self.disaster_recovery = DisasterRecovery(name)
        
        self.logger = logging.getLogger(__name__)
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        cb = CircuitBreaker(name, config)
        self.circuit_breakers[name] = cb
        self.logger.info(f"Created circuit breaker: {name}")
        return cb
    
    def create_bulkhead(self, name: str, config: BulkheadConfig = None) -> Bulkhead:
        """Create and register a bulkhead."""
        bulkhead = Bulkhead(name, config)
        self.bulkheads[name] = bulkhead
        self.logger.info(f"Created bulkhead: {name}")
        return bulkhead
    
    def create_retry_mechanism(self, name: str, config: RetryConfig = None) -> RetryMechanism:
        """Create and register a retry mechanism."""
        retry = RetryMechanism(name, config)
        self.retry_mechanisms[name] = retry
        self.logger.info(f"Created retry mechanism: {name}")
        return retry
    
    def create_health_checker(self, name: str, health_func: Callable[[], bool], config: HealthCheckConfig = None) -> HealthChecker:
        """Create and register a health checker."""
        checker = HealthChecker(name, health_func, config)
        self.health_checkers[name] = checker
        self.logger.info(f"Created health checker: {name}")
        return checker
    
    def create_degradation_manager(self, name: str) -> GracefulDegradation:
        """Create and register a degradation manager."""
        degradation = GracefulDegradation(name)
        self.degradation_managers[name] = degradation
        self.logger.info(f"Created degradation manager: {name}")
        return degradation
    
    @contextmanager
    def resilient_operation(
        self,
        circuit_breaker: str = None,
        bulkhead: str = None,
        retry: str = None,
        fallback: Callable = None
    ):
        """Context manager for resilient operations."""
        try:
            # Acquire bulkhead if specified
            if bulkhead and bulkhead in self.bulkheads:
                if not self.bulkheads[bulkhead].semaphore.acquire(blocking=False):
                    raise BulkheadRejectError(f"Bulkhead {bulkhead} at capacity")
            
            yield
            
        except Exception as e:
            if fallback:
                try:
                    return fallback(e)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed: {fallback_error}")
            raise
        finally:
            # Release bulkhead if acquired
            if bulkhead and bulkhead in self.bulkheads:
                try:
                    self.bulkheads[bulkhead].semaphore.release()
                except:
                    pass
    
    def execute_resilient_operation(
        self,
        func: Callable[..., T],
        *args,
        circuit_breaker: str = None,
        bulkhead: str = None,
        retry: str = None,
        fallback: Callable = None,
        **kwargs
    ) -> T:
        """Execute operation with specified resilience patterns."""
        
        def wrapped_func():
            # Apply circuit breaker if specified
            if circuit_breaker and circuit_breaker in self.circuit_breakers:
                return self.circuit_breakers[circuit_breaker].call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        # Apply retry if specified
        if retry and retry in self.retry_mechanisms:
            wrapped_func = lambda: self.retry_mechanisms[retry].execute(wrapped_func)
        
        # Apply bulkhead if specified
        if bulkhead and bulkhead in self.bulkheads:
            future = self.bulkheads[bulkhead].submit(wrapped_func)
            return future.result()
        else:
            try:
                return wrapped_func()
            except Exception as e:
                if fallback:
                    try:
                        return fallback(e, *args, **kwargs)
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback also failed: {fallback_error}")
                raise
    
    def start_all_health_checkers(self) -> None:
        """Start all registered health checkers."""
        for checker in self.health_checkers.values():
            checker.start_monitoring()
    
    def stop_all_health_checkers(self) -> None:
        """Stop all registered health checkers."""
        for checker in self.health_checkers.values():
            checker.stop_monitoring()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_status = {}
        
        # Circuit breaker status
        cb_status = {name: cb.get_metrics() for name, cb in self.circuit_breakers.items()}
        
        # Bulkhead status
        bulkhead_status = {name: bh.get_metrics() for name, bh in self.bulkheads.items()}
        
        # Health checker status
        health_checker_status = {name: hc.get_status() for name, hc in self.health_checkers.items()}
        
        # Retry mechanism status
        retry_status = {name: rm.get_metrics() for name, rm in self.retry_mechanisms.items()}
        
        # Degradation status
        degradation_status = {name: dm.get_status() for name, dm in self.degradation_managers.items()}
        
        # Calculate overall health score
        unhealthy_components = 0
        total_components = 0
        
        for status in health_checker_status.values():
            total_components += 1
            if status['status'] == 'unhealthy':
                unhealthy_components += 1
        
        for metrics in cb_status.values():
            total_components += 1
            if metrics['state'] == 'open':
                unhealthy_components += 1
        
        overall_health = 1.0 - (unhealthy_components / max(1, total_components))
        
        return {
            'overall_health_score': overall_health,
            'circuit_breakers': cb_status,
            'bulkheads': bulkhead_status,
            'health_checkers': health_checker_status,
            'retry_mechanisms': retry_status,
            'degradation_managers': degradation_status,
            'disaster_recovery': {
                'available_backups': len(self.disaster_recovery.list_backups())
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown all components."""
        self.stop_all_health_checkers()
        
        for bulkhead in self.bulkheads.values():
            bulkhead.shutdown()
        
        self.logger.info(f"Resilience manager {self.name} shut down")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create resilience manager
    resilience = ResilienceManager("test_system")
    
    # Create components
    cb = resilience.create_circuit_breaker("test_cb", CircuitBreakerConfig(failure_threshold=3))
    bulkhead = resilience.create_bulkhead("test_bulkhead", BulkheadConfig(max_concurrent_calls=2))
    retry = resilience.create_retry_mechanism("test_retry", RetryConfig(max_attempts=3))
    
    # Create health checker
    def dummy_health_check():
        return random.random() > 0.3  # 70% chance of being healthy
    
    health_checker = resilience.create_health_checker("test_service", dummy_health_check)
    health_checker.start_monitoring()
    
    # Test resilient operation
    def unreliable_function(x):
        if random.random() < 0.4:  # 40% chance of failure
            raise Exception("Random failure")
        return x * 2
    
    def fallback_function(error, x):
        return x  # Just return the input as fallback
    
    try:
        # Execute with resilience patterns
        result = resilience.execute_resilient_operation(
            unreliable_function,
            5,
            circuit_breaker="test_cb",
            retry="test_retry",
            fallback=fallback_function
        )
        print(f"Result: {result}")
        
        # Get system health
        health = resilience.get_system_health()
        print(f"System health: {health['overall_health_score']:.2f}")
        
    finally:
        resilience.shutdown()
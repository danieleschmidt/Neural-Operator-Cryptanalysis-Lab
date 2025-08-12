"""Generation 3 Advanced Performance Optimization Framework.

This module provides comprehensive performance optimization for neural cryptanalysis
including intelligent caching, memory pooling, concurrent processing, and auto-scaling.
"""

import time
import threading
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from pathlib import Path
import pickle
import hashlib
import json
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from collections import OrderedDict, defaultdict
import warnings
import psutil
import heapq
import queue
import signal

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch
    import torch.multiprocessing as torch_mp
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    torch_mp = None

from ..utils.logging_utils import get_logger
from ..utils.errors import NeuralCryptanalysisError, ResourceError

logger = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    # Caching configuration
    cache_enabled: bool = True
    cache_max_size_mb: int = 2000
    cache_ttl_hours: int = 48
    cache_levels: int = 3  # L1, L2, L3 cache hierarchy
    
    # Memory management
    memory_pool_size_gb: float = 4.0
    memory_auto_cleanup: bool = True
    memory_threshold_percent: float = 85.0
    
    # Concurrent processing
    max_workers: int = None
    async_io_enabled: bool = True
    thread_pool_size: int = None
    process_pool_size: int = None
    
    # Auto-scaling
    auto_scaling_enabled: bool = True
    scaling_threshold_cpu: float = 80.0
    scaling_threshold_memory: float = 80.0
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.8
    
    # Performance monitoring
    profiling_enabled: bool = True
    metrics_collection_interval: float = 1.0
    performance_alerts: bool = True
    
    # Resource limits
    max_batch_size: int = 1024
    max_concurrent_operations: int = 50
    operation_timeout: float = 300.0


@dataclass
class PerformanceMetrics:
    """Enhanced performance metrics."""
    operation: str
    duration: float
    memory_used: float
    cpu_percent: float
    gpu_memory: Optional[float] = None
    throughput: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    concurrency_level: int = 1
    batch_size: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LRUCache:
    """Enhanced LRU Cache with TTL and memory management."""
    
    def __init__(self, max_size: int, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with TTL check."""
        with self.lock:
            if key not in self.cache:
                self._misses += 1
                return default
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                self._misses += 1
                return default
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self._hits += 1
            return value
    
    def set(self, key: str, value: Any):
        """Set item in cache with size management."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
                self.cache[key] = value
                self.timestamps[key] = time.time()
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove oldest
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
                
                self.cache[key] = value
                self.timestamps[key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'fill_ratio': len(self.cache) / self.max_size
        }


class MemoryPool:
    """Advanced memory pool with automatic cleanup and monitoring."""
    
    def __init__(self, pool_size_gb: float, auto_cleanup: bool = True):
        self.pool_size_bytes = int(pool_size_gb * 1024 * 1024 * 1024)
        self.auto_cleanup = auto_cleanup
        self.allocations = {}
        self.free_blocks = []
        self.allocated_size = 0
        self.lock = threading.RLock()
        self.cleanup_threshold = 0.9  # 90% full triggers cleanup
        
        # Weak references for automatic cleanup
        self.weak_refs = weakref.WeakValueDictionary()
        
        logger.info(f"Memory pool initialized: {pool_size_gb}GB")
    
    def allocate(self, name: str, size_bytes: int, alignment: int = 8) -> Optional[int]:
        """Allocate memory from pool."""
        with self.lock:
            # Align size
            aligned_size = ((size_bytes + alignment - 1) // alignment) * alignment
            
            if self.allocated_size + aligned_size > self.pool_size_bytes:
                if self.auto_cleanup:
                    self._cleanup_freed_memory()
                    
                if self.allocated_size + aligned_size > self.pool_size_bytes:
                    logger.warning(f"Memory allocation failed: {name} ({size_bytes} bytes)")
                    return None
            
            # Find or create allocation
            offset = self._find_free_block(aligned_size)
            if offset is None:
                offset = self.allocated_size
                self.allocated_size += aligned_size
            
            allocation = {
                'offset': offset,
                'size': aligned_size,
                'timestamp': time.time(),
                'name': name
            }
            
            self.allocations[name] = allocation
            logger.debug(f"Allocated {aligned_size} bytes for {name} at offset {offset}")
            
            return offset
    
    def deallocate(self, name: str):
        """Deallocate memory."""
        with self.lock:
            if name in self.allocations:
                allocation = self.allocations[name]
                self.free_blocks.append({
                    'offset': allocation['offset'],
                    'size': allocation['size']
                })
                del self.allocations[name]
                logger.debug(f"Deallocated memory for {name}")
    
    def get_usage(self) -> Dict[str, Any]:
        """Get memory pool usage statistics."""
        with self.lock:
            allocated_mb = sum(alloc['size'] for alloc in self.allocations.values()) / 1024 / 1024
            free_mb = sum(block['size'] for block in self.free_blocks) / 1024 / 1024
            
            return {
                'allocated_mb': allocated_mb,
                'free_mb': free_mb,
                'total_mb': self.pool_size_bytes / 1024 / 1024,
                'usage_percent': (allocated_mb / (self.pool_size_bytes / 1024 / 1024)) * 100,
                'fragmentation': len(self.free_blocks),
                'active_allocations': len(self.allocations)
            }
    
    def _find_free_block(self, size: int) -> Optional[int]:
        """Find suitable free block."""
        for i, block in enumerate(self.free_blocks):
            if block['size'] >= size:
                offset = block['offset']
                
                if block['size'] > size:
                    # Split block
                    self.free_blocks[i] = {
                        'offset': offset + size,
                        'size': block['size'] - size
                    }
                else:
                    # Use entire block
                    del self.free_blocks[i]
                
                return offset
        
        return None
    
    def _cleanup_freed_memory(self):
        """Cleanup freed memory and defragment."""
        # Sort free blocks by offset
        self.free_blocks.sort(key=lambda x: x['offset'])
        
        # Merge adjacent blocks
        merged_blocks = []
        for block in self.free_blocks:
            if merged_blocks and merged_blocks[-1]['offset'] + merged_blocks[-1]['size'] == block['offset']:
                # Merge with previous block
                merged_blocks[-1]['size'] += block['size']
            else:
                merged_blocks.append(block)
        
        self.free_blocks = merged_blocks
        logger.debug(f"Memory defragmentation completed: {len(self.free_blocks)} free blocks")


class HierarchicalCache:
    """Multi-level cache hierarchy (L1, L2, L3)."""
    
    def __init__(self, l1_size: int = 100, l2_size: int = 1000, l3_size: int = 10000):
        self.l1_cache = LRUCache(l1_size, ttl_seconds=300)    # 5 min TTL
        self.l2_cache = LRUCache(l2_size, ttl_seconds=3600)   # 1 hour TTL
        self.l3_cache = LRUCache(l3_size, ttl_seconds=86400)  # 24 hour TTL
        
        self.prefetch_queue = queue.Queue()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        
        logger.info("Hierarchical cache initialized")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache hierarchy."""
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.set(key, value)
            return value
        
        # Try L3
        value = self.l3_cache.get(key)
        if value is not None:
            # Promote to L2 and L1
            self.l2_cache.set(key, value)
            self.l1_cache.set(key, value)
            return value
        
        return default
    
    def set(self, key: str, value: Any, level: int = 1):
        """Set value in cache hierarchy."""
        if level >= 1:
            self.l1_cache.set(key, value)
        if level >= 2:
            self.l2_cache.set(key, value)
        if level >= 3:
            self.l3_cache.set(key, value)
    
    def prefetch(self, key: str, loader_func: Callable):
        """Queue key for prefetching."""
        self.prefetch_queue.put((key, loader_func))
    
    def _prefetch_worker(self):
        """Background prefetching worker."""
        while True:
            try:
                key, loader_func = self.prefetch_queue.get(timeout=1)
                
                # Check if already cached
                if self.get(key) is None:
                    try:
                        value = loader_func(key)
                        self.set(key, value, level=3)  # Start in L3
                        logger.debug(f"Prefetched: {key}")
                    except Exception as e:
                        logger.warning(f"Prefetch failed for {key}: {e}")
                
                self.prefetch_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache hierarchy statistics."""
        return {
            'l1': self.l1_cache.get_stats(),
            'l2': self.l2_cache.get_stats(),
            'l3': self.l3_cache.get_stats(),
            'prefetch_queue_size': self.prefetch_queue.qsize()
        }


class ConcurrentProcessor:
    """Advanced concurrent processing with adaptive scaling."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.thread_pool = None
        self.process_pool = None
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_counter = 0
        self.lock = threading.RLock()
        
        # Initialize pools
        self._initialize_pools()
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor(config)
        self.resource_monitor.start()
        
        logger.info("Concurrent processor initialized")
    
    def _initialize_pools(self):
        """Initialize thread and process pools."""
        max_workers = self.config.max_workers or min(32, (mp.cpu_count() or 1) + 4)
        
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size or max_workers,
            thread_name_prefix="neural_crypto"
        )
        
        if self.config.process_pool_size:
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.config.process_pool_size
            )
    
    async def submit_async(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task for asynchronous execution."""
        if not self.config.async_io_enabled:
            return func(*args, **kwargs)
        
        loop = asyncio.get_event_loop()
        
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    def submit_batch(self, func: Callable, items: List[Any], 
                    use_processes: bool = False) -> List[Any]:
        """Submit batch of tasks for concurrent execution."""
        with self.lock:
            task_id = self.task_counter
            self.task_counter += 1
        
        executor = self.process_pool if use_processes and self.process_pool else self.thread_pool
        
        if not executor:
            # Fallback to sequential processing
            return [func(item) for item in items]
        
        # Submit all tasks
        futures = [executor.submit(func, item) for item in items]
        self.active_tasks[task_id] = futures
        
        try:
            # Collect results with timeout
            results = []
            for future in as_completed(futures, timeout=self.config.operation_timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Task failed: {e}")
                    results.append(None)
            
            return results
            
        finally:
            # Cleanup
            with self.lock:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
    
    def process_in_chunks(self, func: Callable, data: List[Any], 
                         chunk_size: int = None) -> List[Any]:
        """Process data in adaptive chunks."""
        if not data:
            return []
        
        # Adaptive chunk size based on system resources
        if chunk_size is None:
            available_memory = psutil.virtual_memory().available
            estimated_item_size = 1024 * 1024  # 1MB estimate
            max_chunk_by_memory = max(1, available_memory // (estimated_item_size * 10))
            chunk_size = min(self.config.max_batch_size, max_chunk_by_memory)
        
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_results = self.submit_batch(func, chunk)
            results.extend(chunk_results)
            
            # Adaptive pause if resources are stressed
            if self.resource_monitor.is_stressed():
                time.sleep(0.1)
        
        return results
    
    def shutdown(self):
        """Shutdown executor pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        self.resource_monitor.stop()
        logger.info("Concurrent processor shutdown")


class ResourceMonitor:
    """Real-time resource monitoring and alerting."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self.alerts = []
        self.lock = threading.RLock()
        
        # Thresholds
        self.cpu_threshold = config.scaling_threshold_cpu
        self.memory_threshold = config.scaling_threshold_memory
        
    def start(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                
                with self.lock:
                    self.metrics_history.append(metrics)
                    
                    # Keep only recent history
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                    
                    # Check for alerts
                    self._check_alerts(metrics)
                
                time.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(1)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / 1024**3,
            'memory_used_gb': memory.used / 1024**3,
            'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }
        
        # GPU metrics if available
        if HAS_TORCH and torch.cuda.is_available():
            try:
                metrics['gpu_memory_used'] = torch.cuda.memory_allocated() / 1024**3
                metrics['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1024**3
                metrics['gpu_utilization'] = self._get_gpu_utilization()
            except Exception:
                pass
        
        return metrics
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return None
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for resource alerts."""
        if not self.config.performance_alerts:
            return
        
        current_time = time.time()
        
        # CPU alert
        if metrics['cpu_percent'] > self.cpu_threshold:
            self.alerts.append({
                'type': 'cpu_high',
                'value': metrics['cpu_percent'],
                'threshold': self.cpu_threshold,
                'timestamp': current_time
            })
        
        # Memory alert
        if metrics['memory_percent'] > self.memory_threshold:
            self.alerts.append({
                'type': 'memory_high',
                'value': metrics['memory_percent'],
                'threshold': self.memory_threshold,
                'timestamp': current_time
            })
        
        # Keep only recent alerts
        cutoff_time = current_time - 3600  # 1 hour
        self.alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
    
    def is_stressed(self) -> bool:
        """Check if system is under stress."""
        if not self.metrics_history:
            return False
        
        recent_metrics = self.metrics_history[-5:]  # Last 5 samples
        avg_cpu = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m['memory_percent'] for m in recent_metrics) / len(recent_metrics)
        
        return avg_cpu > self.cpu_threshold or avg_memory > self.memory_threshold
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else {}
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts."""
        with self.lock:
            return self.alerts.copy()


class AutoScaler:
    """Automatic resource scaling based on load."""
    
    def __init__(self, config: OptimizationConfig, resource_monitor: ResourceMonitor):
        self.config = config
        self.resource_monitor = resource_monitor
        self.scaling_history = []
        self.last_scale_time = 0
        self.cooldown_period = 60  # 1 minute cooldown
        
        # Current scaling factors
        self.current_batch_size = 64
        self.current_worker_count = mp.cpu_count() or 1
        
        logger.info("Auto-scaler initialized")
    
    def should_scale_up(self) -> bool:
        """Check if should scale up resources."""
        if not self.config.auto_scaling_enabled:
            return False
        
        if time.time() - self.last_scale_time < self.cooldown_period:
            return False
        
        metrics = self.resource_monitor.get_current_metrics()
        if not metrics:
            return False
        
        # Scale up if consistently high resource usage
        return (metrics.get('cpu_percent', 0) > self.config.scaling_threshold_cpu and
                metrics.get('memory_percent', 0) < 90)  # Don't scale up if memory is too high
    
    def should_scale_down(self) -> bool:
        """Check if should scale down resources."""
        if not self.config.auto_scaling_enabled:
            return False
        
        if time.time() - self.last_scale_time < self.cooldown_period:
            return False
        
        metrics = self.resource_monitor.get_current_metrics()
        if not metrics:
            return False
        
        # Scale down if consistently low resource usage
        return (metrics.get('cpu_percent', 0) < self.config.scaling_threshold_cpu * 0.5 and
                metrics.get('memory_percent', 0) < self.config.scaling_threshold_memory * 0.5)
    
    def scale_up(self) -> Dict[str, Any]:
        """Scale up resources."""
        old_batch_size = self.current_batch_size
        old_worker_count = self.current_worker_count
        
        self.current_batch_size = min(
            self.config.max_batch_size,
            int(self.current_batch_size * self.config.scale_up_factor)
        )
        
        self.current_worker_count = min(
            self.config.max_concurrent_operations,
            int(self.current_worker_count * self.config.scale_up_factor)
        )
        
        self.last_scale_time = time.time()
        
        scaling_event = {
            'action': 'scale_up',
            'timestamp': self.last_scale_time,
            'old_batch_size': old_batch_size,
            'new_batch_size': self.current_batch_size,
            'old_worker_count': old_worker_count,
            'new_worker_count': self.current_worker_count
        }
        
        self.scaling_history.append(scaling_event)
        logger.info(f"Scaled up: batch_size {old_batch_size} -> {self.current_batch_size}, "
                   f"workers {old_worker_count} -> {self.current_worker_count}")
        
        return scaling_event
    
    def scale_down(self) -> Dict[str, Any]:
        """Scale down resources."""
        old_batch_size = self.current_batch_size
        old_worker_count = self.current_worker_count
        
        self.current_batch_size = max(
            1,
            int(self.current_batch_size * self.config.scale_down_factor)
        )
        
        self.current_worker_count = max(
            1,
            int(self.current_worker_count * self.config.scale_down_factor)
        )
        
        self.last_scale_time = time.time()
        
        scaling_event = {
            'action': 'scale_down',
            'timestamp': self.last_scale_time,
            'old_batch_size': old_batch_size,
            'new_batch_size': self.current_batch_size,
            'old_worker_count': old_worker_count,
            'new_worker_count': self.current_worker_count
        }
        
        self.scaling_history.append(scaling_event)
        logger.info(f"Scaled down: batch_size {old_batch_size} -> {self.current_batch_size}, "
                   f"workers {old_worker_count} -> {self.current_worker_count}")
        
        return scaling_event
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current scaling configuration."""
        return {
            'batch_size': self.current_batch_size,
            'worker_count': self.current_worker_count,
            'scaling_history': self.scaling_history[-10:]  # Last 10 events
        }


class AdvancedPerformanceOptimizer:
    """Generation 3 Advanced Performance Optimization Framework."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # Core components
        self.memory_pool = MemoryPool(
            self.config.memory_pool_size_gb,
            self.config.memory_auto_cleanup
        )
        
        self.cache = HierarchicalCache(
            l1_size=min(1000, self.config.cache_max_size_mb // 10),
            l2_size=min(10000, self.config.cache_max_size_mb // 2),
            l3_size=self.config.cache_max_size_mb * 10
        )
        
        self.concurrent_processor = ConcurrentProcessor(self.config)
        
        self.resource_monitor = self.concurrent_processor.resource_monitor
        self.auto_scaler = AutoScaler(self.config, self.resource_monitor)
        
        # Performance tracking
        self.metrics_collector = PerformanceMetricsCollector()
        
        # Optimization state
        self.optimization_state = {
            'initialized': True,
            'start_time': time.time(),
            'total_operations': 0,
            'total_cache_hits': 0,
            'total_cache_misses': 0,
            'memory_saved_mb': 0,
            'time_saved_seconds': 0
        }
        
        logger.info("Advanced Performance Optimizer initialized")
    
    def optimize_operation(self, operation_name: str, func: Callable, 
                          *args, use_cache: bool = True, use_async: bool = False,
                          **kwargs) -> Any:
        """Optimize a single operation with all available techniques."""
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(operation_name, args, kwargs)
        
        # Try cache first
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.optimization_state['total_cache_hits'] += 1
                self.optimization_state['time_saved_seconds'] += 0.1  # Estimate
                logger.debug(f"Cache hit for {operation_name}")
                return cached_result
            else:
                self.optimization_state['total_cache_misses'] += 1
        
        # Execute operation
        try:
            if use_async and self.config.async_io_enabled:
                # Use async execution
                result = asyncio.run(self.concurrent_processor.submit_async(func, *args, **kwargs))
            else:
                # Check for auto-scaling
                if self.auto_scaler.should_scale_up():
                    self.auto_scaler.scale_up()
                elif self.auto_scaler.should_scale_down():
                    self.auto_scaler.scale_down()
                
                result = func(*args, **kwargs)
            
            # Cache result
            if use_cache and result is not None:
                self.cache.set(cache_key, result)
            
            # Record metrics
            duration = time.time() - start_time
            self._record_operation_metrics(operation_name, duration, args, kwargs)
            
            self.optimization_state['total_operations'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
    
    def optimize_batch(self, operation_name: str, func: Callable, 
                      items: List[Any], use_processes: bool = False) -> List[Any]:
        """Optimize batch processing with adaptive chunking and scaling."""
        start_time = time.time()
        
        if not items:
            return []
        
        # Get current scaling config
        scaling_config = self.auto_scaler.get_current_config()
        chunk_size = min(scaling_config['batch_size'], len(items))
        
        logger.info(f"Processing batch of {len(items)} items with chunk size {chunk_size}")
        
        # Process in optimized chunks
        results = self.concurrent_processor.process_in_chunks(
            func, items, chunk_size
        )
        
        # Record metrics
        duration = time.time() - start_time
        throughput = len(items) / duration if duration > 0 else 0
        
        self.metrics_collector.record_batch_metrics(
            operation_name, len(items), duration, throughput
        )
        
        return results
    
    def warm_cache(self, keys_and_loaders: List[Tuple[str, Callable]]):
        """Warm cache with frequently accessed data."""
        logger.info(f"Warming cache with {len(keys_and_loaders)} items")
        
        for key, loader_func in keys_and_loaders:
            self.cache.prefetch(key, loader_func)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        runtime = time.time() - self.optimization_state['start_time']
        
        report = {
            'runtime_seconds': runtime,
            'total_operations': self.optimization_state['total_operations'],
            'operations_per_second': self.optimization_state['total_operations'] / runtime if runtime > 0 else 0,
            'cache_stats': self.cache.get_stats(),
            'memory_stats': self.memory_pool.get_usage(),
            'resource_stats': self.resource_monitor.get_current_metrics(),
            'scaling_stats': self.auto_scaler.get_current_config(),
            'performance_metrics': self.metrics_collector.get_summary(),
            'optimization_state': self.optimization_state.copy(),
            'alerts': self.resource_monitor.get_alerts()
        }
        
        # Calculate efficiency metrics
        total_requests = self.optimization_state['total_cache_hits'] + self.optimization_state['total_cache_misses']
        cache_hit_rate = self.optimization_state['total_cache_hits'] / total_requests if total_requests > 0 else 0
        
        report['efficiency'] = {
            'cache_hit_rate': cache_hit_rate,
            'time_saved_percent': (self.optimization_state['time_saved_seconds'] / runtime) * 100 if runtime > 0 else 0,
            'memory_efficiency': self.memory_pool.get_usage()['usage_percent'],
            'resource_utilization': {
                'cpu': report['resource_stats'].get('cpu_percent', 0),
                'memory': report['resource_stats'].get('memory_percent', 0)
            }
        }
        
        return report
    
    def _generate_cache_key(self, operation_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for operation."""
        # Create deterministic key from inputs
        key_data = {
            'operation': operation_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def _record_operation_metrics(self, operation_name: str, duration: float, 
                                 args: tuple, kwargs: dict):
        """Record performance metrics for operation."""
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration=duration,
            memory_used=memory_usage,
            cpu_percent=cpu_percent,
            batch_size=len(args[0]) if args and hasattr(args[0], '__len__') else 1
        )
        
        self.metrics_collector.record_metrics(metrics)
    
    def shutdown(self):
        """Shutdown optimizer and cleanup resources."""
        logger.info("Shutting down performance optimizer")
        
        self.concurrent_processor.shutdown()
        self.resource_monitor.stop()
        
        # Save final metrics
        final_report = self.get_optimization_report()
        logger.info(f"Final optimization report: {final_report['efficiency']}")


class PerformanceMetricsCollector:
    """Collects and analyzes performance metrics."""
    
    def __init__(self):
        self.metrics = []
        self.lock = threading.RLock()
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self.lock:
            self.metrics.append(metrics)
            
            # Keep only recent metrics
            if len(self.metrics) > 10000:
                self.metrics = self.metrics[-5000:]
    
    def record_batch_metrics(self, operation: str, batch_size: int, 
                           duration: float, throughput: float):
        """Record batch processing metrics."""
        metrics = PerformanceMetrics(
            operation=f"{operation}_batch",
            duration=duration,
            memory_used=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_percent=psutil.cpu_percent(),
            throughput=throughput,
            batch_size=batch_size
        )
        
        self.record_metrics(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self.lock:
            if not self.metrics:
                return {}
            
            durations = [m.duration for m in self.metrics]
            throughputs = [m.throughput for m in self.metrics if m.throughput]
            
            return {
                'total_operations': len(self.metrics),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0,
                'operations_by_type': self._group_by_operation(),
                'recent_performance': self._get_recent_performance()
            }
    
    def _group_by_operation(self) -> Dict[str, int]:
        """Group metrics by operation type."""
        operation_counts = defaultdict(int)
        for metrics in self.metrics:
            operation_counts[metrics.operation] += 1
        return dict(operation_counts)
    
    def _get_recent_performance(self) -> Dict[str, Any]:
        """Get recent performance trend."""
        if len(self.metrics) < 10:
            return {}
        
        recent_metrics = self.metrics[-100:]  # Last 100 operations
        recent_durations = [m.duration for m in recent_metrics]
        
        return {
            'recent_avg_duration': sum(recent_durations) / len(recent_durations),
            'performance_trend': 'improving' if recent_durations[-10:] < recent_durations[:10] else 'declining'
        }


# Decorator for easy optimization
def optimize(operation_name: str = None, use_cache: bool = True, use_async: bool = False):
    """Decorator to optimize function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create global optimizer
            if not hasattr(wrapper, '_optimizer'):
                wrapper._optimizer = AdvancedPerformanceOptimizer()
            
            op_name = operation_name or func.__name__
            return wrapper._optimizer.optimize_operation(
                op_name, func, *args, use_cache=use_cache, 
                use_async=use_async, **kwargs
            )
        
        return wrapper
    return decorator


# Global optimizer instance
_global_optimizer = None

def get_global_optimizer() -> AdvancedPerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = AdvancedPerformanceOptimizer()
    return _global_optimizer
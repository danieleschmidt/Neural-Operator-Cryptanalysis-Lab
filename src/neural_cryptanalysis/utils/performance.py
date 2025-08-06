"""Performance optimization utilities for neural cryptanalysis."""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
import pickle
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps
import warnings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch
    import torch.multiprocessing as mp
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    mp = None

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation: str
    duration: float
    memory_used: float
    cpu_percent: float
    gpu_memory: Optional[float] = None
    throughput: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PerformanceProfiler:
    """Performance profiler for neural cryptanalysis operations."""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring and HAS_TORCH
        self.metrics = []
        self.start_time = None
        self.start_memory = None
        self.operation_name = None
        
    def __enter__(self):
        self.start_operation()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_operation()
    
    def start_operation(self, operation_name: str = "operation"):
        """Start profiling an operation."""
        self.operation_name = operation_name
        self.start_time = time.time()
        
        # Memory monitoring
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.debug(f"Started profiling: {operation_name}")
    
    def end_operation(self) -> PerformanceMetrics:
        """End profiling and record metrics."""
        if self.start_time is None:
            logger.warning("No operation started")
            return None
        
        duration = time.time() - self.start_time
        
        # Memory and CPU metrics
        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = end_memory - self.start_memory
        cpu_percent = process.cpu_percent()
        
        # GPU memory if available
        gpu_memory = None
        if self.enable_gpu_monitoring and HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            except Exception:
                pass
        
        # Create metrics record
        metrics = PerformanceMetrics(
            operation=self.operation_name,
            duration=duration,
            memory_used=memory_used,
            cpu_percent=cpu_percent,
            gpu_memory=gpu_memory
        )
        
        self.metrics.append(metrics)
        
        logger.debug(f"Completed profiling: {self.operation_name} "
                    f"({duration:.3f}s, {memory_used:.1f}MB)")
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}
        
        total_duration = sum(m.duration for m in self.metrics)
        avg_memory = sum(m.memory_used for m in self.metrics) / len(self.metrics)
        avg_cpu = sum(m.cpu_percent for m in self.metrics) / len(self.metrics)
        
        gpu_metrics = [m for m in self.metrics if m.gpu_memory is not None]
        avg_gpu_memory = (sum(m.gpu_memory for m in gpu_metrics) / len(gpu_metrics)
                         if gpu_metrics else None)
        
        return {
            'total_operations': len(self.metrics),
            'total_duration': total_duration,
            'average_memory_mb': avg_memory,
            'average_cpu_percent': avg_cpu,
            'average_gpu_memory_mb': avg_gpu_memory,
            'operations': [m.operation for m in self.metrics]
        }


def profile_performance(operation_name: str = None):
    """Decorator to profile function performance.
    
    Args:
        operation_name: Name for the operation (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            
            with PerformanceProfiler() as profiler:
                profiler.start_operation(op_name)
                result = func(*args, **kwargs)
                metrics = profiler.end_operation()
            
            # Store metrics in function attribute for later access
            if not hasattr(func, '_performance_metrics'):
                func._performance_metrics = []
            func._performance_metrics.append(metrics)
            
            return result
        
        return wrapper
    
    return decorator


class CacheManager:
    """Intelligent caching system for neural cryptanalysis data."""
    
    def __init__(self, cache_dir: Path = None, max_size_mb: int = 1000,
                 ttl_hours: int = 24):
        self.cache_dir = cache_dir or Path("./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_hours * 3600
        self.cache_index = {}
        self.lock = threading.RLock()
        
        # Load existing cache index
        self._load_cache_index()
        
        logger.info(f"Cache initialized: {self.cache_dir}, max_size: {max_size_mb}MB")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        with self.lock:
            cache_path = self._get_cache_path(key)
            
            if not cache_path.exists():
                return default
            
            # Check TTL
            if key in self.cache_index:
                cache_info = self.cache_index[key]
                age = time.time() - cache_info['timestamp']
                
                if age > self.ttl_seconds:
                    self._remove_from_cache(key)
                    return default
            
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                logger.debug(f"Cache hit: {key}")
                return data
                
            except Exception as e:
                logger.warning(f"Cache read error for {key}: {e}")
                self._remove_from_cache(key)
                return default
    
    def set(self, key: str, value: Any) -> bool:
        """Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if cached successfully
        """
        with self.lock:
            try:
                cache_path = self._get_cache_path(key)
                
                # Serialize data
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Update index
                file_size = cache_path.stat().st_size
                self.cache_index[key] = {
                    'timestamp': time.time(),
                    'size_bytes': file_size,
                    'path': str(cache_path)
                }
                
                # Check cache size and evict if needed
                self._enforce_size_limit()
                
                logger.debug(f"Cache set: {key} ({file_size} bytes)")
                return True
                
            except Exception as e:
                logger.warning(f"Cache write error for {key}: {e}")
                return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and is valid
        """
        with self.lock:
            if key not in self.cache_index:
                return False
            
            # Check TTL
            cache_info = self.cache_index[key]
            age = time.time() - cache_info['timestamp']
            
            if age > self.ttl_seconds:
                self._remove_from_cache(key)
                return False
            
            return Path(cache_info['path']).exists()
    
    def invalidate(self, key: str):
        """Invalidate cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        with self.lock:
            self._remove_from_cache(key)
            logger.debug(f"Cache invalidated: {key}")
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            for key in list(self.cache_index.keys()):
                self._remove_from_cache(key)
            
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(info['size_bytes'] for info in self.cache_index.values())
            
            return {
                'entries': len(self.cache_index),
                'total_size_mb': total_size / 1024 / 1024,
                'max_size_mb': self.max_size_mb,
                'ttl_hours': self.ttl_seconds / 3600,
                'cache_dir': str(self.cache_dir)
            }
    
    def _get_cache_key_hash(self, key: str) -> str:
        """Get hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = self._get_cache_key_hash(key)
        return self.cache_dir / f"{key_hash}.cache"
    
    def _load_cache_index(self):
        """Load cache index from disk."""
        index_path = self.cache_dir / "index.json"
        
        if index_path.exists():
            try:
                import json
                with open(index_path, 'r') as f:
                    self.cache_index = json.load(f)
                logger.debug(f"Loaded cache index: {len(self.cache_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        index_path = self.cache_dir / "index.json"
        
        try:
            import json
            with open(index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _remove_from_cache(self, key: str):
        """Remove entry from cache."""
        if key in self.cache_index:
            cache_info = self.cache_index[key]
            cache_path = Path(cache_info['path'])
            
            if cache_path.exists():
                cache_path.unlink()
            
            del self.cache_index[key]
    
    def _enforce_size_limit(self):
        """Enforce cache size limit by evicting oldest entries."""
        total_size = sum(info['size_bytes'] for info in self.cache_index.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size <= max_size_bytes:
            return
        
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(
            self.cache_index.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # Remove oldest entries until under size limit
        for key, info in sorted_entries:
            if total_size <= max_size_bytes:
                break
            
            total_size -= info['size_bytes']
            self._remove_from_cache(key)
            
            logger.debug(f"Evicted cache entry: {key}")


def cached_function(cache_manager: CacheManager, key_func: Callable = None,
                   ttl_hours: int = None):
    """Decorator to cache function results.
    
    Args:
        cache_manager: Cache manager instance
        key_func: Function to generate cache key from arguments
        ttl_hours: Time to live in hours (overrides cache manager default)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                arg_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}:{hashlib.sha256(arg_str.encode()).hexdigest()[:16]}"
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result)
            
            return result
        
        return wrapper
    
    return decorator


class BatchProcessor:
    """Efficient batch processing for neural cryptanalysis operations."""
    
    def __init__(self, batch_size: int = 64, n_workers: int = None,
                 prefetch_factor: int = 2):
        self.batch_size = batch_size
        self.n_workers = n_workers or min(4, mp.cpu_count() if HAS_TORCH else 4)
        self.prefetch_factor = prefetch_factor
        
        logger.info(f"BatchProcessor initialized: batch_size={batch_size}, "
                   f"workers={self.n_workers}")
    
    def process_traces(self, traces: 'np.ndarray', processor_func: Callable,
                      **kwargs) -> List[Any]:
        """Process traces in batches with multiprocessing.
        
        Args:
            traces: Input trace array
            processor_func: Function to apply to each batch
            **kwargs: Additional arguments for processor function
            
        Returns:
            List of processing results
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for batch processing")
        
        n_traces = len(traces)
        n_batches = (n_traces + self.batch_size - 1) // self.batch_size
        
        results = []
        
        logger.info(f"Processing {n_traces} traces in {n_batches} batches")
        
        # Sequential processing for now (multiprocessing would require picklable functions)
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_traces)
            
            batch = traces[start_idx:end_idx]
            batch_result = processor_func(batch, **kwargs)
            results.append(batch_result)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed batch {i+1}/{n_batches}")
        
        return results
    
    def parallel_map(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """Apply function to iterable in parallel.
        
        Args:
            func: Function to apply
            iterable: Input items
            
        Returns:
            List of results
        """
        if self.n_workers == 1:
            return [func(item) for item in iterable]
        
        # For now, use sequential processing
        # In production, would use proper multiprocessing
        results = []
        for item in iterable:
            result = func(item)
            results.append(result)
        
        return results


class MemoryManager:
    """Memory management utilities for large-scale analysis."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.current_allocations = {}
        self.lock = threading.RLock()
        
        logger.info(f"Memory manager initialized: max_memory={max_memory_gb}GB")
    
    def allocate(self, name: str, size_bytes: int) -> bool:
        """Request memory allocation.
        
        Args:
            name: Allocation name
            size_bytes: Size to allocate
            
        Returns:
            True if allocation successful
        """
        with self.lock:
            current_usage = sum(self.current_allocations.values())
            
            if current_usage + size_bytes > self.max_memory_bytes:
                logger.warning(f"Memory allocation failed: {name} "
                             f"({size_bytes / 1024 / 1024:.1f}MB). "
                             f"Would exceed limit.")
                return False
            
            self.current_allocations[name] = size_bytes
            logger.debug(f"Memory allocated: {name} "
                        f"({size_bytes / 1024 / 1024:.1f}MB)")
            return True
    
    def deallocate(self, name: str):
        """Deallocate memory.
        
        Args:
            name: Allocation name
        """
        with self.lock:
            if name in self.current_allocations:
                size = self.current_allocations[name]
                del self.current_allocations[name]
                logger.debug(f"Memory deallocated: {name} "
                           f"({size / 1024 / 1024:.1f}MB)")
    
    def get_usage(self) -> Dict[str, float]:
        """Get current memory usage.
        
        Returns:
            Memory usage statistics
        """
        with self.lock:
            total_allocated = sum(self.current_allocations.values())
            
            return {
                'total_allocated_mb': total_allocated / 1024 / 1024,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'usage_percent': (total_allocated / self.max_memory_bytes) * 100,
                'allocations': {
                    name: size / 1024 / 1024
                    for name, size in self.current_allocations.items()
                }
            }
    
    def suggest_cleanup(self) -> List[str]:
        """Suggest memory cleanup actions.
        
        Returns:
            List of cleanup suggestions
        """
        usage = self.get_usage()
        suggestions = []
        
        if usage['usage_percent'] > 80:
            suggestions.append("Memory usage high (>80%), consider reducing batch sizes")
        
        if usage['usage_percent'] > 90:
            suggestions.append("Memory usage critical (>90%), cleanup recommended")
        
        # Suggest largest allocations for cleanup
        if self.current_allocations:
            largest = max(self.current_allocations.items(), key=lambda x: x[1])
            if largest[1] > self.max_memory_bytes * 0.5:  # >50% of total
                suggestions.append(f"Consider deallocating large allocation: {largest[0]}")
        
        return suggestions


def optimize_numpy_operations():
    """Optimize NumPy operations for performance."""
    if not HAS_NUMPY:
        return
    
    # Set optimal number of threads for BLAS operations
    try:
        import os
        n_threads = min(4, mp.cpu_count() if HAS_TORCH else 4)
        
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
        os.environ['MKL_NUM_THREADS'] = str(n_threads)
        
        logger.info(f"Optimized NumPy threading: {n_threads} threads")
        
    except Exception as e:
        logger.warning(f"Failed to optimize NumPy threading: {e}")


def optimize_torch_operations():
    """Optimize PyTorch operations for performance."""
    if not HAS_TORCH:
        return
    
    try:
        # Set optimal number of threads
        torch.set_num_threads(min(4, torch.get_num_threads()))
        
        # Enable optimizations
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        logger.info("Optimized PyTorch operations")
        
    except Exception as e:
        logger.warning(f"Failed to optimize PyTorch: {e}")


class PerformanceOptimizer:
    """Centralized performance optimization manager."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cache_manager = None
        self.memory_manager = None
        self.profiler = None
        
        # Initialize components based on configuration
        self._initialize_components()
        
        logger.info("Performance optimizer initialized")
    
    def _initialize_components(self):
        """Initialize performance components."""
        # Cache manager
        cache_config = self.config.get('cache', {})
        if cache_config.get('enabled', True):
            cache_dir = Path(cache_config.get('directory', './cache'))
            max_size = cache_config.get('max_size_mb', 1000)
            ttl_hours = cache_config.get('ttl_hours', 24)
            
            self.cache_manager = CacheManager(
                cache_dir=cache_dir,
                max_size_mb=max_size,
                ttl_hours=ttl_hours
            )
        
        # Memory manager
        memory_config = self.config.get('memory', {})
        max_memory = memory_config.get('max_gb', 8.0)
        self.memory_manager = MemoryManager(max_memory_gb=max_memory)
        
        # Performance profiler
        if self.config.get('profiling', {}).get('enabled', False):
            self.profiler = PerformanceProfiler()
        
        # Apply optimizations
        self._apply_optimizations()
    
    def _apply_optimizations(self):
        """Apply system-level optimizations."""
        optimize_numpy_operations()
        optimize_torch_operations()
        
        # Disable memory warnings if configured
        if self.config.get('suppress_warnings', False):
            warnings.filterwarnings('ignore', category=UserWarning, 
                                  message='.*memory.*')
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate performance optimization report.
        
        Returns:
            Optimization report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'has_numpy': HAS_NUMPY,
                'has_torch': HAS_TORCH,
                'has_cuda': HAS_TORCH and torch.cuda.is_available() if HAS_TORCH else False
            }
        }
        
        # Cache statistics
        if self.cache_manager:
            report['cache'] = self.cache_manager.get_stats()
        
        # Memory statistics
        if self.memory_manager:
            report['memory'] = self.memory_manager.get_usage()
        
        # Performance metrics
        if self.profiler and self.profiler.metrics:
            report['performance'] = self.profiler.get_summary()
        
        return report
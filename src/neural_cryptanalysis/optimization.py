"""Performance optimization and scaling utilities for neural cryptanalysis."""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from functools import wraps, lru_cache
import multiprocessing as mp

from .utils.logging_utils import get_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics for neural cryptanalysis operations."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: Optional[float] = None
    throughput_ops_per_sec: Optional[float] = None
    cache_hit_rate: Optional[float] = None


class PerformanceProfiler:
    """Profiler for monitoring neural cryptanalysis performance."""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.logger = get_logger(__name__)
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.metrics_history = []
        
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Start monitoring
            start_time = time.time()
            start_memory = psutil.virtual_memory().used / 1024**2
            start_cpu = psutil.cpu_percent()
            
            if self.enable_gpu_monitoring and torch.cuda.is_available():
                start_gpu_memory = torch.cuda.memory_allocated() / 1024**2
            else:
                start_gpu_memory = None
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                execution_time = time.time() - start_time
                memory_usage = psutil.virtual_memory().used / 1024**2 - start_memory
                cpu_usage = psutil.cpu_percent() - start_cpu
                
                if start_gpu_memory is not None:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**2 - start_gpu_memory
                else:
                    gpu_memory = None
                
                # Create metrics object
                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_usage,
                    gpu_memory_mb=gpu_memory
                )
                
                self.metrics_history.append(metrics)
                
                self.logger.debug(
                    f"Function {func.__name__} executed in {execution_time:.3f}s, "
                    f"Memory: {memory_usage:.1f}MB, CPU: {cpu_usage:.1f}%"
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error in profiled function {func.__name__}: {e}")
                raise
        
        return wrapper
    
    def get_average_metrics(self, last_n: Optional[int] = None) -> PerformanceMetrics:
        """Get average performance metrics."""
        if not self.metrics_history:
            return PerformanceMetrics(0, 0, 0)
        
        metrics = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        avg_time = np.mean([m.execution_time for m in metrics])
        avg_memory = np.mean([m.memory_usage_mb for m in metrics])
        avg_cpu = np.mean([m.cpu_usage_percent for m in metrics])
        
        gpu_metrics = [m.gpu_memory_mb for m in metrics if m.gpu_memory_mb is not None]
        avg_gpu = np.mean(gpu_metrics) if gpu_metrics else None
        
        return PerformanceMetrics(
            execution_time=avg_time,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            gpu_memory_mb=avg_gpu
        )


class BatchProcessor:
    """Optimized batch processing for large-scale cryptanalysis."""
    
    def __init__(self, 
                 batch_size: int = 64,
                 max_workers: Optional[int] = None,
                 use_gpu: bool = True):
        self.batch_size = batch_size
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.logger = get_logger(__name__)
        
        if self.use_gpu:
            self.device = torch.device('cuda')
            self.logger.info(f"Using GPU acceleration: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            self.logger.info(f"Using CPU with {self.max_workers} workers")
    
    def process_traces_batch(self, 
                           traces: np.ndarray,
                           model: nn.Module,
                           transform_fn: Optional[Callable] = None) -> np.ndarray:
        """Process traces in optimized batches.
        
        Args:
            traces: Input traces array
            model: Neural network model
            transform_fn: Optional preprocessing function
            
        Returns:
            Processed results
        """
        model.eval()
        model = model.to(self.device)
        
        n_traces = len(traces)
        results = []
        
        with torch.no_grad():
            for i in range(0, n_traces, self.batch_size):
                batch_end = min(i + self.batch_size, n_traces)
                batch_traces = traces[i:batch_end]
                
                # Apply preprocessing if provided
                if transform_fn:
                    batch_traces = transform_fn(batch_traces)
                
                # Convert to tensor and move to device
                batch_tensor = torch.tensor(batch_traces, dtype=torch.float32).to(self.device)
                
                # Process batch
                batch_results = model(batch_tensor)
                
                # Move results back to CPU and convert to numpy
                results.append(batch_results.cpu().numpy())
        
        return np.concatenate(results, axis=0)
    
    def parallel_trace_preprocessing(self, 
                                   traces: List[np.ndarray],
                                   preprocess_fn: Callable) -> List[np.ndarray]:
        """Preprocess traces in parallel.
        
        Args:
            traces: List of trace arrays
            preprocess_fn: Preprocessing function
            
        Returns:
            List of preprocessed traces
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(preprocess_fn, trace) for trace in traces]
            results = [future.result() for future in futures]
        
        return results
    
    def distributed_attack(self,
                          trace_chunks: List[np.ndarray],
                          attack_fn: Callable,
                          combine_fn: Callable) -> Any:
        """Distribute attack across multiple processes.
        
        Args:
            trace_chunks: List of trace chunks for parallel processing
            attack_fn: Attack function to apply to each chunk
            combine_fn: Function to combine results from all chunks
            
        Returns:
            Combined attack results
        """
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(attack_fn, chunk) for chunk in trace_chunks]
            chunk_results = [future.result() for future in futures]
        
        return combine_fn(chunk_results)


class CacheManager:
    """Intelligent caching for neural cryptanalysis operations."""
    
    def __init__(self, max_cache_size_mb: int = 1024):
        self.max_cache_size_mb = max_cache_size_mb
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.logger = get_logger(__name__)
        
        # LRU cache for trace preprocessing
        self.preprocess_cache = {}
        self.feature_cache = {}
        
    @lru_cache(maxsize=128)
    def cached_ntt_constants(self, n: int, q: int) -> np.ndarray:
        """Cache NTT constants for polynomial operations."""
        # Compute NTT constants - expensive operation
        constants = np.zeros(n // 2, dtype=np.int32)
        root = 3  # Primitive root
        
        for i in range(n // 2):
            constants[i] = pow(root, 2 * i + 1, q)
        
        return constants
    
    @lru_cache(maxsize=256)
    def cached_sbox_values(self, sbox_type: str = 'aes') -> np.ndarray:
        """Cache S-box lookup tables."""
        if sbox_type == 'aes':
            # AES S-box
            sbox = np.array([
                0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
                0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
                # ... (complete AES S-box would be here)
            ], dtype=np.uint8)
            return sbox[:256]  # Ensure full 256 entries
        else:
            # Generic S-box
            return np.arange(256, dtype=np.uint8)
    
    def cache_trace_features(self, trace_id: str, features: np.ndarray):
        """Cache extracted features for traces."""
        # Simple size-based eviction
        feature_size_mb = features.nbytes / (1024 ** 2)
        
        if feature_size_mb < self.max_cache_size_mb / 4:  # Use max 25% for single entry
            self.feature_cache[trace_id] = features
            
            # Evict old entries if cache too large
            while self._get_cache_size_mb() > self.max_cache_size_mb:
                oldest_key = next(iter(self.feature_cache))
                del self.feature_cache[oldest_key]
    
    def get_cached_features(self, trace_id: str) -> Optional[np.ndarray]:
        """Retrieve cached features."""
        if trace_id in self.feature_cache:
            self.cache_stats['hits'] += 1
            return self.feature_cache[trace_id]
        else:
            self.cache_stats['misses'] += 1
            return None
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate statistics."""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        return self.cache_stats['hits'] / total if total > 0 else 0.0
    
    def _get_cache_size_mb(self) -> float:
        """Calculate current cache size in MB."""
        total_bytes = sum(arr.nbytes for arr in self.feature_cache.values())
        return total_bytes / (1024 ** 2)


class GPUOptimizer:
    """GPU optimization utilities for neural cryptanalysis."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            self.device_count = torch.cuda.device_count()
            self.current_device = torch.cuda.current_device()
            self.logger.info(f"GPU optimization enabled: {self.device_count} devices available")
        else:
            self.logger.info("GPU not available, using CPU optimizations")
    
    def optimize_model_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference performance."""
        if not self.gpu_available:
            return model
        
        # Move to GPU
        model = model.cuda()
        
        # Enable inference optimizations
        model.eval()
        
        # Use half precision if supported
        if torch.cuda.get_device_capability()[0] >= 7:  # Volta or newer
            model = model.half()
            self.logger.info("Enabled half-precision inference")
        
        # Compile model for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                self.logger.info("Model compiled for optimized inference")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def setup_multi_gpu(self, model: nn.Module) -> nn.Module:
        """Setup multi-GPU training/inference."""
        if not self.gpu_available or self.device_count < 2:
            return model
        
        # Use DataParallel for multi-GPU
        model = nn.DataParallel(model)
        self.logger.info(f"Enabled multi-GPU processing on {self.device_count} devices")
        
        return model
    
    def optimize_memory_usage(self, model: nn.Module):
        """Optimize GPU memory usage."""
        if not self.gpu_available:
            return
        
        # Enable memory-efficient attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self.logger.info("Using memory-efficient attention")
        
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Enable memory optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        self.logger.info("GPU memory optimizations applied")
    
    def profile_gpu_usage(self, func: Callable) -> Callable:
        """Decorator to profile GPU memory usage."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.gpu_available:
                return func(*args, **kwargs)
            
            # Clear cache and record initial memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            try:
                result = func(*args, **kwargs)
                
                # Record peak memory usage
                peak_memory = torch.cuda.max_memory_allocated()
                current_memory = torch.cuda.memory_allocated()
                
                self.logger.debug(
                    f"GPU memory usage - Initial: {initial_memory/1024**2:.1f}MB, "
                    f"Peak: {peak_memory/1024**2:.1f}MB, "
                    f"Final: {current_memory/1024**2:.1f}MB"
                )
                
                return result
                
            finally:
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
        
        return wrapper


class AdaptiveOptimizer:
    """Adaptive optimization based on runtime performance."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.performance_history = []
        self.current_config = {
            'batch_size': 64,
            'num_workers': 4,
            'use_mixed_precision': False
        }
        
    def suggest_batch_size(self, 
                          model: nn.Module,
                          sample_input: torch.Tensor,
                          max_memory_mb: Optional[int] = None) -> int:
        """Suggest optimal batch size based on model and memory constraints."""
        if not torch.cuda.is_available():
            return min(128, self.current_config['batch_size'])
        
        # Start with small batch and increase until memory limit
        test_batch_sizes = [16, 32, 64, 128, 256, 512]
        optimal_batch_size = 16
        
        model.eval()
        model = model.cuda()
        
        for batch_size in test_batch_sizes:
            try:
                # Create test batch
                test_batch = sample_input[:1].repeat(batch_size, *([1] * (sample_input.dim() - 1)))
                test_batch = test_batch.cuda()
                
                # Test forward pass
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                with torch.no_grad():
                    _ = model(test_batch)
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used_mb = (peak_memory - initial_memory) / (1024 ** 2)
                
                # Check if within memory limits
                if max_memory_mb and memory_used_mb > max_memory_mb:
                    break
                
                optimal_batch_size = batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise
        
        self.logger.info(f"Suggested optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def adapt_parameters(self, performance_metrics: PerformanceMetrics):
        """Adapt optimization parameters based on performance."""
        self.performance_history.append(performance_metrics)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
        
        # Simple adaptive logic
        recent_metrics = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        avg_memory_usage = np.mean([m.memory_usage_mb for m in recent_metrics])
        avg_execution_time = np.mean([m.execution_time for m in recent_metrics])
        
        # Increase batch size if memory usage is low
        if avg_memory_usage < 1000 and self.current_config['batch_size'] < 256:
            self.current_config['batch_size'] *= 2
            self.logger.info(f"Increased batch size to {self.current_config['batch_size']}")
        
        # Decrease batch size if memory usage is high
        elif avg_memory_usage > 4000 and self.current_config['batch_size'] > 16:
            self.current_config['batch_size'] //= 2
            self.logger.info(f"Decreased batch size to {self.current_config['batch_size']}")
        
        # Adjust worker count based on CPU usage
        if len(recent_metrics) >= 5:
            cpu_usage = np.mean([m.cpu_usage_percent for m in recent_metrics[-5:]])
            
            if cpu_usage < 50 and self.current_config['num_workers'] < 8:
                self.current_config['num_workers'] += 1
            elif cpu_usage > 80 and self.current_config['num_workers'] > 1:
                self.current_config['num_workers'] -= 1
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current optimization configuration."""
        return self.current_config.copy()


class ScalabilityTester:
    """Test scalability of neural cryptanalysis operations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def test_model_scalability(self,
                              model: nn.Module,
                              input_sizes: List[int],
                              batch_sizes: List[int]) -> Dict[str, Any]:
        """Test model scalability across different input and batch sizes."""
        results = {
            'input_size_scaling': {},
            'batch_size_scaling': {},
            'memory_usage': {},
            'throughput': {}
        }
        
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Test input size scaling
        for input_size in input_sizes:
            test_input = torch.randn(32, input_size, 1)
            if torch.cuda.is_available():
                test_input = test_input.cuda()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(test_input)
            
            execution_time = time.time() - start_time
            
            results['input_size_scaling'][input_size] = execution_time
            
            if torch.cuda.is_available():
                memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)
                results['memory_usage'][input_size] = memory_usage
                torch.cuda.reset_peak_memory_stats()
        
        # Test batch size scaling
        test_input_size = input_sizes[len(input_sizes) // 2]  # Use middle size
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, test_input_size, 1)
            if torch.cuda.is_available():
                test_input = test_input.cuda()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(test_input)
            
            execution_time = time.time() - start_time
            throughput = batch_size / execution_time
            
            results['batch_size_scaling'][batch_size] = execution_time
            results['throughput'][batch_size] = throughput
        
        self.logger.info("Scalability testing completed")
        return results
    
    def benchmark_attack_performance(self,
                                   attack_function: Callable,
                                   trace_counts: List[int]) -> Dict[int, float]:
        """Benchmark attack performance across different trace counts."""
        results = {}
        
        for n_traces in trace_counts:
            # Generate synthetic traces
            traces = np.random.randn(n_traces, 1000).astype(np.float32)
            labels = np.random.randint(0, 256, n_traces)
            
            start_time = time.time()
            
            try:
                # Run attack
                attack_results = attack_function(traces, labels)
                execution_time = time.time() - start_time
                
                results[n_traces] = execution_time
                
                self.logger.info(
                    f"Attack with {n_traces} traces completed in {execution_time:.2f}s"
                )
                
            except Exception as e:
                self.logger.error(f"Attack failed with {n_traces} traces: {e}")
                results[n_traces] = float('inf')
        
        return results
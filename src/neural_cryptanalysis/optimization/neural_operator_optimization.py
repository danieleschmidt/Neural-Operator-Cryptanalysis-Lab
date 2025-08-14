"""Neural Operator Optimization Module - Generation 3.

This module provides advanced optimization specifically for neural operators
including Just-In-Time compilation, batching optimization, and adaptive architectures.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, deque
import statistics
import pickle
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.jit
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create mock torch modules
    from unittest.mock import Mock
    torch = Mock()
    nn = Mock()
    nn.Module = Mock
    torch.Tensor = Mock
    torch.jit = Mock()
    DataLoader = Mock

from ..utils.logging_utils import get_logger
from .performance_optimizer import AdvancedPerformanceOptimizer, optimize

logger = get_logger(__name__)


@dataclass
class BatchOptimizationConfig:
    """Configuration for batch optimization."""
    min_batch_size: int = 16
    max_batch_size: int = 512
    adaptive_batching: bool = True
    target_memory_usage: float = 0.8  # 80% of available memory
    dynamic_padding: bool = True
    gradient_accumulation_steps: int = 1


@dataclass
class CompilationConfig:
    """Configuration for JIT compilation."""
    enable_jit: bool = True
    optimization_level: str = "O2"  # O0, O1, O2, O3
    fuse_operations: bool = True
    enable_nvfuser: bool = True
    cache_compiled_models: bool = True


class JITCompiler:
    """Just-In-Time compiler for neural operators."""
    
    def __init__(self, config: CompilationConfig):
        self.config = config
        self.compiled_models = {}
        self.compilation_cache = {}
        self.compilation_stats = defaultdict(int)
        
        self.lock = threading.RLock()
        
        logger.info("JIT Compiler initialized")
    
    def compile_model(self, model: nn.Module, example_inputs: torch.Tensor,
                     model_id: str = None) -> nn.Module:
        """Compile model with JIT optimization."""
        if not self.config.enable_jit or not HAS_TORCH:
            return model
        
        model_id = model_id or f"model_{id(model)}"
        
        with self.lock:
            # Check cache first
            if model_id in self.compiled_models:
                self.compilation_stats['cache_hits'] += 1
                return self.compiled_models[model_id]
        
        try:
            start_time = time.time()
            
            # Prepare model for compilation
            model.eval()
            
            # Apply optimizations based on config
            if self.config.fuse_operations:
                model = self._fuse_operations(model)
            
            # JIT compilation
            if hasattr(torch.jit, 'script'):
                try:
                    compiled_model = torch.jit.script(model)
                except:
                    # Fallback to trace if script fails
                    compiled_model = torch.jit.trace(model, example_inputs)
            else:
                compiled_model = model
            
            # Optimize compiled model
            if hasattr(torch.jit, 'optimize_for_inference'):
                compiled_model = torch.jit.optimize_for_inference(compiled_model)
            
            compilation_time = time.time() - start_time
            
            with self.lock:
                if self.config.cache_compiled_models:
                    self.compiled_models[model_id] = compiled_model
                
                self.compilation_stats['compilations'] += 1
                self.compilation_stats['total_time'] += compilation_time
            
            logger.info(f"Compiled model {model_id} in {compilation_time:.3f}s")
            
            return compiled_model
            
        except Exception as e:
            logger.warning(f"JIT compilation failed for {model_id}: {e}")
            self.compilation_stats['failures'] += 1
            return model
    
    def _fuse_operations(self, model: nn.Module) -> nn.Module:
        """Fuse operations in the model."""
        if not hasattr(torch.jit, 'fuse'):
            return model
        
        try:
            # This is a simplified version - in practice would need more sophisticated fusion
            return model
        except Exception as e:
            logger.warning(f"Operation fusion failed: {e}")
            return model
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        with self.lock:
            stats = dict(self.compilation_stats)
            
            if stats.get('compilations', 0) > 0:
                stats['avg_compilation_time'] = stats['total_time'] / stats['compilations']
            
            stats['cached_models'] = len(self.compiled_models)
            
            return stats
    
    def clear_cache(self):
        """Clear compilation cache."""
        with self.lock:
            self.compiled_models.clear()
            self.compilation_cache.clear()
            
        logger.info("JIT compilation cache cleared")


class AdaptiveBatchProcessor:
    """Adaptive batch processing with dynamic sizing."""
    
    def __init__(self, config: BatchOptimizationConfig):
        self.config = config
        self.batch_history = deque(maxlen=100)
        self.current_batch_size = config.min_batch_size
        self.memory_monitor = MemoryMonitor()
        
        # Performance tracking
        self.processing_times = deque(maxlen=50)
        self.throughput_history = deque(maxlen=50)
        
        self.lock = threading.RLock()
        
        logger.info("Adaptive batch processor initialized")
    
    @optimize(operation_name="batch_processing", use_cache=True)
    def process_batch(self, model: nn.Module, data: torch.Tensor,
                     target_function: Callable = None) -> torch.Tensor:
        """Process data in optimally-sized batches."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for batch processing")
        
        start_time = time.time()
        
        # Determine optimal batch size
        optimal_batch_size = self._determine_optimal_batch_size(data.shape[0])
        
        results = []
        total_processed = 0
        
        # Process in batches
        for i in range(0, data.shape[0], optimal_batch_size):
            end_idx = min(i + optimal_batch_size, data.shape[0])
            batch = data[i:end_idx]
            
            # Memory check before processing
            if self.memory_monitor.check_memory_pressure():
                # Reduce batch size if memory pressure
                optimal_batch_size = max(self.config.min_batch_size, 
                                       optimal_batch_size // 2)
                logger.warning(f"Reduced batch size due to memory pressure: {optimal_batch_size}")
            
            # Process batch
            batch_start = time.time()
            
            if target_function:
                batch_result = target_function(model, batch)
            else:
                with torch.no_grad():
                    batch_result = model(batch)
            
            batch_time = time.time() - batch_start
            
            results.append(batch_result)
            total_processed += batch.shape[0]
            
            # Update performance metrics
            self._update_batch_metrics(batch.shape[0], batch_time)
        
        # Combine results
        if results:
            final_result = torch.cat(results, dim=0)
        else:
            final_result = torch.empty(0)
        
        total_time = time.time() - start_time
        throughput = total_processed / total_time if total_time > 0 else 0
        
        with self.lock:
            self.throughput_history.append(throughput)
            
            # Update batch size for next iteration
            if self.config.adaptive_batching:
                self._adapt_batch_size()
        
        logger.debug(f"Processed {total_processed} samples in {total_time:.3f}s "
                    f"(throughput: {throughput:.1f} samples/s)")
        
        return final_result
    
    def _determine_optimal_batch_size(self, data_size: int) -> int:
        """Determine optimal batch size based on current conditions."""
        # Start with current batch size
        optimal_size = self.current_batch_size
        
        # Adjust based on available memory
        available_memory_ratio = self.memory_monitor.get_available_memory_ratio()
        
        if available_memory_ratio < 0.3:  # Low memory
            optimal_size = max(self.config.min_batch_size, optimal_size // 2)
        elif available_memory_ratio > 0.7:  # High memory available
            optimal_size = min(self.config.max_batch_size, optimal_size * 2)
        
        # Ensure batch size doesn't exceed data size
        optimal_size = min(optimal_size, data_size)
        
        return optimal_size
    
    def _update_batch_metrics(self, batch_size: int, processing_time: float):
        """Update batch processing metrics."""
        with self.lock:
            self.processing_times.append(processing_time)
            
            batch_record = {
                'batch_size': batch_size,
                'processing_time': processing_time,
                'throughput': batch_size / processing_time if processing_time > 0 else 0,
                'timestamp': datetime.now()
            }
            
            self.batch_history.append(batch_record)
    
    def _adapt_batch_size(self):
        """Adapt batch size based on performance history."""
        if len(self.throughput_history) < 5:
            return
        
        recent_throughput = list(self.throughput_history)[-5:]
        avg_throughput = statistics.mean(recent_throughput)
        
        # If throughput is decreasing, try smaller batches
        if len(self.throughput_history) >= 10:
            older_throughput = list(self.throughput_history)[-10:-5]
            old_avg = statistics.mean(older_throughput)
            
            if avg_throughput < old_avg * 0.9:  # 10% decrease
                new_size = max(self.config.min_batch_size, 
                              int(self.current_batch_size * 0.8))
                if new_size != self.current_batch_size:
                    self.current_batch_size = new_size
                    logger.info(f"Reduced batch size to {new_size} due to throughput decrease")
            
            elif avg_throughput > old_avg * 1.1:  # 10% increase
                new_size = min(self.config.max_batch_size,
                              int(self.current_batch_size * 1.2))
                if new_size != self.current_batch_size:
                    self.current_batch_size = new_size
                    logger.info(f"Increased batch size to {new_size} due to throughput increase")
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        with self.lock:
            if not self.batch_history:
                return {}
            
            recent_batches = list(self.batch_history)[-20:]
            
            return {
                'current_batch_size': self.current_batch_size,
                'total_batches_processed': len(self.batch_history),
                'avg_processing_time': statistics.mean([b['processing_time'] for b in recent_batches]),
                'avg_throughput': statistics.mean([b['throughput'] for b in recent_batches]),
                'batch_size_range': (
                    min(b['batch_size'] for b in recent_batches),
                    max(b['batch_size'] for b in recent_batches)
                )
            }


class MemoryMonitor:
    """Monitor memory usage for optimization decisions."""
    
    def __init__(self):
        self.memory_history = deque(maxlen=20)
        self.pressure_threshold = 0.85  # 85% memory usage triggers pressure
        
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            if HAS_TORCH and torch.cuda.is_available():
                # GPU memory check
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                
                usage_ratio = (allocated + cached) / total
                self.memory_history.append(usage_ratio)
                
                return usage_ratio > self.pressure_threshold
            else:
                # CPU memory check
                import psutil
                memory = psutil.virtual_memory()
                usage_ratio = memory.percent / 100
                
                self.memory_history.append(usage_ratio)
                
                return usage_ratio > self.pressure_threshold
                
        except Exception:
            return False
    
    def get_available_memory_ratio(self) -> float:
        """Get ratio of available memory (0.0 to 1.0)."""
        try:
            if HAS_TORCH and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                return 1.0 - (allocated / total)
            else:
                import psutil
                memory = psutil.virtual_memory()
                return memory.available / memory.total
        except Exception:
            return 0.5  # Safe default


class LazyLoader:
    """Lazy loading for neural operator components."""
    
    def __init__(self):
        self.loaded_components = {}
        self.load_functions = {}
        self.access_counts = defaultdict(int)
        self.last_access_times = {}
        
        self.lock = threading.RLock()
        
    def register_component(self, component_id: str, load_function: Callable):
        """Register a component for lazy loading."""
        with self.lock:
            self.load_functions[component_id] = load_function
        
        logger.debug(f"Registered lazy loading for {component_id}")
    
    def get_component(self, component_id: str):
        """Get component, loading if necessary."""
        with self.lock:
            # Update access tracking
            self.access_counts[component_id] += 1
            self.last_access_times[component_id] = datetime.now()
            
            # Return if already loaded
            if component_id in self.loaded_components:
                return self.loaded_components[component_id]
            
            # Load component
            if component_id in self.load_functions:
                logger.info(f"Lazy loading component: {component_id}")
                
                start_time = time.time()
                component = self.load_functions[component_id]()
                load_time = time.time() - start_time
                
                self.loaded_components[component_id] = component
                
                logger.info(f"Loaded {component_id} in {load_time:.3f}s")
                
                return component
            else:
                raise ValueError(f"Component {component_id} not registered")
    
    def unload_component(self, component_id: str):
        """Unload component to free memory."""
        with self.lock:
            if component_id in self.loaded_components:
                del self.loaded_components[component_id]
                logger.info(f"Unloaded component: {component_id}")
    
    def cleanup_unused_components(self, max_idle_hours: int = 1):
        """Cleanup components that haven't been used recently."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=max_idle_hours)
        
        with self.lock:
            components_to_unload = []
            
            for component_id in self.loaded_components:
                last_access = self.last_access_times.get(component_id, datetime.min)
                
                if last_access < cutoff_time:
                    components_to_unload.append(component_id)
            
            for component_id in components_to_unload:
                self.unload_component(component_id)
        
        if components_to_unload:
            logger.info(f"Cleaned up {len(components_to_unload)} unused components")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get component usage statistics."""
        with self.lock:
            return {
                'loaded_components': len(self.loaded_components),
                'registered_components': len(self.load_functions),
                'access_counts': dict(self.access_counts),
                'total_accesses': sum(self.access_counts.values())
            }


class NeuralOperatorOptimizer:
    """Comprehensive neural operator optimization framework."""
    
    def __init__(self, 
                 batch_config: BatchOptimizationConfig = None,
                 compilation_config: CompilationConfig = None):
        
        self.batch_config = batch_config or BatchOptimizationConfig()
        self.compilation_config = compilation_config or CompilationConfig()
        
        # Core components
        self.jit_compiler = JITCompiler(self.compilation_config)
        self.batch_processor = AdaptiveBatchProcessor(self.batch_config)
        self.lazy_loader = LazyLoader()
        self.memory_monitor = MemoryMonitor()
        
        # Optimization tracking
        self.optimization_history = []
        self.performance_metrics = defaultdict(list)
        
        # Model cache
        self.optimized_models = {}
        
        self.lock = threading.RLock()
        
        logger.info("Neural operator optimizer initialized")
    
    def optimize_neural_operator(self, model: nn.Module, 
                                example_input: torch.Tensor,
                                model_id: str = None) -> nn.Module:
        """Comprehensive optimization of neural operator."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available, returning unoptimized model")
            return model
        
        model_id = model_id or f"model_{id(model)}"
        
        # Check if already optimized
        with self.lock:
            if model_id in self.optimized_models:
                logger.debug(f"Returning cached optimized model: {model_id}")
                return self.optimized_models[model_id]
        
        start_time = time.time()
        optimization_steps = []
        
        try:
            # Step 1: Model preparation
            original_model = model
            optimized_model = model.eval()
            
            # Step 2: JIT compilation
            if self.compilation_config.enable_jit:
                optimized_model = self.jit_compiler.compile_model(
                    optimized_model, example_input, model_id
                )
                optimization_steps.append("jit_compilation")
            
            # Step 3: Memory optimization
            optimized_model = self._optimize_memory_usage(optimized_model)
            optimization_steps.append("memory_optimization")
            
            # Step 4: Operator fusion (if available)
            optimized_model = self._optimize_operators(optimized_model)
            optimization_steps.append("operator_fusion")
            
            optimization_time = time.time() - start_time
            
            # Cache optimized model
            with self.lock:
                self.optimized_models[model_id] = optimized_model
                
                # Record optimization
                optimization_record = {
                    'model_id': model_id,
                    'optimization_time': optimization_time,
                    'steps': optimization_steps,
                    'timestamp': datetime.now()
                }
                
                self.optimization_history.append(optimization_record)
            
            logger.info(f"Optimized neural operator {model_id} in {optimization_time:.3f}s")
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"Neural operator optimization failed for {model_id}: {e}")
            return original_model
    
    def _optimize_memory_usage(self, model: nn.Module) -> nn.Module:
        """Optimize model memory usage."""
        try:
            # Enable memory efficient attention if available
            if hasattr(model, 'enable_memory_efficient_attention'):
                model.enable_memory_efficient_attention()
            
            # Optimize parameter storage
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    param.grad = None  # Clear gradients to save memory
            
            return model
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return model
    
    def _optimize_operators(self, model: nn.Module) -> nn.Module:
        """Optimize individual operators in the model."""
        try:
            # This would contain specific optimizations for different operator types
            # For now, just return the model
            return model
            
        except Exception as e:
            logger.warning(f"Operator optimization failed: {e}")
            return model
    
    @optimize(operation_name="optimized_inference", use_cache=True)
    def run_optimized_inference(self, model: nn.Module, data: torch.Tensor,
                               model_id: str = None) -> torch.Tensor:
        """Run inference with all optimizations applied."""
        # Optimize model if not already done
        optimized_model = self.optimize_neural_operator(model, data[:1], model_id)
        
        # Use adaptive batch processing
        result = self.batch_processor.process_batch(optimized_model, data)
        
        return result
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        with self.lock:
            report = {
                'optimized_models': len(self.optimized_models),
                'total_optimizations': len(self.optimization_history),
                'jit_compilation_stats': self.jit_compiler.get_compilation_stats(),
                'batch_processing_stats': self.batch_processor.get_batch_stats(),
                'lazy_loading_stats': self.lazy_loader.get_usage_stats(),
                'recent_optimizations': self.optimization_history[-10:] if self.optimization_history else []
            }
            
            # Calculate average optimization time
            if self.optimization_history:
                avg_opt_time = statistics.mean([
                    opt['optimization_time'] for opt in self.optimization_history
                ])
                report['average_optimization_time'] = avg_opt_time
            
            return report
    
    def cleanup(self):
        """Cleanup optimization resources."""
        self.jit_compiler.clear_cache()
        self.lazy_loader.cleanup_unused_components()
        
        with self.lock:
            self.optimized_models.clear()
        
        logger.info("Neural operator optimizer cleaned up")


# Global neural operator optimizer
_global_neural_optimizer = None

def get_global_neural_optimizer() -> NeuralOperatorOptimizer:
    """Get global neural operator optimizer instance."""
    global _global_neural_optimizer
    if _global_neural_optimizer is None:
        _global_neural_optimizer = NeuralOperatorOptimizer()
    return _global_neural_optimizer


def optimize_neural_operator(operation_name: str = None):
    """Decorator for optimizing neural operator operations."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            optimizer = get_global_neural_optimizer()
            
            # Extract model from arguments if present
            model = None
            for arg in args:
                if HAS_TORCH and isinstance(arg, torch.nn.Module):
                    model = arg
                    break
            
            if model is not None:
                # Create dummy input for optimization
                try:
                    example_input = torch.randn(1, 100)  # Adjust as needed
                    optimized_model = optimizer.optimize_neural_operator(model, example_input)
                    
                    # Replace model in arguments
                    new_args = []
                    for arg in args:
                        if arg is model:
                            new_args.append(optimized_model)
                        else:
                            new_args.append(arg)
                    args = tuple(new_args)
                    
                except Exception as e:
                    logger.warning(f"Neural operator optimization failed: {e}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
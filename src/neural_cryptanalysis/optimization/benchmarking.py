"""Comprehensive Benchmarking Framework for Neural Cryptanalysis.

This module provides advanced benchmarking capabilities including performance testing,
scalability analysis, and automated regression detection.
"""

import time
import threading
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json
import pickle
import uuid
import warnings
from enum import Enum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

from ..utils.logging_utils import get_logger
from ..utils.errors import NeuralCryptanalysisError
from .performance_optimizer import PerformanceMetrics

logger = get_logger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks."""
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    STRESS = "stress"
    REGRESSION = "regression"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    name: str
    benchmark_type: BenchmarkType
    iterations: int = 10
    warmup_iterations: int = 3
    timeout_seconds: float = 300.0
    parallel_execution: bool = False
    max_workers: int = 4
    collect_memory_stats: bool = True
    collect_cpu_stats: bool = True
    collect_gpu_stats: bool = True
    save_artifacts: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from a benchmark execution."""
    benchmark_id: str
    config: BenchmarkConfig
    start_time: datetime
    end_time: datetime
    duration: float
    iterations_completed: int
    success_rate: float
    
    # Performance metrics
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    std_execution_time: float
    
    # Resource usage
    avg_cpu_percent: float
    peak_memory_mb: float
    avg_memory_mb: float
    gpu_memory_mb: Optional[float] = None
    
    # Throughput metrics
    operations_per_second: float = 0.0
    samples_per_second: float = 0.0
    
    # Additional metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Raw data
    execution_times: List[float] = field(default_factory=list)
    memory_snapshots: List[float] = field(default_factory=list)
    cpu_snapshots: List[float] = field(default_factory=list)
    
    status: str = "completed"  # completed, failed, timeout, cancelled


class SystemProfiler:
    """System resource profiler for benchmarks."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.profiling = False
        self.profile_thread = None
        
        # Collected data
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        self.timestamps = []
        
        self.lock = threading.RLock()
    
    def start_profiling(self):
        """Start system profiling."""
        if self.profiling:
            return
        
        self.profiling = True
        self.profile_thread = threading.Thread(target=self._profile_loop, daemon=True)
        self.profile_thread.start()
    
    def stop_profiling(self) -> Dict[str, List[float]]:
        """Stop profiling and return collected data."""
        self.profiling = False
        
        if self.profile_thread:
            self.profile_thread.join(timeout=1)
        
        with self.lock:
            return {
                'cpu_samples': self.cpu_samples.copy(),
                'memory_samples': self.memory_samples.copy(),
                'gpu_samples': self.gpu_samples.copy(),
                'timestamps': self.timestamps.copy()
            }
    
    def _profile_loop(self):
        """Main profiling loop."""
        while self.profiling:
            try:
                timestamp = time.time()
                
                # CPU usage
                cpu_percent = psutil.cpu_percent() if HAS_PSUTIL else 0.0
                
                # Memory usage
                if HAS_PSUTIL:
                    memory = psutil.virtual_memory()
                    memory_mb = memory.used / 1024 / 1024
                else:
                    memory_mb = 0.0
                
                # GPU usage
                gpu_memory = None
                if HAS_TORCH and torch.cuda.is_available():
                    try:
                        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    except:
                        pass
                
                with self.lock:
                    self.timestamps.append(timestamp)
                    self.cpu_samples.append(cpu_percent)
                    self.memory_samples.append(memory_mb)
                    self.gpu_samples.append(gpu_memory)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.warning(f"Profiling error: {e}")
                time.sleep(0.1)
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage during profiling."""
        with self.lock:
            return max(self.memory_samples) if self.memory_samples else 0.0
    
    def get_avg_cpu(self) -> float:
        """Get average CPU usage during profiling."""
        with self.lock:
            return statistics.mean(self.cpu_samples) if self.cpu_samples else 0.0


class BenchmarkExecutor:
    """Executes individual benchmarks with comprehensive monitoring."""
    
    def __init__(self):
        self.profiler = SystemProfiler()
        
    def execute_benchmark(self, benchmark_func: Callable, config: BenchmarkConfig,
                         *args, **kwargs) -> BenchmarkResult:
        """Execute a single benchmark with full monitoring."""
        benchmark_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        logger.info(f"Starting benchmark {config.name} ({benchmark_id})")
        
        # Initialize result
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            config=config,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            duration=0.0,
            iterations_completed=0,
            success_rate=0.0,
            avg_execution_time=0.0,
            min_execution_time=float('inf'),
            max_execution_time=0.0,
            std_execution_time=0.0,
            avg_cpu_percent=0.0,
            peak_memory_mb=0.0,
            avg_memory_mb=0.0
        )
        
        try:
            # Start system profiling
            if config.collect_cpu_stats or config.collect_memory_stats:
                self.profiler.start_profiling()
            
            # Warmup iterations
            logger.debug(f"Running {config.warmup_iterations} warmup iterations")
            for i in range(config.warmup_iterations):
                try:
                    benchmark_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Warmup iteration {i} failed: {e}")
            
            # Main benchmark iterations
            execution_times = []
            successful_iterations = 0
            
            for iteration in range(config.iterations):
                try:
                    iter_start = time.time()
                    
                    # Execute benchmark function
                    if asyncio.iscoroutinefunction(benchmark_func):
                        asyncio.run(benchmark_func(*args, **kwargs))
                    else:
                        benchmark_func(*args, **kwargs)
                    
                    iter_time = time.time() - iter_start
                    execution_times.append(iter_time)
                    successful_iterations += 1
                    
                    # Check timeout
                    if time.time() - start_time.timestamp() > config.timeout_seconds:
                        logger.warning(f"Benchmark timeout reached after {iteration + 1} iterations")
                        result.status = "timeout"
                        break
                    
                except Exception as e:
                    error_msg = f"Iteration {iteration} failed: {e}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)
            
            # Stop profiling and collect system stats
            profile_data = {}
            if config.collect_cpu_stats or config.collect_memory_stats:
                profile_data = self.profiler.stop_profiling()
            
            # Calculate metrics
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            result.end_time = end_time
            result.duration = total_duration
            result.iterations_completed = successful_iterations
            result.success_rate = successful_iterations / config.iterations if config.iterations > 0 else 0
            result.execution_times = execution_times
            
            if execution_times:
                result.avg_execution_time = statistics.mean(execution_times)
                result.min_execution_time = min(execution_times)
                result.max_execution_time = max(execution_times)
                result.std_execution_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                
                # Throughput calculations
                total_operations = successful_iterations
                result.operations_per_second = total_operations / total_duration if total_duration > 0 else 0
            
            # System resource stats
            if profile_data:
                if profile_data['cpu_samples']:
                    result.avg_cpu_percent = statistics.mean(profile_data['cpu_samples'])
                    result.cpu_snapshots = profile_data['cpu_samples']
                
                if profile_data['memory_samples']:
                    result.peak_memory_mb = max(profile_data['memory_samples'])
                    result.avg_memory_mb = statistics.mean(profile_data['memory_samples'])
                    result.memory_snapshots = profile_data['memory_samples']
                
                if profile_data['gpu_samples'] and any(s is not None for s in profile_data['gpu_samples']):
                    gpu_samples = [s for s in profile_data['gpu_samples'] if s is not None]
                    if gpu_samples:
                        result.gpu_memory_mb = max(gpu_samples)
            
            logger.info(f"Benchmark {config.name} completed: "
                       f"{successful_iterations}/{config.iterations} iterations, "
                       f"avg_time={result.avg_execution_time:.4f}s")
            
        except Exception as e:
            result.status = "failed"
            error_msg = f"Benchmark execution failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.end_time = datetime.now()
            result.duration = (result.end_time - start_time).total_seconds()
        
        return result


class BenchmarkSuite:
    """Collection of related benchmarks."""
    
    def __init__(self, suite_name: str, workspace_dir: Path = None):
        self.suite_name = suite_name
        self.workspace_dir = workspace_dir or Path(f"./benchmarks/{suite_name}")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Registered benchmarks
        self.benchmarks = {}
        self.benchmark_dependencies = {}
        
        # Execution state
        self.execution_history = []
        self.baseline_results = {}
        
        self.executor = BenchmarkExecutor()
        
        logger.info(f"Benchmark suite '{suite_name}' initialized")
    
    def register_benchmark(self, benchmark_name: str, benchmark_func: Callable,
                          config: BenchmarkConfig, dependencies: List[str] = None):
        """Register a benchmark in the suite."""
        self.benchmarks[benchmark_name] = {
            'function': benchmark_func,
            'config': config,
            'registered_at': datetime.now()
        }
        
        self.benchmark_dependencies[benchmark_name] = dependencies or []
        
        logger.info(f"Registered benchmark: {benchmark_name}")
    
    def run_benchmark(self, benchmark_name: str, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not registered")
        
        benchmark_info = self.benchmarks[benchmark_name]
        
        # Check dependencies
        self._check_dependencies(benchmark_name)
        
        # Execute benchmark
        result = self.executor.execute_benchmark(
            benchmark_info['function'],
            benchmark_info['config'],
            *args, **kwargs
        )
        
        # Save result
        self._save_result(benchmark_name, result)
        
        return result
    
    def run_suite(self, *args, **kwargs) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks in the suite."""
        logger.info(f"Running benchmark suite: {self.suite_name}")
        
        # Determine execution order based on dependencies
        execution_order = self._resolve_execution_order()
        
        results = {}
        suite_start_time = datetime.now()
        
        for benchmark_name in execution_order:
            try:
                logger.info(f"Running benchmark: {benchmark_name}")
                result = self.run_benchmark(benchmark_name, *args, **kwargs)
                results[benchmark_name] = result
                
                # Log progress
                if result.status == "completed":
                    logger.info(f"✓ {benchmark_name}: {result.avg_execution_time:.4f}s avg")
                else:
                    logger.warning(f"✗ {benchmark_name}: {result.status}")
                
            except Exception as e:
                logger.error(f"Failed to run benchmark {benchmark_name}: {e}")
                results[benchmark_name] = None
        
        suite_duration = (datetime.now() - suite_start_time).total_seconds()
        
        # Generate suite report
        suite_report = self._generate_suite_report(results, suite_duration)
        self._save_suite_report(suite_report)
        
        logger.info(f"Benchmark suite completed in {suite_duration:.2f}s")
        
        return results
    
    def _check_dependencies(self, benchmark_name: str):
        """Check if benchmark dependencies are satisfied."""
        dependencies = self.benchmark_dependencies.get(benchmark_name, [])
        
        for dep in dependencies:
            if dep not in self.benchmarks:
                raise ValueError(f"Dependency '{dep}' not found for benchmark '{benchmark_name}'")
    
    def _resolve_execution_order(self) -> List[str]:
        """Resolve benchmark execution order based on dependencies."""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(benchmark_name):
            if benchmark_name in visited:
                return
            
            visited.add(benchmark_name)
            
            # Visit dependencies first
            for dep in self.benchmark_dependencies.get(benchmark_name, []):
                visit(dep)
            
            order.append(benchmark_name)
        
        for benchmark_name in self.benchmarks:
            visit(benchmark_name)
        
        return order
    
    def _save_result(self, benchmark_name: str, result: BenchmarkResult):
        """Save benchmark result to disk."""
        result_file = self.workspace_dir / f"{benchmark_name}_{result.benchmark_id}.json"
        
        try:
            result_data = {
                'benchmark_name': benchmark_name,
                'suite_name': self.suite_name,
                'result': self._serialize_result(result)
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save benchmark result: {e}")
    
    def _serialize_result(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Serialize benchmark result for JSON storage."""
        return {
            'benchmark_id': result.benchmark_id,
            'config': {
                'name': result.config.name,
                'benchmark_type': result.config.benchmark_type.value,
                'iterations': result.config.iterations,
                'timeout_seconds': result.config.timeout_seconds,
                'tags': result.config.tags,
                'metadata': result.config.metadata
            },
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'duration': result.duration,
            'iterations_completed': result.iterations_completed,
            'success_rate': result.success_rate,
            'avg_execution_time': result.avg_execution_time,
            'min_execution_time': result.min_execution_time,
            'max_execution_time': result.max_execution_time,
            'std_execution_time': result.std_execution_time,
            'avg_cpu_percent': result.avg_cpu_percent,
            'peak_memory_mb': result.peak_memory_mb,
            'avg_memory_mb': result.avg_memory_mb,
            'gpu_memory_mb': result.gpu_memory_mb,
            'operations_per_second': result.operations_per_second,
            'samples_per_second': result.samples_per_second,
            'custom_metrics': result.custom_metrics,
            'errors': result.errors,
            'warnings': result.warnings,
            'status': result.status
        }
    
    def _generate_suite_report(self, results: Dict[str, BenchmarkResult], 
                              suite_duration: float) -> Dict[str, Any]:
        """Generate comprehensive suite report."""
        successful_results = {k: v for k, v in results.items() if v and v.status == "completed"}
        
        report = {
            'suite_name': self.suite_name,
            'execution_time': datetime.now().isoformat(),
            'suite_duration': suite_duration,
            'total_benchmarks': len(self.benchmarks),
            'successful_benchmarks': len(successful_results),
            'success_rate': len(successful_results) / len(self.benchmarks) if self.benchmarks else 0,
            'summary_stats': {},
            'benchmark_results': {}
        }
        
        if successful_results:
            # Aggregate statistics
            all_avg_times = [r.avg_execution_time for r in successful_results.values()]
            all_peak_memory = [r.peak_memory_mb for r in successful_results.values()]
            all_cpu_usage = [r.avg_cpu_percent for r in successful_results.values()]
            
            report['summary_stats'] = {
                'avg_execution_time': statistics.mean(all_avg_times),
                'total_execution_time': sum(all_avg_times),
                'peak_memory_mb': max(all_peak_memory),
                'avg_memory_mb': statistics.mean(all_peak_memory),
                'avg_cpu_percent': statistics.mean(all_cpu_usage)
            }
            
            # Individual benchmark summaries
            for name, result in successful_results.items():
                report['benchmark_results'][name] = {
                    'avg_execution_time': result.avg_execution_time,
                    'operations_per_second': result.operations_per_second,
                    'peak_memory_mb': result.peak_memory_mb,
                    'success_rate': result.success_rate,
                    'status': result.status
                }
        
        return report
    
    def _save_suite_report(self, report: Dict[str, Any]):
        """Save suite report to disk."""
        report_file = self.workspace_dir / f"suite_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save suite report: {e}")
    
    def compare_with_baseline(self, baseline_file: Path = None) -> Dict[str, Any]:
        """Compare current results with baseline."""
        if baseline_file and baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                
                # Compare with latest results
                # This would implement detailed comparison logic
                return {'status': 'comparison_completed'}
                
            except Exception as e:
                logger.error(f"Failed to load baseline: {e}")
                return {'status': 'comparison_failed', 'error': str(e)}
        else:
            return {'status': 'no_baseline', 'message': 'No baseline found for comparison'}


class ScalabilityTester:
    """Specialized tester for scalability benchmarks."""
    
    def __init__(self):
        self.test_parameters = {
            'batch_sizes': [1, 16, 64, 256, 1024],
            'input_sizes': [100, 1000, 10000, 100000],
            'worker_counts': [1, 2, 4, 8, 16],
            'memory_sizes': [1, 10, 100, 1000]  # MB
        }
    
    def test_batch_size_scaling(self, benchmark_func: Callable, 
                               max_batch_size: int = 1024) -> Dict[str, Any]:
        """Test how performance scales with batch size."""
        results = {}
        
        batch_sizes = [bs for bs in self.test_parameters['batch_sizes'] if bs <= max_batch_size]
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            config = BenchmarkConfig(
                name=f"batch_size_{batch_size}",
                benchmark_type=BenchmarkType.SCALABILITY,
                iterations=5
            )
            
            executor = BenchmarkExecutor()
            result = executor.execute_benchmark(benchmark_func, config, batch_size=batch_size)
            
            results[batch_size] = {
                'avg_execution_time': result.avg_execution_time,
                'operations_per_second': result.operations_per_second,
                'peak_memory_mb': result.peak_memory_mb,
                'throughput_per_item': result.operations_per_second / batch_size if batch_size > 0 else 0
            }
        
        # Analyze scaling efficiency
        scaling_analysis = self._analyze_scaling_efficiency(results)
        
        return {
            'results': results,
            'scaling_analysis': scaling_analysis
        }
    
    def test_input_size_scaling(self, benchmark_func: Callable) -> Dict[str, Any]:
        """Test how performance scales with input size."""
        results = {}
        
        for input_size in self.test_parameters['input_sizes']:
            logger.info(f"Testing input size: {input_size}")
            
            config = BenchmarkConfig(
                name=f"input_size_{input_size}",
                benchmark_type=BenchmarkType.SCALABILITY,
                iterations=3
            )
            
            executor = BenchmarkExecutor()
            result = executor.execute_benchmark(benchmark_func, config, input_size=input_size)
            
            results[input_size] = {
                'avg_execution_time': result.avg_execution_time,
                'operations_per_second': result.operations_per_second,
                'peak_memory_mb': result.peak_memory_mb,
                'time_per_input': result.avg_execution_time / input_size if input_size > 0 else 0
            }
        
        return {'results': results}
    
    def _analyze_scaling_efficiency(self, results: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze scaling efficiency from results."""
        batch_sizes = sorted(results.keys())
        
        if len(batch_sizes) < 2:
            return {'analysis': 'insufficient_data'}
        
        # Calculate scaling factors
        base_batch_size = batch_sizes[0]
        base_throughput = results[base_batch_size]['operations_per_second']
        
        scaling_factors = []
        efficiency_ratios = []
        
        for batch_size in batch_sizes[1:]:
            size_ratio = batch_size / base_batch_size
            throughput_ratio = results[batch_size]['operations_per_second'] / base_throughput
            efficiency = throughput_ratio / size_ratio
            
            scaling_factors.append(throughput_ratio)
            efficiency_ratios.append(efficiency)
        
        avg_efficiency = statistics.mean(efficiency_ratios) if efficiency_ratios else 0
        
        # Determine scaling quality
        if avg_efficiency > 0.9:
            scaling_quality = "excellent"
        elif avg_efficiency > 0.7:
            scaling_quality = "good"
        elif avg_efficiency > 0.5:
            scaling_quality = "moderate"
        else:
            scaling_quality = "poor"
        
        return {
            'average_efficiency': avg_efficiency,
            'scaling_quality': scaling_quality,
            'efficiency_ratios': efficiency_ratios,
            'analysis': f"Scaling efficiency: {avg_efficiency:.2f} ({scaling_quality})"
        }


class RegressionDetector:
    """Detects performance regressions in benchmark results."""
    
    def __init__(self, sensitivity: float = 0.1):
        self.sensitivity = sensitivity  # 10% change threshold
        self.baseline_results = {}
        
    def set_baseline(self, benchmark_name: str, result: BenchmarkResult):
        """Set baseline result for regression detection."""
        self.baseline_results[benchmark_name] = {
            'avg_execution_time': result.avg_execution_time,
            'operations_per_second': result.operations_per_second,
            'peak_memory_mb': result.peak_memory_mb,
            'success_rate': result.success_rate,
            'timestamp': result.start_time
        }
        
        logger.info(f"Set baseline for {benchmark_name}")
    
    def check_regression(self, benchmark_name: str, current_result: BenchmarkResult) -> Dict[str, Any]:
        """Check for performance regression."""
        if benchmark_name not in self.baseline_results:
            return {
                'status': 'no_baseline',
                'message': f'No baseline found for {benchmark_name}'
            }
        
        baseline = self.baseline_results[benchmark_name]
        
        regressions = []
        improvements = []
        
        # Check execution time (lower is better)
        time_change = (current_result.avg_execution_time - baseline['avg_execution_time']) / baseline['avg_execution_time']
        if time_change > self.sensitivity:
            regressions.append({
                'metric': 'avg_execution_time',
                'baseline': baseline['avg_execution_time'],
                'current': current_result.avg_execution_time,
                'change_percent': time_change * 100
            })
        elif time_change < -self.sensitivity:
            improvements.append({
                'metric': 'avg_execution_time',
                'baseline': baseline['avg_execution_time'],
                'current': current_result.avg_execution_time,
                'change_percent': time_change * 100
            })
        
        # Check throughput (higher is better)
        if baseline['operations_per_second'] > 0:
            throughput_change = (current_result.operations_per_second - baseline['operations_per_second']) / baseline['operations_per_second']
            if throughput_change < -self.sensitivity:
                regressions.append({
                    'metric': 'operations_per_second',
                    'baseline': baseline['operations_per_second'],
                    'current': current_result.operations_per_second,
                    'change_percent': throughput_change * 100
                })
            elif throughput_change > self.sensitivity:
                improvements.append({
                    'metric': 'operations_per_second',
                    'baseline': baseline['operations_per_second'],
                    'current': current_result.operations_per_second,
                    'change_percent': throughput_change * 100
                })
        
        # Check memory usage (lower is better)
        memory_change = (current_result.peak_memory_mb - baseline['peak_memory_mb']) / baseline['peak_memory_mb']
        if memory_change > self.sensitivity:
            regressions.append({
                'metric': 'peak_memory_mb',
                'baseline': baseline['peak_memory_mb'],
                'current': current_result.peak_memory_mb,
                'change_percent': memory_change * 100
            })
        
        # Determine overall status
        if regressions:
            status = 'regression_detected'
            severity = 'critical' if any(r['change_percent'] > 50 for r in regressions) else 'warning'
        elif improvements:
            status = 'improvement_detected'
            severity = 'info'
        else:
            status = 'no_significant_change'
            severity = 'info'
        
        return {
            'status': status,
            'severity': severity,
            'regressions': regressions,
            'improvements': improvements,
            'baseline_timestamp': baseline['timestamp'].isoformat() if isinstance(baseline['timestamp'], datetime) else str(baseline['timestamp'])
        }


# Example benchmarks for neural cryptanalysis components
def create_neural_operator_benchmarks() -> BenchmarkSuite:
    """Create benchmark suite for neural operators."""
    suite = BenchmarkSuite("neural_operators")
    
    # Benchmark configs
    performance_config = BenchmarkConfig(
        name="neural_operator_performance",
        benchmark_type=BenchmarkType.PERFORMANCE,
        iterations=10,
        warmup_iterations=3
    )
    
    memory_config = BenchmarkConfig(
        name="neural_operator_memory",
        benchmark_type=BenchmarkType.MEMORY,
        iterations=5,
        collect_memory_stats=True
    )
    
    # Register benchmarks (placeholders - would be replaced with actual implementations)
    def dummy_neural_operator_benchmark(*args, **kwargs):
        """Dummy benchmark function."""
        if HAS_TORCH:
            # Simulate neural operator computation
            x = torch.randn(kwargs.get('batch_size', 64), 100)
            model = torch.nn.Linear(100, 10)
            with torch.no_grad():
                result = model(x)
            time.sleep(0.01)  # Simulate computation time
            return result
        else:
            time.sleep(0.01)
            return None
    
    suite.register_benchmark("performance_test", dummy_neural_operator_benchmark, performance_config)
    suite.register_benchmark("memory_test", dummy_neural_operator_benchmark, memory_config)
    
    return suite


# Global benchmark registry
_benchmark_suites = {}

def register_benchmark_suite(suite_name: str, suite: BenchmarkSuite):
    """Register a benchmark suite globally."""
    _benchmark_suites[suite_name] = suite
    logger.info(f"Registered benchmark suite: {suite_name}")

def get_benchmark_suite(suite_name: str) -> Optional[BenchmarkSuite]:
    """Get registered benchmark suite."""
    return _benchmark_suites.get(suite_name)

def list_benchmark_suites() -> List[str]:
    """List all registered benchmark suites."""
    return list(_benchmark_suites.keys())
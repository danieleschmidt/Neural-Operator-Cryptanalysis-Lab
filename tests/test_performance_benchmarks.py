"""Performance tests and benchmarks with regression detection."""

import pytest
import numpy as np
import torch
import time
import psutil
import gc
from pathlib import Path
import sys
import json
import tempfile
from typing import Dict, List, Tuple
from dataclasses import dataclass
from unittest.mock import patch

# Import test fixtures
from conftest import (
    neural_operator_architectures, performance_benchmarks,
    skip_if_no_gpu
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import components for performance testing
from neural_cryptanalysis.core import NeuralSCA, TraceData
from neural_cryptanalysis.neural_operators import FourierNeuralOperator, OperatorConfig
from neural_cryptanalysis.datasets.synthetic import SyntheticDatasetGenerator
from neural_cryptanalysis.utils.performance import PerformanceProfiler
from neural_cryptanalysis.adaptive_rl import AdaptiveAttackEngine
from neural_cryptanalysis.multi_modal_fusion import MultiModalSideChannelAnalyzer


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    test_name: str
    metric_name: str
    value: float
    unit: str
    threshold: float
    passed: bool
    timestamp: float
    system_info: Dict


@dataclass
class SystemInfo:
    """System information for benchmark context."""
    cpu_count: int
    memory_gb: float
    python_version: str
    torch_version: str
    device: str
    
    @classmethod
    def collect(cls):
        """Collect current system information."""
        return cls(
            cpu_count=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )


class PerformanceRegressionTracker:
    """Track performance regression across test runs."""
    
    def __init__(self, results_file: Path = None):
        self.results_file = results_file or Path("performance_regression_results.json")
        self.current_results = []
        self.historical_results = self._load_historical_results()
    
    def _load_historical_results(self) -> List[Dict]:
        """Load historical performance results."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def add_result(self, benchmark: PerformanceBenchmark):
        """Add a benchmark result."""
        self.current_results.append({
            'test_name': benchmark.test_name,
            'metric_name': benchmark.metric_name,
            'value': benchmark.value,
            'unit': benchmark.unit,
            'threshold': benchmark.threshold,
            'passed': benchmark.passed,
            'timestamp': benchmark.timestamp,
            'system_info': benchmark.system_info
        })
    
    def save_results(self):
        """Save current results to file."""
        all_results = self.historical_results + self.current_results
        with open(self.results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    def check_regression(self, test_name: str, metric_name: str, current_value: float, 
                        tolerance: float = 0.2) -> Tuple[bool, float]:
        """Check if current performance shows regression."""
        historical_values = [
            r['value'] for r in self.historical_results
            if r['test_name'] == test_name and r['metric_name'] == metric_name
        ]
        
        if not historical_values:
            return False, 0.0  # No regression if no historical data
        
        # Use median of last 5 runs as baseline
        recent_values = historical_values[-5:]
        baseline = np.median(recent_values)
        
        # For time-based metrics, regression is when current > baseline * (1 + tolerance)
        # For accuracy-based metrics, regression is when current < baseline * (1 - tolerance)
        if 'time' in metric_name.lower() or 'latency' in metric_name.lower():
            regression_threshold = baseline * (1 + tolerance)
            is_regression = current_value > regression_threshold
        else:
            regression_threshold = baseline * (1 - tolerance)
            is_regression = current_value < regression_threshold
        
        regression_percentage = abs(current_value - baseline) / baseline
        return is_regression, regression_percentage


@pytest.fixture
def performance_tracker():
    """Provide performance regression tracker."""
    temp_file = Path(tempfile.mktemp(suffix='.json'))
    tracker = PerformanceRegressionTracker(temp_file)
    yield tracker
    # Cleanup
    if temp_file.exists():
        temp_file.unlink()


@pytest.mark.performance
class TestTrainingPerformance:
    """Test neural operator training performance."""
    
    def test_fno_training_performance(self, performance_tracker, performance_benchmarks):
        """Test FourierNeuralOperator training performance."""
        print("\n=== Testing FNO Training Performance ===")
        
        system_info = SystemInfo.collect().__dict__
        
        # Generate test data
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(n_traces=500, trace_length=1000)
        
        traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
        
        # Configure neural SCA
        neural_sca = NeuralSCA(
            architecture='fourier_neural_operator',
            config={
                'fno': {'modes': 16, 'width': 64, 'n_layers': 4},
                'training': {'batch_size': 32, 'epochs': 5, 'learning_rate': 1e-3}
            }
        )
        
        # Measure training time
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        model = neural_sca.train(traces, labels, validation_split=0.2)
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        training_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Calculate performance metrics
        time_per_epoch = training_time / 5
        time_per_trace = training_time / len(traces)
        
        print(f"✓ Total training time: {training_time:.3f}s")
        print(f"✓ Time per epoch: {time_per_epoch:.3f}s")
        print(f"✓ Time per trace: {time_per_trace*1000:.3f}ms")
        print(f"✓ Memory used: {memory_used:.1f}MB")
        
        # Check against benchmarks
        time_threshold = performance_benchmarks['training_time_per_epoch_seconds']
        memory_threshold = performance_benchmarks['memory_usage_mb']
        
        # Create benchmark results
        benchmarks = [
            PerformanceBenchmark(
                test_name='fno_training',
                metric_name='time_per_epoch_seconds',
                value=time_per_epoch,
                unit='seconds',
                threshold=time_threshold,
                passed=time_per_epoch <= time_threshold,
                timestamp=time.time(),
                system_info=system_info
            ),
            PerformanceBenchmark(
                test_name='fno_training',
                metric_name='memory_usage_mb',
                value=memory_used,
                unit='MB',
                threshold=memory_threshold,
                passed=memory_used <= memory_threshold,
                timestamp=time.time(),
                system_info=system_info
            )
        ]
        
        # Add to tracker and check regression
        for benchmark in benchmarks:
            performance_tracker.add_result(benchmark)
            
            is_regression, regression_pct = performance_tracker.check_regression(
                benchmark.test_name, benchmark.metric_name, benchmark.value
            )
            
            if is_regression:
                print(f"⚠️  Performance regression detected for {benchmark.metric_name}: {regression_pct:.1%}")
            
            # Assert performance requirements
            assert benchmark.passed, f"{benchmark.metric_name} exceeded threshold: {benchmark.value} > {benchmark.threshold}"
        
        assert model is not None
    
    @neural_operator_architectures
    def test_architecture_training_performance(self, architecture, config_params, performance_tracker):
        """Test training performance across different architectures."""
        print(f"\n=== Testing {architecture} Training Performance ===")
        
        system_info = SystemInfo.collect().__dict__
        
        # Generate smaller dataset for comparative testing
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(n_traces=200, trace_length=500)
        
        traces = torch.tensor(dataset['power_traces'], dtype=torch.float32)
        if architecture != 'deep_operator_network':
            traces = traces.unsqueeze(-1)
        
        labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
        
        # Configure neural SCA
        config = {
            'training': {'batch_size': 16, 'epochs': 3},
            architecture.replace('_', ''): config_params
        }
        
        neural_sca = NeuralSCA(architecture=architecture, config=config)
        
        # Measure training performance
        start_time = time.perf_counter()
        
        try:
            model = neural_sca.train(traces, labels, validation_split=0.2)
            training_time = time.perf_counter() - start_time
            
            time_per_epoch = training_time / 3
            
            print(f"✓ {architecture} training time: {training_time:.3f}s")
            print(f"✓ Time per epoch: {time_per_epoch:.3f}s")
            
            # Create benchmark
            benchmark = PerformanceBenchmark(
                test_name=f'{architecture}_training',
                metric_name='time_per_epoch_seconds',
                value=time_per_epoch,
                unit='seconds',
                threshold=60.0,  # 1 minute per epoch threshold
                passed=time_per_epoch <= 60.0,
                timestamp=time.time(),
                system_info=system_info
            )
            
            performance_tracker.add_result(benchmark)
            assert benchmark.passed
            assert model is not None
            
        except Exception as e:
            pytest.fail(f"{architecture} training failed: {e}")
    
    def test_batch_size_scaling_performance(self, performance_tracker):
        """Test performance scaling with different batch sizes."""
        print("\n=== Testing Batch Size Scaling Performance ===")
        
        system_info = SystemInfo.collect().__dict__
        
        # Generate test data
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(n_traces=256, trace_length=500)
        
        traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
        
        batch_sizes = [8, 16, 32, 64]
        performance_results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size {batch_size}...")
            
            neural_sca = NeuralSCA(config={
                'training': {'batch_size': batch_size, 'epochs': 2},
                'fno': {'modes': 8, 'width': 32, 'n_layers': 2}
            })
            
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                model = neural_sca.train(traces, labels, validation_split=0.2)
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                training_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                performance_results[batch_size] = {
                    'training_time': training_time,
                    'memory_used': memory_used,
                    'time_per_sample': training_time / len(traces)
                }
                
                print(f"    ✓ Time: {training_time:.3f}s, Memory: {memory_used:.1f}MB")
                
                # Create benchmark
                benchmark = PerformanceBenchmark(
                    test_name='batch_scaling',
                    metric_name=f'batch_{batch_size}_time_seconds',
                    value=training_time,
                    unit='seconds',
                    threshold=120.0,  # 2 minutes threshold
                    passed=training_time <= 120.0,
                    timestamp=time.time(),
                    system_info=system_info
                )
                
                performance_tracker.add_result(benchmark)
                assert benchmark.passed
                
                # Clean up
                del model, neural_sca
                gc.collect()
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                continue
        
        # Analyze scaling behavior
        if len(performance_results) >= 2:
            batch_sizes_tested = sorted(performance_results.keys())
            times = [performance_results[bs]['training_time'] for bs in batch_sizes_tested]
            
            # Check that larger batch sizes don't drastically increase time
            # (they might actually decrease due to better parallelization)
            max_time = max(times)
            min_time = min(times)
            time_variation = max_time / min_time
            
            print(f"✓ Time variation across batch sizes: {time_variation:.2f}x")
            assert time_variation < 5.0  # Should not vary by more than 5x


@pytest.mark.performance
class TestInferencePerformance:
    """Test neural operator inference performance."""
    
    def test_inference_latency(self, performance_tracker, performance_benchmarks):
        """Test inference latency requirements."""
        print("\n=== Testing Inference Latency ===")
        
        system_info = SystemInfo.collect().__dict__
        
        # Quick training to get model
        neural_sca = NeuralSCA(config={
            'training': {'batch_size': 16, 'epochs': 1},
            'fno': {'modes': 8, 'width': 32, 'n_layers': 2}
        })
        
        # Generate training data
        traces = torch.randn(100, 500, 1)
        labels = torch.randint(0, 256, (100,))
        model = neural_sca.train(traces, labels, validation_split=0.2)
        
        # Generate test data for inference
        test_traces = torch.randn(100, 500, 1)
        
        # Warm up
        with torch.no_grad():
            _ = model(test_traces[:10])
        
        # Measure inference latency
        latencies = []
        
        with torch.no_grad():
            for _ in range(50):  # Multiple measurements for statistics
                start_time = time.perf_counter()
                predictions = model(test_traces[:32])  # Standard batch
                end_time = time.perf_counter()
                
                latency = end_time - start_time
                latencies.append(latency)
        
        # Calculate statistics
        mean_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        latency_per_trace_ms = (mean_latency / 32) * 1000  # Convert to milliseconds
        
        print(f"✓ Mean latency: {mean_latency*1000:.3f}ms")
        print(f"✓ Median latency: {median_latency*1000:.3f}ms")
        print(f"✓ P95 latency: {p95_latency*1000:.3f}ms")
        print(f"✓ Latency per trace: {latency_per_trace_ms:.3f}ms")
        
        # Check against benchmarks
        latency_threshold = performance_benchmarks['inference_time_per_trace_ms']
        
        benchmarks = [
            PerformanceBenchmark(
                test_name='inference_latency',
                metric_name='mean_latency_per_trace_ms',
                value=latency_per_trace_ms,
                unit='milliseconds',
                threshold=latency_threshold,
                passed=latency_per_trace_ms <= latency_threshold,
                timestamp=time.time(),
                system_info=system_info
            ),
            PerformanceBenchmark(
                test_name='inference_latency',
                metric_name='p95_latency_ms',
                value=p95_latency * 1000,
                unit='milliseconds',
                threshold=latency_threshold * 5,  # P95 can be higher
                passed=p95_latency * 1000 <= latency_threshold * 5,
                timestamp=time.time(),
                system_info=system_info
            )
        ]
        
        for benchmark in benchmarks:
            performance_tracker.add_result(benchmark)
            assert benchmark.passed, f"Latency requirement failed: {benchmark.value} > {benchmark.threshold}"
    
    def test_throughput_performance(self, performance_tracker):
        """Test inference throughput performance."""
        print("\n=== Testing Inference Throughput ===")
        
        system_info = SystemInfo.collect().__dict__
        
        # Quick training
        neural_sca = NeuralSCA(config={
            'training': {'batch_size': 16, 'epochs': 1},
            'fno': {'modes': 8, 'width': 32, 'n_layers': 2}
        })
        
        traces = torch.randn(100, 500, 1)
        labels = torch.randint(0, 256, (100,))
        model = neural_sca.train(traces, labels, validation_split=0.2)
        
        # Test different batch sizes for throughput
        batch_sizes = [1, 8, 16, 32, 64]
        throughput_results = {}
        
        for batch_size in batch_sizes:
            test_traces = torch.randn(batch_size, 500, 1)
            
            # Measure throughput
            num_iterations = max(10, 100 // batch_size)  # Adjust iterations based on batch size
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    predictions = model(test_traces)
            
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            total_traces = num_iterations * batch_size
            traces_per_second = total_traces / total_time
            
            throughput_results[batch_size] = traces_per_second
            
            print(f"  ✓ Batch size {batch_size}: {traces_per_second:.1f} traces/sec")
        
        # Find optimal batch size
        optimal_batch_size = max(throughput_results, key=throughput_results.get)
        max_throughput = throughput_results[optimal_batch_size]
        
        print(f"✓ Optimal batch size: {optimal_batch_size}")
        print(f"✓ Maximum throughput: {max_throughput:.1f} traces/sec")
        
        # Create benchmark
        benchmark = PerformanceBenchmark(
            test_name='inference_throughput',
            metric_name='max_throughput_traces_per_second',
            value=max_throughput,
            unit='traces/second',
            threshold=100.0,  # Minimum 100 traces/sec
            passed=max_throughput >= 100.0,
            timestamp=time.time(),
            system_info=system_info
        )
        
        performance_tracker.add_result(benchmark)
        assert benchmark.passed, f"Throughput requirement failed: {max_throughput} < 100 traces/sec"
    
    def test_memory_efficiency_inference(self, performance_tracker):
        """Test memory efficiency during inference."""
        print("\n=== Testing Inference Memory Efficiency ===")
        
        system_info = SystemInfo.collect().__dict__
        
        # Train model
        neural_sca = NeuralSCA(config={
            'training': {'batch_size': 16, 'epochs': 1},
            'fno': {'modes': 8, 'width': 32, 'n_layers': 2}
        })
        
        traces = torch.randn(100, 500, 1)
        labels = torch.randint(0, 256, (100,))
        model = neural_sca.train(traces, labels, validation_split=0.2)
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Test inference with increasing batch sizes
        batch_sizes = [1, 16, 64, 256]
        memory_usage = {}
        
        for batch_size in batch_sizes:
            test_traces = torch.randn(batch_size, 500, 1)
            
            # Measure memory during inference
            with torch.no_grad():
                predictions = model(test_traces)
            
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = current_memory - baseline_memory
            memory_per_trace = memory_used / batch_size
            
            memory_usage[batch_size] = {
                'total_memory_mb': memory_used,
                'memory_per_trace_mb': memory_per_trace
            }
            
            print(f"  ✓ Batch {batch_size}: {memory_used:.1f}MB total, {memory_per_trace:.3f}MB/trace")
            
            # Clean up
            del test_traces, predictions
            gc.collect()
        
        # Check memory scaling
        max_memory_per_trace = max(usage['memory_per_trace_mb'] for usage in memory_usage.values())
        
        benchmark = PerformanceBenchmark(
            test_name='inference_memory',
            metric_name='max_memory_per_trace_mb',
            value=max_memory_per_trace,
            unit='MB',
            threshold=10.0,  # 10MB per trace max
            passed=max_memory_per_trace <= 10.0,
            timestamp=time.time(),
            system_info=system_info
        )
        
        performance_tracker.add_result(benchmark)
        print(f"✓ Maximum memory per trace: {max_memory_per_trace:.3f}MB")
        
        assert benchmark.passed, f"Memory per trace too high: {max_memory_per_trace:.3f}MB > 10MB"


@pytest.mark.performance
class TestScalabilityPerformance:
    """Test system scalability performance."""
    
    def test_dataset_size_scaling(self, performance_tracker):
        """Test performance scaling with increasing dataset sizes."""
        print("\n=== Testing Dataset Size Scaling ===")
        
        system_info = SystemInfo.collect().__dict__
        
        dataset_sizes = [100, 250, 500, 1000]
        scaling_results = {}
        
        for size in dataset_sizes:
            print(f"  Testing dataset size {size}...")
            
            # Generate dataset
            generator = SyntheticDatasetGenerator(random_seed=42)
            dataset = generator.generate_aes_dataset(n_traces=size, trace_length=500)
            
            traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
            labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
            
            # Configure for reasonable training time
            neural_sca = NeuralSCA(config={
                'training': {'batch_size': min(32, size//8), 'epochs': 2},
                'fno': {'modes': 8, 'width': 32, 'n_layers': 2}
            })
            
            # Measure training time
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                model = neural_sca.train(traces, labels, validation_split=0.2)
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                training_time = end_time - start_time
                memory_used = end_memory - start_memory
                time_per_trace = training_time / size
                
                scaling_results[size] = {
                    'training_time': training_time,
                    'memory_used': memory_used,
                    'time_per_trace': time_per_trace
                }
                
                print(f"    ✓ Time: {training_time:.2f}s, Memory: {memory_used:.1f}MB")
                print(f"    ✓ Time per trace: {time_per_trace*1000:.3f}ms")
                
                # Clean up
                del model, neural_sca, traces, labels
                gc.collect()
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                continue
        
        # Analyze scaling behavior
        if len(scaling_results) >= 3:
            sizes = sorted(scaling_results.keys())
            times = [scaling_results[s]['training_time'] for s in sizes]
            
            # Check scaling efficiency (should be roughly linear)
            first_size, last_size = sizes[0], sizes[-1]
            first_time, last_time = times[0], times[-1]
            
            size_ratio = last_size / first_size
            time_ratio = last_time / first_time
            
            scaling_efficiency = size_ratio / time_ratio  # Should be close to 1 for linear scaling
            
            print(f"✓ Size scaling: {size_ratio:.1f}x data, {time_ratio:.1f}x time")
            print(f"✓ Scaling efficiency: {scaling_efficiency:.2f}")
            
            # Create benchmark
            benchmark = PerformanceBenchmark(
                test_name='dataset_scaling',
                metric_name='scaling_efficiency',
                value=scaling_efficiency,
                unit='ratio',
                threshold=0.3,  # Should be at least 0.3 (reasonable scaling)
                passed=scaling_efficiency >= 0.3,
                timestamp=time.time(),
                system_info=system_info
            )
            
            performance_tracker.add_result(benchmark)
            assert benchmark.passed, f"Poor scaling efficiency: {scaling_efficiency:.2f}"
    
    def test_concurrent_training_performance(self, performance_tracker):
        """Test performance under concurrent training workloads."""
        print("\n=== Testing Concurrent Training Performance ===")
        
        system_info = SystemInfo.collect().__dict__
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def train_model_worker(worker_id, start_event):
            """Worker function for concurrent training."""
            try:
                # Wait for start signal
                start_event.wait()
                
                # Generate data
                generator = SyntheticDatasetGenerator(random_seed=42 + worker_id)
                dataset = generator.generate_aes_dataset(n_traces=150, trace_length=300)
                
                traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
                labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
                
                # Create neural SCA
                neural_sca = NeuralSCA(config={
                    'training': {'batch_size': 16, 'epochs': 2},
                    'fno': {'modes': 8, 'width': 24, 'n_layers': 2}
                })
                
                # Train
                start_time = time.perf_counter()
                model = neural_sca.train(traces, labels, validation_split=0.2)
                end_time = time.perf_counter()
                
                training_time = end_time - start_time
                
                results_queue.put({
                    'worker_id': worker_id,
                    'training_time': training_time,
                    'success': True
                })
                
            except Exception as e:
                results_queue.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'success': False
                })
        
        # Test with different numbers of concurrent workers
        num_workers = min(3, psutil.cpu_count())  # Don't overwhelm the system
        
        start_event = threading.Event()
        threads = []
        
        # Start workers
        for i in range(num_workers):
            thread = threading.Thread(target=train_model_worker, args=(i, start_event))
            threads.append(thread)
            thread.start()
        
        # Record start time and signal all workers to start
        concurrent_start_time = time.perf_counter()
        start_event.set()
        
        # Wait for all workers to complete
        for thread in threads:
            thread.join(timeout=300)  # 5 minute timeout
        
        concurrent_end_time = time.perf_counter()
        total_concurrent_time = concurrent_end_time - concurrent_start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        successful_workers = [r for r in results if r.get('success', False)]
        failed_workers = [r for r in results if not r.get('success', False)]
        
        print(f"✓ Concurrent workers: {num_workers}")
        print(f"✓ Successful: {len(successful_workers)}")
        print(f"✓ Failed: {len(failed_workers)}")
        print(f"✓ Total concurrent time: {total_concurrent_time:.2f}s")
        
        if successful_workers:
            avg_worker_time = np.mean([r['training_time'] for r in successful_workers])
            print(f"✓ Average worker time: {avg_worker_time:.2f}s")
            
            # Calculate efficiency (ideal would be total_time ≈ max(worker_times))
            max_worker_time = max(r['training_time'] for r in successful_workers)
            efficiency = max_worker_time / total_concurrent_time
            
            print(f"✓ Concurrency efficiency: {efficiency:.2f}")
            
            # Create benchmark
            benchmark = PerformanceBenchmark(
                test_name='concurrent_training',
                metric_name='concurrency_efficiency',
                value=efficiency,
                unit='ratio',
                threshold=0.7,  # Should be at least 70% efficient
                passed=efficiency >= 0.7,
                timestamp=time.time(),
                system_info=system_info
            )
            
            performance_tracker.add_result(benchmark)
            
            # At least 2/3 of workers should succeed
            success_rate = len(successful_workers) / num_workers
            assert success_rate >= 0.67, f"Too many concurrent training failures: {success_rate:.2f}"


@pytest.mark.performance
class TestRealWorldPerformance:
    """Test performance under realistic workload conditions."""
    
    def test_adaptive_rl_performance(self, performance_tracker):
        """Test adaptive RL system performance."""
        print("\n=== Testing Adaptive RL Performance ===")
        
        system_info = SystemInfo.collect().__dict__
        
        # Create neural SCA
        neural_sca = NeuralSCA(config={
            'training': {'batch_size': 16, 'epochs': 2}
        })
        
        # Create adaptive engine
        adaptive_engine = AdaptiveAttackEngine(neural_sca, epsilon=0.1, device='cpu')
        
        # Generate test data
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(n_traces=200, trace_length=400)
        
        trace_data = TraceData(traces=dataset['power_traces'], labels=dataset['labels'][0])
        
        # Mock evaluation for performance testing
        eval_times = []
        
        async def mock_evaluation(state, traces):
            start_time = time.perf_counter()
            # Simulate realistic evaluation work
            await asyncio.sleep(0.01)  # 10ms simulated work
            end_time = time.perf_counter()
            eval_times.append(end_time - start_time)
            return 0.6, 0.7, 0.5
        
        # Measure adaptive RL performance
        start_time = time.perf_counter()
        
        with patch.object(adaptive_engine, 'evaluate_attack_performance', side_effect=mock_evaluation):
            import asyncio
            results = asyncio.run(adaptive_engine.autonomous_attack(
                traces=trace_data,
                target_success_rate=0.8,
                max_episodes=10,
                patience=5
            ))
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        avg_eval_time = np.mean(eval_times) if eval_times else 0
        episodes_completed = results.get('training_episodes', 0)
        time_per_episode = total_time / max(episodes_completed, 1)
        
        print(f"✓ Total RL time: {total_time:.3f}s")
        print(f"✓ Episodes completed: {episodes_completed}")
        print(f"✓ Time per episode: {time_per_episode:.3f}s")
        print(f"✓ Average evaluation time: {avg_eval_time*1000:.1f}ms")
        
        # Create benchmark
        benchmark = PerformanceBenchmark(
            test_name='adaptive_rl',
            metric_name='time_per_episode_seconds',
            value=time_per_episode,
            unit='seconds',
            threshold=30.0,  # 30 seconds per episode max
            passed=time_per_episode <= 30.0,
            timestamp=time.time(),
            system_info=system_info
        )
        
        performance_tracker.add_result(benchmark)
        assert benchmark.passed, f"RL episode time too high: {time_per_episode:.3f}s"
    
    def test_multimodal_fusion_performance(self, performance_tracker, multimodal_test_data):
        """Test multi-modal fusion performance."""
        print("\n=== Testing Multi-Modal Fusion Performance ===")
        
        system_info = SystemInfo.collect().__dict__
        
        from neural_cryptanalysis.multi_modal_fusion import MultiModalData
        
        # Create larger multi-modal dataset for performance testing
        mm_data = MultiModalData(
            power_traces=np.random.randn(500, 1000) * 0.1,
            em_near_traces=np.random.randn(500, 1000) * 0.08,
            acoustic_traces=np.random.randn(500, 1000) * 0.05
        )
        
        # Test fusion performance
        analyzer = MultiModalSideChannelAnalyzer(fusion_method='adaptive', device='cpu')
        
        start_time = time.perf_counter()
        results = analyzer.analyze_multi_modal(mm_data)
        end_time = time.perf_counter()
        
        fusion_time = end_time - start_time
        time_per_trace = fusion_time / 500
        
        print(f"✓ Fusion time: {fusion_time:.3f}s")
        print(f"✓ Time per trace: {time_per_trace*1000:.3f}ms")
        print(f"✓ Modalities fused: {len(results['modalities_used'])}")
        
        # Create benchmark
        benchmark = PerformanceBenchmark(
            test_name='multimodal_fusion',
            metric_name='time_per_trace_ms',
            value=time_per_trace * 1000,
            unit='milliseconds',
            threshold=50.0,  # 50ms per trace max
            passed=time_per_trace * 1000 <= 50.0,
            timestamp=time.time(),
            system_info=system_info
        )
        
        performance_tracker.add_result(benchmark)
        assert benchmark.passed, f"Fusion time per trace too high: {time_per_trace*1000:.3f}ms"
    
    def test_end_to_end_attack_performance(self, performance_tracker):
        """Test complete end-to-end attack performance."""
        print("\n=== Testing End-to-End Attack Performance ===")
        
        system_info = SystemInfo.collect().__dict__
        
        # Generate dataset
        start_time = time.perf_counter()
        
        generator = SyntheticDatasetGenerator(random_seed=42)
        dataset = generator.generate_aes_dataset(n_traces=300, trace_length=600)
        
        data_gen_time = time.perf_counter() - start_time
        
        # Prepare data
        traces = torch.tensor(dataset['power_traces'], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(dataset['labels'][0], dtype=torch.long)
        
        split_idx = int(0.7 * len(traces))
        train_traces = traces[:split_idx]
        train_labels = labels[:split_idx]
        test_traces = traces[split_idx:]
        
        # Train model
        neural_sca = NeuralSCA(config={
            'fno': {'modes': 8, 'width': 32, 'n_layers': 3},
            'training': {'batch_size': 32, 'epochs': 5}
        })
        
        training_start = time.perf_counter()
        model = neural_sca.train(train_traces, train_labels, validation_split=0.2)
        training_time = time.perf_counter() - training_start
        
        # Perform attack
        attack_start = time.perf_counter()
        attack_results = neural_sca.attack(
            target_traces=test_traces,
            model=model,
            strategy='template',
            target_byte=0,
            plaintexts=dataset['plaintexts'][split_idx:]
        )
        attack_time = time.perf_counter() - attack_start
        
        total_time = data_gen_time + training_time + attack_time
        
        print(f"✓ Data generation: {data_gen_time:.3f}s")
        print(f"✓ Model training: {training_time:.3f}s")
        print(f"✓ Attack execution: {attack_time:.3f}s")
        print(f"✓ Total end-to-end: {total_time:.3f}s")
        print(f"✓ Attack confidence: {attack_results['confidence']:.3f}")
        
        # Create benchmarks
        benchmarks = [
            PerformanceBenchmark(
                test_name='end_to_end_attack',
                metric_name='total_time_seconds',
                value=total_time,
                unit='seconds',
                threshold=300.0,  # 5 minutes total
                passed=total_time <= 300.0,
                timestamp=time.time(),
                system_info=system_info
            ),
            PerformanceBenchmark(
                test_name='end_to_end_attack',
                metric_name='attack_time_seconds',
                value=attack_time,
                unit='seconds',
                threshold=30.0,  # 30 seconds for attack
                passed=attack_time <= 30.0,
                timestamp=time.time(),
                system_info=system_info
            )
        ]
        
        for benchmark in benchmarks:
            performance_tracker.add_result(benchmark)
            assert benchmark.passed, f"{benchmark.metric_name} exceeded threshold"


if __name__ == "__main__":
    # Save performance results after test run
    tracker = PerformanceRegressionTracker()
    
    try:
        pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])
    finally:
        tracker.save_results()
        print(f"\n✓ Performance results saved to {tracker.results_file}")
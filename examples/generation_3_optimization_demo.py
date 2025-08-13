#!/usr/bin/env python3
"""Generation 3 Optimization Framework Demonstration.

This script demonstrates the advanced optimization capabilities of the neural
cryptanalysis framework including performance optimization, research acceleration,
self-healing mechanisms, and comprehensive benchmarking.
"""

import time
import asyncio
import numpy as np
from pathlib import Path

# Import optimization framework
from neural_cryptanalysis.optimization import (
    # Core optimization
    AdvancedPerformanceOptimizer,
    OptimizationConfig,
    optimize,
    
    # Research acceleration  
    ExperimentManager,
    HyperparameterOptimizer,
    ExperimentConfig,
    HyperparameterSpace,
    ResearchPipeline,
    
    # Self-healing
    create_complete_self_healing_system,
    
    # Neural operator optimization
    NeuralOperatorOptimizer,
    BatchOptimizationConfig,
    CompilationConfig,
    
    # Benchmarking
    BenchmarkSuite,
    BenchmarkConfig,
    BenchmarkType,
    create_neural_operator_benchmarks,
    ScalabilityTester
)

# Mock neural cryptanalysis components for demonstration
class MockNeuralOperator:
    """Mock neural operator for demonstration."""
    
    def __init__(self, complexity: int = 1):
        self.complexity = complexity
        
    def forward(self, x):
        """Simulate neural operator computation."""
        # Simulate computation based on complexity
        time.sleep(0.001 * self.complexity)
        
        # Return mock result
        return np.random.random((x.shape[0], 10))
    
    def __call__(self, x):
        return self.forward(x)


class MockTraceData:
    """Mock trace data for demonstration."""
    
    def __init__(self, n_traces: int = 1000, trace_length: int = 5000):
        self.traces = np.random.random((n_traces, trace_length))
        self.labels = np.random.randint(0, 256, n_traces)
        
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return {
            'trace': self.traces[idx],
            'label': self.labels[idx]
        }


def demonstrate_performance_optimization():
    """Demonstrate core performance optimization features."""
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Initialize optimizer with custom configuration
    config = OptimizationConfig(
        cache_max_size_mb=500,
        memory_pool_size_gb=2.0,
        auto_scaling_enabled=True,
        max_workers=4
    )
    
    optimizer = AdvancedPerformanceOptimizer(config)
    
    # Demonstrate function optimization with caching
    @optimize(operation_name="mock_analysis", use_cache=True)
    def mock_cryptanalysis_function(data_size: int, complexity: int = 1):
        """Mock cryptanalysis function."""
        data = np.random.random(data_size)
        operator = MockNeuralOperator(complexity)
        
        # Simulate processing
        result = operator(data.reshape(1, -1))
        return result
    
    print("Testing cached function optimization...")
    
    # First call (should be cached)
    start_time = time.time()
    result1 = mock_cryptanalysis_function(1000, complexity=2)
    time1 = time.time() - start_time
    print(f"First call: {time1:.4f}s")
    
    # Second call (should use cache)
    start_time = time.time()
    result2 = mock_cryptanalysis_function(1000, complexity=2)
    time2 = time.time() - start_time
    print(f"Second call (cached): {time2:.4f}s")
    print(f"Cache speedup: {time1/time2:.1f}x")
    
    # Demonstrate batch optimization
    print("\nTesting batch optimization...")
    data_batches = [np.random.random((50, 1000)) for _ in range(5)]
    
    def process_batch(batch):
        operator = MockNeuralOperator()
        return [operator(row.reshape(1, -1)) for row in batch]
    
    batch_results = optimizer.optimize_batch(
        "batch_processing", 
        process_batch, 
        data_batches
    )
    
    print(f"Processed {len(batch_results)} batches")
    
    # Get optimization report
    report = optimizer.get_optimization_report()
    print(f"\nOptimization Report:")
    print(f"- Total operations: {report['total_operations']}")
    print(f"- Cache hit rate: {report['efficiency']['cache_hit_rate']:.2%}")
    print(f"- Memory efficiency: {report['efficiency']['memory_efficiency']:.1f}%")
    
    return optimizer


def demonstrate_research_acceleration():
    """Demonstrate research acceleration capabilities."""
    print("\n" + "="*60)
    print("RESEARCH ACCELERATION DEMONSTRATION")
    print("="*60)
    
    # Create experiment manager
    workspace_dir = Path("./demo_experiments")
    workspace_dir.mkdir(exist_ok=True)
    
    experiment_manager = ExperimentManager(workspace_dir)
    
    # Define experiment function
    def neural_operator_experiment(architecture: str, learning_rate: float, 
                                 batch_size: int, hidden_dim: int):
        """Mock neural operator training experiment."""
        # Simulate training process
        training_time = np.random.uniform(1, 5)  # 1-5 seconds
        time.sleep(training_time / 100)  # Speed up for demo
        
        # Simulate performance metrics based on parameters
        base_accuracy = 0.7
        lr_factor = 1.0 if 0.001 <= learning_rate <= 0.01 else 0.9
        batch_factor = 1.0 if 32 <= batch_size <= 128 else 0.95
        hidden_factor = min(1.0, hidden_dim / 256)
        
        accuracy = base_accuracy * lr_factor * batch_factor * hidden_factor
        accuracy += np.random.normal(0, 0.05)  # Add noise
        accuracy = max(0.5, min(0.99, accuracy))  # Clamp to reasonable range
        
        return {
            'metrics': {
                'accuracy': accuracy,
                'training_time': training_time,
                'convergence_rate': np.random.uniform(0.8, 1.0)
            },
            'artifacts': {
                'model_size': hidden_dim * 1000,
                'final_loss': np.random.uniform(0.1, 0.5)
            }
        }
    
    # Demonstrate single experiment
    print("Running single experiment...")
    
    config = ExperimentConfig(
        name="baseline_experiment",
        description="Baseline neural operator configuration",
        parameters={
            'architecture': 'fourier_neural_operator',
            'learning_rate': 0.001,
            'batch_size': 64,
            'hidden_dim': 128
        },
        metrics=['accuracy', 'training_time']
    )
    
    exp_id = experiment_manager.create_experiment(config)
    result = experiment_manager.run_experiment(exp_id, neural_operator_experiment, **config.parameters)
    
    print(f"Experiment completed with accuracy: {result.metrics.get('accuracy', 0):.3f}")
    
    # Demonstrate hyperparameter optimization
    print("\nRunning hyperparameter optimization...")
    
    hyperopt = HyperparameterOptimizer(experiment_manager)
    
    # Define search space
    parameter_space = [
        HyperparameterSpace('learning_rate', 'continuous', bounds=(0.0001, 0.1), log_scale=True),
        HyperparameterSpace('batch_size', 'discrete', choices=[16, 32, 64, 128, 256]),
        HyperparameterSpace('hidden_dim', 'discrete', bounds=(64, 512)),
        HyperparameterSpace('architecture', 'categorical', choices=['fourier_neural_operator', 'deep_operator_network'])
    ]
    
    # Run optimization with limited trials for demo
    optimization_result = hyperopt.optimize(
        neural_operator_experiment,
        parameter_space,
        n_trials=8,
        strategy='random',  # Use random for speed
        metric_name='accuracy',
        maximize=True
    )
    
    print(f"Best parameters found:")
    for param, value in optimization_result['best_parameters'].items():
        print(f"  {param}: {value}")
    print(f"Best accuracy: {optimization_result['best_value']:.3f}")
    
    return experiment_manager


def demonstrate_self_healing():
    """Demonstrate self-healing system capabilities."""
    print("\n" + "="*60)
    print("SELF-HEALING SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Create complete self-healing system
    healing_system = create_complete_self_healing_system()
    
    print("Self-healing system started...")
    
    # Let the system run for a few seconds to collect baseline metrics
    time.sleep(3)
    
    # Get health report
    health_report = healing_system['health_monitor'].get_health_report()
    print(f"System health status: {health_report['overall_status']}")
    print(f"Active issues: {health_report['active_issues']}")
    
    if health_report['latest_metrics']:
        metrics = health_report['latest_metrics']
        print(f"Current metrics:")
        print(f"  CPU: {metrics.get('cpu_percent', 0):.1f}%")
        print(f"  Memory: {metrics.get('memory_percent', 0):.1f}%")
    
    # Get optimization report
    optimization_report = healing_system['adaptive_optimizer'].get_optimization_report()
    print(f"\nAdaptive optimization:")
    print(f"  Current batch size: {optimization_report['current_parameters']['batch_size']}")
    print(f"  Learning enabled: {optimization_report['learning_enabled']}")
    
    # Get prediction report
    prediction_report = healing_system['predictive_manager'].get_prediction_report()
    print(f"\nPredictive resource management:")
    print(f"  Active predictions: {prediction_report['active_predictions']}")
    print(f"  Average accuracy: {prediction_report['average_accuracy']:.1f}%")
    
    return healing_system


def demonstrate_neural_operator_optimization():
    """Demonstrate neural operator-specific optimizations."""
    print("\n" + "="*60)
    print("NEURAL OPERATOR OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Create optimizer with custom configuration
    batch_config = BatchOptimizationConfig(
        min_batch_size=16,
        max_batch_size=256,
        adaptive_batching=True
    )
    
    compilation_config = CompilationConfig(
        enable_jit=True,
        fuse_operations=True,
        cache_compiled_models=True
    )
    
    optimizer = NeuralOperatorOptimizer(batch_config, compilation_config)
    
    # Create mock neural operator
    operator = MockNeuralOperator(complexity=2)
    
    # Create test data
    test_data = np.random.random((500, 1000))
    
    print("Testing neural operator optimization...")
    
    # Benchmark unoptimized processing
    start_time = time.time()
    unoptimized_results = []
    for i in range(0, len(test_data), 32):
        batch = test_data[i:i+32]
        for sample in batch:
            result = operator(sample.reshape(1, -1))
            unoptimized_results.append(result)
    unoptimized_time = time.time() - start_time
    
    print(f"Unoptimized processing: {unoptimized_time:.3f}s")
    
    # Benchmark optimized processing
    start_time = time.time()
    optimized_result = optimizer.batch_processor.process_batch(
        operator, test_data, lambda model, batch: [model(row.reshape(1, -1)) for row in batch]
    )
    optimized_time = time.time() - start_time
    
    print(f"Optimized processing: {optimized_time:.3f}s")
    print(f"Speedup: {unoptimized_time/optimized_time:.1f}x")
    
    # Get optimization statistics
    batch_stats = optimizer.batch_processor.get_batch_stats()
    print(f"\nBatch processing statistics:")
    print(f"  Current batch size: {batch_stats['current_batch_size']}")
    print(f"  Average throughput: {batch_stats['avg_throughput']:.1f} samples/s")
    
    return optimizer


def demonstrate_benchmarking():
    """Demonstrate comprehensive benchmarking capabilities."""
    print("\n" + "="*60)
    print("BENCHMARKING FRAMEWORK DEMONSTRATION")
    print("="*60)
    
    # Create benchmark suite
    suite = create_neural_operator_benchmarks()
    
    print("Running neural operator benchmark suite...")
    
    # Add custom benchmark
    def latency_benchmark(batch_size=64):
        """Custom latency benchmark."""
        operator = MockNeuralOperator(complexity=1)
        data = np.random.random((batch_size, 1000))
        
        start_time = time.time()
        results = []
        for sample in data:
            result = operator(sample.reshape(1, -1))
            results.append(result)
        
        return time.time() - start_time
    
    latency_config = BenchmarkConfig(
        name="latency_test",
        benchmark_type=BenchmarkType.LATENCY,
        iterations=5,
        warmup_iterations=2
    )
    
    suite.register_benchmark("latency_test", latency_benchmark, latency_config)
    
    # Run benchmark suite
    results = suite.run_suite()
    
    print(f"\nBenchmark results:")
    for name, result in results.items():
        if result and result.status == "completed":
            print(f"  {name}:")
            print(f"    Average time: {result.avg_execution_time:.4f}s")
            print(f"    Operations/sec: {result.operations_per_second:.1f}")
            print(f"    Peak memory: {result.peak_memory_mb:.1f} MB")
    
    # Demonstrate scalability testing
    print("\nRunning scalability analysis...")
    
    scalability_tester = ScalabilityTester()
    
    def scalable_benchmark(batch_size=64):
        """Benchmark that scales with batch size."""
        operator = MockNeuralOperator()
        data = np.random.random((batch_size, 100))
        
        # Process all samples in the batch
        results = []
        for sample in data:
            result = operator(sample.reshape(1, -1))
            results.append(result)
        
        return len(results)
    
    scaling_results = scalability_tester.test_batch_size_scaling(
        scalable_benchmark, max_batch_size=256
    )
    
    print(f"Scaling analysis:")
    print(f"  Scaling quality: {scaling_results['scaling_analysis']['scaling_quality']}")
    print(f"  Average efficiency: {scaling_results['scaling_analysis']['average_efficiency']:.2f}")
    
    return suite


async def run_async_demo():
    """Run asynchronous demonstration."""
    print("\n" + "="*60)
    print("ASYNCHRONOUS PROCESSING DEMONSTRATION")
    print("="*60)
    
    async def async_neural_operation(data_size: int):
        """Asynchronous neural operation."""
        # Simulate async I/O
        await asyncio.sleep(0.1)
        
        # Simulate computation
        data = np.random.random(data_size)
        operator = MockNeuralOperator()
        result = operator(data.reshape(1, -1))
        
        return result.shape[0]
    
    # Run multiple async operations concurrently
    tasks = [async_neural_operation(100 + i*50) for i in range(5)]
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    async_time = time.time() - start_time
    
    print(f"Processed {len(results)} async operations in {async_time:.3f}s")
    print(f"Results: {results}")


def main():
    """Main demonstration function."""
    print("="*60)
    print("NEURAL CRYPTANALYSIS OPTIMIZATION FRAMEWORK")
    print("GENERATION 3 - COMPREHENSIVE DEMONSTRATION")
    print("="*60)
    
    try:
        # Core performance optimization
        optimizer = demonstrate_performance_optimization()
        
        # Research acceleration
        experiment_manager = demonstrate_research_acceleration()
        
        # Self-healing system
        healing_system = demonstrate_self_healing()
        
        # Neural operator optimization
        neural_optimizer = demonstrate_neural_operator_optimization()
        
        # Benchmarking framework
        benchmark_suite = demonstrate_benchmarking()
        
        # Asynchronous processing
        asyncio.run(run_async_demo())
        
        # Final summary
        print("\n" + "="*60)
        print("OPTIMIZATION FRAMEWORK SUMMARY")
        print("="*60)
        
        # Get comprehensive optimization report
        opt_report = optimizer.get_optimization_report()
        
        print(f"Performance Optimization:")
        print(f"  Total operations optimized: {opt_report['total_operations']}")
        print(f"  Cache efficiency: {opt_report['efficiency']['cache_hit_rate']:.1%}")
        print(f"  Memory efficiency: {opt_report['efficiency']['memory_efficiency']:.1f}%")
        
        print(f"\nResearch Acceleration:")
        print(f"  Experiments completed: {len(experiment_manager.completed_experiments)}")
        print(f"  Failed experiments: {len(experiment_manager.failed_experiments)}")
        
        print(f"\nSelf-Healing System:")
        healing_report = healing_system['self_healing'].get_healing_report()
        print(f"  Healing success rate: {healing_report['success_rate']:.1%}")
        print(f"  Total healings attempted: {healing_report['total_healings_attempted']}")
        
        print(f"\nNeural Operator Optimization:")
        neural_opt_report = neural_optimizer.get_optimization_report()
        print(f"  Models optimized: {neural_opt_report['optimized_models']}")
        print(f"  JIT compilations: {neural_opt_report['jit_compilation_stats'].get('compilations', 0)}")
        
        print(f"\nOverall Performance Improvements:")
        print(f"  ✓ Advanced caching with L1/L2/L3 hierarchy")
        print(f"  ✓ Memory pooling and automatic cleanup")
        print(f"  ✓ Concurrent and asynchronous processing")
        print(f"  ✓ Auto-scaling based on system load")
        print(f"  ✓ Self-healing with predictive management")
        print(f"  ✓ JIT compilation and operator fusion")
        print(f"  ✓ Adaptive batch processing")
        print(f"  ✓ Comprehensive benchmarking and regression detection")
        print(f"  ✓ Research pipeline automation")
        print(f"  ✓ Hyperparameter optimization")
        
        print(f"\nGeneration 3 optimization framework successfully demonstrated!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            optimizer.shutdown()
            if 'healing_system' in locals():
                healing_system['health_monitor'].stop_monitoring()
                healing_system['self_healing'].stop_healing()
        except:
            pass


if __name__ == "__main__":
    main()
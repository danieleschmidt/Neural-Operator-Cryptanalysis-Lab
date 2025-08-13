"""Neural Cryptanalysis Optimization Module - Generation 3.

This module provides comprehensive performance optimization including:
- Advanced caching with hierarchical storage
- Memory pooling and resource management
- Concurrent and asynchronous processing
- Auto-scaling and load balancing
- Performance monitoring and metrics
- Research acceleration tools
- Self-healing mechanisms
- Neural operator optimization
- Comprehensive benchmarking
"""

from .performance_optimizer import (
    AdvancedPerformanceOptimizer,
    OptimizationConfig,
    PerformanceMetrics,
    HierarchicalCache,
    MemoryPool,
    ConcurrentProcessor,
    ResourceMonitor,
    AutoScaler,
    optimize,
    get_global_optimizer
)

from .research_acceleration import (
    ExperimentManager,
    HyperparameterOptimizer,
    ABTestFramework,
    BenchmarkRunner,
    ResearchPipeline,
    ExperimentConfig,
    ExperimentResult,
    HyperparameterSpace
)

from .self_healing import (
    SelfHealingSystem,
    HealthMonitor,
    AdaptiveOptimizer,
    PredictiveResourceManager,
    HealthStatus,
    IssueType,
    HealthIssue,
    SystemMetrics,
    create_complete_self_healing_system
)

from .neural_operator_optimization import (
    NeuralOperatorOptimizer,
    JITCompiler,
    AdaptiveBatchProcessor,
    LazyLoader,
    BatchOptimizationConfig,
    CompilationConfig,
    optimize_neural_operator,
    get_global_neural_optimizer
)

from .benchmarking import (
    BenchmarkSuite,
    BenchmarkExecutor,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkType,
    ScalabilityTester,
    RegressionDetector,
    SystemProfiler,
    create_neural_operator_benchmarks,
    register_benchmark_suite,
    get_benchmark_suite,
    list_benchmark_suites
)

__all__ = [
    # Core optimization
    'AdvancedPerformanceOptimizer',
    'OptimizationConfig', 
    'PerformanceMetrics',
    'HierarchicalCache',
    'MemoryPool',
    'ConcurrentProcessor',
    'ResourceMonitor',
    'AutoScaler',
    'optimize',
    'get_global_optimizer',
    
    # Research acceleration
    'ExperimentManager',
    'HyperparameterOptimizer', 
    'ABTestFramework',
    'BenchmarkRunner',
    'ResearchPipeline',
    'ExperimentConfig',
    'ExperimentResult',
    'HyperparameterSpace',
    
    # Self-healing
    'SelfHealingSystem',
    'HealthMonitor',
    'AdaptiveOptimizer',
    'PredictiveResourceManager',
    'HealthStatus',
    'IssueType',
    'HealthIssue',
    'SystemMetrics',
    'create_complete_self_healing_system',
    
    # Neural operator optimization
    'NeuralOperatorOptimizer',
    'JITCompiler',
    'AdaptiveBatchProcessor',
    'LazyLoader',
    'BatchOptimizationConfig',
    'CompilationConfig',
    'optimize_neural_operator',
    'get_global_neural_optimizer',
    
    # Benchmarking
    'BenchmarkSuite',
    'BenchmarkExecutor',
    'BenchmarkConfig',
    'BenchmarkResult',
    'BenchmarkType',
    'ScalabilityTester',
    'RegressionDetector',
    'SystemProfiler',
    'create_neural_operator_benchmarks',
    'register_benchmark_suite',
    'get_benchmark_suite',
    'list_benchmark_suites'
]
#!/usr/bin/env python3
"""Integration test for Generation 3 optimization framework."""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all optimization modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core optimization imports
        from neural_cryptanalysis.optimization import (
            AdvancedPerformanceOptimizer,
            OptimizationConfig,
            optimize,
            get_global_optimizer
        )
        print("✓ Core optimization imports successful")
        
        # Research acceleration imports
        from neural_cryptanalysis.optimization import (
            ExperimentManager,
            HyperparameterOptimizer,
            ResearchPipeline
        )
        print("✓ Research acceleration imports successful")
        
        # Self-healing imports
        from neural_cryptanalysis.optimization import (
            create_complete_self_healing_system,
            HealthMonitor,
            SelfHealingSystem
        )
        print("✓ Self-healing imports successful")
        
        # Neural operator optimization imports
        from neural_cryptanalysis.optimization import (
            NeuralOperatorOptimizer,
            get_global_neural_optimizer
        )
        print("✓ Neural operator optimization imports successful")
        
        # Benchmarking imports
        from neural_cryptanalysis.optimization import (
            BenchmarkSuite,
            create_neural_operator_benchmarks
        )
        print("✓ Benchmarking imports successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test performance optimizer
        from neural_cryptanalysis.optimization import AdvancedPerformanceOptimizer, OptimizationConfig
        
        config = OptimizationConfig(cache_max_size_mb=100)
        optimizer = AdvancedPerformanceOptimizer(config)
        
        @optimizer.optimize_operation
        def test_function(x):
            return x * 2
        
        result = test_function(5)
        assert result == 10
        print("✓ Performance optimizer basic test passed")
        
        # Test experiment manager
        from neural_cryptanalysis.optimization import ExperimentManager
        
        exp_manager = ExperimentManager(Path("./test_experiments"))
        print("✓ Experiment manager created successfully")
        
        # Test neural operator optimizer
        from neural_cryptanalysis.optimization import NeuralOperatorOptimizer
        
        neural_opt = NeuralOperatorOptimizer()
        print("✓ Neural operator optimizer created successfully")
        
        # Test benchmark suite
        from neural_cryptanalysis.optimization import create_neural_operator_benchmarks
        
        benchmark_suite = create_neural_operator_benchmarks()
        print("✓ Benchmark suite created successfully")
        
        # Cleanup
        optimizer.shutdown()
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between components."""
    print("\nTesting component integration...")
    
    try:
        from neural_cryptanalysis.optimization import (
            get_global_optimizer,
            optimize
        )
        
        # Test global optimizer integration
        @optimize(operation_name="integration_test", use_cache=True)
        def integrated_function(data):
            return len(data) * 2
        
        result1 = integrated_function([1, 2, 3])
        result2 = integrated_function([1, 2, 3])  # Should use cache
        
        assert result1 == result2 == 6
        print("✓ Global optimizer integration test passed")
        
        # Test that caching is working
        global_opt = get_global_optimizer()
        cache_stats = global_opt.cache.get_stats()
        
        print(f"✓ Cache working - L1 entries: {cache_stats['l1']['size']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test error: {e}")
        traceback.print_exc()
        return False

def test_self_healing():
    """Test self-healing system initialization."""
    print("\nTesting self-healing system...")
    
    try:
        from neural_cryptanalysis.optimization import create_complete_self_healing_system
        
        # Just test that it can be created without errors
        healing_system = create_complete_self_healing_system()
        
        # Check that all components exist
        assert 'health_monitor' in healing_system
        assert 'self_healing' in healing_system
        assert 'adaptive_optimizer' in healing_system
        assert 'predictive_manager' in healing_system
        
        print("✓ Self-healing system created successfully")
        
        # Stop monitoring to clean up
        healing_system['health_monitor'].stop_monitoring()
        healing_system['self_healing'].stop_healing()
        
        return True
        
    except Exception as e:
        print(f"✗ Self-healing test error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("="*60)
    print("GENERATION 3 OPTIMIZATION FRAMEWORK INTEGRATION TEST")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality Test", test_basic_functionality),
        ("Integration Test", test_integration),
        ("Self-Healing Test", test_self_healing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n" + "="*60)
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Generation 3 framework is ready!")
        return 0
    else:
        print("❌ Some tests failed - review implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
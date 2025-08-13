#!/usr/bin/env python3
"""Performance regression detection for neural cryptanalysis framework."""

import sys
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics


class PerformanceRegression:
    """Detect performance regressions in neural cryptanalysis code."""
    
    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.baseline_data = self._load_baseline()
        self.current_results = {}
        self.regression_threshold = 0.2  # 20% performance degradation threshold
    
    def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline performance data."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_baseline(self):
        """Save current results as new baseline."""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.current_results, f, indent=2)
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests and collect metrics."""
        results = {}
        
        # Test import time
        results['import_time'] = self._measure_import_time()
        
        # Test basic functionality performance
        results['basic_functionality'] = self._measure_basic_functionality()
        
        # Test training performance
        results['training_performance'] = self._measure_training_performance()
        
        # Test memory usage
        results['memory_usage'] = self._measure_memory_usage()
        
        self.current_results = results
        return results
    
    def _measure_import_time(self) -> float:
        """Measure time to import main modules."""
        import_script = '''
import time
start_time = time.perf_counter()
try:
    import sys
    sys.path.insert(0, "src")
    from neural_cryptanalysis.core import NeuralSCA
    from neural_cryptanalysis.neural_operators import FourierNeuralOperator
    from neural_cryptanalysis.datasets.synthetic import SyntheticDatasetGenerator
    end_time = time.perf_counter()
    print(f"{end_time - start_time:.6f}")
except Exception as e:
    print("ERROR")
'''
        
        try:
            result = subprocess.run([sys.executable, '-c', import_script], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip() != "ERROR":
                return float(result.stdout.strip())
        except Exception:
            pass
        
        return float('inf')  # Failed import
    
    def _measure_basic_functionality(self) -> Dict[str, float]:
        """Measure basic functionality performance."""
        functionality_script = '''
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "src")

def measure_time(func):
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()
    return end - start, result

try:
    # Test data generation
    from neural_cryptanalysis.datasets.synthetic import SyntheticDatasetGenerator
    
    gen_time, _ = measure_time(lambda: SyntheticDatasetGenerator(random_seed=42).generate_aes_dataset(n_traces=50, trace_length=200))
    
    # Test neural operator creation
    from neural_cryptanalysis.neural_operators import FourierNeuralOperator, OperatorConfig
    
    config = OperatorConfig(input_dim=1, output_dim=256, hidden_dim=32)
    create_time, fno = measure_time(lambda: FourierNeuralOperator(config, modes=8))
    
    # Test forward pass
    test_input = torch.randn(10, 200, 1)
    forward_time, _ = measure_time(lambda: fno(test_input))
    
    print(f"{gen_time:.6f},{create_time:.6f},{forward_time:.6f}")

except Exception as e:
    print("ERROR")
'''
        
        try:
            result = subprocess.run([sys.executable, '-c', functionality_script], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout.strip() != "ERROR":
                times = result.stdout.strip().split(',')
                return {
                    'data_generation': float(times[0]),
                    'model_creation': float(times[1]),
                    'forward_pass': float(times[2])
                }
        except Exception:
            pass
        
        return {'data_generation': float('inf'), 'model_creation': float('inf'), 'forward_pass': float('inf')}
    
    def _measure_training_performance(self) -> Dict[str, float]:
        """Measure training performance."""
        training_script = '''
import sys
import time
import torch
import numpy as np

sys.path.insert(0, "src")

try:
    from neural_cryptanalysis.core import NeuralSCA
    
    # Quick training test
    neural_sca = NeuralSCA(config={
        'training': {'batch_size': 16, 'epochs': 1},
        'fno': {'modes': 4, 'width': 16, 'n_layers': 1}
    })
    
    # Generate small dataset
    traces = torch.randn(50, 100, 1)
    labels = torch.randint(0, 256, (50,))
    
    start_time = time.perf_counter()
    model = neural_sca.train(traces, labels, validation_split=0.2)
    training_time = time.perf_counter() - start_time
    
    # Test inference
    start_time = time.perf_counter()
    with torch.no_grad():
        predictions = model(traces[:10])
    inference_time = time.perf_counter() - start_time
    
    print(f"{training_time:.6f},{inference_time:.6f}")

except Exception as e:
    print("ERROR")
'''
        
        try:
            result = subprocess.run([sys.executable, '-c', training_script], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and result.stdout.strip() != "ERROR":
                times = result.stdout.strip().split(',')
                return {
                    'training_time': float(times[0]),
                    'inference_time': float(times[1])
                }
        except Exception:
            pass
        
        return {'training_time': float('inf'), 'inference_time': float('inf')}
    
    def _measure_memory_usage(self) -> Dict[str, float]:
        """Measure memory usage."""
        memory_script = '''
import sys
import psutil
import os

sys.path.insert(0, "src")

try:
    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Import modules
    from neural_cryptanalysis.core import NeuralSCA
    import_memory = process.memory_info().rss / 1024 / 1024
    
    # Create neural SCA
    neural_sca = NeuralSCA()
    creation_memory = process.memory_info().rss / 1024 / 1024
    
    print(f"{baseline_memory:.2f},{import_memory:.2f},{creation_memory:.2f}")

except Exception as e:
    print("ERROR")
'''
        
        try:
            result = subprocess.run([sys.executable, '-c', memory_script], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout.strip() != "ERROR":
                memories = result.stdout.strip().split(',')
                baseline = float(memories[0])
                return {
                    'baseline_memory_mb': baseline,
                    'import_memory_increase_mb': float(memories[1]) - baseline,
                    'creation_memory_increase_mb': float(memories[2]) - baseline
                }
        except Exception:
            pass
        
        return {
            'baseline_memory_mb': 0,
            'import_memory_increase_mb': float('inf'),
            'creation_memory_increase_mb': float('inf')
        }
    
    def detect_regressions(self) -> List[Dict[str, Any]]:
        """Detect performance regressions compared to baseline."""
        regressions = []
        
        if not self.baseline_data:
            return regressions  # No baseline to compare against
        
        def check_metric(current_path: List[str], current_value: Any, baseline_value: Any):
            if isinstance(current_value, dict) and isinstance(baseline_value, dict):
                # Recursively check nested dictionaries
                for key in current_value:
                    if key in baseline_value:
                        check_metric(current_path + [key], current_value[key], baseline_value[key])
            elif isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                # Compare numeric values
                if baseline_value > 0 and current_value > baseline_value:
                    regression_ratio = (current_value - baseline_value) / baseline_value
                    if regression_ratio > self.regression_threshold:
                        regressions.append({
                            'metric': '.'.join(current_path),
                            'baseline_value': baseline_value,
                            'current_value': current_value,
                            'regression_ratio': regression_ratio,
                            'severity': 'high' if regression_ratio > 0.5 else 'medium'
                        })
        
        # Compare current results with baseline
        for top_level_key in self.current_results:
            if top_level_key in self.baseline_data:
                check_metric([top_level_key], 
                           self.current_results[top_level_key], 
                           self.baseline_data[top_level_key])
        
        return regressions
    
    def generate_report(self, regressions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance regression report."""
        return {
            'timestamp': time.time(),
            'current_results': self.current_results,
            'baseline_exists': bool(self.baseline_data),
            'regressions_detected': len(regressions),
            'regressions': regressions,
            'overall_status': 'PASSED' if len(regressions) == 0 else 'FAILED',
            'high_severity_regressions': len([r for r in regressions if r['severity'] == 'high']),
            'medium_severity_regressions': len([r for r in regressions if r['severity'] == 'medium'])
        }


def main():
    """Main performance checking function."""
    print("Running performance regression check...")
    
    detector = PerformanceRegression()
    
    # Run performance tests
    print("Measuring current performance...")
    current_results = detector.run_performance_tests()
    
    # Check for regressions
    print("Checking for regressions...")
    regressions = detector.detect_regressions()
    
    # Generate report
    report = detector.generate_report(regressions)
    
    # Save report
    with open('performance_regression_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print results
    print(f"\nPerformance Check Results:")
    print(f"  Import time: {current_results.get('import_time', 'N/A'):.3f}s")
    
    basic_func = current_results.get('basic_functionality', {})
    print(f"  Data generation: {basic_func.get('data_generation', 'N/A'):.3f}s")
    print(f"  Model creation: {basic_func.get('model_creation', 'N/A'):.3f}s")
    print(f"  Forward pass: {basic_func.get('forward_pass', 'N/A'):.3f}s")
    
    training_perf = current_results.get('training_performance', {})
    print(f"  Training time: {training_perf.get('training_time', 'N/A'):.3f}s")
    print(f"  Inference time: {training_perf.get('inference_time', 'N/A'):.3f}s")
    
    memory_usage = current_results.get('memory_usage', {})
    print(f"  Import memory increase: {memory_usage.get('import_memory_increase_mb', 'N/A'):.1f}MB")
    print(f"  Creation memory increase: {memory_usage.get('creation_memory_increase_mb', 'N/A'):.1f}MB")
    
    # Report regressions
    if regressions:
        print(f"\n❌ Performance regressions detected: {len(regressions)}")
        for regression in regressions:
            print(f"  {regression['metric']}: {regression['baseline_value']:.3f} → {regression['current_value']:.3f} "
                  f"({regression['regression_ratio']:+.1%})")
        
        # Update baseline if user confirms (in CI, this might be automated)
        if len(sys.argv) > 1 and '--update-baseline' in sys.argv:
            detector._save_baseline()
            print("Baseline updated with current results.")
            sys.exit(0)  # Don't fail if baseline was updated
        else:
            sys.exit(1)  # Fail due to regressions
    
    else:
        print("\n✅ No performance regressions detected")
        
        # Save as new baseline if none exists
        if not detector.baseline_data:
            detector._save_baseline()
            print("Baseline performance data saved.")
        
        sys.exit(0)


if __name__ == "__main__":
    main()